from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml
from packaging.requirements import InvalidRequirement
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.models import AuditRequest, MetadataAccess

_SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "dependency_audit.py"
_SPEC = spec_from_file_location("dependency_audit_script", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("dependency audit script cannot be loaded")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


@pytest.fixture(autouse=True)
def _inject_clean_auditor_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(_MODULE, "live_code_provenance"):
        return
    provenance = _MODULE.CodeProvenance(
        commit="a" * 40,
        sources=(_MODULE.SourceHash(path="auditor.py", sha256="b" * 64),),
    )
    monkeypatch.setattr(_MODULE, "live_code_provenance", lambda: provenance)


def _simple(name: str, version: str, size: int) -> bytes:
    filename = f"{name}-{version}-py3-none-any.whl"
    return json.dumps(
        {
            "meta": {"api-version": "1.0"},
            "name": name,
            "files": [
                {
                    "core-metadata": False,
                    "data-dist-info-metadata": False,
                    "filename": filename,
                    "hashes": {"sha256": "a" * 64},
                    "requires-python": ">=3.12",
                    "size": size,
                    "upload-time": "2026-08-01T09:00:00.000000Z",
                    "url": (f"https://files.pythonhosted.org/packages/aa/aa/{'a' * 60}/{filename}"),
                    "yanked": False,
                }
            ],
        }
    ).encode("utf-8")


def _version(
    name: str,
    version: str,
    *,
    license_expression: str | None,
    requires: list[str],
) -> bytes:
    return json.dumps(
        {
            "info": {
                "name": name,
                "version": version,
                "requires_dist": requires,
                "license_expression": license_expression,
                "license": None,
                "classifiers": [],
            },
            "urls": [],
        }
    ).encode("utf-8")


class FixtureMetadataClient:
    def __init__(self, *, child_license: str | None = "MIT") -> None:
        self._simple = {
            "rfdetr": _simple("rfdetr", "1.9.0", 100),
            "trackers": _simple("trackers", "2.5.0.post0", 200),
            "child": _simple("child", "2.0.0", 300),
        }
        self._version = {
            ("rfdetr", "1.9.0"): _version(
                "rfdetr", "1.9.0", license_expression="Apache-2.0", requires=["child>=2"]
            ),
            ("trackers", "2.5.0.post0"): _version(
                "trackers", "2.5.0.post0", license_expression="MIT", requires=[]
            ),
            ("child", "2.0.0"): _version(
                "child", "2.0.0", license_expression=child_license, requires=[]
            ),
        }
        self.metadata_accesses = (
            MetadataAccess(
                url="https://pypi.org/simple/rfdetr/",
                content_type="application/vnd.pypi.simple.v1+json",
                bytes=10,
                sha256="d" * 64,
            ),
        )

    def simple(self, name: str) -> bytes:
        return self._simple[name]

    def version(self, name: str, version: str) -> bytes:
        return self._version[(name, version)]


@pytest.fixture
def passing_client_factory():
    return lambda _request: FixtureMetadataClient()


@pytest.fixture
def unknown_license_client_factory():
    return lambda _request: FixtureMetadataClient(child_license=None)


@pytest.fixture
def legal_review_client_factory():
    return lambda _request: FixtureMetadataClient(child_license="MPL-2.0")


def _request(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "1.0",
        "request_id": "fixture-request",
        "purpose": "synthetic_smoke",
        "target": {
            "os": "macos",
            "os_version": "26.0",
            "architecture": "arm64",
            "python_implementation": "cp",
            "python_version": "3.12.13",
        },
        "roots": ["rfdetr==1.9.0", "trackers==2.5.0.post0"],
        "allowed_licenses": ["Apache-2.0", "MIT"],
        "single_response_bytes_max": 12 * 1024 * 1024,
        "total_metadata_bytes_max": 64 * 1024 * 1024,
        "existing_payload_bytes": 0,
        "installed_size_multipliers": {"pure_python": 4.0, "native": 2.5},
        "free_space_safety_ratio": 0.2,
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_cli_writes_packet_without_archive_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    request_path = _request(tmp_path)

    exit_code = main(
        ["--request", str(request_path), "--output", str(tmp_path / "out")],
        client_factory=passing_client_factory,
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    packet = json.loads(Path(summary["packet_dir"], "packet.json").read_text(encoding="utf-8"))
    assert summary == {
        "archive_bytes": 600,
        "blockers": [],
        "package_count": 3,
        "packet_dir": str(tmp_path / "out" / "fixture-request" / packet["packet_sha256"]),
        "packet_sha256": packet["packet_sha256"],
        "passed": True,
    }
    assert packet["config_sha256"] == sha256(request_path.read_bytes()).hexdigest()
    assert packet["side_effects"] == {
        "archive_downloads": 0,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }


def test_cli_returns_two_for_unknown_licence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], unknown_license_client_factory
) -> None:
    exit_code = main(
        ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
        client_factory=unknown_license_client_factory,
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["blockers"] == ["unknown_license:child==2.0.0"]
    assert summary["passed"] is False
    assert Path(summary["packet_dir"], "packet.json").is_file()


def test_cli_returns_two_for_legal_review_licence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], legal_review_client_factory
) -> None:
    exit_code = main(
        ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
        client_factory=legal_review_client_factory,
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["blockers"] == ["legal_review_required:child==2.0.0"]
    assert summary["passed"] is False
    assert Path(summary["packet_dir"], "packet.json").is_file()


def test_cli_propagates_an_unexpected_os_error_from_the_client_factory(tmp_path: Path) -> None:
    def broken_factory(_request):
        raise OSError("client factory defect")

    with pytest.raises(OSError, match="client factory defect"):
        main(
            ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
            client_factory=broken_factory,
        )


def test_cli_propagates_an_unexpected_os_error_from_the_resolver_boundary(tmp_path: Path) -> None:
    class BrokenMetadataClient:
        metadata_accesses = ()

        def simple(self, name: str) -> bytes:
            raise OSError(f"resolver client defect for {name}")

    with pytest.raises(OSError, match="resolver client defect"):
        main(
            ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
            client_factory=lambda _request: BrokenMetadataClient(),
        )


def test_final_fix_09_cli_maps_expected_live_transport_failure_to_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailedLiveTransportClient:
        metadata_accesses = ()

        def simple(self, name: str) -> bytes:
            raise _MODULE.MetadataTransportError(f"metadata transport failed for {name}")

    exit_code = main(
        ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
        client_factory=lambda _request: FailedLiveTransportClient(),
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    failure = json.loads(Path(summary["failure_dir"], "failure.json").read_text(encoding="utf-8"))
    assert captured.err == ""
    assert summary["passed"] is False
    assert summary["blockers"] == ["resolution:dependency_resolution"]
    assert "metadata transport failed" in failure["error"]
    assert failure["approvable"] is False


def test_cli_returns_two_for_a_missing_request_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    exit_code = main(
        ["--request", str(tmp_path / "missing.yaml"), "--output", str(tmp_path / "out")],
        client_factory=passing_client_factory,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "request read failed" in captured.err


def test_cli_returns_two_for_an_output_publication_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    output = tmp_path / "output-file"
    output.write_text("not a directory", encoding="utf-8")

    exit_code = main(
        ["--request", str(_request(tmp_path)), "--output", str(output)],
        client_factory=passing_client_factory,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "packet output failed" in captured.err


def test_cli_returns_two_for_malformed_yaml_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    request = tmp_path / "malformed.yaml"
    request.write_text("roots: [", encoding="utf-8")

    exit_code = main(
        ["--request", str(request), "--output", str(tmp_path / "out")],
        client_factory=passing_client_factory,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"request parse failed: {request}" in captured.err


def test_cli_returns_two_for_invalid_model_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    request = tmp_path / "invalid-model.yaml"
    request.write_text('schema_version: "1.0"\n', encoding="utf-8")

    exit_code = main(
        ["--request", str(request), "--output", str(tmp_path / "out")],
        client_factory=passing_client_factory,
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"request validation failed: {request}" in captured.err


@pytest.mark.parametrize(
    ("error_factory", "error_type"),
    [
        (lambda: AuditRequest.model_validate({}), ValidationError),
        (lambda: InvalidRequirement("packet construction defect"), InvalidRequirement),
        (lambda: yaml.YAMLError("packet construction defect"), yaml.YAMLError),
    ],
)
def test_cli_propagates_post_load_validation_style_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_client_factory,
    error_factory,
    error_type,
) -> None:
    def broken_packet(*_args, **_kwargs):
        raise error_factory()

    monkeypatch.setattr(_MODULE, "build_approval_packet", broken_packet)

    with pytest.raises(error_type):
        main(
            ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
            client_factory=passing_client_factory,
        )


def test_cli_returns_sixty_four_for_missing_request_argument() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/algorithm/dependency_audit.py"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 64
    assert completed.stdout == ""
    assert "usage:" in completed.stderr


def _initialize_provenance_repository(root: Path) -> tuple[str, ...]:
    script = root / "scripts" / "algorithm" / "dependency_audit.py"
    module = root / "src" / "billiards_ai" / "lab" / "dependency_audit" / "models.py"
    script.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    script.write_text("print('audit')\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Dependency Audit Test",
            "-c",
            "user.email=audit@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return (
        "scripts/algorithm/dependency_audit.py",
        "src/billiards_ai/lab/dependency_audit",
    )


def test_final_fix_10_clean_auditor_sources_record_commit_and_exact_hashes(
    tmp_path: Path,
) -> None:
    source_roots = _initialize_provenance_repository(tmp_path)

    provenance = _MODULE.collect_code_provenance(tmp_path, source_roots)

    assert (
        provenance.commit
        == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    )
    assert provenance.sources == (
        _MODULE.SourceHash(
            path="scripts/algorithm/dependency_audit.py",
            sha256=sha256(b"print('audit')\n").hexdigest(),
        ),
        _MODULE.SourceHash(
            path="src/billiards_ai/lab/dependency_audit/models.py",
            sha256=sha256(b"VALUE = 1\n").hexdigest(),
        ),
    )


@pytest.mark.parametrize("dirty_state", ["staged", "unstaged", "untracked"])
def test_final_fix_10_dirty_auditor_sources_fail_closed(tmp_path: Path, dirty_state: str) -> None:
    source_roots = _initialize_provenance_repository(tmp_path)
    models_path = tmp_path / "src" / "billiards_ai" / "lab" / "dependency_audit" / "models.py"
    if dirty_state == "untracked":
        (models_path.parent / "new_boundary.py").write_text("NEW = True\n", encoding="utf-8")
    else:
        models_path.write_text("VALUE = 2\n", encoding="utf-8")
        if dirty_state == "staged":
            subprocess.run(["git", "add", str(models_path)], cwd=tmp_path, check=True)

    with pytest.raises(_MODULE.AuditProvenanceError, match="auditor source tree is dirty"):
        _MODULE.collect_code_provenance(tmp_path, source_roots)


def test_final_fix_10_cli_maps_provenance_rejection_to_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing_client_factory
) -> None:
    def reject_dirty_source():
        raise _MODULE.AuditProvenanceError("auditor source tree is dirty")

    exit_code = main(
        ["--request", str(_request(tmp_path)), "--output", str(tmp_path / "out")],
        client_factory=passing_client_factory,
        provenance_factory=reject_dirty_source,
    )

    assert exit_code == 2
    assert "auditor source tree is dirty" in capsys.readouterr().err
