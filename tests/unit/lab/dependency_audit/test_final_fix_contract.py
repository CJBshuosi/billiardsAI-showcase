from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable

import pytest
from packaging.specifiers import SpecifierSet
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit import index as index_module
from billiards_ai.lab.dependency_audit import models as models_module
from billiards_ai.lab.dependency_audit import packet as packet_module
from billiards_ai.lab.dependency_audit import resolver as resolver_module


def _target(
    *,
    os_name: str = "macos",
    os_version: str = "26.0",
    architecture: str = "arm64",
    python_version: str = "3.12.13",
):
    return models_module.TargetPlatform.model_construct(
        os=os_name,
        os_version=os_version,
        architecture=architecture,
        python_implementation="cp",
        python_version=python_version,
    )


def _request(*, target=None, roots: tuple[str, ...] | None = None):
    return models_module.AuditRequest.model_construct(
        schema_version="1.0",
        request_id="final-fix-fixture",
        purpose="synthetic_smoke",
        target=target or _target(),
        roots=tuple(
            models_module.RootRequirement.parse(item)
            for item in (roots or ("rfdetr==1.9.0", "trackers==2.5.0.post0"))
        ),
        allowed_licenses=("Apache-2.0", "MIT"),
        single_response_bytes_max=12 * 1024 * 1024,
        total_metadata_bytes_max=64 * 1024 * 1024,
        existing_payload_bytes=10,
        installed_size_multipliers={"pure_python": 4.0, "native": 2.5},
        free_space_safety_ratio=0.20,
    )


def _file_record(
    filename: str,
    *,
    requires_python: str = ">=3.12",
    yanked: bool | str = False,
    hashes: dict[str, str] | None = None,
    size: int = 100,
    url: str | None = None,
) -> dict[str, object]:
    archive_url = f"https://files.pythonhosted.org/packages/aa/aa/{'a' * 60}/{filename}"
    return {
        "core-metadata": False,
        "data-dist-info-metadata": False,
        "filename": filename,
        "hashes": {"sha256": "a" * 64} if hashes is None else hashes,
        "requires-python": requires_python,
        "size": size,
        "upload-time": "2026-08-01T09:00:00.000000Z",
        "url": url or archive_url,
        "yanked": yanked,
    }


def _simple_payload(name: str, records: list[object]) -> bytes:
    return json.dumps(
        {"meta": {"api-version": "1.0"}, "name": name, "files": records},
        sort_keys=True,
    ).encode("utf-8")


def _version_payload(name: str, version: str, requires: list[str]) -> bytes:
    return json.dumps(
        {
            "info": {
                "name": name,
                "version": version,
                "summary": f"{name} summary",
                "requires_python": ">=3.11",
                "requires_dist": requires,
                "license_expression": "MIT",
                "license": None,
                "classifiers": ["License :: OSI Approved :: MIT License"],
            },
            "urls": [],
        },
        sort_keys=True,
    ).encode("utf-8")


def test_final_fix_01_freezes_complete_target_tags_in_the_audit() -> None:
    target = _target()
    snapshot = index_module.resolve_target(target)
    encoded_tags = ("\n".join(snapshot.wheel_tags) + "\n").encode("utf-8")

    assert len(snapshot.wheel_tags) == 1230
    assert hashlib.sha256(encoded_tags).hexdigest() == (
        "716f3d9c2eabe5f8360480a5171c53a4d926bdad3c5dcf2542a8fd80f3431074"
    )
    assert snapshot.wheel_tags[:2] == (
        "cp312-cp312-macosx_26_0_arm64",
        "cp312-cp312-macosx_26_0_universal2",
    )
    assert snapshot.wheel_tags[-1] == "py30-none-any"
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        snapshot.wheel_tags = ()

    assert snapshot.marker_environment.platform_release is None
    assert snapshot.marker_environment.platform_version is None


def test_final_fix_02_preserves_every_nonfatal_candidate_exclusion() -> None:
    snapshot = index_module.resolve_target(_target())
    records = [
        _file_record("demo-3.0.0.tar.gz"),
        _file_record("demo-2.1.0rc1-py3-none-any.whl"),
        _file_record("demo-2.0.0-cp311-cp311-macosx_26_0_arm64.whl"),
        _file_record("demo-1.9.0-py3-none-any.whl", yanked="broken"),
        _file_record("demo-1.8.0-py3-none-any.whl", requires_python=">=3.13"),
        _file_record("demo-1.7.0-py3-none-any.whl"),
    ]

    result = index_module.parse_simple_index(
        _simple_payload("demo", records), snapshot, expected_package="demo"
    )

    assert [candidate.filename for candidate in result.candidates] == [
        "demo-1.7.0-py3-none-any.whl"
    ]
    assert {(item.filename, item.category, item.reason) for item in result.diagnostics} == {
        ("demo-3.0.0.tar.gz", "excluded", "source_distribution"),
        ("demo-2.1.0rc1-py3-none-any.whl", "excluded", "prerelease"),
        (
            "demo-2.0.0-cp311-cp311-macosx_26_0_arm64.whl",
            "excluded",
            "incompatible_tag",
        ),
        ("demo-1.9.0-py3-none-any.whl", "excluded", "yanked"),
        ("demo-1.8.0-py3-none-any.whl", "excluded", "requires_python"),
    }


@pytest.mark.parametrize(
    ("record", "filename", "reason"),
    [
        (_file_record("not-a-valid-wheel.whl"), "not-a-valid-wheel.whl", "malformed_filename"),
        (
            _file_record("demo-1.0.0-py3-none-any.whl", hashes={}),
            "demo-1.0.0-py3-none-any.whl",
            "missing_sha256",
        ),
        (
            _file_record("demo-1.0.0-py3-none-any.whl", size=0),
            "demo-1.0.0-py3-none-any.whl",
            "invalid_size",
        ),
        ({"filename": ""}, "<missing filename>", "malformed_record"),
    ],
)
def test_final_fix_02_fatal_candidate_defects_retain_structured_diagnostics(
    record: object, filename: str, reason: str
) -> None:
    snapshot = index_module.resolve_target(_target())

    with pytest.raises(index_module.WheelIndexError) as captured:
        index_module.parse_simple_index(
            _simple_payload("demo", [record]), snapshot, expected_package="demo"
        )

    assert captured.value.diagnostics[-1].model_dump() == {
        "package": "demo",
        "filename": filename,
        "category": "blocked",
        "reason": reason,
        "detail": captured.value.diagnostics[-1].detail,
    }


def test_final_fix_02_resolver_error_retains_diagnostics_from_every_parsed_package() -> None:
    class MultiPackageClient:
        def simple(self, name: str) -> bytes:
            wheel_name = name.replace("-", "_")
            valid = _file_record(f"{wheel_name}-1.0.0-py3-none-any.whl")
            if name == "root-a":
                return _simple_payload(name, [_file_record("root_a-1.0.0.tar.gz"), valid])
            return _simple_payload(
                name,
                [
                    _file_record(
                        f"{wheel_name}-1.0.0-py3-none-any.whl",
                        hashes={},
                    )
                ],
            )

        def version(self, name: str, version: str) -> bytes:
            return _version_payload(name, version, [])

    request = _request(roots=("root-a==1.0.0", "root-b==1.0.0"))

    with pytest.raises(resolver_module.DependencyResolutionError) as captured:
        resolver_module.resolve_dependency_audit(request, MultiPackageClient())

    assert {(item.package, item.filename, item.reason) for item in captured.value.diagnostics} == {
        ("root-a", "root_a-1.0.0.tar.gz", "source_distribution"),
        ("root-b", "root_b-1.0.0-py3-none-any.whl", "missing_sha256"),
    }


def test_final_fix_03_expression_classifier_conflicts_are_unknown_and_retained() -> None:
    decision = resolver_module.classify_license(
        {
            "license_expression": "MIT",
            "classifiers": [
                "License :: OSI Approved :: Apache Software License",
                "License :: Other/Proprietary License",
            ],
            "license": "MIT",
        },
        ("MIT",),
    )

    assert decision.status == "unknown"
    assert decision.expression == "MIT"
    assert decision.reason == "conflicting_evidence"
    assert decision.evidence == (
        "license_expression: MIT",
        "classifier: License :: OSI Approved :: Apache Software License",
        "classifier: License :: Other/Proprietary License",
        "license: MIT",
    )


def test_final_fix_04_newer_universal_wheel_beats_older_platform_wheel() -> None:
    target = _target()
    payload = _simple_payload(
        "demo",
        [
            _file_record("demo-1.9.0-cp312-cp312-macosx_26_0_arm64.whl"),
            _file_record("demo-2.0.0-py3-none-any.whl"),
        ],
    )

    candidates = index_module.parse_simple_candidates(payload, target)
    selected = index_module.select_best_wheel(candidates, SpecifierSet(">=1"))

    assert [candidate.version for candidate in candidates] == ["2.0.0", "1.9.0"]
    assert selected.filename == "demo-2.0.0-py3-none-any.whl"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            _target(
                os_name="windows",
                os_version="11.0",
                architecture="x86_64",
                python_version="3.11.9",
            ),
            {
                "implementation_name": "cpython",
                "implementation_version": "3.11.9",
                "os_name": "nt",
                "platform_machine": "AMD64",
                "platform_python_implementation": "CPython",
                "platform_release": None,
                "platform_system": "Windows",
                "platform_version": None,
                "python_full_version": "3.11.9",
                "python_version": "3.11",
                "sys_platform": "win32",
            },
        ),
        (
            _target(
                os_name="linux",
                os_version="6.8",
                architecture="arm64",
                python_version="3.13.2",
            ),
            {
                "implementation_name": "cpython",
                "implementation_version": "3.13.2",
                "os_name": "posix",
                "platform_machine": "aarch64",
                "platform_python_implementation": "CPython",
                "platform_release": None,
                "platform_system": "Linux",
                "platform_version": None,
                "python_full_version": "3.13.2",
                "python_version": "3.13",
                "sys_platform": "linux",
            },
        ),
        (
            _target(os_name="macos", architecture="x86_64"),
            {
                "implementation_name": "cpython",
                "implementation_version": "3.12.13",
                "os_name": "posix",
                "platform_machine": "x86_64",
                "platform_python_implementation": "CPython",
                "platform_release": None,
                "platform_system": "Darwin",
                "platform_version": None,
                "python_full_version": "3.12.13",
                "python_version": "3.12",
                "sys_platform": "darwin",
            },
        ),
    ],
)
def test_final_fix_05_marker_environment_is_derived_only_from_the_request_target(
    target, expected: dict[str, str | None]
) -> None:
    snapshot = index_module.resolve_target(target)

    assert snapshot.marker_environment.model_dump() == expected


def test_final_fix_05_multiple_parents_and_extras_use_the_target_marker_environment() -> None:
    class MarkerClient:
        def __init__(self) -> None:
            self.dependencies = {
                "root-a": [
                    'shared>=1; extra == "cpu"',
                    'gpu-child>=1; extra == "gpu"',
                    (
                        'windows-child>=1; sys_platform == "win32" and '
                        'platform_machine == "AMD64" and python_version == "3.11"'
                    ),
                ],
                "root-b": ['shared>=1; os_name == "nt"'],
                "shared": [],
                "gpu-child": [],
                "windows-child": [],
            }

        def simple(self, name: str) -> bytes:
            wheel_name = name.replace("-", "_")
            return _simple_payload(
                name,
                [
                    _file_record(
                        f"{wheel_name}-1.0.0-py3-none-any.whl",
                        requires_python=">=3.11",
                    )
                ],
            )

        def version(self, name: str, version: str) -> bytes:
            return _version_payload(name, version, self.dependencies[name])

    target = _target(
        os_name="windows",
        os_version="11.0",
        architecture="x86_64",
        python_version="3.11.9",
    )
    request = _request(target=target, roots=("root-a[cpu,gpu]==1.0.0", "root-b==1.0.0"))
    snapshot = index_module.resolve_target(target)

    audit = resolver_module.resolve_dependency_audit(request, MarkerClient(), snapshot)

    assert [package.name for package in audit.packages] == [
        "gpu-child",
        "root-a",
        "root-b",
        "shared",
        "windows-child",
    ]
    root_a = next(package for package in audit.packages if package.name == "root-a")
    shared = next(package for package in audit.packages if package.name == "shared")
    assert root_a.extras == ("cpu", "gpu")
    assert {edge.child for edge in root_a.dependencies} == {
        "gpu-child>=1",
        "shared>=1",
        "windows-child>=1",
    }
    assert shared.parents == ("root-a==1.0.0", "root-b==1.0.0")


def _golden_packet():
    target = _target()
    request = _request(target=target)
    snapshot = index_module.resolve_target(target).model_copy(
        update={"wheel_tags": ("py3-none-any",)}
    )
    package = models_module.ResolvedPackage(
        name="rfdetr",
        version="1.9.0",
        extras=("feature",),
        parents=("<root>",),
        wheel=models_module.WheelArtifact(
            name="rfdetr",
            version="1.9.0",
            filename="rfdetr-1.9.0-py3-none-any.whl",
            url=(
                "https://files.pythonhosted.org/packages/11/11/"
                + "1" * 60
                + "/rfdetr-1.9.0-py3-none-any.whl"
            ),
            tags=("py3-none-any",),
            bytes=100,
            sha256="1" * 64,
            requires_python=">=3.12",
            upload_time="2026-08-01T09:00:00Z",
            is_pure_python=True,
        ),
        dependencies=(),
        project_metadata=models_module.ProjectMetadata(
            name="rfdetr",
            version="1.9.0",
            summary="Golden rfdetr",
            requires_python=">=3.12",
            requires_dist=(),
        ),
        license=models_module.LicenseDecision(
            expression="MIT",
            evidence=(
                "license_expression: MIT",
                "classifier: License :: OSI Approved :: MIT License",
            ),
            status="allowed",
            reason="allowlisted",
        ),
    )
    diagnostic = models_module.CandidateDiagnostic(
        package="rfdetr",
        filename="rfdetr-1.8.0-py3-none-any.whl",
        category="excluded",
        reason="not_selected",
        detail="selected rfdetr-1.9.0-py3-none-any.whl",
    )
    trackers = package.model_copy(
        update={
            "name": "trackers",
            "version": "2.5.0.post0",
            "extras": (),
            "wheel": package.wheel.model_copy(
                update={
                    "name": "trackers",
                    "version": "2.5.0.post0",
                    "filename": "trackers-2.5.0.post0-py3-none-any.whl",
                    "url": (
                        "https://files.pythonhosted.org/packages/22/22/"
                        + "2" * 60
                        + "/trackers-2.5.0.post0-py3-none-any.whl"
                    ),
                    "sha256": "7" * 64,
                }
            ),
            "project_metadata": package.project_metadata.model_copy(
                update={
                    "name": "trackers",
                    "version": "2.5.0.post0",
                    "summary": "Golden trackers",
                }
            ),
        }
    )
    audit = models_module.DependencyAuditResult(
        target=snapshot,
        packages=(package, trackers),
        diagnostics=(diagnostic,),
    )
    access = models_module.MetadataAccess(
        url="https://pypi.org/simple/rfdetr/",
        content_type="application/vnd.pypi.simple.v1+json",
        bytes=20,
        sha256="2" * 64,
    )
    runtime = models_module.ToolRuntime(
        python="3.12.13",
        packaging="25.0",
        pydantic="2.11.0",
        pyyaml="6.0.2",
        pip="25.1",
        pip_vendored_resolvelib="1.1.0",
    )
    sources = (
        models_module.SourceHash(path="scripts/algorithm/dependency_audit.py", sha256="3" * 64),
        models_module.SourceHash(
            path="src/billiards_ai/lab/dependency_audit/models.py", sha256="4" * 64
        ),
    )
    return packet_module.build_approval_packet(
        request,
        audit,
        (access,),
        code_commit="5" * 40,
        code_sources=sources,
        config_sha256="6" * 64,
        tool_runtime=runtime,
    )


def test_final_fix_03_conflicting_licence_evidence_creates_a_packet_blocker() -> None:
    base = _golden_packet()
    package = base.packages[0]
    conflicted = package.model_copy(
        update={
            "license": models_module.LicenseDecision(
                expression="MIT",
                evidence=(
                    "license_expression: MIT",
                    "classifier: License :: OSI Approved :: Apache Software License",
                ),
                status="unknown",
                reason="conflicting_evidence",
            )
        }
    )
    packet = packet_module.build_approval_packet(
        base.request,
        models_module.DependencyAuditResult(
            target=base.target,
            packages=(conflicted, *base.packages[1:]),
            diagnostics=base.exclusions,
        ),
        base.metadata_accesses,
        code_commit=base.code_commit,
        code_sources=base.code_sources,
        config_sha256=base.config_sha256,
        tool_runtime=base.tool_runtime,
    )

    assert packet.license_preflight == "BLOCKED"
    assert [item.model_dump() for item in packet.blockers] == [
        {
            "category": "license",
            "package": "rfdetr==1.9.0",
            "filename": None,
            "reason": "unknown_license",
            "detail": "conflicting_evidence",
        }
    ]


def test_final_fix_02_parser_exclusions_flow_through_resolution_json_and_markdown() -> None:
    class DiagnosticClient:
        def simple(self, name: str) -> bytes:
            if name == "trackers":
                return _simple_payload(
                    name,
                    [_file_record("trackers-2.5.0.post0-py3-none-any.whl")],
                )
            return _simple_payload(
                name,
                [
                    _file_record("rfdetr-1.9.0.tar.gz"),
                    _file_record("rfdetr-1.8.0-py3-none-any.whl"),
                    _file_record("rfdetr-1.9.0-py3-none-any.whl"),
                ],
            )

        def version(self, name: str, version: str) -> bytes:
            return _version_payload(name, version, [])

    request = _request()
    audit = resolver_module.resolve_dependency_audit(request, DiagnosticClient())
    base = _golden_packet()
    packet = packet_module.build_approval_packet(
        request,
        audit,
        base.metadata_accesses,
        code_commit=base.code_commit,
        code_sources=base.code_sources,
        config_sha256=base.config_sha256,
        tool_runtime=base.tool_runtime,
    )
    document = json.loads(packet_module.canonical_packet_json(packet))
    markdown = packet_module._packet_markdown(packet).decode("utf-8")

    assert [(item["filename"], item["reason"]) for item in document["exclusions"]] == [
        ("rfdetr-1.8.0-py3-none-any.whl", "not_selected"),
        ("rfdetr-1.9.0.tar.gz", "source_distribution"),
    ]
    assert "rfdetr-1.8.0-py3-none-any.whl" in markdown
    assert "rfdetr-1.9.0.tar.gz" in markdown


def test_final_fix_06_golden_packet_contains_the_complete_typed_contract() -> None:
    packet = _golden_packet()
    document = packet.model_dump(mode="json")

    assert set(document) == {
        "access_hosts",
        "blockers",
        "code_commit",
        "code_sources",
        "config_sha256",
        "exclusions",
        "license_preflight",
        "metadata_accesses",
        "packages",
        "packet_sha256",
        "request",
        "schema_version",
        "side_effects",
        "size",
        "target",
        "tool_runtime",
    }
    assert document["schema_version"] == "2.0"
    assert document["target"]["wheel_tags"] == ["py3-none-any"]
    assert document["code_sources"] == [
        {"path": "scripts/algorithm/dependency_audit.py", "sha256": "3" * 64},
        {
            "path": "src/billiards_ai/lab/dependency_audit/models.py",
            "sha256": "4" * 64,
        },
    ]
    assert document["tool_runtime"] == {
        "python": "3.12.13",
        "packaging": "25.0",
        "pydantic": "2.11.0",
        "pyyaml": "6.0.2",
        "pip": "25.1",
        "pip_vendored_resolvelib": "1.1.0",
    }
    assert document["packages"][0]["parents"] == ["<root>"]
    assert document["packages"][0]["project_metadata"] == {
        "name": "rfdetr",
        "version": "1.9.0",
        "summary": "Golden rfdetr",
        "requires_python": ">=3.12",
        "requires_dist": [],
    }
    assert document["packages"][0]["wheel"] == {
        "name": "rfdetr",
        "version": "1.9.0",
        "filename": "rfdetr-1.9.0-py3-none-any.whl",
        "url": (
            "https://files.pythonhosted.org/packages/11/11/"
            + "1" * 60
            + "/rfdetr-1.9.0-py3-none-any.whl"
        ),
        "tags": ["py3-none-any"],
        "bytes": 100,
        "sha256": "1" * 64,
        "requires_python": ">=3.12",
        "upload_time": "2026-08-01T09:00:00Z",
        "is_pure_python": True,
    }
    assert document["access_hosts"] == ["pypi.org"]
    assert document["license_preflight"] == "PASS"
    assert document["exclusions"] == [
        {
            "package": "rfdetr",
            "filename": "rfdetr-1.8.0-py3-none-any.whl",
            "category": "excluded",
            "reason": "not_selected",
            "detail": "selected rfdetr-1.9.0-py3-none-any.whl",
        }
    ]
    assert document["blockers"] == []
    assert document["size"] == {
        "archive_bytes": 200,
        "metadata_bytes": 20,
        "existing_payload_bytes": 10,
        "estimated_installed_bytes": 800,
        "estimated_temporary_cache_peak_bytes": 220,
        "required_free_bytes": 1236,
    }
    assert document["side_effects"] == {
        "archive_downloads": 0,
        "installs": 0,
        "package_imports": 0,
        "weights_loaded": 0,
    }
    unhashed = dict(document)
    unhashed.pop("packet_sha256")
    expected_payload = (
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert document["packet_sha256"] == hashlib.sha256(expected_payload).hexdigest()


def test_final_fix_06_packet_is_deterministic_when_typed_evidence_is_reordered() -> None:
    packet = _golden_packet()
    reversed_packet = packet_module.build_approval_packet(
        packet.request,
        models_module.DependencyAuditResult(
            target=packet.target,
            packages=tuple(reversed(packet.packages)),
            diagnostics=tuple(reversed(packet.exclusions)),
        ),
        tuple(reversed(packet.metadata_accesses)),
        code_commit=packet.code_commit,
        code_sources=tuple(reversed(packet.code_sources)),
        config_sha256=packet.config_sha256,
        tool_runtime=packet.tool_runtime,
    )

    assert packet_module.canonical_packet_json(reversed_packet) == (
        packet_module.canonical_packet_json(packet)
    )


def test_final_fix_06_complete_packet_rejects_missing_project_metadata() -> None:
    package_document = _golden_packet().packages[0].model_dump(mode="json")
    package_document.pop("project_metadata")

    with pytest.raises(ValidationError, match="project_metadata"):
        models_module.ResolvedPackage.model_validate(package_document)


def test_final_fix_06_complete_packet_rejects_an_empty_source_hash_set() -> None:
    packet_document = _golden_packet().model_dump(mode="json")
    packet_document["code_sources"] = []

    with pytest.raises(ValidationError, match="code_sources"):
        models_module.DependencyApprovalPacket.model_validate(packet_document)


def test_final_fix_07_markdown_renders_the_complete_approval_contract() -> None:
    markdown = packet_module._packet_markdown(_golden_packet()).decode("utf-8")

    for literal in (
        "## Target and ordered wheel tags",
        "&quot;py3-none-any&quot;",
        "## Root requirements",
        "&quot;rfdetr==1.9.0&quot;",
        "## Code, config, and runtime provenance",
        "src/billiards_ai/lab/dependency_audit/models.py",
        "## Access hosts",
        "&quot;pypi.org&quot;",
        "## Packages",
        "Extras: &quot;feature&quot;",
        "Parents: &quot;&lt;root&gt;&quot;",
        "rfdetr-1.9.0-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/11/11/",
        "2026-08-01T09:00:00Z",
        "Golden rfdetr",
        "license_expression: MIT",
        "## Exact and estimated size accounting",
        "Estimated temporary/cache peak bytes: 220",
        "## Candidate exclusions",
        "rfdetr-1.8.0-py3-none-any.whl",
        "## Blockers",
        "## Licence preflight",
        "PASS",
        "## Zero side effects",
        "Archive downloads: 0",
    ):
        assert literal in markdown


@pytest.mark.parametrize(
    "url_factory",
    [
        lambda filename: f"http://files.pythonhosted.org/packages/{filename}",
        lambda filename: f"https://user@files.pythonhosted.org/packages/{filename}",
        lambda filename: f"https://files.pythonhosted.org:443/packages/{filename}",
        lambda filename: f"https://files.pythonhosted.org/packages/{filename}?download=1",
        lambda filename: f"https://files.pythonhosted.org/packages/{filename}#fragment",
        lambda filename: f"https://pypi.org/packages/{filename}",
        lambda filename: f"https://files.pythonhosted.org/simple/{filename}",
        lambda _filename: "https://files.pythonhosted.org/packages/other.whl",
        lambda filename: f"https://files.pythonhosted.org/packages%2fescape/{filename}",
    ],
)
def test_final_fix_08_rejects_unapproved_archive_urls_with_structured_evidence(
    url_factory: Callable[[str], str],
) -> None:
    filename = "demo-1.0.0-py3-none-any.whl"
    record = _file_record(filename, url=url_factory(filename))

    with pytest.raises(index_module.WheelIndexError) as captured:
        index_module.parse_simple_candidates(_simple_payload("demo", [record]), _target())

    assert captured.value.diagnostics[-1].reason == "invalid_archive_url"
    assert captured.value.diagnostics[-1].filename == filename


def test_final_fix_08_validated_archive_url_is_data_only_and_never_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_archive_request(*_args, **_kwargs):
        raise AssertionError("archive URL was requested")

    monkeypatch.setattr(urllib.request, "urlopen", reject_archive_request)
    filename = "demo-1.0.0-py3-none-any.whl"
    candidates = index_module.parse_simple_candidates(
        _simple_payload("demo", [_file_record(filename)]), _target()
    )

    assert candidates[0].url == (
        f"https://files.pythonhosted.org/packages/aa/aa/{'a' * 60}/{filename}"
    )
