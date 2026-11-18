from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from billiards_ai.deployment.field_bundle import DeploymentInputs, prepare_field_bundle

COMMIT = "8f5e9f6a0e2c7b725a5f8f1d4eeac35e5a8bd874"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_replay(root: Path) -> Path:
    root.mkdir(parents=True)
    files = {
        "presentation.mp4": b"development replay video",
        "timeline.json": b'{"schema_version":"1.0","frames":[]}',
        "summary.json": b'{"schema_version":"1.0","shot_count":1}',
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    manifest = {
        "schema_version": "1.0",
        "kind": "field_route_file_replay",
        "artifact_id": "replay-001",
        "inventory": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _inputs(tmp_path: Path) -> DeploymentInputs:
    repository = tmp_path / "repository"
    repository.mkdir()
    # These represent local/untracked data beside the committed tree.
    (repository / "apps/miniprogram").mkdir(parents=True)
    (repository / "apps/miniprogram/project.config.json").write_text(
        '{"appid":"wx-secret-local-app-id"}', encoding="utf-8"
    )
    (repository / "raw-table-code.txt").write_text("raw-table-code-secret", encoding="utf-8")
    (repository / ".venv").mkdir()
    (repository / "node_modules").mkdir()
    (repository / "field.sqlite3").write_bytes(b"private sqlite")
    (repository / "__pycache__").mkdir()
    qr = tmp_path / "table-join.png"
    qr.write_bytes(b"\x89PNG\r\n\x1a\nprivate-qr")
    return DeploymentInputs(
        repository_root=repository,
        artifact_root=_write_replay(tmp_path / "replay"),
        output_root=tmp_path / "output",
        environment_id="cloud1-test",
        sync_base_url="https://sync.example.test",
        device_id="raytheon-01",
        credential_target="billiardsAI/development/raytheon-01",
        table_join_png=qr,
        version="2026.08.27-test.1",
    )


def _fake_git_archive(
    command: list[str] | tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    arguments = list(command)
    if arguments == ["git", "rev-parse", "HEAD"]:
        return subprocess.CompletedProcess(arguments, 0, f"{COMMIT}\n", "")
    if arguments == ["git", "status", "--porcelain", "--untracked-files=no"]:
        return subprocess.CompletedProcess(arguments, 0, "", "")
    if arguments[:3] == ["git", "archive", "--format=zip"]:
        output = Path(arguments[arguments.index("--output") + 1])
        with ZipFile(output, "w") as archive:
            archive.writestr("src/billiards_ai/__init__.py", '"""tracked"""\n')
            archive.writestr(
                "scripts/algorithm/windows/install-field-route-service.ps1", "# tracked\n"
            )
            archive.writestr(
                "scripts/algorithm/windows/install-field-deployment.ps1", "# bootstrap\n"
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")
    raise AssertionError(f"unexpected git command in {cwd}: {arguments}")


def test_bundle_uses_only_committed_files_and_excludes_local_secrets(tmp_path: Path) -> None:
    result = prepare_field_bundle(_inputs(tmp_path), git_runner=_fake_git_archive)
    receipt = json.loads((result / "deployment-receipt.json").read_text(encoding="utf-8"))
    archive_path = result / receipt["release"]["path"]

    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        release_manifest = json.loads(archive.read("release-manifest.json"))
        archive_bytes = b"".join(archive.read(name) for name in sorted(names))

    assert release_manifest == {
        "commit_sha": COMMIT,
        "schema_version": "1.0",
        "version": "2026.08.27-test.1",
    }
    assert "apps/miniprogram/project.config.json" not in names
    assert "raw-table-code.txt" not in names
    assert all("node_modules" not in name and ".venv" not in name for name in names)
    assert b"wx-secret-local-app-id" not in archive_bytes
    assert b"raw-table-code-secret" not in archive_bytes

    serialized = json.dumps(receipt, sort_keys=True)
    assert "table_code" not in serialized
    assert "hmac" not in serialized.lower()
    assert receipt["release"]["sha256"] == _sha256(archive_path)
    assert (result / "install-field-deployment.ps1").read_text() == "# bootstrap\n"
    assert "\n" not in receipt["install"]["raytheon_command"]
    assert "-BundleRoot" in receipt["install"]["raytheon_command"]
    assert "ExpectedSha256" not in receipt["install"]["raytheon_command"]


def test_bundle_copies_the_complete_verified_replay_and_qr(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = prepare_field_bundle(inputs, git_runner=_fake_git_archive)
    receipt = json.loads((result / "deployment-receipt.json").read_text(encoding="utf-8"))
    replay = result / receipt["replay"]["path"]

    assert {path.name for path in replay.iterdir()} == {
        "manifest.json",
        "presentation.mp4",
        "summary.json",
        "timeline.json",
    }
    assert receipt["replay"]["artifact_id"] == "replay-001"
    assert receipt["replay"]["kind"] == "field_route_file_replay"
    assert receipt["replay"]["mode"] == "development_replay"
    qr = result / receipt["table_join_qr"]["path"]
    assert qr.read_bytes().startswith(b"\x89PNG")
    assert receipt["table_join_qr"]["sha256"] == _sha256(qr)


def test_bundle_accepts_the_jpeg_qr_emitted_by_wechat_devtools(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.table_join_png.write_bytes(b"\xff\xd8\xff\xe0wechat-preview-qr")

    result = prepare_field_bundle(inputs, git_runner=_fake_git_archive)
    receipt = json.loads((result / "deployment-receipt.json").read_text(encoding="utf-8"))

    assert (result / receipt["table_join_qr"]["path"]).read_bytes().startswith(b"\xff\xd8")


@pytest.mark.parametrize(
    "url",
    [
        "http://sync.example.test",
        "https://user:pass@sync.example.test",
        "https://sync.example.test/path?token=secret",
        "https://sync.example.test/path#fragment",
    ],
)
def test_bundle_rejects_unsafe_sync_gateway_urls(tmp_path: Path, url: str) -> None:
    inputs = _inputs(tmp_path)
    inputs = DeploymentInputs(
        repository_root=inputs.repository_root,
        artifact_root=inputs.artifact_root,
        output_root=inputs.output_root,
        environment_id=inputs.environment_id,
        sync_base_url=url,
        device_id=inputs.device_id,
        credential_target=inputs.credential_target,
        table_join_png=inputs.table_join_png,
        version=inputs.version,
    )

    with pytest.raises(ValueError, match="HTTPS sync gateway"):
        prepare_field_bundle(inputs, git_runner=_fake_git_archive)


def test_bundle_rejects_tampered_replay_before_publication(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs.artifact_root / "timeline.json").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="inventory mismatch"):
        prepare_field_bundle(inputs, git_runner=_fake_git_archive)

    assert not inputs.output_root.exists()


def test_bundle_allows_only_the_two_known_local_miniprogram_configs_to_be_dirty(
    tmp_path: Path,
) -> None:
    def runner(command: list[str] | tuple[str, ...], cwd: Path):
        if list(command) == ["git", "status", "--porcelain", "--untracked-files=no"]:
            output = (
                " M apps/miniprogram/project.config.json\n"
                " M apps/miniprogram/project.private.config.example.json\n"
            )
            return subprocess.CompletedProcess(list(command), 0, output, "")
        return _fake_git_archive(command, cwd)

    result = prepare_field_bundle(_inputs(tmp_path), git_runner=runner)

    assert (result / "deployment-receipt.json").is_file()


def test_bundle_rejects_any_other_tracked_modification(tmp_path: Path) -> None:
    def runner(command: list[str] | tuple[str, ...], cwd: Path):
        if list(command) == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(list(command), 0, " M src/private-change.py\n", "")
        return _fake_git_archive(command, cwd)

    with pytest.raises(ValueError, match="tracked modifications"):
        prepare_field_bundle(_inputs(tmp_path), git_runner=runner)


@pytest.mark.parametrize("count", [0, 1])
def test_stage1_bundle_rejects_legacy_replay_even_when_summary_claims_one_shot(
    tmp_path: Path,
    count: int,
) -> None:
    inputs = replace(_inputs(tmp_path), expected_shot_count=1)
    _rewrite_replay_json(
        inputs.artifact_root,
        "summary.json",
        {
            "schema_version": "1.0",
            "status": "PASS",
            "shot_count": count,
        },
    )
    with pytest.raises(ValueError, match="v2"):
        prepare_field_bundle(inputs, git_runner=_fake_git_archive)
    assert not inputs.output_root.exists()


def _rewrite_replay_json(root: Path, name: str, payload: dict) -> None:
    data = json.dumps(payload).encode()
    (root / name).write_bytes(data)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["inventory"][name] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _stage1_inputs(tmp_path: Path) -> DeploymentInputs:
    inputs = replace(_inputs(tmp_path), expected_shot_count=1)
    root = inputs.artifact_root
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.update(
        pipeline_version="field-route-file-replay/2",
        source_mainline1_artifact_id="a" * 64,
        events_sha256="b" * 64,
        shot_windows_sha256="c" * 64,
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    _rewrite_replay_json(
        root,
        "summary.json",
        {
            "schema_version": "1.0",
            "status": "PASS",
            "shot_count": 1,
        },
    )
    _rewrite_replay_json(
        root,
        "timeline.json",
        {
            "schema_version": "1.0",
            "frames": [
                {"frame_id": 0, "timestamp_ms": 0, "shot_sequence": 0, "effects": []},
                {
                    "frame_id": 1,
                    "timestamp_ms": 100,
                    "shot_sequence": 0,
                    "effects": ["SHOT_STARTED"],
                },
                {
                    "frame_id": 2,
                    "timestamp_ms": 200,
                    "shot_sequence": 1,
                    "effects": ["SHOT_SETTLED"],
                },
            ],
        },
    )
    return inputs


def test_stage1_bundle_checks_recordable_timeline_not_just_summary(tmp_path: Path) -> None:
    inputs = _stage1_inputs(tmp_path)
    timeline = json.loads((inputs.artifact_root / "timeline.json").read_text())
    timeline["frames"].pop()
    _rewrite_replay_json(inputs.artifact_root, "timeline.json", timeline)
    with pytest.raises(ValueError, match="recordable"):
        prepare_field_bundle(inputs, git_runner=_fake_git_archive)
    assert not inputs.output_root.exists()


def test_stage1_bundle_records_verified_acceptance_expectation(tmp_path: Path) -> None:
    result = prepare_field_bundle(_stage1_inputs(tmp_path), git_runner=_fake_git_archive)
    receipt = json.loads((result / "deployment-receipt.json").read_text())
    assert receipt["replay"]["expected_shot_count"] == 1
    assert receipt["replay"]["pipeline_version"] == "field-route-file-replay/2"
