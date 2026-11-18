from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.deployment.prepare_display_phone_deployment import (
    DeploymentPreparationInputs,
    prepare_complete_deployment,
)

COMMIT = "8f5e9f6a0e2c7b725a5f8f1d4eeac35e5a8bd874"


def _replay(root: Path) -> Path:
    root.mkdir()
    content = b"presentation"
    (root / "presentation.mp4").write_bytes(content)
    manifest = {
        "schema_version": "1.0",
        "kind": "field_route_file_replay",
        "artifact_id": "replay-integration",
        "pipeline_version": "field-route-file-replay/2",
        "source_mainline1_artifact_id": "a" * 64,
        "events_sha256": "b" * 64,
        "shot_windows_sha256": "c" * 64,
        "inventory": {
            "presentation.mp4": {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        },
    }
    for name, payload in {
        "summary.json": {"schema_version": "1.0", "status": "PASS", "shot_count": 1},
        "timeline.json": {
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
    }.items():
        data = json.dumps(payload).encode()
        (root / name).write_bytes(data)
        manifest["inventory"][name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_invalid_stage1_input_is_rejected_before_cloud_changes(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest_path = inputs.artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pipeline_version"] = "field-route-file-replay/1"
    manifest_path.write_text(json.dumps(manifest))
    runner = FakeRunner()
    with pytest.raises(ValueError, match="v2"):
        prepare_complete_deployment(inputs, process_runner=runner, git_runner=runner)
    assert runner.operations == ["verify"]
    assert not inputs.output_root.exists()


def _inputs(tmp_path: Path) -> DeploymentPreparationInputs:
    repository = tmp_path / "repository"
    (repository / "apps/miniprogram/scripts").mkdir(parents=True)
    (repository / "apps/miniprogram/scripts/verify-development.js").write_text("// verify")
    (repository / "apps/miniprogram/scripts/provision-table-code.js").write_text("// provision")
    (repository / "apps/miniprogram/cloudbaserc.json").write_text("{}")
    (repository / "apps/miniprogram/storage").mkdir(parents=True)
    (repository / "apps/miniprogram/storage/security-rules.json").write_text('{"acl":"ADMINONLY"}')
    return DeploymentPreparationInputs(
        repository_root=repository,
        artifact_root=_replay(tmp_path / "replay"),
        output_root=tmp_path / "deployments",
        environment_id="cloud1-test",
        sync_base_url="https://sync.example.test",
        device_id="raytheon-01",
        credential_target="billiardsAI/development/raytheon-01",
        version="2026.08.27-integration.1",
        environment="development",
        state_path=tmp_path / "state" / "raytheon-01.json",
    )


class FakeRunner:
    def __init__(self, *, fail_sync_deploy: bool = False) -> None:
        self.operations: list[str] = []
        self.operation_cwds: dict[str, Path] = {}
        self.provision_output: Path | None = None
        self.provision_arguments: list[str] | None = None
        self.fail_sync_deploy = fail_sync_deploy

    def __call__(
        self, command: list[str] | tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        if arguments[:2] == ["node", "apps/miniprogram/scripts/verify-development.js"]:
            self.operations.append("verify")
            return subprocess.CompletedProcess(arguments, 0, "PASS\n", "")
        if "fn" in arguments and "deploy" in arguments:
            function_name = arguments[arguments.index("deploy") + 1]
            self.operations.append(f"deploy:{function_name}")
            self.operation_cwds[f"deploy:{function_name}"] = cwd
            if function_name == "sync-api" and self.fail_sync_deploy:
                return subprocess.CompletedProcess(arguments, 1, "", "cloud failure")
            return subprocess.CompletedProcess(arguments, 0, "{}", "")
        if "storage" in arguments and "rules" in arguments and "update" in arguments:
            assert arguments[arguments.index("--acl") + 1] == "ADMINONLY"
            assert "--rule" not in arguments
            self.operations.append("storage:rules")
            self.operation_cwds["storage:rules"] = cwd
            return subprocess.CompletedProcess(arguments, 0, "{}", "")
        if arguments[:2] == ["node", "apps/miniprogram/scripts/provision-table-code.js"]:
            self.operations.append("provision")
            self.provision_arguments = arguments
            output = Path(arguments[arguments.index("--output") + 1])
            self.provision_output = output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"\x89PNG\r\n\x1a\npermanent")
            payload = {
                "status": "PASS",
                "image_path": str(output),
                "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "code_kind": "permanent",
                "reused": False,
            }
            return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")
        if arguments == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, f"{COMMIT}\n", "")
        if arguments == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[:3] == ["git", "archive", "--format=zip"]:
            self.operations.append("package")
            output = Path(arguments[arguments.index("--output") + 1])
            with ZipFile(output, "w") as archive:
                archive.writestr("src/billiards_ai/__init__.py", "")
                archive.writestr(
                    "scripts/algorithm/windows/install-field-deployment.ps1", "# bootstrap\n"
                )
            return subprocess.CompletedProcess(arguments, 0, "", "")
        raise AssertionError(f"unexpected command in {cwd}: {arguments}")


def test_complete_preparation_orders_cloud_work_before_atomic_bundle_publication(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    inputs = _inputs(tmp_path)

    result = prepare_complete_deployment(inputs, process_runner=runner, git_runner=runner)

    assert runner.operations == [
        "verify",
        "deploy:client-api",
        "deploy:sync-api",
        "deploy:media-retention",
        "storage:rules",
        "provision",
        "package",
    ]
    expected_cloudbase_cwd = inputs.repository_root / "apps/miniprogram"
    assert runner.operation_cwds == {
        "deploy:client-api": expected_cloudbase_cwd,
        "deploy:sync-api": expected_cloudbase_cwd,
        "deploy:media-retention": expected_cloudbase_cwd,
        "storage:rules": expected_cloudbase_cwd,
    }
    assert runner.provision_output == inputs.state_path.parent / "table-join.png"
    assert runner.provision_arguments is not None
    assert "--permanent-table-code" in runner.provision_arguments
    assert runner.provision_output.is_file()
    assert result.bundle.is_dir()
    assert result.raytheon_command.count("\n") == 0
    serialized = json.dumps(result.as_public_dict(), sort_keys=True)
    assert "table_code" not in serialized
    assert "authorization" not in serialized


def test_complete_preparation_does_not_publish_a_bundle_after_cloud_failure(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(fail_sync_deploy=True)
    inputs = _inputs(tmp_path)

    with pytest.raises(RuntimeError, match="CLOUDBASE_DEPLOY_FAILED"):
        prepare_complete_deployment(inputs, process_runner=runner, git_runner=runner)

    assert runner.operations == [
        "verify",
        "deploy:client-api",
        "deploy:sync-api",
    ]
    assert not inputs.output_root.exists()


def test_check_only_validates_inputs_without_cloud_mutation_or_file_writes(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    inputs = replace(_inputs(tmp_path), check_only=True)

    result = prepare_complete_deployment(inputs, process_runner=runner, git_runner=runner)

    assert result.status == "READY"
    assert runner.operations == ["verify"]
    assert not inputs.output_root.exists()


def test_check_only_rejects_an_insecure_gateway(tmp_path: Path) -> None:
    runner = FakeRunner()
    inputs = replace(_inputs(tmp_path), check_only=True, sync_base_url="http://sync.example.test")

    with pytest.raises(ValueError, match="HTTPS sync gateway"):
        prepare_complete_deployment(inputs, process_runner=runner, git_runner=runner)
