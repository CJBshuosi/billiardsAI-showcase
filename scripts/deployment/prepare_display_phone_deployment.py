#!/usr/bin/env python3
"""Prepare CloudBase and a private Raytheon field deployment directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from billiards_ai.deployment.field_bundle import (
    CommandRunner,
    DeploymentInputs,
    check_field_bundle_source,
    prepare_field_bundle,
)

ProcessRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class DeploymentPreparationInputs:
    repository_root: Path
    artifact_root: Path
    output_root: Path
    environment_id: str
    sync_base_url: str
    device_id: str
    credential_target: str
    version: str
    environment: str
    state_path: Path
    rotate_table_code: bool = False
    check_only: bool = False
    skip_cloud_deploy: bool = False
    existing_table_join_png: Path | None = None
    expected_shot_count: int = 1


@dataclass(frozen=True, slots=True)
class PreparationResult:
    status: str
    bundle: Path | None
    receipt: Path | None
    raytheon_command: str

    def as_public_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "bundle": None if self.bundle is None else str(self.bundle),
            "receipt": None if self.receipt is None else str(self.receipt),
            "raytheon_command": self.raytheon_command,
        }


def _default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, check=False, capture_output=True, text=True)


def _require_success(result: subprocess.CompletedProcess[str], error_code: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(error_code)


def _verify_development(
    inputs: DeploymentPreparationInputs,
    runner: ProcessRunner,
) -> None:
    command = [
        "node",
        "apps/miniprogram/scripts/verify-development.js",
        "--environment",
        "development",
    ]
    _require_success(runner(command, inputs.repository_root), "CLOUDBASE_VERIFY_FAILED")


def _deploy_cloud_functions(
    inputs: DeploymentPreparationInputs,
    runner: ProcessRunner,
) -> None:
    miniprogram_root = inputs.repository_root / "apps/miniprogram"
    config = miniprogram_root / "cloudbaserc.json"
    # All three handlers are packaged from one source unit so their canonical
    # shared retention policy is present in every CloudBase deployment bundle.
    for name in ("client-api", "sync-api", "media-retention"):
        command = [
            "tcb",
            "--env-id",
            inputs.environment_id,
            "--config-file",
            str(config),
            "--yes",
            "fn",
            "deploy",
            name,
            "--force",
            "--json",
        ]
        _require_success(runner(command, miniprogram_root), "CLOUDBASE_DEPLOY_FAILED")
    storage_rules = json.loads(
        (miniprogram_root / "storage/security-rules.json").read_text(encoding="utf-8")
    )
    command = [
        "tcb",
        "--env-id",
        inputs.environment_id,
        "--config-file",
        str(config),
        "--yes",
        "storage",
        "rules",
        "update",
        "--acl",
        str(storage_rules["acl"]),
        "--json",
    ]
    _require_success(runner(command, miniprogram_root), "CLOUDBASE_STORAGE_RULES_FAILED")


def _provision_qr(
    inputs: DeploymentPreparationInputs,
    runner: ProcessRunner,
    output: Path,
) -> Path:
    command = [
        "node",
        "apps/miniprogram/scripts/provision-table-code.js",
        "--environment",
        "development",
        "--environment-id",
        inputs.environment_id,
        "--device-id",
        inputs.device_id,
        "--output",
        str(output),
        "--state",
        str(inputs.state_path),
        "--permanent-table-code",
    ]
    if inputs.rotate_table_code:
        command.append("--rotate-table-code")
    result = runner(command, inputs.repository_root)
    _require_success(result, "TABLE_CODE_PROVISION_FAILED")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TABLE_CODE_PROVISION_FAILED") from exc
    if (
        payload.get("status") != "PASS"
        or payload.get("code_kind") != "permanent"
        or Path(payload.get("image_path", "")) != output
    ):
        raise RuntimeError("TABLE_CODE_PROVISION_FAILED")
    return output


def prepare_complete_deployment(
    inputs: DeploymentPreparationInputs,
    *,
    process_runner: ProcessRunner = _default_runner,
    git_runner: CommandRunner = _default_runner,
) -> PreparationResult:
    """Verify, deploy, provision, and atomically publish one private bundle."""

    if inputs.environment != "development":
        raise ValueError("deployment preparation requires environment=development")
    if inputs.skip_cloud_deploy and inputs.existing_table_join_png is None:
        raise ValueError("skip-cloud-deploy requires an existing table join PNG")
    _verify_development(inputs, process_runner)
    check_field_bundle_source(
        DeploymentInputs(
            repository_root=inputs.repository_root,
            artifact_root=inputs.artifact_root,
            output_root=inputs.output_root,
            environment_id=inputs.environment_id,
            sync_base_url=inputs.sync_base_url,
            device_id=inputs.device_id,
            credential_target=inputs.credential_target,
            table_join_png=inputs.state_path,
            version=inputs.version,
            expected_shot_count=inputs.expected_shot_count,
        ),
        git_runner=git_runner,
        require_qr=False,
    )
    if inputs.check_only:
        return PreparationResult("READY", None, None, "")

    if inputs.skip_cloud_deploy:
        table_join_png = inputs.existing_table_join_png
    else:
        _deploy_cloud_functions(inputs, process_runner)
        table_join_png = _provision_qr(
            inputs, process_runner, inputs.state_path.parent / "table-join.png"
        )
        return _package(inputs, table_join_png, git_runner)
    assert table_join_png is not None
    return _package(inputs, table_join_png, git_runner)


def _package(
    inputs: DeploymentPreparationInputs,
    table_join_png: Path,
    git_runner: CommandRunner,
) -> PreparationResult:
    try:
        bundle = prepare_field_bundle(
            DeploymentInputs(
                repository_root=inputs.repository_root,
                artifact_root=inputs.artifact_root,
                output_root=inputs.output_root,
                environment_id=inputs.environment_id,
                sync_base_url=inputs.sync_base_url,
                device_id=inputs.device_id,
                credential_target=inputs.credential_target,
                table_join_png=table_join_png,
                version=inputs.version,
                expected_shot_count=inputs.expected_shot_count,
            ),
            git_runner=git_runner,
        )
    except Exception as exc:
        raise RuntimeError("BUNDLE_BUILD_FAILED") from exc
    receipt_path = bundle / "deployment-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return PreparationResult(
        status="PASS",
        bundle=bundle,
        receipt=receipt_path,
        raytheon_command=str(receipt["install"]["raytheon_command"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--sync-base-url", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--credential-target", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--environment", choices=("development",), required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--rotate-table-code", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-cloud-deploy", action="store_true")
    parser.add_argument("--existing-table-join-png", type=Path)
    parser.add_argument("--expected-shot-count", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = prepare_complete_deployment(
        DeploymentPreparationInputs(
            repository_root=args.repository_root,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            environment_id=args.environment_id,
            sync_base_url=args.sync_base_url,
            device_id=args.device_id,
            credential_target=args.credential_target,
            version=args.version,
            environment=args.environment,
            state_path=args.state_path,
            rotate_table_code=args.rotate_table_code,
            check_only=args.check_only,
            skip_cloud_deploy=args.skip_cloud_deploy,
            existing_table_join_png=args.existing_table_join_png,
            expected_shot_count=args.expected_shot_count,
        )
    )
    print(json.dumps(result.as_public_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
