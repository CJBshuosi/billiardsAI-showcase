"""Build a private, secret-free field deployment directory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

CommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CREDENTIAL_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,191}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_LOCAL_CONFIGS = {
    "apps/miniprogram/project.config.json",
    "apps/miniprogram/project.private.config.example.json",
}


@dataclass(frozen=True, slots=True)
class DeploymentInputs:
    repository_root: Path
    artifact_root: Path
    output_root: Path
    environment_id: str
    sync_base_url: str
    device_id: str
    credential_target: str
    table_join_png: Path
    version: str
    expected_shot_count: int | None = None


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_replay(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("field replay artifact is missing or unsafe")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("field replay manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "field_route_file_replay":
        raise ValueError("field replay manifest kind is invalid")
    if not isinstance(manifest.get("artifact_id"), str) or not manifest["artifact_id"]:
        raise ValueError("field replay artifact ID is invalid")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("field replay inventory is invalid")
    for name, expected in inventory.items():
        candidate = path / name if isinstance(name, str) else path
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, dict)
            or candidate.is_symlink()
            or not candidate.is_file()
            or expected.get("bytes") != candidate.stat().st_size
            or expected.get("sha256") != _sha256(candidate)
        ):
            raise ValueError(f"field replay inventory mismatch: {name}")
    return manifest


def _verify_expected_shots(path: Path, manifest: dict[str, Any], expected: int) -> None:
    """Check the existing recorder's boundaries, not the video's ability to play."""
    from billiards_ai.edge.route_assistance.recording import ShotTimelineIndex
    from billiards_ai.lab.field_assistance.live_state import load_replay_timeline

    if type(expected) is not int or expected < 1:
        raise ValueError("expected shot count must be a positive integer")
    if manifest.get("pipeline_version") != "field-route-file-replay/2":
        raise ValueError("acceptance requires the authoritative v2 replay")
    for key in ("source_mainline1_artifact_id", "events_sha256", "shot_windows_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise ValueError("acceptance replay is missing Mainline 1 lineage")
    if not {"summary.json", "timeline.json", "presentation.mp4"} <= manifest["inventory"].keys():
        raise ValueError("acceptance replay is missing inventoried record products")
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("schema_version") != "1.0"
            or summary.get("status") != "PASS"
            or type(summary.get("shot_count")) is not int
            or summary["shot_count"] != expected
        ):
            raise ValueError("acceptance replay shot_count does not match expectation")
        rows = json.loads((path / "timeline.json").read_text(encoding="utf-8"))["frames"]
        timeline = load_replay_timeline(
            path / "timeline.json", duration_ms=max(1, int(rows[-1]["timestamp_ms"]))
        )
        index = ShotTimelineIndex.from_timeline(timeline)
        if len(index.boundaries) != 2 * expected:
            raise ValueError("acceptance replay recordable shots do not match expectation")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("acceptance replay record products are invalid") from exc


def _run_git(
    runner: CommandRunner,
    repository: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    result = runner(["git", *arguments], repository)
    if result.returncode != 0:
        raise ValueError("committed release source could not be resolved")
    return result


def _validate_inputs(inputs: DeploymentInputs, *, require_qr: bool = True) -> dict[str, Any]:
    if not inputs.repository_root.is_dir() or inputs.repository_root.is_symlink():
        raise ValueError("repository root is missing or unsafe")
    for label, value in (
        ("environment ID", inputs.environment_id),
        ("device ID", inputs.device_id),
        ("version", inputs.version),
    ):
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"{label} is invalid")
    if not _CREDENTIAL_TARGET.fullmatch(inputs.credential_target):
        raise ValueError("credential target is invalid")

    gateway = urlsplit(inputs.sync_base_url)
    if (
        gateway.scheme != "https"
        or not gateway.hostname
        or gateway.username is not None
        or gateway.password is not None
        or gateway.query
        or gateway.fragment
    ):
        raise ValueError("HTTPS sync gateway is invalid")

    if require_qr:
        qr = inputs.table_join_png
        qr_bytes = qr.read_bytes() if qr.is_file() and not qr.is_symlink() else b""
        if (
            qr.is_symlink()
            or not qr.is_file()
            or not (qr_bytes.startswith(b"\x89PNG\r\n\x1a\n") or qr_bytes.startswith(b"\xff\xd8"))
        ):
            raise ValueError("table join QR is missing or invalid")
    manifest = _verify_replay(inputs.artifact_root)
    if inputs.expected_shot_count is not None:
        _verify_expected_shots(inputs.artifact_root, manifest, inputs.expected_shot_count)
    return manifest


def check_field_bundle_source(
    inputs: DeploymentInputs,
    *,
    git_runner: CommandRunner | None = None,
    require_qr: bool = True,
) -> tuple[dict[str, Any], str]:
    """Validate bundle inputs and committed source without writing files."""

    runner = git_runner or _default_git_runner
    replay_manifest = _validate_inputs(inputs, require_qr=require_qr)
    commit = _run_git(runner, inputs.repository_root, ["rev-parse", "HEAD"]).stdout.strip()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("committed release source returned an invalid commit")
    dirty = _run_git(
        runner,
        inputs.repository_root,
        ["status", "--porcelain", "--untracked-files=no"],
    ).stdout.rstrip("\r\n")
    dirty_lines = dirty.splitlines()
    modified_paths = {line[3:] for line in dirty_lines if len(line) >= 4}
    if len(modified_paths) != len(dirty_lines) or modified_paths - _ALLOWED_LOCAL_CONFIGS:
        raise ValueError("committed release source has tracked modifications")
    return replay_manifest, commit


def _copy_verified_replay(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    destination.mkdir(parents=True)
    shutil.copyfile(source / "manifest.json", destination / "manifest.json")
    for name in sorted(manifest["inventory"]):
        shutil.copyfile(source / name, destination / name)


def _inject_release_manifest(
    source_archive: Path,
    destination_archive: Path,
    release_manifest: dict[str, str],
) -> None:
    with (
        ZipFile(source_archive) as source,
        ZipFile(destination_archive, "w", compression=ZIP_DEFLATED) as destination,
    ):
        names: set[str] = set()
        for entry in source.infolist():
            path = Path(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or entry.filename == "release-manifest.json"
            ):
                raise ValueError("committed release archive contains an unsafe path")
            if entry.filename in names:
                raise ValueError("committed release archive contains duplicate paths")
            names.add(entry.filename)
            destination.writestr(entry, source.read(entry.filename))
        info = ZipInfo("release-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        destination.writestr(info, _canonical_bytes(release_manifest))


def _extract_bootstrap(release_archive: Path, destination: Path) -> None:
    name = "scripts/algorithm/windows/install-field-deployment.ps1"
    with ZipFile(release_archive) as archive:
        try:
            content = archive.read(name)
        except KeyError as exc:
            message = "committed release archive is missing the deployment bootstrap"
            raise ValueError(message) from exc
    destination.write_bytes(content)


def _default_git_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, check=False, capture_output=True, text=True)


def prepare_field_bundle(
    inputs: DeploymentInputs,
    *,
    git_runner: CommandRunner = _default_git_runner,
) -> Path:
    """Build and return a versioned private deployment directory."""

    replay_manifest, commit = check_field_bundle_source(inputs, git_runner=git_runner)

    inputs.output_root.mkdir(parents=True, exist_ok=True)
    destination = inputs.output_root / f"billiardsAI-field-deployment-{inputs.version}"
    if destination.exists():
        raise ValueError("deployment version already exists")
    staging = Path(tempfile.mkdtemp(prefix=".field-deployment-", dir=inputs.output_root))
    try:
        git_archive = staging / ".committed-release.zip"
        _run_git(
            git_runner,
            inputs.repository_root,
            ["archive", "--format=zip", "--output", str(git_archive), commit],
        )
        release_name = f"billiardsAI-field-release-{inputs.version}.zip"
        release_path = staging / release_name
        _inject_release_manifest(
            git_archive,
            release_path,
            {"schema_version": "1.0", "version": inputs.version, "commit_sha": commit},
        )
        git_archive.unlink()
        bootstrap_path = staging / "install-field-deployment.ps1"
        _extract_bootstrap(release_path, bootstrap_path)

        private = staging / "private"
        replay_path = private / "development-replay" / str(replay_manifest["artifact_id"])
        _copy_verified_replay(inputs.artifact_root, replay_path, replay_manifest)
        qr_path = private / "table-join.png"
        qr_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs.table_join_png, qr_path)

        raytheon_command = (
            "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
            '".\\install-field-deployment.ps1" -BundleRoot "."'
        )
        receipt = {
            "schema_version": "1.0",
            "version": inputs.version,
            "commit_sha": commit,
            "cloudbase_environment_id": inputs.environment_id,
            "sync_base_url": inputs.sync_base_url,
            "device_id": inputs.device_id,
            "credential_target": inputs.credential_target,
            "release": {"path": release_name, "sha256": _sha256(release_path)},
            "replay": {
                "path": replay_path.relative_to(staging).as_posix(),
                "manifest_sha256": _sha256(replay_path / "manifest.json"),
                "artifact_id": replay_manifest["artifact_id"],
                "kind": replay_manifest["kind"],
                "mode": "development_replay",
                "pipeline_version": replay_manifest.get("pipeline_version"),
                "expected_shot_count": inputs.expected_shot_count,
                "inventory": replay_manifest["inventory"],
            },
            "table_join_qr": {
                "path": qr_path.relative_to(staging).as_posix(),
                "sha256": _sha256(qr_path),
            },
            "install": {
                "bootstrap": bootstrap_path.relative_to(staging).as_posix(),
                "raytheon_command": raytheon_command,
            },
        }
        (staging / "deployment-receipt.json").write_bytes(_canonical_bytes(receipt))
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
