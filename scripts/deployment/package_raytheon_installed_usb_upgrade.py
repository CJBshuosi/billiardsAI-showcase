#!/usr/bin/env python3
"""Build the single official RER U3 commercial-pilot field-service bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

PACKET_NAME = "billiardsAI-raytheon-commercial-pilot-20260907"
RELEASE_VERSION = "2026.09.07-commercial-pilot.23"
TABLE_DEVICE_ID = "raytheon-01"
DEVICE_LOCATOR = r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000"
DEVICE_FINGERPRINT = "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93"
APPROVED_NATIVE_HELPER_SHA256 = "760839bde41fc5f16c4d73e14c24c35e67dfdb5aefdd7e141112134c6aa7ba7c"
DEVELOPMENT_ENVIRONMENT_ID = "cloud1-public-example"
DETECTOR_ROOT = (
    "bootstrap-training-runs/"
    "2759358fe07f948a3c056a18f2b592e0b0ebad76364159f0a774c4eca70ca186/"
    "rfdetr-seg-small-bootstrap-short-r1/"
    "967612dd02c72a6af9149b6790820c4e998068df4536d3b006b917c774ac4a51"
)
IDENTITY_ROOT = (
    "identity-group-training-runs/2098055fcba31010d88cda072e5ec33b46f125e91c8ea6512673cb2070349b97"
)
DINOV2_ROOT = "dinov2-small-ed25f3a"
DETECTOR_MANIFEST_SHA256 = "355acf72c8e698e03a1f90265537a2e78e4cd883a1df93e8be8e40376e60934b"
DETECTOR_CHECKPOINT_SHA256 = "d9cd5efe6c6deb8b7762e08c0f720cb264a2f41bce9174ddc77154a56f434541"
IDENTITY_MANIFEST_SHA256 = "233a9e63829eceb674dbbe67741d4b4b9afffe84264831f6167e87934c685443"
IDENTITY_HEAD_SHA256 = "2737c03cf7418da8fd308f44c58460261b4801791ce1d0808c0a589eadd73d03"
IDENTITY_METRICS_SHA256 = "c9a86ad22a56a4351e9889bdb3375c2c3fa36b58c9ca632ffb184aa21fc182f6"
DINOV2_MANIFEST_SHA256 = "875d184852d921f3a9cbee54f18771640a511bd71d6f47829dc5d6ecf30cf21c"
DINOV2_MODEL_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
DINOV2_CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
DINOV2_PREPROCESSOR_SHA256 = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"

# This is the product runtime allowlist.  It intentionally excludes diagnostic,
# packaging, replay-launcher and temporary sidecar trees.  The broader shared
# model package is retained because its public __init__ is imported by the live
# runtime and therefore forms part of the installed dependency graph.
RELEASE_TREE_ALLOWLIST = (
    "src/billiards_ai/edge/route_assistance",
    "src/billiards_ai/lab/events",
    "src/billiards_ai/lab/field_assistance",
    "src/billiards_ai/lab/planning",
    "src/billiards_ai/lab/shot_local",
    "src/billiards_ai/lab/vision",
    "src/billiards_ai/shared",
)
RELEASE_FILE_ALLOWLIST = (
    # Required by both field entrypoints, including clean installed USB imports.
    "src/billiards_ai/lab/field_assistance/cloud_runtime.py",
    "scripts/algorithm/serve_mainline1_usb_field.py",
    "scripts/algorithm/run_mainline1_product_realtime.py",
    "scripts/algorithm/run_mainline1_shot_worker.py",
    "scripts/algorithm/retire_existing_field_session.py",
    "scripts/algorithm/windows/install-field-route-service.ps1",
    "scripts/algorithm/windows/serve-field-route-shell.ps1",
    "scripts/algorithm/windows/start-field-route-kiosk.ps1",
    "src/billiards_ai/__init__.py",
    "src/billiards_ai/edge/__init__.py",
    "src/billiards_ai/lab/__init__.py",
    "src/billiards_ai/lab/adapters/__init__.py",
    "src/billiards_ai/lab/adapters/pooltool_physics.py",
    "src/billiards_ai/lab/guidance.py",
    "src/billiards_ai/lab/guidance_render.py",
    "src/billiards_ai/lab/contracts.py",
    "src/billiards_ai/lab/dataset.py",
    "src/billiards_ai/lab/mainline1_calibration.py",
    "src/billiards_ai/lab/mainline1_calibration_gate.py",
    "src/billiards_ai/lab/mainline1_calibration_registry.py",
    "src/billiards_ai/lab/mainline1_capture.py",
    "src/billiards_ai/lab/mainline1_capture_hub.py",
    "src/billiards_ai/lab/mainline1_field_runtime.py",
    "src/billiards_ai/lab/mainline1_frame_source.py",
    "src/billiards_ai/lab/mainline1_live_publisher.py",
    "src/billiards_ai/lab/mainline1_live_window.py",
    "src/billiards_ai/lab/mainline1_native_capture.py",
    "src/billiards_ai/lab/mainline1_product_realtime_runtime.py",
    "src/billiards_ai/lab/mainline1_realtime_anchor.py",
    "src/billiards_ai/lab/mainline1_realtime_capture.py",
    "src/billiards_ai/lab/mainline1_realtime_diagnostics.py",
    "src/billiards_ai/lab/mainline1_recording_media.py",
    "src/billiards_ai/lab/mainline1_recording_processing.py",
    "src/billiards_ai/lab/mainline1_realtime_refinement.py",
    "src/billiards_ai/lab/mainline1_realtime_ring.py",
    "src/billiards_ai/lab/mainline1_realtime_tracker.py",
    "src/billiards_ai/lab/mainline1_runtime.py",
    "src/billiards_ai/lab/mainline1_runtime_support.py",
    "src/billiards_ai/lab/mainline1_identity_backends.py",
    "src/billiards_ai/lab/mainline1_identity_preprocessing.py",
    "src/billiards_ai/lab/mainline1_session_analysis.py",
    "src/billiards_ai/lab/mainline1_session_audit.py",
    "src/billiards_ai/lab/mainline1_session_recording.py",
    "src/billiards_ai/lab/mainline1_shot_jobs.py",
    "src/billiards_ai/lab/mainline1_shot_local_config.py",
    "src/billiards_ai/lab/mainline1_shot_local_video.py",
    "src/billiards_ai/lab/mainline1_shot_product.py",
    "src/billiards_ai/lab/mainline1_shot_worker.py",
    "src/billiards_ai/lab/mainline1_table_validation.py",
    "src/billiards_ai/lab/mainline1_usb_capture.py",
    "src/billiards_ai/lab/training/__init__.py",
    "src/billiards_ai/lab/training/identity_crop_export.py",
    "src/billiards_ai/lab/training/identity_dinov2.py",
    "src/billiards_ai/lab/training/identity_group_backend.py",
    "src/billiards_ai/lab/training/identity_visual_evidence.py",
    "src/billiards_ai/lab/training/rfdetr_bootstrap_train.py",
    "src/billiards_ai/lab/training/rfdetr_train.py",
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required(project: Path, relative: str) -> bytes:
    path = project / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required upgrade source is missing or unsafe: {relative}")
    return path.read_bytes()


def _table_code_asset(
    image_path: Path,
    state_path: Path,
    *,
    allow_development_preview: bool,
) -> tuple[bytes, dict[str, str | bool]]:
    image = image_path.resolve()
    state = state_path.resolve()
    if image_path.is_symlink() or not image.is_file():
        raise ValueError("permanent table code image is missing or unsafe")
    if state_path.is_symlink() or not state.is_file():
        raise ValueError("permanent table code state is missing or unsafe")
    try:
        metadata = json.loads(state.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("permanent table code state is invalid") from exc
    code_kind = metadata.get("code_kind")
    allowed_kinds = {"permanent"}
    if allow_development_preview:
        allowed_kinds.add("development_preview")
    expected = {
        "environment_id": DEVELOPMENT_ENVIRONMENT_ID,
        "device_id": TABLE_DEVICE_ID,
    }
    if code_kind not in allowed_kinds or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("commercial package requires the permanent Raytheon table code")
    configured_image = metadata.get("image_path")
    if not isinstance(configured_image, str) or Path(configured_image).resolve() != image:
        raise ValueError("permanent table code state does not identify the supplied image")
    payload = image.read_bytes()
    if not (payload.startswith(b"\x89PNG\r\n\x1a\n") or payload.startswith(b"\xff\xd8")):
        raise ValueError("permanent table code image format is invalid")
    return payload, {
        **expected,
        "code_kind": code_kind,
        "field_test_only": code_kind != "permanent",
        "launch_gate_status": "PASS" if code_kind == "permanent" else "FAIL",
    }


def _candidate_from_directory(root: Path) -> dict[str, bytes]:
    candidate = root.resolve()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("capture candidate directory is missing or unsafe")
    files: dict[str, bytes] = {}
    for path in sorted(candidate.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"capture candidate contains a link: {path}")
        if path.is_file():
            files[path.relative_to(candidate).as_posix()] = path.read_bytes()
    return files


def _candidate_from_archive(path: Path) -> dict[str, bytes]:
    source = path.resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("capture candidate archive is missing or unsafe")
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        roots = {
            part
            for entry in entries
            for part in PurePosixPath(entry.filename).parts
            if part in {"candidate", "capture"}
        }
        selected_root = "candidate" if "candidate" in roots else "capture"
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            if PurePosixPath(entry.filename).is_absolute() or ".." in parts:
                raise ValueError("capture candidate archive contains an unsafe path")
            positions = [index for index, part in enumerate(parts) if part == selected_root]
            if not positions:
                continue
            if len(positions) != 1 or positions[0] == len(parts) - 1:
                raise ValueError("capture candidate archive is ambiguous")
            relative = PurePosixPath(*parts[positions[0] + 1 :]).as_posix()
            if relative in files:
                raise ValueError("capture candidate archive contains duplicate files")
            files[relative] = archive.read(entry)
    return files


def _validate_candidate(files: dict[str, bytes]) -> None:
    required = {"capture-setup.json", "calibrations/registry.json"}
    missing = required - files.keys()
    if missing:
        raise ValueError(f"capture candidate is incomplete: {sorted(missing)}")
    if not any(
        name.startswith("calibrations/") and name.endswith("/calibration.json") for name in files
    ):
        raise ValueError("capture candidate has no calibration artifact")


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return target.getvalue()


def _release_files(project: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for relative_root in RELEASE_TREE_ALLOWLIST:
        source = project / relative_root
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"release allowlist tree is missing or unsafe: {relative_root}")
        for path in sorted(source.rglob("*.py")):
            if path.is_symlink():
                raise ValueError(f"release source contains a link: {path}")
            files[path.relative_to(project).as_posix()] = path.read_bytes()
    for relative in RELEASE_FILE_ALLOWLIST:
        files[relative] = _required(project, relative)
    files["release-manifest.json"] = _canonical(
        {
            "schema_version": "1.0",
            "version": RELEASE_VERSION,
            "source_identity": "commercial_pilot_explicit_allowlist",
            "files": {
                name: {"bytes": len(payload), "sha256": _sha256(payload)}
                for name, payload in sorted(files.items())
            },
        }
    )
    return files


def build_packet(
    project_root: Path,
    output_root: Path,
    *,
    candidate_root: Path | None = None,
    candidate_archive: Path | None = None,
    table_code_image: Path,
    table_code_state: Path,
    allow_development_preview: bool = False,
) -> Path:
    """Create one official upgrade packet with a pinned permanent table code."""

    if (candidate_root is None) == (candidate_archive is None):
        raise ValueError("provide exactly one capture candidate source")
    project = project_root.resolve()
    if project.is_symlink() or not project.is_dir():
        raise ValueError("project root is missing or unsafe")
    candidate = (
        _candidate_from_directory(candidate_root)
        if candidate_root is not None
        else _candidate_from_archive(candidate_archive)
    )
    candidate = {
        name: payload
        for name, payload in candidate.items()
        if Path(name).name.lower() not in {"join-qr.png", "join-qr.jpg", "join-qr.jpeg"}
        and not name.lower().startswith("native/")
    }
    _validate_candidate(candidate)
    table_code_payload, table_code_metadata = _table_code_asset(
        table_code_image,
        table_code_state,
        allow_development_preview=allow_development_preview,
    )

    release_payload = _zip_bytes(_release_files(project))
    release_name = f"billiardsAI-field-release-{RELEASE_VERSION}.zip"
    files: dict[str, bytes] = {
        "install.ps1": _required(
            project, "scripts/algorithm/windows/activate-mainline1-usb-capture.ps1"
        ),
        "install-field-route-service.ps1": _required(
            project, "scripts/algorithm/windows/install-field-route-service.ps1"
        ),
        release_name: release_payload,
        "capture/shot-local.yaml": _required(project, "configs/algorithm/shot-local.yaml"),
        "join/table-code.png": table_code_payload,
    }
    for name, payload in candidate.items():
        files[f"capture/{name}"] = payload

    manifest = {
        "schema_version": "1.0",
        "kind": "commercial_pilot_field_service",
        "version": RELEASE_VERSION,
        "service": {
            "task_name": "billiardsAI-field-service",
            "kiosk_task_name": "billiardsAI-field-kiosk",
            "port": 8787,
            "runtime_root": r"%LOCALAPPDATA%\billiardsAI\field-runtime",
        },
        "release": {"path": release_name, "sha256": _sha256(release_payload)},
        "capture": {
            "approved_native_helper_sha256": APPROVED_NATIVE_HELPER_SHA256,
            "capture_setup": "capture/capture-setup.json",
            "calibration_root": "capture/calibrations",
            "shot_config": "capture/shot-local.yaml",
            "device_locator": DEVICE_LOCATOR,
            "device_fingerprint": DEVICE_FINGERPRINT,
            "inference_device": "cuda",
            "batch_size": 4,
        },
        "cloud": {
            "development_environment_id": DEVELOPMENT_ENVIRONMENT_ID,
            "production_writes": False,
        },
        "state_policy": {
            "on_version_upgrade": "archive_and_start_new_session",
            "before_archive": "retire_cloud_session_fail_closed",
            "archive_root": r"%LOCALAPPDATA%\billiardsAI\field-runtime\backups",
        },
        "join": {
            **table_code_metadata,
            "image_path": "join/table-code.png",
            "image_sha256": _sha256(table_code_payload),
        },
        "assets": {
            "detector": {
                "relative_root": DETECTOR_ROOT,
                "manifest_sha256": DETECTOR_MANIFEST_SHA256,
                "checkpoint_sha256": DETECTOR_CHECKPOINT_SHA256,
            },
            "identity": {
                "relative_root": IDENTITY_ROOT,
                "manifest_sha256": IDENTITY_MANIFEST_SHA256,
                "head_sha256": IDENTITY_HEAD_SHA256,
                "metrics_sha256": IDENTITY_METRICS_SHA256,
                "gate_status": "PASS",
                "model_generation": "v4",
                "approval_basis": "commercial_candidate_owner_approved",
                "benchmark_gate_status": "FAIL",
            },
            "dinov2": {
                "relative_root": DINOV2_ROOT,
                "manifest_sha256": DINOV2_MANIFEST_SHA256,
                "model_sha256": DINOV2_MODEL_SHA256,
                "config_sha256": DINOV2_CONFIG_SHA256,
                "preprocessor_sha256": DINOV2_PREPROCESSOR_SHA256,
            },
        },
        "files": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in sorted(files.items())
        },
    }
    files["installed-usb-upgrade-manifest.json"] = _canonical(manifest)
    destination = output_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    packet = destination / f"{PACKET_NAME}.zip"
    if packet.exists():
        raise ValueError(f"upgrade packet already exists: {packet}")
    prefixed = {f"{PACKET_NAME}/{name}": payload for name, payload in files.items()}
    packet.write_bytes(_zip_bytes(prefixed))
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--table-code-image", type=Path, required=True)
    parser.add_argument("--table-code-state", type=Path, required=True)
    parser.add_argument("--allow-development-preview-table-code", action="store_true")
    args = parser.parse_args()
    packet = build_packet(
        args.project_root,
        args.output_root,
        candidate_root=args.candidate,
        candidate_archive=args.candidate_archive,
        table_code_image=args.table_code_image,
        table_code_state=args.table_code_state,
        allow_development_preview=args.allow_development_preview_table_code,
    )
    print(packet)
    print(_sha256(packet.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
