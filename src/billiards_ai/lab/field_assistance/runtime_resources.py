"""Inspect immutable local runtime inputs without constructing any runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from billiards_ai.edge.route_assistance.runtime_resources import (
    RESOURCE_PATH_NAMES,
    ManagedRuntimeResourceSpec,
    RuntimeResourceFile,
    managed_live_artifact_id,
    runtime_inventory_sha256,
)
from billiards_ai.lab.mainline1_calibration_registry import (
    CalibrationRegistryRecord,
    resolve_active_calibration,
    resolve_calibration_record,
)
from billiards_ai.lab.mainline1_capture import capture_setup_fingerprint, load_capture_setup
from billiards_ai.lab.mainline1_identity_backends import (
    Dinov2IdentityFeatureBackend,
    verify_dinov2_assets,
)
from billiards_ai.lab.mainline1_runtime import (
    _file_identity,
    _json_object,
    _safe_stat,
    _sha256,
    inspect_development_group_assets,
    inspect_development_vision_package,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config


def _native_path(value: str | Path, *, directory: bool, required: bool = True) -> Path:
    path = Path(value)
    # Reject archive components before any filesystem access, including symlink resolution.
    if ".project-archive" in path.parts:
        raise ValueError("private archive cannot be a runtime resource")
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("runtime resource path is unsafe")
    resolved = path.resolve()
    if ".project-archive" in resolved.parts:
        raise ValueError("private archive cannot be a runtime resource")
    if directory and not required:
        for candidate in (resolved, *resolved.parents):
            if candidate.exists() and not candidate.is_dir():
                raise ValueError("runtime resource output path requires directory ancestors")
    if required and not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError("runtime resource is missing")
    if not directory and required:
        _safe_stat(resolved)
    return resolved


def _resource_file(role: str, path: Path) -> RuntimeResourceFile:
    path = _native_path(path, directory=False)
    before = _safe_stat(path)
    digest = _sha256(path)
    if _file_identity(_safe_stat(path)) != _file_identity(before):
        raise ValueError("runtime resource changed during inspection")
    return RuntimeResourceFile(role, str(path), before.st_size, digest)


def _inspect(
    args: object,
    *,
    rule_profile_id: str | None,
    selected_calibration: CalibrationRegistryRecord | None = None,
) -> ManagedRuntimeResourceSpec:
    paths = {}
    for name in RESOURCE_PATH_NAMES:
        # Output roots may not exist until runtime preparation. Never create them here.
        paths[name] = _native_path(
            getattr(args, name),
            directory=name not in ("capture_setup", "config"),
            required=name not in ("artifact_root", "jobs_root", "ring_root"),
        )
    files = [
        _resource_file("capture_setup", paths["capture_setup"]),
        _resource_file("shot_config", paths["config"]),
    ]
    setup = load_capture_setup(paths["capture_setup"])
    load_mainline1_shot_local_config(paths["config"])
    fingerprint = capture_setup_fingerprint(setup)
    record = selected_calibration or resolve_active_calibration(
        paths["calibration_root"], fingerprint
    )
    if record is None:
        raise ValueError("capture setup has no retained immutable calibration")
    exact = resolve_calibration_record(
        paths["calibration_root"],
        setup_fingerprint=fingerprint,
        artifact_id=record.artifact_id,
        revision=record.revision,
    )
    if exact is None or exact.capture_setup_id != setup.capture_setup_id:
        raise ValueError("capture setup does not match selected calibration")
    calibration = paths["calibration_root"] / exact.artifact_path
    manifest_path = calibration / "manifest.json"
    files.append(_resource_file("calibration_manifest", manifest_path))
    manifest = _json_object(manifest_path, label="calibration manifest")
    for name in sorted(manifest["inventory"]):
        files.append(_resource_file("calibration_inventory:" + name, calibration / name))
    # These helpers are shared by the actual development loaders, not release package readers.
    detector = inspect_development_vision_package(paths["detector_training_run"])
    group = inspect_development_group_assets(paths["identity_training_run"])
    files.extend(
        (
            _resource_file("detector_manifest", detector.manifest_path),
            _resource_file("detector_checkpoint", detector.checkpoint),
            _resource_file("identity_manifest", group.manifest_path),
            _resource_file("identity_metrics", group.metrics_path),
            _resource_file("identity_onnx", group.onnx_path),
        )
    )
    dino = verify_dinov2_assets(paths["dinov2_assets"])
    dino_files = [_resource_file("dinov2:" + name, path) for name, path in sorted(dino.items())]
    files.extend(dino_files)
    verify_dinov2_assets(paths["dinov2_assets"])
    inventory = {
        f.role.removeprefix("dinov2:"): {"bytes": f.byte_size, "sha256": f.sha256}
        for f in dino_files
    }
    # Bind parsed policies/manifest identities to the bytes read for this snapshot.
    by_role = {f.role: f for f in files}
    for name, expected in manifest["inventory"].items():
        file = by_role["calibration_inventory:" + name]
        if file.byte_size != expected["bytes"] or file.sha256 != expected["sha256"]:
            raise ValueError("calibration inventory changed during inspection")
    group_inventory = _json_object(group.manifest_path, label="group manifest")["inventory"]
    if (
        by_role["detector_checkpoint"].sha256 != detector.manifest.artifacts[0].sha256
        or by_role["identity_metrics"].sha256 != group_inventory["metrics.json"]["sha256"]
        or by_role["identity_onnx"].sha256 != group_inventory["group-head.onnx"]["sha256"]
    ):
        raise ValueError("runtime package changed during inspection")
    spec = ManagedRuntimeResourceSpec(
        schema_version="1.0",
        contract="managed_runtime_resources_v1",
        artifact_id=managed_live_artifact_id(str(paths["artifact_root"])),
        rule_profile_id=rule_profile_id,
        paths={k: str(v) for k, v in paths.items()},
        files=tuple(files),
        capture_setup_id=setup.capture_setup_id,
        setup_fingerprint=fingerprint,
        calibration_artifact_id=exact.artifact_id,
        calibration_artifact_path=str(calibration),
        calibration_revision=exact.revision,
        calibration_manifest_sha256=exact.manifest_sha256,
        model_package_id=detector.identity.package_id,
        model_manifest_sha256=detector.identity.manifest_sha256,
        identity_training_package_id=group.root.name,
        identity_training_package_sha256=group.manifest_sha256,
        dinov2_version=Dinov2IdentityFeatureBackend.version,
        dinov2_inventory_sha256=runtime_inventory_sha256(inventory),
        device=args.device,
        batch_size=args.batch_size,
        fps=getattr(args, "fps", 60),
    )
    for file in spec.files:
        if _resource_file(file.role, Path(file.path)) != file:
            raise ValueError("runtime resource changed during inspection")
    return spec


def inspect_managed_runtime_resources(
    args: object, *, rule_profile_id: str | None
) -> ManagedRuntimeResourceSpec:
    """Read only whitelisted runtime inputs; never persist the input namespace."""
    return _inspect(args, rule_profile_id=rule_profile_id)


def verify_managed_runtime_resources(
    spec: ManagedRuntimeResourceSpec, *, purpose: Literal["capture", "analysis", "cleanup"]
) -> None:
    """Verify original inputs, or only declared cleanup roots; perform no mutations."""
    if not isinstance(spec, ManagedRuntimeResourceSpec):
        raise ValueError("invalid managed runtime resource specification")
    if purpose not in ("capture", "analysis", "cleanup"):
        raise ValueError("invalid resource verification purpose")
    # Reject foreign-platform paths at the native boundary, without rewriting metadata.
    for value in spec.paths.values():
        if ".project-archive" in Path(value).parts:
            raise ValueError("private archive cannot be a runtime resource")
        if not Path(value).is_absolute():
            raise ValueError("runtime resource path is not native")
    if purpose == "cleanup":
        for name in ("artifact_root", "jobs_root", "ring_root"):
            path = _native_path(spec.paths[name], directory=True)
            if str(path) != spec.paths[name]:
                raise ValueError("runtime cleanup root changed")
        return
    record = resolve_calibration_record(
        Path(spec.paths["calibration_root"]),
        setup_fingerprint=spec.setup_fingerprint,
        artifact_id=spec.calibration_artifact_id,
        revision=spec.calibration_revision,
    )
    if record is None:
        raise ValueError("recorded calibration is unavailable")
    args = SimpleNamespace(
        **{k: Path(v) for k, v in spec.paths.items()},
        device=spec.device,
        batch_size=spec.batch_size,
        fps=spec.fps,
    )
    current = _inspect(args, rule_profile_id=spec.rule_profile_id, selected_calibration=record)
    if current != spec:
        raise ValueError("managed runtime resources changed")
