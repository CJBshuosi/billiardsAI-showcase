"""Versioned activation registry for immutable Mainline 1 calibrations."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from billiards_ai.lab.mainline1_calibration import Mainline1CalibrationArtifact
from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint
from billiards_ai.lab.mainline1_runtime import _json_object
from billiards_ai.lab.mainline1_runtime import _sha256 as _runtime_sha256


class CalibrationRegistryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    setup_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_setup_id: str
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_path: str
    revision: int = Field(gt=0)
    status: Literal["active", "superseded"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return _runtime_sha256(path)


def _load_payload(root: Path) -> dict[str, object]:
    registry = root / "registry.json"
    if not registry.exists():
        return {"schema_version": "1.0", "records": []}
    if registry.is_symlink():
        raise ValueError("calibration registry is unsafe")
    payload = _json_object(registry, label="calibration registry")
    if payload.get("schema_version") != "1.0" or not isinstance(payload.get("records"), list):
        raise ValueError("calibration registry schema is invalid")
    return payload


def _validated_manifest(root: Path, artifact_path: Path) -> dict[str, object]:
    resolved_root = root.resolve()
    if any(p.is_symlink() for p in (artifact_path, *artifact_path.parents)):
        raise ValueError("calibration artifact is unsafe")
    resolved = artifact_path.resolve()
    if resolved.parent != resolved_root or not resolved.is_dir():
        raise ValueError("calibration artifact must be an immutable child of its registry")
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("calibration manifest is missing or unsafe")
    manifest = _json_object(manifest_path, label="calibration manifest")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("calibration manifest inventory is missing")
    for name, expected in inventory.items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or ".project-archive" in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name
            or name in ("", ".", "manifest.json")
        ):
            raise ValueError("calibration inventory path is unsafe")
        candidate = resolved / name
        if (
            any(p.is_symlink() for p in (candidate, *candidate.parents))
            or not candidate.is_file()
            or not isinstance(expected, dict)
        ):
            raise ValueError("calibration inventory contains an unsafe file")
        if expected.get("bytes") != candidate.stat().st_size or expected.get("sha256") != _sha256(
            candidate
        ):
            raise ValueError("calibration inventory validation failed")
    return manifest


def register_calibration(
    root: Path, artifact: Mainline1CalibrationArtifact, setup: CaptureSetup
) -> CalibrationRegistryRecord:
    resolved_root = root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    manifest = _validated_manifest(resolved_root, artifact.path)
    fingerprint = capture_setup_fingerprint(setup)
    if manifest.get("capture_setup_fingerprint") != fingerprint:
        raise ValueError("calibration artifact does not match the capture setup")
    payload = _load_payload(resolved_root)
    records = [CalibrationRegistryRecord.model_validate(item) for item in payload["records"]]
    revision = 1 + max(
        (item.revision for item in records if item.setup_fingerprint == fingerprint), default=0
    )
    records = [
        item.model_copy(update={"status": "superseded"})
        if item.setup_fingerprint == fingerprint and item.status == "active"
        else item
        for item in records
    ]
    record = CalibrationRegistryRecord(
        setup_fingerprint=fingerprint,
        capture_setup_id=setup.capture_setup_id,
        artifact_id=artifact.artifact_id,
        artifact_path=artifact.path.name,
        revision=revision,
        status="active",
        manifest_sha256=_sha256(artifact.path / "manifest.json"),
        source_video_sha256=str(manifest["source_video_sha256"]),
    )
    records.append(record)
    temporary = resolved_root / f".registry-{uuid4().hex}.json"
    temporary.write_bytes(
        _canonical_bytes(
            {"schema_version": "1.0", "records": [item.model_dump() for item in records]}
        )
    )
    os.replace(temporary, resolved_root / "registry.json")
    return record


def resolve_active_calibration(
    root: Path, setup_fingerprint: str
) -> CalibrationRegistryRecord | None:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        return None
    payload = _load_payload(resolved_root)
    records = [CalibrationRegistryRecord.model_validate(item) for item in payload["records"]]
    active = [
        item
        for item in records
        if item.setup_fingerprint == setup_fingerprint and item.status == "active"
    ]
    if not active:
        return None
    if len(active) != 1:
        raise ValueError("calibration registry has multiple active revisions")
    record = active[0]
    manifest = _validated_manifest(resolved_root, resolved_root / record.artifact_path)
    if _sha256(resolved_root / record.artifact_path / "manifest.json") != record.manifest_sha256:
        raise ValueError("calibration manifest changed after registration")
    if manifest.get("artifact_id") != record.artifact_id:
        raise ValueError("calibration registry artifact identity mismatch")
    return record


def resolve_calibration_record(
    root: Path, *, setup_fingerprint: str, artifact_id: str, revision: int
) -> CalibrationRegistryRecord | None:
    """Validate one retained revision, including superseded records, without writes."""
    if (
        type(revision) is not int
        or revision < 1
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in (setup_fingerprint, artifact_id)
        )
    ):
        raise ValueError("invalid exact calibration identity")
    if ".project-archive" in root.parts:
        raise ValueError("private archive cannot be a runtime resource")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("calibration registry root is unsafe")
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        return None
    records = [
        CalibrationRegistryRecord.model_validate(item)
        for item in _load_payload(resolved_root)["records"]
    ]
    matches = [
        r
        for r in records
        if r.setup_fingerprint == setup_fingerprint
        and r.artifact_id == artifact_id
        and r.revision == revision
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("calibration registry has duplicate exact revisions")
    record = matches[0]
    if record.artifact_path != artifact_id:
        raise ValueError("calibration registry artifact path mismatch")
    artifact = resolved_root / record.artifact_path
    manifest = _validated_manifest(resolved_root, artifact)
    if (
        _sha256(artifact / "manifest.json") != record.manifest_sha256
        or manifest.get("artifact_id") != artifact_id
        or manifest.get("capture_setup_fingerprint") != setup_fingerprint
        or manifest.get("capture_setup_id") != record.capture_setup_id
        or manifest.get("source_video_sha256") != record.source_video_sha256
    ):
        raise ValueError("calibration registry exact artifact identity mismatch")
    return record
