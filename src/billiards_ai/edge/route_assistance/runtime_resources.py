"""Immutable local runtime metadata; no filesystem access or execution authority."""

from __future__ import annotations

import hashlib
import json
import ntpath
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Self

from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
    validate_managed_artifact_id,
    validate_managed_rule_profile_id,
    validate_managed_scope_id,
    validate_managed_utc_timestamp,
    validate_managed_uuid,
)

MAX_RESOURCE_METADATA_BYTES = 1024 * 1024
RESOURCE_PATH_NAMES = frozenset(
    (
        "artifact_root",
        "jobs_root",
        "ring_root",
        "capture_setup",
        "calibration_root",
        "config",
        "detector_training_run",
        "identity_training_run",
        "dinov2_assets",
    )
)
_DINO_NAMES = frozenset(("model.safetensors", "config.json", "preprocessor_config.json"))


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid resource text")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid resource UTF-8") from exc
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid resource SHA256")
    return value


def _path(value: object) -> PurePosixPath | PureWindowsPath:
    text = _text(value)
    path = PurePosixPath(text) if text.startswith("/") else PureWindowsPath(text)
    normal = posixpath.normpath(text) if isinstance(path, PurePosixPath) else ntpath.normpath(text)
    if not path.is_absolute() or normal != text or ".." in path.parts:
        raise ValueError("resource path must be canonical and absolute")
    return path


def _canonical(value: object) -> str:
    try:
        result = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(result.encode("utf-8")) > MAX_RESOURCE_METADATA_BYTES:
            raise ValueError("resource metadata exceeds 1 MiB")
    except (TypeError, UnicodeError) as exc:
        raise ValueError("invalid resource metadata") from exc
    return result


def runtime_inventory_sha256(inventory: Mapping[str, object]) -> str:
    """Hash compact sorted UTF-8 inventory JSON: {name: {bytes, sha256}}."""
    return hashlib.sha256(_canonical(dict(inventory)).encode("utf-8")).hexdigest()


def managed_live_artifact_id(canonical_root_text: str) -> str:
    _path(canonical_root_text)
    return "usb-field-" + hashlib.sha256(canonical_root_text.encode("utf-8")).hexdigest()


class _Payload:
    @classmethod
    def from_payload(cls, payload: object) -> Self:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid resource payload fields")
        return cls(**payload)

    def as_payload(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def canonical_json(self) -> str:
        return _canonical(self.as_payload())


@dataclass(frozen=True, slots=True)
class RuntimeResourceFile(_Payload):
    role: str
    path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        _text(self.role)
        _path(self.path)
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("invalid resource file size")
        _sha(self.sha256)
        _ = self.canonical_json


@dataclass(frozen=True, slots=True)
class ManagedRuntimeResourceSpec(_Payload):
    schema_version: str
    contract: str
    artifact_id: str
    rule_profile_id: str | None
    paths: Mapping[str, str]
    files: tuple[RuntimeResourceFile, ...]
    capture_setup_id: str
    setup_fingerprint: str
    calibration_artifact_id: str
    calibration_artifact_path: str
    calibration_revision: int
    calibration_manifest_sha256: str
    model_package_id: str
    model_manifest_sha256: str
    identity_training_package_id: str
    identity_training_package_sha256: str
    dinov2_version: str
    dinov2_inventory_sha256: str
    device: str
    batch_size: int
    fps: int

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.contract != "managed_runtime_resources_v1":
            raise ValueError("invalid resource contract")
        validate_managed_artifact_id(self.artifact_id)
        validate_managed_rule_profile_id(self.rule_profile_id)
        if not isinstance(self.paths, Mapping) or set(self.paths) != RESOURCE_PATH_NAMES:
            raise ValueError("invalid resource paths")
        paths = dict(self.paths)
        for value in paths.values():
            _path(value)
        object.__setattr__(self, "paths", MappingProxyType(paths))
        if not isinstance(self.files, (tuple, list)) or not all(
            isinstance(f, RuntimeResourceFile) for f in self.files
        ):
            raise ValueError("invalid resource files")
        files = tuple(sorted(self.files, key=lambda f: f.role))
        if len({f.role for f in files}) != len(files):
            raise ValueError("duplicate resource roles")
        object.__setattr__(self, "files", files)
        identities = {}
        for file in files:
            path = _path(file.path)
            identity = (file.byte_size, file.sha256)
            if path in identities and identities[path] != identity:
                raise ValueError("contradictory resource file identities")
            identities[path] = identity
        for name in (
            "setup_fingerprint",
            "calibration_artifact_id",
            "calibration_manifest_sha256",
            "model_package_id",
            "model_manifest_sha256",
            "identity_training_package_id",
            "identity_training_package_sha256",
            "dinov2_inventory_sha256",
        ):
            _sha(getattr(self, name))
        _text(self.capture_setup_id)
        _text(self.dinov2_version)
        if (
            self.device not in ("cpu", "cuda")
            or type(self.batch_size) is not int
            or not 1 <= self.batch_size <= 256
            or type(self.fps) is not int
            or self.fps != 60
            or type(self.calibration_revision) is not int
            or self.calibration_revision < 1
        ):
            raise ValueError("invalid resource runtime configuration")
        cal = _path(self.calibration_artifact_path)
        if (
            self.artifact_id != managed_live_artifact_id(paths["artifact_root"])
            or cal.parent != _path(paths["calibration_root"])
            or cal.name != self.calibration_artifact_id
            or _path(paths["detector_training_run"]).name != self.model_package_id
            or _path(paths["identity_training_run"]).name != self.identity_training_package_id
        ):
            raise ValueError("contradictory resource identity")
        expected = {
            "capture_setup": _path(paths["capture_setup"]),
            "shot_config": _path(paths["config"]),
            "calibration_manifest": cal / "manifest.json",
            "detector_manifest": _path(paths["detector_training_run"]) / "training-manifest.json",
            "detector_checkpoint": _path(paths["detector_training_run"]) / "checkpoint_best.pth",
            "identity_manifest": _path(paths["identity_training_run"]) / "manifest.json",
            "identity_metrics": _path(paths["identity_training_run"]) / "metrics.json",
            "identity_onnx": _path(paths["identity_training_run"]) / "group-head.onnx",
            **{"dinov2:" + n: _path(paths["dinov2_assets"]) / n for n in _DINO_NAMES},
        }
        for file in files:
            if file.role.startswith("calibration_inventory:"):
                relative = file.role.removeprefix("calibration_inventory:")
                part = PurePosixPath(relative)
                if (
                    part.is_absolute()
                    or ".." in part.parts
                    or str(part) != relative
                    or "\\" in relative
                    or relative in ("", ".", "manifest.json")
                ):
                    raise ValueError("invalid calibration inventory role")
                expected[file.role] = cal.joinpath(*part.parts)
        if (
            not any(f.role == "calibration_inventory:calibration.json" for f in files)
            or {f.role for f in files} != set(expected)
            or any(_path(f.path) != expected[f.role] for f in files)
        ):
            raise ValueError("contradictory resource file paths or inventory")
        by_role = {f.role: f for f in files}
        for role, digest in (
            ("calibration_manifest", self.calibration_manifest_sha256),
            ("detector_manifest", self.model_manifest_sha256),
            ("identity_manifest", self.identity_training_package_sha256),
        ):
            if by_role[role].sha256 != digest:
                raise ValueError("contradictory resource manifest SHA")
        inventory = {
            n: {"bytes": by_role["dinov2:" + n].byte_size, "sha256": by_role["dinov2:" + n].sha256}
            for n in _DINO_NAMES
        }
        if runtime_inventory_sha256(inventory) != self.dinov2_inventory_sha256:
            raise ValueError("contradictory DINOv2 inventory SHA")
        _ = self.canonical_json

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid resource specification fields")
        _canonical(dict(payload))
        if not isinstance(payload["files"], (list, tuple)):
            raise ValueError("invalid resource file payload")
        return cls(
            **{
                **payload,
                "files": tuple(RuntimeResourceFile.from_payload(f) for f in payload["files"]),
            }
        )

    def as_payload(self) -> dict[str, object]:
        return {
            **super(ManagedRuntimeResourceSpec, self).as_payload(),
            "paths": dict(self.paths),
            "files": [f.as_payload() for f in self.files],
        }

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def config_sha256(self) -> str:
        return next(f.sha256 for f in self.files if f.role == "shot_config")


def _scope_payload(scope: ManagedSessionScope) -> dict[str, str]:
    if not isinstance(scope, ManagedSessionScope):
        raise ValueError("invalid resource scope")
    return {f.name: getattr(scope, f.name) for f in fields(scope)}


def _decode_scope(value: object) -> ManagedSessionScope:
    if not isinstance(value, Mapping) or set(value) != {
        f.name for f in fields(ManagedSessionScope)
    }:
        raise ValueError("invalid resource scope fields")
    return ManagedSessionScope(**value)


@dataclass(frozen=True, slots=True)
class ManagedResourceSpecRecord(_Payload):
    scope: ManagedSessionScope
    open_request_id: str
    spec: ManagedRuntimeResourceSpec
    spec_sha256: str
    created_at_utc: str

    def __post_init__(self) -> None:
        _scope_payload(self.scope)
        validate_managed_uuid(self.open_request_id, name="open request ID")
        validate_managed_utc_timestamp(self.created_at_utc, name="created timestamp")
        _sha(self.spec_sha256)
        if (
            not isinstance(self.spec, ManagedRuntimeResourceSpec)
            or self.spec_sha256 != self.spec.spec_sha256
        ):
            raise ValueError("invalid resource specification digest")
        _ = self.canonical_json

    def as_payload(self) -> dict[str, object]:
        return {
            **super(ManagedResourceSpecRecord, self).as_payload(),
            "scope": _scope_payload(self.scope),
            "spec": self.spec.as_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid resource record fields")
        return cls(
            **{
                **payload,
                "scope": _decode_scope(payload["scope"]),
                "spec": ManagedRuntimeResourceSpec.from_payload(payload["spec"]),
            }
        )


@dataclass(frozen=True, slots=True)
class ManagedCaptureEpochRecord(_Payload):
    scope: ManagedSessionScope
    open_request_id: str
    session_id: str
    birth_id: str
    source_epoch_id: str
    spec_sha256: str
    created_at_utc: str

    def __post_init__(self) -> None:
        _scope_payload(self.scope)
        validate_managed_uuid(self.open_request_id, name="open request ID")
        validate_managed_scope_id(self.source_epoch_id, name="source epoch ID")
        validate_managed_utc_timestamp(self.created_at_utc, name="created timestamp")
        _sha(self.spec_sha256)
        ManagedSessionBinding(
            "1.0",
            self.session_id,
            self.birth_id,
            "managed_session_birth_v1",
            self.scope.table_id,
            self.scope.device_id,
            "open",
        )
        _ = self.canonical_json

    def as_payload(self) -> dict[str, object]:
        return {
            **super(ManagedCaptureEpochRecord, self).as_payload(),
            "scope": _scope_payload(self.scope),
        }

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise ValueError("invalid resource epoch fields")
        return cls(**{**payload, "scope": _decode_scope(payload["scope"])})
