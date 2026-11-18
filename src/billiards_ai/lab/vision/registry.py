"""Explicit local registry with atomic activation and auditable rollback."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import AwareDatetime, Field, ValidationError, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.vision.package import VerifiedVisionPackage, load_vision_package
from billiards_ai.lab.vision.promotion import VisionPromotionVerdict


class VisionRegistryError(RuntimeError):
    """Raised when model selection cannot be verified or changed safely."""


def _canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


class VisionRegistryEntry(FrozenContract):
    package_id: Sha256
    manifest_sha256: Sha256
    relative_path: NonEmptyStr

    @field_validator("relative_path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."} or "\\" in value:
            raise ValueError("registry package path must be relative and confined")
        return value


class VisionRegistry(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    entries: tuple[VisionRegistryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_are_unique(self) -> Self:
        package_ids = [entry.package_id for entry in self.entries]
        paths = [entry.relative_path for entry in self.entries]
        if len(package_ids) != len(set(package_ids)) or len(paths) != len(set(paths)):
            raise ValueError("registry package identities and paths must be unique")
        return self


class ActiveVisionModel(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    activation_id: Sha256
    package_id: Sha256
    manifest_sha256: Sha256
    verdict_sha256: Sha256
    activated_at: AwareDatetime
    previous_activation_id: Sha256 | None = None
    previous_package_id: Sha256 | None = None
    action: Literal["promote", "rollback"]


def _load_canonical(path: Path, model_type: type[Self], label: str) -> Self:
    if path.is_symlink() or not path.is_file():
        raise VisionRegistryError(f"{label} is missing or is a symlink")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
        if raw != _canonical_bytes(payload):
            raise ValueError("JSON is not canonical")
        return model_type.model_validate(payload)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise VisionRegistryError(f"{label} is invalid: {error}") from error


def load_registry(path: Path) -> VisionRegistry:
    return _load_canonical(Path(path), VisionRegistry, "vision registry")


def _entry_package(registry_path: Path, entry: VisionRegistryEntry) -> VerifiedVisionPackage:
    root = registry_path.parent.absolute()
    candidate = root.joinpath(*PurePosixPath(entry.relative_path).parts)
    current = root
    for part in PurePosixPath(entry.relative_path).parts:
        current /= part
        if current.is_symlink():
            raise VisionRegistryError("registry package path contains a symlink")
    package = load_vision_package(candidate)
    if (
        package.identity.package_id != entry.package_id
        or package.identity.manifest_sha256 != entry.manifest_sha256
    ):
        raise VisionRegistryError("registry entry does not match verified package identity")
    return package


def _find_entry(registry: VisionRegistry, package_id: str) -> VisionRegistryEntry:
    for entry in registry.entries:
        if entry.package_id == package_id:
            return entry
    raise VisionRegistryError("active or candidate package is not registered")


def load_active_model(registry_path: Path, active_path: Path) -> ActiveVisionModel:
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    active = _load_canonical(Path(active_path), ActiveVisionModel, "active vision model")
    entry = _find_entry(registry, active.package_id)
    package = _entry_package(registry_path, entry)
    if package.identity.manifest_sha256 != active.manifest_sha256:
        raise VisionRegistryError("active model manifest identity mismatch")
    return active


def resolve_active_package(
    registry_path: Path,
    active_path: Path,
) -> VerifiedVisionPackage:
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    active = load_active_model(registry_path, active_path)
    return _entry_package(registry_path, _find_entry(registry, active.package_id))


def _new_active(
    *,
    package: VerifiedVisionPackage,
    verdict_sha256: str,
    activated_at: str,
    previous: ActiveVisionModel,
    action: Literal["promote", "rollback"],
) -> ActiveVisionModel:
    unsigned = {
        "schema_version": "1.0",
        "package_id": package.identity.package_id,
        "manifest_sha256": package.identity.manifest_sha256,
        "verdict_sha256": verdict_sha256,
        "activated_at": activated_at,
        "previous_activation_id": previous.activation_id,
        "previous_package_id": previous.package_id,
        "action": action,
    }
    activation_id = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    return ActiveVisionModel(activation_id=activation_id, **unsigned)


def _publish_active(
    active: ActiveVisionModel,
    *,
    active_path: Path,
    history_root: Path,
) -> None:
    active_path = Path(active_path)
    history_root = Path(history_root)
    if active_path.is_symlink() or history_root.is_symlink():
        raise VisionRegistryError("active or history path must not be a symlink")
    history_root.mkdir(parents=True, exist_ok=True)
    history_path = history_root / f"{active.activation_id}.json"
    payload = _canonical_bytes(active.model_dump(mode="json"))
    try:
        descriptor = os.open(history_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise VisionRegistryError("activation history record already exists") from error
    temporary = active_path.with_name(f".{active_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _load_canonical(temporary, ActiveVisionModel, "temporary active vision model")
        os.replace(temporary, active_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def activate_candidate(
    *,
    registry_path: Path,
    active_path: Path,
    history_root: Path,
    verdict_path: Path,
    activated_at: str,
) -> ActiveVisionModel:
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    current = load_active_model(registry_path, active_path)
    verdict = _load_canonical(Path(verdict_path), VisionPromotionVerdict, "promotion verdict")
    if verdict.promotion_status != "passed":
        raise VisionRegistryError("activation requires a passed promotion verdict")
    if verdict.incumbent_package_id != current.package_id:
        raise VisionRegistryError("promotion verdict does not match the active incumbent")
    entry = _find_entry(registry, verdict.candidate_package_id)
    candidate = _entry_package(registry_path, entry)
    verdict_sha256 = hashlib.sha256(Path(verdict_path).read_bytes()).hexdigest()
    active = _new_active(
        package=candidate,
        verdict_sha256=verdict_sha256,
        activated_at=activated_at,
        previous=current,
        action="promote",
    )
    _publish_active(active, active_path=active_path, history_root=history_root)
    return active


def rollback_active(
    *,
    registry_path: Path,
    active_path: Path,
    history_root: Path,
    expected_activation_id: str,
    activated_at: str,
) -> ActiveVisionModel:
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    current = load_active_model(registry_path, active_path)
    if current.activation_id != expected_activation_id:
        raise VisionRegistryError("stale expected activation ID")
    if current.previous_package_id is None:
        raise VisionRegistryError("active model has no recorded previous package")
    entry = _find_entry(registry, current.previous_package_id)
    previous_package = _entry_package(registry_path, entry)
    rolled_back = _new_active(
        package=previous_package,
        verdict_sha256=current.verdict_sha256,
        activated_at=activated_at,
        previous=current,
        action="rollback",
    )
    _publish_active(rolled_back, active_path=active_path, history_root=history_root)
    return rolled_back
