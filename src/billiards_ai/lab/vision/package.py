"""Content-addressed model packages for pluggable vision backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.vision.contracts import VisionPackageIdentity


class VisionPackageError(ValueError):
    """Raised when a model package cannot be verified exactly."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_package_id(payload: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "package_id"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


class VisionArtifact(FrozenContract):
    role: Literal["checkpoint", "onnx"]
    relative_path: NonEmptyStr
    byte_count: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."} or "\\" in value:
            raise ValueError("artifact path must be relative and confined")
        return value


class VisionPreprocess(FrozenContract):
    color_space: Literal["RGB", "BGR"]
    input_width_px: int = Field(gt=0)
    input_height_px: int = Field(gt=0)


class VisionPostprocess(FrozenContract):
    confidence_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    mask_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)


class VisionModelPackageManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    package_id: Sha256
    task: Literal["ball_instance_segmentation"]
    family: NonEmptyStr
    model_version: NonEmptyStr
    model_class: NonEmptyStr
    artifacts: tuple[VisionArtifact, ...] = Field(min_length=1)
    preprocess: VisionPreprocess
    postprocess: VisionPostprocess
    class_map: dict[str, int]
    runtime_requirements: tuple[NonEmptyStr, ...] = Field(min_length=1)
    training_run_sha256: Sha256
    code_sha256: Sha256
    base_weight_sha256: Sha256
    benchmark_artifact_ids: tuple[NonEmptyStr, ...] = ()
    supported_capabilities: tuple[NonEmptyStr, ...] = ()
    unsupported_capabilities: tuple[NonEmptyStr, ...] = ()
    asset_status_ref: NonEmptyStr

    @model_validator(mode="after")
    def package_shape_is_supported(self) -> Self:
        roles = [artifact.role for artifact in self.artifacts]
        if roles.count("onnx") > 1 or len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        if roles.count("checkpoint") != 1:
            raise ValueError("vision package requires exactly one checkpoint")
        if self.class_map != {"ball": 0}:
            raise ValueError("schema 1.0 class map must be exactly {'ball': 0}")
        if len(set(self.class_map.values())) != len(self.class_map):
            raise ValueError("class indices must be unique")
        return self


class VerifiedVisionPackage(FrozenContract):
    root: Path
    manifest_path: Path
    identity: VisionPackageIdentity
    manifest: VisionModelPackageManifest
    checkpoint: Path
    onnx: Path | None = None


def _artifact_path(root: Path, artifact: VisionArtifact) -> Path:
    candidate = root.joinpath(*PurePosixPath(artifact.relative_path).parts)
    current = root
    for part in PurePosixPath(artifact.relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise VisionPackageError("vision package artifact path contains a symlink")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise VisionPackageError("artifact path must be relative and confined") from error
    if not candidate.is_file():
        raise VisionPackageError("vision package artifact is missing")
    return candidate


def _verify_artifact(path: Path, artifact: VisionArtifact) -> None:
    payload = path.read_bytes()
    if len(payload) != artifact.byte_count:
        raise VisionPackageError("artifact byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise VisionPackageError("artifact sha256 mismatch")


def load_vision_package(package_root: Path) -> VerifiedVisionPackage:
    root = Path(package_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise VisionPackageError("vision package root is missing or is a symlink")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise VisionPackageError("vision package manifest is missing or is a symlink")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("manifest must be an object")
        if raw != _canonical_json_bytes(payload):
            raise ValueError("manifest is not canonical JSON")
        manifest = VisionModelPackageManifest.model_validate(payload)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise VisionPackageError(f"invalid manifest: {error}") from error
    if canonical_package_id(payload) != manifest.package_id:
        raise VisionPackageError("package identity mismatch")

    paths: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        path = _artifact_path(root, artifact)
        _verify_artifact(path, artifact)
        paths[artifact.role] = path

    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    return VerifiedVisionPackage(
        root=root,
        manifest_path=manifest_path,
        identity=VisionPackageIdentity(
            package_id=manifest.package_id,
            manifest_sha256=manifest_sha256,
        ),
        manifest=manifest,
        checkpoint=paths["checkpoint"],
        onnx=paths.get("onnx"),
    )
