"""Artifact-first detector training orchestration with an injectable RF-DETR backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectorTrainingError(RuntimeError):
    """Training inputs or outputs violate the reproducibility contract."""


class SmokeTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    model_class: Literal[
        "RFDETRSegSmall",
        "RFDETRSegMedium",
        "RFDETRSegLarge",
        "RFDETRSegXLarge",
    ]
    package_asset_id: str | None = None
    weight_asset_id: str | None = None
    seed: int
    resolution: tuple[int, int] | None = None
    dataset_fraction: float = Field(default=1.0, gt=0, le=1)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    max_train_batches: Literal[1] = 1
    early_stopping_patience: int = Field(default=0, ge=0)
    synthetic_only: Literal[True]
    augmentation_backend: Literal["albumentations", "torchvision", "kornia"] | None = None
    augmentation: dict[str, Any]

    @model_validator(mode="after")
    def geometry_augmentations_are_calibration_safe(self) -> SmokeTrainingConfig:
        if self.augmentation.get("horizontal_flip") is not False:
            raise ValueError("horizontal flip must be explicitly disabled")
        if self.augmentation.get("vertical_flip") is not False:
            raise ValueError("vertical flip must be explicitly disabled")
        if self.augmentation.get("calibration_transform_required") is not True:
            raise ValueError("augmentation must transform calibration with geometry")
        return self


class DetectorTrainingBackend(Protocol):
    name: str
    version: str

    def train(
        self,
        *,
        dataset_dir: Path,
        config: dict[str, Any],
        checkpoint_path: Path,
        log_path: Path,
    ) -> None: ...

    def reload(self, checkpoint_path: Path) -> None: ...

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None: ...


@dataclass(frozen=True)
class DetectorTrainingRun:
    output: Path
    data_hash: str
    config_hash: str
    run_hash: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset(dataset_dir: Path) -> str:
    try:
        derived = json.loads((dataset_dir / "derived-manifest.json").read_text(encoding="utf-8"))
        coco = json.loads(
            (dataset_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8")
        )
        expected_hash = derived.pop("data_hash")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DetectorTrainingError("derived training dataset is incomplete") from exc
    actual_hash = hashlib.sha256(
        _canonical_bytes({"coco": coco, "derived_manifest": derived})
    ).hexdigest()
    if actual_hash != expected_hash:
        raise DetectorTrainingError("derived training dataset hash mismatch")
    if derived.get("purpose") != "local_training":
        raise DetectorTrainingError("derived dataset is not approved for local training")
    if not derived.get("table_cropped") or not derived.get("audio_free"):
        raise DetectorTrainingError("derived dataset violates media minimisation")
    root = dataset_dir.resolve()
    for image in coco.get("images", []):
        path = (dataset_dir / image["file_name"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise DetectorTrainingError("derived image is missing or escapes dataset root")
        if _sha256_file(path) != image["sha256"]:
            raise DetectorTrainingError("derived image sha256 mismatch")
    layout_hashes = derived.get("rfdetr_layout_hashes")
    if layout_hashes is not None:
        if not isinstance(layout_hashes, dict) or not layout_hashes:
            raise DetectorTrainingError("RF-DETR layout hash evidence is invalid")
        required_paths: set[str] = set()
        for split in ("train", "valid"):
            relative_annotations = f"{split}/_annotations.coco.json"
            if relative_annotations not in layout_hashes:
                raise DetectorTrainingError("RF-DETR layout hash evidence is incomplete")
            annotation_path = dataset_dir / relative_annotations
            resolved_annotations = annotation_path.resolve()
            if (
                annotation_path.is_symlink()
                or not resolved_annotations.is_relative_to(root)
                or not resolved_annotations.is_file()
            ):
                raise DetectorTrainingError("RF-DETR layout hash mismatch")
            try:
                split_coco = json.loads(resolved_annotations.read_text(encoding="utf-8"))
                split_images = split_coco["images"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise DetectorTrainingError("RF-DETR layout COCO evidence is invalid") from exc
            if not isinstance(split_images, list) or not split_images:
                raise DetectorTrainingError("RF-DETR layout COCO evidence is invalid")
            required_paths.add(relative_annotations)
            for image in split_images:
                if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
                    raise DetectorTrainingError("RF-DETR layout COCO evidence is invalid")
                file_name = PurePosixPath(image["file_name"])
                if (
                    file_name.is_absolute()
                    or ".." in file_name.parts
                    or str(file_name) in {"", "."}
                ):
                    raise DetectorTrainingError("RF-DETR layout COCO evidence is invalid")
                required_paths.add(f"{split}/{file_name.as_posix()}")
        if set(layout_hashes) != required_paths:
            raise DetectorTrainingError("RF-DETR layout hash evidence is incomplete")
        for relative_path, expected_sha256 in layout_hashes.items():
            if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
                raise DetectorTrainingError("RF-DETR layout hash evidence is invalid")
            unresolved = dataset_dir / relative_path
            path = unresolved.resolve()
            if (
                unresolved.is_symlink()
                or not path.is_relative_to(root)
                or not path.is_file()
                or _sha256_file(path) != expected_sha256
            ):
                raise DetectorTrainingError("RF-DETR layout hash mismatch")
    return expected_hash


def _backend_identity(backend: DetectorTrainingBackend) -> dict[str, str]:
    identity = {"name": backend.name, "version": backend.version}
    evidence_factory = getattr(backend, "identity_evidence", None)
    if evidence_factory is None:
        return identity
    if not callable(evidence_factory):
        raise DetectorTrainingError("backend identity evidence is invalid")
    evidence = evidence_factory()
    if (
        not isinstance(evidence, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in evidence.items()
        )
        or set(evidence) & set(identity)
    ):
        raise DetectorTrainingError("backend identity evidence is invalid")
    return {**identity, **dict(sorted(evidence.items()))}


def _load_config(path: Path) -> tuple[SmokeTrainingConfig, dict[str, Any], str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = SmokeTrainingConfig.model_validate(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise DetectorTrainingError("smoke training config is invalid") from exc
    canonical = config.model_dump(mode="json")
    return config, canonical, hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _verify_training_log(path: Path) -> None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise DetectorTrainingError("training log must be valid JSONL") from exc
    if not rows or any("epoch" not in row or "loss" not in row for row in rows):
        raise DetectorTrainingError("training log lacks epoch/loss evidence")
    if any(
        isinstance(row["loss"], bool)
        or not isinstance(row["loss"], (int, float))
        or not math.isfinite(float(row["loss"]))
        for row in rows
    ):
        raise DetectorTrainingError("training log requires finite numeric loss evidence")


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def run_detector_training_smoke(
    *,
    dataset_dir: Path,
    config_path: Path,
    runs_root: Path,
    backend: DetectorTrainingBackend,
) -> DetectorTrainingRun:
    """Run a synthetic-only lifecycle test; this is never real accuracy evidence."""
    data_hash = _verify_dataset(dataset_dir)
    config, canonical_config, config_hash = _load_config(config_path)
    identity = {
        "data_hash": data_hash,
        "config_hash": config_hash,
        "backend": _backend_identity(backend),
    }
    run_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    parent = runs_root / data_hash / config.candidate_id
    output = parent / run_hash
    if output.exists():
        return DetectorTrainingRun(output, data_hash, config_hash, run_hash)

    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        checkpoint = temporary / "checkpoint.pt"
        log = temporary / "training.jsonl"
        onnx = temporary / "model.onnx"
        backend.train(
            dataset_dir=dataset_dir,
            config=canonical_config,
            checkpoint_path=checkpoint,
            log_path=log,
        )
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise DetectorTrainingError("backend did not produce a checkpoint")
        _verify_training_log(log)
        backend.reload(checkpoint)
        backend.export_onnx(checkpoint_path=checkpoint, output_path=onnx)
        if not onnx.is_file() or onnx.stat().st_size == 0:
            raise DetectorTrainingError("backend did not produce an ONNX export")
        manifest = {
            "schema_version": "1.0",
            "run_hash": run_hash,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "candidate_id": config.candidate_id,
            "model_class": config.model_class,
            "seed": config.seed,
            "synthetic_only": True,
            "backend": identity["backend"],
            "checkpoint_reloaded": True,
            "onnx_exported": True,
            "artifacts": {
                "checkpoint.pt": _sha256_file(checkpoint),
                "model.onnx": _sha256_file(onnx),
                "training.jsonl": _sha256_file(log),
            },
        }
        (temporary / "training-manifest.json").write_bytes(_canonical_bytes(manifest))
        _make_immutable(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return DetectorTrainingRun(output, data_hash, config_hash, run_hash)
