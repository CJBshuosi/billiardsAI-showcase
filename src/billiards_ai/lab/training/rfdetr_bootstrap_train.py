"""Artifact-first RF-DETR short training on reviewed SAM2 bootstrap labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict

from billiards_ai.lab.training.rfdetr_train import (
    DetectorTrainingError,
    _backend_identity,
    _canonical_bytes,
    _make_immutable,
    _sha256_file,
    _verify_training_log,
)


class BootstrapShortTrainingConfig(BaseModel):
    """Frozen GTX 1650 Ti short-run configuration; not a production recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    candidate_id: Literal["rfdetr-seg-small-bootstrap-short-r1"]
    model_class: Literal["RFDETRSegSmall"]
    package_asset_id: Literal["rfdetr-package-1.9.0"]
    weight_asset_id: Literal["rfdetr-seg-small-1.9.0"]
    seed: Literal[20260801]
    resolution: tuple[Literal[384], Literal[384]]
    batch_size: Literal[1]
    grad_accum_steps: Literal[4]
    epochs: Literal[12]
    num_workers: Literal[0]
    accelerator: Literal["cuda"]
    amp: Literal[True]
    synthetic_only: Literal[False]
    dataset_kind: Literal["sam2_rfdetr_bootstrap"]
    training_purpose: Literal["local_bootstrap_short"]
    augmentation_backend: Literal["torchvision"]
    geometry_augmentation: Literal[False]
    use_ema: Literal[False]
    early_stopping: Literal[False]
    baseline_validation: Literal[True]
    export_onnx: Literal[True]
    scientific_use: Literal["development_short_train_not_locked_accuracy"]


class BootstrapTrainingBackend(Protocol):
    name: str
    version: str

    def train_bootstrap_short(
        self,
        *,
        dataset_dir: Path,
        config: dict[str, Any],
        checkpoint_path: Path,
        log_path: Path,
        metrics_path: Path,
        upstream_metrics_path: Path,
    ) -> None: ...

    def reload(self, checkpoint_path: Path) -> None: ...

    def export_onnx(self, *, checkpoint_path: Path, output_path: Path) -> None: ...


@dataclass(frozen=True)
class BootstrapDatasetEvidence:
    dataset_id: str
    image_count: int
    annotation_count: int
    train_image_count: int
    valid_image_count: int


@dataclass(frozen=True)
class BootstrapTrainingRun:
    output: Path
    dataset_id: str
    config_hash: str
    run_hash: str


def _safe_member(root: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or reference in {"", "."}:
        raise DetectorTrainingError("bootstrap dataset inventory is unsafe")
    unresolved = root.joinpath(*relative.parts)
    resolved = unresolved.resolve()
    if (
        unresolved.is_symlink()
        or not resolved.is_relative_to(root.resolve())
        or not resolved.is_file()
    ):
        raise DetectorTrainingError("bootstrap dataset payload is missing or unsafe")
    return resolved


def _load_coco_split(
    dataset_dir: Path,
    *,
    split: str,
    allowed_assets: set[str],
) -> tuple[int, int]:
    try:
        payload = json.loads(
            (dataset_dir / split / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        images = payload["images"]
        annotations = payload["annotations"]
        categories = payload["categories"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DetectorTrainingError("bootstrap dataset COCO split is invalid") from exc
    if (
        not isinstance(images, list)
        or not images
        or not isinstance(annotations, list)
        or not annotations
        or categories != [{"id": 1, "name": "ball", "supercategory": "billiards"}]
    ):
        raise DetectorTrainingError("bootstrap dataset COCO split is invalid")
    image_ids: set[int] = set()
    for image in images:
        if not isinstance(image, dict):
            raise DetectorTrainingError("bootstrap dataset COCO image is invalid")
        image_id = image.get("id")
        file_name = image.get("file_name")
        asset_id = image.get("source_asset_id")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id in image_ids
            or not isinstance(file_name, str)
            or not isinstance(asset_id, str)
            or asset_id not in allowed_assets
        ):
            raise DetectorTrainingError("bootstrap dataset COCO image is invalid")
        image_ids.add(image_id)
        _safe_member(dataset_dir / split, file_name)
    for annotation in annotations:
        if (
            not isinstance(annotation, dict)
            or annotation.get("image_id") not in image_ids
            or annotation.get("category_id") != 1
            or annotation.get("iscrowd") != 0
            or not annotation.get("segmentation")
            or not annotation.get("instance_id")
        ):
            raise DetectorTrainingError("bootstrap dataset COCO annotation is invalid")
    return len(images), len(annotations)


def verify_bootstrap_training_dataset(dataset_dir: Path) -> BootstrapDatasetEvidence:
    """Verify an immutable reviewed bootstrap publication before any GPU work."""

    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise DetectorTrainingError("bootstrap dataset is missing or unsafe")
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        artifact_id = manifest["artifact_id"]
        split_assets = manifest["split_assets"]
        train_assets = set(split_assets["train"])
        valid_assets = set(split_assets["valid"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DetectorTrainingError("bootstrap dataset manifest is invalid") from exc
    if not train_assets or not valid_assets or train_assets & valid_assets:
        raise DetectorTrainingError("bootstrap dataset must remain video-disjoint")
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or artifact_id != dataset_dir.name
        or manifest.get("schema_version") != "1.0"
        or manifest.get("kind") != "sam2_rfdetr_bootstrap"
        or manifest.get("policy")
        != {
            "benchmark_frames_excluded": True,
            "frame_complete_labels_required": True,
            "qa_flags_allowed": False,
            "scientific_use": "development_bootstrap_not_independent_accuracy",
        }
    ):
        raise DetectorTrainingError("bootstrap dataset identity is invalid")
    identity = {key: value for key, value in manifest.items() if key != "artifact_id"}
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != artifact_id:
        raise DetectorTrainingError("bootstrap dataset identity hash changed")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise DetectorTrainingError("bootstrap dataset inventory is invalid")
    expected_files = {"manifest.json"}
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise DetectorTrainingError("bootstrap dataset inventory is invalid")
        reference = item["path"]
        if not isinstance(reference, str):
            raise DetectorTrainingError("bootstrap dataset inventory is invalid")
        path = _safe_member(dataset_dir, reference)
        expected_files.add(reference)
        if path.stat().st_size != item["bytes"] or _sha256_file(path) != item["sha256"]:
            raise DetectorTrainingError("bootstrap dataset payload changed")
    actual_files = {
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise DetectorTrainingError("bootstrap dataset inventory changed")
    if os.name != "nt" and any(
        stat.S_IMODE(path.stat().st_mode) & 0o222 for path in (dataset_dir, *dataset_dir.rglob("*"))
    ):
        raise DetectorTrainingError("bootstrap dataset publication is mutable")
    train_images, train_annotations = _load_coco_split(
        dataset_dir, split="train", allowed_assets=train_assets
    )
    valid_images, valid_annotations = _load_coco_split(
        dataset_dir, split="valid", allowed_assets=valid_assets
    )
    image_count = train_images + valid_images
    annotation_count = train_annotations + valid_annotations
    if (
        manifest.get("image_count") != image_count
        or manifest.get("annotation_count") != annotation_count
    ):
        raise DetectorTrainingError("bootstrap dataset counts changed")
    return BootstrapDatasetEvidence(
        dataset_id=artifact_id,
        image_count=image_count,
        annotation_count=annotation_count,
        train_image_count=train_images,
        valid_image_count=valid_images,
    )


def _load_config(
    config_path: Path,
) -> tuple[BootstrapShortTrainingConfig, dict[str, Any], str]:
    try:
        config = BootstrapShortTrainingConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DetectorTrainingError("bootstrap short training config is invalid") from exc
    canonical = config.model_dump(mode="json")
    return config, canonical, hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _verify_comparison_metrics(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = payload["before_finetune"]
        after = payload["after_short_train"]
        delta = payload["delta"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DetectorTrainingError("bootstrap comparison metrics are invalid") from exc
    if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(delta, dict):
        raise DetectorTrainingError("bootstrap comparison metrics are invalid")
    shared = set(before) & set(after)
    if not shared or set(delta) != shared:
        raise DetectorTrainingError("bootstrap comparison metrics lack matched validation evidence")
    for metric in shared:
        values = (before[metric], after[metric], delta[metric])
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ) or not math.isclose(
            float(delta[metric]),
            float(after[metric]) - float(before[metric]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise DetectorTrainingError("bootstrap comparison metrics are invalid")


def _verify_upstream_metrics(path: Path) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except OSError as exc:
        raise DetectorTrainingError("RF-DETR upstream metrics.csv is missing") from exc
    if not rows:
        raise DetectorTrainingError("RF-DETR upstream metrics.csv is empty")


def _verify_cached_run(
    output: Path,
    *,
    dataset_id: str,
    config_hash: str,
    run_hash: str,
) -> None:
    try:
        manifest = json.loads((output / "training-manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DetectorTrainingError("cached bootstrap training artifact changed") from exc
    if (
        output.is_symlink()
        or not output.is_dir()
        or output.name != run_hash
        or manifest.get("run_hash") != run_hash
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("config_hash") != config_hash
        or not isinstance(artifacts, dict)
    ):
        raise DetectorTrainingError("cached bootstrap training artifact changed")
    expected_files = {"training-manifest.json", *artifacts}
    actual_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise DetectorTrainingError("cached bootstrap training artifact changed")
    for name, expected_sha256 in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise DetectorTrainingError("cached bootstrap training artifact changed")
        path = _safe_member(output, name)
        if _sha256_file(path) != expected_sha256:
            raise DetectorTrainingError("cached bootstrap training artifact changed")


def run_bootstrap_short_training(
    *,
    dataset_dir: Path,
    config_path: Path,
    runs_root: Path,
    backend: BootstrapTrainingBackend,
) -> BootstrapTrainingRun:
    """Run the approved local short train without making a locked accuracy claim."""

    dataset = verify_bootstrap_training_dataset(dataset_dir)
    config, canonical_config, config_hash = _load_config(config_path)
    identity = {
        "dataset_id": dataset.dataset_id,
        "config_hash": config_hash,
        "backend": _backend_identity(backend),
    }
    run_hash = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    parent = runs_root / dataset.dataset_id / config.candidate_id
    output = parent / run_hash
    if output.exists():
        _verify_cached_run(
            output,
            dataset_id=dataset.dataset_id,
            config_hash=config_hash,
            run_hash=run_hash,
        )
        return BootstrapTrainingRun(output, dataset.dataset_id, config_hash, run_hash)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        checkpoint = temporary / "checkpoint_best.pth"
        log = temporary / "training.jsonl"
        metrics = temporary / "metrics.json"
        upstream_metrics = temporary / "metrics.csv"
        onnx = temporary / "model.onnx"
        backend.train_bootstrap_short(
            dataset_dir=dataset_dir,
            config=canonical_config,
            checkpoint_path=checkpoint,
            log_path=log,
            metrics_path=metrics,
            upstream_metrics_path=upstream_metrics,
        )
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise DetectorTrainingError("backend did not produce a bootstrap checkpoint")
        _verify_training_log(log)
        _verify_comparison_metrics(metrics)
        _verify_upstream_metrics(upstream_metrics)
        backend.reload(checkpoint)
        backend.export_onnx(checkpoint_path=checkpoint, output_path=onnx)
        if not onnx.is_file() or onnx.stat().st_size == 0:
            raise DetectorTrainingError("backend did not produce a bootstrap ONNX export")
        artifact_names = (
            "checkpoint_best.pth",
            "metrics.csv",
            "metrics.json",
            "model.onnx",
            "training.jsonl",
        )
        manifest = {
            "schema_version": "1.0",
            "run_hash": run_hash,
            "dataset_id": dataset.dataset_id,
            "config_hash": config_hash,
            "candidate_id": config.candidate_id,
            "model_class": config.model_class,
            "seed": config.seed,
            "synthetic_only": False,
            "scientific_use": config.scientific_use,
            "backend": identity["backend"],
            "dataset_counts": {
                "images": dataset.image_count,
                "annotations": dataset.annotation_count,
                "train_images": dataset.train_image_count,
                "valid_images": dataset.valid_image_count,
            },
            "checkpoint_reloaded": True,
            "onnx_exported": True,
            "artifacts": {name: _sha256_file(temporary / name) for name in artifact_names},
        }
        (temporary / "training-manifest.json").write_bytes(_canonical_bytes(manifest))
        _make_immutable(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return BootstrapTrainingRun(output, dataset.dataset_id, config_hash, run_hash)
