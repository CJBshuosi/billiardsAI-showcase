"""Frozen local comparison of trained RF-DETR, COCO RF-DETR, and classic CV."""

from __future__ import annotations

import gc
import hashlib
import html
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np
from pydantic import Field

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr
from billiards_ai.lab.propagation.contracts import MaskObservation
from billiards_ai.lab.propagation.evaluation import compute_set_metrics, match_frame
from billiards_ai.lab.propagation.frame_mapping import (
    _read_source_geometry,
    _rotate_to_display,
)
from billiards_ai.lab.propagation.pilot_bundle import (
    PilotInputBundle,
    PilotInventoryItem,
    dataset_id_from_inventory,
    verify_pilot_input_bundle,
)
from billiards_ai.lab.propagation.pilot_data import (
    FROZEN_SHOT_COUNT,
    FROZEN_VIDEO_IDS,
    EvaluationCircle,
    EvaluationManifest,
    build_pilot_manifests,
    preflight_frozen_pilot_counts,
)
from billiards_ai.lab.training.rfdetr_bootstrap_train import (
    _load_config,
    _verify_cached_run,
    verify_bootstrap_training_dataset,
)
from billiards_ai.lab.training.rfdetr_train import (
    _canonical_bytes as _training_canonical_bytes,
)

BenchmarkMethod = Literal["trained_rfdetr", "coco_rfdetr", "classic_cv"]
FrameKey = tuple[str, int]
RFDETR_THRESHOLD = 0.25
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_APPROVED_TRAINING_CONFIG = (
    _PROJECT_ROOT / "configs" / "algorithm" / "training" / "rfdetr-seg-small-bootstrap-short.yaml"
)


class PredictionCollection(Protocol):
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    mask: np.ndarray | None

    def __len__(self) -> int: ...


class BenchmarkModel(Protocol):
    def predict(self, image: np.ndarray, **kwargs: object) -> PredictionCollection: ...


class ClassicBenchmarkBackend(Protocol):
    def detect(self, image: object) -> tuple[Mapping[str, Any], ...]: ...


class BenchmarkMethodMetrics(FrozenContract):
    method: BenchmarkMethod
    frame_count: int = Field(ge=0)
    ground_truth_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    true_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    median_normalized_center_error: float | None = Field(default=None, ge=0)
    p90_normalized_center_error: float | None = Field(default=None, ge=0)


class BenchmarkGate(FrozenContract):
    value: float | None
    threshold: float
    comparison: Literal[">=", "<=", ">"]
    passed: bool


class BenchmarkGates(FrozenContract):
    precision: BenchmarkGate
    recall: BenchmarkGate
    f1: BenchmarkGate
    median_center_error: BenchmarkGate
    p90_center_error: BenchmarkGate
    beats_coco_f1: BenchmarkGate
    beats_classic_f1: BenchmarkGate
    passed: bool


@dataclass(frozen=True)
class BenchmarkExecution:
    predictions: dict[BenchmarkMethod, dict[FrameKey, tuple[MaskObservation, ...]]]
    metrics: dict[BenchmarkMethod, BenchmarkMethodMetrics]
    gates: BenchmarkGates


@dataclass(frozen=True)
class FrozenBenchmarkRun:
    path: Path
    artifact_id: str
    reused: bool = False


@dataclass(frozen=True)
class FrozenBenchmarkInputs:
    input_bundle: Path
    evaluation_manifest: Path
    bootstrap_dataset: Path
    training_run: Path
    base_weight: Path
    expected_base_weight_sha256: str
    expected_training_manifest_sha256: str
    expected_trained_checkpoint_sha256: str
    output_root: Path
    trained_model_factory: Callable[[Path], BenchmarkModel] = field(repr=False)
    coco_model_factory: Callable[[Path], BenchmarkModel] = field(repr=False)
    classic_backend_factory: Callable[[], ClassicBenchmarkBackend] = field(repr=False)


@dataclass(frozen=True)
class _VerifiedBenchmarkInputs:
    bundle: PilotInputBundle
    evaluation: EvaluationManifest
    bootstrap_dataset_id: str
    training_run_hash: str
    training_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    base_weight: Path
    base_weight_sha256: str


def _prediction_id(
    *, method: BenchmarkMethod, asset_id: str, frame_index: int, index: int, box: Sequence[float]
) -> str:
    payload = json.dumps(
        {
            "asset_id": asset_id,
            "box": [float(value) for value in box],
            "frame_index": frame_index,
            "index": index,
            "method": method,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _box_observation(
    *,
    asset_id: str,
    frame_index: int,
    method: BenchmarkMethod,
    index: int,
    box: Sequence[float],
    confidence: float,
    area_px: int,
) -> MaskObservation:
    left, top, right, bottom = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)) or (
        right <= left or bottom <= top
    ):
        raise ValueError("benchmark prediction box is invalid")
    prediction_id = _prediction_id(
        method=method,
        asset_id=asset_id,
        frame_index=frame_index,
        index=index,
        box=(left, top, right, bottom),
    )
    return MaskObservation(
        asset_id=asset_id,
        frame_index=frame_index,
        instance_id=f"{method}-{prediction_id}",
        class_name="ball",
        prompt_frame_index=frame_index,
        center_px=((left + right) / 2, (top + bottom) / 2),
        box_xyxy_px=(left, top, right, bottom),
        area_px=max(1, area_px),
        mask_ref=f"predictions/{method}/{prediction_id}.png",
        confidence=confidence,
    )


def adapt_rfdetr_predictions(
    *,
    asset_id: NonEmptyStr,
    frame_index: int,
    detections: PredictionCollection,
    accepted_class_id: int,
    method: Literal["trained_rfdetr", "coco_rfdetr"],
) -> tuple[MaskObservation, ...]:
    """Filter an RF-DETR result through the frozen threshold and class mapping."""

    observations: list[MaskObservation] = []
    for index in range(len(detections)):
        confidence = float(detections.confidence[index])
        if int(detections.class_id[index]) != accepted_class_id or confidence < RFDETR_THRESHOLD:
            continue
        box = tuple(float(value) for value in detections.xyxy[index])
        if detections.mask is None:
            area = round(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        else:
            area = int(np.count_nonzero(detections.mask[index]))
        observations.append(
            _box_observation(
                asset_id=asset_id,
                frame_index=frame_index,
                method=method,
                index=index,
                box=box,
                confidence=confidence,
                area_px=area,
            )
        )
    return tuple(observations)


def adapt_classic_predictions(
    *,
    asset_id: NonEmptyStr,
    frame_index: int,
    detections: Sequence[Mapping[str, Any]],
) -> tuple[MaskObservation, ...]:
    """Convert supported classic-CV contours into the common evaluation contract."""

    observations: list[MaskObservation] = []
    for index, detection in enumerate(detections):
        if detection.get("rejection_reason") is not None:
            continue
        x_px, y_px = (float(value) for value in detection["center_px"])
        radius = float(detection["radius_px"])
        polygon = np.asarray(detection["mask_polygon_px"], dtype=np.float32)
        area = round(abs(float(cv2.contourArea(polygon))))
        observations.append(
            _box_observation(
                asset_id=asset_id,
                frame_index=frame_index,
                method="classic_cv",
                index=index,
                box=(x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                confidence=float(detection["confidence"]),
                area_px=area,
            )
        )
    return tuple(observations)


def evaluate_method(
    *,
    method: BenchmarkMethod,
    circles: Sequence[EvaluationCircle],
    predictions: Mapping[FrameKey, Sequence[MaskObservation]],
) -> BenchmarkMethodMetrics:
    """Evaluate one method over the exact frozen frame population."""

    circles_by_frame: dict[FrameKey, list[EvaluationCircle]] = {}
    for circle in circles:
        circles_by_frame.setdefault((circle.video_id, circle.source_frame_index), []).append(circle)
    unexpected = set(predictions) - set(circles_by_frame)
    if unexpected:
        raise ValueError("benchmark predictions escape the frozen evaluation population")
    assignments = tuple(
        match_frame(frame_circles, predictions.get(frame_key, ()))
        for frame_key, frame_circles in sorted(circles_by_frame.items())
    )
    metrics = compute_set_metrics(assignments)
    precision = (
        metrics.true_positive_support / metrics.prediction_count
        if metrics.prediction_count
        else 0.0
    )
    denominator = precision + metrics.recall
    f1 = 2 * precision * metrics.recall / denominator if denominator else 0.0
    return BenchmarkMethodMetrics(
        method=method,
        frame_count=len(circles_by_frame),
        ground_truth_count=metrics.ground_truth_count,
        prediction_count=metrics.prediction_count,
        true_positive_count=metrics.true_positive_support,
        false_negative_count=metrics.false_negative_count,
        false_positive_count=metrics.false_positive_count,
        precision=precision,
        recall=metrics.recall,
        f1=f1,
        median_normalized_center_error=metrics.median_normalized_center_error,
        p90_normalized_center_error=metrics.p90_normalized_center_error,
    )


def _gate(value: float | None, threshold: float, comparison: Literal[">=", "<=", ">"]):
    passed = value is not None and (
        value >= threshold
        if comparison == ">="
        else value <= threshold
        if comparison == "<="
        else value > threshold
    )
    return BenchmarkGate(
        value=value,
        threshold=threshold,
        comparison=comparison,
        passed=passed,
    )


def compute_benchmark_gates(
    *,
    trained: BenchmarkMethodMetrics,
    coco: BenchmarkMethodMetrics,
    classic: BenchmarkMethodMetrics,
) -> BenchmarkGates:
    """Apply the pre-result absolute gates and strict F1 baseline lift."""

    if (
        trained.method != "trained_rfdetr"
        or coco.method != "coco_rfdetr"
        or classic.method != "classic_cv"
    ):
        raise ValueError("benchmark gate methods are misidentified")
    median = trained.median_normalized_center_error
    p90 = trained.p90_normalized_center_error
    gates = {
        "precision": _gate(trained.precision, 0.85, ">="),
        "recall": _gate(trained.recall, 0.85, ">="),
        "f1": _gate(trained.f1, 0.85, ">="),
        "median_center_error": _gate(median, 0.5, "<="),
        "p90_center_error": _gate(p90, 1.0, "<="),
        "beats_coco_f1": _gate(trained.f1, coco.f1, ">"),
        "beats_classic_f1": _gate(trained.f1, classic.f1, ">"),
    }
    return BenchmarkGates(**gates, passed=all(gate.passed for gate in gates.values()))


def verify_no_frozen_frame_leakage(
    dataset_dir: Path, *, frozen_frames: frozenset[FrameKey]
) -> None:
    """Reject a bootstrap COCO split containing any held-out evaluation frame."""

    seen: set[FrameKey] = set()
    for split in ("train", "valid"):
        try:
            payload = json.loads(
                (dataset_dir / split / "_annotations.coco.json").read_text(encoding="utf-8")
            )
            images = payload["images"]
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ValueError("bootstrap dataset split is invalid") from exc
        if not isinstance(images, list):
            raise ValueError("bootstrap dataset split is invalid")
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("bootstrap dataset split is invalid")
            asset_id = image.get("source_asset_id")
            frame_index = image.get("source_frame_index")
            if (
                not isinstance(asset_id, str)
                or not asset_id
                or isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise ValueError("bootstrap dataset lacks exact frame provenance")
            key = (asset_id, frame_index)
            if key in seen:
                raise ValueError("bootstrap dataset duplicates frame provenance")
            seen.add(key)
            if key in frozen_frames:
                raise ValueError("bootstrap dataset leaks a frozen evaluation frame")


def verify_frozen_evaluation_population(circles: Sequence[EvaluationCircle]) -> None:
    """Require the exact approved held-out frame shape before model execution."""

    frames = {(circle.video_id, circle.source_frame_index) for circle in circles}
    if len(frames) != 50:
        raise ValueError("frozen evaluation requires exactly 50 held-out frames")
    if len(circles) != 510:
        raise ValueError("frozen evaluation requires exactly 510 ball circles")


def _load_bundle(path: Path) -> PilotInputBundle:
    try:
        payload = json.loads((path / "inventory.json").read_text(encoding="utf-8"))
        inventory = tuple(PilotInventoryItem.model_validate(item) for item in payload["files"])
        bundle = PilotInputBundle(bundle_id=path.name, path=path, inventory=inventory)
        verify_pilot_input_bundle(bundle)
        return bundle
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("benchmark input bundle is invalid") from exc


def _load_evaluation(path: Path) -> EvaluationManifest:
    try:
        manifest = EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("benchmark evaluation manifest is invalid") from exc
    identity = manifest.model_dump(mode="json", exclude={"manifest_id"})
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != manifest.manifest_id:
        raise ValueError("benchmark evaluation manifest identity changed")
    verify_frozen_evaluation_population(manifest.circles)
    if len(manifest.shots) != FROZEN_SHOT_COUNT or {
        shot.video_id for shot in manifest.shots
    } != set(FROZEN_VIDEO_IDS):
        raise ValueError("benchmark evaluation manifest is not the frozen pilot population")
    role_frames: dict[tuple[str, int], dict[str, set[int]]] = {}
    for circle in manifest.circles:
        role_frames.setdefault(
            (circle.video_id, circle.shot_index), {"middle": set(), "end": set()}
        )[circle.role].add(circle.source_frame_index)
    shots = {(shot.video_id, shot.shot_index): shot for shot in manifest.shots}
    if set(role_frames) != set(shots):
        raise ValueError("benchmark evaluation manifest lacks frozen shot evidence")
    for key, roles in role_frames.items():
        if len(roles["middle"]) != 1 or len(roles["end"]) != 1:
            raise ValueError("benchmark evaluation manifest role population changed")
        middle = next(iter(roles["middle"]))
        end = next(iter(roles["end"]))
        shot = shots[key]
        if not shot.start_frame < middle < end < shot.end_frame:
            raise ValueError("benchmark evaluation manifest frame order changed")
    return manifest


def resolve_materialized_annotation_export(
    *,
    bundle: PilotInputBundle,
    evaluation_manifest: Path,
) -> Path:
    """Resolve and verify prepare-bundle's review-image materialization."""

    try:
        manifest_parent = evaluation_manifest.parent
        if manifest_parent.parent.name != "evaluation-manifests":
            raise ValueError("evaluation manifest is outside a prepared control root")
        control_root = manifest_parent.parent.parent
        materialized = control_root / "imported-inputs" / bundle.bundle_id / "annotation-export"
        raw_export = bundle.path / "annotation-export"
        annotations = json.loads((raw_export / "annotations.json").read_text(encoding="utf-8"))
        images = annotations["images"]
        if not isinstance(images, list):
            raise ValueError("annotation images are invalid")
        expected = {"annotations.json", "summary.json"}
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
                raise ValueError("annotation image identity is invalid")
            relative = PurePosixPath(image["file_name"])
            if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
                raise ValueError("annotation image path is unsafe")
            expected.add((PurePosixPath("images") / relative).as_posix())
        entries = tuple(materialized.rglob("*"))
        actual = {path.relative_to(materialized).as_posix() for path in entries if path.is_file()}
        if (
            materialized.is_symlink()
            or not materialized.is_dir()
            or any(path.is_symlink() for path in entries)
            or actual != expected
            or any(
                (materialized / name).read_bytes() != (raw_export / name).read_bytes()
                for name in ("annotations.json", "summary.json")
            )
        ):
            raise ValueError("materialized annotation export changed")
        return materialized.resolve(strict=True)
    except (OSError, UnicodeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("benchmark materialized annotation export is invalid") from exc


def verify_rebuilt_evaluation(supplied: EvaluationManifest, rebuilt: EvaluationManifest) -> None:
    """Require the supplied blind labels to equal bundle-derived evidence exactly."""

    if supplied.model_dump(mode="json") != rebuilt.model_dump(mode="json"):
        raise ValueError("benchmark evaluation does not match the verified input bundle")


def verify_training_manifest_identity(
    manifest: Mapping[str, object],
    *,
    dataset_id: str,
    approved_config_hash: str,
) -> str:
    """Recompute the approved short-training identity and return its base-weight hash."""

    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("benchmark training identity is invalid")
    expected_backend_keys = {
        "name",
        "version",
        "model_class",
        "weight_asset_id",
        "weight_sha256",
        "accelerator",
        "precision",
    }
    weight_sha256 = backend.get("weight_sha256")
    expected_values = {
        "name": "rfdetr",
        "version": "1.9.0",
        "model_class": "RFDETRSegSmall",
        "weight_asset_id": "rfdetr-seg-small-1.9.0",
        "accelerator": "cuda",
        "precision": "16-mixed",
    }
    if (
        set(backend) != expected_backend_keys
        or any(backend.get(key) != value for key, value in expected_values.items())
        or not isinstance(weight_sha256, str)
        or len(weight_sha256) != 64
        or any(character not in "0123456789abcdef" for character in weight_sha256)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("config_hash") != approved_config_hash
        or manifest.get("candidate_id") != "rfdetr-seg-small-bootstrap-short-r1"
        or manifest.get("model_class") != "RFDETRSegSmall"
        or manifest.get("seed") != 20260801
        or manifest.get("synthetic_only") is not False
        or manifest.get("scientific_use") != "development_short_train_not_locked_accuracy"
        or manifest.get("checkpoint_reloaded") is not True
        or manifest.get("onnx_exported") is not True
    ):
        raise ValueError("benchmark training identity is invalid")
    run_identity = {
        "dataset_id": dataset_id,
        "config_hash": approved_config_hash,
        "backend": backend,
    }
    if (
        manifest.get("run_hash")
        != hashlib.sha256(_training_canonical_bytes(run_identity)).hexdigest()
    ):
        raise ValueError("benchmark training identity is invalid")
    return weight_sha256


def verify_training_artifact_hashes(
    training_run: Path,
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> tuple[str, str]:
    """Anchor the mutable-on-Windows run directory to independently locked hashes."""

    expected = (expected_manifest_sha256, expected_checkpoint_sha256)
    manifest = training_run / "training-manifest.json"
    checkpoint = training_run / "checkpoint_best.pth"
    if (
        any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in expected
        )
        or manifest.is_symlink()
        or checkpoint.is_symlink()
        or not manifest.is_file()
        or not checkpoint.is_file()
    ):
        raise ValueError("benchmark training artifact does not match its locked hash")
    actual = (_sha256_file(manifest), _sha256_file(checkpoint))
    if actual != expected:
        raise ValueError("benchmark training artifact does not match its locked hash")
    return actual


def snapshot_locked_file(
    source: Path,
    *,
    destination: Path,
    expected_sha256: str,
) -> str:
    """Copy one approved file into a private snapshot while hashing the copied bytes."""

    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or source.is_symlink()
        or not source.is_file()
        or destination.exists()
    ):
        raise ValueError("benchmark locked file cannot be snapshotted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while chunk := input_stream.read(1024 * 1024):
                digest.update(chunk)
                output_stream.write(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256 or _sha256_file(destination) != expected_sha256:
            raise ValueError("benchmark locked file snapshot hash changed")
        return actual_sha256
    except Exception:
        if destination.exists():
            destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
            destination.unlink()
        raise


def verify_rfdetr_runtime(
    *,
    version_resolver: Callable[[str], str] | None = None,
    module_loader: Callable[[str], object] | None = None,
) -> None:
    """Pin the inference runtime and API used by both compared RF-DETR models."""

    resolve = importlib.metadata.version if version_resolver is None else version_resolver
    load = importlib.import_module if module_loader is None else module_loader
    try:
        version = resolve("rfdetr")
        module = load("rfdetr")
    except (ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError("frozen benchmark requires rfdetr==1.9.0") from exc
    if version != "1.9.0" or any(
        not hasattr(module, name) for name in ("RFDETRSegSmall", "from_checkpoint")
    ):
        raise ValueError("frozen benchmark requires rfdetr==1.9.0 with inference APIs")


def _verify_inputs(
    inputs: FrozenBenchmarkInputs, *, snapshot_root: Path
) -> _VerifiedBenchmarkInputs:
    bundle = _load_bundle(inputs.input_bundle)
    evaluation = _load_evaluation(inputs.evaluation_manifest)
    try:
        with tempfile.TemporaryDirectory(prefix="billiards-benchmark-rebuild-") as temporary:
            annotation_export = resolve_materialized_annotation_export(
                bundle=bundle,
                evaluation_manifest=inputs.evaluation_manifest,
            )
            rebuilt = build_pilot_manifests(
                annotation_export=annotation_export,
                review_pack=bundle.path / "review-pack",
                dataset_root=bundle.path,
                output_root=Path(temporary),
            )
            preflight_frozen_pilot_counts(rebuilt)
            verify_rebuilt_evaluation(evaluation, rebuilt.evaluation_manifest)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("benchmark evaluation cannot be rebuilt from the input bundle") from exc
    if evaluation.dataset_id != dataset_id_from_inventory(bundle.inventory):
        raise ValueError("benchmark evaluation and input bundle dataset identities differ")
    dataset = verify_bootstrap_training_dataset(inputs.bootstrap_dataset)
    frozen_frames = frozenset(
        (circle.video_id, circle.source_frame_index) for circle in evaluation.circles
    )
    verify_no_frozen_frame_leakage(inputs.bootstrap_dataset, frozen_frames=frozen_frames)
    manifest_snapshot = snapshot_root / "training-manifest.json"
    checkpoint = snapshot_root / "checkpoint_best.pth"
    base_weight = snapshot_root / "rf-detr-seg-s-ft.pth"
    training_manifest_sha256 = snapshot_locked_file(
        inputs.training_run / "training-manifest.json",
        destination=manifest_snapshot,
        expected_sha256=inputs.expected_training_manifest_sha256,
    )
    checkpoint_sha256 = snapshot_locked_file(
        inputs.training_run / "checkpoint_best.pth",
        destination=checkpoint,
        expected_sha256=inputs.expected_trained_checkpoint_sha256,
    )
    base_weight_sha256 = snapshot_locked_file(
        inputs.base_weight,
        destination=base_weight,
        expected_sha256=inputs.expected_base_weight_sha256,
    )
    try:
        training_manifest = json.loads(manifest_snapshot.read_text(encoding="utf-8"))
        run_hash = training_manifest["run_hash"]
        config_hash = training_manifest["config_hash"]
        training_dataset_id = training_manifest["dataset_id"]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("benchmark training artifact is invalid") from exc
    if training_dataset_id != dataset.dataset_id:
        raise ValueError("benchmark training artifact uses a different bootstrap dataset")
    _, _, approved_config_hash = _load_config(_APPROVED_TRAINING_CONFIG)
    base_weight_sha256 = verify_training_manifest_identity(
        training_manifest,
        dataset_id=dataset.dataset_id,
        approved_config_hash=approved_config_hash,
    )
    try:
        _verify_cached_run(
            inputs.training_run,
            dataset_id=dataset.dataset_id,
            config_hash=config_hash,
            run_hash=run_hash,
        )
    except Exception as exc:
        raise ValueError("benchmark training artifact changed") from exc
    if training_manifest.get("artifacts", {}).get("checkpoint_best.pth") != checkpoint_sha256:
        raise ValueError("benchmark trained checkpoint changed")
    if (
        len(inputs.expected_base_weight_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in inputs.expected_base_weight_sha256
        )
        or inputs.expected_base_weight_sha256 != base_weight_sha256
    ):
        raise ValueError("benchmark COCO base weight changed")
    verify_rfdetr_runtime()
    return _VerifiedBenchmarkInputs(
        bundle=bundle,
        evaluation=evaluation,
        bootstrap_dataset_id=dataset.dataset_id,
        training_run_hash=run_hash,
        training_manifest_sha256=training_manifest_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        base_weight=base_weight,
        base_weight_sha256=base_weight_sha256,
    )


def _extract_frozen_frames(
    *,
    bundle: PilotInputBundle,
    evaluation: EvaluationManifest,
) -> dict[FrameKey, np.ndarray]:
    wanted = {(circle.video_id, circle.source_frame_index) for circle in evaluation.circles}
    frames: dict[FrameKey, np.ndarray] = {}
    for video_id in sorted({key[0] for key in wanted}):
        indices = frozenset(frame_index for asset_id, frame_index in wanted if asset_id == video_id)
        decoded = decode_sparse_video_frames(
            video_path=bundle.path / "videos" / f"{video_id}.mp4",
            wanted_indices=indices,
        )
        frames.update({(video_id, frame_index): image for frame_index, image in decoded.items()})
    if set(frames) != wanted:
        raise ValueError("benchmark frozen frame extraction is incomplete")
    return frames


def decode_sparse_video_frames(
    *,
    video_path: Path,
    wanted_indices: frozenset[int],
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
) -> dict[int, np.ndarray]:
    """Fully verify one video while retaining only explicitly requested frames in memory."""

    if (
        not wanted_indices
        or any(isinstance(index, bool) or index < 0 for index in wanted_indices)
        or video_path.is_symlink()
        or not video_path.is_file()
    ):
        raise ValueError("benchmark sparse video request is invalid")
    before = video_path.stat()
    source_sha256 = _sha256_file(video_path)
    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        raise ValueError("benchmark source video cannot be opened")
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        capture.set(orientation_auto, 0)
    try:
        geometry = _read_source_geometry(capture)
        raw_frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if (
            not math.isfinite(raw_frame_count)
            or raw_frame_count <= 0
            or abs(raw_frame_count - round(raw_frame_count)) > 1e-6
        ):
            raise ValueError("benchmark source video frame count is invalid")
        frame_count = round(raw_frame_count)
        if max(wanted_indices) >= frame_count:
            raise ValueError("benchmark frozen frame exceeds source video")
        decoded: dict[int, np.ndarray] = {}
        decoded_count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.shape[:2] != (
                geometry.source_height,
                geometry.source_width,
            ):
                raise ValueError("benchmark source video geometry changed during decode")
            if decoded_count in wanted_indices:
                display_frame = _rotate_to_display(frame, geometry.rotation_degrees)
                if not isinstance(display_frame, np.ndarray):
                    raise ValueError("benchmark source rotation returned an invalid frame")
                decoded[decoded_count] = display_frame.copy()
            decoded_count += 1
        if decoded_count != frame_count or set(decoded) != set(wanted_indices):
            raise ValueError("benchmark source video decode is incomplete")
    finally:
        capture.release()
    after = video_path.stat()
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or _sha256_file(video_path) != source_sha256
    ):
        raise ValueError("benchmark source video changed during decode")
    return decoded


def verify_output_root_isolated(inputs: FrozenBenchmarkInputs) -> None:
    """Reject output paths that can mutate a verified input tree."""

    unresolved = inputs.output_root.absolute()
    for candidate in (unresolved, *unresolved.parents):
        is_junction = getattr(candidate, "is_junction", lambda: False)
        if candidate.is_symlink() or is_junction():
            raise ValueError("benchmark output root uses a link or junction")
    output = unresolved.resolve(strict=False)
    protected_directories = (
        inputs.input_bundle.resolve(strict=False),
        inputs.evaluation_manifest.parent.resolve(strict=False),
        inputs.bootstrap_dataset.resolve(strict=False),
        inputs.training_run.resolve(strict=False),
        inputs.base_weight.parent.resolve(strict=False),
    )
    if any(
        output == protected or output.is_relative_to(protected)
        for protected in protected_directories
    ):
        raise ValueError("benchmark output root overlaps a frozen input")


def run_frozen_benchmark(inputs: FrozenBenchmarkInputs) -> FrozenBenchmarkRun:
    """Verify every input, run the three methods, and publish local evidence."""

    verify_output_root_isolated(inputs)
    with tempfile.TemporaryDirectory(prefix="billiards-benchmark-model-snapshot-") as temporary:
        verified = _verify_inputs(inputs, snapshot_root=Path(temporary))
        frames = _extract_frozen_frames(
            bundle=verified.bundle,
            evaluation=verified.evaluation,
        )
        execution = run_model_factories_on_frames(
            frames=frames,
            circles=verified.evaluation.circles,
            trained_model_factory=lambda: inputs.trained_model_factory(verified.checkpoint),
            coco_model_factory=lambda: inputs.coco_model_factory(verified.base_weight),
            classic_backend_factory=inputs.classic_backend_factory,
        )
        return publish_benchmark(
            execution=execution,
            frames=frames,
            circles=verified.evaluation.circles,
            provenance={
                "input_bundle_id": verified.bundle.bundle_id,
                "evaluation_manifest_id": verified.evaluation.manifest_id,
                "bootstrap_dataset_id": verified.bootstrap_dataset_id,
                "training_run_hash": verified.training_run_hash,
                "training_manifest_sha256": verified.training_manifest_sha256,
                "trained_checkpoint_sha256": verified.checkpoint_sha256,
                "coco_weight_sha256": verified.base_weight_sha256,
            },
            output_root=inputs.output_root,
        )


def run_models_on_frames(
    *,
    frames: Mapping[FrameKey, np.ndarray],
    circles: Sequence[EvaluationCircle],
    trained_model: BenchmarkModel,
    coco_model: BenchmarkModel,
    classic_backend: ClassicBenchmarkBackend,
) -> BenchmarkExecution:
    """Run all three methods over one in-memory frozen frame mapping."""

    circle_frames = {(circle.video_id, circle.source_frame_index) for circle in circles}
    if set(frames) != circle_frames:
        raise ValueError("decoded frames do not equal the frozen evaluation population")
    predictions: dict[BenchmarkMethod, dict[FrameKey, tuple[MaskObservation, ...]]] = {
        "trained_rfdetr": {},
        "coco_rfdetr": {},
        "classic_cv": {},
    }
    for (asset_id, frame_index), image_bgr in sorted(frames.items()):
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("frozen benchmark frame is invalid")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        trained_detections = trained_model.predict(
            image_rgb,
            threshold=RFDETR_THRESHOLD,
            include_source_image=False,
        )
        coco_detections = coco_model.predict(
            image_rgb,
            threshold=RFDETR_THRESHOLD,
            include_source_image=False,
        )
        predictions["trained_rfdetr"][(asset_id, frame_index)] = adapt_rfdetr_predictions(
            asset_id=asset_id,
            frame_index=frame_index,
            detections=trained_detections,
            accepted_class_id=0,
            method="trained_rfdetr",
        )
        predictions["coco_rfdetr"][(asset_id, frame_index)] = adapt_rfdetr_predictions(
            asset_id=asset_id,
            frame_index=frame_index,
            detections=coco_detections,
            accepted_class_id=37,
            method="coco_rfdetr",
        )
        predictions["classic_cv"][(asset_id, frame_index)] = adapt_classic_predictions(
            asset_id=asset_id,
            frame_index=frame_index,
            detections=classic_backend.detect(image_bgr),
        )
    metrics = {
        method: evaluate_method(method=method, circles=circles, predictions=method_predictions)
        for method, method_predictions in predictions.items()
    }
    gates = compute_benchmark_gates(
        trained=metrics["trained_rfdetr"],
        coco=metrics["coco_rfdetr"],
        classic=metrics["classic_cv"],
    )
    return BenchmarkExecution(predictions=predictions, metrics=metrics, gates=gates)


def _release_accelerator_cache() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = None if torch is None else getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available() and callable(getattr(cuda, "empty_cache", None)):
            cuda.empty_cache()


def _run_rfdetr_factory(
    *,
    frames: Mapping[FrameKey, np.ndarray],
    factory: Callable[[], BenchmarkModel],
    accepted_class_id: int,
    method: Literal["trained_rfdetr", "coco_rfdetr"],
) -> dict[FrameKey, tuple[MaskObservation, ...]]:
    model = factory()
    try:
        predictions: dict[FrameKey, tuple[MaskObservation, ...]] = {}
        for (asset_id, frame_index), image_bgr in sorted(frames.items()):
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(
                image_rgb,
                threshold=RFDETR_THRESHOLD,
                include_source_image=False,
            )
            predictions[(asset_id, frame_index)] = adapt_rfdetr_predictions(
                asset_id=asset_id,
                frame_index=frame_index,
                detections=detections,
                accepted_class_id=accepted_class_id,
                method=method,
            )
        return predictions
    finally:
        del model
        _release_accelerator_cache()


def run_model_factories_on_frames(
    *,
    frames: Mapping[FrameKey, np.ndarray],
    circles: Sequence[EvaluationCircle],
    trained_model_factory: Callable[[], BenchmarkModel],
    coco_model_factory: Callable[[], BenchmarkModel],
    classic_backend_factory: Callable[[], ClassicBenchmarkBackend],
) -> BenchmarkExecution:
    """Run RF-DETR models sequentially so a 4 GB GPU never holds both."""

    circle_frames = {(circle.video_id, circle.source_frame_index) for circle in circles}
    if set(frames) != circle_frames:
        raise ValueError("decoded frames do not equal the frozen evaluation population")
    trained = _run_rfdetr_factory(
        frames=frames,
        factory=trained_model_factory,
        accepted_class_id=0,
        method="trained_rfdetr",
    )
    coco = _run_rfdetr_factory(
        frames=frames,
        factory=coco_model_factory,
        accepted_class_id=37,
        method="coco_rfdetr",
    )
    classic_backend = classic_backend_factory()
    classic = {
        (asset_id, frame_index): adapt_classic_predictions(
            asset_id=asset_id,
            frame_index=frame_index,
            detections=classic_backend.detect(image_bgr),
        )
        for (asset_id, frame_index), image_bgr in sorted(frames.items())
    }
    predictions: dict[BenchmarkMethod, dict[FrameKey, tuple[MaskObservation, ...]]] = {
        "trained_rfdetr": trained,
        "coco_rfdetr": coco,
        "classic_cv": classic,
    }
    metrics = {
        method: evaluate_method(method=method, circles=circles, predictions=method_predictions)
        for method, method_predictions in predictions.items()
    }
    gates = compute_benchmark_gates(
        trained=metrics["trained_rfdetr"],
        coco=metrics["coco_rfdetr"],
        classic=metrics["classic_cv"],
    )
    return BenchmarkExecution(predictions=predictions, metrics=metrics, gates=gates)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_payload(
    predictions: Mapping[FrameKey, Sequence[MaskObservation]],
) -> list[dict[str, object]]:
    return [
        observation.model_dump(mode="json")
        for frame_key in sorted(predictions)
        for observation in predictions[frame_key]
    ]


def _failure_payload(
    *,
    execution: BenchmarkExecution,
    circles_by_frame: Mapping[FrameKey, Sequence[EvaluationCircle]],
) -> dict[BenchmarkMethod, list[dict[str, object]]]:
    failures: dict[BenchmarkMethod, list[dict[str, object]]] = {
        "trained_rfdetr": [],
        "coco_rfdetr": [],
        "classic_cv": [],
    }
    for method, method_predictions in execution.predictions.items():
        for key, circles in sorted(circles_by_frame.items()):
            assignment = match_frame(circles, method_predictions[key])
            if not assignment.unmatched_ground_truth and not assignment.unmatched_predictions:
                continue
            failures[method].append(
                {
                    "asset_id": key[0],
                    "frame_index": key[1],
                    "missed_ground_truth_ids": [
                        item.annotation_id for item in assignment.unmatched_ground_truth
                    ],
                    "false_positive_ids": [
                        item.instance_id for item in assignment.unmatched_predictions
                    ],
                }
            )
    return failures


def _render_overlay(
    *,
    image_bgr: np.ndarray,
    circles: Sequence[EvaluationCircle],
    predictions: Sequence[MaskObservation],
    method: BenchmarkMethod,
) -> np.ndarray:
    overlay = image_bgr.copy()
    for circle in circles:
        cv2.circle(
            overlay,
            (round(circle.center_x_px), round(circle.center_y_px)),
            max(1, round(circle.radius_px)),
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    for prediction in predictions:
        left, top, right, bottom = prediction.box_xyxy_px
        cv2.rectangle(
            overlay,
            (round(left), round(top)),
            (round(right), round(bottom)),
            (0, 0, 240),
            2,
        )
    cv2.putText(
        overlay,
        method,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _write_index(
    *,
    path: Path,
    frames: Sequence[FrameKey],
    metrics: Mapping[BenchmarkMethod, BenchmarkMethodMetrics],
    gates: BenchmarkGates,
) -> None:
    labels = {
        "trained_rfdetr": "trained RF-DETR",
        "coco_rfdetr": "COCO RF-DETR",
        "classic_cv": "classic CV",
    }
    rows = "".join(
        f"<tr><th>{html.escape(labels[method])}</th>"
        f"<td>{metrics[method].precision:.4f}</td>"
        f"<td>{metrics[method].recall:.4f}</td>"
        f"<td>{metrics[method].f1:.4f}</td>"
        f"<td>{metrics[method].false_negative_count}</td>"
        f"<td>{metrics[method].false_positive_count}</td>"
        f"<td>{metrics[method].median_normalized_center_error!s}</td>"
        f"<td>{metrics[method].p90_normalized_center_error!s}</td></tr>"
        for method in labels
    )
    gate_rows = "".join(
        f"<tr><th>{html.escape(name)}</th>"
        f"<td>{'None' if gate.value is None else f'{gate.value:.4f}'}</td>"
        f"<td>{html.escape(gate.comparison)}</td>"
        f"<td>{gate.threshold:.4f}</td>"
        f"<td>{'PASS' if gate.passed else 'FAIL'}</td></tr>"
        for name, gate in (
            ("precision", gates.precision),
            ("recall", gates.recall),
            ("f1", gates.f1),
            ("median_center_error", gates.median_center_error),
            ("p90_center_error", gates.p90_center_error),
            ("beats_coco_f1", gates.beats_coco_f1),
            ("beats_classic_f1", gates.beats_classic_f1),
        )
    )
    cards = "".join(
        "<section><h2>"
        + html.escape(f"{asset_id} / frame {frame_index}")
        + "</h2><div class='grid'>"
        + "".join(
            f"<figure><figcaption>{html.escape(labels[method])}</figcaption>"
            f"<img loading='lazy' src='overlays/{html.escape(asset_id)}-"
            f"{frame_index:08d}-{method}.jpg'></figure>"
            for method in labels
        )
        + "</div></section>"
        for asset_id, frame_index in frames
    )
    path.write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<title>RF-DETR frozen benchmark</title><style>"
        "body{font-family:system-ui;margin:24px;background:#111827;color:#f8fafc}"
        "table{border-collapse:collapse}th,td{padding:8px 14px;border:1px solid #475569}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}"
        "img{width:100%;height:auto}figure{margin:0}figcaption{margin-bottom:6px}"
        "</style><h1>RF-DETR 冻结三路 benchmark</h1>"
        f"<p>Overall gate: {'PASS' if gates.passed else 'FAIL'}</p>"
        "<table><tr><th>方法</th><th>Precision</th><th>Recall</th><th>F1</th>"
        "<th>Miss</th><th>False positives</th><th>Median center error</th>"
        f"<th>P90 center error</th></tr>{rows}</table>"
        "<h2>Frozen gates</h2><table><tr><th>Gate</th><th>Value</th>"
        "<th>Comparison</th><th>Threshold</th><th>Status</th></tr>"
        f"{gate_rows}</table>"
        f"{cards}</html>",
        encoding="utf-8",
    )


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    ]


def _verify_publication(
    path: Path,
    *,
    artifact_id: str,
    require_artifact_directory_name: bool = True,
) -> None:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        identity = report["identity"]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("benchmark publication changed") from exc
    if any(item.is_symlink() for item in (path, *path.rglob("*"))):
        raise ValueError("benchmark publication changed")
    if (
        not path.is_dir()
        or (require_artifact_directory_name and path.name != artifact_id)
        or manifest.get("artifact_id") != artifact_id
        or report.get("artifact_id") != artifact_id
        or hashlib.sha256(_canonical_bytes(identity)).hexdigest() != artifact_id
        or not isinstance(identity, dict)
    ):
        raise ValueError("benchmark publication changed")
    evidence_inventory = _inventory(path, exclude=frozenset({"manifest.json", "report.json"}))
    if identity.get("evidence_inventory") != evidence_inventory:
        raise ValueError("benchmark publication changed")
    payload_inventory = _inventory(path, exclude=frozenset({"manifest.json"}))
    if manifest != {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "inventory": payload_inventory,
    }:
        raise ValueError("benchmark publication changed")
    try:
        stored_predictions = json.loads((path / "predictions.json").read_text(encoding="utf-8"))
        expected_hashes = identity["prediction_hashes"]
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("benchmark publication changed") from exc
    if set(stored_predictions) != set(expected_hashes) or any(
        hashlib.sha256(_canonical_bytes(stored_predictions[method])).hexdigest()
        != expected_hashes[method]
        for method in expected_hashes
    ):
        raise ValueError("benchmark publication changed")
    if os.name != "nt" and any(
        stat.S_IMODE(item.stat().st_mode) & 0o222 for item in (path, *path.rglob("*"))
    ):
        raise ValueError("benchmark publication changed")


def _remove_publication_tree(path: Path) -> None:
    """Remove a possibly read-only staging tree without weakening formal artifacts."""

    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True):
        try:
            item.chmod(item.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass
    try:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    except OSError:
        pass
    shutil.rmtree(path)


def publish_benchmark(
    *,
    execution: BenchmarkExecution,
    frames: Mapping[FrameKey, np.ndarray],
    circles: Sequence[EvaluationCircle],
    provenance: Mapping[str, str],
    output_root: Path,
) -> FrozenBenchmarkRun:
    """Publish deterministic metrics and visual evidence as one immutable artifact."""

    required_provenance = {
        "input_bundle_id",
        "evaluation_manifest_id",
        "bootstrap_dataset_id",
        "training_run_hash",
        "training_manifest_sha256",
        "trained_checkpoint_sha256",
        "coco_weight_sha256",
    }
    if set(provenance) != required_provenance or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in provenance.values()
    ):
        raise ValueError("benchmark provenance is incomplete")
    frame_keys = sorted(frames)
    circles_by_frame: dict[FrameKey, list[EvaluationCircle]] = {}
    for circle in circles:
        circles_by_frame.setdefault((circle.video_id, circle.source_frame_index), []).append(circle)
    if set(frame_keys) != set(circles_by_frame):
        raise ValueError("benchmark frames do not match evaluation circles")
    prediction_hashes = {
        method: hashlib.sha256(
            _canonical_bytes(_prediction_payload(execution.predictions[method]))
        ).hexdigest()
        for method in ("trained_rfdetr", "coco_rfdetr", "classic_cv")
    }
    semantic_identity = {
        "schema_version": "1.0",
        "kind": "rfdetr_frozen_three_way_benchmark",
        "threshold": RFDETR_THRESHOLD,
        "matching_rule": "same_class_center_distance_lte_one_gt_radius",
        "model_contracts": {
            "trained_rfdetr": {
                "model": "RFDETRSegSmall",
                "accepted_class_id": 0,
            },
            "coco_rfdetr": {
                "model": "RFDETRSegSmall",
                "accepted_class_id": 37,
            },
            "rfdetr_version": "1.9.0",
            "classic_cv": {
                "version": "opencv-ball-locator/1.0",
                "minimum_radius_px": 10.0,
                "maximum_radius_px": 40.0,
                "foreground_distance": 28.0,
                "minimum_circularity": 0.82,
                "confidence_abstain_below": 0.5,
            },
        },
        "provenance": dict(sorted(provenance.items())),
        "frame_pixels": [
            {
                "asset_id": key[0],
                "frame_index": key[1],
                "shape": list(frames[key].shape),
                "sha256": hashlib.sha256(frames[key].tobytes()).hexdigest(),
            }
            for key in frame_keys
        ],
        "prediction_hashes": prediction_hashes,
        "metrics": {
            method: execution.metrics[method].model_dump(mode="json")
            for method in ("trained_rfdetr", "coco_rfdetr", "classic_cv")
        },
        "gates": execution.gates.model_dump(mode="json"),
    }
    publication_root = output_root / "frozen-detector-benchmarks"
    publication_root.mkdir(parents=True, exist_ok=True)
    temporary = publication_root / f".pending-{uuid4().hex}"
    temporary.mkdir()
    try:
        overlays = temporary / "overlays"
        overlays.mkdir()
        for key in frame_keys:
            for method in ("trained_rfdetr", "coco_rfdetr", "classic_cv"):
                destination = overlays / f"{key[0]}-{key[1]:08d}-{method}.jpg"
                overlay = _render_overlay(
                    image_bgr=frames[key],
                    circles=circles_by_frame[key],
                    predictions=execution.predictions[method][key],
                    method=method,
                )
                if not cv2.imwrite(str(destination), overlay):
                    raise ValueError("benchmark overlay could not be written")
        (temporary / "predictions.json").write_bytes(
            _canonical_bytes(
                {
                    method: _prediction_payload(execution.predictions[method])
                    for method in ("trained_rfdetr", "coco_rfdetr", "classic_cv")
                }
            )
            + b"\n"
        )
        (temporary / "failures.json").write_bytes(
            _canonical_bytes(
                _failure_payload(execution=execution, circles_by_frame=circles_by_frame)
            )
            + b"\n"
        )
        _write_index(
            path=temporary / "index.html",
            frames=frame_keys,
            metrics=execution.metrics,
            gates=execution.gates,
        )
        identity = {
            **semantic_identity,
            "evidence_inventory": _inventory(temporary),
        }
        artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        target = publication_root / artifact_id
        if target.exists():
            _verify_publication(target, artifact_id=artifact_id)
            return FrozenBenchmarkRun(path=target, artifact_id=artifact_id, reused=True)
        (temporary / "report.json").write_bytes(
            _canonical_bytes({"artifact_id": artifact_id, "identity": identity}) + b"\n"
        )
        manifest = {
            "schema_version": "1.0",
            "artifact_id": artifact_id,
            "inventory": _inventory(temporary),
        }
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        for item in sorted(temporary.rglob("*"), reverse=True):
            item.chmod(0o555 if item.is_dir() else 0o444)
        temporary.chmod(0o555)
        _verify_publication(
            temporary,
            artifact_id=artifact_id,
            require_artifact_directory_name=False,
        )
        try:
            os.replace(temporary, target)
        except OSError:
            if not target.exists():
                raise
            _verify_publication(target, artifact_id=artifact_id)
            return FrozenBenchmarkRun(path=target, artifact_id=artifact_id, reused=True)
        return FrozenBenchmarkRun(path=target, artifact_id=artifact_id)
    finally:
        if temporary.exists():
            _remove_publication_tree(temporary)
