"""Runtime adapters for the already-trained temporary Mainline 1 eye."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Protocol

import cv2
import numpy as np

from billiards_ai.lab.mainline1_identity_preprocessing import (
    normalise_identity_features as _normalise,
)
from billiards_ai.lab.mainline1_identity_preprocessing import square_identity_crop as _square_crop
from billiards_ai.lab.vision.contracts import BallInstanceObservation, VisionPackageIdentity
from billiards_ai.lab.vision.package import (
    VerifiedVisionPackage,
    VisionArtifact,
    VisionModelPackageManifest,
    VisionPostprocess,
    VisionPreprocess,
)
from billiards_ai.lab.vision.rfdetr import RfdetrVisionBackend


class RfdetrDetectionsAdapter:
    """End RF-DETR's private Detections object at the existing vision boundary."""

    def __init__(self, model: object) -> None:
        self._model = model

    def predict(
        self,
        image: np.ndarray | list[np.ndarray],
        *,
        threshold: float,
    ) -> list[dict[str, object]] | list[list[dict[str, object]]]:
        if isinstance(image, list):
            predictions = self._model.predict(  # type: ignore[attr-defined]
                [cv2.cvtColor(item, cv2.COLOR_BGR2RGB) for item in image],
                threshold=threshold,
                include_source_image=False,
            )
            if not isinstance(predictions, list) or len(predictions) != len(image):
                raise ValueError("RF-DETR batch response count is invalid")
            return [self._rows(item) for item in predictions]
        predictions = self._model.predict(  # type: ignore[attr-defined]
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            threshold=threshold,
            include_source_image=False,
        )
        return self._rows(predictions)

    @staticmethod
    def _rows(predictions: object) -> list[dict[str, object]]:
        return [
            {
                "bbox_xyxy": [float(value) for value in predictions.xyxy[index]],
                "confidence": float(predictions.confidence[index]),
                "class_id": int(predictions.class_id[index]),
            }
            for index in range(len(predictions))
        ]


class IdentityFeatureBackend(Protocol):
    def embed(self, crops: tuple[Path, ...]) -> np.ndarray: ...


class IdentityEvidenceBackend(Protocol):
    feature_names: tuple[str, ...]

    def extract(self, crops: tuple[Path, ...]) -> np.ndarray: ...


class OnnxSession(Protocol):
    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> Any: ...


class OnnxGroupIdentityClassifier:
    """Apply the existing DINOv2 + visual-evidence group head to detector crops."""

    _LABELS = ("cue", "solid", "eight", "stripe")

    def __init__(
        self,
        *,
        feature_backend: IdentityFeatureBackend,
        evidence_backend: IdentityEvidenceBackend,
        session: OnnxSession,
        confidence_min: float,
        margin_min: float,
        known_probability_min: float,
        visual_evidence_weight: float,
    ) -> None:
        if (
            any(
                not math.isfinite(value) or value < 0 or value > 1
                for value in (confidence_min, margin_min, known_probability_min)
            )
            or not math.isfinite(visual_evidence_weight)
            or visual_evidence_weight <= 0
        ):
            raise ValueError("group runtime configuration is invalid")
        self._feature_backend = feature_backend
        self._evidence_backend = evidence_backend
        self._session = session
        self._confidence_min = confidence_min
        self._margin_min = margin_min
        self._known_probability_min = known_probability_min
        self._visual_evidence_weight = visual_evidence_weight

    @staticmethod
    def _write_crops(
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
        root: Path,
    ) -> tuple[Path, ...]:
        paths = []
        for index, detection in enumerate(detections):
            box = detection.bbox
            crop = _square_crop(
                image,
                (box.left, box.top, box.right - box.left, box.bottom - box.top),
                context_scale=2.0,
                output_size=224,
            )
            path = root / f"crop-{index:04d}.png"
            if not cv2.imwrite(str(path), crop):
                raise ValueError("identity runtime crop cannot be encoded")
            paths.append(path)
        return tuple(paths)

    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> dict[str, dict[str, object]]:
        if not detections:
            return {}
        with TemporaryDirectory(prefix="billiards-mainline1-identity-") as directory:
            crops = self._write_crops(image, detections, Path(directory))
            semantic = _normalise(self._feature_backend.embed(crops))
            evidence = np.asarray(self._evidence_backend.extract(crops), dtype=np.float32)
        if evidence.shape != (len(crops), len(self._evidence_backend.feature_names)):
            raise ValueError("identity runtime evidence shape is invalid")
        features = _normalise(
            np.concatenate((semantic, (evidence - 0.5) * self._visual_evidence_weight), axis=1)
        ).astype(np.float32)
        probabilities, known = self._session.run(
            ["group_probabilities", "known_probability"], {"features": features}
        )
        probabilities = np.asarray(probabilities, dtype=np.float32)
        known = np.asarray(known, dtype=np.float32).reshape(-1)
        if probabilities.shape != (len(crops), 4) or known.shape != (len(crops),):
            raise ValueError("identity runtime ONNX output shape is invalid")

        output = {}
        for detection, row, known_probability in zip(detections, probabilities, known, strict=True):
            ordered = np.sort(row)
            confidence = float(ordered[-1])
            margin = float(ordered[-1] - ordered[-2])
            group = self._LABELS[int(np.argmax(row))]
            if (
                confidence < self._confidence_min
                or margin < self._margin_min
                or float(known_probability) < self._known_probability_min
            ):
                group = "unknown"
            output[detection.detection_id] = {
                "group": group,
                "confidence": confidence,
                "exact_identity": None,
            }
        return output


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


_STAT_PLATFORM = os.name
_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


class _ResourceChangedError(ValueError):
    """A resource identity failure with diagnostics safe for startup errors."""


def _check_read_stats(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
    phase: str,
    cross_api: bool = False,
) -> None:
    fields = _STAT_FIELDS
    if _STAT_PLATFORM == "nt":
        # Windows path stat ctime is creation time in supported CPython versions;
        # fstat may expose change time. Birthtime has matching meanings instead.
        fields = (*(_STAT_FIELDS[:-1] if cross_api else _STAT_FIELDS), "st_birthtime_ns")
    differences = []
    for field in fields:
        left, right = getattr(before, field, None), getattr(after, field, None)
        if not isinstance(left, int) or not isinstance(right, int):
            differences.append(f"{field}=missing")
        elif left != right:
            differences.append(f"{field}={left}->{right}")
    if differences:
        raise _ResourceChangedError(f"{label} changed {phase}: {', '.join(differences)}")


def _safe_stat(path: Path) -> os.stat_result:
    if ".project-archive" in path.parts:
        raise ValueError("private archive cannot be a runtime resource")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("runtime resource path is unsafe")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("runtime resource is not a regular file")
    return info


@contextmanager
def _guarded_resource_read(path: Path, *, label: str) -> Iterator[BinaryIO]:
    path_before = _safe_stat(path)
    with path.open("rb") as source:
        fd_before = os.fstat(source.fileno())
        _check_read_stats(
            path_before,
            fd_before,
            label=label,
            phase="before read (path/descriptor)",
            cross_api=True,
        )
        yield source
        fd_after = os.fstat(source.fileno())
        path_after = _safe_stat(path)
        _check_read_stats(fd_before, fd_after, label=label, phase="during read (descriptor)")
        _check_read_stats(path_before, path_after, label=label, phase="after read (path)")
        _check_read_stats(
            path_after, fd_after, label=label, phase="after read (path/descriptor)", cross_api=True
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _guarded_resource_read(path, label="runtime resource") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with _guarded_resource_read(path, label="runtime metadata") as source:
            raw = source.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("runtime metadata is oversized or changed")
        payload = json.loads(raw.decode("utf-8"))
    except _ResourceChangedError as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def inspect_development_vision_package(
    training_run: Path,
    *,
    threshold: float = 0.25,
) -> VerifiedVisionPackage:
    """Adapt one verified local short-training run into the pluggable eye contract."""

    if ".project-archive" in training_run.parts:
        raise ValueError("private archive cannot be a runtime resource")
    if any(p.is_symlink() for p in (training_run, *training_run.parents)):
        raise ValueError("detector development root is unsafe")
    root = training_run.resolve()
    manifest_path = root / "training-manifest.json"
    manifest_hash = _sha256(manifest_path)
    manifest = _json_object(manifest_path, label="detector training manifest")
    checkpoint = root / "checkpoint_best.pth"
    artifacts = manifest.get("artifacts")
    backend_identity = manifest.get("backend")
    base_weight_sha256 = (
        backend_identity.get("weight_sha256") if isinstance(backend_identity, dict) else None
    )
    if (
        training_run.is_symlink()
        or not root.is_dir()
        or root.name != manifest.get("run_hash")
        or not isinstance(artifacts, dict)
        or artifacts.get("checkpoint_best.pth") != _sha256(checkpoint)
        or manifest.get("model_class") != "RFDETRSegSmall"
        or not isinstance(base_weight_sha256, str)
        or len(base_weight_sha256) != 64
        or any(character not in "0123456789abcdef" for character in base_weight_sha256)
    ):
        raise ValueError("detector development training run changed")
    checkpoint_hash = _sha256(checkpoint)
    if (
        checkpoint_hash != artifacts["checkpoint_best.pth"]
        or _sha256(manifest_path) != manifest_hash
    ):
        raise ValueError("detector development assets changed during inspection")
    package_manifest = VisionModelPackageManifest(
        package_id=root.name,
        task="ball_instance_segmentation",
        family="rfdetr",
        model_version="1.9.0-development-short",
        model_class="RFDETRSegSmall",
        artifacts=(
            VisionArtifact(
                role="checkpoint",
                relative_path="checkpoint_best.pth",
                byte_count=checkpoint.stat().st_size,
                sha256=checkpoint_hash,
            ),
        ),
        preprocess=VisionPreprocess(color_space="RGB", input_width_px=384, input_height_px=384),
        postprocess=VisionPostprocess(confidence_threshold=threshold, mask_threshold=0.5),
        class_map={"ball": 0},
        runtime_requirements=("rfdetr==1.9.0",),
        training_run_sha256=manifest_hash,
        code_sha256=hashlib.sha256(b"mainline1-development-runtime/1").hexdigest(),
        base_weight_sha256=base_weight_sha256,
        asset_status_ref="local-development-short-not-release-eligible",
    )
    package = VerifiedVisionPackage(
        root=root,
        manifest_path=manifest_path,
        identity=VisionPackageIdentity(
            package_id=root.name,
            manifest_sha256=manifest_hash,
        ),
        manifest=package_manifest,
        checkpoint=checkpoint,
    )
    return package


def load_development_vision_backend(
    training_run: Path,
    *,
    model_factory: Any,
    threshold: float = 0.25,
) -> RfdetrVisionBackend:
    package = inspect_development_vision_package(training_run, threshold=threshold)
    return RfdetrVisionBackend(
        package,
        lambda path: RfdetrDetectionsAdapter(model_factory(Path(path))),
    )


@dataclass(frozen=True, slots=True)
class VerifiedGroupRuntimeAssets:
    root: Path
    manifest_path: Path
    metrics_path: Path
    onnx_path: Path
    manifest_sha256: str
    confidence_min: float
    margin_min: float
    known_probability_min: float
    visual_evidence_weight: float


def inspect_development_group_assets(training_run: Path) -> VerifiedGroupRuntimeAssets:
    """Load the reviewed group-only development head without treating gate FAIL as release proof."""

    if ".project-archive" in training_run.parts:
        raise ValueError("private archive cannot be a runtime resource")
    if any(p.is_symlink() for p in (training_run, *training_run.parents)):
        raise ValueError("group development root is unsafe")
    root = training_run.resolve()
    manifest_path = root / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    manifest = _json_object(manifest_path, label="group training manifest")
    metrics_path = root / "metrics.json"
    onnx_path = root / "group-head.onnx"
    inventory = manifest.get("inventory")
    configuration = manifest.get("backend_configuration")
    if (
        training_run.is_symlink()
        or not root.is_dir()
        or root.name != manifest.get("run_id")
        or manifest.get("status") != "PASS"
        or not isinstance(inventory, dict)
        or not isinstance(configuration, dict)
        or configuration.get("architecture") != "hierarchical_fused_evidence_mlp"
    ):
        raise ValueError("group development training run changed")
    for name, path in (("group-head.onnx", onnx_path), ("metrics.json", metrics_path)):
        record = inventory.get(name)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
            raise ValueError("group development training inventory changed")
    metrics = _json_object(metrics_path, label="group training metrics")
    unknown_policy = metrics.get("unknown_policy")
    visual_policy = metrics.get("visual_evidence_policy")
    if not isinstance(unknown_policy, dict) or not isinstance(visual_policy, dict):
        raise ValueError("group development runtime policy is missing")
    try:
        values = tuple(
            float(unknown_policy[name])
            for name in ("confidence_min", "margin_min", "known_probability_min")
        )
        weight = float(visual_policy["weight"])
        if (
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("group runtime configuration is invalid")
        if (
            _sha256(manifest_path) != manifest_hash
            or _sha256(metrics_path) != inventory["metrics.json"]["sha256"]
            or _sha256(onnx_path) != inventory["group-head.onnx"]["sha256"]
        ):
            raise ValueError("group development assets changed during inspection")
        return VerifiedGroupRuntimeAssets(
            root,
            manifest_path,
            metrics_path,
            onnx_path,
            manifest_hash,
            *values,
            weight,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("group development runtime policy is invalid") from exc


def load_development_group_classifier(
    training_run: Path,
    *,
    feature_backend: IdentityFeatureBackend,
    evidence_backend: IdentityEvidenceBackend,
    session_factory: Any,
) -> OnnxGroupIdentityClassifier:
    assets = inspect_development_group_assets(training_run)
    try:
        return OnnxGroupIdentityClassifier(
            feature_backend=feature_backend,
            evidence_backend=evidence_backend,
            session=session_factory(assets.onnx_path),
            confidence_min=assets.confidence_min,
            margin_min=assets.margin_min,
            known_probability_min=assets.known_probability_min,
            visual_evidence_weight=assets.visual_evidence_weight,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("group development runtime policy is invalid") from exc
