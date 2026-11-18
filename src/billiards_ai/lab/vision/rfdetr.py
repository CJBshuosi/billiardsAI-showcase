"""Explicit RF-DETR adapter that ends all model-private output at the vision boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
from pydantic import ValidationError

from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameInput,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.lab.vision.package import VerifiedVisionPackage


class VisionInferenceError(RuntimeError):
    """Raised when a private backend violates the public observation contract."""


class RfdetrPredictionModel(Protocol):
    def predict(
        self,
        image: np.ndarray | list[np.ndarray],
        *,
        threshold: float,
    ) -> Sequence[object] | Sequence[Sequence[object]]: ...


def _private_value(prediction: object, name: str) -> object:
    if isinstance(prediction, Mapping):
        if name not in prediction:
            raise VisionInferenceError(f"RF-DETR prediction is missing {name}")
        return prediction[name]
    try:
        return getattr(prediction, name)
    except AttributeError as error:
        raise VisionInferenceError(f"RF-DETR prediction is missing {name}") from error


def _normalized_prediction(prediction: object) -> tuple[float, float, float, float, float, int]:
    try:
        raw_box = _private_value(prediction, "bbox_xyxy")
        if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
            raise TypeError("bbox must be a sequence")
        if len(raw_box) != 4:
            raise ValueError("bbox must contain four coordinates")
        left, top, right, bottom = (float(value) for value in raw_box)
        confidence = float(_private_value(prediction, "confidence"))
        class_id = int(_private_value(prediction, "class_id"))
    except (TypeError, ValueError, OverflowError) as error:
        raise VisionInferenceError("RF-DETR prediction is malformed") from error
    values = (left, top, right, bottom, confidence)
    if not all(math.isfinite(value) for value in values):
        raise VisionInferenceError("RF-DETR prediction contains a non-finite value")
    return left, top, right, bottom, confidence, class_id


class RfdetrVisionBackend:
    def __init__(
        self,
        package: VerifiedVisionPackage,
        model_factory: Callable[[object], RfdetrPredictionModel],
    ) -> None:
        if package.manifest.family != "rfdetr":
            raise VisionInferenceError("vision package family must be rfdetr")
        self._package = package
        self._model = model_factory(package.checkpoint)
        self._execution_prepared = False

    @property
    def package_identity(self) -> VisionPackageIdentity:
        return self._package.identity

    def prepare_execution(self, *, width_px: int, height_px: int) -> None:
        """Prepare the original prediction path once without a scene observation.

        This accepts only geometry. All private outputs from the fixed empty
        image are discarded before the public observation boundary.
        """
        if any(type(size) is not int or not 1 <= size <= 4096 for size in (width_px, height_px)):
            raise ValueError("invalid execution preparation geometry")
        if self._execution_prepared:
            return
        empty = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        empty.flags.writeable = False
        self._model.predict(
            empty, threshold=self._package.manifest.postprocess.confidence_threshold
        )
        self._execution_prepared = True

    def infer(self, frame: VisionFrameInput) -> VisionFrameObservation:
        threshold = self._package.manifest.postprocess.confidence_threshold
        try:
            raw_predictions = self._model.predict(frame.image, threshold=threshold)
            return self._observation(frame, raw_predictions, threshold=threshold)
        except VisionInferenceError:
            raise
        except (ValidationError, TypeError, ValueError) as error:
            raise VisionInferenceError("RF-DETR output violates the vision contract") from error

    def infer_batch(
        self,
        frames: Sequence[VisionFrameInput],
    ) -> tuple[VisionFrameObservation, ...]:
        if not frames:
            return ()
        threshold = self._package.manifest.postprocess.confidence_threshold
        try:
            raw_batches = self._model.predict(
                [frame.image for frame in frames],
                threshold=threshold,
            )
            if len(raw_batches) != len(frames):
                raise VisionInferenceError("RF-DETR batch result count is invalid")
            return tuple(
                self._observation(frame, predictions, threshold=threshold)
                for frame, predictions in zip(frames, raw_batches, strict=True)
            )
        except VisionInferenceError:
            raise
        except (ValidationError, TypeError, ValueError) as error:
            raise VisionInferenceError(
                "RF-DETR batch output violates the vision contract"
            ) from error

    def _observation(
        self,
        frame: VisionFrameInput,
        raw_predictions: Sequence[object],
        *,
        threshold: float,
    ) -> VisionFrameObservation:
        normalized = sorted(
            (_normalized_prediction(item) for item in raw_predictions),
            key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
        )
        detections = tuple(
            self._to_observation(frame.frame_id, index, values)
            for index, values in enumerate(normalized)
        )
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=frame.pixel_sha256,
            package=self._package.identity,
            threshold=threshold,
            detections=detections,
            runtime_ref=f"rfdetr/{self._package.manifest.model_version}",
        )

    def _to_observation(
        self,
        frame_id: int,
        index: int,
        values: tuple[float, float, float, float, float, int],
    ) -> BallInstanceObservation:
        left, top, right, bottom, confidence, class_id = values
        if self._package.manifest.class_map.get("ball") != class_id:
            raise VisionInferenceError("RF-DETR prediction uses an unsupported class ID")
        return BallInstanceObservation(
            detection_id=f"frame-{frame_id:06d}-detection-{index:04d}",
            bbox=PixelBox(left=left, top=top, right=right, bottom=bottom),
            center=PixelPoint(x=(left + right) / 2, y=(top + bottom) / 2),
            confidence=confidence,
            class_label="ball",
        )
