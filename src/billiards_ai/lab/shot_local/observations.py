"""Collect detector and coarse-identity evidence exactly once per decoded frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.lab.vision.contracts import BallInstanceObservation, make_vision_frame_input
from billiards_ai.shared.calibration.homography import TableCalibration, pixel_to_table


@dataclass(frozen=True, slots=True)
class DecodedTrackingFrame:
    frame_id: int
    timestamp_ms: int
    image: np.ndarray


class GroupIdentityClassifier(Protocol):
    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> Mapping[str, Mapping[str, Any]]: ...


def _group_evidence(
    row: Mapping[str, Any],
) -> tuple[BallGroup, float]:
    try:
        group = BallGroup(str(row["group"]))
        confidence = float(row["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("identity group is invalid") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("identity group confidence is invalid")
    return group, confidence


def collect_observation_frames(
    *,
    frames: Sequence[DecodedTrackingFrame],
    vision_backend: VisionBackend,
    identity_classifier: GroupIdentityClassifier,
    calibration: TableCalibration,
) -> tuple[ObservationFrame, ...]:
    """Build the only detector/classifier cache consumed by shot-local stages."""

    collected = []
    for frame in frames:
        observation = vision_backend.infer(
            make_vision_frame_input(
                frame.image,
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
            )
        )
        if observation.frame_id != frame.frame_id or observation.timestamp_ms != frame.timestamp_ms:
            raise ValueError("vision observation does not match decoded frame")
        classifications = identity_classifier.classify(
            image=frame.image,
            detections=observation.detections,
        )
        expected_ids = {item.detection_id for item in observation.detections}
        if set(classifications) != expected_ids:
            raise ValueError("identity classifications must match detections exactly")

        calibrated = []
        for detection in observation.detections:
            point = pixel_to_table(
                calibration,
                detection.center.x,
                detection.center.y,
            )
            group, group_confidence = _group_evidence(classifications[detection.detection_id])
            calibrated.append(
                CalibratedBallObservation(
                    detection_id=detection.detection_id,
                    x_mm=point.x_mm,
                    y_mm=point.y_mm,
                    confidence=detection.confidence,
                    group=group,
                    group_confidence=group_confidence,
                    bbox_xyxy_px=(
                        round(detection.bbox.left),
                        round(detection.bbox.top),
                        round(detection.bbox.right),
                        round(detection.bbox.bottom),
                    ),
                )
            )
        collected.append(
            ObservationFrame(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                observations=tuple(sorted(calibrated, key=lambda item: item.detection_id)),
            )
        )
    return tuple(collected)
