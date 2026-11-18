from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    collect_observation_frames,
)
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate


def _decoded_frame(frame_id: int) -> DecodedTrackingFrame:
    return DecodedTrackingFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )


def _detection(detection_id: str, center_x: float) -> BallInstanceObservation:
    return BallInstanceObservation(
        detection_id=detection_id,
        bbox=PixelBox(left=center_x - 4, top=46, right=center_x + 4, bottom=54),
        center=PixelPoint(x=center_x, y=50),
        confidence=0.9,
        class_label="ball",
    )


class RecordingVisionBackend:
    package_identity = VisionPackageIdentity(
        package_id="a" * 64,
        manifest_sha256="b" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer(self, frame) -> VisionFrameObservation:
        self.calls.append(frame.frame_id)
        detections = (
            _detection(f"det-{frame.frame_id}-b", 150),
            _detection(f"det-{frame.frame_id}-a", 50),
        )
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            package=self.package_identity,
            threshold=0.25,
            detections=detections,
            runtime_ref="synthetic/1",
        )


class RecordingClassifier:
    def __init__(self, *, group: str = "cue", drop_last: bool = False) -> None:
        self.group = group
        self.drop_last = drop_last
        self.calls: list[int] = []

    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> Mapping[str, Mapping[str, Any]]:
        self.calls.append(len(detections))
        selected = detections[:-1] if self.drop_last else detections
        return {
            detection.detection_id: {"group": self.group, "confidence": 0.8}
            for detection in selected
        }


def _calibration():
    return calibrate(
        "test-camera",
        ((0, 100), (200, 100), (200, 0), (0, 0)),
        table_length_mm=2000,
        table_width_mm=1000,
    )


def test_collect_observations_runs_eye_and_classifier_once_per_frame() -> None:
    vision = RecordingVisionBackend()
    classifier = RecordingClassifier(group="cue")

    result = collect_observation_frames(
        frames=(_decoded_frame(0), _decoded_frame(1)),
        vision_backend=vision,
        identity_classifier=classifier,
        calibration=_calibration(),
    )

    assert [frame.frame_id for frame in result] == [0, 1]
    assert vision.calls == [0, 1]
    assert classifier.calls == [2, 2]
    assert [item.detection_id for item in result[0].observations] == ["det-0-a", "det-0-b"]
    assert result[0].observations[0].group is BallGroup.CUE
    assert (result[0].observations[0].x_mm, result[0].observations[0].y_mm) == (500, 500)


def test_classifier_keys_must_match_detector_ids_exactly() -> None:
    with pytest.raises(ValueError, match="match detections exactly"):
        collect_observation_frames(
            frames=(_decoded_frame(0),),
            vision_backend=RecordingVisionBackend(),
            identity_classifier=RecordingClassifier(drop_last=True),
            calibration=_calibration(),
        )


def test_unsupported_group_fails_closed() -> None:
    with pytest.raises(ValueError, match="identity group is invalid"):
        collect_observation_frames(
            frames=(_decoded_frame(0),),
            vision_backend=RecordingVisionBackend(),
            identity_classifier=RecordingClassifier(group="purple"),
            calibration=_calibration(),
        )
