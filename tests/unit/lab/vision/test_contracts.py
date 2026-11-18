from __future__ import annotations

import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
    make_vision_frame_input,
)


def _detection(
    *,
    detection_id: str = "detection-0000",
    confidence: float = 0.9,
    left: float = 2,
    top: float = 3,
    right: float = 12,
    bottom: float = 13,
) -> BallInstanceObservation:
    return BallInstanceObservation(
        detection_id=detection_id,
        bbox=PixelBox(left=left, top=top, right=right, bottom=bottom),
        center=PixelPoint(x=(left + right) / 2, y=(top + bottom) / 2),
        confidence=confidence,
        class_label="ball",
    )


def _observation(*, detections: tuple[BallInstanceObservation, ...]) -> VisionFrameObservation:
    return VisionFrameObservation(
        frame_id=7,
        timestamp_ms=233,
        width_px=32,
        height_px=24,
        pixel_sha256="a" * 64,
        package=VisionPackageIdentity(
            package_id="b" * 64,
            manifest_sha256="c" * 64,
        ),
        threshold=0.25,
        detections=detections,
        runtime_ref="synthetic-runtime/1",
    )


def test_frame_factory_copies_pixels_and_binds_their_hash() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)

    frame = make_vision_frame_input(image, frame_id=7, timestamp_ms=233)
    image[0, 0] = 255

    assert frame.width_px == 32
    assert frame.height_px == 24
    assert frame.image.flags.writeable is False
    assert frame.image[0, 0].tolist() == [0, 0, 0]
    assert frame.pixel_sha256 == hashlib.sha256(frame.image.tobytes()).hexdigest()


def test_observation_accepts_zero_detections() -> None:
    observation = _observation(detections=())

    assert observation.detections == ()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_detection_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        _detection(confidence=confidence)


def test_observation_rejects_duplicate_ids() -> None:
    detection = _detection()

    with pytest.raises(ValidationError, match="unique"):
        _observation(detections=(detection, detection))


def test_observation_rejects_out_of_frame_geometry() -> None:
    with pytest.raises(ValidationError, match="source frame"):
        _observation(detections=(_detection(right=33),))


def test_contracts_are_immutable() -> None:
    observation = _observation(detections=(_detection(),))

    with pytest.raises(ValidationError):
        observation.threshold = 0.8  # type: ignore[misc]
