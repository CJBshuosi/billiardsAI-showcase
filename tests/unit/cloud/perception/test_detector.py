"""Regression tests for contour-local classical-CV confidence."""

import cv2
import numpy as np
import pytest


def _image(*, include_strong_ball: bool) -> np.ndarray:
    image = np.empty((400, 600, 3), dtype=np.uint8)
    image[:] = (72, 132, 58)
    # Just over the foreground threshold: a valid but weak local contour.
    cv2.circle(image, (150, 200), 22, (90, 155, 70), thickness=-1, lineType=cv2.LINE_AA)
    if include_strong_ball:
        cv2.circle(image, (450, 200), 22, (242, 242, 242), thickness=-1, lineType=cv2.LINE_AA)
    return image


def _weak_confidence(image: np.ndarray) -> float:
    from billiards_ai.cloud.perception.detector import ClassicCvBallDetector

    detections = ClassicCvBallDetector().detect(image)
    return next(item.confidence for item in detections if item.raw_observation.center_px[0] < 300)


def test_contour_confidence_is_not_inflated_by_a_strong_ball_elsewhere():
    alone = _weak_confidence(_image(include_strong_ball=False))
    alongside_strong = _weak_confidence(_image(include_strong_ball=True))

    assert alongside_strong == pytest.approx(alone)
