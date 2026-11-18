"""Calibrated-table visual motion evidence independent of detector identities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from billiards_ai.shared.calibration.homography import TableCalibration


@dataclass(frozen=True, slots=True)
class VisualMotionConfig:
    pixel_delta_threshold: int = 18
    changed_pixel_ratio: float = 0.0005

    def __post_init__(self) -> None:
        if not 1 <= self.pixel_delta_threshold <= 255:
            raise ValueError("pixel delta threshold must be between 1 and 255")
        if not math.isfinite(self.changed_pixel_ratio) or not 0 < self.changed_pixel_ratio <= 1:
            raise ValueError("changed pixel ratio must be in (0, 1]")


def table_visual_motion_ratio(
    previous: np.ndarray,
    current: np.ndarray,
    calibration: TableCalibration,
    config: VisualMotionConfig,
) -> float:
    """Return changed table-pixel ratio after suppressing compression speckle."""

    if (
        previous.dtype != np.uint8
        or current.dtype != np.uint8
        or previous.ndim != 3
        or current.ndim != 3
        or previous.shape != current.shape
        or previous.shape[2] != 3
    ):
        raise ValueError("visual motion frames must be matching uint8 BGR images")
    mask = _table_mask(current.shape[:2], calibration)
    previous_gray = cv2.GaussianBlur(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    current_gray = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return _changed_ratio(previous_gray, current_gray, mask, config)


def _table_mask(shape, calibration):
    height, width = shape
    polygon = np.asarray(calibration.corners_px, dtype=np.float32)
    if (
        polygon.shape != (4, 2)
        or not np.isfinite(polygon).all()
        or np.any(polygon[:, 0] < 0)
        or np.any(polygon[:, 0] > width)
        or np.any(polygon[:, 1] < 0)
        or np.any(polygon[:, 1] > height)
    ):
        raise ValueError("calibration table ROI is outside the visual motion frame")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    table_pixels = int(cv2.countNonZero(mask))
    if table_pixels <= 0:
        raise ValueError("calibration table ROI is empty")
    return mask


def _changed_ratio(previous_gray, current_gray, mask, config):
    delta = cv2.absdiff(previous_gray, current_gray)
    changed = cv2.threshold(
        delta,
        config.pixel_delta_threshold,
        255,
        cv2.THRESH_BINARY,
    )[1]
    changed = cv2.bitwise_and(changed, mask)
    changed = cv2.morphologyEx(
        changed,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    return cv2.countNonZero(changed) / cv2.countNonZero(mask)


def table_visual_motion_detected(
    previous: np.ndarray,
    current: np.ndarray,
    calibration: TableCalibration,
    config: VisualMotionConfig,
) -> bool:
    ratio = table_visual_motion_ratio(previous, current, calibration, config)
    return ratio >= config.changed_pixel_ratio


class RecordingMotionGap(ValueError):
    """Motion continuity is unknown across an observed camera timestamp gap."""


class RecordingVisualMotion:
    """Confirm fresh displacement within ordinary recording's stability horizon.

    Advance the reference after each confirmation so a held scene change cannot
    confirm movement repeatedly. Subthreshold displacement can accumulate for
    at most one stability window, using one blurred gray reference.
    """

    def __init__(self, calibration, config, *, still_duration_ms):
        if type(still_duration_ms) is not int or still_duration_ms <= 0:
            raise ValueError("recording motion horizon is invalid")
        self._calibration, self._config = calibration, config
        self._duration = still_duration_ms
        self._reference = None
        self._mask = None
        self._last_timestamp_ms = None

    def reset(self):
        self._reference = None
        self._last_timestamp_ms = None

    def push(self, image, *, timestamp_ms):
        if (
            type(timestamp_ms) is not int
            or timestamp_ms < 0
            or (self._last_timestamp_ms is not None and timestamp_ms < self._last_timestamp_ms)
        ):
            raise ValueError("recording motion timestamps must be ordered")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms - self._last_timestamp_ms > self._duration
        ):
            raise RecordingMotionGap("recording motion sample gap exceeds stability horizon")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("recording motion requires uint8 BGR images")
        if self._mask is None:
            self._mask = _table_mask(image.shape[:2], self._calibration)
        elif image.shape[:2] != self._mask.shape:
            raise ValueError("recording motion geometry changed")
        current = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        moving = (
            self._reference is not None
            and _changed_ratio(self._reference[1], current, self._mask, self._config)
            >= self._config.changed_pixel_ratio
        )
        if moving or self._reference is None or timestamp_ms - self._reference[0] >= self._duration:
            self._reference = (timestamp_ms, current)
        self._last_timestamp_ms = timestamp_ms
        return moving
