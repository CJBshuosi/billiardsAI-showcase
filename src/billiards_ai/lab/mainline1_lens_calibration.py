"""OpenCV lens calibration for a fixed Mainline 1 USB camera."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from billiards_ai.shared.models.calibration import (
    DistortionModel,
    DistortionParameters,
)


class LensCalibrationError(ValueError):
    """The sample set cannot establish a trustworthy lens model."""


@dataclass(frozen=True)
class LensCalibrationSample:
    object_points: tuple[tuple[float, float, float], ...]
    image_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.object_points) < 4 or len(self.object_points) != len(self.image_points):
            raise LensCalibrationError("lens sample requires matching object/image points")
        values = tuple(value for point in self.object_points for value in point) + tuple(
            value for point in self.image_points for value in point
        )
        if not all(math.isfinite(value) for value in values):
            raise LensCalibrationError("lens sample points must be finite")


@dataclass(frozen=True)
class LensCalibrationResult:
    distortion: DistortionParameters
    rms_reprojection_error: float
    sample_count: int
    width: int
    height: int


Calibrator = Callable[..., tuple[Any, Any, Any, Any, Any]]


def calibrate_opencv_lens(
    *,
    samples: tuple[LensCalibrationSample, ...],
    width: int,
    height: int,
    calibrator: Calibrator = cv2.calibrateCamera,
    maximum_rms_error: float = 2.0,
) -> LensCalibrationResult:
    """Fit a verified OpenCV distortion model from diverse checkerboard poses."""
    if width <= 0 or height <= 0:
        raise LensCalibrationError("calibration image geometry must be positive")
    if len(samples) < 8:
        raise LensCalibrationError("at least eight lens calibration samples are required")
    centroids = np.asarray([np.mean(np.asarray(sample.image_points), axis=0) for sample in samples])
    if np.ptp(centroids[:, 0]) < width * 0.03 or np.ptp(centroids[:, 1]) < height * 0.02:
        raise LensCalibrationError("lens calibration samples lack pose diversity")

    object_points = [np.asarray(sample.object_points, dtype=np.float32) for sample in samples]
    image_points = [np.asarray(sample.image_points, dtype=np.float32) for sample in samples]
    try:
        rms, camera_matrix, coefficients, _rotations, _translations = calibrator(
            object_points, image_points, (width, height), None, None
        )
    except (cv2.error, TypeError, ValueError) as exc:
        raise LensCalibrationError("OpenCV lens calibration failed") from exc

    matrix = np.asarray(camera_matrix, dtype=np.float64)
    flattened = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    numeric_rms = float(rms)
    if (
        not math.isfinite(numeric_rms)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(flattened))
    ):
        raise LensCalibrationError("lens calibration result must be finite")
    if numeric_rms < 0 or numeric_rms > maximum_rms_error:
        raise LensCalibrationError("lens calibration reprojection error is too high")
    if matrix.shape != (3, 3):
        raise LensCalibrationError("lens calibration camera matrix is invalid")
    if flattened.size not in {4, 5, 8, 12, 14}:
        raise LensCalibrationError("lens calibration coefficients are invalid")

    distortion = DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=True,
        camera_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        coefficients=tuple(float(value) for value in flattened),
    )
    return LensCalibrationResult(
        distortion=distortion,
        rms_reprojection_error=numeric_rms,
        sample_count=len(samples),
        width=width,
        height=height,
    )
