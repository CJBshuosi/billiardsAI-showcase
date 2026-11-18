from __future__ import annotations

import numpy as np
import pytest

from billiards_ai.lab.mainline1_lens_calibration import (
    LensCalibrationError,
    LensCalibrationSample,
    calibrate_opencv_lens,
)
from billiards_ai.shared.models.calibration import DistortionModel


def _sample(offset: float) -> LensCalibrationSample:
    object_points = tuple(
        (float(column), float(row), 0.0) for row in range(6) for column in range(9)
    )
    image_points = tuple(
        (100.0 + column * 20.0 + offset, 80.0 + row * 20.0 + offset / 2)
        for row in range(6)
        for column in range(9)
    )
    return LensCalibrationSample(
        object_points=object_points,
        image_points=image_points,
    )


def test_lens_calibration_rejects_insufficient_pose_diversity() -> None:
    with pytest.raises(LensCalibrationError, match="pose diversity"):
        calibrate_opencv_lens(
            samples=tuple(_sample(0) for _ in range(8)),
            width=1920,
            height=1080,
        )


def test_lens_calibration_builds_verified_opencv_parameters() -> None:
    camera_matrix = np.array(
        ((900.0, 0.0, 960.0), (0.0, 900.0, 540.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    coefficients = np.array(((-0.2, 0.04, 0.0, 0.0, 0.0),), dtype=np.float64)

    result = calibrate_opencv_lens(
        samples=tuple(_sample(float(index * 12)) for index in range(8)),
        width=1920,
        height=1080,
        calibrator=lambda *args, **kwargs: (
            0.42,
            camera_matrix,
            coefficients,
            (),
            (),
        ),
    )

    assert result.rms_reprojection_error == 0.42
    assert result.sample_count == 8
    assert result.distortion.model is DistortionModel.OPENCV
    assert result.distortion.verified is True
    assert result.distortion.coefficients == (-0.2, 0.04, 0.0, 0.0, 0.0)


def test_lens_calibration_rejects_non_finite_result() -> None:
    with pytest.raises(LensCalibrationError, match="finite"):
        calibrate_opencv_lens(
            samples=tuple(_sample(float(index * 12)) for index in range(8)),
            width=1920,
            height=1080,
            calibrator=lambda *args, **kwargs: (
                float("nan"),
                np.eye(3),
                np.zeros((1, 5)),
                (),
                (),
            ),
        )
