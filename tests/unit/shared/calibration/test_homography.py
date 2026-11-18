"""Synthetic-only tests for the shared pixel-to-millimetre boundary."""

import pytest

from billiards_ai.shared.calibration.homography import (
    CalibrationAccuracyError,
    CalibrationDriftError,
    assert_no_drift,
    calibrate,
    pixel_to_table,
    validate_reference_accuracy,
)
from billiards_ai.shared.models.calibration import CalibrationReference
from billiards_ai.shared.models.geometry import Point


def test_legacy_homography_still_returns_integer_millimetres():
    calibration = calibrate(
        "device-legacy",
        [(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)],
    )

    point = pixel_to_table(calibration, 1000.4, 500.6)

    assert point.model_dump() == {"x_mm": 1000, "y_mm": 501}


def test_verified_distortion_is_corrected_before_homography():
    from billiards_ai.shared.models.calibration import DistortionModel, DistortionParameters

    distortion = DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=True,
        camera_matrix=((1000.0, 0.0, 500.0), (0.0, 1000.0, 500.0), (0.0, 0.0, 1.0)),
        coefficients=(0.2, 0.0, 0.0, 0.0, 0.0),
    )
    distorted_corners = [
        (74.4, 74.4),
        (925.6, 74.4),
        (925.6, 925.6),
        (74.4, 925.6),
    ]
    calibration = calibrate(
        "device-distorted",
        distorted_corners,
        table_length_mm=800,
        table_width_mm=800,
        distortion=distortion,
    )

    point = pixel_to_table(calibration, 701.6, 500.0)

    assert point.model_dump() == {"x_mm": 600, "y_mm": 400}


def test_unverified_non_negligible_distortion_never_converts():
    from billiards_ai.shared.calibration.homography import DistortionNotVerifiedError
    from billiards_ai.shared.models.calibration import DistortionModel, DistortionParameters

    distortion = DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=False,
        camera_matrix=((1000.0, 0.0, 500.0), (0.0, 1000.0, 500.0), (0.0, 0.0, 1.0)),
        coefficients=(0.2, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(DistortionNotVerifiedError):
        calibrate(
            "device-unverified",
            [(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)],
            table_length_mm=800,
            table_width_mm=400,
            distortion=distortion,
        )


@pytest.mark.parametrize(
    "corners",
    [
        ((0.0, 0.0),) * 4,
        ((0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)),
    ],
)
def test_degenerate_manual_corners_are_rejected(corners):
    with pytest.raises(ValueError, match="退化"):
        calibrate("device-degenerate", corners)


def test_fitted_corners_cannot_be_used_as_independent_reference():
    calibration = calibrate(
        "device-7", [(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)]
    )

    with pytest.raises(CalibrationAccuracyError, match="independent"):
        validate_reference_accuracy(
            calibration,
            (CalibrationReference(pixel=(0.0, 0.0), table=Point(x_mm=0, y_mm=0)),),
        )


def test_nonfinite_drift_or_threshold_fails_closed():
    calibration = calibrate(
        "device-7", [(0.0, 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)]
    )

    with pytest.raises(CalibrationDriftError, match="finite"):
        assert_no_drift(
            calibration,
            [(float("nan"), 0.0), (2540.0, 0.0), (2540.0, 1270.0), (0.0, 1270.0)],
        )
    with pytest.raises(CalibrationDriftError, match="finite"):
        assert_no_drift(calibration, calibration.corners_px, float("nan"))
