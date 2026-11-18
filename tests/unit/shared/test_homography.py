import pytest

from billiards_ai.shared.calibration.homography import (
    calibrate,
    pixel_to_table,
    table_to_pixel,
)
from billiards_ai.shared.models.calibration import DistortionModel, DistortionParameters


def test_table_to_pixel_round_trips_calibration() -> None:
    calibration = calibrate("cam", [(10, 210), (410, 210), (410, 10), (10, 10)])
    point = pixel_to_table(calibration, 210, 110)
    assert table_to_pixel(calibration, point.x_mm, point.y_mm) == pytest.approx((210, 110), abs=1)


def test_table_to_pixel_round_trips_verified_opencv_distortion() -> None:
    distortion = DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=True,
        camera_matrix=(
            (812.8579598950668, 0.0, 943.185122995636),
            (0.0, 812.8429264333889, 545.2009403149398),
            (0.0, 0.0, 1.0),
        ),
        coefficients=(
            0.05686672213700196,
            -0.0693837862694251,
            0.00010171088484100014,
            0.000523711574196503,
            0.02514533249219027,
        ),
    )
    calibration = calibrate(
        "rer-u3",
        [(100.16, 972.61), (1806.99, 995.28), (1788.20, 112.24), (132.41, 106.42)],
        distortion=distortion,
    )

    for raw_pixel in ((170.0, 160.0), (960.0, 540.0), (1710.0, 920.0)):
        point = pixel_to_table(calibration, *raw_pixel)
        assert table_to_pixel(calibration, point.x_mm, point.y_mm) == pytest.approx(
            raw_pixel, abs=1.5
        )


def test_table_to_pixel_rejects_singular_matrix() -> None:
    calibration = calibrate("cam", [(10, 210), (410, 210), (410, 10), (10, 10)])
    invalid = calibration.model_copy(update={"matrix": [[0.0] * 3 for _ in range(3)]})
    with pytest.raises(ValueError, match="invertible"):
        table_to_pixel(invalid, 100, 100)
