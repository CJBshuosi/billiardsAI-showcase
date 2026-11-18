"""台面标定：图像像素 → 台面毫米坐标（AD-18）。

AD-13a 要求 `Shot` 里的位置量一律是台面毫米整数坐标。类型系统拦不住有人
写入像素值——归一化坐标和像素坐标都能通过 `Point` 的校验。所以契约的真正
落点在这里：**感知实现必须在写入 Shot 之前调用本模块完成转换**。

未标定的设备不得产出任何含距离量的结果（FR19），这条由 `require_calibration`
强制，而不是靠调用方自觉。
"""

import math
from datetime import UTC, datetime

import cv2
import numpy as np
from pydantic import BaseModel, Field

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.calibration import (
    CalibrationReference,
    DistortionModel,
    DistortionParameters,
)
from billiards_ai.shared.models.geometry import Point

Corner = tuple[float, float]


class NotCalibratedError(RuntimeError):
    """设备未标定却试图产出距离量。宁可不给，不可给错（AD-5 同源精神）。"""


class CalibrationDriftError(RuntimeError):
    """标定已失准，结果不可信。"""


class DistortionNotVerifiedError(RuntimeError):
    """非可忽略畸变尚未通过验证，禁止声称毫米精度。"""


class CalibrationAccuracyError(RuntimeError):
    """独立参照点未达到 20mm 精度门槛。"""


class TableCalibration(BaseModel):
    """一台设备的标定结果。随 device_id 持久化。

    corners_px 的顺序是契约的一部分，必须为：左下 → 右下 → 右上 → 左上，
    与台面坐标系 (0,0) → (L,0) → (L,W) → (0,W) 一一对应。顺序错了不会报错，
    只会得到镜像或旋转的坐标系——这是最难查的一类 bug，因此顺序写死在这里。
    """

    device_id: str
    corners_px: list[Corner] = Field(min_length=4, max_length=4)
    table_length_mm: int = TABLE_LENGTH_MM
    table_width_mm: int = TABLE_WIDTH_MM
    matrix: list[list[float]] = Field(min_length=3, max_length=3)
    calibrated_at: datetime
    distortion: DistortionParameters = Field(default_factory=DistortionParameters)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def table_corners_mm(self) -> list[Corner]:
        return _table_corners(self.table_length_mm, self.table_width_mm)


def _table_corners(length_mm: int, width_mm: int) -> list[Corner]:
    """台面四角，顺序与 corners_px 对应。原点为左下袋口中心（AD-13a）。"""
    return [
        (0.0, 0.0),
        (float(length_mm), 0.0),
        (float(length_mm), float(width_mm)),
        (0.0, float(width_mm)),
    ]


def calibrate(
    device_id: str,
    corners_px: list[Corner] | tuple[Corner, ...],
    table_length_mm: int = TABLE_LENGTH_MM,
    table_width_mm: int = TABLE_WIDTH_MM,
    distortion: DistortionParameters | None = None,
) -> TableCalibration:
    """从人工标注的四角求单应矩阵。安装时执行一次。"""
    if len(corners_px) != 4:
        raise ValueError(f"需要恰好 4 个角点，收到 {len(corners_px)}")

    selected_distortion = distortion or DistortionParameters()
    src = _correct_points(corners_px, selected_distortion).astype(np.float32)
    if (
        not np.all(np.isfinite(src))
        or abs(cv2.contourArea(src)) <= 1e-6
        or not cv2.isContourConvex(src)
    ):
        raise ValueError("四角退化（共线、重合或顺序交叉），无法求解单应矩阵")
    dst = np.asarray(_table_corners(table_length_mm, table_width_mm), dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("四角退化（共线或重合），无法求解单应矩阵")

    return TableCalibration(
        device_id=device_id,
        corners_px=[(float(x), float(y)) for x, y in corners_px],
        table_length_mm=table_length_mm,
        table_width_mm=table_width_mm,
        matrix=matrix.tolist(),
        calibrated_at=datetime.now(UTC),
        distortion=selected_distortion,
    )


def require_calibration(calib: TableCalibration | None, device_id: str = "") -> TableCalibration:
    """守卫：未标定就中止，不返回估算值也不返回 0（FR19）。"""
    if calib is None:
        raise NotCalibratedError(
            f"设备 {device_id or '(未知)'} 未完成台面标定，拒绝产出含距离量的结果"
        )
    return calib


def pixel_to_table(calib: TableCalibration, x_px: float, y_px: float) -> Point:
    """单点转换。输出毫米整数，四舍五入。"""
    x_mm, y_mm = _transform_points(calib, [(x_px, y_px)])[0]
    return Point(x_mm=round(float(x_mm)), y_mm=round(float(y_mm)))


def pixels_to_table(calib: TableCalibration, points_px: list[Corner]) -> list[Point]:
    """批量转换。空输入返回空列表，不报错。"""
    if not points_px:
        return []
    dst = _transform_points(calib, points_px)
    return [Point(x_mm=round(float(x)), y_mm=round(float(y))) for x, y in dst]


def table_to_pixel(calib: TableCalibration, x_mm: float, y_mm: float) -> tuple[int, int]:
    """Map one calibrated table-mm point back into source-frame pixels."""
    try:
        inverse = np.linalg.inv(calib.as_array())
    except np.linalg.LinAlgError as exc:
        raise ValueError("calibration matrix must be invertible") from exc
    src = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float64)
    corrected = cv2.perspectiveTransform(src, inverse).reshape(-1, 2)
    x_px, y_px = _distort_points(corrected, calib.distortion)[0]
    if not np.isfinite((x_px, y_px)).all():
        raise ValueError("table coordinate cannot be mapped to a finite pixel")
    return round(float(x_px)), round(float(y_px))


def _distort_points(points_px: np.ndarray, distortion: DistortionParameters) -> np.ndarray:
    """Map undistorted image pixels back onto the raw camera frame."""
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if distortion.model is DistortionModel.NONE:
        return points
    if not distortion.verified:
        raise DistortionNotVerifiedError("非可忽略镜头畸变尚未验证，拒绝坐标转换")
    camera_matrix = np.asarray(distortion.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion.coefficients, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    normalized = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    object_points = np.column_stack(
        (
            normalized[:, 0] / normalized[:, 2],
            normalized[:, 1] / normalized[:, 2],
            np.ones(len(points), dtype=np.float64),
        )
    )
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        coefficients,
    )
    return projected.reshape(-1, 2)


def reference_errors_mm(
    calib: TableCalibration, references: tuple[CalibrationReference, ...]
) -> tuple[float, ...]:
    """Return independent Euclidean reference errors without integer rounding."""
    if not references:
        raise CalibrationAccuracyError(
            "at least one independent reference is required for calibration"
        )
    if any(
        math.isclose(reference.pixel[0], corner[0], abs_tol=1e-6)
        and math.isclose(reference.pixel[1], corner[1], abs_tol=1e-6)
        for reference in references
        for corner in calib.corners_px
    ):
        raise CalibrationAccuracyError(
            "independent reference points must not reuse fitted calibration corners"
        )
    transformed = _transform_points(calib, [item.pixel for item in references])
    return tuple(
        float(np.hypot(x_mm - item.table.x_mm, y_mm - item.table.y_mm))
        for (x_mm, y_mm), item in zip(transformed, references, strict=True)
    )


def validate_reference_accuracy(
    calib: TableCalibration,
    references: tuple[CalibrationReference, ...],
    *,
    max_error_mm: float = 20.0,
) -> float:
    """Enforce the independent-reference 20mm acceptance boundary."""
    largest = max(reference_errors_mm(calib, references))
    if largest > max_error_mm:
        raise CalibrationAccuracyError(
            f"设备 {calib.device_id} 独立参照点误差 {largest:.1f}mm 超过阈值 {max_error_mm:.1f}mm"
        )
    return largest


def _correct_points(
    points_px: list[Corner] | tuple[Corner, ...], distortion: DistortionParameters
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    if distortion.model is DistortionModel.NONE:
        return points.reshape(-1, 2)
    if not distortion.verified:
        raise DistortionNotVerifiedError("非可忽略镜头畸变尚未验证，拒绝坐标转换")
    camera_matrix = np.asarray(distortion.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion.coefficients, dtype=np.float64)
    corrected = cv2.undistortPoints(
        points,
        camera_matrix,
        coefficients,
        P=camera_matrix,
    )
    return corrected.reshape(-1, 2)


def _transform_points(
    calib: TableCalibration, points_px: list[Corner] | tuple[Corner, ...]
) -> np.ndarray:
    corrected = _correct_points(points_px, calib.distortion).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corrected, calib.as_array()).reshape(-1, 2)


def corner_drift_px(calib: TableCalibration, current_corners_px: list[Corner]) -> float:
    """当前检测到的四角与标定时的最大偏移（像素）。用于定期失准检测（FR18）。"""
    if len(current_corners_px) != 4:
        raise ValueError(f"需要恰好 4 个角点，收到 {len(current_corners_px)}")
    old = np.asarray(calib.corners_px, dtype=np.float64)
    new = np.asarray(current_corners_px, dtype=np.float64)
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise CalibrationDriftError("corner drift evidence must contain only finite values")
    drift = float(np.max(np.linalg.norm(new - old, axis=1)))
    if not math.isfinite(drift):
        raise CalibrationDriftError("corner drift evidence must be finite")
    return drift


def assert_no_drift(
    calib: TableCalibration,
    current_corners_px: list[Corner],
    threshold_px: float = 15.0,
) -> None:
    """失准超阈值即中止。摄像头被碰过、球台被挪过都会触发。"""
    if not math.isfinite(threshold_px) or threshold_px < 0:
        raise CalibrationDriftError("corner drift threshold must be finite and non-negative")
    drift = corner_drift_px(calib, current_corners_px)
    if drift > threshold_px:
        raise CalibrationDriftError(
            f"设备 {calib.device_id} 标定失准：四角最大偏移 {drift:.1f}px "
            f"超过阈值 {threshold_px}px，需重新标定"
        )
