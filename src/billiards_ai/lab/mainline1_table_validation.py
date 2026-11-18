"""Current-frame table geometry evidence for the Mainline 1 USB gate."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _robust_line(independent: np.ndarray, dependent: np.ndarray) -> tuple[float, float] | None:
    if len(independent) < 20:
        return None
    keep = np.ones(len(independent), dtype=bool)
    for _ in range(5):
        if int(np.count_nonzero(keep)) < 20:
            return None
        slope, intercept = np.polyfit(independent[keep], dependent[keep], 1)
        residuals = np.abs(dependent - (slope * independent + intercept))
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        next_keep = residuals <= max(3.0, median + 3.5 * max(mad, 1.0))
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if float(np.max(residuals[keep])) > 8.0:
        return None
    return float(slope), float(intercept)


def _intersection(
    horizontal: tuple[float, float],
    vertical: tuple[float, float],
) -> tuple[float, float] | None:
    horizontal_slope, horizontal_intercept = horizontal
    vertical_slope, vertical_intercept = vertical
    denominator = 1.0 - vertical_slope * horizontal_slope
    if abs(denominator) < 1e-6:
        return None
    x = (vertical_slope * horizontal_intercept + vertical_intercept) / denominator
    y = horizontal_slope * x + horizontal_intercept
    return float(x), float(y)


def detect_table_validation_corners(frame: Any) -> tuple[tuple[float, float], ...] | None:
    """Detect a complete dominant green table and return LL, LR, UR, UL corners.

    This detector supplies only current-epoch drift evidence. It does not create
    or replace the metric table calibration used by Mainline 1.
    """

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return None
    height, width = image.shape[:2]
    if height < 64 or width < 64:
        return None
    bgr = image[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (25, 40, 18), (100, 255, 255))
    kernel_size = max(3, round(min(width, height) / 180))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(green)
    if component_count <= 1:
        return None
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left = int(stats[component_index, cv2.CC_STAT_LEFT])
    top = int(stats[component_index, cv2.CC_STAT_TOP])
    component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    component_area = int(stats[component_index, cv2.CC_STAT_AREA])
    coverage = component_area / float(width * height)
    if coverage < 0.35 or coverage > 0.90:
        return None
    component = labels == component_index
    horizontal_step = max(1, component_width // 480)
    vertical_step = max(1, component_height // 270)

    columns: list[float] = []
    upper_edges: list[float] = []
    lower_edges: list[float] = []
    for x in range(
        left + round(component_width * 0.08),
        left + round(component_width * 0.92),
        horizontal_step,
    ):
        rows = np.flatnonzero(component[:, x])
        if len(rows):
            columns.append(float(x))
            upper_edges.append(float(rows[0]))
            lower_edges.append(float(rows[-1]))

    rows_seen: list[float] = []
    left_edges: list[float] = []
    right_edges: list[float] = []
    for y in range(
        top + round(component_height * 0.12),
        top + round(component_height * 0.88),
        vertical_step,
    ):
        columns_at_row = np.flatnonzero(component[y])
        if len(columns_at_row):
            rows_seen.append(float(y))
            left_edges.append(float(columns_at_row[0]))
            right_edges.append(float(columns_at_row[-1]))

    independent_columns = np.asarray(columns, dtype=np.float64)
    independent_rows = np.asarray(rows_seen, dtype=np.float64)
    upper_line = _robust_line(independent_columns, np.asarray(upper_edges))
    lower_line = _robust_line(independent_columns, np.asarray(lower_edges))
    left_line = _robust_line(independent_rows, np.asarray(left_edges))
    right_line = _robust_line(independent_rows, np.asarray(right_edges))
    if None in (upper_line, lower_line, left_line, right_line):
        return None

    assert upper_line is not None
    assert lower_line is not None
    assert left_line is not None
    assert right_line is not None
    upper_left = _intersection(upper_line, left_line)
    upper_right = _intersection(upper_line, right_line)
    lower_left = _intersection(lower_line, left_line)
    lower_right = _intersection(lower_line, right_line)
    if None in (lower_left, lower_right, upper_right, upper_left):
        return None
    assert lower_left is not None
    assert lower_right is not None
    assert upper_right is not None
    assert upper_left is not None
    polygon = np.asarray((lower_left, lower_right, upper_right, upper_left))
    if not cv2.isContourConvex(polygon.astype(np.float32)):
        return None

    margin_x = width * 0.01
    margin_y = height * 0.01
    if any(
        x <= margin_x or x >= width - margin_x or y <= margin_y or y >= height - margin_y
        for x, y in polygon
    ):
        return None
    return tuple((float(x), float(y)) for x, y in polygon)


def _project_x_to_rail(
    x: float,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float] | None:
    delta_x = float(right[0] - left[0])
    if abs(delta_x) < 1e-6:
        return None
    fraction = (x - float(left[0])) / delta_x
    if not 0.25 <= fraction <= 0.75:
        return None
    point = left + fraction * (right - left)
    return float(point[0]), float(point[1])


def detect_middle_pocket_references(
    frame: Any,
    corners: tuple[tuple[float, float], ...] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Detect lower/upper middle pocket mouths as independent fit references.

    The pocket material supplies the longitudinal image coordinate.  That
    coordinate is projected onto the already fitted cushion-nose rail because
    the calibration contract defines a pocket center on the playing boundary.
    These points must be used only as validation references, never as fit input.
    """

    image = np.asarray(frame)
    if corners is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        return None
    height, width = image.shape[:2]
    polygon = np.asarray(corners, dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    table_height = float(
        (np.linalg.norm(polygon[0] - polygon[3]) + np.linalg.norm(polygon[1] - polygon[2])) / 2
    )
    table_width = float(
        (np.linalg.norm(polygon[0] - polygon[1]) + np.linalg.norm(polygon[3] - polygon[2])) / 2
    )
    half_width = max(30, round(table_width * 0.08))
    inward = max(24, round(table_height * 0.10))
    outward = max(10, round(table_height * 0.03))
    minimum_area = max(40, round(width * height * 0.00012))
    maximum_area = round(width * height * 0.012)
    detected: list[tuple[float, float]] = []

    for left, right, direction in (
        (polygon[0], polygon[1], -1),
        (polygon[3], polygon[2], 1),
    ):
        midpoint = (left + right) / 2
        x0 = max(0, round(float(midpoint[0])) - half_width)
        x1 = min(width, round(float(midpoint[0])) + half_width + 1)
        if direction < 0:
            y0 = max(0, round(float(midpoint[1])) - inward)
            y1 = min(height, round(float(midpoint[1])) + outward + 1)
        else:
            y0 = max(0, round(float(midpoint[1])) - outward)
            y1 = min(height, round(float(midpoint[1])) + inward + 1)
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        mask = ((roi[:, :, 1] < 120) & (roi[:, :, 2] >= 40) & (roi[:, :, 2] <= 230)).astype(
            np.uint8
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates: list[tuple[float, float, float]] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if not minimum_area <= area <= maximum_area:
                continue
            if component_width < 8 or component_height < 6:
                continue
            center_x = float(centroids[index, 0] + x0)
            center_y = float(centroids[index, 1] + y0)
            distance = float(np.hypot(center_x - midpoint[0], center_y - midpoint[1]))
            candidates.append((distance, center_x, center_y))
        if not candidates:
            return None
        _distance, center_x, _center_y = min(candidates)
        projected = _project_x_to_rail(center_x, left, right)
        if projected is None:
            return None
        detected.append(projected)
    return detected[0], detected[1]
