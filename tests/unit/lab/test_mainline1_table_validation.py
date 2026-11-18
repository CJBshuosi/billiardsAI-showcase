from __future__ import annotations

import cv2
import numpy as np

from billiards_ai.lab.mainline1_table_validation import (
    detect_middle_pocket_references,
    detect_table_validation_corners,
)


def _table_frame() -> np.ndarray:
    frame = np.full((540, 960, 3), 18, dtype=np.uint8)
    corners = np.asarray(
        ((70, 493), (890, 502), (878, 58), (80, 72)),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, corners, (24, 112, 36))
    cv2.circle(frame, (480, 270), 18, (245, 245, 245), -1)
    cv2.rectangle(frame, (700, 245), (959, 320), (160, 130, 105), -1)
    return frame


def test_detector_returns_live_table_corners_in_calibration_order() -> None:
    # Break caught: returning a bounding box or image corners instead of the
    # observed table polygon would pass no drift evidence to the USB gate.
    corners = detect_table_validation_corners(_table_frame())

    assert corners is not None
    expected = ((70, 493), (890, 502), (878, 58), (80, 72))
    assert np.allclose(np.asarray(corners), np.asarray(expected), atol=3.0)


def test_detector_abstains_when_no_dominant_table_surface_exists() -> None:
    # Break caught: treating an arbitrary green object as the table would let a
    # moved or obstructed camera pass the current-epoch calibration gate.
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30, 30), (180, 140), (20, 120, 35), -1)

    assert detect_table_validation_corners(frame) is None


def test_detector_abstains_when_green_fills_the_entire_frame() -> None:
    # Break caught: accepting the image boundary as table geometry would hide a
    # zoomed or badly aimed camera that no longer contains the complete table.
    frame = np.full((540, 960, 3), (20, 120, 35), dtype=np.uint8)

    assert detect_table_validation_corners(frame) is None


def test_detector_uses_long_rails_instead_of_occluded_corner_vertices() -> None:
    # Break caught: convex-hull vertices move when a pocket or a player hides a
    # corner, even though the four long table rails remain in the same place.
    reference = _table_frame()
    occluded = reference.copy()
    cv2.circle(occluded, (80, 72), 34, (18, 18, 18), -1)
    cv2.circle(occluded, (890, 502), 42, (18, 18, 18), -1)
    cv2.rectangle(occluded, (825, 220), (959, 330), (150, 125, 100), -1)

    reference_corners = detect_table_validation_corners(reference)
    occluded_corners = detect_table_validation_corners(occluded)

    assert reference_corners is not None
    assert occluded_corners is not None
    assert np.allclose(reference_corners, occluded_corners, atol=4.0)


def test_middle_pockets_are_detected_independently_then_projected_to_rails() -> None:
    frame = _table_frame()
    cv2.ellipse(frame, (466, 490), (25, 18), 0, 0, 360, (150, 150, 150), -1)
    cv2.ellipse(frame, (496, 68), (24, 17), 0, 0, 360, (150, 150, 150), -1)
    corners = detect_table_validation_corners(frame)

    references = detect_middle_pocket_references(frame, corners)

    assert references is not None
    lower, upper = references
    assert abs(lower[0] - 466) < 4
    assert abs(upper[0] - 496) < 4
    assert abs(lower[1] - 497) < 4
    assert abs(upper[1] - 65) < 4
