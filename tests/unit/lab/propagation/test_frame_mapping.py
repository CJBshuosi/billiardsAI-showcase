from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from billiards_ai.lab import propagation
from billiards_ai.lab.propagation import (
    ExtractedShot,
    ExtractedWindow,
    Sam2WindowInput,
    sam2_worker_input,
)
from billiards_ai.lab.propagation.frame_mapping import FrameMappingError, FrameTransform


def test_exports_frame_mapping_interfaces_from_the_propagation_package() -> None:
    assert propagation.FrameTransform is FrameTransform
    assert ExtractedShot.__name__ == "ExtractedShot"
    assert ExtractedWindow.__name__ == "ExtractedWindow"


def test_maps_review_pixels_to_unrotated_display_source_pixels() -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
        rotation_degrees=0,
    )

    assert transform.display_size == (1920, 1080)
    assert transform.forward_affine == (4.0, 0.0, 0.0, 0.0, 4.0, 0.0)
    assert transform.map_review_point((120.0, 67.5)) == (480.0, 270.0)
    assert transform.map_source_point((480.0, 270.0)) == (120.0, 67.5)
    assert transform.map_review_box((10.0, 20.0, 100.0, 200.0)) == (
        40.0,
        80.0,
        400.0,
        800.0,
    )


@pytest.mark.parametrize(
    ("rotation_degrees", "expected_size", "expected_point"),
    [
        (90, (1080, 1920), (809.0, 480.0)),
        (180, (1920, 1080), (1439.0, 809.0)),
        (270, (1080, 1920), (270.0, 1439.0)),
    ],
)
def test_maps_review_pixels_into_the_recorded_display_rotation(
    rotation_degrees: int,
    expected_size: tuple[int, int],
    expected_point: tuple[float, float],
) -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
        rotation_degrees=rotation_degrees,
    )

    source_point = transform.map_review_point((120.0, 67.5))

    assert transform.display_size == expected_size
    assert source_point == expected_point
    assert transform.map_source_point(source_point) == (120.0, 67.5)


def test_rejects_review_boxes_that_escape_the_review_image() -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
        rotation_degrees=90,
    )

    with pytest.raises(FrameMappingError, match="review box"):
        transform.map_review_box((-0.1, 20.0, 100.0, 200.0))

    with pytest.raises(FrameMappingError, match="review box"):
        transform.map_review_box((100.0, 20.0, 481.0, 200.0))


def test_rejects_a_source_round_trip_that_exceeds_half_a_pixel() -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
        rotation_degrees=270,
    )

    transform.validate_source_round_trip(
        source_point=(270.0, 1440.0), recovered_source_point=(270.3, 1440.4)
    )

    with pytest.raises(FrameMappingError, match="0.5"):
        transform.validate_source_round_trip(
            source_point=(270.0, 1440.0), recovered_source_point=(270.6, 1440.0)
        )


def test_rotated_boxes_remain_confined_to_display_source_dimensions() -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
        rotation_degrees=90,
    )

    source_box = transform.map_review_box((0.0, 0.0, 480.0, 270.0))
    recovered_box = transform.map_source_box(source_box)

    assert source_box == (0.0, 0.0, 1080.0, 1920.0)
    assert recovered_box == (0.0, 0.0, 480.0, 270.0)
    assert math.dist((source_box[0], source_box[1]), (0.0, 0.0)) == 0.0


def test_points_use_pixel_centers_while_boxes_use_half_open_image_edges() -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
    )

    with pytest.raises(FrameMappingError, match="review point"):
        transform.map_review_point((480.0, 100.0))
    with pytest.raises(FrameMappingError, match="source point"):
        transform.map_source_point((1920.0, 100.0))

    assert transform.map_review_box((0.0, 0.0, 480.0, 270.0)) == (
        0.0,
        0.0,
        1920.0,
        1080.0,
    )


def test_rotated_point_origins_remain_inside_pixel_bounds_while_boxes_use_edges() -> None:
    transform = FrameTransform(
        review_width=4,
        review_height=3,
        source_width=4,
        source_height=3,
        rotation_degrees=90,
    )

    assert transform.map_review_point((0.0, 0.0)) == (2.0, 0.0)
    assert transform.map_source_point((2.0, 0.0)) == (0.0, 0.0)
    assert transform.map_review_box((0.0, 0.0, 4.0, 3.0)) == (0.0, 0.0, 3.0, 4.0)


def test_source_mapping_rejects_a_reverse_round_trip_over_half_a_pixel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = FrameTransform(
        review_width=480,
        review_height=270,
        source_width=1920,
        source_height=1080,
    )
    unrounded_apply = FrameTransform._apply

    def apply_with_source_rounding(
        coefficients: tuple[float, float, float, float, float, float],
        point: tuple[float, float],
    ) -> tuple[float, float]:
        mapped = unrounded_apply(coefficients, point)
        if coefficients == transform.forward_affine:
            return mapped[0] + 0.6, mapped[1]
        return mapped

    monkeypatch.setattr(FrameTransform, "_apply", staticmethod(apply_with_source_rounding))

    with pytest.raises(FrameMappingError, match="0.5"):
        transform.map_source_point((480.0, 270.0))


def test_worker_input_requires_an_extracted_window_not_a_shot_cache() -> None:
    # The full construction is integration-tested; this unit test protects the type boundary.
    with pytest.raises(TypeError, match="ExtractedWindow"):
        sam2_worker_input(object())

    with pytest.raises(ValidationError):
        Sam2WindowInput(window=object())
