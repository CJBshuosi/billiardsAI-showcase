from __future__ import annotations

import cv2
import numpy as np


def _input(frame_index: int, image: np.ndarray) -> object:
    from billiards_ai.lab.motion import MotionInputFrame

    return MotionInputFrame(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100.0,
        image=image,
    )


def _energy_series(values: list[float]) -> tuple[object, ...]:
    from billiards_ai.lab.motion import MotionFrame

    return tuple(
        MotionFrame(
            frame_index=index,
            timestamp_ms=index * 100.0,
            energy=value,
            active_pixel_ratio=0.0,
            centroid_x=None,
            centroid_y=None,
            confidence=min(1.0, value / 10.0),
        )
        for index, value in enumerate(values)
    )


def test_segment_shots_uses_start_stop_hysteresis() -> None:
    from billiards_ai.lab.motion import MotionConfig, segment_shots

    series = _energy_series([0, 0, 1, 8, 9, 3, 1, 0, 0])

    shots = segment_shots(
        series,
        MotionConfig(start_threshold=6, stop_threshold=2, min_shot_frames=1),
    )

    assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(3, 7)]
    assert shots[0].review_status == "unreviewed"
    assert shots[0].peak_energy == 9


def test_human_motion_outside_table_mask_is_ignored() -> None:
    from billiards_ai.lab.motion import MotionConfig, motion_series

    first = np.zeros((64, 64, 3), dtype=np.uint8)
    second = first.copy()
    second[:, :12] = 255
    table_mask = np.zeros((64, 64), dtype=np.uint8)
    table_mask[12:52, 12:52] = 255

    series = motion_series(
        (_input(0, first), _input(1, second)),
        table_mask,
        MotionConfig(pixel_diff_threshold=5, min_component_area=2, blur_kernel=1),
    )

    assert series[1].energy == 0
    assert series[1].active_pixel_ratio == 0
    assert series[1].centroid_x is None


def test_small_inside_table_noise_is_removed() -> None:
    from billiards_ai.lab.motion import MotionConfig, motion_series

    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = first.copy()
    second[16, 16] = 255
    table_mask = np.full((32, 32), 255, dtype=np.uint8)

    series = motion_series(
        (_input(0, first), _input(1, second)),
        table_mask,
        MotionConfig(pixel_diff_threshold=5, min_component_area=4, blur_kernel=1),
    )

    assert series[1].energy == 0
    assert series[1].active_pixel_ratio == 0


def test_large_body_component_is_removed() -> None:
    from billiards_ai.lab.motion import MotionConfig, motion_series

    first = np.zeros((30, 30, 3), dtype=np.uint8)
    second = first.copy()
    second[5:25, 5:25] = 255
    table_mask = np.full((30, 30), 255, dtype=np.uint8)

    series = motion_series(
        (_input(0, first), _input(1, second)),
        table_mask,
        MotionConfig(
            pixel_diff_threshold=5,
            min_component_area=2,
            max_component_area=50,
            blur_kernel=1,
        ),
    )

    assert series[1].energy == 0
    assert series[1].active_pixel_ratio == 0


def test_motion_centroid_tracks_retained_component() -> None:
    from billiards_ai.lab.motion import MotionConfig, motion_series

    first = np.zeros((40, 40, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(second, (24, 10), (29, 15), (255, 255, 255), thickness=-1)
    table_mask = np.full((40, 40), 255, dtype=np.uint8)

    series = motion_series(
        (_input(0, first), _input(1, second)),
        table_mask,
        MotionConfig(pixel_diff_threshold=5, min_component_area=4, blur_kernel=1),
    )

    assert series[1].energy > 0
    assert series[1].active_pixel_ratio == 36 / 1600
    assert series[1].centroid_x == 26.5
    assert series[1].centroid_y == 12.5


def test_motion_rejects_mask_with_wrong_shape() -> None:
    from billiards_ai.lab.motion import MotionConfig, motion_series

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    wrong_mask = np.ones((16, 16), dtype=np.uint8)

    try:
        motion_series(
            (_input(0, image), _input(1, image)),
            wrong_mask,
            MotionConfig(),
        )
    except ValueError as exc:
        assert "mask shape" in str(exc)
    else:
        raise AssertionError("wrong-shaped table mask must be rejected")
