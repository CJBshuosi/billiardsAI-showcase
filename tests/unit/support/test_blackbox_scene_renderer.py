"""Pixel contracts for independent, offline synthetic ball artwork."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tests.support.blackbox_scene_renderer import BallVisual, draw_ball, render_frame

# Literal BGR swatches: expectations do not import the renderer's palette or geometry.
HUE_PAIRS = [
    (1, 9, (0, 210, 255)),
    (2, 10, (200, 70, 25)),
    (3, 11, (35, 35, 220)),
    (4, 12, (140, 45, 110)),
    (5, 13, (0, 125, 245)),
    (6, 14, (65, 140, 25)),
    (7, 15, (45, 35, 110)),
]


def test_cue_is_white_without_numeral_ink():
    frame = np.zeros((81, 81, 3), dtype=np.uint8)
    draw_ball(frame, number=0, center_px=(40, 40), radius_px=16)
    yy, xx = np.indices((81, 81))
    disk = (xx - 40) ** 2 + (yy - 40) ** 2 <= 256
    assert np.all(frame[disk] == 255)
    assert np.all(frame[~disk] == 0)


@pytest.mark.parametrize("solid,stripe,bgr", HUE_PAIRS)
def test_actual_video_size_hue_pairs_keep_colored_middle_and_distinct_caps(solid, stripe, bgr):
    solid_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    stripe_frame = solid_frame.copy()
    draw_ball(solid_frame, number=solid, center_px=(32, 32), radius_px=15.75)
    draw_ball(stripe_frame, number=stripe, center_px=(32, 32), radius_px=15.75)
    # Side of the horizontal middle, outside the central numeral disk.
    for x in (21, 43):
        assert tuple(solid_frame[32, x]) == bgr
        assert tuple(stripe_frame[32, x]) == bgr
    for y in (20, 44):
        assert tuple(solid_frame[y, 32]) == bgr
        assert tuple(stripe_frame[y, 32]) == (255, 255, 255)


def test_eight_has_black_exterior_and_white_number_disk():
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    draw_ball(frame, number=8, center_px=(32, 32), radius_px=16)
    assert tuple(frame[20, 32]) == (0, 0, 0)
    assert tuple(frame[32, 44]) == (0, 0, 0)
    central_disk = frame[26:39, 26:39]
    assert np.any(np.all(central_disk == 255, axis=2))
    assert np.any(np.all(central_disk == 0, axis=2))


@pytest.mark.parametrize("number", range(1, 16))
def test_each_number_has_visible_black_ink_inside_white_disk(number):
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    draw_ball(frame, number=number, center_px=(32, 32), radius_px=15.75)
    # This 9x9 square lies inside the numeral disk, away from black eight's exterior.
    central = frame[28:37, 28:37]
    assert np.any(np.all(central == 0, axis=2))
    assert np.any(np.all(central == 255, axis=2))


@pytest.mark.parametrize("center", [(32.25, 31.75), (0.0, 0.0), (-5.5, 32.25), (65.0, 64.0)])
def test_drawing_changes_only_supplied_circular_footprint_including_cropped_edges(center):
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    draw_ball(frame, number=13, center_px=center, radius_px=16)
    yy, xx = np.indices((64, 64))
    outside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 > 256
    assert np.all(frame[outside] == 90)
    assert np.any(frame[~outside] != 90)


def test_fully_off_image_balls_do_not_leave_ghosts():
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    draw_ball(frame, number=9, center_px=(-100, 32), radius_px=16)
    draw_ball(frame, number=8, center_px=(1e300, -1e300), radius_px=16)
    assert np.all(frame == 90)


def test_fixed_geometry_preserves_off_midline_downward_y_without_relocation():
    # Hand calculation: 635mm -> 350px; 254mm -> 140px, added to (240,147).
    frame = render_frame([BallVisual(0, 635, 254)])
    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous and frame.flags.writeable
    white_y, white_x = np.where(np.all(frame == 255, axis=2))
    assert (white_x.min(), white_x.max(), white_y.min(), white_y.max()) == (575, 605, 272, 302)
    assert tuple(frame[287, 590]) == (255, 255, 255)
    assert tuple(frame[707, 590]) == (45, 110, 45)
    assert tuple(frame[147, 240]) == (45, 110, 45)
    assert tuple(frame[847, 1640]) == (45, 110, 45)
    assert tuple(frame[146, 240]) == (0, 0, 0)
    assert tuple(frame[848, 1640]) == (0, 0, 0)


def test_finite_off_table_position_is_drawn_as_supplied():
    # x=-254mm maps to x=100px, outside the cloth. It must not be moved to a rail.
    frame = render_frame([BallVisual(0, -254, 508)])
    assert tuple(frame[427, 100]) == (255, 255, 255)
    assert tuple(frame[427, 240]) == (45, 110, 45)


def test_invisible_balls_leave_blank_snapshot_unchanged():
    blank = render_frame([])
    invisible = render_frame([BallVisual(8, 635, 254, visible=False)])
    assert np.array_equal(invisible, blank)


def test_frames_are_deterministic_and_accept_one_pass_iterables():
    balls = [BallVisual(3, 635, 254), BallVisual(11, 1270, 762)]
    assert np.array_equal(render_frame(iter(balls)), render_frame(balls))


def test_snapshot_is_immutable():
    ball = BallVisual(1, 635, 254)
    with pytest.raises(FrozenInstanceError):
        ball.x_mm = 0


@pytest.mark.parametrize(
    "balls",
    [
        [BallVisual(1, 0, 0), BallVisual(1, 50, 50)],
        [BallVisual(1, 0, 0, False), BallVisual(1, 50, 50)],
        [BallVisual(True, 0, 0)],
        [BallVisual(16, 0, 0)],
        [BallVisual(1, float("nan"), 0)],
        [BallVisual(1, 0, float("inf"), False)],
        [BallVisual(1, "635", 0)],
        [BallVisual(1, 0, 0, visible="false")],
        [object()],
    ],
)
def test_invalid_snapshots_fail(balls):
    with pytest.raises((TypeError, ValueError)):
        render_frame(balls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"number": True},
        {"number": -1},
        {"number": 16},
        {"number": 1.5},
        {"center_px": (float("nan"), 20)},
        {"center_px": (20, float("inf"))},
        {"center_px": ("20", 20)},
        {"center_px": (True, 20)},
        {"center_px": (20,)},
        {"center_px": None},
        {"radius_px": float("nan")},
        {"radius_px": float("inf")},
        {"radius_px": 0.99},
        {"radius_px": -1},
        {"radius_px": "16"},
    ],
)
def test_invalid_draw_arguments_fail_before_any_pixel_mutation(kwargs):
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    before = frame.copy()
    arguments = {"number": 1, "center_px": (32, 32), "radius_px": 16} | kwargs
    with pytest.raises((TypeError, ValueError)):
        draw_ball(frame, **arguments)
    assert np.array_equal(frame, before)


@pytest.mark.parametrize("kind", ["float", "gray", "rgba", "strided", "readonly", "empty"])
def test_invalid_frames_fail_without_mutation(kind):
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    if kind == "float":
        frame = frame.astype(np.float32)
    elif kind == "gray":
        frame = frame[:, :, 0].copy()
    elif kind == "rgba":
        frame = np.full((64, 64, 4), 90, dtype=np.uint8)
    elif kind == "strided":
        frame = frame[:, ::2, :]
    elif kind == "readonly":
        frame.flags.writeable = False
    elif kind == "empty":
        frame = frame[:0]
    before = frame.copy()
    with pytest.raises((TypeError, ValueError)):
        draw_ball(frame, number=1, center_px=(20, 20), radius_px=16)
    assert np.array_equal(frame, before)
