from __future__ import annotations

import pytest

from billiards_ai.lab.propagation.windows import plan_windows


def test_window_plan_covers_interval_with_exact_overlap() -> None:
    windows = plan_windows(
        asset_id="video-1",
        start_frame=0,
        end_frame=100,
        window_size=30,
        overlap=10,
    )

    assert [(item.start_frame, item.end_frame) for item in windows] == [
        (0, 30),
        (20, 50),
        (40, 70),
        (60, 90),
        (80, 100),
    ]
    assert [item.ordinal for item in windows] == [0, 1, 2, 3, 4]


def test_window_plan_does_not_add_a_duplicate_tail_window() -> None:
    windows = plan_windows(
        asset_id="video-1",
        start_frame=10,
        end_frame=70,
        window_size=30,
        overlap=10,
    )

    assert [(item.start_frame, item.end_frame) for item in windows] == [
        (10, 40),
        (30, 60),
        (50, 70),
    ]


def test_window_plan_defaults_to_the_frozen_initial_30_10_configuration() -> None:
    windows = plan_windows(asset_id="video-1", start_frame=0, end_frame=50)

    assert [(item.start_frame, item.end_frame) for item in windows] == [(0, 30), (20, 50)]


@pytest.mark.parametrize(
    ("start_frame", "end_frame", "window_size", "overlap"),
    [
        (10, 10, 30, 10),
        (10, 9, 30, 10),
        (0, 10, 0, 0),
        (0, 10, 10, -1),
        (0, 10, 10, 10),
        (0, 10, 10, 11),
    ],
)
def test_window_plan_rejects_non_progressing_configurations(
    start_frame: int,
    end_frame: int,
    window_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        plan_windows(
            asset_id="video-1",
            start_frame=start_frame,
            end_frame=end_frame,
            window_size=window_size,
            overlap=overlap,
        )
