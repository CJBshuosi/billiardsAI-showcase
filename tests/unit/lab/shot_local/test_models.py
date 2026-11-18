from __future__ import annotations

import pytest

from billiards_ai.lab.shot_local.models import CueSample, LocalState, ShotWindow


def test_local_ids_must_belong_to_their_shot() -> None:
    with pytest.raises(ValueError, match="shot scope"):
        CueSample(
            shot_id="shot-0001",
            local_id="shot-0002-cue",
            frame_id=4,
            timestamp_ms=160,
            state=LocalState.CONFIRMED,
            x_mm=100,
            y_mm=200,
            confidence=0.9,
            source_detection_id="det-4",
        )


def test_lost_cue_sample_cannot_claim_confirmed_coordinates() -> None:
    with pytest.raises(ValueError, match="lost sample"):
        CueSample(
            shot_id="shot-0001",
            local_id="shot-0001-cue",
            frame_id=5,
            timestamp_ms=200,
            state=LocalState.LOST,
            x_mm=100,
            y_mm=200,
            confidence=0,
            source_detection_id=None,
        )


def test_confirmed_cue_sample_requires_current_detection() -> None:
    with pytest.raises(ValueError, match="observed sample"):
        CueSample(
            shot_id="shot-0001",
            local_id="shot-0001-cue",
            frame_id=5,
            timestamp_ms=200,
            state=LocalState.CONFIRMED,
            x_mm=100,
            y_mm=200,
            confidence=0.9,
            source_detection_id=None,
        )


def test_shot_window_requires_ordered_frames() -> None:
    with pytest.raises(ValueError, match="ordered frames"):
        ShotWindow(
            shot_id="shot-0001",
            before_frame_id=2,
            start_frame_id=1,
            end_frame_id=5,
            after_frame_id=5,
            start_ms=80,
            end_ms=520,
            complete=True,
        )


def test_incomplete_shot_window_has_no_after_frame() -> None:
    with pytest.raises(ValueError, match="incomplete window"):
        ShotWindow(
            shot_id="shot-0001",
            before_frame_id=1,
            start_frame_id=2,
            end_frame_id=5,
            after_frame_id=5,
            start_ms=80,
            end_ms=520,
            complete=False,
        )
