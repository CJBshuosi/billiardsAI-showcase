from __future__ import annotations

from billiards_ai.lab.shot_local.fragments import (
    FragmentTrackingConfig,
    track_motion_fragments,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    LocalState,
    ObservationFrame,
    ShotAnchor,
    ShotWindow,
)


def _window(shot_id: str = "shot-0001", *, complete: bool = True) -> ShotWindow:
    return ShotWindow(
        shot_id=shot_id,
        before_frame_id=1,
        start_frame_id=2,
        end_frame_id=3,
        after_frame_id=3 if complete else None,
        start_ms=80,
        end_ms=120,
        complete=complete,
    )


def _anchor(shot_id: str, number: int, x_mm: int) -> ShotAnchor:
    return ShotAnchor(
        shot_id=shot_id,
        anchor_id=f"{shot_id}-anchor-{number:02d}",
        source_detection_id=f"anchor-{number}",
        group=BallGroup.SOLID,
        x_mm=x_mm,
        y_mm=100,
        confidence=0.9,
    )


def _ball(detection_id: str, x_mm: int) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=100,
        confidence=0.9,
        group=BallGroup.SOLID,
        group_confidence=0.8,
    )


def _frame(frame_id: int, *balls) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        observations=tuple(balls),
    )


def _config() -> FragmentTrackingConfig:
    return FragmentTrackingConfig(
        motion_threshold_mm=10,
        continuation_distance_mm=80,
        ambiguity_mm=5,
    )


def test_equal_continuation_candidates_end_fragment_instead_of_stealing_id() -> None:
    fragments = track_motion_fragments(
        _window(),
        (
            _frame(2, _ball("moving", 120), _ball("static", 500)),
            _frame(3, _ball("candidate-a", 100), _ball("candidate-b", 140), _ball("static", 500)),
        ),
        (_anchor("shot-0001", 1, 100), _anchor("shot-0001", 2, 500)),
        (),
        _config(),
    )

    assert fragments[0].terminal_state is LocalState.AMBIGUOUS
    assert fragments[0].samples[-1].frame_id == 2
    assert all(sample.frame_id != 3 for sample in fragments[0].samples)


def test_fragment_ids_restart_inside_the_next_shot_namespace() -> None:
    first = track_motion_fragments(
        _window("shot-0001"),
        (_frame(2, _ball("a", 120)), _frame(3, _ball("b", 140))),
        (_anchor("shot-0001", 1, 100),),
        (),
        _config(),
    )
    second = track_motion_fragments(
        _window("shot-0002"),
        (_frame(2, _ball("c", 120)), _frame(3, _ball("d", 140))),
        (_anchor("shot-0002", 1, 100),),
        (),
        _config(),
    )

    assert first[0].fragment_id == "shot-0001-fragment-0001"
    assert second[0].fragment_id == "shot-0002-fragment-0001"


def test_static_noncue_observation_does_not_create_motion_fragment() -> None:
    fragments = track_motion_fragments(
        _window(),
        (_frame(2, _ball("a", 100)), _frame(3, _ball("b", 100))),
        (_anchor("shot-0001", 1, 100),),
        (),
        _config(),
    )

    assert fragments == ()


def test_incomplete_window_does_not_label_active_fragment_stopped() -> None:
    fragments = track_motion_fragments(
        _window(complete=False),
        (_frame(2, _ball("a", 120)), _frame(3, _ball("b", 140))),
        (_anchor("shot-0001", 1, 100),),
        (),
        _config(),
    )

    assert fragments[0].terminal_state is LocalState.LOST
