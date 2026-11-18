from __future__ import annotations

from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
    ShotWindow,
)
from billiards_ai.lab.shot_local.window import (
    ShotWindowConfig,
    ShotWindowTransitionCore,
    ShotWindowTransitionKind,
    build_shot_anchors,
    segment_shot_windows,
)


def _ball(
    detection_id: str,
    x_mm: int,
    *,
    group: BallGroup = BallGroup.SOLID,
) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=100,
        confidence=0.9,
        group=group,
        group_confidence=0.8,
    )


def _frame(frame_id: int, timestamp_ms: int, *balls) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        observations=tuple(balls),
    )


def _config() -> ShotWindowConfig:
    return ShotWindowConfig(
        motion_threshold_mm=8,
        motion_confirm_frames=2,
        still_duration_ms=400,
    )


def _stable_motion_stable_frames() -> tuple[ObservationFrame, ...]:
    return (
        _frame(0, 0, _ball("d0", 100)),
        _frame(1, 40, _ball("d1", 100)),
        _frame(2, 80, _ball("d2", 110)),
        _frame(3, 120, _ball("d3", 120)),
        _frame(4, 120, _ball("d4", 120)),
        _frame(5, 520, _ball("d5", 120)),
    )


def test_stable_motion_stable_sequence_forms_one_complete_window() -> None:
    windows = segment_shot_windows(_stable_motion_stable_frames(), _config())

    assert windows == (
        ShotWindow(
            shot_id="shot-0001",
            before_frame_id=1,
            start_frame_id=2,
            end_frame_id=5,
            after_frame_id=5,
            start_ms=80,
            end_ms=520,
            complete=True,
        ),
    )


def test_transition_core_reports_confirmation_and_settle_without_policy_drift() -> None:
    core = ShotWindowTransitionCore(_config())

    transitions = tuple(
        transition for frame in _stable_motion_stable_frames() for transition in core.push(frame)
    )

    assert [transition.kind for transition in transitions] == [
        ShotWindowTransitionKind.SHOT_STARTED,
        ShotWindowTransitionKind.SHOT_SETTLED,
    ]
    assert (
        transitions[-1].window == segment_shot_windows(_stable_motion_stable_frames(), _config())[0]
    )


def test_video_ending_while_moving_forms_incomplete_window() -> None:
    frames = _stable_motion_stable_frames()[:4]

    assert segment_shot_windows(frames, _config()) == (
        ShotWindow(
            shot_id="shot-0001",
            before_frame_id=1,
            start_frame_id=2,
            end_frame_id=3,
            after_frame_id=None,
            start_ms=80,
            end_ms=120,
            complete=False,
        ),
    )


def test_second_window_uses_a_new_shot_namespace() -> None:
    first = _stable_motion_stable_frames()
    second = tuple(
        _frame(frame.frame_id + 6, frame.timestamp_ms + 600, *frame.observations)
        for frame in _stable_motion_stable_frames()[1:]
    )

    windows = segment_shot_windows(first + second, _config())

    assert [window.shot_id for window in windows] == ["shot-0001", "shot-0002"]


def test_count_change_is_motion_without_needing_identity() -> None:
    frames = (
        _frame(0, 0, _ball("a", 100)),
        _frame(1, 40, _ball("b", 100)),
        _frame(2, 80, _ball("c", 100), _ball("d", 400)),
        _frame(3, 120, _ball("e", 100), _ball("f", 400), _ball("g", 700)),
    )

    assert segment_shot_windows(frames, _config())[0].start_frame_id == 2


def test_anchors_use_exact_before_frame_and_deterministic_order() -> None:
    frames = list(_stable_motion_stable_frames())
    frames[1] = _frame(
        1,
        40,
        _ball("z", 300, group=BallGroup.STRIPE),
        _ball("a", 200, group=BallGroup.SOLID),
        _ball("b", 100, group=BallGroup.SOLID),
    )
    window = ShotWindow(
        shot_id="shot-0001",
        before_frame_id=1,
        start_frame_id=2,
        end_frame_id=5,
        after_frame_id=5,
        start_ms=80,
        end_ms=520,
        complete=True,
    )

    anchors = build_shot_anchors(window, tuple(frames))

    assert [anchor.anchor_id for anchor in anchors] == [
        "shot-0001-anchor-01",
        "shot-0001-anchor-02",
        "shot-0001-anchor-03",
    ]
    assert [anchor.source_detection_id for anchor in anchors] == ["b", "a", "z"]
