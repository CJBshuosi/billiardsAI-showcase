from billiards_ai.lab.events.models import BallState, TrackedFrame
from billiards_ai.lab.table_state import build_table_state
from billiards_ai.shared.models.table_state import BallKind, MotionStatus


def _tracked_frame() -> TrackedFrame:
    return TrackedFrame(
        frame_id=7,
        timestamp_ms=233,
        balls=(
            BallState(
                identity="unknown",
                x_mm=420,
                y_mm=610,
                confidence=0.93,
                track_id="track-1",
            ),
            BallState(
                identity="cue",
                x_mm=800,
                y_mm=610,
                confidence=0.97,
                track_id="track-2",
            ),
        ),
    )


def test_builder_preserves_unknown_exact_identity_and_known_group() -> None:
    state = build_table_state(
        tracked_frame=_tracked_frame(),
        identities={
            "track-1": {
                "exact_identity": None,
                "group": "stripe",
                "confidence": 0.62,
            }
        },
        calibration_id="cal-1",
        stationary=True,
        source_refs=("run:a/frame:7",),
    )
    assert state.balls[0].identity == "unknown"
    assert state.balls[0].kind is BallKind.STRIPE
    assert state.balls[0].identity_confidence == 0.62
    assert state.motion_status is MotionStatus.STATIONARY


def test_builder_uses_tracked_identity_when_no_override_exists() -> None:
    state = build_table_state(
        tracked_frame=_tracked_frame(),
        identities={},
        calibration_id="cal-1",
        stationary=False,
        source_refs=("run:a/frame:7",),
    )
    cue = next(ball for ball in state.balls if ball.track_id == "track-2")
    assert cue.identity == "cue"
    assert cue.kind is BallKind.CUE
    assert state.motion_status is MotionStatus.MOVING


def test_builder_does_not_mutate_tracked_frame() -> None:
    tracked = _tracked_frame()
    before = tracked.balls
    build_table_state(
        tracked_frame=tracked,
        identities={"track-1": {"exact_identity": "9", "group": "stripe", "confidence": 0.9}},
        calibration_id="cal-1",
        stationary=True,
        source_refs=("run:a/frame:7",),
    )
    assert tracked.balls == before
