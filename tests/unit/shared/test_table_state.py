import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def _ball(identity: str, track_id: str, kind: BallKind) -> CanonicalBall:
    return CanonicalBall(
        track_id=track_id,
        identity=identity,
        kind=kind,
        identity_confidence=0.9,
        center=Point(x_mm=500, y_mm=600),
        radius_mm=28.5,
        detection_confidence=0.95,
    )


def _state(*, motion_status: MotionStatus = MotionStatus.STATIONARY) -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=12,
        timestamp_ms=400,
        calibration_id="cal-1",
        motion_status=motion_status,
        balls=(_ball("cue", "cue-track", BallKind.CUE),),
        source_refs=("frame:12",),
    )


def test_table_state_rejects_duplicate_tracks() -> None:
    with pytest.raises(ValidationError, match="track IDs must be unique"):
        CanonicalTableState(
            frame_id=12,
            timestamp_ms=400,
            calibration_id="cal-1",
            motion_status=MotionStatus.STATIONARY,
            balls=(
                _ball("cue", "t1", BallKind.CUE),
                _ball("1", "t1", BallKind.SOLID),
            ),
            source_refs=("frame:12",),
        )


def test_ball_identity_must_match_kind() -> None:
    with pytest.raises(ValidationError, match="identity does not match ball kind"):
        _ball("10", "t1", BallKind.SOLID)


def test_planning_guard_requires_stationary_cue_ball() -> None:
    with pytest.raises(ValueError, match="stationary"):
        _state(motion_status=MotionStatus.MOVING).require_plannable()


def test_planning_guard_requires_exactly_one_cue_ball() -> None:
    state = CanonicalTableState(
        frame_id=12,
        timestamp_ms=400,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(_ball("1", "object", BallKind.SOLID),),
        source_refs=("frame:12",),
    )
    with pytest.raises(ValueError, match="exactly one cue ball"):
        state.require_plannable()


def test_contracts_are_frozen_and_forbid_extra_fields() -> None:
    state = _state()
    with pytest.raises(ValidationError, match="frozen"):
        state.frame_id = 13
    with pytest.raises(ValidationError, match="Extra inputs"):
        CanonicalBall(
            track_id="t1",
            identity="unknown",
            kind=BallKind.UNKNOWN,
            identity_confidence=0,
            center=Point(x_mm=0, y_mm=0),
            radius_mm=28.5,
            detection_confidence=0.5,
            pixels=(1, 2),
        )
