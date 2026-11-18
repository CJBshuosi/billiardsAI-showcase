import pytest

from billiards_ai.lab.planning.geometry import plan_direct_shot
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceStatus, PocketId
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def _ball(track: str, identity: str, kind: BallKind, x: int, y: int) -> CanonicalBall:
    return CanonicalBall(
        track_id=track,
        identity=identity,
        kind=kind,
        identity_confidence=0.95,
        center=Point(x_mm=x, y_mm=y),
        radius_mm=28.5,
        detection_confidence=0.98,
    )


def _state(*extra: CanonicalBall) -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=1,
        timestamp_ms=0,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(
            _ball("cue", "cue", BallKind.CUE, 500, 1270),
            _ball("object", "1", BallKind.SOLID, 1957, 1270),
            *extra,
        ),
        source_refs=("frame:1",),
    )


def test_straight_shot_places_ghost_ball_two_radii_behind_object() -> None:
    result = plan_direct_shot(state=_state(), target_track_id="object", pocket=PocketId.TOP_RIGHT)
    assert result.status is GuidanceStatus.PLANNED
    assert result.ghost_ball_center is not None
    assert result.ghost_ball_center.x_mm == pytest.approx(1900, abs=1)
    assert result.cut_angle_deg == pytest.approx(0.0, abs=0.1)


def test_ball_in_object_corridor_rejects_candidate() -> None:
    blocker = _ball("blocker", "9", BallKind.STRIPE, 2250, 1270)
    result = plan_direct_shot(
        state=_state(blocker), target_track_id="object", pocket=PocketId.TOP_RIGHT
    )
    assert result.status is GuidanceStatus.UNSUPPORTED
    assert "object_path_blocked" in result.reasons


def test_ball_in_cue_corridor_rejects_candidate() -> None:
    blocker = _ball("blocker", "9", BallKind.STRIPE, 1200, 1270)
    result = plan_direct_shot(
        state=_state(blocker), target_track_id="object", pocket=PocketId.TOP_RIGHT
    )
    assert result.status is GuidanceStatus.UNSUPPORTED
    assert "cue_path_blocked" in result.reasons


def test_missing_target_is_rejected_explicitly() -> None:
    with pytest.raises(KeyError, match="missing"):
        plan_direct_shot(state=_state(), target_track_id="missing", pocket=PocketId.TOP_RIGHT)
