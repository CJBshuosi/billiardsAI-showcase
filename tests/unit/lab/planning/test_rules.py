from billiards_ai.lab.planning.rules import EightBallContext, legal_targets
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import GuidanceStatus
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def _ball(track: str, identity: str, kind: BallKind) -> CanonicalBall:
    return CanonicalBall(
        track_id=track,
        identity=identity,
        kind=kind,
        identity_confidence=0.9,
        center=Point(x_mm=300 + len(track) * 100, y_mm=600),
        radius_mm=28.5,
        detection_confidence=0.95,
    )


def _state(*balls: CanonicalBall) -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=1,
        timestamp_ms=0,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(_ball("cue", "cue", BallKind.CUE), *balls),
        source_refs=("frame:1",),
    )


def test_open_table_allows_known_solids_and_stripes_but_not_eight() -> None:
    result = legal_targets(
        _state(
            _ball("solid-1", "1", BallKind.SOLID),
            _ball("stripe-1", "9", BallKind.STRIPE),
            _ball("eight", "8", BallKind.EIGHT),
        ),
        EightBallContext(group="open"),
    )
    assert result.track_ids == ("solid-1", "stripe-1")
    assert result.status is GuidanceStatus.PLANNED


def test_unresolved_group_inventory_abstains_before_eight_ball() -> None:
    result = legal_targets(
        _state(
            _ball("unknown", "unknown", BallKind.UNKNOWN),
            _ball("eight", "8", BallKind.EIGHT),
        ),
        EightBallContext(group="solid"),
    )
    assert result.status is GuidanceStatus.ABSTAINED
    assert result.reason == "group_inventory_ambiguous"


def test_eight_is_legal_only_after_assigned_group_is_empty() -> None:
    result = legal_targets(
        _state(_ball("eight", "8", BallKind.EIGHT)),
        EightBallContext(group="solid"),
    )
    assert result.track_ids == ("eight",)
