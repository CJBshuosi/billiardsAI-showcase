from dataclasses import dataclass

from billiards_ai.lab.guidance import ExplicitIntent, build_recommendation
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PhysicsResult,
    PhysicsStatus,
    PocketId,
)
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
        detection_confidence=0.97,
    )


def _state() -> CanonicalTableState:
    return CanonicalTableState(
        frame_id=10,
        timestamp_ms=333,
        calibration_id="cal-1",
        motion_status=MotionStatus.STATIONARY,
        balls=(
            _ball("cue", "cue", BallKind.CUE, 500, 500),
            _ball("solid", "1", BallKind.SOLID, 1500, 650),
            _ball("stripe", "9", BallKind.STRIPE, 2100, 1000),
        ),
        source_refs=("frame:10",),
    )


@dataclass
class _FailingPhysics:
    version: str = "synthetic-failure/1"

    def simulate(self, state, candidate, power):
        return PhysicsResult(
            candidate_id=candidate.candidate_id,
            status=PhysicsStatus.FAILED,
            backend_version=self.version,
            reason="numeric_failure",
        )


def test_service_generates_top_three_automatic_candidates() -> None:
    result = build_recommendation(state=_state(), context=EightBallContext(group="solid"))
    assert result.status is GuidanceStatus.GEOMETRY_ONLY
    assert 1 <= len(result.candidates) <= 3
    assert all(item.target_kind is BallKind.SOLID for item in result.candidates)


def test_explicit_intent_bypasses_automatic_target_selection() -> None:
    result = build_recommendation(
        state=_state(),
        intent=ExplicitIntent(target_track_id="stripe", pocket=PocketId.TOP_RIGHT),
    )
    assert result.candidates[0].target_track_id == "stripe"
    assert result.candidates[0].pocket is PocketId.TOP_RIGHT


def test_physics_failure_is_visible_geometry_fallback() -> None:
    result = build_recommendation(
        state=_state(),
        context=EightBallContext(group="solid"),
        physics=_FailingPhysics(),
    )
    assert result.status is GuidanceStatus.GEOMETRY_ONLY
    assert result.fallback_reason == "numeric_failure"


def test_context_and_intent_are_mutually_exclusive() -> None:
    try:
        build_recommendation(
            state=_state(),
            context=EightBallContext(group="solid"),
            intent=ExplicitIntent(target_track_id="solid", pocket=PocketId.TOP_RIGHT),
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous planning mode was accepted")
