from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance.models import (
    AssistanceEffectKind,
    AssistancePhase,
    AssistanceSnapshot,
    GuidanceDisplay,
    MotionState,
    PowerDisplay,
)
from billiards_ai.edge.route_assistance.session import (
    AcceptGuidance,
    AcceptMotion,
    DegradeSession,
    transition,
)


def _guidance(scene_generation: int = 0) -> GuidanceDisplay:
    return GuidanceDisplay(
        recommendation_id=f"recommendation-{scene_generation}",
        candidate_id=f"candidate-{scene_generation}",
        scene_generation=scene_generation,
        power=PowerDisplay(band="medium", approximate_percent=50),
    )


def _kinds(result) -> list[AssistanceEffectKind]:
    return [effect.kind for effect in result.effects]


def test_route_is_shown_only_for_current_stable_generation() -> None:
    result = transition(
        AssistanceSnapshot(),
        AcceptMotion(MotionState.STABLE, scene_generation=0),
    )
    assert result.snapshot.phase is AssistancePhase.ANALYZING_TABLE
    assert _kinds(result) == [AssistanceEffectKind.REQUEST_ANALYSIS]

    result = transition(result.snapshot, AcceptGuidance(_guidance()))
    assert result.snapshot.phase is AssistancePhase.WAITING_FOR_SHOT
    assert result.snapshot.guidance == _guidance()
    assert _kinds(result) == [AssistanceEffectKind.SHOW_GUIDANCE]

    result = transition(
        result.snapshot,
        AcceptMotion(MotionState.MOVING, scene_generation=0),
    )
    assert result.snapshot.phase is AssistancePhase.BALLS_MOVING
    assert result.snapshot.guidance is None
    assert _kinds(result) == [
        AssistanceEffectKind.CLEAR_GUIDANCE,
        AssistanceEffectKind.SHOT_STARTED,
    ]

    result = transition(
        result.snapshot,
        AcceptMotion(MotionState.STABLE, scene_generation=1),
    )
    assert result.snapshot.phase is AssistancePhase.ANALYZING_TABLE
    assert result.snapshot.shot_sequence == 1
    assert _kinds(result) == [
        AssistanceEffectKind.SHOT_SETTLED,
        AssistanceEffectKind.REQUEST_ANALYSIS,
    ]


def test_repeated_motion_observations_do_not_repeat_shot_effects() -> None:
    snapshot = transition(
        AssistanceSnapshot(),
        AcceptMotion(MotionState.STABLE, scene_generation=0),
    ).snapshot
    snapshot = transition(snapshot, AcceptGuidance(_guidance())).snapshot
    first = transition(snapshot, AcceptMotion(MotionState.MOVING, scene_generation=0))
    repeated = transition(
        first.snapshot,
        AcceptMotion(MotionState.MOVING, scene_generation=0),
    )

    assert repeated.snapshot == first.snapshot
    assert repeated.effects == ()


def test_stale_generation_guidance_is_rejected() -> None:
    snapshot = transition(
        AssistanceSnapshot(scene_generation=2),
        AcceptMotion(MotionState.STABLE, scene_generation=2),
    ).snapshot

    with pytest.raises(ValueError, match="stale"):
        transition(snapshot, AcceptGuidance(_guidance(scene_generation=1)))


def test_abstained_guidance_remains_fail_closed_without_visible_route() -> None:
    snapshot = transition(
        AssistanceSnapshot(),
        AcceptMotion(MotionState.STABLE, scene_generation=0),
    ).snapshot

    result = transition(
        snapshot,
        AcceptGuidance(None, abstained_reason="no_legal_target_visible"),
    )

    assert result.snapshot.phase is AssistancePhase.ANALYZING_TABLE
    assert result.snapshot.guidance is None
    assert result.snapshot.degraded_reason == "no_legal_target_visible"
    assert result.effects == ()


def test_guidance_cannot_be_published_before_analysis_is_requested() -> None:
    with pytest.raises(ValueError, match="ANALYZING_TABLE"):
        transition(AssistanceSnapshot(), AcceptGuidance(_guidance()))


def test_degradation_clears_visible_guidance() -> None:
    snapshot = transition(
        AssistanceSnapshot(),
        AcceptMotion(MotionState.STABLE, scene_generation=0),
    ).snapshot
    snapshot = transition(snapshot, AcceptGuidance(_guidance())).snapshot

    result = transition(snapshot, DegradeSession("camera_disconnected"))

    assert result.snapshot.phase is AssistancePhase.DEGRADED
    assert result.snapshot.guidance is None
    assert result.snapshot.degraded_reason == "camera_disconnected"
    assert _kinds(result) == [AssistanceEffectKind.CLEAR_GUIDANCE]


@pytest.mark.parametrize("scene_generation", (-1, 3))
def test_motion_generation_must_match_expected_transition(scene_generation: int) -> None:
    with pytest.raises(ValueError, match="scene generation"):
        transition(
            AssistanceSnapshot(),
            AcceptMotion(MotionState.STABLE, scene_generation=scene_generation),
        )
