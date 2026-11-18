"""Adapt existing Top-3 planning into truthful Top-1 field presentation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
from billiards_ai.edge.route_assistance.models import GuidanceDisplay, PowerDisplay
from billiards_ai.lab.guidance import build_recommendation
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.models.guidance import (
    GuidanceScene,
    GuidanceStatus,
    PlannedShot,
    Recommendation,
)
from billiards_ai.shared.models.table_state import CanonicalTableState

_POWER_THRESHOLDS = (
    (700.0, "light", 20),
    (1100.0, "medium_light", 35),
    (1600.0, "medium", 50),
    (2200.0, "medium_firm", 70),
)


@dataclass(frozen=True, slots=True)
class FieldGuidanceResult:
    recommendation: Recommendation
    scene: GuidanceScene | None
    display: GuidanceDisplay | None
    abstained_reason: str | None = None


def build_match_field_guidance(
    *,
    state: CanonicalTableState,
    match: MatchControlSnapshot,
    scene_generation: int,
) -> FieldGuidanceResult:
    """Select target context from the authoritative current player, then reuse the planner."""
    if match.active_role is None:
        raise ValueError("active_player_unknown")
    group = match.red_group if match.active_role is WristbandRole.RED else match.blue_group
    return build_field_guidance(
        state=state,
        context=EightBallContext(group="open" if group is None else group.value),
        scene_generation=scene_generation,
    )


def generic_power_display(candidate: PlannedShot) -> PowerDisplay:
    """Map route length to an explicitly generic, non-personalized cue."""

    if candidate.cue_travel_mm is None or candidate.object_travel_mm is None:
        raise ValueError("generic power requires complete route travel")
    travel_mm = candidate.cue_travel_mm + candidate.object_travel_mm
    for maximum, band, percent in _POWER_THRESHOLDS:
        if travel_mm <= maximum:
            return PowerDisplay(band=band, approximate_percent=percent)  # type: ignore[arg-type]
    return PowerDisplay(band="firm", approximate_percent=90)


def _recommendation_identity(
    recommendation: Recommendation,
    scene_generation: int,
) -> str:
    payload = (
        recommendation.model_dump_json(exclude_none=True) + f"|scene_generation={scene_generation}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_field_guidance(
    *,
    state: CanonicalTableState,
    context: EightBallContext,
    scene_generation: int,
) -> FieldGuidanceResult:
    """Keep full Top-3 evidence while exposing only the trustworthy rank one route."""

    if scene_generation < 0:
        raise ValueError("scene generation must be non-negative")
    recommendation = build_recommendation(
        state=state,
        context=context,
        top_n=3,
    )
    if not recommendation.candidates or recommendation.status in {
        GuidanceStatus.ABSTAINED,
        GuidanceStatus.UNSUPPORTED,
    }:
        return FieldGuidanceResult(
            recommendation=recommendation,
            scene=None,
            display=None,
            abstained_reason=recommendation.fallback_reason or "no_supported_candidates",
        )

    selected = recommendation.candidates[0]
    recommendation_id = _recommendation_identity(recommendation, scene_generation)
    scene = GuidanceScene(
        recommendation_id=recommendation_id,
        calibration_id=recommendation.calibration_id,
        routes=(selected,),
        evidence_refs=recommendation.evidence_refs,
    )
    display = GuidanceDisplay(
        recommendation_id=recommendation_id,
        candidate_id=selected.candidate_id,
        scene_generation=scene_generation,
        power=generic_power_display(selected),
    )
    return FieldGuidanceResult(
        recommendation=recommendation,
        scene=scene,
        display=display,
    )
