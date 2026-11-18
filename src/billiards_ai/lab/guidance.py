"""Pure local orchestration for legal direct-shot guidance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from billiards_ai.lab.adapters.pooltool_physics import PhysicsBackend
from billiards_ai.lab.planning.geometry import plan_direct_shot
from billiards_ai.lab.planning.ranking import rank_candidates
from billiards_ai.lab.planning.rules import EightBallContext, legal_targets
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PhysicsResult,
    PhysicsStatus,
    PocketId,
    PowerBand,
    Recommendation,
)
from billiards_ai.shared.models.table_state import CanonicalTableState


@dataclass(frozen=True, slots=True)
class ExplicitIntent:
    target_track_id: str
    pocket: PocketId

    def __post_init__(self) -> None:
        if not self.target_track_id:
            raise ValueError("explicit target track ID is required")


def _abstained(state: CanonicalTableState, reason: str) -> Recommendation:
    return Recommendation(
        status=GuidanceStatus.ABSTAINED,
        candidates=(),
        calibration_id=state.calibration_id,
        evidence_refs=state.source_refs,
        planner_config_hash=hashlib.sha256(b"mainline1-guidance/1").hexdigest(),
        fallback_reason=reason,
    )


def _automatic_candidates(
    state: CanonicalTableState,
    context: EightBallContext,
) -> tuple[list, str | None]:
    targets = legal_targets(state, context)
    if targets.status is GuidanceStatus.ABSTAINED:
        return [], targets.reason
    candidates = []
    for target_track_id in targets.track_ids:
        for pocket in PocketId:
            try:
                candidate = plan_direct_shot(
                    state=state,
                    target_track_id=target_track_id,
                    pocket=pocket,
                )
            except ValueError:
                continue
            if candidate.status is GuidanceStatus.PLANNED:
                candidates.append(candidate)
    return candidates, None


def _simulate(
    backend: PhysicsBackend,
    state: CanonicalTableState,
    candidates: list,
) -> tuple[dict[str, PhysicsResult], str | None]:
    results: dict[str, PhysicsResult] = {}
    fallback_reason = None
    for candidate in candidates:
        result = backend.simulate(state, candidate, PowerBand.MEDIUM)
        results[candidate.candidate_id] = result
        if result.status is not PhysicsStatus.SIMULATED and fallback_reason is None:
            fallback_reason = result.reason
    return results, fallback_reason


def build_recommendation(
    *,
    state: CanonicalTableState,
    context: EightBallContext | None = None,
    intent: ExplicitIntent | None = None,
    physics: PhysicsBackend | None = None,
    top_n: int = 3,
) -> Recommendation:
    state.require_plannable()
    if (context is None) == (intent is None):
        raise ValueError("provide exactly one of context or explicit intent")
    fallback_reason = None
    if intent is not None:
        candidate = plan_direct_shot(
            state=state,
            target_track_id=intent.target_track_id,
            pocket=intent.pocket,
        )
        candidates = [candidate] if candidate.status is GuidanceStatus.PLANNED else []
        if not candidates:
            fallback_reason = ",".join(candidate.reasons) or "unsupported_explicit_intent"
    else:
        assert context is not None
        candidates, fallback_reason = _automatic_candidates(state, context)
    if not candidates:
        return _abstained(state, fallback_reason or "no_supported_candidates")

    physics_results: dict[str, PhysicsResult] = {}
    if physics is not None:
        physics_results, physics_fallback = _simulate(physics, state, candidates)
        fallback_reason = physics_fallback or fallback_reason
    recommendation = rank_candidates(
        candidates,
        physics=physics_results,
        calibration_id=state.calibration_id,
        top_n=top_n,
    )
    if fallback_reason:
        recommendation = recommendation.model_copy(update={"fallback_reason": fallback_reason})
    return recommendation
