"""Deterministic score decomposition and Top-N recommendation selection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import hypot

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PhysicsResult,
    PhysicsStatus,
    PlannedShot,
    Recommendation,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _geometry_components(candidate: PlannedShot) -> dict[str, float]:
    diagonal = hypot(TABLE_LENGTH_MM, TABLE_WIDTH_MM)
    cut_difficulty = _clamp(1.0 - float(candidate.cut_angle_deg or 0) / 90.0)
    pot_tolerance = _clamp(1.0 - float(candidate.object_travel_mm or 0) / diagonal)
    travel_score = _clamp(
        1.0
        - (float(candidate.cue_travel_mm or 0) + float(candidate.object_travel_mm or 0))
        / (2 * diagonal)
    )
    return {
        "pot_tolerance": pot_tolerance,
        "cut_difficulty": cut_difficulty,
        "travel_score": travel_score,
        "evidence_confidence": candidate.evidence_confidence,
    }


def rank_candidates(
    candidates: Sequence[PlannedShot],
    *,
    physics: Mapping[str, PhysicsResult],
    calibration_id: str,
    top_n: int = 3,
) -> Recommendation:
    if not 1 <= top_n <= 3:
        raise ValueError("top_n must be between one and three")
    ranked: list[PlannedShot] = []
    simulated = False
    backend_versions: set[str] = set()
    for candidate in candidates:
        components = _geometry_components(candidate)
        score = (
            0.35 * components["pot_tolerance"]
            + 0.25 * components["cut_difficulty"]
            + 0.20 * components["travel_score"]
            + 0.20 * components["evidence_confidence"]
        )
        result = physics.get(candidate.candidate_id)
        if result is not None:
            backend_versions.add(result.backend_version)
            if result.status is PhysicsStatus.SIMULATED:
                simulated = True
                components["cue_ball_control"] = result.cue_ball_control_score
                score = 0.65 * score + 0.35 * result.cue_ball_control_score
        ranked.append(
            candidate.model_copy(update={"score_components": components, "score": _clamp(score)})
        )
    ranked.sort(key=lambda item: (-item.score, item.candidate_id))
    selected = tuple(ranked[:top_n])
    refs = tuple(sorted({ref for item in selected for ref in item.evidence_refs}))
    if not refs:
        refs = ("planner:no_supported_candidates",)
    config_hash = hashlib.sha256(b"mainline1-ranking/1").hexdigest()
    return Recommendation(
        status=(
            GuidanceStatus.SIMULATED
            if simulated
            else GuidanceStatus.GEOMETRY_ONLY
            if selected
            else GuidanceStatus.ABSTAINED
        ),
        candidates=selected,
        calibration_id=calibration_id,
        evidence_refs=refs,
        planner_config_hash=config_hash,
        backend_versions=tuple(sorted(backend_versions)),
        fallback_reason=None if selected else "no_supported_candidates",
    )
