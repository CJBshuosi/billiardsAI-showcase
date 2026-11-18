"""Pure Story 2.9 merge of objective observations and optional adjudication."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from billiards_ai.shared.constants import (
    CONFIDENCE_ABSTAIN_BELOW,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)
from billiards_ai.shared.models.adjudication import AdjudicationResult, AdjudicationStatus
from billiards_ai.shared.models.observation import (
    ObjectiveShotObservation,
    ObservationConfidence,
    ObservationField,
    UnsupportedReason,
)
from billiards_ai.shared.models.quality import (
    FieldQuality,
    FinalFieldSource,
    SessionQualityStatus,
    SessionQualitySummary,
    ShotAnalysisStatus,
    ShotFieldQuality,
)
from billiards_ai.shared.models.shot import Confidence, Shot, TrajectoryUnavailableReason


def _adjudication_candidates(observation: ObjectiveShotObservation) -> tuple[int, ...]:
    candidates = {
        candidate.ball_id
        for candidate in observation.struck_ball_candidates
        if candidate.confidence in {ObservationConfidence.LOW, ObservationConfidence.MEDIUM}
    }
    if observation.struck_ball.value is not None and observation.struck_ball.confidence in {
        ObservationConfidence.LOW,
        ObservationConfidence.MEDIUM,
    }:
        candidates.add(observation.struck_ball.value)
    return tuple(sorted(candidates))


def _adjudication_signature(observation: ObjectiveShotObservation) -> str:
    # Keep the evidence binding canonical with the provider request, without media access.
    from billiards_ai.cloud.perception.adjudicator import AmbiguityAdjudicator

    return AmbiguityAdjudicator._signature(AmbiguityAdjudicator._candidates(observation))


def _numeric_confidence(confidence: ObservationConfidence) -> float:
    return {
        ObservationConfidence.HIGH: float(CONFIDENCE_HIGH),
        ObservationConfidence.MEDIUM: float(CONFIDENCE_MEDIUM),
        ObservationConfidence.LOW: max(0.0, float(CONFIDENCE_ABSTAIN_BELOW) - 0.01),
        ObservationConfidence.UNKNOWN: 0.0,
    }[confidence]


def _finalize_observation_field[ObservationValue](
    field: ObservationField[ObservationValue],
) -> tuple[ObservationValue | None, FieldQuality]:
    confidence = _numeric_confidence(field.confidence)
    source = FinalFieldSource.CV if field.value is not None else FinalFieldSource.UNKNOWN
    value = field.value if confidence >= CONFIDENCE_ABSTAIN_BELOW else None
    return value, FieldQuality(source=source, confidence=confidence)


def _shot_confidence(field_quality: ShotFieldQuality) -> Confidence:
    minimum = min(quality.confidence for quality in field_quality.values())
    if minimum < CONFIDENCE_ABSTAIN_BELOW:
        return Confidence.ABSTAIN
    if minimum >= CONFIDENCE_HIGH:
        return Confidence.HIGH
    if minimum >= CONFIDENCE_MEDIUM:
        return Confidence.MEDIUM
    return Confidence.LOW


def finalize_shot(
    *,
    shot_id: UUID,
    session_id: UUID,
    sequence: int,
    observation: ObjectiveShotObservation,
    adjudication: AdjudicationResult | None = None,
    created_at: datetime,
) -> Shot:
    """Build one final Shot without media, provider, detector, or network access."""

    if adjudication is not None and adjudication.shot_id != shot_id:
        raise ValueError("adjudication shot_id must match final shot_id")
    if (
        adjudication is not None
        and adjudication.status is AdjudicationStatus.SUCCEEDED
        and observation.struck_ball.confidence is not ObservationConfidence.HIGH
    ):
        if adjudication.evidence_signature != _adjudication_signature(observation):
            raise ValueError("adjudication evidence signature must match observation candidates")
        result_ball_id = adjudication.result.ball_id if adjudication.result is not None else None
        if result_ball_id not in _adjudication_candidates(observation):
            raise ValueError("adjudication result must select an observation candidate")

    before, before_quality = _finalize_observation_field(observation.ball_positions_before)
    after, after_quality = _finalize_observation_field(observation.ball_positions_after)
    potted, potted_quality = _finalize_observation_field(observation.potted_balls)
    cue_end, cue_end_quality = _finalize_observation_field(observation.cue_ball_end)
    struck_ball, struck_quality = _finalize_observation_field(observation.struck_ball)
    cushion_count, cushion_quality = _finalize_observation_field(observation.cushion_count)
    travel_distance, travel_quality = _finalize_observation_field(observation.travel_distance_mm)
    trajectory, trajectory_quality = _finalize_observation_field(observation.trajectory)
    is_foul, foul_quality = _finalize_observation_field(observation.is_foul)

    trajectory_unavailable_reason = None
    if trajectory is None:
        trajectory_unavailable_reason = (
            TrajectoryUnavailableReason.LOW_CONFIDENCE
            if observation.trajectory.value is not None
            else (
                TrajectoryUnavailableReason.INSUFFICIENT_POINTS
                if observation.trajectory.unsupported_reason
                is UnsupportedReason.TRAJECTORY_INSUFFICIENT_POINTS
                else TrajectoryUnavailableReason.NOT_OBSERVED
            )
        )

    if trajectory is not None and len(set(trajectory)) < 2:
        trajectory = None
        trajectory_unavailable_reason = TrajectoryUnavailableReason.INSUFFICIENT_POINTS
        trajectory_quality = FieldQuality(
            source=FinalFieldSource.UNKNOWN,
            confidence=0.0,
        )
    if trajectory is not None and observation.trajectory_calibration is None:
        trajectory = None
        trajectory_unavailable_reason = TrajectoryUnavailableReason.NOT_OBSERVED
        trajectory_quality = FieldQuality(
            source=FinalFieldSource.UNKNOWN,
            confidence=0.0,
        )

    if (
        adjudication is not None
        and adjudication.status is AdjudicationStatus.SUCCEEDED
        and observation.struck_ball.confidence
        in {ObservationConfidence.LOW, ObservationConfidence.MEDIUM, ObservationConfidence.UNKNOWN}
    ):
        adjudication_confidence = _numeric_confidence(adjudication.confidence)
        struck_ball = (
            adjudication.result.ball_id
            if adjudication_confidence >= CONFIDENCE_ABSTAIN_BELOW
            else None
        )
        struck_quality = FieldQuality(
            source=FinalFieldSource.VLM,
            confidence=adjudication_confidence,
        )

    field_quality = ShotFieldQuality(
        ball_positions_before=before_quality,
        ball_positions_after=after_quality,
        cue_ball_end=cue_end_quality,
        struck_ball=struck_quality,
        potted_balls=potted_quality,
        cushion_count=cushion_quality,
        travel_distance_mm=travel_quality,
        is_foul=foul_quality,
        trajectory=trajectory_quality,
    )
    confidence = _shot_confidence(field_quality)
    raw_observation = {"observation": observation.raw_observation.model_dump()}
    if adjudication is not None:
        raw_observation["adjudication"] = adjudication.raw_observation.model_dump()

    return Shot(
        id=shot_id,
        session_id=session_id,
        sequence=sequence,
        t_start_ms=observation.t_start_ms,
        t_end_ms=observation.t_end_ms,
        ball_positions_before=(
            {item.ball_id: item.position for item in before} if before is not None else None
        ),
        ball_positions_after=(
            {item.ball_id: item.position for item in after} if after is not None else None
        ),
        cue_ball_end=cue_end,
        struck_ball=struck_ball,
        potted_balls=list(potted) if potted is not None else None,
        cushion_count=cushion_count,
        travel_distance_mm=travel_distance,
        is_foul=is_foul,
        target_zone=None,
        target_zone_source=observation.target_zone_source,
        deviation_mm=observation.deviation_mm,
        confidence=confidence,
        field_quality=field_quality,
        analysis_status=(
            ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE
            if confidence is Confidence.ABSTAIN
            else ShotAnalysisStatus.ANALYZED
        ),
        trajectory=list(trajectory) if trajectory is not None else None,
        trajectory_calibration=observation.trajectory_calibration,
        trajectory_unavailable_reason=trajectory_unavailable_reason,
        raw_observation=raw_observation,
        created_at=created_at,
    )


def assess_session_quality(shots: tuple[Shot, ...]) -> SessionQualitySummary:
    shot_count = len(shots)
    abstained_count = sum(shot.is_abstained for shot in shots)
    abstention_rate = abstained_count / shot_count if shot_count else 0.0
    return SessionQualitySummary(
        shot_count=shot_count,
        abstained_shot_count=abstained_count,
        abstention_rate=abstention_rate,
        session_quality_status=(
            SessionQualityStatus.LOW_COVERAGE
            if abstention_rate > 0.5
            else SessionQualityStatus.SUFFICIENT_COVERAGE
        ),
    )
