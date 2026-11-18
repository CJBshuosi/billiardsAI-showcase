"""Synthetic-only acceptance tests for Story 2.9."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4


def _segment():
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.segmentation import BallPosition, ShotSegment, TableSnapshot

    def snapshot(timestamp_ms, positions):
        return TableSnapshot(
            timestamp_ms=timestamp_ms,
            positions=tuple(
                BallPosition(ball_id=ball_id, position=Point(x_mm=x, y_mm=y))
                for ball_id, (x, y) in sorted(positions.items())
            ),
        )

    return ShotSegment(
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=snapshot(400, {0: (100, 100), 1: (400, 400), 2: (700, 700)}),
        ball_positions_after=snapshot(900, {0: (200, 100), 1: (450, 400), 2: (700, 700)}),
    )


def _observation(*, fully_supported=False):
    from billiards_ai.cloud.perception.observer import observe_shot

    observation = observe_shot(_segment())
    if not fully_supported:
        return observation

    from billiards_ai.shared.models.observation import (
        ObservationConfidence,
        ObservationField,
        ObservationSource,
    )

    supported = {
        "confidence": ObservationConfidence.HIGH,
        "source": ObservationSource.SEGMENT_SNAPSHOT_DELTA,
    }
    return observation.model_copy(
        update={
            "struck_ball": ObservationField(value=1, **supported),
            "cushion_count": ObservationField(value=1, **supported),
            "is_foul": ObservationField(value=False, **supported),
        }
    )


def _adjudication(shot_id, *, status, confidence="MEDIUM", ball_id=None, evidence_signature=None):
    from billiards_ai.shared.models.adjudication import (
        AdjudicatedStruckBall,
        AdjudicationRawObservation,
        AdjudicationResult,
        AdjudicationSource,
        AdjudicationStatus,
    )
    from billiards_ai.shared.models.observation import ObservationConfidence

    typed_status = AdjudicationStatus(status)
    succeeded = typed_status is AdjudicationStatus.SUCCEEDED
    if evidence_signature is None:
        from billiards_ai.cloud.perception.adjudicator import AmbiguityAdjudicator

        evidence_signature = AmbiguityAdjudicator._signature(
            AmbiguityAdjudicator._candidates(_observation())
        )
    return AdjudicationResult(
        shot_id=shot_id,
        idempotency_key="a" * 64,
        result=AdjudicatedStruckBall(ball_id=ball_id) if succeeded else None,
        source=(AdjudicationSource.EXTERNAL_PROVIDER if succeeded else AdjudicationSource.NONE),
        confidence=(
            ObservationConfidence(confidence) if succeeded else ObservationConfidence.UNKNOWN
        ),
        status=typed_status,
        retryable=False,
        estimated_cost_cny=Decimal("0.00"),
        evidence_signature=evidence_signature,
        raw_observation=AdjudicationRawObservation(
            failure_detail="synthetic" if not succeeded else None
        ),
    )


def _finalize(observation, *, shot_id=None, adjudication=None):
    from billiards_ai.cloud.perception.finalizer import finalize_shot

    return finalize_shot(
        shot_id=shot_id or uuid4(),
        session_id=uuid4(),
        sequence=1,
        observation=observation,
        adjudication=adjudication,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_supported_cv_observation_finalizes_all_scalars_with_field_provenance():
    from billiards_ai.shared.constants import CONFIDENCE_HIGH
    from billiards_ai.shared.models.quality import FinalFieldSource, ShotAnalysisStatus
    from billiards_ai.shared.models.shot import Confidence

    shot = _finalize(_observation(fully_supported=True))

    assert shot.confidence is Confidence.HIGH
    assert shot.analysis_status is ShotAnalysisStatus.ANALYZED
    assert shot.struck_ball == 1
    assert shot.cushion_count == 1
    assert shot.travel_distance_mm == 100
    assert shot.is_foul is False
    assert shot.trajectory is None
    for quality in shot.field_quality.values():
        assert quality.source is FinalFieldSource.CV
        assert quality.confidence == CONFIDENCE_HIGH


def test_unknown_fields_abstain_without_erasing_trusted_cv_fields():
    from billiards_ai.shared.models.quality import FinalFieldSource, ShotAnalysisStatus
    from billiards_ai.shared.models.shot import Confidence

    shot = _finalize(_observation())

    assert shot.confidence is Confidence.ABSTAIN
    assert shot.analysis_status is ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE
    assert shot.has_positional_verdict is False
    assert shot.struck_ball is None
    assert shot.cushion_count is None
    assert shot.is_foul is None
    assert shot.cue_ball_end is not None
    assert shot.travel_distance_mm == 100
    assert shot.ball_positions_before[0].x_mm == 100
    assert shot.field_quality.struck_ball.source is FinalFieldSource.UNKNOWN
    assert shot.field_quality.travel_distance_mm.source is FinalFieldSource.CV


def test_successful_adjudication_only_replaces_struck_ball_and_marks_vlm_source():
    from billiards_ai.shared.constants import CONFIDENCE_MEDIUM
    from billiards_ai.shared.models.quality import FinalFieldSource

    observation = _observation()
    shot_id = uuid4()
    adjudication = _adjudication(shot_id, status="SUCCEEDED", ball_id=1)

    shot = _finalize(observation, shot_id=shot_id, adjudication=adjudication)

    assert shot.struck_ball == 1
    assert shot.field_quality.struck_ball.source is FinalFieldSource.VLM
    assert shot.field_quality.struck_ball.confidence == CONFIDENCE_MEDIUM
    assert shot.travel_distance_mm == 100
    assert shot.field_quality.travel_distance_mm.source is FinalFieldSource.CV


def test_disabled_failed_or_missing_adjudication_never_blocks_trusted_fields():
    observation = _observation()

    missing = _finalize(observation)
    disabled_id = uuid4()
    disabled = _finalize(
        observation,
        shot_id=disabled_id,
        adjudication=_adjudication(disabled_id, status="DISABLED"),
    )
    failed_id = uuid4()
    failed = _finalize(
        observation,
        shot_id=failed_id,
        adjudication=_adjudication(failed_id, status="PARSE_FAILED"),
    )

    for shot in (missing, disabled, failed):
        assert shot.struck_ball is None
        assert shot.cue_ball_end.x_mm == 200
        assert shot.travel_distance_mm == 100
        assert shot.potted_balls == []


def test_adjudication_below_shared_abstain_threshold_cannot_create_a_value():
    from billiards_ai.shared.models.quality import FinalFieldSource
    from billiards_ai.shared.models.shot import Confidence

    shot_id = uuid4()
    shot = _finalize(
        _observation(),
        shot_id=shot_id,
        adjudication=_adjudication(shot_id, status="SUCCEEDED", confidence="LOW", ball_id=1),
    )

    assert shot.struck_ball is None
    assert shot.field_quality.struck_ball.source is FinalFieldSource.VLM
    assert shot.confidence is Confidence.ABSTAIN


def test_low_confidence_collection_fields_remain_none_instead_of_empty_claims():
    from billiards_ai.shared.models.observation import (
        ObservationConfidence,
        ObservationField,
        ObservationSource,
    )

    observation = _observation(fully_supported=True)
    low = {
        "confidence": ObservationConfidence.LOW,
        "source": ObservationSource.SEGMENT_SNAPSHOT_DELTA,
    }
    observation = observation.model_copy(
        update={
            "ball_positions_before": ObservationField(
                value=observation.ball_positions_before.value, **low
            ),
            "ball_positions_after": ObservationField(
                value=observation.ball_positions_after.value, **low
            ),
            "potted_balls": ObservationField(value=observation.potted_balls.value, **low),
        }
    )

    shot = _finalize(observation)

    assert shot.ball_positions_before is None
    assert shot.ball_positions_after is None
    assert shot.potted_balls is None


def test_adjudication_for_another_shot_is_rejected():
    import pytest

    adjudication = _adjudication(uuid4(), status="SUCCEEDED", ball_id=1)

    with pytest.raises(ValueError, match="shot_id"):
        _finalize(_observation(), shot_id=uuid4(), adjudication=adjudication)


def test_adjudication_cannot_override_high_confidence_cv_result():
    observation = _observation(fully_supported=True)
    shot_id = uuid4()

    shot = _finalize(
        observation,
        shot_id=shot_id,
        adjudication=_adjudication(shot_id, status="SUCCEEDED", ball_id=2),
    )

    assert shot.struck_ball == 1


def test_adjudication_must_bind_to_current_candidate_evidence():
    import pytest

    shot_id = uuid4()
    adjudication = _adjudication(
        shot_id,
        status="SUCCEEDED",
        ball_id=1,
        evidence_signature="b" * 64,
    )

    with pytest.raises(ValueError, match="evidence signature"):
        _finalize(_observation(), shot_id=shot_id, adjudication=adjudication)


def test_adjudication_cannot_introduce_ball_outside_observation_candidates():
    import pytest

    shot_id = uuid4()
    adjudication = _adjudication(shot_id, status="SUCCEEDED", ball_id=9)

    with pytest.raises(ValueError, match="candidate"):
        _finalize(_observation(), shot_id=shot_id, adjudication=adjudication)


def test_more_than_half_abstained_shots_produce_low_coverage_but_exactly_half_does_not():
    from billiards_ai.cloud.perception.finalizer import assess_session_quality
    from billiards_ai.shared.models.quality import SessionQualityStatus

    supported = _finalize(_observation(fully_supported=True))
    abstained = _finalize(_observation())

    low = assess_session_quality((abstained, abstained, supported))
    boundary = assess_session_quality((abstained, supported))

    assert low.shot_count == 3
    assert low.abstained_shot_count == 2
    assert low.session_quality_status is SessionQualityStatus.LOW_COVERAGE
    assert boundary.session_quality_status is SessionQualityStatus.SUFFICIENT_COVERAGE
