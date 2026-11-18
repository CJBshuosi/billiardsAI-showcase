"""Behavioral ATDD for Story 8.3 deterministic ability metric projection."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.growth import project_approved_shots
from billiards_ai.cloud.growth.metric_projector import (
    _effective_n,
    _Observation,
    _reliability,
    project_ability_metrics,
)
from billiards_ai.cloud.growth.taxonomy import (
    CALCULATION_POLICY_VERSION,
    TAXONOMY_VERSION,
    classify_angle_band,
    classify_distance_band,
    classify_rail_relation,
    cut_angle_deg,
    shot_distance_mm,
)
from billiards_ai.shared.models import (
    ApprovedEvidenceBatch,
    ApprovedShotData,
    Confidence,
    ReviewStatus,
    ShotAnalysisStatus,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.growth_metrics import (
    AbilityDimension,
    AbilityTaxonomyV2,
    AngleBand,
    DistanceBand,
    MetricCode,
    MetricProjectionStatus,
    MetricReliability,
    MetricSupportStatusV2,
    ProjectedAbilityMetric,
    RailRelation,
    TaxonomyApplicability,
    TechniqueCode,
)
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    GameCode,
    MetricFieldQuality,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    ReviewedFieldQuality,
    ReviewedPoint,
    ReviewedTableGeometry,
    ReviewedTargetZone,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
    TargetZoneShape,
)
from billiards_ai.shared.models.shot import TargetZoneSource

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _quality(confidence: float = 1.0) -> ReviewedFieldQuality:
    source = FieldSource.UNKNOWN if confidence == 0.0 else FieldSource.MANUAL
    return ReviewedFieldQuality(source=source, confidence=confidence)


def _qualities(confidence: float = 1.0) -> MetricFieldQuality:
    quality = _quality(confidence)
    return MetricFieldQuality(
        geometry=quality,
        intent=quality,
        pot_attempt=quality,
        pot_result=quality,
        foul_result=quality,
        target_zone=quality,
        cue_ball_end=quality,
    )


def _calibration() -> dict[str, object]:
    return {
        "calibration_id": _id(1),
        "calibration_revision_id": _id(2),
        "calibration_revision_number": 1,
    }


def _point(x_mm: float, y_mm: float) -> ReviewedPoint:
    return ReviewedPoint(**_calibration(), x_mm=x_mm, y_mm=y_mm)


def _table() -> ReviewedTableGeometry:
    return ReviewedTableGeometry(
        **_calibration(),
        table_length_mm=2540,
        table_width_mm=1270,
        ball_diameter_mm=57.15,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
    )


def _approved_shot(session_id: UUID, shot_id: UUID, sequence: int, **updates: object):
    values: dict[str, object] = {
        "shot_id": shot_id,
        "session_id": session_id,
        "sequence": sequence,
        "struck_ball": 3,
        "potted_balls": (3,),
        "cushion_count": 0,
        "travel_distance_mm": 999,
        "is_foul": False,
        "deviation_mm": 999,
        "target_zone_source": TargetZoneSource.NONE,
        "confidence": Confidence.HIGH,
        "analysis_status": ShotAnalysisStatus.ANALYZED,
        "review_status": ReviewStatus.APPROVED,
    }
    return ApprovedShotData.model_validate(values | updates)


def _metric_shot(session_id: UUID, shot_id: UUID, sequence: int, **updates: object):
    values: dict[str, object] = {
        "shot_id": shot_id,
        "session_id": session_id,
        "sequence": sequence,
        "review_revision": 1,
        "game_code": GameCode.CHINESE_EIGHT_BALL,
        "table_geometry": _table(),
        "cue_ball_start_mm": _point(100.0, 300.0),
        "object_ball_start_mm": _point(500.0, 300.0),
        "intended_object_ball": 3,
        "intended_target_pocket": PocketCode.BOTTOM_RIGHT,
        "pot_attempt_status": PotAttemptStatus.ATTEMPTED,
        "pot_result": PotResult.LEGAL_SUCCESS,
        "foul_result": FoulResult.LEGAL,
        "target_zone_source": TargetZoneSource.NONE,
        "target_zone": None,
        "cue_ball_end_mm": None,
        "target_zone_evaluation_status": TargetZoneEvaluationStatus.NOT_APPLICABLE,
        "target_zone_result": None,
        "field_quality": _qualities(),
    }
    return ApprovedMetricShotEvidence.model_validate(values | updates)


def _batch(
    *,
    batch_number: int,
    session_number: int,
    shots: tuple[ApprovedMetricShotEvidence, ...],
    approved_at: datetime = NOW,
    approval_kind: EvidenceApprovalKind = EvidenceApprovalKind.HUMAN_REVIEW,
) -> ApprovedEvidenceBatchV2:
    session_id = _id(session_number)
    approved_shots = tuple(
        _approved_shot(session_id, item.shot_id, item.sequence) for item in shots
    )
    statistics, scores = project_approved_shots(approved_shots)
    payload = {
        "schema_version": "2.0",
        "batch_id": _id(batch_number),
        "user_id": _id(900),
        "session_id": session_id,
        "report_id": _id(batch_number + 1000),
        "report_revision": 1,
        "review_revision": 1,
        "approved_at": approved_at,
        "approval_kind": approval_kind,
        "approval_policy_version": (
            "human-review-v1"
            if approval_kind is EvidenceApprovalKind.HUMAN_REVIEW
            else "automated-policy-rd-v1"
        ),
        "shots": approved_shots,
        "metric_shots": shots,
        "statistics": statistics,
        "scores": scores,
        "source_envelope_ids": tuple(_id(batch_number * 100 + i) for i in range(1, len(shots) + 1)),
    }
    return ApprovedEvidenceBatchV2.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )


def _pot_batches(
    count: int,
    sessions: int,
    *,
    successes: int | None = None,
    approved_at: datetime = NOW,
) -> tuple[ApprovedEvidenceBatchV2, ...]:
    successes = count if successes is None else successes
    grouped: list[list[ApprovedMetricShotEvidence]] = [[] for _ in range(sessions)]
    for index in range(count):
        session_number = 100 + (index % sessions)
        grouped[index % sessions].append(
            _metric_shot(
                _id(session_number),
                _id(10_000 + index),
                len(grouped[index % sessions]) + 1,
                pot_result=(PotResult.LEGAL_SUCCESS if index < successes else PotResult.MISS),
            )
        )
    return tuple(
        _batch(
            batch_number=200 + index,
            session_number=100 + index,
            shots=tuple(values),
            approved_at=approved_at,
        )
        for index, values in enumerate(grouped)
    )


def _accepted(
    batches: tuple[ApprovedEvidenceBatchV2, ...],
):
    from billiards_ai.cloud.growth.repository import InMemoryGrowthEvidenceRepository
    from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord

    repository = InMemoryGrowthEvidenceRepository()
    ids = iter(_id(800_000 + index) for index in range(len(batches) * 2))
    accepted = []
    for batch in batches:
        result = repository.append_approved_batch(
            batch,
            metric_authority=ApprovedMetricEvidenceAuthorityRecord(
                approval_kind=batch.approval_kind,
                approval_policy_version=batch.approval_policy_version,
                metric_shots=batch.metric_shots,
            ),
            clock=lambda: NOW,
            uuid_factory=lambda: next(ids),
        )
        assert result.accepted_metric_input is not None
        accepted.append(result.accepted_metric_input)
    return tuple(accepted)


def _project(
    batches: tuple[ApprovedEvidenceBatchV2, ...],
    *,
    evidence_cutoff: datetime = NOW,
):
    return project_ability_metrics(_accepted(batches), evidence_cutoff=evidence_cutoff)


def _metric(result, code: MetricCode):
    matches = tuple(item for item in result.metrics if item.metric_code is code)
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, AngleBand.STRAIGHT),
        (5.0, AngleBand.STRAIGHT),
        (5.001, AngleBand.SMALL),
        (30.0, AngleBand.SMALL),
        (30.001, AngleBand.MEDIUM),
        (50.0, AngleBand.MEDIUM),
        (50.001, AngleBand.LARGE),
        (75.0, AngleBand.LARGE),
        (75.001, AngleBand.EXTREME),
        (90.0, AngleBand.EXTREME),
    ],
)
def test_angle_band_boundaries_use_approved_half_up_values(angle: float, expected: AngleBand):
    assert classify_angle_band(angle) is expected


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (449.999, DistanceBand.VERY_SHORT),
        (450.0, DistanceBand.SHORT),
        (900.0, DistanceBand.MEDIUM),
        (1300.0, DistanceBand.LONG),
        (1800.0, DistanceBand.VERY_LONG),
    ],
)
def test_distance_band_boundaries_are_exact(distance: float, expected: DistanceBand):
    assert classify_distance_band(distance) is expected


def test_rail_relation_uses_object_ball_edge_gap_and_rejects_pocket_jaw() -> None:
    assert classify_rail_relation(_point(500.0, 28.575), _table()) is RailRelation.FROZEN_TO_RAIL
    assert classify_rail_relation(_point(500.0, 85.725), _table()) is RailRelation.NEAR_RAIL
    assert classify_rail_relation(_point(500.0, 85.726), _table()) is RailRelation.OPEN_TABLE
    assert classify_rail_relation(_point(20.0, 20.0), _table()) is None


def test_approved_geometry_uses_cue_object_distance_and_object_pocket_cut_angle() -> None:
    assert shot_distance_mm(_point(0.0, 0.0), _point(300.0, 400.0)) == 500.0
    assert cut_angle_deg(_point(0.0, 0.0), _point(100.0, 0.0), PocketCode.BOTTOM_RIGHT) == 0.0
    assert (
        cut_angle_deg(_point(1170.0, 635.0), _point(1270.0, 635.0), PocketCode.TOP_CENTER) == 90.0
    )


def test_v1_batch_fails_closed_without_using_proxy_fields() -> None:
    v2 = _pot_batches(5, 2)[0]
    v1_payload = v2.model_dump(
        mode="python",
        exclude={"metric_shots", "approval_kind", "approval_policy_version", "content_sha256"},
    ) | {"schema_version": "1.0"}
    v1 = ApprovedEvidenceBatch.model_validate(
        v1_payload | {"content_sha256": canonical_sha256(v1_payload)}
    )

    result = project_ability_metrics((v1,), evidence_cutoff=NOW)

    assert result.status is MetricProjectionStatus.UNSUPPORTED_INPUT_CONTRACT
    assert result.metrics == ()


def test_potting_projection_uses_authoritative_results_and_exact_low_threshold() -> None:
    result = _project(_pot_batches(5, 2, successes=3))
    metric = _metric(result, MetricCode.POTTING_SUCCESS_RATE)

    assert result.status is MetricProjectionStatus.PROJECTED
    assert result.taxonomy_version == TAXONOMY_VERSION
    assert result.calculation_policy_version == CALCULATION_POLICY_VERSION
    assert metric.support_status is MetricSupportStatusV2.SUPPORTED
    assert metric.reliability is MetricReliability.LOW
    assert metric.sample_count == 5
    assert metric.distinct_session_count == 2
    assert metric.effective_sample_count == 5.0
    assert metric.coverage == 1.0
    assert metric.success_rate == 0.6


def test_reliability_exact_medium_and_high_equalities() -> None:
    medium = _metric(
        _project(_pot_batches(15, 3, successes=15)),
        MetricCode.POTTING_SUCCESS_RATE,
    )
    high = _metric(
        _project(_pot_batches(40, 5, successes=40)),
        MetricCode.POTTING_SUCCESS_RATE,
    )

    assert medium.reliability is MetricReliability.MEDIUM
    assert high.reliability is MetricReliability.HIGH


@pytest.mark.parametrize(
    ("effective_n", "sessions", "coverage", "uncertainty", "binary", "expected"),
    [
        (40.0, 5, 0.85, 0.15, True, MetricReliability.HIGH),
        (39.999999, 5, 0.85, 0.15, True, MetricReliability.MEDIUM),
        (40.0, 4, 0.85, 0.15, True, MetricReliability.MEDIUM),
        (40.0, 5, 0.849999, 0.15, True, MetricReliability.MEDIUM),
        (40.0, 5, 0.85, 0.150001, True, MetricReliability.MEDIUM),
        (15.0, 3, 0.70, 0.25, True, MetricReliability.MEDIUM),
        (14.999999, 3, 0.70, 0.25, True, MetricReliability.LOW),
        (5.0, 2, 0.50, 999.0, True, MetricReliability.LOW),
        (4.999999, 2, 0.50, 0.0, True, None),
        (40.0, 5, 0.85, 75.0, False, MetricReliability.HIGH),
        (40.0, 5, 0.85, 75.001, False, MetricReliability.MEDIUM),
        (15.0, 3, 0.70, 150.0, False, MetricReliability.MEDIUM),
        (15.0, 3, 0.70, 150.001, False, MetricReliability.LOW),
    ],
)
def test_reliability_policy_exact_equalities(
    effective_n: float,
    sessions: int,
    coverage: float,
    uncertainty: float,
    binary: bool,
    expected: MetricReliability | None,
) -> None:
    assert (
        _reliability(
            effective_n=effective_n,
            sessions=sessions,
            coverage=coverage,
            uncertainty=uncertainty,
            binary=binary,
        )
        is expected
    )


def test_target_zone_sources_are_separate_and_error_quantiles_use_nearest_rank_mad() -> None:
    zone = ReviewedTargetZone(
        **_calibration(),
        shape=TargetZoneShape.RECTANGLE,
        minimum_x_mm=1000.0,
        minimum_y_mm=400.0,
        maximum_x_mm=1200.0,
        maximum_y_mm=600.0,
        vertices=(),
    )
    errors = (0.0, 10.0, 20.0, 100.0, 300.0)
    batches = []
    for session_index in range(2):
        shots = tuple(
            _metric_shot(
                _id(300 + session_index),
                _id(20_000 + session_index * 10 + index),
                index + 1,
                pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
                pot_result=None,
                intended_object_ball=None,
                intended_target_pocket=None,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                target_zone=zone,
                cue_ball_end_mm=_point(1200.0 + error, 500.0),
                target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
                target_zone_result=(
                    TargetZoneResult.INSIDE_OR_TOLERATED
                    if error <= 20
                    else TargetZoneResult.OUTSIDE
                ),
            )
            for index, error in enumerate(errors)
        )
        batches.append(
            _batch(
                batch_number=400 + session_index,
                session_number=300 + session_index,
                shots=shots,
            )
        )
    post_hoc_shot = _metric_shot(
        _id(302),
        _id(20_100),
        1,
        pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
        pot_result=None,
        intended_object_ball=None,
        intended_target_pocket=None,
        target_zone_source=TargetZoneSource.POST_HOC_INFERRED,
        target_zone=zone,
        cue_ball_end_mm=_point(1200.0, 500.0),
        target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
        target_zone_result=TargetZoneResult.INSIDE_OR_TOLERATED,
    )
    batches.append(_batch(batch_number=402, session_number=302, shots=(post_hoc_shot,)))

    result = _project(tuple(batches))
    scene_errors = tuple(
        item
        for item in result.metrics
        if item.metric_code is MetricCode.TARGET_ZONE_ERROR_MM
        and item.taxonomy.target_zone_source is TargetZoneSource.SCENE_PRESET
    )
    post_hoc = tuple(
        item
        for item in result.metrics
        if item.taxonomy.target_zone_source is TargetZoneSource.POST_HOC_INFERRED
    )

    assert len(scene_errors) == 1
    assert post_hoc
    scene_success = tuple(
        item
        for item in result.metrics
        if item.metric_code is MetricCode.TARGET_ZONE_SUCCESS_RATE
        and item.taxonomy.target_zone_source is TargetZoneSource.SCENE_PRESET
    )
    assert len(scene_success) == 1
    assert scene_success[0].success_rate == 0.6
    assert scene_errors[0].error_p50_mm == 20.0
    assert scene_errors[0].error_p75_mm == 100.0
    assert scene_errors[0].error_p90_mm == 300.0
    assert scene_errors[0].mad_mm == 20.0
    assert scene_errors[0].stability == 0.92


def test_support_matrix_preserves_no_sample_unsupported_abstained_and_insufficient() -> None:
    cases = (
        (PotAttemptStatus.NOT_ATTEMPTED, 1.0, MetricSupportStatusV2.NO_SAMPLE),
        (PotAttemptStatus.UNSUPPORTED, 1.0, MetricSupportStatusV2.UNSUPPORTED),
        (PotAttemptStatus.ABSTAINED, 0.0, MetricSupportStatusV2.ABSTAINED),
        (PotAttemptStatus.ATTEMPTED, 1.0, MetricSupportStatusV2.INSUFFICIENT_EVIDENCE),
    )
    for offset, (status, confidence, expected) in enumerate(cases):
        session_id = _id(500 + offset)
        shot = _metric_shot(
            session_id,
            _id(30_000 + offset),
            1,
            pot_attempt_status=status,
            pot_result=(PotResult.LEGAL_SUCCESS if status is PotAttemptStatus.ATTEMPTED else None),
            intended_object_ball=(3 if status is PotAttemptStatus.ATTEMPTED else None),
            intended_target_pocket=(
                PocketCode.BOTTOM_RIGHT if status is PotAttemptStatus.ATTEMPTED else None
            ),
            field_quality=_qualities(confidence),
        )
        result = _project(
            (_batch(batch_number=600 + offset, session_number=500 + offset, shots=(shot,)),)
        )
        metric = _metric(result, MetricCode.POTTING_SUCCESS_RATE)
        assert metric.support_status is expected
        assert metric.reliability is MetricReliability.UNKNOWN
        assert metric.success_rate is None


def test_recency_window_uses_365_day_inclusive_boundary() -> None:
    at_boundary = _project(
        _pot_batches(5, 2, approved_at=NOW - timedelta(days=365)),
    )
    outside = _project(
        _pot_batches(5, 2, approved_at=NOW - timedelta(days=365, microseconds=1)),
    )

    assert _metric(at_boundary, MetricCode.POTTING_SUCCESS_RATE).sample_count == 5
    assert (
        _metric(outside, MetricCode.POTTING_SUCCESS_RATE).support_status
        is MetricSupportStatusV2.NO_SAMPLE
    )
    assert outside.input_batch_count == outside.outside_window_batch_count == 2
    assert outside.input_shot_count == outside.outside_window_shot_count == 5
    assert outside.included_batch_count == outside.included_shot_count == 0


def test_approval_recency_weights_and_kish_effective_n_are_exact() -> None:
    human = _pot_batches(3, 1, successes=3)[0]
    automated = _batch(
        batch_number=701,
        session_number=101,
        shots=tuple(
            _metric_shot(
                _id(101),
                _id(60_000 + index),
                index + 1,
                pot_result=PotResult.MISS,
            )
            for index in range(3)
        ),
        approval_kind=EvidenceApprovalKind.AUTOMATED_POLICY,
    )
    metric = _metric(
        _project((automated, human)),
        MetricCode.POTTING_SUCCESS_RATE,
    )
    assert metric.success_rate == 0.555556
    assert metric.effective_sample_count == 5.926829

    older = _batch(
        batch_number=702,
        session_number=101,
        shots=tuple(
            _metric_shot(
                _id(101),
                _id(61_000 + index),
                index + 1,
                pot_result=PotResult.MISS,
            )
            for index in range(3)
        ),
        approved_at=NOW - timedelta(days=90),
    )
    recency_metric = _metric(
        _project((older, human)),
        MetricCode.POTTING_SUCCESS_RATE,
    )
    assert recency_metric.success_rate == 0.666667
    assert recency_metric.effective_sample_count == 5.4


@pytest.mark.parametrize(
    ("confidence", "expected", "coverage"),
    [
        (0.5, MetricSupportStatusV2.SUPPORTED, 1.0),
        (0.499999, MetricSupportStatusV2.ABSTAINED, 0.0),
    ],
)
def test_field_confidence_half_boundary_is_inclusive(
    confidence: float,
    expected: MetricSupportStatusV2,
    coverage: float,
) -> None:
    grouped: list[list[ApprovedMetricShotEvidence]] = [[], []]
    for index in range(5):
        session_index = index % 2
        grouped[session_index].append(
            _metric_shot(
                _id(710 + session_index),
                _id(62_000 + index),
                len(grouped[session_index]) + 1,
                field_quality=_qualities(confidence),
            )
        )
    batches = tuple(
        _batch(
            batch_number=720 + index,
            session_number=710 + index,
            shots=tuple(shots),
        )
        for index, shots in enumerate(grouped)
    )
    metric = _metric(
        _project(batches),
        MetricCode.POTTING_SUCCESS_RATE,
    )
    assert metric.support_status is expected
    assert metric.coverage == coverage


def test_foul_control_uses_only_reviewed_foul_result() -> None:
    grouped: list[list[ApprovedMetricShotEvidence]] = [[], []]
    for index in range(5):
        session_index = index % 2
        grouped[session_index].append(
            _metric_shot(
                _id(730 + session_index),
                _id(63_000 + index),
                len(grouped[session_index]) + 1,
                pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
                pot_result=None,
                intended_object_ball=None,
                intended_target_pocket=None,
                foul_result=FoulResult.LEGAL if index < 3 else FoulResult.FOUL,
            )
        )
    batches = tuple(
        _batch(
            batch_number=740 + index,
            session_number=730 + index,
            shots=tuple(shots),
        )
        for index, shots in enumerate(grouped)
    )
    metric = _metric(
        _project(batches),
        MetricCode.LEGAL_SHOT_RATE,
    )
    assert metric.success_rate == 0.6


def test_projection_is_order_independent_and_proxy_fields_cannot_change_output() -> None:
    batches = _pot_batches(5, 2, successes=3)
    baseline = _project(batches)
    changed = []
    for batch in reversed(batches):
        proxy_shots = tuple(
            ApprovedShotData.model_validate(
                shot.model_dump(mode="python")
                | {
                    "travel_distance_mm": 1,
                    "cushion_count": 9,
                    "potted_balls": (),
                    "is_foul": True,
                    "deviation_mm": 0,
                }
            )
            for shot in batch.shots
        )
        payload = batch.model_dump(mode="python", exclude={"content_sha256"}) | {
            "shots": proxy_shots
        }
        changed.append(
            ApprovedEvidenceBatchV2.model_validate(
                payload | {"content_sha256": canonical_sha256(payload)}
            )
        )

    changed_result = _project(tuple(changed))
    assert changed_result == baseline
    assert changed_result.model_dump_json() == baseline.model_dump_json()


def test_canonical_batch_identity_deduplicates_replay_and_rejects_conflict() -> None:
    batches = _pot_batches(5, 2, successes=3)
    baseline = _project(batches)
    replayed = project_ability_metrics(
        _accepted((batches[0], batches[1], batches[0])), evidence_cutoff=NOW
    )

    assert replayed.metrics == baseline.metrics
    assert replayed.replayed_batch_count == 1
    assert replayed.replayed_shot_count == len(batches[0].metric_shots)

    conflict = _batch(
        batch_number=990,
        session_number=100,
        shots=batches[0].metric_shots,
    )
    assert (
        project_ability_metrics((batches[0], conflict), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )


def test_bad_digest_or_future_evidence_returns_invalid_input_without_partial_metrics() -> None:
    batch = _pot_batches(5, 2)[0]
    bad = ApprovedEvidenceBatchV2.model_construct(
        **(batch.model_dump(mode="python") | {"content_sha256": "0" * 64})
    )

    assert (
        project_ability_metrics((bad,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )
    assert (
        _project(_pot_batches(5, 2, approved_at=NOW + timedelta(microseconds=1))).status
        is MetricProjectionStatus.INVALID_INPUT
    )


def test_v2_batch_rejects_unapproved_approval_policy_versions() -> None:
    batch = _pot_batches(5, 2)[0]
    with pytest.raises(ValidationError, match="approved policy version"):
        ApprovedEvidenceBatchV2.model_validate(
            batch.model_dump(mode="python") | {"approval_policy_version": "mutable-latest"}
        )


def test_v2_batch_requires_exact_metric_shot_and_calibration_lineage() -> None:
    batch = _pot_batches(5, 2)[0]
    first = batch.metric_shots[0]
    changed_table = ReviewedTableGeometry.model_validate(
        first.table_geometry.model_dump(mode="python") | {"calibration_revision_number": 2}
    )
    changed = ApprovedMetricShotEvidence.model_validate(
        first.model_dump(mode="python")
        | {
            "table_geometry": changed_table,
            "cue_ball_start_mm": first.cue_ball_start_mm.model_copy(
                update={"calibration_revision_number": 2}
            ),
            "object_ball_start_mm": first.object_ball_start_mm.model_copy(
                update={"calibration_revision_number": 2}
            ),
        }
    )
    payload = batch.model_dump(mode="python", exclude={"content_sha256"}) | {
        "metric_shots": (changed,) + batch.metric_shots[1:]
    }

    with pytest.raises(ValidationError, match="calibration"):
        ApprovedEvidenceBatchV2.model_validate(
            payload | {"content_sha256": canonical_sha256(payload)}
        )


def test_reviewed_point_and_zone_calibration_must_match_the_shot_table() -> None:
    batch = _pot_batches(5, 2)[0]
    first = batch.metric_shots[0]
    mismatched = _point(100.0, 300.0).model_copy(update={"calibration_revision_number": 2})

    with pytest.raises(ValidationError, match="calibration"):
        ApprovedMetricShotEvidence.model_validate(
            first.model_dump(mode="python") | {"cue_ball_start_mm": mismatched}
        )


def test_projector_requires_trusted_accepted_input_and_rejects_approval_spoof() -> None:
    batch = _pot_batches(5, 2)[0]
    accepted = _accepted((batch,))[0]

    assert (
        project_ability_metrics((batch,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )
    assert _project((batch,)).status is MetricProjectionStatus.PROJECTED

    caller_declared = type(accepted).model_validate(accepted.model_dump(mode="python"))
    assert caller_declared.is_trusted_admission is False
    assert (
        project_ability_metrics((caller_declared,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )

    spoofed = accepted.model_copy(
        update={
            "approval_kind": EvidenceApprovalKind.AUTOMATED_POLICY,
            "approval_policy_version": "automated-policy-rd-v1",
        }
    )
    assert (
        project_ability_metrics((spoofed,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )


def test_trusted_admission_proof_cannot_transfer_to_a_rebound_batch() -> None:
    original_batch = _pot_batches(5, 1)[0]
    replacement_batch = _pot_batches(5, 1, successes=0)[0]
    accepted = _accepted((original_batch,))[0]

    forged = accepted.model_copy(
        update={
            "batch_id": replacement_batch.batch_id,
            "session_id": replacement_batch.session_id,
            "report_id": replacement_batch.report_id,
            "review_revision": replacement_batch.review_revision,
            "batch_content_sha256": replacement_batch.content_sha256,
            "approval_kind": replacement_batch.approval_kind,
            "approval_policy_version": replacement_batch.approval_policy_version,
            "batch": replacement_batch,
        }
    )

    assert accepted.is_trusted_admission is True
    assert forged.is_trusted_admission is False
    assert (
        project_ability_metrics((forged,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )

    constructed = type(accepted).model_construct(**forged.model_dump(mode="python"))
    constructed.__pydantic_private__ = accepted.__pydantic_private__.copy()
    assert constructed.is_trusted_admission is False
    assert (
        project_ability_metrics((constructed,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )

    deepcopied_then_forged = deepcopy(accepted).model_copy(update={"batch": replacement_batch})
    assert deepcopied_then_forged.is_trusted_admission is False
    assert (
        project_ability_metrics((deepcopied_then_forged,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )


def test_latest_review_revision_supersedes_older_evidence_deterministically() -> None:
    old = _pot_batches(5, 1, successes=5)[0]
    latest_metric_shots = tuple(
        shot.model_copy(update={"review_revision": 2, "pot_result": PotResult.MISS})
        for shot in old.metric_shots
    )
    payload = old.model_dump(mode="python", exclude={"content_sha256"}) | {
        "batch_id": _id(79_001),
        "report_revision": 2,
        "review_revision": 2,
        "approved_at": NOW + timedelta(seconds=1),
        "metric_shots": latest_metric_shots,
    }
    latest = ApprovedEvidenceBatchV2.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )
    accepted = _accepted((old, latest))

    forward = project_ability_metrics(accepted, evidence_cutoff=NOW + timedelta(seconds=1))
    reverse = project_ability_metrics(
        tuple(reversed(accepted)), evidence_cutoff=NOW + timedelta(seconds=1)
    )

    assert forward.model_dump_json() == reverse.model_dump_json()
    assert forward.included_batch_count == 1
    assert forward.superseded_batch_count == 1
    assert forward.superseded_shot_count == 5
    assert len(forward.superseded_evidence) == 1
    reference = forward.superseded_evidence[0]
    assert reference.schema_version == "1.0"
    assert reference.review_revision == 1
    assert reference.superseded_by_admission_id == accepted[1].admission_id
    assert reference.superseded_by_batch_id == latest.batch_id
    assert reference.superseded_by_review_revision == 2
    assert _metric(forward, MetricCode.POTTING_SUCCESS_RATE).sample_count == 5


def test_non_utc_cutoff_returns_typed_invalid_input() -> None:
    result = project_ability_metrics(
        _accepted((_pot_batches(5, 2)[0],)),
        evidence_cutoff=datetime(2026, 8, 5, 12, 0),
    )
    assert result.status is MetricProjectionStatus.INVALID_INPUT
    assert result.evidence_cutoff is None


def test_invalid_nominal_v1_and_mixed_bad_v2_are_invalid_not_unsupported() -> None:
    v2 = _pot_batches(5, 2)[0]
    v1_payload = v2.model_dump(
        mode="python",
        exclude={"metric_shots", "approval_kind", "approval_policy_version", "content_sha256"},
    ) | {"schema_version": "1.0"}
    v1 = ApprovedEvidenceBatch.model_validate(
        v1_payload | {"content_sha256": canonical_sha256(v1_payload)}
    )
    malformed_v1 = ApprovedEvidenceBatch.model_construct(
        **(v1.model_dump(mode="python") | {"review_revision": True})
    )
    malformed_v2 = ApprovedEvidenceBatchV2.model_construct(
        **(v2.model_dump(mode="python") | {"review_revision": True})
    )

    assert (
        project_ability_metrics((malformed_v1,), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )
    assert (
        project_ability_metrics((v1, malformed_v2), evidence_cutoff=NOW).status
        is MetricProjectionStatus.INVALID_INPUT
    )


def test_target_zone_coverage_is_family_wide_and_empty_summaries_preserve_source() -> None:
    shots = tuple(
        _metric_shot(
            _id(810 + index),
            _id(81_000 + index),
            1,
            pot_attempt_status=PotAttemptStatus.NOT_ATTEMPTED,
            pot_result=None,
            intended_object_ball=None,
            intended_target_pocket=None,
            target_zone_source=source,
            target_zone_evaluation_status=TargetZoneEvaluationStatus.UNSUPPORTED,
        )
        for index, source in enumerate(
            (TargetZoneSource.SCENE_PRESET, TargetZoneSource.POST_HOC_INFERRED)
        )
    )
    batches = tuple(
        _batch(
            batch_number=820 + index,
            session_number=810 + index,
            shots=(shot,),
        )
        for index, shot in enumerate(shots)
    )

    result = _project(batches)
    target_metrics = tuple(
        metric
        for metric in result.metrics
        if metric.metric_code
        in {MetricCode.TARGET_ZONE_SUCCESS_RATE, MetricCode.TARGET_ZONE_ERROR_MM}
    )
    assert len(target_metrics) == 4
    assert {metric.taxonomy.target_zone_source for metric in target_metrics} == {
        TargetZoneSource.SCENE_PRESET,
        TargetZoneSource.POST_HOC_INFERRED,
    }
    assert all(metric.coverage == 0.0 for metric in target_metrics)
    assert all(metric.eligible_count == metric.unsupported_count == 2 for metric in target_metrics)


def test_taxonomy_applicability_and_metric_semantics_are_closed() -> None:
    metric = _metric(_project(_pot_batches(5, 2)), MetricCode.POTTING_SUCCESS_RATE)
    assert metric.taxonomy.angle_applicability is TaxonomyApplicability.APPLICABLE
    assert metric.taxonomy.distance_applicability is TaxonomyApplicability.APPLICABLE
    assert metric.taxonomy.rail_applicability is TaxonomyApplicability.APPLICABLE
    assert metric.taxonomy.target_zone_applicability is TaxonomyApplicability.NOT_APPLICABLE

    with pytest.raises(ValidationError, match="metric taxonomy"):
        ProjectedAbilityMetric.model_validate(
            metric.model_dump(mode="python")
            | {
                "taxonomy": AbilityTaxonomyV2(
                    ability_dimension=AbilityDimension.POSITIONING,
                    angle_applicability=TaxonomyApplicability.NOT_APPLICABLE,
                    distance_applicability=TaxonomyApplicability.NOT_APPLICABLE,
                    rail_applicability=TaxonomyApplicability.NOT_APPLICABLE,
                    target_zone_applicability=TaxonomyApplicability.NOT_APPLICABLE,
                    technique_code=TechniqueCode.FOUL_AVOIDANCE,
                    technique_applicability=TaxonomyApplicability.APPLICABLE,
                )
            }
        )

    taxonomy_payload = metric.taxonomy.model_dump(mode="python") | {
        "angle_band": AngleBand.SMALL,
        "technique_code": TechniqueCode.DIRECT_POT,
    }
    with pytest.raises(ValidationError, match="metric taxonomy"):
        ProjectedAbilityMetric.model_validate(
            metric.model_dump(mode="python") | {"taxonomy": taxonomy_payload}
        )


@pytest.mark.parametrize("threshold", [5, 15, 40])
def test_kish_equal_weights_land_exactly_on_reliability_threshold(threshold: int) -> None:
    values = [
        _Observation(session_id=str(index % 5), weight=0.8, value=1.0) for index in range(threshold)
    ]
    assert _effective_n(values) == float(threshold)


def test_projected_metric_rejects_impossible_coverage_and_effective_n() -> None:
    metric = _metric(_project(_pot_batches(5, 2)), MetricCode.POTTING_SUCCESS_RATE)
    with pytest.raises(ValidationError, match="coverage"):
        ProjectedAbilityMetric.model_validate(metric.model_dump(mode="python") | {"coverage": 0.5})
    with pytest.raises(ValidationError, match="effective"):
        ProjectedAbilityMetric.model_validate(
            metric.model_dump(mode="python") | {"effective_sample_count": 5.000001}
        )


def test_geometry_helpers_reject_cross_calibration_points_and_table() -> None:
    other = _point(300.0, 400.0).model_copy(update={"calibration_revision_number": 2})
    with pytest.raises(ValueError, match="calibration"):
        shot_distance_mm(_point(0.0, 0.0), other)
    with pytest.raises(ValueError, match="calibration"):
        cut_angle_deg(_point(0.0, 0.0), other, PocketCode.BOTTOM_RIGHT)
    with pytest.raises(ValueError, match="calibration"):
        classify_rail_relation(other, _table())


def test_v2_batch_rejects_duplicate_shot_sequences_explicitly() -> None:
    batch = _pot_batches(2, 1)[0]
    duplicate_approved = batch.shots[1].model_copy(update={"sequence": 1})
    duplicate_metric = batch.metric_shots[1].model_copy(update={"sequence": 1})
    approved_by_id = {
        batch.shots[0].shot_id: batch.shots[0],
        duplicate_approved.shot_id: duplicate_approved,
    }
    metric_by_id = {
        batch.metric_shots[0].shot_id: batch.metric_shots[0],
        duplicate_metric.shot_id: duplicate_metric,
    }
    current_set_order = tuple(
        item[0]
        for item in sorted(
            {
                (batch.shots[0].shot_id, batch.session_id, 1),
                (duplicate_approved.shot_id, batch.session_id, 1),
            },
            key=lambda item: item[2],
        )
    )
    payload = batch.model_dump(mode="python", exclude={"content_sha256"}) | {
        "shots": tuple(approved_by_id[shot_id] for shot_id in current_set_order),
        "metric_shots": tuple(metric_by_id[shot_id] for shot_id in current_set_order),
    }
    with pytest.raises(ValidationError, match="sequence"):
        ApprovedEvidenceBatchV2.model_validate(
            payload | {"content_sha256": canonical_sha256(payload)}
        )
