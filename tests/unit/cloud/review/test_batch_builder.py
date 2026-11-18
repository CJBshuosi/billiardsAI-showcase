"""Pure RED/Green coverage for the Story 11.2 approved evidence builder."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from billiards_ai.cloud.growth import (
    ApprovedEvidenceLineage,
    ApprovedSourceEnvelopeLineage,
    EvidenceIngestService,
    EvidenceIngestStatus,
    InMemoryGrowthEvidenceRepository,
)
from billiards_ai.cloud.growth.projector import project_approved_shots
from billiards_ai.cloud.review.batch_builder import (
    ApprovedEvidenceBatchBuilderError,
    ApprovedEvidenceBatchV2Builder,
    BatchBuilderErrorCode,
)
from billiards_ai.cloud.review.persistence.publication import (
    ReviewPublicationAuthority,
    ReviewSessionOwnership,
)
from billiards_ai.cloud.review.repository import InMemoryApprovedDataRepository
from billiards_ai.shared.models import (
    Confidence,
    ReportDocument,
    ReportStatus,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    Shot,
    ShotAnalysisStatus,
    TargetZoneSource,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
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
    TargetZoneEvaluationStatus,
)

REPORT_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
APPROVED_AT = datetime(2026, 8, 10, 17, tzinfo=UTC)


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _report() -> ReportDocument:
    draft = ReportDocument.model_validate_json(REPORT_FIXTURE.read_text(encoding="utf-8"))
    return ReportDocument.model_validate(
        draft.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    )


def _shot(
    report: ReportDocument,
    sequence: int,
    status: ReviewStatus,
    *,
    confidence: Confidence = Confidence.HIGH,
    analysis_status: ShotAnalysisStatus = ShotAnalysisStatus.ANALYZED,
) -> Shot:
    return Shot(
        id=_id(100 + sequence),
        session_id=report.session_id,
        sequence=sequence,
        t_start_ms=(sequence - 1) * 1000,
        t_end_ms=sequence * 1000,
        ball_positions_before=None,
        ball_positions_after=None,
        cue_ball_end=None,
        struck_ball=sequence,
        potted_balls=[sequence] if sequence in {1, 3} else [],
        cushion_count=sequence - 1,
        travel_distance_mm=sequence * 100,
        is_foul=False,
        deviation_mm=sequence * 10,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        confidence=confidence,
        analysis_status=analysis_status,
        review_status=status,
        raw_observation={"provider_prompt": "must never enter the batch"},
        created_at=datetime(2026, 8, 10, 16, tzinfo=UTC),
    )


def _quality() -> ReviewedFieldQuality:
    return ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=1.0)


def _metric_quality() -> MetricFieldQuality:
    quality = _quality()
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
        "calibration_id": _id(800),
        "calibration_revision_id": _id(801),
        "calibration_revision_number": 1,
    }


def _metric(shot: Shot, revision: int) -> ApprovedMetricShotEvidence:
    return ApprovedMetricShotEvidence(
        shot_id=shot.id,
        session_id=shot.session_id,
        sequence=shot.sequence,
        review_revision=revision,
        game_code=GameCode.CHINESE_EIGHT_BALL,
        table_geometry=ReviewedTableGeometry(
            **_calibration(),
            table_length_mm=2540,
            table_width_mm=1270,
            ball_diameter_mm=57.15,
            corner_pocket_mouth_mm=86,
            side_pocket_mouth_mm=92,
        ),
        cue_ball_start_mm=ReviewedPoint(**_calibration(), x_mm=100.0, y_mm=300.0),
        object_ball_start_mm=ReviewedPoint(**_calibration(), x_mm=500.0, y_mm=300.0),
        intended_object_ball=shot.sequence,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.LEGAL_SUCCESS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        target_zone_result=None,
        field_quality=_metric_quality(),
    )


def _facts(
    *,
    statuses: tuple[ReviewStatus, ...] = (
        ReviewStatus.APPROVED,
        ReviewStatus.REJECTED,
        ReviewStatus.APPROVED,
        ReviewStatus.REJECTED,
    ),
) -> tuple[
    ReportDocument,
    tuple[Shot, ...],
    dict[UUID, UUID],
    ApprovedMetricEvidenceAuthorityRecord,
]:
    report = _report()
    shots = tuple(_shot(report, index, status) for index, status in enumerate(statuses, 1))
    retained = tuple(shot for shot in shots if shot.review_status is not ReviewStatus.REJECTED)
    envelopes = {shot.id: _id(200 + shot.sequence) for shot in shots}
    metric_authority = ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=tuple(_metric(shot, report.revision) for shot in retained),
    )
    return report, shots, envelopes, metric_authority


def _ownership(report: ReportDocument) -> ReviewSessionOwnership:
    return ReviewSessionOwnership(
        task_id=_id(901),
        user_id=_id(900),
        session_id=report.session_id,
        bound_at=datetime(2026, 8, 10, 16, 30, tzinfo=UTC),
    )


def _authority(
    report: ReportDocument,
    metric_authority: ApprovedMetricEvidenceAuthorityRecord,
    **updates: object,
) -> ReviewPublicationAuthority:
    values: dict[str, object] = {
        "task_id": _id(901),
        "session_id": report.session_id,
        "result_report_id": report.report_id,
        "review_revision": report.revision,
        "task_status": ReviewTaskStatus.APPROVED,
        "batch_id": _id(902),
        "event_id": _id(903),
        "approved_at": APPROVED_AT,
        "metric_authority": metric_authority,
    }
    return ReviewPublicationAuthority(**(values | updates))


def _build(**updates: object):
    report, shots, envelopes, metric_authority = _facts()
    values: dict[str, object] = {
        "result_report": report,
        "result_shots": shots,
        "source_envelope_ids": envelopes,
        "ownership": _ownership(report),
        "authority": _authority(report, metric_authority),
    }
    return ApprovedEvidenceBatchV2Builder.build(**(values | updates))


def test_builder_projects_only_retained_shots_without_renumbering_and_matches_growth_semantics():
    publication = _build()
    batch = publication.batch

    assert batch.schema_version == "2.0"
    assert tuple(shot.sequence for shot in batch.shots) == (1, 3)
    assert tuple(metric.sequence for metric in batch.metric_shots) == (1, 3)
    assert tuple(str(item) for item in batch.source_envelope_ids) == (
        str(_id(201)),
        str(_id(203)),
    )
    assert batch.statistics.total_shot_count == 2
    assert batch.statistics.potted_ball_count == 2
    assert batch.statistics.potting_shot_count == 0
    assert batch.statistics.pot_rate is None
    assert batch.statistics.cue_ball_end_distribution == ()
    assert (batch.statistics, batch.scores) == project_approved_shots(batch.shots)
    assert batch.content_sha256 == canonical_sha256(
        batch.model_dump(mode="json", exclude={"content_sha256"})
    )
    assert "raw_observation" not in str(batch.model_dump(mode="json"))


def test_builder_rejects_a_shifted_full_result_sequence_even_when_metric_matches():
    report, shots, _envelopes, _metric_authority = _facts()
    shifted = tuple(shot.model_copy(update={"sequence": shot.sequence + 1}) for shot in shots)
    retained = tuple(shot for shot in shifted if shot.review_status is not ReviewStatus.REJECTED)
    metric_authority = ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=tuple(_metric(shot, report.revision) for shot in retained),
    )

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(
            result_shots=shifted,
            source_envelope_ids={shot.id: _id(400 + shot.sequence) for shot in shifted},
            authority=_authority(report, metric_authority),
        )

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY


def test_builder_output_is_accepted_by_the_existing_growth_ingest_contract():
    report, shots, envelopes, metric_authority = _facts()
    ownership = _ownership(report)
    authority = _authority(report, metric_authority)
    publication = ApprovedEvidenceBatchV2Builder.build(
        result_report=report,
        result_shots=shots,
        source_envelope_ids=envelopes,
        ownership=ownership,
        authority=authority,
    )
    approved_repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(
            ReviewTask(
                id=authority.task_id,
                session_id=report.session_id,
                report_id=report.report_id,
                report_revision=report.revision,
                status=ReviewTaskStatus.APPROVED,
            ),
        ),
        shots=shots,
    )
    approved = approved_repository.get_approved_session_data(
        report.session_id, report.report_id, report.revision
    )
    lineage = ApprovedEvidenceLineage(
        user_id=ownership.user_id,
        session_id=report.session_id,
        report_id=report.report_id,
        review_revision=report.revision,
        approved_at=authority.approved_at,
        source_envelopes=tuple(
            ApprovedSourceEnvelopeLineage(envelope_id=envelope, shot_id=shot.shot_id)
            for envelope, shot in zip(
                publication.batch.source_envelope_ids,
                publication.batch.shots,
                strict=True,
            )
        ),
    )

    class LineageAuthority:
        def get_approved_evidence_lineage(self, session_id, report_id, revision):
            return lineage

    class MetricAuthority:
        def get_approved_metric_evidence(self, session_id, report_id, revision):
            return metric_authority

    result = EvidenceIngestService(
        approved_repository=approved_repository,
        lineage_repository=LineageAuthority(),
        metric_repository=MetricAuthority(),
        growth_repository=InMemoryGrowthEvidenceRepository(),
        clock=lambda: APPROVED_AT,
        uuid_factory=lambda: _id(999),
    ).ingest(publication.batch, authenticated_user_id=ownership.user_id)

    assert result.status is EvidenceIngestStatus.CREATED
    assert approved.shots[0].session_id == publication.batch.session_id


def test_builder_emits_the_fixed_full_batch_event_and_is_deterministic():
    first = _build()
    second = _build()

    assert first == second
    assert first.event.event_type == "evidence.session.approved.v1"
    assert first.event.aggregate_type == "review_session"
    assert first.event.aggregate_id == first.batch.session_id
    assert first.event.aggregate_version == first.batch.review_revision
    assert first.event.occurred_at == APPROVED_AT
    assert first.event.causation_id == _id(901)
    assert first.event.correlation_id == first.batch.session_id
    assert first.event.idempotency_key == (
        f"evidence-approved:{first.batch.session_id}:{first.batch.review_revision}"
    )
    assert first.event.payload == {"approvedEvidenceBatch": first.batch.model_dump(mode="json")}


def test_builder_accepts_a_corrected_terminal_result_with_rejected_shots():
    report, shots, envelopes, _ignored_metric_authority = _facts(
        statuses=(
            ReviewStatus.CORRECTED,
            ReviewStatus.REJECTED,
            ReviewStatus.APPROVED,
            ReviewStatus.REJECTED,
        )
    )
    corrected_report = ReportDocument.model_validate(
        report.model_dump(mode="python") | {"report_status": ReportStatus.CORRECTED}
    )
    metric_authority = ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=tuple(
            _metric(shot, corrected_report.revision)
            for shot in shots
            if shot.review_status is not ReviewStatus.REJECTED
        ),
    )

    publication = _build(
        result_report=corrected_report,
        result_shots=shots,
        source_envelope_ids=envelopes,
        authority=_authority(
            corrected_report,
            metric_authority,
            task_status=ReviewTaskStatus.CORRECTED,
        ),
    )

    assert tuple(shot.review_status for shot in publication.batch.shots) == (
        ReviewStatus.CORRECTED,
        ReviewStatus.APPROVED,
    )


@pytest.mark.parametrize(
    ("shots", "code"),
    [
        (
            lambda report: (
                _shot(report, 1, ReviewStatus.REJECTED),
                _shot(report, 2, ReviewStatus.REJECTED),
                _shot(report, 3, ReviewStatus.REJECTED),
                _shot(report, 4, ReviewStatus.REJECTED),
            ),
            BatchBuilderErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE,
        ),
        (
            lambda report: (
                _shot(
                    report,
                    1,
                    ReviewStatus.APPROVED,
                    confidence=Confidence.ABSTAIN,
                    analysis_status=ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE,
                ),
                _shot(report, 2, ReviewStatus.REJECTED),
                _shot(report, 3, ReviewStatus.REJECTED),
                _shot(report, 4, ReviewStatus.REJECTED),
            ),
            BatchBuilderErrorCode.NO_PROJECTABLE_APPROVED_EVIDENCE,
        ),
    ],
)
def test_builder_returns_a_typed_failure_when_no_approved_fact_can_be_projected(shots, code):
    report = _report()
    result_shots = shots(report)
    envelopes = {shot.id: _id(300 + shot.sequence) for shot in result_shots}
    retained = tuple(
        shot for shot in result_shots if shot.review_status is not ReviewStatus.REJECTED
    )
    metric_shots = tuple(_metric(shot, report.revision) for shot in retained)
    authority = ApprovedMetricEvidenceAuthorityRecord.model_construct(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=metric_shots,
    )

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(
            result_shots=result_shots,
            source_envelope_ids=envelopes,
            authority=_authority(report, authority),
        )

    assert caught.value.code is code


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_builder_rejects_any_non_exact_full_source_envelope_authority(mutation: str):
    report, shots, envelopes, authority = _facts()
    if mutation == "missing":
        envelopes.pop(shots[1].id)
    elif mutation == "extra":
        envelopes[_id(999)] = _id(998)
    else:
        envelopes[shots[2].id] = envelopes[shots[0].id]

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(
            source_envelope_ids=envelopes,
            authority=_authority(report, authority),
        )

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "reordered", "wrong_revision", "wrong_session", "mixed_calibration"),
)
def test_builder_rejects_any_non_exact_retained_metric_authority(mutation: str):
    report, shots, _envelopes, authority = _facts()
    metrics = list(authority.metric_shots)
    if mutation == "missing":
        metrics.pop()
    elif mutation == "extra":
        metrics.append(_metric(shots[1], report.revision))
    elif mutation == "reordered":
        metrics.reverse()
    elif mutation == "wrong_revision":
        metrics[0] = metrics[0].model_copy(update={"review_revision": report.revision + 1})
    elif mutation == "wrong_session":
        metrics[0] = metrics[0].model_copy(update={"session_id": _id(997)})
    else:
        other_table = metrics[0].table_geometry.model_copy(update={"calibration_id": _id(996)})
        metrics[0] = metrics[0].model_copy(update={"table_geometry": other_table})
    changed = ApprovedMetricEvidenceAuthorityRecord.model_construct(
        approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version="human-review-v1",
        metric_shots=tuple(metrics),
    )

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(authority=_authority(report, changed))

    assert caught.value.code is BatchBuilderErrorCode.INVALID_METRIC_AUTHORITY


def test_builder_rejects_automated_policy_even_when_the_shared_contract_accepts_it():
    _report_value, _shots, _envelopes, authority = _facts()
    automated = ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=EvidenceApprovalKind.AUTOMATED_POLICY,
        approval_policy_version="automated-policy-rd-v1",
        metric_shots=authority.metric_shots,
    )

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        authority_values = _authority(_report_value, authority).model_dump(mode="python")
        _build(
            authority=ReviewPublicationAuthority.model_construct(
                **(authority_values | {"metric_authority": automated})
            )
        )

    assert caught.value.code is BatchBuilderErrorCode.INVALID_METRIC_AUTHORITY


@pytest.mark.parametrize(
    "authority_updates",
    (
        {"task_status": ReviewTaskStatus.CORRECTED},
        {"review_revision": 2},
        {"approved_at": datetime(2026, 8, 10, 17)},
    ),
)
def test_builder_rejects_task_report_revision_or_time_authority_drift(authority_updates):
    report, _shots, _envelopes, metric_authority = _facts()
    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(
            authority=ReviewPublicationAuthority.model_construct(
                **(
                    _authority(report, metric_authority).model_dump(mode="python")
                    | authority_updates
                )
            )
        )

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY


@pytest.mark.parametrize(
    ("ownership_updates", "authority_updates"),
    (
        ({"task_id": _id(990)}, {}),
        ({"session_id": _id(991)}, {}),
        ({"bound_at": datetime(2026, 8, 10, 18, tzinfo=UTC)}, {}),
        ({}, {"task_id": _id(993)}),
        ({}, {"session_id": _id(994)}),
        ({}, {"result_report_id": _id(995)}),
    ),
)
def test_builder_rejects_any_ownership_or_publication_identity_drift(
    ownership_updates,
    authority_updates,
):
    report, _shots, _envelopes, metric_authority = _facts()
    ownership = _ownership(report).model_copy(update=ownership_updates)
    authority = _authority(report, metric_authority).model_copy(update=authority_updates)

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(ownership=ownership, authority=authority)

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_status", "approved"),
        ("task_status", ["approved"]),
        ("approved_at", "2026-08-10T17:00:00Z"),
        ("approved_at", 1786381200),
    ),
)
def test_builder_converts_malformed_status_or_time_authority_to_a_typed_error(field, value):
    report, _shots, _envelopes, metric_authority = _facts()
    authority_values = _authority(report, metric_authority).model_dump(mode="python")
    authority_values[field] = value

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(authority=ReviewPublicationAuthority.model_construct(**authority_values))

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY


@pytest.mark.parametrize("mutation", ("reordered", "pending_status", "short_set"))
def test_builder_rejects_a_noncanonical_or_incomplete_result_set(mutation: str):
    report, shots, _envelopes, _authority = _facts()
    changed = list(shots)
    if mutation == "reordered":
        changed.reverse()
    elif mutation == "pending_status":
        changed[0] = changed[0].model_copy(update={"review_status": ReviewStatus.PENDING})
    else:
        changed.pop()

    with pytest.raises(ApprovedEvidenceBatchBuilderError) as caught:
        _build(result_shots=tuple(changed))

    assert caught.value.code is BatchBuilderErrorCode.INVALID_PUBLICATION_AUTHORITY
