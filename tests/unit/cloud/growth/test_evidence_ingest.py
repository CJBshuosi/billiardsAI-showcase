from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.growth import (
    ApprovedEvidenceLineage,
    ApprovedSourceEnvelopeLineage,
    EvidenceIngestError,
    EvidenceIngestErrorCode,
    EvidenceIngestService,
    EvidenceIngestStatus,
    InMemoryGrowthEvidenceRepository,
    project_ability_metrics,
    project_approved_shots,
)
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.shared.models import (
    ApprovedEvidenceBatch,
    ApprovedEvidenceBatchV2,
    ApprovedMetricShotEvidence,
    ApprovedSessionData,
    ApprovedShotData,
    Confidence,
    DimensionScore,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReviewStatus,
    ScoreStatus,
    ShotAnalysisStatus,
    TargetZoneSource,
)
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.metric_evidence import (
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
NOW = datetime(2026, 8, 4, 9, 0, tzinfo=UTC)
USER_ID = UUID("00000000-0000-4000-8000-000000000101")
OTHER_USER_ID = UUID("00000000-0000-4000-8000-000000000102")
SHOT_ID = UUID("00000000-0000-4000-8000-000000000103")
ENVELOPE_ID = UUID("00000000-0000-4000-8000-000000000104")
BATCH_ID = UUID("00000000-0000-4000-8000-000000000105")
EVENT_ID = UUID("00000000-0000-4000-8000-000000000106")
RECEIPT_ID = UUID("00000000-0000-4000-8000-000000000107")


def _uuid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


class Authority:
    def __init__(self, approved: ApprovedSessionData):
        self.approved = approved
        self.calls: list[tuple[UUID, UUID, int]] = []

    def get_approved_session_data(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedSessionData:
        self.calls.append((session_id, report_id, revision))
        return self.approved


class LineageAuthority:
    def __init__(
        self,
        approved: ApprovedSessionData,
        *,
        user_id: UUID = USER_ID,
        approved_at: datetime = NOW,
        source_envelope_ids: tuple[UUID, ...] = (ENVELOPE_ID,),
    ):
        self.lineage = ApprovedEvidenceLineage(
            user_id=user_id,
            session_id=approved.session_id,
            report_id=approved.report.report_id,
            review_revision=approved.report.revision,
            approved_at=approved_at,
            source_envelopes=tuple(
                ApprovedSourceEnvelopeLineage(envelope_id=envelope_id, shot_id=shot.shot_id)
                for envelope_id, shot in zip(source_envelope_ids, approved.shots, strict=True)
            ),
        )

    def get_approved_evidence_lineage(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedEvidenceLineage:
        return self.lineage


class StaticLineageAuthority:
    def __init__(self, lineage: ApprovedEvidenceLineage):
        self.lineage = lineage

    def get_approved_evidence_lineage(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedEvidenceLineage:
        return self.lineage


class MetricAuthority:
    def __init__(
        self,
        metric_shots: tuple[ApprovedMetricShotEvidence, ...],
        *,
        approval_kind: EvidenceApprovalKind = EvidenceApprovalKind.HUMAN_REVIEW,
        approval_policy_version: str = "human-review-v1",
        error: Exception | None = None,
    ):
        self.metric_shots = metric_shots
        self.approval_kind = approval_kind
        self.approval_policy_version = approval_policy_version
        self.error = error
        self.calls: list[tuple[UUID, UUID, int]] = []

    def get_approved_metric_evidence(self, session_id: UUID, report_id: UUID, revision: int):
        from billiards_ai.shared.models.metric_evidence import (
            ApprovedMetricEvidenceAuthorityRecord,
        )

        self.calls.append((session_id, report_id, revision))
        if self.error is not None:
            raise self.error
        return ApprovedMetricEvidenceAuthorityRecord(
            approval_kind=self.approval_kind,
            approval_policy_version=self.approval_policy_version,
            metric_shots=self.metric_shots,
        )


def _report(*, revision: int = 1, report_id: UUID | None = None) -> ReportDocument:
    payload = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))
    report = ReportDocument.model_validate_json(json.dumps(payload))
    values = report.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    if revision > 1:
        values |= {
            "report_id": report_id or UUID("00000000-0000-4000-8000-000000000120"),
            "revision": revision,
            "parent_revision_id": report.report_id,
        }
    return ReportDocument.model_validate(values)


def _shot(report: ReportDocument, **updates: object) -> ApprovedShotData:
    values: dict[str, object] = {
        "shot_id": SHOT_ID,
        "session_id": report.session_id,
        "sequence": 1,
        "struck_ball": 3,
        "potted_balls": (3,),
        "cushion_count": 0,
        "travel_distance_mm": 500,
        "is_foul": False,
        "deviation_mm": 80,
        "target_zone_source": TargetZoneSource.SCENE_PRESET,
        "confidence": Confidence.HIGH,
        "analysis_status": ShotAnalysisStatus.ANALYZED,
        "review_status": ReviewStatus.APPROVED,
    }
    return ApprovedShotData.model_validate(values | updates)


def _approved(report: ReportDocument | None = None) -> ApprovedSessionData:
    report = report or _report()
    return ApprovedSessionData(session_id=report.session_id, report=report, shots=(_shot(report),))


def _batch(
    approved: ApprovedSessionData | None = None,
    *,
    batch_id: UUID = BATCH_ID,
    user_id: UUID = USER_ID,
    approved_at: datetime = NOW,
    source_envelope_ids: tuple[UUID, ...] = (ENVELOPE_ID,),
) -> ApprovedEvidenceBatch:
    approved = approved or _approved()
    shots = tuple(
        shot
        for shot in approved.shots
        if shot.review_status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
    )
    statistics, scores = project_approved_shots(shots)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "user_id": user_id,
        "session_id": approved.session_id,
        "report_id": approved.report.report_id,
        "report_revision": approved.report.revision,
        "review_revision": approved.report.revision,
        "approved_at": approved_at,
        "shots": shots,
        "statistics": statistics,
        "scores": scores,
        "source_envelope_ids": source_envelope_ids,
    }
    return ApprovedEvidenceBatch.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )


def _metric_evidence(approved: ApprovedSessionData) -> tuple[ApprovedMetricShotEvidence, ...]:
    quality = ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=1.0)
    field_quality = MetricFieldQuality(
        geometry=quality,
        intent=quality,
        pot_attempt=quality,
        pot_result=quality,
        foul_result=quality,
        target_zone=quality,
        cue_ball_end=quality,
    )
    table = ReviewedTableGeometry(
        calibration_id=_uuid(801),
        calibration_revision_id=_uuid(802),
        calibration_revision_number=1,
        table_length_mm=2540,
        table_width_mm=1270,
        ball_diameter_mm=57.15,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
    )
    point_lineage = {
        "calibration_id": table.calibration_id,
        "calibration_revision_id": table.calibration_revision_id,
        "calibration_revision_number": table.calibration_revision_number,
    }
    return tuple(
        ApprovedMetricShotEvidence(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            sequence=shot.sequence,
            review_revision=approved.report.revision,
            game_code=GameCode.CHINESE_EIGHT_BALL,
            table_geometry=table,
            cue_ball_start_mm=ReviewedPoint(**point_lineage, x_mm=100.0, y_mm=300.0),
            object_ball_start_mm=ReviewedPoint(**point_lineage, x_mm=500.0, y_mm=300.0),
            intended_object_ball=3,
            intended_target_pocket=PocketCode.BOTTOM_RIGHT,
            pot_attempt_status=PotAttemptStatus.ATTEMPTED,
            pot_result=PotResult.LEGAL_SUCCESS,
            foul_result=FoulResult.LEGAL,
            target_zone_source=TargetZoneSource.NONE,
            target_zone=None,
            cue_ball_end_mm=None,
            target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
            target_zone_result=None,
            field_quality=field_quality,
        )
        for shot in approved.shots
    )


def _batch_v2(
    approved: ApprovedSessionData | None = None,
    *,
    source_envelope_ids: tuple[UUID, ...] = (ENVELOPE_ID,),
) -> ApprovedEvidenceBatchV2:
    approved = approved or _approved()
    v1 = _batch(approved, source_envelope_ids=source_envelope_ids)
    payload = v1.model_dump(mode="python", exclude={"schema_version", "content_sha256"}) | {
        "schema_version": "2.0",
        "approval_kind": EvidenceApprovalKind.HUMAN_REVIEW,
        "approval_policy_version": "human-review-v1",
        "metric_shots": _metric_evidence(approved),
    }
    return ApprovedEvidenceBatchV2.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )


def _replace(batch: ApprovedEvidenceBatch, **updates: object) -> ApprovedEvidenceBatch:
    payload = batch.model_dump(mode="python", exclude={"content_sha256"}) | updates
    return ApprovedEvidenceBatch.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )


def _service(
    approved: ApprovedSessionData | None = None,
    repository: InMemoryGrowthEvidenceRepository | None = None,
    *,
    clock=lambda: NOW,
    approved_at: datetime = NOW,
    ids: tuple[UUID, ...] = (RECEIPT_ID, EVENT_ID),
) -> tuple[EvidenceIngestService, InMemoryGrowthEvidenceRepository, Authority]:
    authority = Authority(approved or _approved())
    lineage_authority = LineageAuthority(authority.approved, approved_at=approved_at)
    id_values = iter(ids)
    repository = repository or InMemoryGrowthEvidenceRepository()
    return (
        EvidenceIngestService(
            approved_repository=authority,
            lineage_repository=lineage_authority,
            growth_repository=repository,
            clock=clock,
            uuid_factory=lambda: next(id_values),
        ),
        repository,
        authority,
    )


def _assert_empty(repository: InMemoryGrowthEvidenceRepository) -> None:
    assert repository.list_batches(USER_ID) == ()
    assert repository.list_receipts() == ()
    assert repository.list_source_summaries() == ()
    assert repository.list_outbox_events() == ()


def test_ingest_v2_requires_exact_metric_authority_then_stores_and_replays_v2() -> None:
    approved = _approved()
    batch = _batch_v2(approved)
    authority = Authority(approved)
    metric_authority = MetricAuthority(batch.metric_shots)
    repository = InMemoryGrowthEvidenceRepository()
    ids = iter((RECEIPT_ID, EVENT_ID))
    service = EvidenceIngestService(
        approved_repository=authority,
        lineage_repository=LineageAuthority(approved),
        metric_repository=metric_authority,
        growth_repository=repository,
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )

    created = service.ingest(batch, authenticated_user_id=USER_ID)
    replayed = service.ingest(batch, authenticated_user_id=USER_ID)

    assert created.status is EvidenceIngestStatus.CREATED
    assert replayed.status is EvidenceIngestStatus.REPLAYED
    assert created.accepted_metric_input == replayed.accepted_metric_input
    assert created.accepted_metric_input is not None
    assert replayed.accepted_metric_input is not None
    assert created.accepted_metric_input.is_trusted_admission is True
    assert replayed.accepted_metric_input.is_trusted_admission is True
    assert created.accepted_metric_input.schema_version == "1.0"
    assert created.accepted_metric_input.admission_id == created.receipt.receipt_id
    assert created.accepted_metric_input.batch_content_sha256 == batch.content_sha256
    serialized = created.accepted_metric_input.model_dump_json()
    assert not any(
        forbidden in serialized.lower()
        for forbidden in ("raw", "media", "prompt", "video", "image", "audio", "email")
    )
    with pytest.raises(ValidationError, match="frozen"):
        created.accepted_metric_input.review_revision = 99
    assert (
        project_ability_metrics((created.accepted_metric_input,), evidence_cutoff=NOW).status.value
        == "PROJECTED"
    )
    detached = deepcopy(created.accepted_metric_input)
    assert detached.is_trusted_admission is False
    assert project_ability_metrics((detached,), evidence_cutoff=NOW).status.value == "INVALID_INPUT"
    assert repository.list_batches(USER_ID) == (batch,)
    assert metric_authority.calls == [(batch.session_id, batch.report_id, batch.review_revision)]


def test_ingest_v2_rejects_caller_spoofed_approval_metadata() -> None:
    approved = _approved()
    batch = _batch_v2(approved)
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(approved),
        metric_repository=MetricAuthority(
            batch.metric_shots,
            approval_kind=EvidenceApprovalKind.AUTOMATED_POLICY,
            approval_policy_version="automated-policy-rd-v1",
        ),
        growth_repository=repository,
    )

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.LINEAGE_MISMATCH
    _assert_empty(repository)


def test_metric_authority_order_is_canonical_and_not_found_is_typed() -> None:
    report = _report()
    approved = ApprovedSessionData(
        session_id=report.session_id,
        report=report,
        shots=(
            _shot(report),
            _shot(report, shot_id=_uuid(880), sequence=2),
        ),
    )
    batch = _batch_v2(approved, source_envelope_ids=(_uuid(883), _uuid(884)))
    repository = InMemoryGrowthEvidenceRepository()
    ids = iter((_uuid(881), _uuid(882)))
    service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(
            approved, source_envelope_ids=batch.source_envelope_ids
        ),
        metric_repository=MetricAuthority(tuple(reversed(batch.metric_shots))),
        growth_repository=repository,
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    assert service.ingest(batch, authenticated_user_id=USER_ID).accepted_metric_input is not None

    for error in (
        LookupError("missing"),
        ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_APPROVED),
    ):
        empty = InMemoryGrowthEvidenceRepository()
        failing = EvidenceIngestService(
            approved_repository=Authority(approved),
            lineage_repository=LineageAuthority(
                approved, source_envelope_ids=batch.source_envelope_ids
            ),
            metric_repository=MetricAuthority(batch.metric_shots, error=error),
            growth_repository=empty,
        )
        with pytest.raises(EvidenceIngestError) as raised:
            failing.ingest(batch, authenticated_user_id=USER_ID)
        assert raised.value.code is EvidenceIngestErrorCode.NOT_APPROVED
        _assert_empty(empty)


def test_repository_allows_v1_and_authoritative_v2_for_same_revision() -> None:
    approved = _approved()
    v1 = _batch(approved)
    v2 = _batch_v2(approved)
    repository = InMemoryGrowthEvidenceRepository()
    ids = iter((_uuid(890), _uuid(891), _uuid(892), _uuid(893)))

    repository.append_approved_batch(v1, clock=lambda: NOW, uuid_factory=lambda: next(ids))
    v2_result = repository.append_approved_batch(
        v2,
        metric_authority=MetricAuthority(v2.metric_shots).get_approved_metric_evidence(
            v2.session_id, v2.report_id, v2.review_revision
        ),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )

    assert repository.list_batches(USER_ID) == (v1, v2)
    assert v2_result.accepted_metric_input is not None


def test_ingest_v2_rejects_metric_evidence_not_returned_by_approved_authority() -> None:
    approved = _approved()
    batch = _batch_v2(approved)
    authority_shot = batch.metric_shots[0]
    changed = ApprovedMetricShotEvidence.model_validate(
        authority_shot.model_dump(mode="python") | {"pot_result": PotResult.MISS}
    )
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(approved),
        metric_repository=MetricAuthority((changed,)),
        growth_repository=repository,
    )

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.LINEAGE_MISMATCH
    _assert_empty(repository)


def test_ingest_strictly_revalidates_constructed_batch_and_rejects_bool_revision() -> None:
    batch = _batch()
    malformed = ApprovedEvidenceBatch.model_construct(
        **(batch.model_dump(mode="python") | {"review_revision": True})
    )
    service, repository, authority = _service()

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(malformed, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.INVALID_BATCH
    assert authority.calls == []
    _assert_empty(repository)


def test_ingest_deeply_revalidates_constructed_nested_shot() -> None:
    batch = _batch()
    malformed_shot = ApprovedShotData.model_construct(
        **(batch.shots[0].model_dump(mode="python") | {"sequence": True})
    )
    malformed = ApprovedEvidenceBatch.model_construct(
        **(batch.model_dump(mode="python") | {"shots": (malformed_shot,)})
    )
    service, repository, authority = _service()

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(malformed, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.INVALID_BATCH
    assert authority.calls == []
    _assert_empty(repository)


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"schema_version": "2.0"}, EvidenceIngestErrorCode.INVALID_BATCH),
        ({"report_revision": 2}, EvidenceIngestErrorCode.REVISION_MISMATCH),
        ({"user_id": OTHER_USER_ID}, EvidenceIngestErrorCode.LINEAGE_MISMATCH),
    ],
)
def test_ingest_rejects_unknown_major_revision_or_user_lineage(
    updates: dict[str, object], code: EvidenceIngestErrorCode
) -> None:
    batch = _batch()
    malformed = ApprovedEvidenceBatch.model_construct(**(batch.model_dump(mode="python") | updates))
    service, repository, _authority = _service()

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(malformed, authenticated_user_id=USER_ID)

    assert raised.value.code is code
    _assert_empty(repository)


def test_ingest_rejects_nonapproved_shot_even_when_nested_model_was_constructed() -> None:
    batch = _batch()
    rejected = ApprovedShotData.model_construct(
        **(batch.shots[0].model_dump(mode="python") | {"review_status": ReviewStatus.REJECTED})
    )
    malformed = ApprovedEvidenceBatch.model_construct(
        **(batch.model_dump(mode="python") | {"shots": (rejected,)})
    )
    service, repository, _authority = _service()

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(malformed, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.INVALID_BATCH
    _assert_empty(repository)


def test_rejected_authority_is_typed_not_approved_and_writes_nothing() -> None:
    class RejectedAuthority:
        def get_approved_session_data(self, session_id: UUID, report_id: UUID, revision: int):
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVIEW_REJECTED)

    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=RejectedAuthority(),
        lineage_repository=LineageAuthority(_approved()),
        growth_repository=repository,
    )

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(_batch(), authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.NOT_APPROVED
    _assert_empty(repository)


@pytest.mark.parametrize(
    ("lineage_user_id", "lineage_envelope_ids"),
    [
        (OTHER_USER_ID, (ENVELOPE_ID,)),
        (USER_ID, (UUID("00000000-0000-4000-8000-000000000199"),)),
    ],
)
def test_trusted_owner_and_source_envelope_mapping_must_match_batch(
    lineage_user_id: UUID, lineage_envelope_ids: tuple[UUID, ...]
) -> None:
    approved = _approved()
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(
            approved,
            user_id=lineage_user_id,
            source_envelope_ids=lineage_envelope_ids,
        ),
        growth_repository=repository,
    )

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(_batch(approved), authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.LINEAGE_MISMATCH
    _assert_empty(repository)


def test_authority_and_lineage_order_are_canonicalized_and_rejected_shots_are_excluded() -> None:
    report = _report()
    first = _shot(report)
    second = _shot(
        report,
        shot_id=_uuid(210),
        sequence=2,
        review_status=ReviewStatus.CORRECTED,
    )
    rejected = _shot(
        report,
        shot_id=_uuid(211),
        sequence=3,
        review_status=ReviewStatus.REJECTED,
    )
    canonical = ApprovedSessionData(
        session_id=report.session_id,
        report=report,
        shots=(first, second),
    )
    authoritative = ApprovedSessionData(
        session_id=report.session_id,
        report=report,
        shots=(second, rejected, first),
    )
    envelope_ids = (_uuid(212), _uuid(213))
    batch = _batch(canonical, source_envelope_ids=envelope_ids)
    lineage = ApprovedEvidenceLineage(
        user_id=USER_ID,
        session_id=report.session_id,
        report_id=report.report_id,
        review_revision=report.revision,
        approved_at=NOW,
        source_envelopes=(
            ApprovedSourceEnvelopeLineage(envelope_id=envelope_ids[1], shot_id=second.shot_id),
            ApprovedSourceEnvelopeLineage(envelope_id=envelope_ids[0], shot_id=first.shot_id),
        ),
    )
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(authoritative),
        lineage_repository=StaticLineageAuthority(lineage),
        growth_repository=repository,
        clock=lambda: NOW,
        uuid_factory=iter((_uuid(214), _uuid(215))).__next__,
    )

    result = service.ingest(batch, authenticated_user_id=USER_ID)

    assert result.status is EvidenceIngestStatus.CREATED
    assert repository.list_batches(USER_ID) == (batch,)


@pytest.mark.parametrize("authority_variant", ["missing", "extra", "replacement"])
def test_batch_shot_set_must_exactly_match_authoritative_approved_subset(
    authority_variant: str,
) -> None:
    report = _report()
    first = _shot(report)
    second = _shot(report, shot_id=_uuid(220), sequence=2)
    canonical = ApprovedSessionData(
        session_id=report.session_id,
        report=report,
        shots=(first, second),
    )
    envelope_ids = (_uuid(221), _uuid(222))
    batch = _batch(canonical, source_envelope_ids=envelope_ids)
    if authority_variant == "missing":
        authority_shots = (first,)
    elif authority_variant == "extra":
        authority_shots = (first, second, _shot(report, shot_id=_uuid(223), sequence=3))
    else:
        authority_shots = (first, _shot(report, shot_id=_uuid(224), sequence=2))
    authoritative = ApprovedSessionData(
        session_id=report.session_id,
        report=report,
        shots=authority_shots,
    )
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(authoritative),
        lineage_repository=LineageAuthority(canonical, source_envelope_ids=envelope_ids),
        growth_repository=repository,
    )

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.LINEAGE_MISMATCH
    _assert_empty(repository)


def test_ingest_recomputes_digest_statistics_scores_and_source_lineage() -> None:
    batch = _batch()
    wrong_statistics = batch.statistics.model_copy(
        update={"average_cue_ball_travel_distance_mm": 501.0}
    )
    wrong_score = batch.scores.model_copy(
        update={"potting": DimensionScore(score=99, status=ScoreStatus.SCORED)}
    )
    cases = (
        (
            ApprovedEvidenceBatch.model_construct(
                **(batch.model_dump(mode="python") | {"content_sha256": "f" * 64})
            ),
            EvidenceIngestErrorCode.DIGEST_MISMATCH,
        ),
        (
            _replace(batch, statistics=wrong_statistics),
            EvidenceIngestErrorCode.STATISTICS_MISMATCH,
        ),
        (_replace(batch, scores=wrong_score), EvidenceIngestErrorCode.SCORES_MISMATCH),
        (
            _replace(batch, source_envelope_ids=(ENVELOPE_ID, EVENT_ID)),
            EvidenceIngestErrorCode.LINEAGE_MISMATCH,
        ),
    )

    for malformed, expected_code in cases:
        service, repository, _authority = _service()
        with pytest.raises(EvidenceIngestError) as raised:
            service.ingest(malformed, authenticated_user_id=USER_ID)
        assert raised.value.code is expected_code
        _assert_empty(repository)


def test_all_abstained_approved_batch_returns_typed_projection_error() -> None:
    base_approved = _approved()
    base_batch = _batch(base_approved)
    abstained = base_approved.shots[0].model_copy(update={"confidence": Confidence.ABSTAIN})
    approved = ApprovedSessionData(
        session_id=base_approved.session_id,
        report=base_approved.report,
        shots=(abstained,),
    )
    batch = _replace(base_batch, shots=(abstained,))
    service, repository, _authority = _service(approved)

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.PROJECTION_UNAVAILABLE
    _assert_empty(repository)


def test_all_non_analyzed_approved_batch_returns_typed_projection_error() -> None:
    base_approved = _approved()
    pending = base_approved.shots[0].model_copy(
        update={"analysis_status": ShotAnalysisStatus.PENDING}
    )
    approved = ApprovedSessionData(
        session_id=base_approved.session_id,
        report=base_approved.report,
        shots=(pending,),
    )
    batch = _replace(_batch(base_approved), shots=(pending,))
    service, repository, _authority = _service(approved)

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.PROJECTION_UNAVAILABLE
    _assert_empty(repository)


def test_projection_overflow_returns_typed_projection_error() -> None:
    base_approved = _approved()
    enormous = base_approved.shots[0].model_copy(update={"travel_distance_mm": 10**400})
    approved = ApprovedSessionData(
        session_id=base_approved.session_id,
        report=base_approved.report,
        shots=(enormous,),
    )
    batch = _replace(_batch(base_approved), shots=(enormous,))
    service, repository, _authority = _service(approved)

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.PROJECTION_UNAVAILABLE
    _assert_empty(repository)


def test_first_ingest_creates_atomic_safe_facts_and_replay_reuses_original_result() -> None:
    batch = _batch()
    calls = {"clock": 0, "uuid": 0}

    def clock() -> datetime:
        calls["clock"] += 1
        return NOW

    ids = iter((RECEIPT_ID, EVENT_ID))

    def uuid_factory() -> UUID:
        calls["uuid"] += 1
        return next(ids)

    authority = Authority(_approved())
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=authority,
        lineage_repository=LineageAuthority(authority.approved),
        growth_repository=repository,
        clock=clock,
        uuid_factory=uuid_factory,
    )

    created = service.ingest(batch, authenticated_user_id=USER_ID)
    replayed = service.ingest(batch, authenticated_user_id=USER_ID)

    assert created.status is EvidenceIngestStatus.CREATED
    assert replayed.status is EvidenceIngestStatus.REPLAYED
    assert replayed.receipt == created.receipt
    assert calls == {"clock": 1, "uuid": 2}
    assert repository.list_batches(USER_ID) == (batch,)
    assert len(repository.list_receipts()) == 1
    assert len(repository.list_source_summaries()) == 1
    events = repository.list_outbox_events()
    assert len(events) == 1
    assert events[0].event_type == "growth.evidence.batch.accepted.v1"
    assert set(events[0].payload) == {
        "batch_id",
        "user_id",
        "session_id",
        "report_id",
        "review_revision",
        "content_sha256",
    }
    serialized = json.dumps(
        {
            "receipts": [item.model_dump(mode="json") for item in repository.list_receipts()],
            "summaries": [
                item.model_dump(mode="json") for item in repository.list_source_summaries()
            ],
            "events": [item.model_dump(mode="json") for item in events],
        }
    ).lower()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "video",
            "media_locator",
            "prompt",
            "token",
            "openid",
            "raw_observation",
        )
    )


def test_same_identity_different_content_and_reused_batch_id_are_conflicts() -> None:
    approved = _approved()
    first = _batch(approved)
    service, repository, _authority = _service(approved)
    created = service.ingest(first, authenticated_user_id=USER_ID)
    before = repository.snapshot()

    conflict = _replace(
        first,
        batch_id=UUID("00000000-0000-4000-8000-000000000130"),
        source_envelope_ids=(UUID("00000000-0000-4000-8000-000000000131"),),
    )
    with pytest.raises(EvidenceIngestError) as identity_conflict:
        service.ingest(conflict, authenticated_user_id=USER_ID)
    assert identity_conflict.value.code is EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT

    report2 = _report(revision=2)
    approved2 = _approved(report2)
    second = _batch(approved2, batch_id=first.batch_id)
    service2, _, _ = _service(approved2, repository, ids=(_uuid(140), _uuid(141)))
    with pytest.raises(EvidenceIngestError) as batch_id_conflict:
        service2.ingest(second, authenticated_user_id=USER_ID)
    assert batch_id_conflict.value.code is EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT
    assert repository.snapshot() == before
    assert repository.list_receipts() == (created.receipt,)


@pytest.mark.parametrize("changed_field", ["digest", "report_revision", "report_lineage"])
def test_same_owner_identity_conflict_precedes_batch_integrity_errors(changed_field: str) -> None:
    approved = _approved()
    batch = _batch(approved)
    service, repository, _authority = _service(approved)
    service.ingest(batch, authenticated_user_id=USER_ID)

    if changed_field == "digest":
        changed = ApprovedEvidenceBatch.model_construct(
            **(batch.model_dump(mode="python") | {"content_sha256": "f" * 64})
        )
    elif changed_field == "report_revision":
        changed = _replace(batch, report_revision=batch.report_revision + 1)
    else:
        changed = _replace(batch, report_id=_uuid(181))

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(changed, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT


def test_cross_owner_identity_is_indistinguishable_from_unowned_authority_query() -> None:
    approved = _approved()
    populated = InMemoryGrowthEvidenceRepository()
    owner_service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(approved, user_id=OTHER_USER_ID),
        growth_repository=populated,
        clock=lambda: NOW,
        uuid_factory=iter((_uuid(182), _uuid(183))).__next__,
    )
    owner_service.ingest(
        _batch(approved, user_id=OTHER_USER_ID), authenticated_user_id=OTHER_USER_ID
    )

    observed_codes = []
    for repository in (populated, InMemoryGrowthEvidenceRepository()):
        attacker_service = EvidenceIngestService(
            approved_repository=Authority(approved),
            lineage_repository=LineageAuthority(approved, user_id=OTHER_USER_ID),
            growth_repository=repository,
        )
        with pytest.raises(EvidenceIngestError) as raised:
            attacker_service.ingest(_batch(approved), authenticated_user_id=USER_ID)
        observed_codes.append(raised.value.code)

    assert observed_codes == [
        EvidenceIngestErrorCode.LINEAGE_MISMATCH,
        EvidenceIngestErrorCode.LINEAGE_MISMATCH,
    ]


@pytest.mark.parametrize("changed_field", ["report", "shot", "statistics", "scores"])
def test_every_same_identity_canonical_content_change_is_conflict(changed_field: str) -> None:
    approved = _approved()
    batch = _batch(approved)
    service, repository, _authority = _service(approved)
    service.ingest(batch, authenticated_user_id=USER_ID)
    before = repository.snapshot()
    if changed_field == "report":
        changed = _replace(
            batch,
            report_id=UUID("00000000-0000-4000-8000-000000000180"),
        )
    elif changed_field == "shot":
        changed_shot = batch.shots[0].model_copy(update={"travel_distance_mm": 501})
        changed = _replace(batch, shots=(changed_shot,))
    elif changed_field == "statistics":
        changed_statistics = batch.statistics.model_copy(
            update={"average_cue_ball_travel_distance_mm": 501.0}
        )
        changed = _replace(batch, statistics=changed_statistics)
    else:
        changed_scores = batch.scores.model_copy(
            update={"potting": DimensionScore(score=99, status=ScoreStatus.SCORED)}
        )
        changed = _replace(batch, scores=changed_scores)

    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(changed, authenticated_user_id=USER_ID)

    assert raised.value.code is EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT
    assert repository.snapshot() == before


def test_twenty_concurrent_retries_produce_one_business_effect() -> None:
    batch = _batch()
    counter = iter(_uuid(value) for value in range(200, 260))
    authority = Authority(_approved())
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=authority,
        lineage_repository=LineageAuthority(authority.approved),
        growth_repository=repository,
        clock=lambda: NOW,
        uuid_factory=lambda: next(counter),
    )

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = tuple(
            pool.map(lambda _: service.ingest(batch, authenticated_user_id=USER_ID), range(20))
        )

    assert sum(result.status is EvidenceIngestStatus.CREATED for result in results) == 1
    assert sum(result.status is EvidenceIngestStatus.REPLAYED for result in results) == 19
    assert len({result.receipt.receipt_id for result in results}) == 1
    assert len(repository.list_batches(USER_ID)) == 1
    assert len(repository.list_outbox_events()) == 1


@pytest.mark.parametrize("failure_stage", ["batch", "inbox", "source_summary", "outbox"])
def test_atomic_failure_leaves_no_partial_state(failure_stage: str) -> None:
    repository: InMemoryGrowthEvidenceRepository

    def fail(stage: str) -> None:
        if stage == failure_stage:
            observed_counts = (
                len(repository.list_batches(USER_ID)),
                len(repository.list_receipts()),
                len(repository.list_source_summaries()),
                len(repository.list_outbox_events()),
            )
            expected_counts = {
                "batch": (1, 0, 0, 0),
                "inbox": (1, 1, 0, 0),
                "source_summary": (1, 1, 1, 0),
                "outbox": (1, 1, 1, 1),
            }[stage]
            assert observed_counts == expected_counts
            raise RuntimeError(f"synthetic {stage} failure")

    repository = InMemoryGrowthEvidenceRepository(failure_injector=fail)
    service, _, _authority = _service(repository=repository)

    with pytest.raises(RuntimeError, match="synthetic"):
        service.ingest(_batch(), authenticated_user_id=USER_ID)

    _assert_empty(repository)


def test_atomic_failure_restores_preexisting_authority_state() -> None:
    armed = False

    def fail(stage: str) -> None:
        if armed and stage == "outbox":
            raise RuntimeError("synthetic outbox failure")

    repository = InMemoryGrowthEvidenceRepository(failure_injector=fail)
    first_approved = _approved()
    first_service, _, _ = _service(
        first_approved,
        repository,
        ids=(_uuid(230), _uuid(231)),
    )
    first_service.ingest(_batch(first_approved), authenticated_user_id=USER_ID)
    before = repository.snapshot()

    second_approved = _approved(_report(revision=2))
    second_service, _, _ = _service(
        second_approved,
        repository,
        ids=(_uuid(232), _uuid(233)),
    )
    second_batch = _batch(second_approved, batch_id=_uuid(234))
    armed = True

    with pytest.raises(RuntimeError, match="synthetic outbox failure"):
        second_service.ingest(second_batch, authenticated_user_id=USER_ID)

    assert repository.snapshot() == before


def test_list_outbox_events_returns_deeply_detached_payloads() -> None:
    service, repository, _authority = _service()
    service.ingest(_batch(), authenticated_user_id=USER_ID)

    returned = repository.list_outbox_events()[0]
    returned.payload["batch_id"] = "tampered"

    assert repository.list_outbox_events()[0].payload["batch_id"] == str(BATCH_ID)


def test_snapshot_returns_deeply_detached_event_payloads() -> None:
    service, repository, _authority = _service()
    service.ingest(_batch(), authenticated_user_id=USER_ID)

    returned = repository.snapshot()[-1][0][1]
    assert isinstance(returned, DomainEvent)
    returned.payload["batch_id"] = "tampered"

    stored = repository.snapshot()[-1][0][1]
    assert isinstance(stored, DomainEvent)
    assert stored.payload["batch_id"] == str(BATCH_ID)


def test_approved_at_must_match_trusted_lineage_timestamp() -> None:
    approved = _approved()
    trusted_lineage = LineageAuthority(approved, approved_at=NOW).lineage
    repository = InMemoryGrowthEvidenceRepository()
    service = EvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=StaticLineageAuthority(trusted_lineage),
        growth_repository=repository,
        clock=lambda: NOW,
        uuid_factory=iter((_uuid(235), _uuid(236))).__next__,
    )

    created = service.ingest(_batch(approved), authenticated_user_id=USER_ID)
    assert created.status is EvidenceIngestStatus.CREATED

    different_report = _report(revision=2)
    different_approved = _approved(different_report)
    different_lineage = LineageAuthority(different_approved, approved_at=NOW).lineage
    mismatched_service = EvidenceIngestService(
        approved_repository=Authority(different_approved),
        lineage_repository=StaticLineageAuthority(different_lineage),
        growth_repository=repository,
    )
    with pytest.raises(EvidenceIngestError) as raised:
        mismatched_service.ingest(
            _batch(
                different_approved,
                batch_id=_uuid(237),
                approved_at=NOW + timedelta(seconds=1),
            ),
            authenticated_user_id=USER_ID,
        )
    assert raised.value.code is EvidenceIngestErrorCode.LINEAGE_MISMATCH


def test_list_batches_is_user_isolated_frozen_and_stably_sorted() -> None:
    approved = _approved()
    repository = InMemoryGrowthEvidenceRepository()
    first = _batch(
        approved,
        batch_id=UUID("00000000-0000-4000-8000-000000000151"),
        approved_at=NOW + timedelta(minutes=1),
    )
    second_report = _report(revision=2)
    second_approved = _approved(second_report)
    second = _batch(
        second_approved,
        batch_id=UUID("00000000-0000-4000-8000-000000000152"),
        approved_at=NOW,
    )
    service1, _, _ = _service(
        approved,
        repository,
        approved_at=NOW + timedelta(minutes=1),
        ids=(_uuid(301), _uuid(302)),
    )
    service2, _, _ = _service(second_approved, repository, ids=(_uuid(303), _uuid(304)))
    service1.ingest(first, authenticated_user_id=USER_ID)
    service2.ingest(second, authenticated_user_id=USER_ID)

    assert repository.list_batches(USER_ID) == (second, first)
    assert repository.list_batches(OTHER_USER_ID) == ()
    with pytest.raises(ValidationError):
        repository.list_batches(USER_ID)[0].__setattr__("review_revision", 99)
