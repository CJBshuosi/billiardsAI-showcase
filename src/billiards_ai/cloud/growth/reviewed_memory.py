"""Internal Stage 3 consumer of existing, trusted human-reviewed V2 evidence.

No field-ID conversion or new public event. Field replay must first pass the
separate field admission boundary; ownership alone is never review authority.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.cloud.growth.metric_projector import (
    MetricProjectionWithTrace,
    project_ability_metrics_with_trace,
)
from billiards_ai.cloud.growth.repository import EvidenceIngestResult, stored_model_values
from billiards_ai.cloud.growth.service import ApprovedEvidenceAuthority, EvidenceIngestService
from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.persistence.approved_publication import (
    PostgresApprovedEvidencePublisher,
)
from billiards_ai.edge.route_assistance.memory_gate import (
    FieldMemoryAdmissionError,
    require_personal_memory_admission,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.shared.models.events import DomainEvent
from billiards_ai.shared.models.evidence import AcceptedMetricEvidenceInput, ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import (
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    PotAttemptStatus,
    TargetZoneEvaluationStatus,
)


def _require_reviewed_quality(batch: ApprovedEvidenceBatchV2) -> None:
    if (
        batch.approval_kind is not EvidenceApprovalKind.HUMAN_REVIEW
        or batch.approval_policy_version != "human-review-v1"
    ):
        raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
    for shot in batch.metric_shots:
        for name in type(shot.field_quality).model_fields:
            quality = getattr(shot.field_quality, name)
            # Existing HumanReviewMetricAuthorityAssembler emits exactly these
            # two forms. CV confidence, even 1.0, is not a review decision.
            if (quality.source, quality.confidence) not in {
                (FieldSource.MANUAL, 1.0),
                (FieldSource.UNKNOWN, 0.0),
            }:
                raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
        q = shot.field_quality
        if (
            shot.pot_attempt_status is PotAttemptStatus.ATTEMPTED
            and all(
                getattr(q, name).confidence == 1.0
                for name in ("geometry", "intent", "pot_attempt", "pot_result")
            )
            and q.foul_result.confidence != 1.0
        ):
            # The existing metric projector counts pot results independently of
            # foul confidence. Do not feed it a purported reviewed legal result
            # whose legal prerequisite was never confirmed.
            raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)

        if (
            shot.target_zone_evaluation_status is TargetZoneEvaluationStatus.ELIGIBLE
            and q.target_zone.confidence == q.cue_ball_end.confidence == 1.0
            and (q.geometry.confidence != 1.0 or q.intent.confidence != 1.0)
        ):
            raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)


async def consume_reviewed_event(
    event: DomainEvent,
    *,
    authenticated_user_id: UUID,
    approved_repository: ApprovedEvidenceAuthority,
    publication_repository: PostgresApprovedEvidencePublisher,
    ingest_service: EvidenceIngestService,
) -> EvidenceIngestResult:
    """Validate the original publication then use existing atomic inbox/outbox.

    Caller resolves identity independently of the payload. No acknowledgement is
    performed here: only a successfully committed result may be acknowledged.
    """
    try:
        event = DomainEvent.model_validate(stored_model_values(event), strict=True)
        if set(event.payload) != {"approvedEvidenceBatch"}:
            raise ValueError("unexpected publication payload")
        batch = ApprovedEvidenceBatchV2.model_validate_json(
            json.dumps(event.payload["approvedEvidenceBatch"]),
            strict=True,
        )
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH) from exc
    if (
        event.event_type != "evidence.session.approved.v1"
        or event.aggregate_type != "review_session"
        or event.aggregate_id != batch.session_id
        or event.correlation_id != batch.session_id
        or event.aggregate_version != batch.review_revision
        or event.occurred_at != batch.approved_at
        or event.causation_id is None
        or event.idempotency_key != f"evidence-approved:{batch.session_id}:{batch.review_revision}"
        or authenticated_user_id != batch.user_id
    ):
        raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
    _require_reviewed_quality(batch)
    # Resolve the original persisted publication using the existing reader,
    # which rebuilds it from review-task authority. Self-consistent caller IDs
    # and hashes must not reserve another publication's inbox/revision identity.
    publication = await publication_repository.load_publication(batch.batch_id)
    if publication.batch != batch or publication.event != event:
        raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
    # Recheck the approved head even on an ingest replay: a cached receipt must
    # never make a superseded/rejected report newly visible.
    approved = approved_repository.get_approved_session_data(
        batch.session_id,
        batch.report_id,
        batch.review_revision,
    )
    if (
        approved.session_id != batch.session_id
        or approved.report.report_id != batch.report_id
        or approved.report.revision != batch.review_revision
    ):
        raise EvidenceIngestError(EvidenceIngestErrorCode.REVISION_MISMATCH)
    return ingest_service.ingest(batch, authenticated_user_id=authenticated_user_id)


def _confirmed_fields(admission: AcceptedMetricEvidenceInput) -> dict[UUID, frozenset[str]]:
    """Pure internal eligibility, not an authorized or current-head public read."""
    if not isinstance(admission, AcceptedMetricEvidenceInput) or not admission.is_trusted_admission:
        raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
    _require_reviewed_quality(admission.batch)
    result = {}
    for shot in admission.batch.metric_shots:
        confirmed = {
            name
            for name in type(shot.field_quality).model_fields
            if getattr(shot.field_quality, name).source is FieldSource.MANUAL
        }
        eligible = set(confirmed)
        if not {"geometry", "intent", "pot_attempt"} <= confirmed:
            eligible.discard("pot_attempt")
        if (
            not {"geometry", "intent", "pot_attempt", "pot_result", "foul_result"} <= confirmed
            or shot.pot_attempt_status is not PotAttemptStatus.ATTEMPTED
        ):
            eligible.discard("pot_result")
        if shot.pot_attempt_status in {PotAttemptStatus.ABSTAINED, PotAttemptStatus.UNSUPPORTED}:
            eligible.discard("pot_attempt")
        if shot.foul_result not in {FoulResult.LEGAL, FoulResult.FOUL}:
            eligible.discard("foul_result")
        if (
            not {"geometry", "intent", "target_zone", "cue_ball_end"} <= confirmed
            or shot.target_zone_evaluation_status is not TargetZoneEvaluationStatus.ELIGIBLE
        ):
            eligible.discard("target_zone")
        if shot.cue_ball_end_mm is None or "geometry" not in confirmed:
            eligible.discard("cue_ball_end")
        result[shot.shot_id] = frozenset(eligible)
    return result


def consume_field_reviewed_event(
    event: DomainEvent,
    *,
    field_store: SQLiteFieldRecordStore,
    field_session_id: str,
    authenticated_user_id: UUID,
    approved_repository: ApprovedEvidenceAuthority,
    ingest_service: EvidenceIngestService,
) -> EvidenceIngestResult:
    """Field entry point: enforce source exclusion before any growth write.

    Kept separate from the existing UUID-only reviewed entry point. Even a valid
    reviewed event cannot substitute for the missing field-to-review identity
    join. Unused downstream arguments intentionally cannot bypass this gate.
    """
    require_personal_memory_admission(field_store, field_session_id)
    raise FieldMemoryAdmissionError("IDENTITY_MAPPING_REQUIRED")


def project_current_reviewed_metrics(
    admissions: tuple[AcceptedMetricEvidenceInput, ...],
    *,
    authenticated_user_id: UUID,
    approved_repository: ApprovedEvidenceAuthority,
    evidence_cutoff: datetime,
) -> MetricProjectionWithTrace:
    """Reuse latest-revision metric projection with a fresh approved-head fence.

    Revoked/pending heads exclude the entire session, never fall back to an older
    admission. Existing immutable evidence is retained, not deleted or summed.
    This is an internal read, not a Mini Program profile cache or new snapshot.
    """
    heads: dict[UUID, AcceptedMetricEvidenceInput] = {}
    for item in admissions:
        _confirmed_fields(item)
        if item.batch.user_id != authenticated_user_id:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        previous = heads.get(item.session_id)
        if previous is None or item.review_revision > previous.review_revision:
            heads[item.session_id] = item
    visible = set()
    for session_id, head in heads.items():
        try:
            current = approved_repository.get_approved_session_data(
                session_id,
                head.report_id,
                head.review_revision,
            )
        except ReportNotDeliverableError:
            continue
        if (
            current.session_id == session_id
            and current.report.report_id == head.report_id
            and current.report.revision == head.review_revision
        ):
            visible.add(session_id)
    return project_ability_metrics_with_trace(
        tuple(item for item in admissions if item.session_id in visible),
        evidence_cutoff=evidence_cutoff,
    )
