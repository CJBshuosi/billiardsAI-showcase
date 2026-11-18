"""Atomic PostgreSQL publication of approved evidence batch v2."""

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.cloud.review.batch_builder import ApprovedEvidenceBatchV2Builder
from billiards_ai.cloud.review.service import ReviewSubmission
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
    ApprovedMetricShotEvidence,
)
from billiards_ai.shared.models.report import ReportDocument
from billiards_ai.shared.models.review import (
    ReviewCorrection,
    ReviewTaskStatus,
    ShotReviewDecision,
)
from billiards_ai.shared.models.shot import Shot

from .errors import (
    IdempotencyConflictError,
    PersistenceIntegrityError,
    PublicationAuthorityError,
    RevisionConflictError,
)
from .outbox import PostgresOutboxRepository
from .privacy import assert_safe_persistence_payload
from .publication import (
    ReviewPublicationAuthority,
    ReviewPublicationRecord,
    StoredApprovedEvidencePublication,
)
from .repository import (
    ZERO_UUID,
    IdempotencyIdentity,
    PostgresQualityGateRepository,
    StoredReviewTask,
    _validate_json_model,
    _verify_digest,
)


def canonical_finalize_request_digest(
    *,
    source_task: StoredReviewTask,
    submission: ReviewSubmission,
    metric_authority: ApprovedMetricEvidenceAuthorityRecord,
) -> str:
    """Hash only authenticated request facts, excluding server publication identities/time."""

    report_facts = submission.report.model_dump(
        mode="json",
        exclude={
            "report_id",
            "revision",
            "parent_revision_id",
            "generated_at",
            "report_status",
        },
    )
    payload = {
        "actor_principal_id": source_task.claimed_principal_id,
        "actor_credential_id": source_task.claimed_credential_id,
        "task_id": str(source_task.task.id),
        "expected_lock_version": source_task.lock_version,
        "session_id": str(source_task.task.session_id),
        "source_report_id": str(source_task.task.report_id),
        "source_report_revision": source_task.task.report_revision,
        "terminal_status": submission.review_task.status.value,
        "decisions": [item.model_dump(mode="json") for item in submission.decisions],
        "corrections": [item.model_dump(mode="json") for item in submission.corrections],
        "report_facts": report_facts,
        "metric_authority": metric_authority.model_dump(mode="json"),
    }
    return canonical_sha256(payload)


class PostgresApprovedEvidencePublisher:
    """Compose the Story 11.1 staging seam with Batch v2 and terminal CAS."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory
        self._quality = PostgresQualityGateRepository(session_factory)
        self._outbox = PostgresOutboxRepository(session_factory)
        self._builder = ApprovedEvidenceBatchV2Builder()

    async def probe_finalize_receipt(
        self,
        *,
        identity: IdempotencyIdentity,
        source_task: StoredReviewTask,
        submission: ReviewSubmission,
        metric_authority: ApprovedMetricEvidenceAuthorityRecord,
    ) -> StoredApprovedEvidencePublication | None:
        self._validate_identity(identity)
        request_digest = canonical_finalize_request_digest(
            source_task=source_task,
            submission=submission,
            metric_authority=metric_authority,
        )
        async with self._sessions() as session:
            row = await self._receipt_row(session, identity, for_update=False)
        if row is None:
            return None
        return await self._replay_from_receipt(row, identity, request_digest)

    async def probe_finalize_receipt_by_digest(
        self,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredApprovedEvidencePublication | None:
        """Replay an application finalize intent before constructing a submission."""

        self._validate_identity(identity)
        if identity.operation != "finalize_review_intent_v1":
            raise ValueError("digest-only probe requires finalize_review_intent_v1")
        self._validate_request_digest(request_digest)
        async with self._sessions() as session:
            row = await self._receipt_row(session, identity, for_update=False)
        if row is None:
            return None
        return await self._replay_from_receipt(row, identity, request_digest)

    async def commit_finalize(
        self,
        *,
        source_task: StoredReviewTask,
        submission: ReviewSubmission,
        authority: ReviewPublicationAuthority,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredApprovedEvidencePublication:
        """Commit reviewed facts, v2 batch, event, terminal task, and receipt once."""

        self._validate_identity(identity)
        self._validate_request_digest(request_digest)
        if identity.operation == "finalize_review":
            expected_digest = canonical_finalize_request_digest(
                source_task=source_task,
                submission=submission,
                metric_authority=authority.metric_authority,
            )
            if request_digest != expected_digest:
                raise ValueError("finalize request digest does not match canonical request facts")
        authority = ReviewPublicationAuthority.model_validate(
            authority.model_dump(mode="python"), strict=True
        )
        if (
            identity.task_id != source_task.task.id
            or identity.actor_principal_id != source_task.claimed_principal_id
            or authority.task_id != source_task.task.id
            or authority.session_id != source_task.task.session_id
            or authority.result_report_id != submission.report.report_id
            or authority.review_revision != submission.report.revision
            or authority.approved_at != submission.completed_at
            or authority.task_status is not submission.review_task.status
        ):
            raise PublicationAuthorityError("finalize authority does not bind terminal facts")
        if any(decision.reviewed_at > authority.approved_at for decision in submission.decisions):
            raise PublicationAuthorityError("publication cannot precede a review decision")
        replay_batch_id: UUID | None = None
        async with self._sessions() as session, session.begin():
            task_row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_tasks WHERE task_id=:task_id FOR UPDATE"),
                        {"task_id": identity.task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if task_row is None:
                raise RevisionConflictError("review task is unavailable")
            receipt = await self._receipt_row(session, identity, for_update=True)
            if receipt is not None:
                self._validate_receipt(receipt, request_digest)
                replay_batch_id = UUID(receipt["result_identity"])
            else:
                if (
                    task_row["status"] != ReviewTaskStatus.IN_REVIEW.value
                    or task_row["lock_version"] != source_task.lock_version
                    or task_row["claimed_principal_id"] != source_task.claimed_principal_id
                    or task_row["claimed_credential_id"] != source_task.claimed_credential_id
                ):
                    raise RevisionConflictError("finalize lost task ownership or revision")
                await self._lock_source_report_authority(session, source_task=source_task)
                ownership = await self._load_ownership_in_session(
                    session, source_task.task.id, for_update=True
                )
                if ownership.bound_at > authority.approved_at:
                    raise PublicationAuthorityError("ownership cannot be bound after approval")
                await self._quality._stage_review_submission(  # noqa: SLF001
                    session, source_task=source_task, submission=submission
                )
                result_report = await self._load_result_report_authority(
                    session,
                    task_id=source_task.task.id,
                    report_id=submission.report.report_id,
                    revision=submission.report.revision,
                )
                result_shots, envelope_ids = await self._load_result_authority(
                    session,
                    task_id=source_task.task.id,
                    report_id=submission.report.report_id,
                    revision=submission.report.revision,
                )
                decisions, corrections = await self._load_review_audit_authority(
                    session, source_task=source_task, result_shots=result_shots
                )
                if (
                    result_report != submission.report
                    or decisions != submission.decisions
                    or corrections != submission.corrections
                ):
                    raise PersistenceIntegrityError("staged review facts drift from submission")
                if all(shot.review_status.value == "rejected" for shot in result_shots):
                    self._builder.build(
                        result_report=result_report,
                        result_shots=result_shots,
                        source_envelope_ids=envelope_ids,
                        ownership=ownership,
                        authority=authority,
                    )
                await self._stage_metric_authority(
                    session,
                    source_task=source_task,
                    result_report=result_report,
                    result_shots=result_shots,
                    metric_authority=authority.metric_authority,
                )
                stored_metric_authority = await self._load_metric_authority_in_session(
                    session,
                    task_id=source_task.task.id,
                    report_id=result_report.report_id,
                    revision=result_report.revision,
                    approval_kind=authority.metric_authority.approval_kind,
                    approval_policy_version=(authority.metric_authority.approval_policy_version),
                )
                stored_authority = authority.model_copy(
                    update={"metric_authority": stored_metric_authority}
                )
                publication = self._builder.build(
                    result_report=result_report,
                    result_shots=result_shots,
                    source_envelope_ids=envelope_ids,
                    ownership=ownership,
                    authority=stored_authority,
                )
                await self._stage_publication(
                    session,
                    source_task=source_task,
                    submission=submission,
                    user_id=ownership.user_id,
                    batch=publication.batch,
                    event=publication.event,
                    envelope_ids=envelope_ids,
                )
                updated = (
                    await session.execute(
                        text(
                            "UPDATE review_tasks SET status=:status, "
                            "result_report_id=:report_id, result_report_revision=:revision, "
                            "completed_at=:completed_at, updated_at=:completed_at, "
                            "lock_version=lock_version+1 WHERE task_id=:task_id "
                            "AND status='in_review' AND lock_version=:expected "
                            "AND claimed_principal_id=:actor AND "
                            "claimed_credential_id=:credential RETURNING task_id"
                        ),
                        {
                            "status": submission.review_task.status.value,
                            "report_id": submission.report.report_id,
                            "revision": submission.report.revision,
                            "completed_at": submission.completed_at,
                            "task_id": source_task.task.id,
                            "expected": source_task.lock_version,
                            "actor": source_task.claimed_principal_id,
                            "credential": source_task.claimed_credential_id,
                        },
                    )
                ).one_or_none()
                if updated is None:
                    raise RevisionConflictError("finalize terminal CAS failed")
                await self._insert_receipt(
                    session,
                    identity=identity,
                    request_digest=request_digest,
                    batch=publication.batch,
                    created_at=authority.approved_at,
                )
                replay_batch_id = publication.batch.batch_id
        if replay_batch_id is None:
            raise PersistenceIntegrityError("finalize committed without a batch identity")
        return await self.load_publication(replay_batch_id, replayed=receipt is not None)

    async def load_publication(
        self, batch_id: UUID, *, replayed: bool = False
    ) -> StoredApprovedEvidencePublication:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM approved_evidence_batches WHERE batch_id=:batch_id"),
                        {"batch_id": batch_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            lineage_row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_publication_lineage WHERE batch_id=:batch_id"),
                        {"batch_id": batch_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            metric_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM review_metric_authority WHERE task_id=(SELECT "
                            "task_id FROM approved_evidence_batches WHERE batch_id=:batch_id) "
                            "ORDER BY sequence,shot_id"
                        ),
                        {"batch_id": batch_id},
                    )
                )
                .mappings()
                .all()
            )
            source_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM approved_batch_sources WHERE batch_id=:batch_id "
                            "ORDER BY sequence,shot_id"
                        ),
                        {"batch_id": batch_id},
                    )
                )
                .mappings()
                .all()
            )
            receipt_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM quality_gate_idempotency_receipts "
                            "WHERE result_kind='approved_evidence_batch' "
                            "AND result_identity=:batch_id"
                        ),
                        {"batch_id": str(batch_id)},
                    )
                )
                .mappings()
                .all()
            )
        if row is None:
            raise PersistenceIntegrityError("approved batch is missing")
        if lineage_row is None:
            raise PersistenceIntegrityError("approved publication lineage is missing")
        payload = row["payload"]
        _verify_digest(payload, row["payload_sha256"])
        batch = _validate_json_model(ApprovedEvidenceBatchV2, payload)
        without_content = batch.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(without_content) != batch.content_sha256:
            raise PersistenceIntegrityError("approved batch content digest mismatch")
        if (
            batch.batch_id != row["batch_id"]
            or batch.user_id != row["user_id"]
            or batch.session_id != row["session_id"]
            or batch.report_id != row["report_id"]
            or batch.review_revision != row["review_revision"]
            or batch.approved_at != row["approved_at"]
            or batch.content_sha256 != row["content_sha256"]
            or batch.approval_kind.value != row["approval_kind"]
            or batch.approval_policy_version != row["approval_policy_version"]
        ):
            raise PersistenceIntegrityError("approved batch columns drift from payload")
        lineage_payload = lineage_row["payload"]
        _verify_digest(lineage_payload, lineage_row["payload_sha256"])
        lineage = _validate_json_model(ReviewPublicationRecord, lineage_payload)
        calibration = batch.metric_shots[0].table_geometry
        if (
            lineage.task_id != row["task_id"]
            or lineage.user_id != batch.user_id
            or lineage.session_id != batch.session_id
            or lineage.result_report_id != batch.report_id
            or lineage.review_revision != batch.review_revision
            or lineage.batch_id != batch.batch_id
            or lineage.event_id != row["event_id"]
            or lineage.content_sha256 != batch.content_sha256
            or lineage.approval_kind is not batch.approval_kind
            or lineage.approval_policy_version != batch.approval_policy_version
            or lineage.calibration_id != calibration.calibration_id
            or lineage.calibration_revision_id != calibration.calibration_revision_id
            or lineage.calibration_revision_number != calibration.calibration_revision_number
            or lineage.approved_at != batch.approved_at
        ):
            raise PersistenceIntegrityError("publication lineage drifts from the batch")
        stored_metrics: list[ApprovedMetricShotEvidence] = []
        for metric_row in metric_rows:
            _verify_digest(metric_row["payload"], metric_row["payload_sha256"])
            metric = _validate_json_model(ApprovedMetricShotEvidence, metric_row["payload"])
            if (
                metric.shot_id != metric_row["shot_id"]
                or metric.session_id != metric_row["session_id"]
                or metric.sequence != metric_row["sequence"]
                or metric.review_revision != metric_row["review_revision"]
                or metric.table_geometry.calibration_id != metric_row["calibration_id"]
                or metric.table_geometry.calibration_revision_id
                != metric_row["calibration_revision_id"]
                or metric.table_geometry.calibration_revision_number
                != metric_row["calibration_revision_number"]
            ):
                raise PersistenceIntegrityError("metric authority columns drift from payload")
            stored_metrics.append(metric)
        if tuple(stored_metrics) != batch.metric_shots:
            raise PersistenceIntegrityError("metric authority drifts from the batch")
        expected_sources = tuple(
            (
                shot.shot_id,
                shot.session_id,
                shot.sequence,
                envelope_id,
                shot.review_status.value,
            )
            for shot, envelope_id in zip(batch.shots, batch.source_envelope_ids, strict=True)
        )
        actual_sources = tuple(
            (
                source["shot_id"],
                source["session_id"],
                source["sequence"],
                source["source_envelope_id"],
                source["review_status"],
            )
            for source in source_rows
        )
        if actual_sources != expected_sources:
            raise PersistenceIntegrityError("batch source authority drifts from the batch")
        if (
            len(receipt_rows) != 1
            or receipt_rows[0]["task_id"] != lineage.task_id
            or receipt_rows[0]["result_digest"] != batch.content_sha256
        ):
            raise PersistenceIntegrityError("publication receipt authority is incomplete")
        task = await self._quality.load_task(row["task_id"])
        event = await self._outbox.load(row["event_id"])
        if (
            task.result_report_id != batch.report_id
            or task.result_report_revision != batch.review_revision
            or task.task.status.value != lineage.task_status
            or event.event_type != "evidence.session.approved.v1"
            or event.aggregate_type != "review_session"
            or event.aggregate_id != batch.session_id
            or event.aggregate_version != batch.review_revision
            or event.occurred_at != batch.approved_at
            or event.causation_id != task.task.id
            or event.correlation_id != batch.session_id
            or event.idempotency_key
            != f"evidence-approved:{batch.session_id}:{batch.review_revision}"
            or event.payload != {"approvedEvidenceBatch": batch.model_dump(mode="json")}
        ):
            raise PersistenceIntegrityError("approved event does not contain the exact batch")
        ownership = await self._quality.load_session_ownership(task.task.id)
        result_report = await self._quality.load_result_report(
            batch.report_id, batch.review_revision
        )
        async with self._sessions() as session:
            result_shots, envelope_ids = await self._load_result_authority(
                session,
                task_id=task.task.id,
                report_id=batch.report_id,
                revision=batch.review_revision,
            )
            await self._load_review_audit_authority(
                session, source_task=task, result_shots=result_shots
            )
        metric_authority = ApprovedMetricEvidenceAuthorityRecord(
            approval_kind=batch.approval_kind,
            approval_policy_version=batch.approval_policy_version,
            metric_shots=tuple(stored_metrics),
        )
        rebuilt = self._builder.build(
            result_report=result_report,
            result_shots=result_shots,
            source_envelope_ids=envelope_ids,
            ownership=ownership,
            authority=ReviewPublicationAuthority(
                task_id=task.task.id,
                session_id=batch.session_id,
                result_report_id=batch.report_id,
                review_revision=batch.review_revision,
                task_status=task.task.status,
                batch_id=batch.batch_id,
                event_id=event.event_id,
                approved_at=batch.approved_at,
                metric_authority=metric_authority,
            ),
        )
        if rebuilt.batch != batch or rebuilt.event != event:
            raise PersistenceIntegrityError("stored publication cannot be rebuilt from authority")
        return StoredApprovedEvidencePublication(
            task=task, batch=batch, event=event, replayed=replayed
        )

    async def load_publication_for_task(self, task_id: UUID) -> StoredApprovedEvidencePublication:
        """Load the one strict terminal publication bound to a task."""

        async with self._sessions() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT batch_id FROM approved_evidence_batches "
                        "WHERE task_id=:task_id ORDER BY batch_id"
                    ),
                    {"task_id": task_id},
                )
            ).all()
        if len(rows) != 1:
            raise PersistenceIntegrityError("terminal task publication is unavailable")
        return await self.load_publication(rows[0].batch_id)

    @staticmethod
    def _validate_identity(identity: IdempotencyIdentity) -> None:
        if (
            identity.operation not in {"finalize_review", "finalize_review_intent_v1"}
            or identity.shot_id is not None
            or not identity.actor_principal_id
            or not identity.idempotency_key
        ):
            raise ValueError("finalize idempotency identity is invalid")

    @staticmethod
    def _validate_request_digest(request_digest: str) -> None:
        if (
            type(request_digest) is not str
            or len(request_digest) != 64
            or any(character not in "0123456789abcdef" for character in request_digest)
        ):
            raise ValueError("finalize request digest must be canonical sha256")

    @staticmethod
    async def _receipt_row(session, identity, *, for_update: bool):
        query = (
            "SELECT * FROM quality_gate_idempotency_receipts "
            "WHERE actor_principal_id=:actor AND operation=:operation "
            "AND task_id=:task_id AND shot_id_or_zero_uuid=:shot_id "
            "AND idempotency_key=:key FOR UPDATE"
            if for_update
            else "SELECT * FROM quality_gate_idempotency_receipts "
            "WHERE actor_principal_id=:actor AND operation=:operation "
            "AND task_id=:task_id AND shot_id_or_zero_uuid=:shot_id "
            "AND idempotency_key=:key"
        )
        return (
            (
                await session.execute(
                    text(query),
                    {
                        "actor": identity.actor_principal_id,
                        "operation": identity.operation,
                        "task_id": identity.task_id,
                        "shot_id": ZERO_UUID,
                        "key": identity.idempotency_key,
                    },
                )
            )
            .mappings()
            .one_or_none()
        )

    def _validate_receipt(self, row, request_digest: str) -> None:
        if row["request_digest"] != request_digest:
            raise IdempotencyConflictError("finalize key was reused with different facts")
        if row["result_kind"] != "approved_evidence_batch":
            raise PersistenceIntegrityError("finalize receipt points at the wrong result kind")

    async def _replay_from_receipt(self, row, identity, request_digest):
        self._validate_receipt(row, request_digest)
        result = await self.load_publication(UUID(row["result_identity"]), replayed=True)
        if result.batch.content_sha256 != row["result_digest"]:
            raise PersistenceIntegrityError("finalize receipt result digest drift")
        if result.task.task.id != identity.task_id:
            raise PersistenceIntegrityError("finalize receipt task drift")
        return result

    async def _load_ownership_in_session(self, session, task_id, *, for_update):
        query = (
            "SELECT * FROM review_session_ownership WHERE task_id=:task_id FOR UPDATE"
            if for_update
            else "SELECT * FROM review_session_ownership WHERE task_id=:task_id"
        )
        row = (
            (
                await session.execute(
                    text(query),
                    {"task_id": task_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise PublicationAuthorityError("pre-bound session ownership is missing")
        return self._quality._ownership_from_row(row)  # noqa: SLF001

    @staticmethod
    async def _lock_source_report_authority(session, *, source_task):
        row = (
            (
                await session.execute(
                    text(
                        "SELECT * FROM review_reports WHERE report_id=:report_id "
                        "AND revision=:revision FOR UPDATE"
                    ),
                    {
                        "report_id": source_task.task.report_id,
                        "revision": source_task.task.report_revision,
                    },
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise PersistenceIntegrityError("source report authority is missing")
        _verify_digest(row["payload"], row["payload_sha256"])
        report = _validate_json_model(ReportDocument, row["payload"])
        if (
            report.report_id != source_task.task.report_id
            or report.revision != source_task.task.report_revision
            or report.session_id != source_task.task.session_id
            or report.report_id != row["report_id"]
            or report.revision != row["revision"]
            or report.session_id != row["session_id"]
        ):
            raise PersistenceIntegrityError("source report authority drift")
        return report

    @staticmethod
    async def _load_result_authority(session, *, task_id, report_id, revision):
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT result.*, source.source_envelope_id "
                        "FROM review_shot_results result JOIN review_shot_sources source "
                        "ON source.task_id=result.task_id AND source.shot_id=result.shot_id "
                        "WHERE result.task_id=:task_id AND result.result_report_id=:report_id "
                        "AND result.result_report_revision=:revision "
                        "ORDER BY result.sequence, result.shot_id"
                    ),
                    {"task_id": task_id, "report_id": report_id, "revision": revision},
                )
            )
            .mappings()
            .all()
        )
        if not rows:
            raise PersistenceIntegrityError("result shot authority is missing")
        shots: list[Shot] = []
        envelopes: dict[UUID, UUID] = {}
        for row in rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            shot = _validate_json_model(Shot, row["payload"])
            if (
                shot.id != row["shot_id"]
                or shot.session_id != row["session_id"]
                or shot.sequence != row["sequence"]
                or shot.review_status.value != row["review_status"]
            ):
                raise PersistenceIntegrityError("result shot columns drift from payload")
            shots.append(shot)
            envelopes[shot.id] = row["source_envelope_id"]
        return tuple(shots), envelopes

    @staticmethod
    async def _load_result_report_authority(session, *, task_id, report_id, revision):
        row = (
            (
                await session.execute(
                    text(
                        "SELECT result.* FROM review_result_reports result JOIN review_tasks task "
                        "ON task.task_id=:task_id AND task.result_report_id IS NULL "
                        "WHERE result.report_id=:report_id AND result.revision=:revision "
                        "AND result.session_id=task.session_id "
                        "AND result.source_report_id=task.source_report_id "
                        "AND result.source_report_revision=task.source_report_revision"
                    ),
                    {"task_id": task_id, "report_id": report_id, "revision": revision},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise PersistenceIntegrityError("result report authority is missing")
        _verify_digest(row["payload"], row["payload_sha256"])
        report = _validate_json_model(ReportDocument, row["payload"])
        if (
            report.report_id != row["report_id"]
            or report.revision != row["revision"]
            or report.session_id != row["session_id"]
            or report.parent_revision_id != row["parent_report_id"]
            or report.report_status.value != row["status"]
            or report.generated_at != row["generated_at"]
        ):
            raise PersistenceIntegrityError("result report columns drift from payload")
        return report

    @staticmethod
    async def _load_review_audit_authority(session, *, source_task, result_shots):
        decision_rows = (
            (
                await session.execute(
                    text(
                        "SELECT decision.* FROM review_shot_decisions decision "
                        "JOIN review_shot_sources source ON source.task_id=decision.task_id "
                        "AND source.shot_id=decision.shot_id WHERE decision.task_id=:task_id "
                        "ORDER BY source.sequence,decision.shot_id"
                    ),
                    {"task_id": source_task.task.id},
                )
            )
            .mappings()
            .all()
        )
        decisions: list[ShotReviewDecision] = []
        for row in decision_rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            decision = _validate_json_model(ShotReviewDecision, row["payload"])
            if (
                decision.shot_id != row["shot_id"]
                or decision.outcome.value != row["outcome"]
                or decision.reviewer.principal_id != row["reviewer_principal_id"]
                or decision.reviewer.credential_id != row["reviewer_credential_id"]
                or decision.reviewed_at != row["reviewed_at"]
                or decision.trajectory_reviewed != row["trajectory_reviewed"]
                or decision.reviewer.principal_id != source_task.claimed_principal_id
                or decision.reviewer.credential_id != source_task.claimed_credential_id
            ):
                raise PersistenceIntegrityError("review decision authority drift")
            decisions.append(decision)
        if tuple(decision.shot_id for decision in decisions) != tuple(
            shot.id for shot in result_shots
        ):
            raise PersistenceIntegrityError("review decisions do not cover result shots")
        correction_rows = (
            (
                await session.execute(
                    text(
                        "SELECT * FROM review_corrections WHERE task_id=:task_id "
                        "ORDER BY shot_id,field"
                    ),
                    {"task_id": source_task.task.id},
                )
            )
            .mappings()
            .all()
        )
        corrections: list[ReviewCorrection] = []
        for row in correction_rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            correction = _validate_json_model(ReviewCorrection, row["payload"])
            if (
                correction.shot_id != row["shot_id"]
                or correction.field.value != row["field"]
                or correction.ai_value != row["ai_value"]
                or correction.correct_value != row["correct_value"]
                or correction.note != row["note"]
                or correction.reviewer.principal_id != row["reviewer_principal_id"]
                or correction.reviewer.credential_id != row["reviewer_credential_id"]
                or correction.reviewed_at != row["reviewed_at"]
                or correction.reviewer.principal_id != source_task.claimed_principal_id
                or correction.reviewer.credential_id != source_task.claimed_credential_id
            ):
                raise PersistenceIntegrityError("review correction authority drift")
            corrections.append(correction)
        expected_corrections = tuple(
            correction for decision in decisions for correction in decision.corrections
        )
        if tuple(sorted(corrections, key=lambda item: (item.shot_id, item.field.value))) != tuple(
            sorted(expected_corrections, key=lambda item: (item.shot_id, item.field.value))
        ):
            raise PersistenceIntegrityError("review correction authority is incomplete")
        return tuple(decisions), expected_corrections

    @staticmethod
    async def _stage_metric_authority(
        session,
        *,
        source_task,
        result_report,
        result_shots,
        metric_authority,
    ):
        promotable = tuple(shot for shot in result_shots if shot.review_status.value != "rejected")
        expected_keys = tuple((shot.id, shot.session_id, shot.sequence) for shot in promotable)
        actual_keys = tuple(
            (metric.shot_id, metric.session_id, metric.sequence)
            for metric in metric_authority.metric_shots
        )
        if expected_keys != actual_keys:
            raise PublicationAuthorityError("metric authority does not match promotable results")
        calibration = metric_authority.metric_shots[0].table_geometry
        status_by_id = {shot.id: shot.review_status.value for shot in promotable}
        for metric in metric_authority.metric_shots:
            payload = metric.model_dump(mode="json")
            assert_safe_persistence_payload(payload)
            await session.execute(
                text(
                    "INSERT INTO review_metric_authority "
                    "(task_id,shot_id,review_revision,result_report_id,result_report_revision,"
                    "session_id,sequence,review_status,calibration_id,calibration_revision_id,"
                    "calibration_revision_number,payload,payload_sha256) VALUES "
                    "(:task_id,:shot_id,:revision,:report_id,:revision,:session_id,:sequence,"
                    ":review_status,:calibration_id,:calibration_revision_id,"
                    ":calibration_revision_number,CAST(:payload AS jsonb),:digest)"
                ),
                {
                    "task_id": source_task.task.id,
                    "shot_id": metric.shot_id,
                    "revision": result_report.revision,
                    "report_id": result_report.report_id,
                    "session_id": metric.session_id,
                    "sequence": metric.sequence,
                    "review_status": status_by_id[metric.shot_id],
                    "calibration_id": calibration.calibration_id,
                    "calibration_revision_id": calibration.calibration_revision_id,
                    "calibration_revision_number": calibration.calibration_revision_number,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "digest": canonical_sha256(payload),
                },
            )

    @staticmethod
    async def _load_metric_authority_in_session(
        session, *, task_id, report_id, revision, approval_kind, approval_policy_version
    ):
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT * FROM review_metric_authority WHERE task_id=:task_id "
                        "AND result_report_id=:report_id AND result_report_revision=:revision "
                        "ORDER BY sequence,shot_id"
                    ),
                    {"task_id": task_id, "report_id": report_id, "revision": revision},
                )
            )
            .mappings()
            .all()
        )
        metrics: list[ApprovedMetricShotEvidence] = []
        for row in rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            metric = _validate_json_model(ApprovedMetricShotEvidence, row["payload"])
            if (
                metric.shot_id != row["shot_id"]
                or metric.session_id != row["session_id"]
                or metric.sequence != row["sequence"]
                or metric.review_revision != row["review_revision"]
                or metric.table_geometry.calibration_id != row["calibration_id"]
                or metric.table_geometry.calibration_revision_id != row["calibration_revision_id"]
                or metric.table_geometry.calibration_revision_number
                != row["calibration_revision_number"]
            ):
                raise PersistenceIntegrityError("metric authority columns drift from payload")
            metrics.append(metric)
        try:
            return ApprovedMetricEvidenceAuthorityRecord(
                approval_kind=approval_kind,
                approval_policy_version=approval_policy_version,
                metric_shots=tuple(metrics),
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("metric authority is invalid") from exc

    async def _stage_publication(
        self,
        session,
        *,
        source_task,
        submission,
        user_id,
        batch,
        event,
        envelope_ids,
    ):
        metric_by_id = {shot.shot_id: shot for shot in batch.metric_shots}
        calibration = batch.metric_shots[0].table_geometry
        record = ReviewPublicationRecord(
            task_id=source_task.task.id,
            user_id=user_id,
            session_id=batch.session_id,
            result_report_id=batch.report_id,
            review_revision=batch.review_revision,
            task_status=submission.review_task.status.value,
            batch_id=batch.batch_id,
            event_id=event.event_id,
            content_sha256=batch.content_sha256,
            approval_kind=batch.approval_kind,
            approval_policy_version=batch.approval_policy_version,
            calibration_id=calibration.calibration_id,
            calibration_revision_id=calibration.calibration_revision_id,
            calibration_revision_number=calibration.calibration_revision_number,
            approved_at=batch.approved_at,
        )
        record_payload = record.model_dump(mode="json")
        assert_safe_persistence_payload(record_payload)
        await session.execute(
            text(
                "INSERT INTO review_publication_lineage "
                "(task_id,user_id,session_id,report_id,review_revision,approved_at,task_status,"
                "batch_id,event_id,content_sha256,approval_kind,approval_policy_version,"
                "calibration_id,calibration_revision_id,calibration_revision_number,payload,"
                "payload_sha256) VALUES (:task_id,:user_id,:session_id,:report_id,:revision,"
                ":approved_at,:task_status,:batch_id,:event_id,:content_sha256,:approval_kind,"
                ":policy,:calibration_id,:calibration_revision_id,:calibration_revision_number,"
                "CAST(:payload AS jsonb),:digest)"
            ),
            {
                **record.model_dump(mode="python"),
                "report_id": record.result_report_id,
                "revision": record.review_revision,
                "policy": record.approval_policy_version,
                "payload": json.dumps(record_payload, ensure_ascii=False),
                "digest": canonical_sha256(record_payload),
            },
        )
        batch_payload = batch.model_dump(mode="json")
        assert_safe_persistence_payload(batch_payload)
        await session.execute(
            text(
                "INSERT INTO approved_evidence_batches "
                "(batch_id,schema_version,user_id,session_id,report_id,review_revision,"
                "approved_at,content_sha256,task_id,event_id,approval_kind,"
                "approval_policy_version,calibration_id,calibration_revision_id,"
                "calibration_revision_number,payload,payload_sha256) VALUES "
                "(:batch_id,'2.0',:user_id,:session_id,:report_id,:review_revision,"
                ":approved_at,:content_sha256,:task_id,:event_id,:approval_kind,:policy,"
                ":calibration_id,:calibration_revision_id,:calibration_revision_number,"
                "CAST(:payload AS jsonb),:digest)"
            ),
            {
                "batch_id": batch.batch_id,
                "user_id": batch.user_id,
                "session_id": batch.session_id,
                "report_id": batch.report_id,
                "review_revision": batch.review_revision,
                "approved_at": batch.approved_at,
                "content_sha256": batch.content_sha256,
                "task_id": source_task.task.id,
                "event_id": event.event_id,
                "approval_kind": batch.approval_kind.value,
                "policy": batch.approval_policy_version,
                "calibration_id": calibration.calibration_id,
                "calibration_revision_id": calibration.calibration_revision_id,
                "calibration_revision_number": calibration.calibration_revision_number,
                "payload": json.dumps(batch_payload, ensure_ascii=False),
                "digest": canonical_sha256(batch_payload),
            },
        )
        shot_by_id = {shot.shot_id: shot for shot in batch.shots}
        for shot, envelope_id in zip(batch.shots, batch.source_envelope_ids, strict=True):
            metric = metric_by_id[shot.shot_id]
            if envelope_ids[shot.shot_id] != envelope_id or metric.sequence != shot.sequence:
                raise PublicationAuthorityError("batch source authority drift")
            await session.execute(
                text(
                    "INSERT INTO approved_batch_sources "
                    "(batch_id,task_id,shot_id,sequence,source_envelope_id,report_id,"
                    "review_revision,review_status,session_id) VALUES "
                    "(:batch_id,:task_id,:shot_id,:sequence,:envelope_id,:report_id,"
                    ":revision,:status,:session_id)"
                ),
                {
                    "batch_id": batch.batch_id,
                    "task_id": source_task.task.id,
                    "shot_id": shot.shot_id,
                    "sequence": shot.sequence,
                    "envelope_id": envelope_id,
                    "report_id": batch.report_id,
                    "revision": batch.review_revision,
                    "status": shot_by_id[shot.shot_id].review_status.value,
                    "session_id": shot.session_id,
                },
            )
        await PostgresOutboxRepository.enqueue_in_session(
            session, event, available_at=batch.approved_at
        )

    @staticmethod
    async def _insert_receipt(session, *, identity, request_digest, batch, created_at) -> None:
        await session.execute(
            text(
                "INSERT INTO quality_gate_idempotency_receipts "
                "(actor_principal_id,operation,task_id,shot_id_or_zero_uuid,idempotency_key,"
                "request_digest,result_kind,result_identity,result_digest,created_at) VALUES "
                "(:actor,:operation,:task_id,:shot_id,:key,:request_digest,"
                "'approved_evidence_batch',:result_identity,:result_digest,:created_at)"
            ),
            {
                "actor": identity.actor_principal_id,
                "operation": identity.operation,
                "task_id": identity.task_id,
                "shot_id": ZERO_UUID,
                "key": identity.idempotency_key,
                "request_digest": request_digest,
                "result_identity": str(batch.batch_id),
                "result_digest": batch.content_sha256,
                "created_at": created_at,
            },
        )
