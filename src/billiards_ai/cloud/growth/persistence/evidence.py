"""PostgreSQL durability for approved growth evidence and trusted re-admission."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, TypeAdapter
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.cloud.growth.repository import (
    EvidenceInboxReceipt,
    EvidenceIngestResult,
    EvidenceIngestStatus,
    EvidenceSourceSummary,
    revalidate_approved_batch,
)
from billiards_ai.cloud.review.persistence.privacy import assert_safe_persistence_payload
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.evidence import (
    AcceptedMetricEvidenceInput,
    ApprovedEvidenceBatch,
    ApprovedEvidenceBatchAny,
    ApprovedEvidenceBatchV2,
    _seal_accepted_metric_evidence_input,
)
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord

from ._common import event_from_row, insert_outbox, json_text, strict_json_model
from .errors import GrowthPersistenceIntegrityError

_UUID4 = TypeAdapter(UUID4)


class _MetricAuthorityEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["growth-metric-authority.v1"]
    status: Literal["APPROVED"]
    review_revision: int
    approval_kind: str
    approval_policy_version: str
    authority: ApprovedMetricEvidenceAuthorityRecord


def _canonical_metric_shots(values):
    return tuple(sorted(values, key=lambda item: (item.sequence, str(item.shot_id))))


def _authority_for(
    batch: ApprovedEvidenceBatchAny,
    authority: ApprovedMetricEvidenceAuthorityRecord | None,
) -> ApprovedMetricEvidenceAuthorityRecord | None:
    if isinstance(batch, ApprovedEvidenceBatch):
        return None
    if authority is None:
        raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED)
    try:
        validated = ApprovedMetricEvidenceAuthorityRecord.model_validate(
            authority.model_dump(mode="python"), strict=True
        )
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise EvidenceIngestError(EvidenceIngestErrorCode.NOT_APPROVED) from exc
    if (
        _canonical_metric_shots(validated.metric_shots)
        != _canonical_metric_shots(batch.metric_shots)
        or validated.approval_kind is not batch.approval_kind
        or validated.approval_policy_version != batch.approval_policy_version
    ):
        raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
    return ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=batch.approval_kind,
        approval_policy_version=batch.approval_policy_version,
        metric_shots=_canonical_metric_shots(batch.metric_shots),
    )


def _accepted_event(
    batch: ApprovedEvidenceBatchAny,
    receipt: EvidenceInboxReceipt,
    *,
    event_id: UUID,
) -> DomainEvent:
    return DomainEvent(
        event_id=event_id,
        event_type="growth.evidence.batch.accepted.v1",
        aggregate_type="approved_evidence_batch",
        aggregate_id=batch.batch_id,
        aggregate_version=batch.review_revision,
        occurred_at=receipt.received_at,
        causation_id=batch.report_id,
        correlation_id=batch.batch_id,
        idempotency_key=f"{batch.schema_version}:{batch.session_id}:{batch.review_revision}",
        payload={
            "batch_id": str(batch.batch_id),
            "user_id": str(batch.user_id),
            "session_id": str(batch.session_id),
            "report_id": str(batch.report_id),
            "review_revision": batch.review_revision,
            "content_sha256": batch.content_sha256,
        },
    )


def _expected_models(
    batch: ApprovedEvidenceBatchAny,
    *,
    receipt_id: UUID,
    received_at: datetime,
) -> tuple[EvidenceInboxReceipt, EvidenceSourceSummary, AcceptedMetricEvidenceInput | None]:
    receipt = EvidenceInboxReceipt(
        receipt_id=receipt_id,
        batch_id=batch.batch_id,
        user_id=batch.user_id,
        session_id=batch.session_id,
        report_id=batch.report_id,
        review_revision=batch.review_revision,
        content_sha256=batch.content_sha256,
        received_at=received_at,
    )
    source = EvidenceSourceSummary(
        batch_id=batch.batch_id,
        user_id=batch.user_id,
        session_id=batch.session_id,
        report_id=batch.report_id,
        report_revision=batch.report_revision,
        review_revision=batch.review_revision,
        shot_ids=tuple(shot.shot_id for shot in batch.shots),
        source_envelope_ids=batch.source_envelope_ids,
        content_sha256=batch.content_sha256,
    )
    admission = None
    if isinstance(batch, ApprovedEvidenceBatchV2):
        admission = AcceptedMetricEvidenceInput(
            admission_id=receipt.receipt_id,
            admitted_at=receipt.received_at,
            batch_schema_version=batch.schema_version,
            batch_id=batch.batch_id,
            session_id=batch.session_id,
            report_id=batch.report_id,
            review_revision=batch.review_revision,
            batch_content_sha256=batch.content_sha256,
            approval_kind=batch.approval_kind,
            approval_policy_version=batch.approval_policy_version,
            batch=batch,
        )
    return receipt, source, admission


class PostgresGrowthEvidenceRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def resolve_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        authenticated_user_id: UUID,
    ) -> EvidenceIngestResult | None:
        try:
            validated = revalidate_approved_batch(batch)
        except (AttributeError, RecursionError, TypeError, ValueError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH) from exc
        if validated.user_id != authenticated_user_id:
            raise EvidenceIngestError(EvidenceIngestErrorCode.LINEAGE_MISMATCH)
        return await self._find(validated)

    @staticmethod
    def _batch_from_row(row) -> ApprovedEvidenceBatchAny:
        try:
            payload = row["payload"]
            if canonical_sha256(payload) != row["payload_sha256"]:
                raise ValueError("batch row digest mismatch")
            schema = payload.get("schema_version") if isinstance(payload, dict) else None
            model = ApprovedEvidenceBatch if schema == "1.0" else ApprovedEvidenceBatchV2
            batch = strict_json_model(model, payload)
            if (
                batch.batch_id != row["batch_id"]
                or batch.schema_version != row["schema_version"]
                or batch.user_id != row["user_id"]
                or batch.session_id != row["session_id"]
                or batch.report_id != row["report_id"]
                or batch.report_revision != row["report_revision"]
                or batch.review_revision != row["review_revision"]
                or batch.approved_at != row["approved_at"]
                or batch.content_sha256 != row["content_sha256"]
                or batch.content_sha256
                != canonical_sha256(batch.model_dump(mode="json", exclude={"content_sha256"}))
            ):
                raise ValueError("batch columns and payload differ")
            return batch
        except (AttributeError, TypeError, ValueError) as exc:
            raise GrowthPersistenceIntegrityError("invalid persisted evidence batch") from exc

    async def _find(self, batch: ApprovedEvidenceBatchAny):
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_evidence_batches WHERE "
                            "(schema_version=:schema AND session_id=:session_id "
                            "AND review_revision=:revision) OR "
                            "(schema_version=:schema AND batch_id=:batch_id) "
                            "ORDER BY batch_id LIMIT 1"
                        ),
                        {
                            "schema": batch.schema_version,
                            "session_id": batch.session_id,
                            "revision": batch.review_revision,
                            "batch_id": batch.batch_id,
                        },
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            existing = self._batch_from_row(row)
            if existing != batch:
                raise EvidenceIngestError(EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT)
            receipt_row = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_evidence_inbox WHERE "
                            "schema_version=:schema AND batch_id=:batch_id"
                        ),
                        {"schema": batch.schema_version, "batch_id": batch.batch_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if receipt_row is None:
                raise GrowthPersistenceIntegrityError("evidence receipt is missing")
            try:
                if canonical_sha256(receipt_row["payload"]) != receipt_row["payload_sha256"]:
                    raise ValueError("evidence receipt digest mismatch")
                receipt = strict_json_model(EvidenceInboxReceipt, receipt_row["payload"])
                expected_receipt, expected_source, _ = _expected_models(
                    batch,
                    receipt_id=receipt.receipt_id,
                    received_at=receipt.received_at,
                )
                source_row = (
                    (
                        await session.execute(
                            text(
                                "SELECT payload,payload_sha256 FROM growth_evidence_sources "
                                "WHERE schema_version=:schema AND batch_id=:batch_id"
                            ),
                            {"schema": batch.schema_version, "batch_id": batch.batch_id},
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if (
                    receipt != expected_receipt
                    or receipt_row["schema_version"] != batch.schema_version
                    or receipt_row["receipt_id"] != receipt.receipt_id
                    or receipt_row["batch_id"] != receipt.batch_id
                    or receipt_row["user_id"] != receipt.user_id
                    or receipt_row["session_id"] != receipt.session_id
                    or receipt_row["report_id"] != receipt.report_id
                    or receipt_row["review_revision"] != receipt.review_revision
                    or receipt_row["content_sha256"] != receipt.content_sha256
                    or receipt_row["received_at"] != receipt.received_at
                    or source_row is None
                    or canonical_sha256(source_row["payload"]) != source_row["payload_sha256"]
                    or strict_json_model(EvidenceSourceSummary, source_row["payload"])
                    != expected_source
                ):
                    raise ValueError("evidence receipt graph differs from batch")
                event_row = (
                    (
                        await session.execute(
                            text(
                                "SELECT * FROM core_intelligence_outbox WHERE "
                                "owner_user_id=:user_id AND aggregate_namespace=:schema "
                                "AND aggregate_id=:batch_id AND event_type=:event_type"
                            ),
                            {
                                "user_id": batch.user_id,
                                "schema": batch.schema_version,
                                "batch_id": batch.batch_id,
                                "event_type": "growth.evidence.batch.accepted.v1",
                            },
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if event_row is None:
                    raise ValueError("accepted evidence outbox event is missing")
                event = event_from_row(event_row)
                if event != _accepted_event(batch, receipt, event_id=event.event_id):
                    raise ValueError("accepted evidence outbox event differs from batch")
            except (AttributeError, RecursionError, TypeError, ValueError) as exc:
                raise GrowthPersistenceIntegrityError(
                    "persisted evidence replay graph is invalid"
                ) from exc
            accepted = None
            if isinstance(batch, ApprovedEvidenceBatchV2):
                values = await self.list_accepted_metric_inputs(batch.user_id)
                accepted = next((item for item in values if item.batch_id == batch.batch_id), None)
            return EvidenceIngestResult(
                status=EvidenceIngestStatus.REPLAYED,
                receipt=receipt,
                accepted_metric_input=accepted,
            )

    async def append_approved_batch(
        self,
        batch: ApprovedEvidenceBatchAny,
        *,
        metric_authority: ApprovedMetricEvidenceAuthorityRecord | None = None,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> EvidenceIngestResult:
        try:
            validated = revalidate_approved_batch(batch)
        except (AttributeError, RecursionError, TypeError, ValueError) as exc:
            raise EvidenceIngestError(EvidenceIngestErrorCode.INVALID_BATCH) from exc
        if replay := await self._find(validated):
            return replay
        authority = _authority_for(validated, metric_authority)
        received_at = clock()
        receipt_id = _UUID4.validate_python(uuid_factory(), strict=True)
        event_id = _UUID4.validate_python(uuid_factory(), strict=True)
        receipt, source, admission = _expected_models(
            validated, receipt_id=receipt_id, received_at=received_at
        )
        event = _accepted_event(validated, receipt, event_id=event_id)
        batch_payload = validated.model_dump(mode="json")
        receipt_payload = receipt.model_dump(mode="json")
        source_payload = source.model_dump(mode="json")
        for payload in (batch_payload, receipt_payload, source_payload, event.payload):
            assert_safe_persistence_payload(payload)
        authority_envelope = (
            None
            if authority is None
            else _MetricAuthorityEnvelope(
                schema_version="growth-metric-authority.v1",
                status="APPROVED",
                review_revision=validated.review_revision,
                approval_kind=authority.approval_kind.value,
                approval_policy_version=authority.approval_policy_version,
                authority=authority,
            )
        )
        authority_payload = (
            None if authority_envelope is None else authority_envelope.model_dump(mode="json")
        )
        if authority_payload is not None:
            assert_safe_persistence_payload(authority_payload)
        try:
            async with self._sessions() as session, session.begin():
                await session.execute(
                    text(
                        "INSERT INTO growth_evidence_batches "
                        "(batch_id,schema_version,user_id,session_id,report_id,report_revision,"
                        "review_revision,approved_at,content_sha256,payload,payload_sha256) VALUES "
                        "(:batch_id,:schema,:user_id,:session_id,:report_id,:report_revision,"
                        ":review_revision,:approved_at,:content_sha256,CAST(:payload AS jsonb),"
                        ":payload_sha256)"
                    ),
                    {
                        "batch_id": validated.batch_id,
                        "schema": validated.schema_version,
                        "user_id": validated.user_id,
                        "session_id": validated.session_id,
                        "report_id": validated.report_id,
                        "report_revision": validated.report_revision,
                        "review_revision": validated.review_revision,
                        "approved_at": validated.approved_at,
                        "content_sha256": validated.content_sha256,
                        "payload": json_text(batch_payload),
                        "payload_sha256": canonical_sha256(batch_payload),
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO growth_evidence_inbox "
                        "(receipt_id,schema_version,batch_id,user_id,session_id,report_id,review_revision,"
                        "content_sha256,received_at,payload,payload_sha256) VALUES "
                        "(:receipt_id,:schema_version,:batch_id,:user_id,:session_id,:report_id,:review_revision,"
                        ":content_sha256,:received_at,CAST(:payload AS jsonb),:payload_sha256)"
                    ),
                    receipt.model_dump(mode="python")
                    | {
                        "schema_version": validated.schema_version,
                        "payload": json_text(receipt_payload),
                        "payload_sha256": canonical_sha256(receipt_payload),
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO growth_evidence_sources "
                        "(schema_version,batch_id,payload,payload_sha256) VALUES "
                        "(:schema_version,:batch_id,CAST(:payload AS jsonb),:payload_sha256)"
                    ),
                    {
                        "schema_version": validated.schema_version,
                        "batch_id": validated.batch_id,
                        "payload": json_text(source_payload),
                        "payload_sha256": canonical_sha256(source_payload),
                    },
                )
                if authority_payload is not None:
                    await session.execute(
                        text(
                            "INSERT INTO growth_metric_authorities "
                            "(schema_version,batch_id,authority_schema_version,approval_kind,"
                            "approval_policy_version,authority_status,review_revision,payload,"
                            "payload_sha256) VALUES (:schema_version,:batch_id,"
                            ":authority_schema_version,:approval_kind,:approval_policy_version,"
                            ":authority_status,:review_revision,CAST(:payload AS jsonb),"
                            ":payload_sha256)"
                        ),
                        {
                            "schema_version": validated.schema_version,
                            "batch_id": validated.batch_id,
                            "authority_schema_version": authority_envelope.schema_version,
                            "approval_kind": authority_envelope.approval_kind,
                            "approval_policy_version": authority_envelope.approval_policy_version,
                            "authority_status": authority_envelope.status,
                            "review_revision": authority_envelope.review_revision,
                            "payload": json_text(authority_payload),
                            "payload_sha256": canonical_sha256(authority_payload),
                        },
                    )
                await insert_outbox(
                    session,
                    event,
                    owner_user_id=validated.user_id,
                    aggregate_namespace=validated.schema_version,
                )
        except IntegrityError:
            if replay := await self._find(validated):
                return replay
            raise EvidenceIngestError(EvidenceIngestErrorCode.IDEMPOTENCY_CONFLICT) from None
        return EvidenceIngestResult(
            status=EvidenceIngestStatus.CREATED,
            receipt=receipt,
            accepted_metric_input=(
                None if admission is None else _seal_accepted_metric_evidence_input(admission)
            ),
        )

    async def list_batches(self, user_id: UUID) -> tuple[ApprovedEvidenceBatchAny, ...]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_evidence_batches WHERE user_id=:user_id "
                            "ORDER BY approved_at,session_id,review_revision,batch_id,"
                            "schema_version"
                        ),
                        {"user_id": user_id},
                    )
                )
                .mappings()
                .all()
            )
        return tuple(self._batch_from_row(row) for row in rows)

    async def list_accepted_metric_inputs(
        self, user_id: UUID
    ) -> tuple[AcceptedMetricEvidenceInput, ...]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT b.*, i.schema_version AS inbox_schema_version,"
                            "i.receipt_id AS inbox_receipt_id,"
                            "i.user_id AS inbox_user_id,i.session_id AS inbox_session_id,"
                            "i.report_id AS inbox_report_id,"
                            "i.review_revision AS inbox_review_revision,"
                            "i.content_sha256 AS inbox_content_sha256,"
                            "i.received_at AS inbox_received_at,i.payload AS inbox_payload, "
                            "i.payload_sha256 AS inbox_digest,"
                            "s.payload AS source_payload,s.payload_sha256 AS source_digest,"
                            "a.authority_schema_version AS authority_schema_column,"
                            "a.approval_kind AS approval_kind_column,"
                            "a.approval_policy_version AS approval_policy_column,"
                            "a.authority_status AS authority_status_column,"
                            "a.review_revision AS authority_revision_column,"
                            "a.payload AS authority_payload,a.payload_sha256 AS authority_digest "
                            "FROM growth_evidence_batches b "
                            "LEFT JOIN growth_evidence_inbox i ON i.batch_id=b.batch_id "
                            "AND i.schema_version=b.schema_version "
                            "LEFT JOIN growth_evidence_sources s ON s.batch_id=b.batch_id "
                            "AND s.schema_version=b.schema_version "
                            "LEFT JOIN growth_metric_authorities a ON a.batch_id=b.batch_id "
                            "AND a.schema_version=b.schema_version "
                            "WHERE b.user_id=:user_id AND b.schema_version='2.0' "
                            "ORDER BY b.approved_at,b.session_id,b.review_revision,b.batch_id"
                        ),
                        {"user_id": user_id},
                    )
                )
                .mappings()
                .all()
            )
        admitted: list[AcceptedMetricEvidenceInput] = []
        for row in rows:
            try:
                batch = self._batch_from_row(row)
                if not isinstance(batch, ApprovedEvidenceBatchV2):
                    raise ValueError("trusted admission requires v2")
                payloads = (
                    (row["inbox_payload"], row["inbox_digest"]),
                    (row["source_payload"], row["source_digest"]),
                    (row["authority_payload"], row["authority_digest"]),
                )
                if any(
                    value is None or canonical_sha256(value) != digest for value, digest in payloads
                ):
                    raise ValueError("trusted authority digest mismatch")
                receipt = strict_json_model(EvidenceInboxReceipt, row["inbox_payload"])
                source = strict_json_model(EvidenceSourceSummary, row["source_payload"])
                authority_envelope = strict_json_model(
                    _MetricAuthorityEnvelope, row["authority_payload"]
                )
                authority = authority_envelope.authority
                expected_receipt, expected_source, expected_admission = _expected_models(
                    batch, receipt_id=receipt.receipt_id, received_at=receipt.received_at
                )
                expected_authority = ApprovedMetricEvidenceAuthorityRecord(
                    approval_kind=batch.approval_kind,
                    approval_policy_version=batch.approval_policy_version,
                    metric_shots=_canonical_metric_shots(batch.metric_shots),
                )
                if (
                    receipt != expected_receipt
                    or row["inbox_schema_version"] != batch.schema_version
                    or row["inbox_receipt_id"] != receipt.receipt_id
                    or row["inbox_user_id"] != receipt.user_id
                    or row["inbox_session_id"] != receipt.session_id
                    or row["inbox_report_id"] != receipt.report_id
                    or row["inbox_review_revision"] != receipt.review_revision
                    or row["inbox_content_sha256"] != receipt.content_sha256
                    or row["inbox_received_at"] != receipt.received_at
                    or source != expected_source
                    or authority != expected_authority
                    or authority_envelope.schema_version != row["authority_schema_column"]
                    or authority_envelope.approval_kind != row["approval_kind_column"]
                    or authority_envelope.approval_policy_version != row["approval_policy_column"]
                    or authority_envelope.status != row["authority_status_column"]
                    or authority_envelope.review_revision != row["authority_revision_column"]
                    or authority_envelope.review_revision != batch.review_revision
                    or authority_envelope.approval_kind != batch.approval_kind.value
                    or authority_envelope.approval_policy_version != batch.approval_policy_version
                    or expected_admission is None
                ):
                    raise ValueError("trusted authority graph differs from batch")
                admitted.append(_seal_accepted_metric_evidence_input(expected_admission))
            except (AttributeError, RecursionError, TypeError, ValueError) as exc:
                raise GrowthPersistenceIntegrityError(
                    "persisted evidence authority graph is invalid"
                ) from exc
        return tuple(admitted)
