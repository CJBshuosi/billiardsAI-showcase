"""PostgreSQL exactly-once PlanOutcome history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from pydantic import UUID4, TypeAdapter
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.cloud.growth.outcome_repository import (
    OutcomeInboxReceipt,
    OutcomeIngestError,
    OutcomeIngestErrorCode,
    OutcomeIngestStatus,
    PlanOutcomeHistoryRecord,
    PlanOutcomeIngestResult,
    _validate_received_clock,
    validate_outcome_event_lineage,
)
from billiards_ai.cloud.review.persistence.privacy import assert_safe_persistence_payload
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.growth import PlanOutcome

from ._common import json_text, strict_json_model
from .errors import GrowthPersistenceIntegrityError

_UUID4 = TypeAdapter(UUID4)


class PostgresPlanOutcomeRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @staticmethod
    def _record_from_row(row) -> PlanOutcomeHistoryRecord:
        try:
            if canonical_sha256(row["outcome_payload"]) != row["outcome_sha256"]:
                raise ValueError("outcome digest mismatch")
            if canonical_sha256(row["event_payload"]) != row["event_sha256"]:
                raise ValueError("event digest mismatch")
            assert_safe_persistence_payload(row["outcome_payload"])
            assert_safe_persistence_payload(row["event_payload"])
            outcome = strict_json_model(PlanOutcome, row["outcome_payload"])
            event = strict_json_model(DomainEvent, row["event_payload"])
            outcome, event = validate_outcome_event_lineage(outcome, event)
            _validate_received_clock(outcome, event, row["received_at"])
            if (
                row["outcome_id"] != outcome.outcome_id
                or row["event_id"] != event.event_id
                or row["user_id"] != outcome.user_id
                or row["plan_id"] != outcome.plan_id
                or row["session_id"] != outcome.session_id
                or row["idempotency_key"] != event.idempotency_key
                or row["started_at"] != outcome.started_at
                or row["completed_at"] != outcome.completed_at
            ):
                raise ValueError("outcome columns and payload differ")
            return PlanOutcomeHistoryRecord(
                history_id=row["history_id"],
                receipt_id=row["receipt_id"],
                outcome=outcome,
                source_event=event,
                received_at=row["received_at"],
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise GrowthPersistenceIntegrityError("invalid persisted PlanOutcome") from exc

    @staticmethod
    async def _receipt(
        session: AsyncSession, record: PlanOutcomeHistoryRecord
    ) -> OutcomeInboxReceipt:
        row = (
            (
                await session.execute(
                    text("SELECT * FROM growth_outcome_inbox WHERE history_id=:history_id"),
                    {"history_id": record.history_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        try:
            if row is None or canonical_sha256(row["payload"]) != row["payload_sha256"]:
                raise ValueError("outcome receipt missing or corrupt")
            receipt = strict_json_model(OutcomeInboxReceipt, row["payload"])
            expected = OutcomeInboxReceipt(
                receipt_id=record.receipt_id,
                history_id=record.history_id,
                event_id=record.source_event.event_id,
                outcome_id=record.outcome.outcome_id,
                user_id=record.outcome.user_id,
                idempotency_key=record.source_event.idempotency_key,
                received_at=record.received_at,
            )
            if receipt != expected:
                raise ValueError("outcome receipt differs from history")
            if (
                row["receipt_id"] != receipt.receipt_id
                or row["history_id"] != receipt.history_id
                or row["event_id"] != receipt.event_id
                or row["outcome_id"] != receipt.outcome_id
                or row["user_id"] != receipt.user_id
                or row["idempotency_key"] != receipt.idempotency_key
                or row["received_at"] != receipt.received_at
            ):
                raise ValueError("outcome receipt columns differ from history")
            assert_safe_persistence_payload(row["payload"])
            return receipt
        except (AttributeError, TypeError, ValueError) as exc:
            raise GrowthPersistenceIntegrityError("invalid persisted outcome receipt") from exc

    async def _resolve(
        self,
        session: AsyncSession,
        outcome: PlanOutcome,
        event: DomainEvent,
    ) -> PlanOutcomeIngestResult | None:
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT * FROM growth_plan_outcomes WHERE idempotency_key=:key "
                        "OR event_id=:event_id OR outcome_id=:outcome_id "
                        "OR (plan_id=:plan_id AND session_id=:session_id) "
                        "ORDER BY history_id"
                    ),
                    {
                        "key": event.idempotency_key,
                        "event_id": event.event_id,
                        "outcome_id": outcome.outcome_id,
                        "plan_id": outcome.plan_id,
                        "session_id": outcome.session_id,
                    },
                )
            )
            .mappings()
            .all()
        )
        by_key = next(
            (row for row in rows if row["idempotency_key"] == event.idempotency_key), None
        )
        if by_key is not None:
            record = self._record_from_row(by_key)
            if record.outcome != outcome or record.source_event != event:
                raise OutcomeIngestError(OutcomeIngestErrorCode.IDEMPOTENCY_CONFLICT)
            return PlanOutcomeIngestResult(
                status=OutcomeIngestStatus.REPLAYED,
                record=record,
                receipt=await self._receipt(session, record),
            )
        if any(row["event_id"] == event.event_id for row in rows):
            raise OutcomeIngestError(OutcomeIngestErrorCode.EVENT_ID_CONFLICT)
        if any(row["outcome_id"] == outcome.outcome_id for row in rows):
            raise OutcomeIngestError(OutcomeIngestErrorCode.OUTCOME_ID_CONFLICT)
        if any(
            row["plan_id"] == outcome.plan_id and row["session_id"] == outcome.session_id
            for row in rows
        ):
            raise OutcomeIngestError(OutcomeIngestErrorCode.BUSINESS_IDENTITY_CONFLICT)
        return None

    @staticmethod
    async def _generated_id_conflict(
        session: AsyncSession, *, history_id: UUID, receipt_id: UUID
    ) -> OutcomeIngestErrorCode | None:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT "
                        "EXISTS(SELECT 1 FROM growth_plan_outcomes WHERE history_id=:history_id) "
                        "AS history_conflict,"
                        "EXISTS(SELECT 1 FROM growth_plan_outcomes WHERE receipt_id=:receipt_id) "
                        "OR EXISTS(SELECT 1 FROM growth_outcome_inbox "
                        "WHERE receipt_id=:receipt_id) "
                        "AS receipt_conflict"
                    ),
                    {"history_id": history_id, "receipt_id": receipt_id},
                )
            )
            .mappings()
            .one()
        )
        if row["history_conflict"]:
            return OutcomeIngestErrorCode.HISTORY_ID_CONFLICT
        if row["receipt_conflict"]:
            return OutcomeIngestErrorCode.RECEIPT_ID_CONFLICT
        return None

    async def append(
        self,
        outcome: PlanOutcome,
        *,
        source_event: DomainEvent,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> PlanOutcomeIngestResult:
        validated_outcome, validated_event = validate_outcome_event_lineage(outcome, source_event)
        async with self._sessions() as session, session.begin():
            if replay := await self._resolve(session, validated_outcome, validated_event):
                return replay
        received_at = _validate_received_clock(validated_outcome, validated_event, clock())
        history_id = _UUID4.validate_python(uuid_factory(), strict=True)
        receipt_id = _UUID4.validate_python(uuid_factory(), strict=True)
        record = PlanOutcomeHistoryRecord(
            history_id=history_id,
            receipt_id=receipt_id,
            outcome=validated_outcome,
            source_event=validated_event,
            received_at=received_at,
        )
        receipt = OutcomeInboxReceipt(
            receipt_id=receipt_id,
            history_id=history_id,
            event_id=validated_event.event_id,
            outcome_id=validated_outcome.outcome_id,
            user_id=validated_outcome.user_id,
            idempotency_key=validated_event.idempotency_key,
            received_at=received_at,
        )
        outcome_payload = validated_outcome.model_dump(mode="json")
        event_payload = validated_event.model_dump(mode="json")
        receipt_payload = receipt.model_dump(mode="json")
        for payload in (outcome_payload, event_payload, receipt_payload):
            assert_safe_persistence_payload(payload)
        try:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(session, validated_outcome, validated_event):
                    return replay
                await session.execute(
                    text(
                        "INSERT INTO growth_plan_outcomes "
                        "(history_id,receipt_id,outcome_id,event_id,user_id,plan_id,session_id,"
                        "idempotency_key,started_at,completed_at,received_at,outcome_payload,"
                        "outcome_sha256,event_payload,event_sha256) VALUES "
                        "(:history_id,:receipt_id,:outcome_id,:event_id,:user_id,:plan_id,"
                        ":session_id,:key,:started_at,:completed_at,:received_at,"
                        "CAST(:outcome AS jsonb),:outcome_digest,"
                        "CAST(:event AS jsonb),:event_digest)"
                    ),
                    {
                        "history_id": history_id,
                        "receipt_id": receipt_id,
                        "outcome_id": validated_outcome.outcome_id,
                        "event_id": validated_event.event_id,
                        "user_id": validated_outcome.user_id,
                        "plan_id": validated_outcome.plan_id,
                        "session_id": validated_outcome.session_id,
                        "key": validated_event.idempotency_key,
                        "started_at": validated_outcome.started_at,
                        "completed_at": validated_outcome.completed_at,
                        "received_at": received_at,
                        "outcome": json_text(outcome_payload),
                        "outcome_digest": canonical_sha256(outcome_payload),
                        "event": json_text(event_payload),
                        "event_digest": canonical_sha256(event_payload),
                    },
                )
                await session.execute(
                    text(
                        "INSERT INTO growth_outcome_inbox "
                        "(receipt_id,history_id,event_id,outcome_id,user_id,idempotency_key,"
                        "received_at,payload,payload_sha256) VALUES "
                        "(:receipt_id,:history_id,:event_id,:outcome_id,:user_id,:key,"
                        ":received_at,CAST(:payload AS jsonb),:digest)"
                    ),
                    {
                        "receipt_id": receipt_id,
                        "history_id": history_id,
                        "event_id": validated_event.event_id,
                        "outcome_id": validated_outcome.outcome_id,
                        "user_id": validated_outcome.user_id,
                        "key": validated_event.idempotency_key,
                        "received_at": received_at,
                        "payload": json_text(receipt_payload),
                        "digest": canonical_sha256(receipt_payload),
                    },
                )
        except IntegrityError:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(session, validated_outcome, validated_event):
                    return replay
                generated_conflict = await self._generated_id_conflict(
                    session, history_id=history_id, receipt_id=receipt_id
                )
                if generated_conflict is not None:
                    raise OutcomeIngestError(generated_conflict) from None
            raise OutcomeIngestError(OutcomeIngestErrorCode.BUSINESS_IDENTITY_CONFLICT) from None
        return PlanOutcomeIngestResult(
            status=OutcomeIngestStatus.CREATED, record=record, receipt=receipt
        )

    async def list_for_user(self, user_id: UUID) -> tuple[PlanOutcomeHistoryRecord, ...]:
        async with self._sessions() as session, session.begin():
            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_plan_outcomes WHERE user_id=:user_id "
                            "ORDER BY started_at,completed_at,plan_id,session_id,outcome_id"
                        ),
                        {"user_id": user_id},
                    )
                )
                .mappings()
                .all()
            )
            records = tuple(self._record_from_row(row) for row in rows)
            for record in records:
                await self._receipt(session, record)
            return records

    async def list_receipts(self) -> tuple[OutcomeInboxReceipt, ...]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        text("SELECT * FROM growth_outcome_inbox ORDER BY received_at,receipt_id")
                    )
                )
                .mappings()
                .all()
            )
        values: list[OutcomeInboxReceipt] = []
        for row in rows:
            if canonical_sha256(row["payload"]) != row["payload_sha256"]:
                raise GrowthPersistenceIntegrityError("outcome receipt digest mismatch")
            receipt = strict_json_model(OutcomeInboxReceipt, row["payload"])
            if (
                row["receipt_id"] != receipt.receipt_id
                or row["history_id"] != receipt.history_id
                or row["event_id"] != receipt.event_id
                or row["outcome_id"] != receipt.outcome_id
                or row["user_id"] != receipt.user_id
                or row["idempotency_key"] != receipt.idempotency_key
                or row["received_at"] != receipt.received_at
            ):
                raise GrowthPersistenceIntegrityError("outcome receipt column drift")
            values.append(receipt)
        return tuple(values)

    async def list_outbox_events(self) -> tuple[()]:
        return ()
