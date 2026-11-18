"""Transactional outbox storage and lease-based worker reservation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.shared.models.events import DomainEvent, canonical_sha256

from .errors import PersistenceIntegrityError
from .privacy import assert_safe_persistence_payload


@dataclass(frozen=True, slots=True)
class OutboxReservation:
    event: DomainEvent
    attempt_count: int
    claimed_by: str
    claim_until: datetime
    claim_token: UUID


def _event_from_row(row) -> DomainEvent:
    payload = row["payload"]
    if canonical_sha256(payload) != row["payload_sha256"]:
        raise PersistenceIntegrityError("outbox payload digest mismatch")
    try:
        event = DomainEvent(
            schema_version=row["schema_version"],
            event_id=row["event_id"],
            event_type=row["event_type"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            aggregate_version=row["aggregate_version"],
            occurred_at=row["occurred_at"],
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
            idempotency_key=row["idempotency_key"],
            payload=payload,
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise PersistenceIntegrityError("outbox row fails strict DomainEvent validation") from exc
    if canonical_sha256(event.model_dump(mode="json")) != row["event_sha256"]:
        raise PersistenceIntegrityError("outbox event digest mismatch")
    return event


class PostgresOutboxRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def load(self, event_id: UUID) -> DomainEvent:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM quality_gate_outbox WHERE event_id=:event_id"),
                        {"event_id": event_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise PersistenceIntegrityError("outbox event is missing")
        return _event_from_row(row)

    async def enqueue(self, event: DomainEvent, *, available_at: datetime) -> None:
        event = DomainEvent.model_validate(event.model_dump(mode="python"))
        async with self._sessions() as session, session.begin():
            await self.enqueue_in_session(session, event, available_at=available_at)

    @staticmethod
    async def enqueue_in_session(
        session: AsyncSession, event: DomainEvent, *, available_at: datetime
    ) -> None:
        assert_safe_persistence_payload(event.payload)
        event_payload = event.model_dump(mode="json")
        await session.execute(
            text(
                "INSERT INTO quality_gate_outbox "
                "(event_id, schema_version, event_type, aggregate_type, aggregate_id, "
                "aggregate_version, occurred_at, causation_id, correlation_id, idempotency_key, "
                "payload, payload_sha256, event_sha256, available_at, attempt_count) VALUES "
                "(:event_id, :schema_version, :event_type, :aggregate_type, :aggregate_id, "
                ":aggregate_version, :occurred_at, :causation_id, :correlation_id, "
                ":idempotency_key, CAST(:payload AS jsonb), :payload_sha256, :event_sha256, "
                ":available_at, 0)"
            ),
            {
                "event_id": event.event_id,
                "schema_version": event.schema_version,
                "event_type": event.event_type,
                "aggregate_type": event.aggregate_type,
                "aggregate_id": event.aggregate_id,
                "aggregate_version": event.aggregate_version,
                "occurred_at": event.occurred_at,
                "causation_id": event.causation_id,
                "correlation_id": event.correlation_id,
                "idempotency_key": event.idempotency_key,
                "payload": __import__("json").dumps(event.payload, ensure_ascii=False),
                "payload_sha256": canonical_sha256(event.payload),
                "event_sha256": canonical_sha256(event_payload),
                "available_at": available_at,
            },
        )

    async def reserve(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
        limit: int,
    ) -> tuple[OutboxReservation, ...]:
        if not worker_id or limit < 1 or lease_until <= now:
            raise ValueError("outbox reservation requires worker, positive limit, and future lease")
        claim_token = uuid4()
        async with self._sessions() as session, session.begin():
            rows = (
                (
                    await session.execute(
                        text(
                            "WITH candidates AS ("
                            " SELECT event_id FROM quality_gate_outbox"
                            " WHERE published_at IS NULL AND available_at <= :now"
                            " AND (claimed_by IS NULL OR claim_until <= :now)"
                            " ORDER BY available_at, occurred_at, event_id"
                            " FOR UPDATE SKIP LOCKED LIMIT :limit"
                            ") UPDATE quality_gate_outbox AS outbox"
                            " SET claimed_by = :worker, claimed_at = :now,"
                            " claim_until = :lease_until, claim_token = :claim_token,"
                            " attempt_count = outbox.attempt_count + 1"
                            " FROM candidates WHERE outbox.event_id = candidates.event_id"
                            " RETURNING outbox.*"
                        ),
                        {
                            "worker": worker_id,
                            "now": now,
                            "lease_until": lease_until,
                            "claim_token": claim_token,
                            "limit": limit,
                        },
                    )
                )
                .mappings()
                .all()
            )
            reservations = tuple(
                OutboxReservation(
                    event=_event_from_row(row),
                    attempt_count=row["attempt_count"],
                    claimed_by=row["claimed_by"],
                    claim_until=row["claim_until"],
                    claim_token=row["claim_token"],
                )
                for row in rows
            )
        return reservations

    async def ack(
        self,
        *,
        event_id: UUID,
        worker_id: str,
        claim_token: UUID,
        published_at: datetime,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    text(
                        "UPDATE quality_gate_outbox "
                        "SET published_at = COALESCE(published_at, :published_at) "
                        "WHERE event_id = :event_id AND claimed_by = :worker "
                        "AND claim_token = :claim_token "
                        "RETURNING event_id"
                    ),
                    {
                        "event_id": event_id,
                        "worker": worker_id,
                        "claim_token": claim_token,
                        "published_at": published_at,
                    },
                )
            ).one_or_none()
        return row is not None

    async def fail(
        self,
        *,
        event_id: UUID,
        worker_id: str,
        claim_token: UUID,
        failed_at: datetime,
        retry_at: datetime,
        error: str,
    ) -> bool:
        if not error or retry_at <= failed_at:
            raise ValueError("failed delivery requires an error and future retry time")
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    text(
                        "UPDATE quality_gate_outbox SET available_at = :retry_at, "
                        "claimed_by = NULL, claimed_at = NULL, claim_until = NULL, "
                        "claim_token = NULL, "
                        "last_error = :error "
                        "WHERE event_id = :event_id AND claimed_by = :worker "
                        "AND claim_token = :claim_token "
                        "AND published_at IS NULL RETURNING event_id"
                    ),
                    {
                        "event_id": event_id,
                        "worker": worker_id,
                        "claim_token": claim_token,
                        "retry_at": retry_at,
                        "error": error,
                    },
                )
            ).one_or_none()
        return row is not None
