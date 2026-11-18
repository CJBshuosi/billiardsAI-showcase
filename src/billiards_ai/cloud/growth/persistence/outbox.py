"""Lease-based delivery lifecycle for the growth transactional outbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.shared.models.events import DomainEvent

from ._common import event_from_row
from .errors import GrowthPersistenceIntegrityError


@dataclass(frozen=True, slots=True)
class GrowthOutboxReservation:
    event: DomainEvent
    attempt_count: int
    claimed_by: str
    claim_until: datetime
    claim_token: UUID


class PostgresGrowthOutboxRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(self, event_id: UUID) -> DomainEvent:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM core_intelligence_outbox WHERE event_id=:event_id"),
                        {"event_id": event_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise GrowthPersistenceIntegrityError("growth outbox event is missing")
        try:
            return event_from_row(row)
        except (TypeError, ValueError) as exc:
            raise GrowthPersistenceIntegrityError("growth outbox event is invalid") from exc

    async def reserve(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
        limit: int,
    ) -> tuple[GrowthOutboxReservation, ...]:
        if not worker_id or limit < 1 or lease_until <= now:
            raise ValueError("outbox reservation requires worker, positive limit, and future lease")
        token = uuid4()
        async with self._sessions() as session, session.begin():
            rows = (
                (
                    await session.execute(
                        text(
                            "WITH candidates AS (SELECT event_id FROM core_intelligence_outbox "
                            "WHERE published_at IS NULL AND available_at <= :now "
                            "AND (claimed_by IS NULL OR claim_until <= :now) "
                            "ORDER BY available_at,occurred_at,event_id "
                            "FOR UPDATE SKIP LOCKED LIMIT :limit) "
                            "UPDATE core_intelligence_outbox AS outbox SET claimed_by=:worker,"
                            "claimed_at=:now,claim_until=:lease_until,claim_token=:token,"
                            "attempt_count=outbox.attempt_count+1 FROM candidates "
                            "WHERE outbox.event_id=candidates.event_id RETURNING outbox.*"
                        ),
                        {
                            "worker": worker_id,
                            "now": now,
                            "lease_until": lease_until,
                            "token": token,
                            "limit": limit,
                        },
                    )
                )
                .mappings()
                .all()
            )
        values: list[GrowthOutboxReservation] = []
        for row in rows:
            try:
                event = event_from_row(row)
            except (TypeError, ValueError) as exc:
                raise GrowthPersistenceIntegrityError("growth outbox event is invalid") from exc
            values.append(
                GrowthOutboxReservation(
                    event=event,
                    attempt_count=row["attempt_count"],
                    claimed_by=row["claimed_by"],
                    claim_until=row["claim_until"],
                    claim_token=row["claim_token"],
                )
            )
        return tuple(values)

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
                        "UPDATE core_intelligence_outbox SET "
                        "published_at=COALESCE(published_at,:published_at) "
                        "WHERE event_id=:event_id AND claimed_by=:worker "
                        "AND claim_token=:token RETURNING event_id"
                    ),
                    {
                        "event_id": event_id,
                        "worker": worker_id,
                        "token": claim_token,
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
            raise ValueError("failed delivery requires an error and future retry")
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    text(
                        "UPDATE core_intelligence_outbox SET available_at=:retry_at,"
                        "claimed_by=NULL,claimed_at=NULL,claim_until=NULL,claim_token=NULL,"
                        "last_error=:error WHERE event_id=:event_id AND claimed_by=:worker "
                        "AND claim_token=:token AND published_at IS NULL RETURNING event_id"
                    ),
                    {
                        "event_id": event_id,
                        "worker": worker_id,
                        "token": claim_token,
                        "retry_at": retry_at,
                        "error": error,
                    },
                )
            ).one_or_none()
        return row is not None
