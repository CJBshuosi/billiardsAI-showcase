from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from billiards_ai.cloud.review.persistence import (
    PersistenceIntegrityError,
    PostgresOutboxRepository,
)
from billiards_ai.shared.models.events import DomainEvent

NOW = datetime(2026, 8, 10, 10, tzinfo=UTC)


def make_event() -> DomainEvent:
    return DomainEvent(
        event_id=uuid4(),
        event_type="review.session.rejected.v1",
        aggregate_type="review_session",
        aggregate_id=uuid4(),
        aggregate_version=1,
        occurred_at=NOW,
        correlation_id=uuid4(),
        idempotency_key=f"reject:{uuid4()}",
        payload={"session_id": str(uuid4()), "reason": "occluded"},
    )


async def test_skip_locked_reservation_has_one_owner_and_expired_lease_is_reclaimable(
    database_session_factory,
):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    barrier = asyncio.Event()

    async def reserve(worker: str):
        await barrier.wait()
        return await outbox.reserve(
            worker_id=worker,
            now=NOW,
            lease_until=NOW + timedelta(seconds=30),
            limit=1,
        )

    attempts = [asyncio.create_task(reserve("worker-a")), asyncio.create_task(reserve("worker-b"))]
    barrier.set()
    reservations = await asyncio.gather(*attempts)

    assert sum(len(items) for items in reservations) == 1
    first = next(items[0] for items in reservations if items)
    assert first.event == event
    assert first.attempt_count == 1

    reclaimed = await outbox.reserve(
        worker_id="worker-c",
        now=NOW + timedelta(seconds=31),
        lease_until=NOW + timedelta(seconds=61),
        limit=1,
    )
    assert len(reclaimed) == 1
    assert reclaimed[0].event == event
    assert reclaimed[0].attempt_count == 2


async def test_ack_is_owner_cas_and_idempotent(database_session_factory):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    [reservation] = await outbox.reserve(
        worker_id="worker-a",
        now=NOW,
        lease_until=NOW + timedelta(seconds=30),
        limit=1,
    )

    assert not await outbox.ack(
        event_id=reservation.event.event_id,
        worker_id="foreign-worker",
        claim_token=reservation.claim_token,
        published_at=NOW + timedelta(seconds=1),
    )
    assert await outbox.ack(
        event_id=reservation.event.event_id,
        worker_id="worker-a",
        claim_token=reservation.claim_token,
        published_at=NOW + timedelta(seconds=1),
    )
    assert await outbox.ack(
        event_id=reservation.event.event_id,
        worker_id="worker-a",
        claim_token=reservation.claim_token,
        published_at=NOW + timedelta(seconds=2),
    )
    async with database_session_factory() as session:
        published_at = (
            await session.execute(
                text("SELECT published_at FROM quality_gate_outbox WHERE event_id = :event_id"),
                {"event_id": event.event_id},
            )
        ).scalar_one()
    assert published_at == NOW + timedelta(seconds=1)
    assert (
        await outbox.reserve(
            worker_id="worker-b",
            now=NOW + timedelta(minutes=1),
            lease_until=NOW + timedelta(minutes=2),
            limit=1,
        )
        == ()
    )


async def test_tampered_payload_rolls_back_reservation(database_session_factory):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE quality_gate_outbox "
                "SET payload = jsonb_set(payload, '{reason}', '\"wrong_window\"') "
                "WHERE event_id = :event_id"
            ),
            {"event_id": event.event_id},
        )

    with pytest.raises(PersistenceIntegrityError, match="digest"):
        await outbox.reserve(
            worker_id="worker-a",
            now=NOW,
            lease_until=NOW + timedelta(seconds=30),
            limit=1,
        )

    async with database_session_factory() as session:
        claimed_by = (
            await session.execute(
                text("SELECT claimed_by FROM quality_gate_outbox WHERE event_id = :event_id"),
                {"event_id": event.event_id},
            )
        ).scalar_one()
    assert claimed_by is None


async def test_tampered_domain_event_columns_roll_back_reservation(database_session_factory):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    async with database_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE quality_gate_outbox SET aggregate_version = aggregate_version + 1 "
                "WHERE event_id = :event_id"
            ),
            {"event_id": event.event_id},
        )

    with pytest.raises(PersistenceIntegrityError, match="event digest"):
        await outbox.reserve(
            worker_id="worker-a",
            now=NOW,
            lease_until=NOW + timedelta(seconds=30),
            limit=1,
        )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "reviewer_id",
        "raw_observation",
        "media_url",
        "access_token",
        "openid",
        "system_prompt",
        "provider",
    ],
)
async def test_enqueue_rejects_nested_sensitive_public_payloads(
    database_session_factory,
    sensitive_key,
):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event().model_copy(
        update={"payload": {"safe": [{"nested": {sensitive_key: "must-not-persist"}}]}}
    )

    with pytest.raises(ValueError, match="sensitive"):
        await outbox.enqueue(event, available_at=NOW)


async def test_stale_same_worker_token_cannot_ack_or_fail_reclaimed_event(
    database_session_factory,
):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    [first] = await outbox.reserve(
        worker_id="worker-a",
        now=NOW,
        lease_until=NOW + timedelta(seconds=30),
        limit=1,
    )
    [second] = await outbox.reserve(
        worker_id="worker-a",
        now=NOW + timedelta(seconds=31),
        lease_until=NOW + timedelta(seconds=61),
        limit=1,
    )

    assert first.claim_token != second.claim_token
    assert not await outbox.ack(
        event_id=event.event_id,
        worker_id="worker-a",
        claim_token=first.claim_token,
        published_at=NOW + timedelta(seconds=32),
    )
    assert not await outbox.fail(
        event_id=event.event_id,
        worker_id="worker-a",
        claim_token=first.claim_token,
        failed_at=NOW + timedelta(seconds=32),
        retry_at=NOW + timedelta(seconds=40),
        error="stale worker result",
    )
    assert await outbox.ack(
        event_id=event.event_id,
        worker_id="worker-a",
        claim_token=second.claim_token,
        published_at=NOW + timedelta(seconds=32),
    )


async def test_failed_delivery_releases_lease_with_backoff(database_session_factory):
    outbox = PostgresOutboxRepository(database_session_factory)
    event = make_event()
    await outbox.enqueue(event, available_at=NOW)
    [reservation] = await outbox.reserve(
        worker_id="worker-a",
        now=NOW,
        lease_until=NOW + timedelta(seconds=30),
        limit=1,
    )

    retry_at = NOW + timedelta(minutes=2)
    assert await outbox.fail(
        event_id=event.event_id,
        worker_id="worker-a",
        claim_token=reservation.claim_token,
        failed_at=NOW + timedelta(seconds=1),
        retry_at=retry_at,
        error="temporary broker failure",
    )
    assert (
        await outbox.reserve(
            worker_id="worker-b",
            now=retry_at - timedelta(seconds=1),
            lease_until=retry_at + timedelta(seconds=30),
            limit=1,
        )
        == ()
    )
    [retried] = await outbox.reserve(
        worker_id="worker-b",
        now=retry_at,
        lease_until=retry_at + timedelta(seconds=30),
        limit=1,
    )
    assert retried.attempt_count == 2
