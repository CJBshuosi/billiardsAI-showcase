"""Stable PlanOutcome/DomainEvent fixtures for Story 8.5 tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.growth import BlockOutcome, PlanOutcome, PlanOutcomeStatus

NOW = datetime(2026, 8, 16, 7, 30, tzinfo=UTC)


def uid(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def make_outcome(
    number: int,
    *,
    user_number: int = 1,
    plan_number: int | None = None,
    session_number: int | None = None,
    started_at: datetime | None = None,
    status: PlanOutcomeStatus = PlanOutcomeStatus.COMPLETED,
    completed_shots: int = 10,
) -> PlanOutcome:
    started = started_at or (NOW - timedelta(hours=number))
    return PlanOutcome(
        outcome_id=uid(10_000 + number),
        plan_id=uid(plan_number or 20_000 + number),
        plan_version=1,
        user_id=uid(user_number),
        session_id=uid(session_number or 30_000 + number),
        device_id=f"table-{number}",
        started_at=started,
        completed_at=started + timedelta(minutes=30),
        status=status,
        blocks=(
            BlockOutcome(
                block_id=uid(40_000 + number),
                attempted_shots=10,
                completed_shots=completed_shots,
            ),
        ),
        evidence_envelope_ids=(uid(50_000 + number),),
        direct_ability_update_allowed=False,
        content_sha256=f"{number:064x}",
    )


def outcome_event_payload(outcome: PlanOutcome) -> dict[str, object]:
    full_outcome = outcome.model_dump(mode="json")
    authoritative_outcome = outcome.model_dump(mode="json", exclude={"content_sha256"})
    return {
        "plan_outcome": full_outcome,
        "authoritative_sha256": canonical_sha256(authoritative_outcome),
    }


def make_event(
    outcome: PlanOutcome,
    *,
    event_number: int | None = None,
    idempotency_key: str | None = None,
) -> DomainEvent:
    number = event_number or int(str(outcome.outcome_id)[-6:])
    return DomainEvent(
        event_id=uid(60_000 + number),
        event_type="coach.runtime.plan.completed.v1",
        aggregate_type="plan_outcome",
        aggregate_id=outcome.outcome_id,
        aggregate_version=outcome.plan_version,
        occurred_at=outcome.completed_at + timedelta(seconds=1),
        causation_id=outcome.plan_id,
        correlation_id=outcome.session_id,
        idempotency_key=idempotency_key or f"plan-outcome:{outcome.outcome_id}",
        payload=outcome_event_payload(outcome),
    )
