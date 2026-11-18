"""Story 8.5 RED tests for exactly-once PlanOutcome history."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.growth.outcome_repository import (
    InMemoryPlanOutcomeRepository,
    OutcomeIngestError,
    OutcomeIngestErrorCode,
    OutcomeIngestStatus,
)
from tests.unit.cloud.growth.outcome_test_support import NOW, make_event, make_outcome, uid


def _append(repository, outcome, event, *, history_number=1):
    ids = iter((uid(70_000 + history_number), uid(80_000 + history_number)))
    return repository.append(
        outcome,
        source_event=event,
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )


def test_exact_replay_records_one_history_and_does_not_call_factories() -> None:
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    event = make_event(outcome)

    created = _append(repository, outcome, event)
    replayed = repository.append(
        outcome,
        source_event=event,
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )

    assert created.status is OutcomeIngestStatus.CREATED
    assert replayed.status is OutcomeIngestStatus.REPLAYED
    assert replayed.record == created.record
    assert repository.list_for_user(outcome.user_id) == (created.record,)
    assert repository.list_receipts() == (created.receipt,)


def test_out_of_order_arrival_is_retained_once_and_read_in_execution_order() -> None:
    repository = InMemoryPlanOutcomeRepository()
    earlier = make_outcome(1, started_at=NOW - timedelta(days=2))
    later = make_outcome(2, started_at=NOW - timedelta(days=1))

    later_result = _append(repository, later, make_event(later), history_number=2)
    earlier_result = _append(repository, earlier, make_event(earlier), history_number=1)

    assert repository.list_for_user(earlier.user_id) == (
        earlier_result.record,
        later_result.record,
    )


def test_reentrant_uuid_callback_cannot_create_duplicate_business_history() -> None:
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    event = make_event(outcome)
    inner_result = None
    calls = 0

    def reentrant_uuid():
        nonlocal calls, inner_result
        calls += 1
        if calls == 1:
            inner_ids = iter((uid(71_001), uid(81_001)))
            inner_result = repository.append(
                outcome,
                source_event=event,
                clock=lambda: NOW,
                uuid_factory=lambda: next(inner_ids),
            )
        return uid(72_000 + calls)

    outer_result = repository.append(
        outcome,
        source_event=event,
        clock=lambda: NOW,
        uuid_factory=reentrant_uuid,
    )

    assert inner_result is not None
    assert inner_result.status is OutcomeIngestStatus.CREATED
    assert outer_result.status is OutcomeIngestStatus.REPLAYED
    assert outer_result.record == inner_result.record
    assert repository.list_for_user(outcome.user_id) == (inner_result.record,)


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("key", OutcomeIngestErrorCode.IDEMPOTENCY_CONFLICT),
        ("event", OutcomeIngestErrorCode.EVENT_ID_CONFLICT),
        ("outcome", OutcomeIngestErrorCode.OUTCOME_ID_CONFLICT),
        ("business", OutcomeIngestErrorCode.BUSINESS_IDENTITY_CONFLICT),
    ],
)
def test_reused_identity_with_different_content_fails_closed(kind, expected_code) -> None:
    repository = InMemoryPlanOutcomeRepository()
    first = make_outcome(1)
    first_event = make_event(first)
    _append(repository, first, first_event)
    before = repository.snapshot()

    second = make_outcome(2)
    second_event = make_event(second)
    if kind == "key":
        second_event = second_event.model_copy(
            update={"idempotency_key": first_event.idempotency_key}
        )
    elif kind == "event":
        second_event = second_event.model_copy(update={"event_id": first_event.event_id})
    elif kind == "outcome":
        second = second.model_copy(update={"outcome_id": first.outcome_id})
        second_event = make_event(
            second,
            event_number=999,
            idempotency_key="distinct-key-for-reused-outcome",
        )
    else:
        second = second.model_copy(
            update={"plan_id": first.plan_id, "session_id": first.session_id}
        )
        second_event = make_event(second)

    with pytest.raises(OutcomeIngestError) as raised:
        _append(repository, second, second_event, history_number=2)

    assert raised.value.code is expected_code
    assert repository.snapshot() == before


@pytest.mark.parametrize(
    "stage",
    ("history", "receipt", "idempotency", "event", "outcome", "business"),
)
def test_failure_at_each_staging_point_rolls_back_every_authority_map(stage) -> None:
    def fail(selected: str) -> None:
        if selected == stage:
            raise RuntimeError(stage)

    repository = InMemoryPlanOutcomeRepository(failure_injector=fail)
    before = repository.snapshot()
    outcome = make_outcome(1)

    with pytest.raises(RuntimeError, match=stage):
        _append(repository, outcome, make_event(outcome))

    assert repository.snapshot() == before
    assert repository.list_for_user(outcome.user_id) == ()
    assert repository.list_receipts() == ()


def test_history_and_reads_are_frozen_owner_scoped_and_detached() -> None:
    repository = InMemoryPlanOutcomeRepository()
    first = make_outcome(1, user_number=1)
    second = make_outcome(2, user_number=2)
    first_record = _append(repository, first, make_event(first)).record
    _append(repository, second, make_event(second), history_number=2)

    read = repository.list_for_user(first.user_id)
    assert read == (first_record,)
    assert repository.list_for_user(second.user_id)[0].outcome.user_id == second.user_id
    with pytest.raises(ValidationError):
        read[0].outcome.__setattr__("device_id", "forged")
    copied = deepcopy(read)
    assert copied == read
    assert repository.list_for_user(first.user_id) == (first_record,)
