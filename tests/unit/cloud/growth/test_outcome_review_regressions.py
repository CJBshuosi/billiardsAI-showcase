"""Unique Story 8.5 review-patch regressions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import timedelta
from threading import Barrier, Event

import pytest

from billiards_ai.cloud.growth import outcome_repository as outcome_repository_module
from billiards_ai.cloud.growth.outcome_repository import (
    InMemoryPlanOutcomeRepository,
    OutcomeIngestError,
    OutcomeIngestErrorCode,
    OutcomeIngestStatus,
)
from billiards_ai.cloud.growth.outcome_service import PlanOutcomeIngestService
from billiards_ai.shared.models.events import canonical_sha256
from tests.unit.cloud.growth.outcome_test_support import (
    NOW,
    make_event,
    make_outcome,
    outcome_event_payload,
    uid,
)


def _append(repository, outcome, event, *, ids, clock=lambda: NOW):
    values = iter(ids)
    return repository.append(
        outcome,
        source_event=event,
        clock=clock,
        uuid_factory=lambda: next(values),
    )


def test_event_payload_binds_every_plan_outcome_field_with_independent_digest() -> None:
    outcome = make_outcome(1)
    payload = outcome_event_payload(outcome)

    assert payload["plan_outcome"] == outcome.model_dump(mode="json")
    assert set(payload["plan_outcome"]) == set(type(outcome).model_fields)
    assert payload["authoritative_sha256"] == canonical_sha256(
        outcome.model_dump(mode="json", exclude={"content_sha256"})
    )
    same_opaque_hash = make_outcome(2).model_copy(update={"content_sha256": outcome.content_sha256})
    assert (
        outcome_event_payload(same_opaque_hash)["authoritative_sha256"]
        != payload["authoritative_sha256"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "9.0"),
        ("outcome_id", str(uid(110_001))),
        ("plan_id", str(uid(110_002))),
        ("plan_version", 2),
        ("user_id", str(uid(110_003))),
        ("session_id", str(uid(110_004))),
        ("device_id", "spliced-device"),
        ("started_at", "2026-08-15T00:00:00Z"),
        ("completed_at", "2026-08-15T01:00:00Z"),
        ("status", "ABORTED"),
        ("blocks", []),
        ("evidence_envelope_ids", []),
        ("direct_ability_update_allowed", True),
        ("content_sha256", "f" * 64),
    ],
)
def test_every_embedded_outcome_field_splice_fails_even_with_recomputed_digest(
    field, replacement
) -> None:
    repository = InMemoryPlanOutcomeRepository()
    service = PlanOutcomeIngestService(repository=repository)
    outcome = make_outcome(1)
    payload = deepcopy(outcome_event_payload(outcome))
    payload["plan_outcome"][field] = replacement
    authoritative = dict(payload["plan_outcome"])
    authoritative.pop("content_sha256")
    payload["authoritative_sha256"] = canonical_sha256(authoritative)
    event = make_event(outcome).model_copy(update={"payload": payload})

    with pytest.raises(OutcomeIngestError) as raised:
        service.ingest(
            outcome,
            source_event=event,
            authenticated_user_id=outcome.user_id,
        )

    assert raised.value.code is OutcomeIngestErrorCode.LINEAGE_MISMATCH
    assert repository.snapshot() == ((), (), (), (), (), ())


@pytest.mark.parametrize(
    ("collision", "expected_code"),
    [
        ("history", "HISTORY_ID_CONFLICT"),
        ("receipt", "RECEIPT_ID_CONFLICT"),
    ],
)
def test_generated_record_identity_collision_never_overwrites_authority(
    collision, expected_code
) -> None:
    repository = InMemoryPlanOutcomeRepository()
    first = make_outcome(1)
    first_result = _append(
        repository,
        first,
        make_event(first),
        ids=(uid(120_001), uid(120_002)),
    )
    before = repository.snapshot()
    second = make_outcome(2)
    ids = (
        first_result.record.history_id if collision == "history" else uid(120_003),
        first_result.receipt.receipt_id if collision == "receipt" else uid(120_004),
    )

    with pytest.raises(OutcomeIngestError) as raised:
        _append(repository, second, make_event(second), ids=ids)

    assert raised.value.code.value == expected_code
    assert repository.snapshot() == before
    assert repository.list_for_user(first.user_id) == (first_result.record,)


def test_repository_authority_is_one_frozen_six_table_state_and_one_swap() -> None:
    repository = InMemoryPlanOutcomeRepository()
    before = repository._state
    assert len(before.as_tuple()) == 6
    assert not any(
        hasattr(repository, name)
        for name in (
            "_history",
            "_receipts",
            "_by_idempotency",
            "_by_event_id",
            "_by_outcome_id",
            "_by_business_identity",
        )
    )
    with pytest.raises(FrozenInstanceError):
        before.history = ()

    outcome = make_outcome(1)
    _append(
        repository,
        outcome,
        make_event(outcome),
        ids=(uid(121_001), uid(121_002)),
    )

    assert repository._state is not before
    assert before.as_tuple() == ((), (), (), (), (), ())


class FatalProbe(BaseException):
    pass


@pytest.mark.parametrize(
    "stage",
    ("history", "receipt", "idempotency", "event", "outcome", "business"),
)
def test_baseexception_at_every_stage_keeps_the_same_authority_state(stage) -> None:
    def fail(selected: str) -> None:
        if selected == stage:
            raise FatalProbe(stage)

    repository = InMemoryPlanOutcomeRepository(failure_injector=fail)
    before = repository._state
    outcome = make_outcome(1)

    with pytest.raises(FatalProbe):
        _append(
            repository,
            outcome,
            make_event(outcome),
            ids=(uid(122_001), uid(122_002)),
        )

    assert repository._state is before
    assert repository.snapshot() == ((), (), (), (), (), ())


def test_frozen_clock_skew_policy_accepts_exact_upper_boundary() -> None:
    tolerance = outcome_repository_module.PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE
    assert outcome_repository_module.PLAN_OUTCOME_EVENT_CLOCK_POLICY_VERSION == (
        "plan-outcome-event-clock-v1"
    )
    assert tolerance == timedelta(seconds=2)
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    event = make_event(outcome).model_copy(update={"occurred_at": NOW + tolerance})

    created = _append(
        repository,
        outcome,
        event,
        ids=(uid(123_001), uid(123_002)),
    )

    assert created.status is OutcomeIngestStatus.CREATED


def test_frozen_clock_skew_policy_rejects_each_open_boundary() -> None:
    tolerance = outcome_repository_module.PLAN_OUTCOME_EVENT_CLOCK_SKEW_TOLERANCE
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    too_early = make_event(outcome).model_copy(
        update={"occurred_at": outcome.completed_at - timedelta(microseconds=1)}
    )
    too_late = make_event(outcome).model_copy(
        update={"occurred_at": NOW + tolerance + timedelta(microseconds=1)}
    )

    for event in (too_early, too_late):
        with pytest.raises(OutcomeIngestError) as raised:
            _append(
                repository,
                outcome,
                event,
                ids=(uid(124_001), uid(124_002)),
            )
        assert raised.value.code.value == "CLOCK_POLICY_VIOLATION"
        assert repository.snapshot() == ((), (), (), (), (), ())


def test_barrier_same_identity_has_exactly_one_created_business_effect() -> None:
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    event = make_event(outcome)
    barrier = Barrier(2)

    def worker(number: int):
        return _append(
            repository,
            outcome,
            event,
            ids=(uid(130_000 + number * 2), uid(130_001 + number * 2)),
            clock=lambda: (barrier.wait(), NOW)[1],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(worker, (1, 2)))

    assert sorted(result.status.value for result in results) == ["CREATED", "REPLAYED"]
    assert results[0].record == results[1].record
    assert len(repository.list_for_user(outcome.user_id)) == 1


def test_barrier_different_identities_commit_without_lost_write() -> None:
    repository = InMemoryPlanOutcomeRepository()
    barrier = Barrier(2)

    def worker(number: int):
        outcome = make_outcome(number)
        return _append(
            repository,
            outcome,
            make_event(outcome),
            ids=(uid(131_000 + number * 2), uid(131_001 + number * 2)),
            clock=lambda: (barrier.wait(), NOW)[1],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(worker, (1, 2)))

    assert all(result.status is OutcomeIngestStatus.CREATED for result in results)
    assert repository.list_for_user(make_outcome(1).user_id) == tuple(
        sorted(
            (result.record for result in results),
            key=lambda record: record.outcome.started_at,
        )
    )


def test_callback_competition_interleaving_rechecks_identity_before_commit() -> None:
    repository = InMemoryPlanOutcomeRepository()
    outcome = make_outcome(1)
    event = make_event(outcome)
    first_in_callback = Event()
    release_first = Event()

    def delayed_clock():
        first_in_callback.set()
        assert release_first.wait(timeout=5)
        return NOW

    def delayed_writer():
        return _append(
            repository,
            outcome,
            event,
            ids=(uid(132_001), uid(132_002)),
            clock=delayed_clock,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        delayed = executor.submit(delayed_writer)
        assert first_in_callback.wait(timeout=5)
        winner = executor.submit(
            _append,
            repository,
            outcome,
            event,
            ids=(uid(132_003), uid(132_004)),
        ).result(timeout=5)
        release_first.set()
        replay = delayed.result(timeout=5)

    assert winner.status is OutcomeIngestStatus.CREATED
    assert replay.status is OutcomeIngestStatus.REPLAYED
    assert winner.record == replay.record
    assert repository.list_for_user(outcome.user_id) == (winner.record,)
