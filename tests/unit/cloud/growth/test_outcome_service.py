"""Story 8.5 RED tests for PlanOutcome ingestion authority and isolation."""

from __future__ import annotations

from datetime import timedelta

import pytest

from billiards_ai.cloud.growth.outcome_repository import (
    InMemoryPlanOutcomeRepository,
    OutcomeIngestError,
    OutcomeIngestErrorCode,
)
from billiards_ai.cloud.growth.outcome_service import PlanOutcomeIngestService
from billiards_ai.cloud.growth.snapshot_repository import InMemoryGrowthSnapshotRepository
from tests.unit.cloud.growth.outcome_test_support import (
    NOW,
    make_event,
    make_outcome,
    outcome_event_payload,
    uid,
)


def _service(repository=None):
    ids = iter((uid(90_001), uid(90_002), uid(90_003), uid(90_004)))
    repository = repository or InMemoryPlanOutcomeRepository()
    return (
        PlanOutcomeIngestService(
            repository=repository,
            clock=lambda: NOW,
            uuid_factory=lambda: next(ids),
        ),
        repository,
    )


def test_service_accepts_exact_lineage_and_returns_history_only() -> None:
    service, repository = _service()
    outcome = make_outcome(1)

    result = service.ingest(
        outcome,
        source_event=make_event(outcome),
        authenticated_user_id=outcome.user_id,
    )

    assert result.record.outcome == outcome
    assert result.record.source_event.correlation_id == outcome.session_id
    assert repository.list_for_user(outcome.user_id) == (result.record,)
    assert set(type(result).model_fields) == {"status", "record", "receipt"}


def test_cross_user_fails_closed_without_disclosing_or_writing_history() -> None:
    service, repository = _service()
    outcome = make_outcome(1, user_number=1)

    with pytest.raises(OutcomeIngestError) as raised:
        service.ingest(
            outcome,
            source_event=make_event(outcome),
            authenticated_user_id=uid(2),
        )

    assert raised.value.code is OutcomeIngestErrorCode.RESOURCE_NOT_OWNED
    assert repository.snapshot() == ((), (), (), (), (), ())


def test_cross_user_reuse_of_existing_plan_session_identity_fails_closed() -> None:
    service, repository = _service()
    first = make_outcome(1, user_number=1)
    service.ingest(
        first,
        source_event=make_event(first),
        authenticated_user_id=first.user_id,
    )
    before = repository.snapshot()
    other_user = make_outcome(
        2,
        user_number=2,
        plan_number=int(str(first.plan_id)[-6:]),
        session_number=int(str(first.session_id)[-6:]),
    )

    with pytest.raises(OutcomeIngestError) as raised:
        service.ingest(
            other_user,
            source_event=make_event(other_user),
            authenticated_user_id=other_user.user_id,
        )

    assert raised.value.code is OutcomeIngestErrorCode.BUSINESS_IDENTITY_CONFLICT
    assert repository.snapshot() == before


@pytest.mark.parametrize(
    ("update", "expected_code"),
    [
        ({"event_type": "coach.runtime.shot.completed.v1"}, "INVALID_EVENT"),
        ({"aggregate_type": "runtime_session"}, "LINEAGE_MISMATCH"),
        ({"aggregate_id": uid(99_001)}, "LINEAGE_MISMATCH"),
        ({"aggregate_version": 2}, "LINEAGE_MISMATCH"),
        ({"correlation_id": uid(99_002)}, "LINEAGE_MISMATCH"),
        ({"causation_id": uid(99_003)}, "LINEAGE_MISMATCH"),
        ({"occurred_at": NOW - timedelta(days=30)}, "CLOCK_POLICY_VIOLATION"),
        ({"payload": {"outcome_id": str(uid(99_004))}}, "LINEAGE_MISMATCH"),
    ],
)
def test_event_lineage_or_payload_splice_fails_closed(update, expected_code) -> None:
    service, repository = _service()
    outcome = make_outcome(1)
    event = make_event(outcome).model_copy(update=update)

    with pytest.raises(OutcomeIngestError) as raised:
        service.ingest(
            outcome,
            source_event=event,
            authenticated_user_id=outcome.user_id,
        )

    assert raised.value.code.value == expected_code
    assert repository.snapshot() == ((), (), (), (), (), ())


def test_constructed_or_nested_tampered_values_are_revalidated_fail_closed() -> None:
    service, repository = _service()
    outcome = make_outcome(1)
    forged_outcome = outcome.model_copy(
        update={"blocks": (outcome.blocks[0].model_copy(update={"completed_shots": 11}),)}
    )
    event = make_event(outcome)

    with pytest.raises(OutcomeIngestError) as raised:
        service.ingest(
            forged_outcome,
            source_event=event,
            authenticated_user_id=outcome.user_id,
        )

    assert raised.value.code is OutcomeIngestErrorCode.INVALID_OUTCOME
    assert repository.snapshot() == ((), (), (), (), (), ())


def test_plan_outcome_only_cannot_change_any_growth_snapshot_or_emit_growth_event() -> None:
    snapshot_repository = InMemoryGrowthSnapshotRepository()
    snapshot_before = snapshot_repository.snapshot()
    service, outcome_repository = _service()
    outcome = make_outcome(1, completed_shots=10)

    result = service.ingest(
        outcome,
        source_event=make_event(outcome),
        authenticated_user_id=outcome.user_id,
    )

    assert snapshot_repository.snapshot() == snapshot_before
    assert snapshot_repository.list_outbox_events() == ()
    assert outcome_repository.list_outbox_events() == ()
    assert result.record.outcome.direct_ability_update_allowed is False
    assert "snapshot" not in result.model_dump(mode="python")
    assert "metric" not in result.model_dump(mode="python")
    assert "trend" not in result.model_dump(mode="python")


def test_exact_event_payload_is_closed_over_the_existing_outcome_contract() -> None:
    service, _ = _service()
    outcome = make_outcome(1)
    event = make_event(outcome)
    assert event.payload == outcome_event_payload(outcome)

    result = service.ingest(
        outcome,
        source_event=event,
        authenticated_user_id=outcome.user_id,
    )

    assert result.record.outcome.content_sha256 == outcome.content_sha256
