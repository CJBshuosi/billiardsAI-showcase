"""Story 8.4 RED tests for trusted reads, CAS, and atomic outbox."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from billiards_ai.cloud.growth.errors import EvidenceIngestError
from billiards_ai.cloud.growth.repository import InMemoryGrowthEvidenceRepository
from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot
from billiards_ai.cloud.growth.snapshot_repository import (
    InMemoryGrowthSnapshotRepository,
    SnapshotIdempotencyConflict,
    SnapshotIntegrityError,
    SnapshotVersionConflict,
)
from billiards_ai.shared.models.evidence import AcceptedMetricEvidenceInput
from billiards_ai.shared.models.growth import GrowthSnapshotLineage
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid


def _snapshot(number: int, *, version: int = 1, previous=None):
    return project_growth_snapshot(
        admit(make_batch(number)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(
            version=version,
            previous_snapshot_id=previous,
        ),
    )


def test_growth_repository_reissues_a_live_seal_from_trusted_stored_facts() -> None:
    batch = make_batch(1)
    repository = InMemoryGrowthEvidenceRepository()
    ids = iter((uid(700_001), uid(700_002)))
    repository.append_approved_batch(
        batch,
        metric_authority=ApprovedMetricEvidenceAuthorityRecord(
            approval_kind=batch.approval_kind,
            approval_policy_version=batch.approval_policy_version,
            metric_shots=batch.metric_shots,
        ),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )

    first = repository.list_accepted_metric_inputs(batch.user_id)
    second = repository.list_accepted_metric_inputs(batch.user_id)
    assert len(first) == 1
    assert first[0].is_trusted_admission is True
    assert second[0].is_trusted_admission is True
    assert first[0] is not second[0]
    dumped = AcceptedMetricEvidenceInput.model_validate(first[0].model_dump(mode="python"))
    assert dumped.is_trusted_admission is False


@pytest.mark.parametrize(
    ("store_name", "updates"),
    [
        ("_receipts", {"session_id": uid(760_001)}),
        ("_receipts", {"report_id": uid(760_002)}),
        ("_receipts", {"review_revision": 2}),
        ("_source_summaries", {"session_id": uid(760_003)}),
        ("_source_summaries", {"report_id": uid(760_004)}),
        ("_source_summaries", {"report_revision": 2}),
        ("_source_summaries", {"review_revision": 2}),
        ("_source_summaries", {"shot_ids": (uid(760_005),)}),
        ("_source_summaries", {"source_envelope_ids": (uid(760_006),)}),
        ("_metric_authorities", {"approval_policy_version": "forged-policy"}),
    ],
)
def test_trusted_read_rejects_every_tampered_authority_lineage_field(
    store_name: str, updates: dict[str, object]
) -> None:
    batch = make_batch(1)
    repository = InMemoryGrowthEvidenceRepository()
    ids = iter((uid(761_001), uid(761_002)))
    repository.append_approved_batch(
        batch,
        metric_authority=ApprovedMetricEvidenceAuthorityRecord(
            approval_kind=batch.approval_kind,
            approval_policy_version=batch.approval_policy_version,
            metric_shots=batch.metric_shots,
        ),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    store = getattr(repository, store_name)
    key = next(iter(store))
    store[key] = store[key].model_copy(update=updates)
    tampered_state = repository.snapshot()

    with pytest.raises(EvidenceIngestError):
        repository.list_accepted_metric_inputs(batch.user_id)

    assert repository.snapshot() == tampered_state


def test_repository_is_immutable_and_only_exact_content_replays() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    snapshot = _snapshot(1)
    ids = iter((uid(710_001), uid(710_002)))
    created = repository.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="snapshot-1",
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    exact = repository.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="snapshot-1",
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )
    same_evidence_new_lineage = project_growth_snapshot(
        admit(make_batch(1)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(
            version=2,
            previous_snapshot_id=snapshot.snapshot_id,
        ),
    )
    next_version = repository.compare_and_append(
        same_evidence_new_lineage,
        expected_previous_version=1,
        idempotency_key="snapshot-new-content",
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )

    assert created.status.value == "CREATED"
    assert exact.status.value == "REPLAYED"
    assert next_version.status.value == "CREATED"
    assert next_version.snapshot == same_evidence_new_lineage
    assert repository.get_current(snapshot.user_id) == same_evidence_new_lineage
    assert repository.get_version(snapshot.user_id, 1) == snapshot
    assert len(repository.list_outbox_events()) == 2


def test_same_idempotency_with_different_evidence_and_forged_state_fail_closed() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="same-key",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(720_001),
    )
    second = _snapshot(2, version=2, previous=first.snapshot_id)

    with pytest.raises(SnapshotIdempotencyConflict):
        repository.compare_and_append(
            second,
            expected_previous_version=1,
            idempotency_key="same-key",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(720_002),
        )
    with pytest.raises(SnapshotVersionConflict):
        repository.compare_and_append(
            second,
            expected_previous_version=None,
            idempotency_key="stale",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(720_003),
        )
    forged = second.model_copy(update={"previous_snapshot_id": uid(999_999)})
    with pytest.raises(SnapshotIntegrityError):
        repository.compare_and_append(
            forged,
            expected_previous_version=1,
            idempotency_key="forged",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(720_004),
        )
    assert repository.get_current(first.user_id) == first
    assert len(repository.list_outbox_events()) == 1


def test_same_idempotency_key_with_same_evidence_but_new_lineage_is_a_conflict() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="lineage-bound-key",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(722_001),
    )
    different_content = _snapshot(1, version=2, previous=first.snapshot_id)

    with pytest.raises(SnapshotIdempotencyConflict):
        repository.compare_and_append(
            different_content,
            expected_previous_version=1,
            idempotency_key="lineage-bound-key",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(722_002),
        )

    assert repository.get_current(first.user_id) == first
    assert len(repository.list_outbox_events()) == 1


def test_same_evidence_with_new_key_cannot_bypass_stale_cas() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(723_001),
    )
    next_lineage = _snapshot(1, version=2, previous=first.snapshot_id)

    with pytest.raises(SnapshotVersionConflict):
        repository.compare_and_append(
            next_lineage,
            expected_previous_version=None,
            idempotency_key="new-key",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(723_002),
        )

    assert repository.get_current(first.user_id) == first


def test_exact_content_with_new_key_replays_only_after_current_cas_matches() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first-key",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(723_101),
    )

    replay = repository.compare_and_append(
        first,
        expected_previous_version=1,
        idempotency_key="second-key",
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )

    assert replay.status.value == "REPLAYED"
    assert replay.snapshot == first
    assert len(repository.list_outbox_events()) == 1


def test_reentrant_uuid_callback_has_one_created_winner_and_preserves_first_commit() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(724_001),
    )
    outer = _snapshot(2, version=2, previous=first.snapshot_id)
    inner = _snapshot(3, version=2, previous=first.snapshot_id)
    state = {"entered": False, "inner_result": None}

    def reentrant_uuid():
        if not state["entered"]:
            state["entered"] = True
            state["inner_result"] = repository.compare_and_append(
                inner,
                expected_previous_version=1,
                idempotency_key="inner",
                clock=lambda: NOW,
                uuid_factory=lambda: uid(724_002),
            )
        return uid(724_003)

    with pytest.raises(SnapshotVersionConflict):
        repository.compare_and_append(
            outer,
            expected_previous_version=1,
            idempotency_key="outer",
            clock=lambda: NOW,
            uuid_factory=reentrant_uuid,
        )

    assert state["inner_result"].status.value == "CREATED"
    assert repository.get_current(first.user_id) == inner
    assert len(repository.list_outbox_events()) == 2


def test_failure_callback_never_observes_partially_published_snapshot_state() -> None:
    observations = []
    repository = None

    def observe(_stage: str) -> None:
        if repository is not None:
            current = repository.get_current(uid(900))
            observations.append(None if current is None else current.version)

    repository = InMemoryGrowthSnapshotRepository(failure_injector=observe)
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(726_001),
    )
    observations.clear()
    second = _snapshot(2, version=2, previous=first.snapshot_id)

    repository.compare_and_append(
        second,
        expected_previous_version=1,
        idempotency_key="second",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(726_002),
    )

    assert observations
    assert set(observations) == {1}
    assert repository.get_current(first.user_id) == second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snapshot_policy_version", "attacker-policy"),
        ("calculation_policy_version", "attacker-calculation"),
        ("taxonomy_version", "attacker-taxonomy"),
        ("user_id", uid(999_998)),
    ],
)
def test_tampered_user_or_policy_cannot_cross_an_authority_partition(
    field: str, value: object
) -> None:
    repository = InMemoryGrowthSnapshotRepository()
    snapshot = _snapshot(1)

    with pytest.raises(SnapshotIntegrityError):
        repository.compare_and_append(
            snapshot.model_copy(update={field: value}),
            expected_previous_version=None,
            idempotency_key="partition-forgery",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(725_001),
        )

    assert repository.snapshot() == ((), (), (), (), ())


def test_concurrent_writers_from_one_version_have_one_winner_and_no_overwrite() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(730_001),
    )
    candidates = (
        _snapshot(2, version=2, previous=first.snapshot_id),
        _snapshot(3, version=2, previous=first.snapshot_id),
    )
    barrier = Barrier(2)

    def append(index: int):
        barrier.wait()
        try:
            return repository.compare_and_append(
                candidates[index],
                expected_previous_version=1,
                idempotency_key=f"writer-{index}",
                clock=lambda: NOW,
                uuid_factory=lambda: uid(730_010 + index),
            )
        except SnapshotVersionConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(append, range(2)))

    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, SnapshotVersionConflict) for item in results) == 1
    assert repository.get_current(first.user_id).version == 2
    assert repository.get_version(first.user_id, 1) == first
    assert len(repository.list_outbox_events()) == 2


@pytest.mark.parametrize(
    "failure_stage",
    ["snapshot", "version_index", "current_index", "content_index", "idempotency", "outbox"],
)
def test_every_atomic_stage_rolls_back_to_existing_authority_state(failure_stage: str) -> None:
    active = {"stage": None}

    def inject(stage: str) -> None:
        if stage == active["stage"]:
            raise RuntimeError(stage)

    repository = InMemoryGrowthSnapshotRepository(failure_injector=inject)
    first = _snapshot(1)
    repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(740_001),
    )
    before = repository.snapshot()
    active["stage"] = failure_stage
    second = _snapshot(2, version=2, previous=first.snapshot_id)

    with pytest.raises(RuntimeError, match=failure_stage):
        repository.compare_and_append(
            second,
            expected_previous_version=1,
            idempotency_key="second",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(740_002),
        )

    assert repository.snapshot() == before
