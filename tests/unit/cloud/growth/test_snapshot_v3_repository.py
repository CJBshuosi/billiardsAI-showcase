"""Story 8.7 repository/service namespace and atomicity tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from billiards_ai.cloud.growth.snapshot_projector import (
    project_growth_snapshot,
    project_growth_snapshot_v3,
)
from billiards_ai.cloud.growth.snapshot_repository import (
    InMemoryGrowthSnapshotRepository,
    SnapshotIdempotencyConflict,
    SnapshotIntegrityError,
)
from billiards_ai.cloud.growth.snapshot_service import GrowthSnapshotV3Service
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    SNAPSHOT_POLICY_VERSION_V3,
    GrowthSnapshotLineage,
    GrowthSnapshotLineageV3,
)
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid


def _inputs():
    batches = tuple(
        make_batch(
            index,
            session_number=1_000 + index,
            approved_at=NOW - timedelta(days=120 if index < 5 else 30),
            successful=index >= 5,
        )
        for index in range(10)
    )
    return admit(*batches)


def _v3(inputs=None):
    return project_growth_snapshot_v3(
        _inputs() if inputs is None else inputs,
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineageV3(
            version=1,
            previous_snapshot_id=None,
            snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        ),
    )


def test_repository_strictly_isolates_v2_and_v3_policy_namespaces() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    v2 = project_growth_snapshot(
        admit(make_batch(99)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    v3 = _v3()
    repository.compare_and_append(
        v2,
        expected_previous_version=None,
        idempotency_key="v2",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(910_001),
    )
    repository.compare_and_append(
        v3,
        expected_previous_version=None,
        idempotency_key="v3",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(910_002),
    )

    assert repository.get_current(v2.user_id) == v2
    assert (
        repository.get_current(v3.user_id, snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3) == v3
    )
    assert (
        repository.get_version(v3.user_id, 1, snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3)
        == v3
    )
    assert repository.get_version(v2.user_id, 1) == v2
    assert repository.get_current(v2.user_id, snapshot_policy_version=SNAPSHOT_POLICY_VERSION) == v2


def test_v3_repository_exact_replay_idempotency_and_rollback() -> None:
    failure = {"stage": None}

    def inject(stage: str) -> None:
        if failure["stage"] == stage:
            raise RuntimeError("injected")

    repository = InMemoryGrowthSnapshotRepository(failure_injector=inject)
    snapshot = _v3()
    created = repository.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="one",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(911_001),
    )
    replay = repository.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="one",
        clock=lambda: (_ for _ in ()).throw(AssertionError()),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError()),
    )
    assert created.status.value == "CREATED"
    assert replay.status.value == "REPLAYED"

    different = _v3(admit(*tuple(reversed(tuple(item.batch for item in _inputs())))))
    different = different.model_copy(update={"content_sha256": "f" * 64})
    with pytest.raises((SnapshotIdempotencyConflict, SnapshotIntegrityError)):
        repository.compare_and_append(
            different,
            expected_previous_version=1,
            idempotency_key="one",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(911_002),
        )

    before = repository.snapshot()
    failure["stage"] = "outbox"
    second = project_growth_snapshot_v3(
        _inputs(),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineageV3(
            version=2,
            previous_snapshot_id=snapshot.snapshot_id,
            snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        ),
    )
    with pytest.raises(RuntimeError, match="injected"):
        repository.compare_and_append(
            second,
            expected_previous_version=1,
            idempotency_key="two",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(911_003),
        )
    assert repository.snapshot() == before


class _Source:
    def __init__(self, values):
        self.values = values

    def list_accepted_metric_inputs(self, _user_id):
        return self.values


def test_v3_service_build_and_exact_fenced_read() -> None:
    repository = InMemoryGrowthSnapshotRepository()
    service = GrowthSnapshotV3Service(source=_Source(_inputs()), repository=repository)
    created = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="service-v3",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(912_001),
    ).snapshot

    assert created.schema_version == "3.0"
    assert service.read_version(uid(900), 1, expected_fence=created.read_fence) == created
