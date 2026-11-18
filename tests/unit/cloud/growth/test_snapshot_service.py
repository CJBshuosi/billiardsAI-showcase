"""Story 8.4 RED tests for trusted orchestration and fenced historical reads."""

from __future__ import annotations

from copy import deepcopy

import pytest

from billiards_ai.cloud.growth.snapshot_repository import (
    InMemoryGrowthSnapshotRepository,
    SnapshotNotFound,
    SnapshotVersionConflict,
)
from billiards_ai.cloud.growth.snapshot_service import (
    GrowthSnapshotService,
    SnapshotReadFenceMismatch,
    SnapshotSourceError,
)
from billiards_ai.shared.models.growth import GrowthSnapshotReadFence
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch, uid


class RecordingSource:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def list_accepted_metric_inputs(self, user_id):
        self.calls.append(user_id)
        return self.values


def _service(values):
    source = RecordingSource(values)
    repository = InMemoryGrowthSnapshotRepository()
    return GrowthSnapshotService(source=source, repository=repository), source, repository


def test_build_reads_one_consistent_trusted_bundle_and_appends_first_version() -> None:
    admitted = admit(make_batch(2), make_batch(1))
    service, source, repository = _service(admitted)

    result = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="build-1",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(750_001),
    )

    assert source.calls == [uid(900)]
    assert result.snapshot.version == 1
    assert result.snapshot.previous_snapshot_id is None
    assert tuple(item.batch_id for item in result.snapshot.source_batches) == (
        make_batch(1).batch_id,
        make_batch(2).batch_id,
    )
    assert repository.get_current(uid(900)) == result.snapshot


def test_each_new_content_lineage_uses_the_exact_predecessor() -> None:
    values = admit(make_batch(1))
    service, source, repository = _service(values)
    first = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(751_001),
    )
    second = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=1,
        idempotency_key="same-evidence",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(751_002),
    )

    assert source.calls == [uid(900), uid(900)]
    assert second.status.value == "CREATED"
    assert second.snapshot.version == 2
    assert second.snapshot.previous_snapshot_id == first.snapshot.snapshot_id
    assert len(repository.list_outbox_events()) == 2

    source.values = admit(make_batch(1), make_batch(2))
    third = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=2,
        idempotency_key="third",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(751_003),
    )
    assert third.snapshot.version == 3
    assert third.snapshot.previous_snapshot_id == second.snapshot.snapshot_id


def test_service_rejects_empty_forged_cross_user_and_stale_source_reads() -> None:
    empty, _, _ = _service(())
    with pytest.raises(SnapshotSourceError):
        empty.build_and_append(
            uid(900),
            evidence_cutoff=NOW,
            expected_previous_version=None,
            idempotency_key="empty",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(752_001),
        )

    cross_user, _, repository = _service(admit(make_batch(1, user_number=901)))
    with pytest.raises(SnapshotSourceError):
        cross_user.build_and_append(
            uid(900),
            evidence_cutoff=NOW,
            expected_previous_version=None,
            idempotency_key="cross-user",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(752_002),
        )
    assert repository.snapshot() == ((), (), (), (), ())

    forged_values = admit(make_batch(1))
    forged_values = (deepcopy(forged_values[0]),)
    forged, _, _ = _service(forged_values)
    with pytest.raises(SnapshotSourceError):
        forged.build_and_append(
            uid(900),
            evidence_cutoff=NOW,
            expected_previous_version=None,
            idempotency_key="forged",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(752_003),
        )

    valid, _, _ = _service(admit(make_batch(1)))
    with pytest.raises(SnapshotVersionConflict):
        valid.build_and_append(
            uid(900),
            evidence_cutoff=NOW,
            expected_previous_version=1,
            idempotency_key="stale",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(752_004),
        )


def test_read_version_requires_exact_user_version_and_read_fence() -> None:
    service, _, _ = _service(admit(make_batch(1)))
    created = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="readable",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(753_001),
    ).snapshot

    assert service.read_version(uid(900), 1, expected_fence=created.read_fence) == created
    bad_fence = GrowthSnapshotReadFence.model_construct(
        **created.read_fence.model_dump(mode="python") | {"content_sha256": "f" * 64}
    )
    with pytest.raises(SnapshotReadFenceMismatch):
        service.read_version(uid(900), 1, expected_fence=bad_fence)
    with pytest.raises(SnapshotReadFenceMismatch):
        service.read_version(uid(900), 1, expected_fence=None)
    with pytest.raises(SnapshotNotFound):
        service.read_version(uid(900), 2, expected_fence=created.read_fence)
    with pytest.raises(SnapshotNotFound):
        service.read_version(uid(901), 1, expected_fence=created.read_fence)


def test_v2_consumer_seam_preserves_authoritative_snapshot_without_v1_proxy() -> None:
    from billiards_ai.cloud.growth.snapshot_consumer import (
        SnapshotConsumerCompatibilityError,
        prepare_growth_snapshot_for_consumer,
    )

    service, _, _ = _service(admit(make_batch(1)))
    created = service.build_and_append(
        uid(900),
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="consumer-v2",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(754_001),
    ).snapshot
    read = service.read_version(uid(900), 1, expected_fence=created.read_fence)

    adapted = prepare_growth_snapshot_for_consumer(
        read.model_dump_json(), expected_fence=created.read_fence
    )

    assert adapted.schema_version == "2.0"
    assert adapted == created
    assert adapted.model_dump_json() == created.model_dump_json()
    with pytest.raises(SnapshotConsumerCompatibilityError):
        prepare_growth_snapshot_for_consumer(
            read,
            expected_fence=created.read_fence.model_copy(update={"content_sha256": "f" * 64}),
        )
