from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from billiards_ai.cloud.growth.persistence import (
    PostgresGrowthEvidenceRepository,
    PostgresGrowthSnapshotRepository,
    PostgresPlanOutcomeRepository,
)
from billiards_ai.cloud.growth.persistence.errors import GrowthPersistenceIntegrityError
from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot_v3
from billiards_ai.cloud.growth.snapshot_repository import (
    SnapshotIntegrityError,
    SnapshotVersionConflict,
)
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION_V3,
    GrowthSnapshotLineage,
    GrowthSnapshotLineageV3,
)
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord
from tests.unit.cloud.growth.outcome_test_support import make_event, make_outcome, uid
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch


def _authority(batch):
    return ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=batch.approval_kind,
        approval_policy_version=batch.approval_policy_version,
        metric_shots=batch.metric_shots,
    )


def _snapshot(number: int):
    from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot

    return project_growth_snapshot(
        admit(make_batch(number)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )


@pytest.mark.asyncio
async def test_batch_snapshot_and_outcome_survive_repository_recreation(
    growth_session_factory,
) -> None:
    batch = make_batch(1)
    evidence = PostgresGrowthEvidenceRepository(growth_session_factory)
    ids = iter((uid(910_001), uid(910_002)))
    created = await evidence.append_approved_batch(
        batch,
        metric_authority=_authority(batch),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    assert created.status.value == "CREATED"
    replayed = await evidence.append_approved_batch(
        batch,
        metric_authority=_authority(batch),
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )
    assert replayed.status.value == "REPLAYED"
    assert replayed.receipt == created.receipt

    recreated_evidence = PostgresGrowthEvidenceRepository(growth_session_factory)
    admitted = await recreated_evidence.list_accepted_metric_inputs(batch.user_id)
    assert len(admitted) == 1 and admitted[0].is_trusted_admission

    from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot

    snapshot = project_growth_snapshot(
        admitted,
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    snapshots = PostgresGrowthSnapshotRepository(growth_session_factory)
    saved = await snapshots.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="snapshot:first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(910_003),
    )
    assert saved.status.value == "CREATED"
    replayed_snapshot = await snapshots.compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="snapshot:first",
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )
    assert replayed_snapshot.status.value == "REPLAYED"
    assert replayed_snapshot.snapshot == snapshot
    assert (
        await PostgresGrowthSnapshotRepository(growth_session_factory).get_current(batch.user_id)
        == snapshot
    )

    v3_inputs = admit(
        *tuple(
            make_batch(
                100 + index,
                session_number=2_000 + index,
                approved_at=NOW - timedelta(days=120 if index < 5 else 30),
                successful=index >= 5,
            )
            for index in range(10)
        )
    )
    v3 = project_growth_snapshot_v3(
        v3_inputs,
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineageV3(
            version=1,
            previous_snapshot_id=None,
            snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        ),
    )
    saved_v3 = await snapshots.compare_and_append(
        v3,
        expected_previous_version=None,
        idempotency_key="snapshot:v3:first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(910_006),
    )
    assert saved_v3.status.value == "CREATED"
    assert (
        await snapshots.get_current(
            batch.user_id, snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3
        )
        == v3
    )

    outcome = make_outcome(1, user_number=900)
    event = make_event(outcome)
    outcomes = PostgresPlanOutcomeRepository(growth_session_factory)
    outcome_ids = iter((uid(910_004), uid(910_005)))
    saved_outcome = await outcomes.append(
        outcome,
        source_event=event,
        clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
        uuid_factory=lambda: next(outcome_ids),
    )
    assert saved_outcome.status.value == "CREATED"
    replayed_outcome = await outcomes.append(
        outcome,
        source_event=event,
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay called clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay called uuid")),
    )
    assert replayed_outcome.status.value == "REPLAYED"
    assert replayed_outcome.record == saved_outcome.record
    assert await PostgresPlanOutcomeRepository(growth_session_factory).list_for_user(
        outcome.user_id
    ) == (saved_outcome.record,)


@pytest.mark.asyncio
async def test_no_private_admission_proof_is_persisted(growth_session_factory) -> None:
    batch = make_batch(2)
    repository = PostgresGrowthEvidenceRepository(growth_session_factory)
    ids = iter((uid(920_001), uid(920_002)))
    await repository.append_approved_batch(
        batch,
        metric_authority=_authority(batch),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    async with growth_session_factory() as session:
        rows = (
            (
                await session.execute(
                    text(
                        "SELECT payload::text FROM growth_evidence_batches "
                        "UNION ALL SELECT payload::text FROM growth_evidence_inbox "
                        "UNION ALL SELECT payload::text FROM growth_metric_authorities"
                    )
                )
            )
            .scalars()
            .all()
        )
    persisted = "\n".join(rows).lower()
    assert "private" not in persisted
    assert "seal" not in persisted
    assert "pickle" not in persisted


def test_unsealed_serialized_admission_is_never_a_trusted_source() -> None:
    [admission] = admit(make_batch(3))
    rebuilt = admission.__class__.model_validate(admission.model_dump(mode="python"))
    assert rebuilt.is_trusted_admission is False


@pytest.mark.asyncio
async def test_evidence_concurrent_exact_replay_and_tamper_fail_closed(
    growth_session_factory,
) -> None:
    batch = make_batch(4)

    async def append(offset: int):
        ids = iter((uid(930_000 + offset * 10 + 1), uid(930_000 + offset * 10 + 2)))
        return await PostgresGrowthEvidenceRepository(growth_session_factory).append_approved_batch(
            batch,
            metric_authority=_authority(batch),
            clock=lambda: NOW,
            uuid_factory=lambda: next(ids),
        )

    results = await asyncio.gather(append(1), append(2))
    assert sorted(result.status.value for result in results) == ["CREATED", "REPLAYED"]
    first = await PostgresGrowthEvidenceRepository(
        growth_session_factory
    ).list_accepted_metric_inputs(batch.user_id)
    second = await PostgresGrowthEvidenceRepository(
        growth_session_factory
    ).list_accepted_metric_inputs(batch.user_id)
    assert first[0].is_trusted_admission and second[0].is_trusted_admission
    assert first[0] is not second[0]

    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE growth_evidence_sources SET payload_sha256=:digest WHERE batch_id=:batch_id"
            ),
            {"digest": "0" * 64, "batch_id": batch.batch_id},
        )
    with pytest.raises(GrowthPersistenceIntegrityError):
        await PostgresGrowthEvidenceRepository(growth_session_factory).list_accepted_metric_inputs(
            batch.user_id
        )


@pytest.mark.asyncio
async def test_snapshot_two_connection_cas_and_metric_drift_fail_closed(
    growth_session_factory,
) -> None:
    first = _snapshot(5)
    second = _snapshot(6)

    async def append(snapshot, offset: int):
        return await PostgresGrowthSnapshotRepository(growth_session_factory).compare_and_append(
            snapshot,
            expected_previous_version=None,
            idempotency_key=f"snapshot:race:{offset}",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(940_000 + offset),
        )

    results = await asyncio.gather(append(first, 1), append(second, 2), return_exceptions=True)
    assert (
        sum(
            getattr(result, "status", None).value == "CREATED"
            for result in results
            if not isinstance(result, Exception)
        )
        == 1
    )
    assert sum(isinstance(result, SnapshotVersionConflict) for result in results) == 1
    repository = PostgresGrowthSnapshotRepository(growth_session_factory)
    current = await repository.get_current(first.user_id)
    assert current in (first, second)

    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE growth_metrics SET support_status='FORGED' "
                "WHERE snapshot_id=:snapshot_id AND ordinal=0"
            ),
            {"snapshot_id": current.snapshot_id},
        )
    with pytest.raises(SnapshotIntegrityError):
        await repository.get_current(first.user_id)


@pytest.mark.asyncio
async def test_outcome_two_connection_exactly_once_and_no_growth_outbox(
    growth_session_factory,
) -> None:
    outcome = make_outcome(7, user_number=901)
    event = make_event(outcome)

    async def append(offset: int):
        ids = iter((uid(950_000 + offset * 10 + 1), uid(950_000 + offset * 10 + 2)))
        return await PostgresPlanOutcomeRepository(growth_session_factory).append(
            outcome,
            source_event=event,
            clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
            uuid_factory=lambda: next(ids),
        )

    results = await asyncio.gather(append(1), append(2))
    assert sorted(result.status.value for result in results) == ["CREATED", "REPLAYED"]
    async with growth_session_factory() as session:
        counts = (
            await session.execute(
                text(
                    "SELECT (SELECT count(*) FROM growth_plan_outcomes),"
                    "(SELECT count(*) FROM growth_outcome_inbox),"
                    "(SELECT count(*) FROM core_intelligence_outbox)"
                )
            )
        ).one()
    assert tuple(counts) == (1, 1, 0)
