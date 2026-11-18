from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from sqlalchemy import text

from billiards_ai.cloud.growth.api import install_growth_api
from billiards_ai.cloud.growth.errors import EvidenceIngestError
from billiards_ai.cloud.growth.outcome_repository import (
    OutcomeIngestError,
    OutcomeIngestErrorCode,
)
from billiards_ai.cloud.growth.outcome_service import AsyncPlanOutcomeIngestService
from billiards_ai.cloud.growth.persistence import (
    PostgresGrowthEvidenceRepository,
    PostgresGrowthSnapshotRepository,
    PostgresPlanOutcomeRepository,
)
from billiards_ai.cloud.growth.persistence.errors import GrowthPersistenceIntegrityError
from billiards_ai.cloud.growth.persistence.outbox import PostgresGrowthOutboxRepository
from billiards_ai.cloud.growth.read_service import UserGrowthReadService
from billiards_ai.cloud.growth.service import AsyncEvidenceIngestService
from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot
from billiards_ai.cloud.growth.snapshot_repository import SnapshotIntegrityError
from billiards_ai.cloud.growth.snapshot_service import AsyncGrowthSnapshotService
from billiards_ai.cloud.review.persistence.privacy import assert_safe_persistence_payload
from billiards_ai.shared.models.growth import GrowthSnapshotLineage
from billiards_ai.shared.models.metric_evidence import ApprovedMetricEvidenceAuthorityRecord
from tests.unit.cloud.growth.outcome_test_support import make_event, make_outcome, uid
from tests.unit.cloud.growth.snapshot_test_support import NOW, admit, make_batch
from tests.unit.cloud.growth.test_evidence_ingest import (
    USER_ID,
    Authority,
    LineageAuthority,
    MetricAuthority,
    _approved,
    _batch,
    _batch_v2,
)


def _metric_authority(batch):
    return ApprovedMetricEvidenceAuthorityRecord(
        approval_kind=batch.approval_kind,
        approval_policy_version=batch.approval_policy_version,
        metric_shots=tuple(reversed(batch.metric_shots)),
    )


class _Identity:
    def __init__(self, user_id):
        self._user_id = user_id

    def current_user_id(self):
        return self._user_id


@pytest.mark.asyncio
async def test_async_application_services_execute_real_postgres_paths(
    growth_session_factory,
) -> None:
    approved = _approved()
    batch = _batch_v2(approved)
    ids = iter((uid(960_001), uid(960_002)))
    evidence_service = AsyncEvidenceIngestService(
        approved_repository=Authority(approved),
        lineage_repository=LineageAuthority(approved),
        metric_repository=MetricAuthority(tuple(reversed(batch.metric_shots))),
        growth_repository=PostgresGrowthEvidenceRepository(growth_session_factory),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    created = await evidence_service.ingest(batch, authenticated_user_id=USER_ID)
    assert created.status.value == "CREATED"

    snapshots = PostgresGrowthSnapshotRepository(growth_session_factory)
    snapshot_service = AsyncGrowthSnapshotService(
        source=PostgresGrowthEvidenceRepository(growth_session_factory),
        repository=snapshots,
    )
    saved = await snapshot_service.build_and_append(
        USER_ID,
        evidence_cutoff=NOW,
        expected_previous_version=None,
        idempotency_key="application:v2:first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(960_003),
    )
    assert saved.status.value == "CREATED"
    assert await snapshots.get_current(USER_ID) == saved.snapshot
    read_service = UserGrowthReadService(
        identity_provider=_Identity(USER_ID),
        repository=snapshots,
        snapshot_policy_version=saved.snapshot.snapshot_policy_version,
    )
    assert (await read_service.get_current()).snapshot == saved.snapshot
    app = FastAPI()
    install_growth_api(
        app,
        lambda identity: UserGrowthReadService(
            identity_provider=identity,
            repository=snapshots,
            snapshot_policy_version=saved.snapshot.snapshot_policy_version,
        ),
        lambda: _Identity(USER_ID),
    )
    route = next(route for route in app.routes if route.path == "/v1/users/me/growth/current")
    response = await route.endpoint(growth_service=read_service)
    assert response["ok"] is True
    assert response["data"]["snapshot"]["user_id"] == str(USER_ID)

    outcome = make_outcome(60, user_number=960)
    outcome_service = AsyncPlanOutcomeIngestService(
        repository=PostgresPlanOutcomeRepository(growth_session_factory),
        clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
        uuid_factory=iter((uid(960_004), uid(960_005))).__next__,
    )
    result = await outcome_service.ingest(
        outcome,
        source_event=make_event(outcome),
        authenticated_user_id=outcome.user_id,
    )
    assert result.status.value == "CREATED"


@pytest.mark.asyncio
async def test_v1_to_v2_same_revision_and_batch_id_are_schema_namespaced(
    growth_session_factory,
) -> None:
    approved = _approved()
    v1 = _batch(approved)
    v2 = _batch_v2(approved)
    repository = PostgresGrowthEvidenceRepository(growth_session_factory)
    ids = iter((uid(961_001), uid(961_002), uid(961_003), uid(961_004)))
    first = await repository.append_approved_batch(
        v1, clock=lambda: NOW, uuid_factory=lambda: next(ids)
    )
    second = await repository.append_approved_batch(
        v2,
        metric_authority=_metric_authority(v2),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    replay = await repository.append_approved_batch(
        v2,
        metric_authority=None,
        clock=lambda: (_ for _ in ()).throw(AssertionError("replay clock")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("replay uuid")),
    )
    assert (first.status.value, second.status.value, replay.status.value) == (
        "CREATED",
        "CREATED",
        "REPLAYED",
    )
    assert await repository.list_batches(v1.user_id) == (v1, v2)
    async with growth_session_factory() as session:
        authority = (
            (
                await session.execute(
                    text(
                        "SELECT authority_schema_version,authority_status,review_revision,payload "
                        "FROM growth_metric_authorities WHERE schema_version='2.0' "
                        "AND batch_id=:batch_id"
                    ),
                    {"batch_id": v2.batch_id},
                )
            )
            .mappings()
            .one()
        )
    assert authority["authority_schema_version"] == "growth-metric-authority.v1"
    assert authority["authority_status"] == "APPROVED"
    assert authority["review_revision"] == v2.review_revision
    assert authority["payload"]["authority"]["metric_shots"]


@pytest.mark.asyncio
async def test_snapshot_exact_existing_head_race_replays_and_rollback_head_fails_closed(
    growth_session_factory,
) -> None:
    repository = PostgresGrowthSnapshotRepository(growth_session_factory)
    first = project_growth_snapshot(
        admit(make_batch(70)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    await repository.compare_and_append(
        first,
        expected_previous_version=None,
        idempotency_key="head:first",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(962_001),
    )
    second = project_growth_snapshot(
        admit(make_batch(71)),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=2, previous_snapshot_id=first.snapshot_id),
    )

    async def append(offset: int):
        return await PostgresGrowthSnapshotRepository(growth_session_factory).compare_and_append(
            second,
            expected_previous_version=1,
            idempotency_key="head:second",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(962_010 + offset),
        )

    results = await asyncio.gather(append(1), append(2))
    assert sorted(item.status.value for item in results) == ["CREATED", "REPLAYED"]
    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE core_intelligence_outbox SET payload_sha256=:digest "
                "WHERE aggregate_id=:snapshot_id"
            ),
            {"digest": "0" * 64, "snapshot_id": second.snapshot_id},
        )
    with pytest.raises(SnapshotIntegrityError):
        await repository.compare_and_append(
            second,
            expected_previous_version=1,
            idempotency_key="head:second",
            clock=lambda: NOW,
            uuid_factory=lambda: uid(962_099),
        )
    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE growth_snapshot_heads SET version=1,snapshot_id=:snapshot_id "
                "WHERE user_id=:user_id AND snapshot_policy_version=:policy"
            ),
            {
                "snapshot_id": first.snapshot_id,
                "user_id": first.user_id,
                "policy": first.snapshot_policy_version,
            },
        )
    with pytest.raises(SnapshotIntegrityError):
        await repository.get_current(first.user_id)


@pytest.mark.asyncio
async def test_outbox_tamper_and_missing_outcome_receipt_fail_closed(
    growth_session_factory,
) -> None:
    batch = make_batch(72)
    evidence = PostgresGrowthEvidenceRepository(growth_session_factory)
    ids = iter((uid(963_001), uid(963_002)))
    await evidence.append_approved_batch(
        batch,
        metric_authority=_metric_authority(batch),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                "UPDATE core_intelligence_outbox SET payload_sha256=:digest "
                "WHERE aggregate_id=:batch_id"
            ),
            {"digest": "0" * 64, "batch_id": batch.batch_id},
        )
    with pytest.raises(GrowthPersistenceIntegrityError):
        await evidence.append_approved_batch(
            batch,
            metric_authority=None,
            clock=lambda: NOW,
            uuid_factory=lambda: uid(963_003),
        )

    outcome = make_outcome(73, user_number=963)
    event = make_event(outcome)
    outcome_ids = iter((uid(963_004), uid(963_005)))
    outcomes = PostgresPlanOutcomeRepository(growth_session_factory)
    saved = await outcomes.append(
        outcome,
        source_event=event,
        clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
        uuid_factory=lambda: next(outcome_ids),
    )
    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text("DELETE FROM growth_outcome_inbox WHERE history_id=:history_id"),
            {"history_id": saved.record.history_id},
        )
    with pytest.raises(GrowthPersistenceIntegrityError):
        await outcomes.list_for_user(outcome.user_id)


@pytest.mark.asyncio
async def test_generated_outcome_id_conflicts_and_growth_outbox_lifecycle(
    growth_session_factory,
) -> None:
    repository = PostgresPlanOutcomeRepository(growth_session_factory)
    first = make_outcome(74, user_number=964)
    first_ids = iter((uid(964_001), uid(964_002)))
    saved = await repository.append(
        first,
        source_event=make_event(first),
        clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
        uuid_factory=lambda: next(first_ids),
    )
    second = make_outcome(75, user_number=964)
    ids = iter((saved.record.history_id, uid(964_003)))
    with pytest.raises(OutcomeIngestError) as raised:
        await repository.append(
            second,
            source_event=make_event(second),
            clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
            uuid_factory=lambda: next(ids),
        )
    assert raised.value.code is OutcomeIngestErrorCode.HISTORY_ID_CONFLICT
    third = make_outcome(77, user_number=964)
    receipt_collision_ids = iter((uid(964_010), saved.record.receipt_id))
    with pytest.raises(OutcomeIngestError) as receipt_raised:
        await repository.append(
            third,
            source_event=make_event(third),
            clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
            uuid_factory=lambda: next(receipt_collision_ids),
        )
    assert receipt_raised.value.code is OutcomeIngestErrorCode.RECEIPT_ID_CONFLICT

    batch = make_batch(76)
    event_ids = iter((uid(964_004), uid(964_005)))
    await PostgresGrowthEvidenceRepository(growth_session_factory).append_approved_batch(
        batch,
        metric_authority=_metric_authority(batch),
        clock=lambda: NOW,
        uuid_factory=lambda: next(event_ids),
    )
    outbox = PostgresGrowthOutboxRepository(growth_session_factory)
    reservations = await outbox.reserve(
        worker_id="growth-worker",
        now=NOW,
        lease_until=NOW + timedelta(minutes=1),
        limit=10,
    )
    assert reservations and reservations[0].attempt_count == 1
    assert await outbox.fail(
        event_id=reservations[0].event.event_id,
        worker_id="growth-worker",
        claim_token=reservations[0].claim_token,
        failed_at=NOW,
        retry_at=NOW + timedelta(minutes=2),
        error="transient",
    )
    retried = await outbox.reserve(
        worker_id="growth-worker",
        now=NOW + timedelta(minutes=3),
        lease_until=NOW + timedelta(minutes=4),
        limit=10,
    )
    assert retried[0].attempt_count == 2
    assert await outbox.ack(
        event_id=retried[0].event.event_id,
        worker_id="growth-worker",
        claim_token=retried[0].claim_token,
        published_at=NOW + timedelta(minutes=3),
    )


@pytest.mark.asyncio
async def test_full_growth_database_payloads_are_privacy_safe(growth_session_factory) -> None:
    batch = make_batch(78)
    ids = iter((uid(965_001), uid(965_002)))
    evidence = PostgresGrowthEvidenceRepository(growth_session_factory)
    await evidence.append_approved_batch(
        batch,
        metric_authority=_metric_authority(batch),
        clock=lambda: NOW,
        uuid_factory=lambda: next(ids),
    )
    snapshot = project_growth_snapshot(
        await evidence.list_accepted_metric_inputs(batch.user_id),
        evidence_cutoff=NOW,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    await PostgresGrowthSnapshotRepository(growth_session_factory).compare_and_append(
        snapshot,
        expected_previous_version=None,
        idempotency_key="privacy:snapshot",
        clock=lambda: NOW,
        uuid_factory=lambda: uid(965_003),
    )
    outcome = make_outcome(79, user_number=965)
    outcome_ids = iter((uid(965_004), uid(965_005)))
    await PostgresPlanOutcomeRepository(growth_session_factory).append(
        outcome,
        source_event=make_event(outcome),
        clock=lambda: datetime(2026, 8, 16, 8, 1, tzinfo=UTC),
        uuid_factory=lambda: next(outcome_ids),
    )
    async with growth_session_factory() as session:
        tables = (
            "growth_evidence_batches",
            "growth_evidence_inbox",
            "growth_evidence_sources",
            "growth_metric_authorities",
            "growth_snapshots",
            "growth_metrics",
            "growth_plan_outcomes",
            "growth_outcome_inbox",
            "core_intelligence_outbox",
        )
        for table in tables:
            columns = (
                ("outcome_payload", "event_payload")
                if table == "growth_plan_outcomes"
                else ("payload",)
            )
            for column in columns:
                rows = (await session.execute(text(f"SELECT {column} FROM {table}"))).scalars()
                for payload in rows:
                    assert_safe_persistence_payload(payload)


@pytest.mark.parametrize(
    "failed_table",
    (
        "growth_evidence_batches",
        "growth_evidence_inbox",
        "growth_evidence_sources",
        "growth_metric_authorities",
        "core_intelligence_outbox",
    ),
)
@pytest.mark.asyncio
async def test_evidence_each_transaction_stage_rolls_back(
    growth_session_factory, failed_table: str
) -> None:
    allowed = {
        "growth_evidence_batches",
        "growth_evidence_inbox",
        "growth_evidence_sources",
        "growth_metric_authorities",
        "core_intelligence_outbox",
    }
    assert failed_table in allowed
    async with growth_session_factory() as session, session.begin():
        await session.execute(
            text(
                f"ALTER TABLE {failed_table} ADD CONSTRAINT review_injected_failure "
                "CHECK (false) NOT VALID"
            )
        )
    batch = make_batch(80)
    ids = iter((uid(966_001), uid(966_002)))
    with pytest.raises(EvidenceIngestError):
        await PostgresGrowthEvidenceRepository(growth_session_factory).append_approved_batch(
            batch,
            metric_authority=_metric_authority(batch),
            clock=lambda: NOW,
            uuid_factory=lambda: next(ids),
        )
    async with growth_session_factory() as session:
        counts = (
            await session.execute(
                text(
                    "SELECT (SELECT count(*) FROM growth_evidence_batches),"
                    "(SELECT count(*) FROM growth_evidence_inbox),"
                    "(SELECT count(*) FROM growth_evidence_sources),"
                    "(SELECT count(*) FROM growth_metric_authorities),"
                    "(SELECT count(*) FROM core_intelligence_outbox)"
                )
            )
        ).one()
    assert tuple(counts) == (0, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_outcomes_retain_out_of_order_history(growth_session_factory) -> None:
    repository = PostgresPlanOutcomeRepository(growth_session_factory)
    later = make_outcome(81, user_number=967, started_at=NOW - timedelta(days=1))
    earlier = make_outcome(82, user_number=967, started_at=NOW - timedelta(days=2))
    later_ids = iter((uid(967_001), uid(967_002)))
    earlier_ids = iter((uid(967_003), uid(967_004)))
    await repository.append(
        later,
        source_event=make_event(later),
        clock=lambda: NOW,
        uuid_factory=lambda: next(later_ids),
    )
    await repository.append(
        earlier,
        source_event=make_event(earlier),
        clock=lambda: NOW,
        uuid_factory=lambda: next(earlier_ids),
    )
    assert tuple(item.outcome for item in await repository.list_for_user(later.user_id)) == (
        earlier,
        later,
    )
