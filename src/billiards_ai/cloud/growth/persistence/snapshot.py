"""PostgreSQL CAS storage for immutable V2/V3 growth snapshots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from pydantic import UUID4, TypeAdapter
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.cloud.growth.snapshot_repository import (
    SnapshotAppendResult,
    SnapshotAppendStatus,
    SnapshotIdempotencyConflict,
    SnapshotIntegrityError,
    SnapshotNotFound,
    SnapshotVersionConflict,
)
from billiards_ai.cloud.review.persistence.privacy import assert_safe_persistence_payload
from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
    uuid4_from_canonical_payload,
)

from ._common import event_from_row, insert_outbox, json_text, strict_json_model

_UUID4 = TypeAdapter(UUID4)
Snapshot = GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3


def _strict_snapshot(payload: object) -> Snapshot:
    if not isinstance(payload, dict):
        raise SnapshotIntegrityError("snapshot payload is not an object")
    try:
        if payload.get("schema_version") == "2.0":
            return strict_json_model(GrowthEvidenceSnapshotV2, payload)
        if payload.get("schema_version") == "3.0":
            return strict_json_model(GrowthEvidenceSnapshotV3, payload)
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError("invalid snapshot payload") from exc
    raise SnapshotIntegrityError("unsupported snapshot schema")


def _metric_id(snapshot_id: UUID, ordinal: int, payload: object) -> UUID:
    return uuid4_from_canonical_payload(
        {"snapshot_id": str(snapshot_id), "ordinal": ordinal, "metric": payload}
    )


def _snapshot_event(
    snapshot: Snapshot,
    *,
    event_id: UUID,
    occurred_at: datetime,
    idempotency_key: str,
) -> DomainEvent:
    return DomainEvent(
        event_id=event_id,
        event_type="growth.snapshot.created.v1",
        aggregate_type="growth_evidence_snapshot",
        aggregate_id=snapshot.snapshot_id,
        aggregate_version=snapshot.version,
        occurred_at=occurred_at,
        causation_id=snapshot.previous_snapshot_id,
        correlation_id=snapshot.snapshot_id,
        idempotency_key=idempotency_key,
        payload={
            "snapshot_id": str(snapshot.snapshot_id),
            "user_id": str(snapshot.user_id),
            "version": snapshot.version,
            "schema_version": snapshot.schema_version,
            "snapshot_policy_version": snapshot.snapshot_policy_version,
            "taxonomy_version": snapshot.taxonomy_version,
            "calculation_policy_version": snapshot.calculation_policy_version,
            "evidence_sha256": snapshot.evidence_sha256,
            "content_sha256": snapshot.content_sha256,
            "source_bundle_sha256": snapshot.source_bundle_sha256,
            "read_fence": snapshot.read_fence.model_dump(mode="json"),
        },
    )


class PostgresGrowthSnapshotRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @staticmethod
    async def _load_in_session(session: AsyncSession, snapshot_id: UUID) -> Snapshot:
        row = (
            (
                await session.execute(
                    text("SELECT * FROM growth_snapshots WHERE snapshot_id=:snapshot_id"),
                    {"snapshot_id": snapshot_id},
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise SnapshotNotFound("snapshot version not found")
        try:
            if canonical_sha256(row["payload"]) != row["payload_sha256"]:
                raise ValueError("snapshot row digest mismatch")
            assert_safe_persistence_payload(row["payload"])
            snapshot = _strict_snapshot(row["payload"])
            if (
                snapshot.snapshot_id != row["snapshot_id"]
                or snapshot.user_id != row["user_id"]
                or snapshot.schema_version != row["schema_version"]
                or snapshot.snapshot_policy_version != row["snapshot_policy_version"]
                or snapshot.version != row["version"]
                or row["previous_version"]
                != (None if snapshot.version == 1 else snapshot.version - 1)
                or snapshot.previous_snapshot_id != row["previous_snapshot_id"]
                or snapshot.content_sha256 != row["content_sha256"]
                or snapshot.evidence_sha256 != row["evidence_sha256"]
                or snapshot.source_bundle_sha256 != row["source_bundle_sha256"]
                or snapshot.created_at != row["created_at"]
            ):
                raise ValueError("snapshot columns and payload differ")
            metric_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_metrics WHERE snapshot_id=:snapshot_id "
                            "ORDER BY ordinal"
                        ),
                        {"snapshot_id": snapshot.snapshot_id},
                    )
                )
                .mappings()
                .all()
            )
            metrics = snapshot.metric_projection.metrics
            if len(metric_rows) != len(metrics):
                raise ValueError("normalized metric count differs")
            for ordinal, (metric_row, metric) in enumerate(zip(metric_rows, metrics, strict=True)):
                payload = metric.model_dump(mode="json")
                if (
                    metric_row["ordinal"] != ordinal
                    or metric_row["metric_id"] != _metric_id(snapshot.snapshot_id, ordinal, payload)
                    or metric_row["metric_code"] != metric.metric_code.value
                    or metric_row["support_status"] != metric.support_status.value
                    or metric_row["reliability"] != metric.reliability.value
                    or metric_row["payload"] != payload
                    or metric_row["payload_sha256"] != canonical_sha256(payload)
                ):
                    raise ValueError("normalized metric differs from snapshot")
            return snapshot
        except (AttributeError, TypeError, ValueError) as exc:
            if isinstance(exc, SnapshotIntegrityError):
                raise
            raise SnapshotIntegrityError("persisted snapshot graph is invalid") from exc

    @staticmethod
    async def _event_in_session(session: AsyncSession, snapshot: Snapshot) -> DomainEvent:
        row = (
            (
                await session.execute(
                    text(
                        "SELECT o.*,s.idempotency_key AS snapshot_idempotency_key "
                        "FROM core_intelligence_outbox o JOIN growth_snapshots s "
                        "ON s.snapshot_id=o.aggregate_id WHERE "
                        "o.owner_user_id=:user_id AND o.aggregate_namespace=:policy "
                        "AND o.aggregate_id=:snapshot_id "
                        "AND o.event_type='growth.snapshot.created.v1'"
                    ),
                    {
                        "user_id": snapshot.user_id,
                        "policy": snapshot.snapshot_policy_version,
                        "snapshot_id": snapshot.snapshot_id,
                    },
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise SnapshotIntegrityError("snapshot outbox event is missing")
        try:
            event = event_from_row(row)
            expected = _snapshot_event(
                snapshot,
                event_id=event.event_id,
                occurred_at=event.occurred_at,
                idempotency_key=row["snapshot_idempotency_key"],
            )
            assert_safe_persistence_payload(event.payload)
            if event != expected:
                raise ValueError("snapshot outbox differs from snapshot")
            return event
        except (TypeError, ValueError) as exc:
            raise SnapshotIntegrityError("snapshot outbox event is invalid") from exc

    async def get_current(
        self,
        user_id: UUID,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> Snapshot | None:
        async with self._sessions() as session, session.begin():
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT version,snapshot_id FROM growth_snapshot_heads WHERE "
                            "user_id=:user_id AND snapshot_policy_version=:policy"
                        ),
                        {"user_id": user_id, "policy": snapshot_policy_version},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                count = (
                    await session.execute(
                        text(
                            "SELECT count(*) FROM growth_snapshots WHERE user_id=:user_id "
                            "AND snapshot_policy_version=:policy"
                        ),
                        {"user_id": user_id, "policy": snapshot_policy_version},
                    )
                ).scalar_one()
                if count:
                    raise SnapshotIntegrityError("snapshot namespace has no committed head")
                return None
            bounds = (
                (
                    await session.execute(
                        text(
                            "SELECT count(*) AS count,min(version) AS minimum,"
                            "max(version) AS maximum "
                            "FROM growth_snapshots WHERE user_id=:user_id "
                            "AND snapshot_policy_version=:policy"
                        ),
                        {"user_id": user_id, "policy": snapshot_policy_version},
                    )
                )
                .mappings()
                .one()
            )
            if (
                bounds["minimum"] != 1
                or bounds["count"] != bounds["maximum"]
                or row["version"] != bounds["maximum"]
            ):
                raise SnapshotIntegrityError("snapshot head is not the highest continuous revision")
            snapshot = await self._load_in_session(session, row["snapshot_id"])
            if (
                snapshot.user_id != user_id
                or snapshot.snapshot_policy_version != snapshot_policy_version
                or snapshot.version != row["version"]
            ):
                raise SnapshotIntegrityError("snapshot head points outside its namespace")
            return snapshot

    async def get_version(
        self,
        user_id: UUID,
        version: int,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> Snapshot:
        async with self._sessions() as session, session.begin():
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT snapshot_id FROM growth_snapshots WHERE user_id=:user_id "
                            "AND snapshot_policy_version=:policy AND version=:version"
                        ),
                        {
                            "user_id": user_id,
                            "policy": snapshot_policy_version,
                            "version": version,
                        },
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise SnapshotNotFound("snapshot version not found")
            return await self._load_in_session(session, row["snapshot_id"])

    async def _resolve(
        self,
        session: AsyncSession,
        candidate: Snapshot,
        *,
        expected_previous_version: int | None,
        idempotency_key: str,
        lock_head: bool,
    ) -> SnapshotAppendResult | None:
        async def resolve_receipt() -> SnapshotAppendResult | None:
            receipt = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM growth_snapshot_idempotency "
                            "WHERE user_id=:user_id AND snapshot_policy_version=:policy "
                            "AND idempotency_key=:key"
                        ),
                        {
                            "user_id": candidate.user_id,
                            "policy": candidate.snapshot_policy_version,
                            "key": idempotency_key,
                        },
                    )
                )
                .mappings()
                .one_or_none()
            )
            if receipt is None:
                return None
            if receipt["content_sha256"] != candidate.content_sha256:
                raise SnapshotIdempotencyConflict("idempotency key belongs to other content")
            existing = await self._load_in_session(session, receipt["snapshot_id"])
            if (
                receipt["user_id"] != existing.user_id
                or receipt["snapshot_policy_version"] != existing.snapshot_policy_version
                or receipt["snapshot_id"] != existing.snapshot_id
                or receipt["content_sha256"] != existing.content_sha256
                or existing != candidate
            ):
                raise SnapshotIntegrityError("snapshot receipt authority drift")
            return SnapshotAppendResult(
                status=SnapshotAppendStatus.REPLAYED,
                snapshot=existing,
                event=await self._event_in_session(session, existing),
            )

        if replay := await resolve_receipt():
            return replay
        head_query = (
            "SELECT version,snapshot_id FROM growth_snapshot_heads WHERE "
            "user_id=:user_id AND snapshot_policy_version=:policy FOR UPDATE"
            if lock_head
            else "SELECT version,snapshot_id FROM growth_snapshot_heads WHERE "
            "user_id=:user_id AND snapshot_policy_version=:policy"
        )
        head = (
            (
                await session.execute(
                    text(head_query),
                    {"user_id": candidate.user_id, "policy": candidate.snapshot_policy_version},
                )
            )
            .mappings()
            .one_or_none()
        )
        if lock_head and (replay := await resolve_receipt()):
            return replay
        current_version = None if head is None else head["version"]
        if current_version != expected_previous_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        content = (
            (
                await session.execute(
                    text(
                        "SELECT snapshot_id FROM growth_snapshots WHERE user_id=:user_id "
                        "AND snapshot_policy_version=:policy AND content_sha256=:content"
                    ),
                    {
                        "user_id": candidate.user_id,
                        "policy": candidate.snapshot_policy_version,
                        "content": candidate.content_sha256,
                    },
                )
            )
            .mappings()
            .one_or_none()
        )
        if content is not None:
            existing = await self._load_in_session(session, content["snapshot_id"])
            await session.execute(
                text(
                    "INSERT INTO growth_snapshot_idempotency "
                    "(user_id,snapshot_policy_version,idempotency_key,snapshot_id,content_sha256) "
                    "VALUES (:user_id,:policy,:key,:snapshot_id,:content)"
                ),
                {
                    "user_id": candidate.user_id,
                    "policy": candidate.snapshot_policy_version,
                    "key": idempotency_key,
                    "snapshot_id": existing.snapshot_id,
                    "content": existing.content_sha256,
                },
            )
            return SnapshotAppendResult(
                status=SnapshotAppendStatus.REPLAYED,
                snapshot=existing,
                event=await self._event_in_session(session, existing),
            )
        required_version = 1 if head is None else head["version"] + 1
        predecessor = None if head is None else head["snapshot_id"]
        if candidate.version != required_version or candidate.previous_snapshot_id != predecessor:
            raise SnapshotIntegrityError("candidate lineage is not the next committed revision")
        return None

    async def compare_and_append(
        self,
        snapshot: Snapshot,
        *,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        try:
            candidate = _strict_snapshot(snapshot.model_dump(mode="json"))
        except (AttributeError, RecursionError, TypeError, ValueError) as exc:
            raise SnapshotIntegrityError("invalid snapshot contract") from exc
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise SnapshotIntegrityError("idempotency key must be non-empty")
        try:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(
                    session,
                    candidate,
                    expected_previous_version=expected_previous_version,
                    idempotency_key=idempotency_key,
                    lock_head=False,
                ):
                    return replay
        except IntegrityError:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(
                    session,
                    candidate,
                    expected_previous_version=expected_previous_version,
                    idempotency_key=idempotency_key,
                    lock_head=True,
                ):
                    return replay
            raise SnapshotVersionConflict("snapshot preflight CAS lost") from None
        except SQLAlchemyError as exc:
            raise SnapshotIntegrityError("snapshot preflight persistence failure") from exc
        event = _snapshot_event(
            candidate,
            event_id=_UUID4.validate_python(uuid_factory(), strict=True),
            occurred_at=clock(),
            idempotency_key=idempotency_key,
        )
        payload = candidate.model_dump(mode="json")
        assert_safe_persistence_payload(payload)
        assert_safe_persistence_payload(event.payload)
        try:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(
                    session,
                    candidate,
                    expected_previous_version=expected_previous_version,
                    idempotency_key=idempotency_key,
                    lock_head=True,
                ):
                    return replay
                await session.execute(
                    text(
                        "INSERT INTO growth_snapshots "
                        "(snapshot_id,user_id,schema_version,snapshot_policy_version,version,"
                        "previous_version,previous_snapshot_id,content_sha256,evidence_sha256,"
                        "source_bundle_sha256,idempotency_key,created_at,payload,payload_sha256) "
                        "VALUES (:snapshot_id,:user_id,:schema,:policy,:version,:previous_version,"
                        ":previous,:content,"
                        ":evidence,:source,:key,:created_at,CAST(:payload AS jsonb),"
                        ":payload_digest)"
                    ),
                    {
                        "snapshot_id": candidate.snapshot_id,
                        "user_id": candidate.user_id,
                        "schema": candidate.schema_version,
                        "policy": candidate.snapshot_policy_version,
                        "version": candidate.version,
                        "previous_version": (
                            None if candidate.version == 1 else candidate.version - 1
                        ),
                        "previous": candidate.previous_snapshot_id,
                        "content": candidate.content_sha256,
                        "evidence": candidate.evidence_sha256,
                        "source": candidate.source_bundle_sha256,
                        "key": idempotency_key,
                        "created_at": candidate.created_at,
                        "payload": json_text(payload),
                        "payload_digest": canonical_sha256(payload),
                    },
                )
                for ordinal, metric in enumerate(candidate.metric_projection.metrics):
                    metric_payload = metric.model_dump(mode="json")
                    await session.execute(
                        text(
                            "INSERT INTO growth_metrics "
                            "(metric_id,snapshot_id,ordinal,metric_code,support_status,reliability,"
                            "payload,payload_sha256) VALUES "
                            "(:metric_id,:snapshot_id,:ordinal,:metric_code,:support,:reliability,"
                            "CAST(:payload AS jsonb),:digest)"
                        ),
                        {
                            "metric_id": _metric_id(candidate.snapshot_id, ordinal, metric_payload),
                            "snapshot_id": candidate.snapshot_id,
                            "ordinal": ordinal,
                            "metric_code": metric.metric_code.value,
                            "support": metric.support_status.value,
                            "reliability": metric.reliability.value,
                            "payload": json_text(metric_payload),
                            "digest": canonical_sha256(metric_payload),
                        },
                    )
                if expected_previous_version is None:
                    await session.execute(
                        text(
                            "INSERT INTO growth_snapshot_heads "
                            "(user_id,snapshot_policy_version,version,snapshot_id) "
                            "VALUES (:user_id,:policy,:version,:snapshot_id)"
                        ),
                        {
                            "user_id": candidate.user_id,
                            "policy": candidate.snapshot_policy_version,
                            "version": candidate.version,
                            "snapshot_id": candidate.snapshot_id,
                        },
                    )
                else:
                    updated = await session.execute(
                        text(
                            "UPDATE growth_snapshot_heads SET version=:version,"
                            "snapshot_id=:snapshot_id WHERE user_id=:user_id "
                            "AND snapshot_policy_version=:policy AND version=:expected"
                        ),
                        {
                            "version": candidate.version,
                            "snapshot_id": candidate.snapshot_id,
                            "user_id": candidate.user_id,
                            "policy": candidate.snapshot_policy_version,
                            "expected": expected_previous_version,
                        },
                    )
                    if updated.rowcount != 1:
                        raise SnapshotVersionConflict("expected snapshot version is stale")
                await session.execute(
                    text(
                        "INSERT INTO growth_snapshot_idempotency "
                        "(user_id,snapshot_policy_version,idempotency_key,snapshot_id,content_sha256)"
                        " VALUES (:user_id,:policy,:key,:snapshot_id,:content)"
                    ),
                    {
                        "user_id": candidate.user_id,
                        "policy": candidate.snapshot_policy_version,
                        "key": idempotency_key,
                        "snapshot_id": candidate.snapshot_id,
                        "content": candidate.content_sha256,
                    },
                )
                await insert_outbox(
                    session,
                    event,
                    owner_user_id=candidate.user_id,
                    aggregate_namespace=candidate.snapshot_policy_version,
                )
        except IntegrityError:
            async with self._sessions() as session, session.begin():
                if replay := await self._resolve(
                    session,
                    candidate,
                    expected_previous_version=expected_previous_version,
                    idempotency_key=idempotency_key,
                    lock_head=True,
                ):
                    return replay
            raise SnapshotVersionConflict("snapshot CAS lost") from None
        except SQLAlchemyError as exc:
            raise SnapshotIntegrityError("snapshot persistence failure") from exc
        return SnapshotAppendResult(
            status=SnapshotAppendStatus.CREATED, snapshot=candidate, event=event
        )
