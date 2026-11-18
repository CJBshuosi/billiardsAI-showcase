"""PostgreSQL authority repository for the quality gate."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from pydantic import UUID4, TypeAdapter, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from billiards_ai.shared.models.events import DomainEvent, canonical_sha256
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import (
    ReviewTask,
    ReviewTaskStatus,
    SessionRejection,
    SessionRejectionResult,
    ShotReviewDecision,
    UserNotificationEvent,
    UserNotificationStatus,
)
from billiards_ai.shared.models.shot import Shot

from ..service import ReviewSubmission, apply_shot_reviews
from .errors import (
    IdempotencyConflictError,
    PersistenceIntegrityError,
    PublicationAuthorityError,
    ResourceNotFoundError,
    RevisionConflictError,
)
from .outbox import PostgresOutboxRepository, _event_from_row
from .privacy import assert_safe_persistence_payload
from .publication import ReviewSessionOwnership


@dataclass(frozen=True, slots=True)
class StoredReviewTask:
    task: ReviewTask
    lock_version: int
    claimed_principal_id: str | None
    claimed_credential_id: str | None
    claimed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    result_report_id: UUID | None
    result_report_revision: int | None


@dataclass(frozen=True, slots=True)
class StoredSourceShot:
    shot: Shot
    source_envelope_id: UUID


@dataclass(frozen=True, slots=True)
class StoredReviewTaskPage:
    items: tuple[StoredReviewTask, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class IdempotencyIdentity:
    actor_principal_id: str
    operation: str
    task_id: UUID
    shot_id: UUID | None
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class IdempotencyReceipt:
    identity: IdempotencyIdentity
    request_digest: str
    result_kind: str
    result_identity: str
    result_digest: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredRejectedReviewResult:
    """Strict terminal rejection graph returned to application adapters."""

    task: StoredReviewTask
    result: SessionRejectionResult
    notification_event: DomainEvent
    replayed: bool


ZERO_UUID = UUID("00000000-0000-0000-0000-000000000000")
UUID4_ADAPTER = TypeAdapter(UUID4)


def _as_json(model) -> dict:
    return model.model_dump(mode="json")


def _as_safe_shot_json(shot: Shot) -> dict:
    payload = shot.model_dump(mode="json", exclude={"raw_observation"})
    assert_safe_persistence_payload(payload)
    return payload


def _validate_json_model(model_type, payload: dict):
    try:
        return model_type.model_validate_json(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise PersistenceIntegrityError("stored payload fails strict contract validation") from exc


def _verify_digest(payload: dict, stored_digest: str) -> None:
    if canonical_sha256(payload) != stored_digest:
        raise PersistenceIntegrityError("stored payload digest mismatch")


def _require_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must use UTC")
    return value.astimezone(UTC)


def _require_nonblank(value: str, *, field: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field} must be a non-blank canonical string")
    return value


def _require_uuid4(value: UUID, *, field: str) -> UUID:
    try:
        return UUID4_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"{field} must be UUID4") from exc


def _require_request_digest(value: str, *, operation: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{operation} request digest must be canonical sha256")
    return value


def _task_result_digest(*, task_id: UUID, status: str, lock_version: int) -> str:
    return canonical_sha256(
        {
            "task_id": str(task_id),
            "status": status,
            "lock_version": lock_version,
        }
    )


class PostgresQualityGateRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @staticmethod
    def _task_cursor(*, created_at: datetime, task_id: UUID) -> str:
        payload = json.dumps(
            {"created_at": created_at.isoformat(), "task_id": str(task_id)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def _parse_task_cursor(cursor: str | None) -> tuple[datetime, UUID] | None:
        if cursor is None:
            return None
        try:
            if not cursor or len(cursor) > 512:
                raise ValueError("invalid cursor")
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
            if set(payload) != {"created_at", "task_id"}:
                raise ValueError("invalid cursor fields")
            return (
                _require_utc(datetime.fromisoformat(payload["created_at"]), field="cursor"),
                _require_uuid4(UUID(payload["task_id"]), field="cursor task id"),
            )
        except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("review task cursor is invalid") from exc

    async def list_visible_tasks(
        self,
        *,
        actor_principal_id: str,
        credential_id: str,
        cursor: str | None,
        limit: int,
    ) -> StoredReviewTaskPage:
        """List only unclaimed pending work and the caller's active claims."""

        actor_principal_id = _require_nonblank(actor_principal_id, field="actor_principal_id")
        credential_id = _require_nonblank(credential_id, field="credential_id")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("review task page limit is invalid")
        after = self._parse_task_cursor(cursor)
        params = {
            "actor": actor_principal_id,
            "credential": credential_id,
            "limit": limit + 1,
        }
        if after is not None:
            params.update({"after_created": after[0], "after_id": after[1]})
            query = text(
                "SELECT task_id,created_at FROM review_tasks WHERE "
                "(status='pending_review' OR (status='in_review' AND "
                "claimed_principal_id=:actor AND claimed_credential_id=:credential)) "
                "AND (created_at,task_id) > (:after_created,:after_id) "
                "ORDER BY created_at,task_id LIMIT :limit"
            )
        else:
            query = text(
                "SELECT task_id,created_at FROM review_tasks WHERE "
                "(status='pending_review' OR (status='in_review' AND "
                "claimed_principal_id=:actor AND claimed_credential_id=:credential)) "
                "ORDER BY created_at,task_id LIMIT :limit"
            )
        async with self._sessions() as session:
            rows = (await session.execute(query, params)).mappings().all()
        selected = rows[:limit]
        loaded = []
        for row in selected:
            task = await self.load_task(row["task_id"])
            if task.task.status is ReviewTaskStatus.PENDING_REVIEW or (
                task.task.status is ReviewTaskStatus.IN_REVIEW
                and task.claimed_principal_id == actor_principal_id
                and task.claimed_credential_id == credential_id
            ):
                loaded.append(task)
        items = tuple(loaded)
        if len(rows) <= limit or not selected:
            next_cursor = None
        else:
            last = selected[-1]
            next_cursor = self._task_cursor(created_at=last["created_at"], task_id=last["task_id"])
        return StoredReviewTaskPage(items=items, next_cursor=next_cursor)

    async def bind_session_ownership(
        self, authority: ReviewSessionOwnership
    ) -> ReviewSessionOwnership:
        """Persist trusted user ownership before any finalize command can run."""

        authority = ReviewSessionOwnership.model_validate(
            authority.model_dump(mode="python"), strict=True
        )
        payload = authority.model_dump(mode="json")
        assert_safe_persistence_payload(payload)
        async with self._sessions() as session, session.begin():
            task_row = (
                (
                    await session.execute(
                        text(
                            "SELECT task_id, session_id FROM review_tasks "
                            "WHERE task_id=:task_id FOR UPDATE"
                        ),
                        {"task_id": authority.task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if task_row is None:
                raise ResourceNotFoundError("review task not found")
            if task_row["session_id"] != authority.session_id:
                raise PublicationAuthorityError("ownership session does not match task")
            await session.execute(
                text(
                    "INSERT INTO review_session_ownership "
                    "(task_id, session_id, user_id, bound_at, payload, payload_sha256) VALUES "
                    "(:task_id, :session_id, :user_id, :bound_at, CAST(:payload AS jsonb), "
                    ":digest) ON CONFLICT (task_id) DO NOTHING"
                ),
                {
                    "task_id": authority.task_id,
                    "session_id": authority.session_id,
                    "user_id": authority.user_id,
                    "bound_at": authority.bound_at,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "digest": canonical_sha256(payload),
                },
            )
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM review_session_ownership "
                            "WHERE task_id=:task_id FOR UPDATE"
                        ),
                        {"task_id": authority.task_id},
                    )
                )
                .mappings()
                .one()
            )
            stored = self._ownership_from_row(row)
            if stored != authority:
                raise IdempotencyConflictError(
                    "task ownership was already bound to different authority"
                )
        return authority

    async def load_session_ownership(self, task_id: UUID) -> ReviewSessionOwnership:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_session_ownership WHERE task_id=:task_id"),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise ResourceNotFoundError("session ownership not found")
        return self._ownership_from_row(row)

    @staticmethod
    def _ownership_from_row(row) -> ReviewSessionOwnership:
        payload = row["payload"]
        _verify_digest(payload, row["payload_sha256"])
        authority = _validate_json_model(ReviewSessionOwnership, payload)
        if (
            authority.task_id != row["task_id"]
            or authority.session_id != row["session_id"]
            or authority.user_id != row["user_id"]
            or authority.bound_at != row["bound_at"]
        ):
            raise PersistenceIntegrityError("ownership columns drift from payload")
        return authority

    async def stage_review_draft(
        self,
        *,
        report: ReportDocument,
        task: ReviewTask,
        shots: tuple[Shot, ...],
        source_envelope_ids: Mapping[UUID, UUID],
        created_at: datetime,
    ) -> None:
        report = ReportDocument.model_validate(report.model_dump(mode="python"))
        task = ReviewTask.model_validate(task.model_dump(mode="python"))
        validated_shots = tuple(
            Shot.model_validate(shot.model_dump(mode="python")) for shot in shots
        )
        created_at = _require_utc(created_at, field="created_at")
        source_envelope_ids = {
            shot_id: _require_uuid4(envelope_id, field="source_envelope_id")
            for shot_id, envelope_id in source_envelope_ids.items()
        }
        if task.status is not ReviewTaskStatus.PENDING_REVIEW:
            raise ValueError("new review drafts must start pending")
        if (task.session_id, task.report_id, task.report_revision) != (
            report.session_id,
            report.report_id,
            report.revision,
        ):
            raise ValueError("task must bind the exact report revision")
        if not validated_shots or {shot.id for shot in validated_shots} != set(source_envelope_ids):
            raise ValueError("source envelope mapping must cover the exact shot set")
        if {shot.sequence for shot in validated_shots} != set(range(1, len(validated_shots) + 1)):
            raise ValueError("source shots must form a contiguous sequence")
        if any(shot.session_id != report.session_id for shot in validated_shots):
            raise ValueError("source shots must belong to the report session")

        report_payload = _as_json(report)
        assert_safe_persistence_payload(report_payload)
        async with self._sessions() as session, session.begin():
            await session.execute(
                text(
                    "INSERT INTO review_reports "
                    "(report_id, session_id, revision, parent_report_id, schema_version, status, "
                    "generated_at, payload, payload_sha256) VALUES "
                    "(:report_id, :session_id, :revision, :parent_report_id, :schema_version, "
                    ":status, :generated_at, CAST(:payload AS jsonb), :payload_sha256)"
                ),
                {
                    "report_id": report.report_id,
                    "session_id": report.session_id,
                    "revision": report.revision,
                    "parent_report_id": report.parent_revision_id,
                    "schema_version": report.schema_version,
                    "status": report.report_status.value,
                    "generated_at": report.generated_at,
                    "payload": json.dumps(report_payload, ensure_ascii=False),
                    "payload_sha256": canonical_sha256(report_payload),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO review_tasks "
                    "(task_id, session_id, source_report_id, source_report_revision, status, "
                    "lock_version, created_at, updated_at) VALUES "
                    "(:task_id, :session_id, :report_id, :revision, :status, 0, :now, :now)"
                ),
                {
                    "task_id": task.id,
                    "session_id": task.session_id,
                    "report_id": task.report_id,
                    "revision": task.report_revision,
                    "status": task.status.value,
                    "now": created_at,
                },
            )
            for shot in sorted(validated_shots, key=lambda item: item.sequence):
                safe_shot = Shot.model_validate(
                    {**shot.model_dump(mode="python"), "raw_observation": {}}
                )
                payload = _as_safe_shot_json(safe_shot)
                await session.execute(
                    text(
                        "INSERT INTO review_shot_sources "
                        "(task_id, shot_id, session_id, sequence, source_envelope_id, payload, "
                        "payload_sha256) VALUES "
                        "(:task_id, :shot_id, :session_id, :sequence, :envelope_id, "
                        "CAST(:payload AS jsonb), :payload_sha256)"
                    ),
                    {
                        "task_id": task.id,
                        "shot_id": shot.id,
                        "session_id": shot.session_id,
                        "sequence": shot.sequence,
                        "envelope_id": source_envelope_ids[shot.id],
                        "payload": json.dumps(payload, ensure_ascii=False),
                        "payload_sha256": canonical_sha256(payload),
                    },
                )

    async def load_report(self, report_id: UUID) -> ReportDocument:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_reports WHERE report_id = :report_id"),
                        {"report_id": report_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise ResourceNotFoundError("report not found")
        payload = row["payload"]
        _verify_digest(payload, row["payload_sha256"])
        report = _validate_json_model(ReportDocument, payload)
        if (
            report.report_id != row["report_id"]
            or report.session_id != row["session_id"]
            or report.revision != row["revision"]
            or report.parent_revision_id != row["parent_report_id"]
            or report.schema_version != row["schema_version"]
            or report.report_status.value != row["status"]
            or report.generated_at != row["generated_at"]
        ):
            raise PersistenceIntegrityError("report columns drift from the stored payload")
        return report

    async def load_task(self, task_id: UUID) -> StoredReviewTask:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_tasks WHERE task_id = :task_id"),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise ResourceNotFoundError("review task not found")
        try:
            task = ReviewTask(
                id=row["task_id"],
                session_id=row["session_id"],
                report_id=row["source_report_id"],
                report_revision=row["source_report_revision"],
                status=ReviewTaskStatus(row["status"]),
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("stored review task is invalid") from exc
        report = await self.load_report(task.report_id)
        if (task.session_id, task.report_revision) != (report.session_id, report.revision):
            raise PersistenceIntegrityError("task columns drift from the source report")
        result_report_id = row["result_report_id"]
        result_report_revision = row["result_report_revision"]
        if task.status in (ReviewTaskStatus.APPROVED, ReviewTaskStatus.CORRECTED):
            if result_report_id is None or result_report_revision is None:
                raise PersistenceIntegrityError("terminal task is missing its result report")
            result_report = await self.load_result_report(result_report_id, result_report_revision)
            if (
                result_report.session_id != task.session_id
                or result_report.report_status is not ReportStatus[task.status.name]
            ):
                raise PersistenceIntegrityError("terminal task result report drift")
        elif result_report_id is not None or result_report_revision is not None:
            raise PersistenceIntegrityError("non-result task unexpectedly references a result")
        return StoredReviewTask(
            task=task,
            lock_version=row["lock_version"],
            claimed_principal_id=row["claimed_principal_id"],
            claimed_credential_id=row["claimed_credential_id"],
            claimed_at=row["claimed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            result_report_id=result_report_id,
            result_report_revision=result_report_revision,
        )

    async def load_source_shots(self, task_id: UUID) -> tuple[StoredSourceShot, ...]:
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM review_shot_sources WHERE task_id = :task_id "
                            "ORDER BY sequence, shot_id"
                        ),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .all()
            )
        if not rows:
            raise ResourceNotFoundError("source shots not found")
        task = await self.load_task(task_id)
        result: list[StoredSourceShot] = []
        for row in rows:
            payload = row["payload"]
            _verify_digest(payload, row["payload_sha256"])
            shot = _validate_json_model(Shot, payload)
            if (
                shot.id != row["shot_id"]
                or shot.session_id != row["session_id"]
                or shot.sequence != row["sequence"]
                or shot.raw_observation != {}
                or shot.session_id != task.task.session_id
            ):
                raise PersistenceIntegrityError("source shot columns drift from the stored payload")
            result.append(StoredSourceShot(shot=shot, source_envelope_id=row["source_envelope_id"]))
        if {item.shot.sequence for item in result} != set(range(1, len(result) + 1)):
            raise PersistenceIntegrityError("stored source shot set is incomplete")
        return tuple(result)

    @staticmethod
    def _validate_task_scoped_identity(
        identity: IdempotencyIdentity,
        *,
        operation: str,
    ) -> None:
        if (
            identity.operation != operation
            or identity.shot_id is not None
            or not identity.actor_principal_id
            or not identity.idempotency_key
        ):
            raise ValueError(f"{operation} idempotency identity is invalid")

    @staticmethod
    async def _receipt_row(
        session: AsyncSession,
        identity: IdempotencyIdentity,
        *,
        for_update: bool,
    ):
        query = (
            "SELECT * FROM quality_gate_idempotency_receipts "
            "WHERE actor_principal_id=:actor AND operation=:operation "
            "AND task_id=:task_id AND shot_id_or_zero_uuid=:shot_id "
            "AND idempotency_key=:key FOR UPDATE"
            if for_update
            else "SELECT * FROM quality_gate_idempotency_receipts "
            "WHERE actor_principal_id=:actor AND operation=:operation "
            "AND task_id=:task_id AND shot_id_or_zero_uuid=:shot_id "
            "AND idempotency_key=:key"
        )
        return (
            (
                await session.execute(
                    text(query),
                    {
                        "actor": identity.actor_principal_id,
                        "operation": identity.operation,
                        "task_id": identity.task_id,
                        "shot_id": ZERO_UUID,
                        "key": identity.idempotency_key,
                    },
                )
            )
            .mappings()
            .one_or_none()
        )

    @staticmethod
    def _validate_claim_receipt_row(
        row,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> None:
        if row["request_digest"] != request_digest:
            raise IdempotencyConflictError("claim key was reused with different facts")
        if row["result_kind"] != "review_task" or row["result_identity"] != str(identity.task_id):
            raise PersistenceIntegrityError("claim receipt points at the wrong result")

    async def _strict_claim_result(
        self,
        *,
        identity: IdempotencyIdentity,
        receipt_row,
    ) -> StoredReviewTask:
        current = await self.load_task(identity.task_id)
        if current.task.status is ReviewTaskStatus.IN_REVIEW:
            claim_lock_version = current.lock_version
        elif current.task.status in {
            ReviewTaskStatus.APPROVED,
            ReviewTaskStatus.CORRECTED,
            ReviewTaskStatus.REJECTED,
        }:
            claim_lock_version = current.lock_version - 1
        else:
            raise PersistenceIntegrityError("claim receipt task was never claimed")
        if (
            claim_lock_version < 1
            or current.claimed_principal_id != identity.actor_principal_id
            or current.claimed_credential_id is None
            or current.claimed_at is None
            or receipt_row["result_digest"]
            != _task_result_digest(
                task_id=current.task.id,
                status=ReviewTaskStatus.IN_REVIEW.value,
                lock_version=claim_lock_version,
            )
        ):
            raise PersistenceIntegrityError("claim receipt result authority drift")
        ownership = await self.load_session_ownership(identity.task_id)
        if (
            ownership.task_id != current.task.id
            or ownership.session_id != current.task.session_id
            or ownership.bound_at != current.claimed_at
        ):
            raise PersistenceIntegrityError("claim ownership authority drift")
        return StoredReviewTask(
            task=current.task.model_copy(update={"status": ReviewTaskStatus.IN_REVIEW}),
            lock_version=claim_lock_version,
            claimed_principal_id=current.claimed_principal_id,
            claimed_credential_id=current.claimed_credential_id,
            claimed_at=current.claimed_at,
            created_at=current.created_at,
            updated_at=current.claimed_at,
            completed_at=None,
            result_report_id=None,
            result_report_revision=None,
        )

    async def probe_claim_receipt(
        self,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredReviewTask | None:
        """Replay a durable application claim without invoking any candidate factory."""

        self._validate_task_scoped_identity(identity, operation="claim_review_v1")
        _require_request_digest(request_digest, operation="claim")
        async with self._sessions() as session:
            row = await self._receipt_row(session, identity, for_update=False)
        if row is None:
            return None
        self._validate_claim_receipt_row(
            row,
            identity=identity,
            request_digest=request_digest,
        )
        return await self._strict_claim_result(identity=identity, receipt_row=row)

    async def claim_task_with_ownership_and_receipt(
        self,
        *,
        task_id: UUID,
        expected_lock_version: int,
        actor_principal_id: str,
        credential_id: str,
        claimed_at: datetime,
        ownership: ReviewSessionOwnership,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredReviewTask:
        """Atomically bind subject authority, claim the task, and persist its receipt."""

        actor_principal_id = _require_nonblank(actor_principal_id, field="actor_principal_id")
        credential_id = _require_nonblank(credential_id, field="credential_id")
        claimed_at = _require_utc(claimed_at, field="claimed_at")
        ownership = ReviewSessionOwnership.model_validate(
            ownership.model_dump(mode="python"), strict=True
        )
        self._validate_task_scoped_identity(identity, operation="claim_review_v1")
        _require_request_digest(request_digest, operation="claim")
        if (
            type(expected_lock_version) is not int
            or expected_lock_version < 0
            or identity.task_id != task_id
            or identity.actor_principal_id != actor_principal_id
            or ownership.task_id != task_id
            or ownership.bound_at != claimed_at
        ):
            raise ValueError("claim authority does not bind the requested task")

        receipt_row = None
        async with self._sessions() as session, session.begin():
            task_row = (
                (
                    await session.execute(
                        text("SELECT * FROM review_tasks WHERE task_id=:task_id FOR UPDATE"),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if task_row is None:
                raise ResourceNotFoundError("review task not found")
            receipt_row = await self._receipt_row(session, identity, for_update=True)
            if receipt_row is not None:
                self._validate_claim_receipt_row(
                    receipt_row,
                    identity=identity,
                    request_digest=request_digest,
                )
            else:
                if (
                    task_row["status"] != ReviewTaskStatus.PENDING_REVIEW.value
                    or task_row["lock_version"] != expected_lock_version
                    or ownership.session_id != task_row["session_id"]
                ):
                    raise RevisionConflictError(
                        "review task is not claimable at the expected version"
                    )
                ownership_payload = ownership.model_dump(mode="json")
                assert_safe_persistence_payload(ownership_payload)
                await session.execute(
                    text(
                        "INSERT INTO review_session_ownership "
                        "(task_id,session_id,user_id,bound_at,payload,payload_sha256) VALUES "
                        "(:task_id,:session_id,:user_id,:bound_at,CAST(:payload AS jsonb),:digest)"
                    ),
                    {
                        "task_id": ownership.task_id,
                        "session_id": ownership.session_id,
                        "user_id": ownership.user_id,
                        "bound_at": ownership.bound_at,
                        "payload": json.dumps(ownership_payload, ensure_ascii=False),
                        "digest": canonical_sha256(ownership_payload),
                    },
                )
                updated = (
                    (
                        await session.execute(
                            text(
                                "UPDATE review_tasks SET status='in_review', "
                                "claimed_principal_id=:actor,claimed_credential_id=:credential,"
                                "claimed_at=:claimed_at,updated_at=:claimed_at,"
                                "lock_version=lock_version+1 WHERE task_id=:task_id "
                                "AND status='pending_review' AND lock_version=:expected "
                                "RETURNING task_id,status,lock_version"
                            ),
                            {
                                "task_id": task_id,
                                "expected": expected_lock_version,
                                "actor": actor_principal_id,
                                "credential": credential_id,
                                "claimed_at": claimed_at,
                            },
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if updated is None:
                    raise RevisionConflictError(
                        "review task is not claimable at the expected version"
                    )
                result_digest = _task_result_digest(
                    task_id=task_id,
                    status=updated["status"],
                    lock_version=updated["lock_version"],
                )
                await session.execute(
                    text(
                        "INSERT INTO quality_gate_idempotency_receipts "
                        "(actor_principal_id,operation,task_id,shot_id_or_zero_uuid,"
                        "idempotency_key,request_digest,result_kind,result_identity,"
                        "result_digest,created_at) VALUES "
                        "(:actor,:operation,:task_id,:shot_id,:key,:request_digest,"
                        "'review_task',:result_identity,:result_digest,:created_at)"
                    ),
                    {
                        "actor": identity.actor_principal_id,
                        "operation": identity.operation,
                        "task_id": identity.task_id,
                        "shot_id": ZERO_UUID,
                        "key": identity.idempotency_key,
                        "request_digest": request_digest,
                        "result_identity": str(task_id),
                        "result_digest": result_digest,
                        "created_at": claimed_at,
                    },
                )
                receipt_row = {
                    "request_digest": request_digest,
                    "result_kind": "review_task",
                    "result_identity": str(task_id),
                    "result_digest": result_digest,
                }
        if receipt_row is None:
            raise PersistenceIntegrityError("claim committed without a receipt")
        return await self._strict_claim_result(identity=identity, receipt_row=receipt_row)

    async def claim_task(
        self,
        *,
        task_id: UUID,
        expected_lock_version: int,
        actor_principal_id: str,
        credential_id: str,
        claimed_at: datetime,
    ) -> StoredReviewTask:
        actor_principal_id = _require_nonblank(actor_principal_id, field="actor_principal_id")
        credential_id = _require_nonblank(credential_id, field="credential_id")
        claimed_at = _require_utc(claimed_at, field="claimed_at")
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    text(
                        "UPDATE review_tasks SET status = 'in_review', "
                        "claimed_principal_id = :actor, claimed_credential_id = :credential, "
                        "claimed_at = :claimed_at, updated_at = :claimed_at, "
                        "lock_version = lock_version + 1 "
                        "WHERE task_id = :task_id AND status = 'pending_review' "
                        "AND lock_version = :expected RETURNING task_id"
                    ),
                    {
                        "task_id": task_id,
                        "expected": expected_lock_version,
                        "actor": actor_principal_id,
                        "credential": credential_id,
                        "claimed_at": claimed_at,
                    },
                )
            ).one_or_none()
            if row is None:
                raise RevisionConflictError("review task is not claimable at the expected version")
        return await self.load_task(task_id)

    async def record_idempotency_receipt(
        self,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
        result_kind: str,
        result_identity: str,
        result_digest: str,
        created_at: datetime,
    ) -> IdempotencyReceipt:
        if (
            not identity.actor_principal_id
            or not identity.operation
            or not identity.idempotency_key
        ):
            raise ValueError("idempotency identity fields must be non-empty")
        if identity.shot_id == ZERO_UUID:
            raise ValueError("the zero UUID is reserved for task-scoped idempotency")
        created_at = _require_utc(created_at, field="created_at")
        shot_key = identity.shot_id or ZERO_UUID
        async with self._sessions() as session, session.begin():
            await self._validate_receipt_result(
                session,
                result_kind=result_kind,
                result_identity=result_identity,
                result_digest=result_digest,
            )
            await session.execute(
                text(
                    "INSERT INTO quality_gate_idempotency_receipts "
                    "(actor_principal_id, operation, task_id, shot_id_or_zero_uuid, "
                    "idempotency_key, request_digest, result_kind, result_identity, "
                    "result_digest, created_at) VALUES "
                    "(:actor, :operation, :task_id, :shot_id, :key, :request_digest, "
                    ":result_kind, :result_identity, :result_digest, :created_at) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "actor": identity.actor_principal_id,
                    "operation": identity.operation,
                    "task_id": identity.task_id,
                    "shot_id": shot_key,
                    "key": identity.idempotency_key,
                    "request_digest": request_digest,
                    "result_kind": result_kind,
                    "result_identity": result_identity,
                    "result_digest": result_digest,
                    "created_at": created_at,
                },
            )
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM quality_gate_idempotency_receipts "
                            "WHERE actor_principal_id = :actor AND operation = :operation "
                            "AND task_id = :task_id AND shot_id_or_zero_uuid = :shot_id "
                            "AND idempotency_key = :key"
                        ),
                        {
                            "actor": identity.actor_principal_id,
                            "operation": identity.operation,
                            "task_id": identity.task_id,
                            "shot_id": shot_key,
                            "key": identity.idempotency_key,
                        },
                    )
                )
                .mappings()
                .one()
            )
        if (
            row["request_digest"] != request_digest
            or row["result_kind"] != result_kind
            or row["result_identity"] != result_identity
            or row["result_digest"] != result_digest
        ):
            raise IdempotencyConflictError("idempotency identity was reused with different data")
        return IdempotencyReceipt(
            identity=identity,
            request_digest=row["request_digest"],
            result_kind=row["result_kind"],
            result_identity=row["result_identity"],
            result_digest=row["result_digest"],
            created_at=row["created_at"],
        )

    @staticmethod
    async def _validate_receipt_result(
        session: AsyncSession,
        *,
        result_kind: str,
        result_identity: str,
        result_digest: str,
    ) -> None:
        if result_kind == "review_task":
            try:
                task_id = UUID(result_identity)
            except ValueError as exc:
                raise ValueError("review_task result identity must be a UUID") from exc
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT task_id, status, lock_version FROM review_tasks "
                            "WHERE task_id=:task_id"
                        ),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise ResourceNotFoundError("idempotency result task not found")
            expected_digest = canonical_sha256(
                {
                    "task_id": str(row["task_id"]),
                    "status": row["status"],
                    "lock_version": row["lock_version"],
                }
            )
        elif result_kind == "session_rejection":
            try:
                rejection_id = UUID(result_identity)
            except ValueError as exc:
                raise ValueError("rejection result identity must be a UUID") from exc
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT payload, payload_sha256 FROM review_rejections "
                            "WHERE rejection_id=:rejection_id"
                        ),
                        {"rejection_id": rejection_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise ResourceNotFoundError("idempotency rejection not found")
            _verify_digest(row["payload"], row["payload_sha256"])
            expected_digest = row["payload_sha256"]
        else:
            raise ValueError("unsupported idempotency result kind")
        if result_digest != expected_digest:
            raise IdempotencyConflictError("receipt result digest does not match authority")

    @staticmethod
    def _validate_rejection_identity(identity: IdempotencyIdentity) -> None:
        if (
            identity.operation not in {"reject_session", "reject_review_intent_v1"}
            or identity.shot_id is not None
            or not identity.actor_principal_id
            or not identity.idempotency_key
        ):
            raise ValueError("rejection idempotency identity is invalid")

    @staticmethod
    def _validate_rejection_receipt_row(
        row,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> None:
        if row["request_digest"] != request_digest:
            raise IdempotencyConflictError("rejection key was reused with different facts")
        if row["result_kind"] != "session_rejection":
            raise PersistenceIntegrityError("rejection receipt points at the wrong result kind")
        try:
            UUID(row["result_identity"])
        except (TypeError, ValueError) as exc:
            raise PersistenceIntegrityError("rejection receipt identity is invalid") from exc
        if row["task_id"] != identity.task_id:
            raise PersistenceIntegrityError("rejection receipt task authority drift")

    async def probe_rejection_receipt(
        self,
        *,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredRejectedReviewResult | None:
        """Replay a rejected terminal graph before UUID/time factories are invoked."""

        self._validate_rejection_identity(identity)
        _require_request_digest(request_digest, operation="rejection")
        async with self._sessions() as session:
            row = await self._receipt_row(session, identity, for_update=False)
        if row is None:
            return None
        self._validate_rejection_receipt_row(
            row,
            identity=identity,
            request_digest=request_digest,
        )
        stored = await self.load_rejected_result(task_id=identity.task_id, replayed=True)
        if (
            str(stored.result.rejection.rejection_id) != row["result_identity"]
            or canonical_sha256(stored.result.rejection.model_dump(mode="json"))
            != row["result_digest"]
        ):
            raise PersistenceIntegrityError("rejection receipt result digest drift")
        return stored

    async def load_rejected_result(
        self,
        *,
        task_id: UUID,
        replayed: bool = False,
    ) -> StoredRejectedReviewResult:
        """Strictly reconstruct the one rejection, notification event, task, and receipt."""

        async with self._sessions() as session:
            rejection_rows = (
                (
                    await session.execute(
                        text("SELECT * FROM review_rejections WHERE task_id=:task_id"),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .all()
            )
            event_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM quality_gate_outbox WHERE "
                            "event_type='review.session.rejected.v1' "
                            "AND correlation_id=:task_id"
                        ),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .all()
            )
            receipt_rows = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM quality_gate_idempotency_receipts "
                            "WHERE task_id=:task_id AND result_kind='session_rejection'"
                        ),
                        {"task_id": task_id},
                    )
                )
                .mappings()
                .all()
            )
        if len(rejection_rows) != 1 or len(event_rows) != 1 or len(receipt_rows) != 1:
            raise PersistenceIntegrityError("rejection terminal graph is incomplete")
        row = rejection_rows[0]
        payload = row["payload"]
        _verify_digest(payload, row["payload_sha256"])
        rejection = _validate_json_model(SessionRejection, payload)
        if (
            rejection.rejection_id != row["rejection_id"]
            or rejection.review_task_id != row["task_id"]
            or rejection.session_id != row["session_id"]
            or rejection.report_id != row["source_report_id"]
            or rejection.report_revision != row["source_report_revision"]
            or rejection.reason_code.value != row["reason_code"]
            or rejection.reviewer.principal_id != row["reviewer_principal_id"]
            or rejection.reviewer.credential_id != row["reviewer_credential_id"]
            or rejection.rejected_at != row["rejected_at"]
        ):
            raise PersistenceIntegrityError("rejection columns drift from payload")
        event = _event_from_row(event_rows[0])
        expected_event_payload = {
            "session_id": str(rejection.session_id),
            "reason": rejection.reason_code.value,
            "message": rejection.reason_code.notification_message.value,
            "created_at": rejection.rejected_at.isoformat(),
        }
        if (
            event.aggregate_type != "review_session"
            or event.aggregate_id != rejection.session_id
            or event.aggregate_version != rejection.report_revision
            or event.occurred_at != rejection.rejected_at
            or event.causation_id is not None
            or event.correlation_id != rejection.review_task_id
            or event.idempotency_key
            != f"review-rejected:{rejection.review_task_id}:{rejection.report_revision}"
            or event.payload != expected_event_payload
        ):
            raise PersistenceIntegrityError("rejection notification event authority drift")
        notification = UserNotificationEvent(
            event_id=event.event_id,
            session_id=rejection.session_id,
            reason_code=rejection.reason_code,
            message=rejection.reason_code.notification_message,
            created_at=rejection.rejected_at,
            status=UserNotificationStatus.PENDING,
        )
        task = await self.load_task(task_id)
        if (
            task.task.status is not ReviewTaskStatus.REJECTED
            or task.task.session_id != rejection.session_id
            or task.task.report_id != rejection.report_id
            or task.task.report_revision != rejection.report_revision
            or task.claimed_principal_id != rejection.reviewer.principal_id
            or task.claimed_credential_id != rejection.reviewer.credential_id
            or task.completed_at != rejection.rejected_at
        ):
            raise PersistenceIntegrityError("rejection task authority drift")
        result = SessionRejectionResult(
            review_task=task.task,
            rejection=rejection,
            notification=notification,
        )
        receipt = receipt_rows[0]
        if (
            receipt["task_id"] != task_id
            or receipt["actor_principal_id"] != rejection.reviewer.principal_id
            or receipt["operation"] not in {"reject_session", "reject_review_intent_v1"}
            or receipt["shot_id_or_zero_uuid"] != ZERO_UUID
            or receipt["result_kind"] != "session_rejection"
            or receipt["result_identity"] != str(rejection.rejection_id)
            or receipt["result_digest"] != row["payload_sha256"]
        ):
            raise PersistenceIntegrityError("rejection receipt authority drift")
        return StoredRejectedReviewResult(
            task=task,
            result=result,
            notification_event=event,
            replayed=replayed,
        )

    async def commit_rejection(
        self,
        *,
        result: SessionRejectionResult,
        notification_event: DomainEvent,
        expected_lock_version: int,
        identity: IdempotencyIdentity,
        request_digest: str,
    ) -> StoredReviewTask:
        result = SessionRejectionResult.model_validate(result.model_dump(mode="python"))
        notification_event = DomainEvent.model_validate(
            notification_event.model_dump(mode="python")
        )
        task = result.review_task
        rejection = result.rejection
        notification = result.notification
        expected_payload = {
            "session_id": str(task.session_id),
            "reason": rejection.reason_code.value,
            "message": notification.message.value,
            "created_at": notification.created_at.isoformat(),
        }
        self._validate_rejection_identity(identity)
        _require_request_digest(request_digest, operation="rejection")
        if (
            identity.actor_principal_id != rejection.reviewer.principal_id
            or identity.task_id != task.id
        ):
            raise ValueError("rejection idempotency identity does not match the mutation")
        if (
            task.status is not ReviewTaskStatus.REJECTED
            or notification_event.event_id != notification.event_id
            or notification_event.event_type != "review.session.rejected.v1"
            or notification_event.aggregate_type != "review_session"
            or notification_event.aggregate_id != task.session_id
            or notification_event.aggregate_version != task.report_revision
            or notification_event.idempotency_key
            != f"review-rejected:{task.id}:{task.report_revision}"
            or notification_event.payload != expected_payload
            or notification_event.correlation_id != task.id
            or notification_event.causation_id is not None
            or notification_event.occurred_at != notification.created_at
        ):
            raise ValueError("rejection notification event does not match terminal facts")
        payload = _as_json(rejection)
        assert_safe_persistence_payload(payload, allow_reviewer_identity=True)
        receipt_hit = False
        cas_lost = False
        async with self._sessions() as session, session.begin():
            receipt_row = await self._receipt_row(session, identity, for_update=True)
            if receipt_row is not None:
                self._validate_rejection_receipt_row(
                    receipt_row,
                    identity=identity,
                    request_digest=request_digest,
                )
                receipt_hit = True
            else:
                updated = (
                    await session.execute(
                        text(
                            "UPDATE review_tasks SET status = 'rejected', "
                            "completed_at = :completed, "
                            "updated_at = :completed, lock_version = lock_version + 1 "
                            "WHERE task_id = :task_id AND status = 'in_review' "
                            "AND lock_version = :expected AND claimed_principal_id = :actor "
                            "AND claimed_credential_id = :credential RETURNING task_id"
                        ),
                        {
                            "task_id": task.id,
                            "expected": expected_lock_version,
                            "actor": rejection.reviewer.principal_id,
                            "credential": rejection.reviewer.credential_id,
                            "completed": rejection.rejected_at,
                        },
                    )
                ).one_or_none()
                if updated is None:
                    cas_lost = True
                else:
                    await session.execute(
                        text(
                            "INSERT INTO review_rejections "
                            "(rejection_id, task_id, session_id, source_report_id, "
                            "source_report_revision, reason_code, reviewer_principal_id, "
                            "reviewer_credential_id, rejected_at, payload, payload_sha256) VALUES "
                            "(:rejection_id, :task_id, :session_id, :report_id, :revision, "
                            ":reason, :principal, :credential, :rejected_at, "
                            "CAST(:payload AS jsonb), :digest)"
                        ),
                        {
                            "rejection_id": rejection.rejection_id,
                            "task_id": task.id,
                            "session_id": task.session_id,
                            "report_id": task.report_id,
                            "revision": task.report_revision,
                            "reason": rejection.reason_code.value,
                            "principal": rejection.reviewer.principal_id,
                            "credential": rejection.reviewer.credential_id,
                            "rejected_at": rejection.rejected_at,
                            "payload": json.dumps(payload, ensure_ascii=False),
                            "digest": canonical_sha256(payload),
                        },
                    )
                    await PostgresOutboxRepository.enqueue_in_session(
                        session, notification_event, available_at=notification.created_at
                    )
                    await session.execute(
                        text(
                            "INSERT INTO quality_gate_idempotency_receipts "
                            "(actor_principal_id, operation, task_id, shot_id_or_zero_uuid, "
                            "idempotency_key, request_digest, result_kind, result_identity, "
                            "result_digest, created_at) VALUES "
                            "(:actor, :operation, :task_id, :shot_id, :key, :request_digest, "
                            "'session_rejection', :result_identity, :result_digest, :created_at)"
                        ),
                        {
                            "actor": identity.actor_principal_id,
                            "operation": identity.operation,
                            "task_id": identity.task_id,
                            "shot_id": ZERO_UUID,
                            "key": identity.idempotency_key,
                            "request_digest": request_digest,
                            "result_identity": str(rejection.rejection_id),
                            "result_digest": canonical_sha256(payload),
                            "created_at": rejection.rejected_at,
                        },
                    )
        if cas_lost:
            async with self._sessions() as session:
                receipt_row = await self._receipt_row(session, identity, for_update=False)
            if receipt_row is None:
                raise RevisionConflictError("rejection lost task ownership or revision")
            self._validate_rejection_receipt_row(
                receipt_row,
                identity=identity,
                request_digest=request_digest,
            )
        if receipt_hit or cas_lost:
            stored = await self.load_rejected_result(task_id=task.id, replayed=True)
            if (
                str(stored.result.rejection.rejection_id) != receipt_row["result_identity"]
                or canonical_sha256(stored.result.rejection.model_dump(mode="json"))
                != receipt_row["result_digest"]
            ):
                raise PersistenceIntegrityError("rejection receipt result digest drift")
            return stored.task
        return await self.load_task(task.id)

    async def _stage_review_submission(
        self,
        session: AsyncSession,
        *,
        source_task: StoredReviewTask,
        submission: ReviewSubmission,
    ) -> None:
        """Stage reviewed immutable facts inside a caller-owned terminal transaction.

        This private seam intentionally does not transition the task. Story 11.2 must
        compose it with Batch v2 and outbox creation before committing a terminal state.
        """

        if source_task.task.status is not ReviewTaskStatus.IN_REVIEW:
            raise ValueError("submission staging requires an in-review source task")
        report = ReportDocument.model_validate(submission.report.model_dump(mode="python"))
        final_task = ReviewTask.model_validate(submission.review_task.model_dump(mode="python"))
        decisions = tuple(
            ShotReviewDecision.model_validate(item.model_dump(mode="python"))
            for item in submission.decisions
        )
        shots = tuple(
            Shot.model_validate(item.model_dump(mode="python")) for item in submission.shots
        )
        task_row = (
            (
                await session.execute(
                    text("SELECT * FROM review_tasks WHERE task_id=:task_id FOR UPDATE"),
                    {"task_id": source_task.task.id},
                )
            )
            .mappings()
            .one_or_none()
        )
        if task_row is None:
            raise ResourceNotFoundError("review task not found")
        if (
            task_row["status"] != ReviewTaskStatus.IN_REVIEW.value
            or task_row["lock_version"] != source_task.lock_version
            or task_row["claimed_principal_id"] != source_task.claimed_principal_id
            or task_row["claimed_credential_id"] != source_task.claimed_credential_id
            or task_row["source_report_id"] != source_task.task.report_id
            or task_row["source_report_revision"] != source_task.task.report_revision
        ):
            raise RevisionConflictError("submission source task changed before staging")
        source_rows = (
            (
                await session.execute(
                    text(
                        "SELECT * FROM review_shot_sources WHERE task_id=:task_id "
                        "ORDER BY sequence, shot_id FOR UPDATE"
                    ),
                    {"task_id": source_task.task.id},
                )
            )
            .mappings()
            .all()
        )
        source_shots: list[Shot] = []
        for row in source_rows:
            _verify_digest(row["payload"], row["payload_sha256"])
            source_shot = _validate_json_model(Shot, row["payload"])
            if (
                source_shot.id != row["shot_id"]
                or source_shot.session_id != row["session_id"]
                or source_shot.sequence != row["sequence"]
                or source_shot.raw_observation != {}
            ):
                raise PersistenceIntegrityError("source shot authority drift")
            source_shots.append(source_shot)
        if not source_shots:
            raise PersistenceIntegrityError("submission source shot set is empty")
        if (
            final_task.id != source_task.task.id
            or final_task.session_id != source_task.task.session_id
        ):
            raise ValueError("submission must preserve task and session identity")
        if (final_task.report_id, final_task.report_revision) != (
            report.report_id,
            report.revision,
        ):
            raise ValueError("terminal task must bind the exact result report")
        if final_task.status is ReviewTaskStatus.APPROVED:
            if (report.report_id, report.revision) != (
                source_task.task.report_id,
                source_task.task.report_revision,
            ):
                raise ValueError("approved submission must preserve the source report revision")
        elif final_task.status is ReviewTaskStatus.CORRECTED:
            if (
                report.parent_revision_id != source_task.task.report_id
                or report.revision != source_task.task.report_revision + 1
            ):
                raise ValueError("corrected submission must be the direct child of the source")
        else:
            raise ValueError("submission staging only accepts approved or corrected results")
        if not shots or {shot.id for shot in shots} != {decision.shot_id for decision in decisions}:
            raise ValueError("submission decisions must cover the exact result shot set")
        by_decision = {decision.shot_id: decision for decision in decisions}
        if any(
            shot.review_status.value != by_decision[shot.id].outcome.value
            or shot.session_id != final_task.session_id
            for shot in shots
        ):
            raise ValueError("result shot status must match its decision")
        corrections = tuple(
            correction for decision in decisions for correction in decision.corrections
        )
        if corrections != submission.corrections:
            raise ValueError("submission corrections must equal decision corrections")
        if any(
            decision.reviewer.principal_id != source_task.claimed_principal_id
            or decision.reviewer.credential_id != source_task.claimed_credential_id
            for decision in decisions
        ):
            raise ValueError("submission reviewer must own the claimed task")
        application = apply_shot_reviews(
            session_id=source_task.task.session_id,
            shots=tuple(source_shots),
            decisions=decisions,
        )
        expected_shots = tuple(
            Shot.model_validate({**shot.model_dump(mode="python"), "raw_observation": {}})
            for shot in application.shots
        )
        submitted_shots = tuple(
            Shot.model_validate({**shot.model_dump(mode="python"), "raw_observation": {}})
            for shot in shots
        )
        if submitted_shots != expected_shots:
            raise ValueError("submission shots must be the deterministic complete reviewed set")

        report_payload = _as_json(report)
        assert_safe_persistence_payload(report_payload)
        await session.execute(
            text(
                "INSERT INTO review_result_reports "
                "(report_id, revision, source_report_id, source_report_revision, session_id, "
                "parent_report_id, schema_version, status, generated_at, payload, "
                "payload_sha256) VALUES "
                "(:report_id, :revision, :source_report_id, :source_revision, :session_id, "
                ":parent_report_id, :schema_version, :status, :generated_at, "
                "CAST(:payload AS jsonb), :digest)"
            ),
            {
                "report_id": report.report_id,
                "revision": report.revision,
                "source_report_id": source_task.task.report_id,
                "source_revision": source_task.task.report_revision,
                "session_id": report.session_id,
                "parent_report_id": report.parent_revision_id,
                "schema_version": report.schema_version,
                "status": report.report_status.value,
                "generated_at": report.generated_at,
                "payload": json.dumps(report_payload, ensure_ascii=False),
                "digest": canonical_sha256(report_payload),
            },
        )
        for decision in decisions:
            decision_payload = _as_json(decision)
            assert_safe_persistence_payload(decision_payload, allow_reviewer_identity=True)
            await session.execute(
                text(
                    "INSERT INTO review_shot_decisions "
                    "(task_id, shot_id, outcome, reviewer_principal_id, reviewer_credential_id, "
                    "reviewed_at, trajectory_reviewed, payload, payload_sha256) VALUES "
                    "(:task_id, :shot_id, :outcome, :principal, :credential, :reviewed_at, "
                    ":trajectory_reviewed, CAST(:payload AS jsonb), :digest)"
                ),
                {
                    "task_id": source_task.task.id,
                    "shot_id": decision.shot_id,
                    "outcome": decision.outcome.value,
                    "principal": decision.reviewer.principal_id,
                    "credential": decision.reviewer.credential_id,
                    "reviewed_at": decision.reviewed_at,
                    "trajectory_reviewed": decision.trajectory_reviewed,
                    "payload": json.dumps(decision_payload, ensure_ascii=False),
                    "digest": canonical_sha256(decision_payload),
                },
            )
            for correction in decision.corrections:
                correction_payload = _as_json(correction)
                assert_safe_persistence_payload(correction_payload, allow_reviewer_identity=True)
                await session.execute(
                    text(
                        "INSERT INTO review_corrections "
                        "(task_id, shot_id, field, ai_value, correct_value, note, "
                        "reviewer_principal_id, reviewer_credential_id, reviewed_at, payload, "
                        "payload_sha256) VALUES "
                        "(:task_id, :shot_id, :field, CAST(:ai_value AS jsonb), "
                        "CAST(:correct_value AS jsonb), :note, :principal, :credential, "
                        ":reviewed_at, CAST(:payload AS jsonb), :digest)"
                    ),
                    {
                        "task_id": source_task.task.id,
                        "shot_id": decision.shot_id,
                        "field": correction.field.value,
                        "ai_value": json.dumps(correction_payload["ai_value"]),
                        "correct_value": json.dumps(correction_payload["correct_value"]),
                        "note": correction.note,
                        "principal": correction.reviewer.principal_id,
                        "credential": correction.reviewer.credential_id,
                        "reviewed_at": correction.reviewed_at,
                        "payload": json.dumps(correction_payload, ensure_ascii=False),
                        "digest": canonical_sha256(correction_payload),
                    },
                )
        for shot in sorted(shots, key=lambda item: item.sequence):
            safe_shot = Shot.model_validate(
                {**shot.model_dump(mode="python"), "raw_observation": {}}
            )
            payload = _as_safe_shot_json(safe_shot)
            await session.execute(
                text(
                    "INSERT INTO review_shot_results "
                    "(task_id, shot_id, result_report_id, result_report_revision, session_id, "
                    "sequence, review_status, payload, payload_sha256) VALUES "
                    "(:task_id, :shot_id, :report_id, :revision, :session_id, :sequence, "
                    ":review_status, CAST(:payload AS jsonb), :digest)"
                ),
                {
                    "task_id": source_task.task.id,
                    "shot_id": shot.id,
                    "report_id": report.report_id,
                    "revision": report.revision,
                    "session_id": shot.session_id,
                    "sequence": shot.sequence,
                    "review_status": shot.review_status.value,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "digest": canonical_sha256(payload),
                },
            )

    async def load_result_report(self, report_id: UUID, revision: int) -> ReportDocument:
        async with self._sessions() as session:
            row = (
                (
                    await session.execute(
                        text(
                            "SELECT * FROM review_result_reports "
                            "WHERE report_id = :report_id AND revision = :revision"
                        ),
                        {"report_id": report_id, "revision": revision},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise ResourceNotFoundError("result report not found")
        payload = row["payload"]
        _verify_digest(payload, row["payload_sha256"])
        report = _validate_json_model(ReportDocument, payload)
        if (
            report.report_id != row["report_id"]
            or report.revision != row["revision"]
            or report.session_id != row["session_id"]
            or report.schema_version != row["schema_version"]
            or report.report_status.value != row["status"]
            or report.generated_at != row["generated_at"]
            or report.parent_revision_id != row["parent_report_id"]
        ):
            raise PersistenceIntegrityError("result report columns drift from payload")
        if row["status"] == ReportStatus.APPROVED.value:
            if (
                row["report_id"] != row["source_report_id"]
                or row["revision"] != row["source_report_revision"]
                or row["parent_report_id"] is not None
            ):
                raise PersistenceIntegrityError("approved result lineage is invalid")
        elif (
            row["parent_report_id"] != row["source_report_id"]
            or row["revision"] != row["source_report_revision"] + 1
        ):
            raise PersistenceIntegrityError("corrected result lineage is invalid")
        return report
