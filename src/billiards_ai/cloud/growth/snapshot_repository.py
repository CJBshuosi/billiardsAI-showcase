"""Immutable snapshot CAS repository with a transactional outbox reference."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import Protocol
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, TypeAdapter, ValidationError

from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.shared.models.events import DomainEvent
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
)

_UUID4 = TypeAdapter(UUID4)


class SnapshotRepositoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SnapshotAppendStatus(StrEnum):
    CREATED = "CREATED"
    REPLAYED = "REPLAYED"


class SnapshotRepositoryError(RuntimeError):
    """Base type for fail-closed snapshot persistence errors."""


class SnapshotVersionConflict(SnapshotRepositoryError):
    """The expected current version lost a compare-and-swap race."""


class SnapshotIdempotencyConflict(SnapshotRepositoryError):
    """An idempotency key was reused for different evidence."""


class SnapshotIntegrityError(SnapshotRepositoryError):
    """A candidate or stored snapshot violates immutable lineage."""


class SnapshotNotFound(SnapshotRepositoryError):
    """The requested immutable version does not exist."""


class SnapshotAppendResult(SnapshotRepositoryModel):
    status: SnapshotAppendStatus
    snapshot: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3
    event: DomainEvent


class GrowthSnapshotRepository(Protocol):
    def get_current(
        self,
        user_id: UUID,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3 | None: ...

    def get_version(
        self,
        user_id: UUID,
        version: int,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3: ...

    def compare_and_append(
        self,
        snapshot: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        *,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult: ...


def _revalidate_snapshot(value: object) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3:
    try:
        stored = stored_model_values(value)
        if not isinstance(stored, dict):
            raise TypeError("snapshot must be an object")
        if stored.get("schema_version") == "2.0":
            return GrowthEvidenceSnapshotV2.model_validate(stored, strict=True)
        if stored.get("schema_version") == "3.0":
            return GrowthEvidenceSnapshotV3.model_validate(stored, strict=True)
        raise ValueError("unsupported snapshot schema")
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotIntegrityError("invalid snapshot contract") from exc


class InMemoryGrowthSnapshotRepository:
    """Single-process atomic reference; durable storage belongs to Story 8.6."""

    def __init__(self, *, failure_injector: Callable[[str], None] | None = None):
        self._lock = Lock()
        self._failure_injector = failure_injector
        self._snapshots_by_version: dict[
            tuple[UUID, str, int], GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3
        ] = {}
        self._current_versions: dict[tuple[UUID, str], int] = {}
        self._versions_by_content: dict[tuple[UUID, str, str], int] = {}
        self._versions_by_idempotency: dict[tuple[UUID, str, str], int] = {}
        self._outbox_events: dict[tuple[UUID, str, int], DomainEvent] = {}

    def _inject(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)

    def _stored_version(
        self, user_id: UUID, snapshot_policy_version: str, version: int
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3:
        value = self._snapshots_by_version.get((user_id, snapshot_policy_version, version))
        if value is None:
            raise SnapshotIntegrityError("snapshot index refers to a missing version")
        return _revalidate_snapshot(value)

    def _idempotent_replay_locked(
        self,
        candidate: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        idempotency_index: tuple[UUID, str, str],
    ) -> SnapshotAppendResult | None:
        version = self._versions_by_idempotency.get(idempotency_index)
        if version is None:
            return None
        policy_key = (candidate.user_id, candidate.snapshot_policy_version)
        existing = self._stored_version(*policy_key, version)
        if existing.content_sha256 != candidate.content_sha256:
            raise SnapshotIdempotencyConflict(
                "idempotency key belongs to different snapshot content"
            )
        event = self._outbox_events[(*policy_key, version)]
        return SnapshotAppendResult(
            status=SnapshotAppendStatus.REPLAYED,
            snapshot=deepcopy(existing),
            event=deepcopy(event),
        )

    def _validate_expected_version_locked(
        self,
        candidate: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        expected_previous_version: int | None,
    ) -> int | None:
        policy_key = (candidate.user_id, candidate.snapshot_policy_version)
        current_version = self._current_versions.get(policy_key)
        if expected_previous_version != current_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        return current_version

    def _validate_lineage_locked(
        self,
        candidate: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        current_version: int | None,
    ) -> None:
        policy_key = (candidate.user_id, candidate.snapshot_policy_version)
        required_version = 1 if current_version is None else current_version + 1
        if candidate.version != required_version:
            raise SnapshotIntegrityError("candidate version is not the next version")
        expected_predecessor = (
            None
            if current_version is None
            else self._stored_version(*policy_key, current_version).snapshot_id
        )
        if candidate.previous_snapshot_id != expected_predecessor:
            raise SnapshotIntegrityError("candidate predecessor is not current")

    def _content_replay_locked(
        self,
        candidate: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        content_index: tuple[UUID, str, str],
        idempotency_index: tuple[UUID, str, str],
    ) -> SnapshotAppendResult | None:
        version = self._versions_by_content.get(content_index)
        if version is None:
            return None
        policy_key = (candidate.user_id, candidate.snapshot_policy_version)
        existing = self._stored_version(*policy_key, version)
        if existing.content_sha256 != candidate.content_sha256:
            raise SnapshotIntegrityError("snapshot content index is corrupt")
        self._versions_by_idempotency = self._versions_by_idempotency | {idempotency_index: version}
        event = self._outbox_events[(*policy_key, version)]
        return SnapshotAppendResult(
            status=SnapshotAppendStatus.REPLAYED,
            snapshot=deepcopy(existing),
            event=deepcopy(event),
        )

    def get_current(
        self,
        user_id: UUID,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3 | None:
        with self._lock:
            version = self._current_versions.get((user_id, snapshot_policy_version))
            if version is None:
                return None
            return deepcopy(self._stored_version(user_id, snapshot_policy_version, version))

    def get_version(
        self,
        user_id: UUID,
        version: int,
        *,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3:
        with self._lock:
            value = self._snapshots_by_version.get((user_id, snapshot_policy_version, version))
            if value is None:
                raise SnapshotNotFound("snapshot version not found")
            return deepcopy(_revalidate_snapshot(value))

    def compare_and_append(
        self,
        snapshot: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        *,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        candidate = _revalidate_snapshot(snapshot)
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise SnapshotIntegrityError("idempotency key must be non-empty")
        policy_key = (candidate.user_id, candidate.snapshot_policy_version)
        idempotency_index = (*policy_key, idempotency_key)
        content_index = (*policy_key, candidate.content_sha256)

        with self._lock:
            if replay := self._idempotent_replay_locked(candidate, idempotency_index):
                return replay
            current_version = self._validate_expected_version_locked(
                candidate, expected_previous_version
            )
            if replay := self._content_replay_locked(candidate, content_index, idempotency_index):
                return replay
            self._validate_lineage_locked(candidate, current_version)

        event = DomainEvent(
            event_id=_UUID4.validate_python(uuid_factory(), strict=True),
            event_type="growth.snapshot.created.v1",
            aggregate_type="growth_evidence_snapshot",
            aggregate_id=candidate.snapshot_id,
            aggregate_version=candidate.version,
            occurred_at=clock(),
            causation_id=candidate.previous_snapshot_id,
            correlation_id=candidate.snapshot_id,
            idempotency_key=idempotency_key,
            payload={
                "snapshot_id": str(candidate.snapshot_id),
                "user_id": str(candidate.user_id),
                "version": candidate.version,
                "schema_version": candidate.schema_version,
                "snapshot_policy_version": candidate.snapshot_policy_version,
                "taxonomy_version": candidate.taxonomy_version,
                "calculation_policy_version": candidate.calculation_policy_version,
                "evidence_sha256": candidate.evidence_sha256,
                "content_sha256": candidate.content_sha256,
                "source_bundle_sha256": candidate.source_bundle_sha256,
                "read_fence": candidate.read_fence.model_dump(mode="json"),
            },
        )
        for stage in (
            "snapshot",
            "version_index",
            "current_index",
            "content_index",
            "idempotency",
            "outbox",
        ):
            self._inject(stage)

        with self._lock:
            if replay := self._idempotent_replay_locked(candidate, idempotency_index):
                return replay
            current_version = self._validate_expected_version_locked(
                candidate, expected_previous_version
            )
            if replay := self._content_replay_locked(candidate, content_index, idempotency_index):
                return replay
            self._validate_lineage_locked(candidate, current_version)
            version_key = (*policy_key, candidate.version)
            snapshots = self._snapshots_by_version | {version_key: candidate}
            current_versions = self._current_versions | {policy_key: candidate.version}
            versions_by_content = self._versions_by_content | {content_index: candidate.version}
            versions_by_idempotency = self._versions_by_idempotency | {
                idempotency_index: candidate.version
            }
            outbox_events = self._outbox_events | {version_key: event}
            (
                self._snapshots_by_version,
                self._current_versions,
                self._versions_by_content,
                self._versions_by_idempotency,
                self._outbox_events,
            ) = (
                snapshots,
                current_versions,
                versions_by_content,
                versions_by_idempotency,
                outbox_events,
            )
            return SnapshotAppendResult(
                status=SnapshotAppendStatus.CREATED,
                snapshot=deepcopy(candidate),
                event=deepcopy(event),
            )

    def list_outbox_events(self) -> tuple[DomainEvent, ...]:
        with self._lock:
            return deepcopy(
                tuple(self._outbox_events[key] for key in sorted(self._outbox_events, key=str))
            )

    def snapshot(self) -> tuple[object, ...]:
        """Frozen authority-state view used for rollback diagnostics."""

        with self._lock:
            return deepcopy(
                (
                    tuple(self._snapshots_by_version.items()),
                    tuple(self._current_versions.items()),
                    tuple(self._versions_by_content.items()),
                    tuple(self._versions_by_idempotency.items()),
                    tuple(self._outbox_events.items()),
                )
            )


__all__ = [
    "GrowthSnapshotRepository",
    "InMemoryGrowthSnapshotRepository",
    "SnapshotAppendResult",
    "SnapshotAppendStatus",
    "SnapshotIdempotencyConflict",
    "SnapshotIntegrityError",
    "SnapshotNotFound",
    "SnapshotRepositoryError",
    "SnapshotVersionConflict",
]
