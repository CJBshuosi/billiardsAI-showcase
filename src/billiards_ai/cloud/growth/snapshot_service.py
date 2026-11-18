"""Trusted snapshot build orchestration and fenced historical reads."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.cloud.growth.snapshot_projector import (
    SnapshotProjectionError,
    project_growth_snapshot,
    project_growth_snapshot_v3,
)
from billiards_ai.cloud.growth.snapshot_repository import (
    GrowthSnapshotRepository,
    SnapshotAppendResult,
    SnapshotVersionConflict,
)
from billiards_ai.shared.models.evidence import AcceptedMetricEvidenceInput
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    SNAPSHOT_POLICY_VERSION_V3,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
    GrowthSnapshotLineage,
    GrowthSnapshotLineageV3,
    GrowthSnapshotReadFence,
    GrowthSnapshotReadFenceV3,
)


class AcceptedMetricEvidenceSource(Protocol):
    """Trusted port that issues fresh process-local admission proofs."""

    def list_accepted_metric_inputs(
        self, user_id: UUID
    ) -> tuple[AcceptedMetricEvidenceInput, ...]: ...


class AsyncAcceptedMetricEvidenceSource(Protocol):
    async def list_accepted_metric_inputs(
        self, user_id: UUID
    ) -> tuple[AcceptedMetricEvidenceInput, ...]: ...


class AsyncGrowthSnapshotRepository(Protocol):
    async def get_current(
        self, user_id: UUID, *, snapshot_policy_version: str
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3 | None: ...

    async def get_version(
        self, user_id: UUID, version: int, *, snapshot_policy_version: str
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3: ...

    async def compare_and_append(
        self,
        snapshot: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3,
        *,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult: ...


class SnapshotServiceError(RuntimeError):
    """Base type for snapshot application-boundary failures."""


class SnapshotSourceError(SnapshotServiceError):
    """The trusted source did not produce one valid user-owned bundle."""


class SnapshotReadFenceMismatch(SnapshotServiceError):
    """A historical read was attempted without its exact immutable fence."""


class GrowthSnapshotService:
    def __init__(
        self,
        *,
        source: AcceptedMetricEvidenceSource,
        repository: GrowthSnapshotRepository,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ):
        if snapshot_policy_version != SNAPSHOT_POLICY_VERSION:
            raise SnapshotSourceError("unsupported snapshot policy version")
        self._source = source
        self._repository = repository
        self._snapshot_policy_version = snapshot_policy_version

    def build_and_append(
        self,
        user_id: UUID,
        *,
        evidence_cutoff: datetime,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        """Read one trusted bundle, project it once, and atomically CAS append."""

        values = self._source.list_accepted_metric_inputs(user_id)
        if not isinstance(values, tuple) or not values:
            raise SnapshotSourceError("trusted evidence bundle is empty or mutable")
        if any(
            not isinstance(value, AcceptedMetricEvidenceInput)
            or not value.is_trusted_admission
            or value.batch.user_id != user_id
            for value in values
        ):
            raise SnapshotSourceError("trusted evidence bundle is invalid")

        current = self._repository.get_current(
            user_id, snapshot_policy_version=self._snapshot_policy_version
        )
        current_version = None if current is None else current.version
        if current_version != expected_previous_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        lineage = GrowthSnapshotLineage(
            version=1 if current is None else current.version + 1,
            previous_snapshot_id=None if current is None else current.snapshot_id,
            snapshot_policy_version=self._snapshot_policy_version,
        )
        try:
            candidate = project_growth_snapshot(
                values,
                evidence_cutoff=evidence_cutoff,
                lineage=lineage,
            )
        except (SnapshotProjectionError, AttributeError, TypeError, ValueError) as exc:
            raise SnapshotSourceError("trusted evidence bundle failed projection") from exc
        return self._repository.compare_and_append(
            candidate,
            expected_previous_version=expected_previous_version,
            idempotency_key=idempotency_key,
            clock=clock,
            uuid_factory=uuid_factory,
        )

    def read_version(
        self,
        user_id: UUID,
        version: int,
        *,
        expected_fence: GrowthSnapshotReadFence | None,
    ) -> GrowthEvidenceSnapshotV2:
        """Return an exact historical version only when its full fence matches."""

        snapshot = self._repository.get_version(
            user_id,
            version,
            snapshot_policy_version=self._snapshot_policy_version,
        )
        if expected_fence is None:
            raise SnapshotReadFenceMismatch("historical read requires a fence")
        try:
            fence = GrowthSnapshotReadFence.model_validate(
                stored_model_values(expected_fence), strict=True
            )
        except (AttributeError, TypeError, ValidationError, ValueError) as exc:
            raise SnapshotReadFenceMismatch("invalid historical read fence") from exc
        if fence != snapshot.read_fence:
            raise SnapshotReadFenceMismatch("historical read fence does not match")
        return snapshot


class GrowthSnapshotV3Service:
    """Trusted V3 orchestration isolated from the V2 default namespace."""

    def __init__(
        self,
        *,
        source: AcceptedMetricEvidenceSource,
        repository: GrowthSnapshotRepository,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION_V3,
    ):
        if snapshot_policy_version != SNAPSHOT_POLICY_VERSION_V3:
            raise SnapshotSourceError("unsupported v3 snapshot policy version")
        self._source = source
        self._repository = repository
        self._snapshot_policy_version = snapshot_policy_version

    def build_and_append(
        self,
        user_id: UUID,
        *,
        evidence_cutoff: datetime,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        values = self._source.list_accepted_metric_inputs(user_id)
        if not isinstance(values, tuple) or not values:
            raise SnapshotSourceError("trusted evidence bundle is empty or mutable")
        if any(
            not isinstance(value, AcceptedMetricEvidenceInput)
            or not value.is_trusted_admission
            or value.batch.user_id != user_id
            for value in values
        ):
            raise SnapshotSourceError("trusted evidence bundle is invalid")
        current = self._repository.get_current(
            user_id, snapshot_policy_version=self._snapshot_policy_version
        )
        if current is not None and not isinstance(current, GrowthEvidenceSnapshotV3):
            raise SnapshotSourceError("v3 policy namespace contains a non-v3 snapshot")
        current_version = None if current is None else current.version
        if current_version != expected_previous_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        lineage = GrowthSnapshotLineageV3(
            version=1 if current is None else current.version + 1,
            previous_snapshot_id=None if current is None else current.snapshot_id,
            snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        )
        try:
            candidate = project_growth_snapshot_v3(
                values, evidence_cutoff=evidence_cutoff, lineage=lineage
            )
        except (SnapshotProjectionError, AttributeError, TypeError, ValueError) as exc:
            raise SnapshotSourceError("trusted evidence bundle failed v3 projection") from exc
        return self._repository.compare_and_append(
            candidate,
            expected_previous_version=expected_previous_version,
            idempotency_key=idempotency_key,
            clock=clock,
            uuid_factory=uuid_factory,
        )

    def read_version(
        self,
        user_id: UUID,
        version: int,
        *,
        expected_fence: GrowthSnapshotReadFenceV3 | None,
    ) -> GrowthEvidenceSnapshotV3:
        snapshot = self._repository.get_version(
            user_id, version, snapshot_policy_version=self._snapshot_policy_version
        )
        if not isinstance(snapshot, GrowthEvidenceSnapshotV3):
            raise SnapshotReadFenceMismatch("v3 namespace returned a non-v3 snapshot")
        if expected_fence is None:
            raise SnapshotReadFenceMismatch("historical read requires a v3 fence")
        try:
            fence = GrowthSnapshotReadFenceV3.model_validate(
                stored_model_values(expected_fence), strict=True
            )
        except (AttributeError, TypeError, ValidationError, ValueError) as exc:
            raise SnapshotReadFenceMismatch("invalid historical v3 read fence") from exc
        if fence != snapshot.read_fence:
            raise SnapshotReadFenceMismatch("historical v3 read fence does not match")
        return snapshot


class AsyncGrowthSnapshotService:
    """Async V2 orchestration for durable source and repository ports."""

    def __init__(
        self,
        *,
        source: AsyncAcceptedMetricEvidenceSource,
        repository: AsyncGrowthSnapshotRepository,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION,
    ) -> None:
        if snapshot_policy_version != SNAPSHOT_POLICY_VERSION:
            raise SnapshotSourceError("unsupported snapshot policy version")
        self._source = source
        self._repository = repository
        self._policy = snapshot_policy_version

    async def build_and_append(
        self,
        user_id: UUID,
        *,
        evidence_cutoff: datetime,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        values = await self._source.list_accepted_metric_inputs(user_id)
        if not isinstance(values, tuple) or not values:
            raise SnapshotSourceError("trusted evidence bundle is empty or mutable")
        if any(
            not isinstance(value, AcceptedMetricEvidenceInput)
            or not value.is_trusted_admission
            or value.batch.user_id != user_id
            for value in values
        ):
            raise SnapshotSourceError("trusted evidence bundle is invalid")
        current = await self._repository.get_current(user_id, snapshot_policy_version=self._policy)
        current_version = None if current is None else current.version
        if current_version != expected_previous_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        lineage = GrowthSnapshotLineage(
            version=1 if current is None else current.version + 1,
            previous_snapshot_id=None if current is None else current.snapshot_id,
            snapshot_policy_version=self._policy,
        )
        try:
            candidate = project_growth_snapshot(
                values, evidence_cutoff=evidence_cutoff, lineage=lineage
            )
        except (
            SnapshotProjectionError,
            AttributeError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise SnapshotSourceError("trusted evidence bundle failed projection") from exc
        return await self._repository.compare_and_append(
            candidate,
            expected_previous_version=expected_previous_version,
            idempotency_key=idempotency_key,
            clock=clock,
            uuid_factory=uuid_factory,
        )


class AsyncGrowthSnapshotV3Service:
    """Async V3 orchestration isolated by the frozen V3 policy namespace."""

    def __init__(
        self,
        *,
        source: AsyncAcceptedMetricEvidenceSource,
        repository: AsyncGrowthSnapshotRepository,
    ) -> None:
        self._source = source
        self._repository = repository

    async def build_and_append(
        self,
        user_id: UUID,
        *,
        evidence_cutoff: datetime,
        expected_previous_version: int | None,
        idempotency_key: str,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], UUID],
    ) -> SnapshotAppendResult:
        values = await self._source.list_accepted_metric_inputs(user_id)
        if not isinstance(values, tuple) or not values:
            raise SnapshotSourceError("trusted evidence bundle is empty or mutable")
        if any(
            not isinstance(value, AcceptedMetricEvidenceInput)
            or not value.is_trusted_admission
            or value.batch.user_id != user_id
            for value in values
        ):
            raise SnapshotSourceError("trusted evidence bundle is invalid")
        current = await self._repository.get_current(
            user_id, snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3
        )
        if current is not None and not isinstance(current, GrowthEvidenceSnapshotV3):
            raise SnapshotSourceError("v3 namespace contains a non-v3 snapshot")
        current_version = None if current is None else current.version
        if current_version != expected_previous_version:
            raise SnapshotVersionConflict("expected snapshot version is stale")
        lineage = GrowthSnapshotLineageV3(
            version=1 if current is None else current.version + 1,
            previous_snapshot_id=None if current is None else current.snapshot_id,
            snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        )
        try:
            candidate = project_growth_snapshot_v3(
                values, evidence_cutoff=evidence_cutoff, lineage=lineage
            )
        except (
            SnapshotProjectionError,
            AttributeError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise SnapshotSourceError("trusted evidence bundle failed v3 projection") from exc
        return await self._repository.compare_and_append(
            candidate,
            expected_previous_version=expected_previous_version,
            idempotency_key=idempotency_key,
            clock=clock,
            uuid_factory=uuid_factory,
        )


__all__ = [
    "AcceptedMetricEvidenceSource",
    "AsyncAcceptedMetricEvidenceSource",
    "AsyncGrowthSnapshotRepository",
    "AsyncGrowthSnapshotService",
    "AsyncGrowthSnapshotV3Service",
    "GrowthSnapshotService",
    "GrowthSnapshotV3Service",
    "SnapshotReadFenceMismatch",
    "SnapshotServiceError",
    "SnapshotSourceError",
]
