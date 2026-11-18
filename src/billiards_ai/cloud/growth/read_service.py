"""Authorized current committed growth snapshot reads."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import UUID4, BaseModel, ConfigDict, TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from billiards_ai.cloud.growth.snapshot_repository import SnapshotRepositoryError
from billiards_ai.cloud.table_binding.identity import AuthenticatedUserProvider
from billiards_ai.shared.models.growth import (
    SNAPSHOT_POLICY_VERSION,
    SNAPSHOT_POLICY_VERSION_V3,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
)

from .persistence.errors import GrowthIdentityError, GrowthPersistenceIntegrityError

_UUID4 = TypeAdapter(UUID4)


class AsyncCurrentSnapshotRepository(Protocol):
    async def get_current(
        self, user_id: UUID4, *, snapshot_policy_version: str
    ) -> GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3 | None: ...


class CurrentGrowthStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    EMPTY = "EMPTY"


class CurrentGrowthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: CurrentGrowthStatus
    snapshot: GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3 | None


class UserGrowthReadService:
    def __init__(
        self,
        *,
        identity_provider: AuthenticatedUserProvider,
        repository: AsyncCurrentSnapshotRepository,
        snapshot_policy_version: str = SNAPSHOT_POLICY_VERSION_V3,
    ) -> None:
        if snapshot_policy_version not in {
            SNAPSHOT_POLICY_VERSION,
            SNAPSHOT_POLICY_VERSION_V3,
        }:
            raise GrowthPersistenceIntegrityError("unsupported committed snapshot policy")
        self._identity = identity_provider
        self._repository = repository
        self._policy = snapshot_policy_version

    def _user_id(self) -> UUID4:
        try:
            return _UUID4.validate_python(self._identity.current_user_id(), strict=True)
        except (AttributeError, TypeError, ValueError) as exc:
            raise GrowthIdentityError("authenticated user identity is unavailable") from exc

    async def get_current(self) -> CurrentGrowthResult:
        user_id = self._user_id()
        try:
            snapshot = await self._repository.get_current(
                user_id, snapshot_policy_version=self._policy
            )
        except (GrowthPersistenceIntegrityError, SnapshotRepositoryError, SQLAlchemyError):
            raise GrowthPersistenceIntegrityError("committed growth read unavailable") from None
        if snapshot is None:
            return CurrentGrowthResult(status=CurrentGrowthStatus.EMPTY, snapshot=None)
        if snapshot.user_id != user_id or snapshot.snapshot_policy_version != self._policy:
            raise GrowthPersistenceIntegrityError("committed snapshot ownership mismatch")
        return CurrentGrowthResult(status=CurrentGrowthStatus.AVAILABLE, snapshot=snapshot)
