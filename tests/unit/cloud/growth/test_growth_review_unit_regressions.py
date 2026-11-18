from __future__ import annotations

import pytest
from sqlalchemy.exc import SQLAlchemyError

from billiards_ai.cloud.growth.errors import EvidenceIngestError, EvidenceIngestErrorCode
from billiards_ai.cloud.growth.persistence.errors import GrowthPersistenceIntegrityError
from billiards_ai.cloud.growth.read_service import UserGrowthReadService
from billiards_ai.cloud.growth.repository import revalidate_approved_batch
from billiards_ai.cloud.growth.service import EvidenceIngestService
from tests.unit.cloud.growth.snapshot_test_support import make_batch


class _Identity:
    def __init__(self, user_id):
        self.user_id = user_id

    def current_user_id(self):
        return self.user_id


class _BrokenRepository:
    async def get_current(self, user_id, *, snapshot_policy_version):
        raise SQLAlchemyError("database-secret")


def test_constructed_recursive_batch_maps_to_stable_typed_error() -> None:
    batch = make_batch(90)
    object.__setattr__(batch, "shots", (batch,))

    with pytest.raises(ValueError, match="recursive"):
        revalidate_approved_batch(batch)


@pytest.mark.asyncio
async def test_sqlalchemy_read_failure_is_redacted_at_application_boundary() -> None:
    batch = make_batch(91)
    service = UserGrowthReadService(
        identity_provider=_Identity(batch.user_id), repository=_BrokenRepository()
    )
    with pytest.raises(GrowthPersistenceIntegrityError) as raised:
        await service.get_current()
    assert "database-secret" not in str(raised.value)


def test_service_maps_constructed_recursive_error_without_leaking_type() -> None:
    batch = make_batch(92)
    object.__setattr__(batch, "metric_shots", (batch,))
    service = EvidenceIngestService(
        approved_repository=None,  # type: ignore[arg-type]
        lineage_repository=None,  # type: ignore[arg-type]
        growth_repository=None,  # type: ignore[arg-type]
    )
    with pytest.raises(EvidenceIngestError) as raised:
        service.ingest(batch, authenticated_user_id=batch.user_id)
    assert raised.value.code is EvidenceIngestErrorCode.INVALID_BATCH
    assert "RecursionError" not in str(raised.value)
