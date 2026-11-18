from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from billiards_ai.cloud.growth.api import install_growth_api
from billiards_ai.cloud.growth.persistence.errors import (
    GrowthIdentityError,
    GrowthPersistenceIntegrityError,
)
from billiards_ai.cloud.growth.read_service import CurrentGrowthStatus, UserGrowthReadService
from billiards_ai.cloud.growth.snapshot_projector import project_growth_snapshot
from billiards_ai.shared.models.growth import SNAPSHOT_POLICY_VERSION, GrowthSnapshotLineage
from tests.unit.cloud.growth.snapshot_test_support import admit, make_batch


class Identity:
    def __init__(self, user_id):
        self.user_id = user_id

    def current_user_id(self):
        return self.user_id


class Repository:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.calls = []

    async def get_current(self, user_id, *, snapshot_policy_version):
        self.calls.append((user_id, snapshot_policy_version))
        return self.snapshot


@pytest.mark.asyncio
async def test_current_read_uses_only_authenticated_user_and_returns_empty() -> None:
    batch = make_batch(1)
    repository = Repository()
    service = UserGrowthReadService(
        identity_provider=Identity(batch.user_id),
        repository=repository,
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION,
    )
    result = await service.get_current()
    assert result.status is CurrentGrowthStatus.EMPTY
    assert result.snapshot is None
    assert repository.calls == [(batch.user_id, SNAPSHOT_POLICY_VERSION)]


@pytest.mark.asyncio
async def test_invalid_identity_fails_closed_before_repository_access() -> None:
    repository = Repository()
    service = UserGrowthReadService(
        identity_provider=Identity("client-supplied-user"), repository=repository
    )
    with pytest.raises(GrowthIdentityError):
        await service.get_current()
    assert repository.calls == []


@pytest.mark.asyncio
async def test_current_read_rejects_cross_owner_repository_result() -> None:
    own_batch = make_batch(3, user_number=901)
    other_batch = make_batch(4, user_number=902)
    other_snapshot = project_growth_snapshot(
        admit(other_batch),
        evidence_cutoff=other_batch.approved_at,
        lineage=GrowthSnapshotLineage(version=1, previous_snapshot_id=None),
    )
    service = UserGrowthReadService(
        identity_provider=Identity(own_batch.user_id),
        repository=Repository(other_snapshot),
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION,
    )
    with pytest.raises(GrowthPersistenceIntegrityError):
        await service.get_current()


@pytest.mark.asyncio
async def test_growth_api_is_explicit_owner_only_and_redacts_domain_errors() -> None:
    class EmptyService:
        async def get_current(self):
            raise GrowthPersistenceIntegrityError("database-secret")

    app = FastAPI()
    install_growth_api(app, lambda identity: EmptyService(), lambda: object())
    route = next(route for route in app.routes if route.path == "/v1/users/me/growth/current")
    response = await route.endpoint(growth_service=EmptyService())
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "ok": False,
        "data": None,
        "error": {"code": "GROWTH_READ_UNAVAILABLE", "message": "request unavailable"},
    }
    assert b"database-secret" not in response.body
    operation = app.openapi()["paths"]["/v1/users/me/growth/current"]["get"]
    assert operation.get("parameters", []) == []
    assert "requestBody" not in operation


def test_serialized_admission_remains_untrusted_at_read_boundary() -> None:
    [admission] = admit(make_batch(2))
    assert (
        admission.__class__.model_validate(admission.model_dump(mode="python")).is_trusted_admission
        is False
    )
