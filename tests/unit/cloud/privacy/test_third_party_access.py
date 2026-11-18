from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.access_repository import JsonAccessRepository
from billiards_ai.cloud.privacy.access_service import AccessDeniedError, MediaAccessService
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    AccessActorRole,
    AccessDecisionResult,
    AssetContext,
    MediaAssetKind,
    RetainedMediaAsset,
    ReviewerIdentity,
)
from billiards_ai.shared.models.privacy import MaterialUsePurpose

NOW = datetime(2026, 7, 28, 15, 30, tzinfo=UTC)


class FixedIdentityProvider:
    def current_identity(self) -> ReviewerIdentity:
        return ReviewerIdentity(
            principal_id="access-admin-1",
            roles=(AccessActorRole.ACCESS_ADMIN, AccessActorRole.SYSTEM),
            credential_id="phase0-access-admin",
        )


def _asset(kind: MediaAssetKind) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=kind,
        storage_locator=f"objects/{uuid4()}.bin",
        content_sha256=uuid4().hex * 2,
        captured_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _context(asset, approved) -> AssetContext:
    return AssetContext(
        asset_id=str(asset.asset_id),
        asset_sha256=asset.content_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=asset.captured_at,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )


def _service(repo, tmp_path, approved_gate_factory):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    access_repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    service = MediaAccessService(
        access_repository=access_repository,
        retention_service=MediaRetentionService(
            repo,
            gate_repository=gate_service.repository,
            now=lambda: NOW,
        ),
        identity_provider=FixedIdentityProvider(),
        now=lambda: NOW,
    )
    return service, access_repository, approved


def _approve(service: MediaAccessService):
    return service.approve_third_party_processing(
        provider_id="provider-cn-1",
        endpoint_region="cn-shanghai",
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        cross_border=False,
        cross_border_assessment="DOMESTIC_ONLY",
        max_cost_cny=Decimal("2.50"),
        approval_evidence_sha256="c" * 64,
    )


def test_exact_keyframe_approval_produces_decision_without_network(
    repo, tmp_path, approved_gate_factory, monkeypatch
) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset(MediaAssetKind.KEYFRAME)
    repo._insert_asset_for_testing(item)
    _approve(service)
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *args, **kwargs: pytest.fail("third-party guard must not use network"),
    )

    decision = service.authorize_third_party_keyframe_transfer(
        asset_id=item.asset_id,
        gate_revision_id=approved.revision_id,
        asset_context=_context(item, approved),
        reason="adjudicate ambiguous candidate",
        provider_id="provider-cn-1",
        endpoint_region="cn-shanghai",
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        cross_border=False,
        estimated_cost_cny=Decimal("1.25"),
    )

    assert decision.result is AccessDecisionResult.ALLOW
    assert access_repository.list_events()[-1].result is AccessDecisionResult.ALLOW


@pytest.mark.parametrize(
    "override",
    [
        {"provider_id": "other"},
        {"endpoint_region": "us-west"},
        {"purpose": MaterialUsePurpose.TRAINING},
        {"cross_border": True},
        {"estimated_cost_cny": Decimal("2.51")},
    ],
)
def test_third_party_mismatch_is_denied_and_audited(
    repo, tmp_path, approved_gate_factory, override
) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset(MediaAssetKind.KEYFRAME)
    repo._insert_asset_for_testing(item)
    _approve(service)
    arguments = {
        "asset_id": item.asset_id,
        "gate_revision_id": approved.revision_id,
        "asset_context": _context(item, approved),
        "reason": "adjudicate ambiguous candidate",
        "provider_id": "provider-cn-1",
        "endpoint_region": "cn-shanghai",
        "purpose": MaterialUsePurpose.SERVICE_ANALYSIS,
        "cross_border": False,
        "estimated_cost_cny": Decimal("1.25"),
    }
    arguments.update(override)

    with pytest.raises(AccessDeniedError):
        service.authorize_third_party_keyframe_transfer(**arguments)

    assert access_repository.list_events()[-1].result is AccessDecisionResult.DENY


def test_video_cannot_be_disguised_as_keyframe(repo, tmp_path, approved_gate_factory) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset(MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _approve(service)

    with pytest.raises(AccessDeniedError):
        service.authorize_third_party_keyframe_transfer(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason="adjudicate ambiguous candidate",
            provider_id="provider-cn-1",
            endpoint_region="cn-shanghai",
            purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
            cross_border=False,
            estimated_cost_cny=Decimal("1.25"),
        )

    assert access_repository.list_events()[-1].result is AccessDecisionResult.DENY


def test_missing_or_revoked_approval_is_denied(repo, tmp_path, approved_gate_factory) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset(MediaAssetKind.KEYFRAME)
    repo._insert_asset_for_testing(item)
    arguments = dict(
        asset_id=item.asset_id,
        gate_revision_id=approved.revision_id,
        asset_context=_context(item, approved),
        reason="adjudicate ambiguous candidate",
        provider_id="provider-cn-1",
        endpoint_region="cn-shanghai",
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        cross_border=False,
        estimated_cost_cny=Decimal("1.25"),
    )
    with pytest.raises(AccessDeniedError):
        service.authorize_third_party_keyframe_transfer(**arguments)

    approval = _approve(service)
    access_repository.revoke_provider_approval(
        approval.approval_id,
        revoked_at=NOW,
        revocation_reason="approval withdrawn",
    )
    with pytest.raises(AccessDeniedError):
        service.authorize_third_party_keyframe_transfer(**arguments)
