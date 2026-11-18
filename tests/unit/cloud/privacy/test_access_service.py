from datetime import UTC, datetime
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.access_repository import (
    AccessRepositoryError,
    JsonAccessRepository,
)
from billiards_ai.cloud.privacy.access_service import AccessDeniedError, MediaAccessService
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    AccessAction,
    AccessActorRole,
    AccessDecisionResult,
    AssetContext,
    MediaAssetKind,
    MediaDeletionStatus,
    RetainedMediaAsset,
    ReviewerIdentity,
)

NOW = datetime(2026, 7, 28, 15, 30, tzinfo=UTC)


class FixedIdentityProvider:
    def __init__(self, identity: ReviewerIdentity):
        self.identity = identity

    def current_identity(self) -> ReviewerIdentity:
        return self.identity


class BrokenAuditRepository(JsonAccessRepository):
    def append_event(self, event):
        raise AccessRepositoryError("audit unavailable")


class BrokenIdentityProvider:
    def current_identity(self):
        raise AccessRepositoryError("identity unavailable")


def _identity(*roles: AccessActorRole) -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-1",
        roles=roles or (AccessActorRole.REVIEWER, AccessActorRole.ACCESS_ADMIN),
        credential_id="phase0-reviewer",
    )


def _asset(session_id=None) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=session_id or uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator=f"objects/{uuid4()}.mp4",
        content_sha256=uuid4().hex * 2,
        captured_at=NOW,
        deletion_status=MediaDeletionStatus.ACTIVE,
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


def _service(repo, tmp_path, approved_gate_factory, *, access_repository=None, roles=()):
    gate_service, _, approved = approved_gate_factory(tmp_path)
    access_repository = access_repository or JsonAccessRepository(
        tmp_path / "access" / "store.json"
    )
    retention = MediaRetentionService(
        repo,
        gate_repository=gate_service.repository,
        now=lambda: NOW,
    )
    service = MediaAccessService(
        access_repository=access_repository,
        retention_service=retention,
        identity_provider=FixedIdentityProvider(_identity(*roles)),
        now=lambda: NOW,
    )
    return service, access_repository, approved


def test_exact_session_grant_allows_only_after_audit_is_persisted(
    repo, tmp_path, approved_gate_factory
) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset()
    repo._insert_asset_for_testing(item)
    service.grant_current_reviewer(session_id=item.session_id, reason="review assigned session")
    observed = []

    result = service.authorize_and_record_access(
        asset_id=item.asset_id,
        gate_revision_id=approved.revision_id,
        asset_context=_context(item, approved),
        reason="inspect low-confidence shot",
        action=AccessAction.CONTROLLED_REVIEW_VIEW,
        consumer=lambda locator: observed.append((locator, len(access_repository.list_events()))),
    )

    assert result is None
    assert observed == [(item.storage_locator, 1)]
    event = access_repository.list_events()[0]
    assert event.result is AccessDecisionResult.ALLOW
    assert event.session_id == item.session_id
    assert event.actor_id == "reviewer-1"
    assert event.occurred_at == NOW


@pytest.mark.parametrize(
    ("grant_session", "action"),
    [
        (None, AccessAction.CONTROLLED_REVIEW_VIEW),
        (uuid4(), AccessAction.CONTROLLED_REVIEW_VIEW),
        ("actual", AccessAction.EXPORT_OR_DOWNLOAD),
    ],
)
def test_missing_or_wrong_grant_and_export_are_denied_and_audited(
    repo, tmp_path, approved_gate_factory, grant_session, action
) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset()
    repo._insert_asset_for_testing(item)
    if grant_session is not None:
        service.grant_current_reviewer(
            session_id=item.session_id if grant_session == "actual" else grant_session,
            reason="assignment",
        )
    consumed = []

    with pytest.raises(AccessDeniedError) as caught:
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason="review attempt",
            action=action,
            consumer=consumed.append,
        )

    assert consumed == []
    assert caught.value.decision.result is AccessDecisionResult.DENY
    event = access_repository.list_events()[-1]
    assert event.security_event is True
    assert event.session_id == item.session_id


def test_stale_gate_is_denied_after_exact_grant(repo, tmp_path, approved_gate_factory) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset()
    repo._insert_asset_for_testing(item)
    service.grant_current_reviewer(session_id=item.session_id, reason="assignment")

    with pytest.raises(AccessDeniedError):
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=uuid4(),
            asset_context=_context(item, approved),
            reason="review attempt",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=lambda _: pytest.fail("consumer must not be called"),
        )

    assert access_repository.list_events()[-1].result is AccessDecisionResult.DENY


@pytest.mark.parametrize("reason", ["", "   ", "bad\nreason"])
def test_invalid_reason_is_denied_with_stable_code(
    repo, tmp_path, approved_gate_factory, reason
) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset()
    repo._insert_asset_for_testing(item)
    service.grant_current_reviewer(session_id=item.session_id, reason="assignment")

    with pytest.raises(AccessDeniedError) as caught:
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason=reason,
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=lambda _: pytest.fail("invalid reason must not release media"),
        )

    assert caught.value.decision.reason_code.value == "INVALID_REASON"
    assert access_repository.list_events()[-1].reason_code.value == "INVALID_REASON"


def test_revoked_grant_cannot_release_media(repo, tmp_path, approved_gate_factory) -> None:
    service, access_repository, approved = _service(repo, tmp_path, approved_gate_factory)
    item = _asset()
    repo._insert_asset_for_testing(item)
    grant = service.grant_current_reviewer(session_id=item.session_id, reason="assignment")
    service.revoke_grant(grant_id=grant.grant_id, reason="assignment complete")

    with pytest.raises(AccessDeniedError) as caught:
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason="late retry",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=lambda _: pytest.fail("revoked grant must not release media"),
        )

    assert caught.value.decision.reason_code.value == "GRANT_REVOKED"
    assert access_repository.list_events()[-1].security_event is True


def test_audit_failure_never_releases_locator(repo, tmp_path, approved_gate_factory) -> None:
    broken = BrokenAuditRepository(tmp_path / "access" / "store.json")
    service, _, approved = _service(
        repo,
        tmp_path,
        approved_gate_factory,
        access_repository=broken,
    )
    item = _asset()
    repo._insert_asset_for_testing(item)
    service.grant_current_reviewer(session_id=item.session_id, reason="assignment")
    consumed = []

    with pytest.raises(AccessRepositoryError):
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason="review attempt",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=consumed.append,
        )

    assert consumed == []


def test_missing_identity_is_denied_as_audited_security_event(
    repo, tmp_path, approved_gate_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    access_repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    item = _asset()
    repo._insert_asset_for_testing(item)
    service = MediaAccessService(
        access_repository=access_repository,
        retention_service=MediaRetentionService(
            repo, gate_repository=gate_service.repository, now=lambda: NOW
        ),
        identity_provider=BrokenIdentityProvider(),
        now=lambda: NOW,
    )
    consumed = []

    with pytest.raises(AccessDeniedError) as caught:
        service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=_context(item, approved),
            reason="review attempt",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=consumed.append,
        )

    assert caught.value.decision.reason_code.value == "IDENTITY_UNAVAILABLE"
    assert consumed == []
    event = access_repository.list_events()[-1]
    assert event.actor_role is AccessActorRole.SYSTEM
    assert event.security_event is True


def test_only_reviewer_can_view_and_only_admin_can_grant(
    repo, tmp_path, approved_gate_factory
) -> None:
    reviewer_only, _, _ = _service(
        repo,
        tmp_path,
        approved_gate_factory,
        roles=(AccessActorRole.REVIEWER,),
    )
    with pytest.raises(AccessDeniedError):
        reviewer_only.grant_current_reviewer(session_id=uuid4(), reason="self grant")

    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    other.chmod(0o700)
    admin_only, _, _ = _service(
        repo,
        other,
        approved_gate_factory,
        roles=(AccessActorRole.ACCESS_ADMIN,),
    )
    item = _asset()
    repo._insert_asset_for_testing(item)
    admin_only.grant_current_reviewer(session_id=item.session_id, reason="assignment")
    with pytest.raises(AccessDeniedError):
        admin_only.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=uuid4(),
            asset_context=AssetContext(
                asset_id=str(item.asset_id),
                asset_sha256=item.content_sha256,
                gate_scope_fingerprint="a" * 64,
                captured_at=NOW,
                captured_under_revision_id=uuid4(),
                bystanders_present=False,
            ),
            reason="review",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=lambda _: None,
        )
