from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.retention_repository import (
    MediaUnavailableError,
    RetentionConflictError,
    RetentionOwnershipError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.constants import CLIP_RETENTION_DAYS
from billiards_ai.shared.models import (
    AssetContext,
    DeletionRequestStatus,
    DeletionScope,
    MaterialUsePurpose,
    MediaAssetKind,
    RetainedMediaAsset,
    RetainedMediaAssetCreate,
)

NOW = datetime(2026, 7, 27, 18, tzinfo=UTC)


def _put_synthetic(media, locator: str) -> None:
    path = media.root / locator
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")
    path.chmod(0o600)


def asset(*, owner: str = "user-1", session_id=None, sources=()) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=session_id or uuid4(),
        owner_user_id=owner,
        kind=MediaAssetKind.VIDEO if not sources else MediaAssetKind.FRAME,
        storage_locator=f"objects/{uuid4()}.bin",
        content_sha256=uuid4().hex * 2,
        source_asset_ids=sources,
        captured_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def asset_create(item: RetainedMediaAsset) -> RetainedMediaAssetCreate:
    return RetainedMediaAssetCreate.model_validate(
        item.model_dump(
            include={
                "asset_id",
                "session_id",
                "owner_scope",
                "kind",
                "storage_locator",
                "content_sha256",
                "captured_at",
                "source_asset_ids",
            }
        )
    )


def test_retention_starts_after_both_milestones_and_never_silently_extends(repo) -> None:
    item = asset()
    repo._insert_asset_for_testing(item)
    service = MediaRetentionService(repo, now=lambda: NOW + timedelta(minutes=4))
    reviewed = service.record_reviewed(item.asset_id, NOW + timedelta(minutes=1))
    assert reviewed.retention_expires_at is None
    delivered = service.record_delivered(item.asset_id, NOW + timedelta(minutes=2))
    expected = NOW + timedelta(minutes=2, days=CLIP_RETENTION_DAYS)
    assert delivered.retention_expires_at == expected
    unchanged = service.record_reviewed(item.asset_id, NOW + timedelta(minutes=3))
    assert unchanged.retention_expires_at == expected


def test_future_milestone_is_rejected(repo) -> None:
    item = asset()
    repo._insert_asset_for_testing(item)
    service = MediaRetentionService(repo, now=lambda: NOW)
    with pytest.raises(ValueError, match="future"):
        service.record_delivered(item.asset_id, NOW + timedelta(minutes=6))


def test_submit_is_idempotent_and_key_reuse_with_other_scope_conflicts(repo) -> None:
    first, second = asset(), asset()
    repo._insert_asset_for_testing(first)
    repo._insert_asset_for_testing(second)
    service = MediaRetentionService(repo, now=lambda: NOW)
    one = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(first.asset_id,)),
        idempotency_key="same-key",
    )
    repeated = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(first.asset_id,)),
        idempotency_key="same-key",
    )
    assert repeated.request_id == one.request_id
    with pytest.raises(RetentionConflictError):
        service.submit_deletion(
            current_user_id="user-1",
            scope=DeletionScope(asset_ids=(second.asset_id,)),
            idempotency_key="same-key",
        )


def test_mixed_owner_scope_is_rejected_as_a_whole(repo) -> None:
    owned, foreign = asset(), asset(owner="user-2")
    repo._insert_asset_for_testing(owned)
    repo._insert_asset_for_testing(foreign)
    service = MediaRetentionService(repo, now=lambda: NOW)
    with pytest.raises(RetentionOwnershipError):
        service.submit_deletion(
            current_user_id="user-1",
            scope=DeletionScope(asset_ids=(owned.asset_id, foreign.asset_id)),
            idempotency_key="mixed",
        )
    assert repo.list_requests() == ()


def test_due_sweep_creates_one_repeatable_request(repo) -> None:
    item = asset()
    repo._insert_asset_for_testing(item)
    service = MediaRetentionService(repo, now=lambda: NOW)
    service.record_reviewed(item.asset_id, NOW)
    matured = service.record_delivered(item.asset_id, NOW)
    due_at = matured.retention_expires_at
    created = service.submit_due_deletions(at=due_at)
    assert len(created) == 1
    assert created[0].idempotency_key == f"retention-expiry:{item.asset_id}"
    assert service.submit_due_deletions(at=due_at) == ()


def test_empty_all_user_media_request_completes_immediately_and_idempotently(repo) -> None:
    service = MediaRetentionService(repo, now=lambda: NOW)
    first = service.submit_deletion(
        current_user_id="user-with-no-media",
        scope=DeletionScope(all_user_media=True),
        idempotency_key="empty-delete",
    )
    repeated = service.submit_deletion(
        current_user_id="user-with-no-media",
        scope=DeletionScope(all_user_media=True),
        idempotency_key="empty-delete",
    )
    assert first.status is DeletionRequestStatus.COMPLETED
    assert first.frozen_asset_ids == ()
    assert repeated.request_id == first.request_id


def test_real_media_registration_reuses_story_1_1_authorization(
    repo, tmp_path, approved_gate_factory, controlled_media_store_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    item = asset()
    context = AssetContext(
        asset_id=str(item.asset_id),
        asset_sha256=item.content_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=item.captured_at,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    service = MediaRetentionService(repo, gate_repository=gate_service.repository, now=lambda: NOW)
    assert (
        service.register_asset(
            asset_create(item),
            gate_revision_id=approved.revision_id,
            asset_context=context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )
        == item
    )
    media = controlled_media_store_factory(tmp_path)
    _put_synthetic(media, item.storage_locator)
    deletion_service = MediaRetentionService(
        repo,
        gate_repository=gate_service.repository,
        media_store=media,
        now=lambda: NOW,
    )
    request = deletion_service.submit_deletion(
        current_user_id=item.owner_user_id,
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="delete-before-late-upload",
    )
    pending_derivative = asset(session_id=item.session_id, sources=(item.asset_id,))
    pending_context = context.model_copy(
        update={
            "asset_id": str(pending_derivative.asset_id),
            "asset_sha256": pending_derivative.content_sha256,
        }
    )
    with pytest.raises(MediaUnavailableError):
        deletion_service.register_asset(
            asset_create(pending_derivative),
            gate_revision_id=approved.revision_id,
            asset_context=pending_context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )
    deletion_service.execute_deletion(request_id=request.request_id, executor_id="Boyu")

    late = asset(session_id=item.session_id)
    late_context = context.model_copy(
        update={
            "asset_id": str(late.asset_id),
            "asset_sha256": late.content_sha256,
            "captured_at": late.captured_at,
        }
    )
    with pytest.raises(MediaUnavailableError):
        deletion_service.register_asset(
            asset_create(late),
            gate_revision_id=approved.revision_id,
            asset_context=late_context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )
