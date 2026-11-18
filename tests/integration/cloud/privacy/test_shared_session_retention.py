from datetime import UTC, datetime
from uuid import uuid4

from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models.retention import (
    DeletionScope,
    MediaOwnerScope,
    RetainedMediaAsset,
    RetainedMediaAssetCreate,
)


def test_owner_scope_is_nonempty_unique_sorted_and_legacy_value_is_lossless() -> None:
    first, second = str(uuid4()), str(uuid4())
    scope = MediaOwnerScope(subject_user_ids=(second, first))
    assert scope.subject_user_ids == tuple(sorted((first, second)))
    assert MediaOwnerScope.from_legacy(" legacy-owner ").subject_user_ids == ("legacy-owner",)


def test_shared_asset_create_has_one_owner_scope_not_a_primary_owner() -> None:
    users = tuple(sorted((str(uuid4()), str(uuid4()))))
    asset = RetainedMediaAssetCreate(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_scope={"subject_user_ids": users},
        kind="VIDEO",
        storage_locator="objects/a.mp4",
        content_sha256="a" * 64,
        captured_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    assert asset.owner_scope.subject_user_ids == users
    assert "owner_user_id" not in asset.model_dump()


def test_either_subject_can_select_the_one_shared_asset_for_deletion(repo) -> None:
    users = tuple(sorted((str(uuid4()), str(uuid4()))))
    now = datetime(2026, 7, 29, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_scope=MediaOwnerScope(subject_user_ids=users),
        kind="VIDEO",
        storage_locator="objects/shared.mp4",
        content_sha256="b" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    request = MediaRetentionService(repo, now=lambda: now).submit_deletion(
        current_user_id=users[1],
        scope=DeletionScope(asset_ids=(asset.asset_id,)),
        idempotency_key="shared-subject-deletion",
    )
    assert request.frozen_asset_ids == (asset.asset_id,)
