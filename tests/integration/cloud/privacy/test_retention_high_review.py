from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy import media_store as media_store_module
from billiards_ai.cloud.privacy.media_store import ControlledFilesystemMediaStore
from billiards_ai.cloud.privacy.retention_repository import (
    RetentionConflictError,
    RetentionRepositoryError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    DeletionRequestStatus,
    DeletionScope,
    MediaAssetKind,
    RetainedMediaAsset,
)

NOW = datetime(2026, 7, 27, 18, tzinfo=UTC)


def _asset(*, owner="user-1", locator=None) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id=owner,
        kind=MediaAssetKind.VIDEO,
        storage_locator=locator or f"high/{uuid4()}.mp4",
        content_sha256=uuid4().hex * 2,
        captured_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _put(media: ControlledFilesystemMediaStore, asset: RetainedMediaAsset) -> None:
    target = media.root / asset.storage_locator
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_bytes(b"synthetic")
    target.chmod(0o600)


@pytest.mark.parametrize("reference_kind", ["asset_id", "hash", "locator", "url"])
def test_aggregate_registration_rejects_every_unindexed_media_reference(
    repo, reference_kind
) -> None:
    item = _asset()
    repo._insert_asset_for_testing(item)
    references = {
        "asset_id": str(item.asset_id),
        "hash": item.content_sha256,
        "locator": item.storage_locator,
        "url": f"https://media.example/{item.storage_locator}",
    }

    with pytest.raises(RetentionConflictError):
        repo.register_aggregate(
            "unindexed",
            "user-1",
            {"media_ref": references[reference_kind]},
            (),
            media_ref_paths=(),
        )


def test_aggregate_registration_validates_asset_existence_owner_and_pointer_index(repo) -> None:
    owned = _asset()
    foreign = _asset(owner="user-2")
    repo._insert_asset_for_testing(owned)
    repo._insert_asset_for_testing(foreign)
    with pytest.raises(RetentionConflictError):
        repo.register_aggregate(
            "missing",
            "user-1",
            {"asset": str(uuid4())},
            (uuid4(),),
            media_ref_paths=("/asset",),
        )
    with pytest.raises(RetentionConflictError):
        repo.register_aggregate(
            "foreign",
            "user-1",
            {"asset": str(foreign.asset_id)},
            (foreign.asset_id,),
            media_ref_paths=("/asset",),
        )
    with pytest.raises(RetentionConflictError):
        repo.register_aggregate(
            "mismatch",
            "user-1",
            {"asset": str(foreign.asset_id)},
            (owned.asset_id,),
            media_ref_paths=("/asset",),
        )


def test_non_deleted_locator_is_globally_unique_across_owners(repo) -> None:
    first = _asset(locator="shared/object.mp4")
    second = _asset(owner="user-2", locator="shared/object.mp4")
    repo._insert_asset_for_testing(first)

    with pytest.raises(RetentionConflictError):
        repo._insert_asset_for_testing(second)


@pytest.mark.parametrize("failure_site", ["resolve", "expand"])
def test_post_claim_scope_failure_records_typed_failed_attempt(
    repo, tmp_path, controlled_media_store_factory, monkeypatch, failure_site
) -> None:
    media = controlled_media_store_factory(tmp_path)
    item = _asset()
    repo._insert_asset_for_testing(item)
    _put(media, item)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key=f"scope-failure-{failure_site}",
    )
    if failure_site == "resolve":
        monkeypatch.setattr(
            service,
            "_resolve_scope",
            lambda *_: (_ for _ in ()).throw(
                RetentionRepositoryError("synthetic scope recompute failure")
            ),
        )
    else:
        monkeypatch.setattr(
            repo,
            "expand_in_progress_scope",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RetentionConflictError("synthetic scope CAS failure")
            ),
        )

    failed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")

    assert failed.status is DeletionRequestStatus.FAILED
    assert failed.attempts[-1].failure_code.value == "INVALID_STATE"
    assert failed.attempts[-1].planned_asset_count == 1
    assert failed.attempts[-1].verified_deleted_count == 0


def test_failed_attempt_reports_verified_count_reached_before_scrub_failure(
    repo, tmp_path, controlled_media_store_factory, monkeypatch
) -> None:
    media = controlled_media_store_factory(tmp_path)
    items = (_asset(), _asset())
    for item in items:
        repo._insert_asset_for_testing(item)
        _put(media, item)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=tuple(item.asset_id for item in items)),
        idempotency_key="verified-before-scrub-failure",
    )
    monkeypatch.setattr(
        repo,
        "scrub_media_references",
        lambda *_: (_ for _ in ()).throw(RetentionRepositoryError("synthetic scrub failure")),
    )

    failed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")

    assert failed.status is DeletionRequestStatus.FAILED
    assert failed.attempts[-1].verified_deleted_count == 2


def test_delete_is_safe_when_intermediate_directory_is_replaced_by_symlink(
    tmp_path, controlled_media_store_factory, monkeypatch
) -> None:
    media = controlled_media_store_factory(tmp_path)
    nested = media.root / "nested"
    nested.mkdir(mode=0o700)
    target = nested / "video.mp4"
    target.write_bytes(b"inside")
    target.chmod(0o600)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "video.mp4"
    outside_target.write_bytes(b"outside")
    parked = media.root / "parked"
    real_unlink = Path.unlink
    swapped = False

    def racing_unlink(path: Path, *args, **kwargs):
        nonlocal swapped
        if path == target and not swapped:
            swapped = True
            nested.rename(parked)
            nested.symlink_to(outside, target_is_directory=True)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(media_store_module.Path, "unlink", racing_unlink)
    media.delete("nested/video.mp4")

    assert outside_target.read_bytes() == b"outside"
