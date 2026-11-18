import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.deletion_cli import main
from billiards_ai.cloud.privacy.media_store import ControlledFilesystemMediaStore
from billiards_ai.cloud.privacy.retention_repository import (
    JsonRetentionRepository,
    RetentionRepositoryError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    DeletionRequest,
    DeletionRequestStatus,
    DeletionScope,
    MediaAssetKind,
    RetainedMediaAsset,
)


def _put_synthetic(media, locator: str) -> None:
    path = media.root / locator
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")
    path.chmod(0o600)


def test_status_cli_outputs_public_projection_without_locator(tmp_path, capsys) -> None:
    store = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(store)
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="secret/video.mp4",
        content_sha256="a" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    assert (
        main(
            [
                "--store",
                str(store),
                "submit",
                "--user",
                "user-1",
                "--asset-id",
                str(asset.asset_id),
                "--idempotency-key",
                "cli-key",
            ]
        )
        == 0
    )
    submitted = json.loads(capsys.readouterr().out)
    assert "secret" not in json.dumps(submitted)
    assert (
        main(
            [
                "--store",
                str(store),
                "status",
                "--user",
                "user-1",
                "--request-id",
                submitted["request_id"],
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "RECEIVED"
    assert "locator" not in json.dumps(status)


def test_verify_requires_owner_and_uses_public_projection(tmp_path, capsys) -> None:
    store = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(store)
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="secret/video.mp4",
        content_sha256="b" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    request = MediaRetentionService(repo, now=lambda: now).submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(asset.asset_id,)),
        idempotency_key="verify-owner",
    )
    assert (
        main(
            [
                "--store",
                str(store),
                "verify",
                "--user",
                "user-2",
                "--request-id",
                str(request.request_id),
            ]
        )
        == 2
    )
    assert "evidence_summary" not in capsys.readouterr().err


def test_list_due_requires_explicit_owner_filter(tmp_path) -> None:
    store = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(store)
    captured = datetime(2026, 6, 1, 18, tzinfo=UTC)
    anchor = captured + timedelta(minutes=1)
    repo._insert_asset_for_testing(
        RetainedMediaAsset(
            asset_id=uuid4(),
            session_id=uuid4(),
            owner_user_id="user-1",
            kind=MediaAssetKind.VIDEO,
            storage_locator="secret/due.mp4",
            content_sha256="c" * 64,
            captured_at=captured,
            reviewed_at=anchor,
            delivered_at=anchor,
            retention_expires_at=anchor + timedelta(days=30),
            created_at=captured,
            updated_at=anchor,
        )
    )
    with pytest.raises(SystemExit):
        main(["--store", str(store), "list-due"])


def test_execute_verify_and_owner_filtered_list_due_cli(
    tmp_path, capsys, monkeypatch, controlled_media_store_factory
) -> None:
    store = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(store)
    media = controlled_media_store_factory(tmp_path)
    monkeypatch.setenv(ControlledFilesystemMediaStore.ENVIRONMENT_VARIABLE, str(media.root))
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="cli/video.mp4",
        content_sha256="d" * 64,
        captured_at=now,
        reviewed_at=now,
        delivered_at=now,
        retention_expires_at=now + timedelta(days=30),
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    _put_synthetic(media, asset.storage_locator)
    service = MediaRetentionService(repo, now=lambda: now)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(asset.asset_id,)),
        idempotency_key="cli-execute",
    )
    assert (
        main(
            [
                "--store",
                str(store),
                "execute",
                "--request-id",
                str(request.request_id),
                "--executor",
                "Boyu",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETED"
    assert (
        main(
            [
                "--store",
                str(store),
                "verify",
                "--user",
                "user-1",
                "--request-id",
                str(request.request_id),
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "COMPLETED"
    assert "evidence" not in json.dumps(verified)
    assert main(["--store", str(store), "list-due", "--owner", "user-1"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_retry_and_reclaim_cli(
    tmp_path, capsys, monkeypatch, controlled_media_store_factory
) -> None:
    class FailingStore(ControlledFilesystemMediaStore):
        def delete(self, locator: str) -> None:
            raise RetentionRepositoryError("synthetic failure")

    store = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(store)
    failing_media = controlled_media_store_factory(tmp_path, FailingStore)
    monkeypatch.setenv(ControlledFilesystemMediaStore.ENVIRONMENT_VARIABLE, str(failing_media.root))
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    first = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="cli/retry.mp4",
        content_sha256="e" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(first)
    _put_synthetic(failing_media, first.storage_locator)
    failing_service = MediaRetentionService(repo, media_store=failing_media, now=lambda: now)
    retry_request = failing_service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(first.asset_id,)),
        idempotency_key="cli-retry",
    )
    assert (
        failing_service.execute_deletion(
            request_id=retry_request.request_id, executor_id="Boyu"
        ).status
        is DeletionRequestStatus.FAILED
    )
    assert (
        main(
            [
                "--store",
                str(store),
                "retry",
                "--request-id",
                str(retry_request.request_id),
                "--executor",
                "Boyu",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETED"

    second = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="cli/reclaim.mp4",
        content_sha256="f" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(second)
    service = MediaRetentionService(repo, now=lambda: now)
    reclaim_request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(second.asset_id,)),
        idempotency_key="cli-reclaim",
    )
    repo.save_request(
        DeletionRequest.model_validate(
            reclaim_request.model_copy(
                update={"status": DeletionRequestStatus.IN_PROGRESS}
            ).model_dump()
        )
    )
    assert (
        main(
            [
                "--store",
                str(store),
                "reclaim",
                "--request-id",
                str(reclaim_request.request_id),
                "--executor",
                "Boyu",
                "--reason",
                "previous process confirmed stopped",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "FAILED"
