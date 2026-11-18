import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.retention_repository import (
    JsonRetentionRepository,
    RetentionConflictError,
    RetentionRepositoryError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    DeletionScope,
    MediaAssetKind,
    RetainedMediaAsset,
)


def test_repository_writes_private_versioned_store(tmp_path: Path) -> None:
    path = tmp_path / "private" / "retention.json"
    repo = JsonRetentionRepository(path)
    assert repo.list_assets() == ()
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_corrupt_or_symlink_store_fails_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not-json", encoding="utf-8")
    os.chmod(corrupt, 0o600)
    with pytest.raises(RetentionRepositoryError):
        JsonRetentionRepository(corrupt).list_assets()

    target = tmp_path / "target.json"
    target.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    os.chmod(target, 0o600)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RetentionRepositoryError):
        JsonRetentionRepository(link).list_assets()


def test_concurrent_idempotent_submission_creates_one_request(tmp_path: Path) -> None:
    repo = JsonRetentionRepository(tmp_path / "retention" / "store.json")
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="session/video.mp4",
        content_sha256="a" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    service = MediaRetentionService(repo, now=lambda: now)

    def submit():
        return service.submit_deletion(
            current_user_id="user-1",
            scope=DeletionScope(asset_ids=(asset.asset_id,)),
            idempotency_key="concurrent",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = tuple(pool.map(lambda _: submit(), range(24)))
    assert len({item.request_id for item in requests}) == 1
    assert len(repo.list_requests()) == 1


def test_repository_does_not_chmod_unsafe_parent_directory(tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    with pytest.raises(RetentionRepositoryError):
        JsonRetentionRepository(parent / "store.json").list_assets()
    assert parent.stat().st_mode & 0o777 == 0o755


def test_directory_fsync_failure_after_replace_does_not_lie_about_commit(
    tmp_path: Path, monkeypatch
) -> None:
    repo = JsonRetentionRepository(tmp_path / "retention" / "store.json")
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="session/video.mp4",
        content_sha256="b" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(
        "billiards_ai.cloud.privacy.retention_repository.os.fsync",
        fail_directory_fsync,
    )
    with pytest.warns(RuntimeWarning, match="directory fsync failed"):
        assert repo._insert_asset_for_testing(asset) == asset
    assert repo.get_asset(asset.asset_id) == asset


def test_repository_cannot_rewrite_existing_milestone_anchor(tmp_path: Path) -> None:
    repo = JsonRetentionRepository(tmp_path / "retention" / "store.json")
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    original = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="session/video.mp4",
        content_sha256="c" * 64,
        captured_at=now,
        reviewed_at=now + timedelta(minutes=1),
        created_at=now,
        updated_at=now + timedelta(minutes=1),
    )
    repo._insert_asset_for_testing(original)
    rewritten = RetainedMediaAsset.model_validate(
        original.model_copy(
            update={
                "reviewed_at": now + timedelta(minutes=2),
                "updated_at": now + timedelta(minutes=2),
            }
        ).model_dump()
    )
    with pytest.raises(RetentionConflictError, match="milestone"):
        repo.save_asset(rewritten, expected_current=original)


def test_processes_share_the_same_flock_idempotency_boundary(tmp_path: Path) -> None:
    store = tmp_path / "retention" / "store.json"
    repo = JsonRetentionRepository(store)
    now = datetime(2026, 7, 27, 18, tzinfo=UTC)
    asset = RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="process/video.mp4",
        content_sha256="d" * 64,
        captured_at=now,
        created_at=now,
        updated_at=now,
    )
    repo._insert_asset_for_testing(asset)
    script = """
import sys
from uuid import UUID
from billiards_ai.cloud.privacy.retention_repository import JsonRetentionRepository
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import DeletionScope
repo = JsonRetentionRepository(sys.argv[1])
request = MediaRetentionService(repo).submit_deletion(
    current_user_id="user-1",
    scope=DeletionScope(asset_ids=(UUID(sys.argv[2]),)),
    idempotency_key="cross-process",
)
print(request.request_id)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(store), str(asset.asset_id)],
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip())
    assert len(set(outputs)) == 1
    assert len(repo.list_requests()) == 1
