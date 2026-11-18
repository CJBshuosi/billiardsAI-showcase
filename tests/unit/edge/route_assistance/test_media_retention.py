from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.media_retention import (
    EdgeDeletionReceipt,
    EdgeDeletionTask,
    EdgeRetentionClient,
    HttpResult,
    LocalMediaRecord,
    SQLiteDeletionReceiptStore,
    execute_local_deletion,
    process_next_deletion,
)


def _record(path: Path, content: bytes, **changes: object) -> LocalMediaRecord:
    values: dict[str, object] = {
        "clip_id": "clip-1",
        "local_path": path,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "sync_status": "complete",
        "source_kind": "synthetic_retention_test",
    }
    values.update(changes)
    return LocalMediaRecord(**values)  # type: ignore[arg-type]


def _task(**changes: object) -> EdgeDeletionTask:
    values: dict[str, object] = {
        "job_id": "delete-1",
        "clip_id": "clip-1",
        "device_id": "device-1",
        "content_sha256": hashlib.sha256(b"synthetic-video").hexdigest(),
        "claim_token": "a" * 64,
    }
    values.update(changes)
    return EdgeDeletionTask(**values)  # type: ignore[arg-type]


def test_local_deletion_is_default_off_and_idempotently_verifies_absence(tmp_path: Path) -> None:
    media_root = tmp_path / "clips"
    media_root.mkdir()
    clip = media_root / "clip-1.mp4"
    clip.write_bytes(b"synthetic-video")
    record = _record(clip, b"synthetic-video")

    with pytest.raises(ValueError, match="disabled"):
        execute_local_deletion(_task(), record, media_root=media_root, device_id="device-1")
    assert clip.exists()

    receipt = execute_local_deletion(
        _task(),
        record,
        media_root=media_root,
        device_id="device-1",
        execution_mode="synthetic-only",
    )
    assert receipt.outcome == "deleted"
    assert not clip.exists()
    repeated = execute_local_deletion(
        _task(),
        record,
        media_root=media_root,
        device_id="device-1",
        execution_mode="synthetic-only",
    )
    assert repeated.outcome == "already_absent"


@pytest.mark.parametrize("case", ["outside", "symlink", "hash", "status", "source", "device"])
def test_local_deletion_rejects_untrusted_authority(tmp_path: Path, case: str) -> None:
    media_root = tmp_path / "clips"
    media_root.mkdir()
    inside = media_root / "clip.mp4"
    inside.write_bytes(b"synthetic-video")
    path = inside
    record_changes: dict[str, object] = {}
    task_changes: dict[str, object] = {}
    device_id = "device-1"
    if case == "outside":
        path = tmp_path / "outside.mp4"
        path.write_bytes(b"synthetic-video")
    elif case == "symlink":
        target = tmp_path / "target.mp4"
        target.write_bytes(b"synthetic-video")
        inside.unlink()
        inside.symlink_to(target)
    elif case == "hash":
        task_changes["content_sha256"] = "b" * 64
    elif case == "status":
        record_changes["sync_status"] = "uploaded"
    elif case == "source":
        record_changes["source_kind"] = "recorded_replay"
    else:
        device_id = "other-device"
    record = _record(path, b"synthetic-video", **record_changes)
    with pytest.raises(ValueError):
        execute_local_deletion(
            _task(**task_changes),
            record,
            media_root=media_root,
            device_id=device_id,
            execution_mode="synthetic-only",
        )
    assert path.exists() or path.is_symlink()


def test_receipt_store_is_durable_and_rejects_conflicting_replay(tmp_path: Path) -> None:
    store_path = tmp_path / "deletion-receipts.sqlite3"
    store = SQLiteDeletionReceiptStore(store_path)
    receipt = execute_local_deletion(
        _task(),
        _record(tmp_path / "missing.mp4", b"synthetic-video"),
        media_root=tmp_path,
        device_id="device-1",
        execution_mode="synthetic-only",
    )
    store.save(receipt)
    assert SQLiteDeletionReceiptStore(store_path).load("delete-1") == receipt
    with pytest.raises(ValueError, match="conflicting"):
        store.save(replace(receipt, clip_id="other"))


def test_retention_client_uses_signed_device_routes_without_leaking_secret() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.requests: list[tuple[str, dict[str, str], bytes]] = []

        def post_json(self, path: str, headers: dict[str, str], body: bytes) -> HttpResult:
            self.requests.append((path, headers, body))
            if path.endswith("/claim"):
                return HttpResult(
                    200,
                    (
                        b'{"data":{"schema_version":"1.0","job_id":"delete-1",'
                        b'"clip_id":"clip-1","content_sha256":"'
                        + b"a" * 64
                        + b'","claim_token":"'
                        + b"b" * 64
                        + b'"}}'
                    ),
                )
            return HttpResult(200, b'{"data":{"status":"completed"}}')

    transport = Transport()
    client = EdgeRetentionClient(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(("nonce-claim", "nonce-complete")).__next__,
    )
    task = client.claim_next()
    assert task == EdgeDeletionTask(
        job_id="delete-1",
        clip_id="clip-1",
        device_id="device-1",
        content_sha256="a" * 64,
        claim_token="b" * 64,
    )
    receipt = EdgeDeletionReceipt(
        job_id="delete-1",
        clip_id="clip-1",
        device_id="device-1",
        claim_token="b" * 64,
        outcome="deleted",
        completed_at_utc="now",
    )
    client.complete(receipt)
    assert [request[0] for request in transport.requests] == [
        "/v1/media-deletions/claim",
        "/v1/media-deletions/delete-1/complete",
    ]
    assert "fixture-secret" not in repr(transport.requests)


def test_processing_persists_receipt_before_network_completion(tmp_path: Path) -> None:
    content = b"synthetic-video"
    media_root = tmp_path / "clips"
    media_root.mkdir()
    clip = media_root / "clip.mp4"
    clip.write_bytes(content)
    record = _record(clip, content)
    task = _task()

    class Client:
        def claim_next(self) -> EdgeDeletionTask:
            return task

        def complete(self, _receipt: EdgeDeletionReceipt) -> None:
            raise OSError("offline")

    store = SQLiteDeletionReceiptStore(tmp_path / "receipts.sqlite3")
    with pytest.raises(OSError, match="offline"):
        process_next_deletion(
            client=Client(),
            record_lookup=lambda _clip_id: record,
            receipt_store=store,
            media_root=media_root,
            device_id="device-1",
            execution_mode="synthetic-only",
        )
    assert not clip.exists()
    assert store.load(task.job_id) is not None
