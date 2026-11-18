"""Fail-closed deletion of media previously registered by the field sync ledger."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol


def _require_hash(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("media digest is invalid")
    return value


@dataclass(frozen=True, slots=True)
class EdgeDeletionTask:
    job_id: str
    clip_id: str
    device_id: str
    content_sha256: str
    claim_token: str

    def __post_init__(self) -> None:
        if not self.job_id or not self.clip_id or not self.device_id:
            raise ValueError("deletion task identity is invalid")
        _require_hash(self.content_sha256)
        _require_hash(self.claim_token)


@dataclass(frozen=True, slots=True)
class LocalMediaRecord:
    clip_id: str
    local_path: Path
    content_sha256: str
    sync_status: str
    source_kind: str


@dataclass(frozen=True, slots=True)
class EdgeDeletionReceipt:
    job_id: str
    clip_id: str
    device_id: str
    claim_token: str
    outcome: Literal["deleted", "already_absent"]
    completed_at_utc: str


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: bytes


class DeviceCredentialProvider(Protocol):
    def device_id(self) -> str: ...

    def secret(self) -> bytes: ...


class RetentionHttpTransport(Protocol):
    def post_json(self, path: str, headers: Mapping[str, str], body: bytes) -> HttpResult: ...


class EdgeRetentionClient:
    """Signed device client; cloud paths never cross this interface."""

    def __init__(
        self,
        *,
        credentials: DeviceCredentialProvider,
        transport: RetentionHttpTransport,
        now_seconds: Callable[[], int] = lambda: int(time.time()),
        nonce_factory: Callable[[], str] = lambda: f"nonce-{secrets.token_hex(16)}",
    ) -> None:
        self._credentials = credentials
        self._transport = transport
        self._now_seconds = now_seconds
        self._nonce_factory = nonce_factory

    def _post(self, path: str, payload: dict[str, object]) -> HttpResult:
        body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(self._now_seconds())
        nonce = self._nonce_factory()
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"POST\n{path}\n{timestamp}\n{nonce}\n{body_hash}".encode()
        secret = self._credentials.secret()
        if len(secret) < 8:
            raise OSError("device credential is invalid")
        signature = hmac.new(secret, canonical, hashlib.sha256).hexdigest()
        headers = {
            "content-type": "application/json",
            "x-device-id": self._credentials.device_id(),
            "x-timestamp": timestamp,
            "x-nonce": nonce,
            "x-body-sha256": body_hash,
            "x-signature": signature,
        }
        return self._transport.post_json(path, headers, body)

    def claim_next(self) -> EdgeDeletionTask | None:
        result = self._post("/v1/media-deletions/claim", {})
        if not 200 <= result.status_code < 300:
            raise OSError("media deletion claim failed")
        try:
            data = json.loads(result.body)["data"]
            if data is None:
                return None
            return EdgeDeletionTask(
                job_id=str(data["job_id"]),
                clip_id=str(data["clip_id"]),
                device_id=self._credentials.device_id(),
                content_sha256=str(data["content_sha256"]),
                claim_token=str(data["claim_token"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("media deletion claim response is invalid") from exc

    def complete(self, receipt: EdgeDeletionReceipt) -> None:
        if receipt.device_id != self._credentials.device_id():
            raise ValueError("media deletion receipt belongs to another device")
        result = self._post(
            f"/v1/media-deletions/{receipt.job_id}/complete",
            {
                "clip_id": receipt.clip_id,
                "claim_token": receipt.claim_token,
                "outcome": receipt.outcome,
            },
        )
        if not 200 <= result.status_code < 300:
            raise OSError("media deletion completion failed")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def execute_local_deletion(
    task: EdgeDeletionTask,
    record: LocalMediaRecord,
    *,
    media_root: Path,
    device_id: str,
    execution_mode: str = "disabled",
    now: datetime | None = None,
) -> EdgeDeletionReceipt:
    if execution_mode != "synthetic-only" or record.source_kind != "synthetic_retention_test":
        raise ValueError("physical deletion execution is disabled")
    if task.device_id != device_id or task.clip_id != record.clip_id:
        raise ValueError("deletion task does not match local authority")
    if record.sync_status != "complete" or task.content_sha256 != record.content_sha256:
        raise ValueError("local sync authority is incomplete")
    root = media_root.resolve(strict=True)
    candidate = record.local_path
    if not candidate.is_absolute() or not _is_within(candidate.parent.resolve(strict=False), root):
        raise ValueError("media path is outside the controlled root")
    outcome: Literal["deleted", "already_absent"]
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        outcome = "already_absent"
    else:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("media path is not a regular file")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != task.content_sha256:
            raise ValueError("local media digest does not match sync authority")
        os.unlink(candidate)
        if candidate.exists() or candidate.is_symlink():
            raise OSError("local media deletion could not be verified")
        outcome = "deleted"
    completed = (now or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
    return EdgeDeletionReceipt(
        job_id=task.job_id,
        clip_id=task.clip_id,
        device_id=device_id,
        claim_token=task.claim_token,
        outcome=outcome,
        completed_at_utc=completed,
    )


class SQLiteDeletionReceiptStore:
    """Durable idempotency ledger; it intentionally stores no media path."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS media_deletion_receipts (
                  job_id TEXT PRIMARY KEY,
                  clip_id TEXT NOT NULL,
                  device_id TEXT NOT NULL,
                  claim_token TEXT NOT NULL,
                  outcome TEXT NOT NULL CHECK(outcome IN ('deleted','already_absent')),
                  completed_at_utc TEXT NOT NULL
                )"""
            )

    def save(self, receipt: EdgeDeletionReceipt) -> None:
        with sqlite3.connect(self.path) as connection:
            existing = connection.execute(
                "SELECT clip_id, device_id, claim_token, outcome, completed_at_utc "
                "FROM media_deletion_receipts WHERE job_id=?",
                (receipt.job_id,),
            ).fetchone()
            values = (
                receipt.clip_id,
                receipt.device_id,
                receipt.claim_token,
                receipt.outcome,
                receipt.completed_at_utc,
            )
            if existing is not None:
                if tuple(existing) != values:
                    raise ValueError("conflicting deletion receipt replay")
                return
            connection.execute(
                "INSERT INTO media_deletion_receipts VALUES(?,?,?,?,?,?)",
                (receipt.job_id, *values),
            )

    def load(self, job_id: str) -> EdgeDeletionReceipt | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT job_id, clip_id, device_id, claim_token, outcome, completed_at_utc "
                "FROM media_deletion_receipts WHERE job_id=?",
                (job_id,),
            ).fetchone()
        return EdgeDeletionReceipt(*row) if row is not None else None


def process_next_deletion(
    *,
    client: EdgeRetentionClient,
    record_lookup: Callable[[str], LocalMediaRecord | None],
    receipt_store: SQLiteDeletionReceiptStore,
    media_root: Path,
    device_id: str,
    execution_mode: str = "disabled",
) -> EdgeDeletionReceipt | None:
    """Process once; a failed completion leaves a durable receipt for replay."""

    task = client.claim_next()
    if task is None:
        return None
    existing = receipt_store.load(task.job_id)
    if existing is not None:
        client.complete(existing)
        return existing
    record = record_lookup(task.clip_id)
    if record is None:
        raise ValueError("deletion task has no local sync authority")
    receipt = execute_local_deletion(
        task,
        record,
        media_root=media_root,
        device_id=device_id,
        execution_mode=execution_mode,
    )
    receipt_store.save(receipt)
    client.complete(receipt)
    return receipt
