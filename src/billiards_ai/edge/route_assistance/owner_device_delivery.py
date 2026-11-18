"""Metadata-only completion codec and durable original-device receipt delivery."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask

if TYPE_CHECKING:
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

_ERRORS = frozenset(
    {
        "transport_unavailable",
        "transport_timeout",
        "invalid_response",
        "invalid_ack",
        "invalid_completion",
        "scope_conflict",
        "remote_conflict",
        "unauthorized",
        "attestation_failed",
        "configuration_invalid",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "contract",
        "deletion_job_id",
        "clip_id",
        "task_sha256",
        "cloud_erase_receipt_sha256",
        "journal_sha256",
        "reclaimed_bytes",
    }
)
_ACK_KEYS = frozenset(
    {
        "contract",
        "deletion_job_id",
        "clip_id",
        "task_sha256",
        "cloud_erase_receipt_sha256",
        "local_receipt_sha256",
        "received_at_utc",
    }
)


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _hash(raw: str) -> str:
    return sha256(raw.encode("ascii")).hexdigest()


def _time(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("owner delivery clock requires aware datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def completion_json(task: OwnerDeleteDeviceTask, local_receipt: object) -> str:
    """Exact Task2 completion wire. No task lease or filesystem evidence is inferred."""
    if not isinstance(task, OwnerDeleteDeviceTask):
        raise ValueError("invalid completion task")
    if not isinstance(local_receipt, Mapping) or set(local_receipt) != _RECEIPT_KEYS:
        raise ValueError("invalid completion receipt fields")
    receipt = dict(local_receipt)
    reclaimed = receipt["reclaimed_bytes"]
    if (
        receipt["contract"] != "local_owner_erase_receipt_v1"
        or receipt["deletion_job_id"] != task.deletion_job_id
        or receipt["clip_id"] != task.clip_id
        or receipt["task_sha256"] != task.task_sha256
        or receipt["cloud_erase_receipt_sha256"] != task.cloud_erase_receipt_sha256
        or not _digest(receipt["journal_sha256"])
        or type(reclaimed) is not int
        or not 0 <= reclaimed <= 9007199254740991
    ):
        raise ValueError("invalid completion receipt identity")
    return _canonical(
        dict(
            contract="owner_delete_device_completion_v1",
            task_sha256=task.task_sha256,
            local_receipt=receipt,
            local_receipt_sha256=_hash(_canonical(receipt)),
        )
    )


def _state(connection, key: str, turn: str):
    connection.execute(
        "INSERT OR IGNORE INTO field_local_media_schedule_state(state_key,turn) VALUES(?,?)",
        (key, turn),
    )
    return connection.execute(
        "SELECT * FROM field_local_media_schedule_state WHERE state_key=?", (key,)
    ).fetchone()


def _state_key(kind: str, scope: tuple | None, *extra) -> str:
    return kind + ":" + _hash(_canonical([scope, *extra]))


def _completed_in(store, connection, row):
    state = store._owner_delete_from_row(row)
    if state.status != "completed":
        raise ValueError("invalid completion state")
    task = state.task
    scope = task.environment_id, task.device_id, task.table_id
    selection, item = store._validate_owner_delete_authority_in(connection, task, scope)
    if (
        row["selection_json"] != selection
        or state.local_path != item.local_path
        or item.status != "blocked"
        or item.last_error_code != "OWNER_CLIP_DELETE_REQUESTED"
        or item.claim_token is not None
        or item.lease_until_utc is not None
    ):
        raise ValueError("invalid completion accepted authority")
    progress = store._owner_progress(state, state.progress_json, terminal=True)
    receipt = json.loads(state.receipt_json)
    body = completion_json(task, receipt)
    if (
        state.progress_json != progress
        or _canonical(receipt) != state.receipt_json
        or _hash(state.receipt_json) != state.receipt_sha256
        or receipt["journal_sha256"] != _hash(progress)
        or receipt["reclaimed_bytes"]
        != sum(o["reclaimed_bytes"] for o in json.loads(progress)["objects"])
    ):
        raise ValueError("invalid completion terminal receipt")
    return state, body


def enqueue_owner_completion_in(store, connection, clip_id: str) -> bool:
    """Join the caller's transaction; failure rolls back its terminal metadata too."""
    row = connection.execute(
        "SELECT * FROM field_owner_media_deletions WHERE clip_id=?", (clip_id,)
    ).fetchone()
    state, body = _completed_in(store, connection, row)
    task = state.task
    immutable = (
        task.deletion_job_id,
        task.clip_id,
        task.environment_id,
        task.device_id,
        task.table_id,
        task.task_sha256,
        task.cloud_erase_receipt_sha256,
        state.receipt_json,
        state.receipt_sha256,
        body,
        _hash(body),
    )
    existing = connection.execute(
        "SELECT * FROM field_owner_completion_outbox WHERE deletion_job_id=? OR clip_id=?",
        (task.deletion_job_id, task.clip_id),
    ).fetchone()
    if existing is not None:
        if tuple(existing)[:11] != immutable:
            raise ValueError("invalid completion outbox conflict")
        return False
    connection.execute(
        "INSERT INTO field_owner_completion_outbox "
        "(deletion_job_id,clip_id,environment_id,device_id,table_id,task_sha256,"
        "cloud_erase_receipt_sha256,local_receipt_json,local_receipt_sha256,completion_json,"
        "completion_sha256,status,attempts,next_attempt_at_utc) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,'pending',0,'1970-01-01T00:00:00Z')",
        immutable,
    )
    return True


@dataclass(frozen=True, slots=True)
class OwnerDeletionReceiptItem:
    deletion_job_id: str
    clip_id: str
    environment_id: str
    device_id: str
    table_id: str
    task_sha256: str
    cloud_erase_receipt_sha256: str
    local_receipt_json: str
    local_receipt_sha256: str
    completion_json: str
    completion_sha256: str
    status: str
    attempts: int
    claim_token: str | None
    lease_until_utc: str | None
    next_attempt_at_utc: str
    last_error_code: str | None
    ack_json: str | None


@dataclass(frozen=True, slots=True)
class OwnerDeletionReconcileResult:
    scanned: int
    queued: int
    diagnostics: tuple[str, ...]


class OwnerDeletionReceiptOutbox:
    def __init__(self, store: SQLiteFieldRecordStore, scope: ManagedSessionScope):
        self.store = store
        self.scope = store._managed_scope_values(scope)
        self.key = _state_key("owner-reconcile", self.scope)

    def reconcile(self, limit: int = 16) -> OwnerDeletionReconcileResult:
        if type(limit) is not int or limit <= 0:
            raise ValueError("invalid reconciliation limit")
        scanned, queued, diagnostics = 0, 0, []
        # Commit progress before validating each row. A corrupt row/storage fault
        # cannot restart the sweep at a poisoned head after process restart.
        for _ in range(min(limit, 16)):
            with self.store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                state = _state(db, self.key, "receive")
                upper = state["upper_key"]
                cursor = int(state["cursor"] or "0")
                query = (
                    " FROM field_owner_media_deletions d WHERE d.status='completed' AND EXISTS "
                    "(SELECT 1 FROM field_managed_open_requests m WHERE m.session_id=d.session_id "
                    "AND m.environment_id=? AND m.device_id=? AND m.table_id=? "
                    "AND m.phase='materialized') "
                )
                if upper is None:
                    upper = db.execute("SELECT MAX(d.rowid)" + query, self.scope).fetchone()[0]
                    cursor = 0
                row = (
                    None
                    if upper is None
                    else db.execute(
                        "SELECT d.rowid AS reconcile_rowid,d.*"
                        + query
                        + "AND d.rowid>? AND d.rowid<=? ORDER BY d.rowid LIMIT 1",
                        (*self.scope, cursor, int(upper)),
                    ).fetchone()
                )
                if row is None:
                    db.execute(
                        "UPDATE field_local_media_schedule_state SET cursor='',upper_key=NULL "
                        "WHERE state_key=?",
                        (self.key,),
                    )
                    break
                scanned += 1
                db.execute(
                    "UPDATE field_local_media_schedule_state SET cursor=?,upper_key=? "
                    "WHERE state_key=?",
                    (str(row["reconcile_rowid"]), str(upper), self.key),
                )
            try:
                with self.store._connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    current = db.execute(
                        "SELECT * FROM field_owner_media_deletions WHERE rowid=?",
                        (row["reconcile_rowid"],),
                    ).fetchone()
                    if current is None:
                        raise ValueError("invalid completion missing row")
                    task = OwnerDeleteDeviceTask.from_payload(json.loads(current["task_json"]))
                    if (task.environment_id, task.device_id, task.table_id) != self.scope:
                        raise ValueError("invalid completion scope")
                    queued += int(enqueue_owner_completion_in(self.store, db, current["clip_id"]))
            except (ValueError, TypeError, KeyError, AttributeError, IndexError, RecursionError):
                diagnostics.append("invalid_completion")
        return OwnerDeletionReconcileResult(scanned, queued, tuple(diagnostics))

    @staticmethod
    def _item(row):
        return OwnerDeletionReceiptItem(
            **{f.name: row[f.name] for f in fields(OwnerDeletionReceiptItem)}
        )

    @staticmethod
    def _valid_wire(row) -> bool:
        try:
            receipt = json.loads(row["local_receipt_json"])
            body = json.loads(row["completion_json"])
            if (
                not isinstance(receipt, dict)
                or set(receipt) != _RECEIPT_KEYS
                or receipt["contract"] != "local_owner_erase_receipt_v1"
                or any(
                    receipt[k] != row[k]
                    for k in (
                        "deletion_job_id",
                        "clip_id",
                        "task_sha256",
                        "cloud_erase_receipt_sha256",
                    )
                )
                or not all(
                    _digest(row[k])
                    for k in (
                        "task_sha256",
                        "cloud_erase_receipt_sha256",
                        "local_receipt_sha256",
                        "completion_sha256",
                    )
                )
                or not _digest(receipt["journal_sha256"])
                or type(receipt["reclaimed_bytes"]) is not int
                or not 0 <= receipt["reclaimed_bytes"] <= 9007199254740991
                or _canonical(receipt) != row["local_receipt_json"]
                or _hash(row["local_receipt_json"]) != row["local_receipt_sha256"]
            ):
                return False
            expected = dict(
                contract="owner_delete_device_completion_v1",
                task_sha256=row["task_sha256"],
                local_receipt=receipt,
                local_receipt_sha256=row["local_receipt_sha256"],
            )
            return (
                body == expected
                and row["completion_json"] == _canonical(expected)
                and _hash(row["completion_json"]) == row["completion_sha256"]
            )
        except (TypeError, ValueError, KeyError, RecursionError):
            return False

    def claim_due(self, now: datetime, lease_seconds: int = 60) -> OwnerDeletionReceiptItem | None:
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("invalid completion lease")
        due = _time(now)
        until = _time(now + timedelta(seconds=lease_seconds))
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM field_owner_completion_outbox WHERE environment_id=? AND "
                "device_id=? AND table_id=? AND ((status='pending' AND next_attempt_at_utc<=?) "
                "OR (status='leased' AND lease_until_utc<=?)) "
                "ORDER BY next_attempt_at_utc,deletion_job_id LIMIT 1",
                (*self.scope, due, due),
            ).fetchone()
            if row is None:
                return None
            if not self._valid_wire(row):
                db.execute(
                    "UPDATE field_owner_completion_outbox SET status='blocked',"
                    "claim_token=NULL,lease_until_utc=NULL,last_error_code='invalid_completion' "
                    "WHERE deletion_job_id=?",
                    (row["deletion_job_id"],),
                )
                return None
            db.execute(
                "UPDATE field_owner_completion_outbox SET status='leased',attempts=attempts+1,"
                "claim_token=?,lease_until_utc=?,last_error_code=NULL WHERE deletion_job_id=?",
                (str(uuid.uuid4()), until, row["deletion_job_id"]),
            )
            return self._item(
                db.execute(
                    "SELECT * FROM field_owner_completion_outbox WHERE deletion_job_id=?",
                    (row["deletion_job_id"],),
                ).fetchone()
            )

    def _current(self, db, item):
        if not isinstance(item, OwnerDeletionReceiptItem) or not item.claim_token:
            return None
        row = db.execute(
            "SELECT * FROM field_owner_completion_outbox WHERE deletion_job_id=? "
            "AND environment_id=? AND device_id=? AND table_id=?",
            (item.deletion_job_id, *self.scope),
        ).fetchone()
        if row is None or row["claim_token"] != item.claim_token:
            return None
        # Settlement is bound to exact immutable bytes, not merely a mutable job ID.
        for f in fields(OwnerDeletionReceiptItem)[:11]:
            if row[f.name] != getattr(item, f.name):
                return None
        return row

    def acknowledge(self, item: OwnerDeletionReceiptItem, ack: object) -> bool:
        if not isinstance(ack, Mapping) or set(ack) != _ACK_KEYS:
            raise ValueError("invalid ack fields")
        if ack["contract"] != "owner_delete_device_ack_v1" or any(
            ack[name] != getattr(item, name)
            for name in (
                "deletion_job_id",
                "clip_id",
                "task_sha256",
                "cloud_erase_receipt_sha256",
                "local_receipt_sha256",
            )
        ):
            raise ValueError("invalid ack identity")
        time = ack["received_at_utc"]
        if (
            not isinstance(time, str)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", time) is None
        ):
            raise ValueError("invalid ack time")
        try:
            datetime.fromisoformat(time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid ack time") from exc
        raw = _canonical(dict(ack))
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._current(db, item)
            if row is None:
                return False
            if row["status"] == "acknowledged":
                return row["ack_json"] == raw
            if row["status"] != "leased":
                return False
            db.execute(
                "UPDATE field_owner_completion_outbox SET status='acknowledged',"
                "lease_until_utc=NULL,last_error_code=NULL,ack_json=? WHERE deletion_job_id=?",
                (raw, item.deletion_job_id),
            )
            return True

    def _finish(self, item, status, error_code, due=None):
        if not isinstance(error_code, str) or error_code not in _ERRORS:
            raise ValueError("invalid completion error code")
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._current(db, item)
            if row is None or row["status"] != "leased":
                return False
            db.execute(
                "UPDATE field_owner_completion_outbox SET status=?,claim_token=NULL,"
                "lease_until_utc=NULL,last_error_code=?,next_attempt_at_utc=? "
                "WHERE deletion_job_id=?",
                (status, error_code, due or row["next_attempt_at_utc"], item.deletion_job_id),
            )
            return True

    def retry(
        self, item: OwnerDeletionReceiptItem, error_code: str, next_attempt_at_utc: datetime
    ) -> bool:
        return self._finish(item, "pending", error_code, _time(next_attempt_at_utc))

    def block(self, item: OwnerDeletionReceiptItem, error_code: str) -> bool:
        return self._finish(item, "blocked", error_code)

    def next_turn(self) -> Literal["receive", "send"]:
        key = _state_key("owner-delivery-turn", self.scope)
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            turn = _state(db, key, "receive")["turn"]
            db.execute(
                "UPDATE field_local_media_schedule_state SET turn=? WHERE state_key=?",
                ("send" if turn == "receive" else "receive", key),
            )
            return turn
