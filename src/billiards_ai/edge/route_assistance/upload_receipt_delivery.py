"""Permanent terminal-evidence delivery metadata, independent of media authority.

Only validated retained ledger facts project to wire bytes. Nothing in this module
opens source/model files, admits an upload, or changes the original terminal bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionScope,
    validate_managed_scope_id,
    validate_managed_utc_timestamp,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

if TYPE_CHECKING:
    from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestRecord

# Sender errors are fixed public classifications, never transport exception strings.
UPLOAD_RECEIPT_ERROR_CODES = frozenset(
    {
        "receipt_unavailable",
        "receipt_conflict",
        "receipt_invalid",
        "receipt_auth_failed",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}")
_ACK_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z")


class UploadReceiptDeliveryError(ValueError):
    """Safe classification suitable for a caller diagnostic, without raw errors."""

    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _utc(value: str) -> str:
    validate_managed_utc_timestamp(value, name="receipt timestamp")
    return datetime.fromisoformat(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _now(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise UploadReceiptDeliveryError("invalid_context")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def project_upload_receipt(record: SavedUploadRequestRecord) -> str:
    """Project already validated original evidence; preserve its local representation."""
    from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestRecord

    if not isinstance(record, SavedUploadRequestRecord):
        raise ValueError("invalid receipt record")
    record = replace(record)  # Also revalidate direct dataclass construction.
    if record.state == "succeeded":
        terminal = json.loads(record.terminal_json)
        evidence = dict(
            kind="http200",
            local_binding_sha256=record.binding_sha256,
            started_at_utc=_utc(record.started_at_utc),
            received_at_utc=_utc(terminal["received_at_utc"]),
            status_code=200,
        )
    elif record.state == "retired":
        evidence = dict(kind="retired_before_start", local_binding_sha256=record.binding_sha256)
    else:
        raise ValueError("nonterminal receipt")
    return _canonical(
        dict(
            contract="managed_upload_receipt_v1",
            upload_request=dict(contract="managed_upload_request_v1", request_id=record.request_id),
            upload_attempt=None if record.upload_attempt is None else dict(record.upload_attempt),
            evidence=evidence,
        )
    )


def enqueue_receipt(db, scope: ManagedSessionScope, clip_id: str, record) -> bool:
    """Pure DB helper: use the caller's original terminal-transition transaction.

    It neither acquires a lock nor opens/commits a connection. Existing immutable
    queue bytes and identity must match even after delivery or blocking.
    """
    scope_values = SQLiteFieldRecordStore._managed_scope_values(scope)
    body = project_upload_receipt(record)
    values = (record.request_id, *scope_values, clip_id, record.cloud_job_id, body, _sha(body))
    old = db.execute(
        "SELECT request_id,environment_id,device_id,table_id,clip_id,cloud_job_id,"
        "receipt_json,receipt_sha256 FROM field_upload_receipt_outbox WHERE request_id=?",
        (record.request_id,),
    ).fetchone()
    if old is not None:
        if tuple(old) != values:
            raise ValueError("changed terminal receipt")
        return False
    db.execute(
        "INSERT INTO field_upload_receipt_outbox "
        "(request_id,environment_id,device_id,table_id,clip_id,cloud_job_id,receipt_json,"
        "receipt_sha256,status) VALUES(?,?,?,?,?,?,?,?,'pending')",
        values,
    )
    return True


@dataclass(frozen=True, slots=True)
class UploadReceiptOutboxItem:
    request_id: str
    cloud_job_id: str
    clip_id: str
    receipt_json: str
    receipt_sha256: str
    claim_token: str
    scope: ManagedSessionScope
    attempts: int
    lease_until_utc: str
    next_attempt_at_utc: str | None
    last_error_code: str | None

    def __post_init__(self):
        validate_managed_uuid(self.request_id, name="upload request ID")
        validate_managed_uuid(self.claim_token, name="receipt claim token")
        validate_managed_scope_id(self.cloud_job_id, name="cloud job ID")
        validate_managed_scope_id(self.clip_id, name="clip ID")
        SQLiteFieldRecordStore._managed_scope_values(self.scope)
        if (
            type(self.attempts) is not int
            or self.attempts < 1
            or not isinstance(self.receipt_json, str)
            or not isinstance(self.receipt_sha256, str)
            or _SHA.fullmatch(self.receipt_sha256) is None
            or _sha(self.receipt_json) != self.receipt_sha256
            or _canonical(json.loads(self.receipt_json)) != self.receipt_json
        ):
            raise ValueError("invalid receipt claim")
        if _utc(self.lease_until_utc) != self.lease_until_utc:
            raise ValueError("invalid receipt lease")
        if self.next_attempt_at_utc is not None:
            if _utc(self.next_attempt_at_utc) != self.next_attempt_at_utc:
                raise ValueError("invalid receipt due time")
        if (
            self.last_error_code is not None
            and self.last_error_code not in UPLOAD_RECEIPT_ERROR_CODES
        ):
            raise ValueError("invalid receipt error")


@dataclass(frozen=True, slots=True)
class UploadReceiptReconcileResult:
    next_request_id: str | None
    scanned: int
    queued: int
    diagnostics: tuple[str, ...]


def validate_upload_receipt_ack(item, ack) -> None:
    """Validate the exact server evidence without settling a claim token."""
    if not isinstance(item, UploadReceiptOutboxItem):
        raise UploadReceiptDeliveryError("invalid_context")
    if (
        not isinstance(ack, Mapping)
        or set(ack)
        != {"contract", "upload_request", "receipt_sha256", "received_at_utc", "erasable"}
        or ack["contract"] != "managed_upload_receipt_ack_v1"
        or ack["upload_request"]
        != dict(contract="managed_upload_request_v1", request_id=item.request_id)
        or ack["receipt_sha256"] != item.receipt_sha256
        or ack["erasable"] is not False
        or not isinstance(ack["received_at_utc"], str)
        or _ACK_TIME.fullmatch(ack["received_at_utc"]) is None
    ):
        raise UploadReceiptDeliveryError("invalid_ack")
    try:
        validate_managed_utc_timestamp(ack["received_at_utc"], name="receipt ACK timestamp")
    except ValueError:
        raise UploadReceiptDeliveryError("invalid_ack") from None


class UploadReceiptOutbox:
    def __init__(self, store: SQLiteFieldRecordStore, scope: ManagedSessionScope):
        if not isinstance(store, SQLiteFieldRecordStore):
            raise UploadReceiptDeliveryError("invalid_context")
        self._values = store._managed_scope_values(scope)
        self._store = store
        self._scope = scope

    @staticmethod
    def _error(exc):
        if isinstance(exc, sqlite3.Error):
            busy = (getattr(exc, "sqlite_errorcode", 0) & 0xFF) in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }
            return UploadReceiptDeliveryError(
                "database_busy" if busy else "invalid_state", retryable=busy
            )
        code = getattr(exc, "code", "invalid_state")
        if code not in {"database_busy", "control_unavailable"}:
            code = "invalid_state"
        return UploadReceiptDeliveryError(code, retryable=getattr(exc, "retryable", False))

    def _candidates(self, after, limit):
        with self._store._connect() as db:
            # Scope derives from original managed session joins, never untrusted
            # binding JSON or a current cloud admission. A bad binding remains a
            # scanned candidate so that the caller's cursor can advance past it.
            sql = (
                "SELECT r.request_id,r.clip_id FROM field_saved_upload_requests r "
                "JOIN field_postgame_clip_availability a ON a.clip_id=r.clip_id "
                "JOIN field_managed_open_requests m ON m.session_id=a.session_id "
                "WHERE m.environment_id=? AND m.device_id=? AND m.table_id=? "
                "AND r.state IN ('succeeded','retired') "
                "AND NOT EXISTS (SELECT 1 FROM field_upload_receipt_outbox o "
                "WHERE o.request_id=r.request_id) "
            )
            return db.execute(
                sql + "AND r.request_id>? ORDER BY r.request_id LIMIT ?",
                (*self._values, after or "", limit),
            ).fetchall()

    def _validated(self, request_id, clip_id, action):
        from billiards_ai.edge.route_assistance.saved_upload_requests import (
            SavedUploadRequestLedger,
        )

        # Read candidate metadata and close its connection BEFORE the canonical
        # clip pin. Never wait for that pin while holding a SQLite write lock.
        saved = self._store.get_saved_media_outbox(clip_id)
        if saved is None:
            raise ValueError("missing retained selection")
        with SavedUploadRequestLedger(self._store, self._scope).use(saved) as session:
            with session._transaction() as (db, _, records, _):
                record = next((r for r in records if r.request_id == request_id), None)
                if record is None:
                    raise ValueError("missing retained request")
                return action(db, record)

    def reconcile(self, after_request_id=None, limit=16) -> UploadReceiptReconcileResult:
        if type(limit) is not int or not 1 <= limit <= 16:
            raise UploadReceiptDeliveryError("invalid_context")
        # Cursor is a local ordering key, not a wire identity. Return/accept the
        # exact scanned key even when that candidate's UUID itself is corrupt.
        if after_request_id is not None and not isinstance(after_request_id, str):
            raise UploadReceiptDeliveryError("invalid_context")
        try:
            candidates = self._candidates(after_request_id, limit)
            if not candidates and after_request_id is not None:
                candidates = self._candidates(None, limit)
        except (sqlite3.Error, ValueError) as exc:
            raise self._error(exc) from None
        queued, diagnostics = 0, []
        for candidate in candidates:
            try:
                queued += self._validated(
                    candidate["request_id"],
                    candidate["clip_id"],
                    lambda db, record, clip=candidate["clip_id"]: enqueue_receipt(
                        db, self._scope, clip, record
                    ),
                )
            except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError) as exc:
                diagnostics.append(self._error(exc).code)
        return UploadReceiptReconcileResult(
            candidates[-1]["request_id"] if candidates else None,
            len(candidates),
            queued,
            tuple(diagnostics),
        )

    def claim_due(self, now, lease_seconds=60) -> UploadReceiptOutboxItem | None:
        stamp = _now(now)
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise UploadReceiptDeliveryError("invalid_context")
        expiry = _now(now + timedelta(seconds=lease_seconds))
        try:
            with self._store._connect() as db:
                candidates = db.execute(
                    "SELECT request_id,clip_id FROM field_upload_receipt_outbox "
                    "WHERE environment_id=? AND device_id=? AND table_id=? "
                    "AND ((status='pending' AND (next_attempt_at_utc IS NULL "
                    "OR next_attempt_at_utc<=?)) OR (status='leased' AND lease_until_utc<=?)) "
                    "ORDER BY request_id LIMIT 16",
                    (*self._values, stamp, stamp),
                ).fetchall()
            for candidate in candidates:
                try:
                    result = self._validated(
                        candidate["request_id"],
                        candidate["clip_id"],
                        lambda db, record, clip=candidate["clip_id"]: self._claim_in(
                            db, record, clip, stamp, expiry
                        ),
                    )
                    if result is not None:
                        return result
                except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError) as exc:
                    error = self._error(exc)
                    if error.retryable:
                        raise error from None
                    # Corrupt retained history is never sent. Do not settle a
                    # competing live claim that arose after candidate selection.
                    with self._store._connect() as db:
                        db.execute(
                            "UPDATE field_upload_receipt_outbox SET status='blocked',"
                            "claim_token=NULL,lease_until_utc=NULL,"
                            "last_error_code='receipt_invalid' "
                            "WHERE request_id=? AND environment_id=? "
                            "AND device_id=? AND table_id=? "
                            "AND (status='pending' OR (status='leased' AND lease_until_utc<=?))",
                            (candidate["request_id"], *self._values, stamp),
                        )
            return None
        except UploadReceiptDeliveryError:
            raise
        except (sqlite3.Error, ValueError) as exc:
            raise self._error(exc) from None

    def _claim_in(self, db, record, clip_id, stamp, expiry):
        row = db.execute(
            "SELECT * FROM field_upload_receipt_outbox WHERE request_id=? AND environment_id=? "
            "AND device_id=? AND table_id=?",
            (record.request_id, *self._values),
        ).fetchone()
        if row is None or row["status"] not in {"pending", "leased"}:
            return None
        if row["status"] == "leased":
            if row["lease_until_utc"] is None or _utc(row["lease_until_utc"]) > stamp:
                return None
        elif row["next_attempt_at_utc"] is not None and _utc(row["next_attempt_at_utc"]) > stamp:
            return None
        if (
            row["cloud_job_id"] != record.cloud_job_id
            or row["clip_id"] != clip_id
            or row["receipt_json"] != project_upload_receipt(record)
            or row["receipt_sha256"] != _sha(row["receipt_json"])
        ):
            raise ValueError("changed receipt")
        token = str(uuid.uuid4())
        item = UploadReceiptOutboxItem(
            record.request_id,
            row["cloud_job_id"],
            clip_id,
            row["receipt_json"],
            row["receipt_sha256"],
            token,
            self._scope,
            row["attempts"] + 1,
            expiry,
            row["next_attempt_at_utc"],
            row["last_error_code"],
        )
        db.execute(
            "UPDATE field_upload_receipt_outbox SET status='leased',attempts=attempts+1,"
            "claim_token=?,lease_until_utc=? WHERE request_id=?",
            (token, expiry, record.request_id),
        )
        return item

    def _settle(self, item, status, *, error=None, due=None):
        if not isinstance(item, UploadReceiptOutboxItem) or item.scope != self._scope:
            raise UploadReceiptDeliveryError("invalid_context")
        if error is not None and error not in UPLOAD_RECEIPT_ERROR_CODES:
            raise UploadReceiptDeliveryError("invalid_context")
        try:
            with self._store._connect() as db:
                changed = db.execute(
                    "UPDATE field_upload_receipt_outbox SET status=?,claim_token=NULL,"
                    "lease_until_utc=NULL,next_attempt_at_utc=?,last_error_code=? "
                    "WHERE request_id=? AND environment_id=? AND device_id=? AND table_id=? "
                    "AND clip_id=? AND cloud_job_id=? AND receipt_json=? AND receipt_sha256=? "
                    "AND status='leased' AND claim_token=?",
                    (
                        status,
                        due,
                        error,
                        item.request_id,
                        *self._values,
                        item.clip_id,
                        item.cloud_job_id,
                        item.receipt_json,
                        item.receipt_sha256,
                        item.claim_token,
                    ),
                ).rowcount
            return changed == 1
        except sqlite3.Error as exc:
            raise self._error(exc) from None

    def acknowledge(self, item, ack):
        validate_upload_receipt_ack(item, ack)
        return self._settle(item, "acknowledged")

    def retry(self, item, error_code, next_attempt_at_utc):
        if not isinstance(error_code, str) or error_code not in UPLOAD_RECEIPT_ERROR_CODES:
            raise UploadReceiptDeliveryError("invalid_context")
        return self._settle(item, "pending", error=error_code, due=_utc(next_attempt_at_utc))

    def block(self, item, error_code):
        if not isinstance(error_code, str) or error_code not in UPLOAD_RECEIPT_ERROR_CODES:
            raise UploadReceiptDeliveryError("invalid_context")
        return self._settle(item, "blocked", error=error_code)
