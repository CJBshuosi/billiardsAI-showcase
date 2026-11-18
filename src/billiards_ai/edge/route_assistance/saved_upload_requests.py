"""Permanent nonsecret upload evidence; held per-clip pins admit new actions.

The outbox lease controls admission, not exclusion. A terminal response is evidence
only and can be retained after losing the lease or accepting an owner deletion.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType

from billiards_ai.edge.route_assistance.local_media_lock import (
    _assert_directory_chain,
    session_media_lock,
)
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionScope,
    validate_managed_scope_id,
    validate_managed_utc_timestamp,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import SavedMediaOutboxItem

_STATES = {"requested", "received", "started", "succeeded", "unknown", "retired"}


class SavedUploadRequestError(ValueError):
    """Fixed safe error code; retryable distinguishes contention from invalid state."""

    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attempt(value) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"contract", "operation_id", "sequence"}
        or value["contract"] != "managed_upload_attempt_v1"
        or type(value["sequence"]) is not int
        or not 1 <= value["sequence"] <= 64
    ):
        raise ValueError("invalid upload attempt")
    validate_managed_uuid(value["operation_id"], name="upload operation ID")
    return MappingProxyType(dict(value))


def _timestamp(now: datetime) -> str:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("invalid upload timestamp")
    return now.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class SavedUploadRequestRecord:
    request_id: str
    cloud_job_id: str
    state: str
    upload_attempt: Mapping[str, object] | None
    binding_json: str
    binding_sha256: str
    created_at_utc: str
    started_at_utc: str | None = None
    terminal_json: str | None = None
    terminal_sha256: str | None = None

    def __post_init__(self):
        validate_managed_uuid(self.request_id, name="upload request ID")
        validate_managed_scope_id(self.cloud_job_id, name="cloud job ID")
        validate_managed_utc_timestamp(self.created_at_utc, name="request timestamp")
        if self.state not in _STATES:
            raise ValueError("invalid upload request state")
        if self.upload_attempt is not None:
            object.__setattr__(self, "upload_attempt", _attempt(self.upload_attempt))
        binding = json.loads(self.binding_json)
        if (
            not isinstance(binding, dict)
            or _canonical(binding) != self.binding_json
            or _sha(self.binding_json) != self.binding_sha256
            or binding.get("request_id") != self.request_id
            or binding.get("cloud_job_id") != self.cloud_job_id
            or binding.get("created_at_utc") != self.created_at_utc
        ):
            raise ValueError("invalid upload request binding")
        if self.state == "requested" and self.upload_attempt is not None:
            raise ValueError("requested upload already has attempt")
        if self.state in {"received", "started", "succeeded", "unknown"}:
            if self.upload_attempt is None:
                raise ValueError("upload request attempt missing")
        if self.state in {"started", "succeeded", "unknown"}:
            validate_managed_utc_timestamp(self.started_at_utc, name="start timestamp")
            if datetime.fromisoformat(self.started_at_utc) < datetime.fromisoformat(
                self.created_at_utc
            ):
                raise ValueError("invalid upload timestamp order")
        elif self.started_at_utc is not None:
            raise ValueError("unstarted upload has start evidence")
        if self.state == "succeeded":
            terminal = json.loads(self.terminal_json)
            if (
                not isinstance(terminal, dict)
                or set(terminal)
                != {
                    "contract",
                    "request_id",
                    "cloud_job_id",
                    "binding_sha256",
                    "upload_attempt",
                    "started_at_utc",
                    "status_code",
                    "received_at_utc",
                }
                or terminal["contract"] != "local_saved_upload_http200_v1"
                or terminal["request_id"] != self.request_id
                or terminal["cloud_job_id"] != self.cloud_job_id
                or terminal["binding_sha256"] != self.binding_sha256
                or _attempt(terminal["upload_attempt"]) != self.upload_attempt
                or terminal["started_at_utc"] != self.started_at_utc
                or type(terminal["status_code"]) is not int
                or terminal["status_code"] != 200
                or _canonical(terminal) != self.terminal_json
                or _sha(self.terminal_json) != self.terminal_sha256
            ):
                raise ValueError("invalid upload terminal evidence")
            validate_managed_utc_timestamp(terminal["received_at_utc"], name="response timestamp")
            if datetime.fromisoformat(terminal["received_at_utc"]) < datetime.fromisoformat(
                self.started_at_utc
            ):
                raise ValueError("invalid upload response timestamp order")
        elif self.terminal_json is not None or self.terminal_sha256 is not None:
            raise ValueError("non-success upload has success evidence")


class SavedUploadRequestLedger:
    def __init__(self, store: SQLiteFieldRecordStore, scope: ManagedSessionScope):
        if not isinstance(store, SQLiteFieldRecordStore):
            raise SavedUploadRequestError("invalid_context")
        store._managed_scope_values(scope)
        self._store = store
        self._scope = scope
        self._held_sessions = set()

    @contextmanager
    def use(self, item: SavedMediaOutboxItem):
        if not isinstance(item, SavedMediaOutboxItem):
            raise SavedUploadRequestError("invalid_context")
        with ExitStack() as stack:
            try:
                parent = self._store.path.parent
                _assert_directory_chain(parent)
                control = parent / ".saved-upload-request-control"
                control.mkdir(exist_ok=True, mode=0o700)
                identity = _sha(_canonical([str(self._store.path), item.clip_id]))
                stack.enter_context(session_media_lock(control, identity, exclusive=True))
            except (OSError, ValueError, AttributeError):
                raise SavedUploadRequestError("control_unavailable", retryable=True) from None
            session = SavedUploadRequestSession(self, item)
            self._held_sessions.add(session)
            try:
                # Validate and recover any abandoned started row under this pin.
                session.current()
                yield session
            finally:
                session._held = False
                self._held_sessions.discard(session)


class SavedUploadRequestSession:
    """Only the ledger's held context admits calls; escaped sessions fail closed."""

    def __init__(self, ledger: SavedUploadRequestLedger, item: SavedMediaOutboxItem):
        self._ledger = ledger
        self._item = item
        self._held = True
        self._started_request = None

    @contextmanager
    def _transaction(self, *, admit=False):
        if not self._held or self not in self._ledger._held_sessions:
            raise SavedUploadRequestError("session_released")
        try:
            with self._ledger._store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                binding = self._binding(db, admit=admit)
                rows = db.execute(
                    "SELECT * FROM field_saved_upload_requests WHERE clip_id=? "
                    "ORDER BY rowid LIMIT 65",
                    (self._item.clip_id,),
                ).fetchall()
                if len(rows) > 64:
                    raise ValueError("excess request history")
                records = [self._from_row(row, binding) for row in rows]
                active = [row for row in records if row.state != "retired"]
                if len(active) > 1:
                    raise ValueError("multiple active requests")
                yield db, binding, records, active[0] if active else None
        except SavedUploadRequestError:
            raise
        except sqlite3.Error as exc:
            busy = (getattr(exc, "sqlite_errorcode", 0) & 0xFF) in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }
            raise SavedUploadRequestError(
                "database_busy" if busy else "invalid_state",
                retryable=busy,
            ) from None
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise SavedUploadRequestError("invalid_state") from None

    def _binding(self, db, *, admit):
        store, scope, item = self._ledger._store, self._ledger._scope, self._item
        request = store._managed_resource_session_in(db, scope, item.session_id)
        spec = store._managed_resource_spec_in(db, request)
        row = db.execute(
            "SELECT a.*,o.*,s.payload_json AS shot_payload_json,"
            "r.receipt_json AS save_receipt_json,"
            "w.payload_json AS ownership_json,m.payload_json AS live_media_json "
            "FROM field_saved_media_outbox o "
            "JOIN field_postgame_clip_availability a USING(clip_id) "
            "JOIN field_clip_save_requests r ON r.request_id=o.save_request_id "
            "JOIN field_shots s ON s.session_id=a.session_id AND s.shot_id=a.shot_id "
            "JOIN field_shot_ownership w ON w.session_id=a.session_id AND w.shot_id=a.shot_id "
            "JOIN field_live_shot_media m ON m.session_id=a.session_id AND m.shot_id=a.shot_id "
            "WHERE o.clip_id=?",
            (item.clip_id,),
        ).fetchone()
        if row is None:
            raise ValueError("selected media missing")
        persisted = store._saved_media_from_row(row, connection=db)
        expected_job = store._validate_postgame_clip_identity(
            db,
            item.session_id,
            item.shot_id,
            item.postgame_manifest_sha256,
        )
        if expected_job != item.job_id:
            raise ValueError("original postgame job changed")
        if (
            persisted.descriptor_json != item.descriptor_json
            or persisted.local_path != item.local_path
        ):
            raise ValueError("selected source changed")
        epoch_id = persisted.ownership_receipt.source_epoch_id
        epoch = db.execute(
            "SELECT * FROM field_managed_capture_epochs WHERE environment_id=? AND device_id=? "
            "AND source_epoch_id=?",
            (scope.environment_id, scope.device_id, epoch_id),
        ).fetchone()
        if (
            epoch is None
            or epoch["table_id"] != scope.table_id
            or epoch["session_id"] != item.session_id
            or epoch["open_request_id"] != request.open_request_id
            or epoch["birth_id"] != request.binding.birth_id
            or epoch["spec_sha256"] != spec.spec_sha256
        ):
            raise ValueError("original resource epoch missing")
        validate_managed_utc_timestamp(epoch["created_at_utc"], name="epoch timestamp")
        if admit:
            if (
                persisted.status != "leased"
                or not item.claim_token
                or persisted.claim_token != item.claim_token
                or persisted.lease_until_utc is None
                or datetime.fromisoformat(persisted.lease_until_utc) <= store._now_utc()
            ):
                raise SavedUploadRequestError("stale_claim")
            store._reject_owner_clip_in(db, item.clip_id)
        return _canonical(
            dict(
                environment_id=scope.environment_id,
                device_id=scope.device_id,
                table_id=scope.table_id,
                session_id=item.session_id,
                birth_id=request.binding.birth_id,
                birth_contract=request.binding.birth_contract,
                open_request_id=request.open_request_id,
                resource_spec_sha256=spec.spec_sha256,
                source_epoch_id=epoch_id,
                clip_id=item.clip_id,
                source_job_id=item.job_id,
                artifact_id=item.artifact_id,
                source_descriptor_sha256=_sha(persisted.descriptor_json),
                local_path_sha256=_sha(str(persisted.local_path)),
                owner_user_id=item.owner_user_id,
                save_request_id=item.save_request_id,
                save_receipt_sha256=item.save_receipt_sha256,
                ownership_receipt_sha256=item.ownership_receipt_sha256,
                postgame_manifest_sha256=item.postgame_manifest_sha256,
            )
        )

    @staticmethod
    def _from_row(row, binding):
        attempt = row["upload_attempt_json"]
        parsed = None if attempt is None else _attempt(json.loads(attempt))
        if attempt is not None and _canonical(dict(parsed)) != attempt:
            raise ValueError("noncanonical attempt")
        record = SavedUploadRequestRecord(
            row["request_id"],
            row["cloud_job_id"],
            row["state"],
            parsed,
            row["binding_json"],
            row["binding_sha256"],
            row["created_at_utc"],
            row["started_at_utc"],
            row["terminal_json"],
            row["terminal_sha256"],
        )
        expected_binding = _canonical(
            json.loads(binding)
            | {
                "request_id": record.request_id,
                "cloud_job_id": record.cloud_job_id,
                "created_at_utc": record.created_at_utc,
            }
        )
        if record.binding_json != expected_binding:
            raise ValueError("original binding changed")
        return record

    @staticmethod
    def _write(db, record):
        db.execute(
            "UPDATE field_saved_upload_requests SET state=?,upload_attempt_json=?,"
            "started_at_utc=?,terminal_json=?,terminal_sha256=? WHERE request_id=?",
            (
                record.state,
                None if record.upload_attempt is None else _canonical(dict(record.upload_attempt)),
                record.started_at_utc,
                record.terminal_json,
                record.terminal_sha256,
                record.request_id,
            ),
        )
        return record

    @staticmethod
    def _exact(active, request_id):
        validate_managed_uuid(request_id, name="upload request ID")
        if active is None or active.request_id != request_id:
            raise SavedUploadRequestError("request_mismatch")
        return active

    def require_current_claim(self) -> None:
        """Revalidate retained identity and current owner/lease before a remote mutation."""
        with self._transaction(admit=True):
            pass

    def current(self) -> SavedUploadRequestRecord | None:
        with self._transaction() as (db, _, _, active):
            if (
                active is not None
                and active.state == "started"
                and (self._started_request != active.request_id)
            ):
                return self._write(db, replace(active, state="unknown"))
            return active

    def get(self, request_id: str) -> SavedUploadRequestRecord | None:
        """Evidence-only held read, including permanently retired history."""
        validate_managed_uuid(request_id, name="upload request ID")
        with self._transaction() as (_, _, records, _):
            return next((r for r in records if r.request_id == request_id), None)

    def _write_terminal(self, db, record):
        # Local import keeps the queue projection independent of this ledger at import time.
        from billiards_ai.edge.route_assistance.upload_receipt_delivery import enqueue_receipt

        self._write(db, record)
        enqueue_receipt(db, self._ledger._scope, self._item.clip_id, record)
        return record

    def load_or_create(self, cloud_job_id: str) -> SavedUploadRequestRecord:
        with self._transaction(admit=True) as (db, binding, records, active):
            validate_managed_scope_id(cloud_job_id, name="cloud job ID")
            if active is not None:
                if active.cloud_job_id != cloud_job_id:
                    raise SavedUploadRequestError("request_mismatch")
                return active
            if len(records) >= 64:
                raise SavedUploadRequestError("request_limit")
            request_id = str(uuid.uuid4())
            created_at_utc = _timestamp(self._ledger._store._now_utc())
            binding = _canonical(
                json.loads(binding)
                | {
                    "request_id": request_id,
                    "cloud_job_id": cloud_job_id,
                    "created_at_utc": created_at_utc,
                }
            )
            record = SavedUploadRequestRecord(
                request_id,
                cloud_job_id,
                "requested",
                None,
                binding,
                _sha(binding),
                created_at_utc,
            )
            db.execute(
                "INSERT INTO field_saved_upload_requests VALUES(?,?,?,?,?,?,NULL,?,NULL,NULL,NULL)",
                (
                    record.request_id,
                    self._item.clip_id,
                    cloud_job_id,
                    binding,
                    record.binding_sha256,
                    record.state,
                    record.created_at_utc,
                ),
            )
            return record

    def record_received(self, request_id: str, upload_attempt) -> SavedUploadRequestRecord:
        with self._transaction(admit=True) as (db, _, _, active):
            active = self._exact(active, request_id)
            attempt = _attempt(upload_attempt)
            if active.state == "received" and active.upload_attempt == attempt:
                return active
            if active.state != "requested":
                raise SavedUploadRequestError("invalid_transition")
            return self._write(db, replace(active, state="received", upload_attempt=attempt))

    def record_started(self, request_id: str) -> SavedUploadRequestRecord:
        with self._transaction(admit=True) as (db, _, _, active):
            active = self._exact(active, request_id)
            if active.state != "received":
                raise SavedUploadRequestError("invalid_transition")
            result = self._write(
                db,
                replace(
                    active,
                    state="started",
                    started_at_utc=_timestamp(self._ledger._store._now_utc()),
                ),
            )
        self._started_request = request_id
        return result

    def record_succeeded(self, request_id: str, status_code: int) -> SavedUploadRequestRecord:
        with self._transaction() as (db, _, _, active):
            active = self._exact(active, request_id)
            if type(status_code) is not int or status_code != 200:
                raise SavedUploadRequestError("invalid_terminal")
            if active.state == "succeeded":
                return self._write_terminal(db, active)
            if active.state != "started" or self._started_request != request_id:
                raise SavedUploadRequestError("invalid_transition")
            terminal = _canonical(
                dict(
                    contract="local_saved_upload_http200_v1",
                    request_id=request_id,
                    cloud_job_id=active.cloud_job_id,
                    binding_sha256=active.binding_sha256,
                    upload_attempt=dict(active.upload_attempt),
                    started_at_utc=active.started_at_utc,
                    status_code=200,
                    received_at_utc=_timestamp(self._ledger._store._now_utc()),
                )
            )
            return self._write_terminal(
                db,
                replace(
                    active,
                    state="succeeded",
                    terminal_json=terminal,
                    terminal_sha256=_sha(terminal),
                ),
            )

    def record_unknown(self, request_id: str) -> SavedUploadRequestRecord:
        with self._transaction() as (db, _, _, active):
            active = self._exact(active, request_id)
            if active.state == "unknown":
                return active
            if active.state != "started":
                raise SavedUploadRequestError("invalid_transition")
            return self._write(db, replace(active, state="unknown"))

    def retire(self, request_id: str, upload_attempt=None) -> SavedUploadRequestRecord:
        with self._transaction() as (db, _, records, active):
            # A repeated retirement is evidence-only and cannot revive a request.
            retired = next(
                (r for r in records if r.request_id == request_id and r.state == "retired"), None
            )
            if retired is not None:
                if (
                    upload_attempt is not None
                    and _attempt(upload_attempt) != retired.upload_attempt
                ):
                    raise SavedUploadRequestError("request_mismatch")
                return self._write_terminal(db, retired)
            active = self._exact(active, request_id)
            if active.state not in {"requested", "received"}:
                raise SavedUploadRequestError("invalid_transition")
            attempt = active.upload_attempt if upload_attempt is None else _attempt(upload_attempt)
            if active.upload_attempt is not None and attempt != active.upload_attempt:
                raise SavedUploadRequestError("request_mismatch")
            return self._write_terminal(
                db, replace(active, state="retired", upload_attempt=attempt)
            )
