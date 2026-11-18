"""Encrypted SQLite persistence for one deterministic CoachRuntime session.

The database is deliberately an opaque delivery/recovery cache.  Session identities and
authority payloads are never indexed in plaintext; callers must already know a session id.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, final
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel

from billiards_ai.edge.coach_runtime.errors import RuntimeErrorCode, RuntimeTransitionError
from billiards_ai.edge.coach_runtime.execution_policy import (
    AbstentionAction,
    DrillExecutionProjection,
    EvaluationConclusion,
    ExecutionBlockProjection,
    ExecutionDecision,
    ExecutionDecisionKind,
    ExecutionPolicyState,
    MetricCombination,
    MetricEvaluation,
    MetricRuleProjection,
    ObservationEvaluation,
    PolicyReceipt,
    ReplanContinueAction,
    ReplanIntent,
    SuccessPolicyProjection,
    VerifiedExecutionProjection,
    _issue_verified_execution_projection,
)
from billiards_ai.edge.coach_runtime.state_machine import (
    AcceptObservation,
    AdmittedPlanIdentity,
    AdmitVerifiedPlan,
    AdvanceToNextShot,
    BeginCompletion,
    BeginPlanLoad,
    CoachRuntimeState,
    CompletionReceipt,
    FailRuntime,
    FinishCompletion,
    ObservationKey,
    ObservationOutcomeReceipt,
    ObservationReceipt,
    PauseRuntime,
    PresentFeedback,
    PresentSetup,
    RecoverRuntime,
    ResumeRuntime,
    RuntimeCommand,
    RuntimeEffect,
    RuntimeEffectKind,
    RuntimeSnapshot,
    RuntimeTransition,
    RuntimeTransitionResult,
    VerifiedPlanAdmission,
    WaitForShot,
    _snapshot_invariant_error,
    transition,
)
from billiards_ai.shared.models.growth import BlockOutcome, PlanOutcome, PlanOutcomeStatus
from billiards_ai.shared.models.realtime import (
    CoachingEventType,
    RealtimeCoachingEventV2,
    RealtimeShotObservation,
)

SCHEMA_VERSION = 2
CODEC_VERSION = "edge-runtime-sealed-json/v1"
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
_ACK_QUARANTINE_AFTER = timedelta(days=7)
_DELETE_AFTER_ACK = timedelta(hours=24)


class PersistenceErrorCode(StrEnum):
    DB_LOCKED = "DB_LOCKED"
    UNSUPPORTED_MIGRATION = "UNSUPPORTED_MIGRATION"
    CORRUPT_DATABASE = "CORRUPT_DATABASE"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    COMMIT_UNKNOWN = "COMMIT_UNKNOWN"
    KEY_UNAVAILABLE = "KEY_UNAVAILABLE"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    QUARANTINED = "QUARANTINED"


class RecoveryStatus(StrEnum):
    EXACT = "EXACT"
    PAUSED = "PAUSED"
    DEGRADED = "DEGRADED"


class RuntimePersistenceError(RuntimeError):
    def __init__(
        self,
        code: PersistenceErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@final
@dataclass(frozen=True, slots=True)
class PersistenceReceipt:
    replayed: bool
    snapshot: RuntimeSnapshot
    transition_id: str
    outbox_ids: tuple[str, ...]
    storage_revision: int
    previous_snapshot: RuntimeSnapshot
    command: RuntimeCommand
    result: RuntimeTransitionResult

    def with_replayed(self, replayed: bool) -> PersistenceReceipt:
        return replace(self, replayed=replayed)


@final
@dataclass(frozen=True, slots=True)
class PendingOutboxItem:
    event_id: str
    event_type: str
    schema_version: str
    sequence: int
    content_sha256: str
    causation_id: str
    payload: RealtimeCoachingEventV2 | ReplanIntent | PlanOutcome
    plaintext: bytes


@final
@dataclass(frozen=True, slots=True)
class RecoveryResult:
    status: RecoveryStatus
    snapshot: RuntimeSnapshot
    policy_state: ExecutionPolicyState


@final
@dataclass(frozen=True, slots=True)
class RecoveryFailureResult:
    """Fail-closed recovery projection that cannot be executed as runtime state."""

    status: RecoveryStatus
    session_id: UUID
    error: PersistenceErrorCode
    executable: bool = False


@final
@dataclass(frozen=True, slots=True)
class RetentionReport:
    deleted_session_ids: tuple[UUID, ...]
    quarantined_session_ids: tuple[UUID, ...]


_DATACLASS_TYPES: dict[str, type[Any]] = {
    cls.__name__: cls
    for cls in (
        AcceptObservation,
        AdmittedPlanIdentity,
        AdmitVerifiedPlan,
        AdvanceToNextShot,
        BeginCompletion,
        BeginPlanLoad,
        CompletionReceipt,
        DrillExecutionProjection,
        ExecutionBlockProjection,
        ExecutionDecision,
        ExecutionPolicyState,
        FailRuntime,
        FinishCompletion,
        MetricEvaluation,
        MetricRuleProjection,
        ObservationEvaluation,
        ObservationKey,
        ObservationOutcomeReceipt,
        ObservationReceipt,
        PauseRuntime,
        PersistenceReceipt,
        PolicyReceipt,
        PresentFeedback,
        PresentSetup,
        RecoverRuntime,
        ReplanIntent,
        ResumeRuntime,
        RuntimeEffect,
        RuntimeSnapshot,
        RuntimeTransition,
        RuntimeTransitionError,
        RuntimeTransitionResult,
        SuccessPolicyProjection,
        VerifiedExecutionProjection,
        VerifiedPlanAdmission,
        WaitForShot,
    )
}
_ENUM_TYPES: dict[str, type[Enum]] = {
    cls.__name__: cls
    for cls in (
        AbstentionAction,
        CoachRuntimeState,
        CoachingEventType,
        EvaluationConclusion,
        ExecutionDecisionKind,
        MetricCombination,
        PlanOutcomeStatus,
        ReplanContinueAction,
        RuntimeEffectKind,
        RuntimeErrorCode,
    )
}
_MODEL_TYPES: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in (
        BlockOutcome,
        PlanOutcome,
        RealtimeCoachingEventV2,
        RealtimeShotObservation,
    )
}


def _encode(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, BaseModel):
        return {
            "$model": type(value).__name__,
            "value": _encode(value.model_dump(mode="json")),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": type(value).__name__,
            "value": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
                if not field.name.startswith("_")
            },
        }
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("only string-keyed dictionaries are serializable")
        return {key: _encode(item) for key, item in value.items()}
    raise TypeError(f"unsupported persistence type: {type(value).__name__}")


def _decode(value: object) -> object:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$bytes"}:
        return base64.b64decode(str(value["$bytes"]), validate=True)
    if set(value) == {"$uuid"}:
        return UUID(str(value["$uuid"]))
    if set(value) == {"$datetime"}:
        parsed = datetime.fromisoformat(str(value["$datetime"]))
        if parsed.tzinfo is None:
            raise ValueError("persisted datetime must be timezone aware")
        return parsed
    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise ValueError("invalid tuple envelope")
        return tuple(_decode(item) for item in items)
    if set(value) == {"$enum", "value"}:
        enum_type = _ENUM_TYPES.get(str(value["$enum"]))
        if enum_type is None:
            raise ValueError("unknown enum type")
        return enum_type(value["value"])
    if set(value) == {"$model", "value"}:
        model_type = _MODEL_TYPES.get(str(value["$model"]))
        if model_type is None:
            raise ValueError("unknown model type")
        decoded = _decode(value["value"])
        # The sealed envelope contains canonical JSON values (UUID/datetime strings), so
        # Pydantic must perform its normal JSON-compatible parsing before validating all
        # model invariants.  The outer type tag remains a closed allowlist.
        return model_type.model_validate_json(
            json.dumps(decoded, ensure_ascii=False, separators=(",", ":"))
        )
    if set(value) == {"$dataclass", "value"}:
        name = str(value["$dataclass"])
        dataclass_type = _DATACLASS_TYPES.get(name)
        if dataclass_type is None or not isinstance(value["value"], dict):
            raise ValueError("unknown dataclass type")
        decoded_values = {key: _decode(item) for key, item in value["value"].items()}
        if dataclass_type is VerifiedExecutionProjection:
            decoded_values.pop("schema_version", None)
            decoded_values.pop("projection_sha256", None)
            return _issue_verified_execution_projection(**decoded_values)
        return dataclass_type(**decoded_values)
    return {key: _decode(item) for key, item in value.items()}


def _codec_dump(value: object) -> bytes:
    data = json.dumps(
        {"codec": CODEC_VERSION, "payload": _encode(value)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not data or len(data) > _MAX_PAYLOAD_BYTES:
        raise ValueError("persistence payload exceeds size limit")
    return data


def _codec_load(data: bytes) -> object:
    if not data or len(data) > _MAX_PAYLOAD_BYTES:
        raise ValueError("invalid persistence payload size")
    raw = json.loads(data)
    if not isinstance(raw, dict) or set(raw) != {"codec", "payload"}:
        raise ValueError("invalid persistence codec envelope")
    if raw["codec"] != CODEC_VERSION:
        raise ValueError("unsupported persistence codec")
    return _decode(raw["payload"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("persisted timestamp must be timezone aware")
    return parsed


class SqliteRuntimeStore:
    """One-device encrypted runtime repository with transactional inbox/outbox."""

    def __init__(
        self,
        path: str | Path,
        *,
        device_root_key: bytes,
        now: Callable[[], datetime] | None = None,
        busy_timeout_ms: int = 250,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if type(device_root_key) is not bytes or len(device_root_key) != 32:
            raise RuntimePersistenceError(
                PersistenceErrorCode.KEY_UNAVAILABLE,
                "device root key must be 32 bytes",
            )
        if type(busy_timeout_ms) is not int or busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self.path = Path(path)
        self._root_key = device_root_key
        self._now = now or (lambda: datetime.now(UTC))
        self._fault = fault_injector or (lambda _stage: None)
        self._connection: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                timeout=busy_timeout_ms / 1000,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA secure_delete=ON")
            self._migrate()
        except RuntimePersistenceError:
            self.close()
            raise
        except sqlite3.OperationalError as exc:
            self.close()
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite database is busy during initialization",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "SQLite database cannot be opened or migrated",
            ) from exc
        except sqlite3.DatabaseError as exc:
            self.close()
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "SQLite database cannot be opened or migrated",
            ) from exc

    @property
    def schema_version(self) -> int:
        row = self._db.execute(
            "SELECT schema_version FROM runtime_catalog WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE, "runtime catalog is missing"
            )
        return int(row[0])

    @property
    def accepting_new_sessions(self) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM runtime_sessions WHERE quarantined_at IS NOT NULL LIMIT 1"
        ).fetchone()
        return row is None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("persistence store is closed")
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def checkpoint(self) -> None:
        self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @staticmethod
    def _is_locked(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    @staticmethod
    def _schema_columns() -> dict[str, tuple[str, ...]]:
        return {
            "runtime_catalog": ("singleton", "schema_version", "codec_version"),
            "runtime_sessions": (
                "session_token",
                "state",
                "snapshot_sequence",
                "storage_revision",
                "wrapped_key_nonce",
                "wrapped_key_ciphertext",
                "snapshot_sha256",
                "snapshot_nonce",
                "snapshot_ciphertext",
                "policy_sha256",
                "policy_nonce",
                "policy_ciphertext",
                "committed_at",
                "completed_at",
                "last_ack_at",
                "quarantined_at",
            ),
            "runtime_events": (
                "event_id",
                "session_token",
                "event_type",
                "schema_version",
                "storage_revision",
                "event_ordinal",
                "sequence",
                "content_sha256",
                "metadata_sha256",
                "causation_id",
                "payload_nonce",
                "payload_ciphertext",
                "created_at",
            ),
            "runtime_receipts": (
                "transition_id",
                "session_token",
                "storage_revision",
                "command_type",
                "command_sha256",
                "previous_snapshot_sha256",
                "result_sha256",
                "receipt_sha256",
                "receipt_nonce",
                "receipt_ciphertext",
                "created_at",
            ),
            "runtime_inbox": (
                "event_id",
                "session_token",
                "transition_id",
                "event_type",
                "content_sha256",
                "metadata_sha256",
            ),
            "runtime_outbox": (
                "event_id",
                "session_token",
                "event_type",
                "schema_version",
                "storage_revision",
                "sequence",
                "content_sha256",
                "metadata_sha256",
                "causation_id",
                "payload_nonce",
                "payload_ciphertext",
                "manifest_sha256",
                "manifest_nonce",
                "manifest_ciphertext",
                "created_at",
            ),
            "runtime_outbox_acks": (
                "event_id",
                "event_type",
                "acked_at",
                "session_token",
                "content_sha256",
                "ack_sha256",
                "ack_nonce",
                "ack_ciphertext",
            ),
            "deletion_receipts": ("session_token", "erased_at", "reason_code"),
        }

    def _validate_schema_manifest(self) -> None:
        expected = self._schema_columns()
        objects = self._db.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        tables = {str(row[1]) for row in objects if row[0] == "table"}
        if tables != set(expected):
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime schema table manifest is invalid",
            )
        for table, columns in expected.items():
            actual = tuple(str(row[1]) for row in self._db.execute(f"PRAGMA table_info({table})"))
            if actual != columns:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.CORRUPT_DATABASE,
                    f"runtime schema columns are invalid for {table}",
                )
        required_indexes = {
            "runtime_events_revision",
            "runtime_outbox_pending",
            "runtime_receipts_revision",
            "runtime_ack_session",
        }
        indexes = {str(row[1]) for row in objects if row[0] == "index"}
        if indexes != required_indexes:
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime schema index manifest is invalid",
            )
        required_triggers = {
            "runtime_events_no_update",
            "runtime_events_no_delete",
            "runtime_receipts_no_update",
            "runtime_receipts_no_delete",
            "runtime_outbox_no_update",
            "runtime_outbox_no_delete",
            "runtime_outbox_acks_no_update",
            "runtime_outbox_acks_no_delete",
            "deletion_receipts_no_update",
            "deletion_receipts_no_delete",
        }
        triggers = {str(row[1]) for row in objects if row[0] == "trigger"}
        if triggers != required_triggers:
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime schema trigger manifest is invalid",
            )
        expected_primary_keys = {
            "runtime_catalog": ("singleton",),
            "runtime_sessions": ("session_token",),
            "runtime_events": ("event_id", "event_type"),
            "runtime_receipts": ("transition_id",),
            "runtime_inbox": ("event_id",),
            "runtime_outbox": ("event_id", "event_type"),
            "runtime_outbox_acks": ("event_id", "event_type", "acked_at"),
            "deletion_receipts": ("session_token",),
        }
        for table, expected_pk in expected_primary_keys.items():
            actual_pk = tuple(
                str(row[1])
                for row in sorted(
                    self._db.execute(f"PRAGMA table_info({table})"),
                    key=lambda item: int(item[5]) if int(item[5]) else 999,
                )
                if int(row[5]) > 0
            )
            if actual_pk != expected_pk:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.CORRUPT_DATABASE,
                    f"runtime schema primary key manifest is invalid for {table}",
                )
        foreign_keys = {
            table: tuple(
                (str(row[2]), str(row[3]), str(row[4]), str(row[6]))
                for row in self._db.execute(f"PRAGMA foreign_key_list({table})")
            )
            for table in expected
        }
        for table in ("runtime_events", "runtime_receipts", "runtime_inbox", "runtime_outbox"):
            session_fk = (
                "runtime_sessions",
                "session_token",
                "session_token",
                "CASCADE",
            )
            if session_fk not in foreign_keys[table]:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.CORRUPT_DATABASE,
                    f"runtime schema foreign key manifest is invalid for {table}",
                )
        if (
            "runtime_receipts",
            "transition_id",
            "transition_id",
            "CASCADE",
        ) not in foreign_keys["runtime_inbox"]:
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime inbox receipt foreign key manifest is invalid",
            )
        ack_fks = foreign_keys["runtime_outbox_acks"]
        if (
            "runtime_sessions",
            "session_token",
            "session_token",
            "CASCADE",
        ) not in ack_fks or not {
            ("runtime_outbox", "event_id", "event_id", "CASCADE"),
            ("runtime_outbox", "event_type", "event_type", "CASCADE"),
        } <= set(ack_fks):
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime ACK foreign key manifest is invalid",
            )
        integrity = self._db.execute("PRAGMA foreign_key_check").fetchall()
        if integrity:
            raise RuntimePersistenceError(
                PersistenceErrorCode.CORRUPT_DATABASE,
                "runtime schema contains foreign key violations",
            )

    def _migrate(self) -> None:
        existing = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runtime_catalog'"
        ).fetchone()
        if existing is not None:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                columns = {
                    str(row[1]) for row in self._db.execute("PRAGMA table_info(runtime_catalog)")
                }
                if columns != {"singleton", "schema_version", "codec_version"}:
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.CORRUPT_DATABASE,
                        "runtime catalog shape is invalid",
                    )
                row = self._db.execute(
                    "SELECT singleton, schema_version, codec_version FROM runtime_catalog"
                ).fetchall()
                if len(row) != 1 or int(row[0][0]) != 1:
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.CORRUPT_DATABASE,
                        "runtime catalog singleton is invalid",
                    )
                if int(row[0][1]) != SCHEMA_VERSION or row[0][2] != CODEC_VERSION:
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.UNSUPPORTED_MIGRATION,
                        "runtime database uses an unsupported schema",
                    )
                self._validate_schema_manifest()
                self._db.commit()
            except Exception:
                if self._db.in_transaction:
                    self._db.rollback()
                raise
            return

        try:
            self._db.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE runtime_catalog(
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL,
                    codec_version TEXT NOT NULL
                );

                CREATE TABLE runtime_sessions(
                    session_token TEXT PRIMARY KEY CHECK(length(session_token) = 64),
                    state TEXT NOT NULL,
                    snapshot_sequence INTEGER NOT NULL CHECK(snapshot_sequence >= 0),
                    storage_revision INTEGER NOT NULL CHECK(storage_revision >= 1),
                    wrapped_key_nonce BLOB NOT NULL,
                    wrapped_key_ciphertext BLOB NOT NULL,
                    snapshot_sha256 TEXT NOT NULL CHECK(length(snapshot_sha256) = 64),
                    snapshot_nonce BLOB NOT NULL,
                    snapshot_ciphertext BLOB NOT NULL,
                    policy_sha256 TEXT NOT NULL CHECK(length(policy_sha256) = 64),
                    policy_nonce BLOB NOT NULL,
                    policy_ciphertext BLOB NOT NULL,
                    committed_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_ack_at TEXT,
                    quarantined_at TEXT
                );
                CREATE TABLE runtime_events(
                    event_id TEXT NOT NULL,
                    session_token TEXT NOT NULL REFERENCES runtime_sessions(session_token)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    storage_revision INTEGER NOT NULL CHECK(storage_revision >= 1),
                    event_ordinal INTEGER NOT NULL CHECK(event_ordinal >= 0),
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
                    metadata_sha256 TEXT NOT NULL CHECK(length(metadata_sha256) = 64),
                    causation_id TEXT NOT NULL,
                    payload_nonce BLOB NOT NULL,
                    payload_ciphertext BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(event_id, event_type),
                    UNIQUE(session_token, storage_revision, event_ordinal)
                );
                CREATE UNIQUE INDEX runtime_events_revision
                    ON runtime_events(session_token, storage_revision, event_ordinal);
                CREATE TRIGGER runtime_events_no_update
                BEFORE UPDATE ON runtime_events BEGIN
                    SELECT RAISE(ABORT, 'runtime_events append-only');
                END;
                CREATE TRIGGER runtime_events_no_delete
                BEFORE DELETE ON runtime_events
                WHEN EXISTS (SELECT 1 FROM runtime_sessions WHERE session_token = OLD.session_token)
                BEGIN
                    SELECT RAISE(ABORT, 'runtime_events append-only');
                END;
                CREATE TABLE runtime_receipts(
                    transition_id TEXT PRIMARY KEY,
                    session_token TEXT NOT NULL REFERENCES runtime_sessions(session_token)
                        ON DELETE CASCADE,
                    storage_revision INTEGER NOT NULL CHECK(storage_revision >= 1),
                    command_type TEXT NOT NULL,
                    command_sha256 TEXT NOT NULL CHECK(length(command_sha256) = 64),
                    previous_snapshot_sha256 TEXT NOT NULL
                        CHECK(length(previous_snapshot_sha256) = 64),
                    result_sha256 TEXT NOT NULL CHECK(length(result_sha256) = 64),
                    receipt_sha256 TEXT NOT NULL CHECK(length(receipt_sha256) = 64),
                    receipt_nonce BLOB NOT NULL,
                    receipt_ciphertext BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_token, storage_revision)
                );
                CREATE UNIQUE INDEX runtime_receipts_revision
                    ON runtime_receipts(session_token, storage_revision);
                CREATE TRIGGER runtime_receipts_no_update
                BEFORE UPDATE ON runtime_receipts BEGIN
                    SELECT RAISE(ABORT, 'runtime_receipts append-only');
                END;
                CREATE TRIGGER runtime_receipts_no_delete
                BEFORE DELETE ON runtime_receipts
                WHEN EXISTS (SELECT 1 FROM runtime_sessions WHERE session_token = OLD.session_token)
                BEGIN
                    SELECT RAISE(ABORT, 'runtime_receipts append-only');
                END;
                CREATE TABLE runtime_inbox(
                    event_id TEXT PRIMARY KEY,
                    session_token TEXT NOT NULL REFERENCES runtime_sessions(session_token)
                        ON DELETE CASCADE,
                    transition_id TEXT NOT NULL REFERENCES runtime_receipts(transition_id)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
                    metadata_sha256 TEXT NOT NULL CHECK(length(metadata_sha256) = 64)
                );
                CREATE TABLE runtime_outbox(
                    event_id TEXT NOT NULL,
                    session_token TEXT NOT NULL REFERENCES runtime_sessions(session_token)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    storage_revision INTEGER NOT NULL CHECK(storage_revision >= 1),
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
                    metadata_sha256 TEXT NOT NULL CHECK(length(metadata_sha256) = 64),
                    causation_id TEXT NOT NULL,
                    payload_nonce BLOB NOT NULL,
                    payload_ciphertext BLOB NOT NULL,
                    manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
                    manifest_nonce BLOB NOT NULL,
                    manifest_ciphertext BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(event_id, event_type)
                );
                CREATE INDEX runtime_outbox_pending
                    ON runtime_outbox(session_token, storage_revision, sequence);
                CREATE TRIGGER runtime_outbox_no_update
                BEFORE UPDATE ON runtime_outbox BEGIN
                    SELECT RAISE(ABORT, 'runtime_outbox append-only');
                END;
                CREATE TRIGGER runtime_outbox_no_delete
                BEFORE DELETE ON runtime_outbox
                WHEN EXISTS (SELECT 1 FROM runtime_sessions WHERE session_token = OLD.session_token)
                BEGIN
                    SELECT RAISE(ABORT, 'runtime_outbox append-only');
                END;
                CREATE TABLE runtime_outbox_acks(
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    acked_at TEXT NOT NULL,
                    session_token TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
                    ack_sha256 TEXT NOT NULL CHECK(length(ack_sha256) = 64),
                    ack_nonce BLOB NOT NULL,
                    ack_ciphertext BLOB NOT NULL,
                    PRIMARY KEY(event_id, event_type, acked_at),
                    FOREIGN KEY(event_id, event_type)
                        REFERENCES runtime_outbox(event_id, event_type) ON DELETE CASCADE,
                    FOREIGN KEY(session_token)
                        REFERENCES runtime_sessions(session_token) ON DELETE CASCADE
                );
                CREATE INDEX runtime_ack_session
                    ON runtime_outbox_acks(session_token, acked_at);
                CREATE TRIGGER runtime_outbox_acks_no_update
                BEFORE UPDATE ON runtime_outbox_acks BEGIN
                    SELECT RAISE(ABORT, 'runtime_outbox_acks append-only');
                END;
                CREATE TRIGGER runtime_outbox_acks_no_delete
                BEFORE DELETE ON runtime_outbox_acks
                WHEN EXISTS (SELECT 1 FROM runtime_sessions WHERE session_token = OLD.session_token)
                BEGIN
                    SELECT RAISE(ABORT, 'runtime_outbox_acks append-only');
                END;
                CREATE TABLE deletion_receipts(
                    session_token TEXT PRIMARY KEY,
                    erased_at TEXT NOT NULL,
                    reason_code TEXT NOT NULL
                );
                CREATE TRIGGER deletion_receipts_no_update
                BEFORE UPDATE ON deletion_receipts BEGIN
                    SELECT RAISE(ABORT, 'deletion_receipts append-only');
                END;
                CREATE TRIGGER deletion_receipts_no_delete
                BEFORE DELETE ON deletion_receipts BEGIN
                    SELECT RAISE(ABORT, 'deletion_receipts append-only');
                END;
                INSERT INTO runtime_catalog(singleton, schema_version, codec_version)
                    VALUES(1, 2, 'edge-runtime-sealed-json/v1');
                COMMIT;
                """
            )
            self._validate_schema_manifest()
        except sqlite3.DatabaseError:
            if self._db.in_transaction:
                self._db.rollback()
            raise

    def _session_token(self, session_id: UUID) -> str:
        if not isinstance(session_id, UUID) or session_id.version != 4:
            raise RuntimePersistenceError(
                PersistenceErrorCode.INVALID_PAYLOAD, "session id must be UUID4"
            )
        return hmac.new(
            self._root_key,
            b"edge-runtime-session-token/v1\0" + session_id.bytes,
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _aad(
        token: str,
        sequence: int,
        kind: str,
        content_sha256: str,
        metadata_sha256: str = "0" * 64,
    ) -> bytes:
        return (
            f"{CODEC_VERSION}|{token}|{sequence}|{kind}|{content_sha256}|{metadata_sha256}"
        ).encode("ascii")

    @staticmethod
    def _metadata_sha256(**values: object) -> str:
        return _sha256(
            json.dumps(
                values,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )

    def _wrap_key(self, token: str, data_key: bytes) -> tuple[bytes, bytes]:
        nonce = secrets.token_bytes(12)
        aad = self._aad(token, 0, "session-key", "0" * 64)
        return nonce, AESGCM(self._root_key).encrypt(nonce, data_key, aad)

    def _unwrap_key(self, row: sqlite3.Row) -> bytes:
        token = str(row["session_token"])
        aad = self._aad(token, 0, "session-key", "0" * 64)
        try:
            return AESGCM(self._root_key).decrypt(
                bytes(row["wrapped_key_nonce"]),
                bytes(row["wrapped_key_ciphertext"]),
                aad,
            )
        except InvalidTag as exc:
            raise RuntimePersistenceError(
                PersistenceErrorCode.KEY_UNAVAILABLE,
                "session data key cannot be unwrapped",
            ) from exc

    def _seal(
        self,
        data_key: bytes,
        token: str,
        sequence: int,
        kind: str,
        plaintext: bytes,
        metadata_sha256: str = "0" * 64,
    ) -> tuple[str, bytes, bytes]:
        digest = _sha256(plaintext)
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(data_key).encrypt(
            nonce, plaintext, self._aad(token, sequence, kind, digest, metadata_sha256)
        )
        return digest, nonce, ciphertext

    def _open(
        self,
        data_key: bytes,
        token: str,
        sequence: int,
        kind: str,
        digest: str,
        nonce: bytes,
        ciphertext: bytes,
        metadata_sha256: str = "0" * 64,
    ) -> bytes:
        try:
            plaintext = AESGCM(data_key).decrypt(
                nonce,
                ciphertext,
                self._aad(token, sequence, kind, digest, metadata_sha256),
            )
        except InvalidTag as exc:
            raise RuntimePersistenceError(
                PersistenceErrorCode.INTEGRITY_MISMATCH,
                "sealed runtime record failed AEAD verification",
            ) from exc
        if not hmac.compare_digest(_sha256(plaintext), digest):
            raise RuntimePersistenceError(
                PersistenceErrorCode.INTEGRITY_MISMATCH,
                "sealed runtime record content hash mismatch",
            )
        return plaintext

    @staticmethod
    def _authority_item(
        payload: object,
    ) -> tuple[str, str, str, int, bytes]:
        if type(payload) is RealtimeCoachingEventV2:
            return (
                str(payload.event_id),
                type(payload).__name__,
                payload.schema_version,
                payload.sequence,
                _codec_dump(payload),
            )
        if type(payload) is ReplanIntent:
            return (
                payload.intent_id,
                type(payload).__name__,
                payload.schema_version,
                payload.shot_sequence,
                _codec_dump(payload),
            )
        if type(payload) is PlanOutcome:
            if payload.direct_ability_update_allowed is not False:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INVALID_PAYLOAD,
                    "PlanOutcome cannot directly update long-term ability",
                )
            return (
                str(payload.outcome_id),
                type(payload).__name__,
                payload.schema_version,
                1,
                _codec_dump(payload),
            )
        raise RuntimePersistenceError(
            PersistenceErrorCode.INVALID_PAYLOAD,
            "outbox payload is not a frozen authority contract",
        )

    @staticmethod
    def _projection_structure_valid(projection: object) -> bool:
        if type(projection) is not VerifiedExecutionProjection or not projection.integrity_valid():
            return False
        if (
            type(projection.admission_id) is not str
            or not projection.admission_id.strip()
            or type(projection.plan_version) is not int
            or projection.plan_version < 1
            or type(projection.publication_sequence) is not int
            or projection.publication_sequence < 1
            or not projection.blocks
            or any(type(block) is not ExecutionBlockProjection for block in projection.blocks)
            or tuple(block.order for block in projection.blocks)
            != tuple(range(1, len(projection.blocks) + 1))
        ):
            return False
        for block in projection.blocks:
            drill = block.drill
            if (
                type(block.target_repetitions) is not int
                or block.target_repetitions < 1
                or type(drill) is not DrillExecutionProjection
                or not (
                    drill.minimum_runtime_difficulty
                    <= drill.initial_difficulty
                    <= drill.maximum_runtime_difficulty
                )
                or not drill.success_policy.metric_rules
            ):
                return False
        return True

    @classmethod
    def _policy_invariant_error(
        cls, policy: ExecutionPolicyState, snapshot: RuntimeSnapshot
    ) -> str | None:
        projection = policy.projection
        if not cls._projection_structure_valid(projection):
            return "policy projection structure is invalid"
        if snapshot.plan is None or snapshot.plan.execution_projection != projection:
            return "policy projection does not match runtime snapshot"
        if policy.block_index < 0 or policy.block_index >= len(projection.blocks):
            return "policy block index is outside projection"
        receipt_sequences = tuple(receipt.shot_sequence for receipt in policy.receipts)
        if receipt_sequences != tuple(range(1, len(receipt_sequences) + 1)):
            return "policy receipt sequence is not contiguous"
        if policy.expected_shot_sequence != len(policy.receipts) + 1:
            return "policy expected sequence does not follow receipts"
        if any(
            receipt.decision.next_shot_sequence != receipt.shot_sequence + 1
            or not receipt.evidence_sha256
            for receipt in policy.receipts
        ):
            return "policy receipt decision sequence is invalid"
        drill = (
            projection.terminal_fallback
            if policy.fallback_active
            else projection.blocks[policy.block_index].drill
        )
        if policy.current_drill_key != drill.key:
            return "policy drill key does not match selected block"
        if not (
            drill.minimum_runtime_difficulty
            <= policy.difficulty
            <= drill.maximum_runtime_difficulty
        ):
            return "policy difficulty is outside published bounds"
        if policy.receipts:
            last = policy.receipts[-1].decision
            if last.difficulty != policy.difficulty:
                return "policy difficulty does not match latest receipt"
            if policy.safe_paused != (last.kind is ExecutionDecisionKind.SAFE_PAUSE):
                return "policy pause state does not match latest receipt"
            if policy.session_complete != (last.kind is ExecutionDecisionKind.BEGIN_COMPLETION):
                return "policy completion state does not match latest receipt"
        effective_state = snapshot.state
        if snapshot.state is CoachRuntimeState.PAUSED:
            effective_state = snapshot.resume_state
        elif snapshot.state is CoachRuntimeState.DEGRADED:
            effective_state = snapshot.last_trusted_state
        if effective_state in {
            CoachRuntimeState.ANALYZING_SHOT,
            CoachRuntimeState.PRESENTING_FEEDBACK,
        }:
            expected = snapshot.shot_sequence + 1
        elif snapshot.state in {CoachRuntimeState.COMPLETING, CoachRuntimeState.COMPLETED} and (
            policy.receipts
        ):
            expected = snapshot.shot_sequence + 1
        else:
            expected = snapshot.shot_sequence
        if policy.expected_shot_sequence != expected:
            return "policy and runtime shot sequences do not reconcile"
        return None

    @staticmethod
    def _outbox_semantics_valid(
        payload: object,
        snapshot: RuntimeSnapshot,
        policy: ExecutionPolicyState,
    ) -> bool:
        if snapshot.plan is None or snapshot.session_id is None:
            return False
        projection = snapshot.plan.execution_projection
        if projection is None:
            return False
        if type(payload) is ReplanIntent:
            return (
                payload.session_id == snapshot.session_id
                and payload.plan_id == snapshot.plan.plan_id
                and payload.admission_id == snapshot.plan.admission_id
                and payload.block_index == policy.block_index
                and payload.shot_sequence
                in {
                    snapshot.shot_sequence,
                    policy.expected_shot_sequence - 1,
                }
            )
        if type(payload) is RealtimeCoachingEventV2:
            lineage = payload.lineage
            block = projection.blocks[policy.block_index]
            return (
                lineage.session_id == snapshot.session_id
                and lineage.plan_id == snapshot.plan.plan_id
                and lineage.admission_id == snapshot.plan.admission_id
                and lineage.block_id == block.block_id
                and lineage.block_order == block.order
                and (
                    lineage.shot_sequence is None or lineage.shot_sequence == snapshot.shot_sequence
                )
            )
        if type(payload) is PlanOutcome:
            return (
                payload.session_id == snapshot.session_id
                and payload.plan_id == snapshot.plan.plan_id
                and payload.user_id == projection.user_id
                and payload.direct_ability_update_allowed is False
            )
        return False

    def commit_transition(
        self,
        result: RuntimeTransitionResult,
        policy_state: ExecutionPolicyState,
        *,
        previous_snapshot: RuntimeSnapshot,
        command: RuntimeCommand,
        input_observation: RealtimeShotObservation | None = None,
        outbox: tuple[object, ...] = (),
        committed_at: datetime | None = None,
    ) -> PersistenceReceipt:
        recomputed = (
            transition(previous_snapshot, command)
            if type(previous_snapshot) is RuntimeSnapshot
            else None
        )
        if (
            type(result) is not RuntimeTransitionResult
            or result.error is not None
            or result.transition is None
            or type(previous_snapshot) is not RuntimeSnapshot
            or _snapshot_invariant_error(previous_snapshot) is not None
            or recomputed != result
            or type(result.snapshot) is not RuntimeSnapshot
            or _snapshot_invariant_error(result.snapshot) is not None
            or result.snapshot.session_id is None
            or result.snapshot.plan is None
            or type(policy_state) is not ExecutionPolicyState
            or policy_state.projection != result.snapshot.plan.execution_projection
            or self._policy_invariant_error(policy_state, result.snapshot) is not None
        ):
            raise RuntimePersistenceError(
                PersistenceErrorCode.INVALID_PAYLOAD,
                "only a valid authoritative runtime transition can be persisted",
            )
        session_id = result.snapshot.session_id
        token = self._session_token(session_id)
        now = committed_at or self._now()
        timestamp = _iso(now)
        transition_id = result.transition.transition_id
        authority = tuple(self._authority_item(item) for item in outbox)
        if any(
            not self._outbox_semantics_valid(item, result.snapshot, policy_state) for item in outbox
        ):
            raise RuntimePersistenceError(
                PersistenceErrorCode.INVALID_PAYLOAD,
                "outbox authority does not belong to the current runtime transition",
            )
        observation_bytes = None
        observation_id = None
        observation_hash = None
        if input_observation is not None:
            if (
                type(input_observation) is not RealtimeShotObservation
                or input_observation.session_id != session_id
                or type(command) is not AcceptObservation
                or command.observation != input_observation
                or result.transition.command_type != AcceptObservation.__name__
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INVALID_PAYLOAD,
                    "input observation does not belong to runtime session",
                )
            observation_bytes = _codec_dump(input_observation)
            observation_id = str(input_observation.event_id)
            observation_hash = _sha256(observation_bytes)
        try:
            self._db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy",
                    retryable=True,
                ) from exc
            raise
        committed = False
        try:
            if (
                self._db.execute(
                    "SELECT 1 FROM deletion_receipts WHERE session_token = ?", (token,)
                ).fetchone()
                is not None
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.QUARANTINED,
                    "deleted runtime session cannot be resurrected",
                )
            session_row = self._db.execute(
                "SELECT * FROM runtime_sessions WHERE session_token = ?", (token,)
            ).fetchone()
            existing_receipt = self._db.execute(
                "SELECT * FROM runtime_receipts WHERE transition_id = ?", (transition_id,)
            ).fetchone()
            if existing_receipt is not None:
                if session_row is None or existing_receipt["session_token"] != token:
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.IDENTITY_CONFLICT,
                        "transition receipt owner does not match runtime session",
                    )
                data_key = self._unwrap_key(session_row)
                storage_revision = int(existing_receipt["storage_revision"])
                plaintext = self._open(
                    data_key,
                    token,
                    storage_revision,
                    "persistence-receipt",
                    str(existing_receipt["receipt_sha256"]),
                    bytes(existing_receipt["receipt_nonce"]),
                    bytes(existing_receipt["receipt_ciphertext"]),
                )
                receipt = _codec_load(plaintext)
                command_sha = _sha256(_codec_dump(command))
                expected = PersistenceReceipt(
                    False,
                    result.snapshot,
                    transition_id,
                    tuple(item[0] for item in authority),
                    storage_revision,
                    previous_snapshot,
                    command,
                    result,
                )
                if (
                    type(receipt) is not PersistenceReceipt
                    or receipt != expected
                    or existing_receipt["command_type"] != type(command).__name__
                    or not hmac.compare_digest(str(existing_receipt["command_sha256"]), command_sha)
                ):
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.IDENTITY_CONFLICT,
                        "transition replay does not match persisted authority",
                    )
                if observation_id is not None:
                    inbox = self._db.execute(
                        "SELECT * FROM runtime_inbox WHERE event_id = ?", (observation_id,)
                    ).fetchone()
                    if (
                        inbox is None
                        or inbox["session_token"] != token
                        or inbox["transition_id"] != transition_id
                        or inbox["event_type"] != type(input_observation).__name__
                        or not hmac.compare_digest(
                            str(inbox["content_sha256"]), str(observation_hash)
                        )
                    ):
                        raise RuntimePersistenceError(
                            PersistenceErrorCode.IDENTITY_CONFLICT,
                            "inbox replay does not bind the current transition",
                        )
                for event_id, event_type, _schema, _sequence, payload_bytes in authority:
                    outbox_row = self._db.execute(
                        "SELECT session_token, content_sha256 FROM runtime_outbox "
                        "WHERE event_id = ? AND event_type = ?",
                        (event_id, event_type),
                    ).fetchone()
                    if (
                        outbox_row is None
                        or outbox_row["session_token"] != token
                        or not hmac.compare_digest(
                            str(outbox_row["content_sha256"]), _sha256(payload_bytes)
                        )
                    ):
                        raise RuntimePersistenceError(
                            PersistenceErrorCode.IDENTITY_CONFLICT,
                            "replayed outbox authority is incomplete",
                        )
                self._db.rollback()
                return receipt.with_replayed(True)

            if session_row is None:
                if (
                    self._db.execute(
                        "SELECT 1 FROM runtime_sessions WHERE quarantined_at IS NOT NULL LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.QUARANTINED,
                        "device has overdue authority outbox and cannot start another session",
                    )
                data_key = secrets.token_bytes(32)
                wrapped_nonce, wrapped_ciphertext = self._wrap_key(token, data_key)
                storage_revision = 1
            else:
                data_key = self._unwrap_key(session_row)
                previous_revision = int(session_row["storage_revision"])
                previous_plaintext = self._open(
                    data_key,
                    token,
                    previous_revision,
                    "runtime-snapshot",
                    str(session_row["snapshot_sha256"]),
                    bytes(session_row["snapshot_nonce"]),
                    bytes(session_row["snapshot_ciphertext"]),
                )
                previous = _codec_load(previous_plaintext)
                if type(previous) is not RuntimeSnapshot or previous != previous_snapshot:
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.IDENTITY_CONFLICT,
                        "runtime session authority changed across transitions",
                    )
                storage_revision = previous_revision + 1
                wrapped_nonce = bytes(session_row["wrapped_key_nonce"])
                wrapped_ciphertext = bytes(session_row["wrapped_key_ciphertext"])

            snapshot_plaintext = _codec_dump(result.snapshot)
            snapshot_sha, snapshot_nonce, snapshot_ciphertext = self._seal(
                data_key,
                token,
                storage_revision,
                "runtime-snapshot",
                snapshot_plaintext,
            )
            policy_plaintext = _codec_dump(policy_state)
            policy_sha, policy_nonce, policy_ciphertext = self._seal(
                data_key,
                token,
                storage_revision,
                "policy-state",
                policy_plaintext,
            )
            completed_at = (
                timestamp if result.snapshot.state is CoachRuntimeState.COMPLETED else None
            )
            self._db.execute(
                """
                INSERT INTO runtime_sessions(
                    session_token, state, snapshot_sequence,
                    storage_revision,
                    wrapped_key_nonce, wrapped_key_ciphertext,
                    snapshot_sha256, snapshot_nonce, snapshot_ciphertext,
                    policy_sha256, policy_nonce, policy_ciphertext,
                    committed_at, completed_at, last_ack_at, quarantined_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(session_token) DO UPDATE SET
                    state=excluded.state,
                    snapshot_sequence=excluded.snapshot_sequence,
                    storage_revision=excluded.storage_revision,
                    snapshot_sha256=excluded.snapshot_sha256,
                    snapshot_nonce=excluded.snapshot_nonce,
                    snapshot_ciphertext=excluded.snapshot_ciphertext,
                    policy_sha256=excluded.policy_sha256,
                    policy_nonce=excluded.policy_nonce,
                    policy_ciphertext=excluded.policy_ciphertext,
                    committed_at=excluded.committed_at,
                    completed_at=COALESCE(excluded.completed_at, runtime_sessions.completed_at)
                """,
                (
                    token,
                    result.snapshot.state.value,
                    result.snapshot.transition_count,
                    storage_revision,
                    wrapped_nonce,
                    wrapped_ciphertext,
                    snapshot_sha,
                    snapshot_nonce,
                    snapshot_ciphertext,
                    policy_sha,
                    policy_nonce,
                    policy_ciphertext,
                    timestamp,
                    completed_at,
                ),
            )
            self._fault("after_session")

            transition_plaintext = _codec_dump(result)
            transition_metadata = self._metadata_sha256(
                event_id=transition_id,
                event_type="RuntimeTransitionResult",
                schema_version="1.0",
                storage_revision=storage_revision,
                event_ordinal=0,
                sequence=result.snapshot.transition_count,
                causation_id=observation_id or transition_id,
                exists=True,
            )
            transition_sha, transition_nonce, transition_ciphertext = self._seal(
                data_key,
                token,
                storage_revision,
                "RuntimeTransitionResult",
                transition_plaintext,
                transition_metadata,
            )
            self._insert_event(
                event_id=transition_id,
                token=token,
                event_type="RuntimeTransitionResult",
                schema_version="1.0",
                storage_revision=storage_revision,
                event_ordinal=0,
                sequence=result.snapshot.transition_count,
                digest=transition_sha,
                metadata_sha256=transition_metadata,
                causation_id=observation_id or transition_id,
                nonce=transition_nonce,
                ciphertext=transition_ciphertext,
                created_at=timestamp,
            )
            if observation_bytes is not None and observation_id is not None:
                observation_metadata = self._metadata_sha256(
                    event_id=observation_id,
                    event_type="RealtimeShotObservation",
                    schema_version=input_observation.schema_version,
                    storage_revision=storage_revision,
                    event_ordinal=1,
                    sequence=input_observation.shot_sequence,
                    causation_id=observation_id,
                    transition_id=transition_id,
                    exists=True,
                )
                obs_sha, obs_nonce, obs_ciphertext = self._seal(
                    data_key,
                    token,
                    storage_revision,
                    "RealtimeShotObservation",
                    observation_bytes,
                    observation_metadata,
                )
                self._insert_event(
                    event_id=observation_id,
                    token=token,
                    event_type="RealtimeShotObservation",
                    schema_version=input_observation.schema_version,
                    storage_revision=storage_revision,
                    event_ordinal=1,
                    sequence=input_observation.shot_sequence,
                    digest=obs_sha,
                    metadata_sha256=observation_metadata,
                    causation_id=observation_id,
                    nonce=obs_nonce,
                    ciphertext=obs_ciphertext,
                    created_at=timestamp,
                )
            self._fault("after_event")

            for event_id, event_type, schema_version, sequence, plaintext in authority:
                metadata = self._metadata_sha256(
                    event_id=event_id,
                    event_type=event_type,
                    schema_version=schema_version,
                    storage_revision=storage_revision,
                    sequence=sequence,
                    causation_id=transition_id,
                    exists=True,
                )
                digest, nonce, ciphertext = self._seal(
                    data_key, token, storage_revision, event_type, plaintext, metadata
                )
                manifest_plaintext = _codec_dump(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "schema_version": schema_version,
                        "storage_revision": storage_revision,
                        "sequence": sequence,
                        "content_sha256": digest,
                        "metadata_sha256": metadata,
                        "causation_id": transition_id,
                        "exists": True,
                    }
                )
                manifest_sha, manifest_nonce, manifest_ciphertext = self._seal(
                    data_key,
                    token,
                    storage_revision,
                    "outbox-manifest",
                    manifest_plaintext,
                    metadata,
                )
                self._db.execute(
                    """
                    INSERT INTO runtime_outbox(
                        event_id, session_token, event_type, schema_version, sequence,
                        storage_revision, content_sha256, metadata_sha256, causation_id,
                        payload_nonce, payload_ciphertext, manifest_sha256, manifest_nonce,
                        manifest_ciphertext, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id, event_type) DO NOTHING
                    """,
                    (
                        event_id,
                        token,
                        event_type,
                        schema_version,
                        sequence,
                        storage_revision,
                        digest,
                        metadata,
                        transition_id,
                        nonce,
                        ciphertext,
                        manifest_sha,
                        manifest_nonce,
                        manifest_ciphertext,
                        timestamp,
                    ),
                )
                row = self._db.execute(
                    "SELECT session_token, content_sha256 FROM runtime_outbox "
                    "WHERE event_id = ? AND event_type = ?",
                    (event_id, event_type),
                ).fetchone()
                if (
                    row is None
                    or row["session_token"] != token
                    or not hmac.compare_digest(str(row["content_sha256"]), digest)
                ):
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.IDENTITY_CONFLICT,
                        "outbox event identity conflict",
                    )
            self._fault("after_outbox")

            receipt = PersistenceReceipt(
                False,
                result.snapshot,
                transition_id,
                tuple(item[0] for item in authority),
                storage_revision,
                previous_snapshot,
                command,
                result,
            )
            receipt_plaintext = _codec_dump(receipt)
            receipt_sha, receipt_nonce, receipt_ciphertext = self._seal(
                data_key,
                token,
                storage_revision,
                "persistence-receipt",
                receipt_plaintext,
            )
            self._db.execute(
                """
                INSERT INTO runtime_receipts(
                    transition_id, session_token, storage_revision, command_type,
                    command_sha256, previous_snapshot_sha256, result_sha256,
                    receipt_sha256, receipt_nonce, receipt_ciphertext, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    token,
                    storage_revision,
                    type(command).__name__,
                    _sha256(_codec_dump(command)),
                    _sha256(_codec_dump(previous_snapshot)),
                    _sha256(_codec_dump(result)),
                    receipt_sha,
                    receipt_nonce,
                    receipt_ciphertext,
                    timestamp,
                ),
            )
            if observation_id is not None and observation_hash is not None:
                inbox_metadata = self._metadata_sha256(
                    event_id=observation_id,
                    event_type=type(input_observation).__name__,
                    transition_id=transition_id,
                    content_sha256=observation_hash,
                    exists=True,
                )
                self._db.execute(
                    """
                    INSERT INTO runtime_inbox(
                        event_id, session_token, transition_id, event_type,
                        content_sha256, metadata_sha256
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        token,
                        transition_id,
                        type(input_observation).__name__,
                        observation_hash,
                        inbox_metadata,
                    ),
                )
            self._fault("before_commit")
            self._db.commit()
            committed = True
            self._fault("after_commit")
            return receipt
        except RuntimePersistenceError:
            if not committed:
                self._db.rollback()
            raise
        except sqlite3.OperationalError as exc:
            if not committed:
                self._db.rollback()
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED,
                "SQLite transaction failed",
            ) from exc
        except Exception as exc:
            if not committed:
                self._db.rollback()
                raise RuntimePersistenceError(
                    PersistenceErrorCode.TRANSACTION_FAILED,
                    "runtime transaction was rolled back",
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.COMMIT_UNKNOWN,
                "SQLite commit succeeded but caller acknowledgement is unknown",
                retryable=True,
            ) from exc

    def _insert_event(
        self,
        *,
        event_id: str,
        token: str,
        event_type: str,
        schema_version: str,
        storage_revision: int,
        event_ordinal: int,
        sequence: int,
        digest: str,
        metadata_sha256: str,
        causation_id: str,
        nonce: bytes,
        ciphertext: bytes,
        created_at: str,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO runtime_events(
                event_id, session_token, event_type, schema_version, sequence,
                storage_revision, event_ordinal, content_sha256, metadata_sha256,
                causation_id, payload_nonce, payload_ciphertext, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id, event_type) DO NOTHING
            """,
            (
                event_id,
                token,
                event_type,
                schema_version,
                sequence,
                storage_revision,
                event_ordinal,
                digest,
                metadata_sha256,
                causation_id,
                nonce,
                ciphertext,
                created_at,
            ),
        )
        row = self._db.execute(
            "SELECT session_token, content_sha256 FROM runtime_events "
            "WHERE event_id = ? AND event_type = ?",
            (event_id, event_type),
        ).fetchone()
        if (
            row is None
            or row["session_token"] != token
            or not hmac.compare_digest(str(row["content_sha256"]), digest)
        ):
            raise RuntimePersistenceError(
                PersistenceErrorCode.IDENTITY_CONFLICT,
                "runtime event identity conflict",
            )

    def recover_session(self, session_id: UUID) -> RecoveryResult | RecoveryFailureResult:
        token = self._session_token(session_id)
        try:
            self._validate_schema_manifest()
            row = self._db.execute(
                "SELECT * FROM runtime_sessions WHERE session_token = ?", (token,)
            ).fetchone()
        except sqlite3.DatabaseError:
            return self._safe_recovery(
                session_id, RecoveryStatus.DEGRADED, PersistenceErrorCode.CORRUPT_DATABASE
            )
        if row is None:
            return self._safe_recovery(
                session_id, RecoveryStatus.PAUSED, PersistenceErrorCode.KEY_UNAVAILABLE
            )
        try:
            key = self._unwrap_key(row)
            sequence = int(row["snapshot_sequence"])
            storage_revision = int(row["storage_revision"])
            snapshot_plaintext = self._open(
                key,
                token,
                storage_revision,
                "runtime-snapshot",
                str(row["snapshot_sha256"]),
                bytes(row["snapshot_nonce"]),
                bytes(row["snapshot_ciphertext"]),
            )
            policy_plaintext = self._open(
                key,
                token,
                storage_revision,
                "policy-state",
                str(row["policy_sha256"]),
                bytes(row["policy_nonce"]),
                bytes(row["policy_ciphertext"]),
            )
            snapshot = _codec_load(snapshot_plaintext)
            policy = _codec_load(policy_plaintext)
            if (
                type(snapshot) is not RuntimeSnapshot
                or type(policy) is not ExecutionPolicyState
                or _snapshot_invariant_error(snapshot) is not None
                or snapshot.session_id != session_id
                or snapshot.plan is None
                or snapshot.plan.execution_projection != policy.projection
                or not self._projection_structure_valid(policy.projection)
                or self._policy_invariant_error(policy, snapshot) is not None
                or snapshot.transition_count != sequence
                or _codec_dump(snapshot) != snapshot_plaintext
                or _codec_dump(policy) != policy_plaintext
                or snapshot.state.value != row["state"]
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime snapshot invariants do not reconcile",
                )
            self._validate_session_records(row, key)
            return RecoveryResult(RecoveryStatus.EXACT, snapshot, policy)
        except RuntimePersistenceError as exc:
            status = (
                RecoveryStatus.PAUSED
                if exc.code is PersistenceErrorCode.KEY_UNAVAILABLE
                else RecoveryStatus.DEGRADED
            )
            return self._safe_recovery(session_id, status, exc.code)
        except (ValueError, TypeError, json.JSONDecodeError, sqlite3.DatabaseError):
            return self._safe_recovery(
                session_id, RecoveryStatus.DEGRADED, PersistenceErrorCode.INTEGRITY_MISMATCH
            )

    def _validate_session_records(self, session_row: sqlite3.Row, key: bytes) -> None:
        token = str(session_row["session_token"])
        rows = self._db.execute(
            "SELECT * FROM runtime_events WHERE session_token = ? "
            "ORDER BY storage_revision, event_ordinal",
            (token,),
        ).fetchall()
        known_ids = {str(row["event_id"]) for row in rows}
        for row in rows:
            plaintext = self._open(
                key,
                token,
                int(row["storage_revision"]),
                str(row["event_type"]),
                str(row["content_sha256"]),
                bytes(row["payload_nonce"]),
                bytes(row["payload_ciphertext"]),
                str(row["metadata_sha256"]),
            )
            if _codec_dump(_codec_load(plaintext)) != plaintext:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime event payload is not canonical",
                )
            causation = str(row["causation_id"])
            if causation not in known_ids and causation != str(row["event_id"]):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime event causation lineage is incomplete",
                )
        receipt_rows = self._db.execute(
            "SELECT * FROM runtime_receipts WHERE session_token = ? ORDER BY storage_revision",
            (token,),
        ).fetchall()
        revisions = tuple(int(row["storage_revision"]) for row in receipt_rows)
        if revisions != tuple(range(1, int(session_row["storage_revision"]) + 1)):
            raise RuntimePersistenceError(
                PersistenceErrorCode.SEQUENCE_GAP,
                "runtime storage revisions are not contiguous",
            )
        prior_result: RuntimeTransitionResult | None = None
        for row in receipt_rows:
            revision = int(row["storage_revision"])
            receipt_plaintext = self._open(
                key,
                token,
                revision,
                "persistence-receipt",
                str(row["receipt_sha256"]),
                bytes(row["receipt_nonce"]),
                bytes(row["receipt_ciphertext"]),
            )
            receipt = _codec_load(receipt_plaintext)
            if (
                type(receipt) is not PersistenceReceipt
                or receipt.storage_revision != revision
                or receipt.transition_id != row["transition_id"]
                or type(receipt.command).__name__ != row["command_type"]
                or not hmac.compare_digest(
                    _sha256(_codec_dump(receipt.command)), str(row["command_sha256"])
                )
                or not hmac.compare_digest(
                    _sha256(_codec_dump(receipt.previous_snapshot)),
                    str(row["previous_snapshot_sha256"]),
                )
                or not hmac.compare_digest(
                    _sha256(_codec_dump(receipt.result)), str(row["result_sha256"])
                )
                or transition(receipt.previous_snapshot, receipt.command) != receipt.result
                or receipt.snapshot != receipt.result.snapshot
                or (prior_result is not None and receipt.previous_snapshot != prior_result.snapshot)
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime transition receipt chain is invalid",
                )
            transition_event = self._db.execute(
                "SELECT 1 FROM runtime_events WHERE session_token = ? "
                "AND storage_revision = ? AND event_ordinal = 0 AND event_id = ? "
                "AND event_type = 'RuntimeTransitionResult'",
                (token, revision, receipt.transition_id),
            ).fetchone()
            if transition_event is None:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime transition manifest is incomplete",
                )
            for outbox_id in receipt.outbox_ids:
                if (
                    self._db.execute(
                        "SELECT 1 FROM runtime_outbox WHERE session_token = ? "
                        "AND event_id = ? AND causation_id = ?",
                        (token, outbox_id, receipt.transition_id),
                    ).fetchone()
                    is None
                ):
                    raise RuntimePersistenceError(
                        PersistenceErrorCode.INTEGRITY_MISMATCH,
                        "runtime receipt outbox manifest is incomplete",
                    )
            prior_result = receipt.result
        if prior_result is None or prior_result.snapshot != _codec_load(
            self._open(
                key,
                token,
                int(session_row["storage_revision"]),
                "runtime-snapshot",
                str(session_row["snapshot_sha256"]),
                bytes(session_row["snapshot_nonce"]),
                bytes(session_row["snapshot_ciphertext"]),
            )
        ):
            raise RuntimePersistenceError(
                PersistenceErrorCode.INTEGRITY_MISMATCH,
                "runtime receipt head does not match session snapshot",
            )
        for row in self._db.execute(
            "SELECT * FROM runtime_inbox WHERE session_token = ?", (token,)
        ):
            receipt = self._db.execute(
                "SELECT 1 FROM runtime_receipts WHERE transition_id = ? AND session_token = ?",
                (row["transition_id"], token),
            ).fetchone()
            event = self._db.execute(
                "SELECT 1 FROM runtime_events WHERE event_id = ? "
                "AND event_type = ? AND session_token = ? AND content_sha256 = ?",
                (row["event_id"], row["event_type"], token, row["content_sha256"]),
            ).fetchone()
            expected_metadata = self._metadata_sha256(
                event_id=row["event_id"],
                event_type=row["event_type"],
                transition_id=row["transition_id"],
                content_sha256=row["content_sha256"],
                exists=True,
            )
            if (
                receipt is None
                or event is None
                or not hmac.compare_digest(str(row["metadata_sha256"]), expected_metadata)
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime inbox binding is incomplete",
                )
        for row in self._db.execute(
            "SELECT * FROM runtime_outbox WHERE session_token = ?", (token,)
        ):
            plaintext = self._open(
                key,
                token,
                int(row["storage_revision"]),
                str(row["event_type"]),
                str(row["content_sha256"]),
                bytes(row["payload_nonce"]),
                bytes(row["payload_ciphertext"]),
                str(row["metadata_sha256"]),
            )
            payload = _codec_load(plaintext)
            if type(payload) not in {RealtimeCoachingEventV2, ReplanIntent, PlanOutcome}:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime outbox payload has an invalid authority type",
                )
            if str(row["causation_id"]) not in known_ids:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime outbox causation lineage is incomplete",
                )
            manifest = _codec_load(
                self._open(
                    key,
                    token,
                    int(row["storage_revision"]),
                    "outbox-manifest",
                    str(row["manifest_sha256"]),
                    bytes(row["manifest_nonce"]),
                    bytes(row["manifest_ciphertext"]),
                    str(row["metadata_sha256"]),
                )
            )
            expected_manifest = {
                "event_id": str(row["event_id"]),
                "event_type": str(row["event_type"]),
                "schema_version": str(row["schema_version"]),
                "storage_revision": int(row["storage_revision"]),
                "sequence": int(row["sequence"]),
                "content_sha256": str(row["content_sha256"]),
                "metadata_sha256": str(row["metadata_sha256"]),
                "causation_id": str(row["causation_id"]),
                "exists": True,
            }
            if manifest != expected_manifest:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime outbox manifest is invalid",
                )
        ack_rows = self._db.execute(
            "SELECT * FROM runtime_outbox_acks WHERE session_token = ? ORDER BY acked_at",
            (token,),
        ).fetchall()
        for row in ack_rows:
            outbox = self._db.execute(
                "SELECT * FROM runtime_outbox WHERE event_id = ? AND event_type = ?",
                (row["event_id"], row["event_type"]),
            ).fetchone()
            if outbox is None or not hmac.compare_digest(
                str(outbox["content_sha256"]), str(row["content_sha256"])
            ):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime ACK has no exact outbox authority",
                )
            ack = _codec_load(
                self._open(
                    key,
                    token,
                    int(outbox["storage_revision"]),
                    "outbox-ack",
                    str(row["ack_sha256"]),
                    bytes(row["ack_nonce"]),
                    bytes(row["ack_ciphertext"]),
                    str(outbox["metadata_sha256"]),
                )
            )
            if ack != {
                "event_id": str(row["event_id"]),
                "event_type": str(row["event_type"]),
                "acked_at": str(row["acked_at"]),
                "session_token": token,
                "content_sha256": str(row["content_sha256"]),
                "exists": True,
                "acknowledged": True,
            }:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "runtime ACK manifest is invalid",
                )
        maximum_ack = max((str(row["acked_at"]) for row in ack_rows), default=None)
        stored_last_ack = (
            None if session_row["last_ack_at"] is None else str(session_row["last_ack_at"])
        )
        if maximum_ack != stored_last_ack:
            raise RuntimePersistenceError(
                PersistenceErrorCode.INTEGRITY_MISMATCH,
                "runtime ACK head does not match append-only ledger",
            )

    @staticmethod
    def _safe_recovery(
        session_id: UUID,
        status: RecoveryStatus,
        error: PersistenceErrorCode,
    ) -> RecoveryFailureResult:
        return RecoveryFailureResult(status, session_id, error)

    def pending_outbox(self, session_id: UUID) -> tuple[PendingOutboxItem, ...]:
        token = self._session_token(session_id)
        session_row = self._db.execute(
            "SELECT * FROM runtime_sessions WHERE session_token = ?", (token,)
        ).fetchone()
        if session_row is None:
            return ()
        key = self._unwrap_key(session_row)
        results: list[PendingOutboxItem] = []
        for row in self._db.execute(
            "SELECT * FROM runtime_outbox o WHERE session_token = ? "
            "AND NOT EXISTS (SELECT 1 FROM runtime_outbox_acks a "
            "WHERE a.event_id = o.event_id AND a.event_type = o.event_type) "
            "ORDER BY sequence, event_type, event_id",
            (token,),
        ):
            plaintext = self._open(
                key,
                token,
                int(row["storage_revision"]),
                str(row["event_type"]),
                str(row["content_sha256"]),
                bytes(row["payload_nonce"]),
                bytes(row["payload_ciphertext"]),
                str(row["metadata_sha256"]),
            )
            payload = _codec_load(plaintext)
            if type(payload) not in {RealtimeCoachingEventV2, ReplanIntent, PlanOutcome}:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.INTEGRITY_MISMATCH,
                    "outbox record is not an authority contract",
                )
            results.append(
                PendingOutboxItem(
                    event_id=str(row["event_id"]),
                    event_type=str(row["event_type"]),
                    schema_version=str(row["schema_version"]),
                    sequence=int(row["sequence"]),
                    content_sha256=str(row["content_sha256"]),
                    causation_id=str(row["causation_id"]),
                    payload=payload,
                    plaintext=plaintext,
                )
            )
        return tuple(results)

    def ack_outbox(
        self,
        event_id: str,
        event_type: str,
        content_sha256: str,
        *,
        acked_at: datetime,
    ) -> None:
        timestamp = _iso(acked_at)
        try:
            self._db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy during outbox ACK",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED,
                "outbox ACK transaction failed",
            ) from exc
        try:
            row = self._db.execute(
                "SELECT * FROM runtime_outbox WHERE event_id = ? AND event_type = ?",
                (event_id, event_type),
            ).fetchone()
            if row is None or not hmac.compare_digest(str(row["content_sha256"]), content_sha256):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.IDENTITY_CONFLICT,
                    "outbox ACK does not bind the stored authority event",
                )
            session_row = self._db.execute(
                "SELECT * FROM runtime_sessions WHERE session_token = ?",
                (row["session_token"],),
            ).fetchone()
            if session_row is None:
                raise RuntimePersistenceError(
                    PersistenceErrorCode.CORRUPT_DATABASE,
                    "outbox ACK has no owning session",
                )
            data_key = self._unwrap_key(session_row)
            ack_manifest = {
                "event_id": event_id,
                "event_type": event_type,
                "acked_at": timestamp,
                "session_token": str(row["session_token"]),
                "content_sha256": content_sha256,
                "exists": True,
                "acknowledged": True,
            }
            ack_sha, ack_nonce, ack_ciphertext = self._seal(
                data_key,
                str(row["session_token"]),
                int(row["storage_revision"]),
                "outbox-ack",
                _codec_dump(ack_manifest),
                str(row["metadata_sha256"]),
            )
            self._db.execute(
                "INSERT INTO runtime_outbox_acks("
                "event_id, event_type, acked_at, session_token, content_sha256, "
                "ack_sha256, ack_nonce, ack_ciphertext) VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_id, event_type, acked_at) DO NOTHING",
                (
                    event_id,
                    event_type,
                    timestamp,
                    row["session_token"],
                    content_sha256,
                    ack_sha,
                    ack_nonce,
                    ack_ciphertext,
                ),
            )
            self._db.execute(
                "UPDATE runtime_sessions SET last_ack_at = CASE "
                "WHEN last_ack_at IS NULL OR last_ack_at < ? THEN ? ELSE last_ack_at END "
                "WHERE session_token = ?",
                (timestamp, timestamp, row["session_token"]),
            )
            self._db.commit()
        except RuntimePersistenceError:
            self._db.rollback()
            raise
        except sqlite3.OperationalError as exc:
            self._db.rollback()
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy during outbox ACK",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED, "outbox ACK transaction failed"
            ) from exc
        except sqlite3.DatabaseError as exc:
            self._db.rollback()
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED, "outbox ACK transaction failed"
            ) from exc

    def run_retention(self, *, now: datetime) -> RetentionReport:
        current = now.astimezone(UTC)
        deleted: list[UUID] = []
        quarantined: list[UUID] = []
        rows = self._db.execute("SELECT * FROM runtime_sessions ORDER BY session_token").fetchall()
        for row in rows:
            token = str(row["session_token"])
            recovered_id = self._recover_identity(row)
            unacked = self._db.execute(
                "SELECT MIN(o.created_at) FROM runtime_outbox o "
                "WHERE o.session_token = ? AND NOT EXISTS ("
                "SELECT 1 FROM runtime_outbox_acks a WHERE a.event_id = o.event_id "
                "AND a.event_type = o.event_type)",
                (token,),
            ).fetchone()[0]
            if unacked is not None and current - _parse_time(str(unacked)) > _ACK_QUARANTINE_AFTER:
                if row["quarantined_at"] is None:
                    self._db.execute(
                        "UPDATE runtime_sessions SET quarantined_at = ? WHERE session_token = ?",
                        (_iso(current), token),
                    )
                if recovered_id is not None:
                    quarantined.append(recovered_id)
                continue
            if row["quarantined_at"] is not None:
                self._db.execute(
                    "UPDATE runtime_sessions SET quarantined_at = NULL WHERE session_token = ?",
                    (token,),
                )
            if row["completed_at"] is not None and unacked is None:
                retention_start = _parse_time(str(row["completed_at"]))
                if row["last_ack_at"] is not None:
                    retention_start = max(
                        retention_start,
                        _parse_time(str(row["last_ack_at"])),
                    )
                if current - retention_start >= _DELETE_AFTER_ACK:
                    self._erase_token(token, current, "ACK_TTL")
                    if recovered_id is not None:
                        deleted.append(recovered_id)
        return RetentionReport(tuple(sorted(deleted)), tuple(sorted(quarantined)))

    def _recover_identity(self, row: sqlite3.Row) -> UUID | None:
        try:
            key = self._unwrap_key(row)
            plaintext = self._open(
                key,
                str(row["session_token"]),
                int(row["storage_revision"]),
                "runtime-snapshot",
                str(row["snapshot_sha256"]),
                bytes(row["snapshot_nonce"]),
                bytes(row["snapshot_ciphertext"]),
            )
            snapshot = _codec_load(plaintext)
            return snapshot.session_id if type(snapshot) is RuntimeSnapshot else None
        except (RuntimePersistenceError, ValueError, TypeError):
            return None

    def erase_session(self, session_id: UUID, *, erased_at: datetime) -> None:
        self._erase_token(self._session_token(session_id), erased_at, "USER_ERASURE")

    def _erase_token(self, token: str, erased_at: datetime, reason: str) -> None:
        try:
            self._db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy during session erasure",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED,
                "session crypto-erasure failed",
            ) from exc
        try:
            existing = self._db.execute(
                "SELECT erased_at, reason_code FROM deletion_receipts WHERE session_token = ?",
                (token,),
            ).fetchone()
            if existing is not None:
                self._db.rollback()
                return
            self._db.execute("DELETE FROM runtime_sessions WHERE session_token = ?", (token,))
            self._db.execute(
                "INSERT INTO deletion_receipts(session_token, erased_at, reason_code) "
                "VALUES(?, ?, ?)",
                (token, _iso(erased_at), reason),
            )
            self._db.commit()
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.execute("VACUUM")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:
            self._db.rollback()
            if self._is_locked(exc):
                raise RuntimePersistenceError(
                    PersistenceErrorCode.DB_LOCKED,
                    "SQLite writer is busy during session erasure",
                    retryable=True,
                ) from exc
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED, "session crypto-erasure failed"
            ) from exc
        except sqlite3.DatabaseError as exc:
            self._db.rollback()
            raise RuntimePersistenceError(
                PersistenceErrorCode.TRANSACTION_FAILED, "session crypto-erasure failed"
            ) from exc

    def catalog_counts(self) -> dict[str, int]:
        return {
            "sessions": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_sessions").fetchone()[0]
            ),
            "runtime_events": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_events").fetchone()[0]
            ),
            "runtime_inbox": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_inbox").fetchone()[0]
            ),
            "runtime_outbox": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_outbox").fetchone()[0]
            ),
            "runtime_receipts": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_receipts").fetchone()[0]
            ),
            "runtime_outbox_acks": int(
                self._db.execute("SELECT COUNT(*) FROM runtime_outbox_acks").fetchone()[0]
            ),
        }
