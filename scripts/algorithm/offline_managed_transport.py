"""One-scene, signed in-memory remote for offline managed development runs.

This is a protocol peer only. Product commands are applied by the real local
worker and its SQLite ledger; this object accepts only their actual receipts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
from collections import deque
from dataclasses import asdict, dataclass
from uuid import uuid4

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
)
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionCloseState,
    validate_managed_uuid,
)
from billiards_ai.edge.route_assistance.sync import HttpResult, SyncRequestToSign

BASE_URL = "https://offline-development.invalid/sync-api"
ENVIRONMENT_ID = "offline-development"
DEVICE_ID = "offline-device"
TABLE_ID = "offline-table"
CREDENTIAL_TARGET = "billiardsAI/development/offline-device"
_MAX_BODY = 65_536
_MAX_CALLS = 16_384
_MAX_COMMANDS = 4
_MAX_REVISION = 9_007_199_254_740_991
_HEADERS = frozenset(
    {"content-type", "x-device-id", "x-timestamp", "x-nonce", "x-body-sha256", "x-signature"}
)
_REJECTION_CODES = frozenset(
    {
        "SESSION_ENDED",
        "STALE_REVISION",
        "TURN_CONFIRMATION_MISMATCH",
        "COMMAND_FORBIDDEN",
        "COMMAND_INVALID",
        "SHOT_UNSETTLED",
    }
)


@dataclass(frozen=True, slots=True)
class _Credentials:
    _secret: bytes

    def device_id(self) -> str:
        return DEVICE_ID

    def secret(self) -> bytes:
        return self._secret


class _DenyTransport:
    def __init__(self, owner: OfflineManagedRemote) -> None:
        self._owner = owner

    def post_json(self, path, headers, body):
        self._owner._deny()

    def put_json(self, path, headers, body):
        self._owner._deny()

    def put_file(self, url, path, content_type, headers):
        self._owner._deny()

    def put_bytes(self, url, payload, content_type, headers):
        self._owner._deny()


class OfflineManagedRemote:
    """A fixed-scope signed HTTP protocol with one managed birth and four commands."""

    def __init__(self, *, observer_mode="single-shot-v1") -> None:
        if observer_mode not in {"single-shot-v1", "fullgame-v1"}:
            raise ValueError("unknown offline observer mode")
        self._fullgame = observer_mode == "fullgame-v1"
        self._lock = threading.RLock()
        self.credentials = _Credentials(secrets.token_bytes(32))
        self.failure: str | None = None
        self._events: deque[str] = deque(maxlen=256)
        self._calls = 0
        self._open_request_id: str | None = None
        self._binding: ManagedSessionBinding | None = None
        self._closed_target: int | None = None
        self._commands: list[FieldCommandEnvelope] = []
        self._claimed: str | None = None
        self._applications: dict[str, FieldCommandApplication] = {}
        self._completed_prefix = 0

    def _fail(self, code: str, *, error: type[Exception] = OSError) -> None:
        # Codes are fixed internal literals; no headers, tokens or input bodies.
        if self.failure is None:
            self.failure = code
        self._events.append(code)
        raise error(code)

    def _count_call(self) -> None:
        if self._calls >= _MAX_CALLS:
            self._fail("CALL_LIMIT")
        self._calls += 1

    def _deny(self) -> None:
        with self._lock:
            self._count_call()
            self._fail("DENIED_TRANSPORT")

    def credentials_factory(self, *, device_id, target_name):
        with self._lock:
            if device_id != DEVICE_ID or target_name != CREDENTIAL_TARGET:
                self._fail("CREDENTIAL_SCOPE", error=ValueError)
            return self.credentials

    def transport_factory(self, base_url, *, allow_redirects=False, timeout_seconds=None):
        with self._lock:
            if (
                base_url != BASE_URL
                or allow_redirects is not False
                or (
                    timeout_seconds is not None
                    and (
                        type(timeout_seconds) not in (int, float)
                        or not math.isfinite(timeout_seconds)
                        or timeout_seconds <= 0
                    )
                )
            ):
                self._fail("TRANSPORT_SCOPE", error=ValueError)
            return self

    def deny_transport_factory(self, base_url):
        # This factory is deliberately independent of the URL. No outbound client exists.
        return _DenyTransport(self)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "synthetic_remote": True,
                "environment_id": ENVIRONMENT_ID,
                "device_id": DEVICE_ID,
                "table_id": TABLE_ID,
                "birth_count": int(self._binding is not None),
                "issued_count": len(self._commands),
                "completed_prefix": self._completed_prefix,
                "closed_target": self._closed_target,
                "call_count": self._calls,
                "failure": self.failure,
                "events": tuple(self._events),
            }

    def _revision(self, value: object) -> bool:
        return type(value) is int and 0 <= value <= _MAX_REVISION

    def issue(
        self,
        *,
        session_id: str,
        actor_user_id: str,
        command_type: str,
        payload: dict,
        expected_revision: int,
    ) -> str:
        with self._lock:
            if self._binding is None or session_id != self._binding.session_id:
                self._fail("ISSUE_SESSION", error=ValueError)
            if self._closed_target is not None:
                self._fail("ISSUE_AFTER_CLOSE", error=ValueError)
            if len(self._commands) >= (3 if self._fullgame else _MAX_COMMANDS):
                self._fail("ISSUE_CAPACITY", error=ValueError)
            if self._completed_prefix != len(self._commands):
                self._fail("ISSUE_PENDING", error=ValueError)
            if not self._revision(expected_revision):
                self._fail("ISSUE_REVISION", error=ValueError)
            if (
                not isinstance(actor_user_id, str)
                or actor_user_id not in {"offline-red", "offline-blue"}
                or type(payload) is not dict
            ):
                self._fail("ISSUE_SCOPE", error=ValueError)
            sequence = len(self._commands)
            if (
                (
                    sequence < 2
                    and (
                        command_type != "claim_role"
                        or any(item.actor_user_id == actor_user_id for item in self._commands)
                        or (
                            self._fullgame
                            and actor_user_id != ("offline-red", "offline-blue")[sequence]
                        )
                    )
                )
                or (
                    sequence == 2
                    and (
                        command_type != "set_breaker"
                        or (self._fullgame and actor_user_id != "offline-red")
                    )
                )
                or (sequence == 3 and command_type != "end_session")
            ):
                self._fail("ISSUE_ORDER", error=ValueError)
            if command_type == "claim_role":
                expected_role = "red" if actor_user_id == "offline-red" else "blue"
                allowed = payload == {"role": expected_role}
            elif command_type == "set_breaker":
                allowed = (
                    payload
                    == {
                        "role": "red",
                        "mode": "assistance",
                        "format": "single",
                        "race_target": None,
                        "ranking_intent": False,
                        "settings_revision": 1,
                    }
                    and type(payload.get("ranking_intent")) is bool
                    and type(payload.get("settings_revision")) is int
                )
            elif command_type == "end_session":
                allowed = payload == {}
            else:
                allowed = False
            if not allowed:
                self._fail("ISSUE_COMMAND", error=ValueError)
            command_id = str(uuid4())
            envelope = FieldCommandEnvelope(
                command_id=command_id,
                session_id=session_id,
                actor_user_id=actor_user_id,
                command_type=command_type,
                payload=dict(payload),
                expected_revision=expected_revision,
                server_sequence=len(self._commands) + 1,
            )
            self._commands.append(envelope)
            self._events.append("ISSUED")
            return command_id

    def completion(self, command_id: str) -> FieldCommandApplication | None:
        with self._lock:
            if not any(item.command_id == command_id for item in self._commands):
                self._fail("UNKNOWN_COMPLETION")
            return self._applications.get(command_id)

    def _verify(self, method: str, path: object, headers: object, body: object) -> object:
        if not isinstance(path, str) or not path.startswith("/") or type(body) is not bytes:
            self._fail("REQUEST_SHAPE")
        if len(body) > _MAX_BODY:
            self._fail("BODY_LIMIT")
        if (
            not isinstance(headers, dict)
            or set(headers) != _HEADERS
            or any(not isinstance(value, str) for value in headers.values())
        ):
            self._fail("HEADER_SHAPE")
        if headers["content-type"] != "application/json" or headers["x-device-id"] != DEVICE_ID:
            self._fail("DEVICE_SCOPE")
        digest = hashlib.sha256(body).hexdigest()
        if not hmac.compare_digest(headers["x-body-sha256"], digest):
            self._fail("BODY_DIGEST")
        try:
            request = SyncRequestToSign(
                method, path, headers["x-timestamp"], headers["x-nonce"], digest
            )
        except ValueError:
            self._fail("SIGNATURE_SHAPE")
        expected = hmac.new(
            self.credentials.secret(), request.canonical, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(headers["x-signature"], expected):
            self._fail("SIGNATURE_INVALID")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        try:
            parsed = json.loads(body, object_pairs_hook=unique)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._fail("BODY_JSON")
        if not isinstance(parsed, dict):
            self._fail("BODY_OBJECT")
        return parsed

    def _response(self, data: object) -> HttpResult:
        return HttpResult(
            200, json.dumps({"ok": True, "data": data}, separators=(",", ":")).encode()
        )

    def post_json(self, path, headers, body) -> HttpResult:
        with self._lock:
            self._count_call()
            payload = self._verify("POST", path, headers, body)
            if path == "/v1/runtime-attestation":
                if payload != {}:
                    self._fail("ATTESTATION_BODY")
                data = {
                    "schema_version": "1.0",
                    "runtime_environment": "development",
                    "cloudbase_environment_id": ENVIRONMENT_ID,
                }
            elif path == "/v2/field-sessions/open":
                if (
                    set(payload) != {"schema_version", "open_request_id"}
                    or payload["schema_version"] != "1.0"
                ):
                    self._fail("OPEN_BODY")
                try:
                    validate_managed_uuid(payload["open_request_id"], name="open request ID")
                except ValueError:
                    self._fail("OPEN_REQUEST_ID")
                request_id = payload["open_request_id"]
                if self._open_request_id is None:
                    birth_id = str(uuid4())
                    self._binding = ManagedSessionBinding(
                        "1.0",
                        f"managed-{birth_id}",
                        birth_id,
                        "managed_session_birth_v1",
                        TABLE_ID,
                        DEVICE_ID,
                        "open",
                    )
                    self._open_request_id = request_id
                elif request_id != self._open_request_id:
                    self._fail("SECOND_BIRTH")
                data = asdict(self._binding)
            elif (
                self._binding is not None
                and path == f"/v2/field-sessions/{self._binding.session_id}/close"
            ):
                if payload != {"schema_version": "1.0"}:
                    self._fail("CLOSE_BODY")
                if self._closed_target is None:
                    self._closed_target = len(self._commands)
                close = ManagedSessionCloseState(
                    **{
                        key: value
                        for key, value in asdict(self._binding).items()
                        if key != "status"
                    },
                    coverage="sealed"
                    if self._completed_prefix == self._closed_target
                    else "closing",
                    closed_target=self._closed_target,
                    contiguous_ack_sequence=self._completed_prefix,
                )
                data = asdict(close)
            elif (
                self._binding is not None
                and path == f"/v1/field-sessions/{self._binding.session_id}/join-requests/query"
            ):
                if payload != {}:
                    self._fail("JOIN_BODY")
                data = {"requests": []}
            elif path == "/v1/field-commands/claim":
                if payload != {}:
                    self._fail("CLAIM_BODY")
                if self._completed_prefix == len(self._commands):
                    data = None
                else:
                    envelope = self._commands[self._completed_prefix]
                    self._claimed = envelope.command_id
                    data = {
                        "command_id": envelope.command_id,
                        "session_id": envelope.session_id,
                        "actor_user_id": envelope.actor_user_id,
                        "command_type": envelope.command_type,
                        "validated_payload": dict(envelope.payload),
                        "expected_revision": envelope.expected_revision,
                        "server_sequence": envelope.server_sequence,
                    }
            elif path.startswith("/v1/field-commands/") and path.endswith("/complete"):
                command_id = path.removeprefix("/v1/field-commands/").removesuffix("/complete")
                data = self._complete(command_id, payload)
            else:
                self._fail("UNKNOWN_ROUTE")
            self._events.append("POST_OK")
            return self._response(data)

    def _complete(self, command_id: str, payload: dict) -> dict:
        if set(payload) != {"status", "public_result_code", "resulting_revision"}:
            self._fail("COMPLETION_BODY")
        outcome, code, revision = (
            payload["status"],
            payload["public_result_code"],
            payload["resulting_revision"],
        )
        if not self._revision(revision) or not (
            (outcome == "applied" and code == "APPLIED")
            or (outcome == "rejected" and isinstance(code, str) and code in _REJECTION_CODES)
        ):
            self._fail("COMPLETION_RESULT")
        if command_id not in {item.command_id for item in self._commands}:
            self._fail("COMPLETION_UNKNOWN")
        application = FieldCommandApplication(command_id, outcome, code, revision)
        prior = self._applications.get(command_id)
        if prior is not None:
            if prior != application:
                self._fail("COMPLETION_CONFLICT")
            return {
                "command_id": command_id,
                "status": outcome,
                "public_result_code": code,
                "resulting_revision": revision,
            }
        expected = self._commands[self._completed_prefix]
        if command_id != expected.command_id or self._claimed != command_id:
            self._fail("COMPLETION_UNCLAIMED")
        self._applications[command_id] = application
        self._completed_prefix += 1
        self._claimed = None
        self._events.append("COMPLETED")
        return {
            "command_id": command_id,
            "status": outcome,
            "public_result_code": code,
            "resulting_revision": revision,
        }

    def put_json(self, path, headers, body):
        self._deny()

    def put_file(self, url, path, content_type, headers):
        self._deny()

    def put_bytes(self, url, payload, content_type, headers):
        self._deny()
