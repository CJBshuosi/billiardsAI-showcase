"""Signed, retry-safe field-session synchronization ports and value objects."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID

from billiards_ai.edge.route_assistance.postgame import (
    MAX_SAVED_CLIP_BYTES,
    SavedMediaOutboxItem,
)
from billiards_ai.edge.route_assistance.records import LiveShotFactOutboxItem

if TYPE_CHECKING:
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.edge.route_assistance.owner_device_delivery import OwnerDeletionReceiptItem
    from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask
    from billiards_ai.edge.route_assistance.saved_upload_requests import (
        SavedUploadRequestError,
        SavedUploadRequestLedger,
        SavedUploadRequestRecord,
        SavedUploadRequestSession,
    )
    from billiards_ai.edge.route_assistance.upload_receipt_delivery import UploadReceiptOutboxItem

_HASH_LENGTH = 64


def _require_text(value: str, name: str) -> str:
    if not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not be empty or contain control characters")
    return value


@dataclass(frozen=True, slots=True)
class SyncRequestToSign:
    method: str
    path: str
    timestamp: str
    nonce: str
    body_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.method, "method")
        if not self.path.startswith("/"):
            raise ValueError("sync path must be absolute")
        if not self.timestamp.isdigit():
            raise ValueError("sync timestamp must contain seconds since epoch")
        _require_text(self.nonce, "nonce")
        if len(self.body_sha256) != _HASH_LENGTH or any(
            character not in "0123456789abcdef" for character in self.body_sha256
        ):
            raise ValueError("body SHA-256 must be lowercase hexadecimal")

    @property
    def canonical(self) -> bytes:
        return (
            f"{self.method.upper()}\n{self.path}\n{self.timestamp}\n"
            f"{self.nonce}\n{self.body_sha256}"
        ).encode()


@dataclass(frozen=True, slots=True)
class SignedSyncRequest:
    request: SyncRequestToSign
    signature: str


def sign_sync_request(request: SyncRequestToSign, *, secret: bytes) -> SignedSyncRequest:
    if len(secret) < 8:
        raise ValueError("device secret is too short")
    signature = hmac.new(secret, request.canonical, hashlib.sha256).hexdigest()
    return SignedSyncRequest(request=request, signature=signature)


SyncJobStatus = Literal["pending", "claimed", "partial", "complete", "blocked"]
SyncMediaStatus = Literal["pending", "uploaded", "complete"]


@dataclass(frozen=True, slots=True)
class SyncMediaItem:
    clip_id: str
    shot_id: str
    local_path: Path
    content_type: str
    content_sha256: str
    byte_size: int
    status: SyncMediaStatus = "pending"


@dataclass(frozen=True, slots=True)
class SyncOutboxJob:
    sync_job_id: str
    session_id: str
    manifest_hash: str
    manifest_json: str
    status: SyncJobStatus
    retry_count: int
    next_attempt_at_utc: str
    last_error_code: str | None
    media: tuple[SyncMediaItem, ...] = ()


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: bytes


@dataclass(frozen=True, slots=True)
class SyncAttemptResult:
    outcome: Literal["complete", "retry", "blocked"]
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class UploadReceiptSyncResult(SyncAttemptResult):
    receipt_ack: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class OwnerDevicePollResult:
    outcome: Literal["idle", "task", "retry", "blocked"]
    error_code: str | None = None
    task: OwnerDeleteDeviceTask | None = None


@dataclass(frozen=True, slots=True)
class OwnerDeviceCompletionResult:
    outcome: Literal["ack", "retry", "blocked"]
    error_code: str | None = None
    ack: dict[str, object] | None = None


def _owner_wire_time(value: object) -> bool:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value) is None
    ):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _owner_unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid owner response")
        result[key] = value
    return result


class DeviceCredentialProvider(Protocol):
    def device_id(self) -> str: ...

    def secret(self) -> bytes: ...


def _normalize_windows_credential_blob(blob: bytes) -> bytes:
    """Return portable HMAC bytes from generic or cmdkey credentials."""

    if (
        len(blob) >= 8
        and len(blob) % 2 == 0
        and all(blob[index] == 0 for index in range(1, len(blob), 2))
    ):
        try:
            decoded = blob.decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise OSError("device credential is invalid") from exc
        _require_text(decoded, "device credential")
        blob = decoded.encode()
    if len(blob) < 8:
        raise OSError("device credential is invalid")
    return blob


class WindowsCredentialManagerProvider:
    """Read one generic credential from Windows Credential Manager on demand."""

    def __init__(self, *, device_id: str, target_name: str) -> None:
        self._device_id = _require_text(device_id, "device ID")
        self._target_name = _require_text(target_name, "credential target")

    def device_id(self) -> str:
        return self._device_id

    def secret(self) -> bytes:
        if sys.platform != "win32":
            raise OSError("Windows Credential Manager is unavailable")
        import ctypes
        from ctypes import wintypes

        class Credential(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        credential_pointer = ctypes.POINTER(Credential)()
        advapi = ctypes.WinDLL("Advapi32.dll")
        advapi.CredReadW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(Credential)),
        ]
        advapi.CredReadW.restype = wintypes.BOOL
        advapi.CredFree.argtypes = [ctypes.c_void_p]
        if not advapi.CredReadW(self._target_name, 1, 0, ctypes.byref(credential_pointer)):
            raise OSError("device credential is unavailable")
        try:
            credential = credential_pointer.contents
            secret = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return _normalize_windows_credential_blob(secret)
        finally:
            advapi.CredFree(credential_pointer)


class SyncHttpTransport(Protocol):
    def post_json(self, path: str, headers: Mapping[str, str], body: bytes) -> HttpResult: ...

    def put_json(self, path: str, headers: Mapping[str, str], body: bytes) -> HttpResult: ...

    def put_file(
        self,
        url: str,
        path: Path,
        content_type: str,
        headers: Mapping[str, str],
    ) -> HttpResult: ...

    def put_bytes(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str],
    ) -> HttpResult: ...


def classify_http_status(status_code: int) -> Literal["success", "retry", "blocked"]:
    if 200 <= status_code < 300:
        return "success"
    if status_code in {408, 429} or 500 <= status_code < 600:
        return "retry"
    return "blocked"


class FieldSyncClient:
    """Run one outbox attempt without persisting credentials or upload URLs."""

    def __init__(
        self,
        *,
        credentials: DeviceCredentialProvider,
        transport: SyncHttpTransport,
        now_seconds: Callable[[], int] = lambda: int(time.time()),
        nonce_factory: Callable[[], str] = lambda: f"nonce-{secrets.token_hex(16)}",
        before_mutation: Callable[[], object] = lambda: None,
        owner_cancelled: Callable[[], bool] = lambda: False,
        owner_scope: ManagedSessionScope | None = None,
    ) -> None:
        self._credentials = credentials
        self._transport = transport
        self._now_seconds = now_seconds
        self._nonce_factory = nonce_factory
        self._before_mutation = before_mutation
        self._owner_cancelled = owner_cancelled
        self._owner_scope = owner_scope

    def _headers(self, path: str, body: bytes, *, method: str = "POST") -> dict[str, str]:
        timestamp = str(self._now_seconds())
        nonce = self._nonce_factory()
        body_hash = hashlib.sha256(body).hexdigest()
        signed = sign_sync_request(
            SyncRequestToSign(
                method=method,
                path=path,
                timestamp=timestamp,
                nonce=nonce,
                body_sha256=body_hash,
            ),
            secret=self._credentials.secret(),
        )
        return {
            "content-type": "application/json",
            "x-device-id": self._credentials.device_id(),
            "x-timestamp": timestamp,
            "x-nonce": nonce,
            "x-body-sha256": body_hash,
            "x-signature": signed.signature,
        }

    def _owner_post(self, path, body):
        # Both HTTP calls are separately bounded by the dedicated transport.
        if self._owner_cancelled():
            return None, ("retry", "transport_unavailable")
        try:
            self._before_mutation()
        except TimeoutError:
            return None, ("retry", "transport_timeout")
        except (OSError, ValueError, RuntimeError):
            return None, ("retry", "attestation_failed")
        if self._owner_cancelled():
            return None, ("retry", "transport_unavailable")
        try:
            headers = self._headers(path, body)
        except (OSError, ValueError, RuntimeError):
            return None, ("retry", "unauthorized")
        if self._owner_cancelled():
            return None, ("retry", "transport_unavailable")
        try:
            result = self._transport.post_json(path, headers, body)
        except TimeoutError:
            return None, ("retry", "transport_timeout")
        except OSError:
            return None, ("retry", "transport_unavailable")
        except ValueError:
            return None, ("blocked", "invalid_response")
        if self._owner_cancelled():
            return None, ("retry", "transport_unavailable")
        if result.status_code in {401, 403}:
            return None, ("retry", "unauthorized")
        if classify_http_status(result.status_code) == "retry":
            return None, ("retry", "transport_unavailable")
        if result.status_code == 409:
            return None, ("blocked", "remote_conflict")
        try:
            if not 200 <= result.status_code < 300 or len(result.body) > 16384:
                raise ValueError("invalid owner response")
            payload = json.loads(
                result.body.decode("utf-8"), object_pairs_hook=_owner_unique_object
            )
            if (
                not isinstance(payload, dict)
                or set(payload) != {"ok", "data"}
                or payload["ok"] is not True
            ):
                raise ValueError("invalid owner response")
            return payload["data"], None
        except (TypeError, ValueError, RecursionError):
            return None, ("blocked", "invalid_response")

    def poll_owner_delete_once(self, scope: ManagedSessionScope) -> OwnerDevicePollResult:
        from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
        from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask

        if not isinstance(scope, ManagedSessionScope):
            return OwnerDevicePollResult("blocked", "configuration_invalid")
        if self._owner_scope is not None and self._owner_scope != scope:
            return OwnerDevicePollResult("blocked", "scope_conflict")
        try:
            if self._credentials.device_id() != scope.device_id:
                return OwnerDevicePollResult("blocked", "scope_conflict")
        except (OSError, ValueError):
            return OwnerDevicePollResult("retry", "unauthorized")
        data, error = self._owner_post(
            "/v1/owner-media-deletions/claim", b'{"contract":"owner_delete_device_poll_v1"}'
        )
        if error:
            return OwnerDevicePollResult(*error)
        if data is None:
            return OwnerDevicePollResult("idle")
        try:
            if (
                not isinstance(data, dict)
                or set(data)
                != {"contract", "task", "task_sha256", "delivery_token", "lease_until_utc"}
                or data["contract"] != "owner_delete_device_delivery_v1"
                or not _owner_wire_time(data["lease_until_utc"])
            ):
                raise ValueError("invalid delivery")
            token = data["delivery_token"]
            if not isinstance(token, str) or str(UUID(token)) != token or UUID(token).version != 4:
                raise ValueError("invalid delivery")
            task = OwnerDeleteDeviceTask.from_payload(data["task"])
            if data["task_sha256"] != task.task_sha256:
                raise ValueError("invalid delivery")
        except (KeyError, TypeError, ValueError):
            return OwnerDevicePollResult("blocked", "invalid_response")
        if (task.environment_id, task.device_id, task.table_id) != (
            scope.environment_id,
            scope.device_id,
            scope.table_id,
        ):
            return OwnerDevicePollResult("blocked", "scope_conflict")
        return OwnerDevicePollResult("task", task=task)

    def complete_owner_delete_once(
        self, item: OwnerDeletionReceiptItem
    ) -> OwnerDeviceCompletionResult:
        from billiards_ai.edge.route_assistance.owner_device_delivery import (
            _ACK_KEYS,
            OwnerDeletionReceiptItem,
            OwnerDeletionReceiptOutbox,
        )

        if not isinstance(
            item, OwnerDeletionReceiptItem
        ) or not OwnerDeletionReceiptOutbox._valid_wire(asdict(item)):
            return OwnerDeviceCompletionResult("blocked", "invalid_completion")
        scope = self._owner_scope
        if scope is None:
            return OwnerDeviceCompletionResult("blocked", "configuration_invalid")
        if (item.environment_id, item.device_id, item.table_id) != (
            scope.environment_id,
            scope.device_id,
            scope.table_id,
        ):
            return OwnerDeviceCompletionResult("blocked", "scope_conflict")
        try:
            if self._credentials.device_id() != item.device_id:
                return OwnerDeviceCompletionResult("blocked", "scope_conflict")
        except (OSError, ValueError):
            return OwnerDeviceCompletionResult("retry", "unauthorized")
        if re.fullmatch(r"owner-delete-[0-9a-f]{32}", item.deletion_job_id) is None:
            return OwnerDeviceCompletionResult("blocked", "invalid_completion")
        data, error = self._owner_post(
            f"/v1/owner-media-deletions/{item.deletion_job_id}/complete",
            item.completion_json.encode("ascii"),
        )
        if error:
            return OwnerDeviceCompletionResult(*error)
        if (
            not isinstance(data, dict)
            or set(data) != _ACK_KEYS
            or data["contract"] != "owner_delete_device_ack_v1"
            or not _owner_wire_time(data["received_at_utc"])
            or any(
                data[key] != getattr(item, key)
                for key in (
                    "deletion_job_id",
                    "clip_id",
                    "task_sha256",
                    "cloud_erase_receipt_sha256",
                    "local_receipt_sha256",
                )
            )
        ):
            return OwnerDeviceCompletionResult("blocked", "invalid_ack")
        return OwnerDeviceCompletionResult("ack", ack=data)

    def close_owner_delete_transport(self):
        self._transport.close()

    def run_live_shot_fact_once(
        self,
        item: LiveShotFactOutboxItem,
    ) -> SyncAttemptResult:
        """Publish one facts-only shot projection with its stable retry key."""
        fact = item.fact
        path = f"/v1/field-sessions/{fact.session_id}/shots/{fact.shot_id}"
        body = item.payload_json.encode()
        headers = self._headers(path, body, method="PUT")
        headers["x-idempotency-key"] = fact.idempotency_key
        try:
            self._before_mutation()
            result = self._transport.put_json(path, headers, body)
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        classification = classify_http_status(result.status_code)
        if classification != "success":
            return SyncAttemptResult(classification, self._response_error(result))
        try:
            data = json.loads(result.body)["data"]
            if (
                data["session_id"] != fact.session_id
                or data["shot_id"] != fact.shot_id
                or data["idempotency_key"] != fact.idempotency_key
            ):
                raise ValueError("live shot fact acknowledgement mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        return SyncAttemptResult("complete")

    @staticmethod
    def _response_error(result: HttpResult) -> str:
        try:
            payload = json.loads(result.body)
            return str(payload["error"]["code"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return f"HTTP_{result.status_code}"

    def _post(self, path: str, body: bytes) -> HttpResult:
        self._before_mutation()
        return self._transport.post_json(path, self._headers(path, body), body)

    def run_upload_receipt_once(self, item: UploadReceiptOutboxItem) -> UploadReceiptSyncResult:
        """Send original terminal metadata without acquiring any media authority."""
        from billiards_ai.edge.route_assistance.upload_receipt_delivery import (
            UploadReceiptOutboxItem,
            validate_upload_receipt_ack,
        )

        if not isinstance(item, UploadReceiptOutboxItem):
            return UploadReceiptSyncResult("blocked", "receipt_invalid")
        try:
            if self._credentials.device_id() != item.scope.device_id:
                return UploadReceiptSyncResult("retry", "receipt_auth_failed")
            # Attestation failures keep the original body pending under actual auth.
            self._before_mutation()
            path = f"/v1/sync/saved-clips/{item.cloud_job_id}/upload-receipt"
            body = item.receipt_json.encode("ascii")
            headers = self._headers(path, body)
        except (OSError, ValueError, RuntimeError):
            return UploadReceiptSyncResult("retry", "receipt_auth_failed")
        try:
            result = self._transport.post_json(path, headers, body)
        except (OSError, TimeoutError):
            return UploadReceiptSyncResult("retry", "receipt_unavailable")
        if result.status_code in {401, 403}:
            return UploadReceiptSyncResult("retry", "receipt_auth_failed")
        if classify_http_status(result.status_code) == "retry":
            return UploadReceiptSyncResult("retry", "receipt_unavailable")
        if not 200 <= result.status_code < 300:
            code = self._response_error(result)
            if code == "SYNC_CONFLICT":
                return UploadReceiptSyncResult("blocked", "receipt_conflict")
            if code == "SYNC_INVALID":
                return UploadReceiptSyncResult("blocked", "receipt_invalid")
            # Missing mapping/storage and unknown errors carry no permanent evidence.
            return UploadReceiptSyncResult("retry", "receipt_unavailable")
        try:
            data = json.loads(result.body)["data"]
            validate_upload_receipt_ack(item, data)
        except (KeyError, TypeError, ValueError):
            return UploadReceiptSyncResult("blocked", "receipt_invalid")
        return UploadReceiptSyncResult("complete", receipt_ack=data)

    @staticmethod
    def _saved_result(result: HttpResult) -> SyncAttemptResult | None:
        classification = classify_http_status(result.status_code)
        if classification == "success":
            return None
        error_code = FieldSyncClient._response_error(result)
        if error_code == "SYNC_PARTIAL":
            classification = "retry"
        return SyncAttemptResult(classification, error_code)

    @staticmethod
    def _validate_saved_ack(
        payload: bytes,
        item: SavedMediaOutboxItem,
        *,
        allowed_statuses: set[str],
    ) -> tuple[str, str]:
        try:
            data = json.loads(payload)["data"]
            job_id = str(data["job_id"])
            status_value = str(data["status"])
            if (
                not job_id
                or len(job_id) > 128
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                    for character in job_id
                )
                or status_value not in allowed_statuses
                or data["clip_id"] != item.clip_id
                or data["session_id"] != item.session_id
                or data["shot_id"] != item.shot_id
                or data["content_sha256"] != item.content_sha256
                or data["publication_kind"] != "saved_postgame_clip"
            ):
                raise ValueError("saved clip acknowledgement mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("saved clip acknowledgement is invalid") from exc
        return job_id, status_value

    @staticmethod
    def _read_saved_media(
        item: SavedMediaOutboxItem,
        trusted_postgame_root: Path,
    ) -> bytes:
        root = trusted_postgame_root
        path = item.local_path
        if not root.is_absolute() or not path.is_absolute():
            raise ValueError("saved media path is not absolute")
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("saved media path escapes trusted root") from exc
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("saved media path is unsafe")
        candidates = (
            root,
            *(root / Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)),
        )
        try:
            pre_open = None
            for index, candidate in enumerate(candidates):
                current = candidate.lstat()
                if stat.S_ISLNK(current.st_mode):
                    raise ValueError("saved media path contains a symlink")
                if index < len(candidates) - 1 and not stat.S_ISDIR(current.st_mode):
                    raise ValueError("saved media ancestor is not a directory")
                if index == len(candidates) - 1:
                    pre_open = current
            if pre_open is None or not stat.S_ISREG(pre_open.st_mode):
                raise ValueError("saved media is not a regular file")
            flags = os.O_RDONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                    pre_open.st_dev,
                    pre_open.st_ino,
                ):
                    raise ValueError("saved media changed before open")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    payload = stream.read(item.byte_size + 1)
                after = os.fstat(descriptor)
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                ):
                    raise ValueError("saved media changed while reading")
            finally:
                os.close(descriptor)
        except (OSError, ValueError) as exc:
            raise ValueError("saved media file validation failed") from exc
        if (
            len(payload) != item.byte_size
            or hashlib.sha256(payload).hexdigest() != item.content_sha256
        ):
            raise ValueError("saved media content validation failed")
        return payload

    @staticmethod
    def _upload_authorization(payload: bytes) -> tuple[str, dict[str, str]]:
        try:
            data = json.loads(payload)["data"]
            if data.get("method") != "PUT":
                raise ValueError("unsupported upload method")
            url = str(data["url"])
            parsed = urlsplit(url)
            if (
                len(url) > 4096
                or parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.fragment
            ):
                raise ValueError("unsafe upload URL")
            raw_headers = data["headers"]
            if (
                not isinstance(raw_headers, dict)
                or len(raw_headers) > 32
                or not all(
                    isinstance(key, str)
                    and isinstance(value, str)
                    and 0 < len(key) <= 128
                    and len(value) <= 4096
                    and all(32 <= ord(character) < 127 for character in key)
                    and all(ord(character) >= 32 and character not in "\r\n" for character in value)
                    for key, value in raw_headers.items()
                )
            ):
                raise ValueError("unsafe upload headers")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("saved clip upload authorization is invalid") from exc
        return url, dict(raw_headers)

    @staticmethod
    def _saved_upload_attempt(payload: bytes) -> dict[str, object] | None:
        """Validate correlation before PUT without changing authorization's port."""
        try:
            data = json.loads(payload)["data"]
            if "upload_attempt" not in data:
                return None
            attempt = data["upload_attempt"]
            if (
                not isinstance(attempt, dict)
                or set(attempt) != {"contract", "operation_id", "sequence"}
                or attempt["contract"] != "managed_upload_attempt_v1"
                or not isinstance(attempt["operation_id"], str)
                or re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                    attempt["operation_id"],
                )
                is None
                or type(attempt["sequence"]) is not int
                or not 1 <= attempt["sequence"] <= 64
            ):
                raise ValueError("invalid attempt")
            return dict(attempt)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("saved clip upload attempt is invalid") from exc

    def run_saved_media_once(
        self,
        item: SavedMediaOutboxItem,
        *,
        trusted_postgame_root: Path,
        upload_requests: SavedUploadRequestLedger | None = None,
    ) -> SyncAttemptResult:
        """Append one selected clip without reopening bytes after validation."""
        if upload_requests is not None:
            return self._run_managed_saved_media_once(item, trusted_postgame_root, upload_requests)
        # Explicit compatibility path: no durable single-use evidence is claimed.
        if item.byte_size > MAX_SAVED_CLIP_BYTES:
            return SyncAttemptResult("blocked", "SYNC_MEDIA_TOO_LARGE")
        try:
            media_bytes = self._read_saved_media(item, trusted_postgame_root)
        except ValueError:
            return SyncAttemptResult("blocked", "SYNC_LOCAL_MEDIA_INVALID")

        create_path = "/v1/sync/saved-clips"
        create_body = item.descriptor_json.encode()
        try:
            headers = self._headers(create_path, create_body)
            headers["x-idempotency-key"] = item.idempotency_key
            self._before_mutation()
            created = self._transport.post_json(create_path, headers, create_body)
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        failure = self._saved_result(created)
        if failure is not None:
            return failure
        try:
            cloud_job_id, create_status = self._validate_saved_ack(
                created.body,
                item,
                allowed_statuses={"pending", "uploading", "completed"},
            )
        except ValueError:
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        if create_status == "completed":
            return SyncAttemptResult("complete")

        authorization_path = f"/v1/sync/saved-clips/{cloud_job_id}/media-authorization"
        try:
            authorization = self._post(authorization_path, b"{}")
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        failure = self._saved_result(authorization)
        if failure is not None:
            return failure
        try:
            upload_url, upload_headers = self._upload_authorization(authorization.body)
            upload_attempt = self._saved_upload_attempt(authorization.body)
            self._before_mutation()
            uploaded = self._transport.put_bytes(
                upload_url,
                media_bytes,
                item.content_type,
                upload_headers,
            )
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        except ValueError:
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        failure = self._saved_result(uploaded)
        if failure is not None:
            return failure

        complete_path = f"/v1/sync/saved-clips/{cloud_job_id}/complete"
        try:
            complete_body = (
                b"{}"
                if upload_attempt is None
                else json.dumps(
                    {"upload_attempt": upload_attempt}, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            completed = self._post(complete_path, complete_body)
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        failure = self._saved_result(completed)
        if failure is not None:
            return failure
        try:
            self._validate_saved_ack(
                completed.body,
                item,
                allowed_statuses={"published", "completed"},
            )
        except ValueError:
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        return SyncAttemptResult("complete")

    @staticmethod
    def _upload_request_failure(error: SavedUploadRequestError) -> SyncAttemptResult:
        if error.retryable:
            return SyncAttemptResult("retry", "SYNC_UPLOAD_REQUEST_UNAVAILABLE")
        if error.code == "stale_claim":
            return SyncAttemptResult("retry", "SYNC_SAVED_MEDIA_STALE_CLAIM")
        if error.code in {"invalid_context", "invalid_state", "session_released"}:
            return SyncAttemptResult("blocked", "SYNC_LOCAL_MEDIA_INVALID")
        if error.code == "request_limit":
            return SyncAttemptResult("blocked", "SYNC_UPLOAD_REQUEST_LIMIT")
        return SyncAttemptResult("blocked", "SYNC_UPLOAD_REQUEST_INVALID")

    def _managed_saved_post(
        self,
        session: SavedUploadRequestSession,
        path: str,
        body: bytes,
        *,
        idempotency_key: str | None = None,
    ) -> HttpResult:
        self._before_mutation()
        session.require_current_claim()
        headers = self._headers(path, body)
        if idempotency_key is not None:
            headers["x-idempotency-key"] = idempotency_key
        return self._transport.post_json(path, headers, body)

    @classmethod
    def _managed_saved_authorization(
        cls,
        payload: bytes,
        request: dict[str, str],
    ) -> tuple[str, dict[str, object], str | None, dict[str, str] | None]:
        """Accept the exact tracked issued/reserved response, without retaining secrets."""
        try:
            data = json.loads(payload)["data"]
            status = data["request_status"]
            common = {"upload_request", "upload_attempt", "request_status"}
            if (
                not isinstance(data, dict)
                or data["upload_request"] != request
                or status not in {"issued", "reserved"}
                or set(data)
                != (
                    common
                    if status == "reserved"
                    else common
                    | {
                        "method",
                        "url",
                        "headers",
                        "expires_in_seconds",
                    }
                )
            ):
                raise ValueError("invalid tracked authorization")
            upload_attempt = cls._saved_upload_attempt(payload)
            if upload_attempt is None:
                raise ValueError("missing tracked attempt")
            if status == "reserved":
                return status, upload_attempt, None, None
            if type(data["expires_in_seconds"]) is not int or data["expires_in_seconds"] <= 0:
                raise ValueError("invalid authorization lifetime")
            url, headers = cls._upload_authorization(payload)
            return status, upload_attempt, url, headers
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved clip tracked authorization is invalid") from exc

    def _complete_managed_saved_media(
        self,
        session: SavedUploadRequestSession,
        item: SavedMediaOutboxItem,
        record: SavedUploadRequestRecord,
    ) -> SyncAttemptResult:
        complete_path = f"/v1/sync/saved-clips/{record.cloud_job_id}/complete"
        complete_body = json.dumps(
            {"upload_attempt": dict(record.upload_attempt)}, sort_keys=True, separators=(",", ":")
        ).encode()
        completed = self._managed_saved_post(session, complete_path, complete_body)
        failure = self._saved_result(completed)
        if failure is not None:
            return failure
        try:
            job_id, _ = self._validate_saved_ack(
                completed.body,
                item,
                allowed_statuses={"published", "completed"},
            )
            if job_id != record.cloud_job_id:
                raise ValueError("saved clip job mismatch")
        except ValueError:
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        return SyncAttemptResult("complete")

    def _run_managed_saved_media_once(
        self,
        item: SavedMediaOutboxItem,
        trusted_postgame_root: Path,
        upload_requests: SavedUploadRequestLedger,
    ) -> SyncAttemptResult:
        from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestError

        transport_unresolved = False
        try:
            with upload_requests.use(item) as session:
                record = session.current()
                # Durable uncertainty takes precedence over create/completed responses
                # and over file availability. Only terminal success permits complete.
                if record is not None:
                    if record.state in {"started", "unknown"}:
                        return SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
                    if record.state == "succeeded":
                        return self._complete_managed_saved_media(session, item, record)
                    if record.state == "received":
                        session.require_current_claim()
                        session.retire(record.request_id)
                        return SyncAttemptResult("retry", "SYNC_UPLOAD_REQUEST_RETIRED")
                put_once = getattr(self._transport, "put_bytes_once", None)
                if not callable(put_once):
                    return SyncAttemptResult("blocked", "SYNC_UPLOAD_TRANSPORT_UNAVAILABLE")
                if item.byte_size > MAX_SAVED_CLIP_BYTES:
                    return SyncAttemptResult("blocked", "SYNC_MEDIA_TOO_LARGE")
                try:
                    media_bytes = self._read_saved_media(item, trusted_postgame_root)
                except ValueError:
                    return SyncAttemptResult("blocked", "SYNC_LOCAL_MEDIA_INVALID")
                if record is None:
                    created = self._managed_saved_post(
                        session,
                        "/v1/sync/saved-clips",
                        item.descriptor_json.encode(),
                        idempotency_key=item.idempotency_key,
                    )
                    failure = self._saved_result(created)
                    if failure is not None:
                        return failure
                    try:
                        cloud_job_id, create_status = self._validate_saved_ack(
                            created.body,
                            item,
                            allowed_statuses={"pending", "uploading", "completed"},
                        )
                    except ValueError:
                        return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
                    if create_status == "completed":
                        return SyncAttemptResult("complete")
                    record = session.load_or_create(cloud_job_id)
                request = {"contract": "managed_upload_request_v1", "request_id": record.request_id}
                authorization_body = json.dumps(
                    {"upload_request": request}, sort_keys=True, separators=(",", ":")
                ).encode()
                authorization = self._managed_saved_post(
                    session,
                    f"/v1/sync/saved-clips/{record.cloud_job_id}/media-authorization",
                    authorization_body,
                )
                failure = self._saved_result(authorization)
                if failure is not None:
                    return failure
                try:
                    status, attempt, url, headers = self._managed_saved_authorization(
                        authorization.body,
                        request,
                    )
                except ValueError:
                    return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
                if status == "reserved":
                    session.require_current_claim()
                    session.retire(record.request_id, attempt)
                    return SyncAttemptResult("retry", "SYNC_UPLOAD_REQUEST_RETIRED")
                session.record_received(record.request_id, attempt)
                self._before_mutation()
                session.record_started(record.request_id)
                transport_unresolved = True
                # Nothing between the committed start and this single transport call
                # may recover credentials, follow redirects, or retry a write.
                upload_status = None
                try:
                    uploaded = put_once(url, media_bytes, item.content_type, headers)
                    upload_status = uploaded.status_code
                except Exception:
                    # A thrown transport error proves neither success nor no write.
                    pass
                if type(upload_status) is not int or upload_status != 200:
                    session.record_unknown(record.request_id)
                    return SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
                succeeded = session.record_succeeded(record.request_id, upload_status)
                transport_unresolved = False
                return self._complete_managed_saved_media(session, item, succeeded)
        except SavedUploadRequestError as error:
            if transport_unresolved:
                # A failed result commit leaves durable start evidence. Its next use
                # will recover as unknown, and this caller must already stop writes.
                return SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
            return self._upload_request_failure(error)
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")

    def run_once(self, job: SyncOutboxJob) -> SyncAttemptResult:
        body = job.manifest_json.encode()
        headers = self._headers("/v1/sync/jobs", body)
        headers["x-idempotency-key"] = job.sync_job_id
        headers["x-manifest-sha256"] = job.manifest_hash
        try:
            self._before_mutation()
            created = self._transport.post_json("/v1/sync/jobs", headers, body)
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        classification = classify_http_status(created.status_code)
        if classification != "success":
            return SyncAttemptResult(classification, self._response_error(created))
        try:
            created_data = json.loads(created.body)["data"]
            cloud_job_id = str(created_data["job_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
        if created_data.get("status") == "completed":
            return SyncAttemptResult("complete")

        for media in job.media:
            authorization_path = f"/v1/sync/jobs/{cloud_job_id}/media-authorizations"
            authorization_body = json.dumps(
                {"clip_id": media.clip_id}, separators=(",", ":")
            ).encode()
            try:
                authorization = self._post(authorization_path, authorization_body)
            except OSError:
                return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
            classification = classify_http_status(authorization.status_code)
            if classification != "success":
                return SyncAttemptResult(classification, self._response_error(authorization))
            try:
                upload_data = json.loads(authorization.body)["data"]
                if upload_data.get("method") != "PUT":
                    raise ValueError("unsupported upload method")
                upload_url = str(upload_data["url"])
                upload_headers = upload_data["headers"]
                if not isinstance(upload_headers, dict) or not all(
                    isinstance(key, str)
                    and isinstance(value, str)
                    and key
                    and "\r" not in key
                    and "\n" not in key
                    and "\r" not in value
                    and "\n" not in value
                    for key, value in upload_headers.items()
                ):
                    raise ValueError("invalid upload headers")
                self._before_mutation()
                uploaded = self._transport.put_file(
                    upload_url, media.local_path, media.content_type, upload_headers
                )
            except OSError:
                return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
            classification = classify_http_status(uploaded.status_code)
            if classification != "success":
                return SyncAttemptResult(classification, f"HTTP_{uploaded.status_code}")

        complete_path = f"/v1/sync/jobs/{cloud_job_id}/complete"
        try:
            completed = self._post(complete_path, b"{}")
        except OSError:
            return SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
        classification = classify_http_status(completed.status_code)
        return SyncAttemptResult(
            "complete" if classification == "success" else classification,
            None if classification == "success" else self._response_error(completed),
        )


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibSyncHttpTransport:
    """Small synchronous production transport suitable for a background worker."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        allow_redirects: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._allow_redirects = allow_redirects
        self._opener = (
            None if allow_redirects else urllib.request.build_opener(_RejectRedirectHandler())
        )

    def _request(self, request: urllib.request.Request) -> HttpResult:
        try:
            open_request = urllib.request.urlopen if self._opener is None else self._opener.open
            with open_request(request, timeout=self._timeout_seconds) as response:
                return HttpResult(response.status, response.read())
        except urllib.error.HTTPError as error:
            return HttpResult(error.code, error.read())

    def post_json(self, path: str, headers: Mapping[str, str], body: bytes) -> HttpResult:
        return self._request(
            urllib.request.Request(
                f"{self._base_url}{path}", data=body, headers=dict(headers), method="POST"
            )
        )

    def put_json(self, path: str, headers: Mapping[str, str], body: bytes) -> HttpResult:
        return self._request(
            urllib.request.Request(
                f"{self._base_url}{path}", data=body, headers=dict(headers), method="PUT"
            )
        )

    def put_file(
        self,
        url: str,
        path: Path,
        content_type: str,
        headers: Mapping[str, str],
    ) -> HttpResult:
        request_headers = dict(headers)
        request_headers["content-type"] = content_type
        return self._request(
            urllib.request.Request(
                url,
                data=path.read_bytes(),
                headers=request_headers,
                method="PUT",
            )
        )

    def put_bytes(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str],
    ) -> HttpResult:
        request_headers = dict(headers)
        request_headers["content-type"] = content_type
        return self._request(
            urllib.request.Request(
                url,
                data=payload,
                headers=request_headers,
                method="PUT",
            )
        )

    def put_bytes_once(
        self,
        url: str,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str],
    ) -> HttpResult:
        """One PUT with terminal response: no redirects and no automatic retries."""
        request_headers = dict(headers)
        request_headers["content-type"] = content_type
        request = urllib.request.Request(
            url,
            data=payload,
            headers=request_headers,
            method="PUT",
        )
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        try:
            with opener.open(request, timeout=self._timeout_seconds) as response:
                return HttpResult(response.status, response.read())
        except urllib.error.HTTPError as error:
            return HttpResult(error.code, error.read())
