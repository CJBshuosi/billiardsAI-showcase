"""Process-safe Phase 0 store for media-access grants and audit events."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.shared.models import (
    AccessAuditEvent,
    SessionAccessGrant,
    ThirdPartyProcessingApproval,
)

ACCESS_STORE_SCHEMA_VERSION = "1.0"
MAX_ACCESS_STORE_BYTES = 1_048_576
MAX_ACCESS_STORE_DEPTH = 32
T = TypeVar("T")


class AccessError(RuntimeError):
    """Base access-domain error."""


class AccessRepositoryError(AccessError):
    """The access store or trusted access identity cannot be used safely."""


class AccessConflictError(AccessError):
    """An access record conflicts with current authoritative state."""


class AccessNotFoundError(AccessError):
    """A private access record does not exist."""


def _structure_depth(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_ACCESS_STORE_DEPTH:
        return depth
    if isinstance(value, dict):
        return max(
            (_structure_depth(item, depth=depth + 1) for item in value.values()),
            default=depth,
        )
    if isinstance(value, list):
        return max((_structure_depth(item, depth=depth + 1) for item in value), default=depth)
    return depth


class JsonAccessRepository:
    """Versioned JSON repository with atomic mutation and append-only events."""

    def __init__(self, path: Path | str):
        self.path = Path(path).absolute()
        self._lock_path = self.path.with_name(f".{self.path.name}-lock")

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": ACCESS_STORE_SCHEMA_VERSION,
            "revision": 0,
            "grants": {},
            "provider_approvals": {},
            "events": [],
        }

    @contextmanager
    def _lock(self) -> Iterator[None]:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
        fd = -1
        try:
            parent = self.path.parent.lstat()
            if (
                stat.S_ISLNK(parent.st_mode)
                or not stat.S_ISDIR(parent.st_mode)
                or parent.st_uid != os.geteuid()
                or stat.S_IMODE(parent.st_mode) != 0o700
            ):
                raise AccessRepositoryError("access directory is unsafe")
            if self.path.is_symlink() or self._lock_path.is_symlink():
                raise AccessRepositoryError("access paths must not be symlinks")
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self._lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise AccessRepositoryError("access lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        except AccessRepositoryError:
            raise
        except OSError as exc:
            raise AccessRepositoryError("access lock could not be opened") from exc
        finally:
            if fd >= 0:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        fd = -1
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.path, flags)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise AccessRepositoryError("access store is unsafe")
            if info.st_size > MAX_ACCESS_STORE_BYTES:
                raise AccessRepositoryError("access store exceeds the Phase 0 size limit")
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                raw = handle.read(MAX_ACCESS_STORE_BYTES + 1)
            if len(raw) > MAX_ACCESS_STORE_BYTES:
                raise AccessRepositoryError("access store exceeds the Phase 0 size limit")
            payload = json.loads(raw.decode("utf-8"))
            if _structure_depth(payload) > MAX_ACCESS_STORE_DEPTH:
                raise AccessRepositoryError("access store exceeds the structure depth limit")
            self._validate(payload)
            return payload
        except AccessRepositoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise AccessRepositoryError("access store could not be trusted") from exc
        finally:
            if fd >= 0:
                os.close(fd)

    @classmethod
    def _validate(cls, payload: object) -> None:
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != ACCESS_STORE_SCHEMA_VERSION
        ):
            raise AccessRepositoryError("unsupported access store schema")
        if set(payload) != {
            "schema_version",
            "revision",
            "grants",
            "provider_approvals",
            "events",
        }:
            raise AccessRepositoryError("access store contains unknown fields")
        if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
            raise AccessRepositoryError("invalid access store revision")
        if not isinstance(payload.get("grants"), dict):
            raise AccessRepositoryError("invalid access grants structure")
        if not isinstance(payload.get("provider_approvals"), dict):
            raise AccessRepositoryError("invalid provider approvals structure")
        if not isinstance(payload.get("events"), list):
            raise AccessRepositoryError("invalid access events structure")
        try:
            grants = {
                key: SessionAccessGrant.model_validate(value)
                for key, value in payload["grants"].items()
            }
            approvals = {
                key: ThirdPartyProcessingApproval.model_validate(value)
                for key, value in payload["provider_approvals"].items()
            }
            events = [AccessAuditEvent.model_validate(value) for value in payload["events"]]
        except (ValidationError, RecursionError, ValueError) as exc:
            raise AccessRepositoryError("access store contains invalid records") from exc
        if any(str(grant.grant_id) != key for key, grant in grants.items()):
            raise AccessRepositoryError("grant key does not match grant ID")
        if any(str(item.approval_id) != key for key, item in approvals.items()):
            raise AccessRepositoryError("approval key does not match approval ID")
        if [event.sequence for event in events] != list(range(1, len(events) + 1)):
            raise AccessRepositoryError("access event sequence is not append-only")
        if len({event.event_id for event in events}) != len(events):
            raise AccessRepositoryError("access event IDs must be unique")

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self._validate(payload)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                os.fchmod(handle.fileno(), 0o600)
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            temporary_name = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, TypeError, ValueError) as exc:
            raise AccessRepositoryError("access store could not be persisted") from exc
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    def _snapshot(self) -> dict[str, Any]:
        with self._lock():
            return self._read_unlocked()

    def _mutate(self, operation: Callable[[dict[str, Any]], T]) -> T:
        with self._lock():
            payload = self._read_unlocked()
            result = operation(payload)
            payload["revision"] += 1
            self._write_unlocked(payload)
            return result

    def create_grant(self, grant: SessionAccessGrant) -> SessionAccessGrant:
        def operation(payload: dict[str, Any]) -> SessionAccessGrant:
            key = str(grant.grant_id)
            if key in payload["grants"]:
                raise AccessConflictError("grant ID already exists")
            active = [
                SessionAccessGrant.model_validate(value)
                for value in payload["grants"].values()
                if value.get("reviewer_id") == grant.reviewer_id
                and value.get("session_id") == str(grant.session_id)
                and value.get("revoked_at") is None
            ]
            if active:
                raise AccessConflictError(
                    "an active grant already exists for this reviewer session"
                )
            payload["grants"][key] = grant.model_dump(mode="json")
            return grant

        return self._mutate(operation)

    def revoke_grant(
        self,
        grant_id: UUID,
        *,
        revoked_at: datetime,
        revocation_reason: str,
    ) -> SessionAccessGrant:
        def operation(payload: dict[str, Any]) -> SessionAccessGrant:
            raw = payload["grants"].get(str(grant_id))
            if raw is None:
                raise AccessNotFoundError("grant was not found")
            current = SessionAccessGrant.model_validate(raw)
            if not current.is_active:
                return current
            updated = SessionAccessGrant.model_validate(
                {
                    **current.model_dump(),
                    "revoked_at": revoked_at,
                    "revocation_reason": revocation_reason,
                }
            )
            payload["grants"][str(grant_id)] = updated.model_dump(mode="json")
            return updated

        return self._mutate(operation)

    def find_active_grant(self, reviewer_id: str, session_id: UUID) -> SessionAccessGrant | None:
        grants = (
            SessionAccessGrant.model_validate(value)
            for value in self._snapshot()["grants"].values()
        )
        return next(
            (
                grant
                for grant in grants
                if grant.reviewer_id == reviewer_id
                and grant.session_id == session_id
                and grant.is_active
            ),
            None,
        )

    def list_grants(self) -> tuple[SessionAccessGrant, ...]:
        return tuple(
            SessionAccessGrant.model_validate(value)
            for value in self._snapshot()["grants"].values()
        )

    def create_provider_approval(
        self, approval: ThirdPartyProcessingApproval
    ) -> ThirdPartyProcessingApproval:
        def operation(payload: dict[str, Any]) -> ThirdPartyProcessingApproval:
            key = str(approval.approval_id)
            if key in payload["provider_approvals"]:
                raise AccessConflictError("provider approval ID already exists")
            if any(
                item.get("provider_id") == approval.provider_id and item.get("revoked_at") is None
                for item in payload["provider_approvals"].values()
            ):
                raise AccessConflictError("an active provider approval already exists")
            payload["provider_approvals"][key] = approval.model_dump(mode="json")
            return approval

        return self._mutate(operation)

    def revoke_provider_approval(
        self,
        approval_id: UUID,
        *,
        revoked_at: datetime,
        revocation_reason: str,
    ) -> ThirdPartyProcessingApproval:
        def operation(payload: dict[str, Any]) -> ThirdPartyProcessingApproval:
            raw = payload["provider_approvals"].get(str(approval_id))
            if raw is None:
                raise AccessNotFoundError("provider approval was not found")
            current = ThirdPartyProcessingApproval.model_validate(raw)
            if not current.is_active:
                return current
            updated = ThirdPartyProcessingApproval.model_validate(
                {
                    **current.model_dump(),
                    "revoked_at": revoked_at,
                    "revocation_reason": revocation_reason,
                }
            )
            payload["provider_approvals"][str(approval_id)] = updated.model_dump(mode="json")
            return updated

        return self._mutate(operation)

    def find_active_provider_approval(
        self, provider_id: str
    ) -> ThirdPartyProcessingApproval | None:
        approvals = (
            ThirdPartyProcessingApproval.model_validate(value)
            for value in self._snapshot()["provider_approvals"].values()
        )
        return next(
            (item for item in approvals if item.provider_id == provider_id and item.is_active),
            None,
        )

    def append_event(self, event: AccessAuditEvent) -> AccessAuditEvent:
        def operation(payload: dict[str, Any]) -> AccessAuditEvent:
            events = [AccessAuditEvent.model_validate(value) for value in payload["events"]]
            if any(item.event_id == event.event_id for item in events):
                raise AccessConflictError("access event ID already exists")
            stored = AccessAuditEvent.model_validate(
                {**event.model_dump(), "sequence": len(events) + 1}
            )
            payload["events"].append(stored.model_dump(mode="json"))
            return stored

        return self._mutate(operation)

    def list_events(self, *, session_id: UUID | None = None) -> tuple[AccessAuditEvent, ...]:
        events = tuple(
            AccessAuditEvent.model_validate(value) for value in self._snapshot()["events"]
        )
        if session_id is None:
            return events
        return tuple(event for event in events if event.session_id == session_id)
