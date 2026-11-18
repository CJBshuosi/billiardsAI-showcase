"""Trusted, persistent asset-context evidence boundary for staged media."""

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol
from uuid import UUID

from billiards_ai.shared.models import AssetContext
from billiards_ai.shared.models.session_capture import SessionCaptureErrorCode

from .errors import SessionCaptureServiceError
from .identity import AuthenticatedStaffProvider, StaffPrincipal


class AssetContextProvider(Protocol):
    def resolve(self, *, attempt_id: UUID, asset_id: UUID, content_sha256: str) -> AssetContext: ...


class AssetContextEvidenceRepository(Protocol):
    def save(self, *, attempt_id: UUID, context: AssetContext, operator_id: UUID) -> None: ...
    def get(self, *, attempt_id: UUID) -> AssetContext | None: ...


class JsonAssetContextEvidenceRepository:
    """Phase 0 operator-controlled evidence inventory; staff requests never populate it."""

    def __init__(self, path: Path | str):
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.lock_path = self.path.with_name(f".{self.path.name}-lock")

    @staticmethod
    def _pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("asset evidence store contains duplicate keys")
            result[key] = value
        return result

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        current = Path(self.path.anchor)
        for component in self.path.parent.parts[1:]:
            current /= component
            ancestor = current.lstat()
            mode = stat.S_IMODE(ancestor.st_mode)
            if stat.S_ISLNK(ancestor.st_mode) or not stat.S_ISDIR(ancestor.st_mode):
                raise RuntimeError("asset evidence store ancestor is unsafe")
            if current != self.path.parent and ancestor.st_uid not in {0, os.geteuid()}:
                raise RuntimeError("asset evidence store ancestor is unsafe")
            if (
                current != self.path.parent
                and mode & 0o022
                and not (mode & stat.S_ISVTX and ancestor.st_uid == 0)
            ):
                raise RuntimeError("asset evidence store ancestor is unsafe")
        info = self.path.parent.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError("asset evidence store parent is unsafe")

    @contextmanager
    def _locked(self):
        self._ensure_parent()
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
                raise RuntimeError("asset evidence lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"schema_version": "1.0", "evidence": {}}
        info = self.path.lstat()
        if self.path.is_symlink() or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise RuntimeError("asset evidence store is unsafe")
        if info.st_size > 4 * 1024 * 1024 or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("asset evidence store is unsafe")
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=self._pairs)
        if set(payload) != {"schema_version", "evidence"} or payload["schema_version"] != "1.0":
            raise RuntimeError("asset evidence store schema is unsupported")
        if not isinstance(payload["evidence"], dict):
            raise RuntimeError("asset evidence store is invalid")
        for key, value in payload["evidence"].items():
            UUID(key)
            if not isinstance(value, dict) or set(value) != {"context", "operator_id"}:
                raise RuntimeError("asset evidence record is invalid")
            AssetContext.model_validate(value["context"])
            UUID(value["operator_id"])
        return payload

    def save(self, *, attempt_id: UUID, context: AssetContext, operator_id: UUID) -> None:
        with self._locked():
            payload = self._read()
            key = str(attempt_id)
            record = {"context": context.model_dump(mode="json"), "operator_id": str(operator_id)}
            existing = payload["evidence"].get(key)
            if existing is not None and existing != record:
                raise RuntimeError("asset evidence is append-only")
            payload["evidence"][key] = record
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                temp = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def get(self, *, attempt_id: UUID) -> AssetContext | None:
        with self._locked():
            raw = self._read()["evidence"].get(str(attempt_id))
        return None if raw is None else AssetContext.model_validate(raw["context"])


class RepositoryAssetContextProvider:
    def __init__(self, repository: AssetContextEvidenceRepository):
        self.repository = repository

    def resolve(self, *, attempt_id: UUID, asset_id: UUID, content_sha256: str) -> AssetContext:
        context = self.repository.get(attempt_id=attempt_id)
        if (
            context is None
            or context.asset_id != str(asset_id)
            or context.asset_sha256 != content_sha256
        ):
            raise RuntimeError("trusted asset evidence is unavailable")
        return context


class TrustedAssetContextEvidenceService:
    """Closed operator entry point; request payloads cannot supply the principal."""

    def __init__(
        self,
        repository: JsonAssetContextEvidenceRepository,
        identity_provider: AuthenticatedStaffProvider,
    ):
        self.repository = repository
        self.identity_provider = identity_provider

    def record(self, *, attempt_id: UUID, context: AssetContext) -> None:
        try:
            raw = self.identity_provider.current_staff()
            principal = StaffPrincipal(raw.staff_id, raw.role, tuple(raw.venue_ids))
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED
            ) from None
        self.repository.save(attempt_id=attempt_id, context=context, operator_id=principal.staff_id)
