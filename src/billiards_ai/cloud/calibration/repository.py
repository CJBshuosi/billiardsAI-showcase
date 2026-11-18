"""Private, versioned Phase 0 JSON persistence for calibration revisions."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.shared.models.calibration import CalibrationRecord

from .errors import CalibrationConflictError, CalibrationRepositoryError

STORE_SCHEMA_VERSION = "1.0"


class CalibrationRepository(Protocol):
    """Stable persistence boundary shared by JSON and future database stores."""

    def append(
        self,
        record: CalibrationRecord,
        *,
        expected_current_revision_id: UUID | None,
    ) -> CalibrationRecord: ...

    def get_current(self, device_id: str) -> CalibrationRecord | None: ...

    def get_history(self, device_id: str) -> tuple[CalibrationRecord, ...]: ...


class JsonCalibrationRepository:
    """Process-safe append-only calibration history with atomic replacement."""

    def __init__(self, path: Path | str):
        try:
            self.path = Path(os.path.abspath(os.fspath(path)))
        except (OSError, RuntimeError, TypeError) as exc:
            raise CalibrationRepositoryError(
                "calibration store path could not be resolved"
            ) from exc
        self._lock_path = self.path.with_name(f".{self.path.name}-lock")

    @staticmethod
    def _empty() -> dict[str, object]:
        return {"schema_version": STORE_SCHEMA_VERSION, "revision": 0, "devices": {}}

    @staticmethod
    def _require_private_directory(path: Path) -> None:
        try:
            info = path.lstat()
        except OSError as exc:
            raise CalibrationRepositoryError(
                "calibration store directory could not be inspected"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise CalibrationRepositoryError("calibration store directory is unsafe")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        fd = -1
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._require_private_directory(self.path.parent)
            if self.path.is_symlink() or self._lock_path.is_symlink():
                raise CalibrationRepositoryError("calibration paths must not be symlinks")
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self._lock_path, flags, 0o600)
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise CalibrationRepositoryError("calibration lock is unsafe")
        except CalibrationRepositoryError:
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise CalibrationRepositoryError("calibration lock could not be opened") from exc
        locked = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        except CalibrationRepositoryError:
            raise
        except OSError as exc:
            raise CalibrationRepositoryError("calibration lock operation failed") from exc
        finally:
            try:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_unlocked(self) -> dict[str, object]:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return self._empty()
        except OSError as exc:
            raise CalibrationRepositoryError("calibration store could not be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CalibrationRepositoryError("calibration store must not be a symlink")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(self.path, flags)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise CalibrationRepositoryError("calibration store is unsafe")
            with os.fdopen(fd, encoding="utf-8") as handle:
                fd = -1
                payload = json.load(handle)
        except CalibrationRepositoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationRepositoryError("calibration store could not be trusted") from exc
        finally:
            if fd >= 0:
                os.close(fd)
        self._validated_store(payload)
        return payload

    @classmethod
    def _validated_history(
        cls, device_id: str, raw_history: object
    ) -> tuple[CalibrationRecord, ...]:
        if not isinstance(raw_history, list) or not raw_history:
            raise CalibrationRepositoryError("calibration history must be a non-empty list")
        try:
            history = tuple(CalibrationRecord.model_validate(item) for item in raw_history)
        except (ValidationError, RecursionError, ValueError) as exc:
            raise CalibrationRepositoryError(
                "calibration history contains invalid records"
            ) from exc
        for index, record in enumerate(history, start=1):
            if record.device_id != device_id:
                raise CalibrationRepositoryError(
                    "calibration history does not match its device key"
                )
            if record.revision_number != index:
                raise CalibrationRepositoryError("calibration revision sequence is not contiguous")
            if index == 1:
                if record.supersedes_revision_id is not None:
                    raise CalibrationRepositoryError("first calibration revision cannot supersede")
            else:
                previous = history[index - 2]
                if record.supersedes_revision_id != previous.revision_id:
                    raise CalibrationRepositoryError("calibration revision chain is broken")
                if record.calibration_id != previous.calibration_id:
                    raise CalibrationRepositoryError("calibration identity changed within history")
                if record.created_at < previous.created_at:
                    raise CalibrationRepositoryError(
                        "calibration revision timeline moves backwards"
                    )
        return history

    @classmethod
    def _validated_store(cls, payload: object) -> dict[str, tuple[CalibrationRecord, ...]]:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "revision",
            "devices",
        }:
            raise CalibrationRepositoryError("invalid calibration store structure")
        if payload.get("schema_version") != STORE_SCHEMA_VERSION:
            raise CalibrationRepositoryError("unsupported calibration store schema")
        revision = payload.get("revision")
        devices = payload.get("devices")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise CalibrationRepositoryError("invalid calibration store revision")
        if not isinstance(devices, dict) or not all(
            isinstance(key, str) and key for key in devices
        ):
            raise CalibrationRepositoryError("invalid calibration devices structure")
        histories = {
            device_id: cls._validated_history(device_id, raw_history)
            for device_id, raw_history in devices.items()
        }
        revision_ids = [item.revision_id for history in histories.values() for item in history]
        if len(revision_ids) != len(set(revision_ids)):
            raise CalibrationRepositoryError("calibration revision IDs must be globally unique")
        if revision != len(revision_ids):
            raise CalibrationRepositoryError("calibration store revision does not match history")
        return histories

    def _write_unlocked(self, payload: dict[str, object]) -> None:
        self._validated_store(payload)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
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
            os.replace(temporary_path, self.path)
            temporary_path = None
            directory_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                directory_flags |= os.O_DIRECTORY
            directory_fd = os.open(self.path.parent, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise CalibrationRepositoryError(
                "calibration store could not be written atomically"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise CalibrationRepositoryError(
                        "calibration temporary file could not be removed"
                    ) from exc

    def append(
        self,
        record: CalibrationRecord,
        *,
        expected_current_revision_id: UUID | None,
    ) -> CalibrationRecord:
        with self._lock():
            payload = self._read_unlocked()
            histories = self._validated_store(payload)
            history = histories.get(record.device_id, ())
            current_id = history[-1].revision_id if history else None
            if current_id != expected_current_revision_id:
                raise CalibrationConflictError("calibration revision changed before append")
            if record.revision_number != len(history) + 1:
                raise CalibrationConflictError("calibration revision number is stale")
            if history:
                if record.supersedes_revision_id != current_id:
                    raise CalibrationConflictError("calibration revision does not extend current")
                if record.calibration_id != history[-1].calibration_id:
                    raise CalibrationConflictError("calibration identity changed")
            elif record.supersedes_revision_id is not None:
                raise CalibrationConflictError("first calibration revision cannot supersede")
            devices = payload["devices"]
            assert isinstance(devices, dict)
            devices.setdefault(record.device_id, []).append(record.model_dump(mode="json"))
            payload["revision"] = int(payload["revision"]) + 1
            self._write_unlocked(payload)
            return record

    def get_history(self, device_id: str) -> tuple[CalibrationRecord, ...]:
        with self._lock():
            histories = self._validated_store(self._read_unlocked())
            return histories.get(device_id, ())

    def get_current(self, device_id: str) -> CalibrationRecord | None:
        history = self.get_history(device_id)
        return history[-1] if history else None
