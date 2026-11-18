"""Narrow deployment-owned upload source and controlled local staging adapter."""

from __future__ import annotations

import hashlib
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID

from billiards_ai.cloud.media.runner import SubprocessFFmpegRunner
from billiards_ai.cloud.privacy.media_store import ControlledFilesystemMediaStore
from billiards_ai.cloud.privacy.retention_repository import RetentionRepositoryError
from billiards_ai.shared.models.session_capture import SessionCaptureLimits

__all__ = [
    "BinarySource",
    "ControlledStagingObjectStore",
    "ObjectStore",
    "SessionCaptureLimits",
]


class BinarySource(Protocol):
    def open(self) -> BinaryIO: ...


class ControlledStagingObjectStore:
    """Extends the existing validated media root; it does not accept caller paths."""

    def __init__(self, media_store: ControlledFilesystemMediaStore):
        self._media_store = media_store
        staged = media_store._resolved_root / "staged"
        try:
            staged.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = staged.lstat()
        if staged.is_symlink() or not staged.is_dir() or info.st_uid != os.geteuid():
            raise RetentionRepositoryError("staging namespace is unsafe")
        if info.st_mode & 0o077:
            raise RetentionRepositoryError("staging namespace permissions are unsafe")

    def stage(
        self, *, attempt_id: UUID, source: BinarySource, max_bytes: int
    ) -> tuple[str, str, int]:
        locator = f"staged/{attempt_id}.bin"
        parent_fd, leaf = self._media_store._open_parent(locator)
        fd = -1
        digest = hashlib.sha256()
        count = 0
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
            with closing(source.open()) as stream:
                while chunk := stream.read(1024 * 1024):
                    if not isinstance(chunk, bytes):
                        raise RetentionRepositoryError("binary source returned non-bytes")
                    count += len(chunk)
                    if count > max_bytes:
                        raise RetentionRepositoryError("media exceeds upload limit")
                    digest.update(chunk)
                    remaining = memoryview(chunk)
                    while remaining:
                        try:
                            written = os.write(fd, remaining)
                        except InterruptedError:
                            continue
                        if written <= 0:
                            raise RetentionRepositoryError("media object write made no progress")
                        remaining = remaining[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.fsync(parent_fd)
            return locator, digest.hexdigest(), count
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(leaf, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(parent_fd)

    def delete(self, locator: str) -> None:
        self._media_store.delete(locator)

    def internal_path(self, locator: str) -> Path:
        if not self._media_store.exists(locator):
            raise RetentionRepositoryError("media object is unavailable")
        return self._media_store._resolved_root.joinpath(*self._media_store._locator_parts(locator))


class ObjectStore(Protocol):
    def stage(
        self, *, attempt_id: UUID, source: BinarySource, max_bytes: int
    ) -> tuple[str, str, int]: ...
    def delete(self, locator: str) -> None: ...


@dataclass(frozen=True, slots=True)
class InspectedMediaFacts:
    container: str
    video_codec: str
    duration_seconds: float
    audio_present: bool


class FFmpegMediaInspector:
    """Compose the existing ffprobe adapter; no shell or duplicate probe implementation."""

    def __init__(self, object_store: ControlledStagingObjectStore):
        self.object_store = object_store

    def inspect(self, locator: str, *, timeout_seconds: float) -> InspectedMediaFacts:
        probe = SubprocessFFmpegRunner(timeout_seconds=timeout_seconds).probe(
            self.object_store.internal_path(locator)
        )
        pts = [item for item in probe.frame_pts if item is not None]
        if not pts:
            raise RetentionRepositoryError("media duration is unavailable")
        scale = probe.time_base.numerator / probe.time_base.denominator
        duration = getattr(probe, "duration_seconds", None)
        if duration is None:
            durations = getattr(probe, "frame_durations", ())
            if len(durations) == len(probe.frame_pts) and all(
                item is not None for item in durations
            ):
                duration = sum(durations) * scale
            else:
                duration = (max(pts) - min(pts) + 1) * scale
        aliases = {item.strip().lower() for item in probe.container.split(",")}
        container = "mp4" if "mp4" in aliases else sorted(aliases)[0]
        return InspectedMediaFacts(
            container=container,
            video_codec=probe.video_codec,
            duration_seconds=float(duration),
            audio_present=probe.audio_present,
        )
