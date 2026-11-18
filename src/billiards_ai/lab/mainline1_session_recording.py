"""Continuous segmented session recording with reference-only shot intervals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError, CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import (
    _MAGIC,
    _RECORD,
    FrozenCaptureInterval,
    RealtimeRingError,
    RealtimeRingHealth,
    RealtimeRingState,
)


@dataclass(frozen=True, slots=True)
class SessionRecordingConfig:
    segment_duration_seconds: float = 1.0
    memory_buffer_seconds: float = 10.0
    reserve_bytes: int = 2 * 1024**3
    minimum_write_mbps: float = 64.0
    shutdown_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.segment_duration_seconds)
            or self.segment_duration_seconds <= 0
            or not math.isfinite(self.memory_buffer_seconds)
            or self.memory_buffer_seconds < self.segment_duration_seconds
            or self.reserve_bytes <= 0
            or not math.isfinite(self.minimum_write_mbps)
            or self.minimum_write_mbps < 0
            or not math.isfinite(self.shutdown_timeout_seconds)
            or self.shutdown_timeout_seconds <= 0
        ):
            raise ValueError("session recording configuration is invalid")


@dataclass(frozen=True, slots=True)
class RecordingPreflight:
    available_bytes: int
    reserved_bytes: int
    measured_write_mbps: float
    minimum_write_mbps: float


class RecordingHealthState(StrEnum):
    RUNNING = "running"
    DEGRADED = "degraded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RecordingHealth:
    state: RecordingHealthState
    committed_segments: int
    committed_frames: int
    buffered_frames: int
    pending_chunks: int
    maximum_pending_chunks: int
    reason: str | None = None


class StorageProbe(Protocol):
    def available_bytes(self, root: Path) -> int: ...

    def benchmark_write_mbps(self, root: Path) -> float: ...


class LocalStorageProbe:
    """Measure the exact filesystem used by the session recording."""

    _BENCHMARK_BYTES = 8 * 1024**2

    def available_bytes(self, root: Path) -> int:
        return shutil.disk_usage(root).free

    def benchmark_write_mbps(self, root: Path) -> float:
        path = root / f".recording-preflight-{uuid4().hex}"
        block = bytes(1024**2)
        started = time.perf_counter()
        try:
            with path.open("xb") as stream:
                for _ in range(self._BENCHMARK_BYTES // len(block)):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            elapsed = time.perf_counter() - started
        finally:
            path.unlink(missing_ok=True)
        return self._BENCHMARK_BYTES / 1_000_000 / max(elapsed, 1e-9)


@dataclass(frozen=True, slots=True)
class SessionSegment:
    name: str
    first_frame_id: int
    last_frame_id: int
    first_captured_at_ns: int
    last_captured_at_ns: int
    frame_count: int
    payload_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or Path(self.name).name != self.name
            or self.first_frame_id < 0
            or self.last_frame_id < self.first_frame_id
            or self.first_captured_at_ns < 0
            or self.last_captured_at_ns < self.first_captured_at_ns
            or self.frame_count != self.last_frame_id - self.first_frame_id + 1
            or self.payload_bytes <= 0
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("session recording segment is invalid")

    def row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "first_captured_at_ns": self.first_captured_at_ns,
            "last_captured_at_ns": self.last_captured_at_ns,
            "frame_count": self.frame_count,
            "payload_bytes": self.payload_bytes,
            "sha256": self.sha256,
        }


class SessionSegmentObserver(Protocol):
    """Receive durable segment metadata without blocking the capture writer."""

    def submit(self, segment: SessionSegment) -> None: ...

    def close_input(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionRecordingManifest:
    session_id: str
    source_epoch_id: str
    width: int
    height: int
    segments: tuple[SessionSegment, ...]


@dataclass(frozen=True, slots=True)
class RecordedShotInterval:
    session_id: str
    shot_id: str
    root: Path
    source_epoch_id: str
    width: int
    height: int
    start_frame_id: int
    end_frame_id: int
    segment_sha256: dict[str, str]

    @property
    def frame_count(self) -> int:
        return self.end_frame_id - self.start_frame_id + 1

    @property
    def media_sha256(self) -> str:
        return self._frozen().media_sha256

    def frames(self):
        return self._frozen().frames()

    def _frozen(self) -> FrozenCaptureInterval:
        return FrozenCaptureInterval(
            root=self.root,
            source_epoch_id=self.source_epoch_id,
            width=self.width,
            height=self.height,
            start_frame_id=self.start_frame_id,
            end_frame_id=self.end_frame_id,
            frame_count=self.frame_count,
            segment_sha256=dict(self.segment_sha256),
        )


class ContinuousSessionRecorder:
    """Write fixed-duration chunks once and reference them for every shot."""

    def __init__(
        self,
        *,
        root: Path,
        session_id: str,
        expected_fps: int,
        config: SessionRecordingConfig | None = None,
        storage_probe: StorageProbe | None = None,
        read_only: bool = False,
        guard_writer: bool = False,
    ) -> None:
        if not session_id.strip() or expected_fps <= 0:
            raise ValueError("session recording identity is invalid")
        self._root = root.resolve()
        if read_only:
            if not (self._root / "session-recording.json").is_file():
                raise RealtimeRingError("processing snapshot requires a committed manifest")
        else:
            self._root.mkdir(parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise RealtimeRingError("session recording root must be a real directory")
        self._session_id = session_id
        self._expected_fps = expected_fps
        self._config = config or SessionRecordingConfig()
        self._storage_probe = storage_probe or LocalStorageProbe()
        self._segment_frames = max(
            1,
            round(self._expected_fps * self._config.segment_duration_seconds),
        )
        maximum_chunks = max(
            1,
            math.ceil(self._config.memory_buffer_seconds / self._config.segment_duration_seconds),
        )
        self._chunks: queue.Queue[tuple[CaptureFrame, ...] | None] = queue.Queue(
            maxsize=maximum_chunks
        )
        self._maximum_pending_chunks = maximum_chunks
        self._append_lock = threading.Lock()
        self._condition = threading.Condition()
        self._pending: list[CaptureFrame] = []
        self._segments: list[SessionSegment] = []
        self._source_epoch_id: str | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._first_frame_id: int | None = None
        self._last_frame_id: int | None = None
        self._manifest_committed_frame_id = -1
        self._last_timestamp_ns: int | None = None
        self._writer_failure: BaseException | None = None
        self._failure_reason: str | None = None
        self._closed_manifest: SessionRecordingManifest | None = None
        self._segment_receipts = {}
        self._segment_observer: SessionSegmentObserver | None = None
        self._observer_input_closed = False
        self._writer_ownership = None
        if guard_writer and not read_only:
            from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

            self._writer_ownership = session_media_lock(
                self._root.parent, "recording-writer", exclusive=True
            )
            self._writer_ownership.__enter__()
        try:
            self._load_manifest()
            if read_only:
                # A finite committed-prefix view, not proof capture ended normally.
                # It owns no writer, runs no benchmark and never rewrites the manifest.
                self._closed_manifest = self._snapshot_manifest()
                self._writer = None
                return
            self.preflight()
            self._writer = threading.Thread(
                target=self._write_chunks,
                name="mainline1-session-recorder",
                daemon=True,
            )
            self._writer.start()
        except BaseException:
            self._release_writer_ownership()
            raise

    def _release_writer_ownership(self):
        if self._writer_ownership is not None:
            self._writer_ownership.__exit__(None, None, None)
            self._writer_ownership = None

    @property
    def health(self) -> RecordingHealth:
        with self._condition:
            segments = tuple(self._segments)
            writer_failure = self._writer_failure
            failure_reason = self._failure_reason
        if writer_failure is not None:
            state = RecordingHealthState.DEGRADED
            reason = f"{type(writer_failure).__name__}: {writer_failure}"
        elif failure_reason is not None:
            state = RecordingHealthState.DEGRADED
            reason = failure_reason
        elif self._closed_manifest is not None:
            state = RecordingHealthState.CLOSED
            reason = None
        else:
            state = RecordingHealthState.RUNNING
            reason = None
        return RecordingHealth(
            state=state,
            committed_segments=len(segments),
            committed_frames=sum(segment.frame_count for segment in segments),
            buffered_frames=len(self._pending),
            pending_chunks=self._chunks.qsize(),
            maximum_pending_chunks=self._maximum_pending_chunks,
            reason=reason,
        )

    @property
    def pending_command_count(self) -> int:
        """Count queued and in-flight writes without changing queue health fields."""
        with self._chunks.all_tasks_done:
            return self._chunks.unfinished_tasks

    @property
    def first_frame_id(self) -> int | None:
        with self._append_lock:
            return self._first_frame_id

    def preflight(self) -> RecordingPreflight:
        available_bytes = self._storage_probe.available_bytes(self._root)
        if available_bytes < self._config.reserve_bytes:
            raise RealtimeRingError("session recording reserved space is unavailable")
        measured_write_mbps = (
            math.inf
            if self._config.minimum_write_mbps == 0
            else self._storage_probe.benchmark_write_mbps(self._root)
        )
        if measured_write_mbps < self._config.minimum_write_mbps:
            raise RealtimeRingError("session recording write throughput is unsupported")
        return RecordingPreflight(
            available_bytes=available_bytes,
            reserved_bytes=self._config.reserve_bytes,
            measured_write_mbps=measured_write_mbps,
            minimum_write_mbps=self._config.minimum_write_mbps,
        )

    def set_segment_observer(self, observer: SessionSegmentObserver) -> None:
        """Attach one asynchronous consumer and replay already durable segments."""
        with self._condition:
            if self._segment_observer is not None:
                raise ValueError("session recording segment observer is already configured")
            if self._observer_input_closed:
                raise RealtimeRingError("session recording observer input is closed")
            self._segment_observer = observer
            existing = tuple(self._segments)
        for segment in existing:
            observer.submit(segment)

    def append(self, frame: CaptureFrame) -> None:
        with self._append_lock:
            self._require_writable()
            if self._source_epoch_id is None:
                self._source_epoch_id = frame.source_epoch_id
                self._width = frame.width
                self._height = frame.height
                self._first_frame_id = frame.capture_frame_id
            if (
                frame.source_epoch_id != self._source_epoch_id
                or frame.width != self._width
                or frame.height != self._height
            ):
                raise RealtimeRingError("session recording source epoch or geometry changed")
            if (
                self._last_frame_id is not None
                and frame.capture_frame_id != self._last_frame_id + 1
            ):
                raise RealtimeRingError("session recording frames must be contiguous")
            if (
                self._last_timestamp_ns is not None
                and frame.captured_at_ns < self._last_timestamp_ns
            ):
                raise RealtimeRingError("session recording timestamps must be monotonic")
            self._pending.append(frame)
            self._last_frame_id = frame.capture_frame_id
            self._last_timestamp_ns = frame.captured_at_ns
            with self._condition:
                self._condition.notify_all()
            if len(self._pending) >= self._segment_frames:
                self._enqueue_pending()

    def close(self, *, timeout_seconds=None) -> SessionRecordingManifest:
        deadline = time.monotonic() + (
            self._config.shutdown_timeout_seconds if timeout_seconds is None else timeout_seconds
        )

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise RealtimeRingError("session recording close exceeded timeout")
            return budget

        if not self._append_lock.acquire(timeout=remaining()):
            raise RealtimeRingError("session recording close exceeded timeout")
        try:
            if self._closed_manifest is not None:
                return self._closed_manifest
            self._require_writer_healthy()
            self._enqueue_pending(timeout_seconds=remaining())
            try:
                self._chunks.put(None, timeout=remaining())
            except queue.Full as exc:
                raise RealtimeRingError("session recorder could not schedule shutdown") from exc
        finally:
            self._append_lock.release()
        self._writer.join(timeout=remaining())
        if self._writer.is_alive():
            raise RealtimeRingError("session recorder did not stop within timeout")
        self._require_writer_healthy()
        manifest = self._snapshot_manifest()
        remaining()
        self._write_manifest(manifest, deadline=deadline)
        with self._condition:
            self._closed_manifest = manifest
            self._condition.notify_all()
        self._close_observer_input()
        self._release_writer_ownership()
        return manifest

    @property
    def finalized_manifest_path(self):
        if self._closed_manifest is None:
            raise RealtimeRingError("session recording is not finalized")
        self._require_writer_healthy()
        return (self._root / "session-recording.json").resolve()

    @staticmethod
    def segment_file_identity(path):
        if path.is_symlink() or not path.is_file():
            raise RealtimeRingError("recording segment is missing or unsafe")
        value = path.stat()
        return dict(
            size=value.st_size,
            device=value.st_dev,
            inode=value.st_ino,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    @property
    def finalized_segment_receipts(self):
        _ = self.finalized_manifest_path
        return [
            dict(name=segment.name, sha256=segment.sha256, **self._segment_receipts[segment.name])
            for segment in self._segments
        ]

    def read_segment(self, segment: SessionSegment) -> tuple[CaptureFrame, ...]:
        """Verify and read one atomically committed segment."""
        with self._condition:
            if segment not in self._segments:
                raise RealtimeRingError("session recording segment is not committed")
            source_epoch_id = self._source_epoch_id
            width = self._width
            height = self._height
        if source_epoch_id is None or width is None or height is None:
            raise RealtimeRingError("session recording identity is unavailable")
        interval = FrozenCaptureInterval(
            root=self._root,
            source_epoch_id=source_epoch_id,
            width=width,
            height=height,
            start_frame_id=segment.first_frame_id,
            end_frame_id=segment.last_frame_id,
            frame_count=segment.frame_count,
            segment_sha256={segment.name: segment.sha256},
        )
        path = self._root / segment.name
        receipt = self._segment_receipts[segment.name]
        if self.segment_file_identity(path) != receipt:
            raise RealtimeRingError("session recording segment identity changed")
        frames = tuple(interval.frames())
        if self.segment_file_identity(path) != receipt:
            raise RealtimeRingError("session recording segment identity changed")
        return frames

    def wait_for_segment(
        self,
        index: int,
        *,
        timeout_seconds: float = 0.1,
    ) -> SessionSegment | None:
        """Follow the durable manifest without queueing media or waiting for ACKs."""
        if (
            type(index) is not int
            or index < 0
            or (not math.isfinite(timeout_seconds) or timeout_seconds <= 0)
        ):
            raise ValueError("session recording segment wait is invalid")
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    index < len(self._segments)
                    or self._closed_manifest is not None
                    or self._writer_failure is not None
                    or self._failure_reason is not None
                ),
                timeout=timeout_seconds,
            )
            if index < len(self._segments):
                return self._segments[index]
            self._require_writer_healthy()
            if self._failure_reason is not None:
                raise RealtimeRingError(self._failure_reason)
            return None

    def commit_pending(self, *, timeout_seconds: float = 5.0) -> int | None:
        """Close the current partial segment and wait until it is durable."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("session recording commit timeout is invalid")
        deadline = time.monotonic() + timeout_seconds

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise RealtimeRingError("session recording commit exceeded timeout")
            return budget

        if not self._append_lock.acquire(timeout=remaining()):
            raise RealtimeRingError("session recording commit exceeded timeout")
        try:
            self._require_writable()
            target_frame_id = self._last_frame_id
            self._enqueue_pending(timeout_seconds=remaining())
        finally:
            self._append_lock.release()
        if target_frame_id is None:
            return None
        if not self.wait_until_committed(target_frame_id, timeout_seconds=remaining()):
            raise RealtimeRingError("session recording commit exceeded timeout")
        return target_frame_id

    def commit_through(self, frame_id: int, *, timeout_seconds: float = 5.0) -> None:
        """Fence an explicit received frame and its durable write on one deadline."""
        if (
            type(frame_id) is not int
            or frame_id < 0
            or (not math.isfinite(timeout_seconds) or timeout_seconds <= 0)
        ):
            raise ValueError("recording target fence is invalid")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            if self._manifest_committed_frame_id >= frame_id:
                return
            while self._last_frame_id is None or self._last_frame_id < frame_id:
                self._require_writable()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RealtimeRingError("recording target frame was not received")
                self._condition.wait(remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._append_lock.acquire(timeout=remaining):
            raise RealtimeRingError("recording target commit exceeded timeout")
        try:
            self._require_writable()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RealtimeRingError("recording target commit exceeded timeout")
            self._enqueue_pending(timeout_seconds=remaining)
        finally:
            self._append_lock.release()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.wait_until_committed(frame_id, timeout_seconds=remaining):
            raise RealtimeRingError("recording target commit exceeded timeout")

    def wait_until_committed(self, frame_id: int, *, timeout_seconds: float = 5.0) -> bool:
        if frame_id < 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("session recording commit wait is invalid")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._manifest_committed_frame_id < frame_id:
                self._require_writer_healthy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def reference_interval(
        self,
        *,
        session_id: str,
        shot_id: str,
        source_epoch_id: str,
        start_frame_id: int,
        end_frame_id: int,
    ) -> RecordedShotInterval:
        if (
            session_id != self._session_id
            or not shot_id.strip()
            or source_epoch_id != self._source_epoch_id
            or start_frame_id < 0
            or end_frame_id < start_frame_id
        ):
            raise ValueError("recorded shot interval identity is invalid")
        with self._condition:
            selected = tuple(
                segment
                for segment in self._segments
                if segment.last_frame_id >= start_frame_id
                and segment.first_frame_id <= end_frame_id
            )
        if (
            not selected
            or selected[0].first_frame_id > start_frame_id
            or selected[-1].last_frame_id < end_frame_id
        ):
            raise RealtimeRingError("recorded shot interval is not committed")
        interval = RecordedShotInterval(
            session_id=session_id,
            shot_id=shot_id,
            root=self._root,
            source_epoch_id=source_epoch_id,
            width=self._width or 0,
            height=self._height or 0,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            segment_sha256={segment.name: segment.sha256 for segment in selected},
        )
        tuple(interval.frames())
        return interval

    def _enqueue_pending(self, *, timeout_seconds: float | None = None) -> None:
        if not self._pending:
            return
        chunk = tuple(self._pending)
        try:
            if timeout_seconds is None:
                self._chunks.put_nowait(chunk)
            else:
                self._chunks.put(chunk, timeout=timeout_seconds)
        except queue.Full as exc:
            reason = "session recording segment buffer is exhausted"
            with self._condition:
                self._failure_reason = reason
                self._condition.notify_all()
            raise RealtimeRingError(reason) from exc
        self._pending.clear()

    def _require_writable(self) -> None:
        if self._closed_manifest is not None:
            raise RealtimeRingError("session recorder is closed")
        if self._failure_reason is not None:
            raise RealtimeRingError(self._failure_reason)
        self._require_writer_healthy()

    def _require_writer_healthy(self) -> None:
        if self._writer_failure is not None:
            raise RealtimeRingError(
                f"session recorder writer failed: {type(self._writer_failure).__name__}: "
                f"{self._writer_failure}"
            ) from self._writer_failure

    def _write_chunks(self) -> None:
        try:
            while True:
                chunk = self._chunks.get()
                try:
                    if chunk is None:
                        return
                    segment = self._write_segment(chunk)
                    with self._condition:
                        manifest = replace(
                            self._snapshot_manifest(),
                            segments=(*self._segments, segment),
                        )
                    self._write_manifest(manifest)
                    with self._condition:
                        # Publish only after both file and manifest are durable.
                        # Readers, references and health share this same boundary.
                        self._segments.append(segment)
                        self._manifest_committed_frame_id = segment.last_frame_id
                        self._condition.notify_all()
                    observer = self._segment_observer
                    if observer is not None:
                        observer.submit(segment)
                finally:
                    self._chunks.task_done()
        except BaseException as exc:
            with self._condition:
                self._writer_failure = exc
                self._condition.notify_all()

    def _close_observer_input(self) -> None:
        with self._condition:
            if self._observer_input_closed:
                return
            self._observer_input_closed = True
            observer = self._segment_observer
        if observer is not None:
            observer.close_input()

    def _write_segment(self, frames: tuple[CaptureFrame, ...]) -> SessionSegment:
        first = frames[0]
        last = frames[-1]
        name = f"{first.capture_frame_id:020d}-{last.capture_frame_id:020d}.mjpgseg"
        destination = self._root / name
        temporary = self._root / f".pending-{uuid4().hex}.mjpgseg"
        required_bytes = len(_MAGIC) + sum(_RECORD.size + len(frame.mjpeg) for frame in frames)
        if (
            self._storage_probe.available_bytes(self._root)
            < self._config.reserve_bytes + required_bytes
        ):
            raise RealtimeRingError("session recording reserved space would be exhausted")
        digest = hashlib.sha256()
        payload_bytes = 0
        try:
            with temporary.open("xb") as stream:
                stream.write(_MAGIC)
                digest.update(_MAGIC)
                for frame in frames:
                    header = _RECORD.pack(
                        frame.capture_frame_id,
                        frame.captured_at_ns,
                        len(frame.mjpeg),
                    )
                    stream.write(header)
                    stream.write(frame.mjpeg)
                    digest.update(header)
                    digest.update(frame.mjpeg)
                    payload_bytes += len(frame.mjpeg)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._segment_receipts[name] = self.segment_file_identity(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return SessionSegment(
            name=name,
            first_frame_id=first.capture_frame_id,
            last_frame_id=last.capture_frame_id,
            first_captured_at_ns=first.captured_at_ns,
            last_captured_at_ns=last.captured_at_ns,
            frame_count=len(frames),
            payload_bytes=payload_bytes,
            sha256=digest.hexdigest(),
        )

    def _snapshot_manifest(self) -> SessionRecordingManifest:
        return SessionRecordingManifest(
            session_id=self._session_id,
            source_epoch_id=self._source_epoch_id or "",
            width=self._width or 0,
            height=self._height or 0,
            segments=tuple(self._segments),
        )

    def _write_manifest(
        self,
        manifest: SessionRecordingManifest,
        *,
        deadline: float | None = None,
    ) -> None:
        payload = {
            "schema_version": "1.0",
            "kind": "mainline1_session_recording",
            "session_id": manifest.session_id,
            "source_epoch_id": manifest.source_epoch_id,
            "width": manifest.width,
            "height": manifest.height,
            "segments": [segment.row() for segment in manifest.segments],
        }
        temporary = self._root / f".pending-manifest-{uuid4().hex}.json"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # A short-lived Windows reader/scan can deny atomic replacement.
            # Retry only this already-fsynced file, outside capture admission
            # and condition locks; persistent denial must still fail recording.
            if deadline is not None and time.monotonic() >= deadline:
                raise RealtimeRingError("session recording close exceeded timeout")
            replace_deadline = time.monotonic() + 0.25
            if deadline is not None:
                replace_deadline = min(replace_deadline, deadline)
            for attempt in range(26):
                try:
                    os.replace(temporary, self._root / "session-recording.json")
                    break
                except OSError as exc:
                    remaining = replace_deadline - time.monotonic()
                    if (
                        os.name != "nt"
                        or getattr(exc, "winerror", None) not in {5, 32, 33}
                        or attempt == 25
                        or remaining <= 0
                    ):
                        raise
                    time.sleep(min(0.01, remaining))
                    if time.monotonic() >= replace_deadline:
                        raise
            # Publish the fence only after both segment and manifest directory
            # entries are durable. Windows uses the existing file-fsync boundary.
            try:
                descriptor = os.open(self._root, os.O_RDONLY)
            except OSError:
                if os.name != "nt":
                    raise
            else:
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Preserve the publication failure; leave the pending file as evidence.
            raise
        else:
            temporary.unlink(missing_ok=True)

    def _load_manifest(self) -> None:
        path = self._root / "session-recording.json"
        if not path.exists():
            if any(self._root.glob("*.mjpgseg")):
                raise RealtimeRingError("session recording manifest is missing")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != "1.0"
                or payload.get("kind") != "mainline1_session_recording"
                or payload.get("session_id") != self._session_id
                or not isinstance(payload.get("source_epoch_id"), str)
                or not isinstance(payload.get("width"), int)
                or not isinstance(payload.get("height"), int)
                or not isinstance(payload.get("segments"), list)
            ):
                raise ValueError("manifest shape")
            segments = tuple(SessionSegment(**row) for row in payload["segments"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RealtimeRingError("session recording manifest is invalid") from exc
        expected_first = 0 if not segments else segments[0].first_frame_id
        last_timestamp_ns: int | None = None
        for segment in segments:
            if segment.first_frame_id != expected_first:
                raise RealtimeRingError("session recording manifest frames are not contiguous")
            if last_timestamp_ns is not None and segment.first_captured_at_ns < last_timestamp_ns:
                raise RealtimeRingError("session recording manifest timestamps reversed")
            segment_path = self._root / segment.name
            if _sha256_file(segment_path) != segment.sha256:
                raise RealtimeRingError("session recording segment hash changed")
            self._segment_receipts[segment.name] = self.segment_file_identity(segment_path)
            expected_first = segment.last_frame_id + 1
            last_timestamp_ns = segment.last_captured_at_ns
        self._segments.extend(segments)
        self._source_epoch_id = payload["source_epoch_id"] or None
        self._width = payload["width"] or None
        self._height = payload["height"] or None
        if segments:
            self._first_frame_id = segments[0].first_frame_id
            self._last_frame_id = segments[-1].last_frame_id
            self._manifest_committed_frame_id = segments[-1].last_frame_id
            self._last_timestamp_ns = segments[-1].last_captured_at_ns


@dataclass(frozen=True, slots=True)
class RecordingReadProgress:
    """Delivery telemetry only; this is not a recovered detector checkpoint."""

    session_id: str
    source_epoch_id: str
    last_delivered_frame_id: int | None
    committed_frame_id: int | None
    input_closed: bool
    cancelled: bool
    failure_reason: str | None


class CommittedRecordingSource:
    """Sequentially read durable media; keep at most one segment in memory.

    The writer publishes metadata after fsync and never calls this consumer.
    Cancelling a reader does not cancel capture, close the writer or erase media.
    A new reader starts at the beginning: frame delivery alone cannot restore
    motion references, active shots or publication state after a process crash.
    """

    allow_accounted_frame_gaps = False

    def __init__(
        self,
        *,
        recorder: ContinuousSessionRecorder,
        session_id: str,
        source_epoch_id: str,
        capture_failure: Callable[[], str | None] | None = None,
    ) -> None:
        if session_id != recorder._session_id or not source_epoch_id.strip():
            raise ValueError("recording reader session identity is invalid")
        self._recorder = recorder
        self._session_id = session_id
        self._source_epoch_id = source_epoch_id
        self._capture_failure = capture_failure
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._last_delivered: CaptureFrame | None = None
        self._sealed = False
        self._cutoff: CaptureFrame | None = None
        self._failure_reason: str | None = None
        self._input_digest = self._new_input_digest()
        self._current_segment_input = None
        self._terminal_capture_failure = None

    def _new_input_digest(self):
        return hashlib.sha256(
            json.dumps(
                dict(
                    root=str(self._recorder._root),
                    session_id=self._session_id,
                    source_epoch_id=self._source_epoch_id,
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    def _segment_input(self, index, segment, digest):
        identity = dict(
            segment=segment.row(),
            receipt=self._recorder._segment_receipts[segment.name],
        )
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        return dict(
            segment_index=index,
            last_frame_id=segment.last_frame_id,
            last_captured_at_ns=segment.last_captured_at_ns,
            prefix_sha256=digest.hexdigest(),
        )

    @property
    def terminal_capture_failure(self):
        """A real capture stop observed only after its durable prefix was drained."""
        with self._lock:
            return self._terminal_capture_failure

    @property
    def current_segment_input(self):
        with self._lock:
            return (
                None if self._current_segment_input is None else dict(self._current_segment_input)
            )

    def validate_input_prefix(self, expected):
        """Bind restored processing to the same committed session/epoch/media prefix."""
        index = expected.get("segment_index")
        if type(index) is not int or index < 0:
            raise RealtimeRingError("recording processing input prefix is invalid")
        with self._recorder._condition:
            segments = tuple(self._recorder._segments[: index + 1])
        if len(segments) != index + 1:
            raise RealtimeRingError("recording processing input prefix is unavailable")
        digest = self._new_input_digest()
        for position, segment in enumerate(segments):
            actual = self._segment_input(position, segment, digest)
        if actual != expected:
            raise RealtimeRingError("recording processing input prefix changed")

    @property
    def progress(self) -> RecordingReadProgress:
        with self._lock, self._recorder._condition:
            watermark = self._recorder._manifest_committed_frame_id
            return RecordingReadProgress(
                session_id=self._session_id,
                source_epoch_id=self._source_epoch_id,
                last_delivered_frame_id=(
                    None if self._last_delivered is None else self._last_delivered.capture_frame_id
                ),
                committed_frame_id=None if watermark < 0 else watermark,
                input_closed=self._recorder._closed_manifest is not None,
                cancelled=self._cancelled.is_set(),
                failure_reason=self._failure_reason,
            )

    def frames(self, stop: threading.Event):
        with self._lock:
            if self._started:
                raise RealtimeRingError("recording reader is single-use")
            self._started = True
        index = 0
        try:
            while not stop.is_set() and not self._cancelled.is_set():
                with self._lock:
                    if self._sealed and self._last_delivered == self._cutoff:
                        return
                segment = self._recorder.wait_for_segment(index)
                if segment is None:
                    with self._lock:
                        sealed = self._sealed
                    if not sealed and self._capture_failure is not None:
                        reason = self._capture_failure()
                        if reason is not None:
                            with self._lock:
                                # The callback may span the original admission
                                # fence. Later shared-preview faults do not end
                                # a disk input that already owns its cutoff.
                                if not self._sealed:
                                    self._terminal_capture_failure = reason
                                    raise CaptureContractError(reason)
                    if self.progress.input_closed:
                        with self._lock:
                            if self._sealed and self._last_delivered != self._cutoff:
                                raise RealtimeRingError("recording ended before capture cutoff")
                        return
                    continue
                frames = self._recorder.read_segment(segment)
                with self._lock:
                    self._current_segment_input = self._segment_input(
                        index,
                        segment,
                        self._input_digest,
                    )
                for frame in frames:
                    if stop.is_set() or self._cancelled.is_set():
                        return
                    previous = self._last_delivered
                    if frame.source_epoch_id != self._source_epoch_id or (
                        previous is not None
                        and (
                            frame.capture_frame_id != previous.capture_frame_id + 1
                            or frame.captured_at_ns < previous.captured_at_ns
                        )
                    ):
                        raise RealtimeRingError("recording reader epoch or frame order changed")
                    with self._lock:
                        if self._sealed and (
                            self._cutoff is None
                            or frame.capture_frame_id > self._cutoff.capture_frame_id
                        ):
                            raise RealtimeRingError("recording reader passed capture cutoff")
                        self._last_delivered = frame
                    yield frame
                index += 1
                del frames
        except Exception as exc:
            with self._lock:
                self._failure_reason = f"{type(exc).__name__}: {exc}"
            raise

    def seal(self, cutoff: CaptureFrame | None) -> None:
        """Set an immutable admission cutoff; never flush or close the writer."""
        with self._lock:
            if self._sealed:
                if self._cutoff != cutoff:
                    raise RealtimeRingError("recording reader cutoff changed")
                return
            if cutoff is not None and cutoff.source_epoch_id != self._source_epoch_id:
                raise RealtimeRingError("recording reader cutoff epoch changed")
            if self._last_delivered is not None and (
                cutoff is None or self._last_delivered.capture_frame_id > cutoff.capture_frame_id
            ):
                raise RealtimeRingError("recording reader cutoff precedes delivered frame")
            self._cutoff = cutoff
            self._sealed = True

    def cancel(self) -> None:
        self._cancelled.set()


class SessionRecordingEvidenceStore:
    """Expose continuous session recording through the shot-evidence interface."""

    def __init__(
        self,
        *,
        recorder: ContinuousSessionRecorder,
        session_id: str,
        pre_roll_frames: int,
        post_roll_frames: int,
        maximum_shot_frames: int,
        append_frames: bool = True,
        close_recorder: bool = True,
        commit_timeout_seconds: float = 5.0,
    ) -> None:
        if (
            not session_id.strip()
            or pre_roll_frames < 0
            or post_roll_frames < 0
            or maximum_shot_frames <= 0
            or not math.isfinite(commit_timeout_seconds)
            or commit_timeout_seconds <= 0
        ):
            raise ValueError("session recording evidence configuration is invalid")
        self._recorder = recorder
        self._session_id = session_id
        self._pre_roll_frames = pre_roll_frames
        self._post_roll_frames = post_roll_frames
        self._maximum_shot_frames = maximum_shot_frames
        self._append_frames = append_frames
        self._close_recorder = close_recorder
        self._commit_timeout_seconds = commit_timeout_seconds
        self._last_frame: CaptureFrame | None = None
        self._shot_start_frame_id: int | None = None
        self._shot_settle_frame_id: int | None = None
        self.maximum_queue_depth = 0
        self._closed = False

    @property
    def last_admitted_frame(self) -> CaptureFrame | None:
        return self._last_frame

    def commit_recording_tail(self, *, timeout_seconds: float) -> None:
        self._recorder.commit_pending(timeout_seconds=timeout_seconds)

    @property
    def recording_manifest_path(self):
        return self._recorder._root / "session-recording.json"

    @property
    def first_recorded_frame_id(self):
        return self._recorder.first_frame_id

    def close_recording_input(self, *, timeout_seconds):
        # Keep the evidence adapter open for late disk-based detection/windows.
        return self._recorder.close(timeout_seconds=timeout_seconds)

    def finish_capture_input(self) -> None:
        """Capture has stopped; persist its partial tail independently of detection."""
        self._recorder.commit_pending(timeout_seconds=self._commit_timeout_seconds)

    def committed_source(
        self,
        *,
        source_epoch_id: str,
        capture_failure: Callable[[], str | None] | None = None,
    ) -> CommittedRecordingSource:
        return CommittedRecordingSource(
            recorder=self._recorder,
            session_id=self._session_id,
            source_epoch_id=source_epoch_id,
            capture_failure=capture_failure,
        )

    @property
    def writer_health(self) -> RecordingHealth:
        """Expose writer pressure without changing the shot-ring health contract."""
        return self._recorder.health

    @property
    def pending_command_count(self) -> int:
        return self._recorder.pending_command_count

    @property
    def health(self) -> RealtimeRingHealth:
        health = self._recorder.health
        state = {
            RecordingHealthState.RUNNING: RealtimeRingState.RUNNING,
            RecordingHealthState.DEGRADED: RealtimeRingState.DEGRADED,
            RecordingHealthState.CLOSED: RealtimeRingState.CLOSED,
        }[health.state]
        return RealtimeRingHealth(
            state=state,
            stored_segments=health.committed_segments,
            stored_frames=health.committed_frames + health.buffered_frames,
            stored_payload_bytes=0,
            reason=health.reason,
        )

    def append(self, frame: CaptureFrame) -> None:
        self._require_open()
        if self._last_frame is not None:
            expected_frame_id = self._last_frame.capture_frame_id + 1
            frame_id_is_valid = (
                frame.capture_frame_id == expected_frame_id
                if self._append_frames
                else frame.capture_frame_id >= expected_frame_id
            )
            if (
                frame.source_epoch_id != self._last_frame.source_epoch_id
                or frame.width != self._last_frame.width
                or frame.height != self._last_frame.height
                or not frame_id_is_valid
                or frame.captured_at_ns < self._last_frame.captured_at_ns
            ):
                raise RealtimeRingError("session evidence frames are not contiguous")
        if self._append_frames:
            self._recorder.append(frame)
        self._last_frame = frame
        recorder_health = self._recorder.health
        self.maximum_queue_depth = max(
            self.maximum_queue_depth,
            recorder_health.pending_chunks,
        )
        if (
            self._shot_start_frame_id is not None
            and self._shot_settle_frame_id is None
            and frame.capture_frame_id - self._shot_start_frame_id + 1 > self._maximum_shot_frames
        ):
            raise RealtimeRingError("session evidence maximum shot duration exceeded")

    def begin_shot(self, frame_id: int) -> None:
        self._require_available_frame(frame_id)
        if self._shot_start_frame_id is not None:
            raise RealtimeRingError("session evidence already has an active shot")
        self._shot_start_frame_id = frame_id
        self._shot_settle_frame_id = None

    def settle_shot(self, frame_id: int) -> None:
        if self._shot_start_frame_id is None:
            raise RealtimeRingError("session evidence has no active shot")
        self._require_available_frame(frame_id)
        if frame_id < self._shot_start_frame_id:
            raise RealtimeRingError("session evidence settle frame precedes shot start")
        if frame_id - self._shot_start_frame_id + 1 > self._maximum_shot_frames:
            raise RealtimeRingError("session evidence maximum shot duration exceeded")
        self._shot_settle_frame_id = frame_id

    def freeze_shot(self) -> RecordedShotInterval:
        self._require_open()
        if self._shot_start_frame_id is None or self._shot_settle_frame_id is None:
            raise RealtimeRingError("session evidence shot is not settled")
        end_frame_id = self._shot_settle_frame_id + self._post_roll_frames
        self._require_available_frame(end_frame_id)
        self._recorder.commit_pending(timeout_seconds=self._commit_timeout_seconds)
        assert self._last_frame is not None
        first_frame_id = self._recorder.first_frame_id
        if first_frame_id is None:
            raise RealtimeRingError("session evidence has no recorded frames")
        return self._recorder.reference_interval(
            session_id=self._session_id,
            shot_id=(f"shot-{self._shot_start_frame_id:020d}-{self._shot_settle_frame_id:020d}"),
            source_epoch_id=self._last_frame.source_epoch_id,
            start_frame_id=max(
                first_frame_id,
                self._shot_start_frame_id - self._pre_roll_frames,
            ),
            end_frame_id=end_frame_id,
        )

    def release_shot(self) -> None:
        self._shot_start_frame_id = None
        self._shot_settle_frame_id = None

    def freeze_window(
        self,
        *,
        shot_id: str,
        source_epoch_id: str,
        start_frame_id: int,
        end_frame_id: int,
        timeout_seconds: float = 5.0,
    ) -> RecordedShotInterval:
        """Reference one explicit window without sharing the single active-shot slot."""
        self._require_open()
        self._recorder.commit_through(end_frame_id, timeout_seconds=timeout_seconds)
        first = self._recorder.first_frame_id
        if first is None:
            raise RealtimeRingError("session evidence has no recorded frames")
        return self._recorder.reference_interval(
            session_id=self._session_id,
            shot_id=shot_id,
            source_epoch_id=source_epoch_id,
            start_frame_id=max(first, start_frame_id - self._pre_roll_frames),
            end_frame_id=end_frame_id,
        )

    def close(self, *, timeout_seconds=None) -> None:
        if self._closed:
            return
        if self._close_recorder:
            self._recorder.close(timeout_seconds=timeout_seconds)
        self._closed = True

    @property
    def finalized_manifest_path(self):
        return self._recorder.finalized_manifest_path

    @property
    def finalized_segment_receipts(self):
        return self._recorder.finalized_segment_receipts

    def _require_open(self) -> None:
        if self._closed:
            raise RealtimeRingError("session evidence store is closed")

    def _require_available_frame(self, frame_id: int) -> None:
        self._require_open()
        if frame_id < 0 or self._last_frame is None or frame_id > self._last_frame.capture_frame_id:
            raise RealtimeRingError("session evidence frame is not available")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RealtimeRingError("session recording segment is unavailable") from exc
    return digest.hexdigest()
