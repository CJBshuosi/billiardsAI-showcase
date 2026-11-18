"""Bounded atomic MJPEG segment ring for product realtime evidence."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import stat
import struct
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame

_MAGIC = b"BAMJPG01"
_RECORD = struct.Struct(">QQI")


class RealtimeRingError(RuntimeError):
    """The rolling source evidence is incomplete, corrupt or exhausted."""


class RealtimeRingState(StrEnum):
    RUNNING = "running"
    DEGRADED = "degraded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RealtimeRingHealth:
    state: RealtimeRingState
    stored_segments: int
    stored_frames: int
    stored_payload_bytes: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Segment:
    name: str
    first_frame_id: int
    last_frame_id: int
    first_captured_at_ns: int
    last_captured_at_ns: int
    frame_count: int
    payload_bytes: int
    sha256: str

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


@dataclass(frozen=True, slots=True)
class FrozenCaptureInterval:
    root: Path
    source_epoch_id: str
    width: int
    height: int
    start_frame_id: int
    end_frame_id: int
    frame_count: int
    segment_sha256: dict[str, str]

    @property
    def media_sha256(self) -> str:
        """Return a stable content identity for this exact source interval."""

        payload = {
            "schema_version": "1.0",
            "source_epoch_id": self.source_epoch_id,
            "width": self.width,
            "height": self.height,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "frame_count": self.frame_count,
            "segments": [
                {"name": name, "sha256": sha256} for name, sha256 in self.segment_sha256.items()
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def frames(self, *, before_segment: Callable[[], None] | None = None) -> Iterator[CaptureFrame]:
        expected_frame_id = self.start_frame_id
        for name, expected_sha256 in self.segment_sha256.items():
            if before_segment is not None:
                before_segment()
            path = self.root / name
            for frame_id, captured_at_ns, payload in _read_verified_segment(
                path, expected_sha256, continue_check=before_segment
            ):
                if self.start_frame_id <= frame_id <= self.end_frame_id:
                    if frame_id != expected_frame_id:
                        raise RealtimeRingError("frozen capture interval is not contiguous")
                    expected_frame_id += 1
                    yield CaptureFrame(
                        source_epoch_id=self.source_epoch_id,
                        capture_frame_id=frame_id,
                        captured_at_ns=captured_at_ns,
                        width=self.width,
                        height=self.height,
                        mjpeg=payload,
                    )
        if expected_frame_id != self.end_frame_id + 1:
            raise RealtimeRingError("frozen capture interval is not contiguous")


class MjpegSegmentRing:
    """Persist source MJPEG in atomic one-second segments."""

    def __init__(
        self,
        root: Path,
        *,
        expected_fps: int,
        pre_roll_frames: int,
        post_roll_frames: int,
        maximum_shot_frames: int,
        maximum_bytes: int,
        persist_idle_segments: bool = True,
        idle_onset_lookback_frames: int = 0,
    ) -> None:
        if (
            expected_fps <= 0
            or pre_roll_frames < 0
            or post_roll_frames < 0
            or maximum_shot_frames <= 0
            or maximum_bytes <= 0
            or type(idle_onset_lookback_frames) is not int
            or idle_onset_lookback_frames < 0
        ):
            raise ValueError("realtime ring configuration is invalid")
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir() or self._root.is_symlink():
            raise RealtimeRingError("capture ring root must be a real directory")
        self._expected_fps = expected_fps
        self._pre_roll_frames = pre_roll_frames
        self._idle_onset_lookback_frames = idle_onset_lookback_frames
        self._post_roll_frames = post_roll_frames
        self._maximum_shot_frames = maximum_shot_frames
        self._maximum_bytes = maximum_bytes
        self._persist_idle_segments = persist_idle_segments
        self._state = RealtimeRingState.RUNNING
        self._reason: str | None = None
        self._source_epoch_id: str | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._last_frame_id: int | None = None
        self._last_timestamp_ns: int | None = None
        self._segments: list[_Segment] = []
        self._pending: list[CaptureFrame] = []
        self._stored_payload_bytes = 0
        self._pending_payload_bytes = 0
        self._shot_start_frame_id: int | None = None
        self._shot_settle_frame_id: int | None = None
        self._write_index()

    @property
    def health(self) -> RealtimeRingHealth:
        return RealtimeRingHealth(
            state=self._state,
            stored_segments=len(self._segments),
            stored_frames=sum(segment.frame_count for segment in self._segments)
            + len(self._pending),
            stored_payload_bytes=self._stored_payload_bytes + self._pending_payload_bytes,
            reason=self._reason,
        )

    def append(self, frame: CaptureFrame) -> None:
        self._require_running()
        if self._source_epoch_id is None:
            self._source_epoch_id = frame.source_epoch_id
            self._width = frame.width
            self._height = frame.height
        if (
            frame.source_epoch_id != self._source_epoch_id
            or frame.width != self._width
            or frame.height != self._height
        ):
            self._degrade("capture ring source epoch or geometry changed")
        if self._last_frame_id is not None and frame.capture_frame_id != self._last_frame_id + 1:
            self._degrade("capture ring frames must be contiguous")
        if self._last_timestamp_ns is not None and frame.captured_at_ns < self._last_timestamp_ns:
            self._degrade("capture ring timestamps must be monotonic")
        projected = self._stored_payload_bytes + self._pending_payload_bytes + len(frame.mjpeg)
        if projected > self._maximum_bytes:
            if self._shot_start_frame_id is not None:
                self._degrade("capture ring byte capacity exhausted during shot")
            self._flush_pending()
            self._retire_idle_segments(frame.capture_frame_id)
            projected = self._stored_payload_bytes + len(frame.mjpeg)
            if projected > self._maximum_bytes:
                self._degrade("capture ring byte capacity exhausted")
        if (
            self._shot_start_frame_id is not None
            and self._shot_settle_frame_id is None
            and frame.capture_frame_id - self._shot_start_frame_id + 1 > self._maximum_shot_frames
        ):
            self._degrade("capture ring maximum shot duration exceeded")
        self._pending.append(frame)
        self._pending_payload_bytes += len(frame.mjpeg)
        self._last_frame_id = frame.capture_frame_id
        self._last_timestamp_ns = frame.captured_at_ns
        if self._shot_start_frame_id is None and not self._persist_idle_segments:
            retain = self._pre_roll_frames + self._idle_onset_lookback_frames + 1
            while len(self._pending) > retain:
                retired = self._pending.pop(0)
                self._pending_payload_bytes -= len(retired.mjpeg)
        elif len(self._pending) >= self._expected_fps:
            self._flush_pending()
            if self._shot_start_frame_id is None:
                self._retire_idle_segments(frame.capture_frame_id)

    def begin_shot(self, frame_id: int) -> None:
        self._require_running()
        if self._last_frame_id is None or frame_id > self._last_frame_id or frame_id < 0:
            raise RealtimeRingError("shot start frame is not available")
        if self._shot_start_frame_id is not None:
            raise RealtimeRingError("capture ring already has an active shot")
        self._shot_start_frame_id = frame_id
        self._shot_settle_frame_id = None

    def settle_shot(self, frame_id: int) -> None:
        self._require_running()
        if self._shot_start_frame_id is None:
            raise RealtimeRingError("capture ring has no active shot")
        if (
            self._last_frame_id is None
            or frame_id < self._shot_start_frame_id
            or frame_id > self._last_frame_id
        ):
            raise RealtimeRingError("shot settle frame is not available")
        if frame_id - self._shot_start_frame_id + 1 > self._maximum_shot_frames:
            self._degrade("capture ring maximum shot duration exceeded")
        self._shot_settle_frame_id = frame_id

    def freeze_shot(self) -> FrozenCaptureInterval:
        self._require_running()
        if self._shot_start_frame_id is None or self._shot_settle_frame_id is None:
            raise RealtimeRingError("capture ring shot is not settled")
        end_frame_id = self._shot_settle_frame_id + self._post_roll_frames
        if self._last_frame_id is None or self._last_frame_id < end_frame_id:
            raise RealtimeRingError("capture ring post-roll is incomplete")
        start_frame_id = max(0, self._shot_start_frame_id - self._pre_roll_frames)
        self._flush_pending()
        selected = [
            segment
            for segment in self._segments
            if segment.last_frame_id >= start_frame_id and segment.first_frame_id <= end_frame_id
        ]
        frozen = FrozenCaptureInterval(
            root=self._root,
            source_epoch_id=self._source_epoch_id or "",
            width=self._width or 0,
            height=self._height or 0,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            frame_count=end_frame_id - start_frame_id + 1,
            segment_sha256={segment.name: segment.sha256 for segment in selected},
        )
        tuple(frozen.frames())
        return frozen

    def close(self) -> None:
        if self._state is RealtimeRingState.CLOSED:
            return
        if self._state is RealtimeRingState.RUNNING:
            if self._persist_idle_segments:
                self._flush_pending()
                if self._shot_start_frame_id is None and self._last_frame_id is not None:
                    self._retire_idle_segments(self._last_frame_id)
            else:
                # The commercial field path keeps pre-roll in memory and only
                # materializes a completed shot. A restart must not preserve an
                # idle or incomplete interval as if it were valid evidence.
                self._pending = []
                self._pending_payload_bytes = 0
                for segment in self._segments:
                    (self._root / segment.name).unlink(missing_ok=True)
                self._segments = []
                self._stored_payload_bytes = 0
            self._state = RealtimeRingState.CLOSED
        self._write_index()

    def release_shot(self) -> None:
        """Release a frozen shot only after its consumer has copied or published it."""

        self._require_running()
        if self._shot_start_frame_id is None:
            raise RealtimeRingError("capture ring has no active shot")
        self._shot_start_frame_id = None
        self._shot_settle_frame_id = None
        if self._last_frame_id is not None:
            self._retire_idle_segments(self._last_frame_id)

    def _require_running(self) -> None:
        if self._state is RealtimeRingState.DEGRADED:
            raise RealtimeRingError("capture ring is degraded")
        if self._state is RealtimeRingState.CLOSED:
            raise RealtimeRingError("capture ring is closed")

    def _degrade(self, reason: str) -> None:
        self._state = RealtimeRingState.DEGRADED
        self._reason = reason
        self._write_index()
        raise RealtimeRingError(reason)

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        first = self._pending[0]
        last = self._pending[-1]
        name = f"{first.capture_frame_id:020d}-{last.capture_frame_id:020d}.mjpgseg"
        destination = self._root / name
        temporary = self._root / f".pending-{uuid4().hex}.mjpgseg"
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as stream:
                stream.write(_MAGIC)
                digest.update(_MAGIC)
                for frame in self._pending:
                    header = _RECORD.pack(
                        frame.capture_frame_id,
                        frame.captured_at_ns,
                        len(frame.mjpeg),
                    )
                    stream.write(header)
                    digest.update(header)
                    stream.write(frame.mjpeg)
                    digest.update(frame.mjpeg)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        segment = _Segment(
            name=name,
            first_frame_id=first.capture_frame_id,
            last_frame_id=last.capture_frame_id,
            first_captured_at_ns=first.captured_at_ns,
            last_captured_at_ns=last.captured_at_ns,
            frame_count=len(self._pending),
            payload_bytes=self._pending_payload_bytes,
            sha256=digest.hexdigest(),
        )
        self._segments.append(segment)
        self._stored_payload_bytes += self._pending_payload_bytes
        self._pending = []
        self._pending_payload_bytes = 0
        self._write_index()

    def _retire_idle_segments(self, latest_frame_id: int) -> None:
        # An onset can be observed after capture has advanced. This explicit
        # budget admits that lookback while preserving the requested public
        # pre-roll interval; it does not bound analysis scheduling latency.
        extra = self._idle_onset_lookback_frames
        # Preserve legacy direct-call segment retention when no budget is
        # admitted. A positive budget also retains the onset frame itself.
        retained_frames = self._pre_roll_frames + (extra + 1 if extra else 0)
        retain_from = max(0, latest_frame_id - retained_frames + 1)
        retained: list[_Segment] = []
        changed = False
        for segment in self._segments:
            if segment.last_frame_id < retain_from:
                (self._root / segment.name).unlink(missing_ok=True)
                self._stored_payload_bytes -= segment.payload_bytes
                changed = True
            else:
                retained.append(segment)
        self._segments = retained
        if changed:
            self._write_index()

    def _write_index(self) -> None:
        payload = {
            "schema_version": "1.0",
            "state": self._state.value,
            "reason": self._reason,
            "source_epoch_id": self._source_epoch_id,
            "width": self._width,
            "height": self._height,
            "segments": [segment.row() for segment in self._segments],
        }
        temporary = self._root / f".pending-index-{uuid4().hex}.json"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._root / "index.json")
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class _AsyncRingCommand:
    operation: str
    argument: object | None = None
    completed: threading.Event | None = None
    result: list[object] | None = None


class AsyncMjpegSegmentRing:
    """Serialize ring writes off the capture/tracking coordinator thread."""

    def __init__(
        self,
        ring: MjpegSegmentRing,
        *,
        maximum_pending_frames: int = 180,
    ) -> None:
        if maximum_pending_frames <= 0:
            raise ValueError("async ring queue capacity must be positive")
        self._ring = ring
        self._commands: queue.Queue[_AsyncRingCommand] = queue.Queue(maxsize=maximum_pending_frames)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._health = ring.health
        self._failure: BaseException | None = None
        self._close_requested = threading.Event()
        self._close_command: _AsyncRingCommand | None = None
        self._close_enqueued = False
        self.maximum_queue_depth = 0
        self._thread = threading.Thread(
            target=self._run,
            name="mainline1-mjpeg-ring-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def health(self) -> RealtimeRingHealth:
        with self._lock:
            return self._health

    @property
    def pending_command_count(self) -> int:
        """Include the in-flight write as well as queued ring commands."""
        with self._commands.all_tasks_done:
            return self._commands.unfinished_tasks

    def append(self, frame: CaptureFrame) -> None:
        self._enqueue(_AsyncRingCommand("append", frame))

    def begin_shot(self, frame_id: int) -> None:
        self._enqueue(_AsyncRingCommand("begin_shot", frame_id))

    def settle_shot(self, frame_id: int) -> None:
        self._enqueue(_AsyncRingCommand("settle_shot", frame_id))

    def freeze_shot(self, *, timeout_seconds: float = 5.0) -> FrozenCaptureInterval:
        result = self._request("freeze_shot", timeout_seconds=timeout_seconds)
        if not isinstance(result, FrozenCaptureInterval):
            raise RealtimeRingError("async ring freeze returned an invalid interval")
        return result

    def release_shot(self) -> None:
        self._enqueue(_AsyncRingCommand("release_shot"))

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        self._close_requested.set()
        if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise RealtimeRingError("async ring close exceeded timeout")
        try:
            if self._close_command is None:
                self._close_command = _AsyncRingCommand(
                    "close",
                    completed=threading.Event(),
                    result=[],
                )
            command = self._close_command
        finally:
            self._lock.release()
        if not self._close_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise RealtimeRingError("async ring close exceeded timeout")
        try:
            if not self._close_enqueued:
                # All admissions before revocation finished under _lock. Never
                # hold that state lock while waiting for the bounded queue.
                try:
                    self._commands.put(command, timeout=max(0.0, deadline - time.monotonic()))
                except queue.Full as exc:
                    raise RealtimeRingError("async ring writer queue capacity exhausted") from exc
                self._close_enqueued = True
            assert command.completed is not None
            if not command.completed.wait(max(0.0, deadline - time.monotonic())):
                raise RealtimeRingError("async ring close exceeded timeout")
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise RealtimeRingError("async ring writer did not stop within timeout")
            if command.result and isinstance(command.result[0], BaseException):
                error = command.result[0]
                raise RealtimeRingError(
                    f"async ring close failed: {type(error).__name__}: {error}"
                ) from error
            # The worker has joined; no producer can mutate its failure now.
            failure = self._failure
            if failure is not None:
                raise RealtimeRingError(
                    f"async ring writer failed: {type(failure).__name__}: {failure}"
                ) from failure
        finally:
            self._close_lock.release()

    def _enqueue(self, command: _AsyncRingCommand) -> None:
        with self._lock:
            failure = self._failure
            if failure is not None:
                raise RealtimeRingError(
                    f"async ring writer failed: {type(failure).__name__}: {failure}"
                ) from failure
            if self._close_requested.is_set():
                raise RealtimeRingError("async ring writer is closed")
            try:
                self._commands.put_nowait(command)
            except queue.Full as exc:
                raise RealtimeRingError("async ring writer queue capacity exhausted") from exc
            self.maximum_queue_depth = max(
                self.maximum_queue_depth,
                self._commands.qsize(),
            )

    def _request(
        self,
        operation: str,
        *,
        timeout_seconds: float,
    ) -> object | None:
        completed = threading.Event()
        result: list[object] = []
        started = time.monotonic()
        command = _AsyncRingCommand(operation, completed=completed, result=result)
        self._enqueue(command)
        remaining = max(0.0, timeout_seconds - (time.monotonic() - started))
        if not completed.wait(remaining):
            raise RealtimeRingError(f"async ring {operation} exceeded timeout")
        if result and isinstance(result[0], BaseException):
            error = result[0]
            raise RealtimeRingError(
                f"async ring {operation} failed: {type(error).__name__}: {error}"
            ) from error
        return None if not result else result[0]

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            try:
                if command.operation == "append":
                    assert isinstance(command.argument, CaptureFrame)
                    value: object | None = self._ring.append(command.argument)
                elif command.operation == "begin_shot":
                    assert isinstance(command.argument, int)
                    value = self._ring.begin_shot(command.argument)
                elif command.operation == "settle_shot":
                    assert isinstance(command.argument, int)
                    value = self._ring.settle_shot(command.argument)
                elif command.operation == "freeze_shot":
                    value = self._ring.freeze_shot()
                elif command.operation == "release_shot":
                    value = self._ring.release_shot()
                elif command.operation == "close":
                    value = self._ring.close()
                else:
                    raise RealtimeRingError("async ring operation is invalid")
                with self._lock:
                    self._health = self._ring.health
                if command.result is not None:
                    command.result.append(value)
            except BaseException as exc:
                with self._lock:
                    if self._failure is None:
                        self._failure = exc
                    self._health = self._ring.health
                if command.result is not None:
                    command.result.append(exc)
            finally:
                self._commands.task_done()
                if command.completed is not None:
                    command.completed.set()
            if command.operation == "close":
                return


def _read_verified_segment(
    path: Path, expected_sha256: str, *, continue_check: Callable[[], None] | None = None
) -> Iterator[tuple[int, int, bytes]]:
    descriptor = None
    try:
        original = path.lstat()
        if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
            raise RealtimeRingError("capture ring segment is unsafe")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not _same_file_identity(
                original, before, path_and_handle=True
            ):
                raise RealtimeRingError("capture ring segment identity changed")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                if continue_check is not None:
                    continue_check()
                digest.update(chunk)
            hashed = os.fstat(stream.fileno())
            if digest.hexdigest() != expected_sha256 or not _same_file_identity(before, hashed):
                raise RealtimeRingError("capture ring segment hash changed")
            stream.seek(0)
            yield from _read_segment_stream(stream)
            after = os.fstat(stream.fileno())
            current = path.lstat()
            if not _same_file_identity(before, after) or not _same_file_identity(
                after, current, path_and_handle=True
            ):
                raise RealtimeRingError("capture ring segment changed while reading")
    except OSError as exc:
        raise RealtimeRingError("capture ring segment cannot be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_segment_stream(stream) -> Iterator[tuple[int, int, bytes]]:
    if stream.read(len(_MAGIC)) != _MAGIC:
        raise RealtimeRingError("capture ring segment magic is invalid")
    while True:
        header = stream.read(_RECORD.size)
        if not header:
            return
        if len(header) != _RECORD.size:
            raise RealtimeRingError("capture ring segment is truncated")
        frame_id, captured_at_ns, payload_bytes = _RECORD.unpack(header)
        payload = stream.read(payload_bytes)
        if len(payload) != payload_bytes:
            raise RealtimeRingError("capture ring segment payload is truncated")
        yield frame_id, captured_at_ns, payload


def _same_file_identity(first, second, *, path_and_handle=False) -> bool:
    # CPython 3.12 on Windows exposes birth time in path stat's ctime, but
    # metadata change time in fd stat's ctime. Compare their shared birth clock
    # only at that boundary; fd-to-fd checks retain the change-time fence.
    clock = (
        "st_birthtime_ns"
        if path_and_handle
        and hasattr(first, "st_birthtime_ns")
        and hasattr(second, "st_birthtime_ns")
        else "st_ctime_ns"
    )
    if (first.st_size, first.st_mtime_ns, getattr(first, clock)) != (
        second.st_size,
        second.st_mtime_ns,
        getattr(second, clock),
    ):
        return False
    if first.st_ino and second.st_ino:
        return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
    return True


def _sha256(path: Path) -> str:
    """Retained private seam for write-path no-reread regression tests."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RealtimeRingError("capture ring segment cannot be hashed") from exc
    return digest.hexdigest()
