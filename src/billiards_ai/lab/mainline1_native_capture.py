"""Strict Python boundary for the Windows Media Foundation capture helper."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureFrame,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)

_FRAME_HEADER = struct.Struct(">8sQQIII32s")
_READY_LENGTH = struct.Struct(">I")
_MAGIC = b"BAUVC001"


class NativeCaptureProtocolError(RuntimeError):
    """The native helper failed identity, mode, framing or timing validation."""


def _enqueue_capture_item(
    items: queue.Queue[Any],
    item: Any,
    cancel_requested: threading.Event,
) -> bool:
    """Offer an item without letting a full consumer queue block cancellation."""
    while not cancel_requested.is_set():
        try:
            items.put(item, timeout=0.01)
            return True
        except queue.Full:
            continue
    return False


def _read_exact(
    stream: BinaryIO,
    byte_count: int,
    *,
    allow_clean_eof: bool = False,
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_clean_eof and not chunks:
                return None
            raise NativeCaptureProtocolError("native capture stream is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NativeCaptureStreamParser:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        expected_device_locator: str,
        expected_fingerprint: str,
        capture_config: RealtimeCaptureConfig,
        source_epoch_id: str,
    ) -> None:
        self._stream = stream
        self._expected_locator = expected_device_locator
        self._expected_fingerprint = expected_fingerprint
        self._config = capture_config
        self._epoch = source_epoch_id
        self._ready = False
        self._last_frame_id: int | None = None
        self._last_timestamp_ns: int | None = None

    def read_readiness(self) -> NativeCaptureIdentity:
        if self._ready:
            raise NativeCaptureProtocolError("native capture readiness was repeated")
        length_bytes = _read_exact(self._stream, _READY_LENGTH.size)
        assert length_bytes is not None
        (length,) = _READY_LENGTH.unpack(length_bytes)
        if length <= 0 or length > 64 * 1024:
            raise NativeCaptureProtocolError("native capture readiness length is invalid")
        encoded = _read_exact(self._stream, length)
        assert encoded is not None
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise NativeCaptureProtocolError("native capture readiness JSON is invalid") from exc
        if not isinstance(payload, dict):
            raise NativeCaptureProtocolError("native capture readiness JSON is invalid")
        if (
            payload.get("schema_version") != "1.0"
            or payload.get("kind") != "mainline1_native_capture_ready"
        ):
            raise NativeCaptureProtocolError("native capture readiness schema is invalid")
        if payload.get("device_locator") != self._expected_locator:
            raise NativeCaptureProtocolError("native capture device locator does not match PnP")
        if payload.get("durable_fingerprint") != self._expected_fingerprint:
            raise NativeCaptureProtocolError("native capture fingerprint changed")
        try:
            actual_fps = int(payload["fps_numerator"]) / int(payload["fps_denominator"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise NativeCaptureProtocolError("native capture mode is invalid") from exc
        if (
            payload.get("width") != self._config.width
            or payload.get("height") != self._config.height
            or payload.get("pixel_format") != self._config.pixel_format
            or abs(actual_fps - self._config.fps) > 0.01
        ):
            raise NativeCaptureProtocolError("native capture mode does not match request")
        if payload.get("controls_locked") is not True:
            raise NativeCaptureProtocolError("native capture controls were not locked")
        try:
            identity = NativeCaptureIdentity(
                device_locator=str(payload["device_locator"]),
                native_handle_ref=str(payload["native_handle_ref"]),
                durable_fingerprint=str(payload["durable_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeCaptureProtocolError("native capture identity is invalid") from exc
        self._ready = True
        return identity

    def read_frame(self) -> CaptureFrame | None:
        if not self._ready:
            raise NativeCaptureProtocolError("native capture readiness was not validated")
        header = _read_exact(
            self._stream,
            _FRAME_HEADER.size,
            allow_clean_eof=True,
        )
        if header is None:
            return None
        magic, frame_id, timestamp_ns, width, height, payload_length, expected_hash = (
            _FRAME_HEADER.unpack(header)
        )
        if magic != _MAGIC:
            raise NativeCaptureProtocolError("native capture frame magic is invalid")
        if (width, height) != (self._config.width, self._config.height):
            raise NativeCaptureProtocolError("native capture frame geometry changed")
        if payload_length <= 0 or payload_length > 32 * 1024 * 1024:
            raise NativeCaptureProtocolError("native capture frame size is invalid")
        payload = _read_exact(self._stream, payload_length)
        assert payload is not None
        if hashlib.sha256(payload).digest() != expected_hash:
            raise NativeCaptureProtocolError("native capture frame SHA-256 mismatch")
        expected_frame_id = 0 if self._last_frame_id is None else self._last_frame_id + 1
        if frame_id != expected_frame_id:
            raise NativeCaptureProtocolError("native capture frame ID is not contiguous")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise NativeCaptureProtocolError("native capture timestamp reversed")
        self._last_frame_id = frame_id
        self._last_timestamp_ns = timestamp_ns
        return CaptureFrame(
            source_epoch_id=self._epoch,
            capture_frame_id=frame_id,
            captured_at_ns=timestamp_ns,
            width=width,
            height=height,
            mjpeg=payload,
        )


class NativeCaptureSubprocessBackend:
    """Run one trusted helper and turn its stdout protocol into capture frames."""

    def __init__(
        self,
        *,
        helper: Path,
        device_locator: str,
        expected_fingerprint: str,
        capture_config: RealtimeCaptureConfig,
        source_epoch_id: str,
        image_controls: tuple[tuple[str, float], ...] = (),
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        resolved = helper.resolve()
        if helper.is_symlink() or not resolved.is_file():
            raise ValueError("native capture helper is missing or unsafe")
        self._helper = resolved
        self._device_locator = device_locator
        self._expected_fingerprint = expected_fingerprint
        self._config = capture_config
        self._epoch = source_epoch_id
        self._image_controls = image_controls
        self._process_factory = process_factory
        self._process: Any | None = None
        self._identity: NativeCaptureIdentity | None = None
        self._items: queue.Queue[CaptureFrame | BaseException | None] = queue.Queue(maxsize=64)
        self._reader: threading.Thread | None = None
        self._cancel_requested = threading.Event()
        self._startup_failed = False

    @property
    def identity(self) -> NativeCaptureIdentity:
        if self._identity is None:
            raise NativeCaptureProtocolError("native capture helper is not ready")
        return self._identity

    @property
    def pending_frame_count(self) -> int:
        return self._items.qsize()

    def start(self) -> NativeCaptureIdentity:
        self._start()
        return self.identity

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
        self._start()
        while not stop.is_set():
            try:
                item = self._items.get(timeout=self._config.cancel_timeout_seconds)
            except queue.Empty as exc:
                raise NativeCaptureProtocolError("native camera read exceeded timeout") from exc
            if item is None:
                return
            if isinstance(item, BaseException):
                raise NativeCaptureProtocolError(
                    f"native capture reader failed: {type(item).__name__}: {item}"
                ) from item
            yield item

    def cancel(self) -> None:
        process = self._process
        if process is None:
            return
        self._cancel_requested.set()
        if process.poll() is not None:
            return
        deadline = time.monotonic() + self._config.cancel_timeout_seconds
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(b"STOP\n")
                process.stdin.flush()
                process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        graceful_timeout = self._config.cancel_timeout_seconds * 0.8
        try:
            process.wait(timeout=graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            process.terminate()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeCaptureProtocolError("native capture helper did not cancel within timeout")
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise NativeCaptureProtocolError(
                "native capture helper did not cancel within timeout"
            ) from exc

    def _start(self) -> None:
        if self._startup_failed:
            raise NativeCaptureProtocolError("native capture startup previously failed")
        try:
            self._start_owned()
        except BaseException:
            self._startup_failed = True
            self._cleanup_failed_start()
            raise

    def _cleanup_failed_start(self) -> None:
        """Keep ownership until an unsuccessfully started helper is reaped."""
        self._cancel_requested.set()
        process = self._process
        if process is None:
            return
        try:
            try:
                self.cancel()
            except (OSError, NativeCaptureProtocolError):
                # cancel may have escalated to kill at its deadline. Reap it
                # below before releasing the blocked stdout reader.
                pass
            try:
                process.wait(timeout=self._config.cancel_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._config.cancel_timeout_seconds)
        finally:
            reader = self._reader
            if reader is not None:
                reader.join(timeout=self._config.cancel_timeout_seconds)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            if reader is not None and reader.is_alive():
                reader.join(timeout=self._config.cancel_timeout_seconds)
                if reader.is_alive():
                    raise NativeCaptureProtocolError("native capture startup reader did not stop")

    def _start_owned(self) -> None:
        if self._process is not None:
            return
        command = [
            str(self._helper),
            "--device-locator",
            self._device_locator,
            "--width",
            str(self._config.width),
            "--height",
            str(self._config.height),
            "--fps",
            str(round(self._config.fps)),
            "--pixel-format",
            self._config.pixel_format,
            "--fingerprint",
            self._expected_fingerprint,
        ]
        for name, value in self._image_controls:
            if not name or not math.isfinite(value):
                raise ValueError("native capture image control is invalid")
            command.extend(("--control", f"{name}={round(value)}"))
        self._process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdout is None:
            raise NativeCaptureProtocolError("native capture stdout is unavailable")
        parser = NativeCaptureStreamParser(
            self._process.stdout,
            expected_device_locator=self._device_locator,
            expected_fingerprint=self._expected_fingerprint,
            capture_config=self._config,
            source_epoch_id=self._epoch,
        )
        readiness: queue.Queue[NativeCaptureIdentity | BaseException] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                identity = parser.read_readiness()
                readiness.put(identity)
                while True:
                    frame = parser.read_frame()
                    if frame is None:
                        if not self._cancel_requested.is_set():
                            _enqueue_capture_item(
                                self._items,
                                NativeCaptureProtocolError(
                                    "native capture helper ended unexpectedly"
                                ),
                                self._cancel_requested,
                            )
                        return
                    _enqueue_capture_item(
                        self._items,
                        frame,
                        self._cancel_requested,
                    )
            except BaseException as exc:
                if readiness.empty():
                    readiness.put(exc)
                elif not self._cancel_requested.is_set():
                    _enqueue_capture_item(
                        self._items,
                        exc,
                        self._cancel_requested,
                    )

        self._reader = threading.Thread(target=read, name="native-uvc-reader", daemon=True)
        self._reader.start()
        try:
            ready = readiness.get(timeout=self._config.startup_timeout_seconds)
        except queue.Empty as exc:
            raise NativeCaptureProtocolError("native capture readiness timed out") from exc
        if isinstance(ready, BaseException):
            raise NativeCaptureProtocolError("native capture readiness failed") from ready
        self._identity = ready
