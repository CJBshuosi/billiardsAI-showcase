#!/usr/bin/env python3
"""Validate the native capture helper against the scoped RER camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO

_READY = struct.Struct(">I")
_FRAME = struct.Struct(">8sQQIII32s")
_LOCATOR = r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000"
_FINGERPRINT = "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93"


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise RuntimeError("native capture stream ended early")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


class StreamValidator:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        expected_locator: str,
        expected_fingerprint: str,
    ) -> None:
        self._stream = stream
        self._locator = expected_locator
        self._fingerprint = expected_fingerprint
        self._last_frame = -1
        self._last_timestamp = -1

    def read_readiness(self) -> dict[str, object]:
        (length,) = _READY.unpack(_read_exact(self._stream, _READY.size))
        if length <= 0 or length > 65_536:
            raise RuntimeError("invalid readiness length")
        payload = json.loads(_read_exact(self._stream, length).decode("utf-8"))
        expected = {
            "schema_version": "1.0",
            "kind": "mainline1_native_capture_ready",
            "device_locator": self._locator,
            "durable_fingerprint": self._fingerprint,
            "width": 1920,
            "height": 1080,
            "fps_numerator": 60,
            "fps_denominator": 1,
            "pixel_format": "MJPG",
            "controls_locked": True,
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("native readiness identity, mode, or controls did not match")
        handle = payload.get("native_handle_ref")
        if not isinstance(handle, str) or not handle.startswith("dshow://"):
            raise RuntimeError("native capture handle is not an exact DirectShow handle")
        return payload

    def read_frame(self) -> tuple[int, int, bytes]:
        values = _FRAME.unpack(_read_exact(self._stream, _FRAME.size))
        magic, frame_id, timestamp_ns, width, height, size, digest = values
        if magic != b"BAUVC001" or (width, height) != (1920, 1080):
            raise RuntimeError("native frame header changed")
        if size <= 0 or size > 32 * 1024 * 1024:
            raise RuntimeError("native frame size is invalid")
        payload = _read_exact(self._stream, size)
        if hashlib.sha256(payload).digest() != digest:
            raise RuntimeError("native frame hash mismatch")
        if frame_id != self._last_frame + 1:
            raise RuntimeError("native frame IDs are not contiguous")
        if timestamp_ns < self._last_timestamp:
            raise RuntimeError("native frame timestamp reversed")
        if not payload.startswith(b"\xff\xd8"):
            raise RuntimeError("native frame is not MJPEG")
        self._last_frame = frame_id
        self._last_timestamp = timestamp_ns
        return frame_id, timestamp_ns, payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.helper),
        "--device-locator",
        _LOCATOR,
        "--fingerprint",
        _FINGERPRINT,
        "--width",
        "1920",
        "--height",
        "1080",
        "--fps",
        "60",
        "--pixel-format",
        "MJPG",
    ]
    for control in (
        "exposure=-8",
        "gain=140",
        "auto_white_balance=0",
        "white_balance_temperature=4650",
        "focus=100",
        "auto_focus=2",
        "zoom=100",
    ):
        command.extend(("--control", control))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if process.stdout is None or process.stdin is None or process.stderr is None:
        raise RuntimeError("native helper pipes are unavailable")
    outcomes: queue.Queue[object] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            validator = StreamValidator(
                process.stdout,
                expected_locator=_LOCATOR,
                expected_fingerprint=_FINGERPRINT,
            )
            ready = validator.read_readiness()
            frames = [validator.read_frame() for _ in range(args.frames)]
            outcomes.put((ready, frames))
        except BaseException as exc:
            outcomes.put(exc)

    started = time.monotonic()
    thread = threading.Thread(target=read, name="native-smoke-reader", daemon=True)
    thread.start()
    status = "FAIL"
    reason: str | None = None
    report: dict[str, object] = {}
    try:
        try:
            outcome = outcomes.get(timeout=15.0)
        except queue.Empty as exc:
            raise RuntimeError(
                "native capture did not deliver 120 frames within 15 seconds"
            ) from exc
        if isinstance(outcome, BaseException):
            raise RuntimeError(f"native capture reader failed: {outcome}") from outcome
        readiness, frames = outcome
        first_id, first_ns, first_payload = frames[0]
        last_id, last_ns, last_payload = frames[-1]
        elapsed_ns = last_ns - first_ns
        measured_fps = (len(frames) - 1) * 1_000_000_000 / elapsed_ns if elapsed_ns > 0 else 0.0
        if measured_fps < 55.0 or measured_fps > 65.0:
            raise RuntimeError(f"native capture cadence was {measured_fps:.3f} FPS")
        (args.output / "first.jpg").write_bytes(first_payload)
        (args.output / "last.jpg").write_bytes(last_payload)
        process.stdin.write(b"STOP\n")
        process.stdin.flush()
        process.stdin.close()
        cancel_started = time.monotonic()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise RuntimeError("native helper did not cancel within 500 ms") from exc
        cancel_ms = (time.monotonic() - cancel_started) * 1000
        if process.returncode != 0:
            raise RuntimeError(f"native helper exited with {process.returncode}")
        status = "PASS"
        report = {
            "readiness": readiness,
            "delivered_frames": len(frames),
            "first_frame_id": first_id,
            "last_frame_id": last_id,
            "measured_fps": measured_fps,
            "cancel_ms": cancel_ms,
            "first_jpeg_sha256": hashlib.sha256(first_payload).hexdigest(),
            "last_jpeg_sha256": hashlib.sha256(last_payload).hexdigest(),
        }
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    (args.output / "native.stderr.log").write_text(stderr, encoding="utf-8")
    report.update(
        {
            "schema_version": "1.0",
            "kind": "raytheon_csharp_capture_live",
            "status": status,
            "reason": reason,
            "wall_seconds": time.monotonic() - started,
            "production_ready": False,
            "service_or_production_data_mutated": False,
        }
    )
    (args.output / "live.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
