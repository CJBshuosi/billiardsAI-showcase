from __future__ import annotations

import hashlib
import io
import json
import queue
import re
import struct
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_native_capture import (
    NativeCaptureProtocolError,
    NativeCaptureStreamParser,
    NativeCaptureSubprocessBackend,
    _enqueue_capture_item,
)
from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity


def _readiness(**changes: object) -> bytes:
    payload = {
        "schema_version": "1.0",
        "kind": "mainline1_native_capture_ready",
        "device_locator": "pnp://rer-u3",
        "native_handle_ref": "mf://rer-u3-symbolic-link",
        "durable_fingerprint": "c" * 64,
        "width": 16,
        "height": 12,
        "fps_numerator": 60,
        "fps_denominator": 1,
        "pixel_format": "MJPG",
        "controls_locked": True,
    }
    payload.update(changes)
    encoded = json.dumps(payload, sort_keys=True).encode()
    return struct.pack(">I", len(encoded)) + encoded


def _record(frame_id: int, timestamp_ns: int, payload: bytes, **changes: object) -> bytes:
    values = {
        "magic": b"BAUVC001",
        "width": 16,
        "height": 12,
        "digest": hashlib.sha256(payload).digest(),
    }
    values.update(changes)
    return (
        struct.pack(
            ">8sQQIII32s",
            values["magic"],
            frame_id,
            timestamp_ns,
            values["width"],
            values["height"],
            len(payload),
            values["digest"],
        )
        + payload
    )


def _jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _parser(data: bytes) -> NativeCaptureStreamParser:
    return NativeCaptureStreamParser(
        io.BytesIO(data),
        expected_device_locator="pnp://rer-u3",
        expected_fingerprint="c" * 64,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        source_epoch_id="epoch-1",
    )


def test_parser_binds_readiness_identity_mode_controls_and_frame_hash() -> None:
    payload = _jpeg()
    parser = _parser(_readiness() + _record(0, 10, payload))

    identity = parser.read_readiness()
    frame = parser.read_frame()

    assert identity.device_locator == "pnp://rer-u3"
    assert identity.native_handle_ref == "mf://rer-u3-symbolic-link"
    assert frame.capture_frame_id == 0
    assert frame.decode_bgr().shape == (12, 16, 3)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_readiness(device_locator="pnp://other"), "device locator"),
        (_readiness(controls_locked=False), "controls"),
        (_readiness(fps_numerator=30), "capture mode"),
        (_readiness() + b"\x00", "truncated"),
    ],
)
def test_parser_rejects_untrusted_readiness_or_truncation(data: bytes, message: str) -> None:
    parser = _parser(data)
    if message == "truncated":
        parser.read_readiness()
        operation = parser.read_frame
    else:
        operation = parser.read_readiness

    with pytest.raises(NativeCaptureProtocolError, match=message):
        operation()


def test_parser_rejects_hash_frame_gap_and_reversed_timestamp() -> None:
    payload = _jpeg()
    bad_hash = _parser(_readiness() + _record(0, 10, payload, digest=b"x" * 32))
    bad_hash.read_readiness()
    with pytest.raises(NativeCaptureProtocolError, match="SHA-256"):
        bad_hash.read_frame()

    gap = _parser(_readiness() + _record(1, 10, payload))
    gap.read_readiness()
    with pytest.raises(NativeCaptureProtocolError, match="frame ID"):
        gap.read_frame()

    reversed_time = _parser(_readiness() + _record(0, 10, payload) + _record(1, 9, payload))
    reversed_time.read_readiness()
    reversed_time.read_frame()
    with pytest.raises(NativeCaptureProtocolError, match="timestamp"):
        reversed_time.read_frame()


def test_backend_uses_startup_timeout_separately_from_read_cancel_timeout() -> None:
    source = Path("src/billiards_ai/lab/mainline1_native_capture.py").read_text(encoding="utf-8")

    assert "readiness.get(timeout=self._config.startup_timeout_seconds)" in source
    assert "self._items.get(timeout=self._config.cancel_timeout_seconds)" in source
    assert "graceful_timeout = self._config.cancel_timeout_seconds * 0.8" in source
    assert "process.terminate()" in source
    assert "remaining = deadline - time.monotonic()" in source


def test_native_failures_preserve_the_specific_reader_and_helper_reason() -> None:
    python_source = Path("src/billiards_ai/lab/mainline1_native_capture.py").read_text(
        encoding="utf-8"
    )
    helper_source = Path("native/windows/mainline1_uvc_capture_csharp/main.cs").read_text(
        encoding="utf-8"
    )

    assert 'f"native capture reader failed: {type(item).__name__}: {item}"' in python_source
    assert 'failureMessage = "native capture callback queue exceeded 32 frames"' in helper_source
    assert "if (graph.FailureMessage != null)" in helper_source


def test_full_reader_queue_stops_blocking_as_soon_as_cancel_is_requested() -> None:
    items: queue.Queue[object] = queue.Queue(maxsize=1)
    items.put(object())
    cancelled = threading.Event()
    cancelled.set()

    accepted = _enqueue_capture_item(items, object(), cancelled)

    assert accepted is False


def test_native_reader_has_bounded_headroom_for_explicit_lag_measurement() -> None:
    source = Path("src/billiards_ai/lab/mainline1_native_capture.py").read_text(encoding="utf-8")

    assert "queue.Queue(maxsize=64)" in source
    assert "def pending_frame_count" in source


def test_cancel_is_idempotent_after_the_helper_control_pipe_has_closed(tmp_path) -> None:
    helper = tmp_path / "helper.exe"
    helper.touch()

    class Process:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdin.close()
            self.returncode = None
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls += 1
            self.returncode = 0
            return self.returncode

    process = Process()
    backend = NativeCaptureSubprocessBackend(
        helper=helper,
        device_locator="pnp://rer-u3",
        expected_fingerprint="c" * 64,
        capture_config=RealtimeCaptureConfig(
            width=16,
            height=12,
            cancel_timeout_seconds=0.1,
        ),
        source_epoch_id="epoch-1",
    )
    backend._process = process

    backend.cancel()
    backend.cancel()

    assert process.wait_calls == 1


class _StartupPipe(io.BytesIO):
    def __init__(self, ended: threading.Event, *, malformed: bool) -> None:
        super().__init__(b"\x00" if malformed else b"")
        self.ended = ended
        self.malformed = malformed

    def read(self, size=-1):
        if not self.malformed:
            self.ended.wait()
        return super().read(size)


class _StartupProcess:
    def __init__(self, *, malformed: bool, stops_at: str) -> None:
        self.ended = threading.Event()
        self.stdin = io.BytesIO()
        self.stdout = _StartupPipe(self.ended, malformed=malformed)
        self.stderr = io.BytesIO()
        self.stops_at = stops_at
        self.actions: list[str] = []
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if not self.actions:
            assert self.stdin.closed
            self.actions.append("stop")
        if self.actions[-1] != self.stops_at:
            raise subprocess.TimeoutExpired("helper", timeout)
        self.returncode = 0
        self.ended.set()
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")


@pytest.mark.parametrize("malformed", [False, True], ids=["timeout", "failure"])
@pytest.mark.parametrize("stops_at", ["stop", "terminate", "kill"])
def test_failed_start_owns_child_until_reaped_and_reader_joined(
    tmp_path, malformed: bool, stops_at: str
) -> None:
    helper = tmp_path / "helper.exe"
    helper.touch()
    process = _StartupProcess(malformed=malformed, stops_at=stops_at)
    backend = NativeCaptureSubprocessBackend(
        helper=helper,
        device_locator="pnp://rer-u3",
        expected_fingerprint="c" * 64,
        capture_config=RealtimeCaptureConfig(
            width=16,
            height=12,
            startup_timeout_seconds=0.01,
            cancel_timeout_seconds=0.1,
        ),
        source_epoch_id="epoch-1",
        process_factory=lambda *args, **kwargs: process,
    )
    try:
        with pytest.raises(NativeCaptureProtocolError):
            backend.start()
        assert process.poll() is not None
        assert backend._reader is not None and not backend._reader.is_alive()
        assert (
            process.actions
            == ["stop", "terminate", "kill"][: ["stop", "terminate", "kill"].index(stops_at) + 1]
        )
        assert process.stdout.closed and process.stderr.closed
        with pytest.raises(NativeCaptureProtocolError):
            backend.start()
    finally:
        # Release the pre-fix reader even when the ownership assertion fails.
        process.ended.set()
        if backend._reader is not None:
            backend._reader.join(timeout=1)


def test_helper_binds_canonical_fingerprint_to_selected_device_before_ready() -> None:
    source = Path("native/windows/mainline1_uvc_capture_csharp/main.cs").read_text()
    assert "DeviceFingerprint(locator)" in source
    assert "fingerprint.Equals(expectedFingerprint, StringComparison.Ordinal)" in source
    assert source.index("fingerprint.Equals(expectedFingerprint") < source.index(
        "moniker.BindToObject"
    )
    assert "SelectDevice(args.DeviceLocator, args.Fingerprint)" in source
    assert "WriteReady(args, graph.DevicePath, graph.DurableFingerprint)" in source
    ready = source.split("private static void WriteReady", 1)[1].split(
        "private static void WriteFrame", 1
    )[0]
    assert "args.Fingerprint" not in ready
    assert "durableFingerprint" in ready
    assert "Regex.Match(locator," in source
    assert 'canonical = "{\\"device_locator\\":"' in source
    assert '"\\"product_id\\":"' in source
    assert '"\\"serial_number\\":"' in source
    assert '"\\"vendor_id\\":"' in source
    assert "sha.ComputeHash(Encoding.UTF8.GetBytes(canonical))" in source
    assert "DurableFingerprint = selected.DurableFingerprint" in source
    assert "DurableFingerprint = fingerprint" in source
    assert "value.Trim().ToLowerInvariant()" in source
    assert 'Regex.IsMatch(args.Fingerprint, @"\\A[0-9a-f]{64}\\z")' in source


@pytest.mark.parametrize(
    ("locator", "canonical"),
    [
        (
            r"USB\VID_1234&PID_ABCD\CAMERA-A",
            r'{"device_locator":"usb\\vid_1234&pid_abcd\\camera-a",'
            r'"product_id":"abcd","serial_number":"camera-a","vendor_id":"1234"}',
        ),
        (
            r"USB\VID_1234&PID_ABCD&MI_00\6&123&0&0000",
            r'{"device_locator":"usb\\vid_1234&pid_abcd&mi_00\\6&123&0&0000",'
            r'"product_id":"abcd","serial_number":null,"vendor_id":"1234"}',
        ),
        (
            r"ROOT\CAMERA\0000",
            r'{"device_locator":"root\\camera\\0000",'
            r'"product_id":null,"serial_number":null,"vendor_id":null}',
        ),
    ],
)
def test_helper_identity_regex_and_json_match_python_capture_setup_contract(
    locator: str, canonical: str
) -> None:
    # Check the helper's source regex against hand-written canonical fixtures.
    # This is a cross-language source contract, not execution of C# on Windows.
    source = Path("native/windows/mainline1_uvc_capture_csharp/main.cs").read_text()
    pattern = re.search(r'Regex.Match\(locator, @"([^"]+)"', source).group(1)
    match = re.search(pattern, locator, re.IGNORECASE)
    identity = UsbDeviceIdentity(
        device_locator=locator,
        display_name="ignored display name",
        vendor_id=match.group(1) if match else None,
        product_id=match.group(2) if match else None,
        serial_number=match.group(3) if match else None,
    )
    assert identity.durable_fingerprint == hashlib.sha256(canonical.encode()).hexdigest()
