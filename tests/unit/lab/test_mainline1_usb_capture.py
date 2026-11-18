from __future__ import annotations

from collections.abc import Callable
from subprocess import CompletedProcess
from threading import Event, get_ident

import cv2
import numpy as np
import pytest

import billiards_ai.lab.mainline1_usb_capture as capture_module
from billiards_ai.lab.mainline1_usb_capture import (
    CaptureReadTimeout,
    UsbCaptureErrorCode,
    UsbCaptureRequest,
    UsbDeviceIdentity,
    enumerate_usb_devices,
    open_trusted_usb_capture,
    open_usb_capture,
    probe_usb_capture,
)


class _FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        actual_width: int = 1920,
        actual_height: int = 1080,
        actual_fps: float = 60.0,
        frames: int = 3,
        device_fingerprint: str | None = None,
    ) -> None:
        self.opened = opened
        self.actual_width = actual_width
        self.actual_height = actual_height
        self.actual_fps = actual_fps
        self.frames = frames
        self.read_count = 0
        self.release_count = 0
        self.settings: dict[int, float] = {}
        self.set_calls: list[tuple[int, float]] = []
        self.device_fingerprint = device_fingerprint or _identity().durable_fingerprint

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV-compatible boundary
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.settings[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        import cv2

        mode_properties = {
            cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS,
        }
        if property_id in self.settings and property_id not in mode_properties:
            return self.settings[property_id]
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.actual_width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.actual_height),
            cv2.CAP_PROP_FPS: self.actual_fps,
            cv2.CAP_PROP_BACKEND: float(cv2.CAP_DSHOW),
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_count >= self.frames:
            return False, None
        self.read_count += 1
        return True, np.zeros((self.actual_height, self.actual_width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_count += 1


def _identity(*, runtime_index: int | None = 2) -> UsbDeviceIdentity:
    return UsbDeviceIdentity(
        device_locator=r"USB\VID_32E4&PID_9230\FIT1044-A",
        display_name="DFRobot FIT1044",
        runtime_index=runtime_index,
        vendor_id="32e4",
        product_id="9230",
        serial_number="FIT1044-A",
    )


def _request(
    *,
    sample_frames: int = 3,
    warmup_frames: int = 0,
    image_controls: tuple[tuple[str, float], ...] = (),
) -> UsbCaptureRequest:
    return UsbCaptureRequest(
        device=_identity(),
        width=1920,
        height=1080,
        fps=60.0,
        sample_frames=sample_frames,
        warmup_frames=warmup_frames,
        image_controls=image_controls,
    )


def _factory(capture: _FakeCapture) -> Callable[[int], _FakeCapture]:
    def create(runtime_index: int) -> _FakeCapture:
        assert runtime_index == 2
        return capture

    return create


def test_windows_capture_prefers_directshow_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _FakeCapture()
    calls: list[tuple[int, int]] = []

    def video_capture(runtime_index: int, backend: int) -> _FakeCapture:
        calls.append((runtime_index, backend))
        return capture

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", video_capture)

    opened = open_usb_capture(4)

    assert opened is capture
    assert calls == [(4, cv2.CAP_DSHOW)]


def test_windows_capture_falls_back_to_media_foundation_when_directshow_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_show = _FakeCapture(opened=False)
    media_foundation = _FakeCapture()
    captures = {
        cv2.CAP_MSMF: media_foundation,
        cv2.CAP_DSHOW: direct_show,
    }
    calls: list[tuple[int, int]] = []

    def video_capture(runtime_index: int, backend: int) -> _FakeCapture:
        calls.append((runtime_index, backend))
        return captures[backend]

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", video_capture)

    opened = open_usb_capture(4)

    assert opened is media_foundation
    assert calls == [(4, cv2.CAP_DSHOW), (4, cv2.CAP_MSMF)]
    assert direct_show.release_count == 1


def test_identity_fingerprint_ignores_runtime_index_and_display_name() -> None:
    first = _identity(runtime_index=0)
    rebound = first.model_copy(update={"runtime_index": 7, "display_name": "USB Camera"})
    changed = first.model_copy(update={"serial_number": "FIT1044-B"})

    assert first.durable_fingerprint == rebound.durable_fingerprint
    assert first.durable_fingerprint != changed.durable_fingerprint


def test_trusted_windows_binding_stamps_handle_only_for_unique_camera() -> None:
    requested = _identity(runtime_index=0)
    capture = _FakeCapture(device_fingerprint=requested.durable_fingerprint)

    bound = open_trusted_usb_capture(
        requested,
        device_enumerator=lambda: (requested.model_copy(update={"runtime_index": None}),),
        capture_opener=lambda index: capture if index == 0 else pytest.fail("wrong index"),
    )

    assert bound.device_fingerprint == requested.durable_fingerprint
    assert bound.read()[0] is True


def test_trusted_capture_reuses_one_worker_for_all_driver_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DirectShow reads must stay on one thread or its cadence can collapse."""

    requested = _identity(runtime_index=0)
    capture = _FakeCapture(frames=3, device_fingerprint=requested.durable_fingerprint)
    real_thread = capture_module.Thread
    created_threads = []

    def recording_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(capture_module, "Thread", recording_thread)
    bound = open_trusted_usb_capture(
        requested,
        device_enumerator=lambda: (requested.model_copy(update={"runtime_index": None}),),
        capture_opener=lambda _index: capture,
    )

    assert bound.read()[0] is True
    assert bound.read()[0] is True
    assert bound.read()[0] is True
    bound.release()

    assert len(created_threads) == 1


def test_trusted_capture_opens_and_reads_directshow_on_the_same_thread() -> None:
    requested = _identity(runtime_index=0)
    owner_thread: list[int] = []

    class ThreadAffineCapture(_FakeCapture):
        def read(self):
            assert get_ident() == owner_thread[0]
            return super().read()

    capture = ThreadAffineCapture(device_fingerprint=requested.durable_fingerprint)

    def open_capture(_index: int):
        owner_thread.append(get_ident())
        return capture

    bound = open_trusted_usb_capture(
        requested,
        device_enumerator=lambda: (requested.model_copy(update={"runtime_index": None}),),
        capture_opener=open_capture,
    )

    assert bound.read()[0] is True
    bound.release()


def test_trusted_windows_binding_rejects_ambiguous_numeric_index() -> None:
    requested = _identity(runtime_index=0)
    another = requested.model_copy(
        update={
            "device_locator": r"USB\VID_32E4&PID_6678\SECOND",
            "serial_number": "SECOND",
        }
    )

    with pytest.raises(ValueError, match="exactly one present camera"):
        open_trusted_usb_capture(
            requested,
            device_enumerator=lambda: (requested, another),
            capture_opener=lambda _index: pytest.fail("ambiguous index must not be opened"),
        )


def test_trusted_windows_capture_cancels_a_blocked_driver_read() -> None:
    class BlockingCapture(_FakeCapture):
        def read(self):
            Event().wait(10)
            return False, None

    requested = _identity(runtime_index=0)
    capture = BlockingCapture(device_fingerprint=requested.durable_fingerprint)
    bound = open_trusted_usb_capture(
        requested,
        device_enumerator=lambda: (requested.model_copy(update={"runtime_index": None}),),
        capture_opener=lambda _index: capture,
        read_timeout_seconds=0.01,
    )

    with pytest.raises(CaptureReadTimeout, match="timed out"):
        bound.read()

    assert capture.release_count == 1


def test_probe_reports_read_timeout_instead_of_hanging_or_raising() -> None:
    class TimedOutCapture(_FakeCapture):
        def read(self):
            raise CaptureReadTimeout("camera read timed out after 2.000s")

    capture = TimedOutCapture()

    report = probe_usb_capture(_request(), capture_factory=_factory(capture))

    assert report.status is UsbCaptureErrorCode.FRAME_READ_TIMEOUT
    assert report.delivered_frames == 0
    assert report.reasons == ("camera read timed out after 2.000s",)


def test_probe_rejects_silent_1080p30_fallback_without_reading_frames() -> None:
    capture = _FakeCapture(actual_fps=30.0)

    report = probe_usb_capture(_request(), capture_factory=_factory(capture))

    assert report.status is UsbCaptureErrorCode.CAPTURE_GEOMETRY_MISMATCH
    assert report.observed_width == 1920
    assert report.observed_height == 1080
    assert report.observed_fps == 30.0
    assert report.delivered_frames == 0
    assert capture.read_count == 0
    assert capture.release_count == 1


def test_probe_uses_the_field_verified_directshow_negotiation_order() -> None:
    capture = _FakeCapture(frames=3)

    probe_usb_capture(_request(), capture_factory=_factory(capture))

    assert capture.set_calls[:4] == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1920.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 1080.0),
        (cv2.CAP_PROP_FPS, 60.0),
        (cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*"MJPG"))),
    ]


def test_probe_excludes_field_verified_warmup_frames_from_cadence() -> None:
    capture = _FakeCapture(frames=5)
    clock = iter((0, 0, 16_000_000, 33_000_000, 50_000_000))

    report = probe_usb_capture(
        _request(warmup_frames=2),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert capture.read_count == 5
    assert report.delivered_frames == 3
    assert report.measured_fps == 60.606


def test_probe_reports_ordered_sample_quality_and_releases_capture() -> None:
    capture = _FakeCapture(frames=3)
    clock = iter(
        (
            1_000_000_000,
            1_000_000_000,
            1_016_000_000,
            1_033_000_000,
            1_050_000_000,
        )
    )

    report = probe_usb_capture(
        _request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert report.delivered_frames == 3
    assert report.failed_reads == 0
    assert report.timestamp_reversals == 0
    assert report.mean_frame_gap_ms == 16.5
    assert report.maximum_frame_gap_ms == 17.0
    assert report.elapsed_ms == 50.0
    assert report.measured_fps == 60.606
    assert report.device_fingerprint == _identity().durable_fingerprint
    assert capture.release_count == 1


def test_probe_rejects_runtime_index_bound_to_a_different_device() -> None:
    capture = _FakeCapture(device_fingerprint="e" * 64)

    report = probe_usb_capture(_request(), capture_factory=_factory(capture))

    assert report.status is UsbCaptureErrorCode.CAMERA_IDENTITY_MISMATCH
    assert report.delivered_frames == 0
    assert report.reasons == ("capture handle does not match the configured device",)


def test_probe_locks_requested_controls_and_reports_the_verified_profile() -> None:
    # Break caught: merely observing values set earlier in AMCap would allow a
    # restart or reconnect to run with automatic controls enabled again.
    capture = _FakeCapture(frames=3)
    clock = iter((0, 0, 16_000_000, 33_000_000, 50_000_000))
    profile = (
        ("auto_focus", 2.0),
        ("auto_white_balance", 0.0),
        ("exposure", -7.0),
        ("focus", 100.0),
    )

    report = probe_usb_capture(
        _request(image_controls=profile),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert report.image_controls == profile
    assert capture.settings[cv2.CAP_PROP_AUTOFOCUS] == 2.0
    assert capture.settings[cv2.CAP_PROP_AUTO_WB] == 0.0
    assert capture.settings[cv2.CAP_PROP_EXPOSURE] == -7.0
    assert capture.settings[cv2.CAP_PROP_FOCUS] == 100.0


def test_probe_fails_closed_when_driver_accepts_but_does_not_apply_control() -> None:
    # Break caught: OpenCV set() returning True is not proof that the UVC driver
    # applied the requested value.
    capture = _FakeCapture(frames=3)
    original_set = capture.set

    def accept_without_applying(property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_EXPOSURE:
            return True
        return original_set(property_id, value)

    capture.set = accept_without_applying  # type: ignore[method-assign]

    report = probe_usb_capture(
        _request(image_controls=(("exposure", -7.0),)),
        capture_factory=_factory(capture),
    )

    assert report.status is UsbCaptureErrorCode.IMAGE_CONTROL_MISMATCH
    assert report.delivered_frames == 0
    assert report.reasons == ("image control could not be locked: exposure",)


def test_probe_records_current_epoch_frame_evidence_and_controls() -> None:
    capture = _FakeCapture(frames=3)
    clock = iter((0, 0, 16_000_000, 33_000_000, 50_000_000))

    report = probe_usb_capture(
        _request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
        corner_detector=lambda _frame: ((1.0, 1079.0), (1919.0, 1079.0), (1919.0, 1.0), (1.0, 1.0)),
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert len(report.source_epoch_id) == 32
    assert len(report.evidence_frame_sha256) == 64
    assert report.validation_corners_px == (
        (1.0, 1079.0),
        (1919.0, 1079.0),
        (1919.0, 1.0),
        (1.0, 1.0),
    )
    assert dict(report.image_controls)["exposure"] == 0.0


def test_probe_retries_current_epoch_geometry_after_cadence_sample() -> None:
    capture = _FakeCapture(frames=5)
    clock = iter((0, 0, 16_000_000, 33_000_000, 50_000_000))
    corners = ((1.0, 1079.0), (1919.0, 1079.0), (1919.0, 1.0), (1.0, 1.0))
    detector_calls = 0
    evidence_frames = []

    def detect(_frame):
        nonlocal detector_calls
        detector_calls += 1
        return corners if detector_calls == 3 else None

    report = probe_usb_capture(
        _request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
        corner_detector=detect,
        evidence_frame_sink=evidence_frames.append,
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert report.measured_fps == 60.606
    assert report.delivered_frames == 3
    assert report.validation_corners_px == corners
    assert capture.read_count == 5
    assert len(evidence_frames) == 1


def test_probe_measures_steady_cadence_without_counting_first_frame_startup() -> None:
    capture = _FakeCapture(frames=3)
    clock = iter(
        (
            0,
            200_000_000,
            216_666_667,
            233_333_333,
            233_333_333,
        )
    )

    report = probe_usb_capture(
        _request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.READY
    assert report.elapsed_ms == 233.333333
    assert report.measured_fps == 60.0


def test_probe_rejects_driver_reported_60fps_when_measured_delivery_is_slow() -> None:
    capture = _FakeCapture(frames=3)
    clock = iter((0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 3_100_000_000))

    report = probe_usb_capture(
        _request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.CAPTURE_CADENCE_MISMATCH
    assert report.observed_fps == 60.0
    assert report.measured_fps == 1.0
    assert report.reasons == ("measured frame delivery did not match the requested fps",)
    assert capture.release_count == 1


def test_probe_reports_camera_busy_and_still_releases_handle() -> None:
    capture = _FakeCapture(opened=False)

    report = probe_usb_capture(_request(), capture_factory=_factory(capture))

    assert report.status is UsbCaptureErrorCode.CAMERA_BUSY
    assert report.delivered_frames == 0
    assert report.reasons == ("camera could not be opened",)
    assert capture.release_count == 1


def test_probe_reports_read_failure_without_claiming_ready() -> None:
    capture = _FakeCapture(frames=0)
    clock = iter((1_000_000_000, 1_010_000_000))

    report = probe_usb_capture(
        _request(sample_frames=1),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
    )

    assert report.status is UsbCaptureErrorCode.FRAME_READ_FAILED
    assert report.delivered_frames == 0
    assert report.failed_reads == 1
    assert capture.release_count == 1


def test_windows_enumeration_extracts_durable_pnp_identity() -> None:
    seen = {}

    def runner(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return CompletedProcess(
            command,
            0,
            stdout=(
                '[{"FriendlyName":"DFRobot FIT1044",'
                '"InstanceId":"USB\\\\VID_32E4&PID_9230\\\\FIT1044-A",'
                '"Status":"OK"}]'
            ),
            stderr="",
        )

    devices = enumerate_usb_devices(platform_name="win32", command_runner=runner)

    assert len(devices) == 1
    assert devices[0].device_locator == r"USB\VID_32E4&PID_9230\FIT1044-A"
    assert devices[0].vendor_id == "32E4"
    assert devices[0].product_id == "9230"
    assert devices[0].serial_number == "FIT1044-A"
    assert devices[0].runtime_index is None
    assert seen["command"][:3] == ["powershell.exe", "-NoProfile", "-Command"]
    assert seen["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
    }


def test_non_windows_enumeration_never_runs_powershell() -> None:
    def runner(*args, **kwargs):  # pragma: no cover - must remain unused
        raise AssertionError((args, kwargs))

    assert enumerate_usb_devices(platform_name="darwin", command_runner=runner) == ()
