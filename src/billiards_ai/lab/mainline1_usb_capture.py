"""USB capture identity and preflight quality probe for Mainline 1."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from enum import StrEnum
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UsbCaptureErrorCode(StrEnum):
    READY = "ready"
    CAMERA_NOT_FOUND = "camera_not_found"
    CAMERA_BUSY = "camera_busy"
    CAMERA_IDENTITY_MISMATCH = "camera_identity_mismatch"
    CAPTURE_GEOMETRY_MISMATCH = "capture_geometry_mismatch"
    CAPTURE_CADENCE_MISMATCH = "capture_cadence_mismatch"
    FRAME_READ_FAILED = "frame_read_failed"
    FRAME_READ_TIMEOUT = "frame_read_timeout"
    TIMESTAMP_INVALID = "timestamp_invalid"
    IMAGE_CONTROL_MISMATCH = "image_control_mismatch"


class UsbDeviceIdentity(_FrozenModel):
    """Durable device identity plus an explicitly ephemeral runtime index."""

    schema_version: Literal["1.0"] = "1.0"
    device_locator: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    runtime_index: int | None = Field(default=None, ge=0)
    vendor_id: str | None = None
    product_id: str | None = None
    serial_number: str | None = None

    @property
    def durable_fingerprint(self) -> str:
        payload = {
            "device_locator": self.device_locator.strip().casefold(),
            "product_id": _normalized_optional(self.product_id),
            "serial_number": _normalized_optional(self.serial_number),
            "vendor_id": _normalized_optional(self.vendor_id),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class UsbCaptureRequest(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    device: UsbDeviceIdentity
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    sample_frames: int = Field(gt=0)
    warmup_frames: int = Field(default=15, ge=0)
    image_controls: tuple[tuple[str, float], ...] = ()

    @model_validator(mode="after")
    def validate_image_controls(self) -> UsbCaptureRequest:
        names: set[str] = set()
        for name, value in self.image_controls:
            if name not in _CONTROL_PROPERTY_IDS:
                raise ValueError(f"unsupported image control: {name}")
            if name in names:
                raise ValueError(f"duplicate image control: {name}")
            if not math.isfinite(value):
                raise ValueError(f"image control must be finite: {name}")
            names.add(name)
        return self


class UsbCaptureReport(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: UsbCaptureErrorCode
    device_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_width: int = Field(gt=0)
    requested_height: int = Field(gt=0)
    requested_fps: float = Field(gt=0)
    observed_width: int | None = Field(default=None, gt=0)
    observed_height: int | None = Field(default=None, gt=0)
    observed_fps: float | None = Field(default=None, gt=0)
    measured_fps: float | None = Field(default=None, gt=0)
    delivered_frames: int = Field(ge=0)
    failed_reads: int = Field(ge=0)
    timestamp_reversals: int = Field(ge=0)
    mean_frame_gap_ms: float | None = Field(default=None, ge=0)
    maximum_frame_gap_ms: float | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(ge=0)
    backend_name: str | None = None
    reasons: tuple[str, ...] = ()
    source_epoch_id: str | None = None
    evidence_frame_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    validation_corners_px: tuple[tuple[float, float], ...] | None = None
    image_controls: tuple[tuple[str, float], ...] = ()


class CaptureHandle(Protocol):
    def isOpened(self) -> bool: ...  # noqa: N802

    def set(self, property_id: int, value: float) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...

    def getBackendName(self) -> str: ...  # noqa: N802


CaptureFactory = Callable[[int], CaptureHandle]


class CaptureReadTimeout(TimeoutError):
    """Raised after cancelling a capture whose driver read stopped responding."""


_USB_ID_PATTERN = re.compile(r"VID_([0-9A-F]{4})&PID_([0-9A-F]{4})(?:\\([^\\]+))?", re.IGNORECASE)


def _normalized_optional(value: str | None) -> str | None:
    return value.strip().casefold() if value is not None else None


def open_usb_capture(runtime_index: int) -> CaptureHandle:
    """Open the resolved camera through the best supported Windows backend."""

    if sys.platform == "win32":
        capture = cv2.VideoCapture(runtime_index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(runtime_index, cv2.CAP_MSMF)


def enumerate_usb_devices(
    *,
    platform_name: str = sys.platform,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[UsbDeviceIdentity, ...]:
    """Enumerate durable Windows camera identities without guessing OpenCV indices."""

    if platform_name != "win32":
        return ()
    command = (
        "Get-PnpDevice -PresentOnly | "
        "Where-Object { $_.Class -in @('Camera','Image') } | "
        "Select-Object FriendlyName,InstanceId,Status | ConvertTo-Json -Compress"
    )
    completed = command_runner(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Windows camera enumeration failed")
    raw = completed.stdout.strip()
    if not raw:
        return ()
    payload = json.loads(raw)
    rows = payload if isinstance(payload, list) else [payload]
    devices: list[UsbDeviceIdentity] = []
    for row in rows:
        locator = str(row.get("InstanceId") or "").strip()
        display_name = str(row.get("FriendlyName") or "").strip()
        if not locator or not display_name:
            continue
        match = _USB_ID_PATTERN.search(locator)
        devices.append(
            UsbDeviceIdentity(
                device_locator=locator,
                display_name=display_name,
                vendor_id=match.group(1) if match else None,
                product_id=match.group(2) if match else None,
                serial_number=match.group(3) if match else None,
            )
        )
    return tuple(sorted(devices, key=lambda device: device.durable_fingerprint))


class _TrustedCaptureHandle:
    """Run every DirectShow operation on one bounded, driver-facing thread."""

    def __init__(
        self,
        capture_opener: CaptureFactory,
        runtime_index: int,
        device_fingerprint: str,
        *,
        read_timeout_seconds: float,
    ) -> None:
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        self._capture_opener = capture_opener
        self._runtime_index = runtime_index
        self._capture: CaptureHandle | None = None
        self.device_fingerprint = device_fingerprint
        self._read_timeout_seconds = read_timeout_seconds
        self._released = False
        self._requests: Queue[tuple[str, tuple[Any, ...]]] = Queue(maxsize=1)
        self._results: Queue[Any | BaseException] = Queue(maxsize=1)
        self._operation_lock = Lock()
        self._worker = Thread(
            target=self._run_capture,
            name="mainline1-usb-capture",
            daemon=True,
        )
        self._worker.start()
        try:
            opened = self._results.get(timeout=max(10.0, read_timeout_seconds))
        except Empty as exc:
            self._emergency_release()
            raise CaptureReadTimeout("camera open timed out") from exc
        if isinstance(opened, BaseException):
            raise opened

    def isOpened(self) -> bool:  # noqa: N802
        return bool(self._invoke("isOpened"))

    def set(self, property_id: int, value: float) -> bool:
        return bool(self._invoke("set", property_id, value))

    def get(self, property_id: int) -> float:
        return float(self._invoke("get", property_id))

    def read(self) -> tuple[bool, Any]:
        if self._released:
            return False, None
        return self._invoke("read")

    def _invoke(self, command: str, *arguments: Any) -> Any:
        with self._operation_lock:
            if self._released:
                raise RuntimeError("camera capture has been released")
            self._requests.put((command, arguments))
            try:
                outcome = self._results.get(timeout=self._read_timeout_seconds)
            except Empty as exc:
                self._released = True
                try:
                    self._requests.put_nowait(("stop", ()))
                except Full:
                    pass
                self._emergency_release()
                raise CaptureReadTimeout(
                    f"camera {command} timed out after {self._read_timeout_seconds:.3f}s"
                ) from exc
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    def _run_capture(self) -> None:
        try:
            capture = self._capture_opener(self._runtime_index)
            self._capture = capture
        except BaseException as exc:  # pragma: no cover - defensive driver boundary
            self._results.put(exc)
            return
        self._results.put(True)
        while True:
            command, arguments = self._requests.get()
            if command == "stop":
                return
            try:
                operation = getattr(capture, command)
                outcome: Any | BaseException = operation(*arguments)
            except BaseException as exc:  # pragma: no cover - defensive driver boundary
                outcome = exc
            self._results.put(outcome)
            if command == "release":
                return

    def _emergency_release(self) -> None:
        capture = self._capture
        if capture is not None:
            capture.release()

    def release(self) -> None:
        with self._operation_lock:
            if self._released:
                return
            self._released = True
            self._requests.put(("release", ()))
            try:
                self._results.get(timeout=self._read_timeout_seconds)
            except Empty:
                self._emergency_release()

    def getBackendName(self) -> str:  # noqa: N802
        return str(self._invoke("getBackendName"))


def open_trusted_usb_capture(
    device: UsbDeviceIdentity,
    *,
    device_enumerator: Callable[[], tuple[UsbDeviceIdentity, ...]] = enumerate_usb_devices,
    capture_opener: CaptureFactory = open_usb_capture,
    read_timeout_seconds: float = 2.0,
) -> CaptureHandle:
    """Bind PnP identity to OpenCV index when the mapping is provably unique.

    OpenCV does not expose the Windows PnP locator of an opened numeric capture
    handle.  The safe compatibility path therefore accepts index 0 only when
    Windows reports exactly one present camera and it is the configured device.
    Multi-camera machines fail closed instead of guessing.
    """

    if device.runtime_index != 0:
        raise ValueError("trusted Windows binding requires runtime index 0")
    devices = device_enumerator()
    if len(devices) != 1:
        raise ValueError("trusted Windows binding requires exactly one present camera")
    present = devices[0]
    if present.device_locator.casefold() != device.device_locator.casefold():
        raise ValueError("the only present camera does not match the configured device")
    return _TrustedCaptureHandle(
        capture_opener,
        0,
        device.durable_fingerprint,
        read_timeout_seconds=read_timeout_seconds,
    )


def _report(
    request: UsbCaptureRequest,
    *,
    status: UsbCaptureErrorCode,
    observed_width: int | None = None,
    observed_height: int | None = None,
    observed_fps: float | None = None,
    measured_fps: float | None = None,
    delivered_frames: int = 0,
    failed_reads: int = 0,
    timestamp_reversals: int = 0,
    frame_gaps_ms: tuple[float, ...] = (),
    elapsed_ms: float = 0.0,
    backend_name: str | None = None,
    reasons: tuple[str, ...] = (),
    source_epoch_id: str | None = None,
    evidence_frame_sha256: str | None = None,
    validation_corners_px: tuple[tuple[float, float], ...] | None = None,
    image_controls: tuple[tuple[str, float], ...] = (),
) -> UsbCaptureReport:
    return UsbCaptureReport(
        status=status,
        device_fingerprint=request.device.durable_fingerprint,
        requested_width=request.width,
        requested_height=request.height,
        requested_fps=request.fps,
        observed_width=observed_width,
        observed_height=observed_height,
        observed_fps=observed_fps,
        measured_fps=measured_fps,
        delivered_frames=delivered_frames,
        failed_reads=failed_reads,
        timestamp_reversals=timestamp_reversals,
        mean_frame_gap_ms=(sum(frame_gaps_ms) / len(frame_gaps_ms) if frame_gaps_ms else None),
        maximum_frame_gap_ms=max(frame_gaps_ms) if frame_gaps_ms else None,
        elapsed_ms=elapsed_ms,
        backend_name=backend_name,
        reasons=reasons,
        source_epoch_id=source_epoch_id,
        evidence_frame_sha256=evidence_frame_sha256,
        validation_corners_px=validation_corners_px,
        image_controls=image_controls,
    )


_CONTROL_PROPERTIES = (
    ("brightness", cv2.CAP_PROP_BRIGHTNESS),
    ("contrast", cv2.CAP_PROP_CONTRAST),
    ("hue", cv2.CAP_PROP_HUE),
    ("saturation", cv2.CAP_PROP_SATURATION),
    ("sharpness", cv2.CAP_PROP_SHARPNESS),
    ("gamma", cv2.CAP_PROP_GAMMA),
    ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("exposure", cv2.CAP_PROP_EXPOSURE),
    ("gain", cv2.CAP_PROP_GAIN),
    ("auto_white_balance", cv2.CAP_PROP_AUTO_WB),
    ("white_balance_temperature", cv2.CAP_PROP_WB_TEMPERATURE),
    ("backlight_compensation", cv2.CAP_PROP_BACKLIGHT),
    ("focus", cv2.CAP_PROP_FOCUS),
    ("auto_focus", cv2.CAP_PROP_AUTOFOCUS),
    ("zoom", cv2.CAP_PROP_ZOOM),
)
_CONTROL_PROPERTY_IDS = dict(_CONTROL_PROPERTIES)


def _read_control(capture: CaptureHandle, name: str) -> float | None:
    value = float(capture.get(_CONTROL_PROPERTY_IDS[name]))
    return value if math.isfinite(value) else None


def _control_matches(actual: float | None, expected: float) -> bool:
    return actual is not None and math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.25)


def _lock_image_controls(
    capture: CaptureHandle,
    requested: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...] | None:
    for name, expected in requested:
        property_id = _CONTROL_PROPERTY_IDS[name]
        if not capture.set(property_id, expected):
            return None
        actual = _read_control(capture, name)
        if not _control_matches(actual, expected):
            return None
    return tuple((name, float(expected)) for name, expected in requested)


def _controls_still_locked(
    capture: CaptureHandle,
    requested: tuple[tuple[str, float], ...],
) -> str | None:
    for name, expected in requested:
        if not _control_matches(_read_control(capture, name), expected):
            return name
    return None


def probe_usb_capture(
    request: UsbCaptureRequest,
    *,
    capture_factory: CaptureFactory | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    corner_detector: Callable[[Any], tuple[tuple[float, float], ...] | None] | None = None,
    capture_sink: Callable[[CaptureHandle], None] | None = None,
    evidence_frame_sink: Callable[[Any], None] | None = None,
) -> UsbCaptureReport:
    """Probe an exact USB mode and fail closed on fallback or unstable reads."""

    runtime_index = request.device.runtime_index
    if runtime_index is None:
        return _report(
            request,
            status=UsbCaptureErrorCode.CAMERA_NOT_FOUND,
            reasons=("device has no resolved runtime index",),
        )

    capture = (
        open_trusted_usb_capture(request.device)
        if capture_factory is None
        else capture_factory(runtime_index)
    )
    retained = False
    try:
        if not capture.isOpened():
            return _report(
                request,
                status=UsbCaptureErrorCode.CAMERA_BUSY,
                reasons=("camera could not be opened",),
            )
        resolved_fingerprint = getattr(capture, "device_fingerprint", None)
        if resolved_fingerprint != request.device.durable_fingerprint:
            return _report(
                request,
                status=UsbCaptureErrorCode.CAMERA_IDENTITY_MISMATCH,
                reasons=("capture handle does not match the configured device",),
            )

        # This RER U3 rebuilds its DirectShow graph as geometry/FPS change.
        # The field-verified 60fps path requests MJPG last.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.height))
        capture.set(cv2.CAP_PROP_FPS, request.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        observed_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        observed_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
        backend_name = _backend_name(capture)
        if (
            observed_width != request.width
            or observed_height != request.height
            or abs(observed_fps - request.fps) > 0.5
        ):
            return _report(
                request,
                status=UsbCaptureErrorCode.CAPTURE_GEOMETRY_MISMATCH,
                observed_width=observed_width,
                observed_height=observed_height,
                observed_fps=observed_fps,
                backend_name=backend_name,
                reasons=("capture mode did not match the exact requested geometry",),
            )

        locked_controls = _lock_image_controls(capture, request.image_controls)
        if request.image_controls and locked_controls is None:
            failed_name = next(
                (
                    name
                    for name, expected in request.image_controls
                    if not _control_matches(_read_control(capture, name), expected)
                ),
                request.image_controls[0][0],
            )
            return _report(
                request,
                status=UsbCaptureErrorCode.IMAGE_CONTROL_MISMATCH,
                observed_width=observed_width,
                observed_height=observed_height,
                observed_fps=observed_fps,
                backend_name=backend_name,
                reasons=(f"image control could not be locked: {failed_name}",),
            )

        for _ in range(request.warmup_frames):
            try:
                ok, warmup_frame = capture.read()
            except CaptureReadTimeout as exc:
                return _report(
                    request,
                    status=UsbCaptureErrorCode.FRAME_READ_TIMEOUT,
                    observed_width=observed_width,
                    observed_height=observed_height,
                    observed_fps=observed_fps,
                    failed_reads=1,
                    backend_name=backend_name,
                    reasons=(str(exc),),
                    image_controls=locked_controls or (),
                )
            if not ok or warmup_frame is None:
                return _report(
                    request,
                    status=UsbCaptureErrorCode.FRAME_READ_FAILED,
                    observed_width=observed_width,
                    observed_height=observed_height,
                    observed_fps=observed_fps,
                    failed_reads=1,
                    backend_name=backend_name,
                    reasons=("camera failed during mode-negotiation warmup",),
                    image_controls=locked_controls or (),
                )

        started_ns = monotonic_ns()
        frame_timestamps_ns: list[int] = []
        evidence_frame: Any | None = None
        failed_reads = 0
        read_timeout_reason: str | None = None
        for _ in range(request.sample_frames):
            try:
                ok, frame = capture.read()
            except CaptureReadTimeout as exc:
                failed_reads += 1
                read_timeout_reason = str(exc)
                break
            if not ok or frame is None:
                failed_reads += 1
                break
            frame_timestamps_ns.append(monotonic_ns())
            evidence_frame = frame
        ended_ns = monotonic_ns()
        elapsed_ms = max(0.0, (ended_ns - started_ns) / 1_000_000)
        delivery_span_ns = (
            frame_timestamps_ns[-1] - frame_timestamps_ns[0] if len(frame_timestamps_ns) > 1 else 0
        )
        measured_fps = (
            round((len(frame_timestamps_ns) - 1) / (delivery_span_ns / 1_000_000_000), 3)
            if delivery_span_ns > 0
            else None
        )
        gaps_ms = tuple(
            (current - previous) / 1_000_000
            for previous, current in zip(frame_timestamps_ns, frame_timestamps_ns[1:], strict=False)
        )
        reversals = sum(gap < 0 for gap in gaps_ms)
        if read_timeout_reason is not None:
            status = UsbCaptureErrorCode.FRAME_READ_TIMEOUT
            reasons = (read_timeout_reason,)
        elif reversals:
            status = UsbCaptureErrorCode.TIMESTAMP_INVALID
            reasons = ("frame timestamps were not monotonic",)
        elif len(frame_timestamps_ns) != request.sample_frames:
            status = UsbCaptureErrorCode.FRAME_READ_FAILED
            reasons = ("camera did not deliver the requested sample",)
        elif measured_fps is None or abs(measured_fps - request.fps) > request.fps * 0.05:
            status = UsbCaptureErrorCode.CAPTURE_CADENCE_MISMATCH
            reasons = ("measured frame delivery did not match the requested fps",)
        else:
            status = UsbCaptureErrorCode.READY
            reasons = ()
        validation_corners = None
        if corner_detector is not None and evidence_frame is not None:
            # A single decoded frame can land while DirectShow is still
            # settling after mode/control negotiation. Geometry work is kept
            # outside the cadence window and retried on fresh frames.
            for attempt in range(5):
                validation_corners = corner_detector(evidence_frame)
                if validation_corners is not None or attempt == 4:
                    break
                try:
                    ok, retry_frame = capture.read()
                except CaptureReadTimeout as exc:
                    failed_reads += 1
                    status = UsbCaptureErrorCode.FRAME_READ_TIMEOUT
                    reasons = (str(exc),)
                    break
                if not ok or retry_frame is None:
                    failed_reads += 1
                    status = UsbCaptureErrorCode.FRAME_READ_FAILED
                    reasons = ("camera failed during current-epoch geometry retry",)
                    break
                evidence_frame = retry_frame
        evidence_sha256 = (
            hashlib.sha256(evidence_frame.tobytes()).hexdigest()
            if evidence_frame is not None
            else None
        )
        if evidence_frame_sink is not None and evidence_frame is not None:
            evidence_frame_sink(evidence_frame)
        drifted_control = _controls_still_locked(capture, request.image_controls)
        if drifted_control is not None:
            status = UsbCaptureErrorCode.IMAGE_CONTROL_MISMATCH
            reasons = (f"image control drifted after capture: {drifted_control}",)
        controls = locked_controls or tuple(
            (name, value)
            for name, property_id in _CONTROL_PROPERTIES
            if math.isfinite(value := float(capture.get(property_id)))
        )
        report = _report(
            request,
            status=status,
            observed_width=observed_width,
            observed_height=observed_height,
            observed_fps=observed_fps,
            measured_fps=measured_fps,
            delivered_frames=len(frame_timestamps_ns),
            failed_reads=failed_reads,
            timestamp_reversals=reversals,
            frame_gaps_ms=gaps_ms,
            elapsed_ms=elapsed_ms,
            backend_name=backend_name,
            reasons=reasons,
            source_epoch_id=uuid4().hex,
            evidence_frame_sha256=evidence_sha256,
            validation_corners_px=validation_corners,
            image_controls=controls,
        )
        if report.status is UsbCaptureErrorCode.READY and capture_sink is not None:
            capture_sink(capture)
            retained = True
        return report
    finally:
        if not retained:
            capture.release()


def _backend_name(capture: CaptureHandle) -> str | None:
    get_backend_name = getattr(capture, "getBackendName", None)
    if callable(get_backend_name):
        try:
            return str(get_backend_name())
        except cv2.error:
            return None
    backend = round(capture.get(cv2.CAP_PROP_BACKEND))
    backend_names = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
    }
    return backend_names.get(backend, str(backend))
