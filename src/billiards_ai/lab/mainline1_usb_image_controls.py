"""Inspect and explicitly test UVC image controls on the Mainline 1 USB camera."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from billiards_ai.lab.mainline1_usb_capture import (
    CaptureFactory,
    CaptureHandle,
    UsbDeviceIdentity,
    _backend_name,
    open_usb_capture,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UsbImageControlName(StrEnum):
    AUTO_EXPOSURE = "auto_exposure"
    EXPOSURE = "exposure"
    GAIN = "gain"
    AUTO_WHITE_BALANCE = "auto_white_balance"
    WHITE_BALANCE_TEMPERATURE = "white_balance_temperature"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    SHARPNESS = "sharpness"


class UsbImageControlSetting(_FrozenModel):
    name: UsbImageControlName
    value: float

    @field_validator("value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("image control value must be finite")
        return value


class UsbImageControlsRequest(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    device: UsbDeviceIdentity
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    warmup_frames: int = Field(default=60, gt=0)
    settings: tuple[UsbImageControlSetting, ...] = ()

    @field_validator("settings")
    @classmethod
    def _unique_settings(
        cls, settings: tuple[UsbImageControlSetting, ...]
    ) -> tuple[UsbImageControlSetting, ...]:
        names = [setting.name for setting in settings]
        if len(names) != len(set(names)):
            raise ValueError("each image control may be requested only once")
        return settings


class UsbImageControlStatus(StrEnum):
    OBSERVED = "observed"
    APPLIED = "applied"
    REJECTED = "rejected"
    NOT_APPLIED = "not_applied"
    UNREADABLE = "unreadable"


class UsbImageControlObservation(_FrozenModel):
    before: float | None = None
    requested: float | None = None
    after: float | None = None
    setter_accepted: bool | None = None
    status: UsbImageControlStatus


class UsbFrameQuality(_FrozenModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    luminance_mean: float = Field(ge=0, le=255)
    luminance_p05: float = Field(ge=0, le=255)
    luminance_p50: float = Field(ge=0, le=255)
    luminance_p95: float = Field(ge=0, le=255)
    clipped_black_ratio: float = Field(ge=0, le=1)
    clipped_white_ratio: float = Field(ge=0, le=1)


class UsbImageControlsStatus(StrEnum):
    PASS = "PASS"
    CAMERA_NOT_FOUND = "camera_not_found"
    CAMERA_BUSY = "camera_busy"
    CAPTURE_MODE_MISMATCH = "capture_mode_mismatch"
    FRAME_READ_FAILED = "frame_read_failed"
    CONTROL_WRITE_FAILED = "control_write_failed"


class UsbImageControlsReport(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: UsbImageControlsStatus
    device_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_name: str | None = None
    observed_width: int | None = Field(default=None, gt=0)
    observed_height: int | None = Field(default=None, gt=0)
    observed_fps: float | None = Field(default=None, gt=0)
    control_mode: Literal["read_only", "explicit_write"]
    controls: dict[str, UsbImageControlObservation]
    quality: UsbFrameQuality | None = None
    reasons: tuple[str, ...] = ()


_CONTROL_PROPERTY_IDS: dict[UsbImageControlName, int] = {
    UsbImageControlName.AUTO_EXPOSURE: cv2.CAP_PROP_AUTO_EXPOSURE,
    UsbImageControlName.EXPOSURE: cv2.CAP_PROP_EXPOSURE,
    UsbImageControlName.GAIN: cv2.CAP_PROP_GAIN,
    UsbImageControlName.AUTO_WHITE_BALANCE: cv2.CAP_PROP_AUTO_WB,
    UsbImageControlName.WHITE_BALANCE_TEMPERATURE: cv2.CAP_PROP_WB_TEMPERATURE,
    UsbImageControlName.BRIGHTNESS: cv2.CAP_PROP_BRIGHTNESS,
    UsbImageControlName.CONTRAST: cv2.CAP_PROP_CONTRAST,
    UsbImageControlName.SATURATION: cv2.CAP_PROP_SATURATION,
    UsbImageControlName.SHARPNESS: cv2.CAP_PROP_SHARPNESS,
}


def _read_control(capture: CaptureHandle, property_id: int) -> float | None:
    value = float(capture.get(property_id))
    return value if math.isfinite(value) else None


def _frame_quality(frame: Any) -> UsbFrameQuality:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"camera frame has invalid shape: {array.shape!r}")
    if array.shape[2] not in (3, 4):
        raise ValueError(f"camera frame has invalid channel count: {array.shape[2]}")
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return UsbFrameQuality(
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        luminance_mean=float(np.mean(gray)),
        luminance_p05=float(np.percentile(gray, 5)),
        luminance_p50=float(np.percentile(gray, 50)),
        luminance_p95=float(np.percentile(gray, 95)),
        clipped_black_ratio=float(np.mean(gray <= 1)),
        clipped_white_ratio=float(np.mean(gray >= 254)),
    )


def _report(
    request: UsbImageControlsRequest,
    *,
    status: UsbImageControlsStatus,
    backend_name: str | None = None,
    observed_width: int | None = None,
    observed_height: int | None = None,
    observed_fps: float | None = None,
    controls: dict[str, UsbImageControlObservation] | None = None,
    quality: UsbFrameQuality | None = None,
    reasons: tuple[str, ...] = (),
) -> UsbImageControlsReport:
    return UsbImageControlsReport(
        status=status,
        device_fingerprint=request.device.durable_fingerprint,
        backend_name=backend_name,
        observed_width=observed_width,
        observed_height=observed_height,
        observed_fps=observed_fps,
        control_mode="explicit_write" if request.settings else "read_only",
        controls=controls or {},
        quality=quality,
        reasons=reasons,
    )


def probe_usb_image_controls(
    request: UsbImageControlsRequest,
    *,
    capture_factory: CaptureFactory = open_usb_capture,
) -> tuple[UsbImageControlsReport, Any | None]:
    """Read image controls and optionally apply only explicitly requested values."""

    runtime_index = request.device.runtime_index
    if runtime_index is None:
        return (
            _report(
                request,
                status=UsbImageControlsStatus.CAMERA_NOT_FOUND,
                reasons=("device has no resolved runtime index",),
            ),
            None,
        )

    capture = capture_factory(runtime_index)
    try:
        if not capture.isOpened():
            return (
                _report(
                    request,
                    status=UsbImageControlsStatus.CAMERA_BUSY,
                    reasons=("camera could not be opened",),
                ),
                None,
            )

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.height))
        capture.set(cv2.CAP_PROP_FPS, request.fps)
        observed_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        observed_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
        backend_name = _backend_name(capture)
        if (
            observed_width != request.width
            or observed_height != request.height
            or abs(observed_fps - request.fps) > 0.5
        ):
            return (
                _report(
                    request,
                    status=UsbImageControlsStatus.CAPTURE_MODE_MISMATCH,
                    backend_name=backend_name,
                    observed_width=observed_width,
                    observed_height=observed_height,
                    observed_fps=observed_fps,
                    reasons=("capture mode did not match the exact requested geometry",),
                ),
                None,
            )

        before = {
            name: _read_control(capture, property_id)
            for name, property_id in _CONTROL_PROPERTY_IDS.items()
        }
        requested = {setting.name: setting.value for setting in request.settings}
        accepted = {
            name: bool(capture.set(_CONTROL_PROPERTY_IDS[name], value))
            for name, value in requested.items()
        }

        frame = None
        frame_failure: str | None = None
        for _ in range(request.warmup_frames):
            ok, candidate = capture.read()
            if not ok or candidate is None:
                frame_failure = "camera did not deliver a warmup frame"
                break
            candidate_array = np.asarray(candidate)
            if (
                candidate_array.ndim != 3
                or candidate_array.shape[0] <= 0
                or candidate_array.shape[1] <= 0
                or candidate_array.shape[2] not in (3, 4)
            ):
                frame_failure = (
                    f"camera delivered a frame with invalid shape: {candidate_array.shape!r}"
                )
                break
            frame = candidate

        after = {
            name: _read_control(capture, property_id)
            for name, property_id in _CONTROL_PROPERTY_IDS.items()
        }
        observations: dict[str, UsbImageControlObservation] = {}
        for name in _CONTROL_PROPERTY_IDS:
            requested_value = requested.get(name)
            accepted_value = accepted.get(name)
            if requested_value is None:
                status = (
                    UsbImageControlStatus.OBSERVED
                    if before[name] is not None
                    else UsbImageControlStatus.UNREADABLE
                )
            elif not accepted_value:
                status = UsbImageControlStatus.REJECTED
            elif after[name] is not None and math.isclose(
                after[name], requested_value, abs_tol=0.01
            ):
                status = UsbImageControlStatus.APPLIED
            else:
                status = UsbImageControlStatus.NOT_APPLIED
            observations[name.value] = UsbImageControlObservation(
                before=before[name],
                requested=requested_value,
                after=after[name],
                setter_accepted=accepted_value,
                status=status,
            )

        if frame_failure is not None:
            return (
                _report(
                    request,
                    status=UsbImageControlsStatus.FRAME_READ_FAILED,
                    backend_name=backend_name,
                    observed_width=observed_width,
                    observed_height=observed_height,
                    observed_fps=observed_fps,
                    controls=observations,
                    reasons=(frame_failure,),
                ),
                None,
            )

        assert frame is not None
        control_status = (
            UsbImageControlsStatus.CONTROL_WRITE_FAILED
            if any(
                observation.requested is not None
                and observation.status is not UsbImageControlStatus.APPLIED
                for observation in observations.values()
            )
            else UsbImageControlsStatus.PASS
        )
        return (
            _report(
                request,
                status=control_status,
                backend_name=backend_name,
                observed_width=observed_width,
                observed_height=observed_height,
                observed_fps=observed_fps,
                controls=observations,
                quality=_frame_quality(frame),
            ),
            frame,
        )
    finally:
        capture.release()
