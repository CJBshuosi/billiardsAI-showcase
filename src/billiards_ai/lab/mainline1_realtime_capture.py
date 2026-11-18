"""Product realtime capture contracts independent of a platform backend."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import cv2
import numpy as np


class CaptureContractError(ValueError):
    """A native capture frame or health transition violated its contract."""


class CaptureHealthState(StrEnum):
    RUNNING = "running"
    DEGRADED = "degraded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class NativeCaptureIdentity:
    device_locator: str
    native_handle_ref: str
    durable_fingerprint: str

    def __post_init__(self) -> None:
        fingerprint = self.durable_fingerprint
        if (
            not self.device_locator.strip()
            or not self.native_handle_ref.strip()
            or self.native_handle_ref.strip().isdigit()
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("native capture identity is invalid")


@dataclass(frozen=True, slots=True)
class RealtimeCaptureConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    pixel_format: str = "MJPG"
    startup_timeout_seconds: float = 5.0
    cancel_timeout_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("capture geometry must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("capture fps must be finite and positive")
        if self.pixel_format != "MJPG":
            raise ValueError("product realtime capture requires MJPG")
        if not math.isfinite(self.startup_timeout_seconds) or self.startup_timeout_seconds <= 0:
            raise ValueError("capture startup timeout must be finite and positive")
        if not math.isfinite(self.cancel_timeout_seconds) or self.cancel_timeout_seconds <= 0:
            raise ValueError("capture cancellation timeout must be finite and positive")


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    source_epoch_id: str
    capture_frame_id: int
    captured_at_ns: int
    width: int
    height: int
    mjpeg: bytes

    def __post_init__(self) -> None:
        if not self.source_epoch_id.strip():
            raise CaptureContractError("capture source epoch is empty")
        if self.capture_frame_id < 0 or self.captured_at_ns < 0:
            raise CaptureContractError("capture frame identity is invalid")
        if self.width <= 0 or self.height <= 0:
            raise CaptureContractError("capture frame geometry is invalid")
        if not self.mjpeg:
            raise CaptureContractError("capture MJPEG payload is empty")

    @property
    def timestamp_ms(self) -> int:
        return round(self.captured_at_ns / 1_000_000)

    def decode_bgr(self) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self.mjpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureContractError("capture MJPEG cannot be decoded")
        if image.shape[:2] != (self.height, self.width):
            raise CaptureContractError("capture MJPEG violates declared geometry")
        return image

    def decode_working_gray(self, *, width: int, height: int) -> np.ndarray:
        """Decode MJPEG at the closest native JPEG reduction before any resize."""
        if width <= 0 or height <= 0:
            raise ValueError("working decode geometry must be positive")
        reduction = next(
            (
                factor
                for factor in (1, 2, 4, 8)
                if self.width // factor == width and self.height // factor == height
            ),
            1,
        )
        flags = {
            1: cv2.IMREAD_GRAYSCALE,
            2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
            4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
            8: cv2.IMREAD_REDUCED_GRAYSCALE_8,
        }
        image = cv2.imdecode(
            np.frombuffer(self.mjpeg, dtype=np.uint8),
            flags[reduction],
        )
        expected_height = (self.height + reduction - 1) // reduction
        expected_width = (self.width + reduction - 1) // reduction
        if image is None or image.shape[:2] != (expected_height, expected_width):
            raise CaptureContractError("capture MJPEG violates declared geometry")
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    state: CaptureHealthState
    received_frames: int
    measured_fps: float | None
    maximum_gap_ms: float
    timestamp_reversals: int
    unaccounted_frame_gaps: int
    accounted_frame_gaps: int = 0
    decode_failures: int = 0
    ring_write_failures: int = 0
    control_drifts: int = 0
    reason: str | None = None


class CaptureCadenceMonitor:
    """Observe native capture continuity before frames reach perception."""

    def __init__(
        self,
        *,
        expected_fps: float,
        allow_accounted_frame_gaps: bool = False,
    ) -> None:
        if not math.isfinite(expected_fps) or expected_fps <= 0:
            raise ValueError("expected capture fps must be finite and positive")
        self._expected_fps = expected_fps
        self._allow_accounted_frame_gaps = allow_accounted_frame_gaps
        self._state = CaptureHealthState.RUNNING
        self._received_frames = 0
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_frame_id: int | None = None
        self._maximum_gap_ns = 0
        self._timestamp_reversals = 0
        self._unaccounted_frame_gaps = 0
        self._accounted_frame_gaps = 0
        self._reason: str | None = None

    @property
    def health(self) -> CaptureHealth:
        measured_fps: float | None = None
        if (
            self._received_frames > 1
            and self._first_timestamp_ns is not None
            and self._last_timestamp_ns is not None
            and self._last_timestamp_ns > self._first_timestamp_ns
        ):
            measured_fps = (
                (self._received_frames - 1)
                * 1_000_000_000
                / (self._last_timestamp_ns - self._first_timestamp_ns)
            )
        return CaptureHealth(
            state=self._state,
            received_frames=self._received_frames,
            measured_fps=measured_fps,
            maximum_gap_ms=self._maximum_gap_ns / 1_000_000,
            timestamp_reversals=self._timestamp_reversals,
            unaccounted_frame_gaps=self._unaccounted_frame_gaps,
            accounted_frame_gaps=self._accounted_frame_gaps,
            reason=self._reason,
        )

    def observe(self, frame: CaptureFrame) -> None:
        if self._state is not CaptureHealthState.RUNNING:
            raise CaptureContractError("capture cadence monitor is not running")
        if self._last_frame_id is not None:
            frame_gap = frame.capture_frame_id - self._last_frame_id - 1
            if frame_gap and self._allow_accounted_frame_gaps and frame_gap > 0:
                self._accounted_frame_gaps += frame_gap
            elif frame_gap:
                self._unaccounted_frame_gaps += 1
                self._degrade("capture frame gap is unaccounted")
        if self._last_timestamp_ns is not None and frame.captured_at_ns < self._last_timestamp_ns:
            self._timestamp_reversals += 1
            self._degrade("capture timestamp reversed")
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = frame.captured_at_ns
        if self._last_timestamp_ns is not None:
            self._maximum_gap_ns = max(
                self._maximum_gap_ns,
                frame.captured_at_ns - self._last_timestamp_ns,
            )
        self._last_frame_id = frame.capture_frame_id
        self._last_timestamp_ns = frame.captured_at_ns
        self._received_frames += 1

    def close(self) -> None:
        if self._state is CaptureHealthState.RUNNING:
            self._state = CaptureHealthState.CLOSED

    def _degrade(self, reason: str) -> None:
        self._state = CaptureHealthState.DEGRADED
        self._reason = reason
        raise CaptureContractError(reason)


class CaptureBackend(Protocol):
    @property
    def identity(self) -> NativeCaptureIdentity: ...

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]: ...

    def cancel(self) -> None: ...
