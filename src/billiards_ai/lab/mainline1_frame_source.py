"""Source-neutral ordered frame streams for Mainline 1."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from billiards_ai.lab.mainline1_capture import CaptureSourceKind
from billiards_ai.lab.mainline1_usb_capture import (
    CaptureFactory,
    CaptureHandle,
    UsbCaptureRequest,
    _controls_still_locked,
    _lock_image_controls,
    open_trusted_usb_capture,
)
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame


class FrameSourceContractError(ValueError):
    """Raised before invalid source data can reach perception."""


@dataclass(frozen=True, slots=True)
class FrameSourceMetadata:
    source_kind: CaptureSourceKind
    source_ref: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class FrameSourceHealth:
    delivered_frames: int
    stopped_cleanly: bool
    reason: str | None = None


class FrameSource(Protocol):
    @property
    def metadata(self) -> FrameSourceMetadata: ...

    @property
    def health(self) -> FrameSourceHealth: ...

    def __iter__(self) -> Iterator[DecodedTrackingFrame]: ...


def validate_frame_order(
    frames: Iterable[DecodedTrackingFrame],
) -> Iterator[DecodedTrackingFrame]:
    previous_frame_id: int | None = None
    previous_timestamp_ms: int | None = None
    for frame in frames:
        if previous_frame_id is not None and frame.frame_id <= previous_frame_id:
            raise FrameSourceContractError("frame_id must be strictly increasing")
        if previous_timestamp_ms is not None and frame.timestamp_ms < previous_timestamp_ms:
            raise FrameSourceContractError("timestamp must be monotonic")
        if frame.image.ndim != 3 or frame.image.shape[2] != 3:
            raise FrameSourceContractError("frame image must be BGR HxWx3")
        previous_frame_id = frame.frame_id
        previous_timestamp_ms = frame.timestamp_ms
        yield frame


class FileFrameSource:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._metadata: FrameSourceMetadata | None = None
        self._health = FrameSourceHealth(delivered_frames=0, stopped_cleanly=False)

    @property
    def metadata(self) -> FrameSourceMetadata:
        if self._metadata is None:
            raise FrameSourceContractError("file source has not been decoded")
        return self._metadata

    @property
    def health(self) -> FrameSourceHealth:
        return self._health

    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        capture = cv2.VideoCapture(str(self._path))
        delivered = 0
        try:
            if not capture.isOpened():
                raise FrameSourceContractError("shot-local video cannot be decoded")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._metadata = FrameSourceMetadata(
                source_kind=CaptureSourceKind.FILE,
                source_ref=str(self._path.resolve()),
                width=width,
                height=height,
                fps=fps,
            )
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield DecodedTrackingFrame(
                    frame_id=delivered,
                    timestamp_ms=round(delivered * 1000 / fps),
                    image=image,
                )
                delivered += 1
            if delivered == 0:
                raise FrameSourceContractError("shot-local video contains no decodable frames")
            self._health = FrameSourceHealth(
                delivered_frames=delivered,
                stopped_cleanly=True,
            )
        except Exception as exc:
            self._health = FrameSourceHealth(
                delivered_frames=delivered,
                stopped_cleanly=False,
                reason=str(exc),
            )
            raise
        finally:
            capture.release()


class UsbFrameSource:
    def __init__(
        self,
        *,
        request: UsbCaptureRequest,
        capture_factory: CaptureFactory | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        max_frames: int | None = None,
        maximum_identical_frames: int = 120,
        image_control_check_interval_frames: int = 120,
        capture_handle: CaptureHandle | None = None,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if maximum_identical_frames <= 1:
            raise ValueError("maximum_identical_frames must be greater than one")
        if image_control_check_interval_frames <= 0:
            raise ValueError("image_control_check_interval_frames must be positive")
        self._request = request
        self._capture_factory = capture_factory
        self._monotonic_ns = monotonic_ns
        self._max_frames = max_frames
        self._maximum_identical_frames = maximum_identical_frames
        self._image_control_check_interval_frames = image_control_check_interval_frames
        self._capture_handle = capture_handle
        self._started = False
        self._metadata: FrameSourceMetadata | None = None
        self._health = FrameSourceHealth(delivered_frames=0, stopped_cleanly=False)

    @property
    def metadata(self) -> FrameSourceMetadata:
        if self._metadata is None:
            raise FrameSourceContractError("USB source has not started")
        return self._metadata

    @property
    def health(self) -> FrameSourceHealth:
        return self._health

    def close(self) -> None:
        self._started = True
        capture = self._capture_handle
        self._capture_handle = None
        if capture is not None:
            capture.release()

    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        if self._started:
            raise FrameSourceContractError("USB source is single-use")
        self._started = True
        runtime_index = self._request.device.runtime_index
        if runtime_index is None:
            raise FrameSourceContractError("USB device has no resolved runtime index")
        capture = self._capture_handle
        self._capture_handle = None
        if capture is None:
            capture = (
                open_trusted_usb_capture(self._request.device)
                if self._capture_factory is None
                else self._capture_factory(runtime_index)
            )
        delivered = 0
        try:
            if not capture.isOpened():
                raise FrameSourceContractError("USB camera could not be opened")
            if (
                getattr(capture, "device_fingerprint", None)
                != self._request.device.durable_fingerprint
            ):
                raise FrameSourceContractError(
                    "USB capture handle does not match configured device"
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._request.width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._request.height))
            capture.set(cv2.CAP_PROP_FPS, self._request.fps)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if (
                width != self._request.width
                or height != self._request.height
                or abs(fps - self._request.fps) > 0.5
            ):
                raise FrameSourceContractError(
                    "USB camera did not negotiate the exact requested mode"
                )
            if (
                self._request.image_controls
                and _lock_image_controls(capture, self._request.image_controls) is None
            ):
                raise FrameSourceContractError("USB image controls could not be locked")
            for _ in range(self._request.warmup_frames):
                ok, warmup_frame = capture.read()
                if not ok or warmup_frame is None:
                    raise FrameSourceContractError(
                        "USB frame read failed during mode-negotiation warmup"
                    )
            self._metadata = FrameSourceMetadata(
                source_kind=CaptureSourceKind.USB,
                source_ref=self._request.device.durable_fingerprint,
                width=width,
                height=height,
                fps=fps,
            )
            origin_ns: int | None = None
            previous_digest: str | None = None
            identical_frames = 0
            while self._max_frames is None or delivered < self._max_frames:
                ok, image = capture.read()
                if not ok or image is None:
                    raise FrameSourceContractError("USB frame read failed")
                if image.shape[:2] != (height, width):
                    raise FrameSourceContractError("USB frame geometry changed after negotiation")
                if delivered % self._image_control_check_interval_frames == 0:
                    drifted_control = _controls_still_locked(capture, self._request.image_controls)
                    if drifted_control is not None:
                        raise FrameSourceContractError(
                            f"USB image control drifted: {drifted_control}"
                        )
                captured_ns = self._monotonic_ns()
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                identical_frames = identical_frames + 1 if digest == previous_digest else 1
                previous_digest = digest
                if identical_frames >= self._maximum_identical_frames:
                    raise FrameSourceContractError("USB frame stream is frozen")
                if origin_ns is None:
                    origin_ns = captured_ns
                yield DecodedTrackingFrame(
                    frame_id=delivered,
                    timestamp_ms=round((captured_ns - origin_ns) / 1_000_000),
                    image=image,
                )
                delivered += 1
            self._health = FrameSourceHealth(
                delivered_frames=delivered,
                stopped_cleanly=True,
            )
        except Exception as exc:
            self._health = FrameSourceHealth(
                delivered_frames=delivered,
                stopped_cleanly=False,
                reason=str(exc),
            )
            raise
        finally:
            capture.release()


def decode_frame_source(
    source: FrameSource,
) -> tuple[tuple[DecodedTrackingFrame, ...], FrameSourceMetadata]:
    frames = tuple(validate_frame_order(source))
    return frames, source.metadata
