"""Paced local scenario video at the product's raw capture boundary."""

from __future__ import annotations

import hashlib
import math
import stat
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

import cv2
import numpy as np

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureContractError,
    CaptureFrame,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)


@dataclass(frozen=True, slots=True)
class ScenarioCaptureConfig:
    source_path: Path
    capture_config: RealtimeCaptureConfig = RealtimeCaptureConfig()
    startup_preparation_seconds: float | None = None

    def __post_init__(self) -> None:
        seconds = self.startup_preparation_seconds
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("scenario startup preparation must have a finite positive budget")


def _source_stat(path: Path) -> tuple[int, int, int, int, int]:
    try:
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise CaptureContractError("scenario source is unsafe: symlink")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise CaptureContractError("scenario source must be a regular file")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except OSError as exc:
        raise CaptureContractError("scenario source is missing or unsafe") from exc


def _content_hash(path: Path) -> str:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as exc:
        raise CaptureContractError("scenario source fingerprint cannot be read") from exc


def _validated_mjpeg(packet: np.ndarray, width: int, height: int) -> bytes:
    """Validate complete JPEGs; retain bytes unless APP1 requires conversion."""
    if (
        not isinstance(packet, np.ndarray)
        or packet.dtype != np.uint8
        or packet.ndim != 2
        or packet.shape[0] != 1
        or packet.size < 4
    ):
        raise CaptureContractError("scenario MJPEG packet cannot be decoded")
    payload = packet.tobytes()
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise CaptureContractError("scenario MJPEG packet cannot be decoded")
    # Container rotation does not describe EXIF embedded in individual JPEGs.
    # FFmpeg ignored it in the old video path, while downstream imdecode applies
    # it. Convert APP1-bearing packets to preserve the old orientation and strip
    # embedded EXIF metadata; never forward those payloads unchanged.
    metadata = b"\xff\xe1" in payload
    flags = (cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) if metadata else cv2.IMREAD_GRAYSCALE
    try:
        # Full resolution validates every image even on the unchanged-packet path.
        image = cv2.imdecode(packet, flags)
    except cv2.error as exc:
        raise CaptureContractError("scenario MJPEG cannot be decoded") from exc
    shape = (height, width, 3) if metadata else (height, width)
    if image is None or image.shape != shape:
        raise CaptureContractError("scenario MJPEG violates capture geometry")
    if metadata:
        try:
            ok, encoded = cv2.imencode(".jpg", image)
        except cv2.error as exc:
            raise CaptureContractError("scenario MJPEG encoding failed") from exc
        if not ok or encoded is None or not encoded.size:
            raise CaptureContractError("scenario MJPEG encoding failed")
        return encoded.tobytes()
    return payload


def _wait(event: threading.Event, seconds: float) -> bool:
    return event.wait(seconds)


class ScenarioPlaybackGate:
    """One-way, thread-safe raw playback latch, owned by one managed scene."""

    def __init__(self) -> None:
        self._released = threading.Event()

    def release(self) -> None:
        self._released.set()

    def is_released(self) -> bool:
        return self._released.is_set()


class ScenarioVideoCaptureBackend:
    """Decode one immutable source and emit only paced ``CaptureFrame`` values.

    Instances are single-use, like an opened camera epoch. Timestamps use the
    opening monotonic clock plus the nominal source cadence, even if downstream
    processing is late. Geometry is exact; FPS permits only 1e-6 absolute error
    for container floating-point representation (no rate conversion). A closed
    playback gate repeats decoded frame zero at that cadence without advancing
    the source cursor; capture IDs and timestamps continue to advance.

    Supported, unrotated FFmpeg MJPEG sources preserve original JPEG packets
    after full decode validation; APP1-bearing packets are converted to retain
    video orientation. Other sources retain decode/re-encode.
    Cancellation interrupts pacing and is checked around codec work. A hung
    native OpenCV call cannot itself be forcibly preempted by this adapter.
    """

    def __init__(
        self,
        config: ScenarioCaptureConfig,
        *,
        source_epoch_id: str,
        playback_gate: ScenarioPlaybackGate | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        waiter: Callable[[threading.Event, float], bool] = _wait,
    ) -> None:
        if not source_epoch_id.strip():
            raise ValueError("scenario source epoch is empty")
        self._config = config.capture_config
        self._startup_preparation_seconds = config.startup_preparation_seconds
        self._preparation_completed_at_ns = None
        self._source = Path(config.source_path).absolute()
        self._source_stat = _source_stat(self._source)
        self._fingerprint = _content_hash(self._source)
        self._validate_source()
        self._identity = NativeCaptureIdentity(
            device_locator=self._source.as_uri(),
            native_handle_ref=f"scenario://sha256/{self._fingerprint}",
            durable_fingerprint=self._fingerprint,
        )
        self._epoch = source_epoch_id
        self._playback_gate = playback_gate
        self._clock_ns = clock_ns
        self._waiter = waiter
        self._cancelled = threading.Event()
        self._started = False
        self._diagnostic_lock = threading.Lock()
        self._startup_snapshot: dict[str, object] = {
            "stage": "not_started",
            "status": "not_started",
            "stage_started_at_ns": None,
            "stage_finished_at_ns": None,
            "cancel_requested_at_ns": None,
            "released_at_ns": None,
            "thread_id": None,
            "failure_type": None,
            "failure_stage": None,
        }

    @property
    def startup_preparation_seconds(self) -> float | None:
        return self._startup_preparation_seconds

    @property
    def preparation_completed_at_ns(self) -> int | None:
        with self._diagnostic_lock:
            return self._preparation_completed_at_ns

    @property
    def startup_diagnostics(self) -> dict[str, object]:
        with self._diagnostic_lock:
            return dict(self._startup_snapshot)

    def _observe_stage(self, stage: str, status: str = "in_progress") -> None:
        with self._diagnostic_lock:
            self._startup_snapshot.update(
                stage=stage,
                status=status,
                stage_started_at_ns=monotonic_ns(),
                stage_finished_at_ns=None,
                thread_id=threading.get_ident(),
            )

    def _observe_failure(self, exc: BaseException) -> None:
        with self._diagnostic_lock:
            self._startup_snapshot.update(
                status="failed",
                stage_finished_at_ns=monotonic_ns(),
                failure_type=type(exc).__name__,
                failure_stage=self._startup_snapshot["stage"],
            )

    @property
    def identity(self) -> NativeCaptureIdentity:
        return self._identity

    def cancel(self) -> None:
        self._cancelled.set()
        with self._diagnostic_lock:
            if self._startup_snapshot["cancel_requested_at_ns"] is None:
                self._startup_snapshot["cancel_requested_at_ns"] = monotonic_ns()

    def _validate_source(self, *, content: bool = False) -> None:
        if _source_stat(self._source) != self._source_stat:
            raise CaptureContractError("scenario source changed or was replaced")
        if content:
            if _content_hash(self._source) != self._fingerprint:
                raise CaptureContractError("scenario source content fingerprint changed")
            if _source_stat(self._source) != self._source_stat:
                raise CaptureContractError("scenario source changed during fingerprint validation")

    def _stopped(self, stop: threading.Event) -> bool:
        return self._cancelled.is_set() or stop.is_set()

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
        if self._started:
            raise CaptureContractError("scenario capture epoch was already opened")
        self._started = True
        if self._stopped(stop):
            return
        capture = None
        try:
            preparation_started = (
                monotonic_ns() if self._startup_preparation_seconds is not None else None
            )
            self._observe_stage("source_validation")
            self._validate_source(content=True)
            if self._startup_preparation_seconds is not None:
                if self._stopped(stop):
                    return
                completed = monotonic_ns()
                if completed - preparation_started > self._startup_preparation_seconds * 1e9:
                    raise CaptureContractError("scenario startup preparation exceeded deadline")
                with self._diagnostic_lock:
                    self._preparation_completed_at_ns = completed
            opened_at_ns = self._clock_ns()
            self._observe_stage("video_open")
            capture = cv2.VideoCapture(str(self._source))
            self._observe_stage("mode_validation")
            if not capture.isOpened():
                raise CaptureContractError("scenario video cannot be opened or decoded")
            self._validate_source()
            if (
                capture.get(cv2.CAP_PROP_FRAME_WIDTH) != self._config.width
                or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) != self._config.height
            ):
                raise CaptureContractError("scenario video geometry does not match capture config")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or not math.isclose(
                fps,
                self._config.fps,
                rel_tol=0,
                abs_tol=1e-6,
            ):
                raise CaptureContractError("scenario video fps does not match capture config")
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(frame_count) or frame_count < 0 or not frame_count.is_integer():
                raise CaptureContractError("scenario video frame count cannot establish clean EOF")
            raw_mjpeg = bool(
                capture.get(cv2.CAP_PROP_BACKEND) == cv2.CAP_FFMPEG
                and capture.get(cv2.CAP_PROP_FOURCC) == cv2.VideoWriter_fourcc(*"MJPG")
                and capture.get(cv2.CAP_PROP_ORIENTATION_META) == 0
                and capture.set(cv2.CAP_PROP_FORMAT, -1)
            )
            if raw_mjpeg and capture.get(cv2.CAP_PROP_FORMAT) != -1:
                raise CaptureContractError("scenario MJPEG raw mode was not established")
            frame_id = 0
            source_cursor = 0
            mjpeg = None
            while not self._stopped(stop):
                self._validate_source()
                if (
                    mjpeg is None
                    or self._playback_gate is None
                    or self._playback_gate.is_released()
                ):
                    if frame_id == 0:
                        self._observe_stage("first_read")
                    ok, image = capture.read()
                    self._validate_source()
                    if self._stopped(stop):
                        return
                    if not ok:
                        self._validate_source(content=True)
                        if source_cursor != int(frame_count):
                            raise CaptureContractError(
                                "scenario video decode failed before expected EOF"
                            )
                        return
                    if source_cursor >= frame_count:
                        raise CaptureContractError("scenario video decoded beyond declared EOF")
                    source_cursor += 1
                    if raw_mjpeg:
                        mjpeg = _validated_mjpeg(image, self._config.width, self._config.height)
                    else:
                        if image is None or image.shape != (
                            self._config.height,
                            self._config.width,
                            3,
                        ):
                            raise CaptureContractError(
                                "scenario decoded frame violates capture geometry"
                            )
                        if frame_id == 0:
                            self._observe_stage("first_encode")
                        try:
                            encoded_ok, encoded = cv2.imencode(".jpg", image)
                        except cv2.error as exc:
                            raise CaptureContractError("scenario MJPEG encoding failed") from exc
                        if not encoded_ok or encoded is None or not encoded.size:
                            raise CaptureContractError("scenario MJPEG encoding failed")
                        mjpeg = encoded.tobytes()
                timestamp_ns = opened_at_ns + round(frame_id * 1_000_000_000 / self._config.fps)
                while not self._stopped(stop):
                    remaining_ns = timestamp_ns - self._clock_ns()
                    if remaining_ns <= 0:
                        break
                    self._waiter(
                        self._cancelled,
                        min(
                            remaining_ns / 1_000_000_000,
                            self._config.cancel_timeout_seconds,
                            0.01,
                        ),
                    )
                if self._stopped(stop):
                    return
                self._validate_source()
                if frame_id == 0:
                    self._observe_stage("frame_ready", "ready")
                yield CaptureFrame(
                    source_epoch_id=self._epoch,
                    capture_frame_id=frame_id,
                    captured_at_ns=timestamp_ns,
                    width=self._config.width,
                    height=self._config.height,
                    mjpeg=mjpeg,
                )
                frame_id += 1
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                self._observe_failure(exc)
            raise
        finally:
            if capture is not None:
                self._observe_stage("video_release")
                try:
                    capture.release()
                except BaseException as exc:
                    self._observe_failure(exc)
                    raise
                else:
                    with self._diagnostic_lock:
                        released_at_ns = monotonic_ns()
                        self._startup_snapshot.update(
                            status="released",
                            stage_finished_at_ns=released_at_ns,
                            released_at_ns=released_at_ns,
                        )
