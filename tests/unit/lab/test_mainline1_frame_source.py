from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_frame_source import (
    FileFrameSource,
    FrameSourceContractError,
    UsbFrameSource,
    decode_frame_source,
    validate_frame_order,
)
from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureRequest,
    UsbDeviceIdentity,
)
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame


def _synthetic_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (160, 120),
    )
    assert writer.isOpened()
    for value in (0, 64, 128):
        writer.write(np.full((120, 160, 3), value, dtype=np.uint8))
    writer.release()
    return path


class _FakeCapture:
    def __init__(
        self,
        frames: tuple[np.ndarray, ...],
        *,
        device_fingerprint: str | None = None,
    ) -> None:
        self.frames = list(frames)
        self.release_count = 0
        self.settings: dict[int, float] = {}
        self.device_fingerprint = device_fingerprint or _usb_request().device.durable_fingerprint

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, property_id: int, value: float) -> bool:
        self.settings[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        if property_id in self.settings:
            return self.settings[property_id]
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 60.0,
            cv2.CAP_PROP_BACKEND: float(cv2.CAP_DSHOW),
        }
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.release_count += 1


def _usb_request(
    *,
    image_controls: tuple[tuple[str, float], ...] = (),
) -> UsbCaptureRequest:
    return UsbCaptureRequest(
        device=UsbDeviceIdentity(
            device_locator=r"USB\VID_32E4&PID_9230\FIT1044-A",
            display_name="DFRobot FIT1044",
            runtime_index=2,
            vendor_id="32e4",
            product_id="9230",
            serial_number="FIT1044-A",
        ),
        width=1920,
        height=1080,
        fps=60,
        sample_frames=120,
        warmup_frames=0,
        image_controls=image_controls,
    )


def _factory(capture: _FakeCapture) -> Callable[[int], _FakeCapture]:
    def create(runtime_index: int) -> _FakeCapture:
        assert runtime_index == 2
        return capture

    return create


def test_file_source_preserves_existing_frame_ids_and_timestamps(tmp_path: Path) -> None:
    source = FileFrameSource(_synthetic_video(tmp_path / "input.avi"))

    frames, metadata = decode_frame_source(source)

    assert [frame.frame_id for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_ms for frame in frames] == [0, 100, 200]
    assert metadata.source_kind == "file"
    assert metadata.width == 160
    assert metadata.height == 120
    assert metadata.fps == 10.0


def test_usb_source_uses_host_monotonic_timestamps_and_durable_identity() -> None:
    capture = _FakeCapture(
        (
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.ones((1080, 1920, 3), dtype=np.uint8),
        )
    )
    clock = iter((1_000_000_000, 1_016_000_000))
    source = UsbFrameSource(
        request=_usb_request(),
        capture_factory=_factory(capture),
        monotonic_ns=lambda: next(clock),
        max_frames=2,
    )

    frames, metadata = decode_frame_source(source)

    assert [frame.frame_id for frame in frames] == [0, 1]
    assert [frame.timestamp_ms for frame in frames] == [0, 16]
    assert metadata.source_kind == "usb"
    assert metadata.source_ref == _usb_request().device.durable_fingerprint
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == 60.0
    assert capture.release_count == 1


def test_frame_contract_rejects_duplicate_frame_id() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = (
        DecodedTrackingFrame(frame_id=1, timestamp_ms=10, image=image),
        DecodedTrackingFrame(frame_id=1, timestamp_ms=20, image=image),
    )

    with pytest.raises(FrameSourceContractError, match="frame_id"):
        tuple(validate_frame_order(frames))


def test_frame_contract_rejects_timestamp_reversal() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = (
        DecodedTrackingFrame(frame_id=1, timestamp_ms=20, image=image),
        DecodedTrackingFrame(frame_id=2, timestamp_ms=19, image=image),
    )

    with pytest.raises(FrameSourceContractError, match="timestamp"):
        tuple(validate_frame_order(frames))


def test_usb_source_rejects_silent_mode_fallback_before_yielding() -> None:
    class FallbackCapture(_FakeCapture):
        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 30.0
            return super().get(property_id)

    capture = FallbackCapture((np.zeros((1080, 1920, 3), dtype=np.uint8),))
    source = UsbFrameSource(
        request=_usb_request(),
        capture_factory=_factory(capture),
        max_frames=1,
    )

    with pytest.raises(FrameSourceContractError, match="exact requested mode"):
        tuple(source)

    assert capture.release_count == 1


def test_usb_source_rejects_reopened_handle_for_a_different_device() -> None:
    capture = _FakeCapture(
        (np.zeros((1080, 1920, 3), dtype=np.uint8),),
        device_fingerprint="e" * 64,
    )
    source = UsbFrameSource(
        request=_usb_request(),
        capture_factory=_factory(capture),
        max_frames=1,
    )

    with pytest.raises(FrameSourceContractError, match="configured device"):
        tuple(source)


def test_usb_source_rejects_a_frozen_frame_stream() -> None:
    frozen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    capture = _FakeCapture((frozen, frozen.copy(), frozen.copy()))
    source = UsbFrameSource(
        request=_usb_request(),
        capture_factory=_factory(capture),
        max_frames=3,
        maximum_identical_frames=2,
    )

    with pytest.raises(FrameSourceContractError, match="frozen"):
        tuple(source)


def test_usb_source_fails_closed_when_a_locked_control_drifts() -> None:
    class DriftingCapture(_FakeCapture):
        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_EXPOSURE and len(self.frames) < 2:
                return -6.0
            return super().get(property_id)

    capture = DriftingCapture(
        (
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            np.ones((1080, 1920, 3), dtype=np.uint8),
        )
    )
    source = UsbFrameSource(
        request=_usb_request(image_controls=(("exposure", -7.0),)),
        capture_factory=_factory(capture),
        max_frames=2,
        image_control_check_interval_frames=1,
    )

    with pytest.raises(FrameSourceContractError, match="image control drifted: exposure"):
        tuple(source)

    assert source.health.delivered_frames == 0
    assert capture.release_count == 1


def test_retained_usb_epoch_handle_is_single_use() -> None:
    capture = _FakeCapture((np.zeros((1080, 1920, 3), dtype=np.uint8),))
    source = UsbFrameSource(
        request=_usb_request(),
        capture_handle=capture,
        capture_factory=lambda _index: pytest.fail("must not reopen the USB epoch"),
        max_frames=1,
    )

    tuple(source)

    with pytest.raises(FrameSourceContractError, match="single-use"):
        tuple(source)
