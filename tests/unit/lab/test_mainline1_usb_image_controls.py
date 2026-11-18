from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity
from billiards_ai.lab.mainline1_usb_image_controls import (
    UsbImageControlSetting,
    UsbImageControlsRequest,
    probe_usb_image_controls,
)


class _FakeCapture:
    def __init__(
        self,
        *,
        reject: set[int] | None = None,
        fail_reads: bool = False,
        malformed_frame: bool = False,
    ) -> None:
        self.opened = True
        self.release_count = 0
        self.reject = reject or set()
        self.fail_reads = fail_reads
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 60.0,
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
            cv2.CAP_PROP_EXPOSURE: -5.0,
            cv2.CAP_PROP_GAIN: 4.0,
            cv2.CAP_PROP_AUTO_WB: 1.0,
            cv2.CAP_PROP_WB_TEMPERATURE: 4600.0,
            cv2.CAP_PROP_BRIGHTNESS: 128.0,
            cv2.CAP_PROP_CONTRAST: 32.0,
            cv2.CAP_PROP_SATURATION: 64.0,
            cv2.CAP_PROP_SHARPNESS: 3.0,
        }
        self.frame = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[10, 20, 30], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        if malformed_frame:
            self.frame = np.array([1, 2, 3], dtype=np.uint8)

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        if property_id in self.reject:
            return False
        self.values[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_BACKEND:
            return float(cv2.CAP_MSMF)
        return self.values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray]:
        if self.fail_reads:
            return False, self.frame.copy()
        return True, self.frame.copy()

    def release(self) -> None:
        self.release_count += 1


def _request(*settings: UsbImageControlSetting) -> UsbImageControlsRequest:
    return UsbImageControlsRequest(
        device=UsbDeviceIdentity(
            device_locator=r"USB\VID_A000&PID_B111&MI_00\fixture",
            display_name="FIT1044",
            runtime_index=0,
            vendor_id="A000",
            product_id="B111",
        ),
        width=1920,
        height=1080,
        fps=60.0,
        warmup_frames=2,
        settings=settings,
    )


def _factory(capture: _FakeCapture) -> Callable[[int], _FakeCapture]:
    def create(runtime_index: int) -> _FakeCapture:
        assert runtime_index == 0
        return capture

    return create


def test_read_only_probe_observes_controls_without_mutating_them() -> None:
    capture = _FakeCapture()
    before = {
        property_id: capture.values[property_id]
        for property_id in (
            cv2.CAP_PROP_AUTO_EXPOSURE,
            cv2.CAP_PROP_EXPOSURE,
            cv2.CAP_PROP_GAIN,
            cv2.CAP_PROP_AUTO_WB,
            cv2.CAP_PROP_WB_TEMPERATURE,
            cv2.CAP_PROP_BRIGHTNESS,
            cv2.CAP_PROP_CONTRAST,
            cv2.CAP_PROP_SATURATION,
            cv2.CAP_PROP_SHARPNESS,
        )
    }

    report, frame = probe_usb_image_controls(_request(), capture_factory=_factory(capture))

    assert report.status == "PASS"
    assert report.backend_name == "MSMF"
    assert report.observed_width == 1920
    assert report.observed_height == 1080
    assert report.observed_fps == 60.0
    assert report.control_mode == "read_only"
    assert report.controls["exposure"].before == -5.0
    assert report.controls["exposure"].requested is None
    assert report.controls["exposure"].status == "observed"
    assert report.quality.width == 2
    assert report.quality.height == 2
    assert report.quality.clipped_black_ratio == 0.25
    assert report.quality.clipped_white_ratio == 0.25
    assert frame.shape == (2, 2, 3)
    assert {property_id: capture.values[property_id] for property_id in before} == before
    assert capture.release_count == 1


def test_explicit_control_writes_are_read_back_and_rejections_are_reported() -> None:
    capture = _FakeCapture(reject={cv2.CAP_PROP_GAIN})

    report, _frame = probe_usb_image_controls(
        _request(
            UsbImageControlSetting(name="exposure", value=-7.0),
            UsbImageControlSetting(name="gain", value=1.0),
        ),
        capture_factory=_factory(capture),
    )

    assert report.status == "control_write_failed"
    assert report.control_mode == "explicit_write"
    assert report.controls["exposure"].before == -5.0
    assert report.controls["exposure"].after == -7.0
    assert report.controls["exposure"].requested == -7.0
    assert report.controls["exposure"].setter_accepted is True
    assert report.controls["exposure"].status == "applied"
    assert report.controls["gain"].before == 4.0
    assert report.controls["gain"].after == 4.0
    assert report.controls["gain"].requested == 1.0
    assert report.controls["gain"].setter_accepted is False
    assert report.controls["gain"].status == "rejected"
    assert capture.release_count == 1


def test_setter_success_is_not_treated_as_applied_when_readback_differs() -> None:
    capture = _FakeCapture()
    original_set = capture.set

    def accept_without_applying(property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_EXPOSURE:
            return True
        return original_set(property_id, value)

    capture.set = accept_without_applying  # type: ignore[method-assign]

    report, _frame = probe_usb_image_controls(
        _request(UsbImageControlSetting(name="exposure", value=-7.0)),
        capture_factory=_factory(capture),
    )

    assert report.status == "control_write_failed"
    assert report.controls["exposure"].setter_accepted is True
    assert report.controls["exposure"].after == -5.0
    assert report.controls["exposure"].status == "not_applied"


def test_readback_is_preserved_when_frame_delivery_fails_after_a_write() -> None:
    capture = _FakeCapture(fail_reads=True)

    report, frame = probe_usb_image_controls(
        _request(UsbImageControlSetting(name="exposure", value=-7.0)),
        capture_factory=_factory(capture),
    )

    assert report.status == "frame_read_failed"
    assert report.controls["exposure"].before == -5.0
    assert report.controls["exposure"].requested == -7.0
    assert report.controls["exposure"].after == -7.0
    assert report.controls["exposure"].status == "applied"
    assert frame is None
    assert capture.release_count == 1


def test_malformed_frame_fails_closed_and_releases_camera() -> None:
    capture = _FakeCapture(malformed_frame=True)

    report, frame = probe_usb_image_controls(_request(), capture_factory=_factory(capture))

    assert report.status == "frame_read_failed"
    assert "invalid shape" in report.reasons[0]
    assert frame is None
    assert capture.release_count == 1


def test_control_probe_fails_closed_when_camera_cannot_open() -> None:
    capture = _FakeCapture()
    capture.opened = False

    report, frame = probe_usb_image_controls(_request(), capture_factory=_factory(capture))

    assert report.status == "camera_busy"
    assert report.reasons == ("camera could not be opened",)
    assert frame is None
    assert capture.release_count == 1
