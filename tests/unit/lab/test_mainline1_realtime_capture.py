from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureCadenceMonitor,
    CaptureContractError,
    CaptureFrame,
    CaptureHealthState,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _frame(frame_id: int, captured_at_ns: int) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=captured_at_ns,
        width=16,
        height=12,
        mjpeg=_jpeg(np.full((12, 16, 3), frame_id, dtype=np.uint8)),
    )


def test_capture_frame_decodes_exact_declared_geometry_and_rejects_invalid_mjpeg() -> None:
    frame = _frame(7, 116_666_667)

    decoded = frame.decode_bgr()

    assert decoded.shape == (12, 16, 3)
    with pytest.raises(CaptureContractError, match="MJPEG"):
        replace(frame, mjpeg=b"not-an-image").decode_bgr()
    with pytest.raises(CaptureContractError, match="geometry"):
        replace(frame, width=17).decode_bgr()


def test_capture_identity_requires_native_binding_not_a_numeric_index() -> None:
    identity = NativeCaptureIdentity(
        device_locator=r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000",
        native_handle_ref="mf://symbolic-link/4k-u3-camera",
        durable_fingerprint="a" * 64,
    )

    assert identity.durable_fingerprint == "a" * 64
    with pytest.raises(ValueError, match="native capture identity"):
        NativeCaptureIdentity(
            device_locator="",
            native_handle_ref="0",
            durable_fingerprint="a" * 64,
        )


def test_capture_config_fixes_product_cadence_and_cancellation_contract() -> None:
    config = RealtimeCaptureConfig()

    assert (config.width, config.height, config.fps, config.pixel_format) == (
        1920,
        1080,
        60.0,
        "MJPG",
    )
    assert config.startup_timeout_seconds == 5.0
    assert config.cancel_timeout_seconds == 0.5
    with pytest.raises(ValueError, match="capture fps"):
        replace(config, fps=0)
    with pytest.raises(ValueError, match="startup timeout"):
        replace(config, startup_timeout_seconds=0)


def test_capture_monitor_reports_sixty_fps_and_rejects_unaccounted_frame_gap() -> None:
    monitor = CaptureCadenceMonitor(expected_fps=60.0)
    for frame_id in range(61):
        monitor.observe(_frame(frame_id, round(frame_id * 1_000_000_000 / 60)))

    health = monitor.health

    assert health.state is CaptureHealthState.RUNNING
    assert health.received_frames == 61
    assert health.measured_fps == pytest.approx(60.0, abs=0.001)
    assert health.maximum_gap_ms == pytest.approx(16.666667, abs=0.00001)
    with pytest.raises(CaptureContractError, match="capture frame gap"):
        monitor.observe(_frame(62, 1_033_333_333))
    assert monitor.health.state is CaptureHealthState.DEGRADED


def test_capture_monitor_accounts_for_frames_coalesced_after_validated_capture() -> None:
    monitor = CaptureCadenceMonitor(
        expected_fps=60.0,
        allow_accounted_frame_gaps=True,
    )
    monitor.observe(_frame(10, 166_666_667))

    monitor.observe(_frame(13, 216_666_668))

    health = monitor.health
    assert health.state is CaptureHealthState.RUNNING
    assert health.received_frames == 2
    assert health.accounted_frame_gaps == 2
    assert health.unaccounted_frame_gaps == 0


def test_capture_monitor_rejects_timestamp_reversal_without_advancing_count() -> None:
    monitor = CaptureCadenceMonitor(expected_fps=60.0)
    monitor.observe(_frame(0, 100))

    with pytest.raises(CaptureContractError, match="timestamp"):
        monitor.observe(_frame(1, 99))

    assert monitor.health.received_frames == 1
    assert monitor.health.timestamp_reversals == 1
    assert monitor.health.state is CaptureHealthState.DEGRADED
