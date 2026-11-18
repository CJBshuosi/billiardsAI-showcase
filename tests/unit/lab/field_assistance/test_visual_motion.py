from __future__ import annotations

import tracemalloc

import cv2
import numpy as np
import pytest

from billiards_ai.lab.field_assistance.visual_motion import (
    RecordingVisualMotion,
    VisualMotionConfig,
    table_visual_motion_detected,
    table_visual_motion_ratio,
)
from billiards_ai.lab.mainline1_live_window import VisualShotBoundaryDetector
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeRuntime
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.shared.calibration.homography import calibrate


def _calibration():
    return calibrate("camera", ((0, 99), (199, 99), (199, 0), (0, 0)))


def test_static_frames_remain_stable_despite_small_pixel_noise() -> None:
    first = np.full((100, 200, 3), 80, dtype=np.uint8)
    second = first.copy()
    second[20, 20] = 90

    assert not table_visual_motion_detected(first, second, _calibration(), VisualMotionConfig())


def test_ball_sized_change_inside_table_is_motion() -> None:
    first = np.full((100, 200, 3), 80, dtype=np.uint8)
    second = first.copy()
    cv2.circle(second, (100, 50), 6, (240, 240, 240), -1)
    config = VisualMotionConfig(changed_pixel_ratio=0.0005)

    assert table_visual_motion_ratio(first, second, _calibration(), config) > 0.0005
    assert table_visual_motion_detected(first, second, _calibration(), config)


def test_temporal_motion_returns_to_still_and_reset_discards_prior_scene():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    first = np.zeros((100, 200, 3), dtype=np.uint8)
    second = first.copy()
    cv2.circle(second, (100, 50), 6, (240, 240, 240), -1)
    assert not detector.push(first, timestamp_ms=0)
    assert detector.push(second, timestamp_ms=17)
    for timestamp in range(34, 501, 17):
        moving = detector.push(second, timestamp_ms=timestamp)
    assert not moving
    detector.reset()
    assert not detector.push(first, timestamp_ms=0)


def test_temporal_history_is_bounded_even_when_timestamps_repeat():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracemalloc.start()
    try:
        for _ in range(1000):
            assert not detector.push(image, timestamp_ms=10)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024**2
    with pytest.raises(ValueError, match="ordered"):
        detector.push(image, timestamp_ms=9)


def test_one_exposure_step_is_only_one_independent_motion_confirmation():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    flags = [
        detector.push(np.full((100, 200, 3), 80 if i < 10 else 110, np.uint8), timestamp_ms=i * 17)
        for i in range(80)
    ]
    assert sum(flags) == 1


def test_short_legal_still_gap_keeps_two_separate_shots():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    windows = VisualShotBoundaryDetector(
        ShotWindowConfig(motion_threshold_mm=8, motion_confirm_frames=2, still_duration_ms=400)
    )
    events = []
    for i in range(145):
        image = np.full((100, 200, 3), 80, np.uint8)
        x = 30 + 2 * min(10, max(0, i - 10)) + 2 * min(10, max(0, i - 56))
        cv2.circle(image, (x, 50), 6, (240, 240, 240), -1)
        events.extend(
            windows.push(
                frame_id=i, timestamp_ms=i * 17, moving=detector.push(image, timestamp_ms=i * 17)
            )
        )
    assert sum(e.kind.value == "SHOT_SETTLED" for e in events) == 2


def test_periodic_motion_cannot_alias_to_stillness():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    flags = []
    for i in range(80):
        image = np.full((100, 200, 3), 80, np.uint8)
        cv2.circle(image, (30 + 10 * (i % 5), 50), 6, (240, 240, 240), -1)
        flags.append(detector.push(image, timestamp_ms=i * 10))
    assert all(flags[40:])


def test_long_sample_gap_cannot_be_interpreted_as_stability():
    detector = RecordingVisualMotion(_calibration(), VisualMotionConfig(), still_duration_ms=400)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detector.push(image, timestamp_ms=0)
    with pytest.raises(ValueError, match="gap"):
        detector.push(image, timestamp_ms=500)


def test_runtime_interrupts_gap_window_and_accepts_fresh_pixel_motion():
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "recording-v1"
    runtime._visual_motion_config = VisualMotionConfig()
    runtime._calibration = _calibration()
    runtime._recording_motion = RecordingVisualMotion(
        runtime._calibration, runtime._visual_motion_config, still_duration_ms=400
    )
    runtime._visual_boundary_detector = VisualShotBoundaryDetector(
        ShotWindowConfig(8, 2, 400), motion_confirmation_window_ms=400
    )
    runtime._event_channel = None
    runtime._visual_previous_bgr = None
    runtime._apply_boundary = lambda boundary, events: events.append(boundary)
    events = []
    for index in range(9):
        image = np.full((100, 200, 3), 80, dtype=np.uint8)
        cv2.circle(image, (30 + 5 * index, 50), 6, (240, 240, 240), -1)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        timestamp = index * 17 if index < 6 else 600 + (index - 6) * 17
        runtime._handle_visual_boundary(
            CaptureFrame("epoch", index, timestamp * 1_000_000, 200, 100, encoded.tobytes()), events
        )
    assert [event.kind.value for event in events] == [
        "SHOT_STARTED",
        "SHOT_INTERRUPTED",
        "SHOT_STARTED",
    ]
    assert events[1].reason == "recording_motion_sample_gap"
    assert not events[1].window.complete
