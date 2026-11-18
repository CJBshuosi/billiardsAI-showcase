from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_anchor import (
    AnchorReplayCoordinator,
    AnchorSchedule,
    HeavyAnchorWorker,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_refinement import RealtimeShotRefiner
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorBall,
    AnchorObservation,
    RealtimeTrackerConfig,
    TrackerHealthState,
)
from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate


def _image(ball_x: int) -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (ball_x, 60), 8, (240, 240, 240), -1)
    cv2.line(image, (ball_x - 5, 60), (ball_x + 5, 60), (10, 10, 10), 2)
    cv2.line(image, (ball_x, 55), (ball_x, 65), (10, 10, 10), 2)
    return image


def _capture(frame_id: int) -> CaptureFrame:
    ok, encoded = cv2.imencode(".jpg", _image(40 + frame_id * 2))
    assert ok
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 16_666_667,
        width=160,
        height=120,
        mjpeg=encoded.tobytes(),
    )


class _BlockingBackend:
    package_identity = VisionPackageIdentity(package_id="a" * 64, manifest_sha256="b" * 64)

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def infer(self, frame: object) -> VisionFrameObservation:
        self._release.wait(timeout=2)
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=frame.pixel_sha256,
            package=self.package_identity,
            threshold=0.25,
            detections=(),
            runtime_ref="fake/1",
        )


class _DetectOneBackend:
    package_identity = VisionPackageIdentity(package_id="d" * 64, manifest_sha256="e" * 64)

    def infer(self, frame: object) -> VisionFrameObservation:
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=frame.pixel_sha256,
            package=self.package_identity,
            threshold=0.25,
            detections=(
                BallInstanceObservation(
                    detection_id="ball-one",
                    bbox=PixelBox(left=30, top=50, right=50, bottom=70),
                    center=PixelPoint(x=40, y=60),
                    confidence=0.96,
                    class_label="ball",
                ),
            ),
            runtime_ref="fake/one",
        )


class _ThreadRecordingClassifier:
    def __init__(self) -> None:
        self.thread_name: str | None = None

    def classify(self, *, image: np.ndarray, detections: tuple[object, ...]):
        self.thread_name = threading.current_thread().name
        return {"ball-one": {"group": "solid", "confidence": 0.9}}


class _CountingClassifier:
    def __init__(self) -> None:
        self.frame_markers: list[int] = []

    def classify(self, *, image: np.ndarray, detections: tuple[object, ...]):
        self.frame_markers.append(int(image[0, 0, 0]))
        return {"ball-one": {"group": "cue", "confidence": 0.9}}


def test_scheduler_uses_four_fps_idle_and_eight_fps_during_motion() -> None:
    schedule = AnchorSchedule(source_fps=60, idle_fps=4, active_fps=8)

    assert [index for index in range(60) if schedule.due(index, active=False)] == [
        0,
        15,
        30,
        45,
    ]
    assert [index for index in range(60) if schedule.due(index, active=True)] == [
        0,
        8,
        15,
        23,
        30,
        38,
        45,
        53,
    ]


def test_slow_anchor_worker_never_paces_capture_and_reports_skips() -> None:
    release = threading.Event()
    worker = HeavyAnchorWorker(
        _BlockingBackend(release),
        AnchorSchedule(source_fps=60, idle_fps=4, active_fps=8),
        maximum_pending=1,
    )
    started = time.perf_counter()

    accepted = [worker.submit(_capture(index), active=True) for index in range(60)]

    assert time.perf_counter() - started < 0.5
    assert sum(accepted) < 8
    assert worker.health.skipped_scheduled_frames == 8 - sum(accepted)
    release.set()
    worker.close(timeout_seconds=2)


def test_full_anchor_queue_replaces_pending_work_with_latest_scheduled_frame() -> None:
    release = threading.Event()
    worker = HeavyAnchorWorker(
        _BlockingBackend(release),
        AnchorSchedule(source_fps=60, idle_fps=4, active_fps=8),
        maximum_pending=1,
    )

    assert worker.submit(_capture(0), active=True)
    deadline = time.monotonic() + 1
    while worker._pending.qsize() and time.monotonic() < deadline:  # type: ignore[attr-defined]
        time.sleep(0.005)
    assert worker.submit(_capture(8), active=True)
    assert worker.submit(_capture(15), active=True) is False

    release.set()
    deadline = time.monotonic() + 2
    results = ()
    while len(results) < 2 and time.monotonic() < deadline:
        results += worker.poll()
        time.sleep(0.005)
    worker.close(timeout_seconds=2)

    assert tuple(row.capture_frame.capture_frame_id for row in results) == (0, 15)
    assert worker.health.skipped_scheduled_frames == 1


def test_identity_refresh_is_cadenced_independently_from_detector_anchors() -> None:
    classifier = _CountingClassifier()
    worker = HeavyAnchorWorker(
        _DetectOneBackend(),
        identity_classifier=classifier,
        identity_refresh_interval_ns=1_000_000_000,
    )

    results = []
    for frame_id in (0, 15, 60):
        assert worker.submit(_capture(frame_id), active=False)
        deadline = time.monotonic() + 2
        while len(results) < (1 if frame_id == 0 else 2 if frame_id == 15 else 3):
            results.extend(worker.poll())
            assert time.monotonic() < deadline
            time.sleep(0.005)
    worker.close(timeout_seconds=2)

    assert tuple(bool(row.identities) for row in results) == (True, False, True)
    assert len(classifier.frame_markers) == 2
    assert worker.health.identity_classified_anchors == 2
    assert worker.health.p50_inference_ms is not None
    assert worker.health.p95_inference_ms is not None
    assert worker.health.maximum_inference_ms is not None


def test_identity_classification_runs_inside_heavy_anchor_worker() -> None:
    classifier = _ThreadRecordingClassifier()
    worker = HeavyAnchorWorker(
        _DetectOneBackend(),
        identity_classifier=classifier,
    )

    assert worker.submit(_capture(0), active=False)
    deadline = time.monotonic() + 2
    results = ()
    while not results and time.monotonic() < deadline:
        results = worker.poll()
        time.sleep(0.005)
    worker.close(timeout_seconds=2)

    assert len(results) == 1
    assert results[0].identities["ball-one"]["group"] == "solid"
    assert classifier.thread_name == "mainline1-heavy-anchor"


def test_late_anchor_is_replayed_to_current_frame_before_state_changes() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            minimum_features=4,
            ambiguity_distance_px=12,
        ),
    )
    for frame_id in range(5):
        coordinator.retain(_capture(frame_id))
    anchor = AnchorObservation(
        capture_frame_id=1,
        captured_at_ns=16_666_667,
        balls=(
            AnchorBall(
                track_id="ball-0",
                center_px=(42, 60),
                confidence=0.96,
                group=BallGroup.SOLID,
                bbox_xyxy_px=(32, 50, 52, 70),
            ),
        ),
    )

    assert coordinator.accept(anchor) is True
    assert coordinator.current is not None
    assert coordinator.current.capture_frame_id == 4
    assert abs((coordinator.current.balls[0].center_px or (0, 0))[0] - 48.0) <= 1.5
    assert coordinator.current.balls[0].anchor_frame_id == 1


def test_model_latency_does_not_count_as_live_tracker_queue_lag() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            minimum_features=4,
            maximum_source_lag_frames=2,
        ),
    )
    for frame_id in range(10):
        coordinator.retain(_capture(frame_id))

    accepted = coordinator.accept(
        AnchorObservation(
            capture_frame_id=0,
            captured_at_ns=0,
            balls=(AnchorBall("ball-0", (40, 60), 0.96),),
        )
    )

    assert accepted is True
    assert coordinator.current is not None
    assert coordinator.current.capture_frame_id == 9


@pytest.mark.parametrize("recovered_id", ["ball-0", "replacement"])
def test_fresh_anchor_replay_cannot_erase_roster_loss_from_rule_coverage(
    recovered_id: str,
) -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            enforce_processing_budget=False,
        ),
    )
    frames = tuple(_capture(frame_id) for frame_id in range(5))
    anchors = (
        AnchorObservation(0, 0, (AnchorBall("ball-0", (40, 60), 0.96),)),
        AnchorObservation(2, frames[2].captured_at_ns, ()),
        AnchorObservation(4, frames[4].captured_at_ns, (AnchorBall(recovered_id, (48, 60), 0.96),)),
    )
    try:
        for frame in frames:
            coordinator.retain(frame)
        for anchor in anchors:
            assert coordinator.accept(anchor)

        history = coordinator.tracked_history
        assert history[2].balls == ()
        assert history[4].balls[0].track_id == recovered_id
        result = RealtimeShotRefiner().refine(
            interval=SimpleNamespace(
                start_frame_id=0,
                end_frame_id=4,
                frame_count=5,
                segment_sha256={"unit-segment": "a" * 64},
                frames=lambda: iter(frames),
            ),
            tracked=history,
            anchors=anchors,
        )
        assert (2, "ball-0", "lost") in result.coverage.uncertain_track_states
        assert result.fact_coverage["first_contact"].reason == "tracker_uncertain"
        assert result.fact_coverage["cushion_contacts"].reason == "tracker_uncertain"
        if recovered_id == "replacement":
            assert (4, recovered_id, "ambiguous") in result.coverage.uncertain_track_states
    finally:
        coordinator.close(timeout_seconds=1)


def test_replay_rejects_anchor_when_history_has_a_gap() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(working_width=160, working_height=120),
    )
    coordinator.retain(_capture(1))
    coordinator.retain(_capture(3))
    anchor = AnchorObservation(
        capture_frame_id=1,
        captured_at_ns=16_666_667,
        balls=(),
    )

    assert coordinator.accept(anchor) is False
    assert coordinator.health.unreplayable_anchors == 1


def test_replay_tracks_across_gaps_from_an_accounted_coalescing_source() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            allow_accounted_frame_gaps=True,
        ),
    )
    coordinator.retain(_capture(1))
    coordinator.retain(_capture(3))
    anchor = AnchorObservation(
        capture_frame_id=1,
        captured_at_ns=16_666_667,
        balls=(AnchorBall("ball-0", (42, 60), 0.96),),
    )

    assert coordinator.accept(anchor) is True
    assert coordinator.current is not None
    assert coordinator.current.capture_frame_id == 3
    assert coordinator.current.balls[0].state == "confirmed"
    assert coordinator.current.balls[0].center_px is not None
    coordinator.close(timeout_seconds=1)


def test_accounted_gap_beyond_safe_flow_window_revokes_stale_tracker() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            allow_accounted_frame_gaps=True,
            maximum_accounted_frame_gap=2,
        ),
    )
    coordinator.retain(_capture(0))
    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=0,
            captured_at_ns=0,
            balls=(AnchorBall("ball-0", (40, 60), 0.96),),
        )
    )

    coordinator.retain(_capture(3))
    assert coordinator.wait_until_preprocessed(3, timeout_seconds=1)

    assert coordinator.current is None
    assert coordinator.tracker_health is None
    coordinator.close(timeout_seconds=1)


def test_later_anchor_replay_reuses_retained_tracker_grayscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(working_width=160, working_height=120),
    )
    for frame_id in range(8):
        coordinator.retain(_capture(frame_id))
        if frame_id == 4:
            assert coordinator.accept(
                AnchorObservation(
                    capture_frame_id=1,
                    captured_at_ns=16_666_667,
                    balls=(AnchorBall("ball-0", (42, 60), 0.96),),
                )
            )

    decode_calls = 0
    original = CaptureFrame.decode_working_gray

    def counted(self: CaptureFrame, *, width: int, height: int) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original(self, width=width, height=height)

    monkeypatch.setattr(CaptureFrame, "decode_working_gray", counted)

    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=5,
            captured_at_ns=5 * 16_666_667,
            balls=(AnchorBall("ball-0", (50, 60), 0.96),),
        )
    )
    decode_calls_after_catchup = decode_calls
    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=6,
            captured_at_ns=6 * 16_666_667,
            balls=(AnchorBall("ball-0", (52, 60), 0.96),),
        )
    )
    assert decode_calls == decode_calls_after_catchup


def test_first_late_anchor_reuses_grayscale_decoded_once_at_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(working_width=160, working_height=120),
    )
    decode_calls = 0
    original = CaptureFrame.decode_working_gray

    def counted(self: CaptureFrame, *, width: int, height: int) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original(self, width=width, height=height)

    monkeypatch.setattr(CaptureFrame, "decode_working_gray", counted)
    for frame_id in range(8):
        coordinator.retain(_capture(frame_id))

    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=1,
            captured_at_ns=16_666_667,
            balls=(AnchorBall("ball-0", (42, 60), 0.96),),
        )
    )
    assert decode_calls == 8
    coordinator.close(timeout_seconds=1)


def test_replay_retention_drops_full_mjpeg_after_working_decode() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(working_width=160, working_height=120),
    )
    frame = _capture(0)
    try:
        coordinator.retain(frame)
        assert coordinator.wait_until_preprocessed(0, timeout_seconds=1)
        retained = coordinator._frames[0]  # type: ignore[attr-defined]
        assert len(retained.source.mjpeg) < len(frame.mjpeg)
        assert retained.working_gray.shape == (120, 160)
    finally:
        coordinator.close(timeout_seconds=1)


def test_async_anchor_replay_does_not_block_capture_and_catches_current_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            processing_budget_ms=100,
        ),
    )
    for frame_id in range(10):
        coordinator.retain(_capture(frame_id))
    original = coordinator._new_tracker  # type: ignore[attr-defined]

    def slow_tracker():
        tracker = original()
        original_track = tracker.track_working

        def slow_track(*args: object, **kwargs: object):
            time.sleep(0.01)
            return original_track(*args, **kwargs)

        monkeypatch.setattr(tracker, "track_working", slow_track)
        return tracker

    monkeypatch.setattr(coordinator, "_new_tracker", slow_tracker)
    anchor = AnchorObservation(
        capture_frame_id=0,
        captured_at_ns=0,
        balls=(AnchorBall("ball-0", (40, 60), 0.96),),
    )

    started = time.monotonic()
    assert coordinator.submit(anchor) is True
    assert time.monotonic() - started < 0.02
    for frame_id in range(10, 13):
        coordinator.retain(_capture(frame_id))
    deadline = time.monotonic() + 2
    accepted = ()
    while not accepted and time.monotonic() < deadline:
        accepted = coordinator.poll_accepted()
        time.sleep(0.005)

    coordinator.close(timeout_seconds=2)
    assert accepted == (anchor,)
    assert coordinator.current is not None
    assert coordinator.current.capture_frame_id == 12


class _OneFrameSpikeClock:
    def __init__(self) -> None:
        self._values = iter((0, 20_000_000))

    def __call__(self) -> int:
        return next(self._values, 20_000_000)


def test_replay_evicts_observations_with_frames_and_late_anchor_cannot_restore_them() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            history_frames=3,
            perf_counter_ns=lambda: 0,
        ),
    )
    try:
        coordinator.retain(_capture(0))
        assert coordinator.accept(AnchorObservation(0, 0, ()))
        for frame_id in range(1, 20):
            coordinator.retain(_capture(frame_id))
            assert coordinator.wait_until_preprocessed(frame_id, timeout_seconds=1)
            expected = list(range(max(0, frame_id - 2), frame_id + 1))
            assert [row.capture_frame_id for row in coordinator.tracked_history] == expected
            # Check retained storage, not merely the filtered public projection.
            assert sorted(coordinator._tracked) == expected
        assert coordinator.accept(AnchorObservation(18, 18 * 16_666_667, ()))
        assert sorted(coordinator._tracked) == [17, 18, 19]
        assert not coordinator.accept(AnchorObservation(0, 0, ()))
        assert sorted(coordinator._tracked) == [17, 18, 19]
    finally:
        coordinator.close(timeout_seconds=1)


def test_idle_tracker_budget_spike_waits_for_next_anchor_without_stopping_capture() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            processing_budget_ms=16.7,
            perf_counter_ns=_OneFrameSpikeClock(),
        ),
    )
    coordinator.retain(_capture(0))
    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=0,
            captured_at_ns=0,
            balls=(AnchorBall("ball-0", (40, 60), 0.96),),
        )
    )

    coordinator.retain(_capture(1))
    coordinator.retain(_capture(2))

    assert coordinator.accept(
        AnchorObservation(
            capture_frame_id=2,
            captured_at_ns=2 * 16_666_667,
            balls=(AnchorBall("ball-0", (44, 60), 0.96),),
        )
    )

    assert coordinator.tracker_health is not None
    assert coordinator.tracker_health.state is TrackerHealthState.RUNNING
    assert coordinator.current is not None
    assert coordinator.current.capture_frame_id == 2
    coordinator.close(timeout_seconds=1)


def test_degraded_replay_candidate_is_not_installed_as_current_tracker() -> None:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    coordinator = AnchorReplayCoordinator(
        calibration,
        RealtimeTrackerConfig(
            working_width=160,
            working_height=120,
            processing_budget_ms=16.7,
            perf_counter_ns=_OneFrameSpikeClock(),
        ),
    )
    coordinator.retain(_capture(0))
    coordinator.retain(_capture(1))

    accepted = coordinator.accept(
        AnchorObservation(
            capture_frame_id=0,
            captured_at_ns=0,
            balls=(AnchorBall("ball-0", (40, 60), 0.96),),
        )
    )

    assert accepted is False
    assert coordinator.current is None
    assert coordinator.health.unreplayable_anchors == 1
    coordinator.close(timeout_seconds=1)
