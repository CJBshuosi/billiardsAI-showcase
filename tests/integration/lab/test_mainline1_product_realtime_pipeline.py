from __future__ import annotations

import hashlib
import importlib
import json
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.field_assistance.visual_motion import VisualMotionConfig
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeRuntime,
    ProductRealtimeRuntimeConfig,
    _detect_unique_cue_ball,
)
from billiards_ai.lab.mainline1_realtime_anchor import HeavyAnchorResult
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureFrame,
    NativeCaptureIdentity,
    RealtimeCaptureConfig,
)
from billiards_ai.lab.mainline1_realtime_tracker import (
    TrackedBall,
    TrackedObservation,
    TrackerHealthState,
)
from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.shot_local.cue import CueTrackingConfig
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.fragments import FragmentTrackingConfig
from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate


class _VirtualSixtyFpsCapture:
    identity = NativeCaptureIdentity(
        device_locator="virtual://rer-u3",
        native_handle_ref="virtual-handle-rer-u3",
        durable_fingerprint="c" * 64,
    )

    def __init__(self, frames: int) -> None:
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        self._payload = encoded.tobytes()
        self._frames = frames
        self.cancelled = False

    def frames(self, stop: threading.Event):
        for frame_id in range(self._frames):
            if stop.is_set():
                return
            yield CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=frame_id,
                captured_at_ns=frame_id * 16_666_667,
                width=16,
                height=12,
                mjpeg=self._payload,
            )

    def cancel(self) -> None:
        self.cancelled = True


class _PrimaryAndCancelFailureCapture(_VirtualSixtyFpsCapture):
    def frames(self, stop: threading.Event):
        raise RuntimeError("primary capture failure")
        yield

    def cancel(self) -> None:
        raise RuntimeError("cleanup cancellation failure")


class _ReportedLagCapture(_VirtualSixtyFpsCapture):
    @property
    def pending_frame_count(self) -> int:
        return 3


class _AccountedCoalescedCapture(_VirtualSixtyFpsCapture):
    allow_accounted_frame_gaps = True

    def frames(self, stop: threading.Event):
        for frame_id in (0, 3):
            if stop.is_set():
                return
            yield CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=frame_id,
                captured_at_ns=frame_id * 16_666_667,
                width=16,
                height=12,
                mjpeg=self._payload,
            )


class _RealtimeQueuedCapture(_VirtualSixtyFpsCapture):
    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self._queue: queue.Queue[CaptureFrame | None] = queue.Queue(maxsize=64)
        self._cancel = threading.Event()
        self._producer: threading.Thread | None = None

    @property
    def pending_frame_count(self) -> int:
        return sum(item is not None for item in tuple(self._queue.queue))

    def frames(self, stop: threading.Event):
        started = time.monotonic()

        def produce() -> None:
            for frame_id in range(self._frames):
                deadline = started + frame_id / 60
                while not self._cancel.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.002))
                if self._cancel.is_set():
                    return
                self._queue.put(
                    CaptureFrame(
                        source_epoch_id="epoch-1",
                        capture_frame_id=frame_id,
                        captured_at_ns=frame_id * 16_666_667,
                        width=16,
                        height=12,
                        mjpeg=self._payload,
                    )
                )
            self._queue.put(None)

        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()
        while not stop.is_set():
            item = self._queue.get(timeout=1)
            if item is None:
                return
            yield item

    def cancel(self) -> None:
        super().cancel()
        self._cancel.set()
        if self._producer is not None:
            self._producer.join(timeout=1)


class _SlowEmptyDetector:
    package_identity = VisionPackageIdentity(package_id="a" * 64, manifest_sha256="b" * 64)

    def infer(self, frame: object) -> VisionFrameObservation:
        time.sleep(0.002)
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=frame.pixel_sha256,
            package=self.package_identity,
            threshold=0.25,
            detections=(),
            runtime_ref="virtual-eight-fps/1",
        )


class _BackloggedEmptyDetector(_SlowEmptyDetector):
    def infer(self, frame: object) -> VisionFrameObservation:
        time.sleep(0.02)
        return super().infer(frame)


class _CancelReleasedDetector(_SlowEmptyDetector):
    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def infer(self, frame: object) -> VisionFrameObservation:
        self._release.wait(timeout=2)
        time.sleep(0.6)
        return super().infer(frame)


class _ReleaseOnCancelCapture(_VirtualSixtyFpsCapture):
    def __init__(self, release: threading.Event) -> None:
        super().__init__(frames=1)
        self._release = release

    def cancel(self) -> None:
        super().cancel()
        self._release.set()


class _ShotCapture:
    identity = NativeCaptureIdentity(
        device_locator="virtual://shot-camera",
        native_handle_ref="virtual-handle-shot-camera",
        durable_fingerprint="d" * 64,
    )

    def __init__(self, frames: int = 52) -> None:
        self.cancelled = False
        self._frames = frames
        self.emitted_frames = 0

    @staticmethod
    def _image(frame_id: int) -> np.ndarray:
        cue_x = 40 if frame_id <= 10 else min(60, 40 + (frame_id - 10) * 2)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        for x, shade in ((cue_x, 245), (105, 180)):
            cv2.circle(image, (x, 60), 8, (shade, shade, shade), -1)
            cv2.line(image, (x - 5, 60), (x + 5, 60), (10, 10, 10), 2)
            cv2.line(image, (x, 55), (x, 65), (10, 10, 10), 2)
        return image

    def frames(self, stop: threading.Event):
        for frame_id in range(self._frames):
            if stop.is_set():
                return
            self.emitted_frames += 1
            ok, encoded = cv2.imencode(".jpg", self._image(frame_id))
            assert ok
            yield CaptureFrame(
                source_epoch_id="shot-epoch",
                capture_frame_id=frame_id,
                captured_at_ns=frame_id * 16_666_667,
                width=160,
                height=120,
                mjpeg=encoded.tobytes(),
            )
            time.sleep(0.001)

    def cancel(self) -> None:
        self.cancelled = True


class _QueuedShotCapture(_ShotCapture):
    def __init__(self, frames: int) -> None:
        super().__init__(frames)
        self._queue: queue.Queue[CaptureFrame | None] = queue.Queue(maxsize=64)
        self._cancel = threading.Event()
        self._producer: threading.Thread | None = None
        self.produced_frames = 0
        self.maximum_pending_frame_count = 0

    @property
    def pending_frame_count(self) -> int:
        return sum(item is not None for item in tuple(self._queue.queue))

    def frames(self, stop: threading.Event):
        started = time.monotonic()

        def produce() -> None:
            for frame_id in range(self._frames):
                deadline = started + frame_id / 60
                while not self._cancel.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.002))
                if self._cancel.is_set():
                    return
                ok, encoded = cv2.imencode(".jpg", self._image(frame_id))
                assert ok
                self._queue.put(
                    CaptureFrame(
                        source_epoch_id="shot-epoch",
                        capture_frame_id=frame_id,
                        captured_at_ns=frame_id * 16_666_667,
                        width=160,
                        height=120,
                        mjpeg=encoded.tobytes(),
                    )
                )
                self.produced_frames += 1
                self.maximum_pending_frame_count = max(
                    self.maximum_pending_frame_count,
                    self.pending_frame_count,
                )
            self._queue.put(None)

        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()
        while not stop.is_set():
            item = self._queue.get(timeout=1)
            if item is None:
                return
            self.emitted_frames += 1
            yield item

    def cancel(self) -> None:
        super().cancel()
        self._cancel.set()
        if self._producer is not None:
            self._producer.join(timeout=1)


class _ShotDetector:
    package_identity = VisionPackageIdentity(package_id="e" * 64, manifest_sha256="f" * 64)

    def infer(self, frame: object) -> VisionFrameObservation:
        frame_id = frame.frame_id
        cue_x = 40 if frame_id <= 10 else min(60, 40 + (frame_id - 10) * 2)
        detections = tuple(
            BallInstanceObservation(
                detection_id=f"{name}-{frame_id}",
                bbox=PixelBox(left=x - 8, top=52, right=x + 8, bottom=68),
                center=PixelPoint(x=x, y=60),
                confidence=0.96,
                class_label="ball",
            )
            for name, x in (("cue", cue_x), ("target", 105))
        )
        return VisionFrameObservation(
            frame_id=frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            package=self.package_identity,
            threshold=0.25,
            detections=detections,
            runtime_ref="virtual-shot/1",
        )


class _Classifier:
    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> Mapping[str, Mapping[str, object]]:
        return {
            detection.detection_id: {
                "group": "cue" if detection.detection_id.startswith("cue") else "solid",
                "confidence": 0.96,
            }
            for detection in detections
        }


def test_detector_only_anchor_inherits_matched_trusted_group() -> None:
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._next_track_id = 1
    runtime._calibration = calibrate(
        "camera-1",
        [(0, 120), (160, 120), (160, 0), (0, 0)],
    )
    runtime._replay = SimpleNamespace(
        current=TrackedObservation(
            capture_frame_id=14,
            captured_at_ns=233_333_338,
            balls=(
                TrackedBall(
                    track_id="cue-1",
                    group=BallGroup.CUE,
                    state="confirmed",
                    provenance="flow",
                    center_px=(40.0, 60.0),
                    table_mm=(400, 600),
                    confidence=0.9,
                    anchor_frame_id=0,
                ),
            ),
        )
    )
    frame = CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=15,
        captured_at_ns=250_000_005,
        width=160,
        height=120,
        mjpeg=b"unused",
    )
    observation = VisionFrameObservation(
        frame_id=15,
        timestamp_ms=250,
        width_px=160,
        height_px=120,
        pixel_sha256="a" * 64,
        package=_ShotDetector.package_identity,
        threshold=0.25,
        detections=(
            BallInstanceObservation(
                detection_id="ball-15",
                bbox=PixelBox(left=34, top=52, right=50, bottom=68),
                center=PixelPoint(x=42, y=60),
                confidence=0.96,
                class_label="ball",
            ),
        ),
        runtime_ref="test/1",
    )

    anchor = runtime._to_anchor(
        HeavyAnchorResult(
            capture_frame=frame,
            observation=observation,
            inference_duration_ms=1.0,
            active=True,
            identities={},
        )
    )

    assert anchor.balls[0].track_id == "cue-1"
    assert anchor.balls[0].group is BallGroup.CUE


def test_unique_unmatched_white_ball_supplements_a_missing_cue_detection() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 30)
    cv2.circle(image, (90, 90), 13, (235, 235, 235), -1)
    cv2.circle(image, (86, 87), 2, (20, 20, 210), -1)
    cv2.circle(image, (94, 93), 2, (20, 20, 210), -1)
    cv2.circle(image, (220, 90), 13, (30, 30, 220), -1)
    calibration = calibrate(
        "camera-1",
        [(20, 160), (300, 160), (300, 20), (20, 20)],
    )
    detections = (
        BallInstanceObservation(
            detection_id="object",
            bbox=PixelBox(left=207, top=77, right=233, bottom=103),
            center=PixelPoint(x=220, y=90),
            confidence=0.95,
            class_label="ball",
        ),
    )

    candidate = _detect_unique_cue_ball(image, detections, calibration)

    assert candidate is not None
    assert candidate.center_px == pytest.approx((90, 90), abs=2)


def test_cue_supplement_abstains_when_two_unmatched_white_balls_are_plausible() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 30)
    cv2.circle(image, (90, 90), 13, (235, 235, 235), -1)
    cv2.circle(image, (150, 90), 13, (235, 235, 235), -1)
    cv2.circle(image, (220, 90), 13, (30, 30, 220), -1)
    calibration = calibrate(
        "camera-1",
        [(20, 160), (300, 160), (300, 20), (20, 20)],
    )
    detections = (
        BallInstanceObservation(
            detection_id="object",
            bbox=PixelBox(left=207, top=77, right=233, bottom=103),
            center=PixelPoint(x=220, y=90),
            confidence=0.95,
            class_label="ball",
        ),
    )

    assert _detect_unique_cue_ball(image, detections, calibration) is None


def test_anchor_filters_rail_false_positive_and_adds_unique_missing_cue() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 30)
    cv2.circle(image, (90, 90), 13, (235, 235, 235), -1)
    cv2.circle(image, (86, 87), 2, (20, 20, 210), -1)
    cv2.circle(image, (94, 93), 2, (20, 20, 210), -1)
    cv2.circle(image, (220, 90), 13, (30, 30, 220), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._next_track_id = 0
    runtime._calibration = calibrate(
        "camera-1",
        [(20, 160), (300, 160), (300, 20), (20, 20)],
    )
    runtime._replay = SimpleNamespace(current=None)
    detections = (
        BallInstanceObservation(
            detection_id="rail",
            bbox=PixelBox(left=0, top=77, right=16, bottom=103),
            center=PixelPoint(x=8, y=90),
            confidence=0.70,
            class_label="ball",
        ),
        BallInstanceObservation(
            detection_id="object",
            bbox=PixelBox(left=207, top=77, right=233, bottom=103),
            center=PixelPoint(x=220, y=90),
            confidence=0.95,
            class_label="ball",
        ),
    )
    observation = VisionFrameObservation(
        frame_id=1,
        timestamp_ms=16,
        width_px=320,
        height_px=180,
        pixel_sha256="a" * 64,
        package=_ShotDetector.package_identity,
        threshold=0.25,
        detections=detections,
        runtime_ref="test/1",
    )

    anchor = runtime._to_anchor(
        HeavyAnchorResult(
            capture_frame=CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=1,
                captured_at_ns=16_666_667,
                width=320,
                height=180,
                mjpeg=encoded.tobytes(),
            ),
            observation=observation,
            inference_duration_ms=1.0,
            active=False,
            identities={"object": {"group": "solid", "confidence": 0.95}},
        )
    )

    assert len(anchor.balls) == 2
    assert {ball.group for ball in anchor.balls} == {BallGroup.CUE, BallGroup.SOLID}


def test_anchor_relabels_detected_unique_white_ball_as_cue() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[:] = (30, 110, 30)
    cv2.circle(image, (90, 90), 13, (235, 235, 235), -1)
    cv2.circle(image, (86, 87), 2, (20, 20, 210), -1)
    cv2.circle(image, (94, 93), 2, (20, 20, 210), -1)
    cv2.circle(image, (220, 90), 13, (30, 30, 220), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._next_track_id = 0
    runtime._calibration = calibrate(
        "camera-1",
        [(20, 160), (300, 160), (300, 20), (20, 20)],
    )
    runtime._replay = SimpleNamespace(current=None)
    detections = (
        BallInstanceObservation(
            detection_id="white-ball",
            bbox=PixelBox(left=77, top=77, right=103, bottom=103),
            center=PixelPoint(x=90, y=90),
            confidence=0.94,
            class_label="ball",
        ),
        BallInstanceObservation(
            detection_id="object",
            bbox=PixelBox(left=207, top=77, right=233, bottom=103),
            center=PixelPoint(x=220, y=90),
            confidence=0.95,
            class_label="ball",
        ),
    )
    observation = VisionFrameObservation(
        frame_id=1,
        timestamp_ms=16,
        width_px=320,
        height_px=180,
        pixel_sha256="a" * 64,
        package=_ShotDetector.package_identity,
        threshold=0.25,
        detections=detections,
        runtime_ref="test/1",
    )

    anchor = runtime._to_anchor(
        HeavyAnchorResult(
            capture_frame=CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=1,
                captured_at_ns=16_666_667,
                width=320,
                height=180,
                mjpeg=encoded.tobytes(),
            ),
            observation=observation,
            inference_duration_ms=1.0,
            active=False,
            identities={
                "white-ball": {"group": "unknown", "confidence": 0.90},
                "object": {"group": "solid", "confidence": 0.95},
            },
        )
    )

    assert len(anchor.balls) == 2
    groups_by_x = {round(ball.center_px[0]): ball.group for ball in anchor.balls}
    assert groups_by_x == {90: BallGroup.CUE, 220: BallGroup.SOLID}


def _shot_config() -> Mainline1ShotLocalConfig:
    return Mainline1ShotLocalConfig(
        window=ShotWindowConfig(8, 2, 100),
        cue=CueTrackingConfig(0.08, 0.55),
        fragment=FragmentTrackingConfig(5, 80, 3),
        event=ShotLocalEventConfig(
            first_contact_distance_mm=70,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=120,
            cushion_distance_mm=20,
            pocket_distance_mm=30,
            pot_missing_frames=3,
            after_match_distance_mm=50,
            table_length_mm=1590,
            table_width_mm=1190,
        ),
    )


def _job_options(
    tmp_path: Path,
    *,
    job_store: object | None = None,
) -> dict[str, object]:
    return {
        "job_store": (
            SQLiteShotAnalysisJobStore(tmp_path / "analysis-jobs.sqlite3")
            if job_store is None
            else job_store
        ),
        "job_media_root": tmp_path / "analysis-media",
        "quarantine_root": tmp_path / "analysis-quarantine",
        "session_id": "session-0001",
        "capture_setup_id": "setup-1",
        "calibration_artifact_id": "calibration-1",
        "config_sha256": "1" * 64,
    }


def _runtime_failure_diagnostics(runtime, events) -> str:
    """Keep asynchronous failure evidence bounded and visible in CI assertions."""
    diagnostics = asdict(runtime.diagnostics)
    # Per-shot coverage can contain every source frame. Keep the lifecycle and
    # health evidence readable instead of truncating it behind that large blob.
    diagnostics.pop("first_shot_diagnostics", None)
    diagnostics["accepted_anchor_frame_ids"] = diagnostics["accepted_anchor_frame_ids"][-16:]
    return json.dumps(
        {
            "events": [
                {
                    "kind": event.kind,
                    "frame_id": event.frame_id,
                    "reason": None if event.reason is None else event.reason[:512],
                }
                for event in events[-16:]
            ],
            "capture": asdict(runtime.capture_health),
            "anchor": asdict(runtime.anchor_health),
            "replay": asdict(runtime.anchor_replay_health),
            "tracker": (None if runtime.tracker_health is None else asdict(runtime.tracker_health)),
            "diagnostics": diagnostics,
        },
        default=str,
        sort_keys=True,
    )[:16_384]


def test_sixty_fps_capture_is_not_paced_by_eight_fps_detector(tmp_path) -> None:
    source = _VirtualSixtyFpsCapture(frames=36_000)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_BackloggedEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=120,
            post_roll_frames=120,
            maximum_shot_frames=1800,
            maximum_ring_bytes=64 * 1024 * 1024,
            maximum_pending_ring_frames=36_001,
            tracker_width=16,
            tracker_height=12,
        ),
    )

    runtime.run()

    assert runtime.capture_health.received_frames == 36_000
    assert runtime.capture_health.measured_fps == pytest.approx(60.0, abs=0.01)
    assert runtime.maximum_capture_queue_lag_frames <= 2
    assert runtime.maximum_detector_queue_depth <= 1
    assert runtime.anchor_health.skipped_scheduled_frames > 0
    assert source.cancelled is True


def test_diagnostic_duration_stops_capture_without_operator_interrupt(tmp_path) -> None:
    source = _VirtualSixtyFpsCapture(frames=600)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            maximum_pending_ring_frames=600,
            tracker_width=16,
            tracker_height=12,
            diagnostic_duration_seconds=0.1,
        ),
    )
    assert runtime._replay._tracker_config.history_frames == 720  # type: ignore[attr-defined]

    events = runtime.run()

    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)
    assert runtime.capture_health.received_frames == 7
    assert runtime.current_capture_frame is not None
    assert runtime.current_capture_frame.capture_frame_id == 6
    assert runtime.current_capture_frame.mjpeg == source._payload
    assert source.cancelled is True


def test_runtime_diagnostics_distinguish_balls_cue_and_motion_threshold(tmp_path) -> None:
    source = _ShotCapture(frames=80)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
        ),
    )

    runtime.run()

    diagnostics = runtime.diagnostics
    assert diagnostics.accepted_anchor_frames > 0
    assert diagnostics.anchor_frames_with_balls > 0
    assert diagnostics.anchor_frames_with_cue_ball > 0
    assert diagnostics.tracked_observation_frames > 0
    assert diagnostics.tracked_frames_with_confirmed_balls > 0
    assert diagnostics.tracked_frames_with_cue_ball > 0
    assert diagnostics.maximum_confirmed_balls == 2
    assert diagnostics.maximum_frame_change_mm >= 8
    assert diagnostics.maximum_frame_change_frame_id is not None
    assert diagnostics.motion_threshold_mm == 8
    assert diagnostics.motion_threshold_crossings > 0
    expected_stages = {
        "capture_health",
        "ring_enqueue",
        "tracker_retain",
        "anchor_transfer",
        "tracking_window",
        "shot_finalize",
        "anchor_schedule",
        "frame_processing",
    }
    assert set(diagnostics.maximum_stage_duration_ms) == expected_stages
    assert set(diagnostics.maximum_stage_frame_id) == expected_stages
    assert all(value >= 0 for value in diagnostics.maximum_stage_duration_ms.values())
    assert diagnostics.maximum_frame_processing_ms >= max(
        diagnostics.maximum_stage_duration_ms[name]
        for name in expected_stages - {"frame_processing"}
    )
    assert diagnostics.maximum_frame_processing_frame_id is not None


def test_sixty_fps_capture_is_not_blocked_by_slow_segment_fsync(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing

    source = _RealtimeQueuedCapture(frames=180)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    original = MjpegSegmentRing._flush_pending

    def slow_flush(ring: MjpegSegmentRing) -> None:
        time.sleep(0.3)
        original(ring)

    monkeypatch.setattr(MjpegSegmentRing, "_flush_pending", slow_flush)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=120,
            post_roll_frames=120,
            maximum_shot_frames=1800,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )

    events = runtime.run()

    assert runtime.capture_health.received_frames == 180
    assert runtime.maximum_capture_queue_lag_frames <= 2
    assert runtime.maximum_ring_queue_depth >= 10
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)


def test_ring_queue_exhaustion_fails_closed_instead_of_discarding_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing

    source = _VirtualSixtyFpsCapture(frames=20)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    original = MjpegSegmentRing.append

    def slow_append(ring: MjpegSegmentRing, frame: CaptureFrame) -> None:
        time.sleep(0.02)
        original(ring, frame)

    monkeypatch.setattr(MjpegSegmentRing, "append", slow_append)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            maximum_pending_ring_frames=2,
            tracker_width=16,
            tracker_height=12,
        ),
    )

    events = runtime.run()

    assert runtime.capture_health.received_frames < 20
    assert any(
        event.kind == "CAPTURE_DEGRADED"
        and event.reason is not None
        and "async ring writer queue capacity exhausted" in event.reason
        for event in events
    )


def test_ring_queue_exhaustion_never_settles_a_shot_without_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing

    source = _ShotCapture(frames=200)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    original = MjpegSegmentRing.append

    def slow_append(ring: MjpegSegmentRing, frame: CaptureFrame) -> None:
        time.sleep(0.02)
        original(ring, frame)

    monkeypatch.setattr(MjpegSegmentRing, "append", slow_append)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            maximum_pending_ring_frames=2,
            tracker_width=160,
            tracker_height=120,
        ),
    )

    events = runtime.run()

    assert not any(event.kind == "SHOT_SETTLED" for event in events)
    assert any(
        event.kind == "CAPTURE_DEGRADED"
        and event.reason is not None
        and "async ring writer queue capacity exhausted" in event.reason
        for event in events
    )


def test_sixty_fps_capture_is_not_blocked_by_slow_grayscale_decode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _RealtimeQueuedCapture(frames=120)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    original = CaptureFrame.decode_working_gray

    def slow_decode(
        frame: CaptureFrame,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        time.sleep(0.03)
        return original(frame, width=width, height=height)

    monkeypatch.setattr(CaptureFrame, "decode_working_gray", slow_decode)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=120,
            post_roll_frames=120,
            maximum_shot_frames=1800,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )
    runtime._active_until_frame_id = 1_000  # type: ignore[attr-defined]

    events = runtime.run()

    assert runtime.capture_health.received_frames == 120, events
    assert runtime.maximum_capture_queue_lag_frames <= 2, _runtime_failure_diagnostics(
        runtime, events
    )
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)


def test_capture_cleanup_failure_does_not_hide_primary_runtime_failure(tmp_path) -> None:
    source = _PrimaryAndCancelFailureCapture(frames=0)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )
    events = runtime.run()

    reasons = [event.reason or "" for event in events if event.kind == "CAPTURE_DEGRADED"]
    assert any("primary capture failure" in reason for reason in reasons)
    assert any("cleanup cancellation failure" in reason for reason in reasons)


def test_capture_stops_before_waiting_for_slow_anchor_worker(tmp_path) -> None:
    release = threading.Event()
    source = _ReleaseOnCancelCapture(release)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_CancelReleasedDetector(release),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )

    events = runtime.run()

    assert source.cancelled is True
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)


def test_native_queue_lag_above_two_frames_does_not_stop_live_capture(tmp_path) -> None:
    source = _ReportedLagCapture(frames=10)
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )
    runtime._active_until_frame_id = 100  # type: ignore[attr-defined]

    events = runtime.run()

    assert runtime.maximum_capture_queue_lag_frames > 2
    assert runtime.capture_health.received_frames == 10
    assert runtime.current_capture_frame is not None
    assert runtime.current_capture_frame.capture_frame_id == 9
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)


def test_validated_coalescing_keeps_analysis_live_while_external_recording_is_complete(
    tmp_path: Path,
) -> None:
    from billiards_ai.lab.mainline1_session_recording import (
        ContinuousSessionRecorder,
        SessionRecordingConfig,
        SessionRecordingEvidenceStore,
    )

    source = _AccountedCoalescedCapture(frames=4)
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "recording",
        session_id="session-1",
        expected_fps=60,
        config=SessionRecordingConfig(
            segment_duration_seconds=0.1,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    for frame_id in range(4):
        recorder.append(
            CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=frame_id,
                captured_at_ns=frame_id * 16_666_667,
                width=16,
                height=12,
                mjpeg=source._payload,
            )
        )
    evidence_store = SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session-1",
        pre_roll_frames=1,
        post_roll_frames=1,
        maximum_shot_frames=60,
        append_frames=False,
        close_recorder=False,
    )
    calibration = calibrate(
        "virtual",
        [(0, 11), (15, 11), (15, 0), (0, 0)],
        table_length_mm=150,
        table_width_mm=110,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        evidence_store=evidence_store,
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "unused-ring",
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=16,
            tracker_height=12,
        ),
    )

    events = runtime.run()
    recorder.close()

    assert runtime.capture_health.received_frames == 2
    assert runtime.capture_health.accounted_frame_gaps == 2
    assert runtime.current_capture_frame is not None
    assert runtime.current_capture_frame.capture_frame_id == 3
    assert runtime.tracker_health is not None
    assert runtime.tracker_health.state is TrackerHealthState.RUNNING
    assert runtime.current_tracking is not None
    assert runtime.current_tracking.capture_frame_id == 3
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in events)


@pytest.mark.parametrize("identity_pair", [None, ("identity-1", "e" * 64)])
def test_settle_enqueues_without_calling_canonical_builders(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    identity_pair,
) -> None:
    from billiards_ai.lab import mainline1_product_realtime_runtime as product_runtime

    def forbidden_call(*args: object, **kwargs: object) -> object:
        raise AssertionError("canonical analysis must not run in the capture thread")

    for name in (
        "build_shot_evidence",
        "build_canonical_shot_analyses",
        "build_mainline1_shot_product",
        "publish_live_shot",
    ):
        monkeypatch.setattr(product_runtime, name, forbidden_call, raising=False)
    source = _ShotCapture(frames=200)
    boundary_ledger = ShotBoundaryLedger(
        tmp_path / "shot-boundaries.sqlite3",
        session_id="session-0001",
        source_epoch_id="shot-epoch",
    )
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        boundary_ledger=boundary_ledger,
        **_job_options(tmp_path),
        **(
            {}
            if identity_pair is None
            else dict(
                identity_training_package_id=identity_pair[0],
                identity_training_package_sha256=identity_pair[1],
            )
        ),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
            stop_after_terminal_shots=1,
            initial_shot_number=7,
        ),
    )

    events = runtime.run()

    shot_events = [event for event in events if event.kind.startswith("SHOT_")]
    assert [event.kind for event in shot_events] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
        "SHOT_ENQUEUED",
    ], _runtime_failure_diagnostics(runtime, events)
    assert source.emitted_frames < 200
    settled = shot_events[1].settled
    assert settled is not None
    assert settled.shot_id == shot_events[0].shot_id
    assert shot_events[2].shot_id == settled.shot_id
    assert shot_events[2].settled is None
    assert shot_events[0].started_ms is not None
    assert shot_events[0].started_ms <= settled.settled_at_ns // 1_000_000
    assert shot_events[2].job.job_id == settled.job_id
    assert shot_events[2].job.source_epoch_id == settled.source_epoch_id
    assert shot_events[2].job.shot_id == settled.shot_id
    assert (
        shot_events[2].job.identity_training_package_id,
        shot_events[2].job.identity_training_package_sha256,
    ) == ((None, None) if identity_pair is None else identity_pair)
    assert settled.shot_id == "shot-0008"
    diagnostic = runtime.diagnostics.first_shot_diagnostics
    assert diagnostic.status == "OBSERVED_METADATA"
    assert diagnostic.shot_id == settled.shot_id
    assert diagnostic.source.count == len(settled.rule_coverage.source_frame_ids)
    assert diagnostic.rule.count == len(settled.rule_frames)
    assert diagnostic.tracker_policy.enforce_processing_budget is True
    assert (
        json.loads(json.dumps(asdict(runtime.diagnostics), allow_nan=False))[
            "first_shot_diagnostics"
        ]["shot_id"]
        == settled.shot_id
    )
    assert (tmp_path / "analysis-media" / settled.job_id / "interval.json").is_file()
    assert [record.kind for record in boundary_ledger.records()] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
    ]


def test_raw_visual_boundary_survives_empty_heavy_detector(tmp_path: Path) -> None:
    """Catches shot boundaries disappearing whenever RF-DETR returns no balls."""
    source = _ShotCapture(frames=200)
    boundary_ledger = ShotBoundaryLedger(
        tmp_path / "shot-boundaries.sqlite3",
        session_id="session-0001",
        source_epoch_id="shot-epoch",
    )
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_SlowEmptyDetector(),
        shot_config=_shot_config(),
        visual_motion_config=VisualMotionConfig(
            pixel_delta_threshold=10,
            changed_pixel_ratio=0.0001,
        ),
        boundary_ledger=boundary_ledger,
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
            stop_after_terminal_shots=1,
        ),
    )

    events = runtime.run()

    assert [record.kind for record in boundary_ledger.records()] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
    ]
    assert [event.kind for event in events if event.kind.startswith("SHOT_")][:2] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
    ]
    assert runtime.diagnostics.tracked_observation_frames > 0


def test_product_path_references_continuous_session_recording_without_copying_video(
    tmp_path: Path,
) -> None:
    """Catches port-8787 analysis jobs reverting to per-shot media copies."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    assert hasattr(recording, "SessionRecordingEvidenceStore"), (
        "the product runtime has no continuous session evidence adapter"
    )
    recording_root = tmp_path / "session-recording"
    recorder = recording.ContinuousSessionRecorder(
        root=recording_root,
        session_id="session-0001",
        expected_fps=60,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=0.1,
            memory_buffer_seconds=2.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    evidence_store = recording.SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session-0001",
        pre_roll_frames=20,
        post_roll_frames=4,
        maximum_shot_frames=120,
    )
    source = _ShotCapture(frames=200)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        evidence_store=evidence_store,
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "unused-legacy-ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
            stop_after_terminal_shots=1,
            initial_shot_number=7,
        ),
    )

    events = runtime.run()

    enqueued = next(event for event in events if event.kind == "SHOT_ENQUEUED")
    job_root = tmp_path / "analysis-media" / enqueued.job.job_id
    manifest = (job_root / "interval.json").read_text(encoding="utf-8")
    assert '"storage_kind":"session_segment_reference"' in manifest
    assert {path.name for path in job_root.iterdir()} == {"interval.json"}
    assert tuple(recording_root.glob("*.mjpgseg"))
    assert not (tmp_path / "unused-legacy-ring").exists()


@pytest.mark.parametrize("consumer_pause_seconds", [0.0, 0.08])
def test_slow_job_store_does_not_block_capture(
    tmp_path, monkeypatch, consumer_pause_seconds
) -> None:
    class SlowJobStore:
        def __init__(self) -> None:
            self.jobs: list[object] = []
            self.producer_progress: tuple[int, int] | None = None

        def enqueue(self, job: object) -> SimpleNamespace:
            before = source.produced_frames
            time.sleep(0.02)
            deadline = time.monotonic() + 0.05
            while source.produced_frames <= before and time.monotonic() < deadline:
                time.sleep(0.001)
            self.producer_progress = (before, source.produced_frames)
            self.jobs.append(job)
            return SimpleNamespace(job=job, status="pending", created=True)

    from billiards_ai.lab import mainline1_product_realtime_runtime as product_runtime

    source = _QueuedShotCapture(frames=120)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    store = SlowJobStore()
    event_channel = product_runtime.ProductRealtimeEventChannel(maximum_pending_events=8)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        event_channel=event_channel,
        **_job_options(tmp_path, job_store=store),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
        ),
    )

    returned_events: list[object] = []
    validate_frame = runtime._validate_frame_mode

    def delayed_consumer(frame):
        if frame.capture_frame_id == 30:
            time.sleep(consumer_pause_seconds)
        return validate_frame(frame)

    monkeypatch.setattr(runtime, "_validate_frame_mode", delayed_consumer)
    runner = threading.Thread(
        target=lambda: returned_events.extend(runtime.run()),
        daemon=True,
    )
    runner.start()
    delivered = []
    deadline = time.monotonic() + 3
    while len(delivered) < 3 and time.monotonic() < deadline:
        event = event_channel.receive(timeout_seconds=0.1)
        if event is not None:
            delivered.append(event)

    assert [event.kind for event in delivered] == [
        "SHOT_STARTED",
        "SHOT_SETTLED",
        "SHOT_ENQUEUED",
    ]
    assert runner.is_alive()
    runner.join(timeout=5)
    assert not runner.is_alive()
    assert len(store.jobs) == 1
    assert runtime.capture_health.received_frames == 120
    assert runtime.capture_health.received_frames > (delivered[-1].frame_id or 0) + 20
    assert source.maximum_pending_frame_count >= 1
    # Prove the camera producer advances during the actual store call. The
    # whole-run queue high-water also includes unrelated consumer scheduling.
    assert store.producer_progress is not None
    assert store.producer_progress[1] > store.producer_progress[0]
    if consumer_pause_seconds:
        assert runtime.maximum_capture_queue_lag_frames > 2
    assert runtime.diagnostics.maximum_stage_duration_ms["shot_finalize"] < 100.0
    assert not any(event.kind == "CAPTURE_DEGRADED" for event in returned_events)


@pytest.mark.parametrize("first_anchor_at_eof", [False, True])
def test_event_channel_overflow_degrades_capture_instead_of_dropping_boundaries(
    tmp_path,
    first_anchor_at_eof,
) -> None:
    from billiards_ai.lab import mainline1_product_realtime_runtime as product_runtime

    cancelled = threading.Event()
    anchor_completed = threading.Event()
    anchor_replayed = threading.Event()

    class NotifyingResults(queue.SimpleQueue):
        def __init__(self, ready):
            super().__init__()
            self.ready = ready

        def put(self, value):
            super().put(value)
            self.ready.set()

    class Capture(_ShotCapture):
        def frames(self, stop):
            for frame in super().frames(stop):
                yield frame
                if first_anchor_at_eof:
                    continue
                # This case checks online event backpressure, not throughput.
                # Coordinate real worker/replay completion instead of assuming
                # that a 1ms fixture delay delivers an anchor before source EOF.
                deadline = runtime._config.background_shutdown_timeout_seconds
                assert runtime._replay.wait_until_preprocessed(
                    frame.capture_frame_id, timeout_seconds=deadline
                )
                if frame.capture_frame_id == 0:
                    assert anchor_completed.wait(deadline)
                elif frame.capture_frame_id == 1:
                    assert anchor_replayed.wait(deadline)

        def cancel(self):
            super().cancel()
            cancelled.set()

    class Detector(_ShotDetector):
        def infer(self, frame):
            if first_anchor_at_eof:
                assert cancelled.wait(1), "capture must cancel before waiting on the anchor"
            return super().infer(frame)

    source = Capture(frames=200)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    event_channel = product_runtime.ProductRealtimeEventChannel(maximum_pending_events=1)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=Detector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        event_channel=event_channel,
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
        ),
    )
    runtime._worker._results = NotifyingResults(anchor_completed)
    runtime._replay._accepted_results = NotifyingResults(anchor_replayed)

    events = runtime.run()

    if first_anchor_at_eof:
        assert source.emitted_frames == 200
    else:
        assert source.emitted_frames < 200, _runtime_failure_diagnostics(runtime, events)
    assert any(
        event.kind == "CAPTURE_DEGRADED"
        and event.reason is not None
        and "event channel capacity" in event.reason
        for event in events
    )
    assert not any(event.kind == "SHOT_ENQUEUED" for event in events)
    delivered = event_channel.receive(timeout_seconds=0.1)
    assert delivered is not None
    assert delivered.kind == "SHOT_STARTED"
    with pytest.raises(
        product_runtime.ProductRealtimeEventChannelFailure,
        match="event channel capacity",
    ):
        event_channel.receive(timeout_seconds=0.1)


@pytest.mark.parametrize(
    ("frames", "failure"),
    [(18, None), (28, None), (52, None), (52, "capture"), (52, "cleanup")],
)
def test_source_end_drains_inflight_first_anchor_before_closing_shot(
    tmp_path, frames, failure
) -> None:
    """Anchor completion during cancellation must retain the source's real boundary."""
    cancelled = threading.Event()

    class Capture(_ShotCapture):
        def frames(self, stop):
            yield from super().frames(stop)
            if failure == "capture":
                raise RuntimeError("injected source failure after final captured frame")

        def cancel(self):
            super().cancel()
            cancelled.set()
            if failure == "cleanup":
                raise RuntimeError("injected capture cleanup failure")

    class Detector(_ShotDetector):
        def infer(self, frame):
            assert cancelled.wait(1), "capture must cancel before waiting on the anchor"
            return super().infer(frame)

    source = Capture(frames=frames)
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibrate(
            "virtual-shot",
            [(0, 119), (159, 119), (159, 0), (0, 0)],
            table_length_mm=1590,
            table_width_mm=1190,
        ),
        detector=Detector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
        ),
    )

    events = runtime.run()

    complete = frames == 52 and failure is None
    expected = (
        ["SHOT_STARTED", "SHOT_SETTLED", "SHOT_ENQUEUED"]
        if complete
        else ["SHOT_STARTED", "SHOT_INCOMPLETE"]
    )
    if failure is not None:
        expected.append("CAPTURE_DEGRADED")
    assert [event.kind for event in events] == expected, _runtime_failure_diagnostics(
        runtime, events
    )
    assert runtime.capture_health.received_frames == frames
    assert source.cancelled
    assert not runtime._worker._thread.is_alive()
    assert not runtime._replay._thread.is_alive()
    assert not runtime._replay._working_commit_thread.is_alive()
    assert all(not thread.is_alive() for thread in runtime._replay._working_threads)
    if not complete:
        incomplete = next(event for event in events if event.kind == "SHOT_INCOMPLETE")
        assert incomplete.reason == (
            "source_stopped" if frames == 18 else "source_stopped_before_post_roll"
        )
        assert not list((tmp_path / "analysis-media").glob("*"))
    else:
        assert events[1].settled.shot_id == events[2].job.shot_id
        assert events[1].settled.source_epoch_id == events[2].job.source_epoch_id == "shot-epoch"
        assert (events[2].job.media_root / "interval.json").is_file()


def test_product_realtime_disconnect_during_motion_is_incomplete(tmp_path) -> None:
    source = _ShotCapture(frames=18)
    calibration = calibrate(
        "virtual-shot",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    runtime = ProductRealtimeRuntime(
        source=source,
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibration,
        detector=_ShotDetector(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        **_job_options(tmp_path),
        config=ProductRealtimeRuntimeConfig(
            ring_root=tmp_path / "ring",
            pre_roll_frames=20,
            post_roll_frames=4,
            maximum_shot_frames=120,
            maximum_ring_bytes=64 * 1024 * 1024,
            tracker_width=160,
            tracker_height=120,
        ),
    )

    events = runtime.run()

    assert any(event.kind == "SHOT_STARTED" for event in events), _runtime_failure_diagnostics(
        runtime, events
    )
    assert any(event.kind == "SHOT_INCOMPLETE" for event in events), _runtime_failure_diagnostics(
        runtime, events
    )
    assert not list((tmp_path / "analysis-media").glob("*"))
