import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle, build_live_runtime_factory
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEventChannel
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig
from billiards_ai.lab.mainline1_realtime_ring import AsyncMjpegSegmentRing
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.shared.calibration.homography import calibrate


def test_real_writer_health_survives_evidence_adapter_and_media_idle_gate(tmp_path, monkeypatch):
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(
            reserve_bytes=1, minimum_write_mbps=0, segment_duration_seconds=1 / 60
        ),
    )
    started, release = threading.Event(), threading.Event()
    original = recorder._write_segment

    def delayed_segment(frames):
        started.set()
        assert release.wait(3)
        return original(frames)

    monkeypatch.setattr(recorder, "_write_segment", delayed_segment)
    evidence = SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session",
        pre_roll_frames=120,
        post_roll_frames=120,
        maximum_shot_frames=1800,
    )
    hub = SimpleNamespace(
        subscribe=lambda *args, **kwargs: object(),
        recording_health=evidence.health,
        capture_health=SimpleNamespace(state="running", reason=None),
    )
    handle = RuntimeHandle(object(), lambda: None, capture_hub=hub, recording_store=evidence)
    source = object.__new__(USBLiveProjectionSource)
    source.runtime, source.lock = handle, threading.RLock()
    source._stopped, source._restart_reason = threading.Event(), None
    try:
        assert not hasattr(handle.recording_health, "pending_chunks")
        assert handle.recording_writer_health.pending_chunks == 0
        assert source.media_encoding_idle()
        ok, image = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
        assert ok
        recorder.append(CaptureFrame("epoch", 0, 0, 16, 12, image.tobytes()))
        assert started.wait(1)
        assert handle.recording_writer_health.pending_chunks == 0
        assert not source.media_encoding_idle()
    finally:
        release.set()
        recorder.close()


def test_legacy_real_writer_idle_and_inflight_pressure_are_visible(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[3] / "configs/algorithm/shot-local.yaml"
    factory = build_live_runtime_factory(
        capture_factory=lambda _: SimpleNamespace(cancel=lambda: None),
        capture_config=RealtimeCaptureConfig(width=16, height=12),
        calibration=calibrate("camera", [(0, 11), (15, 11), (15, 0), (0, 0)]),
        detector=object(),
        identity_classifier=object(),
        shot_config=load_mainline1_shot_local_config(config_path),
        capture_setup_id="capture",
        calibration_artifact_id="calibration",
        config_path=config_path,
        ring_root=tmp_path / "ring",
        jobs_root=tmp_path / "jobs",
    )
    handle = factory("session", ProductRealtimeEventChannel(), 0, "epoch")
    assert handle._recording_store is None  # Legacy must not gain ordinary seal authority.
    writer = handle.capture_hub._recorder
    assert isinstance(writer, AsyncMjpegSegmentRing)
    ring = writer._ring
    started, release = threading.Event(), threading.Event()
    original = ring.append

    def delayed_append(frame):
        started.set()
        assert release.wait(3)
        original(frame)

    monkeypatch.setattr(ring, "append", delayed_append)
    source = object.__new__(USBLiveProjectionSource)
    source.runtime, source.lock = handle, threading.RLock()
    source._stopped, source._restart_reason = threading.Event(), None
    try:
        assert source.media_encoding_idle()
        ok, image = cv2.imencode(".jpg", np.zeros((12, 16, 3), dtype=np.uint8))
        assert ok
        writer.append(CaptureFrame("epoch", 0, 0, 16, 12, image.tobytes()))
        assert started.wait(1)
        assert not source.media_encoding_idle()
        release.set()
        deadline = time.monotonic() + 1
        while not source.media_encoding_idle() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert source.media_encoding_idle()
    finally:
        release.set()
        handle.shutdown()
