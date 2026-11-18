"""A capture cutoff must survive a transient tail-commit wait timeout."""

import threading

import pytest

from billiards_ai.lab.mainline1_capture_hub import CaptureHub
from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig
from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
    SessionRecordingEvidenceStore,
)


def test_tail_timeout_retry_keeps_original_admission_cutoff_after_detector_finishes(
    tmp_path, monkeypatch
):
    frame = CaptureFrame("epoch", 0, 0, 16, 12, b"jpeg")
    entered, release = threading.Event(), threading.Event()
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    store = SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session",
        pre_roll_frames=0,
        post_roll_frames=0,
        maximum_shot_frames=60,
    )
    original = recorder._write_manifest

    def delayed_manifest(manifest, **kwargs):
        entered.set()
        assert release.wait(2)
        original(manifest, **kwargs)

    monkeypatch.setattr(recorder, "_write_manifest", delayed_manifest)

    class Camera:
        def frames(self, stop):
            yield frame
            stop.wait(3)

        def cancel(self):
            pass

    class Detector:
        current_capture_frame = None

        def run(self):
            for row in disk.frames(threading.Event()):
                self.current_capture_frame = row

        def request_recording_end(self):
            pass

    camera = Camera()
    hub = CaptureHub(
        source=camera,
        recorder=store,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )
    disk = store.committed_source(source_epoch_id="epoch")
    handle = RuntimeHandle(
        camera,
        lambda _: Detector(),
        capture_hub=hub,
        recording_store=store,
        recording_source=disk,
    )
    failures = []

    def run():
        try:
            handle.run()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert handle.wait_until_ready(timeout_seconds=1)
        with pytest.raises(RealtimeRingError, match="commit exceeded timeout"):
            handle.seal_recording(timeout_seconds=0.02)
        assert entered.is_set() and not hub._uses_recorder(store)
        release.set()
        assert recorder.wait_until_committed(0, timeout_seconds=1)
        thread.join(1)
        assert not thread.is_alive() and failures == []
        assert handle.seal_recording(timeout_seconds=1) == frame
        assert handle.seal_recording(timeout_seconds=1) == frame
        assert recorder.health.committed_frames == 1
    finally:
        release.set()
        handle.shutdown()
        thread.join(1)
