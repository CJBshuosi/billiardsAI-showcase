import threading

import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)


def _recorder(tmp_path):
    return ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )


def _frame(number):
    return CaptureFrame("epoch", number, number * 16_666_667, 16, 12, b"jpeg")


def test_target_fence_waits_for_actual_append_and_durable_manifest(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    original = recorder._write_manifest
    in_manifest, release, complete = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def delayed_manifest(manifest, **kwargs):
        in_manifest.set()
        assert release.wait(1)
        original(manifest, **kwargs)

    monkeypatch.setattr(recorder, "_write_manifest", delayed_manifest)

    def fence():
        try:
            recorder.commit_through(1, timeout_seconds=1)
            complete.set()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=fence)
    thread.start()
    try:
        assert not in_manifest.wait(0.02) and not complete.is_set()
        recorder.append(_frame(1))
        assert in_manifest.wait(1)
        assert not complete.wait(0.02), "segment alone is not the manifest fence"
        release.set()
        thread.join(1)
        assert complete.is_set() and errors == []
        assert (
            recorder.reference_interval(
                session_id="session",
                shot_id="shot-1",
                source_epoch_id="epoch",
                start_frame_id=0,
                end_frame_id=1,
            ).frame_count
            == 2
        )
    finally:
        release.set()
        thread.join(1)
        recorder.close()


def test_unreceived_target_and_failed_writer_never_return_a_reference(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    try:
        with pytest.raises(RealtimeRingError, match="not received"):
            recorder.commit_through(1, timeout_seconds=0.01)

        def broken_write(chunk):
            raise OSError("controlled disk failure")

        monkeypatch.setattr(recorder, "_write_segment", broken_write)
        with pytest.raises(RealtimeRingError, match="writer failed"):
            recorder.commit_through(0, timeout_seconds=1)
    finally:
        with pytest.raises(RealtimeRingError):
            recorder.close()


def test_each_chunk_preserves_disk_reserve_and_prior_durable_evidence(tmp_path):
    class Probe:
        free = 100000

        def available_bytes(self, root):
            return self.free

    probe = Probe()
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=100, minimum_write_mbps=0),
        storage_probe=probe,
    )
    recorder.append(_frame(0))
    recorder.commit_through(0, timeout_seconds=1)
    committed = {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()}
    probe.free = 100
    recorder.append(_frame(1))
    with pytest.raises(RealtimeRingError, match="reserved space"):
        recorder.commit_through(1, timeout_seconds=1)
    assert {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()} == committed
    with pytest.raises(RealtimeRingError):
        recorder.close()


def test_pending_window_capacity_keeps_every_boundary_and_marks_only_excess_failed():
    from collections import deque
    from types import SimpleNamespace

    from billiards_ai.edge.route_assistance.records import RecordingBoundaryV1
    from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeRuntime

    runtime = ProductRealtimeRuntime.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "recording-v1"
    runtime._recording_windows = deque()
    runtime._recording_active = None
    runtime._event_channel = None
    runtime._event_channel_failed = False
    runtime._latest_capture_frame = _frame(200)
    events = []
    for i in range(33):
        window = SimpleNamespace(
            shot_id=f"shot-{i}",
            start_frame_id=i * 3,
            end_frame_id=i * 3 + 1,
            start_ms=i * 50,
            end_ms=i * 50 + 20,
            complete=True,
        )
        runtime._apply_recording_boundary(
            SimpleNamespace(
                kind=object(),
                window=window,
                reason=None,
            ),
            events,
        )
    assert len(runtime._recording_windows) == 32
    assert [row.shot_id for row in runtime._recording_windows] == [f"shot-{i}" for i in range(32)]
    boundaries = [e.recording_boundary for e in events if e.kind == "SHOT_RECORDING_BOUNDARY"]
    assert len(boundaries) == 33 and all(isinstance(b, RecordingBoundaryV1) for b in boundaries)
    assert events[-1].recorded_media.boundary == boundaries[-1]
    assert events[-1].recorded_media.failure_code == "RECORDING_WINDOW_CAPACITY"


def test_reference_to_already_committed_frame_does_not_flush_the_current_capture_tail(tmp_path):
    """A disk detector reading old frames must not change writer segmentation."""
    recorder = _recorder(tmp_path)
    try:
        recorder.append(_frame(0))
        recorder.commit_through(0, timeout_seconds=1)
        recorder.append(_frame(1))
        recorder.commit_through(0, timeout_seconds=1)
        assert recorder.health.committed_frames == 1
        assert recorder.health.buffered_frames == 1
    finally:
        recorder.close()


def test_commit_pending_timeout_includes_waiting_for_writer_admission_lock(tmp_path):
    recorder = _recorder(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def hold_admission():
        with recorder._append_lock:
            entered.set()
            release.wait(1)

    thread = threading.Thread(target=hold_admission)
    thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RealtimeRingError, match="commit exceeded timeout"):
            recorder.commit_pending(timeout_seconds=0.02)
    finally:
        release.set()
        thread.join(1)
        recorder.close()
