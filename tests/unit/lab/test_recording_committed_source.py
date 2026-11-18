"""Disk consumers must never see uncommitted raw media or gate its writer."""

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
        session_id="session-1",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )


def _frame(number):
    return CaptureFrame("epoch-1", number, number * 16_666_667, 16, 12, b"jpeg")


def test_file_without_durable_manifest_is_neither_readable_nor_referenceable(tmp_path, monkeypatch):
    """Catches publishing a segment after rename but before its manifest fsync."""
    recorder = _recorder(tmp_path)
    entered, release = threading.Event(), threading.Event()
    segments = []
    original = recorder._write_manifest

    def blocked_manifest(manifest, **kwargs):
        segments.extend(manifest.segments)
        entered.set()
        assert release.wait(2)
        original(manifest, **kwargs)

    monkeypatch.setattr(recorder, "_write_manifest", blocked_manifest)
    try:
        for number in range(60):
            recorder.append(_frame(number))
        assert entered.wait(1)
        assert list((tmp_path / "raw").glob("*.mjpgseg"))
        assert recorder.health.committed_segments == 0
        with pytest.raises(RealtimeRingError, match="not committed"):
            recorder.read_segment(segments[0])
        with pytest.raises(RealtimeRingError, match="not committed"):
            recorder.reference_interval(
                session_id="session-1",
                shot_id="shot-1",
                source_epoch_id="epoch-1",
                start_frame_id=0,
                end_frame_id=59,
            )
        release.set()
        assert recorder.wait_until_committed(59, timeout_seconds=1)
        assert [row.capture_frame_id for row in recorder.read_segment(segments[0])] == list(
            range(60)
        )
    finally:
        release.set()
        recorder.close()


def _source(recorder, *, session_id="session-1", epoch="epoch-1"):
    from billiards_ai.lab import mainline1_session_recording as recording

    assert hasattr(recording, "CommittedRecordingSource"), "disk-following source is missing"
    return recording.CommittedRecordingSource(
        recorder=recorder,
        session_id=session_id,
        source_epoch_id=epoch,
    )


def test_disk_follower_reads_first_segment_before_close_then_catches_up_over_180_frames(tmp_path):
    """Catches waiting for the whole match, coalescing, and another bounded FIFO."""
    import cv2
    import numpy as np

    recorder = _recorder(tmp_path)
    source = _source(recorder)
    originals = []
    for number in range(241):
        image = np.full((12, 16, 3), number % 256, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        assert ok
        originals.append(
            CaptureFrame(
                "epoch-1",
                number,
                number * 16_666_667,
                16,
                12,
                jpeg.tobytes(),
            )
        )
    rows = source.frames(threading.Event())
    try:
        for frame in originals[:60]:
            recorder.append(frame)
        assert recorder.wait_until_committed(59, timeout_seconds=1)
        first = next(rows)
        assert first == originals[0]
        # Keep the generator paused while capture stores > the old FIFO capacity.
        for frame in originals[60:]:
            recorder.append(frame)
        assert recorder.wait_until_committed(239, timeout_seconds=1)
        assert recorder.health.committed_frames == 240
        assert recorder.health.state == "running"
        assert source.progress.last_delivered_frame_id == 0
        assert source.progress.committed_frame_id == 239
        recorder.close()  # Commits the final one-frame tail.
        assert [first, *rows] == originals
        assert source.progress.last_delivered_frame_id == 240
        assert source.progress.input_closed
        assert len(list((tmp_path / "raw").glob("*.mjpgseg"))) == 5
    finally:
        source.cancel()
        rows.close()
        recorder.close()


def test_disk_follower_waits_for_partial_tail_and_cancel_never_closes_writer(tmp_path):
    """Catches reading pending memory, polling after cancel, or cancelling capture."""
    recorder = _recorder(tmp_path)
    source = _source(recorder)
    recorder.append(_frame(0))
    done = threading.Event()
    received, errors = [], []

    def read():
        try:
            received.extend(source.frames(threading.Event()))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert not done.wait(0.03)
        assert received == []
        source.cancel()
        assert done.wait(1)
        assert received == [] and errors == []
        recorder.append(_frame(1))
        assert recorder.close().segments[0].last_frame_id == 1
    finally:
        source.cancel()
        reader.join(1)
        recorder.close()


@pytest.mark.parametrize("mismatch", ["session", "epoch"])
def test_disk_follower_rejects_foreign_recording_without_delivering_a_frame(tmp_path, mismatch):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    recorder.close()
    with pytest.raises((ValueError, RealtimeRingError), match="identity|epoch|session"):
        source = _source(
            recorder,
            session_id="other" if mismatch == "session" else "session-1",
            epoch="other" if mismatch == "epoch" else "epoch-1",
        )
        next(source.frames(threading.Event()))


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "replaced"])
def test_disk_follower_rejects_changed_segment_instead_of_skipping_it(tmp_path, mutation):
    """Catches hash bypass and accepting a replaced file with the same bytes."""
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    manifest = recorder.close()
    source = _source(recorder)
    path = tmp_path / "raw" / manifest.segments[0].name
    if mutation == "missing":
        path.unlink()
    elif mutation == "corrupt":
        path.write_bytes(b"bad")
    else:
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    with pytest.raises(RealtimeRingError):
        next(source.frames(threading.Event()))
    assert source.progress.last_delivered_frame_id is None
    assert source.progress.failure_reason


def test_failed_manifest_never_becomes_readable_even_when_segment_file_exists(
    tmp_path, monkeypatch
):
    recorder = _recorder(tmp_path)
    source = _source(recorder)

    def failed_manifest(manifest):
        raise OSError("controlled manifest failure")

    monkeypatch.setattr(recorder, "_write_manifest", failed_manifest)
    for number in range(60):
        recorder.append(_frame(number))
    with pytest.raises(RealtimeRingError, match="writer failed"):
        next(source.frames(threading.Event()))
    assert recorder.health.committed_frames == 0
    assert source.progress.last_delivered_frame_id is None
    assert list((tmp_path / "raw").glob("*.mjpgseg"))
    with pytest.raises(RealtimeRingError, match="writer failed"):
        recorder.close()


def test_disk_follower_preserves_nonzero_rollover_frame_ids_and_capture_time(tmp_path):
    recorder = _recorder(tmp_path)
    for number in range(100, 105):
        recorder.append(_frame(number))
    recorder.close()
    source = _source(recorder)
    rows = list(source.frames(threading.Event()))
    assert [row.capture_frame_id for row in rows] == [100, 101, 102, 103, 104]
    assert [row.captured_at_ns for row in rows] == [
        1_666_666_700,
        1_683_333_367,
        1_700_000_034,
        1_716_666_701,
        1_733_333_368,
    ]
    with pytest.raises(RealtimeRingError, match="single-use"):
        next(source.frames(threading.Event()))


def test_stopped_reader_wakes_without_a_segment_and_does_not_wait_for_writer_close(tmp_path):
    recorder = _recorder(tmp_path)
    source = _source(recorder)
    stop, started, done = threading.Event(), threading.Event(), threading.Event()
    received = []

    def read():
        started.set()
        received.extend(source.frames(stop))
        done.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert started.wait(1)
        stop.set()
        assert done.wait(1)
        assert received == [] and recorder.health.state == "running"
    finally:
        stop.set()
        source.cancel()
        reader.join(1)
        recorder.close()


def test_sealed_disk_reader_waits_for_durable_tail_then_stops_at_exact_cutoff(tmp_path):
    recorder = _recorder(tmp_path)
    source = _source(recorder)
    recorder.append(_frame(0))
    assert hasattr(source, "seal"), "disk source cannot observe a capture cutoff"
    source.seal(_frame(0))
    done = threading.Event()
    received, errors = [], []

    def read():
        try:
            received.extend(source.frames(threading.Event()))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert not done.wait(0.03)
        recorder.commit_pending(timeout_seconds=1)
        assert done.wait(1)
        assert received == [_frame(0)] and errors == []
        # Reader exhaustion is not writer closure; no detector drain dependency.
        assert recorder.health.state == "running"
    finally:
        source.cancel()
        reader.join(1)
        recorder.close()


def test_closed_recording_short_of_cutoff_is_a_failure_not_completed_detection(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    recorder.close()
    source = _source(recorder)
    source.seal(_frame(1))
    with pytest.raises(RealtimeRingError, match="cutoff"):
        list(source.frames(threading.Event()))
    assert source.progress.last_delivered_frame_id == 0
    assert source.progress.failure_reason


def test_disk_source_cutoff_is_idempotent_but_cannot_change_or_switch_epoch(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    recorder.close()
    source = _source(recorder)
    source.seal(_frame(0))
    source.seal(_frame(0))
    with pytest.raises(RealtimeRingError, match="cutoff changed"):
        source.seal(_frame(1))
    other = _source(recorder, epoch="other-epoch")
    with pytest.raises(RealtimeRingError, match="epoch"):
        other.seal(_frame(0))
    assert list(source.frames(threading.Event())) == [_frame(0)]


def test_saved_recording_opens_for_processing_without_writer_or_storage_preflight(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    recorder.append(_frame(1))
    recorder.close()
    before = {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()}

    class NoWrites:
        def available_bytes(self, root):
            pytest.fail("processing snapshot must not preflight or write capture storage")

        def benchmark_write_mbps(self, root):
            pytest.fail("processing snapshot must not benchmark storage")

    snapshot = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session-1",
        expected_fps=60,
        read_only=True,
        storage_probe=NoWrites(),
    )
    assert list(_source(snapshot).frames(threading.Event())) == [_frame(0), _frame(1)]
    with pytest.raises(RealtimeRingError, match="closed"):
        snapshot.append(_frame(2))
    snapshot.close()
    assert {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()} == before


def test_capture_failure_is_not_hidden_when_original_writer_has_closed(tmp_path):
    from billiards_ai.lab.mainline1_realtime_capture import CaptureContractError
    from billiards_ai.lab.mainline1_session_recording import CommittedRecordingSource

    recorder = _recorder(tmp_path)
    recorder.append(_frame(0))
    recorder.close()
    source = CommittedRecordingSource(
        recorder=recorder,
        session_id="session-1",
        source_epoch_id="epoch-1",
        capture_failure=lambda: "controlled capture interruption",
    )
    frames = source.frames(threading.Event())
    assert next(frames) == _frame(0)
    with pytest.raises(CaptureContractError, match="controlled capture interruption"):
        next(frames)
