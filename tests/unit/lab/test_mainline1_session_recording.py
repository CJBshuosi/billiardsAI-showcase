from __future__ import annotations

import importlib
import importlib.util
import threading
from pathlib import Path

import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError


def _frame(frame_id: int, *, fps: int = 10) -> CaptureFrame:
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * (1_000_000_000 // fps),
        width=16,
        height=12,
        mjpeg=f"jpeg-payload-{frame_id}".encode(),
    )


def test_session_segments_are_written_once_and_shots_only_reference_them(
    tmp_path: Path,
) -> None:
    """Catches per-shot media copies and per-frame disk command persistence."""
    module_spec = importlib.util.find_spec("billiards_ai.lab.mainline1_session_recording")
    assert module_spec is not None, "continuous session recorder is missing"
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    root = tmp_path / "session-recording"
    recorder = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-1",
        expected_fps=10,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=0.2,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )

    for frame_id in range(6):
        recorder.append(_frame(frame_id))
    manifest = recorder.close()
    first = recorder.reference_interval(
        session_id="session-1",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        start_frame_id=1,
        end_frame_id=2,
    )
    second = recorder.reference_interval(
        session_id="session-1",
        shot_id="shot-2",
        source_epoch_id="epoch-1",
        start_frame_id=3,
        end_frame_id=4,
    )

    assert [(row.first_frame_id, row.last_frame_id) for row in manifest.segments] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert [frame.capture_frame_id for frame in first.frames()] == [1, 2]
    assert [frame.capture_frame_id for frame in second.frames()] == [3, 4]
    assert first.root == root and second.root == root
    assert first.segment_sha256 != second.segment_sha256
    assert {path.name for path in root.iterdir()} == {
        "00000000000000000000-00000000000000000001.mjpgseg",
        "00000000000000000002-00000000000000000003.mjpgseg",
        "00000000000000000004-00000000000000000005.mjpgseg",
        "session-recording.json",
    }


def test_simulated_sixty_minute_session_keeps_one_timeline_for_one_hundred_shots(
    tmp_path: Path,
) -> None:
    """Catches long-session rollover and O(shots) media duplication."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    root = tmp_path / "sixty-minute-session"
    recorder = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-hour",
        expected_fps=60,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=60.0,
            memory_buffer_seconds=3600.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
            shutdown_timeout_seconds=10.0,
        ),
    )
    frame_count = 60 * 60 * 60
    for frame_id in range(frame_count):
        recorder.append(_frame(frame_id, fps=60))
    manifest = recorder.close()

    intervals = tuple(
        recorder.reference_interval(
            session_id="session-hour",
            shot_id=f"shot-{shot_number:03d}",
            source_epoch_id="epoch-1",
            start_frame_id=shot_number * 2_000 + 10,
            end_frame_id=shot_number * 2_000 + 69,
        )
        for shot_number in range(100)
    )

    assert len(manifest.segments) == 60
    assert manifest.segments[0].first_frame_id == 0
    assert manifest.segments[-1].last_frame_id == frame_count - 1
    assert all(interval.root == root for interval in intervals)
    assert len({interval.shot_id for interval in intervals}) == 100
    assert len(tuple(root.glob("*.mjpgseg"))) == 60
    assert not tuple(root.glob("shot-*"))


@pytest.mark.parametrize(
    ("available_bytes", "write_mbps", "message"),
    [
        (999, 100.0, "reserved space"),
        (2_000, 79.9, "write throughput"),
    ],
)
def test_recorder_rejects_the_match_before_capture_when_storage_is_unsupported(
    tmp_path: Path,
    available_bytes: int,
    write_mbps: float,
    message: str,
) -> None:
    """Catches starting an authoritative match on storage that will fall behind."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")

    class StorageProbe:
        def available_bytes(self, root: Path) -> int:
            assert root == (tmp_path / "session-recording").resolve()
            return available_bytes

        def benchmark_write_mbps(self, root: Path) -> float:
            assert root == (tmp_path / "session-recording").resolve()
            return write_mbps

    with pytest.raises(RealtimeRingError, match=message):
        recording.ContinuousSessionRecorder(
            root=tmp_path / "session-recording",
            session_id="session-1",
            expected_fps=60,
            config=recording.SessionRecordingConfig(
                reserve_bytes=1_000,
                minimum_write_mbps=80.0,
            ),
            storage_probe=StorageProbe(),
        )


def test_reopen_recovers_committed_segments_and_continues_without_overwrite(
    tmp_path: Path,
) -> None:
    """Catches a service restart orphaning or replacing an unfinished session."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    root = tmp_path / "session-recording"
    config = recording.SessionRecordingConfig(
        segment_duration_seconds=0.2,
        memory_buffer_seconds=1.0,
        reserve_bytes=1,
        minimum_write_mbps=0.0,
    )
    first = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-1",
        expected_fps=10,
        config=config,
    )
    for frame_id in range(4):
        first.append(_frame(frame_id))
    first.close()

    resumed = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-1",
        expected_fps=10,
        config=config,
    )
    resumed.append(_frame(4))
    resumed.append(_frame(5))
    manifest = resumed.close()
    interval = resumed.reference_interval(
        session_id="session-1",
        shot_id="shot-across-restart",
        source_epoch_id="epoch-1",
        start_frame_id=1,
        end_frame_id=4,
    )

    assert [(row.first_frame_id, row.last_frame_id) for row in manifest.segments] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert [frame.capture_frame_id for frame in interval.frames()] == [1, 2, 3, 4]


def test_new_match_recording_can_begin_at_current_physical_camera_frame(
    tmp_path: Path,
) -> None:
    """Catches match rollover requiring the shared camera frame counter to reset."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    root = tmp_path / "rolled-session"
    config = recording.SessionRecordingConfig(
        segment_duration_seconds=0.2,
        memory_buffer_seconds=1.0,
        reserve_bytes=1,
        minimum_write_mbps=0.0,
    )
    recorder = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-2",
        expected_fps=10,
        config=config,
    )
    recorder.append(_frame(100))
    recorder.append(_frame(101))
    recorder.close()

    resumed = recording.ContinuousSessionRecorder(
        root=root,
        session_id="session-2",
        expected_fps=10,
        config=config,
    )
    resumed.append(_frame(102))
    resumed.append(_frame(103))
    manifest = resumed.close()

    assert [(item.first_frame_id, item.last_frame_id) for item in manifest.segments] == [
        (100, 101),
        (102, 103),
    ]


def test_shot_pre_roll_clamps_to_first_frame_recorded_after_camera_preflight(
    tmp_path: Path,
) -> None:
    """Catches the twelve startup-validation frames making the first shot incomplete."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    recorder = recording.ContinuousSessionRecorder(
        root=tmp_path / "recording",
        session_id="session-1",
        expected_fps=10,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=0.2,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    evidence = recording.SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session-1",
        pre_roll_frames=20,
        post_roll_frames=2,
        maximum_shot_frames=100,
    )
    for frame_id in range(100, 106):
        evidence.append(_frame(frame_id))
        if frame_id == 102:
            evidence.begin_shot(frame_id)
        if frame_id == 103:
            evidence.settle_shot(frame_id)

    interval = evidence.freeze_shot()
    evidence.close()

    assert interval.start_frame_id == 100
    assert interval.end_frame_id == 105
    assert [frame.capture_frame_id for frame in interval.frames()] == list(range(100, 106))


def test_external_continuous_recording_accepts_coalesced_analysis_cursor(
    tmp_path: Path,
) -> None:
    """Analysis may skip old frames while the independently owned evidence stays complete."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    recorder = recording.ContinuousSessionRecorder(
        root=tmp_path / "recording",
        session_id="session-1",
        expected_fps=10,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=0.2,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    for frame_id in range(4):
        recorder.append(_frame(frame_id))
    evidence = recording.SessionRecordingEvidenceStore(
        recorder=recorder,
        session_id="session-1",
        pre_roll_frames=1,
        post_roll_frames=0,
        maximum_shot_frames=10,
        append_frames=False,
        close_recorder=False,
    )

    evidence.append(_frame(0))
    evidence.append(_frame(3))
    evidence.begin_shot(3)
    evidence.settle_shot(3)
    interval = evidence.freeze_shot()
    evidence.close()
    recorder.close()

    assert [frame.capture_frame_id for frame in interval.frames()] == [2, 3]


def test_full_segment_buffer_fails_immediately_instead_of_blocking_capture(
    tmp_path: Path,
) -> None:
    """Catches a stalled disk turning the recorder into a camera backpressure loop."""
    recording = importlib.import_module("billiards_ai.lab.mainline1_session_recording")
    recorder = recording.ContinuousSessionRecorder(
        root=tmp_path / "session-recording",
        session_id="session-1",
        expected_fps=10,
        config=recording.SessionRecordingConfig(
            segment_duration_seconds=0.1,
            memory_buffer_seconds=0.1,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    writer_started = threading.Event()
    release_writer = threading.Event()
    write_segment = recorder._write_segment

    def blocked_write(frames):
        writer_started.set()
        assert release_writer.wait(timeout=1)
        return write_segment(frames)

    recorder._write_segment = blocked_write
    recorder.append(_frame(0))
    assert writer_started.wait(timeout=1)
    recorder.append(_frame(1))

    finished = threading.Event()
    failures: list[BaseException] = []

    def append_while_full() -> None:
        try:
            recorder.append(_frame(2))
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    append_thread = threading.Thread(target=append_while_full)
    append_thread.start()
    returned_without_disk = finished.wait(timeout=0.1)
    release_writer.set()
    append_thread.join(timeout=1)

    assert returned_without_disk is True
    assert len(failures) == 1
    assert isinstance(failures[0], RealtimeRingError)
    assert "segment buffer" in str(failures[0])
    assert recorder.health.state.value == "degraded"
    assert recorder.health.reason == "session recording segment buffer is exhausted"
    manifest = recorder.close()
    assert manifest.segments[-1].last_frame_id == 2
