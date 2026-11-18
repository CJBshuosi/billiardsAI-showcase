from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
from billiards_ai.lab.mainline1_shot_worker import load_materialized_interval


def _frame(frame_id: int) -> CaptureFrame:
    image = np.full((12, 16, 3), frame_id, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 100_000_000,
        width=16,
        height=12,
        mjpeg=encoded.tobytes(),
    )


def test_worker_reads_session_segment_reference_without_copied_video(
    tmp_path: Path,
) -> None:
    """Catches analysis job creation duplicating session recording bytes per shot."""
    shot_jobs = importlib.import_module("billiards_ai.lab.mainline1_shot_jobs")
    assert hasattr(shot_jobs, "materialize_recorded_interval_reference"), (
        "reference-only shot job persistence is missing"
    )
    recording_root = tmp_path / "recording"
    recorder = ContinuousSessionRecorder(
        root=recording_root,
        session_id="session-1",
        expected_fps=10,
        config=SessionRecordingConfig(
            segment_duration_seconds=0.2,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0.0,
        ),
    )
    for frame_id in range(8):
        recorder.append(_frame(frame_id))
    recorder.close()
    interval = recorder.reference_interval(
        session_id="session-1",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        start_frame_id=1,
        end_frame_id=6,
    )
    job_root = tmp_path / "jobs" / "job-1"
    shot_jobs.materialize_recorded_interval_reference(interval, job_root)
    job = ShotAnalysisJob(
        job_id="job-1",
        session_id="session-1",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        start_frame_id=1,
        end_frame_id=6,
        media_root=job_root,
        media_sha256=interval.media_sha256,
        capture_setup_id="capture-1",
        calibration_artifact_id="calibration-1",
        config_sha256="a" * 64,
        model_package_id="model-1",
    )

    loaded = load_materialized_interval(job)

    assert [frame.capture_frame_id for frame in loaded.frames()] == list(range(1, 7))
    assert {path.name for path in job_root.iterdir()} == {"interval.json"}
    assert len(tuple(recording_root.glob("*.mjpgseg"))) == 4
