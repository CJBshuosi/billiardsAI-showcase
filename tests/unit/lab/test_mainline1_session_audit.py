from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.field_assistance.visual_motion import VisualMotionConfig
from billiards_ai.lab.mainline1_live_window import IncrementalShotWindowCoordinator
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.shared.calibration.homography import calibrate


def _frame(frame_id: int, timestamp_ms: int, x_mm: int) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        observations=(
            CalibratedBallObservation(
                detection_id=f"ball-{frame_id}",
                x_mm=x_mm,
                y_mm=100,
                confidence=0.9,
                group=BallGroup.CUE,
                group_confidence=0.9,
            ),
        ),
    )


def _config() -> ShotWindowConfig:
    return ShotWindowConfig(8, 2, 400)


def _shot(frame_offset: int, time_offset: int) -> tuple[ObservationFrame, ...]:
    return (
        _frame(frame_offset, time_offset, 100),
        _frame(frame_offset + 1, time_offset + 40, 100),
        _frame(frame_offset + 2, time_offset + 80, 110),
        _frame(frame_offset + 3, time_offset + 120, 120),
        _frame(frame_offset + 4, time_offset + 160, 120),
        _frame(frame_offset + 5, time_offset + 560, 120),
    )


def test_audit_does_not_turn_pre_match_motion_into_recovered_match_shots():
    audit = importlib.import_module("billiards_ai.lab.mainline1_session_audit")
    minimum = None
    auditor = audit.ClosedSegmentAuditor(
        session_id="session-1",
        source_epoch_id="epoch-1",
        config=_config(),
        minimum_frame_id=lambda: minimum,
    )
    before = _shot(0, 0)
    assert auditor.audit(before, ()) == ()
    minimum = 10
    result = auditor.audit(before + _shot(10, 1000), ())
    assert len(result) == 2
    assert all(item.boundary.window.shot_id == "shot-0001" for item in result)


def test_dropped_realtime_boundary_is_recovered_once_across_segment_overlap(
    tmp_path: Path,
) -> None:
    """Catches one lost channel event silently deleting a physical shot."""
    audit = importlib.import_module("billiards_ai.lab.mainline1_session_audit")
    ledger = audit.ShotBoundaryLedger(
        tmp_path / "boundaries.sqlite3",
        session_id="session-1",
        source_epoch_id="epoch-1",
    )
    realtime = IncrementalShotWindowCoordinator(_config())
    sequence = _shot(0, 0)
    emitted = tuple(boundary for frame in sequence for boundary in realtime.push(frame))
    assert len(emitted) == 2
    ledger.record(emitted[0])
    auditor = audit.ClosedSegmentAuditor(
        session_id="session-1",
        source_epoch_id="epoch-1",
        config=_config(),
    )

    assert auditor.audit(sequence[:4], ledger.records()) == ()
    recovered = auditor.audit(sequence[3:], ledger.records())
    ledger.record(recovered[0].boundary, recovered=True)
    assert auditor.audit(sequence[3:], ledger.records()) == ()

    records = ledger.records()
    assert [record.kind for record in records] == ["SHOT_STARTED", "SHOT_SETTLED"]
    assert [record.recovered for record in records] == [False, True]
    assert len({record.boundary_id for record in records}) == 2


def test_boundary_ledger_restart_deduplicates_one_hundred_ordered_shots(
    tmp_path: Path,
) -> None:
    """Catches restart replay producing duplicate or reordered shot identities."""
    audit = importlib.import_module("billiards_ai.lab.mainline1_session_audit")
    path = tmp_path / "boundaries.sqlite3"
    ledger = audit.ShotBoundaryLedger(
        path,
        session_id="session-1",
        source_epoch_id="epoch-1",
    )
    coordinator = IncrementalShotWindowCoordinator(_config())
    frame_offset = 0
    time_offset = 0
    boundaries = []
    for _ in range(100):
        sequence = _shot(frame_offset, time_offset)
        boundaries.extend(boundary for frame in sequence for boundary in coordinator.push(frame))
        frame_offset += 6
        time_offset += 600
    for boundary in boundaries:
        ledger.record(boundary)

    reopened = audit.ShotBoundaryLedger(
        path,
        session_id="session-1",
        source_epoch_id="epoch-1",
    )
    for boundary in boundaries:
        reopened.record(boundary, recovered=True)

    records = reopened.records()
    assert len(records) == 200
    assert len({record.boundary_id for record in records}) == 200
    settled = [record for record in records if record.kind == "SHOT_SETTLED"]
    assert len(settled) == 100
    assert [record.shot_id for record in settled] == [
        f"shot-{number:04d}" for number in range(1, 101)
    ]
    assert all(record.recovered is False for record in records)


def _capture_frame(frame_id: int, x: int) -> CaptureFrame:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (x, 60), 8, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 100_000_000,
        width=160,
        height=120,
        mjpeg=encoded.tobytes(),
    )


def test_committed_recording_segments_automatically_recover_visual_boundaries(
    tmp_path: Path,
) -> None:
    """Catches a closed recording existing while its physical shot vanishes."""
    audit = importlib.import_module("billiards_ai.lab.mainline1_session_audit")
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "recording",
        session_id="session-1",
        expected_fps=10,
        config=SessionRecordingConfig(
            segment_duration_seconds=0.2,
            memory_buffer_seconds=1.0,
            reserve_bytes=1,
            minimum_write_mbps=0,
        ),
    )
    ledger = audit.ShotBoundaryLedger(
        tmp_path / "boundaries.sqlite3",
        session_id="session-1",
        source_epoch_id="epoch-1",
    )
    calibration = calibrate(
        "virtual",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    auditor = audit.ClosedSegmentAuditor(
        session_id="session-1",
        source_epoch_id="epoch-1",
        config=ShotWindowConfig(8, 2, 200),
        calibration=calibration,
        visual_motion_config=VisualMotionConfig(
            pixel_delta_threshold=10,
            changed_pixel_ratio=0.0001,
        ),
    )
    worker = audit.SessionSegmentAuditWorker(
        recorder=recorder,
        ledger=ledger,
        auditor=auditor,
    )
    recorder.set_segment_observer(worker)

    for frame_id, x in enumerate((40, 40, 50, 60, 60, 60, 60, 60)):
        recorder.append(_capture_frame(frame_id, x))
    recorder.close()
    worker.close(timeout_seconds=2)

    records = ledger.records()
    assert [record.kind for record in records] == ["SHOT_STARTED", "SHOT_SETTLED"]
    assert all(record.recovered for record in records)
    assert worker.health.audited_segments == 4
    assert worker.health.recovered_boundaries == 2
    assert worker.health.status == "closed"
