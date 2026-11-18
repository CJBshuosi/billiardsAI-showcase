"""Durable, idempotent recovery ledger for live shot boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from billiards_ai.lab.field_assistance.visual_motion import (
    VisualMotionConfig,
    table_visual_motion_detected,
)
from billiards_ai.lab.mainline1_live_window import (
    IncrementalShotWindowCoordinator,
    LiveShotBoundary,
    VisualShotBoundaryDetector,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionSegment,
)
from billiards_ai.lab.shot_local.models import ObservationFrame
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.shared.calibration.homography import TableCalibration


@dataclass(frozen=True, slots=True)
class ShotBoundaryRecord:
    boundary_id: str
    session_id: str
    source_epoch_id: str
    kind: str
    shot_id: str
    start_frame_id: int
    end_frame_id: int | None
    recovered: bool


@dataclass(frozen=True, slots=True)
class RecoveredShotBoundary:
    boundary_id: str
    boundary: LiveShotBoundary


@dataclass(frozen=True, slots=True)
class SegmentAuditHealth:
    status: str
    audited_segments: int
    recovered_boundaries: int
    reason: str | None = None


class ShotBoundaryLedger:
    """Persist each physical start/settle boundary exactly once."""

    def __init__(self, path: Path, *, session_id: str, source_epoch_id: str) -> None:
        if not session_id.strip() or not source_epoch_id.strip():
            raise ValueError("shot boundary ledger identity is invalid")
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.source_epoch_id = source_epoch_id
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shot_boundary_ledger (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    boundary_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    source_epoch_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    start_frame_id INTEGER NOT NULL,
                    end_frame_id INTEGER,
                    recovered INTEGER NOT NULL CHECK(recovered IN (0, 1)),
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS shot_boundary_ledger_session
                ON shot_boundary_ledger(session_id, source_epoch_id, sequence);
                """
            )

    def record(
        self,
        boundary: LiveShotBoundary,
        *,
        recovered: bool = False,
    ) -> ShotBoundaryRecord:
        record = _boundary_record(
            boundary,
            session_id=self.session_id,
            source_epoch_id=self.source_epoch_id,
            recovered=recovered,
        )
        payload = _canonical_payload(record)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO shot_boundary_ledger "
                "(boundary_id, session_id, source_epoch_id, kind, shot_id, "
                "start_frame_id, end_frame_id, recovered, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.boundary_id,
                    record.session_id,
                    record.source_epoch_id,
                    record.kind,
                    record.shot_id,
                    record.start_frame_id,
                    record.end_frame_id,
                    int(record.recovered),
                    payload,
                ),
            )
            row = connection.execute(
                "SELECT payload_json FROM shot_boundary_ledger WHERE boundary_id=?",
                (record.boundary_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("shot boundary ledger insert disappeared")
        stored = _record_from_payload(row[0])
        if _record_identity(stored) != _record_identity(record):
            raise ValueError("shot boundary identity conflicts with persisted record")
        return stored

    def records(self) -> tuple[ShotBoundaryRecord, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM shot_boundary_ledger "
                "WHERE session_id=? AND source_epoch_id=? ORDER BY sequence",
                (self.session_id, self.source_epoch_id),
            ).fetchall()
        return tuple(_record_from_payload(row[0]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


class ClosedSegmentAuditor:
    """Replay ordered closed-segment observations and recover missing boundaries."""

    def __init__(
        self,
        *,
        session_id: str,
        source_epoch_id: str,
        config: ShotWindowConfig,
        initial_shot_number: int = 0,
        calibration: TableCalibration | None = None,
        visual_motion_config: VisualMotionConfig | None = None,
        minimum_frame_id: Callable[[], int | None] | None = None,
    ) -> None:
        if not session_id.strip() or not source_epoch_id.strip():
            raise ValueError("closed segment auditor identity is invalid")
        self._session_id = session_id
        self._source_epoch_id = source_epoch_id
        if (calibration is None) != (visual_motion_config is None):
            raise ValueError("visual segment audit requires calibration and motion config")
        self._coordinator = (
            IncrementalShotWindowCoordinator(
                config,
                initial_shot_number=initial_shot_number,
            )
            if calibration is None
            else None
        )
        self._visual_detector = (
            None
            if calibration is None
            else VisualShotBoundaryDetector(
                config,
                initial_shot_number=initial_shot_number,
            )
        )
        self._calibration = calibration
        self._visual_motion_config = visual_motion_config
        self._previous_bgr = None
        self._last_frame_id: int | None = None
        self._minimum_frame_id = minimum_frame_id

    def audit(
        self,
        segment: Sequence[ObservationFrame | CaptureFrame],
        existing_boundaries: Iterable[ShotBoundaryRecord],
    ) -> tuple[RecoveredShotBoundary, ...]:
        existing_ids = {record.boundary_id for record in existing_boundaries}
        recovered: list[RecoveredShotBoundary] = []
        for frame in sorted(
            segment,
            key=lambda item: (
                item.capture_frame_id if isinstance(item, CaptureFrame) else item.frame_id
            ),
        ):
            frame_id = frame.capture_frame_id if isinstance(frame, CaptureFrame) else frame.frame_id
            if self._minimum_frame_id is not None:
                minimum = self._minimum_frame_id()
                if minimum is None or frame_id < minimum:
                    continue
            if self._last_frame_id is not None and frame_id <= self._last_frame_id:
                continue
            boundaries = self._boundaries(frame)
            for boundary in boundaries:
                record = _boundary_record(
                    boundary,
                    session_id=self._session_id,
                    source_epoch_id=self._source_epoch_id,
                    recovered=True,
                )
                if record.boundary_id not in existing_ids:
                    recovered.append(RecoveredShotBoundary(record.boundary_id, boundary))
                    existing_ids.add(record.boundary_id)
            self._last_frame_id = frame_id
        return tuple(recovered)

    def _boundaries(
        self,
        frame: ObservationFrame | CaptureFrame,
    ) -> tuple[LiveShotBoundary, ...]:
        if isinstance(frame, ObservationFrame):
            if self._coordinator is None:
                raise TypeError("visual closed segment auditor requires capture frames")
            return self._coordinator.push(frame)
        detector = self._visual_detector
        calibration = self._calibration
        config = self._visual_motion_config
        if detector is None or calibration is None or config is None:
            raise TypeError("observation closed segment auditor requires observation frames")
        current = frame.decode_bgr()
        previous = self._previous_bgr
        self._previous_bgr = current
        if previous is None:
            return ()
        moving = table_visual_motion_detected(previous, current, calibration, config)
        return detector.push(
            frame_id=frame.capture_frame_id,
            timestamp_ms=frame.timestamp_ms,
            moving=moving,
        )


class SessionSegmentAuditWorker:
    """Audit committed segment files on a separate thread and persist recoveries."""

    def __init__(
        self,
        *,
        recorder: ContinuousSessionRecorder,
        ledger: ShotBoundaryLedger,
        auditor: ClosedSegmentAuditor,
    ) -> None:
        self._recorder = recorder
        self._ledger = ledger
        self._auditor = auditor
        self._segments: queue.SimpleQueue[SessionSegment | None] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._input_closed = False
        self._audited_segments = 0
        self._recovered_boundaries = 0
        self._failure: BaseException | None = None
        self._previous_segment: SessionSegment | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="mainline1-session-segment-audit",
            daemon=True,
        )
        self._thread.start()

    @property
    def health(self) -> SegmentAuditHealth:
        with self._condition:
            failure = self._failure
            status = (
                "degraded"
                if failure is not None
                else "closed"
                if not self._thread.is_alive()
                else "running"
            )
            return SegmentAuditHealth(
                status=status,
                audited_segments=self._audited_segments,
                recovered_boundaries=self._recovered_boundaries,
                reason=(None if failure is None else f"{type(failure).__name__}: {failure}"),
            )

    def submit(self, segment: SessionSegment) -> None:
        with self._condition:
            if self._input_closed:
                raise RuntimeError("segment audit input is closed")
        self._segments.put(segment)

    def close_input(self) -> None:
        with self._condition:
            if self._input_closed:
                return
            self._input_closed = True
        self._segments.put(None)

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("segment audit close timeout is invalid")
        self.close_input()
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("segment audit did not stop within timeout")
        if self._failure is not None:
            raise RuntimeError(
                f"segment audit failed: {type(self._failure).__name__}: {self._failure}"
            ) from self._failure

    def _run(self) -> None:
        try:
            while True:
                segment = self._segments.get()
                if segment is None:
                    return
                frames: tuple[CaptureFrame, ...] = ()
                previous = self._previous_segment
                if previous is not None:
                    frames += self._recorder.read_segment(previous)
                frames += self._recorder.read_segment(segment)
                recovered = self._auditor.audit(frames, self._ledger.records())
                for item in recovered:
                    self._ledger.record(item.boundary, recovered=True)
                with self._condition:
                    self._audited_segments += 1
                    self._recovered_boundaries += len(recovered)
                    self._previous_segment = segment
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()


def _boundary_record(
    boundary: LiveShotBoundary,
    *,
    session_id: str,
    source_epoch_id: str,
    recovered: bool,
) -> ShotBoundaryRecord:
    window = boundary.window
    if window is None:
        raise ValueError("persisted shot boundary requires a window")
    end_frame_id = window.end_frame_id if boundary.kind.value == "SHOT_SETTLED" else None
    identity = {
        "session_id": session_id,
        "source_epoch_id": source_epoch_id,
        "kind": boundary.kind.value,
        "shot_id": window.shot_id,
        "start_frame_id": window.start_frame_id,
        "end_frame_id": end_frame_id,
    }
    boundary_id = (
        "boundary-"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return ShotBoundaryRecord(
        boundary_id=boundary_id,
        session_id=session_id,
        source_epoch_id=source_epoch_id,
        kind=boundary.kind.value,
        shot_id=window.shot_id,
        start_frame_id=window.start_frame_id,
        end_frame_id=end_frame_id,
        recovered=recovered,
    )


def _record_identity(record: ShotBoundaryRecord) -> tuple[object, ...]:
    return (
        record.boundary_id,
        record.session_id,
        record.source_epoch_id,
        record.kind,
        record.shot_id,
        record.start_frame_id,
        record.end_frame_id,
    )


def _canonical_payload(record: ShotBoundaryRecord) -> str:
    return json.dumps(
        {
            "boundary_id": record.boundary_id,
            "session_id": record.session_id,
            "source_epoch_id": record.source_epoch_id,
            "kind": record.kind,
            "shot_id": record.shot_id,
            "start_frame_id": record.start_frame_id,
            "end_frame_id": record.end_frame_id,
            "recovered": record.recovered,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_from_payload(payload: str) -> ShotBoundaryRecord:
    try:
        raw = json.loads(payload)
        return ShotBoundaryRecord(**raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("shot boundary ledger payload is invalid") from exc
