"""The whole-match observer samples committed work, not an obsolete live FIFO."""

import threading
from types import SimpleNamespace

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.mainline1_field_runtime import RuntimeHandle
from billiards_ai.lab.mainline1_recording_processing import RecordingProcessingJournal
from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger
from billiards_ai.lab.mainline1_session_recording import SessionRecordingEvidenceStore
from scripts.algorithm import run_recording_first_fullgame as runner
from tests.unit.lab.test_mainline1_shot_jobs import _frame
from tests.unit.lab.test_recording_committed_source import _recorder


def test_capture_sample_distinguishes_processed_checkpoint_from_delivered_and_uncommitted_frames(
    tmp_path,
):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    session = store.open_or_resume_session(artifact_id="sample", session_policy="recording-v1")
    raw = _recorder(tmp_path)
    evidence = SessionRecordingEvidenceStore(
        recorder=raw,
        session_id=session.session_id,
        pre_roll_frames=0,
        post_roll_frames=0,
        maximum_shot_frames=1800,
    )
    source = evidence.committed_source(source_epoch_id="epoch-1")
    ledger = ShotBoundaryLedger(
        tmp_path / "boundaries.sqlite3",
        session_id=session.session_id,
        source_epoch_id="epoch-1",
    )
    journal = RecordingProcessingJournal(
        path=ledger.path,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        config_sha256="a" * 64,
        initial_shot_number=0,
    )
    try:
        for index in range(121):
            evidence.append(_frame(index))
        assert raw.wait_until_committed(119, timeout_seconds=1)
        rows = source.frames(threading.Event())
        assert next(rows).capture_frame_id == 0
        first = source.current_segment_input
        journal.admit_segment(first)
        for _ in range(59):
            next(rows)
        journal.checkpoint(first)
        for _ in range(6):
            next(rows)
        journal.admit_segment(source.current_segment_input)
        handle = RuntimeHandle(
            None,
            None,
            recording_store=evidence,
            recording_source=source,
        )
        handle._runtime = SimpleNamespace(
            current_capture_frame=_frame(120),
            _recording_processing=journal,
        )
        state = SimpleNamespace(record_store=store, session_id=session.session_id)
        sample = runner._capture_sample(state, handle, started=0)
        assert sample["frame"] == 120
        assert sample["committed_frame_id"] == 119
        assert sample["delivered_frame_id"] == 65
        assert sample["processed_frame_id"] == 59
        assert sample["detection_backlog_frames"] == 60
        assert sample["buffered_frames"] == 1
        assert sample["shots"] == 0
        assert journal.state["checkpoint"]["last_frame_id"] == 59
    finally:
        source.cancel()
        evidence.close()


def test_observer_http_request_obeys_remaining_budget_when_service_is_waiting():
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import pytest

    entered, release = threading.Event(), threading.Event()

    class HeldRequest(BaseHTTPRequestHandler):
        def do_GET(self):
            entered.set()
            release.wait(2)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), HeldRequest)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            runner._request(server, "/held", timeout_seconds=0.1)
        assert entered.is_set()
        assert time.monotonic() - started < 0.7
    finally:
        release.set()
        server.shutdown()
        thread.join(2)
        server.server_close()
