"""Durable capture stop is independent of a final detected-shot manifest."""

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.unit.edge.route_assistance.test_recording_evidence import _store


def _stop(store, session, root):
    root.mkdir(exist_ok=True)
    return store.record_recording_capture_end(
        session,
        "epoch",
        {"frame_id": 9, "captured_at_ns": 150_000_003},
        root / "session-recording.json",
        first_frame_id=0,
        width=16,
        height=12,
        last_frame_sha256="a" * 64,
    )


def test_capture_stop_is_immutable_and_survives_reopen_before_tail_or_detection(tmp_path):
    store, session = _store(tmp_path)
    receipt = _stop(store, session, tmp_path / "raw")
    assert receipt["contract"] == "recording-capture-end/v1"
    assert receipt["tail_status"] == "PENDING"
    assert receipt["processing_status"] == "PROCESSING"
    assert store.get_recording_end_manifest(session) is None
    store.end_session(session)
    reopened = SQLiteFieldRecordStore(store.path)
    assert reopened.load_session(session).status.value == "ended"
    assert reopened.get_recording_capture_end(session) == receipt
    head = reopened.get_postgame_manifest(session)
    assert head is not None and head.expected_jobs == ()
    assert reopened.get_recording_end_manifest(session) is None
    assert _stop(reopened, session, tmp_path / "raw") == receipt
    with pytest.raises(ValueError, match="conflict"):
        reopened.record_recording_capture_end(
            session,
            "epoch",
            {"frame_id": 10, "captured_at_ns": 166_666_670},
            tmp_path / "raw/session-recording.json",
            first_frame_id=0,
            width=16,
            height=12,
            last_frame_sha256="a" * 64,
        )


def test_ended_capture_accepts_only_original_late_start_within_its_cutoff(tmp_path):
    store, session = _store(tmp_path)
    _stop(store, session, tmp_path / "raw")
    store.end_session(session)
    store.record_recording_start(session, "late-1", "epoch", 4, 67)
    assert store.get_shot_ownership(session, "late-1").evidence_status == "unknown"
    for epoch, frame, at in [("other", 5, 84), ("epoch", 10, 167), ("epoch", 5, 151)]:
        with pytest.raises(ValueError, match="cutoff|epoch"):
            store.record_recording_start(session, "forbidden", epoch, frame, at)
    assert [row.shot_id for row in store.list_shots(session)] == ["late-1"]


def test_unended_historical_session_still_requires_old_drain_or_new_capture_proof(tmp_path):
    store, session = _store(tmp_path)
    with pytest.raises(ValueError, match="RECORDING_DRAIN_REQUIRED"):
        store.end_session(session)
    assert store.get_recording_capture_end(session) is None


def test_progress_cannot_claim_processing_complete_without_a_final_manifest(tmp_path):
    store, session = _store(tmp_path)
    _stop(store, session, tmp_path / "raw")
    store.end_session(session)
    with pytest.raises(ValueError, match="final|complete|progress"):
        store.set_recording_capture_progress(session, processing_status="COMPLETE")
    assert store.get_recording_capture_end(session)["processing_status"] == "PROCESSING"


def test_schema29_upgrade_adds_no_capture_receipt_to_historical_recording(tmp_path):
    import sqlite3

    from tests.unit.edge.route_assistance.test_recording_end import _prepared

    store, session, _ = _prepared(tmp_path)
    store.end_session(session)
    original = store.get_recording_end_manifest(session)
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TABLE field_recording_captures")
        db.execute("PRAGMA user_version=29")
    upgraded = SQLiteFieldRecordStore(store.path)
    assert upgraded.get_recording_capture_end(session) is None
    assert upgraded.get_recording_end_manifest(session) == original
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


def test_late_start_cannot_use_frames_from_before_this_capture(tmp_path):
    store, session = _store(tmp_path)
    root = tmp_path / "raw"
    root.mkdir()
    store.record_recording_capture_end(
        session,
        "epoch",
        {"frame_id": 9, "captured_at_ns": 150_000_003},
        root / "session-recording.json",
        first_frame_id=5,
        width=16,
        height=12,
        last_frame_sha256="a" * 64,
    )
    store.end_session(session)
    with pytest.raises(ValueError, match="cutoff|range"):
        store.record_recording_start(session, "foreign-old-frame", "epoch", 4, 67)
    assert store.list_shots(session) == ()


def _background_complete(tmp_path):
    from billiards_ai.lab.mainline1_session_recording import (
        ContinuousSessionRecorder,
        SessionRecordingConfig,
    )

    store, session = _store(tmp_path)
    raw = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id=session,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    raw.close()
    store.record_recording_capture_end(
        session,
        "epoch",
        None,
        raw.finalized_manifest_path,
        first_frame_id=None,
        width=0,
        height=0,
        last_frame_sha256=None,
    )
    store.end_session(session)
    store.prepare_recording_end(
        session,
        "epoch",
        None,
        raw.finalized_manifest_path,
        raw.finalized_segment_receipts,
        background=True,
    )
    return store, session


def test_late_progress_cannot_downgrade_completed_background_processing(tmp_path):
    store, session = _background_complete(tmp_path)
    with pytest.raises(ValueError, match="complete|final"):
        store.set_recording_capture_progress(session, processing_status="FAILED")
    assert store.get_recording_capture_end(session)["processing_status"] == "COMPLETE"


def test_background_finalization_does_not_expire_later_end_retries(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance import persistence

    store, session = _background_complete(tmp_path)
    later = persistence.time.monotonic() + 7
    monkeypatch.setattr(persistence.time, "monotonic", lambda: later)
    assert store.end_session(session).status.value == "ended"
    assert store.get_recording_end_manifest(session)["media_status"] == "EMPTY"
