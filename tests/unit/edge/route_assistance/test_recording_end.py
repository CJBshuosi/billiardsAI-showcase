"""Recording end proof gates the existing session and command authorities."""

import sqlite3
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)
from tests.unit.edge.route_assistance.test_recording_evidence import _store


def _prepared(tmp_path, *, empty=False):
    store, session = _store(tmp_path)
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id=session,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    if not empty:
        for frame in range(3):
            recorder.append(CaptureFrame("epoch", frame, frame * 16666667, 16, 12, b"pixels"))
    recorder.close()
    store.prepare_recording_end(
        session,
        "epoch",
        None if empty else {"frame_id": 2, "captured_at_ns": 33333334},
        recorder.finalized_manifest_path,
        recorder.finalized_segment_receipts,
    )
    return store, session, recorder


def test_direct_end_and_phone_store_cannot_bypass_recording_drain(tmp_path):
    store, session = _store(tmp_path)
    with pytest.raises(ValueError, match="RECORDING_DRAIN_REQUIRED"):
        store.end_session(session)
    envelope = FieldCommandEnvelope("end", session, "alice", "end_session", {}, 1, 1)
    with pytest.raises(ValueError, match="RECORDING_DRAIN_REQUIRED"):
        store.apply_field_command(envelope, reduce_field_command)
    assert store.load_session(session).status.value == "open"
    assert store.get_postgame_manifest(session) is None
    assert not store.list_command_ledger(session)


@pytest.mark.parametrize("empty", [False, True])
def test_prepared_end_is_internal_and_final_commit_is_atomic_and_reopenable(
    tmp_path, monkeypatch, empty
):
    store, session, recorder = _prepared(tmp_path, empty=empty)
    assert store.get_recording_end_manifest(session) is None
    assert store.load_session(session).status.value == "open"
    original = store._freeze_postgame_manifest

    def fail_after_freeze(*args):
        original(*args)
        raise sqlite3.OperationalError("injected final transaction failure")

    monkeypatch.setattr(store, "_freeze_postgame_manifest", fail_after_freeze)
    with pytest.raises(ValueError, match="database"):
        store.end_session(session)
    assert store.load_session(session).status.value == "open"
    assert store.get_postgame_manifest(session) is None
    assert store.get_recording_end_manifest(session) is None
    reopened = SQLiteFieldRecordStore(store.path)
    reopened.end_session(session)
    frozen = reopened.get_recording_end_manifest(session)
    assert frozen["media_status"] == ("EMPTY" if empty else "RECORDED")
    assert (frozen["cutoff"] is None) is empty
    assert frozen["source_epoch_id"] == "epoch"
    assert frozen["analysis_status"] == "NOT_REQUESTED"
    reopened.end_session(session)
    assert reopened.get_recording_end_manifest(session) == frozen
    assert reopened.active_postgame_review("recording").session_id == session
    reopened.complete_postgame_review(
        "recording", session, reopened.get_postgame_manifest(session).manifest_sha256
    )
    assert reopened.active_postgame_review("recording") is None


@pytest.mark.parametrize("change", ["manifest", "segment", "shot", "epoch"])
def test_prepared_proof_rejects_changed_original_evidence(tmp_path, change):
    store, session, recorder = _prepared(tmp_path)
    if change == "manifest":
        path = recorder.finalized_manifest_path
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "segment":
        path = (
            recorder.finalized_manifest_path.parent / recorder.finalized_segment_receipts[0]["name"]
        )
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif change == "epoch":
        with pytest.raises(ValueError, match="cutoff|lineage|conflicts"):
            store.prepare_recording_end(
                session,
                "foreign",
                {"frame_id": 2, "captured_at_ns": 33333334},
                recorder.finalized_manifest_path,
                recorder.finalized_segment_receipts,
            )
        return
    else:
        with pytest.raises(ValueError, match="sealed"):
            store.record_recording_start(session, "late", "epoch", 2, 33)
        return
    with pytest.raises((ValueError, RuntimeError), match="changed|hash"):
        store.end_session(session)
    assert store.load_session(session).status.value == "open"
    assert store.get_postgame_manifest(session) is None


def test_phone_end_retains_original_ledger_and_exact_retry(tmp_path):
    store, session, _ = _prepared(tmp_path)
    envelope = FieldCommandEnvelope("end", session, "alice", "end_session", {}, 1, 1)
    first = store.apply_field_command(envelope, reduce_field_command)
    assert first.outcome == "applied"
    frozen = store.get_recording_end_manifest(session)
    reopened = SQLiteFieldRecordStore(store.path)
    assert reopened.apply_field_command(envelope, reduce_field_command) == first
    conflict = reopened.apply_field_command(
        replace(envelope, actor_user_id="bob"), reduce_field_command
    )
    assert conflict.result_code == "COMMAND_IDEMPOTENCY_CONFLICT"
    assert reopened.get_recording_end_manifest(session) == frozen
    assert len(reopened.list_command_ledger(session)) == 1


def test_recording_end_reuses_writer_receipts_for_hour_scale_without_raw_reads(
    tmp_path, monkeypatch
):
    import time
    from pathlib import Path

    store, session = _store(tmp_path)
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id=session,
        expected_fps=1,
        config=SessionRecordingConfig(
            reserve_bytes=1, minimum_write_mbps=0, memory_buffer_seconds=10
        ),
    )
    try:
        for frame in range(3600):
            recorder.append(CaptureFrame("epoch", frame, frame * 1_000_000_000, 16, 12, b"pixel"))
            if frame % 8 == 7:
                recorder.commit_through(frame)
        recorder.close()
        assert len(recorder.finalized_segment_receipts) == 3600
        original_open = Path.open

        def refuse_raw_read(path, mode="r", *args, **kwargs):
            if path.suffix == ".mjpgseg" and "r" in mode:
                raise AssertionError("end rescanned durable raw media")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse_raw_read)
        started = time.monotonic()
        store.prepare_recording_end(
            session,
            "epoch",
            {"frame_id": 3599, "captured_at_ns": 3_599_000_000_000},
            recorder.finalized_manifest_path,
            recorder.finalized_segment_receipts,
            deadline=started + 6,
        )
        store.end_session(session)
        elapsed = time.monotonic() - started
        assert elapsed < 6
        frozen = store.get_recording_end_manifest(session)
        assert len(frozen["raw_segments"]) == 3600
        print(f"3600 writer segments prepared and frozen in {elapsed:.3f}s; raw reads=0")
    finally:
        recorder.close()


def test_expired_budget_and_changed_writer_receipt_never_prepare_end(tmp_path):
    import time

    store, session, recorder = _prepared(tmp_path)
    with pytest.raises(ValueError, match="TIMEOUT"):
        store.prepare_recording_end(
            session,
            "epoch",
            {"frame_id": 2, "captured_at_ns": 33333334},
            recorder.finalized_manifest_path,
            recorder.finalized_segment_receipts,
            deadline=time.monotonic() - 1,
        )
    assert store.get_recording_end_manifest(session) is None


def test_prepared_crash_refuses_new_capture_until_explicit_original_proof_end(tmp_path):
    store, session, _ = _prepared(tmp_path)
    reopened = SQLiteFieldRecordStore(store.path)
    with pytest.raises(ValueError, match="recovery"):
        reopened.open_or_resume_session(artifact_id="recording", session_policy="recording-v1")
    assert reopened.get_recording_end_manifest(session) is None
    reopened.end_session(session)
    assert reopened.get_recording_end_manifest(session)["source_epoch_id"] == "epoch"


def _original_shot_reference(tmp_path, *, incomplete=False):
    from hashlib import sha256

    from billiards_ai.edge.route_assistance.records import RecordedMediaV1, RecordingBoundaryV1
    from billiards_ai.lab.mainline1_session_recording import SessionRecordingEvidenceStore
    from billiards_ai.lab.mainline1_shot_jobs import materialize_recorded_interval_reference

    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "epoch", 1, 16)
    boundary = RecordingBoundaryV1(
        "shot-1",
        "epoch",
        1,
        2,
        16,
        33,
        not incomplete,
        "USER_ENDED_RECORDING" if incomplete else None,
    )
    store.record_recording_boundary(session, boundary)
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id=session,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    try:
        for frame in range(6):
            recorder.append(CaptureFrame("epoch", frame, frame * 16_666_667, 16, 12, b"jpeg"))
        evidence = SessionRecordingEvidenceStore(
            recorder=recorder,
            session_id=session,
            pre_roll_frames=1,
            post_roll_frames=3,
            maximum_shot_frames=60,
        )
        interval = evidence.freeze_window(
            shot_id="shot-1",
            source_epoch_id="epoch",
            start_frame_id=1,
            end_frame_id=5,
        )
        materialize_recorded_interval_reference(interval, tmp_path / "ref", recording_identity=True)
        path = tmp_path / "ref" / "interval.json"
        store.record_recording_media(
            session,
            RecordedMediaV1(
                boundary,
                "FAILED" if incomplete else "RECORDED",
                "recording-review",
                str(path),
                sha256(path.read_bytes()).hexdigest(),
                interval.media_sha256,
                interval.start_frame_id,
                interval.end_frame_id,
                "INCOMPLETE_RECORDING_WINDOW" if incomplete else None,
            ),
        )
    finally:
        recorder.close()
    return store, session, recorder, path


def _prepare_original_reference(store, session, recorder):
    store.prepare_recording_end(
        session,
        "epoch",
        dict(frame_id=5, captured_at_ns=5 * 16_666_667),
        recorder.finalized_manifest_path,
        recorder.finalized_segment_receipts,
    )


@pytest.mark.parametrize("timing", ["before_prepare", "local_commit", "phone_commit"])
@pytest.mark.parametrize("mutation", ["delete", "change"])
def test_end_rejects_missing_or_changed_original_interval_reference(tmp_path, timing, mutation):
    store, session, recorder, path = _original_shot_reference(tmp_path)
    original = path.read_bytes()
    evidence = store.get_recording_evidence(session, "shot-1")
    if timing != "before_prepare":
        _prepare_original_reference(store, session, recorder)
    if mutation == "delete":
        path.unlink()
    else:
        path.write_text("{}")
    envelope = FieldCommandEnvelope("end-ref", session, "alice", "end_session", {}, 1, 1)
    with pytest.raises(ValueError, match="recording reference"):
        if timing == "before_prepare":
            _prepare_original_reference(store, session, recorder)
        elif timing == "local_commit":
            store.end_session(session)
        else:
            store.apply_field_command(envelope, reduce_field_command)
    assert store.load_session(session).status.value == "open"
    assert store.get_recording_end_manifest(session) is None
    assert store.get_postgame_manifest(session) is None
    assert store.active_postgame_review("recording") is None
    assert store.get_recording_evidence(session, "shot-1") == evidence
    assert not store.list_command_ledger(session)
    assert recorder.finalized_manifest_path.is_file()
    assert list(recorder.finalized_manifest_path.parent.glob("*.mjpgseg"))
    # Restoring the exact original small reference allows explicit original-proof recovery.
    path.write_bytes(original)
    reopened = SQLiteFieldRecordStore(store.path)
    if timing == "before_prepare":
        _prepare_original_reference(reopened, session, recorder)
    reopened.end_session(session)
    assert reopened.get_recording_end_manifest(session)["media_status"] == "RECORDED"


@pytest.mark.parametrize("incomplete", [False, True])
def test_end_reference_validation_preserves_valid_windows_without_raw_rescan(
    tmp_path,
    monkeypatch,
    incomplete,
):
    from pathlib import Path

    store, session, recorder, _ = _original_shot_reference(tmp_path, incomplete=incomplete)
    original_open = Path.open

    def refuse_raw_read(path, mode="r", *args, **kwargs):
        if path.suffix == ".mjpgseg" and "r" in mode:
            raise AssertionError("end rescanned raw media")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_raw_read)
    _prepare_original_reference(store, session, recorder)
    store.end_session(session)
    frozen = store.get_recording_end_manifest(session)
    assert frozen["media_status"] == ("RETAINED_INCOMPLETE" if incomplete else "RECORDED")
    assert frozen["shots"][0]["shot_id"] == "shot-1"
    assert (store.list_shots(session)[0].settled_ms is None) is incomplete


@pytest.mark.parametrize("mutation", ["symlink", "oversized"])
def test_end_reference_requires_safe_bounded_original_file(tmp_path, mutation):
    store, session, recorder, path = _original_shot_reference(tmp_path)
    if mutation == "symlink":
        target = path.with_name("moved-interval.json")
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="recording reference"):
        _prepare_original_reference(store, session, recorder)
    assert store.load_session(session).status.value == "open"
    assert store.get_recording_end_manifest(session) is None
    assert store.get_postgame_manifest(session) is None


def test_end_reference_must_use_the_original_raw_manifest_root(tmp_path):
    store, session, _, path = _original_shot_reference(tmp_path)
    other = ContinuousSessionRecorder(
        root=tmp_path / "other-raw",
        session_id=session,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    try:
        for frame in range(6):
            other.append(CaptureFrame("epoch", frame, frame * 16_666_667, 16, 12, b"jpeg"))
    finally:
        other.close()
    with pytest.raises(ValueError, match="recording reference lineage"):
        _prepare_original_reference(store, session, other)
    assert path.is_file()
    assert store.load_session(session).status.value == "open"
    assert store.get_postgame_manifest(session) is None


@pytest.mark.parametrize("timing", ["prepare", "local_commit", "phone_commit"])
@pytest.mark.parametrize("mutate", [False, True])
def test_end_accepts_windows_reference_clocks_but_rejects_in_read_mutation(
    tmp_path, monkeypatch, timing, mutate
):
    from tests.unit.lab.test_recording_windows_stat_identity import _windows_stat_views

    store, session, recorder, path = _original_shot_reference(tmp_path)
    if timing != "prepare":
        _prepare_original_reference(store, session, recorder)
    _windows_stat_views(monkeypatch, path, mutate_after_read=mutate, path_stat=True)

    def end():
        if timing == "prepare":
            _prepare_original_reference(store, session, recorder)
        if timing == "phone_commit":
            envelope = FieldCommandEnvelope(
                "windows-end", session, "alice", "end_session", {}, 1, 1
            )
            store.apply_field_command(envelope, reduce_field_command)
        else:
            store.end_session(session)

    if mutate:
        # The real rewrite preserves size and mtime, and occurs after the read.
        # Only the fd-to-fd change-time fence can detect it before SHA validation.
        with pytest.raises(ValueError, match="recording reference changed while reading"):
            end()
        assert store.load_session(session).status.value == "open"
        assert store.get_recording_end_manifest(session) is None
        assert store.get_postgame_manifest(session) is None
        assert not store.list_command_ledger(session)
    else:
        end()
        frozen = store.get_recording_end_manifest(session)
        assert frozen["media_status"] == "RECORDED"
        assert frozen["cutoff"]["frame_id"] == 5
        assert frozen["shots"][0]["shot_id"] == "shot-1"
        assert store.load_session(session).status.value == "ended"
