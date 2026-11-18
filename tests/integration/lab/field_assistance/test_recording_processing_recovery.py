"""A lost detector process must recover active/pending state from saved pixels."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)


def _prepare(root, *, frames=360):
    store = SQLiteFieldRecordStore(root / "field.sqlite3")
    session = store.open_or_resume_session(artifact_id="processing", session_policy="recording-v1")
    raw = ContinuousSessionRecorder(
        root=root / "raw",
        session_id=session.session_id,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    for i in range(frames):
        x = 30 + 2 * min(30, max(0, i - 90)) - 2 * min(30, max(0, i - 180))
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (x - 10, 50), (x + 10, 70), (180, 180, 180), -1)
        ok, jpeg = cv2.imencode(".jpg", image)
        assert ok
        raw.append(CaptureFrame("epoch", i, i * 16_666_667, 160, 120, jpeg.tobytes()))
    raw.close()
    return store, session.session_id


def _worker(root, session, crash, *, crash_after="frame", fault="exit", sink="real"):
    project = Path(__file__).resolve().parents[4]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / "src"), str(project)]))
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.integration.lab.field_assistance.recording_processing_worker",
            str(root),
            session,
            str(crash),
            crash_after,
            fault,
            sink,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=project,
        env=environment,
    )


@pytest.mark.parametrize("crash_frame", [110, 185])
def test_real_process_loss_restores_active_shot_and_pending_media_without_new_owner(
    tmp_path, crash_frame
):
    store, session = _prepare(tmp_path)
    raw_before = {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()}
    crashed = _worker(tmp_path, session, crash_frame)
    assert crashed.returncode == 23, crashed.stdout + crashed.stderr
    before = store.get_shot_ownership(session, "shot-0001")
    assert before is not None and before.evidence_status == "unknown"
    with sqlite3.connect(tmp_path / "boundaries.sqlite3") as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "recording_processing" in tables, "no durable origin/checkpoint before active shot"
        state = json.loads(
            db.execute("SELECT payload_json FROM recording_processing").fetchone()[0]
        )
    assert state["initial_shot_number"] == 0
    assert state["checkpoint"]["last_frame_id"] == (59 if crash_frame == 110 else 179)
    assert state["input_prefix"]["last_frame_id"] == (119 if crash_frame == 110 else 239)
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    rows = store.list_shots(session)
    assert [row.shot_id for row in rows] == ["shot-0001", "shot-0002"]
    assert store.get_shot_ownership(session, "shot-0001") == before
    assert all(
        store.get_recording_evidence(session, row.shot_id)["media_status"] == "RECORDED"
        for row in rows
    )
    assert {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()} == raw_before
    with sqlite3.connect(tmp_path / "boundaries.sqlite3") as db:
        state = json.loads(
            db.execute("SELECT payload_json FROM recording_processing").fetchone()[0]
        )
    assert state["checkpoint"]["last_frame_id"] == 359


def test_crash_after_final_checkpoint_still_finalizes_real_incomplete_tail(tmp_path):
    store, session = _prepare(tmp_path, frames=120)
    crashed = _worker(tmp_path, session, 119, crash_after="checkpoint")
    assert crashed.returncode == 23, crashed.stdout + crashed.stderr
    assert store.get_recording_evidence(session, "shot-0001")["media_status"] == "PENDING"
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    evidence = store.get_recording_evidence(session, "shot-0001")
    assert evidence["media_status"] == "FAILED"
    assert evidence["boundary"]["complete"] is False
    assert evidence["boundary"]["reason"] == "USER_ENDED_RECORDING"


@pytest.mark.parametrize("mutation", ["content", "identity"])
def test_changed_media_cannot_reinterpret_checkpointed_history(tmp_path, mutation):
    store, session = _prepare(tmp_path)
    crashed = _worker(tmp_path, session, 110)
    assert crashed.returncode == 23, crashed.stdout + crashed.stderr
    original_evidence = store.get_recording_evidence(session, "shot-0001")
    original_owner = store.get_shot_ownership(session, "shot-0001")
    path = tmp_path / "raw/session-recording.json"
    manifest = json.loads(path.read_bytes())
    segment = tmp_path / "raw" / manifest["segments"][0]["name"]
    if mutation == "content":
        payload = bytearray(segment.read_bytes())
        payload[-10] ^= 1
        segment.write_bytes(payload)
        manifest["segments"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
        path.write_text(json.dumps(manifest))
    else:
        replacement = segment.with_suffix(".replacement")
        replacement.write_bytes(segment.read_bytes())
        replacement.replace(segment)
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode != 0
    assert "input prefix changed" in recovered.stderr
    assert store.get_recording_evidence(session, "shot-0001") == original_evidence
    assert store.get_shot_ownership(session, "shot-0001") == original_owner
    assert len(store.list_shots(session)) == 1


def test_missing_replay_origin_cannot_renumber_existing_recorded_shots(tmp_path):
    store, session = _prepare(tmp_path)
    crashed = _worker(tmp_path, session, 110)
    assert crashed.returncode == 23, crashed.stdout + crashed.stderr
    original = store.get_recording_evidence(session, "shot-0001")
    with sqlite3.connect(tmp_path / "boundaries.sqlite3") as db:
        db.execute("DELETE FROM recording_processing")
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode != 0
    assert "replay origin" in recovered.stderr
    assert len(store.list_shots(session)) == 1
    assert store.get_recording_evidence(session, "shot-0001") == original


def test_checkpoint_position_must_match_the_bound_input_prefix(tmp_path):
    store, session = _prepare(tmp_path)
    crashed = _worker(tmp_path, session, 110)
    assert crashed.returncode == 23, crashed.stdout + crashed.stderr
    original = store.get_recording_evidence(session, "shot-0001")
    with sqlite3.connect(tmp_path / "boundaries.sqlite3") as db:
        state = json.loads(
            db.execute("SELECT payload_json FROM recording_processing").fetchone()[0]
        )
        state["checkpoint"]["prefix_sha256"] = "d" * 64
        db.execute("UPDATE recording_processing SET payload_json=?", (json.dumps(state),))
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode != 0
    assert "input prefix changed" in recovered.stderr
    assert store.get_recording_evidence(session, "shot-0001") == original


@pytest.mark.parametrize("fault", ["exception", "materialize"])
def test_processing_error_retains_replayable_evidence_instead_of_terminalizing_capture(
    tmp_path, fault
):
    store, session = _prepare(tmp_path)
    failed = _worker(tmp_path, session, 110 if fault == "exception" else -1, fault=fault)
    assert failed.returncode == 0, failed.stdout + failed.stderr
    assert json.loads(failed.stdout)["degraded"] is True
    before = store.get_shot_ownership(session, "shot-0001")
    assert store.get_recording_evidence(session, "shot-0001")["media_status"] == "PENDING"
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    rows = store.list_shots(session)
    assert [row.shot_id for row in rows] == ["shot-0001", "shot-0002"]
    assert store.get_shot_ownership(session, "shot-0001") == before
    assert all(
        store.get_recording_evidence(session, row.shot_id)["media_status"] == "RECORDED"
        for row in rows
    )


@pytest.mark.parametrize("sink", ["none", "no_sink"])
def test_persistent_checkpoint_requires_synchronous_recording_evidence_sink(tmp_path, sink):
    store, session = _prepare(tmp_path)
    failed = _worker(tmp_path, session, -1, sink=sink)
    assert failed.returncode != 0
    assert "synchronous evidence sink" in failed.stderr
    assert store.list_shots(session) == ()


@pytest.mark.parametrize("fault", ["foreign_session", "foreign_epoch"])
def test_processing_rejects_mixed_session_or_epoch_before_recording_events(tmp_path, fault):
    store, session = _prepare(tmp_path)
    failed = _worker(tmp_path, session, -1, fault=fault)
    assert failed.returncode != 0
    assert "session/epoch" in failed.stderr
    assert store.list_shots(session) == ()


def test_reader_cancellation_does_not_create_a_fake_capture_terminal_boundary(tmp_path):
    store, session = _prepare(tmp_path)
    stopped = _worker(tmp_path, session, 110, fault="cancel")
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert store.get_recording_evidence(session, "shot-0001")["media_status"] == "PENDING"
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert [row.shot_id for row in store.list_shots(session)] == ["shot-0001", "shot-0002"]


def test_second_detector_cannot_process_the_same_original_recording_concurrently(tmp_path):
    store, session = _prepare(tmp_path)
    project = Path(__file__).resolve().parents[4]
    holder = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.integration.lab.field_assistance.recording_processing_worker",
            str(tmp_path),
            session,
            "110",
            "frame",
            "hold",
            "real",
        ],
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / "src"), str(project)])),
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "processing-held").exists():
            assert holder.poll() is None, holder.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        before = store.get_recording_evidence(session, "shot-0001")
        duplicate = _worker(tmp_path, session, -1, fault="probe")
        assert duplicate.returncode == 0, duplicate.stderr
        assert json.loads(duplicate.stdout)["degraded"] is True
        assert store.get_recording_evidence(session, "shot-0001") == before
        assert len(store.list_shots(session)) == 1
    finally:
        (tmp_path / "release-processing").write_text("release")
        out, err = holder.communicate(timeout=12)
        assert holder.returncode == 23, out + err
    recovered = _worker(tmp_path, session, -1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert [row.shot_id for row in store.list_shots(session)] == ["shot-0001", "shot-0002"]
