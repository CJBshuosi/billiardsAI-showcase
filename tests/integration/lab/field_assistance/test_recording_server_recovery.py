"""A newly built field service must resume the stopped original disk job."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import _until
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


@pytest.mark.parametrize(
    "crash_frame,stop_stage,processing_start_delay_seconds",
    [(110, "ended", 0), (185, "ended", 0), (110, "intent", 0), (185, "ended", 4.0)],
)
def test_server_restart_finishes_original_stopped_capture_without_reopening_its_camera(
    tmp_path, monkeypatch, crash_frame, stop_stage, processing_start_delay_seconds
):
    capture = _crashed_capture(tmp_path, crash_frame, stop_stage, processing_start_delay_seconds)
    session = capture["session_id"]
    raw = tuple((tmp_path / "rings/sessions" / session).rglob("session-recording/*"))
    original = {p: p.read_bytes() for p in raw if p.is_file()}
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        source_epoch_id="restarted-epoch",
    )
    try:
        state = server.field_state
        assert state.session_id != session
        before = state.record_store.get_shot_ownership(session, "shot-0001")
        assert before.evidence_status == "unknown"
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(session),
            timeout_seconds=5,
            label="field service resumes stopped original processing",
        )
        assert final["source_epoch_id"] == "recording-epoch"
        assert final["cutoff"] == capture["cutoff"]
        assert [row["shot_id"] for row in final["shots"]] == ["shot-0001", "shot-0002"]
        assert state.record_store.get_shot_ownership(session, "shot-0001") == before
        assert all(
            state.record_store.get_recording_evidence(session, row["shot_id"])["media_status"]
            == "RECORDED"
            for row in final["shots"]
        )
        assert (
            state.record_store.get_recording_capture_end(session)["processing_status"] == "COMPLETE"
        )
        assert {p: p.read_bytes() for p in original} == original
        assert not semantic.is_set()
    finally:
        release.set()
        server.server_close()


def _crashed_capture(
    tmp_path, crash_frame=110, stop_stage="ended", processing_start_delay_seconds=0
):
    project = Path(__file__).resolve().parents[4]
    lost = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.integration.lab.field_assistance.recording_server_recovery_worker",
            str(tmp_path),
            str(crash_frame),
            stop_stage,
            str(processing_start_delay_seconds),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=18,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / "src"), str(project)])),
    )
    assert lost.returncode == 23, lost.stdout + lost.stderr
    return json.loads((tmp_path / "crashed-session.json").read_text())


@pytest.mark.parametrize(
    "fault,tail",
    [
        ("missing_tail", "FAILED"),
        ("config_changed", "SAVED"),
        ("missing_origin", "SAVED"),
    ],
)
def test_recovery_preserves_failure_and_raw_without_a_false_final_manifest(
    tmp_path, monkeypatch, fault, tail
):
    from dataclasses import replace

    from billiards_ai.lab import mainline1_field_runtime as field_runtime

    capture = _crashed_capture(tmp_path)
    session = capture["session_id"]
    manifest = Path(capture["raw_manifest_path"])
    if fault == "missing_tail":
        payload = json.loads(manifest.read_text())
        payload["segments"].pop()
        manifest.write_text(json.dumps(payload))
    elif fault == "config_changed":
        original = field_runtime.VisualMotionConfig
        monkeypatch.setattr(
            field_runtime,
            "VisualMotionConfig",
            lambda: replace(original(), pixel_delta_threshold=24),
        )
    else:
        with sqlite3.connect(manifest.parent.parent / "shot-boundaries.sqlite3") as db:
            db.execute("DELETE FROM recording_processing")
            db.execute("DELETE FROM shot_boundary_ledger")
    server, release, _ = _recording_server(tmp_path, monkeypatch, source_epoch_id="restarted-epoch")
    try:
        state = server.field_state
        _until(
            lambda: (
                True
                if state.record_store.get_recording_capture_end(session)["processing_status"]
                == "FAILED"
                else None
            ),
            timeout_seconds=4,
            label="honest old-scene recovery failure",
        )
        result = state.record_store.get_recording_capture_end(session)
        assert result["tail_status"] == tail
        assert result["cutoff"] == capture["cutoff"]
        assert state.record_store.get_recording_end_manifest(session) is None
        assert state.record_store.get_postgame_manifest(session) is not None
        view = state.postgame_review_payload()
        assert view["phase"] == "RECORDING_PROCESSING"
        assert view["processing"]["status"] == "FAILED" and not view["can_complete"]
        assert (
            state.record_store.get_shot_ownership(session, "shot-0001").evidence_status == "unknown"
        )
        assert manifest.exists() and tuple(manifest.parent.glob("*.mjpgseg"))
    finally:
        release.set()
        server.server_close()
