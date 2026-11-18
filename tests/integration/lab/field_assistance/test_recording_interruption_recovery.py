"""Recover committed pixels after real capture death without a user stop receipt."""

import http.client
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import _until
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


def _lost_unstopped_capture(root, pause_after=380, *, managed=False):
    project = Path(__file__).resolve().parents[4]
    lost = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.integration.lab.field_assistance.recording_server_recovery_worker",
            str(root),
            "110",
            "interrupted_managed" if managed else "interrupted",
            "0",
            str(pause_after),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=18,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / "src"), str(project)])),
    )
    assert lost.returncode == 23, lost.stdout + lost.stderr
    original = json.loads((root / "interrupted-session.json").read_text())
    store = SQLiteFieldRecordStore(root / "field.sqlite3")
    assert store.get_recording_capture_end(original["session_id"]) is None
    assert store.load_session(original["session_id"]).status.value == "open"
    assert (
        store.get_recording_evidence(original["session_id"], "shot-0001")["media_status"]
        == "PENDING"
    )
    raw = (
        root
        / "rings/sessions"
        / original["session_id"]
        / original["source_epoch_id"]
        / "session-recording"
    )
    original["raw"] = raw
    original["bytes"] = {p.name: p.read_bytes() for p in raw.iterdir() if p.is_file()}
    manifest = json.loads(original["bytes"]["session-recording.json"])
    original["committed"] = manifest["segments"][-1]["last_frame_id"]
    assert original["committed"] < 380
    return original


@pytest.mark.parametrize("pause_after", [140, 380])
def test_real_unstopped_capture_recovers_only_committed_prefix(tmp_path, monkeypatch, pause_after):
    original = _lost_unstopped_capture(tmp_path, pause_after)
    for name in (".pending-uncommitted", "orphan-uncommitted.mjpgseg"):
        (original["raw"] / name).write_bytes(b"not admitted video")
        original["bytes"][name] = b"not admitted video"
    sid = original["session_id"]
    server, release, semantic = _recording_server(
        tmp_path, monkeypatch, source_epoch_id="new-camera-epoch"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = server.field_state
        assert state.session_id != sid
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(sid),
            timeout_seconds=8,
            label="recover interrupted committed prefix",
        )
        receipt = state.record_store.get_recording_capture_end(sid)
        assert receipt["contract"] == "recording-capture-end/v2"
        assert receipt["end_reason"] == "PROCESS_INTERRUPTED"
        assert receipt["tail_completeness"] == "UNKNOWN"
        assert receipt["cutoff"]["frame_id"] == original["committed"]
        assert receipt["source_epoch_id"] == original["source_epoch_id"]
        assert final["contract"] == "recording-end/v2"
        assert final["end_reason"] == "PROCESS_INTERRUPTED"
        assert final["tail_completeness"] == "UNKNOWN"
        assert final["last_frame_id"] == original["committed"]
        assert state.record_store.get_shot_ownership(sid, "shot-0001").evidence_status == "unknown"
        if pause_after == 380:
            review = _until(
                lambda: v if (v := state.postgame_review_payload())["available_count"] else None,
                timeout_seconds=8,
                label="interrupted original playable media",
            )
            clip = next(c for c in review["clips"] if c["media_url"])
            c = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                c.request("GET", clip["media_url"])
                response = c.getresponse()
                assert response.status == 200
                assert response.getheader("Content-Type") == "video/mp4"
                encoded = response.read()
                assert len(encoded) > 0
                returned = tmp_path / "interrupted-http-return.mp4"
                returned.write_bytes(encoded)
                import cv2

                player = cv2.VideoCapture(str(returned))
                decoded = 0
                try:
                    assert player.isOpened()
                    while True:
                        ok, image = player.read()
                        if not ok:
                            break
                        assert image.shape == (120, 160, 3)
                        decoded += 1
                finally:
                    player.release()
                job = next(
                    j
                    for j in state.media_job_store.list_session_media_jobs(sid)
                    if j.shot_id == clip["shot_id"]
                )
                assert decoded == job.end_frame_id - job.start_frame_id + 1
            finally:
                c.close()
        else:
            review = state.postgame_review_payload()
            evidence = state.record_store.get_recording_evidence(sid, "shot-0001")
            assert evidence["boundary"]["reason"] == "PROCESS_INTERRUPTED"
            assert evidence["media_status"] == "FAILED"
            assert not evidence["boundary"]["complete"]
        assert review["phase"] == "RECORDING_INTERRUPTED"
        assert not review["can_complete"]
        head = state.record_store.get_postgame_manifest(sid)
        with pytest.raises(ValueError, match="tail is unknown"):
            state.record_store.complete_postgame_review(
                state.artifact_id, sid, head.manifest_sha256, media_jobs=state.media_job_store
            )
        with pytest.raises(ValueError, match="ownership"):
            state.record_store.save_postgame_clip(
                request_id="interrupted-save",
                session_id=sid,
                shot_id="shot-0001",
                postgame_manifest_sha256=head.manifest_sha256,
            )
        assert not state.record_store.list_saved_media_outbox()
        assert {
            p.name: p.read_bytes() for p in original["raw"].iterdir() if p.is_file()
        } == original["bytes"]
        assert not semantic.is_set()
    finally:
        release.set()
        server.shutdown()
        thread.join(6)
        server.server_close()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_origin",
        "changed_config",
        "missing_segment",
        "active_writer",
        "active_detector",
        "unproven_writer",
        "manifest_symlink",
    ],
)
def test_unverified_interrupted_prefix_stays_protected(tmp_path, monkeypatch, fault):
    import sqlite3
    from contextlib import ExitStack
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from billiards_ai.lab import mainline1_field_runtime as field_runtime

    original = _lost_unstopped_capture(tmp_path)
    sid = original["session_id"]
    context = original["raw"].parent
    with ExitStack() as held:
        if fault == "missing_origin":
            with sqlite3.connect(context / "shot-boundaries.sqlite3") as db:
                db.execute("DELETE FROM recording_processing")
        elif fault == "changed_config":
            prior = field_runtime.VisualMotionConfig
            monkeypatch.setattr(
                field_runtime,
                "VisualMotionConfig",
                lambda: replace(prior(), pixel_delta_threshold=24),
            )
        elif fault == "missing_segment":
            manifest = json.loads(original["bytes"]["session-recording.json"])
            (original["raw"] / manifest["segments"][0]["name"]).unlink()
        elif fault == "manifest_symlink":
            target = tmp_path / "moved-manifest.json"
            original_path = original["raw"] / "session-recording.json"
            original_path.rename(target)
            original_path.symlink_to(target)
        elif fault == "unproven_writer":
            import hashlib

            (
                context
                / ".session-media-locks"
                / (hashlib.sha256(b"recording-writer").hexdigest() + ".lock")
            ).unlink()
        else:
            held.enter_context(
                session_media_lock(
                    context,
                    "recording-writer" if fault == "active_writer" else "recording-detector",
                    exclusive=True,
                )
            )
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            _recording_server(tmp_path, monkeypatch, source_epoch_id="forbidden-new-epoch")
        store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
        assert store.get_recording_capture_end(sid) is None
        assert store.load_session(sid).status.value == "open"
        assert store.get_recording_evidence(sid, "shot-0001")["media_status"] == "PENDING"
        assert (original["raw"] / "session-recording.json").read_bytes() == original["bytes"][
            "session-recording.json"
        ]
        for name, data in original["bytes"].items():
            if (original["raw"] / name).exists():
                assert (original["raw"] / name).read_bytes() == data


def test_second_real_process_loss_resumes_same_interruption_and_job_identities(
    tmp_path, monkeypatch
):
    original = _lost_unstopped_capture(tmp_path)
    project = Path(__file__).resolve().parents[4]
    lost = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tests.integration.lab.field_assistance.recording_server_recovery_worker",
            str(tmp_path),
            "110",
            "recover_interrupted",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=18,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(project / "src"), str(project)])),
    )
    assert lost.returncode == 24, lost.stdout + lost.stderr
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    sid = original["session_id"]
    receipt = store.get_recording_capture_end(sid)
    assert receipt["end_reason"] == "PROCESS_INTERRUPTED"
    owner = store.get_shot_ownership(sid, "shot-0001")
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        source_epoch_id="restart-after-recovery-loss",
        fresh_session_on_start=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = server.field_state
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(sid),
            timeout_seconds=8,
            label="resume interrupted recovery after a second process loss",
        )
        assert final["end_reason"] == "PROCESS_INTERRUPTED"
        after = state.record_store.get_recording_capture_end(sid)
        assert after["capture_end_sha256"] == receipt["capture_end_sha256"]
        assert after["cutoff"] == receipt["cutoff"]
        assert [s.shot_id for s in state.record_store.list_shots(sid)] == ["shot-0001", "shot-0002"]
        assert state.record_store.get_shot_ownership(sid, "shot-0001") == owner
        _until(
            lambda: True if len(state.media_job_store.list_session_media_jobs(sid)) == 2 else None,
            timeout_seconds=5,
            label="same media inventory after recovery retry",
        )
        jobs = state.media_job_store.list_session_media_jobs(sid)
        assert len({j.job_id for j in jobs}) == 2
        assert all(j.source_epoch_id == original["source_epoch_id"] for j in jobs)
        assert {
            p.name: p.read_bytes() for p in original["raw"].iterdir() if p.is_file()
        } == original["bytes"]
        assert not semantic.is_set()
    finally:
        release.set()
        server.shutdown()
        thread.join(6)
        server.server_close()


@pytest.mark.parametrize("fresh", [False, True])
def test_managed_deferred_runtime_recovers_before_signed_new_capture_admission(
    tmp_path, monkeypatch, fresh
):
    from tests.unit.lab.field_assistance.test_managed_startup import SCOPE, Remote, coordinator

    original = _lost_unstopped_capture(tmp_path, managed=True)
    remote = Remote()
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        source_epoch_id="new-managed-epoch",
        deferred_runtime=True,
        managed_startup=coordinator(remote),
        fresh_session_on_start=fresh,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = server.field_state
        sid = original["session_id"]
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(sid),
            timeout_seconds=8,
            label="managed original committed prefix recovery",
        )
        assert final["end_reason"] == "PROCESS_INTERRUPTED"
        assert state.session_id != sid
        assert (
            state.record_store.get_latest_retired_managed_open_request(SCOPE).binding.status
            == "ended"
        )
        assert any(path.endswith("/close") for path, _ in remote.calls)
        assert {
            p.name: p.read_bytes() for p in original["raw"].iterdir() if p.is_file()
        } == original["bytes"]
        assert not semantic.is_set()
    finally:
        release.set()
        server.shutdown()
        thread.join(6)
        server.server_close()
