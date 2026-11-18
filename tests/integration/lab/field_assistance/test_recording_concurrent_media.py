"""Committed pixels may encode during play; user access stays after match end."""

import http.client
import threading

from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_first_runtime import (
    _end_http,
    _recording_server,
    _wait_frozen,
)


def test_live_committed_shots_generate_before_end_but_review_waits(tmp_path, monkeypatch):
    paused, resume = threading.Event(), threading.Event()
    server, release, semantic = _recording_server(
        tmp_path, monkeypatch, total_frames=900, pause_after=380, paused=paused, resume=resume
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    state = server.field_state
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(7)
        session = state.session_id
        available = _until(
            lambda: (
                tuple(
                    job
                    for job in state.media_job_store.list_session_media_jobs(session)
                    if state.media_job_store.snapshot(job.job_id).status.value == "available"
                )
                or None
            ),
            timeout_seconds=5,
            label="media is generated while ordinary match remains open",
        )
        assert state.record_store.load_session(session).status.value == "open"
        assert state.record_store.get_recording_capture_end(session) is None
        assert state.record_store.get_postgame_manifest(session) is None
        live = state.postgame_review_payload()
        assert live["phase"] == "LIVE_MATCH"
        assert live["expected_count"] is None
        assert live["generated_count"] >= 1
        assert live["discovered_count"] >= live["generated_count"]
        assert live["clips"] == [] and live["available_count"] == 0
        before = {
            job.job_id: state.media_job_store.snapshot(job.job_id).artifact_id for job in available
        }
        # Continue real capture after generation without restarting or injecting events.
        resume.set()
        _until(
            lambda: (
                state.projection_source.runtime.current_capture_frame.capture_frame_id > 400 or None
            ),
            timeout_seconds=2,
            label="capture keeps moving while playback generation runs",
        )
        status, reply = _end_http(server, session)
        assert status == 200, reply
        _wait_frozen(state, session)
        review = _until(
            lambda: (
                result
                if (result := state.postgame_review_payload())["available_count"] == 2
                else None
            ),
            timeout_seconds=8,
            label="same generated tasks become visible after end",
        )
        assert not semantic.is_set()
        assert all(not clip["can_save"] for clip in review["clips"])
        after = {
            job.job_id: state.media_job_store.snapshot(job.job_id).artifact_id
            for job in state.media_job_store.list_session_media_jobs(session)
        }
        assert all(after[key] == value for key, value in before.items())
        assert len(after) == 2
        for clip in review["clips"]:
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                connection.request("GET", clip["media_url"])
                response = connection.getresponse()
                assert response.status == 200 and response.read()
            finally:
                connection.close()
    finally:
        release.set()
        resume.set()
        server.shutdown()
        server.server_close()
        serving.join(3)


def test_uncommitted_postroll_cannot_admit_live_media_job(tmp_path, monkeypatch):
    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path, monkeypatch, total_frames=900, pause_after=200, paused=paused, resume=resume
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    state = server.field_state
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(7)
        session = state.session_id
        first = _until(
            lambda: shots[0] if (shots := state.record_store.list_shots(session)) else None,
            timeout_seconds=2,
            label="original shot start from committed pixels",
        )
        evidence = state.record_store.get_recording_evidence(session, first.shot_id)
        assert evidence["media"] is None
        assert state.projection_source.runtime.recording_writer_health.committed_frames == 180
        assert state.media_job_store.list_session_media_jobs(session) == ()
        assert state.postgame_review_payload()["clips"] == []
    finally:
        release.set()
        resume.set()
        server.shutdown()
        server.server_close()
        serving.join(3)
