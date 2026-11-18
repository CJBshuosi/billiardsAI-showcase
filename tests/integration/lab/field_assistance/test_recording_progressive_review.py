"""An ended scene exposes playable committed shots while detection still trails."""

import http.client
import json
import threading

import pytest

from billiards_ai.lab import mainline1_product_realtime_runtime as product
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_background_end import _end
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


def _review(server):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request("GET", "/api/postgame-review")
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


def test_original_first_clip_plays_before_late_detection_and_completion_stays_blocked(
    tmp_path, monkeypatch
):
    blocked, detect = threading.Event(), threading.Event()
    paused, capture = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed(runtime, frame, events):
        if frame.capture_frame_id == 300:
            blocked.set()
            assert detect.wait(20)
        original(runtime, frame, events)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed)
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=1200,
        pause_after=500,
        paused=paused,
        resume=capture,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert blocked.wait(7) and paused.wait(9)
        session = state.session_id
        status, reply = _end(server, session)
        assert status == 200, reply
        view = _review(server)
        assert view["phase"] == "RECORDING_PROCESSING", view
        assert view["review_session_id"] == session
        assert view["expected_count"] is None
        assert view["playback"]["expected_count"] is None
        assert view["playback"]["discovered_count"] == 2
        assert view["discovered_count"] == 2
        assert not view["can_complete"]
        assert view["processing"]["processed_frame_id"] == 299
        assert view["processing"]["cutoff_frame_id"] == 500
        assert view["processing"]["remaining_frames"] == 201
        assert state.record_store.get_recording_end_manifest(session) is None
        view = _until(
            lambda: value if (value := _review(server))["available_count"] >= 1 else None,
            timeout_seconds=8,
            label="first original clip while later detection is blocked",
        )
        first = view["clips"][0]
        assert first["shot_id"] == "shot-0001" and not first["can_save"]
        assert not detect.is_set()
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            connection.request("GET", first["media_url"], headers={"Range": "bytes=0-127"})
            response = connection.getresponse()
            assert response.status == 206
            assert response.getheader("Content-Type") == "video/mp4"
            assert b"avc1" in response.read()
        finally:
            connection.close()
        with pytest.raises(ValueError, match="processing|ready|complete"):
            state.record_store.complete_postgame_review(
                state.artifact_id,
                session,
                view["postgame_manifest_sha256"],
            )
        assert tuple((tmp_path / "rings").rglob("*.mjpgseg"))
        first_job = state.media_job_store.list_session_media_jobs(session)[0]
        detect.set()
        final = _until(
            lambda: (
                value
                if (
                    (value := _review(server))["phase"] == "RECORDING_ENDED"
                    and value["available_count"] == 2
                )
                else None
            ),
            timeout_seconds=8,
            label="original growing inventory reaches its final count",
        )
        assert final["expected_count"] == 2
        assert final["processing"]["remaining_frames"] == 0
        assert final["postgame_manifest_sha256"] == view["postgame_manifest_sha256"]
        jobs = state.media_job_store.list_session_media_jobs(session)
        assert len(jobs) == 2 and first_job in jobs
        assert not semantic.is_set()
    finally:
        detect.set()
        capture.set()
        release.set()
        server.shutdown()
        serving.join(2)
        server.server_close()
