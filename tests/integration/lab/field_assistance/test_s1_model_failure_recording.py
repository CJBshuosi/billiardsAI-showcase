"""Failed optional preparation must not stop ordinary raw-pixel recording.

The existing rectangle video is a recording fault fixture, not a billiards
demonstration or evidence of successful model inference/recommendations.
"""

import http.client
import json
import threading
import time

import cv2
import pytest

from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotGuidanceFactory
from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory
from tests.integration.lab.field_assistance import test_recording_first_runtime as recording
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_background_end import _end
from tests.integration.lab.field_assistance.test_recording_progressive_review import _review


def fail_numeric_preparation():
    raise RuntimeError("controlled numerical preparation failure")


def wait_numeric_preparation():
    threading.Event().wait()


@pytest.mark.parametrize(
    "failure", ["missing_models", "numeric_preparation", "waiting_numeric_preparation"]
)
def test_optional_preparation_leaves_recording_and_playable_clips_available(
    tmp_path, monkeypatch, failure
):
    original_factory = recording.build_live_runtime_factory

    def with_guidance(**kwargs):
        if failure in {"numeric_preparation", "waiting_numeric_preparation"}:
            guidance = SnapshotGuidanceFactory(
                object,
                resource_preparer=(
                    wait_numeric_preparation
                    if failure == "waiting_numeric_preparation"
                    else fail_numeric_preparation
                ),
            )
        else:
            guidance = SnapshotGuidanceFactory(
                ModelSnapshotObserverFactory(
                    tmp_path / "missing-detector",
                    tmp_path / "missing-identity",
                    tmp_path / "missing-dinov2",
                )
            )
        return original_factory(
            **kwargs,
            snapshot_guidance_factory=guidance,
        )

    monkeypatch.setattr(recording, "build_live_runtime_factory", with_guidance)
    server, release, semantic = recording._recording_server(tmp_path, monkeypatch, total_frames=900)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()

        def unavailable():
            value = state.projection_source.current().frame.payload.get("snapshot_guidance")
            expected = "preparing" if failure == "waiting_numeric_preparation" else "failed"
            return value if value and value["model_state"] == expected else None

        failed = _until(unavailable, timeout_seconds=9, label="optional preparation state")
        if failure == "waiting_numeric_preparation":
            assert failed["error"] is None
            assert failed["worker_pid"] is not None
        elif failure == "numeric_preparation":
            assert "controlled numerical preparation failure" in failed["error"]
        else:
            assert "missing-detector" in failed["error"] or "missing" in failed["error"]
        assert failed["recommendation"] is None
        assert failed["model_loads"] == 1
        assert failed["tasks_started"] == 0, "preparation must not submit a snapshot"
        assert failed["tasks_completed"] == 0
        runtime = state.projection_source.runtime
        before = runtime.current_capture_frame.capture_frame_id
        time.sleep(0.2)
        assert runtime.current_capture_frame.capture_frame_id > before
        assert runtime.recording_writer_health.reason is None

        def recorded_shots():
            shots = state.record_store.list_shots(state.session_id)
            if len(shots) == 2 and all(
                state.record_store.get_recording_evidence(state.session_id, shot.shot_id)[
                    "media_status"
                ]
                == "RECORDED"
                for shot in shots
            ):
                return shots
            return None

        _until(
            recorded_shots,
            timeout_seconds=7,
            label="both original intervals committed during optional preparation state",
        )
        status, reply = _end(server, state.session_id)
        assert status == 200, reply
        guidance = runtime.snapshot_guidance
        assert guidance._stop.is_set(), "capture cutoff must revoke old-scene guidance"
        view = _until(
            lambda: (
                value
                if (value := _review(server))["available_count"] == 2
                and value["phase"] == "RECORDING_ENDED"
                else None
            ),
            timeout_seconds=9,
            label="ordinary clips complete independently of optional preparation",
        )
        assert not semantic.is_set()
        assert not guidance._thread.is_alive(), "finished recording cannot retain a guidance reader"
        assert guidance.active_pid is None
        assert guidance.projection()["worker_pid"] is None
        assert guidance.cleanup_complete
        assert guidance.projection()["phase"] == "ended"
        for index, clip in enumerate(view["clips"]):
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                connection.request("GET", clip["media_url"])
                response = connection.getresponse()
                assert response.status == 200
                media = response.read()
            finally:
                connection.close()
            path = tmp_path / f"verified-original-{index}.mp4"
            path.write_bytes(media)
            video = cv2.VideoCapture(str(path))
            decoded = 0
            try:
                while True:
                    ok, image = video.read()
                    if not ok:
                        break
                    assert image.shape == (120, 160, 3)
                    decoded += 1
            finally:
                video.release()
            assert decoded > 30
    finally:
        release.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


@pytest.mark.parametrize("fault", ["cleanup", "projection", "overlay"])
def test_optional_projection_failure_preserves_real_reader_second_shot_and_playback(
    tmp_path, monkeypatch, fault
):
    from types import SimpleNamespace

    from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotGuidance

    paused, resume = threading.Event(), threading.Event()
    server, release, semantic = recording._recording_server(
        tmp_path, monkeypatch, total_frames=900, pause_after=170, paused=paused, resume=resume
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    alive = True
    guidance = None
    cleanup_calls = []
    owner = SimpleNamespace(
        pid=17,
        is_alive=lambda: alive,
        terminate=lambda: cleanup_calls.append("terminate"),
        kill=lambda: cleanup_calls.append("kill"),
        join=lambda timeout=None: None,
        close=lambda: None,
    )
    connection = SimpleNamespace(close=lambda: None)
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert paused.wait(5)
        _until(
            lambda: len(state.record_store.list_shots(state.session_id)) == 1,
            timeout_seconds=2,
            label="first real pixel shot registered before projection fault",
        )
        source = state.projection_source
        runtime = source.runtime
        reader = runtime._recording_source
        assert not reader._cancelled.is_set()
        # Only the optional child/IPC fault is controlled. Capture, durable
        # reader, shot handling, HTTP, encoding and all media remain real.
        guidance = SnapshotGuidance(
            hub=None,
            session_id=state.session_id,
            source_epoch_id=source.source_epoch_id,
            calibration=runtime.calibration,
            observer_factory=None,
            journal_path=tmp_path / "fault-guidance.json",
        )
        guidance._process, guidance._connection = owner, connection
        guidance._model_state = "ready"
        guidance._busy = fault == "cleanup"
        guidance._ticket = "owned-fault-ticket" if fault == "cleanup" else None
        if fault != "cleanup":
            guidance._context = "open"
            guidance._context_key = (state.match.revision, state.match.active_role)
        if fault == "cleanup":

            def broken_cleanup_projection():
                guidance._worker_fault("controlled hard worker fault")

            monkeypatch.setattr(guidance, "projection", broken_cleanup_projection)
        elif fault == "projection":

            def broken_projection():
                raise RuntimeError("controlled optional projection fault")

            monkeypatch.setattr(guidance, "projection", broken_projection)
        elif fault == "overlay":
            guidance._recommendation = {
                "route": {
                    "cue_path": [{"x_mm": None, "y_mm": 100}],
                    "object_path": [{"x_mm": 300, "y_mm": 300}],
                },
                "region": None,
            }
        runtime.snapshot_guidance = guidance
        exception = None
        try:
            payload = state.payload()
        except Exception as exc:
            exception = exc
        assert not reader._cancelled.is_set(), f"optional guidance cancelled reader: {exception}"
        assert not runtime._cancelled.is_set() and not source._stopped.is_set()
        assert source.failure_reason is None
        assert exception is None
        assert payload["frame"]["snapshot_guidance"]["phase"] == "failed"
        assert payload["frame"]["snapshot_guidance"]["recommendation"] is None
        assert payload["frame"]["guidance_visible"] is False
        assert guidance._stop.is_set()
        assert runtime.snapshot_guidance is guidance
        assert guidance._process is owner and guidance._connection is connection
        assert not runtime.snapshot_guidance_cleanup_complete
        previous_cleanup_calls = list(cleanup_calls)
        response = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            response.request("GET", "/api/screen")
            reply = response.getresponse()
            assert reply.status == 200
            assert json.loads(reply.read())["frame"]["guidance_visible"] is False
        finally:
            response.close()
        assert cleanup_calls == previous_cleanup_calls, "failed optional projection must not retry"
        resume.set()

        def both_committed():
            rows = state.record_store.list_shots(state.session_id)
            return (
                rows
                if len(rows) == 2
                and all(
                    state.record_store.get_recording_evidence(state.session_id, row.shot_id)[
                        "media_status"
                    ]
                    == "RECORDED"
                    for row in rows
                )
                else None
            )

        _until(both_committed, timeout_seconds=7, label="second real shot survives guidance fault")
        assert not semantic.is_set() and not reader._cancelled.is_set()
        assert runtime.recording_writer_health.reason is None
        assert not runtime.snapshot_guidance_cleanup_complete
        # Keep the fault unreapable through end/media completion too. The
        # owner must remain reachable without blocking ordinary finalization.
        status, reply = _end(server, state.session_id)
        assert status == 200, reply
        view = _until(
            lambda: (
                v
                if (v := _review(server))["available_count"] == 2
                and v["phase"] == "RECORDING_ENDED"
                else None
            ),
            timeout_seconds=9,
            label="normal end and both playable clips",
        )
        assert not runtime.snapshot_guidance_cleanup_complete
        assert guidance._process is owner
        for index, clip in enumerate(view["clips"]):
            client = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                client.request("GET", clip["media_url"])
                reply = client.getresponse()
                assert reply.status == 200
                video_path = tmp_path / f"projection-fault-original-{index}.mp4"
                video_path.write_bytes(reply.read())
            finally:
                client.close()
            capture = cv2.VideoCapture(str(video_path))
            frames = 0
            try:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    assert image.shape == (120, 160, 3)
                    frames += 1
            finally:
                capture.release()
            assert frames > 30
    finally:
        alive = False
        release.set()
        resume.set()
        server.shutdown()
        serving.join(2)
        server.server_close()

    assert runtime.snapshot_guidance_cleanup_complete, (
        "service shutdown must reap the retained owner"
    )
