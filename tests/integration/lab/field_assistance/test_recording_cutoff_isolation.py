"""Later preview EOF must not alter an accepted original capture cutoff."""

import threading


def test_original_accepted_end_survives_later_preview_eof(tmp_path, monkeypatch):
    from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
        _setup_players,
        _until,
    )
    from tests.integration.lab.field_assistance.test_recording_background_end import _end
    from tests.integration.lab.field_assistance.test_recording_first_runtime import (
        _recording_server,
    )

    paused, resume, tail_release, tail_started = (threading.Event() for _ in range(4))
    server, release, _ = _recording_server(
        tmp_path, monkeypatch, total_frames=368, pause_after=358, paused=paused, resume=resume
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert paused.wait(8)
        session = state.session_id
        raw = state.projection_source.runtime._recording_store._recorder
        original = raw._write_manifest

        def slow_tail(manifest, **kwargs):
            if manifest.segments[-1].last_frame_id == 358:
                tail_started.set()
                assert tail_release.wait(4)
            return original(manifest, **kwargs)

        monkeypatch.setattr(raw, "_write_manifest", slow_tail)
        status, body = _end(server, session)
        assert status == 200, body
        assert tail_started.wait(1)
        resume.set()
        _until(
            lambda: state.projection_source.runtime.capture_hub.failure_reason,
            timeout_seconds=2,
            label="later preview EOF",
        )
        state.projection_source._runtime_exited.wait(1)
        tail_release.set()
        value = _until(
            lambda: (
                c
                if (c := state.record_store.get_recording_capture_end(session))["processing_status"]
                != "PROCESSING"
                else None
            ),
            timeout_seconds=3,
            label="original end result",
        )
        assert value["processing_status"] == "COMPLETE", value
        _assert_original_two_shots(state, session)
    finally:
        tail_release.set()
        resume.set()
        release.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_delayed_detector_constructor_ignores_post_end_preview_eof(tmp_path, monkeypatch):
    from billiards_ai.lab import mainline1_product_realtime_runtime as product
    from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
        _setup_players,
        _until,
    )
    from tests.integration.lab.field_assistance.test_recording_background_end import _end
    from tests.integration.lab.field_assistance.test_recording_first_runtime import (
        _recording_server,
    )

    built, construct, paused, resume = (threading.Event() for _ in range(4))
    original = product.ProductRealtimeRuntime.__init__

    def delayed(runtime, *args, **kwargs):
        original(runtime, *args, **kwargs)
        built.set()
        assert construct.wait(12)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "__init__", delayed)
    server, release, _ = _recording_server(
        tmp_path, monkeypatch, total_frames=368, pause_after=358, paused=paused, resume=resume
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert built.wait(3) and paused.wait(7)
        session = state.session_id
        status, body = _end(server, session)
        assert status == 200, body
        assert state.projection_source.runtime.wait_for_recording_tail(timeout_seconds=2)
        resume.set()
        _until(
            lambda: state.projection_source.runtime.capture_hub.failure_reason,
            timeout_seconds=2,
            label="post-end preview EOF",
        )
        construct.set()
        value = _until(
            lambda: (
                c
                if (c := state.record_store.get_recording_capture_end(session))["processing_status"]
                != "PROCESSING"
                else None
            ),
            timeout_seconds=3,
            label="delayed detector result",
        )
        assert value["processing_status"] == "COMPLETE", value
        _assert_original_two_shots(state, session)
    finally:
        construct.set()
        resume.set()
        release.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_capture_stop_marks_disk_input_before_post_fence_preview_eof(tmp_path, monkeypatch):
    from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
        _setup_players,
        _until,
    )
    from tests.integration.lab.field_assistance.test_recording_background_end import _end
    from tests.integration.lab.field_assistance.test_recording_first_runtime import (
        _recording_server,
    )

    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=360,
        pause_after=358,
        paused=paused,
        resume=resume,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    state = server.field_state
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(8)
        source = state.projection_source
        hub = source.runtime.capture_hub
        original = hub.seal_recording_input

        def eof_after_original_fence(recorder, *, timeout_seconds, on_sealed):
            cutoff = original(recorder, timeout_seconds=timeout_seconds, on_sealed=on_sealed)
            resume.set()
            _until(lambda: hub.failure_reason, timeout_seconds=1, label="preview EOF after fence")
            source._runtime_exited.wait(0.25)
            return cutoff

        monkeypatch.setattr(hub, "seal_recording_input", eof_after_original_fence)
        session = state.session_id
        status, reply = _end(server, session)
        assert status == 200, reply
        result = _until(
            lambda: (
                value
                if (value := state.record_store.get_recording_capture_end(session))[
                    "processing_status"
                ]
                != "PROCESSING"
                else None
            ),
            timeout_seconds=3,
            label="original background result after later preview EOF",
        )
        assert result["processing_status"] == "COMPLETE", result
        _assert_original_two_shots(state, session)
        assert all(
            state.record_store.get_recording_evidence(session, shot.shot_id)["media_status"]
            == "RECORDED"
            for shot in state.record_store.list_shots(session)
        )
    finally:
        resume.set()
        release.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def _assert_original_two_shots(state, session):
    frozen = state.record_store.get_recording_end_manifest(session)
    assert frozen["last_frame_id"] == 358
    shots = state.record_store.list_shots(session)
    assert [row.shot_id for row in shots] == ["shot-0001", "shot-0002"]
    media = [state.record_store.get_recording_evidence(session, row.shot_id) for row in shots]
    assert all(row["media_status"] == "RECORDED" for row in media)
    assert media[1]["media"]["end_frame_id"] == 354
    assert all(
        state.record_store.get_shot_ownership(session, row.shot_id).evidence_status == "unknown"
        for row in shots
    )
