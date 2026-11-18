"""Ending capture must not wait for pixel detection or the final shot list."""

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from billiards_ai.lab import mainline1_product_realtime_runtime as product
from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


def _end(server, session, *, token="recording-control"):
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request(
            "POST",
            "/api/control",
            json.dumps({"command": "end_session", "session_id": session}),
            {"Content-Type": "application/json", "X-Control-Token": token},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize("processing_fault", [False, True])
def test_http_end_persists_cutoff_before_detection_and_allows_late_shots(
    tmp_path, monkeypatch, processing_fault
):
    blocked, detect = threading.Event(), threading.Event()
    paused, resume_capture = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed(runtime, frame, events):
        if frame.capture_frame_id == 0:
            blocked.set()
            assert detect.wait(15)
            if processing_fault:
                raise RuntimeError("controlled post-end processing fault")
        original(runtime, frame, events)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed)
    if processing_fault:
        original_finalizer = USBLiveProjectionSource._finish_recording_end

        def delayed_finalizer(source):
            # The real consumer observes the fault before the finalizer wakes.
            assert source._stopped.wait(10)
            original_finalizer(source)

        monkeypatch.setattr(USBLiveProjectionSource, "_finish_recording_end", delayed_finalizer)
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume_capture,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert blocked.wait(3) and paused.wait(7)
        session = state.session_id
        assert state.record_store.list_shots(session) == ()
        assert _end(server, session, token="wrong")[0] == 403
        started = time.monotonic()
        status, reply = _end(server, session)
        assert status == 200, reply
        assert time.monotonic() - started < 1
        assert not detect.is_set()
        receipt = reply["recording_end"]
        assert receipt["contract"] == "recording-capture-end/v1"
        assert receipt["cutoff"]["frame_id"] == 380
        assert receipt["capture_status"] == "STOPPED"
        assert receipt["processing_status"] == "PROCESSING"
        assert state.record_store.load_session(session).status.value == "ended"
        assert state.record_store.get_recording_end_manifest(session) is None
        head = state.record_store.get_postgame_manifest(session)
        assert head is not None and head.expected_jobs == ()
        view = state.postgame_review_payload()
        assert view["phase"] == "RECORDING_PROCESSING"
        assert view["expected_count"] is None and not view["can_complete"]
        status, retry = _end(server, session)
        assert status == 200 and retry["recording_end"]["cutoff"] == receipt["cutoff"]
        resume_capture.set()
        _until(
            lambda: (
                True
                if state.projection_source.runtime.current_capture_frame.capture_frame_id > 380
                else None
            ),
            timeout_seconds=1,
            label="preview survives match capture cutoff",
        )
        detect.set()
        if processing_fault:
            _until(
                lambda: (
                    True
                    if state.record_store.get_recording_capture_end(session)["processing_status"]
                    == "FAILED"
                    else None
                ),
                timeout_seconds=3,
                label="durable failure after accepted capture end",
            )
            assert state.record_store.get_recording_end_manifest(session) is None
            assert list((tmp_path / "rings").rglob("session-recording.json"))
            return
        frozen = _until(
            lambda: state.record_store.get_recording_end_manifest(session),
            timeout_seconds=4,
            label="background detection finalizes original match",
        )
        assert frozen["cutoff"] == receipt["cutoff"]
        assert len(frozen["shots"]) == 2
        assert all(
            state.record_store.get_recording_evidence(session, row["shot_id"])["media_status"]
            == "RECORDED"
            for row in frozen["shots"]
        )
        raw = json.loads(Path(frozen["raw_manifest_path"]).read_bytes())
        assert raw["segments"][-1]["last_frame_id"] == 380
        assert not semantic.is_set()
    finally:
        detect.set()
        release.set()
        resume_capture.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_end_during_detector_construction_keeps_original_manual_cutoff_reason(
    tmp_path, monkeypatch
):
    built, construct = threading.Event(), threading.Event()
    paused, resume = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime.__init__

    def slow_constructor(runtime, *args, **kwargs):
        original(runtime, *args, **kwargs)
        built.set()
        assert construct.wait(12)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "__init__", slow_constructor)
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=200,
        paused=paused,
        resume=resume,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert built.wait(3) and paused.wait(5)
        session = state.session_id
        assert state.projection_source.runtime._runtime is None
        status, receipt = _end(server, session)
        assert status == 200, receipt
        assert receipt["recording_end"]["cutoff"]["frame_id"] == 200
        construct.set()
        _until(
            lambda: state.record_store.get_recording_end_manifest(session),
            timeout_seconds=4,
            label="constructor-delayed detector reaches accepted cutoff",
        )
        rows = state.record_store.list_shots(session)
        assert len(rows) == 2
        last = state.record_store.get_recording_evidence(session, rows[-1].shot_id)
        assert last["boundary"]["reason"] == "USER_ENDED_RECORDING"
        assert rows[-1].settled_ms is None
    finally:
        construct.set()
        release.set()
        resume.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_next_scene_does_not_cancel_or_relabel_previous_disk_processing(tmp_path, monkeypatch):
    blocked, detect = threading.Event(), threading.Event()
    paused, resume = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed(runtime, frame, events):
        if frame.capture_frame_id == 0:
            blocked.set()
            assert detect.wait(15)
        original(runtime, frame, events)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed)
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert blocked.wait(3) and paused.wait(7)
        old_session, old_source = state.session_id, state.projection_source
        state.mutate({"command": "end_session", "session_id": old_session})
        _until(
            lambda: (
                True
                if state.record_store.get_recording_capture_end(old_session)["tail_status"]
                == "SAVED"
                else None
            ),
            timeout_seconds=2,
            label="old capture tail durable before next live writer",
        )
        state.session_ended_at_monotonic -= 100
        resume.set()
        state.payload()
        assert state.session_id != old_session
        new_session = state.session_id
        assert state.projection_source is not old_source
        assert old_source._runner.is_alive() and not old_source._stopped.is_set()
        assert not detect.is_set()
        assert not state.projection_source.runtime._recording_sealed
        detect.set()
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(old_session),
            timeout_seconds=4,
            label="old background work completes under original namespace",
        )
        assert final["cutoff"]["frame_id"] == 380
        assert len(final["shots"]) == 2
        assert state.session_id == new_session
        assert state.record_store.list_shots(new_session) == ()
        assert all(
            state.record_store.get_recording_evidence(old_session, row["shot_id"])["media_status"]
            == "RECORDED"
            for row in final["shots"]
        )
    finally:
        detect.set()
        release.set()
        resume.set()
        server.server_close()


@pytest.mark.parametrize("receipt_failure", [False, True])
def test_http_end_does_not_wait_for_tail_writer_admission_lock(
    tmp_path, monkeypatch, receipt_failure
):
    paused, resume = threading.Event(), threading.Event()
    entered, finish_tail = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=50,
        paused=paused,
        resume=resume,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    state = server.field_state
    store = state.projection_source.runtime._recording_store
    original_close = store.close_recording_input
    original_start = threading.Thread.start
    original_persist = state.record_store.record_recording_capture_end
    attempts = 0

    def persist(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if receipt_failure and attempts == 1:
            raise ValueError("controlled capture receipt persistence retry")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(state.record_store, "record_recording_capture_end", persist)

    def delayed_close(*, timeout_seconds):
        with store._recorder._append_lock:
            entered.set()
            assert finish_tail.wait(5)
        return original_close(timeout_seconds=timeout_seconds)

    def start(thread):
        original_start(thread)
        if thread.name == "recording-tail-save":
            assert entered.wait(1)

    monkeypatch.setattr(store, "close_recording_input", delayed_close)
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(3)
        if receipt_failure:
            assert _end(server, state.session_id)[0] == 400
            assert state.projection_source.runtime._recording_sealed
            assert state.record_store.load_session(state.session_id).status.value == "open"
        started = time.monotonic()
        status, reply = _end(server, state.session_id)
        assert status == 200, reply
        assert time.monotonic() - started < 1
        assert entered.is_set() and not finish_tail.is_set()
        assert reply["recording_end"]["tail_status"] == "PENDING"
        finish_tail.set()
        _until(
            lambda: state.record_store.get_recording_end_manifest(state.session_id),
            timeout_seconds=3,
            label="delayed tail eventually finishes outside end request",
        )
    finally:
        finish_tail.set()
        release.set()
        resume.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


@pytest.mark.parametrize("worker_name", ["recording-tail-save", "recording-end-finalizer"])
def test_end_retry_starts_original_background_worker_after_thread_start_failure(
    tmp_path, monkeypatch, worker_name
):
    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    original = threading.Thread.start
    injected = []

    def fail_once(thread):
        if thread.name == worker_name and not injected:
            injected.append(thread)
            raise RuntimeError("controlled background thread start failure")
        return original(thread)

    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert paused.wait(7)
        session = state.session_id
        monkeypatch.setattr(threading.Thread, "start", fail_once)
        status, _ = _end(server, session)
        assert status == 400
        assert injected
        status, reply = _end(server, session)
        assert status == 200, reply
        assert reply["recording_end"]["cutoff"]["frame_id"] == 380
        final = _until(
            lambda: state.record_store.get_recording_end_manifest(session),
            timeout_seconds=4,
            label="same end retry starts missing original worker",
        )
        assert len(final["shots"]) == 2
    finally:
        monkeypatch.setattr(threading.Thread, "start", original)
        release.set()
        resume.set()
        server.shutdown()
        serving.join(2)
        server.server_close()


def test_capture_binds_processing_origin_before_detector_construction(tmp_path, monkeypatch):
    import sqlite3

    constructing, finish = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime.__init__

    def held(runtime, *args, **kwargs):
        constructing.set()
        assert finish.wait(10)
        original(runtime, *args, **kwargs)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "__init__", held)
    server, release, _ = _recording_server(tmp_path, monkeypatch)
    try:
        release.set()
        assert constructing.wait(3)
        boundary = next((tmp_path / "rings").rglob("shot-boundaries.sqlite3"))
        with sqlite3.connect(boundary) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
            assert "recording_processing" in tables, "capture has no durable processing context"
            state = json.loads(
                db.execute("SELECT payload_json FROM recording_processing").fetchone()[0]
            )
        assert state["initial_shot_number"] == 0
        assert state["input_prefix"] is state["checkpoint"] is None
        assert len(state["config_sha256"]) == 64
    finally:
        finish.set()
        release.set()
        server.server_close()
