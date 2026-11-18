"""Recording acceptance starts with raw pixels, never injected product events."""

import http.client
import json
import threading
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab import mainline1_field_runtime as field_runtime
from billiards_ai.lab import mainline1_product_realtime_runtime as product
from billiards_ai.lab.field_assistance.server import build_live_field_assistance_server
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory
from billiards_ai.lab.mainline1_scenario_capture import (
    ScenarioCaptureConfig,
    ScenarioVideoCaptureBackend,
)
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    LocalStorageProbe,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.shared.calibration.homography import calibrate
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _CAPTURE_CONFIG,
    _SHOT_CONFIG_PATH,
    _EmptyClassifier,
    _EmptyDetector,
    _setup_players,
    _until,
)


def _two_shots(path, total_frames=450):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 120))
    assert writer.isOpened()
    try:
        for frame_id in range(total_frames):
            x = 30 + 2 * min(30, max(0, frame_id - 90))
            x -= 2 * min(30, max(0, frame_id - 180))
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (x - 10, 50), (x + 10, 70), (180, 180, 180), -1)
            writer.write(image)
    finally:
        writer.release()


def _recording_server(
    tmp_path,
    monkeypatch,
    *,
    total_frames=450,
    pause_after=None,
    paused=None,
    resume=None,
    source_epoch_id="recording-epoch",
    fresh_session_on_start=True,
    deferred_runtime=False,
    managed_startup=None,
):
    class FunctionalStorageProbe(LocalStorageProbe):
        # Functional lifecycle tests assume admitted storage; real writes/fsync
        # remain real, while hardware throughput has separate admission tests.
        def benchmark_write_mbps(self, root):
            return 1024.0

    monkeypatch.setattr(
        field_runtime,
        "ContinuousSessionRecorder",
        partial(ContinuousSessionRecorder, storage_probe=FunctionalStorageProbe()),
    )
    video = tmp_path / "two-shots.avi"
    _two_shots(video, total_frames=total_frames)
    release = threading.Event()
    semantic_failure = threading.Event()

    def broken_refiner(*args, **kwargs):
        semantic_failure.set()
        raise RuntimeError("R1 semantic failure control")

    monkeypatch.setattr(product.RealtimeShotRefiner, "refine", broken_refiner)
    policy = {"session_policy": "recording-v1"}
    for name in ("HeavyAnchorWorker", "AnchorReplayCoordinator"):
        monkeypatch.setattr(product, name, broken_refiner)

    class GatedPixels(ScenarioVideoCaptureBackend):
        def frames(self, stop):
            for frame in super().frames(stop):
                yield frame
                if frame.capture_frame_id == pause_after:
                    held_at_ns = self._clock_ns()
                    paused.set()
                    while not stop.is_set() and not resume.wait(0.02):
                        pass
                    # This fixture pauses virtual camera time, not just delivery.
                    # Resume at native cadence instead of bursting all held time
                    # into the production FIFO after browser/media inspection.
                    prior_clock = self._clock_ns
                    held_ns = prior_clock() - held_at_ns
                    self._clock_ns = lambda clock=prior_clock, held=held_ns: clock() - held

    factory = build_live_runtime_factory(
        capture_factory=lambda epoch: GatedPixels(
            ScenarioCaptureConfig(video, _CAPTURE_CONFIG),
            source_epoch_id=epoch,
        ),
        capture_config=_CAPTURE_CONFIG,
        calibration=calibrate(
            "recording-test",
            [(0, 119), (159, 119), (159, 0), (0, 0)],
            table_length_mm=2540,
            table_width_mm=1270,
        ),
        detector=_EmptyDetector(),
        identity_classifier=_EmptyClassifier(),
        shot_config=load_mainline1_shot_local_config(_SHOT_CONFIG_PATH),
        capture_setup_id="recording-file",
        calibration_artifact_id="recording-cal",
        config_path=_SHOT_CONFIG_PATH,
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
        capture_starter=lambda _: release.wait(3),
        **policy,
    )
    server = build_live_field_assistance_server(
        artifact_path=tmp_path / "artifacts",
        state_db_path=tmp_path / "field.sqlite3",
        host="127.0.0.1",
        port=0,
        control_token="recording-control",
        source_epoch_factory=lambda: source_epoch_id,
        calibration_id="recording-cal",
        runtime_factory=None if deferred_runtime else factory,
        runtime_preparer=(lambda: ("recording-cal", factory, None)) if deferred_runtime else None,
        recording_recovery_preparer=(lambda: factory) if deferred_runtime else None,
        managed_startup=managed_startup,
        fresh_session_on_start=fresh_session_on_start,
        **policy,
    )
    return server, release, semantic_failure


def test_functional_camera_fixture_is_independent_of_host_benchmark_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(LocalStorageProbe, "benchmark_write_mbps", lambda *_: 0.0)
    server, release, _ = _recording_server(tmp_path, monkeypatch, total_frames=120)
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        state.wait_until_capture_ready(timeout_seconds=3)
        assert state.projection_source.runtime.current_capture_frame is not None
    finally:
        release.set()
        server.server_close()


@pytest.mark.parametrize("delay_stage", ["none", "publication", "metadata"])
def test_recording_keeps_two_real_pixel_shots_when_semantics_unavailable(
    tmp_path, monkeypatch, delay_stage
):
    server, release, semantic_failure = _recording_server(tmp_path, monkeypatch)
    try:
        state = server.field_state
        _setup_players(state)
        delayed = False
        record_media = state.record_store.record_recording_media

        def delayed_media_commit(session_id, media):
            nonlocal delayed
            if delay_stage == "metadata" and media.boundary.shot_id == "shot-0002" and not delayed:
                delayed = True
                assert Path(media.reference_path).is_file()
                time.sleep(1.2)
            return record_media(session_id, media)

        monkeypatch.setattr(state.record_store, "record_recording_media", delayed_media_commit)
        if delay_stage == "publication":
            from billiards_ai.lab import mainline1_shot_jobs as shot_jobs

            rename = shot_jobs.os.rename
            publications = 0

            def delayed_publication(source, destination, *args, **kwargs):
                nonlocal publications, delayed
                if Path(source).name.startswith(".pending-recording-"):
                    publications += 1
                    if publications == 2:
                        delayed = True
                        assert (Path(source) / "interval.json").is_file()
                        time.sleep(1.2)
                return rename(source, destination, *args, **kwargs)

            monkeypatch.setattr(shot_jobs.os, "rename", delayed_publication)

        release.set()
        state.wait_until_capture_ready(timeout_seconds=3)

        def committed_shots():
            if semantic_failure.is_set():
                return ()
            rows = state.record_store.list_shots(state.session_id)
            if len(rows) == 2 and all(
                state.record_store.get_recording_evidence(state.session_id, row.shot_id)[
                    "media_status"
                ]
                == "RECORDED"
                for row in rows
            ):
                return rows
            return None

        # A recursive file count can see .pending directories during atomic
        # publication. Wait for committed authority within the same 7+1 budget.
        shots = _until(
            committed_shots,
            timeout_seconds=8,
            label="two committed field-shot recording metadata rows",
        )
        assert len(shots) == 2, "semantic failure/nearby next shot lost real recorded media"
        media = list((tmp_path / "jobs" / "media").rglob("interval.json"))
        assert len(media) == 2
        assert delayed is (delay_stage != "none")
        metadata = [
            state.record_store.get_recording_evidence(
                state.session_id,
                shot.shot_id,
            )
            for shot in shots
        ]
        assert metadata[0]["boundary"]["end_frame_id"] + 120 > metadata[1]["start"]["frame_id"]
        assert all(row["analysis_status"] == "NOT_REQUESTED" for row in metadata)
        assert all(row["encoding_status"] == "NOT_SCHEDULED" for row in metadata)
        assert all(shot.live_job_id is None and shot.actor_role is None for shot in shots)
        # The shared queue now also stores lightweight media tasks. No semantic
        # analysis work may be scheduled for ordinary recording.
        assert state.media_job_store.list_session_jobs(state.session_id) == ()
        for shot, row in zip(shots, metadata, strict=True):
            ownership = state.record_store.get_shot_ownership(state.session_id, shot.shot_id)
            assert ownership.evidence_status == "unknown"
            assert ownership.actor_role is ownership.actor_user_id is None
            assert state.record_store.get_rule_decision(state.session_id, shot.shot_id) is None
            reference = json.loads(Path(row["media"]["reference_path"]).read_bytes())
            assert reference["storage_kind"] == "session_segment_reference"
            assert reference["source_epoch_id"] == "recording-epoch"
        assert not semantic_failure.is_set()
    finally:
        release.set()
        server.server_close()


def test_pixel_source_eof_preserves_interrupted_shots_and_continuous_raw(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    server, release, semantic_failure = _recording_server(tmp_path, monkeypatch, total_frames=200)
    state = server.field_state
    session = state.session_id
    try:
        _setup_players(state)
        release.set()
        state.wait_until_capture_ready(timeout_seconds=3)
        rows = _until(
            lambda: (
                rows
                if len(rows := state.record_store.list_shots(session)) == 2
                and all(
                    state.record_store.get_recording_evidence(session, row.shot_id)["media_status"]
                    == "FAILED"
                    for row in rows
                )
                else None
            ),
            timeout_seconds=7,
            label="both EOF-affected recording intervals preserved",
        )
        assert not semantic_failure.is_set()
        assert rows[1].shot_id != rows[0].shot_id
        assert (
            state.record_store.get_recording_evidence(session, rows[1].shot_id)["boundary"][
                "complete"
            ]
            is False
        )
    finally:
        release.set()
        server.server_close()
    restarted = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    assert len(restarted.list_shots(session)) == 2
    assert all(
        restarted.get_recording_evidence(session, row.shot_id)["media_status"] == "FAILED"
        for row in restarted.list_shots(session)
    )
    manifests = list((tmp_path / "rings").rglob("session-recording.json"))
    assert len(manifests) == 1
    raw = json.loads(manifests[0].read_bytes())
    assert sum(segment["frame_count"] for segment in raw["segments"]) == 200
    assert list(manifests[0].parent.glob("*.mjpgseg"))


def _end_http(server, session_id, *, token="recording-control"):
    connection = http.client.HTTPConnection(*server.server_address, timeout=15)
    try:
        connection.request(
            "POST",
            "/api/control",
            json.dumps({"command": "end_session", "session_id": session_id}),
            {"Content-Type": "application/json", "X-Control-Token": token},
        )
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _wait_frozen(state, session):
    return _until(
        lambda: state.record_store.get_recording_end_manifest(session),
        timeout_seconds=4,
        label="background processing freezes final recording manifest",
    )


def test_public_end_drains_real_pixels_and_freezes_original_recording(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    processing, drain = threading.Event(), threading.Event()
    paused, resume = threading.Event(), threading.Event()
    original_boundary = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed_boundary(runtime, frame, events):
        if frame.capture_frame_id == 210:
            processing.set()
            assert drain.wait(10)
        return original_boundary(runtime, frame, events)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed_boundary)
    server, release, semantic_failure = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    # This test isolates cutoff/tail persistence before media scheduling. The
    # separate public-review test exercises the real independent encoder.
    monkeypatch.setattr(server.field_state, "_maintain_recording_media", lambda: None)
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    state = server.field_state
    session_id = state.session_id
    try:
        _setup_players(state)
        release.set()
        state.wait_until_capture_ready(timeout_seconds=3)
        assert processing.wait(5) and paused.wait(5)
        read_progress = state.projection_source.runtime.recording_read_progress
        assert read_progress.committed_frame_id - read_progress.last_delivered_frame_id > 100
        replies = []
        request = threading.Thread(target=lambda: replies.append(_end_http(server, session_id)))
        request.start()
        _until(
            lambda: True if state.projection_source.runtime._recording_sealed else None,
            timeout_seconds=1,
            label="public request seals before draining",
        )
        drain.set()
        request.join(10)
        assert not request.is_alive()
        status, body = replies[0]
        assert status == 200, body
        frozen = _wait_frozen(state, session_id)
        assert frozen["contract"] == "recording-end/v1"
        assert frozen["end_reason"] == "USER_ENDED_RECORDING"
        assert frozen["session_id"] == session_id
        assert frozen["source_epoch_id"] == "recording-epoch"
        assert len(frozen["shots"]) == 2
        assert frozen["analysis_status"] == "NOT_REQUESTED"
        assert frozen["encoding_status"] == "NOT_SCHEDULED"
        raw = json.loads(Path(frozen["raw_manifest_path"]).read_bytes())
        assert raw["segments"][-1]["last_frame_id"] == frozen["cutoff"]["frame_id"]
        assert sum(s["frame_count"] for s in raw["segments"]) == frozen["cutoff"]["frame_id"] + 1
        assert not semantic_failure.is_set()
        rows = state.record_store.list_shots(session_id)
        assert [shot.shot_id for shot in rows] == [shot["shot_id"] for shot in frozen["shots"]]
        for shot in rows:
            assert (
                state.record_store.get_recording_evidence(session_id, shot.shot_id)["media_status"]
                == "RECORDED"
            )
            assert state.record_store.get_rule_decision(session_id, shot.shot_id) is None
            assert (
                state.record_store.get_shot_ownership(session_id, shot.shot_id).evidence_status
                == "unknown"
            )
            assert shot.live_job_id is None
        assert _end_http(server, session_id)[0] == 200
        reopened = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
        assert reopened.get_recording_end_manifest(session_id) == frozen
        assert reopened.load_session(session_id).status.value == "ended"
        review = state.postgame_review_payload(kiosk=True)
        assert review["analysis_status"] == "NOT_REQUESTED"
        assert review["encoding_status"] == "NOT_SCHEDULED"
        assert not review["can_complete"]
        manifest = state.record_store.get_postgame_manifest(session_id)
        with pytest.raises(ValueError, match="recording encoding is not ready"):
            state.record_store.complete_postgame_review(
                state.artifact_id,
                session_id,
                manifest.manifest_sha256,
            )
    finally:
        drain.set()
        resume.set()
        release.set()
        server.shutdown()
        http_thread.join(2)
        server.server_close()


@pytest.mark.parametrize("cutoff,active", [(200, True), (250, False), (50, False)])
def test_user_end_keeps_real_incomplete_tail_and_preview(tmp_path, monkeypatch, cutoff, active):
    paused, resume = threading.Event(), threading.Event()
    server, release, semantic = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=cutoff,
        paused=paused,
        resume=resume,
    )
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    state = server.field_state
    session = state.session_id
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(7)
        status, body = _end_http(server, session, token="wrong")
        assert status == 403, body
        assert state.record_store.get_recording_end_manifest(session) is None
        status, body = _end_http(server, session)
        assert status == 200, body
        capture = body["recording_end"]
        assert capture["contract"] == "recording-capture-end/v1"
        assert capture["cutoff"]["frame_id"] == cutoff
        frozen = _wait_frozen(state, session)
        assert frozen["cutoff"] == capture["cutoff"]
        rows = state.record_store.list_shots(session)
        assert len(rows) == (0 if cutoff == 50 else 2)
        if rows:
            tail = state.record_store.get_recording_evidence(session, rows[-1].shot_id)
            assert tail["media_status"] == "FAILED"
            assert tail["media"]["failure_code"] == "INCOMPLETE_RECORDING_WINDOW"
            assert tail["boundary"]["complete"] is not active
            if active:
                assert tail["boundary"]["end_frame_id"] == cutoff
                assert tail["boundary"]["reason"] == "USER_ENDED_RECORDING"
                assert rows[-1].settled_ms is None
            review_tail = next(
                clip
                for clip in state.postgame_review_payload()["clips"]
                if clip["shot_id"] == rows[-1].shot_id
            )
            assert review_tail["encoding_status"] == "FAILED"
            assert review_tail["terminal_failure"] is True
            assert review_tail["error_code"] == "INCOMPLETE_RECORDING_WINDOW"
            assert review_tail["media_url"] is None
        resume.set()
        runtime = state.projection_source.runtime
        assert runtime.capture_hub.wait_for_frame(after_frame_id=cutoff, timeout_seconds=1)
        assert state.record_store.get_recording_end_manifest(session) == frozen
        raw = json.loads(Path(frozen["raw_manifest_path"]).read_bytes())
        assert raw["segments"][-1]["last_frame_id"] == cutoff
        assert not semantic.is_set()
    finally:
        release.set()
        resume.set()
        server.shutdown()
        http_thread.join(2)
        server.server_close()


def test_phone_end_checks_original_authorization_order_and_retries_before_seal(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=200,
        paused=paused,
        resume=resume,
    )
    state = server.field_state
    session = state.session_id
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(5)
        revision = state.record_store.load_session(session).match.revision
        sequence = (
            max(e.origin_sequence or 0 for e in state.record_store.list_command_ledger(session)) + 1
        )
        base = FieldCommandEnvelope(
            "phone-end", session, "alice", "end_session", {}, revision, sequence
        )

        def apply(envelope):
            with state.match_lock:
                state.before_field_command(envelope)
                result = state.record_store.apply_field_command(envelope, state.reduce_live_command)
                state.handle_command_application(envelope, result)
                return result

        denied = [
            (replace(base, command_id="outsider", actor_user_id="outsider"), "COMMAND_FORBIDDEN"),
            (
                replace(
                    base,
                    command_id="stale",
                    expected_revision=revision + 1,
                    server_sequence=sequence + 1,
                ),
                "STALE_REVISION",
            ),
            (
                replace(
                    base,
                    command_id="payload",
                    payload={"force": True},
                    server_sequence=sequence + 2,
                ),
                "COMMAND_INVALID",
            ),
            (replace(base, command_id="order", server_sequence=99), "COMMAND_OUT_OF_ORDER"),
            (replace(base, command_id="outsider"), "COMMAND_IDEMPOTENCY_CONFLICT"),
        ]
        for envelope, expected in denied:
            result = apply(envelope)
            assert result.result_code == expected
            assert not state.projection_source.runtime._recording_sealed
            assert state.record_store.load_session(session).status.value == "open"
        valid = replace(base, server_sequence=sequence + 3)
        result = apply(valid)
        assert result.outcome == "applied", result
        frozen = _wait_frozen(state, session)
        assert frozen["media_status"] == "RETAINED_INCOMPLETE"
        assert apply(valid) == result
        assert state.record_store.get_recording_end_manifest(session) == frozen
        assert state.record_store.list_shots(session)[-1].settled_ms is None
    finally:
        release.set()
        resume.set()
        server.server_close()


def test_delayed_local_end_retry_cannot_stop_rollover_session(tmp_path, monkeypatch):
    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=50,
        paused=paused,
        resume=resume,
    )
    state = server.field_state
    session = state.session_id
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(3)
        result = state.mutate({"command": "end_session", "session_id": session})
        original = state.projection_source
        frozen = _wait_frozen(state, session)
        state.session_ended_at_monotonic -= 100
        resume.set()
        state.payload()
        assert state.session_id != session
        assert state.projection_source is not original
        new_session = state.session_id
        retried = state.mutate({"command": "end_session", "session_id": session})
        assert retried["recording_end"]["cutoff"] == result["recording_end"]["cutoff"]
        assert (
            retried["recording_end"]["accepted_at_utc"]
            == result["recording_end"]["accepted_at_utc"]
        )
        assert state.session_id == new_session
        assert not state.projection_source.runtime._recording_sealed
        assert state.record_store.load_session(new_session).status.value == "open"
        assert state.record_store.active_postgame_review(state.artifact_id).session_id == session
        assert state.record_store.get_recording_end_manifest(session) == frozen
        assert (
            state.projection_source.runtime._recording_store
            is not original.runtime._recording_store
        )
    finally:
        release.set()
        resume.set()
        server.server_close()


def test_genuine_pixel_eof_cannot_be_renamed_manual_end(tmp_path, monkeypatch):
    server, release, _ = _recording_server(tmp_path, monkeypatch, total_frames=60)
    state = server.field_state
    try:
        _setup_players(state)
        release.set()
        assert state.projection_source._runtime_exited.wait(4)
        with pytest.raises(ValueError):
            state.mutate({"command": "end_session", "session_id": state.session_id})
        assert state.record_store.load_session(state.session_id).status.value == "open"
        assert state.record_store.get_recording_end_manifest(state.session_id) is None
    finally:
        release.set()
        server.server_close()


def test_tail_failure_keeps_capture_stop_but_cannot_publish_final_manifest(tmp_path, monkeypatch):
    paused, resume = threading.Event(), threading.Event()
    server, release, _ = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=50,
        paused=paused,
        resume=resume,
    )
    state = server.field_state
    try:
        _setup_players(state)
        release.set()
        assert paused.wait(3)
        recorder = state.projection_source.runtime._recording_store._recorder
        original = recorder._write_segment

        def broken_write(frames):
            raise OSError("injected disk failure at final writer fence")

        monkeypatch.setattr(recorder, "_write_segment", broken_write)
        reply = state.mutate({"command": "end_session", "session_id": state.session_id})
        assert reply["recording_end"]["capture_status"] == "STOPPED"
        assert state.record_store.load_session(state.session_id).status.value == "ended"
        _until(
            lambda: (
                True
                if state.record_store.get_recording_capture_end(state.session_id)["tail_status"]
                == "FAILED"
                else None
            ),
            timeout_seconds=3,
            label="failed tail is separate from accepted capture stop",
        )
        assert (
            state.record_store.get_recording_capture_end(state.session_id)["processing_status"]
            == "FAILED"
        )
        assert state.record_store.get_recording_end_manifest(state.session_id) is None
        assert state.record_store.get_postgame_manifest(state.session_id) is not None
        view = state.postgame_review_payload()
        assert view["processing"]["tail_status"] == "FAILED" and not view["can_complete"]
        monkeypatch.setattr(recorder, "_write_segment", original)
    finally:
        release.set()
        resume.set()
        with pytest.raises(Exception, match="writer failed|cleanup failed"):
            server.server_close()


def test_public_review_lists_and_plays_original_recorded_intervals(tmp_path, monkeypatch):
    """Catches ordinary end hiding real intervals behind semantic publication."""
    paused, resume = threading.Event(), threading.Event()
    server, release, semantic_failure = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    state = server.field_state
    session_id = state.session_id

    def get_review():
        connection = http.client.HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request("GET", "/api/postgame-review")
            response = connection.getresponse()
            assert response.status == 200
            return json.loads(response.read())
        finally:
            connection.close()

    try:
        _setup_players(state)
        release.set()
        state.wait_until_capture_ready(timeout_seconds=3)
        assert paused.wait(7), "synthetic capture never reached the manual-end cutoff"
        from dataclasses import replace

        from billiards_ai.lab.field_assistance.live_source import session_has_live_activity

        # Already-arrived pixel starts remain authoritative even if readiness
        # metadata later becomes incomplete; no product start is injected here.
        assert state.record_store.list_shots(session_id)
        match = state.record_store.load_session(session_id).match
        state.record_store.save_match(
            session_id,
            replace(match, breaker=None, active_role=None, revision=match.revision + 1),
            expected_revision=match.revision,
        )
        assert session_has_live_activity(state.record_store, session_id)
        status, reply = _end_http(server, session_id)
        assert status == 200, reply
        frozen = _wait_frozen(state, session_id)
        original_shots = [shot["shot_id"] for shot in frozen["shots"]]
        assert len(original_shots) == 2
        assert all(
            state.record_store.get_recording_evidence(session_id, shot_id)["media_status"]
            == "RECORDED"
            for shot_id in original_shots
        )
        assert frozen["analysis_status"] == "NOT_REQUESTED"
        assert frozen["encoding_status"] == "NOT_SCHEDULED"
        assert not semantic_failure.is_set()
        review = get_review()
        assert review["review_session_id"] == session_id
        assert review["expected_count"] == 2
        assert [clip["shot_id"] for clip in review["clips"]] == original_shots, (
            "manual end must list every original recorded interval even before encoding; "
            f"received {review}"
        )
        # R3 must wire the real lightweight worker before this becomes GREEN.
        # No clip rows, task success or media bytes are injected by this fixture.
        review = _until(
            lambda: value if (value := get_review())["available_count"] == 2 else None,
            timeout_seconds=10,
            label="independent local publication of both original intervals",
        )
        assert review["analysis_status"] == "NOT_REQUESTED"
        assert review["can_complete"]  # Terminal processing and all real media are ready.
        for clip in review["clips"]:
            assert not clip["can_save"]
            assert clip["media_url"]
            connection = http.client.HTTPConnection(*server.server_address, timeout=10)
            try:
                connection.request("GET", clip["media_url"])
                response = connection.getresponse()
                media = response.read()
                assert response.status == 200
                assert response.getheader("Content-Type") == "video/mp4"
                assert len(media) > 0
                assert b"avc1" in media, "local playback must contain actual H.264 media"
            finally:
                connection.close()
        first = review["clips"][0]
        connection = http.client.HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request("GET", first["media_url"], headers={"Range": "bytes=0-31"})
            response = connection.getresponse()
            assert response.status == 206
            assert response.getheader("Accept-Ranges") == "bytes"
            assert response.getheader("Content-Range").startswith("bytes 0-31/")
            assert len(response.read()) == 32
        finally:
            connection.close()
        stale = first["media_url"].split("?", 1)[0] + "?manifest=" + "f" * 64
        connection = http.client.HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request("GET", stale)
            response = connection.getresponse()
            assert response.status == 404
            response.read()
        finally:
            connection.close()
        assert state.record_store.get_recording_end_manifest(session_id) == frozen
        # Keep the old review durable across empty rollover, then apply actual
        # participant/breaker commands before the next physical shot occurs.
        resume.set()
        with state.match_lock:
            state.session_ended_at_monotonic -= 100

        def rollover():
            state.payload()
            return state.session_id if state.session_id != session_id else None

        _until(
            rollover,
            timeout_seconds=3,
            label="empty recording rollover",
        )
        assert get_review()["review_session_id"] == session_id
        _setup_players(state)
        assert get_review()["review_session_id"] is None
        assert state.record_store.active_postgame_review(state.artifact_id).session_id == session_id
        connection = http.client.HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request("GET", first["media_url"])
            response = connection.getresponse()
            assert response.status == 404
            response.read()
        finally:
            connection.close()
        next_id = state.session_id
        status, reply = _end_http(server, next_id)
        assert status == 200, (
            reply,
            state.projection_source._failure,
            state.projection_source.runtime.failure_reason,
            state.projection_source.runtime.recording_failure_reason,
            state.projection_source._validated_ready,
        )
        assert get_review()["review_session_id"] == session_id
    finally:
        resume.set()
        release.set()
        server.shutdown()
        http_thread.join(2)
        server.server_close()
