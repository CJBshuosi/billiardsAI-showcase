import hashlib
import http.client
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.mainline1_native_capture import NativeCaptureProtocolError
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    LiveSettledShot,
    ProductRealtimeEvent,
)
from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.shared.calibration.homography import calibrate


def test_waiting_profiled_match_does_not_admit_shots_until_setup(live_server):
    from dataclasses import replace

    from billiards_ai.edge.route_assistance.control import MatchFormat, MatchMode

    server, captures, _ = live_server
    state = server.field_state
    store = state.record_store
    store.save_match(
        state.session_id,
        MatchControlSnapshot(rule_profile_id=PROFILE_ID, revision=2),
        expected_revision=1,
    )
    channel = captures[0].channel
    assert channel.can_start_shots() is False
    match = store.load_session(state.session_id).match
    store.save_match(
        state.session_id,
        replace(
            match,
            red_user_id="alice",
            blue_user_id="bob",
            breaker=WristbandRole.RED,
            active_role=WristbandRole.RED,
            revision=3,
        ),
        expected_revision=2,
    )
    assert channel.can_start_shots() is False  # legacy role-only command is not a setup
    match = store.load_session(state.session_id).match
    store.save_match(
        state.session_id,
        replace(
            match,
            mode=MatchMode.ASSISTANCE,
            match_format=MatchFormat.SINGLE,
            settings_revision=1,
            revision=4,
        ),
        expected_revision=3,
    )
    assert channel.can_start_shots() is True


def test_profiled_live_service_rejects_legacy_breaker_without_match_mode(live_server):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, _captures, _ = live_server
    match = MatchControlSnapshot(
        red_user_id="alice",
        blue_user_id="bob",
        rule_profile_id=PROFILE_ID,
        revision=2,
    )
    legacy = FieldCommandEnvelope(
        command_id="legacy-breaker",
        session_id=server.field_state.session_id,
        actor_user_id="alice",
        command_type="set_breaker",
        payload={"role": "red"},
        expected_revision=2,
        server_sequence=1,
    )

    with pytest.raises(ValueError, match="MATCH_SETTINGS_REQUIRED"):
        server.field_state.reduce_live_command(
            match,
            legacy,
            has_unsettled_shot=False,
            has_started=False,
        )


def test_recording_failure_overrides_running_health():
    from billiards_ai.lab.field_assistance.live_source import _component_health

    health = _component_health(
        SimpleNamespace(state="running", reason=None),
        fallback_status="degraded",
        fallback_reason="epoch changed",
    )
    assert health == {"status": "degraded", "reason": "epoch changed"}


def test_route_health_exposes_anchor_worker_and_replay_failure_details():
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource

    source = object.__new__(USBLiveProjectionSource)
    source.lock = threading.RLock()
    source._failure = None
    source._restart_reason = None
    source.coordinator = SimpleNamespace(
        snapshot=SimpleNamespace(route_abstained_reason=None),
    )
    source.runtime = SimpleNamespace(
        video_health=SimpleNamespace(state="running", reason=None),
        recording_health=SimpleNamespace(state="running", reason=None),
        recording_failure_reason=None,
        segment_audit_health=None,
        current_observation=None,
        tracker_health=None,
        anchor_health=SimpleNamespace(
            scheduled_frames=12,
            accepted_frames=4,
            skipped_scheduled_frames=8,
            completed_anchors=4,
            failed_anchors=0,
            identity_classified_anchors=1,
            p50_inference_ms=5100.0,
            p95_inference_ms=5300.0,
            maximum_inference_ms=5400.0,
            high_speed_coverage=False,
            last_error=None,
        ),
        anchor_replay_health=SimpleNamespace(
            accepted_anchors=0,
            unreplayable_anchors=4,
            skipped_anchors=0,
        ),
        diagnostics=None,
    )

    health = source.subsystem_health()

    assert health["analysis_health"]["anchor_worker"]["completed_anchors"] == 4
    assert health["analysis_health"]["anchor_worker"]["p95_inference_ms"] == 5300.0
    assert health["analysis_health"]["anchor_replay"] == {
        "status": "unavailable",
        "reason": "no detector anchor accepted",
        "accepted_anchors": 0,
        "unreplayable_anchors": 4,
        "skipped_anchors": 0,
    }
    assert health["route_health"]["reason"] == "no detector anchor accepted"


class Capture:
    def __init__(self, session_id, channel, source_epoch_id):
        self.session_id = session_id
        self.channel = channel
        self.source_epoch_id = source_epoch_id
        self.cancelled = threading.Event()
        self.running = threading.Event()
        self.validated = threading.Event()
        self.current_capture_frame = None
        self.calibration = None

    def run(self):
        self.running.set()
        self.validated.set()
        self.cancelled.wait(10)

    def close(self):
        self.cancelled.set()

    def wait_until_ready(self, *, timeout_seconds):
        return self.validated.wait(timeout_seconds) and not self.cancelled.is_set()


@pytest.fixture
def live_server(tmp_path):
    captures = []
    epoch_number = 0

    def source_epoch_factory():
        nonlocal epoch_number
        epoch_number += 1
        return f"epoch-{epoch_number}"

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        capture = Capture(session_id, channel, source_epoch_id)
        capture.initial_shot_number = initial_shot_number
        captures.append(capture)
        return capture

    assert hasattr(service, "build_live_field_assistance_server")
    qr = tmp_path / "qr.png"
    qr.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "artifacts",
        state_db_path=tmp_path / "field.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        join_qr_asset_path=qr,
    )
    state = server.field_state
    state.record_store.save_match(
        state.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            active_role=WristbandRole.RED,
            breaker=WristbandRole.RED,
            red_group=BallGroupAssignment.SOLID,
            blue_group=BallGroupAssignment.STRIPE,
            revision=1,
        ),
        expected_revision=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, captures, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert all(c.cancelled.is_set() for c in captures)


def request(server, path="/api/screen", payload=None, token="secret"):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request(
        "GET" if payload is None else "POST",
        path,
        body=None if payload is None else json.dumps(payload),
        headers={"X-Control-Token": token},
    )
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, json.loads(body) if "json" in response.getheader(
        "Content-Type", ""
    ) else body


def test_scenario_capture_diagnostics_reach_live_http_screen(tmp_path, monkeypatch):
    """Real paced pixels and live HTTP; inference is stubbed, no events are injected."""
    from functools import partial
    from uuid import uuid4

    import cv2
    import numpy as np

    from billiards_ai.lab import mainline1_field_runtime as assembly
    from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
    from billiards_ai.lab.mainline1_scenario_capture import (
        ScenarioCaptureConfig,
        ScenarioVideoCaptureBackend,
    )
    from scripts.algorithm.serve_mainline1_blackbox_field import _capture_diagnostics

    video = tmp_path / "scenario.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (64, 48))
    assert writer.isOpened()
    for _ in range(600):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    config = RealtimeCaptureConfig(width=64, height=48, fps=60)
    epoch = uuid4().hex
    sources, consumed = [], threading.Event()

    def capture_factory(source_epoch):
        source = ScenarioVideoCaptureBackend(
            ScenarioCaptureConfig(video, config), source_epoch_id=source_epoch
        )
        sources.append(source)
        return source

    class Runtime:
        def __init__(self, **kwargs):
            self.source = kwargs["source"]

        def run(self):
            for frame in self.source.frames(threading.Event()):
                assert frame.source_epoch_id == epoch
                consumed.set()

    monkeypatch.setattr(assembly, "ProductRealtimeRuntime", Runtime)
    config_path = tmp_path / "config"
    config_path.write_text("product-config")
    factory = assembly.build_live_runtime_factory(
        capture_factory=capture_factory,
        capture_config=config,
        calibration=None,
        detector=None,
        identity_classifier=None,
        shot_config=None,
        capture_setup_id="setup",
        calibration_artifact_id="calibration",
        config_path=config_path,
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
        capture_diagnostics_factory=partial(_capture_diagnostics, capture_config=config),
    )
    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "artifacts",
        state_db_path=tmp_path / "field.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: epoch,
        calibration_id="calibration",
        runtime_factory=factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.field_state.wait_until_capture_ready(timeout_seconds=3) == epoch
        assert consumed.wait(1)
        status, payload = request(server)
        assert status == 200
        assert payload["frame"]["source_kind"] == "usb_live"
        assert payload["frame"]["source_epoch_id"] == epoch
        assert payload["capture_diagnostics"] == {
            "source_kind": "blackbox_video",
            "runtime_environment": "development",
            "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "source_epoch_id": epoch,
            "width": 64,
            "height": 48,
            "fps": 60,
        }
        assert str(tmp_path) not in json.dumps(payload["capture_diagnostics"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
    assert len(sources) == 1


def test_usb_default_screen_omits_capture_diagnostics(live_server):
    server, _, _ = live_server
    status, payload = request(server)
    assert status == 200
    assert "capture_diagnostics" not in payload


def review_request(server, path="/api/postgame-review", payload=None, nonce="", headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request(
        "GET" if payload is None else "POST",
        path,
        body=None if payload is None else json.dumps(payload),
        headers={"Content-Type": "application/json", "X-Kiosk-Nonce": nonce, **(headers or {})},
    )
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, json.loads(body) if "json" in response.getheader(
        "Content-Type", ""
    ) else body


def ended_review(server, capture):
    from billiards_ai.edge.route_assistance.records import LiveShotMediaReference

    settle(server, capture)
    state = server.field_state
    state.record_store.attach_live_shot_media(
        LiveShotMediaReference(
            state.session_id,
            "shot-1",
            capture.source_epoch_id,
            1,
            "job-1",
            str(state.artifact_path / "synthetic-usb"),
            "d" * 64,
        )
    )
    assert request(server, "/api/control", {"command": "end_session"})[0] == 200
    status, review = review_request(server)
    assert status == 200
    return review


def seed_review_media(server, review):
    import hashlib

    state = server.field_state
    clip = state.record_store.list_postgame_clips(review["review_session_id"])[0]
    path = state.artifact_path / "postgame" / clip.session_id / "clip.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0123456789abcdef")
    state.record_store.record_postgame_clip_available(
        session_id=clip.session_id,
        shot_id=clip.shot_id,
        postgame_manifest_sha256=clip.postgame_manifest_sha256,
        job_id=clip.job_id,
        artifact_id="c" * 64,
        local_path=path,
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        byte_size=path.stat().st_size,
        content_type="video/mp4",
    )
    return path


def test_postgame_review_save_rollover_and_complete_are_independent(live_server):
    server, captures, _ = live_server
    status, empty = review_request(server)
    # The fixture has a live session; an empty review is not a completed match.
    assert status == 200 and empty["phase"] == "LIVE_MATCH"
    assert empty["review_session_id"] is None
    assert empty["can_complete"] is False
    assert "kiosk_nonce" not in empty and empty["clips"] == []
    review = ended_review(server, captures[0])
    assert review["phase"] == "POSTGAME_PROCESSING"
    assert review["clips"][0]["save_status"] == "NOT_SAVED"
    assert review["clips"][0]["media_url"] is None
    payload = dict(
        request_id="save-1",
        session_id=review["review_session_id"],
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    for nonce in ("", "wrong"):
        assert review_request(server, "/api/postgame-review/save", payload, nonce)[0] == 403
    status, receipt = review_request(
        server, "/api/postgame-review/save", payload, review["kiosk_nonce"]
    )
    assert status == 200 and receipt["status"] == "SAVE_PENDING"
    seed_review_media(server, review)
    store = server.field_state.record_store
    store.record_postgame_finalization(payload["session_id"], '{"clips":[],"report":{}}')
    _, available = review_request(server)
    assert available["phase"] == "POSTGAME_REVIEW"
    assert available["clips"][0]["save_status"] == "SAVED"
    assert "local_path" not in json.dumps(available)
    assert "artifact_id" not in json.dumps(available)
    assert (
        review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])[1]
        == receipt
    )
    assert len(store.list_saved_media_outbox()) == 1
    for change in (
        {"session_id": "other"},
        {"shot_id": "other"},
        {"postgame_manifest_sha256": "0" * 64},
    ):
        assert (
            review_request(
                server, "/api/postgame-review/save", {**payload, **change}, review["kiosk_nonce"]
            )[0]
            == 409
        )
    server.field_state.session_ended_at_monotonic -= 4
    _, current = request(server)
    assert current["session_id"] != payload["session_id"]
    assert review_request(server)[1] == available
    store.end_session(current["session_id"])
    complete = {key: payload[key] for key in ("session_id", "postgame_manifest_sha256")}
    assert (
        review_request(server, "/api/postgame-review/complete", complete, review["kiosk_nonce"])[0]
        == 200
    )
    _, next_review = review_request(server)
    assert next_review["review_session_id"] == current["session_id"]
    assert next_review["kiosk_nonce"] != review["kiosk_nonce"]
    assert (
        review_request(
            server, "/api/postgame-review/complete", complete, next_review["kiosk_nonce"]
        )[0]
        == 409
    )
    assert review_request(server)[1] == next_review
    assert (
        review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])[0]
        == 403
    )
    assert review_request(server, available["clips"][0]["media_url"])[0] == 404


def test_postgame_media_is_bounded_and_rejects_tampering(live_server):
    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    path = seed_review_media(server, review)
    _, review = review_request(server)
    url = review["clips"][0]["media_url"]
    assert review_request(server, url) == (200, b"0123456789abcdef")
    for requested, expected in (
        ("bytes=2-5", b"2345"),
        ("bytes=12-", b"cdef"),
        ("bytes=-3", b"def"),
    ):
        assert review_request(server, url, headers={"Range": requested}) == (206, expected)
    for requested in ("bytes=0-1,3-4", "bytes=-0", "bytes=100-", "bad"):
        assert review_request(server, url, headers={"Range": requested})[0] == 416
    assert review_request(server, url.replace("manifest=", "manifest=x"))[0] == 404
    path.write_bytes(b"xxxxxxxxxxxxxxxx")
    assert review_request(server, url)[0] == 404
    path.write_bytes(b"short")
    assert review_request(server, url)[0] == 404
    path.unlink()
    assert review_request(server, url)[0] == 404
    target = path.parent / "target.mp4"
    target.write_bytes(b"0123456789abcdef")
    path.symlink_to(target)
    assert review_request(server, url)[0] == 404


def test_postgame_review_failure_and_outbox_retry_projection(live_server):
    from datetime import UTC, datetime, timedelta

    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    state = server.field_state
    manifest = state.record_store.get_postgame_manifest(review["review_session_id"])
    # Deterministic read clock; no publisher or worker is invoked by polling.
    state.postgame_now_utc = lambda: manifest.deadline_at_utc + timedelta(seconds=1)
    assert review_request(server)[1]["phase"] == "FAILED_RETRYING"
    seed_review_media(server, review)
    payload = dict(
        request_id="retry-save",
        session_id=review["review_session_id"],
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    assert (
        review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])[0]
        == 200
    )
    item = state.record_store.claim_due_saved_media(datetime.now(UTC))
    state.record_store.retry_saved_media(item.clip_id, item.claim_token, error_code="offline")
    assert review_request(server)[1]["clips"][0]["save_status"] == "SAVE_FAILED_RETRYING"
    reclaimed = state.record_store.claim_due_saved_media(datetime.now(UTC))
    state.record_store.block_saved_media(
        reclaimed.clip_id,
        reclaimed.claim_token,
        error_code="SYNC_IDENTITY_CONFLICT",
    )
    assert review_request(server)[1]["clips"][0]["save_status"] == "SAVE_BLOCKED"
    duplicate = {**payload, "request_id": "retry-save-new-request"}
    assert (
        review_request(
            server,
            "/api/postgame-review/save",
            duplicate,
            review["kiosk_nonce"],
        )[0]
        == 409
    )
    with sqlite3.connect(state.record_store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0]
        assert count == 1


@pytest.mark.parametrize("request_id", ("contains space", "slash/not-allowed", "雪", "a" * 129))
def test_postgame_save_rejects_non_cloud_safe_request_id_at_http_boundary(live_server, request_id):
    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    payload = dict(
        request_id=request_id,
        session_id=review["review_session_id"],
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )

    assert (
        review_request(
            server,
            "/api/postgame-review/save",
            payload,
            review["kiosk_nonce"],
        )[0]
        == 400
    )
    with sqlite3.connect(server.field_state.record_store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0]
        assert count == 0


@pytest.mark.parametrize("fresh", [False, True])
def test_postgame_review_restart_rotates_nonce_and_recovers_saved_receipt(live_server, fresh):
    server, captures, tmp_path = live_server
    review = ended_review(server, captures[0])
    seed_review_media(server, review)
    state = server.field_state
    payload = dict(
        request_id="restart-save",
        session_id=review["review_session_id"],
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    _, receipt = review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])
    state.record_store.record_postgame_finalization(payload["session_id"], "{}")
    state.session_ended_at_monotonic -= 4
    _, current = request(server)
    server.shutdown()
    server.server_close()
    restarted = service.build_live_field_assistance_server(
        artifact_path=state.artifact_path,
        state_db_path=tmp_path / "field.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        calibration_id="cal-1",
        source_epoch_factory=lambda: "restarted-epoch",
        fresh_session_on_start=fresh,
        runtime_factory=lambda session_id, channel, initial_shot_number, source_epoch_id: Capture(
            session_id, channel, source_epoch_id
        ),
    )
    try:
        recovered = restarted.field_state.postgame_review_payload(kiosk=True)
        assert (restarted.field_state.session_id != current["session_id"]) is fresh
        assert recovered["review_session_id"] == payload["session_id"]
        assert recovered["phase"] == "POSTGAME_REVIEW"
        assert recovered["kiosk_nonce"] != review["kiosk_nonce"]
        with pytest.raises(PermissionError):
            restarted.field_state.mutate_postgame_review(
                payload, review["kiosk_nonce"], complete=False
            )
        assert (
            restarted.field_state.mutate_postgame_review(
                payload, recovered["kiosk_nonce"], complete=False
            )
            == receipt
        )
    finally:
        restarted.server_close()


def test_remote_review_reads_omit_nonce_and_spoofed_headers_cannot_write(live_server):
    from io import BytesIO

    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    handler = object.__new__(service._FieldAssistanceHandler)
    handler.server = server
    handler.client_address = ("192.168.10.2", 1234)
    handler.path = "/api/postgame-review"
    handler.headers = {
        "Host": "localhost",
        "Origin": "http://localhost",
        "X-Forwarded-For": "127.0.0.1",
        "Forwarded": "for=127.0.0.1",
        "X-Control-Token": "secret",
        "X-Kiosk-Nonce": review["kiosk_nonce"],
    }
    responses = []
    handler._send_json = lambda status, payload: responses.append((status, payload))
    handler.do_GET()
    assert responses[-1] == (200, {k: v for k, v in review.items() if k != "kiosk_nonce"})
    handler.path = "/api/postgame-review/save"
    handler.rfile = BytesIO(b"{}")
    handler.do_POST()
    assert responses[-1][0] == 403
    assert (
        server.field_state.record_store.list_postgame_clips(review["review_session_id"])[
            0
        ].save_status
        == "NOT_SAVED"
    )


@pytest.mark.parametrize(
    "body,extra",
    [
        (b"{", {}),
        (b"[]", {}),
        (b"{}", {}),
        (b"{}", {"Content-Length": "-1"}),
        (b"{}", {"Content-Length": "32769"}),
        (b"{}", {"Content-Length": "invalid"}),
        (b"{}", {"Content-Type": "text/plain"}),
        (b"{}", {"Transfer-Encoding": "chunked"}),
    ],
)
def test_postgame_body_parser_rejects_invalid_requests(live_server, body, extra):
    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request(
        "POST",
        "/api/postgame-review/save",
        body=body,
        headers={
            "Content-Type": "application/json",
            "X-Kiosk-Nonce": review["kiosk_nonce"],
            **extra,
        },
    )
    response = conn.getresponse()
    assert response.status in (400, 413)
    response.read()
    conn.close()
    assert (
        server.field_state.record_store.list_postgame_clips(review["review_session_id"])[
            0
        ].save_status
        == "NOT_SAVED"
    )


def test_postgame_duplicate_json_identity_is_rejected(live_server):
    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    payload = dict(
        request_id="duplicate",
        session_id=review["review_session_id"],
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    body = '{"session_id":"forged",' + json.dumps(payload)[1:]
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request(
        "POST",
        "/api/postgame-review/save",
        body=body,
        headers={"Content-Type": "application/json", "X-Kiosk-Nonce": review["kiosk_nonce"]},
    )
    response = conn.getresponse()
    assert response.status == 400
    response.read()
    conn.close()


def test_concurrent_postgame_save_has_one_receipt_and_preserves_match(live_server):
    from concurrent.futures import ThreadPoolExecutor

    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    seed_review_media(server, review)
    store = server.field_state.record_store
    session = store.load_session(review["review_session_id"])
    shots = store.list_shots(session.session_id)
    payload = dict(
        request_id="concurrent",
        session_id=session.session_id,
        shot_id=review["clips"][0]["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: review_request(
                    server, "/api/postgame-review/save", payload, review["kiosk_nonce"]
                ),
                range(2),
            )
        )
    assert results[0] == results[1]
    assert results[0][0] == 200
    assert len(store.list_saved_media_outbox()) == 1
    assert store.load_session(session.session_id) == session
    assert store.list_shots(session.session_id) == shots


def test_zero_shot_finalized_review_is_available(live_server):
    server, _, _ = live_server
    state = server.field_state
    state.record_store.end_session(state.session_id)
    state.record_store.record_postgame_finalization(state.session_id, "{}")
    status, review = review_request(server)
    assert status == 200 and review["phase"] == "POSTGAME_REVIEW"
    assert review["expected_count"] == review["available_count"] == 0
    assert review["clips"] == []


def until(server, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, body = request(server)
        assert status == 200
        if predicate(body):
            return body
        time.sleep(0.01)
    pytest.fail(f"live projection did not arrive: {body}")


def settle(server, capture):
    capture.channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
    )
    until(server, lambda p: p["frame"]["phase"] == "moving")
    frame = ObservationFrame(
        20,
        200,
        (
            CalibratedBallObservation("cue", 500, 500, 0.95, BallGroup.CUE, 0.95),
            CalibratedBallObservation("solid", 1500, 650, 0.95, BallGroup.SOLID, 0.95),
            CalibratedBallObservation("stripe", 1900, 900, 0.95, BallGroup.STRIPE, 0.95),
        ),
    )
    shot = LiveSettledShot(
        "shot-1", 10, 20, 200_000_000, (frame,), True, None, {}, "epoch-1", "job-1"
    )
    capture.channel.publish(
        ProductRealtimeEvent("SHOT_SETTLED", shot_id="shot-1", frame_id=20, settled=shot)
    )
    return until(server, lambda p: p.get("confirmation_ref") is not None)


def test_live_start_settle_confirmation_and_new_start_clear_guidance(live_server):
    server, captures, root = live_server
    pending = settle(server, captures[0])
    assert captures[0].running.is_set() and not captures[0].cancelled.is_set()
    assert pending["rule_status"] == "RULES_PENDING"
    assert pending["guidance"] is None
    command = {
        "command": "continue_turn",
        "command_id": "cmd-1",
        "confirmation_ref": pending["confirmation_ref"],
    }
    assert request(server, "/api/control", command, token="bad")[0] == 403
    captures[0].current_capture_frame = type(
        "Frame",
        (),
        {
            "capture_frame_id": 41,
            "width": 1100,
            "height": 600,
            "mjpeg": b"live-camera-jpeg",
        },
    )()
    captures[0].calibration = calibrate(
        "camera",
        [(50, 550), (1050, 550), (1050, 50), (50, 50)],
        table_length_mm=2540,
        table_width_mm=1270,
    )
    status, confirmed = request(server, "/api/control", command)
    assert status == 200
    assert confirmed["rule_status"] == "READY" and confirmed["guidance"] is not None
    screen = request(server)[1]
    assert screen["frame"]["media_frame_id"] == 41
    assert screen["frame"]["route_overlay"]["routes"]
    status, body = request(server, "/media/frame/41.jpg")
    assert status == 200
    assert body == b"live-camera-jpeg"
    page = request(server, "/tablet")[1].decode()
    assert 'data-command="continue_turn"' in page and "confirmation_ref" in page
    assert request(server, "/api/tablet")[1] == confirmed
    captures[0].channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-2", frame_id=30, started_ms=321)
    )
    moving = until(server, lambda p: p["frame"]["phase"] == "moving")
    assert moving["display"]["last_action"] == "已检测第2杆击球开始"
    assert moving["guidance"] is None and not moving["frame"]["guidance_visible"]
    assert request(server, "/api/control", command)[0] == 400
    assert (
        server.field_state.record_store.list_shots(server.field_state.session_id)[-1].started_ms
        == 321
    )


def test_live_screen_reports_settled_shot_count_without_waiting_for_rules(live_server):
    server, captures, _ = live_server

    pending = settle(server, captures[0])

    assert pending["frame"]["shot_sequence"] == 1
    assert pending["display"]["last_action"] == "第1杆已停止，证据已保留"


def test_live_screen_streams_camera_frames_over_one_multipart_response(live_server):
    server, captures, _ = live_server
    captures[0].current_capture_frame = type(
        "Frame",
        (),
        {
            "capture_frame_id": 41,
            "width": 1100,
            "height": 600,
            "mjpeg": b"live-camera-jpeg",
        },
    )()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request("GET", "/media/live.mjpg?epoch=epoch-1")
    response = conn.getresponse()

    assert response.status == 200
    assert response.getheader("Content-Type") == (
        "multipart/x-mixed-replace; boundary=billiards-frame"
    )
    chunk = response.read1(4096)
    conn.close()
    assert b"Content-Type: image/jpeg" in chunk
    assert b"live-camera-jpeg" in chunk


def test_analysis_failure_does_not_remove_a_healthy_live_camera_frame(live_server):
    """Catches downstream projection health being promoted to a black screen."""
    server, captures, _ = live_server
    captures[0].current_capture_frame = type(
        "Frame",
        (),
        {
            "capture_frame_id": 52,
            "width": 1100,
            "height": 600,
            "mjpeg": b"camera-survives-analysis-failure",
        },
    )()
    captures[0].channel.fail("analysis worker failed")
    degraded = until(server, lambda payload: payload["frame"]["phase"] == "degraded")

    assert degraded["frame"]["media_frame_id"] == 52
    assert degraded["frame"]["degraded_reason"]
    assert degraded["video_health"]["status"] == "ready"
    assert degraded["analysis_health"]["status"] == "degraded"
    assert degraded["recording_health"]["status"] == "unavailable"
    assert degraded["route_health"]["status"] == "degraded"
    assert "sync_health" in degraded
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    conn.request("GET", "/media/live.mjpg?epoch=epoch-1")
    response = conn.getresponse()
    chunk = response.read1(4096)
    conn.close()

    assert response.status == 200
    assert b"camera-survives-analysis-failure" in chunk


def test_enqueued_media_uses_job_session_and_start_revision(live_server):
    server, captures, root = live_server
    pending = settle(server, captures[0])
    request(
        server,
        "/api/control",
        {
            "command": "switch_turn",
            "command_id": "cmd-1",
            "confirmation_ref": pending["confirmation_ref"],
        },
    )
    job = ShotAnalysisJob(
        "job-1",
        pending["session_id"],
        "shot-1",
        "epoch-1",
        1,
        25,
        root / "media",
        "a" * 64,
        "setup",
        "cal-1",
        "b" * 64,
        "model",
    )
    captures[0].channel.publish(
        ProductRealtimeEvent("SHOT_ENQUEUED", shot_id="shot-1", frame_id=20, job=job)
    )
    store = server.field_state.record_store
    until(server, lambda _: store.get_live_shot_media(pending["session_id"], "shot-1") is not None)
    media = store.get_live_shot_media(pending["session_id"], "shot-1")
    assert media.match_revision == 1 and media.job_id == "job-1"
    assert media.media_path == str(root / "media")
    assert request(server, "/api/table-code.png")[1].startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "failure", ["channel", "capture", "missing_start_time", "missing_frame_id"]
)
def test_terminal_failure_clears_authority_and_rejects_confirmation(live_server, failure):
    server, captures, _ = live_server
    pending = settle(server, captures[0])
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "continue_turn",
                "command_id": "confirmed",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[1]["guidance"]
        is not None
    )
    channel = captures[0].channel
    if failure == "channel":
        channel.fail("capture channel failed")
    elif failure == "capture":
        channel.publish(ProductRealtimeEvent("CAPTURE_DEGRADED", reason="device lost"))
    elif failure == "missing_frame_id":
        channel.publish(ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-2", started_ms=300))
    else:
        channel.publish(ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-2", frame_id=30))
    state = until(server, lambda p: p["display"]["phase"] == "DEGRADED")
    assert state["guidance"] is None and state["confirmation_ref"] is None
    if failure == "missing_frame_id":
        assert [
            shot.shot_id for shot in server.field_state.record_store.list_shots(state["session_id"])
        ] == ["shot-1"]
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "continue_turn",
                "command_id": "cmd-1",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[0]
        == 400
    )


@pytest.mark.parametrize("event_kind", ["SHOT_INCOMPLETE", "CAPTURE_DEGRADED"])
def test_native_disconnect_abstains_active_shot_and_reopens_fresh_epoch(tmp_path, event_kind):
    captures = []
    epoch_number = 0

    def source_epoch_factory():
        nonlocal epoch_number
        epoch_number += 1
        return f"epoch-{epoch_number}"

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        capture = Capture(session_id, channel, source_epoch_id)
        capture.initial_shot_number = initial_shot_number
        captures.append(capture)
        return capture

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "recoverable-artifacts",
        state_db_path=tmp_path / "recoverable.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    state = server.field_state
    state.record_store.save_match(
        state.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            active_role=WristbandRole.RED,
            breaker=WristbandRole.RED,
            red_group=BallGroupAssignment.SOLID,
            blue_group=BallGroupAssignment.STRIPE,
            revision=1,
            rule_profile_id=PROFILE_ID,
            rack_phase=RackPhase.ASSIGNED,
        ),
        expected_revision=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session_id = state.session_id
        captures[0].channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0001", frame_id=10, started_ms=100)
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if state.record_store.list_shots(session_id):
                break
            time.sleep(0.01)
        else:
            pytest.fail("shot start was not consumed without an HTTP request")
        captures[0].channel.publish(
            ProductRealtimeEvent(
                event_kind,
                shot_id=None if event_kind == "CAPTURE_DEGRADED" else "shot-0001",
                frame_id=12,
                reason="NativeCaptureProtocolError: native capture stream is truncated",
            )
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if len(captures) == 2:
                break
            time.sleep(0.01)
        else:
            pytest.fail("native capture did not recover without an HTTP request")
        recovered = request(server)[1]
        shots = state.record_store.list_shots(session_id)
        receipt = state.record_store.get_rule_decision(session_id, "shot-0001")
        match = state.record_store.load_session(session_id).match

        assert recovered["session_id"] == session_id
        assert recovered["rule_status"] == "RULES_PENDING"
        assert recovered["guidance"] is None
        assert len(shots) == 1 and shots[0].settled_ms is not None
        assert shots[0].media.source_epoch_id == "epoch-1"
        assert receipt is not None and receipt.status == "abstained"
        assert match.revision == 1 and match.active_role is WristbandRole.RED
        assert captures[0].cancelled.is_set()
        assert captures[1].running.is_set()
        assert captures[1].source_epoch_id == "epoch-2"
        assert captures[1].initial_shot_number == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert all(capture.cancelled.is_set() for capture in captures)


def test_native_recovery_is_serialized_with_server_close(tmp_path):
    captures = []
    factory_calls = 0
    replacement_entered = threading.Event()
    release_replacement = threading.Event()

    def source_epoch_factory():
        return f"epoch-{len(captures) + 1}"

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            replacement_entered.set()
            release_replacement.wait(3)
        capture = Capture(session_id, channel, source_epoch_id)
        capture.initial_shot_number = initial_shot_number
        captures.append(capture)
        return capture

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "race-artifacts",
        state_db_path=tmp_path / "race.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    captures[0].channel.publish(
        ProductRealtimeEvent(
            "CAPTURE_DEGRADED",
            frame_id=12,
            reason="NativeCaptureProtocolError: native capture stream is truncated",
        )
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if server.field_state.projection_source.restart_required:
            break
        time.sleep(0.01)
    else:
        pytest.fail("native failure did not request recovery")

    recovering = threading.Thread(target=server.service_actions)
    closing = threading.Thread(target=server.server_close)
    try:
        recovering.start()
        assert replacement_entered.wait(3)
        closing.start()
        time.sleep(0.05)
        assert closing.is_alive()
        release_replacement.set()
        recovering.join(3)
        closing.join(3)
        assert not recovering.is_alive() and not closing.is_alive()
        assert len(captures) == 2
        assert len({capture.source_epoch_id for capture in captures}) == 2
        assert all(capture.cancelled.is_set() for capture in captures)
        server.service_actions()
        server.server_close()
        assert len(captures) == 2
    finally:
        release_replacement.set()
        server.server_close()


def test_native_runtime_start_retries_fresh_epochs_until_camera_returns(tmp_path):
    runtimes = []
    epoch_number = 0
    modes = iter(("stable", "native_error", "native_error", "stable"))

    def source_epoch_factory():
        nonlocal epoch_number
        epoch_number += 1
        return f"epoch-{epoch_number}"

    class Runtime(Capture):
        def __init__(self, session_id, channel, source_epoch_id, mode):
            super().__init__(session_id, channel, source_epoch_id)
            self.mode = mode
            self.created_at = time.monotonic()

        def run(self):
            self.running.set()
            if self.mode == "native_error":
                raise NativeCaptureProtocolError("native capture helper is not ready")
            self.validated.set()
            self.cancelled.wait(10)

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        runtime = Runtime(session_id, channel, source_epoch_id, next(modes))
        runtime.initial_shot_number = initial_shot_number
        runtimes.append(runtime)
        return runtime

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "retry-artifacts",
        state_db_path=tmp_path / "retry.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        assert runtimes[0].running.wait(1)
        runtimes[0].channel.publish(
            ProductRealtimeEvent(
                "CAPTURE_DEGRADED",
                reason="NativeCaptureProtocolError: native capture stream is truncated",
            )
        )
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if len(runtimes) == 4 and runtimes[-1].running.is_set():
                break
            time.sleep(0.01)
        else:
            pytest.fail("native startup retries did not reach a stable fresh epoch")
        assert [runtime.source_epoch_id for runtime in runtimes] == [
            "epoch-1",
            "epoch-2",
            "epoch-3",
            "epoch-4",
        ]
        assert runtimes[2].created_at - runtimes[1].created_at >= 0.15
        assert runtimes[3].created_at - runtimes[2].created_at >= 0.35
        assert all(runtime.cancelled.is_set() for runtime in runtimes[:-1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert all(runtime.cancelled.is_set() for runtime in runtimes)


def test_reconnect_stays_recovering_until_replacement_validation_completes(tmp_path):
    runtimes = []
    epoch_number = 0

    def source_epoch_factory():
        nonlocal epoch_number
        epoch_number += 1
        return f"epoch-{epoch_number}"

    class DelayedRuntime(Capture):
        def run(self):
            self.running.set()
            if len(runtimes) == 1:
                self.validated.set()
            self.cancelled.wait(10)

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        runtime = DelayedRuntime(session_id, channel, source_epoch_id)
        runtime.initial_shot_number = initial_shot_number
        runtimes.append(runtime)
        return runtime

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "delayed-ready-artifacts",
        state_db_path=tmp_path / "delayed-ready.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    assert server.field_state.wait_until_capture_ready(timeout_seconds=1) == "epoch-1"
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        runtimes[0].channel.publish(
            ProductRealtimeEvent(
                "CAPTURE_DEGRADED",
                reason="NativeCaptureProtocolError: native capture stream is truncated",
            )
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if len(runtimes) == 2 and runtimes[1].running.is_set():
                break
            time.sleep(0.01)
        else:
            pytest.fail("replacement runtime did not start")
        recovering = server.field_state.projection_source.current().frame.payload
        assert recovering["rule_status"] == "DEGRADED"
        assert recovering["phase"] == "recovering"
        assert recovering["degraded_reason"] == "capture validation pending"
        time.sleep(1.1)
        assert server.field_state.capture_recovery_attempts == 1
        still_recovering = server.field_state.projection_source.current().frame.payload
        assert still_recovering["rule_status"] == "DEGRADED"

        runtimes[1].validated.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if (
                server.field_state.capture_recovery_attempts == 0
                and server.field_state.projection_source.current().frame.payload["rule_status"]
                != "DEGRADED"
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail("validated replacement did not become authoritative")
        assert server.field_state.capture_recovery_not_before == 0.0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert all(runtime.cancelled.is_set() for runtime in runtimes)


def test_non_native_runtime_start_failure_remains_terminal(tmp_path):
    runtimes = []

    class InvalidRuntime(Capture):
        def run(self):
            self.running.set()
            raise ValueError("calibration package is invalid")

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        runtime = InvalidRuntime(session_id, channel, source_epoch_id)
        runtime.initial_shot_number = initial_shot_number
        runtimes.append(runtime)
        return runtime

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "invalid-artifacts",
        state_db_path=tmp_path / "invalid.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "epoch-1",
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            projection = server.field_state.projection_source.current().frame.payload
            if (
                projection["rule_status"] == "DEGRADED"
                and "ValueError" in projection["degraded_reason"]
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail("non-native startup failure did not revoke authority")
        for _ in range(5):
            server.service_actions()
            time.sleep(0.05)
        assert len(runtimes) == 1
        assert not server.field_state.projection_source.restart_required
        assert "ValueError" in projection["degraded_reason"]
    finally:
        server.server_close()


def test_server_close_cancels_pending_native_start_retries(tmp_path):
    runtimes = []
    epoch_number = 0

    def source_epoch_factory():
        nonlocal epoch_number
        epoch_number += 1
        return f"epoch-{epoch_number}"

    class Runtime(Capture):
        def run(self):
            self.running.set()
            if len(runtimes) > 1:
                raise NativeCaptureProtocolError("native capture helper is not ready")
            self.validated.set()
            self.cancelled.wait(10)

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        runtime = Runtime(session_id, channel, source_epoch_id)
        runtime.initial_shot_number = initial_shot_number
        runtimes.append(runtime)
        return runtime

    server = service.build_live_field_assistance_server(
        artifact_path=tmp_path / "cancel-retry-artifacts",
        state_db_path=tmp_path / "cancel-retry.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=source_epoch_factory,
        calibration_id="cal-1",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    runtimes[0].channel.publish(
        ProductRealtimeEvent(
            "CAPTURE_DEGRADED",
            reason="NativeCaptureProtocolError: native capture stream is truncated",
        )
    )
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server.field_state.projection_source.restart_required:
                break
            time.sleep(0.01)
        else:
            pytest.fail("initial disconnect did not request recovery")
        server.service_actions()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server.field_state.projection_source.restart_required:
                break
            time.sleep(0.01)
        else:
            pytest.fail("native startup failure did not schedule another retry")
        server.server_close()
        runtime_count = len(runtimes)
        time.sleep(0.5)
        for _ in range(5):
            server.service_actions()
        assert len(runtimes) == runtime_count == 2
        assert all(runtime.cancelled.is_set() for runtime in runtimes)
    finally:
        server.server_close()


@pytest.mark.parametrize("stage", ["freeze", "live_settle"])
def test_incomplete_active_shot_is_abstained_and_capture_continues(live_server, stage):
    server, captures, _ = live_server
    state = server.field_state
    match = state.record_store.load_session(state.session_id).match
    state.record_store.save_match(
        state.session_id,
        MatchControlSnapshot(
            red_user_id=match.red_user_id,
            blue_user_id=match.blue_user_id,
            active_role=match.active_role,
            breaker=match.breaker,
            red_group=match.red_group,
            blue_group=match.blue_group,
            revision=2,
            rule_profile_id=PROFILE_ID,
            rack_phase=RackPhase.ASSIGNED,
        ),
        expected_revision=1,
    )
    captures[0].channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
    )
    until(server, lambda payload: payload["frame"]["phase"] == "moving")

    captures[0].channel.publish(
        ProductRealtimeEvent(
            "SHOT_INCOMPLETE",
            shot_id="shot-1",
            frame_id=12,
            reason=json.dumps({"error_type": "OSError", "message": "disk", "stage": stage}),
        )
    )

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and len(captures) < 2:
        time.sleep(0.01)
    assert len(captures) == 2
    continued = request(server)[1]
    shots = server.field_state.record_store.list_shots(continued["session_id"])

    assert continued["analysis_health"]["status"] == "ready"
    assert continued["route_health"]["status"] != "degraded"
    assert shots == ()
    assert captures[0].cancelled.is_set()
    assert (
        server.field_state.record_store.list_admin_audit(continued["session_id"])[-1][
            "command_type"
        ]
        == "discard_incomplete_shot"
    )
    captures[1].channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=20, started_ms=200)
    )
    until(server, lambda payload: payload["frame"]["phase"] == "moving")


def test_quick_next_shot_discards_prior_post_roll_without_restarting_capture(live_server):
    server, captures, _ = live_server
    first_capture = captures[0]
    first_capture.channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-1", frame_id=10, started_ms=100)
    )
    until(server, lambda payload: payload["frame"]["phase"] == "moving")

    first_capture.channel.publish(
        ProductRealtimeEvent(
            "SHOT_INCOMPLETE",
            shot_id="shot-1",
            frame_id=20,
            reason=json.dumps(
                {
                    "error_type": "IncompleteShotEvidence",
                    "message": "next shot started before prior post-roll completed",
                    "stage": "post_roll_superseded",
                }
            ),
        )
    )
    first_capture.channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-2", frame_id=30, started_ms=300)
    )

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        shots = server.field_state.record_store.list_shots(server.field_state.session_id)
        if [(shot.shot_id, shot.settled_ms) for shot in shots] == [("shot-2", None)]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("the next shot did not replace superseded post-roll evidence")
    continued = request(server)[1]
    shots = server.field_state.record_store.list_shots(continued["session_id"])

    assert [(shot.shot_id, shot.settled_ms) for shot in shots] == [("shot-2", None)]
    assert captures == [first_capture]
    assert not first_capture.cancelled.is_set()
    assert not server.field_state.projection_source.restart_required
    assert continued["analysis_health"]["status"] == "ready"
    assert continued["route_health"]["status"] != "degraded"


@pytest.mark.parametrize("stage", ["materialization", "enqueue"])
def test_post_settle_media_failure_does_not_degrade_capture(live_server, stage):
    server, captures, _ = live_server
    pending = settle(server, captures[0])
    captures[0].channel.publish(
        ProductRealtimeEvent(
            "SHOT_INCOMPLETE",
            shot_id="shot-1",
            frame_id=20,
            reason=json.dumps({"error_type": "OSError", "message": "disk", "stage": stage}),
        )
    )

    continued = until(
        server,
        lambda payload: "录像未保存" in payload["display"]["last_action"],
    )

    assert continued["rule_status"] == pending["rule_status"]
    assert continued["analysis_health"]["status"] == "ready"
    assert continued["route_health"]["status"] != "degraded"


def test_unclassified_incomplete_failure_remains_terminal(live_server):
    server, captures, _ = live_server
    captures[0].channel.publish(
        ProductRealtimeEvent(
            "SHOT_INCOMPLETE",
            shot_id="shot-0001",
            frame_id=12,
            reason="source_stopped_before_post_roll",
        )
    )

    degraded = until(server, lambda payload: payload["rule_status"] == "DEGRADED")

    assert degraded["guidance"] is None
    assert len(captures) == 1


def test_end_rollover_rebinds_capture_session_and_rejects_old_refs(live_server):
    server, captures, _ = live_server
    pending = settle(server, captures[0])
    status, ended = request(server, "/api/control", {"command": "end_session"})
    assert status == 200 and ended["session"]["status"] == "ended", ended
    assert captures[0].cancelled.is_set()
    server.field_state.session_ended_at_monotonic -= 4
    _, current = request(server)
    assert current["session_id"] != pending["session_id"]
    assert captures[-1].session_id == current["session_id"]
    assert captures[-1].source_epoch_id == captures[0].source_epoch_id
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "switch_turn",
                "command_id": "old",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[0]
        == 400
    )


def test_cloud_confirmation_uses_exact_reference_and_existing_command_ledger(live_server):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, captures, _ = live_server
    pending = settle(server, captures[0])

    class Gateway:
        completed = []

        def claim_field_command(self):
            return FieldCommandEnvelope(
                "cloud-1",
                pending["session_id"],
                "alice",
                "continue_turn",
                {"confirmation_ref": pending["confirmation_ref"]},
                1,
                1,
            )

        def complete_field_command(self, application):
            self.completed.append(application)

    state = server.field_state
    gateway = Gateway()
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        confirmation_resolver=state.resolve_command_confirmation,
        command_reducer=state.reduce_live_command,
    )
    assert worker.claim_and_apply_once()
    assert gateway.completed[0].outcome == "applied"
    ready = request(server)[1]
    assert ready["rule_status"] == "READY" and ready["guidance"] is not None
    assert worker.claim_and_apply_once()
    assert request(server)[1]["match"]["revision"] == 2
    captures[0].channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0002", frame_id=30, started_ms=300)
    )
    until(server, lambda p: p["frame"]["phase"] == "moving")
    assert worker.claim_and_apply_once()
    assert request(server)[1]["guidance"] is None


def test_process_restart_resumes_persisted_shot_sequence(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
    from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot

    database = tmp_path / "resume.sqlite"
    store = SQLiteFieldRecordStore(database)
    session = store.open_or_resume_session(artifact_id="fixture")
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
    )
    coordinator.shot_started(shot_id="shot-0001", started_ms=100)
    coordinator.settle(
        LiveSettledShot(
            "shot-0001", 10, 20, 200_000_000, (), False, "unknown", {}, "old-epoch", "job-1"
        ),
        facts=(),
    )
    seen = []

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        seen.append(initial_shot_number)
        return Capture(session_id, channel, source_epoch_id)

    source = USBLiveProjectionSource(
        store=SQLiteFieldRecordStore(database),
        session_id=session.session_id,
        source_epoch_id="new-epoch",
        calibration_id="cal",
        runtime_factory=factory,
        match_lock=threading.RLock(),
    )
    assert seen == [1]
    source.start()
    source.channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0002", frame_id=10, started_ms=100)
    )
    deadline = time.monotonic() + 2
    while len(store.list_shots(session.session_id)) != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    source.close()
    shots = store.list_shots(session.session_id)
    assert [(s.shot_id, s.sequence) for s in shots] == [("shot-0001", 1), ("shot-0002", 2)]
    assert shots[-1].media.source_epoch_id == "new-epoch"
    with pytest.raises(ValueError, match="unsettled"):
        USBLiveProjectionSource(
            store=SQLiteFieldRecordStore(database),
            session_id=session.session_id,
            source_epoch_id="third-epoch",
            calibration_id="cal",
            runtime_factory=factory,
            match_lock=threading.RLock(),
        )


def test_process_restart_resumes_after_audited_incomplete_shot_discard(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource

    database = tmp_path / "resume-after-discard.sqlite"
    store = SQLiteFieldRecordStore(database)
    session = store.open_or_resume_session(artifact_id="fixture")
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
    )
    coordinator.shot_started(shot_id="shot-0001", started_ms=100)
    store.discard_latest_unsettled_shot(
        session.session_id,
        "shot-0001",
        reason="incomplete shot discarded after post_roll_superseded",
    )
    coordinator.shot_started(shot_id="shot-0002", started_ms=200)
    coordinator.settle(
        LiveSettledShot(
            "shot-0002", 20, 30, 300_000_000, (), False, "unknown", {}, "old-epoch", "job-2"
        ),
        facts=(),
    )
    seen = []

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        seen.append(initial_shot_number)
        return Capture(session_id, channel, source_epoch_id)

    source = USBLiveProjectionSource(
        store=SQLiteFieldRecordStore(database),
        session_id=session.session_id,
        source_epoch_id="new-epoch",
        calibration_id="cal",
        runtime_factory=factory,
        match_lock=threading.RLock(),
    )

    assert seen == [2]
    source.start()
    source.channel.publish(
        ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0003", frame_id=40, started_ms=400)
    )
    deadline = time.monotonic() + 2
    while len(store.list_shots(session.session_id)) != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    projection = source.current()
    assert projection.frame.payload["shot_sequence"] == 3
    assert projection.frame.payload["capture_action"] == "已检测第3杆击球开始"
    source.close()


def test_process_restart_rejects_unaccounted_shot_id_gap(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource

    database = tmp_path / "resume-with-unaccounted-gap.sqlite"
    store = SQLiteFieldRecordStore(database)
    session = store.open_or_resume_session(artifact_id="fixture")
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
    )
    coordinator.shot_started(shot_id="shot-0002", started_ms=200)
    coordinator.settle(
        LiveSettledShot(
            "shot-0002", 20, 30, 300_000_000, (), False, "unknown", {}, "old-epoch", "job-2"
        ),
        facts=(),
    )

    with pytest.raises(ValueError, match="restart offset"):
        USBLiveProjectionSource(
            store=SQLiteFieldRecordStore(database),
            session_id=session.session_id,
            source_epoch_id="new-epoch",
            calibration_id="cal",
            runtime_factory=lambda *args: None,
            match_lock=threading.RLock(),
        )


def test_live_startup_failure_reports_the_underlying_capture_reason(tmp_path):
    del tmp_path

    class FailedCapture:
        failure_reason = "NativeCaptureProtocolError: camera reopen failed"
        restart_required = False

        def wait_until_ready(self, *, timeout_seconds):
            del timeout_seconds
            return False

    state = object.__new__(service._LiveServerState)
    state.projection_source = FailedCapture()
    state.match_lock = threading.RLock()

    with pytest.raises(RuntimeError, match="camera reopen failed"):
        state.wait_until_capture_ready(timeout_seconds=0.1)


def test_process_restart_rolls_over_unsettled_session_without_discarding_it(tmp_path):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator

    artifact = (tmp_path / "restart-artifacts").resolve()
    artifact.mkdir()
    database = tmp_path / "restart.sqlite"
    artifact_id = "usb-field-" + hashlib.sha256(str(artifact).encode()).hexdigest()
    store = SQLiteFieldRecordStore(database, session_id_factory=lambda: "interrupted-session")
    interrupted = store.open_or_resume_session(
        artifact_id=artifact_id,
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(
        interrupted.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            active_role=WristbandRole.RED,
            breaker=WristbandRole.RED,
            red_group=BallGroupAssignment.SOLID,
            blue_group=BallGroupAssignment.STRIPE,
            revision=1,
            rule_profile_id=PROFILE_ID,
            rack_phase=RackPhase.ASSIGNED,
        ),
        expected_revision=0,
    )
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=interrupted.session_id,
        source_epoch_id="interrupted-epoch",
        calibration_id="cal",
    )
    coordinator.shot_started(shot_id="shot-0001", started_ms=100)
    store.record_cloud_session_open(interrupted.session_id)
    captures = []

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        capture = Capture(session_id, channel, source_epoch_id)
        capture.initial_shot_number = initial_shot_number
        captures.append(capture)
        return capture

    server = service.build_live_field_assistance_server(
        artifact_path=artifact,
        state_db_path=database,
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "replacement-epoch",
        calibration_id="cal",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    try:
        state = server.field_state
        assert state.session_id != interrupted.session_id
        assert store.load_session(interrupted.session_id).status.value == "ended"
        assert store.list_shots(interrupted.session_id)[0].settled_ms is None
        assert interrupted.session_id in store.list_pending_cloud_close_session_ids()
        assert captures[0].session_id == state.session_id
        assert captures[0].initial_shot_number == 0
        assert state.last_applied_action == "上次异常中断的对局已保存，请重新扫码加入新对局"
    finally:
        server.server_close()


def test_cloud_close_retry_survives_rollover_and_permanent_qr(live_server):
    from billiards_ai.lab.field_assistance.cloud_gateway import (
        FieldJoinCoordinator,
        FieldSessionLease,
    )

    server, captures, _ = live_server
    pending = settle(server, captures[0])
    now = [100.0]

    class Gateway:
        attempts = 0
        opened = []

        def open_field_session(self, session_id):
            self.opened.append(session_id)
            return FieldSessionLease(session_id, "table", "open")

        def close_field_session(self, session_id):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("offline")
            return FieldSessionLease(session_id, "table", "ended")

        def refresh_join_requests(self, session_id):
            return ()

    gateway = Gateway()
    state = server.field_state
    state.monotonic = lambda: now[0]
    state.join_coordinator = FieldJoinCoordinator(session_id=state.session_id, gateway=gateway)
    assert request(server)[1]["join"]["status"] == "open"
    qr = request(server, "/api/table-code.png")[1]
    assert request(server, "/api/control", {"command": "end_session"})[0] == 200
    assert pending["session_id"] in state.record_store.list_pending_cloud_close_session_ids()
    now[0] += 4
    new = request(server)[1]
    assert new["session_id"] != pending["session_id"]
    assert gateway.opened == [pending["session_id"], new["session_id"]]
    now[0] += 10
    request(server)
    assert pending["session_id"] not in state.record_store.list_pending_cloud_close_session_ids()
    assert request(server, "/api/table-code.png")[1] == qr


def test_capture_cancel_error_still_joins_runtime_and_serves_degraded(live_server):
    server, captures, _ = live_server
    capture = captures[0]
    original_close = capture.close

    def close():
        original_close()
        raise OSError("capture cancellation receipt failed")

    capture.close = close
    capture.channel.fail("device disconnected")
    assert until(server, lambda p: p["display"]["phase"] == "DEGRADED")["guidance"] is None
    source = server.field_state.projection_source
    # Cancellation errors remain observable and cleanup stays retryable. Both
    # the consumer's original cancellation and this explicit attempt are kept.
    with pytest.raises(BaseExceptionGroup) as caught:
        source.close()
    assert all(isinstance(exc, OSError) for exc in caught.value.exceptions)
    assert all("capture cancellation receipt failed" in str(exc) for exc in caught.value.exceptions)
    assert not source._closed
    assert not source._runner.is_alive() and not source._consumer.is_alive()
    capture.close = original_close
    source.close()
    assert source._closed and not source.ready_to_end


@pytest.mark.parametrize("operation", ["open", "close", "refresh"])
@pytest.mark.parametrize("caller", ["payload", "maintain_background"])
def test_cloud_io_never_holds_live_boundary_lock(live_server, operation, caller):
    from billiards_ai.lab.field_assistance.cloud_gateway import (
        FieldJoinCoordinator,
        FieldSessionLease,
    )

    server, captures, _ = live_server
    pending = settle(server, captures[0])
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "continue_turn",
                "command_id": "ready",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[1]["guidance"]
        is not None
    )
    entered, release = threading.Event(), threading.Event()

    class Gateway:
        def call(self, kind, session_id):
            if operation == kind:
                entered.set()
                assert release.wait(5)
            return FieldSessionLease(session_id, "table", "open" if kind == "open" else "ended")

        def open_field_session(self, session_id):
            return self.call("open", session_id)

        def close_field_session(self, session_id):
            return self.call("close", session_id)

        def refresh_join_requests(self, session_id):
            self.call("refresh", session_id)
            return ()

    state = server.field_state
    state.join_coordinator = FieldJoinCoordinator(session_id=state.session_id, gateway=Gateway())
    if operation == "refresh":
        state.cloud_session_status = "open"
    elif operation == "close":
        old = state.record_store.open_or_resume_session(artifact_id="old-fixture")
        state.record_store.end_session(old.session_id)
        state.record_store.enqueue_cloud_close_intent(old.session_id)
        state.pending_cloud_closes[old.session_id] = 0
    task = threading.Thread(target=getattr(state, caller))
    task.start()
    try:
        assert entered.wait(2)
        captures[0].channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0002", frame_id=30, started_ms=300)
        )
        deadline = time.monotonic() + 1
        while (
            len(state.record_store.list_shots(state.session_id)) < 2 and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert len(state.record_store.list_shots(state.session_id)) == 2
        projection = request(server)[1]
        assert projection["guidance"] is None and projection["frame"]["phase"] == "moving"
        assert not release.is_set()
    finally:
        release.set()
        task.join(3)


@pytest.mark.parametrize("complete", [False, True])
def test_cloud_end_drains_arrived_boundary_before_persisting_end(live_server, complete):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, captures, _ = live_server
    state = server.field_state

    class Gateway:
        result = None

        def claim_field_command(self):
            return FieldCommandEnvelope("end", state.session_id, "alice", "end_session", {}, 1, 1)

        def complete_field_command(self, result):
            self.result = result

    gateway = Gateway()
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    session_id = state.session_id
    with state.match_lock:
        captures[0].channel.publish(
            ProductRealtimeEvent("SHOT_STARTED", shot_id="shot-0001", frame_id=10, started_ms=100)
        )
        if complete:
            shot = LiveSettledShot(
                "shot-0001",
                10,
                20,
                200_000_000,
                (),
                False,
                "not_publishable",
                {},
                "epoch-1",
                "job-1",
            )
            captures[0].channel.publish(
                ProductRealtimeEvent(
                    "SHOT_SETTLED", shot_id=shot.shot_id, frame_id=20, settled=shot
                )
            )
        assert worker.claim_and_apply_once()
    shots = state.record_store.list_shots(session_id)
    assert len(shots) == 1 and shots[0].started_ms == 100
    if complete:
        assert shots[0].settled_ms == 200
        assert gateway.result.outcome == "applied"
        assert state.record_store.load_session(session_id).status.value == "ended"
    else:
        assert gateway.result.outcome == "rejected"
        assert gateway.result.result_code == "SHOT_UNSETTLED"
        assert state.record_store.load_session(session_id).status.value == "open"
        assert request(server)[1]["rule_status"] == "DEGRADED"
    assert captures[0].cancelled.is_set()


def test_non_active_joined_player_can_prepare_and_end_live_session(live_server):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, captures, _ = live_server
    state = server.field_state

    class Gateway:
        result = None

        def claim_field_command(self):
            return FieldCommandEnvelope(
                "end-by-blue", state.session_id, "bob", "end_session", {}, 1, 1
            )

        def complete_field_command(self, result):
            self.result = result

    gateway = Gateway()
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )

    assert worker.claim_and_apply_once()
    assert gateway.result.outcome == "applied"
    assert state.record_store.load_session(state.session_id).status.value == "ended"
    assert captures[0].cancelled.is_set()


def test_persistence_failure_is_terminal_and_clears_previously_ready_route(
    live_server, monkeypatch
):
    import sqlite3

    server, captures, root = live_server
    pending = settle(server, captures[0])
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "continue_turn",
                "command_id": "ready",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[1]["guidance"]
        is not None
    )

    def fail(_media):
        raise sqlite3.OperationalError("disk is full")

    monkeypatch.setattr(server.field_state.record_store, "attach_live_shot_media", fail)
    job = ShotAnalysisJob(
        "job-1",
        pending["session_id"],
        "shot-1",
        "epoch-1",
        1,
        25,
        root / "media",
        "a" * 64,
        "setup",
        "cal-1",
        "b" * 64,
        "model",
    )
    captures[0].channel.publish(ProductRealtimeEvent("SHOT_ENQUEUED", shot_id="shot-1", job=job))
    projection = until(server, lambda p: p["rule_status"] == "DEGRADED")
    assert projection["guidance"] is None and projection["confirmation_ref"] is None
    assert (
        "OperationalError" in projection["degraded_reason"]
        and "disk is full" in projection["degraded_reason"]
    )
    assert captures[0].cancelled.is_set()

    def unavailable(_session_id):
        raise sqlite3.OperationalError("database unavailable")

    source = server.field_state.projection_source
    with monkeypatch.context() as reads:
        reads.setattr(server.field_state.record_store, "load_session", unavailable)
        failed = source.current().frame.payload
        assert failed["rule_status"] == "DEGRADED" and failed["guidance"] is None


@pytest.mark.parametrize("operation", ["open", "refresh"])
@pytest.mark.parametrize("change", ["rollover", "revision"])
def test_late_cloud_response_cannot_write_into_changed_authority(live_server, operation, change):
    from dataclasses import replace

    from billiards_ai.lab.field_assistance.cloud_gateway import (
        FieldJoinCoordinator,
        FieldSessionLease,
        JoinRequest,
    )

    server, _, _ = live_server
    state = server.field_state
    entered, release = threading.Event(), threading.Event()
    old_id = state.session_id

    class Gateway:
        def open_field_session(self, session_id):
            if operation == "open":
                entered.set()
                assert release.wait(5)
            return FieldSessionLease(session_id, "table", "open")

        def refresh_join_requests(self, session_id):
            entered.set()
            assert release.wait(5)
            return (JoinRequest("old-request", "old-name", "pending", None),)

    state.join_coordinator = FieldJoinCoordinator(session_id=old_id, gateway=Gateway())
    if operation == "refresh":
        state.cloud_session_status = "open"
    task = threading.Thread(target=state.maintain_background)
    task.start()
    try:
        assert entered.wait(2)
        with state.match_lock:
            if change == "rollover":
                state.projection_source.end_session()
                ended = state.record_store.end_session(old_id)
                state.session_ended_at_monotonic = state.monotonic() - 4
                _, rolled = state._rollover_ended_session(ended)
                assert rolled
            else:
                session = state.record_store.load_session(old_id)
                state.record_store.save_match(
                    old_id, replace(session.match, revision=2), expected_revision=1
                )
        release.set()
        task.join(3)
        assert not task.is_alive()
        assert state.join_coordinator.pending == ()
        assert state.cloud_session_status == (
            "open" if operation == "refresh" and change == "revision" else "unavailable"
        )
        if change == "rollover" and operation == "open":
            assert old_id in state.record_store.list_pending_cloud_close_session_ids()
    finally:
        release.set()
        task.join(3)


@pytest.mark.parametrize("rejection", ["order", "id_conflict", "duplicate_rejected"])
def test_rejected_cloud_end_does_not_stop_open_capture(live_server, rejection):
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, captures, _ = live_server
    state = server.field_state
    pending = settle(server, captures[0])
    ready = request(
        server,
        "/api/control",
        {
            "command": "continue_turn",
            "command_id": "ready",
            "confirmation_ref": pending["confirmation_ref"],
        },
    )[1]
    assert ready["guidance"] is not None
    envelope = FieldCommandEnvelope(
        "ready" if rejection == "id_conflict" else "end",
        state.session_id,
        "alice",
        "end_session",
        {},
        2,
        99 if rejection == "order" else 2,
    )
    if rejection == "duplicate_rejected":

        def reject(*args, **kwargs):
            raise ValueError("PREVIOUS_END_REJECTION")

        state.record_store.apply_field_command(envelope, reject)

    class Gateway:
        def claim_field_command(self):
            return envelope

        def complete_field_command(self, result):
            self.result = result

    gateway = Gateway()
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    assert worker.claim_and_apply_once()
    assert (
        gateway.result.result_code
        == {
            "order": "COMMAND_OUT_OF_ORDER",
            "id_conflict": "COMMAND_IDEMPOTENCY_CONFLICT",
            "duplicate_rejected": "PREVIOUS_END_REJECTION",
        }[rejection]
    )
    assert not captures[0].cancelled.is_set()
    captures[0].channel.publish(
        ProductRealtimeEvent(
            "SHOT_STARTED",
            shot_id="shot-0002",
            frame_id=30,
            started_ms=300,
        )
    )
    moving = until(server, lambda p: p["frame"]["phase"] == "moving")
    assert moving["guidance"] is None
    assert state.record_store.list_shots(state.session_id)[-1].started_ms == 300


@pytest.mark.parametrize("failure", ["rejected", "database"])
def test_cloud_end_failure_after_freeze_cannot_retain_ready_route(
    live_server, monkeypatch, failure
):
    import sqlite3

    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope

    server, captures, _ = live_server
    state = server.field_state
    pending = settle(server, captures[0])
    ready = request(
        server,
        "/api/control",
        {
            "command": "continue_turn",
            "command_id": "ready",
            "confirmation_ref": pending["confirmation_ref"],
        },
    )[1]
    assert ready["guidance"] is not None

    class Gateway:
        def claim_field_command(self):
            return FieldCommandEnvelope("end", state.session_id, "alice", "end_session", {}, 2, 2)

        def complete_field_command(self, result):
            assert result.outcome == "rejected"

    def reject(*args, **kwargs):
        raise ValueError("END_REJECTED_AFTER_FREEZE")

    worker = service._BackgroundFieldCommandWorker(
        gateway=Gateway(),
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=reject,
    )
    if failure == "database":

        def unavailable(*args, **kwargs):
            raise sqlite3.OperationalError("end commit unavailable")

        monkeypatch.setattr(state.record_store, "apply_field_command", unavailable)
        with pytest.raises(sqlite3.OperationalError):
            worker.claim_and_apply_once()
    else:
        assert worker.claim_and_apply_once()
    assert captures[0].cancelled.is_set()
    projection = request(server)[1]
    assert projection["rule_status"] == "DEGRADED"
    assert projection["guidance"] is None and projection["confirmation_ref"] is None
    assert projection["frame"]["frame_id"] != ready["frame"]["frame_id"]


@pytest.mark.parametrize("endpoint", ["/api/screen", "/api/tablet"])
def test_http_database_failure_returns_terminal_safe_projection(live_server, monkeypatch, endpoint):
    import sqlite3

    server, captures, _ = live_server
    pending = settle(server, captures[0])
    assert (
        request(
            server,
            "/api/control",
            {
                "command": "continue_turn",
                "command_id": "ready",
                "confirmation_ref": pending["confirmation_ref"],
            },
        )[1]["guidance"]
        is not None
    )

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("projection database unavailable")

    with monkeypatch.context() as broken:
        broken.setattr(server.field_state.record_store, "load_session", unavailable)
        for _ in range(2):
            status, projection = request(server, endpoint)
            assert status == 200
            assert projection["rule_status"] == projection["display"]["phase"] == "DEGRADED"
            assert projection["guidance"] is None and projection["confirmation_ref"] is None
            assert not projection["frame"]["guidance_visible"]
            assert "OperationalError" in projection["degraded_reason"]
            assert "projection database unavailable" in projection["degraded_reason"]
            assert captures[0].cancelled.is_set()
    assert request(server)[1]["rule_status"] == "DEGRADED"


@pytest.mark.parametrize("failure_at", ["commit", "wrapped_commit", "after_freeze_read"])
def test_local_admin_end_failure_is_terminal_before_source_is_marked_ended(
    live_server, monkeypatch, failure_at
):
    import sqlite3

    server, captures, _ = live_server
    state = server.field_state
    pending = settle(server, captures[0])
    ready = request(
        server,
        "/api/control",
        {
            "command": "continue_turn",
            "command_id": "ready",
            "confirmation_ref": pending["confirmation_ref"],
        },
    )[1]
    assert ready["guidance"] is not None

    def unavailable(*args, **kwargs):
        if failure_at == "wrapped_commit":
            try:
                raise sqlite3.OperationalError("local end database unavailable")
            except sqlite3.DatabaseError as exc:
                raise ValueError("field record database is invalid") from exc
        raise sqlite3.OperationalError("local end database unavailable")

    with monkeypatch.context() as broken:
        if failure_at in {"commit", "wrapped_commit"}:
            broken.setattr(state.record_store, "end_session", unavailable)
        else:
            original_prepare = state.projection_source.prepare_session_end

            def prepare_then_break_read():
                original_prepare()
                broken.setattr(state.record_store, "list_shots", unavailable)

            broken.setattr(state.projection_source, "prepare_session_end", prepare_then_break_read)
        # This is the real local administrator entrypoint used by /api/control.
        with pytest.raises((ValueError, sqlite3.OperationalError)):
            state.mutate({"command": "end_session"})
    assert state.record_store.load_session(state.session_id).status.value == "open"
    assert captures[0].cancelled.is_set()
    projection = request(server)[1]
    assert projection["rule_status"] == projection["display"]["phase"] == "DEGRADED"
    assert not state.projection_source._ended
    assert projection["guidance"] is None and projection["confirmation_ref"] is None
    assert not projection["frame"]["guidance_visible"]
    assert projection["frame"]["frame_id"] != ready["frame"]["frame_id"]
    assert "OperationalError" in projection["degraded_reason"]
    assert "local end database unavailable" in projection["degraded_reason"]


def test_local_admin_end_unsettled_rejection(live_server):
    server, captures, _ = live_server
    state = server.field_state
    captures[0].channel.publish(
        ProductRealtimeEvent(
            "SHOT_STARTED",
            shot_id="shot-0001",
            frame_id=10,
            started_ms=100,
        )
    )
    until(server, lambda p: p["frame"]["phase"] == "moving")
    status, rejected = request(server, "/api/control", {"command": "end_session"})
    assert status == 400 and rejected["error"] == "SHOT_UNSETTLED"
    assert state.record_store.load_session(state.session_id).status.value == "open"
    assert request(server)[1]["rule_status"] == "DEGRADED"
    assert captures[0].cancelled.is_set()


def test_local_admin_duplicate_successful_end_remains_ended(live_server):
    server, captures, _ = live_server
    state = server.field_state
    for _ in range(2):
        status, ended = request(server, "/api/control", {"command": "end_session"})
        assert status == 200 and ended["session"]["status"] == "ended"
        assert ended["guidance"] is None and ended["confirmation_ref"] is None
    assert captures[0].cancelled.is_set()
    assert state.record_store.load_session(state.session_id).status.value == "ended"


@pytest.mark.parametrize("invalid", ["missing", "unknown", "epoch", "job", "revision"])
def test_postgame_http_revalidates_frozen_ownership_and_keeps_replay(live_server, invalid):
    server, captures, _ = live_server
    review = ended_review(server, captures[0])
    seed_review_media(server, review)
    store = server.field_state.record_store
    with sqlite3.connect(store.path) as db:
        if invalid == "missing":
            db.execute("DELETE FROM field_shot_ownership")
        elif invalid == "unknown":
            row = json.loads(
                db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
            )
            row.update(evidence_status="unknown", actor_user_id=None, actor_role=None)
            db.execute("UPDATE field_shot_ownership SET payload_json=?", (json.dumps(row),))
        else:
            row = json.loads(
                db.execute("SELECT payload_json FROM field_live_shot_media").fetchone()[0]
            )
            row.update(
                {
                    {"epoch": "source_epoch_id", "job": "job_id", "revision": "match_revision"}[
                        invalid
                    ]: 2 if invalid == "revision" else "wrong"
                }
            )
            db.execute("UPDATE field_live_shot_media SET payload_json=?", (json.dumps(row),))
    _, projection = review_request(server)
    clip = projection["clips"][0]
    assert clip["can_save"] is False
    assert clip["ownership_unavailable_reason"] == "SHOT_OWNERSHIP_UNAVAILABLE"
    assert review_request(server, clip["media_url"])[0] == 200
    payload = dict(
        request_id="invalid-owner",
        session_id=review["review_session_id"],
        shot_id=clip["shot_id"],
        postgame_manifest_sha256=review["postgame_manifest_sha256"],
    )
    assert (
        review_request(server, "/api/postgame-review/save", payload, review["kiosk_nonce"])[0]
        == 409
    )
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0] == 0


def _startup_fixture(tmp_path, *, old_id="old-session", shots=()):
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator

    artifact = (tmp_path / "fresh-artifacts").resolve()
    artifact.mkdir()
    artifact_id = "usb-field-" + hashlib.sha256(str(artifact).encode()).hexdigest()
    store = SQLiteFieldRecordStore(tmp_path / "fresh.sqlite", session_id_factory=lambda: old_id)
    session = store.open_or_resume_session(artifact_id=artifact_id)
    store.save_match(
        session.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            active_role=WristbandRole.RED,
            breaker=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    session = store.load_session(session.session_id)
    store.bind_artifact_path(artifact_id, artifact)
    store.record_cloud_session_open(session.session_id)
    coordinator = LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
    )
    for number, settled in enumerate(shots, 1):
        shot_id = f"shot-{number:04d}"
        coordinator.shot_started(shot_id=shot_id, started_ms=number * 100)
        if settled:
            coordinator.settle(
                LiveSettledShot(
                    shot_id,
                    number * 10,
                    number * 10 + 5,
                    number * 200_000_000,
                    (),
                    False,
                    "unknown",
                    {},
                    "old-epoch",
                    f"job-{number}",
                ),
                facts=(),
            )
    captures = []

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        # Runtime construction must observe the committed successor, never the old scene.
        assert store.load_session(session_id).status.value == "open"
        capture = Capture(session_id, channel, source_epoch_id)
        capture.initial_shot_number = initial_shot_number
        captures.append(capture)
        return capture

    options = dict(
        artifact_path=artifact,
        state_db_path=store.path,
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "new-epoch",
        calibration_id="cal",
        runtime_factory=factory,
        rule_profile_id=PROFILE_ID,
    )
    return store, session, captures, options


def _database_rows(store):
    with sqlite3.connect(store.path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {
            name: connection.execute(f'SELECT * FROM "{name}"').fetchall() for (name,) in tables
        }


@pytest.mark.parametrize("shots", [(), (True,), (False,)])
def test_fresh_start_ends_old_open_atomically_and_clears_match(tmp_path, shots):
    store, old, captures, options = _startup_fixture(tmp_path, shots=shots)
    old_shots = store.list_shots(old.session_id)
    server = service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    try:
        state = server.field_state
        assert state.session_id != old.session_id
        ended = store.load_session(old.session_id)
        assert ended.status.value == "ended" and ended.match == old.match
        assert store.list_shots(old.session_id) == old_shots
        assert old.session_id in store.list_pending_cloud_close_session_ids()
        assert store.active_postgame_review(old.artifact_id).session_id == old.session_id
        assert store.load_session(state.session_id).match == MatchControlSnapshot(
            rule_profile_id=PROFILE_ID
        )
        assert len(captures) == 1
        assert captures[0].initial_shot_number == 0
        assert captures[0].session_id == state.session_id
        assert len(_database_rows(store)["field_sessions"]) == 2
    finally:
        server.server_close()


def test_fresh_first_start_creates_only_one_session(tmp_path):
    store, old, captures, options = _startup_fixture(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        for name in _database_rows(store):
            connection.execute(f'DELETE FROM "{name}"')
    server = service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    try:
        rows = _database_rows(store)
        assert len(rows["field_sessions"]) == 1
        assert rows["field_postgame_manifests"] == []
        assert captures[0].initial_shot_number == 0
    finally:
        server.server_close()


def test_default_usb_builder_resumes_settled_open_and_shot_number(tmp_path):
    store, old, captures, options = _startup_fixture(tmp_path, shots=(True,))
    server = service.build_live_field_assistance_server(**options)
    try:
        assert server.field_state.session_id == old.session_id
        assert store.load_session(old.session_id).status.value == "open"
        assert captures[0].initial_shot_number == 1
    finally:
        server.server_close()


def test_fresh_bind_failure_changes_no_rows_or_runtime(tmp_path):
    import socket

    store, old, captures, options = _startup_fixture(tmp_path, shots=(False,))
    before = _database_rows(store)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        options["port"] = listener.getsockname()[1]
        with pytest.raises(OSError):
            service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    assert _database_rows(store) == before
    assert captures == []


def test_fresh_successor_write_failure_rolls_back_and_releases_listener(tmp_path):
    import socket

    store, old, captures, options = _startup_fixture(tmp_path, shots=(True,))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_successor BEFORE INSERT ON field_match_state "
            "BEGIN SELECT RAISE(ABORT, 'successor rejected'); END"
        )
    before = _database_rows(store)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        options["port"] = listener.getsockname()[1]
    with pytest.raises(ValueError):
        service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    assert _database_rows(store) == before
    assert captures == []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", options["port"]))


@pytest.mark.parametrize(
    "old_id,shots,error",
    [
        ("managed-reserved", (), "managed"),
        ("old-session", (False, True), "ambiguous"),
        ("old-session", (False, False), "ambiguous"),
    ],
)
def test_fresh_refuses_managed_or_ambiguous_without_changes(tmp_path, old_id, shots, error):
    store, old, captures, options = _startup_fixture(tmp_path, old_id=old_id, shots=shots)
    before = _database_rows(store)
    with pytest.raises(ValueError, match=error):
        service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    assert _database_rows(store) == before
    assert captures == []


@pytest.mark.parametrize("failure", ["runtime", "state", "start"])
def test_fresh_construction_failure_releases_listener_and_prepared_runtime(
    tmp_path, monkeypatch, failure
):
    import socket

    store, old, captures, options = _startup_fixture(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        options["port"] = listener.getsockname()[1]
    factory = options["runtime_factory"]

    def prepared_runtime(*args):
        capture = factory(*args)
        capture.shutdown_called = False
        capture.shutdown = lambda: setattr(capture, "shutdown_called", True)
        return capture

    options["runtime_factory"] = prepared_runtime
    if failure == "runtime":

        def fail_runtime(*args):
            raise RuntimeError("runtime construction failed")

        options["runtime_factory"] = fail_runtime
    elif failure == "state":
        options["join_qr_asset_path"] = tmp_path / "missing-qr.png"
    else:

        def fail_start(_source):
            raise RuntimeError("runtime start failed")

        monkeypatch.setattr(service.USBLiveProjectionSource, "start", fail_start)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        service.build_live_field_assistance_server(**options, fresh_session_on_start=True)
    assert all(capture.cancelled.is_set() for capture in captures)
    assert all(capture.shutdown_called for capture in captures)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", options["port"]))
