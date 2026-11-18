from __future__ import annotations

import hashlib
import http.client
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import (
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.lab.field_assistance.cloud_gateway import (
    FieldMatchResultProposal,
    FieldSessionLease,
    JoinRequest,
)
from billiards_ai.lab.field_assistance.server import build_field_assistance_server


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _artifact(root: Path) -> Path:
    root.mkdir()
    timeline = {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": 0,
                "timestamp_ms": 0,
                "phase": "stable",
                "guidance_visible": True,
                "shot_sequence": 0,
                "recommendation_id": "recommendation-1",
                "effects": ["SHOW_GUIDANCE"],
            },
            {
                "frame_id": 1,
                "timestamp_ms": 500,
                "phase": "moving",
                "guidance_visible": False,
                "shot_sequence": 0,
                "recommendation_id": None,
                "effects": ["CLEAR_GUIDANCE", "SHOT_STARTED"],
            },
            {
                "frame_id": 2,
                "timestamp_ms": 900,
                "phase": "stable",
                "guidance_visible": False,
                "shot_sequence": 1,
                "recommendation_id": None,
                "effects": ["SHOT_SETTLED"],
            },
        ],
    }
    recommendations = {
        "schema_version": "1.0",
        "generations": [
            {
                "display": {
                    "recommendation_id": "recommendation-1",
                    "power": {
                        "band": "medium_light",
                        "approximate_percent": 40,
                        "personalized": False,
                    },
                }
            }
        ],
    }
    files = {
        "timeline.json": _canonical(timeline),
        "recommendations.json": _canonical(recommendations),
    }
    media = root / "presentation.mp4"
    writer = cv2.VideoWriter(str(media), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
    assert writer.isOpened()
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
    writer.write(np.full((48, 64, 3), 90, dtype=np.uint8))
    writer.release()
    files["presentation.mp4"] = media.read_bytes()
    inventory = {}
    for name, content in files.items():
        (root / name).write_bytes(content)
        inventory[name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (root / "manifest.json").write_bytes(
        _canonical(
            {
                "kind": "field_route_file_replay",
                "artifact_id": "fixture",
                "inventory": inventory,
            }
        )
    )
    return root


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def running_server(tmp_path: Path):
    clock = _Clock()
    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=clock,
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, clock
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def _request(server, method: str, path: str, *, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    connection.close()
    return result


def test_join_endpoint_serves_immutable_public_table_code_without_control_token(
    tmp_path: Path,
) -> None:
    image = b"\xff\xd8\xffmini-program-code"
    asset = tmp_path / "table-code.jpg"
    asset.write_bytes(image)

    class Gateway:
        def open_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=Gateway(),  # type: ignore[arg-type]
        join_qr_asset_path=asset,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, state_payload = _request(server, "GET", "/api/tablet")
        state = json.loads(state_payload)
        assert status == 200
        assert state["join"]["status"] == "open"
        assert state["display"]["phase"] == "WAITING_FOR_PLAYERS"
        status, headers, payload = _request(server, "GET", "/api/table-code.png")
        assert status == 200
        assert headers["Content-Type"] == "image/jpeg"
        assert payload == image
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_pending_phone_scan_changes_the_public_screen_join_prompt(tmp_path: Path) -> None:
    asset = tmp_path / "table-code.jpg"
    asset.write_bytes(b"\xff\xd8\xffmini-program-code")

    class Gateway:
        def open_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return (JoinRequest("join-1", "球友 87BC6C", "pending", None),)

    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=Gateway(),  # type: ignore[arg-type]
        join_qr_asset_path=asset,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, raw_state = _request(server, "GET", "/api/screen")
        state = json.loads(raw_state)

        assert status == 200
        assert state["display"]["phase"] == "WAITING_FOR_PLAYERS"
        assert state["display"]["join_prompt"] == ("球友 87BC6C 已加入，请在手机认领红方或蓝方")
        assert state["display"]["red_slot"] == "红方待认领"
        assert state["display"]["blue_slot"] == "蓝方待认领"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_confirmed_phone_players_keep_their_names_on_the_public_screen(
    tmp_path: Path,
) -> None:
    class Gateway:
        def open_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return (
                JoinRequest("join-red", "球友 87BC6C", "confirmed", WristbandRole.RED),
                JoinRequest("join-blue", "球友 42AF10", "confirmed", WristbandRole.BLUE),
            )

    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=Gateway(),  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, raw_state = _request(server, "GET", "/api/screen")
        state = json.loads(raw_state)

        assert status == 200
        assert state["display"]["red_slot"] == "红方：球友 87BC6C"
        assert state["display"]["blue_slot"] == "蓝方：球友 42AF10"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_cloud_session_open_failure_is_retried_without_breaking_screen_state(
    tmp_path: Path,
) -> None:
    clock = _Clock()

    class Gateway:
        attempts = 0

        def open_field_session(self, _session_id: str) -> FieldSessionLease:
            self.attempts += 1
            raise OSError("SYNC_CONFIGURATION_REQUIRED")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            raise AssertionError(session_id)

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            raise AssertionError("requests cannot refresh before the cloud session opens")

    gateway = Gateway()
    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=clock,
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=gateway,  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first_status, _, first_payload = _request(server, "GET", "/api/tablet")
        second_status, _, second_payload = _request(server, "GET", "/api/tablet")
        assert first_status == second_status == 200
        assert json.loads(first_payload)["join"] == json.loads(second_payload)["join"]
        assert json.loads(first_payload)["join"]["status"] == "retrying"
        assert json.loads(first_payload)["join"]["error"] == (
            "云端加入服务暂时不可用，系统正在自动重试"
        )
        assert gateway.attempts == 1

        clock.value += 10
        _request(server, "GET", "/api/tablet")
        assert gateway.attempts == 2
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_serves_separate_screen_and_tablet_surfaces(running_server) -> None:
    server, _ = running_server

    screen_status, _, screen = _request(server, "GET", "/screen")
    tablet_status, _, tablet = _request(server, "GET", "/tablet")

    assert screen_status == tablet_status == 200
    assert b"/api/screen" in screen and b"/api/control" not in screen
    assert b"/api/tablet" in tablet and b"/api/control" in tablet


def test_control_token_protects_mutation_and_state_is_synchronized(
    running_server,
) -> None:
    server, clock = running_server
    body = json.dumps(
        {
            "command": "emergency_set_breaker",
            "role": "red",
            "reason": "开发回放应急初始化",
        }
    ).encode()
    common = {"Content-Type": "application/json"}

    assert _request(server, "POST", "/api/control", body=body, headers=common)[0] == 403
    status, _, _ = _request(
        server,
        "POST",
        "/api/control",
        body=body,
        headers={**common, "X-Control-Token": "secret-token"},
    )
    assert status == 200
    start = json.dumps({"command": "start"}).encode()
    assert (
        _request(
            server,
            "POST",
            "/api/control",
            body=start,
            headers={**common, "X-Control-Token": "secret-token"},
        )[0]
        == 200
    )
    clock.value = 10.6

    screen_status, _, screen = _request(server, "GET", "/api/screen")
    tablet_status, _, tablet = _request(server, "GET", "/api/tablet")
    screen_state = json.loads(screen)
    tablet_state = json.loads(tablet)

    assert screen_status == tablet_status == 200
    assert screen_state["match"]["active_role"] == "red"
    assert tablet_state["match"] == screen_state["match"]
    assert screen_state["playback"] == {"playing": True, "position_ms": 600}
    assert screen_state["frame"]["frame_id"] == 1
    assert (
        server.field_state.record_store.list_admin_audit(server.field_state.session_id)[0]["reason"]
        == "开发回放应急初始化"
    )


def test_tablet_token_rejects_ordinary_player_control(running_server) -> None:
    server, _ = running_server

    assert _mutate(server, {"command": "switch_turn"}) == 400


def test_media_supports_one_bounded_range(running_server) -> None:
    server, _ = running_server
    media = (server.field_state.artifact_path / "presentation.mp4").read_bytes()

    status, headers, body = _request(
        server,
        "GET",
        "/media/presentation.mp4",
        headers={"Range": "bytes=2-5"},
    )

    assert status == 206
    assert headers["Content-Range"] == f"bytes 2-5/{len(media)}"
    assert headers["Accept-Ranges"] == "bytes"
    assert body == media[2:6]
    assert (
        _request(
            server,
            "GET",
            "/media/presentation.mp4",
            headers={"Range": f"bytes={len(media) - 2}-"},
        )[2]
        == media[-2:]
    )
    assert (
        _request(
            server,
            "GET",
            "/media/presentation.mp4",
            headers={"Range": "bytes=1-2,4-5"},
        )[0]
        == 416
    )


def test_rejects_oversized_control_body(running_server) -> None:
    server, _ = running_server

    status, _, _ = _request(
        server,
        "POST",
        "/api/control",
        body=b"x" * 32769,
        headers={
            "Content-Type": "application/json",
            "X-Control-Token": "secret-token",
        },
    )

    assert status == 413


def test_unknown_route_is_not_exposed(running_server) -> None:
    server, _ = running_server

    assert _request(server, "GET", "/manifest.json")[0] == 404


def test_serves_browser_native_jpeg_frame(running_server) -> None:
    server, _ = running_server

    status, headers, payload = _request(server, "GET", "/media/frame/0.jpg")

    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    assert payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


def test_refuses_tampered_artifact_before_binding(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    (artifact / "presentation.mp4").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="inventory mismatch"):
        build_field_assistance_server(
            artifact_path=artifact,
            host="127.0.0.1",
            port=0,
            control_token="secret-token",
            monotonic=_Clock(),
            duration_probe=lambda _: 1_500,
            state_db_path=tmp_path / "field.sqlite3",
        )


def _mutate(server, payload: dict[str, object]) -> int:
    return _request(
        server,
        "POST",
        "/api/control",
        body=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Control-Token": "secret-token",
        },
    )[0]


def _apply_breaker_command(server, *, role: str = "red", command_id: str = "command-breaker"):
    reason = "开发回放应急初始化"
    for participant_role in ("red", "blue"):
        assert (
            _mutate(
                server,
                {
                    "command": "emergency_set_participant",
                    "role": participant_role,
                    "user_id": f"user-{participant_role}",
                    "reason": reason,
                },
            )
            == 200
        )
    envelope = FieldCommandEnvelope(
        command_id=command_id,
        session_id=server.field_state.session_id,
        actor_user_id=f"user-{role}",
        command_type="set_breaker",
        payload={"role": role},
        expected_revision=2,
        server_sequence=1,
    )
    application = server.field_state.record_store.apply_field_command(
        envelope, reduce_field_command
    )
    server.field_state.set_command_match(
        envelope.session_id,
        server.field_state.record_store.load_session(server.field_state.session_id).match,
    )
    server.field_state.handle_command_application(envelope, application)
    assert application.outcome == "applied"
    return envelope, application


def test_applied_breaker_command_starts_existing_replay_clock_once(running_server) -> None:
    server, clock = running_server

    envelope, application = _apply_breaker_command(server)
    clock.value = 10.4
    first = server.field_state.payload()
    server.field_state.handle_command_application(envelope, application)
    clock.value = 10.6
    second = server.field_state.payload()

    assert first["playback"] == {"position_ms": 400, "playing": True}
    assert second["playback"] == {"position_ms": 600, "playing": True}


def test_ended_receipt_rolls_over_to_a_clean_session_without_restarting_server(
    running_server,
) -> None:
    server, clock = running_server
    original = server.field_state.payload()
    original_session_id = original["session"]["session_id"]
    table_code_image = original["display"]["table_code_image"]
    _apply_breaker_command(server)
    end = FieldCommandEnvelope(
        command_id="command-end",
        session_id=original_session_id,
        actor_user_id="user-red",
        command_type="end_session",
        payload={},
        expected_revision=3,
        server_sequence=2,
    )
    application = server.field_state.record_store.apply_field_command(end, reduce_field_command)

    server.field_state.handle_command_application(end, application)
    ended = server.field_state.payload()
    clock.value += 2.9
    receipt = server.field_state.payload()
    clock.value += 0.2
    next_match = server.field_state.payload()

    assert application.outcome == "applied"
    assert ended["session"] == {
        "session_id": original_session_id,
        "status": "ended",
    }
    assert ended["display"]["phase"] == "SESSION_ENDED"
    assert receipt["session"]["session_id"] == original_session_id
    assert next_match["session"]["status"] == "open"
    assert next_match["session"]["session_id"] != original_session_id
    assert next_match["display"]["phase"] == "WAITING_FOR_PLAYERS"
    assert next_match["display"]["table_code_image"] == table_code_image
    assert next_match["display"]["last_action"] is None
    assert next_match["match"] == {
        "red_user_id": None,
        "blue_user_id": None,
        "breaker": None,
        "active_role": None,
        "red_group": None,
        "blue_group": None,
        "rule_profile_id": None,
        "rack_phase": "break",
        "cue_ball_placement": "none",
        "winner_role": None,
        "terminal_reason": None,
        "mode": None,
        "match_format": None,
        "race_target": None,
        "ranking_intent": False,
        "settings_revision": 0,
        "revision": 0,
    }
    assert next_match["playback"] == {"position_ms": 0, "playing": False}


def test_rollover_reuses_permanent_qr_and_retries_previous_cloud_close(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    image = b"\xff\xd8\xffpermanent-table-code"
    asset = tmp_path / "table-code.jpg"
    asset.write_bytes(image)

    class Gateway:
        def __init__(self) -> None:
            self.opened: list[str] = []
            self.closed: list[str] = []

        def open_field_session(self, session_id: str) -> FieldSessionLease:
            self.opened.append(session_id)
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            self.closed.append(session_id)
            if self.closed.count(session_id) == 1:
                raise OSError("temporary close failure")
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    gateway = Gateway()
    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=clock,
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=gateway,  # type: ignore[arg-type]
        join_qr_asset_path=asset,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        original = server.field_state.payload()
        original_session_id = original["session"]["session_id"]
        _apply_breaker_command(server)
        end = FieldCommandEnvelope(
            command_id="command-end-cloud",
            session_id=original_session_id,
            actor_user_id="user-red",
            command_type="end_session",
            payload={},
            expected_revision=3,
            server_sequence=2,
        )
        application = server.field_state.record_store.apply_field_command(end, reduce_field_command)
        server.field_state.handle_command_application(end, application)
        ended = server.field_state.payload()
        clock.value += 3.1
        next_match = server.field_state.payload()
        qr_status, _, qr_after_rollover = _request(server, "GET", "/api/table-code.png")
        clock.value += 7.0
        server.field_state.payload()

        assert ended["display"]["phase"] == "SESSION_ENDED"
        assert next_match["session"]["status"] == "open"
        assert next_match["session"]["session_id"] != original_session_id
        assert gateway.opened == [
            original_session_id,
            next_match["session"]["session_id"],
        ]
        assert gateway.closed == [original_session_id, original_session_id]
        assert qr_status == 200
        assert qr_after_rollover == image
        assert next_match["join"]["qr_available"] is True
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_two_consecutive_sessions_publish_distinct_facts_and_clips(running_server) -> None:
    server, clock = running_server
    published = []

    for match_number in (1, 2):
        session_id = server.field_state.session_id
        _apply_breaker_command(
            server,
            command_id=f"command-breaker-{match_number}",
        )
        clock.value += 1.0
        state = server.field_state.payload()
        assert state["playback"] == {"position_ms": 1000, "playing": True}
        end = FieldCommandEnvelope(
            command_id=f"command-end-{match_number}",
            session_id=session_id,
            actor_user_id="user-red",
            command_type="end_session",
            payload={},
            expected_revision=3,
            server_sequence=2,
        )
        application = server.field_state.record_store.apply_field_command(end, reduce_field_command)
        server.field_state.handle_command_application(end, application)
        queued = server.field_state.record_store.claim_due_sync_job(datetime.now(UTC))

        assert application.outcome == "applied"
        assert queued is not None
        assert queued.session_id == session_id
        assert len(queued.media) == 1
        assert queued.media[0].local_path.is_file()
        manifest = json.loads(queued.manifest_json)
        assert manifest["session"]["session_id"] == session_id
        assert len(manifest["session"]["shots"]) == 1
        assert len(manifest["media"]) == 1
        published.append((session_id, queued.media[0].local_path))

        if match_number == 1:
            blocked = server.field_state.record_store.mark_sync_blocked(
                queued.sync_job_id,
                error_code="DEVELOPMENT_CLOUD_OFFLINE",
            )
            assert blocked.status == "blocked"
            clock.value += 3.1
            rolled = server.field_state.payload()
            assert rolled["session"]["status"] == "open"
            assert rolled["session"]["session_id"] != session_id

    assert published[0][0] != published[1][0]
    assert published[0][1] != published[1][1]


def test_duplicate_end_acknowledgement_enqueues_only_one_sync_job(running_server) -> None:
    server, _ = running_server
    session_id = server.field_state.session_id
    _apply_breaker_command(server)
    end = FieldCommandEnvelope(
        command_id="command-end-once",
        session_id=session_id,
        actor_user_id="user-red",
        command_type="end_session",
        payload={},
        expected_revision=3,
        server_sequence=2,
    )
    application = server.field_state.record_store.apply_field_command(end, reduce_field_command)

    server.field_state.handle_command_application(end, application)
    server.field_state.handle_command_application(end, application)
    first = server.field_state.record_store.claim_due_sync_job(datetime.now(UTC))
    second = server.field_state.record_store.claim_due_sync_job(datetime.now(UTC))

    assert first is not None
    assert first.session_id == session_id
    assert server.field_state.record_store.list_shots(session_id) == ()
    manifest = json.loads(first.manifest_json)
    assert manifest["session"]["evidence_source_kind"] == "development_replay"
    assert second is None


def test_failed_initial_sync_enqueue_is_retried_from_ended_session(
    running_server, monkeypatch
) -> None:
    server, _ = running_server
    state = server.field_state
    session_id = state.session_id
    _apply_breaker_command(server)
    original = state._prepare_sync_media
    attempts = 0

    def fail_once(session, shots):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary encoder outage")
        return original(session, shots)

    monkeypatch.setattr(
        type(state), "_prepare_sync_media", lambda self, s, shots: fail_once(s, shots)
    )
    end = FieldCommandEnvelope(
        command_id="command-end-retry-enqueue",
        session_id=session_id,
        actor_user_id="user-red",
        command_type="end_session",
        payload={},
        expected_revision=3,
        server_sequence=2,
    )
    application = state.record_store.apply_field_command(end, reduce_field_command)

    state.handle_command_application(end, application)
    assert attempts == 1
    state.payload()

    queued = state.record_store.claim_due_sync_job(datetime.now(UTC))
    assert attempts == 2
    assert queued is not None and queued.session_id == session_id


def test_missing_outbox_blocks_rollover_until_recovery_is_durable(
    running_server, monkeypatch
) -> None:
    server, clock = running_server
    state = server.field_state
    session_id = state.session_id
    _apply_breaker_command(server)
    monkeypatch.setattr(
        type(state),
        "_prepare_sync_media",
        lambda _self, _session, _shots: (_ for _ in ()).throw(OSError("disk offline")),
    )
    state.record_store.end_session(session_id)
    state.session_ended_at_monotonic = clock.value
    clock.value += 10

    payload = state.payload()

    assert payload["session"] == {"session_id": session_id, "status": "ended"}
    assert state.record_store.list_ended_sessions_without_sync()[0].session_id == session_id


def test_server_startup_recovers_missing_outbox_without_ui_request(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    database = tmp_path / "field.sqlite3"
    first = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
    )
    session_id = first.field_state.session_id
    first.field_state.record_store.end_session(session_id)
    first.server_close()

    restarted = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
    )
    try:
        assert restarted.field_state.record_store.has_sync_job(session_id)
    finally:
        restarted.server_close()


def test_sync_media_rejects_a_partially_decoded_clip(running_server, monkeypatch) -> None:
    server, clock = running_server
    state = server.field_state
    _apply_breaker_command(server)
    clock.value += 1.0
    state.payload()
    session = state.record_store.load_session(state.session_id)
    shots = state.record_store.list_shots(state.session_id)
    real_capture = cv2.VideoCapture

    class TruncatedCapture:
        def __init__(self, path: str) -> None:
            self.inner = real_capture(path)
            self.reads = 0

        def isOpened(self):
            return self.inner.isOpened()

        def get(self, key):
            return self.inner.get(key)

        def set(self, key, value):
            self.reads = 0
            return self.inner.set(key, value)

        def read(self):
            if self.reads >= 1:
                return False, None
            self.reads += 1
            return self.inner.read()

        def release(self):
            self.inner.release()

    monkeypatch.setattr(
        "billiards_ai.lab.field_assistance.server.cv2.VideoCapture", TruncatedCapture
    )

    with pytest.raises(ValueError, match="complete frame range"):
        state._prepare_sync_media(session, shots)


def test_multiple_failed_cloud_closes_remain_queued(tmp_path: Path) -> None:
    clock = _Clock()

    class Gateway:
        def __init__(self) -> None:
            self.closed: list[str] = []

        def open_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            self.closed.append(session_id)
            raise OSError("offline")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    gateway = Gateway()
    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=clock,
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=gateway,  # type: ignore[arg-type]
    )
    try:
        state = server.field_state
        state.payload()
        current = state.session_id
        state.pending_cloud_closes["older-session"] = 0
        state.record_store.end_session(current)

        state._maintain_join(ended=True)
        clock.value += 10.1
        state._maintain_join(ended=True)

        assert gateway.closed == [
            "older-session",
            current,
            "older-session",
            current,
        ]
    finally:
        server.server_close()


def test_cloud_close_intent_is_replayed_after_server_restart(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    database = tmp_path / "field.sqlite3"

    class FailingGateway:
        def open_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, _session_id: str) -> FieldSessionLease:
            raise OSError("offline")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    first = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
        cloud_gateway=FailingGateway(),  # type: ignore[arg-type]
    )
    old_session_id = first.field_state.session_id
    first.field_state.payload()
    first.field_state.record_store.end_session(old_session_id)
    first.field_state._maintain_join(ended=True)
    first.server_close()

    closed: list[str] = []

    class RecoveredGateway(FailingGateway):
        def close_field_session(self, session_id: str) -> FieldSessionLease:
            closed.append(session_id)
            return FieldSessionLease(session_id, "table-1", "ended")

    restarted = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
        cloud_gateway=RecoveredGateway(),  # type: ignore[arg-type]
    )
    try:
        restarted.field_state._maintain_join(ended=True)
        assert closed == [old_session_id]
        assert restarted.field_state.record_store.list_pending_cloud_close_session_ids() == ()
    finally:
        restarted.server_close()


def test_ranked_terminal_result_is_proposed_before_the_cloud_session_closes(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    class Gateway:
        def open_field_session(self, session_id: str) -> FieldSessionLease:
            calls.append(("open", session_id))
            return FieldSessionLease(session_id, "table-1", "open")

        def propose_match_result(self, **payload) -> FieldMatchResultProposal:
            calls.append(("result", payload))
            return FieldMatchResultProposal(
                session_id=payload["session_id"],
                status="awaiting_result_confirmations",
                winner_role=payload["winner_role"],
                red_score=payload["red_score"],
                blue_score=payload["blue_score"],
                finish_reason=payload["finish_reason"],
            )

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            calls.append(("close", session_id))
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=Gateway(),  # type: ignore[arg-type]
    )
    try:
        state = server.field_state
        state._maintain_join(ended=False)
        session_id = state.session_id
        terminal = MatchControlSnapshot(
            red_user_id="user-red",
            blue_user_id="user-blue",
            breaker=WristbandRole.RED,
            active_role=WristbandRole.RED,
            revision=1,
            rule_profile_id="cbsa-chinese-pool-2017-observable-v1",
            rack_phase=RackPhase.ENDED,
            winner_role=WristbandRole.RED,
            terminal_reason="legal_eight",
            mode=MatchMode.OBSERVATION,
            match_format=MatchFormat.SINGLE,
            ranking_intent=True,
            settings_revision=1,
        )
        state.record_store.save_match(session_id, terminal, expected_revision=0)
        state.record_store.end_session(session_id)
        state.match = terminal

        # Never invite confirmations before the immutable session/media package
        # that ranking qualification reads has reached CloudBase.
        state._maintain_join(ended=True)
        assert calls == [("open", session_id)]
        assert session_id in state.pending_cloud_closes

        state.record_store.is_session_sync_complete = lambda _session_id: True
        state.pending_cloud_closes[session_id] = 0
        state._maintain_join(ended=True)

        assert calls == [
            ("open", session_id),
            (
                "result",
                {
                    "session_id": session_id,
                    "winner_role": WristbandRole.RED,
                    "red_score": 1,
                    "blue_score": 0,
                    "finish_reason": "normal",
                },
            ),
            ("close", session_id),
        ]
    finally:
        server.server_close()


def test_cloud_open_is_not_marked_live_until_close_intent_is_durable(
    tmp_path: Path, monkeypatch
) -> None:
    class Gateway:
        attempts = 0

        def open_field_session(self, session_id: str) -> FieldSessionLease:
            self.attempts += 1
            return FieldSessionLease(session_id, "table-1", "open")

        def close_field_session(self, session_id: str) -> FieldSessionLease:
            return FieldSessionLease(session_id, "table-1", "ended")

        def refresh_join_requests(self, _session_id: str) -> tuple[JoinRequest, ...]:
            return ()

    gateway = Gateway()
    server = build_field_assistance_server(
        artifact_path=_artifact(tmp_path / "artifact"),
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=tmp_path / "field.sqlite3",
        cloud_gateway=gateway,  # type: ignore[arg-type]
    )
    state = server.field_state
    original = state.record_store.record_cloud_session_open
    calls = 0

    def fail_once(session_id: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("sqlite busy")
        original(session_id)

    monkeypatch.setattr(state.record_store, "record_cloud_session_open", fail_once)
    try:
        with pytest.raises(ValueError, match="sqlite busy"):
            state._maintain_join(ended=False)
        assert state.cloud_session_status != "open"

        state._maintain_join(ended=False)

        assert gateway.attempts == 2
        assert state.cloud_session_status == "open"
    finally:
        server.server_close()


def test_restart_resumes_open_session_and_skips_ended_session(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    database = tmp_path / "field.sqlite3"
    first = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
    )
    first_session_id = first.field_state.session_id
    first.server_close()

    resumed = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
    )
    assert resumed.field_state.session_id == first_session_id
    resumed.field_state.record_store.end_session(first_session_id)
    resumed.field_state.record_store.enqueue_sync_job(
        session_id=first_session_id,
        manifest_hash="a" * 64,
        manifest_json="{}",
    )
    resumed.server_close()

    restarted = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        duration_probe=lambda _: 1_500,
        state_db_path=database,
    )
    try:
        state = restarted.field_state.payload()
        assert state["session"]["status"] == "open"
        assert state["session"]["session_id"] != first_session_id
        assert state["match"]["revision"] == 0
        assert (
            restarted.field_state.record_store.load_session(first_session_id).status.value
            == "ended"
        )
    finally:
        restarted.server_close()


def test_breaker_is_enough_to_enter_first_shot_before_groups_are_assigned(
    running_server,
) -> None:
    server, _ = running_server

    _apply_breaker_command(server)
    state = server.field_state.payload()

    assert state["match"]["red_group"] is None
    assert state["display"]["phase"] == "WAITING_FOR_SHOT"


def test_fact_apis_are_protected_and_capture_actor_before_role_switch(
    running_server,
) -> None:
    server, clock = running_server
    reason = "开发回放应急初始化"
    assert (
        _mutate(
            server,
            {
                "command": "emergency_set_participant",
                "role": "red",
                "user_id": "user-red",
                "reason": reason,
            },
        )
        == 200
    )
    assert (
        _mutate(
            server,
            {
                "command": "emergency_set_participant",
                "role": "blue",
                "user_id": "user-blue",
                "reason": reason,
            },
        )
        == 200
    )
    assert (
        _mutate(
            server,
            {
                "command": "emergency_set_breaker",
                "role": "red",
                "reason": reason,
            },
        )
        == 200
    )
    assert _mutate(server, {"command": "start"}) == 200
    clock.value = 11.0
    assert _request(server, "GET", "/api/screen")[0] == 200
    switch = FieldCommandEnvelope(
        command_id="command-1",
        session_id=server.field_state.session_id,
        actor_user_id="user-red",
        command_type="switch_turn",
        payload={},
        expected_revision=3,
        server_sequence=1,
    )
    switch_result = server.field_state.record_store.apply_field_command(
        switch, reduce_field_command
    )
    server.field_state.set_command_match(
        switch.session_id,
        server.field_state.record_store.load_session(server.field_state.session_id).match,
    )
    assert switch_result.outcome == "applied"
    end = FieldCommandEnvelope(
        command_id="command-2",
        session_id=server.field_state.session_id,
        actor_user_id="user-blue",
        command_type="end_session",
        payload={},
        expected_revision=4,
        server_sequence=2,
    )
    end_result = server.field_state.record_store.apply_field_command(end, reduce_field_command)
    server.field_state.handle_command_application(end, end_result)
    assert end_result.outcome == "applied"
    queued = server.field_state.record_store.claim_due_sync_job(datetime.now(UTC))
    assert queued is not None
    assert queued.session_id == server.field_state.session_id
    assert queued.media[0].shot_id == "shot-0001"

    assert _request(server, "GET", "/api/local/shots")[0] == 403
    status, _, raw_shots = _request(
        server,
        "GET",
        "/api/local/shots",
        headers={"X-Control-Token": "secret-token"},
    )
    assert status == 200
    shots = json.loads(raw_shots)["shots"]
    assert len(shots) == 1
    assert shots[0]["actor_role"] == "red"
    assert shots[0]["power_band"] == "medium_light"
    assert shots[0]["media"]["source_kind"] == "development_replay"

    report_status, _, raw_report = _request(
        server,
        "GET",
        "/api/local/report",
        headers={"X-Control-Token": "secret-token"},
    )
    assert report_status == 200
    assert json.loads(raw_report)["report_type"] == "field_session_facts"


def test_server_restart_restores_match_and_does_not_duplicate_shot(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "artifact")
    state_db = tmp_path / "field.sqlite3"
    first_clock = _Clock()
    first = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=first_clock,
        duration_probe=lambda _: 1_500,
        state_db_path=state_db,
    )
    try:
        first.field_state.mutate(
            {
                "command": "emergency_set_breaker",
                "role": "red",
                "reason": "开发回放应急初始化",
            }
        )
        first.field_state.mutate({"command": "start"})
        first_clock.value = 11.0
        first.field_state.payload()
        session_id = first.field_state.session_id
    finally:
        first.server_close()

    second_clock = _Clock()
    second = build_field_assistance_server(
        artifact_path=artifact,
        host="127.0.0.1",
        port=0,
        control_token="secret-token",
        monotonic=second_clock,
        duration_probe=lambda _: 1_500,
        state_db_path=state_db,
    )
    try:
        assert second.field_state.session_id == session_id
        assert second.field_state.match.active_role.value == "red"
        second.field_state.mutate({"command": "start"})
        second_clock.value = 11.0
        second.field_state.payload()
        assert len(second.field_state.record_store.list_shots(session_id)) == 1
    finally:
        second.server_close()
