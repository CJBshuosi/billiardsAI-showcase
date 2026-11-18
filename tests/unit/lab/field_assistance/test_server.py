from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import replace
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
)
from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import ClipSaveStatus, PostgameClip
from billiards_ai.lab.field_assistance import server as field_server
from billiards_ai.lab.field_assistance.server import (
    _BackgroundFieldCommandWorker,
    _FieldAssistanceHandler,
    _load_join_qr_asset,
)


class _AbortedWriter:
    def write(self, payload: bytes) -> None:
        del payload
        raise ConnectionAbortedError(10053, "client closed request")


def test_media_maintenance_dispatches_once_without_blocking_and_joins_on_close(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    state = object.__new__(field_server._ServerState)
    state.media_job_store = object()
    state._media_thread = None
    state._media_thread_lock = threading.Lock()
    state._media_stopped = threading.Event()
    calls = []

    def encode(_state):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(field_server._ServerState, "_run_recording_media", encode)
    try:
        state.maintain_service()
        assert entered.wait(1)
        state.maintain_service()
        assert len(calls) == 1
        assert calls[0] != threading.get_ident()
    finally:
        release.set()
        state.close_media_worker()
    assert not state._media_thread.is_alive()
    state.maintain_service()
    assert len(calls) == 1


def test_ended_media_probe_uses_one_closed_connection_and_sees_new_live_session(
    tmp_path, monkeypatch
):
    # Reopening three connections per frame is the measured regression. A later
    # committed live session must still pause the very next probe (no cached idle).
    import sqlite3

    from tests.unit.lab.test_mainline1_recording_media_worker import _ordinary_scene

    store, session_id, _ = _ordinary_scene(tmp_path)
    state = SimpleNamespace(
        match_lock=threading.RLock(),
        _media_stopped=threading.Event(),
        record_store=store,
        session_id=session_id,
        projection_lifecycle_closing=False,
        projection_lifecycle_closed=False,
    )
    gate = field_server._FieldMediaResourceGate(state)
    original_connect = store._connect
    connections = []

    def connect(**kwargs):
        connection = original_connect(**kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", connect)
    assert gate.may_run_media_encoding(session_id)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")

    peer = SQLiteFieldRecordStore(store.path)
    peer.open_or_resume_session(artifact_id="next-table", session_policy="recording-v1")
    assert not gate.may_run_media_encoding(session_id)
    assert len(connections) == 2
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[-1].execute("SELECT 1")


@pytest.mark.parametrize("fault", ["missing-session", "corrupt-match", "open-session"])
def test_ended_media_probe_fails_closed_for_invalid_or_active_target(tmp_path, fault):
    import sqlite3

    from tests.unit.lab.test_mainline1_recording_media_worker import _ordinary_scene

    store, session_id, _ = _ordinary_scene(tmp_path)
    with sqlite3.connect(store.path) as connection:
        if fault == "corrupt-match":
            connection.execute(
                "UPDATE field_match_state SET payload_json='{}' WHERE session_id=?", (session_id,)
            )
        elif fault == "open-session":
            connection.execute(
                "UPDATE field_sessions SET status='open',ended_at_utc=NULL WHERE session_id=?",
                (session_id,),
            )
    if fault == "missing-session":
        session_id = "missing"
    state = SimpleNamespace(
        match_lock=threading.RLock(),
        _media_stopped=threading.Event(),
        record_store=store,
        session_id=session_id,
        projection_lifecycle_closing=False,
        projection_lifecycle_closed=False,
    )
    assert not field_server._FieldMediaResourceGate(state).may_run_media_encoding(session_id)


@pytest.mark.parametrize("start_kind", ["ready", "shot", "legacy"])
def test_recording_rollover_activity_hides_review_and_pauses_media(tmp_path, start_kind):
    from billiards_ai.edge.route_assistance.commands import reduce_field_command
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
    from tests.unit.lab.test_mainline1_recording_media_worker import _ordinary_scene

    store, old_id, root = _ordinary_scene(tmp_path)
    new = store.open_or_resume_session(
        artifact_id="recording",
        session_policy="legacy" if start_kind == "legacy" else "recording-v1",
    )

    class Source(USBLiveProjectionSource):
        def __init__(self):
            pass

        def media_encoding_idle(self):
            return True

        def current(self):
            return SimpleNamespace(
                frame=SimpleNamespace(
                    payload=dict(
                        session_id=new.session_id,
                        source_kind="usb_live",
                        phase="stable",
                        rule_status="WAITING_FOR_SHOT",
                        guidance_visible=False,
                    )
                )
            )

    state = field_server._ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=None,
        match=new.match,
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id=new.session_id,
        artifact_id="recording",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        projection_source=Source(),
    )
    gate = field_server._FieldMediaResourceGate(state)
    if start_kind == "legacy":
        assert not gate.may_run_media_encoding(old_id)
        assert state.postgame_review_payload()["review_session_id"] == old_id
        store.save_match(
            new.session_id,
            replace(
                new.match,
                red_user_id="alice",
                blue_user_id="bob",
                breaker=WristbandRole.RED,
                active_role=WristbandRole.RED,
                revision=1,
            ),
            expected_revision=0,
        )
        assert state.postgame_review_payload()["phase"] == "LIVE_MATCH"
        assert store.active_postgame_review("recording").session_id == old_id
        return
    assert gate.may_run_media_encoding(old_id)
    assert state.postgame_review_payload()["review_session_id"] == old_id
    for sequence, (user, role) in enumerate((("alice", "red"), ("bob", "blue")), 1):
        result = store.apply_field_command(
            FieldCommandEnvelope(
                f"join-{role}",
                new.session_id,
                user,
                "claim_role",
                {"role": role},
                sequence - 1,
                sequence,
            ),
            reduce_field_command,
        )
        assert result.outcome == "applied"
    assert gate.may_run_media_encoding(old_id)
    assert state.postgame_review_payload()["review_session_id"] == old_id
    if start_kind == "ready":
        result = store.apply_field_command(
            FieldCommandEnvelope(
                "breaker",
                new.session_id,
                "alice",
                "set_breaker",
                {
                    "role": "red",
                    "mode": "assistance",
                    "format": "single",
                    "race_target": None,
                    "ranking_intent": False,
                    "settings_revision": 1,
                },
                2,
                3,
            ),
            reduce_field_command,
        )
        assert result.outcome == "applied"
    else:
        store.record_recording_start(new.session_id, "real-start", "epoch-next", 10, 166)
    assert not gate.may_run_media_encoding(old_id)
    assert state.postgame_review_payload()["review_session_id"] is None
    assert store.active_postgame_review("recording").session_id == old_id


@pytest.mark.parametrize(
    "pending,status,expected",
    [
        (0, "running", True),
        (1, "running", False),
        (0, "degraded", False),
        (None, "running", False),
    ],
)
def test_media_idle_health_requires_healthy_capture_and_empty_recording_queue(
    pending, status, expected
):
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource

    source = object.__new__(USBLiveProjectionSource)
    source.lock = threading.RLock()
    source._stopped = threading.Event()
    source._restart_reason = None
    source.runtime = SimpleNamespace(
        video_health=SimpleNamespace(state="ready", reason=None),
        recording_health=SimpleNamespace(state=status, reason=None, pending_chunks=pending),
    )
    assert source.media_encoding_idle() is expected


def test_review_completion_readiness_and_stream_pin_gate(tmp_path: Path):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from tests.unit.edge.route_assistance.test_persistence import (
        _available_clip,
        _postgame_store,
    )

    store = _postgame_store(tmp_path / "field.sqlite3")
    store.bind_artifact_path("artifact-1", tmp_path)
    state = field_server._ServerState(
        artifact_path=tmp_path,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id="session-1",
        artifact_id="artifact-1",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
    )
    projection = state.postgame_review_payload(kiosk=True)
    body = {
        "session_id": projection["review_session_id"],
        "postgame_manifest_sha256": projection["postgame_manifest_sha256"],
    }
    nonce = projection["kiosk_nonce"]
    assert projection["can_complete"] is True
    overlay = tmp_path / "postgame" / "session-1" / ("c" * 64) / "overlay.mp4"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"clip")
    _available_clip(
        store,
        tmp_path,
        local_path=overlay,
        byte_size=4,
        content_sha256=hashlib.sha256(b"clip").hexdigest(),
    )
    assert state.postgame_review_payload()["can_complete"] is True
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    assert state.postgame_review_payload()["can_complete"] is True
    handler = object.__new__(_FieldAssistanceHandler)
    handler.server = SimpleNamespace(field_state=state)
    streamed = []

    def stream_under_pin(source, size, *, continue_check):
        assert size == 4 and source.read() == b"clip"
        completed = state.mutate_postgame_review(body, nonce, complete=True)
        assert completed["phase"] == "REVIEW_COMPLETE"
        assert source.read() == b""
        streamed.append(True)

    handler._stream_media = stream_under_pin
    handler._error = lambda *args: pytest.fail("valid pinned stream refused")
    clip = store.list_postgame_clips("session-1")[0]
    handler._serve_postgame_media(clip.clip_id, "manifest=" + body["postgame_manifest_sha256"])
    assert streamed == [True]
    with session_media_lock(tmp_path, "session-1", exclusive=False):
        store.complete_postgame_review("artifact-1", "session-1", body["postgame_manifest_sha256"])
    with pytest.raises(ValueError, match="unavailable"):
        state.review_clip(clip.clip_id, body["postgame_manifest_sha256"])
    assert state.postgame_review_payload()["can_complete"] is False
    assert store.is_postgame_discarded("session-1", "shot-0001")


@pytest.mark.parametrize(
    "peer,expected",
    [
        ("127.0.0.1", True),
        ("127.25.1.2", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("192.168.1.2", False),
        ("::ffff:192.168.1.2", False),
        ("localhost", False),
        ("", False),
    ],
)
def test_review_authority_uses_socket_peer_only(peer, expected):
    handler = object.__new__(_FieldAssistanceHandler)
    handler.client_address = (peer, 12345)
    handler.headers = {
        "Host": "localhost",
        "X-Forwarded-For": "127.0.0.1",
        "Forwarded": "for=127.0.0.1",
        "Origin": "http://localhost",
    }
    assert handler._is_loopback_peer() is expected


@pytest.mark.parametrize(
    "kind",
    [
        "escape",
        "traversal",
        "other_session",
        "directory",
        "parent_symlink",
        "type",
        "fifo",
        "empty",
        "session_traversal",
    ],
)
def test_review_media_rejects_unsafe_recorded_paths(tmp_path, kind):
    root = tmp_path / "artifact"
    folder = root / "postgame" / "session-1"
    folder.mkdir(parents=True)
    path = folder / "clip.mp4"
    path.write_bytes(b"test")
    clip = PostgameClip(
        "session-1",
        "shot-1",
        "a" * 64,
        "job-1",
        ClipSaveStatus.NOT_SAVED,
        "clip-id",
        "analysis",
        path,
        hashlib.sha256(b"test").hexdigest(),
        4,
        "video/mp4",
    )
    if kind == "escape":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"test")
        clip = replace(clip, local_path=outside)
    elif kind == "traversal":
        clip = replace(clip, local_path=folder / ".." / "session-1" / "clip.mp4")
    elif kind == "other_session":
        clip = replace(clip, session_id="session-2")
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "parent_symlink":
        folder.rename(root / "target")
        folder.symlink_to(root / "target", target_is_directory=True)
    elif kind == "type":
        clip = replace(clip, content_type="text/plain")
    elif kind == "empty":
        path.write_bytes(b"")
        clip = replace(clip, byte_size=0)
    elif kind == "session_traversal":
        (root / "clip.mp4").write_bytes(b"test")
        clip = replace(clip, session_id="..", local_path=root / "postgame" / ".." / "clip.mp4")
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        with field_server._open_review_media(root, clip):
            pytest.fail("unsafe media was opened")


def test_review_stream_disconnect_is_bounded_and_does_not_escape():
    from io import BytesIO

    handler = object.__new__(_FieldAssistanceHandler)
    handler.wfile = _AbortedWriter()
    handler.headers = {}
    handler._send_media_headers = lambda *args, **kwargs: None
    handler._stream_media(BytesIO(b"123456"), 6)


@pytest.mark.parametrize("change", ["none", "symlink", "junction", "identity"])
def test_review_media_portable_open_validates_links_and_identity(tmp_path, monkeypatch, change):
    root = tmp_path / "artifact"
    path = root / "postgame" / "session-1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"test")
    clip = PostgameClip(
        "session-1",
        "shot-1",
        "a" * 64,
        "job-1",
        ClipSaveStatus.NOT_SAVED,
        "clip-id",
        "analysis",
        path,
        hashlib.sha256(b"test").hexdigest(),
        4,
        "video/mp4",
    )
    monkeypatch.delattr(os, "O_NOFOLLOW")
    if change == "symlink":
        target = path.parent / "target.mp4"
        path.rename(target)
        path.symlink_to(target)
    elif change == "junction":
        monkeypatch.setattr(Path, "is_junction", lambda self: self == path.parent, raising=False)
    elif change == "identity":
        original_open = os.open

        def replace_before_open(name, flags, *args, **kwargs):
            if Path(name) == path:
                path.rename(path.with_suffix(".old"))
                path.write_bytes(b"test")
            return original_open(name, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", replace_before_open)
    if change == "none":
        with field_server._open_review_media(root, clip) as source:
            assert source.read() == b"test"
    else:
        with pytest.raises((ValueError, OSError)):
            with field_server._open_review_media(root, clip):
                pytest.fail("unsafe portable media was opened")


def test_review_media_portable_open_rejects_parent_a_b_a_replacement(tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    folder = root / "postgame" / "session-1"
    folder.mkdir(parents=True)
    path = folder / "clip.mp4"
    path.write_bytes(b"test")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "clip.mp4").write_bytes(b"test")
    parked = tmp_path / "original-session"
    clip = PostgameClip(
        "session-1",
        "shot-1",
        "a" * 64,
        "job-1",
        ClipSaveStatus.NOT_SAVED,
        "clip-id",
        "analysis",
        path,
        hashlib.sha256(b"test").hexdigest(),
        4,
        "video/mp4",
    )
    monkeypatch.delattr(os, "O_NOFOLLOW")
    original_snapshot = field_server._portable_review_path_snapshot
    original_open = os.open
    replaced = False

    def swap_after_initial_snapshot(*args):
        nonlocal replaced
        snapshot = original_snapshot(*args)
        if not replaced:
            folder.rename(parked)
            outside.rename(folder)
            replaced = True
        return snapshot

    def restore_after_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if Path(name) == path:
            folder.rename(outside)
            parked.rename(folder)
        return descriptor

    monkeypatch.setattr(field_server, "_portable_review_path_snapshot", swap_after_initial_snapshot)
    monkeypatch.setattr(os, "open", restore_after_open)
    with pytest.raises(ValueError, match="identity changed"):
        with field_server._open_review_media(root, clip):
            pytest.fail("replacement descriptor escaped the initial validated identity")


def test_public_screen_poll_client_abort_does_not_emit_a_server_exception() -> None:
    handler = object.__new__(_FieldAssistanceHandler)
    handler.wfile = _AbortedWriter()
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    handler.request = SimpleNamespace()

    handler._send_bytes(
        HTTPStatus.OK,
        b"payload",
        content_type="application/json",
    )


def _application(revision: int) -> FieldCommandApplication:
    return FieldCommandApplication("command-1", "applied", "APPLIED", revision)


def _envelope(
    command_type: str, payload: dict[str, str], actor: str = "user-red"
) -> FieldCommandEnvelope:
    return FieldCommandEnvelope(
        command_id="command-1",
        session_id="session-1",
        actor_user_id=actor,
        command_type=command_type,
        payload=payload,
        expected_revision=4,
        server_sequence=5,
    )


def test_applied_phone_commands_have_user_visible_screen_feedback() -> None:
    assert hasattr(field_server, "_describe_applied_command")
    describe = field_server._describe_applied_command
    match = MatchControlSnapshot(
        red_user_id="user-red",
        blue_user_id="user-blue",
        breaker=WristbandRole.RED,
        active_role=WristbandRole.RED,
        red_group=BallGroupAssignment.STRIPE,
        blue_group=BallGroupAssignment.SOLID,
        revision=5,
    )

    assert (
        describe(_envelope("set_breaker", {"role": "red"}), _application(3), match)
        == "红方开球已设置"
    )
    assert (
        describe(_envelope("assign_groups", {"red_group": "stripe"}), _application(4), match)
        == "球组已分配：红方花色，蓝方全色"
    )
    assert describe(_envelope("continue_turn", {}), _application(5), match) == "红方继续击球"
    assert (
        describe(
            _envelope("switch_turn", {}),
            _application(6),
            MatchControlSnapshot(
                red_user_id="user-red",
                blue_user_id="user-blue",
                active_role=WristbandRole.BLUE,
                revision=6,
            ),
        )
        == "已切换至蓝方击球"
    )


def test_permanent_join_qr_asset_is_validated_once_and_served_as_public_bytes(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "table-code.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\npermanent-table-code")

    loaded = _load_join_qr_asset(asset)

    assert loaded.image_bytes == asset.read_bytes()
    assert loaded.content_type == "image/png"


def test_permanent_join_qr_asset_rejects_symlinks_and_unknown_images(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"not-an-image")
    link = tmp_path / "table-code.png"
    link.symlink_to(target)

    for path in (target, link):
        try:
            _load_join_qr_asset(path)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe table code image must fail closed")


def test_end_session_can_enqueue_sync_without_cloud_network(tmp_path: Path) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: datetime(2026, 8, 26, 14, 0, tzinfo=UTC),
        session_id_factory=lambda: "session-1",
        sync_job_id_factory=lambda: "sync-1",
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session("session-1")
    queued = store.enqueue_sync_job(
        session_id="session-1",
        manifest_hash="a" * 64,
        manifest_json='{"schema_version":"1.0"}',
    )

    assert store.load_session("session-1").status.value == "ended"
    assert store.load_sync_job(queued.sync_job_id).status == "pending"


def test_command_worker_persists_before_ack_and_is_restart_idempotent(
    tmp_path: Path,
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    command = FieldCommandEnvelope(
        command_id="command-1",
        session_id="session-1",
        actor_user_id="user-a",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
        server_sequence=1,
    )

    class Gateway:
        def __init__(self) -> None:
            self.completed = []

        def claim_field_command(self):
            return command

        def complete_field_command(self, application):
            self.completed.append(application)
            return application

    gateway = Gateway()
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )

    assert worker.claim_and_apply_once() is True
    restarted = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )
    assert restarted.claim_and_apply_once() is True

    assert [item.resulting_revision for item in gateway.completed] == [1, 1]
    assert store.load_session("session-1").match.revision == 1


def test_command_worker_cloud_outage_does_not_mutate_local_facts(tmp_path: Path) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")

    class OfflineGateway:
        def claim_field_command(self):
            raise OSError("offline")

    health = []
    worker = _BackgroundFieldCommandWorker(
        gateway=OfflineGateway(),
        store=store,
        match_lock=threading.RLock(),
        health_changed=health.append,
    )

    assert worker.claim_and_apply_once() is False
    assert health[-1] == "offline"
    assert store.load_session("session-1").match.revision == 0


@pytest.mark.parametrize("result", ["command", "empty", "offline"])
def test_command_shutdown_cancels_blocked_claim_without_application_or_callbacks(tmp_path, result):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    entered, release = threading.Event(), threading.Event()
    callbacks, acknowledgements = [], []
    command = FieldCommandEnvelope(
        "command-1", "session-1", "alice", "claim_role", {"role": "red"}, 0, 1
    )

    class Gateway:
        def claim_field_command(self):
            entered.set()
            assert release.wait(2)
            if result == "offline":
                raise OSError("offline after cancellation")
            return command if result == "command" else None

        def complete_field_command(self, application):
            acknowledgements.append(application)

    worker = _BackgroundFieldCommandWorker(
        gateway=Gateway(),
        store=store,
        match_lock=threading.RLock(),
        join_timeout_seconds=0.01,
        before_application=lambda item: callbacks.append("before"),
        health_changed=lambda health: callbacks.append(health),
        match_changed=lambda *args: callbacks.append("match"),
        application_changed=lambda *args: callbacks.append("application"),
    )
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="command"):
            worker.stop()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(2)
        worker.stop()
    assert not worker._thread.is_alive()
    assert callbacks == acknowledgements == []
    assert store.load_session("session-1").match.revision == 0
    # The unacknowledged lease remains replayable by a fresh worker.
    if result == "command":
        replayed = _BackgroundFieldCommandWorker(
            gateway=Gateway(), store=store, match_lock=threading.RLock()
        )
        assert replayed.claim_and_apply_once()
        assert store.load_session("session-1").match.red_user_id == "alice"
        assert len(acknowledgements) == 1


@pytest.mark.parametrize("stage", ["before", "confirmation"])
def test_command_cancelled_during_preparation_does_not_apply(tmp_path, stage):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    entered, release = threading.Event(), threading.Event()

    def prepare(envelope):
        entered.set()
        assert release.wait(2)

    worker = _BackgroundFieldCommandWorker(
        gateway=SimpleNamespace(
            claim_field_command=lambda: FieldCommandEnvelope(
                "command-1", "session-1", "alice", "claim_role", {"role": "red"}, 0, 1
            )
        ),
        store=store,
        match_lock=threading.RLock(),
        join_timeout_seconds=0.01,
        **{"before_application" if stage == "before" else "confirmation_resolver": prepare},
    )
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError):
            worker.stop()
    finally:
        release.set()
        worker._thread.join(2)
        worker.stop()
    assert store.load_session("session-1").match.revision == 0


def test_command_cancelled_during_session_read_has_no_late_callback_and_replays(
    tmp_path, monkeypatch
):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    load_session = store.load_session
    entered, release = threading.Event(), threading.Event()
    callbacks, acknowledgements = [], []
    command = FieldCommandEnvelope(
        "command-1", "session-1", "alice", "claim_role", {"role": "red"}, 0, 1
    )

    def blocked_read(session_id):
        entered.set()
        assert release.wait(2)
        return load_session(session_id)

    monkeypatch.setattr(store, "load_session", blocked_read)
    gateway = SimpleNamespace(
        claim_field_command=lambda: command,
        complete_field_command=acknowledgements.append,
    )
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway,
        store=store,
        match_lock=threading.RLock(),
        join_timeout_seconds=0.01,
        match_changed=lambda *args: callbacks.append("match"),
        application_changed=lambda *args: callbacks.append("application"),
        health_changed=lambda health: callbacks.append(health),
    )
    worker.start()
    try:
        assert entered.wait(1)
        # The command has committed; cancellation must preserve its durable ledger.
        assert load_session("session-1").match.revision == 1
        with pytest.raises(TimeoutError, match="command"):
            worker.stop()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(2)
        worker.stop()
    assert not worker._thread.is_alive()
    assert callbacks == acknowledgements == []

    monkeypatch.setattr(store, "load_session", load_session)
    replayed = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )
    assert replayed.claim_and_apply_once()
    assert store.load_session("session-1").match.revision == 1
    assert store.load_session("session-1").match.red_user_id == "alice"
    assert acknowledgements == [FieldCommandApplication("command-1", "applied", "APPLIED", 1)]


def test_command_worker_closes_a_claimed_session_missing_from_local_state(
    tmp_path: Path,
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-current"
    )
    current = store.open_or_resume_session(artifact_id="artifact-1")
    command = FieldCommandEnvelope(
        command_id="command-old-end",
        session_id="session-previous",
        actor_user_id="user-red",
        command_type="end_session",
        payload={},
        expected_revision=4,
        server_sequence=5,
    )

    class Gateway:
        def __init__(self) -> None:
            self.closed: list[str] = []
            self.completed: list[FieldCommandApplication] = []

        def claim_field_command(self):
            return command

        def close_field_session(self, session_id):
            self.closed.append(session_id)
            return SimpleNamespace(session_id=session_id, status="ended")

        def complete_field_command(self, application):
            self.completed.append(application)
            return application

    gateway = Gateway()
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )

    assert worker.claim_and_apply_once() is True
    assert gateway.closed == ["session-previous"]
    assert gateway.completed == [
        FieldCommandApplication("command-old-end", "applied", "STALE_SESSION_CLOSED", 4)
    ]
    assert store.load_session(current.session_id).match == MatchControlSnapshot()


def test_command_worker_identifies_the_session_whose_match_changed(tmp_path: Path) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-1"
    )
    store.open_or_resume_session(artifact_id="artifact-1")
    command = FieldCommandEnvelope(
        command_id="command-1",
        session_id="session-1",
        actor_user_id="user-a",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
        server_sequence=1,
    )

    class Gateway:
        def claim_field_command(self):
            return command

        def complete_field_command(self, application):
            return application

    changed: list[tuple[str, MatchControlSnapshot]] = []
    worker = _BackgroundFieldCommandWorker(
        gateway=Gateway(),
        store=store,
        match_lock=threading.RLock(),
        match_changed=lambda session_id, match: changed.append((session_id, match)),
    )

    assert worker.claim_and_apply_once() is True
    assert changed == [("session-1", store.load_session("session-1").match)]


def test_late_command_from_previous_session_cannot_change_new_session_state(
    tmp_path: Path,
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3", session_id_factory=lambda: "session-current"
    )
    store.open_or_resume_session(artifact_id="artifact-1")

    class ReplayClockProbe:
        def __init__(self) -> None:
            self.start_count = 0

        def start(self) -> None:
            self.start_count += 1

    replay_clock = ReplayClockProbe()
    state = field_server._ServerState(
        artifact_path=tmp_path,
        control_token="token",
        replay_clock=replay_clock,
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id="session-current",
        artifact_id="artifact-1",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
    )
    previous_match = MatchControlSnapshot(
        red_user_id="old-red",
        blue_user_id="old-blue",
        breaker=WristbandRole.RED,
        active_role=WristbandRole.RED,
        revision=7,
    )
    previous_envelope = FieldCommandEnvelope(
        command_id="old-command",
        session_id="session-previous",
        actor_user_id="old-red",
        command_type="set_breaker",
        payload={"role": "red"},
        expected_revision=6,
        server_sequence=8,
    )
    previous_application = FieldCommandApplication("old-command", "applied", "APPLIED", 7)

    state.set_command_match("session-previous", previous_match)
    state.handle_command_application(previous_envelope, previous_application)

    assert state.match == MatchControlSnapshot()
    assert state.last_applied_action is None
    assert replay_clock.start_count == 0


def test_recording_gate_reads_writer_pressure_outside_match_lock_without_database():
    class Lock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *args):
            self.held = False

    lock = Lock()
    capacity = [True]

    def can_encode():
        assert not lock.held
        return capacity[0]

    state = SimpleNamespace(
        match_lock=lock,
        _media_stopped=threading.Event(),
        projection_lifecycle_closing=False,
        projection_lifecycle_closed=False,
        session_policy="recording-v1",
        projection_source=SimpleNamespace(recording_media_capacity_available=can_encode),
    )
    gate = field_server._FieldMediaResourceGate(state)
    assert gate.cooperative_pause and gate.may_run_media_encoding("current-or-prior")
    capacity[0] = False
    assert not gate.may_run_media_encoding("current-or-prior")
    state._media_stopped.set()
    assert gate.cancellation_requested()
    assert not gate.may_run_media_encoding("current-or-prior")


def test_stream_revocation_stops_before_next_chunk():
    from io import BytesIO

    handler = object.__new__(_FieldAssistanceHandler)
    handler.headers = {}
    handler._send_media_headers = lambda *a, **kw: None
    allowed = [True]

    class DeleteAfterFirstWrite(BytesIO):
        def write(self, chunk):
            result = super().write(chunk)
            allowed[0] = False
            return result

    handler.wfile = DeleteAfterFirstWrite()
    handler._stream_media(
        BytesIO(b"x" * (3 * 1024 * 1024)), 3 * 1024 * 1024, continue_check=lambda: allowed[0]
    )
    assert len(handler.wfile.getvalue()) == 1024 * 1024
    assert handler.close_connection is True


def test_server_close_reaps_owned_idle_and_partial_request_connections():
    import socket
    import time

    state = SimpleNamespace(
        maintain_service=lambda: None,
        close_media_worker=lambda: None,
        close_projection=lambda: None,
        session_payload=lambda: {},
        shots_payload=lambda: {},
        report_payload=lambda: {},
    )
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    threads = []
    original = server.process_request_thread
    entered = threading.Event()

    def tracked(*args):
        threads.append(threading.current_thread())
        if len(threads) == 5:
            entered.set()
        return original(*args)

    server.process_request_thread = tracked
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    clients = []
    serving.start()
    try:
        for number in range(5):
            client = socket.create_connection(server.server_address, timeout=2)
            clients.append(client)
            if number % 2:
                client.sendall(b"GET /scr")
        assert entered.wait(2)
        server.shutdown()
        server.server_close()
        serving.join(2)
        deadline = time.monotonic() + 1
        while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(t.is_alive() for t in threads), "owned HTTP request threads survived close"
    finally:
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()
        serving.join(2)


def test_shutdown_cancels_open_descriptor_hash_and_releases_original_pin(tmp_path, monkeypatch):
    import http.client
    import time

    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from tests.unit.edge.route_assistance.test_persistence import _available_clip, _postgame_store

    store = _postgame_store(tmp_path / "field.sqlite3")
    store.bind_artifact_path("artifact-1", tmp_path)
    path = tmp_path / "postgame/session-1" / ("c" * 64) / "overlay.mp4"
    path.parent.mkdir(parents=True)
    data = b"x" * (3 * 1024**2)
    path.write_bytes(data)
    _available_clip(
        store,
        tmp_path,
        local_path=path,
        byte_size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
    state = field_server._ServerState(
        artifact_path=tmp_path,
        control_token="token",
        replay_clock=object(),
        match=MatchControlSnapshot(),
        match_lock=threading.RLock(),
        frame_store=SimpleNamespace(close=lambda: None),
        record_store=store,
        session_id="session-1",
        artifact_id="artifact-1",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
    )
    clip = store.list_postgame_clips("session-1")[0]
    entered, release = threading.Event(), threading.Event()
    actual_sha = hashlib.sha256

    class SlowHash:
        def __init__(self, *a, **kw):
            self.digest = actual_sha(*a, **kw)

        def __getattr__(self, name):
            return getattr(self.digest, name)

        def update(self, chunk):
            self.digest.update(chunk)
            if len(chunk) == 1024**2 and not entered.is_set():
                entered.set()
                assert release.wait(3)

    monkeypatch.setattr(field_server.hashlib, "sha256", SlowHash)
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)

    def request():
        try:
            connection.request(
                "GET",
                f"/media/postgame/{clip.clip_id}.mp4?manifest={clip.postgame_manifest_sha256}",
            )
            connection.getresponse().read()
        except (OSError, http.client.HTTPException):
            pass

    client = threading.Thread(target=request, daemon=True)
    client.start()
    close_errors = []

    def close():
        try:
            server.server_close()
        except BaseException as exc:
            close_errors.append(exc)

    closer = None
    try:
        assert entered.wait(2)
        server.shutdown()
        closer = threading.Thread(target=close)
        closer.start()
        deadline = time.monotonic() + 2
        while not state.projection_lifecycle_closing and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.projection_lifecycle_closing
        release.set()
        closer.join(2)
        client.join(2)
        serving.join(2)
        assert not closer.is_alive() and not close_errors and not server._request_connections
        with session_media_lock(tmp_path, "session-1", exclusive=True, timeout_seconds=0):
            pass
    finally:
        release.set()
        connection.close()
        server.shutdown()
        server.server_close()
        serving.join(2)
        if closer:
            closer.join(2)


def test_close_cancels_buffered_read_when_native_shutdown_does_not_wake_it():
    """Windows makefile readers can retain a socket until peer EOF/15s timeout."""
    import socket

    accepted, client = socket.socketpair()
    entered = threading.Event()

    class DeferredNativeClose:
        def __getattr__(self, name):
            return getattr(accepted, name)

        def makefile(self, *args, **kwargs):
            result = accepted.makefile(*args, **kwargs)
            entered.set()
            return result

        def shutdown(self, _):
            pass  # Model the observed Windows buffered reader receiving no EOF.

        def close(self):
            pass

    state = SimpleNamespace(
        close_media_worker=lambda: None,
        close_projection=lambda: None,
        session_payload=lambda: {},
        shots_payload=lambda: {},
        report_payload=lambda: {},
    )
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    request = DeferredNativeClose()
    try:
        server.process_request(request, ("127.0.0.1", 0))
        assert entered.wait(2)
        client.sendall(b"GET /scr")
        with server._request_lock:
            threads = tuple(server._request_connections.values())
        assert len(threads) == 1
        server.server_close()
        assert not any(thread.is_alive() for thread in threads)
    finally:
        client.close()
        accepted.close()
        for thread in tuple(server._request_connections.values()):
            thread.join(2)
        server.server_close()


def test_close_aborts_partial_request_body_without_dispatching_mutation():
    import socket

    waiting = threading.Event()
    mutations = []

    class Handler(field_server._FieldAssistanceHandler):
        def do_POST(self):
            waiting.set()
            body = self.rfile.read(int(self.headers["Content-Length"]))
            mutations.append(body)
            self._send_bytes(200, b"ok", content_type="text/plain")

    state = SimpleNamespace(
        maintain_service=lambda: None,
        close_media_worker=lambda: None,
        close_projection=lambda: None,
    )
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    server.RequestHandlerClass = Handler
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = socket.create_connection(server.server_address, timeout=2)
    try:
        client.sendall(b"POST /test HTTP/1.0\r\nContent-Length: 20\r\n\r\n{}")
        assert waiting.wait(2)
        server.shutdown()
        server.server_close()
        assert mutations == []
        assert not server._request_connections
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(2)


def test_normal_request_body_can_arrive_more_than_one_second_apart():
    import socket

    waiting = threading.Event()
    delivered = threading.Event()
    mutations = []

    class Handler(field_server._FieldAssistanceHandler):
        def do_POST(self):
            waiting.set()
            mutations.append(self.rfile.read(int(self.headers["Content-Length"])))
            self._send_bytes(200, b"ok", content_type="text/plain")
            delivered.set()

    state = SimpleNamespace(
        maintain_service=lambda: None,
        close_media_worker=lambda: None,
        close_projection=lambda: None,
    )
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    server.RequestHandlerClass = Handler
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = socket.create_connection(server.server_address, timeout=3)
    try:
        client.sendall(b"POST /test HTTP/1.0\r\nContent-Length: 4\r\n\r\nab")
        assert waiting.wait(2)
        assert not delivered.wait(1.1)
        client.sendall(b"cd")
        assert delivered.wait(2)
        assert mutations == [b"abcd"]
        assert b"200 OK" in client.recv(4096)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(2)


def test_request_reader_accepts_unix_socket_descriptor_above_select_limit():
    import socket

    fcntl = pytest.importorskip("fcntl")
    accepted, client = socket.socketpair()
    high = None
    try:
        high = socket.socket(fileno=fcntl.fcntl(accepted.fileno(), fcntl.F_DUPFD, 1100))
        high.settimeout(2)
        client.sendall(b"body")
        reader = field_server._ClosingRequestReader(high, threading.Event())
        try:
            data = bytearray(4)
            assert reader.readinto(data) == 4
            assert data == b"body"
        finally:
            reader.close()
    finally:
        if high is not None:
            high.close()
        accepted.close()
        client.close()


def test_close_during_native_receive_never_returns_a_truncated_json_body():
    import io
    import socket

    accepted, client = socket.socketpair()
    accepted.settimeout(15)
    closing = threading.Event()

    class CloseDuringReceive:
        calls = 0

        def __getattr__(self, name):
            return getattr(accepted, name)

        def recv_into(self, buffer):
            self.calls += 1
            if self.calls == 1:
                return accepted.recv_into(buffer, 2)
            closing.set()
            return 0  # Native receive returns EOF while server shutdown races it.

    reader = io.BufferedReader(field_server._ClosingRequestReader(CloseDuringReceive(), closing))
    try:
        client.sendall(b"{}x")
        with pytest.raises(ConnectionAbortedError):
            reader.read(20)
    finally:
        reader.close()
        client.close()
        accepted.close()


@pytest.mark.parametrize("snapshot", [None, (1, b"static-jpeg")])
def test_close_reaps_live_stream_waiting_forever_for_a_new_frame(snapshot):
    import http.client

    source = SimpleNamespace(source_epoch_id="frozen-epoch", mjpeg_snapshot=lambda: snapshot)
    state = object.__new__(field_server._LiveServerState)
    state.projection_source = source
    state.match_lock = threading.RLock()
    state.maintain_service = lambda: None
    state.close_media_worker = lambda: None
    state.close_projection = lambda: None
    server = field_server.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        client.request("GET", "/media/live.mjpg?epoch=frozen-epoch")
        response = client.getresponse()
        assert response.status == 200
        if snapshot is not None:
            part = (
                b"--billiards-frame\r\nContent-Type: image/jpeg\r\n"
                b"Content-Length: 11\r\n\r\nstatic-jpeg\r\n"
            )
            assert response.read(len(part)) == part
        with server._request_lock:
            threads = tuple(server._request_connections.values())
        assert len(threads) == 1
        server.shutdown()
        server.server_close()
        assert not any(thread.is_alive() for thread in threads)
    finally:
        state.projection_source = None  # Release the old-loop regression on failure.
        client.close()
        for thread in tuple(server._request_connections.values()):
            thread.join(2)
        server.shutdown()
        server.server_close()
        serving.join(2)
