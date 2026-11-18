"""Listener-owned assembly with real retained SQLite and signed startup RPCs."""

import hashlib
import socket
import sqlite3
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance import server as service
from tests.integration.lab.field_assistance.test_live_server import Capture
from tests.unit.lab.field_assistance.test_managed_startup import (
    BINDING,
    REQUEST,
    SCOPE,
    Remote,
    coordinator,
)


def options_at(root):
    captures = []

    def factory(session_id, channel, initial_shot_number, source_epoch_id):
        capture = Capture(session_id, channel, source_epoch_id)
        captures.append(capture)
        return capture

    return dict(
        artifact_path=root / "artifacts",
        state_db_path=root / "field.sqlite",
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "fresh-epoch",
        calibration_id="cal-1",
        runtime_factory=factory,
    ), captures


def artifact_id(path):
    return "usb-field-" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def seed(options, *, materialized=True):
    store = SQLiteFieldRecordStore(
        options["state_db_path"], managed_open_request_id_factory=lambda: REQUEST
    )
    store.reserve_managed_open(SCOPE, artifact_id=artifact_id(options["artifact_path"]))
    if materialized:
        store.record_managed_binding(SCOPE, REQUEST, BINDING)
        store.materialize_managed_session(SCOPE, REQUEST)
        options["artifact_path"].mkdir(exist_ok=True)
        store.bind_artifact_path(artifact_id(options["artifact_path"]), options["artifact_path"])
    return store


def rows(store):
    with sqlite3.connect(store.path) as connection:
        return tuple(connection.iterdump())


def test_occupied_listener_does_not_prepare_runtime_create_artifact_or_allocate(tmp_path):
    options, captures = options_at(tmp_path)
    remote = Remote()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        options["port"] = listener.getsockname()[1]
        with pytest.raises(OSError):
            service.build_live_field_assistance_server(
                **options,
                managed_startup=coordinator(remote),
                runtime_preparer=lambda: pytest.fail("prepared before listener ownership"),
            )
    assert remote.calls == []
    assert captures == []
    assert not options["artifact_path"].exists()
    assert not options["state_db_path"].exists()


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_unflagged_builder_cannot_escape_managed_occupancy(tmp_path, materialized, fresh):
    options, captures = options_at(tmp_path)
    store = seed(options, materialized=materialized)
    before = rows(store)
    with pytest.raises(ValueError, match="managed"):
        service.build_live_field_assistance_server(**options, fresh_session_on_start=fresh)
    assert rows(store) == before
    assert captures == []


def test_usb_resumes_actual_binding_with_fresh_capture_epoch_and_cloud_status(tmp_path):
    options, captures = options_at(tmp_path)
    store = seed(options)
    remote = Remote()
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(remote)
    )
    try:
        state = server.field_state
        assert state.session_id == BINDING.session_id
        assert state.managed_session_scope == SCOPE
        assert state.cloud_session_status == "open"
        assert state.resource_session_ids == ()
        assert captures[0].session_id == BINDING.session_id
        assert captures[0].source_epoch_id == "fresh-epoch"
        assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST
        assert [p for p, _ in remote.calls if p != "/v1/runtime-attestation"] == [
            "/v2/field-sessions/open"
        ]
    finally:
        server.server_close()
    assert captures[0].cancelled.is_set()


@pytest.mark.parametrize("changed", [False, True])
def test_blackbox_closes_and_retires_before_runtime_even_empty_or_changed_context(
    tmp_path, changed
):
    options, captures = options_at(tmp_path)
    store = seed(options)
    old_artifact = artifact_id(options["artifact_path"])
    if changed:
        options["artifact_path"] = tmp_path / "new-artifacts"
        options["rule_profile_id"] = PROFILE_ID
    remote = Remote()
    original = options["runtime_factory"]

    def factory(*args):
        assert store.get_latest_retired_managed_open_request(SCOPE).binding.status == "ended"
        assert store.load_session(BINDING.session_id).status.value == "ended"
        return original(*args)

    options["runtime_factory"] = factory
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(remote), fresh_session_on_start=True
    )
    try:
        assert server.field_state.session_id != BINDING.session_id
        assert server.field_state.resource_session_ids == (server.field_state.session_id,)
        assert captures[0].session_id == server.field_state.session_id
        assert store.load_session(BINDING.session_id).artifact_id == old_artifact
        assert store.get_latest_retired_managed_open_request(SCOPE).open_request_id == REQUEST
        review = server.field_state.postgame_review_payload(kiosk=True)
        assert review["review_session_id"] == BINDING.session_id
        server.field_state.mutate_postgame_review(
            dict(
                session_id=BINDING.session_id,
                postgame_manifest_sha256=review["postgame_manifest_sha256"],
            ),
            review["kiosk_nonce"],
            complete=True,
        )
        assert store.active_postgame_review(old_artifact) is None
    finally:
        server.server_close()


def test_preparer_failure_preserves_materialized_birth_and_listener_can_be_reused(tmp_path):
    options, captures = options_at(tmp_path)
    store = seed(options, materialized=False)
    remote = Remote()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        options["port"] = listener.getsockname()[1]

    def fail_prepare():
        assert store.get_active_managed_open_request(SCOPE).phase == "materialized"
        assert store.load_session(BINDING.session_id).status.value == "open"
        raise RuntimeError("model preparation failed")

    with pytest.raises(RuntimeError, match="model preparation"):
        service.build_live_field_assistance_server(
            **options,
            managed_startup=coordinator(remote),
            runtime_preparer=fail_prepare,
        )
    assert captures == []
    assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(remote)
    )
    try:
        assert server.field_state.session_id == BINDING.session_id
        assert [
            body["open_request_id"]
            for path, body in remote.calls
            if path == "/v2/field-sessions/open"
        ] == [REQUEST, REQUEST]
    finally:
        server.server_close()


def test_prepared_backend_is_cancelled_if_source_construction_fails(tmp_path):
    from types import SimpleNamespace

    options, _ = options_at(tmp_path)
    seed(options, materialized=False)
    cancelled = []

    def fail_runtime(*_args):
        raise RuntimeError("source construction failed")

    with pytest.raises(RuntimeError, match="source construction"):
        service.build_live_field_assistance_server(
            **options,
            managed_startup=coordinator(Remote()),
            runtime_preparer=lambda: (
                "cal",
                fail_runtime,
                SimpleNamespace(cancel=lambda: cancelled.append(True)),
            ),
        )
    assert cancelled == [True]


@pytest.mark.parametrize("settled", [False, True])
def test_blackbox_preserves_settled_or_interrupted_shot_before_fresh_runtime(tmp_path, settled):
    from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
    from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
    from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot

    options, captures = options_at(tmp_path)
    store = seed(options)
    store.save_match(
        BINDING.session_id,
        MatchControlSnapshot(
            red_user_id="old-red",
            blue_user_id="old-blue",
            active_role=WristbandRole.RED,
            breaker=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    match = LiveMatchCoordinator(
        store=store,
        session_id=BINDING.session_id,
        source_epoch_id="old-epoch",
        calibration_id="cal",
    )
    match.shot_started(shot_id="shot-0001", started_ms=100)
    if settled:
        match.settle(
            LiveSettledShot(
                "shot-0001", 1, 10, 200_000_000, (), False, "unknown", {}, "old-epoch", "old-job"
            ),
            facts=(),
        )
    shots = store.list_shots(BINDING.session_id)
    owner = store.get_shot_ownership(BINDING.session_id, "shot-0001")
    server = service.build_live_field_assistance_server(
        **options,
        managed_startup=coordinator(Remote()),
        fresh_session_on_start=True,
    )
    try:
        assert captures[0].session_id != BINDING.session_id
        assert store.list_shots(BINDING.session_id) == shots
        assert store.get_shot_ownership(BINDING.session_id, "shot-0001") == owner
        assert store.list_shots(server.field_state.session_id) == ()
    finally:
        server.server_close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_bootstrap_timeout_or_cancellation_releases_listener_without_preparation(
    tmp_path, cancelled
):
    from functools import partial
    from types import SimpleNamespace

    options, captures = options_at(tmp_path)
    store = seed(options, materialized=False)
    remote = Remote()
    ticks, stop = [0.0], [False]
    startup = coordinator(remote, monotonic=lambda: ticks[0])

    def expire(_path):
        if cancelled:
            stop[0] = True
        else:
            ticks[0] = 61.0

    remote.hook = expire
    startup = SimpleNamespace(
        scope=SCOPE, prepare=partial(startup.prepare, cancelled=lambda: stop[0])
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        options["port"] = listener.getsockname()[1]
    with pytest.raises(OSError, match="cancelled|deadline"):
        service.build_live_field_assistance_server(
            **options,
            managed_startup=startup,
            runtime_preparer=lambda: pytest.fail("expired startup prepared capture"),
        )
    assert captures == []
    assert store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST
    assert store.get_active_managed_open_request(SCOPE).phase == "pending"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", options["port"]))


def test_lost_allocation_response_keeps_uuid_and_releases_listener_without_capture(tmp_path):
    options, captures = options_at(tmp_path)
    store = seed(options, materialized=False)
    remote = Remote()
    remote.lost = "/v2/field-sessions/open"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        options["port"] = listener.getsockname()[1]
    with pytest.raises(OSError, match="response lost"):
        service.build_live_field_assistance_server(**options, managed_startup=coordinator(remote))
    assert captures == []
    assert store.get_active_managed_open_request(SCOPE).phase == "pending"
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(remote), fresh_session_on_start=True
    )
    try:
        assert server.field_state.session_id == BINDING.session_id
        assert [
            body["open_request_id"]
            for path, body in remote.calls
            if path == "/v2/field-sessions/open"
        ] == [REQUEST, REQUEST]
    finally:
        server.server_close()


def test_managed_blackbox_never_recovers_frame_zero_under_the_old_session(tmp_path):
    options, captures = options_at(tmp_path)
    seed(options, materialized=False)
    server = service.build_live_field_assistance_server(
        **options, managed_startup=coordinator(Remote()), fresh_session_on_start=True
    )
    try:
        source = server.field_state.projection_source
        source._restart_reason = "finite EOF"
        assert source.restart_required
        server.field_state.maintain_service()
        assert len(captures) == 1
        assert server.field_state.projection_source is source
    finally:
        server.server_close()


def test_foreign_and_unknown_reviews_are_excluded_from_managed_scope(tmp_path):
    options, _ = options_at(tmp_path)
    store = seed(options)
    store.end_session(BINDING.session_id)
    # The original valid request belongs to another table, so it cannot authorize review.
    other = replace(SCOPE, table_id="table-other")
    options["artifact_path"] = tmp_path / "other-artifacts"
    remote = Remote()
    binding = replace(BINDING, table_id=other.table_id)
    from billiards_ai.lab.field_assistance.managed_startup import ManagedStartupCoordinator

    gateway = type("Gateway", (), {"open_managed_session": lambda self, **_: binding})()
    startup = ManagedStartupCoordinator(scope=other, gateway_factory=lambda **_: gateway)
    # Separate birth to keep permanent uniqueness and prove scope rather than artifact filtering.
    from tests.unit.lab.field_assistance.test_managed_startup import NEXT_BIRTH

    binding = replace(binding, session_id=f"managed-{NEXT_BIRTH}", birth_id=NEXT_BIRTH)
    server = service.build_live_field_assistance_server(**options, managed_startup=startup)
    try:
        assert server.field_state.postgame_review_payload()["review_session_id"] is None
        assert store.active_postgame_review(artifact_id(tmp_path / "artifacts")) is not None
        assert remote.calls == []
    finally:
        server.server_close()
