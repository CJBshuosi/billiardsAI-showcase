"""Durable managed startup with real resource inspection, SQLite and signed HTTP.

Model/native execution is replaced only at the injected runtime boundary here.
"""

import shutil
import socket
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.field_assistance.runtime_resources import inspect_managed_runtime_resources
from tests.integration.lab.field_assistance.test_live_server import Capture
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.lab.field_assistance.test_managed_startup import (
    BINDING,
    NEXT_REQUEST,
    REQUEST,
    SCOPE,
    Remote,
    coordinator,
)

resource_assets_fixture = resource_fixtures.assets


@pytest.fixture
def context(resource_assets_fixture):
    assets = resource_assets_fixture
    spec = inspect_managed_runtime_resources(assets, rule_profile_id=None)
    request_ids = iter([REQUEST, NEXT_REQUEST])
    store = SQLiteFieldRecordStore(
        assets.artifact_root.parent / "field.sqlite",
        managed_open_request_id_factory=lambda: next(request_ids),
        session_id_factory=lambda: pytest.fail("local identity escape"),
    )
    captures = []

    def runtime(session_id, channel, initial_shot_number, source_epoch_id):
        # Registration must be durable before projection constructs its runtime.
        epoch = store.get_managed_capture_epoch(SCOPE, session_id, source_epoch_id)
        assert epoch.spec_sha256 == spec.spec_sha256
        capture = Capture(session_id, channel, source_epoch_id)
        captures.append(capture)
        return capture

    runtime.managed_resource_spec_sha256 = spec.spec_sha256
    options = dict(
        artifact_path=assets.artifact_root,
        state_db_path=store.path,
        host="127.0.0.1",
        port=0,
        control_token="secret",
        source_epoch_factory=lambda: "epoch-1",
        calibration_id=None,
        runtime_factory=None,
        runtime_resource_inspector=lambda: inspect_managed_runtime_resources(
            assets, rule_profile_id=None
        ),
        runtime_preparer=lambda *, resource_spec: ("cal", runtime, None),
    )
    return SimpleNamespace(
        args=assets, spec=spec, store=store, captures=captures, runtime=runtime, options=options
    )


def reserve(ctx, *, materialized=False, with_spec=True):
    ctx.store.reserve_managed_open(
        SCOPE,
        artifact_id=ctx.spec.artifact_id,
        resource_spec=ctx.spec if with_spec else None,
    )
    if materialized:
        ctx.store.record_managed_binding(SCOPE, REQUEST, BINDING)
        ctx.store.materialize_managed_session(SCOPE, REQUEST)


def build(ctx, remote=None, **changes):
    return service.build_live_field_assistance_server(
        **(ctx.options | changes), managed_startup=coordinator(remote or Remote())
    )


def test_occupied_listener_never_inspects_or_transitions(context):
    ctx = context
    before = ctx.store.get_active_managed_open_request(SCOPE)
    remote = Remote()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            build(
                ctx,
                remote,
                port=listener.getsockname()[1],
                runtime_resource_inspector=lambda: pytest.fail("inspection before ownership"),
            )
    assert ctx.store.get_active_managed_open_request(SCOPE) == before
    assert remote.calls == [] and ctx.captures == []


@pytest.mark.parametrize("phase", ["pending", "bound", "materialized"])
@pytest.mark.parametrize("missing", [False, True])
def test_pending_and_same_scene_drift_refuse_before_rpc_or_prepare(context, phase, missing):
    ctx = context
    reserve(ctx, materialized=phase == "materialized", with_spec=not missing)
    if phase == "bound":
        ctx.store.record_managed_binding(SCOPE, REQUEST, BINDING)
    old = ctx.store.get_active_managed_open_request(SCOPE)
    remote = Remote()
    if not missing:
        ctx.args.batch_size += 1
    with pytest.raises(ValueError):
        build(ctx, remote, runtime_preparer=lambda **_: pytest.fail("prepared mismatched scene"))
    assert remote.calls == [] and ctx.captures == []
    assert ctx.store.get_active_managed_open_request(SCOPE) == old
    if phase == "materialized":
        assert ctx.store.load_session(BINDING.session_id).status.value == "open"
    if not missing:
        assert ctx.store.get_managed_resource_spec(SCOPE, REQUEST).spec == ctx.spec


@pytest.mark.parametrize("failure", ["preparation", "response"])
def test_original_intent_and_uuid_survive_failure_then_exact_retry(context, failure):
    ctx = context
    reserve(ctx)
    remote = Remote()
    options = {}
    if failure == "preparation":

        def fail(**_):
            raise RuntimeError("model preparation failed")

        options["runtime_preparer"] = fail
    else:
        remote.lost = "/v2/field-sessions/open"
    with pytest.raises((RuntimeError, OSError)):
        build(ctx, remote, **options)
    assert ctx.captures == []
    original = ctx.store.get_managed_resource_spec(SCOPE, REQUEST)
    assert original.spec == ctx.spec
    server = build(ctx, remote)
    try:
        assert server.field_state.session_id == BINDING.session_id
        assert server.field_state.managed_resource_spec_record == original
        assert ctx.store.get_active_managed_open_request(SCOPE).open_request_id == REQUEST
        assert [
            body["open_request_id"]
            for path, body in remote.calls
            if path == "/v2/field-sessions/open"
        ] == [REQUEST, REQUEST]
        assert ctx.store.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1")
        assert len(ctx.captures) == 1
    finally:
        server.server_close()


def test_usb_same_scene_restarts_and_automatic_recovery_register_fresh_epochs(context):
    ctx = context
    reserve(ctx, materialized=True)
    server = build(ctx)
    server.server_close()
    epochs = iter(["epoch-2", "epoch-3"])
    server = build(ctx, source_epoch_factory=lambda: next(epochs))
    try:
        state = server.field_state
        state.projection_source._restart_reason = "capture fault"
        state.maintain_service()
        assert [c.source_epoch_id for c in ctx.captures] == ["epoch-1", "epoch-2", "epoch-3"]
        assert state.session_id == BINDING.session_id
        for epoch in ("epoch-1", "epoch-2", "epoch-3"):
            assert ctx.store.get_managed_capture_epoch(SCOPE, BINDING.session_id, epoch)
    finally:
        server.server_close()


def test_epoch_commit_loss_consumes_epoch_before_any_capture(context, monkeypatch):
    ctx = context
    reserve(ctx)
    register = SQLiteFieldRecordStore.register_managed_capture_epoch

    def lost(store, *args):
        register(store, *args)
        raise OSError("lost epoch commit return")

    with monkeypatch.context() as patch:
        patch.setattr(SQLiteFieldRecordStore, "register_managed_capture_epoch", lost)
        with pytest.raises(OSError, match="epoch commit"):
            build(ctx)
    assert ctx.store.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1")
    with pytest.raises(ValueError):
        build(ctx)
    assert ctx.captures == []
    server = build(ctx, source_epoch_factory=lambda: "epoch-2")
    server.server_close()
    assert len(ctx.captures) == 1


@pytest.mark.parametrize("tag", ["wrong", "drift"])
def test_postpreparation_validation_cancels_owned_source_before_epoch(context, tag):
    ctx = context
    reserve(ctx)
    cancelled = []

    def prepare(*, resource_spec):
        assert resource_spec == ctx.spec
        if tag == "wrong":
            ctx.runtime.managed_resource_spec_sha256 = "0" * 64
        else:
            ctx.args.config.write_text(ctx.args.config.read_text() + "\n# changed")
        return "cal", ctx.runtime, SimpleNamespace(cancel=lambda: cancelled.append(True))

    with pytest.raises(ValueError):
        build(ctx, runtime_preparer=prepare)
    assert cancelled == [True] and ctx.captures == []
    with pytest.raises(ValueError):
        ctx.store.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1")
    assert ctx.store.get_managed_resource_spec(SCOPE, REQUEST).spec == ctx.spec


@pytest.mark.parametrize("with_spec", [False, True])
def test_blackbox_new_resources_close_old_without_requiring_old_models(context, with_spec):
    ctx = context
    reserve(ctx, materialized=True, with_spec=with_spec)
    old_detector = ctx.args.detector_training_run
    ctx.args.detector_training_run = old_detector.parent / "new-models" / old_detector.name
    shutil.copytree(old_detector, ctx.args.detector_training_run)
    (old_detector / "checkpoint_best.pth").unlink()
    ctx.args.batch_size += 1
    new = inspect_managed_runtime_resources(ctx.args, rule_profile_id=None)
    ctx.runtime.managed_resource_spec_sha256 = new.spec_sha256

    def runtime(session_id, channel, initial_shot_number, source_epoch_id):
        epoch = ctx.store.get_managed_capture_epoch(SCOPE, session_id, source_epoch_id)
        assert epoch.spec_sha256 == new.spec_sha256
        assert ctx.store.load_session(BINDING.session_id).status.value == "ended"
        return Capture(session_id, channel, source_epoch_id)

    runtime.managed_resource_spec_sha256 = new.spec_sha256
    server = build(
        ctx,
        fresh_session_on_start=True,
        runtime_preparer=lambda *, resource_spec: ("cal", runtime, None),
    )
    try:
        assert server.field_state.session_id != BINDING.session_id
        request = ctx.store.get_active_managed_open_request(SCOPE)
        assert ctx.store.get_managed_resource_spec(SCOPE, request.open_request_id).spec == new
        if with_spec:
            assert ctx.store.get_managed_resource_spec(SCOPE, REQUEST).spec == ctx.spec
        else:
            with pytest.raises(ValueError):
                ctx.store.get_managed_resource_spec(SCOPE, REQUEST)
    finally:
        server.server_close()


@pytest.mark.parametrize("ended", [False, True])
def test_usb_capture_free_closure_does_not_require_old_spec(context, ended):
    ctx = context
    reserve(ctx, materialized=True, with_spec=False)
    if ended:
        ctx.store.end_session(BINDING.session_id)
    else:
        from dataclasses import replace

        from tests.unit.edge.route_assistance.test_turn_ownership import shot

        start = shot()
        ctx.store.record_shot_start(
            replace(
                start,
                session_id=BINDING.session_id,
                media=replace(start.media, artifact_id=ctx.spec.artifact_id),
            )
        )
    # An ended or interrupted scene cannot resume capture; normal lifecycle closes it.
    remote = Remote()
    result = coordinator(remote).prepare(
        ctx.store,
        artifact_id=ctx.spec.artifact_id,
        rule_profile_id=None,
        policy="usb_resume",
        resource_spec=ctx.spec,
    )
    assert result.session_id != BINDING.session_id
    assert ctx.store.load_session(BINDING.session_id).status.value == "ended"
    assert ctx.store.get_latest_retired_managed_open_request(SCOPE).open_request_id == REQUEST


def test_projection_constructor_failure_cancels_prepared_source_and_consumes_epoch(context):
    ctx = context
    reserve(ctx)
    cancelled = []

    def fail_runtime(*_):
        # Constructor failure is after registration, before starting any product source.
        assert ctx.store.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1")
        raise RuntimeError("projection runtime construction failed")

    fail_runtime.managed_resource_spec_sha256 = ctx.spec.spec_sha256
    with pytest.raises(RuntimeError, match="projection runtime"):
        build(
            ctx,
            runtime_preparer=lambda *, resource_spec: (
                "cal",
                fail_runtime,
                SimpleNamespace(cancel=lambda: cancelled.append(True)),
            ),
        )
    assert cancelled == [True] and ctx.captures == []
    with pytest.raises(ValueError):
        build(ctx)
    server = build(ctx, source_epoch_factory=lambda: "epoch-2")
    server.server_close()
