from dataclasses import replace
from types import SimpleNamespace

import pytest


def _args(tmp_path, **overrides):
    values = dict(
        sync_base_url="https://cloud-development.example/cloud-development/sync-api",
        device_id="raytheon-01",
        credential_target="billiardsAI/development/raytheon-01",
        runtime_environment="development",
        cloudbase_environment_id="cloud-development",
        expected_development_environment_id="cloud-development",
        existing_development_control_only=False,
        observation_only=False,
        artifact_root=tmp_path,
        jobs_root=tmp_path / "jobs",
    )
    return SimpleNamespace(**(values | overrides))


def _dependencies(*, failure=None, stop_failure=None):
    from billiards_ai.lab.field_assistance.cloud_runtime import FieldCloudFactories

    calls = []
    workers = {}

    def credentials(**kwargs):
        calls.append(("credentials", kwargs))
        return object()

    def transport(url, **kwargs):
        assert kwargs == {"allow_redirects": False}
        calls.append(("transport", url))
        return object()

    class Signed:
        def __init__(self, **kwargs):
            calls.append("signed")

        def runtime_attestation(self):
            calls.append("attest")
            return SimpleNamespace(
                runtime_environment="development", cloudbase_environment_id="cloud-development"
            )

    class Worker:
        def __init__(self, kind, **kwargs):
            calls.append((kind, "construct"))
            if failure == (kind, "construct"):
                raise RuntimeError("construction failed")
            self.kind = kind
            self.kwargs = kwargs
            workers[kind] = self

        def start(self):
            calls.append((self.kind, "start"))
            if failure == (self.kind, "start"):
                raise RuntimeError("start failed")

        def stop(self):
            calls.append((self.kind, "stop"))
            if stop_failure == self.kind:
                raise RuntimeError("stop failed")

    factories = FieldCloudFactories(
        credentials=credentials,
        transport=transport,
        signed_gateway=Signed,
        sync_client=lambda **kwargs: SimpleNamespace(**kwargs),
        postgame_worker=lambda **kwargs: Worker("postgame", **kwargs),
        saved_media_worker=lambda **kwargs: Worker("saved", **kwargs),
        upload_receipt_worker=lambda **kwargs: Worker("receipts", **kwargs),
        local_cleanup_worker=lambda **kwargs: Worker("cleanup", **kwargs),
        owner_deletion_worker=lambda **kwargs: Worker("owner_deletion", **kwargs),
        owner_deletion_transport=lambda url: object(),
        command_worker=lambda **kwargs: Worker("commands", **kwargs),
        live_fact_worker=lambda **kwargs: Worker("facts", **kwargs),
    )
    return factories, calls, workers


def _server():
    return SimpleNamespace(
        field_state=SimpleNamespace(
            record_store=object(),
            match_lock=object(),
            before_field_command=object(),
            set_command_match=object(),
            handle_command_application=object(),
            set_cloud_control_status=object(),
            maintain_background=object(),
            resolve_command_confirmation=object(),
            reduce_live_command=object(),
        )
    )


def test_preparation_attests_before_constructing_workers_and_wakeup_only_sets_event(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)

    assert calls == [
        (
            "credentials",
            {"device_id": "raytheon-01", "target_name": "billiardsAI/development/raytheon-01"},
        ),
        ("transport", "https://cloud-development.example/cloud-development/sync-api"),
        "signed",
        "attest",
    ]
    assert workers == {}
    assert runtime.gateway is not None
    runtime.sync_wakeup()
    assert runtime.sync_wakeup.__self__.is_set()
    runtime.stop()


def test_recording_start_constructs_media_supervisor_without_invoking_analysis_command(
    tmp_path,
):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()

    def unrequested(**kwargs):
        pytest.fail("recording mode requested an analysis/model factory")

    runtime = prepare_cloud_runtime(
        _args(tmp_path, session_policy="recording-v1"),
        factories=factories,
    )
    try:
        runtime.start(_server(), postgame_command_factory=unrequested)
        assert set(workers) == {"postgame", "cleanup", "saved", "commands", "facts"}
        assert workers["postgame"].kwargs["command_factory"] is unrequested
        assert all((kind, "start") in calls for kind in workers)
    finally:
        runtime.stop()
    assert all((kind, "stop") in calls for kind in workers)


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_id": None},
        {"expected_development_environment_id": None},
        {"runtime_environment": "production"},
        {"cloudbase_environment_id": "production"},
        {"credential_target": "billiardsAI/development/another-device"},
        {"sync_base_url": "http://cloud-development.invalid"},
        {"existing_development_control_only": True},
    ],
)
def test_invalid_scope_fails_before_credentials_transport_or_workers(tmp_path, overrides):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, _ = _dependencies()
    with pytest.raises(ValueError):
        prepare_cloud_runtime(_args(tmp_path, **overrides), factories=factories)
    assert calls == []


def test_wrong_authenticated_environment_fails_before_any_worker(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    signed = SimpleNamespace(
        runtime_attestation=lambda: SimpleNamespace(
            runtime_environment="production", cloudbase_environment_id="cloud-development"
        )
    )
    factories = replace(factories, signed_gateway=lambda **kwargs: signed)
    with pytest.raises(OSError, match="development"):
        prepare_cloud_runtime(_args(tmp_path), factories=factories)
    assert workers == {}
    assert len(calls) == 2


def test_workers_share_server_dependencies_and_shutdown_order_is_idempotent(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)
    server = _server()

    def command(session_id):
        return [session_id]

    runtime.start(server, postgame_command_factory=command)
    runtime.start(server, postgame_command_factory=command)
    assert workers["commands"].kwargs["command_reducer"] is server.field_state.reduce_live_command
    assert workers["postgame"].kwargs["command_factory"] is command
    assert workers["postgame"].kwargs["exclude_managed"] is True
    assert workers["cleanup"].kwargs == {
        "store": server.field_state.record_store,
        "jobs_db": tmp_path / "jobs" / "analysis-jobs.sqlite3",
        "artifact_root": tmp_path,
        "jobs_root": tmp_path / "jobs",
        "exclude_managed": True,
    }
    saved = workers["saved"].kwargs
    assert saved["store"] is server.field_state.record_store
    assert saved["trusted_postgame_root"] == tmp_path / "postgame"
    assert saved["exclude_managed"] is True
    assert saved["wake_event"] is runtime.sync_wakeup.__self__
    assert saved["client"].before_mutation.__self__ is runtime.gateway
    runtime.stop()
    runtime.stop()
    assert [
        event for event in calls if isinstance(event, tuple) and event[1] in ("start", "stop")
    ] == [
        ("postgame", "start"),
        ("cleanup", "start"),
        ("saved", "start"),
        ("commands", "start"),
        ("facts", "start"),
        ("commands", "stop"),
        ("facts", "stop"),
        ("postgame", "stop"),
        ("saved", "stop"),
        ("cleanup", "stop"),
    ]


@pytest.mark.parametrize("kind", ["postgame", "cleanup", "saved", "commands", "facts"])
@pytest.mark.parametrize("stage", ["construct", "start"])
def test_partial_start_stops_every_constructed_worker_once(tmp_path, kind, stage):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies(failure=(kind, stage))
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)
    with pytest.raises(RuntimeError, match="failed"):
        runtime.start(_server(), postgame_command_factory=lambda session_id: [session_id])
    runtime.stop()
    assert [event for event in calls if isinstance(event, tuple) and event[1] == "stop"] == [
        (name, "stop")
        for name in ("commands", "facts", "postgame", "saved", "cleanup")
        if name in workers
    ]


def test_stop_failure_does_not_skip_remaining_resources(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, _ = _dependencies(stop_failure="commands")
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)
    runtime.start(_server(), postgame_command_factory=lambda session_id: [session_id])
    with pytest.raises(RuntimeError, match="stop failed"):
        runtime.stop()
    with pytest.raises(RuntimeError, match="stop failed"):
        runtime.stop()
    assert list(runtime._workers) == ["commands"]
    workers = runtime._workers
    workers["commands"].stop = lambda: calls.append(("commands", "recovered"))
    runtime.stop()
    assert workers == {}
    assert calls[-2:] == [("commands", "stop"), ("commands", "recovered")]
    assert calls[-7:-2] == [
        ("commands", "stop"),
        ("facts", "stop"),
        ("postgame", "stop"),
        ("saved", "stop"),
        ("cleanup", "stop"),
    ]


@pytest.mark.parametrize("runtime_environment", [None, "development"])
def test_local_mode_constructs_only_postgame_and_needs_no_credentials(
    tmp_path, runtime_environment
):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    runtime = prepare_cloud_runtime(
        _args(
            tmp_path,
            sync_base_url=None,
            device_id=None,
            credential_target=None,
            runtime_environment=runtime_environment,
            cloudbase_environment_id=None,
            expected_development_environment_id=None,
        ),
        factories=factories,
    )
    assert runtime.gateway is None
    assert runtime.sync_wakeup is None
    assert calls == []
    runtime.start(_server(), postgame_command_factory=lambda session_id: [session_id])
    runtime.stop()
    assert list(workers) == ["postgame", "cleanup"]


def test_cleanup_worker_shutdown_retains_ownership_until_running_attempt_exits(tmp_path):
    import threading

    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker

    entered, release = threading.Event(), threading.Event()

    def process(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        assert kwargs["cancelled"]()
        return SimpleNamespace(outcome="retry")

    worker = _BackgroundLocalMediaCleanupWorker(
        store=object(),
        jobs_db=tmp_path / "jobs.db",
        artifact_root=tmp_path,
        jobs_root=tmp_path,
        process_once=process,
        scheduler=SimpleNamespace(
            next_candidate=lambda now: SimpleNamespace(kind="discard", session_id="synthetic")
        ),
        join_timeout_seconds=0.001,
    )
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="cleanup"):
            worker.stop()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(2)
    worker.stop()
    assert not worker._thread.is_alive()


def test_cleanup_worker_idle_shutdown_and_partial_thread_start_are_safe(tmp_path, monkeypatch):
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker

    def create():
        return _BackgroundLocalMediaCleanupWorker(
            store=object(),
            jobs_db=tmp_path / "jobs.db",
            artifact_root=tmp_path,
            jobs_root=tmp_path,
            process_once=lambda *a, **kw: SimpleNamespace(outcome="idle"),
            poll_seconds=60,
        )

    worker = create()
    worker.start()
    worker.stop()
    assert not worker._thread.is_alive()
    worker = create()
    monkeypatch.setattr(
        worker._thread, "start", lambda: (_ for _ in ()).throw(RuntimeError("start"))
    )
    with pytest.raises(RuntimeError):
        worker.start()
    worker.stop()


def test_usb_scope_policy_rejects_local_development_assertion_before_side_effects(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, _ = _dependencies()
    with pytest.raises(ValueError, match="without sync"):
        prepare_cloud_runtime(
            _args(
                tmp_path,
                sync_base_url=None,
                device_id=None,
                credential_target=None,
                cloudbase_environment_id=None,
                expected_development_environment_id=None,
            ),
            factories=factories,
            allow_local_development_environment=False,
        )
    assert calls == []


@pytest.mark.parametrize("kind", ["postgame", "saved", "facts"])
def test_worker_stop_is_safe_when_native_thread_start_fails(tmp_path, monkeypatch, kind):
    import threading

    from billiards_ai.lab.field_assistance.cloud_runtime import (
        _BackgroundLiveShotFactSyncWorker,
        _BackgroundPostgameAnalysisWorker,
        _BackgroundSavedMediaSyncWorker,
    )

    if kind == "postgame":
        worker = _BackgroundPostgameAnalysisWorker(store=object(), command_factory=lambda _: [])
    elif kind == "saved":
        worker = _BackgroundSavedMediaSyncWorker(
            store=object(),
            client=object(),
            trusted_postgame_root=tmp_path,
            wake_event=threading.Event(),
        )
    else:
        worker = _BackgroundLiveShotFactSyncWorker(store=object(), client=object())

    def fail():
        raise RuntimeError("native thread unavailable")

    monkeypatch.setattr(worker._thread, "start", fail)
    with pytest.raises(RuntimeError, match="native thread"):
        worker.start()
    worker.stop()


@pytest.mark.parametrize("kind", ["facts", "saved"])
def test_sync_shutdown_reports_live_thread_until_blocked_request_returns(tmp_path, kind):
    import threading

    from billiards_ai.lab.field_assistance.cloud_runtime import (
        _BackgroundLiveShotFactSyncWorker,
        _BackgroundSavedMediaSyncWorker,
    )

    entered, release = threading.Event(), threading.Event()
    callbacks = []
    item = SimpleNamespace(
        idempotency_key="fact-key", clip_id="clip", claim_token="token", session_id="legacy-session"
    )

    def request(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(outcome="complete")

    store = SimpleNamespace(
        claim_due_live_shot_fact=lambda now: item,
        acknowledge_live_shot_fact=lambda *args: callbacks.append("ack"),
        retry_live_shot_fact=lambda *args: callbacks.append("retry"),
        claim_due_saved_media=lambda *args, **kwargs: item,
        has_managed_session_identity=lambda session_id: False,
        is_saved_media_claim_current=lambda clip_id, claim_token: True,
        acknowledge_saved_media=lambda *args: callbacks.append("ack"),
    )
    client = SimpleNamespace(run_live_shot_fact_once=request, run_saved_media_once=request)
    if kind == "facts":
        worker = _BackgroundLiveShotFactSyncWorker(
            store=store, client=client, join_timeout_seconds=0.01
        )
    else:
        worker = _BackgroundSavedMediaSyncWorker(
            store=store,
            client=client,
            trusted_postgame_root=tmp_path,
            wake_event=threading.Event(),
            join_timeout_seconds=0.01,
        )
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="sync"):
            worker.stop()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(2)
        worker.stop()
    assert not worker._thread.is_alive()
    assert callbacks == []


def test_postgame_shutdown_reaps_process_returned_by_inflight_launcher():
    import threading

    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundPostgameAnalysisWorker

    entered, release = threading.Event(), threading.Event()
    terminated, errors = [], []

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            terminated.append(True)
            self.returncode = 0

        def wait(self, *, timeout):
            return self.returncode

    def launch(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return Process()

    worker = _BackgroundPostgameAnalysisWorker(
        store=SimpleNamespace(
            list_ended_sessions_requiring_postgame=lambda: [
                SimpleNamespace(session_id="ended-session")
            ]
        ),
        command_factory=lambda session: [session],
        process_factory=launch,
    )

    def stop():
        try:
            worker.stop()
        except Exception as exc:
            errors.append(exc)

    worker.start()
    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(1)
        stopper.start()
        assert worker._stopped.wait(1)
    finally:
        release.set()
        stopper.join(2)
        worker._thread.join(2)
    assert not stopper.is_alive() and not worker._thread.is_alive()
    assert errors == []
    assert terminated == [True]


def test_recording_only_supervisor_stops_after_successful_media_child():
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundPostgameAnalysisWorker

    launches = []
    store = SimpleNamespace(
        list_ended_sessions_requiring_postgame=lambda: [
            SimpleNamespace(session_id="recording-session")
        ],
        get_session_policy=lambda session_id: (
            "recording-v1" if session_id == "recording-session" else "legacy"
        ),
    )
    worker = _BackgroundPostgameAnalysisWorker(
        store=store,
        command_factory=lambda session_id, *, purpose: [session_id, purpose],
        process_factory=lambda command, **_kwargs: (
            launches.append(command) or SimpleNamespace(poll=lambda: 0)
        ),
    )

    worker.run_once()
    worker.run_once()
    worker.run_once()

    assert launches == [["recording-session", "media"]]


@pytest.mark.parametrize("source", ["blackbox", "usb"])
@pytest.mark.parametrize("times_out", [False, True])
def test_cli_keeps_capture_open_until_blocked_command_worker_terminates(
    tmp_path, monkeypatch, source, times_out, capsys
):
    import importlib
    import threading

    from billiards_ai.edge.route_assistance.commands import (
        FieldCommandEnvelope,
        reduce_field_command,
    )
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab.field_assistance.cloud_runtime import (
        FieldCloudFactories,
        FieldCloudRuntime,
    )
    from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker

    cli = importlib.import_module(f"scripts.algorithm.serve_mainline1_{source}_field")
    args = _args(
        tmp_path,
        rule_profile=None,
        identity_gate_status="PASS",
        shutdown_stdin=False,
        state_db=tmp_path / "field.sqlite3",
        host="127.0.0.1",
        port=0,
        join_qr_asset=None,
    )
    entered, release = threading.Event(), threading.Event()
    callbacks, closes, commands = [], [], []
    store = SQLiteFieldRecordStore(args.state_db, session_id_factory=lambda: "session-1")
    store.open_or_resume_session(artifact_id="artifact-1")

    class Gateway:
        def runtime_attestation(self):
            return None

        def claim_field_command(self):
            entered.set()
            assert release.wait(2)
            return FieldCommandEnvelope(
                "command-1", "session-1", "alice", "claim_role", {"role": "red"}, 0, 1
            )

        def complete_field_command(self, application):
            callbacks.append("ack")

    def command_worker(**kwargs):
        worker = _BackgroundFieldCommandWorker(
            **kwargs, join_timeout_seconds=0.02 if times_out else 1
        )
        commands.append(worker)
        return worker

    def inert(**kwargs):
        return SimpleNamespace(start=lambda: None, stop=lambda: None)

    runtime = FieldCloudRuntime(
        args=args,
        gateway=Gateway(),
        factories=FieldCloudFactories(
            postgame_worker=inert,
            command_worker=command_worker,
            live_fact_worker=inert,
            saved_media_worker=inert,
            sync_client=lambda **kwargs: object(),
        ),
    )
    state = _server().field_state
    state.record_store, state.match_lock = store, threading.RLock()
    for name in (
        "before_field_command",
        "set_command_match",
        "handle_command_application",
        "set_cloud_control_status",
        "maintain_background",
        "resolve_command_confirmation",
    ):
        setattr(state, name, lambda *args: None)
    state.reduce_live_command = reduce_field_command
    state.wait_until_capture_ready = lambda **kwargs: "epoch"

    def close():
        assert not commands[0]._thread.is_alive()
        closes.append("close")

    def serve():
        assert entered.wait(1)

    server = SimpleNamespace(
        field_state=state, server_port=12345, serve_forever=serve, server_close=close
    )
    monkeypatch.setattr(cli, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
    monkeypatch.setattr(cli, "_validate_paths", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_control_token", lambda args: "control")
    monkeypatch.setattr(cli, "prepare_cloud_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(
        cli,
        "prepare_runtime",
        lambda *args, **kwargs: (
            "calibration",
            "factory",
            SimpleNamespace(cancel=lambda: closes.append("cancel")),
        ),
    )
    monkeypatch.setattr(cli, "build_live_field_assistance_server", lambda **kwargs: server)

    def release_when_stopping():
        assert entered.wait(1)
        assert commands[0]._stopped.wait(1)
        release.set()

    releaser = None
    if not times_out:
        releaser = threading.Thread(target=release_when_stopping)
        releaser.start()
    try:
        if times_out:
            assert cli.main([]) == 2
            error = capsys.readouterr().err
            assert "cleanup failed (cloud workers)" in error
            assert "shutdown incomplete" in error
            assert closes == []
            assert list(runtime._workers) == ["commands"]
        else:
            assert cli.main([]) == 0
            assert closes == ["close"]
    finally:
        release.set()
        if commands:
            commands[0]._thread.join(2)
        runtime.stop()
        if releaser is not None:
            releaser.join(2)
    if times_out:
        server.server_close()
    assert closes == ["close"]
    assert callbacks == []
    assert store.load_session("session-1").match.revision == 0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(
            sync_base_url=None,
            device_id=None,
            credential_target=None,
            cloudbase_environment_id=None,
            expected_development_environment_id=None,
        ),
        dict(table_id=None),
        dict(existing_development_control_only=True),
    ],
)
def test_managed_configuration_refuses_incomplete_or_control_only_before_credentials(
    tmp_path, overrides
):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    with pytest.raises(ValueError):
        prepare_cloud_runtime(
            _args(tmp_path, **(dict(managed_sessions=True, table_id="table-1") | overrides)),
            factories=factories,
            allow_existing_development_control_only=True,
        )
    assert calls == [] and workers == {}


def test_bootstrap_gateway_is_separate_per_http_budgeted_and_does_not_start_workers(tmp_path):
    import json

    from billiards_ai.edge.route_assistance.sync import HttpResult
    from billiards_ai.lab.field_assistance.cloud_gateway import SignedFieldCloudGateway
    from billiards_ai.lab.field_assistance.cloud_runtime import (
        FieldCloudFactories,
        prepare_cloud_runtime,
    )

    clock = [0.0]
    timeouts = []

    class Credentials:
        def __init__(self, **_kwargs):
            pass

        def device_id(self):
            return "raytheon-01"

        def secret(self):
            return b"synthetic-secret"

    class Transport:
        def __init__(self, _url, *, allow_redirects, timeout_seconds=None):
            assert allow_redirects is False
            self.timeout = timeout_seconds

        def post_json(self, path, headers, body):
            assert path == "/v1/runtime-attestation"
            timeouts.append(self.timeout)
            clock[0] += 0.4
            data = dict(
                schema_version="1.0",
                runtime_environment="development",
                cloudbase_environment_id="cloud-development",
            )
            return HttpResult(200, json.dumps(dict(ok=True, data=data)).encode())

    factories = FieldCloudFactories(
        credentials=Credentials, transport=Transport, signed_gateway=SignedFieldCloudGateway
    )
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)
    ordinary = runtime.gateway
    factory = getattr(runtime, "make_managed_startup_gateway", None)
    assert callable(factory), "separate bounded bootstrap gateway factory is missing"
    bootstrap = factory(deadline=1, cancelled=lambda: False, monotonic=lambda: clock[0])
    assert bootstrap is not ordinary and runtime._workers == {}
    with pytest.raises(TimeoutError):
        bootstrap.runtime_attestation()
    # Initial ordinary attestation has no startup deadline. Bootstrap construction
    # attests, then a second attestation only has the actual remaining 0.2 seconds.
    assert timeouts == pytest.approx([None, 0.6, 0.2])
    assert runtime.gateway is ordinary and runtime._workers == {}


def test_managed_receipt_worker_owns_original_scope_independent_of_resources(tmp_path):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()
    observed = {}

    class ReceiptWorker:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def start(self):
            calls.append(("receipt", "start"))

        def stop(self):
            calls.append(("receipt", "stop"))

    factories = replace(factories, upload_receipt_worker=ReceiptWorker)
    args = _args(tmp_path, managed_sessions=True, table_id="table-1")
    runtime = prepare_cloud_runtime(args, factories=factories)
    server = _server()
    scope = ManagedSessionScope(args.cloudbase_environment_id, args.device_id, args.table_id)
    server.field_state.managed_session_scope = scope
    runtime.start(server, postgame_command_factory=lambda session: [session])
    assert observed["store"] is server.field_state.record_store
    assert observed["managed_scope"] == scope
    assert observed["client"].credentials is runtime._credentials
    assert observed["client"].before_mutation.__self__ is runtime.gateway
    runtime.stop()
    assert calls.count(("receipt", "start")) == calls.count(("receipt", "stop")) == 1


@pytest.mark.parametrize("stage", ["construct", "start"])
def test_managed_receipt_partial_start_is_owned_and_all_workers_stop(tmp_path, stage):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, calls, workers = _dependencies()

    class ReceiptWorker:
        def __init__(self, **kwargs):
            if stage == "construct":
                raise RuntimeError("receipt failed")

        def start(self):
            raise RuntimeError("receipt failed")

        def stop(self):
            calls.append(("receipt", "stop"))

    runtime = prepare_cloud_runtime(
        _args(tmp_path, managed_sessions=True, table_id="table-1"),
        factories=replace(factories, upload_receipt_worker=ReceiptWorker),
    )
    server = _server()
    server.field_state.managed_session_scope = ManagedSessionScope(
        "cloud-development", "raytheon-01", "table-1"
    )
    with pytest.raises(RuntimeError, match="receipt failed"):
        runtime.start(server, postgame_command_factory=lambda session: [session])
    assert runtime._workers == {}
    assert calls.count(("receipt", "stop")) == (stage == "start")


def test_unmanaged_receipt_factory_never_constructed_even_with_cloud_and_store(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime

    factories, _, workers = _dependencies()
    factories = replace(
        factories, upload_receipt_worker=lambda **kw: pytest.fail("unmanaged receipt")
    )
    runtime = prepare_cloud_runtime(_args(tmp_path), factories=factories)
    runtime.start(_server(), postgame_command_factory=lambda session: [session])
    runtime.stop()
    assert "receipts" not in workers
