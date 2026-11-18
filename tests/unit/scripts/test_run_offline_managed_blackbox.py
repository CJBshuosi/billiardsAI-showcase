import hashlib
import importlib.util
import json
import threading
from types import SimpleNamespace

import pytest

from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint
from billiards_ai.shared.calibration.homography import calibrate
from scripts.algorithm import offline_managed_transport as protocol
from scripts.algorithm import serve_mainline1_blackbox_field as cli
from tests.support.development_acceptance import write_identity_assets
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures

assets = resource_fixtures.assets


def launcher():
    name = "scripts.algorithm.run_offline_managed_blackbox"
    assert importlib.util.find_spec(name), "missing scoped offline episode launcher"
    return __import__(name, fromlist=["run_episode"])


def arguments(root, resources, video):
    paths = {
        name: getattr(resources, name)
        for name in (
            "capture_setup",
            "calibration_root",
            "config",
            "detector_training_run",
            "identity_training_run",
            "dinov2_assets",
        )
    }
    values = {
        **paths,
        "scenario_video": video,
        "state_db": root / "run/state.sqlite3",
        "jobs_root": root / "run/jobs",
        "ring_root": root / "run/ring",
        "artifact_root": root / "run/artifacts",
        "host": "127.0.0.1",
        "port": 0,
        "device": "cpu",
        "runtime_environment": "development",
        "rule_profile": cli.PROFILE_ID,
        "identity_gate_status": "FAIL",
        "sync_base_url": protocol.BASE_URL,
        "device_id": protocol.DEVICE_ID,
        "credential_target": protocol.CREDENTIAL_TARGET,
        "table_id": protocol.TABLE_ID,
        "cloudbase_environment_id": protocol.ENVIRONMENT_ID,
        "expected_development_environment_id": protocol.ENVIRONMENT_ID,
    }
    argv = ["--managed-sessions", "--development-model-acceptance"]
    for key, value in values.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    return argv


def prepare_file_setup(assets):
    setup = CaptureSetup.model_validate(
        dict(
            capture_setup_id="offline-file",
            table_id=protocol.TABLE_ID,
            camera_device_id="test-camera",
            lens_id="lens",
            mount_revision="1",
            distortion_revision="1",
            source_kind="file",
            requested_fps=60,
            geometry=dict(width=1920, height=1080, crop=[0, 0, 1920, 1080]),
        )
    )
    assets.capture_setup.write_text(setup.model_dump_json())
    directory = assets.calibration_root / ("b" * 64)
    directory.mkdir()
    calibration = (
        calibrate(
            "test-camera",
            [(0, 1079), (1919, 1079), (1919, 0), (0, 0)],
            table_length_mm=2540,
            table_width_mm=1270,
        )
        .model_dump_json()
        .encode()
    )
    (directory / "calibration.json").write_bytes(calibration)
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                artifact_id=directory.name,
                capture_setup_id=setup.capture_setup_id,
                capture_setup_fingerprint=capture_setup_fingerprint(setup),
                source_video_sha256="c" * 64,
                inventory={
                    "calibration.json": {
                        "bytes": len(calibration),
                        "sha256": hashlib.sha256(calibration).hexdigest(),
                    }
                },
            )
        )
    )
    register_calibration(
        assets.calibration_root, SimpleNamespace(path=directory, artifact_id=directory.name), setup
    )
    write_identity_assets(assets.identity_training_run)


@pytest.fixture
def invocation(tmp_path, assets, monkeypatch):
    # Some failure fixtures deliberately leave CLI termination unprovable.
    monkeypatch.setattr(launcher(), "_ACTIVE", threading.Lock())
    monkeypatch.setattr(launcher(), "_RETAINED", None, raising=False)
    root = tmp_path.resolve() / "fresh"
    video = root.parent / "scenario.avi"
    video.write_bytes(b"not decoded in launcher validation tests")
    prepare_file_setup(assets)
    return root, arguments(root, assets, video)


@pytest.mark.parametrize(
    "change",
    [
        "root",
        "symlink",
        "state-db",
        "ring-root",
        "jobs-root",
        "artifact-root",
        "token",
        "token-file",
        "device-id",
        "host",
        "scenario-video",
        "observation",
        "stdin",
    ],
)
def test_scope_errors_never_start_or_allocate(invocation, change):
    root, argv = invocation
    if change == "root":
        root.mkdir()
    elif change == "symlink":
        root.symlink_to(root.parent / "absent")
    elif change.startswith("token"):
        argv += ["--control-" + change + "=private"]
    elif change in {"observation", "stdin"}:
        argv += ["--observation-only" if change == "observation" else "--shutdown-stdin"]
    else:
        argv[argv.index("--" + change) + 1] = str(root.parent / "wrong")
    with pytest.raises(ValueError):
        launcher().run_episode(argv, output_root=root)
    assert not (root / "run").exists()


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), 0, -1, 61, True])
def test_budget_validation_precedes_allocation(invocation, budget):
    root, argv = invocation
    with pytest.raises(ValueError):
        launcher().run_episode(argv, output_root=root, setup_seconds=budget)
    assert not root.exists()


@pytest.mark.parametrize("outcome", ["pre-serve", "exception", "cancel"])
def test_pre_serve_exit_never_shuts_down_and_restores_bindings(invocation, monkeypatch, outcome):
    root, argv = invocation
    marker = object()
    server = SimpleNamespace(service_actions=marker, shutdown=lambda: pytest.fail("not serving"))
    original_builder = lambda **_: server  # noqa: E731
    monkeypatch.setattr(cli, "build_live_field_assistance_server", original_builder)
    prepare, command = cli.prepare_cloud_runtime, cli._postgame_worker_command

    def early_exit(actual):
        assert actual == argv
        with pytest.raises(RuntimeError):
            launcher().run_episode(argv, output_root=root.parent / "second")
        assert not (root.parent / "second").exists()
        cli.build_live_field_assistance_server(
            control_token="sensitive-ephemeral", scenario_playback_gate=object()
        )
        if outcome == "pre-serve":
            return 2
        raise (KeyboardInterrupt if outcome == "cancel" else LookupError)("sensitive-ephemeral")

    monkeypatch.setattr(cli, "main", early_exit)
    if outcome == "pre-serve":
        assert launcher().run_episode(argv, output_root=root) != 0
    else:
        with pytest.raises(KeyboardInterrupt if outcome == "cancel" else LookupError):
            launcher().run_episode(argv, output_root=root)
    assert (
        cli.prepare_cloud_runtime,
        cli.build_live_field_assistance_server,
        cli._postgame_worker_command,
    ) == (prepare, original_builder, command)
    assert server.service_actions is marker
    assert not any(t.name == "offline-managed-observer" for t in threading.enumerate())
    data = (root / "evidence/episode.json").read_text()
    assert "sensitive-ephemeral" not in data
    assert len(data) < 32768 and json.loads(data)["formal_field_acceptance"] == "NOT_ASSERTED"


def test_provider_factories_cannot_acknowledge_delivery_or_use_independent_http():
    remote = protocol.OfflineManagedRemote()
    factories = launcher()._factories(remote)
    from billiards_ai.lab.field_assistance.cloud_runtime import FieldCloudFactories

    for name in (
        "command_worker",
        "local_cleanup_worker",
        "postgame_worker",
        "signed_gateway",
        "attested_gateway",
    ):
        assert getattr(factories, name) is getattr(FieldCloudFactories(), name)
    for name in (
        "live_fact_worker",
        "upload_receipt_worker",
        "saved_media_worker",
        "owner_deletion_worker",
    ):
        worker = getattr(factories, name)(client=object())
        worker.start()
        worker.stop()
    assert remote.snapshot()["call_count"] == 0
    with pytest.raises(OSError):
        factories.owner_deletion_transport(protocol.BASE_URL).post_json("/provider", {}, b"{}")
    assert remote.snapshot()["failure"]


def test_fullgame_requires_frozen_oracle_before_allocating_or_starting(invocation):
    root, argv = invocation
    with pytest.raises(ValueError, match="oracle"):
        launcher().run_episode(argv, output_root=root, observer_mode="fullgame-v1")
    assert not root.exists()


@pytest.mark.parametrize("budget", [True, float("inf"), 601])
def test_fullgame_match_budget_rejected_before_allocation(invocation, budget):
    root, argv = invocation
    with pytest.raises(ValueError, match="match budget"):
        launcher().run_episode(
            argv, output_root=root, observer_mode="fullgame-v1", match_seconds=budget
        )
    assert not root.exists()


def test_fullgame_report_is_versioned_and_formal_gates_stay_negative(invocation, monkeypatch):
    from tests.unit.scripts.test_offline_fullgame_observer import oracle_file

    root, argv = invocation
    path, video, _ = oracle_file(root.parent)
    argv[argv.index("--scenario-video") + 1] = str(video)
    monkeypatch.setattr(cli, "main", lambda _: 2)  # No server, worker, capture or model starts.
    assert (
        launcher().run_episode(
            argv,
            output_root=root,
            observer_mode="fullgame-v1",
            input_oracle=path,
            input_oracle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        != 0
    )
    report = json.loads((root / "evidence/episode.json").read_text())
    assert report["fullgame"]["contract"] == "offline_fullgame_observer_v1"
    assert report["fullgame"]["status"] == "FAILED"
    assert report["complete_game_acceptance"] is False
    assert report["formal_model_qualified"] is False
    assert report["provider_delivery"] == "NOT_ASSERTED"


def test_unavailable_screen_after_serving_stops_owned_server_only(invocation, monkeypatch):
    root, argv = invocation
    stopped = threading.Event()
    actions = []
    server = SimpleNamespace(
        service_actions=lambda: actions.append("service_actions"),
        shutdown=stopped.set,
        server_port=0,
    )
    original = server.service_actions
    driver = launcher()
    monkeypatch.setattr(cli, "build_live_field_assistance_server", lambda **_: server)
    monkeypatch.setattr(driver._Actor, "request", lambda *_: None)
    # A process-wide interrupt could hit the next unrelated invocation.
    import _thread

    monkeypatch.setattr(_thread, "interrupt_main", lambda: actions.append("interrupt_main"))

    def serve(_):
        cli.build_live_field_assistance_server(
            control_token="ephemeral", scenario_playback_gate=None
        )
        server.service_actions()
        stopped.wait(1)
        return 0

    monkeypatch.setattr(cli, "main", serve)
    assert driver.run_episode(argv, output_root=root, setup_seconds=0.1) != 0
    report = json.loads((root / "evidence/episode.json").read_text())
    assert actions == ["service_actions"] and stopped.is_set()
    assert server.service_actions is original
    assert report["failure"] == "SCREEN_TIMEOUT"
    assert not report["commands"]


@pytest.mark.parametrize("purpose", ["analysis", "media"])
def test_only_child_sync_is_disabled_and_context_is_forwarded(invocation, purpose):
    root, argv = invocation
    args = cli._parser().parse_args(argv)
    original = vars(args).copy()
    for directory in (args.jobs_root, args.artifact_root):
        directory.mkdir(parents=True)
    spec = cli.inspect_managed_runtime_resources(args, rule_profile_id=cli.PROFILE_ID)
    scope = SimpleNamespace(
        environment_id=protocol.ENVIRONMENT_ID,
        device_id=protocol.DEVICE_ID,
        table_id=protocol.TABLE_ID,
    )
    record = SimpleNamespace(
        open_request_id="test-open", spec_sha256=spec.spec_sha256, scope=scope, spec=spec
    )
    context = SimpleNamespace(spec_record=record)

    def forward(local, **kwargs):
        assert vars(local) == {**original, "sync_base_url": None} and local is not args
        assert kwargs["resource_context"] is context
        return cli._postgame_worker_command(local, **kwargs)

    command = launcher()._local_command(
        forward,
        args,
        session_id="managed-original",
        server_port=1234,
        resource_context=context,
        purpose=purpose,
    )
    assert command[command.index("--purpose") + 1] == purpose
    assert ("--config" in command) is (purpose == "analysis")
    assert vars(args) == original
    for forbidden in (
        "--development-analysis-sync",
        "--sync-base-url",
        "--device-id",
        "--credential-target",
        "--development-cloudbase-environment-id",
    ):
        assert forbidden not in command
    for flag, value in (
        ("--managed-environment-id", protocol.ENVIRONMENT_ID),
        ("--managed-open-request-id", "test-open"),
        ("--managed-resource-spec-sha256", spec.spec_sha256),
        ("--live-resource-url", "http://127.0.0.1:1234/api/screen"),
    ):
        assert command[command.index(flag) + 1] == value
    with pytest.raises(ValueError):
        launcher()._local_command(
            lambda *_, **__: command + ["--device-id", "other"],
            args,
            session_id="managed-original",
            server_port=1234,
            resource_context=context,
        )


def test_actual_cli_failure_before_ready_restores_service_and_joins(invocation, monkeypatch):
    root, argv = invocation
    events = []

    def fail_ready(**_):
        events.append("capture_wait")
        raise RuntimeError("controlled readiness failure")

    original = lambda: events.append("service_actions")  # noqa: E731
    server = SimpleNamespace(
        service_actions=original,
        server_close=lambda: events.append("server_close"),
        shutdown=lambda: events.append("unsafe_shutdown"),
        field_state=SimpleNamespace(wait_until_capture_ready=fail_ready),
    )
    monkeypatch.setattr(cli, "build_live_field_assistance_server", lambda **_: server)
    assert launcher().run_episode(argv, output_root=root) == 2
    assert events == ["capture_wait", "server_close"]
    assert server.service_actions is original
    report = json.loads((root / "evidence/episode.json").read_text())
    assert report["cleanup"]["cli_exit"] == 2 and report["cleanup"]["observer_joined"]
    assert report["remote"]["issued_count"] == 0


def test_caught_unknown_route_is_sticky_in_final_report(invocation, monkeypatch):
    root, argv = invocation

    def hidden_request(args, *, factories):
        peer = factories.transport(protocol.BASE_URL)
        with pytest.raises(OSError):
            peer.post_json("/unknown", {}, b"{}")
        raise RuntimeError("foreground failure")

    monkeypatch.setattr(cli, "prepare_cloud_runtime", hidden_request)
    assert launcher().run_episode(argv, output_root=root) == 2
    report = json.loads((root / "evidence/episode.json").read_text())
    assert report["failure"] == "REMOTE_PROTOCOL_FAILURE" and report["remote"]["failure"]


def test_mixed_case_archive_rejected_before_any_path_inspection(invocation, monkeypatch):
    root, argv = invocation
    argv[argv.index("--scenario-video") + 1] = str(root.parent / ".PrOjEcT-ArChIvE" / "video")
    with monkeypatch.context() as guard:

        def deny_inspection(*_):
            pytest.fail("filesystem inspected before lexical archive rejection")

        guard.setattr(type(root), "is_symlink", deny_inspection)
        with pytest.raises(ValueError):
            launcher().run_episode(argv, output_root=root)
    assert not root.exists()


def test_case_alias_of_input_directory_cannot_contain_output(invocation, assets, monkeypatch):
    original, argv = invocation
    alias = assets.calibration_root.with_name(assets.calibration_root.name.upper())
    if not alias.exists() or not alias.samefile(assets.calibration_root):
        pytest.skip("case-insensitive volume required")
    root = alias / "disposable-output"
    argv = [item.replace(str(original), str(root)) for item in argv]
    monkeypatch.setattr(cli, "main", lambda _: 0)
    with pytest.raises(ValueError):
        launcher().run_episode(argv, output_root=root)
    assert not root.exists()


def test_incompatible_consistent_capture_metadata_never_allocates(tmp_path, assets, monkeypatch):
    # This original resource fixture has a consistent registered 420x220 FILE calibration.
    root = tmp_path.resolve() / "remains-fresh"
    video = root.parent / "not-decoded.avi"
    video.touch()
    write_identity_assets(assets.identity_training_run)
    monkeypatch.setattr(cli, "main", lambda _: 0)
    with pytest.raises(ValueError):
        launcher().run_episode(arguments(root, assets, video), output_root=root)
    assert not root.exists()


@pytest.mark.parametrize("expire", [False, True])
def test_postgame_read_uses_remaining_deadline_and_retries_transient_get(monkeypatch, expire):
    import time

    driver = launcher()
    budget = 0.20 if expire else 2.0
    actor = driver._Actor(
        protocol.OfflineManagedRemote(),
        {"boundary": {"shot_id": "shot"}},
        threading.Event(),
        (1, 1, 1, budget),
        None,
    )
    actor.session, actor.server, actor.token = "scene", object(), "unit-only"
    digest, clip_id = "b" * 64, "clip-" + "a" * 64
    url = f"/media/postgame/{clip_id}.mp4?manifest={digest}"
    requests = []

    # Isolated read-client contract; no product scene or successful episode is manufactured.
    def respond(path, timeout, *, media=False):
        requests.append((path, timeout))
        if len(requests) == 1:
            raise ConnectionResetError()
        if media:
            assert path == url
            delay = 1.10
            time.sleep(min(delay, timeout))
            if timeout < delay:
                raise TimeoutError()
            return True
        return dict(
            review_session_id="scene",
            postgame_manifest_sha256=digest,
            phase="POSTGAME_REVIEW",
            analysis_status="AVAILABLE",
            expected_count=1,
            available_count=1,
            clips=[dict(clip_id=clip_id, shot_id="shot", media_url=url)],
        )

    # Exercise the bounded transport callbacks that the selection state machine receives.
    def read_only(*, read, wait, report, **_):
        def probe():
            return read("/api/postgame-review") and read(url, media=True)

        wait(probe)
        report["playable_clip_ids"] = [clip_id]

    actor.store = None
    monkeypatch.setattr(driver, "run_postgame_selection", read_only)
    monkeypatch.setattr(actor, "request", respond)
    started = time.monotonic()
    if expire:
        with pytest.raises(driver._EpisodeFailure, match="POSTGAME_TIMEOUT"):
            actor.review()
        assert time.monotonic() - started < budget + 0.25
    else:
        actor.review()
        assert actor.report["postgame"]["playable_clip_ids"] == [clip_id]
    assert requests[-1][0] == url
    assert 0 < requests[-1][1] <= budget
    assert actor.remote.snapshot()["issued_count"] == 0


@pytest.mark.parametrize(
    "at", ["before_registration", "response_headers", "response_body", "post_headers", "post_body"]
)
def test_cli_cancellation_interrupts_only_its_owned_read(invocation, monkeypatch, at):
    import socket
    import time
    from http.client import HTTPResponse

    driver = launcher()
    root, argv = invocation
    connecting, reading_body = threading.Event(), threading.Event()
    seen, shutdowns = {}, []
    server = SimpleNamespace(
        service_actions=lambda: None, server_port=0, shutdown=lambda: shutdowns.append(True)
    )
    monkeypatch.setattr(cli, "build_live_field_assistance_server", lambda **_: server)

    attempts = []
    connections = []

    def selection(*, read, post, **_):
        if at.startswith("post_"):
            attempts.append("save")
            post(
                "/api/postgame-review/save",
                {
                    "request_id": "request",
                    "session_id": "scene",
                    "shot_id": "shot",
                    "postgame_manifest_sha256": "a" * 64,
                },
                nonce="nonce",
            )
            attempts.append("completion reached")
        else:
            read("/api/postgame-review")

    monkeypatch.setattr(driver, "run_postgame_selection", selection)

    def review_only(actor):
        actor.store, actor.session = None, "scene"
        actor.report["boundary"]["shot_id"] = "shot"
        seen["stop"] = actor.stop
        actor.review()

    monkeypatch.setattr(driver._Actor, "run", review_only)
    read = HTTPResponse.read

    def observed_read(response, *args):
        reading_body.set()
        return read(response, *args)

    monkeypatch.setattr(HTTPResponse, "read", observed_read)
    client, peer = socket.socketpair()
    with client, peer:
        client.settimeout(2.0)
        peer.settimeout(2.0)

        def connect(connection):
            connections.append(True)
            connection.sock = client.dup()
            connection.sock.settimeout(connection.timeout)
            connecting.set()
            if at == "before_registration":
                assert seen["stop"].wait(2), "CLI cancellation never began"

        monkeypatch.setattr(driver.HTTPConnection, "connect", connect)

        def exiting_cli(_):
            cli.build_live_field_assistance_server(
                control_token="unit", scenario_playback_gate=None
            )
            server.service_actions()
            assert connecting.wait(2)
            if at != "before_registration":
                received = b""
                while b"\r\n\r\n" not in received:
                    received += peer.recv(2048)
                first_line = (
                    b"POST /api/postgame-review/save"
                    if at.startswith("post_")
                    else b"GET /api/postgame-review"
                )
                assert received.startswith(first_line + b" HTTP/1.1")
            if at in {"response_body", "post_body"}:
                # HTTP/1.0 detaches the response socket from HTTPConnection before body read.
                peer.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 100\r\n\r\n{")
                assert reading_body.wait(2)
            seen["exit_at"] = time.monotonic()
            return 130

        monkeypatch.setattr(cli, "main", exiting_cli)
        assert driver.run_episode(argv, output_root=root, postgame_seconds=2.0) == 130
        assert time.monotonic() - seen["exit_at"] < 0.8
        report = json.loads((root / "evidence/episode.json").read_text())
        assert report["failure"] == "CLI_CANCELLED" and report["cleanup"]["observer_joined"]
        assert connections == [True]
        assert attempts == (["save"] if at.startswith("post_") else [])
        assert not report["commands"] and not shutdowns
        if at == "before_registration":
            peer.setblocking(False)
            with pytest.raises(BlockingIOError):
                peer.recv(1)
