import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    prepare_cloud_runtime,
)

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/algorithm/serve_mainline1_blackbox_field.py"
REQUIRED = (
    "scenario-video",
    "capture-setup",
    "calibration-root",
    "detector-training-run",
    "identity-training-run",
    "dinov2-assets",
    "config",
    "state-db",
    "ring-root",
    "jobs-root",
    "artifact-root",
    "host",
    "port",
    "runtime-environment",
)


def load_script():
    assert SCRIPT.is_file(), "black-box field entrypoint is missing"
    spec = importlib.util.spec_from_file_location("blackbox_field_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(root):
    result = []
    for flag in REQUIRED:
        path = root / flag
        if flag in ("scenario-video", "capture-setup", "config"):
            path.write_bytes(b"fixture")
        elif flag.endswith("root") or flag.endswith("run") or flag == "dinov2-assets":
            path.mkdir(exist_ok=True)
        value = {"host": "127.0.0.1", "port": "0", "runtime-environment": "development"}.get(
            flag, str(path)
        )
        result.extend(("--" + flag, value))
    return result


def stub_loaders(module, args, monkeypatch):
    setup = SimpleNamespace(
        source_kind=module.realtime.CaptureSourceKind.FILE,
        geometry=SimpleNamespace(width=1920, height=1080),
        requested_fps=60,
        camera_device_id="camera",
        capture_setup_id="setup",
    )
    record = SimpleNamespace(artifact_id="calibration", artifact_path="artifact")
    (args.calibration_root / "artifact").mkdir(exist_ok=True)
    (args.calibration_root / "artifact/calibration.json").write_text("{}")
    calibration = SimpleNamespace(device_id="camera")
    monkeypatch.setattr(module.realtime, "load_capture_setup", lambda _: setup)
    monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda _: "setup-sha")
    monkeypatch.setattr(module.realtime, "resolve_active_calibration", lambda *a: record)
    monkeypatch.setattr(
        module.realtime.TableCalibration, "model_validate_json", lambda _: calibration
    )
    models = [object(), object(), object()]
    monkeypatch.setattr(module.realtime, "load_mainline1_shot_local_config", lambda _: models[0])
    monkeypatch.setattr(module.realtime, "Dinov2IdentityFeatureBackend", lambda **kw: object())
    monkeypatch.setattr(
        module.realtime, "load_development_vision_backend", lambda *a, **kw: models[1]
    )
    monkeypatch.setattr(
        module.realtime, "load_development_group_classifier", lambda *a, **kw: models[2]
    )
    monkeypatch.setattr(
        module.realtime, "_capture", lambda *a, **kw: pytest.fail("native capture invoked")
    )
    return setup, calibration, models


def test_help_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for flag in REQUIRED:
        assert "--" + flag in result.stdout


def test_model_paths_alone_do_not_enable_snapshot_child(tmp_path):
    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path) + ["--session-policy", "recording-v1"])
    assert module._snapshot_factory(args) is None


def test_explicit_s1_uses_spawn_safe_model_factory(tmp_path):
    module = load_script()
    args = module._parser().parse_args(
        arguments(tmp_path) + ["--session-policy", "recording-v1", "--s1-guidance"]
    )
    service_factory = module._snapshot_factory(args)
    assert service_factory.observer_factory.detector_training_run == args.detector_training_run
    from billiards_ai.lab.planning.s1_validation import prepare_s1_physics

    assert service_factory.resource_preparer is prepare_s1_physics


@pytest.mark.parametrize("flag", REQUIRED)
def test_each_required_argument_is_required(tmp_path, flag):
    module = load_script()
    argv = arguments(tmp_path)
    index = argv.index("--" + flag)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2
    assert not (tmp_path / "state-db").exists()


@pytest.mark.parametrize(
    "flag",
    [
        "events",
        "events-jsonl",
        "event-input",
        "facts",
        "rule-facts",
        "truth",
        "truth-manifest",
        "timeline",
        "replay",
        "replay-artifact",
        "observations",
        "detections",
        "decisions",
        "routes",
        "native-helper",
        "device-locator",
        "existing-development-control-only",
        "scenario-vid",
    ],
)
def test_injection_native_legacy_and_abbreviated_flags_rejected(tmp_path, flag):
    module = load_script()
    with pytest.raises(SystemExit) as exc:
        module.main(arguments(tmp_path) + ["--" + flag, "forbidden"])
    assert exc.value.code == 2
    assert not (tmp_path / "state-db").exists()


@pytest.mark.parametrize("environment", ["production", "unknown", ""])
def test_development_guard_before_preparation(tmp_path, monkeypatch, environment):
    module = load_script()
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("prepared"))
    argv = arguments(tmp_path)
    argv[argv.index("--runtime-environment") + 1] = environment
    with pytest.raises(SystemExit):
        module.main(argv)


@pytest.mark.parametrize("kind", ["missing", "directory", "leaf-symlink", "parent-symlink"])
def test_original_scenario_path_is_checked_before_models_or_state(tmp_path, monkeypatch, kind):
    module = load_script()
    argv = arguments(tmp_path)
    args = module._parser().parse_args(argv)
    stub_loaders(module, args, monkeypatch)
    monkeypatch.setattr(
        module.realtime, "Dinov2IdentityFeatureBackend", lambda **kw: pytest.fail("models loaded")
    )
    video = args.scenario_video
    video.unlink()
    if kind == "directory":
        video.mkdir()
    elif kind == "leaf-symlink":
        target = tmp_path / "target"
        target.write_bytes(b"video")
        video.symlink_to(target)
    elif kind == "parent-symlink":
        target = tmp_path / "real"
        target.mkdir()
        (target / "video").write_bytes(b"video")
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)
        argv[argv.index("--scenario-video") + 1] = str(link / "video")
    assert module.main(argv) == 2
    assert not args.state_db.exists()


@pytest.mark.parametrize("development_acceptance", [False, True])
def test_prepare_pins_source_identity_and_passes_product_dependencies(
    tmp_path, monkeypatch, development_acceptance
):
    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    setup, calibration, models = stub_loaders(module, args, monkeypatch)
    acceptance = {}
    if development_acceptance:
        from billiards_ai.lab.field_assistance.development_acceptance import (
            resolve_acceptance_policy,
        )
        from tests.support.development_acceptance import write_identity_assets

        acceptance = {"acceptance": write_identity_assets(args.identity_training_run)}
        args.development_model_acceptance = True
        args.identity_gate_status = "FAIL"
        args.rule_profile = module.PROFILE_ID
        args._development_acceptance_policy = resolve_acceptance_policy(args)
    options = {}
    monkeypatch.setattr(
        module, "build_live_runtime_factory", lambda **kw: options.update(kw) or "factory"
    )
    calibration_id, factory, prepared = module.prepare_runtime(
        args, initial_source_epoch_id="epoch-1"
    )
    assert (calibration_id, factory) == ("calibration", "factory")
    assert options["capture_factory"]("epoch-1") is prepared
    second = options["capture_factory"]("epoch-2")
    assert second is not prepared
    assert second._epoch == "epoch-2"
    assert options.get("capture_starter") is None
    assert options.get("startup_validator") is None
    assert options["calibration"] is calibration
    assert options["shot_config"] is models[0]
    assert options["detector"] is models[1]
    assert options["identity_classifier"] is models[2]
    assert options["capture_setup_id"] == setup.capture_setup_id
    assert options["config_path"] == args.config
    diagnostic = options["capture_diagnostics_factory"](second, "epoch-2")
    assert diagnostic == dict(
        source_kind="blackbox_video",
        runtime_environment="development",
        source_sha256=hashlib.sha256(b"fixture").hexdigest(),
        source_epoch_id="epoch-2",
        width=1920,
        height=1080,
        fps=60,
        **acceptance,
    )
    assert str(tmp_path) not in json.dumps(diagnostic)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    replacement.replace(args.scenario_video)
    with pytest.raises(ValueError, match="identity|fingerprint"):
        options["capture_factory"]("epoch-3")
    prepared.cancel()
    second.cancel()


@pytest.mark.parametrize("mismatch", ["geometry", "fps", "source", "calibration"])
def test_setup_and_calibration_must_match(tmp_path, monkeypatch, mismatch):
    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    setup, calibration, _ = stub_loaders(module, args, monkeypatch)
    if mismatch == "geometry":
        setup.geometry.width = 640
    elif mismatch == "fps":
        setup.requested_fps = 30
    elif mismatch == "source":
        setup.source_kind = module.realtime.CaptureSourceKind.USB
    else:
        calibration.device_id = "other-camera"
    with pytest.raises(ValueError):
        module.prepare_runtime(args, initial_source_epoch_id="epoch")
    assert not args.state_db.exists()


@pytest.mark.parametrize("failure", [None, "bind", "ready", "interrupt"])
def test_server_lifecycle_and_fresh_random_epochs(tmp_path, monkeypatch, capsys, failure):
    module = load_script()
    argv = arguments(tmp_path)
    calls, options, epochs = [], {}, []
    prepared = SimpleNamespace(cancel=lambda: calls.append("cancel"))

    def prepare(args, *, initial_source_epoch_id):
        epochs.append(initial_source_epoch_id)
        return "calibration", "factory", prepared

    monkeypatch.setattr(module, "prepare_runtime", prepare)

    def ready(**kw):
        assert kw == {"timeout_seconds": 30.0}
        calls.append("ready")
        if failure == "ready":
            raise RuntimeError("not ready")
        return epochs[0]

    def serve():
        calls.append("serve")
        if failure == "interrupt":
            raise KeyboardInterrupt

    server = SimpleNamespace(
        field_state=SimpleNamespace(record_store=object(), wait_until_capture_ready=ready),
        server_port=12345,
        serve_forever=serve,
        server_close=lambda: calls.append("close"),
    )

    def build(**kw):
        options.update(kw)
        observed = [kw["source_epoch_factory"]() for _ in range(3)]
        assert observed[0] == epochs[0]
        assert len(set(observed)) == 3
        assert all(len(value) == 32 for value in observed)
        if failure == "bind":
            raise OSError("bind failed")
        return server

    monkeypatch.setattr(module, "build_live_field_assistance_server", build)

    def worker(**kw):
        command = kw["command_factory"]("ended-session")
        assert "http://127.0.0.1:12345/api/screen" in command
        assert "--development-analysis-sync" not in command
        return SimpleNamespace(
            start=lambda: calls.append("start"), stop=lambda: calls.append("stop")
        )

    monkeypatch.setattr(
        module,
        "prepare_cloud_runtime",
        lambda args: prepare_cloud_runtime(
            args, factories=replace(FieldCloudFactories(), postgame_worker=worker)
        ),
    )
    assert module.main(argv) == {None: 0, "bind": 2, "ready": 2, "interrupt": 130}[failure]
    assert options["fresh_session_on_start"] is True
    assert options["cloud_gateway"] is None
    assert options["runtime_factory"] == "factory"
    assert "join_qr_asset_path" not in options
    assert (
        calls
        == {
            None: ["ready", "start", "serve", "stop", "close"],
            "bind": ["cancel"],
            "ready": ["ready", "close"],
            "interrupt": ["ready", "start", "serve", "stop", "close"],
        }[failure]
    )
    assert ("BLACKBOX_FIELD_READY" in capsys.readouterr().out) == (failure in (None, "interrupt"))


@pytest.mark.parametrize("status", ["FAIL", "UNDECLARED"])
def test_identity_gate_blocks_rule_authority(tmp_path, monkeypatch, status):
    module = load_script()
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("prepared"))
    assert (
        module.main(
            arguments(tmp_path)
            + ["--rule-profile", module.PROFILE_ID, "--identity-gate-status", status]
        )
        == 2
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("port", "-1"),
        ("port", "65536"),
        ("batch-size", "0"),
        ("capture-setup", "missing"),
        ("config", "missing"),
        ("calibration-root", "missing"),
        ("detector-training-run", "missing"),
        ("identity-training-run", "missing"),
        ("dinov2-assets", "missing"),
        ("ring-root", "missing"),
        ("jobs-root", "missing"),
        ("artifact-root", "missing"),
        ("state-db", "missing-parent/state.sqlite"),
        ("join-qr-asset", "missing"),
        ("control-token-file", "missing"),
    ],
)
def test_invalid_inputs_fail_before_preparation(tmp_path, monkeypatch, option, value):
    module = load_script()
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("prepared"))
    argv = arguments(tmp_path)
    if "--" + option in argv:
        argv[argv.index("--" + option) + 1] = value
    else:
        argv.extend(("--" + option, value))
    assert module.main(argv) == 2
    assert not (tmp_path / "state-db").exists()


def test_existing_state_db_directory_fails_before_models_or_jobs(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path)
    state_db = tmp_path / "state-db"
    state_db.mkdir()
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("prepared"))
    monkeypatch.setattr(
        module, "prepare_cloud_runtime", lambda *a, **kw: pytest.fail("cloud prepared")
    )

    assert module.main(argv) == 2
    assert state_db.is_dir()


def test_token_transport_conflict_fails_before_preparation(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path)
    token = tmp_path / "token"
    token.write_text("a" * 32)
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("prepared"))
    conflicting = argv + ["--control-token", "secret", "--control-token-file", str(token)]
    assert module.main(conflicting) == 2


def test_prepared_first_epoch_rechecks_source_before_decoding(tmp_path, monkeypatch):
    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    stub_loaders(module, args, monkeypatch)
    options = {}
    monkeypatch.setattr(module, "build_live_runtime_factory", lambda **kw: options.update(kw))
    _, _, prepared = module.prepare_runtime(args, initial_source_epoch_id="first")
    args.scenario_video.write_bytes(b"changed-after-prepare")
    first = options["capture_factory"]("first")
    try:
        with pytest.raises(ValueError, match="changed|replaced"):
            next(first.frames(threading.Event()))
    finally:
        prepared.cancel()


def test_model_loading_failure_cancels_prepared_source(tmp_path, monkeypatch):
    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    stub_loaders(module, args, monkeypatch)
    calls = []
    monkeypatch.setattr(
        module,
        "ScenarioVideoCaptureBackend",
        lambda *a, **kw: SimpleNamespace(
            identity=SimpleNamespace(durable_fingerprint="a" * 64),
            cancel=lambda: calls.append("cancel"),
        ),
    )

    def failed_model(**kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(module.realtime, "Dinov2IdentityFeatureBackend", failed_model)
    with pytest.raises(RuntimeError, match="model unavailable"):
        module.prepare_runtime(args, initial_source_epoch_id="first")
    assert calls == ["cancel"]


def cloud_arguments():
    return [
        "--sync-base-url",
        "https://cloud-development.example/sync-api",
        "--device-id",
        "blackbox-01",
        "--credential-target",
        "billiardsAI/development/blackbox-01",
        "--cloudbase-environment-id",
        "cloud-development",
        "--expected-development-environment-id",
        "cloud-development",
    ]


@pytest.mark.parametrize(
    "change",
    [
        "sync-base-url",
        "device-id",
        "credential-target",
        "cloudbase-environment-id",
        "expected-development-environment-id",
        "wrong-environment",
        "wrong-target",
        "http",
        "scope-only",
    ],
)
def test_invalid_cloud_scope_fails_before_credentials_models_and_server(
    tmp_path, monkeypatch, change
):
    module = load_script()
    flags = cloud_arguments()
    if change in ("wrong-environment", "wrong-target", "http"):
        flag, value = {
            "wrong-environment": ("cloudbase-environment-id", "production"),
            "wrong-target": ("credential-target", "billiardsAI/development/other-device"),
            "http": ("sync-base-url", "http://cloud-development.example/sync-api"),
        }[change]
        flags[flags.index("--" + flag) + 1] = value
    elif change == "scope-only":
        flags = flags[-4:]
    else:
        index = flags.index("--" + change)
        del flags[index : index + 2]
    argv = arguments(tmp_path) + flags
    args = module._parser().parse_args(argv)
    monkeypatch.setattr(
        module,
        "prepare_cloud_runtime",
        lambda args: prepare_cloud_runtime(
            args,
            factories=replace(
                FieldCloudFactories(), credentials=lambda **kw: pytest.fail("credentials read")
            ),
        ),
    )
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("models loaded"))
    monkeypatch.setattr(
        module, "build_live_field_assistance_server", lambda **kw: pytest.fail("server built")
    )
    assert module.main(argv) == 2
    assert not args.state_db.exists()


@pytest.mark.parametrize("failure", [None, "prepare", "models", "bind", "ready", "start", "stop"])
def test_cloud_lifecycle_prepares_before_models_and_stops_before_capture_close(
    tmp_path, monkeypatch, failure, capsys
):
    module = load_script()
    calls, options = [], {}
    argv = arguments(tmp_path) + cloud_arguments()
    gateway = object()
    wakeup = threading.Event().set

    def prepare_cloud(args):
        calls.append("cloud.prepare")
        assert args.sync_base_url == "https://cloud-development.example/sync-api"
        if failure == "prepare":
            raise OSError("attestation unavailable")
        return runtime

    def start(server, *, postgame_command_factory):
        calls.append("cloud.start")
        command = postgame_command_factory("ended-session")
        assert "--development-analysis-sync" in command
        assert "--development-cloudbase-environment-id" in command
        assert command[command.index("--credential-target") + 1] == (
            "billiardsAI/development/blackbox-01"
        )
        if failure == "start":
            raise RuntimeError("worker start failed")

    def stop():
        calls.append("cloud.stop")
        if failure == "stop":
            raise RuntimeError("worker stop failed")

    runtime = SimpleNamespace(gateway=gateway, sync_wakeup=wakeup, start=start, stop=stop)
    monkeypatch.setattr(module, "prepare_cloud_runtime", prepare_cloud)

    def prepare(*args, **kwargs):
        calls.append("models")
        if failure == "models":
            raise RuntimeError("models unavailable")
        return "calibration", "factory", SimpleNamespace(cancel=lambda: calls.append("cancel"))

    monkeypatch.setattr(module, "prepare_runtime", prepare)

    def ready(**kwargs):
        calls.append("ready")
        if failure == "ready":
            raise RuntimeError("not ready")
        return "epoch"

    server = SimpleNamespace(
        field_state=SimpleNamespace(wait_until_capture_ready=ready),
        server_port=12345,
        serve_forever=lambda: calls.append("serve"),
        server_close=lambda: calls.append("close"),
    )

    def build(**kwargs):
        options.update(kwargs)
        calls.append("bind")
        if failure == "bind":
            raise OSError("bind failed")
        return server

    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    # Cleanup-only failures now select a deliberate nonzero CLI outcome and
    # report their owner independently, just like primary-plus-cleanup errors.
    assert module.main(argv) == (0 if failure is None else 2)
    if failure == "stop":
        error = capsys.readouterr().err
        assert "cleanup failed (cloud workers)" in error
        assert "worker stop failed" in error
    assert (
        calls
        == {
            None: [
                "cloud.prepare",
                "models",
                "bind",
                "ready",
                "cloud.start",
                "serve",
                "cloud.stop",
                "close",
            ],
            "prepare": ["cloud.prepare"],
            "models": ["cloud.prepare", "models", "cloud.stop"],
            "bind": ["cloud.prepare", "models", "bind", "cloud.stop", "cancel"],
            "ready": ["cloud.prepare", "models", "bind", "ready", "cloud.stop", "close"],
            "start": [
                "cloud.prepare",
                "models",
                "bind",
                "ready",
                "cloud.start",
                "cloud.stop",
                "close",
            ],
            "stop": [
                "cloud.prepare",
                "models",
                "bind",
                "ready",
                "cloud.start",
                "serve",
                "cloud.stop",
                "close",
            ],
        }[failure]
    )
    if options:
        assert options["fresh_session_on_start"] is True
        assert options["cloud_gateway"] is gateway
        assert options["sync_wakeup"] is wakeup


def test_prepared_and_later_backends_share_managed_scene_playback_gate(tmp_path, monkeypatch):
    from billiards_ai.lab.mainline1_scenario_capture import ScenarioPlaybackGate

    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    stub_loaders(module, args, monkeypatch)
    options = {}
    monkeypatch.setattr(module, "build_live_runtime_factory", lambda **kw: options.update(kw))
    gate = ScenarioPlaybackGate()
    _, _, prepared = module.prepare_runtime(
        args,
        initial_source_epoch_id="first",
        playback_gate=gate,
    )
    first = options["capture_factory"]("first")
    second = options["capture_factory"]("second")
    try:
        assert first is prepared
        assert first._playback_gate is second._playback_gate is gate
        assert not gate.is_released()
    finally:
        first.cancel()
        second.cancel()


@pytest.mark.parametrize(
    "conflict", ["profile", "observation", "PASS", "UNDECLARED", "actual-PASS"]
)
def test_development_acceptance_rejected_before_cloud(tmp_path, monkeypatch, conflict):
    from tests.support.development_acceptance import write_identity_assets

    module = load_script()
    argv = arguments(tmp_path) + [
        "--development-model-acceptance",
        "--identity-gate-status",
        "FAIL",
    ]
    if conflict != "profile":
        argv += ["--rule-profile", module.PROFILE_ID]
    if conflict == "observation":
        argv += ["--observation-only"]
    if conflict in ("PASS", "UNDECLARED"):
        argv[argv.index("--identity-gate-status") + 1] = conflict
    write_identity_assets(
        tmp_path / "identity-training-run",
        gate_status="PASS" if conflict == "actual-PASS" else "FAIL",
    )
    monkeypatch.setattr(
        module, "prepare_cloud_runtime", lambda *a, **kw: pytest.fail("cloud prepared")
    )
    assert module.main(argv) == 2


def test_development_acceptance_ready_retains_full_rules(tmp_path, monkeypatch, capsys):
    from tests.support.development_acceptance import write_identity_assets

    module = load_script()
    argv = arguments(tmp_path) + [
        "--development-model-acceptance",
        "--identity-gate-status",
        "FAIL",
        "--rule-profile",
        module.PROFILE_ID,
    ]
    expected = write_identity_assets(tmp_path / "identity-training-run")
    observed = {}
    monkeypatch.setattr(
        module,
        "prepare_cloud_runtime",
        lambda *a, **kw: SimpleNamespace(
            gateway=None,
            sync_wakeup=None,
            start=lambda *a, **kw: None,
            stop=lambda: None,
        ),
    )

    def prepare(args, *, initial_source_epoch_id):
        assert args._development_acceptance_policy.as_payload() == expected
        return "cal", "factory", SimpleNamespace(cancel=lambda: None)

    def build(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            field_state=SimpleNamespace(wait_until_capture_ready=lambda **kw: "validated"),
            server_port=1234,
            serve_forever=lambda: None,
            server_close=lambda: None,
        )

    monkeypatch.setattr(module, "prepare_runtime", prepare)
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    assert module.main(argv) == 0
    ready = json.loads(capsys.readouterr().out.strip())
    assert ready["acceptance"] == expected
    assert observed["rule_profile_id"] == module.PROFILE_ID


@pytest.mark.parametrize("change_at", ["before", "during"])
def test_development_acceptance_rechecks_preparation_and_cleans_source(
    tmp_path, monkeypatch, change_at
):
    from billiards_ai.lab.field_assistance.development_acceptance import resolve_acceptance_policy
    from tests.support.development_acceptance import write_identity_assets

    module = load_script()
    args = module._parser().parse_args(arguments(tmp_path))
    stub_loaders(module, args, monkeypatch)
    write_identity_assets(args.identity_training_run)
    args.development_model_acceptance = True
    args.identity_gate_status = "FAIL"
    args.rule_profile = module.PROFILE_ID
    args._development_acceptance_policy = resolve_acceptance_policy(args)
    calls = []
    source = SimpleNamespace(
        identity=SimpleNamespace(durable_fingerprint="a" * 64),
        cancel=lambda: calls.append("cancel"),
    )
    monkeypatch.setattr(
        module, "ScenarioVideoCaptureBackend", lambda *a, **kw: calls.append("source") or source
    )
    monkeypatch.setattr(module, "build_live_runtime_factory", lambda **kw: object())

    def change(*a, **kw):
        write_identity_assets(args.identity_training_run, gate_violations=["changed failure"])
        return object()

    if change_at == "before":
        change()
    else:
        monkeypatch.setattr(module.realtime, "load_development_group_classifier", change)
    with pytest.raises(ValueError, match="changed"):
        module.prepare_runtime(args, initial_source_epoch_id="epoch")
    assert calls == ([] if change_at == "before" else ["source", "cancel"])
