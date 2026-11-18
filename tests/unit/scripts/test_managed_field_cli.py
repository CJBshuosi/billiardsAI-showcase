"""Both real CLI parsers and their shared assembly, without native models or cloud."""

from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.runtime_resources import RESOURCE_PATH_NAMES
from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    prepare_cloud_runtime,
)
from billiards_ai.lab.field_assistance.runtime_resources import inspect_managed_runtime_resources
from tests.integration.lab.field_assistance.test_live_server import Capture
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.lab.field_assistance.test_managed_startup import (
    BINDING,
    REQUEST,
    SCOPE,
    Credentials,
    Remote,
)
from tests.unit.scripts import test_serve_mainline1_blackbox_field as blackbox
from tests.unit.scripts import test_serve_mainline1_usb_field as usb

resource_assets_fixture = resource_fixtures.assets


@pytest.fixture(params=[usb, blackbox], ids=["usb", "blackbox"])
def cli(request):
    return request.param


def cloud_arguments():
    return [
        "--sync-base-url",
        "https://dev.example/sync-api",
        "--device-id",
        SCOPE.device_id,
        "--credential-target",
        f"billiardsAI/development/{SCOPE.device_id}",
        "--cloudbase-environment-id",
        SCOPE.environment_id,
        "--expected-development-environment-id",
        SCOPE.environment_id,
    ]


def test_managed_flags_default_off_and_accept_only_explicit_complete_scope(cli, tmp_path):
    module = cli.load_script()
    argv = cli.arguments(tmp_path)
    args = module._parser().parse_args(argv)
    assert args.managed_sessions is False
    assert args.table_id is None
    args = module._parser().parse_args(argv + ["--managed-sessions", "--table-id", SCOPE.table_id])
    assert args.managed_sessions is True and args.table_id == SCOPE.table_id


def test_managed_cli_without_cloud_rejects_before_models_or_database(cli, tmp_path, monkeypatch):
    module = cli.load_script()
    argv = cli.arguments(tmp_path) + ["--managed-sessions", "--table-id", SCOPE.table_id]
    monkeypatch.setattr(module, "prepare_runtime", lambda *a, **kw: pytest.fail("models prepared"))
    assert module.main(argv) == 2
    assert not module._parser().parse_args(argv).state_db.exists()


def test_real_cli_defers_preparer_until_managed_binding_and_preserves_source_policy(
    cli,
    tmp_path,
    monkeypatch,
    resource_assets_fixture,
):
    assets = resource_assets_fixture
    module = cli.load_script()
    argv = cli.arguments(tmp_path)
    if cli is usb:
        argv += ["--runtime-environment", "development"]
    argv += cloud_arguments() + ["--managed-sessions", "--table-id", SCOPE.table_id]
    for name in RESOURCE_PATH_NAMES:
        argv[argv.index("--" + name.replace("_", "-")) + 1] = str(getattr(assets, name))
    args = module._parser().parse_args(argv)
    if args.join_qr_asset is not None:
        args.join_qr_asset.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    spec = inspect_managed_runtime_resources(args, rule_profile_id=args.rule_profile)
    store = SQLiteFieldRecordStore(args.state_db, managed_open_request_id_factory=lambda: REQUEST)
    store.reserve_managed_open(
        SCOPE,
        artifact_id=spec.artifact_id,
        rule_profile_id=spec.rule_profile_id,
        resource_spec=spec,
    )
    store.record_managed_binding(SCOPE, REQUEST, BINDING)
    store.materialize_managed_session(SCOPE, REQUEST)
    remote = Remote()
    factories = replace(
        FieldCloudFactories(),
        credentials=lambda **_: Credentials(),
        transport=lambda *a, **kw: remote,
    )
    runtime = prepare_cloud_runtime(args, factories=factories)
    monkeypatch.setattr(module, "prepare_cloud_runtime", lambda *a, **kw: runtime)
    monkeypatch.setattr(runtime, "start", lambda *a, **kw: None)
    seen = []
    gates = []

    def prepare(_args, *, initial_source_epoch_id, playback_gate=None, resource_spec=None):
        assert resource_spec == spec
        assert (playback_gate is not None) is (cli is blackbox)
        if cli is blackbox:
            assert playback_gate is gates[0] and not playback_gate.is_released()
        current = store.get_active_managed_open_request(SCOPE)
        assert current.phase == "materialized"
        assert (current.binding.session_id != BINDING.session_id) is (cli is blackbox)
        seen.append(current.binding.session_id)

        def factory(session_id, channel, initial_shot_number, source_epoch_id):
            assert session_id == current.binding.session_id
            return Capture(session_id, channel, source_epoch_id)

        factory.managed_resource_spec_sha256 = spec.spec_sha256
        return "cal", factory, None

    monkeypatch.setattr(module, "prepare_runtime", prepare)
    builder = module.build_live_field_assistance_server

    def build(**kwargs):
        if cli is blackbox:
            gates.append(kwargs.get("scenario_playback_gate"))
            assert gates[0] is not None
        assert callable(kwargs.get("runtime_resource_inspector"))
        assert "runtime_preparer" in kwargs
        assert kwargs["managed_startup"].scope == SCOPE
        server = builder(**kwargs)
        server.serve_forever = lambda: None
        return server

    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    assert module.main(argv) == 0
    assert len(seen) == 1


def test_usb_preflight_start_failure_cancels_owned_backend(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = usb.load_script()
    args = module._parser().parse_args(usb.arguments(tmp_path))
    setup = SimpleNamespace(
        source_kind=module.realtime.CaptureSourceKind.USB,
        geometry=SimpleNamespace(width=1920, height=1080),
        requested_fps=60,
        device_locator_fingerprint=args.device_fingerprint,
        camera_device_id="camera",
        image_controls=(),
    )
    record = SimpleNamespace(artifact_id="cal", artifact_path="cal")
    (args.calibration_root / "cal").mkdir()
    (args.calibration_root / "cal/calibration.json").write_text("{}")
    monkeypatch.setattr(module.realtime, "load_capture_setup", lambda _: setup)
    monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(module.realtime, "resolve_active_calibration", lambda *a: record)
    monkeypatch.setattr(
        module.realtime.TableCalibration,
        "model_validate_json",
        lambda _: SimpleNamespace(device_id="camera"),
    )
    cancelled = []

    def fail_start():
        raise RuntimeError("native preflight start failed")

    monkeypatch.setattr(
        module.realtime,
        "_capture",
        lambda *a, **kw: SimpleNamespace(
            start=fail_start,
            cancel=lambda: cancelled.append(True),
        ),
    )
    with pytest.raises(RuntimeError, match="preflight start"):
        module.prepare_runtime(args, initial_source_epoch_id="epoch")
    assert cancelled == [True]


@pytest.mark.parametrize("flag", ["--skip-resource-check", "--allow-resource-drift"])
def test_managed_cli_has_no_resource_context_bypass(cli, tmp_path, flag):
    module = cli.load_script()
    with pytest.raises(SystemExit):
        module._parser().parse_args(cli.arguments(tmp_path) + [flag])


def test_both_entrypoint_shared_runtime_owns_receipts_without_resource_sessions(cli, tmp_path):
    from types import SimpleNamespace

    from tests.unit.lab.field_assistance.test_cloud_runtime import _dependencies, _server

    module = cli.load_script()
    argv = cli.arguments(tmp_path) + cloud_arguments()
    argv += ["--managed-sessions", "--table-id", SCOPE.table_id]
    if cli is usb:
        argv += ["--runtime-environment", "development"]
    args = module._parser().parse_args(argv)
    factories, calls, workers = _dependencies()
    observed = []

    def receipts(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(
            start=lambda: calls.append("receipts-start"), stop=lambda: calls.append("receipts-stop")
        )

    # Both production entrypoints expose and call this one shared preparer.
    assert module.prepare_cloud_runtime is prepare_cloud_runtime
    runtime = module.prepare_cloud_runtime(
        args,
        factories=replace(
            factories,
            upload_receipt_worker=receipts,
            attested_gateway=lambda **kwargs: kwargs["delegate"],
        ),
    )
    server = _server()
    server.field_state.managed_session_scope = SCOPE
    runtime.start(server, postgame_command_factory=lambda session: [session])
    assert len(observed) == 1
    assert observed[0]["managed_scope"] == SCOPE
    assert observed[0]["store"] is server.field_state.record_store
    assert observed[0]["client"].before_mutation.__self__ is runtime.gateway
    runtime.stop()
    assert calls.count("receipts-start") == calls.count("receipts-stop") == 1
