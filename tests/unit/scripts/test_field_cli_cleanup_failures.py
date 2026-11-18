"""Both actual CLI entrypoints preserve primary outcomes when owned cleanup fails."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.cloud_runtime import FieldCloudShutdownError


@pytest.fixture(params=["usb", "blackbox"])
def cli(request, tmp_path, monkeypatch):
    path = Path(__file__).with_name(f"test_serve_mainline1_{request.param}_field.py")
    spec = importlib.util.spec_from_file_location(f"cleanup_helpers_{request.param}", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    module = helper.load_script()
    argv = helper.arguments(tmp_path)
    monkeypatch.setattr(module, "resolve_acceptance_policy", lambda args: None)
    monkeypatch.setattr(module, "_validate_paths", lambda args: None)
    monkeypatch.setattr(module, "_resolve_control_token", lambda args: "secret")
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    calls = []
    prepared = SimpleNamespace(cancel=lambda: calls.append("cancel"))
    monkeypatch.setattr(
        module, "prepare_runtime", lambda *args, **kwargs: ("cal", object(), prepared)
    )
    cloud = SimpleNamespace(
        gateway=None,
        sync_wakeup=None,
        start=lambda *args, **kwargs: None,
        stop=lambda: calls.append("stop"),
    )
    monkeypatch.setattr(module, "prepare_cloud_runtime", lambda *args, **kwargs: cloud)
    server = SimpleNamespace(
        server_port=12345,
        field_state=SimpleNamespace(wait_until_capture_ready=lambda **kw: "epoch"),
        serve_forever=lambda: None,
        server_close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(module, "build_live_field_assistance_server", lambda **kwargs: server)
    return module, argv, server, cloud, calls


def raise_error(error):
    raise error


@pytest.mark.parametrize("phase", ["startup", "runtime"])
@pytest.mark.parametrize(
    "primary, expected",
    [(RuntimeError("primary runtime failed"), 2), (KeyboardInterrupt(), 130), (None, 2)],
)
def test_primary_outcome_and_capture_cleanup_are_independent(cli, capsys, phase, primary, expected):
    module, argv, server, cloud, calls = cli
    if primary is not None:
        if phase == "startup":
            server.field_state.wait_until_capture_ready = lambda **kw: raise_error(primary)
        else:
            server.serve_forever = lambda: raise_error(primary)
    cleanup = ExceptionGroup("capture owners failed", [OSError("decoder"), RuntimeError("writer")])
    server.server_close = lambda: raise_error(cleanup)
    assert module.main(argv) == expected
    error = capsys.readouterr().err
    assert "cleanup failed" in error.lower()
    assert "decoder" in error and "writer" in error
    if isinstance(primary, RuntimeError):
        assert "primary runtime failed" in error
    assert calls == ["stop"]


def test_unknown_primary_keeps_original_traceback_when_cleanup_also_fails(cli, capsys):
    module, argv, server, cloud, calls = cli
    primary = LookupError("unexpected primary")
    server.serve_forever = lambda: raise_error(primary)
    server.server_close = lambda: raise_error(OSError("capture cleanup"))
    with pytest.raises(LookupError) as caught:
        module.main(argv)
    assert caught.value is primary
    assert any(frame.name == "raise_error" for frame in caught.traceback)
    assert "cleanup failed" in capsys.readouterr().err.lower()


def test_cloud_nontermination_leaves_capture_dependency_alive_and_primary_intact(cli, capsys):
    module, argv, server, cloud, calls = cli
    server.serve_forever = lambda: raise_error(KeyboardInterrupt())
    cloud.stop = lambda: raise_error(FieldCloudShutdownError("worker did not terminate"))
    assert module.main(argv) == 130
    assert calls == []
    assert "cleanup failed" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    "primary, expected", [(RuntimeError("bind failed"), 2), (KeyboardInterrupt(), 130)]
)
def test_prepared_source_cleanup_failure_preserves_bind_outcome(
    cli, monkeypatch, capsys, primary, expected
):
    module, argv, server, cloud, calls = cli
    prepared = SimpleNamespace(cancel=lambda: raise_error(OSError("prepared capture cleanup")))
    monkeypatch.setattr(module, "prepare_runtime", lambda *args, **kw: ("cal", object(), prepared))
    monkeypatch.setattr(
        module, "build_live_field_assistance_server", lambda **kw: raise_error(primary)
    )
    assert module.main(argv) == expected
    assert calls == ["stop"]
    error = capsys.readouterr().err
    assert "cleanup failed (capture/service)" in error
    assert "prepared capture cleanup" in error


@pytest.mark.parametrize("nested", [False, True])
def test_cloud_cleanup_only_failure_is_nonzero_and_retains_dependencies(cli, capsys, nested):
    module, argv, server, cloud, calls = cli
    failure = FieldCloudShutdownError("worker still alive")
    if nested:
        failure = ExceptionGroup("worker failures", [OSError("other worker"), failure])
    cloud.stop = lambda: raise_error(failure)
    assert module.main(argv) == 2
    assert calls == []
    error = capsys.readouterr().err
    assert "cleanup failed (cloud workers)" in error
    assert "worker still alive" in error


@pytest.mark.parametrize(
    "primary_kind, workers_blocked",
    [
        ("handled", False),
        ("interrupt", False),
        ("unknown", False),
        ("handled", True),
    ],
)
def test_actual_main_drains_original_builder_owner_only_after_cloud_workers_stop(
    cli,
    monkeypatch,
    primary_kind,
    workers_blocked,
):
    import threading

    from billiards_ai.lab.field_assistance import server as service
    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource

    module, argv, server, cloud, calls = cli
    if "--join-qr-asset" in argv:
        Path(argv[argv.index("--join-qr-asset") + 1]).write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    release, running = threading.Event(), threading.Event()
    sources = []
    primary = {"handled": RuntimeError, "interrupt": KeyboardInterrupt, "unknown": LookupError}[
        primary_kind
    ]("original startup failure")

    class Runtime:
        def run(self):
            running.set()
            assert release.wait(2)

        def close(self):
            pass

    original_start = USBLiveProjectionSource.start

    def fail_start(source):
        sources.append(source)
        original_join = source._runner.join
        monkeypatch.setattr(
            source._runner, "join", lambda timeout: original_join(min(timeout, 0.02))
        )
        original_start(source)
        assert running.wait(1)
        raise primary

    monkeypatch.setattr(USBLiveProjectionSource, "start", fail_start)
    prepared = SimpleNamespace(cancel=lambda: pytest.fail("partial server already owns capture"))
    monkeypatch.setattr(
        module, "prepare_runtime", lambda *args, **kw: ("cal", lambda *args: Runtime(), prepared)
    )
    monkeypatch.setattr(
        module, "build_live_field_assistance_server", service.build_live_field_assistance_server
    )

    def stop():
        calls.append("stop")
        if workers_blocked:
            raise FieldCloudShutdownError("cloud worker still alive")
        release.set()
        assert sources[0]._runtime_exited.wait(1)

    cloud.stop = stop
    try:
        if primary_kind == "unknown":
            with pytest.raises(LookupError) as caught:
                module.main(argv)
            assert caught.value is primary
        else:
            assert module.main(argv) == (130 if primary_kind == "interrupt" else 2)
        source = sources[0]
        assert calls == ["stop"]
        assert len(primary._field_cleanup_failures) == 1
        if workers_blocked:
            assert source._runner.is_alive()
            assert not source._shutdown_complete
            assert len(primary._field_pending_cleanup) == 1
        else:
            assert not source._runner.is_alive()
            assert source._closed and source._shutdown_complete
            assert primary._field_pending_cleanup == ()
    finally:
        release.set()
        if sources:
            assert sources[0]._runtime_exited.wait(1)
        service._retry_failed_startup_cleanup(primary)
    assert primary._field_pending_cleanup == ()


@pytest.fixture(params=["usb", "blackbox"])
def actual_preparation_cli(request, tmp_path, monkeypatch):
    """Keep main and prepare_runtime real; stop before every model constructor."""
    path = Path(__file__).with_name(f"test_serve_mainline1_{request.param}_field.py")
    spec = importlib.util.spec_from_file_location(f"preparation_helpers_{request.param}", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    module = helper.load_script()
    argv = helper.arguments(tmp_path)
    args = module._parser().parse_args(argv)
    monkeypatch.setattr(module, "resolve_acceptance_policy", lambda args: None)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    setup = SimpleNamespace(
        source_kind=(
            module.realtime.CaptureSourceKind.USB
            if request.param == "usb"
            else module.realtime.CaptureSourceKind.FILE
        ),
        geometry=SimpleNamespace(width=1920, height=1080),
        requested_fps=60,
        device_locator_fingerprint=getattr(args, "device_fingerprint", None),
        camera_device_id="camera",
        capture_setup_id="setup",
        image_controls={},
    )
    record = SimpleNamespace(artifact_id="calibration", artifact_path="artifact")
    artifact = args.calibration_root / "artifact"
    artifact.mkdir()
    (artifact / "calibration.json").write_text("{}")
    monkeypatch.setattr(module.realtime, "load_capture_setup", lambda path: setup)
    monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda setup: "setup-sha")
    monkeypatch.setattr(module.realtime, "resolve_active_calibration", lambda *args: record)
    monkeypatch.setattr(
        module.realtime.TableCalibration,
        "model_validate_json",
        lambda raw: SimpleNamespace(device_id="camera"),
    )
    for name in (
        "Dinov2IdentityFeatureBackend",
        "load_development_vision_backend",
        "load_development_group_classifier",
    ):
        monkeypatch.setattr(
            module.realtime, name, lambda *args, **kwargs: pytest.fail("model boundary reached")
        )
    monkeypatch.setattr(
        module,
        "build_live_field_assistance_server",
        lambda **kwargs: pytest.fail("preparation unexpectedly returned"),
    )
    return request.param, module, argv


@pytest.mark.parametrize("primary_kind", ["handled", "interrupt", "unknown"])
@pytest.mark.parametrize("cloud_stop", ["ready", "release", "blocked"])
def test_actual_preparation_preserves_primary_and_original_pending_cancel_owner(
    actual_preparation_cli,
    monkeypatch,
    capsys,
    primary_kind,
    cloud_stop,
):
    from billiards_ai.lab.field_assistance import server as service

    kind, module, argv = actual_preparation_cli
    primary = {"handled": RuntimeError, "interrupt": KeyboardInterrupt, "unknown": LookupError}[
        primary_kind
    ]("original preparation failure")
    cause = ValueError("original preparation cause")
    primary.__cause__ = cause
    cleanup_failure = OSError("preparation cancellation failed")
    cloud_calls, sources = [], []

    class Source:
        def __init__(self):
            self.identity = SimpleNamespace(durable_fingerprint="a" * 64)
            self.blocked = True
            self.closed = False
            self.cancel_calls = 0

        def start(self):
            raise primary

        def cancel(self):
            self.cancel_calls += 1
            if self.blocked:
                raise cleanup_failure
            self.closed = True

    source = Source()

    def capture(*args, **kwargs):
        sources.append(source)
        return source

    if kind == "usb":
        monkeypatch.setattr(module.realtime, "_capture", capture)
    else:
        monkeypatch.setattr(module, "ScenarioVideoCaptureBackend", capture)

    def config(*args, **kwargs):
        if kind == "usb":
            pytest.fail("USB failure bypassed preflight")
        raise primary

    monkeypatch.setattr(module.realtime, "load_mainline1_shot_local_config", config)

    def stop():
        cloud_calls.append("stop")
        if cloud_stop == "blocked":
            raise FieldCloudShutdownError("owned cloud worker is still alive")
        if cloud_stop == "release":
            source.blocked = False

    cloud = SimpleNamespace(gateway=None, sync_wakeup=None, stop=stop)
    monkeypatch.setattr(module, "prepare_cloud_runtime", lambda *args, **kwargs: cloud)
    try:
        if primary_kind == "unknown":
            with pytest.raises(LookupError) as caught:
                module.main(argv)
            assert caught.value is primary
            assert any(frame.name == "prepare_runtime" for frame in caught.traceback)
        else:
            assert module.main(argv) == (130 if primary_kind == "interrupt" else 2)
        assert primary.__cause__ is cause
        assert sources == [source]
        assert cloud_calls == ["stop"]
        assert primary._field_cleanup_failures[0][1] is cleanup_failure
        pending = primary._field_pending_cleanup
        if cloud_stop == "release":
            assert source.closed and pending == ()
        else:
            assert not source.closed and len(pending) == 1
            assert pending[0][1].__self__ is source
            assert pending[0][1].__func__ is Source.cancel
        assert source.cancel_calls == (1 if cloud_stop == "blocked" else 2)
        error = capsys.readouterr().err
        assert "startup cleanup failed" in error
        assert "preparation cancellation failed" in error
    finally:
        source.blocked = False
        service._retry_failed_startup_cleanup(primary)
    assert source.closed and primary._field_pending_cleanup == ()
    assert source.cancel_calls == {"ready": 3, "release": 2, "blocked": 2}[cloud_stop]


def test_actual_usb_preflight_cleanup_only_failure_keeps_retry_owner(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A validated preflight still owns cleanup when cancellation alone fails."""
    path = Path(__file__).with_name("test_serve_mainline1_usb_field.py")
    spec = importlib.util.spec_from_file_location("usb_preflight_cleanup_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    module = helper.load_script()
    argv = helper.arguments(tmp_path)
    args = module._parser().parse_args(argv)
    record = SimpleNamespace(artifact_id="calibration", artifact_path="artifact")
    (args.calibration_root / "artifact").mkdir()
    (args.calibration_root / "artifact/calibration.json").write_text("{}")
    monkeypatch.setattr(module, "resolve_acceptance_policy", lambda args: None)
    monkeypatch.setattr(
        module.realtime,
        "load_capture_setup",
        lambda path: SimpleNamespace(
            source_kind=module.realtime.CaptureSourceKind.USB,
            geometry=SimpleNamespace(width=1920, height=1080),
            requested_fps=60,
            device_locator_fingerprint=args.device_fingerprint,
            camera_device_id="camera",
            image_controls={},
        ),
    )
    monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda setup: "setup-sha")
    monkeypatch.setattr(module.realtime, "resolve_active_calibration", lambda *args: record)
    monkeypatch.setattr(
        module.realtime.TableCalibration,
        "model_validate_json",
        lambda raw: SimpleNamespace(device_id="camera"),
    )
    monkeypatch.setattr(
        module,
        "_validate_started_usb_source",
        lambda *args, **kwargs: (SimpleNamespace(calibration_artifact_id="calibration"), ()),
    )
    monkeypatch.setattr(
        module.realtime,
        "load_mainline1_shot_local_config",
        lambda *args, **kwargs: pytest.fail("cleanup failure reached models"),
    )
    failure = OSError("USB preflight cancellation alone failed")

    class Source:
        blocked = True
        closed = False
        calls = 0

        def start(self):
            pass

        def cancel(self):
            self.calls += 1
            if self.blocked:
                raise failure
            self.closed = True

    source = Source()
    monkeypatch.setattr(module.realtime, "_capture", lambda *args, **kwargs: source)
    cloud = SimpleNamespace(gateway=None, sync_wakeup=None, stop=lambda: None)
    monkeypatch.setattr(module, "prepare_cloud_runtime", lambda *args, **kwargs: cloud)
    try:
        assert module.main(argv) == 2
        assert failure._field_pending_cleanup[0][1].__self__ is source
        assert failure._field_cleanup_failures[0][1] is failure
        assert source.calls == 2 and not source.closed
        assert "startup cleanup failed" in capsys.readouterr().err
    finally:
        source.blocked = False
        from billiards_ai.lab.field_assistance.server import _retry_failed_startup_cleanup

        _retry_failed_startup_cleanup(failure)
    assert source.closed and failure._field_pending_cleanup == ()
