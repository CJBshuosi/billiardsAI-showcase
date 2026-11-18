from types import SimpleNamespace

import pytest

from tests.unit.scripts import test_serve_mainline1_blackbox_field as blackbox_tests
from tests.unit.scripts import test_serve_mainline1_usb_field as usb_tests


def _without_models(argv):
    result = []
    iterator = iter(argv)
    for value in iterator:
        if value in {"--detector-training-run", "--identity-training-run", "--dinov2-assets"}:
            next(iterator)
        else:
            result.append(value)
    return result + ["--session-policy", "recording-v1"]


@pytest.mark.parametrize("backend", ["usb", "blackbox"])
def test_recording_cli_prepares_real_capture_configuration_without_any_model(
    tmp_path,
    monkeypatch,
    backend,
):
    tests = usb_tests if backend == "usb" else blackbox_tests
    module = tests.load_script()
    args = module._parser().parse_args(_without_models(tests.arguments(tmp_path)))
    module._validate_paths(args)
    assert args.detector_training_run is args.identity_training_run is args.dinov2_assets is None
    calls = []

    def unrequested(*args, **kwargs):
        pytest.fail("recording startup invoked a model/identity/acceptance operation")

    if backend == "blackbox":
        blackbox_tests.stub_loaders(module, args, monkeypatch)
    else:
        setup = SimpleNamespace(
            source_kind=module.realtime.CaptureSourceKind.USB,
            geometry=SimpleNamespace(width=1920, height=1080),
            requested_fps=60,
            device_locator_fingerprint=args.device_fingerprint,
            camera_device_id="camera",
            capture_setup_id="setup",
            image_controls=object(),
        )
        (args.calibration_root / "artifact").mkdir()
        (args.calibration_root / "artifact/calibration.json").write_text("{}")
        monkeypatch.setattr(module.realtime, "load_capture_setup", lambda _: setup)
        monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda _: "setup-sha")
        monkeypatch.setattr(
            module.realtime,
            "resolve_active_calibration",
            lambda *a: SimpleNamespace(
                artifact_id="calibration",
                artifact_path="artifact",
            ),
        )
        monkeypatch.setattr(
            module.realtime.TableCalibration,
            "model_validate_json",
            lambda _: SimpleNamespace(device_id="camera"),
        )
        monkeypatch.setattr(
            module.realtime,
            "_capture",
            lambda *a, **kw: SimpleNamespace(
                start=lambda: calls.append("capture-preflight"),
                cancel=lambda: calls.append("cancel"),
            ),
        )
        monkeypatch.setattr(
            module,
            "_validate_started_usb_source",
            lambda *a, **kw: (
                SimpleNamespace(calibration_artifact_id="calibration"),
                (SimpleNamespace(decode_bgr=unrequested),),
            ),
        )
        monkeypatch.setattr(module.realtime, "load_mainline1_shot_local_config", lambda _: object())
    for name in (
        "Dinov2IdentityFeatureBackend",
        "load_development_vision_backend",
        "load_development_group_classifier",
        "make_vision_frame_input",
    ):
        monkeypatch.setattr(module.realtime, name, unrequested)
    monkeypatch.setattr(module, "resolve_acceptance_policy", unrequested)
    options = []
    monkeypatch.setattr(
        module, "build_live_runtime_factory", lambda **kw: options.append(kw) or "recording-factory"
    )
    calibration_id, factory, prepared = module.prepare_runtime(
        args, initial_source_epoch_id="epoch"
    )
    assert calibration_id == "calibration" and factory == "recording-factory"
    assert options[0]["session_policy"] == "recording-v1"
    assert options[0]["detector"] is options[0]["identity_classifier"] is None
    if prepared is not None:
        prepared.cancel()
    if backend == "usb":
        assert calls == ["capture-preflight", "cancel"]
