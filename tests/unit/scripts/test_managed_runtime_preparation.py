"""Real CLI preparation and metadata verification with explicit native/model fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from billiards_ai.lab.field_assistance.runtime_resources import inspect_managed_runtime_resources
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import CaptureSetup, capture_setup_fingerprint
from tests.unit.lab.field_assistance import test_runtime_resources as resource_fixtures
from tests.unit.scripts import test_serve_mainline1_blackbox_field as blackbox
from tests.unit.scripts import test_serve_mainline1_usb_field as usb

resource_assets_fixture = resource_fixtures.assets


@pytest.fixture(params=[usb, blackbox], ids=["usb", "blackbox"])
def prepared(request, resource_assets_fixture, monkeypatch):
    cli = request.param
    module = cli.load_script()
    args = resource_assets_fixture
    args.runtime_environment = "development"
    args.device_fingerprint = "d" * 64
    setup = CaptureSetup.model_validate(
        dict(
            capture_setup_id="native-fixture",
            table_id="table-a",
            camera_device_id="camera-a",
            lens_id="lens",
            mount_revision="1",
            distortion_revision="1",
            source_kind="usb" if cli is usb else "file",
            requested_fps=60,
            device_locator_fingerprint=args.device_fingerprint,
            geometry=dict(width=1920, height=1080, crop=[0, 0, 1920, 1080]),
        )
    )
    args.capture_setup.write_text(setup.model_dump_json())
    args.calibration_root = args.calibration_root.parent / "native-calibration"
    directory = args.calibration_root / ("b" * 64)
    directory.mkdir(parents=True)
    (directory / "calibration.json").write_bytes(b"{}")
    manifest = dict(
        artifact_id="b" * 64,
        capture_setup_id=setup.capture_setup_id,
        capture_setup_fingerprint=capture_setup_fingerprint(setup),
        source_video_sha256="c" * 64,
        inventory={"calibration.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}},
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    register_calibration(
        args.calibration_root, SimpleNamespace(path=directory, artifact_id="b" * 64), setup
    )
    cancelled = []

    class NativeSource:
        identity = SimpleNamespace(durable_fingerprint="d" * 64)

        def __init__(self, *a, **kw):
            self.epoch = kw.get("source_epoch_id", kw.get("epoch"))

        def start(self):
            pass

        def cancel(self):
            cancelled.append(self.epoch)

    # Calibration numerical/native and heavy model construction are fixture boundaries;
    # all resource files, active/exact calibration inventory and config parsing remain real.
    monkeypatch.setattr(
        module.realtime.TableCalibration,
        "model_validate_json",
        lambda _: SimpleNamespace(device_id="camera-a"),
    )
    monkeypatch.setattr(module.realtime, "Dinov2IdentityFeatureBackend", lambda **_: object())
    monkeypatch.setattr(
        module.realtime,
        "load_development_vision_backend",
        lambda *a, **kw: SimpleNamespace(infer=lambda _: SimpleNamespace(detections=())),
    )
    monkeypatch.setattr(
        module.realtime,
        "load_development_group_classifier",
        lambda *a, **kw: SimpleNamespace(classify=lambda **_: None),
    )
    if cli is usb:
        monkeypatch.setattr(module.realtime, "_capture", lambda *a, **kw: NativeSource(**kw))
        monkeypatch.setattr(module.realtime, "make_vision_frame_input", lambda *a, **kw: object())
        monkeypatch.setattr(
            module,
            "_validate_started_usb_source",
            lambda *a, **kw: (
                SimpleNamespace(calibration_artifact_id="b" * 64),
                (SimpleNamespace(decode_bgr=lambda: object(), capture_frame_id=1, timestamp_ms=1),),
            ),
        )
    else:
        args.scenario_video = args.capture_setup.parent / "fixture-video.avi"
        args.scenario_video.write_bytes(b"native fixture")
        monkeypatch.setattr(module, "ScenarioVideoCaptureBackend", NativeSource)
    spec = inspect_managed_runtime_resources(args, rule_profile_id=None)
    return SimpleNamespace(module=module, args=args, spec=spec, cancelled=cancelled, cli=cli)


def test_actual_prepare_verifies_spec_and_tags_factory(prepared):
    p = prepared
    cal, factory, source = p.module.prepare_runtime(
        p.args, initial_source_epoch_id="product-epoch", resource_spec=p.spec
    )
    assert cal == "b" * 64
    assert factory.managed_resource_spec_sha256 == p.spec.spec_sha256
    if source is not None:
        source.cancel()
    assert len(p.cancelled) == 1  # USB preflight or still-unopened prepared blackbox source.


@pytest.mark.parametrize("drift", ["arguments", "before", "during"])
def test_actual_prepare_refuses_resource_drift_and_cancels_owned_source(
    prepared, monkeypatch, drift
):
    p = prepared
    if drift == "arguments":
        p.args.batch_size += 1
    elif drift == "before":
        p.args.config.write_text(p.args.config.read_text() + "\n# before")
    else:
        loader = p.module.realtime.load_development_vision_backend

        def mutate(*a, **kw):
            p.args.config.write_text(p.args.config.read_text() + "\n# during")
            return loader(*a, **kw)

        monkeypatch.setattr(p.module.realtime, "load_development_vision_backend", mutate)
    with pytest.raises(ValueError):
        p.module.prepare_runtime(
            p.args, initial_source_epoch_id="product-epoch", resource_spec=p.spec
        )
    if drift == "during":
        assert len(p.cancelled) == 1
    else:
        assert p.cancelled == []


def test_actual_prepare_freezes_parsed_config_for_deferred_runtime(prepared, monkeypatch):
    from billiards_ai.lab import mainline1_field_runtime as runtime_module
    from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEventChannel

    p = prepared
    _, factory, _ = p.module.prepare_runtime(
        p.args, initial_source_epoch_id="product-epoch", resource_spec=p.spec
    )
    # Execute the real factory's deferred construction while replacing only the
    # product/model execution constructor; no capture frames/native I/O are started.
    monkeypatch.setattr(
        runtime_module, "ProductRealtimeRuntime", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    p.args.config.unlink()
    handle = factory("scene-a", ProductRealtimeEventChannel(), 0, "product-epoch")
    try:
        runtime = handle.runtime_factory(object())
        assert runtime.config_sha256 == p.spec.config_sha256
        assert runtime.shot_config.window.motion_confirm_frames == 2
    finally:
        handle.shutdown()
