from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateResult,
    CalibrationGateStatus,
    CalibrationValidationMode,
)
from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    CaptureSourceKind,
    FrameGeometry,
)
from billiards_ai.lab.mainline1_live_runtime import (
    LiveRuntimeEvent,
    LiveRuntimeEventKind,
)
from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureErrorCode,
    UsbCaptureReport,
    UsbDeviceIdentity,
)


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_mainline1_usb_live.py"
    spec = importlib.util.spec_from_file_location("run_mainline1_usb_live", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> UsbDeviceIdentity:
    return UsbDeviceIdentity(
        device_locator=r"USB\VID_32E4&PID_9230\FIT1044-A",
        display_name="DFRobot FIT1044",
        runtime_index=None,
    )


def _setup() -> CaptureSetup:
    return CaptureSetup(
        capture_setup_id="setup-live",
        table_id="table-1",
        camera_device_id="camera-1",
        lens_id="fit1044-120",
        mount_revision="mount-1",
        distortion_revision="distortion-1",
        source_kind=CaptureSourceKind.USB,
        geometry=FrameGeometry(width=1920, height=1080, crop=(0, 0, 1920, 1080)),
        requested_fps=60,
        observed_fps=60,
        device_locator_fingerprint=_device().durable_fingerprint,
        validation_corners_px=((0, 1080), (1920, 1080), (1920, 0), (0, 0)),
        image_controls=(("exposure", -6.0),),
    )


def _ready_report(request) -> UsbCaptureReport:
    return UsbCaptureReport(
        status=UsbCaptureErrorCode.READY,
        device_fingerprint=request.device.durable_fingerprint,
        requested_width=1920,
        requested_height=1080,
        requested_fps=60,
        observed_width=1920,
        observed_height=1080,
        observed_fps=60,
        delivered_frames=120,
        failed_reads=0,
        timestamp_reversals=0,
        elapsed_ms=2000,
        source_epoch_id="epoch-live",
        evidence_frame_sha256="f" * 64,
        validation_corners_px=((0, 1080), (1920, 1080), (1920, 0), (0, 0)),
        image_controls=(("exposure", -6.0),),
    )


def _args(tmp_path: Path) -> list[str]:
    return [
        "--device-locator",
        _device().device_locator,
        "--runtime-index",
        "2",
        "--capture-setup",
        str(tmp_path / "setup.json"),
        "--calibration-root",
        str(tmp_path / "calibrations"),
        "--detector-training-run",
        str(tmp_path / "detector"),
        "--identity-training-run",
        str(tmp_path / "identity"),
        "--dinov2-assets",
        str(tmp_path / "dinov2"),
        "--config",
        str(tmp_path / "shot.yaml"),
        "--output-root",
        str(tmp_path / "artifacts"),
        "--spool-root",
        str(tmp_path / "spool"),
    ]


def test_default_entrypoint_detects_current_table_instead_of_always_abstaining() -> None:
    # Break caught: restoring the old placeholder that returned None would make
    # every real USB epoch fail before Mainline 1 starts.
    frame = np.full((540, 960, 3), 18, dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((70, 493), (890, 502), (878, 58), (80, 72)), dtype=np.int32),
        (24, 112, 36),
    )

    corners = _module()._detect_validation_corners(frame)

    assert corners is not None
    assert np.allclose(
        np.asarray(corners),
        np.asarray(((70, 493), (890, 502), (878, 58), (80, 72))),
        atol=3.0,
    )


def test_ready_gate_runs_live_runtime_and_emits_events(tmp_path: Path, capsys) -> None:
    module = _module()
    calibration_path = tmp_path / "calibration.json"
    gate = CalibrationGateResult(
        status=CalibrationGateStatus.READY,
        capture_setup_id="setup-live",
        setup_fingerprint="c" * 64,
        calibration_path=calibration_path,
        calibration_artifact_id="d" * 64,
        calibration_revision=3,
        validation_mode=CalibrationValidationMode.USB_QUICK_CHECK,
    )
    seen = {}
    retained_handle = SimpleNamespace(release=lambda: None)

    evidence_frame = np.full((1080, 1920, 3), 23, dtype=np.uint8)

    def ready_probe(request, *, capture_sink, corner_detector, evidence_frame_sink):
        assert callable(corner_detector)
        assert request.image_controls == _setup().image_controls
        capture_sink(retained_handle)
        evidence_frame_sink(evidence_frame)
        return _ready_report(request)

    def warmup_runner(**kwargs):
        frames = tuple(kwargs["frames"])
        assert len(frames) == 1
        assert frames[0].image is evidence_frame
        assert kwargs["vision_backend"] == "vision"
        assert kwargs["identity_classifier"] == "identity"
        assert kwargs["calibration"] == "calibration"
        seen["warmup_complete"] = True
        return ("observation",)

    class Runtime:
        def __init__(self, **kwargs):
            assert seen["warmup_complete"] is True
            seen.update(kwargs)

        def run(self, *, event_sink):
            events = (
                LiveRuntimeEvent(
                    kind=LiveRuntimeEventKind.SHOT_PUBLISHED,
                    shot_id="shot-0001",
                    artifact_id="e" * 64,
                    artifact_path=tmp_path / "artifacts" / "shot",
                ),
                LiveRuntimeEvent(kind=LiveRuntimeEventKind.CAPTURE_STOPPED, reason="stopped"),
            )
            for event in events:
                event_sink(event)
            return events

    exit_code = module.main(
        [*_args(tmp_path), "--max-frames", "600"],
        enumerator=lambda: (_device(),),
        prober=ready_probe,
        setup_loader=lambda _: _setup(),
        gate_evaluator=lambda **_: gate,
        calibration_loader=lambda _: "calibration",
        config_loader=lambda _: "config",
        vision_loader=lambda *_args, **_kwargs: "vision",
        identity_loader=lambda *_args, **_kwargs: "identity",
        feature_factory=lambda **_kwargs: "features",
        evidence_factory=lambda: "evidence",
        source_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        runtime_factory=Runtime,
        corner_detector=lambda _frame: _setup().validation_corners_px,
        warmup_runner=warmup_runner,
    )

    payload = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert payload[0] == {
        "calibration_artifact_id": "d" * 64,
        "capture_setup_id": "setup-live",
        "source_epoch_id": "epoch-live",
        "status": "LIVE_RUNTIME_READY",
    }
    assert payload[1]["kind"] == "SHOT_PUBLISHED"
    assert payload[1]["artifact_id"] == "e" * 64
    assert seen["source"].request.device.runtime_index == 2
    assert seen["source"].capture_handle is retained_handle
    assert seen["source"].max_frames == 600
    assert seen["capture_setup_id"] == "setup-live"
    assert seen["calibration_revision"] == 3
    assert seen["warmup_complete"] is True


def test_capture_probe_failure_prevents_model_and_runtime_loading(tmp_path: Path, capsys) -> None:
    module = _module()
    called = []

    def rejected(request, **_kwargs):
        return _ready_report(request).model_copy(
            update={"status": UsbCaptureErrorCode.CAPTURE_GEOMETRY_MISMATCH}
        )

    def load_setup(_path):
        called.append("setup")
        return _setup()

    exit_code = module.main(
        _args(tmp_path),
        enumerator=lambda: (_device(),),
        prober=rejected,
        setup_loader=load_setup,
    )

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 3
    assert payload["status"] == "capture_geometry_mismatch"
    assert called == ["setup"]


def test_calibration_gate_failure_prevents_runtime_start(tmp_path: Path, capsys) -> None:
    module = _module()
    gate = CalibrationGateResult(
        status=CalibrationGateStatus.CALIBRATION_REQUIRED,
        capture_setup_id="setup-live",
        setup_fingerprint="c" * 64,
        reasons=("distortion_not_verified",),
    )

    exit_code = module.main(
        _args(tmp_path),
        enumerator=lambda: (_device(),),
        prober=lambda request, **_kwargs: _ready_report(request),
        setup_loader=lambda _: _setup(),
        gate_evaluator=lambda **_: gate,
    )

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 3
    assert payload["status"] == "CALIBRATION_REQUIRED"
    assert payload["reasons"] == ["distortion_not_verified"]


def test_runtime_reconnect_repeats_preflight_and_keeps_the_new_handle(
    tmp_path: Path,
) -> None:
    module = _module()
    calibration_path = tmp_path / "calibration.json"
    gate = CalibrationGateResult(
        status=CalibrationGateStatus.READY,
        capture_setup_id="setup-live",
        setup_fingerprint="c" * 64,
        calibration_path=calibration_path,
        calibration_artifact_id="d" * 64,
        calibration_revision=3,
        validation_mode=CalibrationValidationMode.USB_QUICK_CHECK,
    )
    handles = [SimpleNamespace(release=lambda: None), SimpleNamespace(release=lambda: None)]
    epochs = iter(("epoch-initial", "epoch-reconnected"))
    seen = {}

    def ready_probe(request, *, capture_sink, corner_detector, evidence_frame_sink=None):
        assert request.image_controls == _setup().image_controls
        capture_sink(handles.pop(0))
        if evidence_frame_sink is not None:
            evidence_frame_sink(np.zeros((1080, 1920, 3), dtype=np.uint8))
        return _ready_report(request).model_copy(update={"source_epoch_id": next(epochs)})

    class Runtime:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self, *, event_sink):
            replacement = seen["reconnect_source_factory"]()
            seen["replacement"] = replacement
            return ()

    assert (
        module.main(
            [*_args(tmp_path), "--max-frames", "600"],
            enumerator=lambda: (_device(),),
            prober=ready_probe,
            setup_loader=lambda _: _setup(),
            gate_evaluator=lambda **_: gate,
            calibration_loader=lambda _: "calibration",
            config_loader=lambda _: "config",
            vision_loader=lambda *_args, **_kwargs: "vision",
            identity_loader=lambda *_args, **_kwargs: "identity",
            feature_factory=lambda **_kwargs: "features",
            evidence_factory=lambda: "evidence",
            source_factory=lambda **kwargs: SimpleNamespace(**kwargs),
            runtime_factory=Runtime,
            corner_detector=lambda _frame: _setup().validation_corners_px,
            warmup_runner=lambda **_kwargs: ("observation",),
        )
        == 0
    )
    replacement = seen["replacement"]
    assert replacement.source_epoch_id == "epoch-reconnected"
    assert replacement.source.capture_handle is not seen["source"].capture_handle
    assert replacement.capture_setup_id == "setup-live"
    assert replacement.source.max_frames == 600
    assert seen["maximum_reconnect_attempts"] > 0
