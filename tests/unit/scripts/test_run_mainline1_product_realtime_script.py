import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEvent,
    ProductRealtimeEventChannelFailure,
)
from billiards_ai.shared.calibration.homography import calibrate
from scripts.algorithm import run_mainline1_product_realtime as runtime_script
from scripts.algorithm.run_mainline1_product_realtime import (
    _detector,
    _parse_arguments,
    _select_table_corners,
)


def test_table_corner_selector_uses_canonical_twelve_pixel_default() -> None:
    import inspect

    assert inspect.signature(_select_table_corners).parameters["drift_threshold_px"].default == 12.0


def _required_arguments() -> list[str]:
    return [
        "--native-helper",
        "capture.exe",
        "--device-locator",
        "camera-1",
        "--device-fingerprint",
        "f" * 64,
        "--capture-setup",
        "setup.json",
        "--calibration-root",
        "calibrations",
        "--detector-training-run",
        "detector",
        "--identity-training-run",
        "identity",
        "--dinov2-assets",
        "dinov2",
        "--config",
        "config.json",
        "--output-root",
        "output",
        "--ring-root",
        "ring",
    ]


def test_diagnostic_stop_options_require_named_mode_gate() -> None:
    with pytest.raises(SystemExit):
        _parse_arguments(_required_arguments() + ["--stop-after-terminal-shots", "1"])

    arguments = _parse_arguments(
        _required_arguments() + ["--diagnostic-mode", "--stop-after-terminal-shots", "1"]
    )

    assert arguments.diagnostic_mode is True
    assert arguments.stop_after_terminal_shots == 1


def test_default_runner_has_no_automatic_stop_condition() -> None:
    arguments = _parse_arguments(_required_arguments())

    assert arguments.diagnostic_mode is False
    assert arguments.stop_after_terminal_shots is None
    assert arguments.diagnostic_duration_seconds is None


def test_cuda_detector_enters_public_fp16_inference_mode(monkeypatch) -> None:
    calls = []

    class Model:
        def inference(self, **kwargs):
            calls.append(kwargs)

    model = Model()
    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        SimpleNamespace(from_checkpoint=lambda path: model),
    )
    float16 = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16=float16))

    loaded = _detector(Path("checkpoint.pth"), device="cuda")

    assert loaded is model
    assert calls == [{"compile": False, "inplace": True, "dtype": float16}]


def test_runtime_event_forwarder_emits_before_runtime_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = runtime_script.ProductRealtimeEventChannel(maximum_pending_events=4)
    completed = threading.Event()
    failures: list[BaseException] = []
    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(runtime_script, "_emit", lambda payload: emitted.append(payload))
    forwarder = threading.Thread(
        target=runtime_script._forward_runtime_events,
        args=(channel, completed, failures),
        daemon=True,
    )
    forwarder.start()

    channel.publish(ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-00000001"))
    deadline = time.monotonic() + 1
    while not emitted and time.monotonic() < deadline:
        time.sleep(0.001)

    assert emitted[0]["kind"] == "SHOT_STARTED"
    assert completed.is_set() is False
    completed.set()
    forwarder.join(timeout=1)
    assert not forwarder.is_alive()
    assert not failures


def test_final_drain_output_failure_is_propagated_to_cli_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = runtime_script.ProductRealtimeEventChannel(maximum_pending_events=4)
    completed = threading.Event()
    completed.set()
    failures: list[BaseException] = []
    channel.publish(ProductRealtimeEvent(kind="SHOT_STARTED", shot_id="shot-00000001"))

    def fail_output(payload: object) -> None:
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr(runtime_script, "_emit", fail_output)
    runtime_script._forward_runtime_events(channel, completed, failures)

    assert len(failures) == 1
    with pytest.raises(RuntimeError, match="stdout closed"):
        runtime_script._raise_event_forwarding_failure(failures)
    with pytest.raises(
        ProductRealtimeEventChannelFailure,
        match="event consumer failed",
    ):
        channel.receive(timeout_seconds=0.01)


def _table_frame() -> np.ndarray:
    frame = np.full((540, 960, 3), 18, dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((70, 493), (890, 502), (878, 58), (80, 72)), dtype=np.int32),
        (24, 112, 36),
    )
    return frame


def _calibration():
    return calibrate(
        "camera",
        [(70, 493), (890, 502), (878, 58), (80, 72)],
        table_length_mm=2540,
        table_width_mm=1270,
    )


def test_preflight_accepts_multiple_consistent_detections_among_startup_misses() -> None:
    missing = np.zeros((540, 960, 3), dtype=np.uint8)

    corners = _select_table_corners(
        (missing, _table_frame(), missing, _table_frame(), _table_frame()),
        _calibration(),
        minimum_confirmations=3,
    )

    assert np.allclose(corners, ((70, 493), (890, 502), (878, 58), (80, 72)), atol=3)


def test_preflight_fails_closed_without_enough_confirmed_frames() -> None:
    missing = np.zeros((540, 960, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="only 2 of 5 frames"):
        _select_table_corners(
            (_table_frame(), missing, _table_frame(), missing, missing),
            _calibration(),
            minimum_confirmations=3,
        )
