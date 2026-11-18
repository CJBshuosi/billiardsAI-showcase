from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureErrorCode,
    UsbCaptureReport,
    UsbDeviceIdentity,
)


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "probe_mainline1_usb.py"
    spec = importlib.util.spec_from_file_location("probe_mainline1_usb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> UsbDeviceIdentity:
    return UsbDeviceIdentity(
        device_locator=r"USB\VID_32E4&PID_9230\FIT1044-A",
        display_name="DFRobot FIT1044",
        vendor_id="32e4",
        product_id="9230",
        serial_number="FIT1044-A",
    )


def test_list_emits_durable_identity_without_inventing_runtime_index(capsys) -> None:
    module = _module()

    exit_code = module.main(["--list"], enumerator=lambda: (_device(),))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "devices": [
            {
                "device_fingerprint": _device().durable_fingerprint,
                "device_locator": _device().device_locator,
                "display_name": "DFRobot FIT1044",
                "product_id": "9230",
                "serial_number": "FIT1044-A",
                "vendor_id": "32e4",
            }
        ],
        "schema_version": "1.0",
        "status": "PASS",
    }


def test_probe_binds_ephemeral_index_and_prints_compact_ready_report(capsys) -> None:
    module = _module()
    seen = {}

    def prober(request):
        seen["request"] = request
        return UsbCaptureReport(
            status=UsbCaptureErrorCode.READY,
            device_fingerprint=request.device.durable_fingerprint,
            requested_width=1920,
            requested_height=1080,
            requested_fps=60,
            observed_width=1920,
            observed_height=1080,
            observed_fps=60,
            measured_fps=60,
            delivered_frames=120,
            failed_reads=0,
            timestamp_reversals=0,
            mean_frame_gap_ms=16.67,
            maximum_frame_gap_ms=18.2,
            elapsed_ms=2000,
            backend_name="DSHOW",
        )

    exit_code = module.main(
        [
            "--device-locator",
            _device().device_locator,
            "--runtime-index",
            "3",
        ],
        enumerator=lambda: (_device(),),
        prober=prober,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["observed_fps"] == 60.0
    assert payload["measured_fps"] == 60.0
    assert seen["request"].device.runtime_index == 3
    assert seen["request"].sample_frames == 120


def test_probe_returns_two_when_camera_mode_is_not_exact(capsys) -> None:
    module = _module()

    def prober(request):
        return UsbCaptureReport(
            status=UsbCaptureErrorCode.CAPTURE_GEOMETRY_MISMATCH,
            device_fingerprint=request.device.durable_fingerprint,
            requested_width=1920,
            requested_height=1080,
            requested_fps=60,
            observed_width=1920,
            observed_height=1080,
            observed_fps=30,
            delivered_frames=0,
            failed_reads=0,
            timestamp_reversals=0,
            elapsed_ms=0,
            backend_name="DSHOW",
            reasons=("capture mode did not match the exact requested geometry",),
        )

    exit_code = module.main(
        [
            "--device-locator",
            _device().device_locator,
            "--runtime-index",
            "2",
        ],
        enumerator=lambda: (_device(),),
        prober=prober,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "capture_geometry_mismatch"
    assert payload["observed_fps"] == 30.0


def test_probe_refuses_locator_not_present_in_current_enumeration(capsys) -> None:
    module = _module()

    exit_code = module.main(
        ["--device-locator", "missing", "--runtime-index", "0"],
        enumerator=lambda: (_device(),),
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload == {
        "error": "configured USB device locator is not present",
        "schema_version": "1.0",
        "status": "camera_not_found",
    }
