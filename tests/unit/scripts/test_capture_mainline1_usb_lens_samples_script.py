from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from billiards_ai.lab.mainline1_lens_calibration import LensCalibrationSample
from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity


def _module():
    path = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "capture_mainline1_usb_lens_samples.py"
    )
    spec = importlib.util.spec_from_file_location("capture_mainline1_usb_lens_samples", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_purchased_board_uses_88_corners_and_preserves_metric_spacing(tmp_path):
    import numpy as np

    from billiards_ai.lab.mainline1_lens_sample_capture import detect_checkerboard_sample

    module = _module()
    device = UsbDeviceIdentity(device_locator="fixture", display_name="4K U3 Camera")
    output = tmp_path / "new-board.json"

    def collector(**kwargs):
        spec = kwargs["spec"]
        corners = np.array(
            [[[100 + c * 20, 200 + r * 20]] for r in range(8) for c in range(11)], dtype=np.float32
        )
        sample = detect_checkerboard_sample(
            np.zeros((1080, 1920, 3), dtype=np.uint8),
            spec,
            detector=lambda _g, pattern, **_k: (pattern == (11, 8), corners),
        )
        assert sample is not None
        return (sample,) * 24

    status = module.main(
        [
            "--device-locator",
            "fixture",
            "--runtime-index",
            "0",
            "--output",
            str(output),
            "--rer-profile",
            "--checkerboard-columns",
            "11",
            "--checkerboard-rows",
            "8",
            "--target-samples",
            "24",
        ],
        enumerator=lambda: (device,),
        collector=collector,
    )
    assert status == 0
    payload = json.loads(output.read_bytes())
    assert payload["checkerboard"] == {"columns": 11, "rows": 8, "square_size_mm": 25.0}
    assert len(payload["samples"][0]["image_points"]) == 88
    assert payload["samples"][0]["object_points"][-1] == [250.0, 175.0, 0.0]


def test_rer_profile_uses_directshow_and_manual_controls(tmp_path, monkeypatch, capsys):
    import cv2

    module = _module()
    device = UsbDeviceIdentity(
        device_locator="USB\\VID_32E4&PID_6678\\fixture", display_name="4K U3 Camera"
    )
    calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda index, backend: calls.append((index, backend)))

    def collector(**kwargs):
        kwargs["capture_factory"](0)
        assert kwargs["require_mjpg"] is True
        assert callable(kwargs["roi_selector"])
        assert kwargs["evidence_directory"] == output.with_suffix(".frames")
        assert kwargs["expected_controls"][cv2.CAP_PROP_EXPOSURE] == -8
        assert kwargs["expected_controls"][cv2.CAP_PROP_AUTOFOCUS] == 2
        assert kwargs["expected_controls"][cv2.CAP_PROP_AUTO_WB] == 0
        kwargs["capture_evidence"]["measured_fps"] = 60.0
        return ()

    output = tmp_path / "samples.json"
    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output",
            str(output),
            "--rer-profile",
        ],
        enumerator=lambda: (device,),
        collector=collector,
    )
    assert status == 0
    assert calls == [(0, cv2.CAP_DSHOW)]
    payload = json.loads(output.read_bytes())
    assert payload["capture_evidence"]["measured_fps"] == 60.0
    assert payload["capture_backend"] == "DSHOW"
    assert payload["identity_binding"] == "enumeration_and_operator_confirmation_only"
    assert json.loads(capsys.readouterr().out)["status"] == "CAPTURED_NOT_CALIBRATED"


def test_rer_profile_does_not_overwrite_previous_samples(tmp_path):
    module = _module()
    device = UsbDeviceIdentity(device_locator="fixture", display_name="4K U3 Camera")
    output = tmp_path / "samples.json"
    output.write_text("old evidence")
    status = module.main(
        [
            "--device-locator",
            "fixture",
            "--runtime-index",
            "0",
            "--output",
            str(output),
            "--rer-profile",
        ],
        enumerator=lambda: (device,),
        collector=lambda **_: (),
    )
    assert status == 2
    assert output.read_text() == "old evidence"


def test_cli_writes_samples_for_exact_present_device(tmp_path: Path, capsys) -> None:
    module = _module()
    output = tmp_path / "samples.json"
    device = UsbDeviceIdentity(
        device_locator=r"USB\VID_A000&PID_B111&MI_00\fixture",
        display_name="FIT1044",
        vendor_id="A000",
        product_id="B111",
    )
    sample = LensCalibrationSample(
        object_points=((0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (0.0, 25.0, 0.0), (25.0, 25.0, 0.0)),
        image_points=((10.0, 20.0), (30.0, 20.0), (10.0, 40.0), (30.0, 40.0)),
    )
    observed = {}

    def collector(**kwargs):
        observed.update(kwargs)
        return (sample,) * 12

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output",
            str(output),
        ],
        enumerator=lambda: (device,),
        collector=collector,
    )

    assert status == 0
    assert observed["runtime_index"] == 0
    assert observed["target_samples"] == 12
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["width"] == 1920
    assert stored["height"] == 1080
    assert stored["runtime_index"] == 0
    assert stored["device_fingerprint"] == device.durable_fingerprint
    assert stored["checkerboard"] == {"columns": 9, "rows": 6, "square_size_mm": 25.0}
    assert len(stored["samples"]) == 12
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_cli_fails_closed_when_configured_device_is_missing(tmp_path: Path, capsys) -> None:
    module = _module()
    output = tmp_path / "samples.json"

    status = module.main(
        [
            "--device-locator",
            r"USB\VID_A000&PID_B111&MI_00\missing",
            "--runtime-index",
            "0",
            "--output",
            str(output),
        ],
        enumerator=lambda: (),
    )

    assert status == 2
    assert not output.exists()
    assert "not present" in capsys.readouterr().err


def test_cli_allows_eight_samples_as_the_minimum(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "samples.json"
    device = UsbDeviceIdentity(
        device_locator=r"USB\VID_A000&PID_B111&MI_00\fixture",
        display_name="FIT1044",
        vendor_id="A000",
        product_id="B111",
    )
    sample = LensCalibrationSample(
        object_points=((0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (0.0, 25.0, 0.0), (25.0, 25.0, 0.0)),
        image_points=((10.0, 20.0), (30.0, 20.0), (10.0, 40.0), (30.0, 40.0)),
    )

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output",
            str(output),
            "--target-samples",
            "8",
        ],
        enumerator=lambda: (device,),
        collector=lambda **_kwargs: (sample,) * 8,
    )

    assert status == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))["samples"]) == 8


def test_cli_rejects_ambiguous_present_camera_set(tmp_path: Path, capsys) -> None:
    module = _module()
    output = tmp_path / "samples.json"
    first = UsbDeviceIdentity(
        device_locator=r"USB\VID_A000&PID_B111&MI_00\first",
        display_name="FIT1044",
        vendor_id="A000",
        product_id="B111",
    )
    second = UsbDeviceIdentity(
        device_locator=r"USB\VID_2222&PID_3333&MI_00\second",
        display_name="Other camera",
        vendor_id="2222",
        product_id="3333",
    )

    status = module.main(
        [
            "--device-locator",
            first.device_locator,
            "--runtime-index",
            "0",
            "--output",
            str(output),
        ],
        enumerator=lambda: (first, second),
    )

    assert status == 2
    assert not output.exists()
    assert "exactly one present camera" in capsys.readouterr().err


def test_cli_removes_stale_output_before_failed_collection(tmp_path: Path, capsys) -> None:
    module = _module()
    output = tmp_path / "samples.json"
    output.write_text("stale", encoding="utf-8")
    device = UsbDeviceIdentity(
        device_locator=r"USB\VID_A000&PID_B111&MI_00\fixture",
        display_name="FIT1044",
        vendor_id="A000",
        product_id="B111",
    )

    def collector(**_kwargs):
        raise module.LensSampleCaptureError("cancelled")

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output",
            str(output),
        ],
        enumerator=lambda: (device,),
        collector=collector,
    )

    assert status == 2
    assert not output.exists()
    assert "cancelled" in capsys.readouterr().err


def test_cli_has_no_overrides_for_fixed_capture_or_checkerboard_contract(
    tmp_path: Path, capsys
) -> None:
    module = _module()

    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--device-locator",
                "fixture",
                "--runtime-index",
                "0",
                "--output",
                str(tmp_path / "samples.json"),
                "--width",
                "1280",
            ]
        )

    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
