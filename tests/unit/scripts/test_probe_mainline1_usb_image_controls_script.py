from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity
from billiards_ai.lab.mainline1_usb_image_controls import (
    UsbFrameQuality,
    UsbImageControlObservation,
    UsbImageControlsReport,
    UsbImageControlsStatus,
)


def _module():
    path = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "probe_mainline1_usb_image_controls.py"
    )
    spec = importlib.util.spec_from_file_location("probe_mainline1_usb_image_controls", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> UsbDeviceIdentity:
    return UsbDeviceIdentity(
        device_locator=r"USB\VID_A000&PID_B111&MI_00\fixture",
        display_name="FIT1044",
        vendor_id="A000",
        product_id="B111",
    )


def _report(device: UsbDeviceIdentity) -> UsbImageControlsReport:
    return UsbImageControlsReport(
        status="PASS",
        device_fingerprint=device.durable_fingerprint,
        backend_name="MSMF",
        observed_width=1920,
        observed_height=1080,
        observed_fps=60.0,
        control_mode="explicit_write",
        controls={
            "exposure": UsbImageControlObservation(
                before=-5.0,
                requested=-7.0,
                after=-7.0,
                setter_accepted=True,
                status="applied",
            )
        },
        quality=UsbFrameQuality(
            width=2,
            height=2,
            luminance_mean=100.0,
            luminance_p05=10.0,
            luminance_p50=100.0,
            luminance_p95=240.0,
            clipped_black_ratio=0.0,
            clipped_white_ratio=0.0,
        ),
    )


def test_cli_writes_atomic_json_and_baseline_image_for_exact_device(tmp_path: Path, capsys) -> None:
    module = _module()
    device = _device()
    output = tmp_path / "controls"
    observed = {}

    def prober(request):
        observed["request"] = request
        return _report(device), np.zeros((2, 2, 3), dtype=np.uint8)

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(output),
            "--control",
            "exposure=-7",
        ],
        enumerator=lambda: (device,),
        prober=prober,
        image_writer=lambda path, _frame: path.write_bytes(b"jpeg") or True,
    )

    assert status == 0
    assert output.joinpath("baseline.jpg").read_bytes() == b"jpeg"
    stored = json.loads(output.joinpath("controls.json").read_text(encoding="utf-8"))
    assert stored["status"] == "PASS"
    assert stored["baseline_image"] == "baseline.jpg"
    assert observed["request"].settings[0].name == "exposure"
    assert observed["request"].settings[0].value == -7.0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_cli_rejects_unknown_control_without_opening_camera(tmp_path: Path, capsys) -> None:
    module = _module()
    device = _device()

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(tmp_path / "controls"),
            "--control",
            "magic=1",
        ],
        enumerator=lambda: (device,),
        prober=lambda _request: (_ for _ in ()).throw(AssertionError("must not probe")),
    )

    assert status == 2
    assert "unknown image control" in capsys.readouterr().err


def test_cli_refuses_to_mix_new_evidence_with_existing_directory(tmp_path: Path, capsys) -> None:
    module = _module()
    device = _device()
    output = tmp_path / "controls"
    output.mkdir()

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(output),
        ],
        enumerator=lambda: (device,),
    )

    assert status == 2
    assert "already exists" in capsys.readouterr().err


def test_explicit_writes_require_one_camera_and_runtime_index_zero(tmp_path: Path, capsys) -> None:
    module = _module()
    device = _device()
    second = device.model_copy(update={"device_locator": r"USB\VID_2222&PID_3333&MI_00\second"})

    many_status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(tmp_path / "many"),
            "--control",
            "exposure=-7",
        ],
        enumerator=lambda: (device, second),
        prober=lambda _request: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    index_status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "1",
            "--output-dir",
            str(tmp_path / "index"),
            "--control",
            "exposure=-7",
        ],
        enumerator=lambda: (device,),
        prober=lambda _request: (_ for _ in ()).throw(AssertionError("must not probe")),
    )

    assert many_status == 2
    assert index_status == 2
    errors = capsys.readouterr().err
    assert "exactly one present camera" in errors
    assert "runtime index 0" in errors


def test_read_only_probe_remains_available_with_multiple_cameras(
    tmp_path: Path,
) -> None:
    module = _module()
    device = _device()
    second = device.model_copy(update={"device_locator": r"USB\VID_2222&PID_3333&MI_00\second"})
    output = tmp_path / "read-only"

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(output),
        ],
        enumerator=lambda: (device, second),
        prober=lambda _request: (_report(device), np.zeros((2, 2, 3), dtype=np.uint8)),
        image_writer=lambda path, _frame: path.write_bytes(b"jpeg") or True,
    )

    assert status == 0
    assert output.joinpath("controls.json").exists()


def test_control_write_failure_keeps_evidence_but_returns_failure(
    tmp_path: Path,
) -> None:
    module = _module()
    device = _device()
    output = tmp_path / "rejected"
    report = _report(device).model_copy(
        update={"status": UsbImageControlsStatus.CONTROL_WRITE_FAILED}
    )

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(output),
            "--control",
            "exposure=-7",
        ],
        enumerator=lambda: (device,),
        prober=lambda _request: (report, np.zeros((2, 2, 3), dtype=np.uint8)),
        image_writer=lambda path, _frame: path.write_bytes(b"jpeg") or True,
    )

    assert status == 2
    assert output.joinpath("baseline.jpg").read_bytes() == b"jpeg"
    stored = json.loads(output.joinpath("controls.json").read_text(encoding="utf-8"))
    assert stored["status"] == "control_write_failed"


def test_opencv_failure_cleans_new_output_directory(tmp_path: Path, capsys) -> None:
    module = _module()
    device = _device()
    output = tmp_path / "opencv-failure"

    status = module.main(
        [
            "--device-locator",
            device.device_locator,
            "--runtime-index",
            "0",
            "--output-dir",
            str(output),
        ],
        enumerator=lambda: (device,),
        prober=lambda _request: (_ for _ in ()).throw(cv2.error("camera failure")),
    )

    assert status == 2
    assert not output.exists()
    assert "camera failure" in capsys.readouterr().err
