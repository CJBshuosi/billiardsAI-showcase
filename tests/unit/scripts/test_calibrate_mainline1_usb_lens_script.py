from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from billiards_ai.lab.mainline1_lens_calibration import LensCalibrationResult
from billiards_ai.shared.models.calibration import (
    DistortionModel,
    DistortionParameters,
)


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "calibrate_mainline1_usb_lens.py"
    spec = importlib.util.spec_from_file_location("calibrate_mainline1_usb_lens", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_verified_distortion_report(tmp_path: Path, capsys) -> None:
    module = _module()
    samples = tmp_path / "samples.json"
    output = tmp_path / "distortion.json"
    samples.write_text(
        json.dumps(
            {
                "height": 1080,
                "samples": [
                    {
                        "image_points": [[10, 20], [30, 20], [30, 40], [10, 40]],
                        "object_points": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                    }
                ],
                "width": 1920,
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    def calibrator(**kwargs):
        observed.update(kwargs)
        return LensCalibrationResult(
            distortion=DistortionParameters(
                model=DistortionModel.OPENCV,
                verified=True,
                camera_matrix=(
                    (900.0, 0.0, 960.0),
                    (0.0, 900.0, 540.0),
                    (0.0, 0.0, 1.0),
                ),
                coefficients=(-0.2, 0.04, 0.0, 0.0, 0.0),
            ),
            rms_reprojection_error=0.42,
            sample_count=8,
            width=1920,
            height=1080,
        )

    status = module.main(
        ["--samples", str(samples), "--output", str(output)],
        calibrator=calibrator,
    )

    assert status == 0
    assert observed["width"] == 1920
    assert observed["height"] == 1080
    assert len(observed["samples"]) == 1
    stored = json.loads(output.read_text())
    assert stored["distortion"]["model"] == "OPENCV"
    assert stored["sample_count"] == 8
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_cli_fails_closed_for_invalid_sample_payload(tmp_path: Path, capsys) -> None:
    module = _module()
    samples = tmp_path / "samples.json"
    samples.write_text("{}", encoding="utf-8")

    status = module.main(["--samples", str(samples), "--output", str(tmp_path / "out.json")])

    assert status == 2
    assert "failed" in capsys.readouterr().err.lower()
