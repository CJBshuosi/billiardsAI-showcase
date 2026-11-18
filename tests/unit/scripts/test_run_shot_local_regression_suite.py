from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab.mainline1_calibration_gate import CalibrationGateStatus
from billiards_ai.lab.shot_local.regression_suite import (
    RegressionCalibrationBlocked,
    RegressionSuiteArtifact,
)


def _module():
    path = (
        Path(__file__).parents[3] / "scripts" / "algorithm" / "run_shot_local_regression_suite.py"
    )
    spec = importlib.util.spec_from_file_location("run_shot_local_regression_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_returns_zero_and_prints_pass_artifact(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text("{}")
    output = tmp_path / "output"
    calls = []

    def runner(manifest_path, output_root, calibration_root, product_runner):
        calls.append((manifest_path, output_root, calibration_root, product_runner))
        return RegressionSuiteArtifact(output / "artifact", "abc123", "PASS", 3)

    module = _module()
    assert (
        module.main(
            ["--manifest", str(manifest), "--output-root", str(output)],
            suite_runner=runner,
            product_runner="product-runner",
        )
        == 0
    )
    assert calls == [(manifest, output, output / "calibrations", "product-runner")]
    assert json.loads(capsys.readouterr().out) == {
        "artifact_id": "abc123",
        "case_count": 3,
        "output": str(output / "artifact"),
        "status": "PASS",
    }


def test_cli_returns_one_for_completed_quality_gate_failure(tmp_path: Path) -> None:
    module = _module()

    def runner(*_):
        return RegressionSuiteArtifact(tmp_path / "artifact", "failed", "FAIL", 2)

    assert (
        module.main(
            ["--manifest", str(tmp_path / "suite.json"), "--output-root", str(tmp_path)],
            suite_runner=runner,
            product_runner="product-runner",
        )
        == 1
    )


def test_cli_returns_two_for_execution_error(tmp_path: Path, capsys) -> None:
    module = _module()

    def runner(*_):
        raise ValueError("bad suite")

    assert (
        module.main(
            ["--manifest", str(tmp_path / "suite.json"), "--output-root", str(tmp_path)],
            suite_runner=runner,
            product_runner="product-runner",
        )
        == 2
    )
    assert "bad suite" in capsys.readouterr().err


def test_cli_accepts_explicit_calibration_root(tmp_path: Path) -> None:
    module = _module()
    seen = []

    def runner(manifest, output, calibration_root, product_runner):
        seen.append((manifest, output, calibration_root, product_runner))
        return RegressionSuiteArtifact(output / "artifact", "ready", "PASS", 1)

    manifest = tmp_path / "suite.json"
    calibration_root = tmp_path / "registered-calibrations"
    assert (
        module.main(
            [
                "--manifest",
                str(manifest),
                "--output-root",
                str(tmp_path / "output"),
                "--calibration-root",
                str(calibration_root),
            ],
            suite_runner=runner,
            product_runner="product-runner",
        )
        == 0
    )
    assert seen[0][2] == calibration_root


def test_default_product_runner_stops_before_product_when_calibration_is_blocked(
    tmp_path: Path,
) -> None:
    module = _module()
    product_calls = []
    runner = module._product_runner(
        source_probe=lambda path: f"source:{path.name}",
        setup_loader=lambda path: f"setup:{path.name}",
        gate_evaluator=lambda **_: SimpleNamespace(
            status=CalibrationGateStatus.CALIBRATION_REQUIRED,
            capture_setup_id="setup-a",
            setup_fingerprint="f" * 64,
            reasons=("calibration_not_registered",),
            calibration_artifact_id=None,
            calibration_revision=None,
            validation_mode=None,
            maximum_corner_drift_px=None,
        ),
        config_loader=lambda path: f"config:{path.name}",
        cached_runner=lambda **kwargs: product_calls.append(kwargs),
    )

    with pytest.raises(RegressionCalibrationBlocked) as captured:
        runner(
            video=tmp_path / "video.mp4",
            source_artifact=tmp_path / "source",
            capture_setup=tmp_path / "setup.json",
            calibration_root=tmp_path / "calibrations",
            config_path=tmp_path / "config.yaml",
            output_root=tmp_path / "output",
        )

    assert captured.value.status == "CALIBRATION_REQUIRED"
    assert product_calls == []
