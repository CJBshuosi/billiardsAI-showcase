from __future__ import annotations

from pathlib import Path


def test_windows_entrypoint_exposes_manifest_output_and_exit_code() -> None:
    script = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "windows"
        / "run-shot-local-regression-suite.ps1"
    ).read_text(encoding="utf-8")

    assert "[Parameter(Mandatory = $true)][string]$Manifest" in script
    assert "[Parameter(Mandatory = $true)][string]$OutputRoot" in script
    assert "[string]$CalibrationRoot" in script
    assert '"--calibration-root"' in script
    assert "run_shot_local_regression_suite.py" in script
    assert "$LASTEXITCODE" in script
