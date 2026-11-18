from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
WRAPPER = ROOT / "scripts" / "algorithm" / "windows" / "probe-mainline1-usb.ps1"


def test_wrapper_forwards_exact_mode_and_uses_project_python_entrypoint() -> None:
    source = WRAPPER.read_text(encoding="ascii")

    assert "[Parameter(Mandatory = $true)] [string]$DeviceLocator" in source
    assert "[Parameter(Mandatory = $true)] [int]$RuntimeIndex" in source
    assert '"--width", "1920"' in source
    assert '"--height", "1080"' in source
    assert '"--fps", "60"' in source
    assert '"--sample-frames", $SampleFrames' in source
    assert "[ValidateRange(1, 120000)]" in source
    assert "probe_mainline1_usb.py" in source
    assert "Start-Process" not in source


def test_wrapper_parses_on_available_powershell() -> None:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("pwsh is unavailable; Raytheon execution remains a hardware gate")

    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-Command",
            (f"[void][scriptblock]::Create((Get-Content -Raw -LiteralPath '{WRAPPER}'))"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
