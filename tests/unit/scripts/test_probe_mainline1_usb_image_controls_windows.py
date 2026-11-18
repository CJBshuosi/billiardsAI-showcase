from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
WRAPPER = ROOT / "scripts" / "algorithm" / "windows" / "probe-mainline1-usb-image-controls.ps1"


def test_wrapper_forwards_exact_camera_output_and_explicit_controls() -> None:
    source = WRAPPER.read_text(encoding="ascii")

    assert "probe_mainline1_usb_image_controls.py" in source
    assert '"--device-locator", $DeviceLocator' in source
    assert '"--runtime-index", $RuntimeIndex' in source
    assert '"--output-dir", $OutputDir' in source
    assert '"--width", "1920"' in source
    assert '"--height", "1080"' in source
    assert '"--fps", "60"' in source
    assert '"--control", $setting' in source
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
            f"[void][scriptblock]::Create((Get-Content -Raw -LiteralPath '{WRAPPER}'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
