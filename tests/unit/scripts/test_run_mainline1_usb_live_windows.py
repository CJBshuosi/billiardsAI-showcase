from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_wrapper_forwards_durable_device_and_live_roots() -> None:
    source = Path("scripts/algorithm/windows/run-mainline1-usb-live.ps1").read_text(
        encoding="utf-8"
    )

    assert "run_mainline1_usb_live.py" in source
    assert "--device-locator" in source
    assert "--runtime-index" in source
    assert "--spool-root" in source
    assert "--output-root" in source
    assert "--capture-setup" in source
    assert "RuntimeIndex -ne 0" in source
    assert "exactly one present camera" in source


def test_product_wrapper_requires_signed_hash_bound_native_helper() -> None:
    source = Path("scripts/algorithm/windows/run-mainline1-product-realtime.ps1").read_text(
        encoding="utf-8"
    )

    assert "run_mainline1_product_realtime.py" in source
    assert "Get-FileHash" in source
    assert "Get-AuthenticodeSignature" in source
    assert "SignatureStatus]::Valid" in source
    assert "SignerCertificate.Subject -ne $NativeCaptureHelperSigner" in source
    assert '"--device-locator"' in source
    assert '"--device-fingerprint"' in source
    assert '"--native-helper"' in source


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is unavailable")
def test_wrapper_parses_under_powershell() -> None:
    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "'scripts/algorithm/windows/run-mainline1-usb-live.ps1',[ref]$null,[ref]$null)"
            "| Out-Null",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is unavailable")
def test_product_wrapper_parses_under_powershell() -> None:
    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "'scripts/algorithm/windows/run-mainline1-product-realtime.ps1',"
            "[ref]$null,[ref]$null)| Out-Null",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
