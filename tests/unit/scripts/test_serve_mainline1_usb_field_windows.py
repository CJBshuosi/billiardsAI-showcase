import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/algorithm/windows/serve-mainline1-usb-field.ps1"
)


def test_wrapper_has_literal_path_trust_and_owned_process_contract():
    assert SCRIPT.is_file(), "live service Windows wrapper is missing"
    source = SCRIPT.read_text()
    assert "Get-FileHash -LiteralPath" in source and "-Algorithm SHA256" in source
    assert "Get-AuthenticodeSignature -LiteralPath" in source
    assert "SignatureStatus]::Valid" in source
    assert "SignerCertificate.Subject -ne $NativeHelperSigner" in source
    assert "ArgumentList.Add" in source and "$RunnerArguments = @(" in source
    assert "RedirectStandardError = $true" in source
    assert "StandardError.ReadLineAsync" in source
    assert "[Console]::Error.WriteLine" in source
    assert "finally" in source and ".Kill($true)" in source
    assert "--camera-index" not in source and "--runtime-index" not in source
    for line in source.splitlines():
        if "Test-Path " in line or "Resolve-Path " in line:
            assert "-LiteralPath" in line


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell unavailable on macOS")
def test_wrapper_parses_without_powershell_errors():
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "$errors=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile('"
            + str(SCRIPT)
            + "',[ref]$null,[ref]$errors)|Out-Null;"
            "if($errors){$errors|Out-String|Write-Error;exit 1}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
