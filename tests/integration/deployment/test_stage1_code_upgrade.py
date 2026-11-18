from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("legacy", "refresh_exit", "preflight_exit"),
    [(True, 0, 0), (True, 2, 0), (False, 0, 0), (False, 0, 2)],
)
@pytest.mark.parametrize("bundle_argument", ["explicit", "omitted", "empty"])
def test_code_upgrade_reuses_input_and_stops_on_failure(
    tmp_path, legacy, refresh_exit, preflight_exit, bundle_argument
):
    shell = os.environ.get("BILLIARDS_TEST_PWSH") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell runtime required")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    artifact = tmp_path / "legacy"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps({"pipeline_version": f"field-route-file-replay/{1 if legacy else 2}"})
    )
    (artifact / "summary.json").write_text(json.dumps({"shot_count": 0 if legacy else 1}))
    configuration = {
        "artifact": str(artifact),
        "current_release": str(tmp_path / "old-release"),
        "python_path": "python.exe",
        "join_qr_asset": "existing-qr.png",
        "sync_base_url": "https://sync.example.test",
        "device_id": "raytheon-01",
        "credential_target": "billiardsAI/development/raytheon-01",
        "browser_path": "browser.exe",
    }
    (runtime / "current.json").write_text(json.dumps(configuration))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = bundle / "upgrade-stage1-code.ps1"
    shutil.copyfile(ROOT / "scripts/algorithm/windows/upgrade-stage1-code.ps1", script)
    archive = bundle / "release.zip"
    archive.write_bytes(b"archive boundary stub")
    (bundle / "upgrade-receipt.json").write_text(
        json.dumps(
            {
                "release": "release.zip",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )
    # Only external child processes are substituted; the real orchestration script runs.
    harness = tmp_path / "harness.ps1"
    harness.write_text("""
param($Script, $Bundle, $Runtime, [int]$RefreshExit, [int]$PreflightExit, $BundleArgument)
$ErrorActionPreference = 'Stop'
function powershell.exe {
  $arguments = @($args)
  if ($arguments -match 'refresh-video43-field-replay.ps1') {
    Write-Output 'REFRESH'
    if ($RefreshExit -eq 0) {
      $pointer = Join-Path $Runtime 'current.json'
      $current = Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json
      $current.artifact = Join-Path $Runtime 'refreshed'
      $current | ConvertTo-Json | Set-Content -LiteralPath $pointer
    }
    $global:LASTEXITCODE = $RefreshExit
  } else {
    Write-Output ('INSTALL_ARGS=' + ($arguments | ConvertTo-Json -Compress))
    $global:LASTEXITCODE = if ($arguments -contains '-CheckOnly') { $PreflightExit } else { 0 }
  }
}
switch ($BundleArgument) {
  'explicit' { & $Script -BundleRoot $Bundle -RuntimeRoot $Runtime }
  'omitted' { & $Script -RuntimeRoot $Runtime }
  'empty' { & $Script -BundleRoot '' -RuntimeRoot $Runtime }
}
""")
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-File",
            str(harness),
            str(script),
            str(bundle),
            str(runtime),
            str(refresh_exit),
            str(preflight_exit),
            bundle_argument,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "LOCALAPPDATA": str(tmp_path)},
        timeout=20,
        cwd=tmp_path,
    )
    assert ("REFRESH" in result.stdout) is legacy, result.stderr
    if refresh_exit:
        assert result.returncode != 0
        assert "INSTALL_ARGS" not in result.stdout
    else:
        assert (result.returncode != 0) is bool(preflight_exit), result.stderr
        assert result.stdout.count("INSTALL_ARGS=") == (1 if preflight_exit else 2)
        arguments = json.loads(
            next(
                line.removeprefix("INSTALL_ARGS=")
                for line in result.stdout.splitlines()
                if line.startswith("INSTALL_ARGS=")
            )
        )
        assert arguments[arguments.index("-Artifact") + 1] == str(
            runtime / "refreshed" if legacy else artifact
        )
        assert arguments[arguments.index("-JoinQrAsset") + 1] == "existing-qr.png"
        assert arguments[arguments.index("-ExpectedShotCount") + 1] == 1
