"""Execute installer preflight; Windows task APIs are not needed for -CheckOnly."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _artifact(path: Path, *, count: int = 1, version: int = 2) -> Path:
    path.mkdir()
    data = json.dumps({"schema_version": "1.0", "status": "PASS", "shot_count": count})
    (path / "summary.json").write_text(data)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "field_route_file_replay",
                "pipeline_version": f"field-route-file-replay/{version}",
                "inventory": {
                    "summary.json": {
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data.encode()).hexdigest(),
                    }
                },
            }
        )
    )
    return path


def _preflight(tmp_path: Path, *, active: str | None, count: int = 0, tamper=False):
    shell = os.environ.get("BILLIARDS_TEST_PWSH") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell runtime required; set BILLIARDS_TEST_PWSH")
    asset = tmp_path / "asset"
    asset.write_bytes(b"placeholder: CheckOnly verifies existence only")
    bundled = _artifact(tmp_path / "bundled", count=count)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    if active is not None:
        current = _artifact(tmp_path / "active", version=1 if active == "legacy" else 2)
        if tamper:
            (current / "summary.json").write_text("tampered")
        (runtime / "current.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "artifact": str(current),
                    "join_qr_asset": str(asset),
                    "browser_path": str(asset),
                    "python_path": "",
                    "device_id": "raytheon-01",
                    "sync_base_url": "https://sync.example.test",
                    "credential_target": "billiardsAI/development/raytheon-01",
                }
            )
        )
    archive = tmp_path / "release.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("release-manifest.json", '{"schema_version":"1.0","version":"test"}')
    command = [
        shell,
        "-NoProfile",
        "-File",
        str(ROOT / "scripts/algorithm/windows/install-field-route-service.ps1"),
        "-CheckOnly",
        "-ReleaseArchive",
        str(archive),
        "-ExpectedSha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "-Artifact",
        str(bundled),
        "-JoinQrAsset",
        str(asset),
        "-BrowserPath",
        str(asset),
        "-RuntimeRoot",
        str(runtime),
        "-ReleasesRoot",
        str(tmp_path / "releases"),
        "-SyncBaseUrl",
        "https://sync.example.test",
        "-DeviceId",
        "raytheon-01",
        "-CredentialTarget",
        "billiardsAI/development/raytheon-01",
        "-ExpectedShotCount",
        "1",
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "LOCALAPPDATA": str(tmp_path)},
        timeout=20,
        cwd=tmp_path,
    )


def test_upgrade_preserves_valid_active_replay_instead_of_zero_shot_bundle(tmp_path):
    result = _preflight(tmp_path, active="v2")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["artifact"] == str(tmp_path / "active")
    assert output["shot_count"] == 1


def test_upgrade_does_not_silently_replace_tampered_active_replay(tmp_path):
    result = _preflight(tmp_path, active="v2", count=1, tamper=True)
    assert result.returncode != 0
    assert "mismatch" in result.stderr


def test_first_install_cannot_claim_one_shot_with_zero_shot_bundle(tmp_path):
    result = _preflight(tmp_path, active=None)
    assert result.returncode != 0
    assert "shot_count" in result.stderr


def test_upgrade_requires_explicit_refresh_of_legacy_active_replay(tmp_path):
    result = _preflight(tmp_path, active="legacy", count=1)
    assert result.returncode != 0
    assert "v2" in result.stderr
