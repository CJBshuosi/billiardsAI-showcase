"""Execute the installed-service USB selection preflight without Windows task APIs."""

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


def _write_replay(path: Path) -> Path:
    path.mkdir()
    summary = json.dumps({"schema_version": "1.0", "status": "PASS", "shot_count": 1})
    (path / "summary.json").write_text(summary, encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "field_route_file_replay",
                "pipeline_version": "field-route-file-replay/2",
                "inventory": {
                    "summary.json": {
                        "bytes": len(summary),
                        "sha256": hashlib.sha256(summary.encode()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("escape", [None, "sibling-prefix", "parent", "root-itself"])
def test_check_only_selects_usb_and_preserves_installed_business_fields(
    tmp_path: Path, escape: str | None
) -> None:
    shell = os.environ.get("BILLIARDS_TEST_PWSH") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell runtime required; set BILLIARDS_TEST_PWSH")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    replay = _write_replay(tmp_path / "replay")
    existing_files = {}
    for name in ("qr.jpg", "browser.exe", "python.exe"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        existing_files[name] = path
    current = {
        "schema_version": "1.0",
        "artifact": str(replay),
        "join_qr_asset": str(existing_files["qr.jpg"]),
        "sync_base_url": "https://sync.example.test",
        "device_id": "raytheon-01",
        "credential_target": "billiardsAI/development/raytheon-01",
        "browser_path": str(existing_files["browser.exe"]),
        "python_path": str(existing_files["python.exe"]),
    }
    (runtime / "current.json").write_text(json.dumps(current), encoding="utf-8")
    capture = runtime / "capture"
    capture.mkdir()
    directories = {}
    for name in ("calibration", "detector", "identity", "dinov2"):
        path = capture / name
        path.mkdir()
        directories[name] = path
    files = {}
    for name in ("capture.exe", "capture-setup.json", "shot.yaml"):
        path = capture / name
        path.write_bytes(name.encode())
        files[name] = path
    profile = {
        "schema_version": "1.0",
        "mode": "mainline1_usb_native",
        "native_helper": str(files["capture.exe"]),
        "device_locator": r"USB\VID_32E4&PID_6678&MI_00\test",
        "device_fingerprint": "a" * 64,
        "capture_setup": str(files["capture-setup.json"]),
        "calibration_root": str(directories["calibration"]),
        "detector_training_run": str(directories["detector"]),
        "identity_training_run": str(directories["identity"]),
        "dinov2_assets": str(directories["dinov2"]),
        "shot_config": str(files["shot.yaml"]),
        "ring_root": str(runtime / "ring"),
        "jobs_root": str(runtime / "jobs"),
        "live_artifact_root": str(runtime / "live"),
        "device": "cuda",
        "batch_size": 4,
        "identity_gate_status": "FAIL",
        "runtime_environment": "development",
        "cloudbase_environment_id": "cloud1-test",
        "expected_development_environment_id": "cloud1-test",
    }
    if escape == "sibling-prefix":
        profile["ring_root"] = str(runtime.with_name(runtime.name + "-foreign") / "ring")
    elif escape == "parent":
        profile["ring_root"] = str(runtime / ".." / "foreign-ring")
    elif escape == "root-itself":
        profile["ring_root"] = str(runtime)
    profile_path = tmp_path / "capture-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    archive = tmp_path / "release.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("release-manifest.json", '{"schema_version":"1.0","version":"usb-test"}')
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
        "-CaptureProfile",
        str(profile_path),
        "-RuntimeRoot",
        str(runtime),
        "-ReleasesRoot",
        str(tmp_path / "releases"),
        "-ExpectedShotCount",
        "1",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "LOCALAPPDATA": str(tmp_path)},
        timeout=20,
        cwd=tmp_path,
    )

    if escape is not None:
        assert result.returncode != 0
        assert "Path escapes the approved root" in result.stderr
        assert not (tmp_path / "runtime-foreign").exists()
        assert not (tmp_path / "foreign-ring").exists()
        assert json.loads((runtime / "current.json").read_text()) == current
        return
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["capture_mode"] == "mainline1_usb_native"
    assert output["artifact"] == str(replay)
    assert output["credential_target"] == current["credential_target"]
