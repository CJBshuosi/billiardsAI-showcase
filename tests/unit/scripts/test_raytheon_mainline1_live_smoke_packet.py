from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_live_smoke_packet_is_side_by_side_bounded_and_hash_complete(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "calibrations" / "artifact").mkdir(parents=True)
    (candidate / "capture-setup.json").write_text("{}", encoding="utf-8")
    (candidate / "calibrations" / "registry.json").write_text("{}", encoding="utf-8")
    (candidate / "calibrations" / "artifact" / "calibration.json").write_text(
        "{}", encoding="utf-8"
    )
    target = tmp_path / "packet.zip"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_mainline1_live_smoke.py"),
            "--candidate",
            str(candidate),
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["kind"] == "mainline1_usb_live_single_shot_diagnostic"
        assert manifest["baseline_commit"].startswith("9a14495")
        assert manifest["development_evidence_only"] is True
        assert manifest["production_ready"] is False
        assert manifest["writes_installed_service_or_state_db"] is False
        assert manifest["run"]["max_frames"] == 600
        assert manifest["run"]["device_locator"] == (
            r"USB\VID_32E4&PID_6678&MI_00\6&B920475&0&0000"
        )
        assert manifest["assets"]["detector"]["checkpoint_sha256"] == (
            "d9cd5efe6c6deb8b7762e08c0f720cb264a2f41bce9174ddc77154a56f434541"
        )
        assert manifest["assets"]["identity"]["gate_status"] == "FAIL"
        assert manifest["assets"]["identity"]["development_only"] is True
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        assert "scripts/algorithm/run_mainline1_usb_live.py" in archive.namelist()
        assert "configs/algorithm/shot-local.yaml" in archive.namelist()
        assert "candidate/capture-setup.json" in archive.namelist()
        launcher = archive.read("run.ps1").decode("utf-8-sig")
        assert '[string]$OutputParent = ""' in launcher
        assert (
            "if ([string]::IsNullOrWhiteSpace($OutputParent)) { $OutputParent = $PacketRoot }"
            in launcher
        )
        native_call = launcher.index("& $PythonPath @Arguments")
        assert launcher.index("$NativeErrorPreference = $ErrorActionPreference") < native_call
        assert launcher.index('$ErrorActionPreference = "Continue"') < native_call
        assert (
            launcher.index("$ErrorActionPreference = $NativeErrorPreference", native_call)
            > native_call
        )
        assert '$_.PSObject.Properties["status"]' in launcher
        assert '$_.PSObject.Properties["kind"]' in launcher
