import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
BUILDER = ROOT / "scripts/diagnostics/package_raytheon_lens.py"


def test_reused_packet_imports_from_clean_extraction_and_has_no_business_data(tmp_path):
    output = tmp_path / "lens.zip"
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all(not n.endswith((".sqlite3", ".db", ".avi", ".mp4")) for n in names)
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["baseline_commit"] == "9a144955377b289ce47de71769d1305dd8585de4"
        assert manifest["acceptance"] == "NOT_EVALUATED"
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        archive.extractall(tmp_path / "clean extraction")
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path / "clean extraction/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for script in [
        "capture_mainline1_usb_lens_samples.py",
        "calibrate_mainline1_usb_lens.py",
        "probe_mainline1_usb.py",
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(tmp_path / "clean extraction/scripts/algorithm" / script),
                "--help",
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    # Never replace an already-delivered package.
    old = output.read_bytes()
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)], capture_output=True
    )
    assert result.returncode != 0
    assert output.read_bytes() == old


def test_existing_powershell_wrapper_forwards_rer_profile(tmp_path):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    fake_python = tmp_path / "python stub.py"
    fake_python.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(ROOT / "scripts/algorithm/windows/capture-mainline1-usb-lens-samples.ps1"),
            "-ProjectRoot",
            str(ROOT),
            "-PythonPath",
            str(fake_python),
            "-DeviceLocator",
            "USB\\VID_32E4&PID_6678\\fixture",
            "-RuntimeIndex",
            "0",
            "-Output",
            str(tmp_path / "new samples.json"),
            "-TargetSamples",
            "24",
            "-RerProfile",
            "-CheckerboardColumns",
            "11",
            "-CheckerboardRows",
            "8",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout.strip())
    assert args[-1] == "--rer-profile"
    assert args[args.index("--checkerboard-columns") + 1] == "11"
    assert args[args.index("--checkerboard-rows") + 1] == "8"
    assert args[args.index("--target-samples") + 1] == "24"
    assert args[args.index("--output") + 1] == str(tmp_path / "new samples.json")


@pytest.mark.parametrize("enumeration_ok", [True, False])
def test_delivered_one_liner_archives_success_and_failure(tmp_path, enumeration_ok):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    zip_path = downloads / "billiardsAI-rer-lens-reuse-20260905-v3.zip"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(zip_path)], check=True, capture_output=True
    )
    fake = tmp_path / "python with spaces"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,sys,pathlib\n"
        "if '--list' in sys.argv:\n"
        " print(json.dumps({'devices': [{'device_locator': 'USB/VID_32E4&PID_6678/fixture'}]"
        f" if {enumeration_ok!r} else []}}))\n"
        "else:\n pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n"
        " print('CAPTURED_NOT_CALIBRATED')\n"
    )
    fake.chmod(0o755)
    text = (ROOT / "docs/operations/rer-lens-reuse-20260905.md").read_text()
    command = text.split("```powershell\n")[1].split("\n```")[0]
    assert "\n" not in command
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe", str(fake)
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; $env:PYTHONPATH='original value'; "
        f"function powershell.exe {{ & '{pwsh}' @args; "
        "$global:LASTEXITCODE=$LASTEXITCODE }; "
    )
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            prefix
            + command
            + "; if($env:PYTHONPATH -ne 'original value'){throw 'Environment not restored'}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    results = list(downloads.glob("rer-lens-*/SEND_BACK-*.zip"))
    assert len(results) == 1, result.stdout + result.stderr
    with zipfile.ZipFile(results[0]) as archive:
        names = archive.namelist()
        assert any(n.endswith("devices.json") for n in names)
        assert any(n.endswith("packet-manifest.json") for n in names)
        assert any(n.endswith("lens-samples.json") for n in names) == enumeration_ok
        assert any(n.endswith("error.txt") for n in names) != enumeration_ok
