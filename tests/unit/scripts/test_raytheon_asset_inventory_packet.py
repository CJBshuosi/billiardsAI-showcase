import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/diagnostics/raytheon_inventory/inventory_check.py"


@pytest.fixture
def module():
    assert SCRIPT.exists(), "Asset inventory diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("inventory_check", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inventory_finds_runtime_assets_and_exports_no_manifest_secrets(module, tmp_path):
    detector = tmp_path / "algorithm-artifacts/rfdetr-training-runs/run-a"
    detector.mkdir(parents=True)
    checkpoint = detector / "checkpoint_best.pth"
    checkpoint.write_bytes(b"detector")
    (detector / "training-manifest.json").write_text(
        json.dumps(
            {
                "run_hash": "run-a",
                "model_class": "RFDETRSegSmall",
                "status": "PASS",
                "api_token": "TOP_SECRET",
                "artifacts": {"checkpoint_best.pth": sha(checkpoint)},
            }
        )
    )
    identity = tmp_path / "algorithm-artifacts/identity-group-training-runs/run-b"
    identity.mkdir(parents=True)
    (identity / "group-head.onnx").write_bytes(b"identity")
    (identity / "metrics.json").write_text('{"private_player_id":"SECRET_PLAYER"}')
    (identity / "manifest.json").write_text(
        json.dumps({"run_id": "run-b", "status": "PASS", "credential": "SECRET"})
    )
    dinov2 = tmp_path / "approved-assets/dinov2-small-ed25f3a"
    dinov2.mkdir(parents=True)
    (dinov2 / "model.safetensors").write_bytes(b"features")
    (dinov2 / "config.json").write_text('{"hidden_size":384}')
    (dinov2 / "manifest.json").write_text(
        json.dumps({"source": "facebook/dinov2-small", "revision": "ed25f3a", "secret": "X"})
    )
    capture = tmp_path / "project/private-data/capture-setups"
    capture.mkdir(parents=True)
    (capture / "table-a-rer-usb.json").write_text(
        json.dumps({"capture_setup_id": "table-a-rer", "access_token": "DO_NOT_EXPORT"})
    )
    calibration = tmp_path / "algorithm-artifacts/mainline1-calibrations/table-a"
    calibration.mkdir(parents=True)
    (calibration / "calibration.json").write_text(
        json.dumps({"schema_version": "1.0", "status": "READY", "player": "PRIVATE"})
    )
    config = tmp_path / "project/configs/algorithm"
    config.mkdir(parents=True)
    (config / "shot-local.yaml").write_text("secret: not_exported")

    result = module.scan_roots(
        {
            "project": tmp_path / "project",
            "algorithm_artifacts": tmp_path / "algorithm-artifacts",
            "approved_assets": tmp_path / "approved-assets",
        }
    )

    by_kind = {}
    for item in result["candidates"]:
        by_kind.setdefault(item["kind"], []).append(item)
    assert by_kind["detector_checkpoint"][0]["sha256"] == sha(checkpoint)
    assert by_kind["detector_manifest"][0]["metadata"] == {
        "model_class": "RFDETRSegSmall",
        "run_hash": "run-a",
        "status": "PASS",
    }
    assert by_kind["identity_head"][0]["bytes"] == len(b"identity")
    assert by_kind["dinov2_weights"][0]["bytes"] == len(b"features")
    assert by_kind["capture_setup"][0]["metadata"] == {"capture_setup_id": "table-a-rer"}
    assert by_kind["calibration_json"][0]["metadata"] == {
        "schema_version": "1.0",
        "status": "READY",
    }
    assert by_kind["shot_local_config"][0]["sha256"] == sha(config / "shot-local.yaml")
    exported = json.dumps(result)
    for secret in ("TOP_SECRET", "SECRET_PLAYER", "DO_NOT_EXPORT", "not_exported"):
        assert secret not in exported


def test_inventory_skips_symlinks_and_fails_closed_on_limit(module, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "checkpoint_best.pth").write_bytes(b"private")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    result = module.scan_roots({"project": root})
    assert result["candidates"] == []
    assert result["skipped_links"] == [str(root / "linked")]
    for index in range(3):
        (root / f"capture-setup-{index}.json").write_text("{}")
    with pytest.raises(ValueError, match="limit"):
        module.scan_roots({"project": root}, maximum_candidates=2)


def test_inventory_rejects_root_below_linked_ancestor_and_signed_source_metadata(module, tmp_path):
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    (nested / "capture-setup.json").write_text("{}")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    result = module.scan_roots({"project": alias / "nested"})
    assert result["candidates"] == []
    assert result["roots"]["project"]["status"] == "MISSING_OR_UNSAFE"
    metadata = module._safe_metadata(
        {"source": "https://assets.example/model?token=TEST_SECRET", "status": "PASS"}
    )
    assert metadata == {"status": "PASS"}
    assert module._safe_local_path_value(r"\\server\share\release") is None
    local_windows_path = r"C:\billiardsAI\release"
    assert module._safe_local_path_value(local_windows_path) == local_windows_path


def test_hash_manifest_relations_marks_changed_detector_and_valid_dinov2(module, tmp_path):
    detector = tmp_path / "run-a"
    detector.mkdir()
    (detector / "checkpoint_best.pth").write_bytes(b"current")
    (detector / "training-manifest.json").write_text(
        json.dumps({"artifacts": {"checkpoint_best.pth": hashlib.sha256(b"old").hexdigest()}})
    )
    dino = tmp_path / "dinov2"
    dino.mkdir()
    model = dino / "model.safetensors"
    model.write_bytes(b"model")
    (dino / "manifest.json").write_text(
        json.dumps({"payloads": [{"name": "model.safetensors", "bytes": 5, "sha256": sha(model)}]})
    )
    result = module.scan_roots({"algorithm": tmp_path})
    relations = {item["kind"]: item["status"] for item in result["relations"]}
    assert relations["detector_run"] == "HASH_MISMATCH"
    assert relations["dinov2_assets"] == "MANIFEST_HASHES_MATCH"


def test_output_directory_accepts_bootstrap_manifest_but_no_other_existing_files(module, tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "packet-manifest.json").write_text("{}")
    module.prepare_output_directory(output)
    (output / "unexpected.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="not empty"):
        module.prepare_output_directory(output)


def build_packet(target):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_inventory.py"),
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_packet_clean_import_hashes_and_single_line_powershell(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    target = downloads / "billiardsAI-raytheon-asset-inventory-20260905-v1.zip"
    build_packet(target)
    extracted = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["packet_version"] == "raytheon-asset-inventory-20260905-v1"
        assert manifest["read_only"] is True
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        archive.extractall(extracted)
    help_result = subprocess.run(
        [sys.executable, str(extracted / "inventory_check.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    fake = tmp_path / "python with spaces"
    fake.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    command = (
        (ROOT / "docs/operations/raytheon-asset-inventory-20260905.md")
        .read_text()
        .split("```powershell\n")[1]
        .split("\n```")[0]
    )
    assert "\n" not in command
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe",
        str(fake),
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; $env:LOCALAPPDATA='{tmp_path / 'LocalAppData'}'; "
        f"function powershell.exe {{ & '{pwsh}' @args; $global:LASTEXITCODE=$LASTEXITCODE }}; "
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", prefix + command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
    assert args[0].endswith("inventory_check.py")
    assert args[1:3] == ["--output", args[2]]
    assert "--project-root" in args and "--runtime-root" in args


def test_powershell_launcher_rejects_python_below_linked_ancestor(tmp_path):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    packet = tmp_path / "packet.zip"
    build_packet(packet)
    extracted = tmp_path / "packet"
    with zipfile.ZipFile(packet) as archive:
        archive.extractall(extracted)
    real = tmp_path / "real"
    real.mkdir()
    fake = real / "python"
    fake.write_text("#!/bin/sh\necho EXECUTED\n")
    fake.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = tmp_path / "output"
    output.mkdir()
    environment = dict(os.environ)
    environment.update(
        {"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path / "LocalAppData")}
    )
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(extracted / "run.ps1"),
            "-ProjectRoot",
            str(tmp_path / "project"),
            "-PythonPath",
            str(alias / "python"),
            "-OutputParent",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXECUTED" not in result.stdout
    assert "reparse" in result.stdout.casefold()
