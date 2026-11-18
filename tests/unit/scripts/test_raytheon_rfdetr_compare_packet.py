import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/diagnostics/raytheon_rfdetr_compare/rfdetr_compare.py"


@pytest.fixture
def module():
    assert SCRIPT.exists(), "RF-DETR comparison diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("rfdetr_compare", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def test_frame_selection_combines_stride_dense_motion_and_boundaries(module):
    assert module.choose_frame_indices(25, peak_index=12, stride=10, dense_radius=2) == (
        0,
        10,
        11,
        12,
        13,
        14,
        20,
        24,
    )


def test_model_verification_requires_manifest_checkpoint_hash_and_safe_path(module, tmp_path):
    run = tmp_path / "run-a"
    run.mkdir()
    checkpoint = run / "checkpoint_best.pth"
    checkpoint.write_bytes(b"checkpoint")
    manifest = run / "training-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_hash": "run-a",
                "model_class": "RFDETRSegSmall",
                "artifacts": {"checkpoint_best.pth": digest(b"checkpoint")},
            }
        )
    )
    spec = {
        "run_id": "run-a",
        "manifest_sha256": digest(manifest.read_bytes()),
        "checkpoint_sha256": digest(b"checkpoint"),
    }
    verified = module.verify_model_run(run, spec)
    assert verified == {
        "run_id": "run-a",
        "manifest_sha256": spec["manifest_sha256"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_bytes": 10,
    }
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        module.verify_model_run(run, spec)
    alias = tmp_path / "linked-run"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        module.verify_model_run(alias, {**spec, "run_id": "linked-run"})


def test_media_resolution_prefers_existing_exact_directory_and_verifies_hashes(module, tmp_path):
    downloads = tmp_path / "Downloads"
    root = downloads / "nested" / "rer-table-result-fixed"
    for name, payload in (("slow", b"s"), ("fast_cushion", b"f"), ("pocket", b"p")):
        path = root / name / "camera.avi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    specs = {
        "slow": digest(b"s"),
        "fast_cushion": digest(b"f"),
        "pocket": digest(b"p"),
    }
    result = module.resolve_media(
        downloads,
        result_directory_name="rer-table-result-fixed",
        archive_name="unused.zip",
        archive_sha256="0" * 64,
        media_sha256=specs,
    )
    assert result.source_kind == "existing_result_directory"
    assert {key: path.read_bytes() for key, path in result.paths.items()} == {
        "slow": b"s",
        "fast_cushion": b"f",
        "pocket": b"p",
    }
    result.close()
    (root / "slow" / "camera.avi").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        module.resolve_media(
            downloads,
            result_directory_name="rer-table-result-fixed",
            archive_name="unused.zip",
            archive_sha256="0" * 64,
            media_sha256=specs,
        )


def test_media_resolution_extracts_only_verified_videos_from_exact_archive(module, tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    archive = downloads / "motion.zip"
    prefix = "rer-table-result-fixed"
    payloads = {"slow": b"s", "fast_cushion": b"f", "pocket": b"p"}
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in payloads.items():
            output.writestr(f"{prefix}/{name}/camera.avi", payload)
        output.writestr(f"{prefix}/summary.json", "{}")
    result = module.resolve_media(
        downloads,
        result_directory_name=prefix,
        archive_name=archive.name,
        archive_sha256=digest(archive.read_bytes()),
        media_sha256={name: digest(payload) for name, payload in payloads.items()},
    )
    assert result.source_kind == "verified_send_back_archive"
    assert {name: path.read_bytes() for name, path in result.paths.items()} == payloads
    temporary_root = result.temporary_root
    result.close()
    assert not temporary_root.exists()


class FakePredictions:
    xyxy = np.asarray([[1.0, 2.0, 5.0, 8.0], [10.0, 11.0, 15.0, 18.0]])
    confidence = np.asarray([0.75, 0.5])
    class_id = np.asarray([0, 0])

    def __len__(self):
        return 2


class FakeModel:
    def predict(self, image, *, threshold, include_source_image):
        assert image.shape == (4, 6, 3)
        assert threshold == 0.25
        assert include_source_image is False
        return FakePredictions()


class FakeParameter:
    device = "cuda:0"


class FakeCudaModule:
    def __init__(self, device="cuda:0"):
        self.parameter = FakeParameter()
        self.parameter.device = device

    def parameters(self):
        return iter((self.parameter,))


class FakeCudaContext:
    device = "cuda:0"
    model = FakeCudaModule()


class FakeCudaModel:
    model = FakeCudaContext()


class FakeLazyCudaModel:
    def __init__(self):
        self.model = FakeCudaContext()
        self.model.model = FakeCudaModule(device="cpu")
        self.predict_calls = 0

    def predict(self, image, *, threshold, include_source_image):
        assert image.shape == (4, 6, 3)
        assert threshold == 0.25
        assert include_source_image is False
        self.predict_calls += 1
        self.model.model.parameter.device = "cuda:0"
        return FakePredictions()


def test_prediction_boundary_exports_only_normalized_ball_detections(module):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert module.predict_frame(FakeModel(), image, threshold=0.25) == [
        {"bbox_xyxy": [1.0, 2.0, 5.0, 8.0], "class_id": 0, "confidence": 0.75},
        {"bbox_xyxy": [10.0, 11.0, 15.0, 18.0], "class_id": 0, "confidence": 0.5},
    ]


def test_checkpoint_loader_forces_cuda_and_device_evidence_rejects_cpu(module, tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pth"
    checkpoint.write_bytes(b"model")

    def factory(path, *, device):
        assert path == str(checkpoint)
        assert device == "cuda:0"
        return FakeCudaModel()

    model = module.load_cuda_checkpoint(checkpoint, factory)
    assert module.cuda_model_evidence(model) == {
        "declared_device": "cuda:0",
        "parameter_device": "cuda:0",
    }
    model.model.device = "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        module.cuda_model_evidence(model)
    model.model.device = "cuda:0"


def test_cuda_warmup_allows_rfdetr_lazy_parameter_migration(module):
    model = FakeLazyCudaModel()
    with pytest.raises(RuntimeError, match="CUDA"):
        module.cuda_model_evidence(model)

    evidence = module.warmup_cuda_model(
        model,
        np.zeros((4, 6, 3), dtype=np.uint8),
    )

    assert model.predict_calls == 1
    assert evidence == {
        "declared_device": "cuda:0",
        "parameter_device": "cuda:0",
    }


def test_result_packaging_refuses_original_video_and_marks_non_authoritative(module, tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "comparison.json").write_text("{}")
    target = module.pack_result(output, {"status": "COLLECTED_REVIEW_REQUIRED"})
    with zipfile.ZipFile(target) as archive:
        summary = json.loads(archive.read(f"{output.name}/summary.json"))
        assert summary["identity_evaluated"] is False
        assert summary["official_results_written"] is False
        assert summary["production_ready"] is False
        assert summary["acceptance"] == "NOT_EVALUATED"
    second = tmp_path / "unsafe-result"
    second.mkdir()
    (second / "input.avi").write_bytes(b"video")
    with pytest.raises(ValueError, match="video"):
        module.pack_result(second, {"status": "FAILED"})
    weights = tmp_path / "unsafe-weights"
    weights.mkdir()
    (weights / "checkpoint.pth").write_bytes(b"weights")
    with pytest.raises(ValueError, match="weight"):
        module.pack_result(weights, {"status": "FAILED"})


def test_windows_timeout_terminates_worker_without_ctrl_break_group(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module.os, "name", "nt")
    result = module._run_worker(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "worker",
        timeout_seconds=0.01,
        stall_seconds=30,
    )
    assert result["status"] == "TIMEOUT"
    assert result["exit_code"] is not None


def build_packet(target):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_rfdetr_compare.py"),
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_packet_hashes_clean_import_and_single_line_powershell(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (tmp_path / "LocalAppData" / "billiardsAI" / "algorithm-artifacts").mkdir(parents=True)
    target = downloads / "billiardsAI-raytheon-rfdetr-compare-20260905-v1.zip"
    build_packet(target)
    extracted = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["packet_version"] == "raytheon-rfdetr-compare-20260905-v1"
        assert manifest["identity_evaluated"] is False
        assert manifest["official_results_written"] is False
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        archive.extractall(extracted)
    help_result = subprocess.run(
        [sys.executable, str(extracted / "rfdetr_compare.py"), "--help"],
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
        (ROOT / "docs/operations/raytheon-rfdetr-compare-20260905.md")
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
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
    assert args[0].endswith("rfdetr_compare.py")
    assert "--downloads-root" in args
    assert "--algorithm-root" in args
    assert "--output" in args
