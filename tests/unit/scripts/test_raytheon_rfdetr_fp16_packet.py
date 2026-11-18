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
SCRIPT = ROOT / "scripts/diagnostics/raytheon_rfdetr_fp16/rfdetr_fp16.py"


@pytest.fixture
def module():
    assert SCRIPT.exists(), "RF-DETR FP16 diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("rfdetr_fp16", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def test_detection_comparison_matches_by_iou_and_reports_unmatched(module):
    baseline = [
        {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "class_id": 0, "confidence": 0.9},
        {"bbox_xyxy": [20.0, 20.0, 30.0, 30.0], "class_id": 0, "confidence": 0.8},
    ]
    candidate = [
        {"bbox_xyxy": [1.0, 1.0, 11.0, 11.0], "class_id": 0, "confidence": 0.85},
        {"bbox_xyxy": [40.0, 40.0, 50.0, 50.0], "class_id": 0, "confidence": 0.7},
    ]

    result = module.compare_detection_sets(baseline, candidate, minimum_iou=0.5)

    assert result == {
        "baseline_count": 2,
        "candidate_count": 2,
        "matched_count": 1,
        "unmatched_baseline_count": 1,
        "unmatched_candidate_count": 1,
        "matched_iou": pytest.approx([81.0 / 119.0]),
        "matched_confidence_delta": pytest.approx([-0.05]),
    }


class FakeDtype:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FakeParameter:
    device = "cuda:0"

    def __init__(self, dtype):
        self.dtype = dtype


class FakeModule:
    def __init__(self, dtype):
        self.parameter = FakeParameter(dtype)

    def parameters(self):
        return iter((self.parameter,))


class FakeContext:
    device = "cuda:0"

    def __init__(self, dtype):
        self.model = FakeModule(dtype)


class FakeModel:
    def __init__(self):
        self.model = FakeContext(FakeDtype("torch.float32"))
        self.calls = []

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        self.model.model.parameter.dtype = kwargs["dtype"]


class FakeTorch:
    float16 = FakeDtype("torch.float16")


def test_enable_fp16_uses_official_inplace_api_and_proves_parameter_dtype(module):
    model = FakeModel()

    evidence = module.enable_fp16(model, FakeTorch)

    assert model.calls == [{"compile": False, "inplace": True, "dtype": FakeTorch.float16}]
    assert evidence == {
        "declared_device": "cuda:0",
        "parameter_device": "cuda:0",
        "parameter_dtype": "torch.float16",
    }


def test_dtype_evidence_fails_closed_if_optimization_stays_float32(module):
    model = FakeModel()
    with pytest.raises(RuntimeError, match="float16"):
        module.cuda_model_evidence(model, expected_dtype="torch.float16")


class FakeCuda:
    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1


def test_timed_prediction_synchronizes_cuda_around_measurement(module, monkeypatch):
    cuda = FakeCuda()
    monkeypatch.setattr(
        module.base,
        "predict_frame",
        lambda model, frame, threshold: [
            {"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "class_id": 0, "confidence": 0.9}
        ],
    )

    detections, elapsed_ms = module.timed_predict(object(), object(), threshold=0.25, cuda=cuda)

    assert cuda.calls == 2
    assert detections[0]["confidence"] == 0.9
    assert elapsed_ms >= 0


def test_precision_summary_is_review_only_and_has_frame_budgets(module):
    summary = module.summarize_precision_comparison(
        fp32_timings=[120.0, 100.0],
        fp16_timings=[50.0, 40.0],
        comparisons=[
            {
                "baseline_count": 2,
                "candidate_count": 2,
                "matched_count": 2,
                "unmatched_baseline_count": 0,
                "unmatched_candidate_count": 0,
                "matched_iou": [0.9, 1.0],
                "matched_confidence_delta": [-0.01, 0.02],
            }
        ],
    )

    assert summary["fp32"]["inference_ms_median"] == 100.0
    assert summary["fp16"]["inference_ms_median"] == 40.0
    assert summary["fp16"]["effective_fps_from_median"] == 25.0
    assert summary["frame_budget"]["30fps_ms"] == pytest.approx(1000.0 / 30.0)
    assert summary["frame_budget"]["60fps_ms"] == pytest.approx(1000.0 / 60.0)
    assert summary["frame_budget"]["fp16_median_within_30fps_detector_budget"] is False
    assert summary["frame_budget"]["fp16_p90_within_30fps_detector_budget"] is False
    assert summary["frame_budget"]["fp16_frames_within_30fps_budget_rate"] == 0.0
    assert summary["frame_budget"]["sustained_realtime_acceptance"] == "NOT_EVALUATED"
    assert summary["matched_iou_median"] == 0.9
    assert summary["acceptance"] == "NOT_EVALUATED"
    assert summary["production_ready"] is False


def test_frame_budget_reports_tail_and_rate_not_only_median(module):
    summary = module.summarize_precision_comparison(
        fp32_timings=[120.0, 120.0, 120.0, 120.0],
        fp16_timings=[10.0, 10.0, 100.0, 100.0],
        comparisons=[],
    )

    budget = summary["frame_budget"]
    assert budget["fp16_median_within_30fps_detector_budget"] is True
    assert budget["fp16_p90_within_30fps_detector_budget"] is False
    assert budget["fp16_frames_within_30fps_budget_rate"] == 0.5
    assert budget["fp16_frames_within_60fps_budget_rate"] == 0.5
    assert "fp16_meets_30fps_detector_only" not in budget


def build_packet(target):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_rfdetr_fp16.py"),
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
    target = downloads / "billiardsAI-raytheon-rfdetr-fp16-20260906-v1.zip"
    build_packet(target)
    extracted = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["packet_version"] == "raytheon-rfdetr-fp16-20260906-v1"
        assert manifest["precision_modes"] == ["fp32", "fp16"]
        assert manifest["identity_evaluated"] is False
        assert manifest["official_results_written"] is False
        assert manifest["production_ready"] is False
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        archive.extractall(extracted)
    help_result = subprocess.run(
        [sys.executable, str(extracted / "rfdetr_fp16.py"), "--help"],
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
        (ROOT / "docs/operations/raytheon-rfdetr-fp16-20260906.md")
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
    assert args[0].endswith("rfdetr_fp16.py")
    assert "--downloads-root" in args
    assert "--algorithm-root" in args
    assert "--output" in args


def test_run_ps1_rejects_windows_backslash_parent_traversal(tmp_path):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    target = tmp_path / "packet.zip"
    build_packet(target)
    extracted = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        archive.extractall(extracted)
    output = tmp_path / "output"
    output.mkdir()

    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(extracted / "run.ps1"),
            "-PythonPath",
            r"C:\safe\..\evil.exe",
            "-DownloadsRoot",
            str(tmp_path),
            "-AlgorithmRoot",
            str(tmp_path),
            "-OutputParent",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "Parent traversal is not allowed" in result.stdout + result.stderr
