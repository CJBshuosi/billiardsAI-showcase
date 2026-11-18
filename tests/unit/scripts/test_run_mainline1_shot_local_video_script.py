from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateStatus,
    CalibrationValidationMode,
)


def _module():
    path = Path(__file__).parents[3] / "scripts" / "algorithm" / "run_mainline1_shot_local_video.py"
    spec = importlib.util.spec_from_file_location("run_mainline1_shot_local_video", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_wires_existing_runtime_into_shot_local_video_seam(tmp_path: Path, capsys) -> None:
    module = _module()
    video = tmp_path / "input.mp4"
    calibration = tmp_path / "calibration.json"
    capture_setup = tmp_path / "capture-setup.json"
    calibration_root = tmp_path / "calibrations"
    detector_run = tmp_path / "detector"
    identity_run = tmp_path / "identity"
    assets = tmp_path / "assets"
    config = tmp_path / "shot-local.yaml"
    output = tmp_path / "output"
    seen = {}

    class Artifact:
        path = output / "artifact"
        artifact_id = "a" * 64
        frame_count = 17
        shot_count = 1

    def runner(**kwargs):
        seen.update(kwargs)
        return Artifact()

    exit_code = module.main(
        [
            "--video",
            str(video),
            "--calibration",
            str(calibration),
            "--capture-setup",
            str(capture_setup),
            "--calibration-root",
            str(calibration_root),
            "--detector-training-run",
            str(detector_run),
            "--identity-training-run",
            str(identity_run),
            "--dinov2-assets",
            str(assets),
            "--config",
            str(config),
            "--output-root",
            str(output),
            "--device",
            "cuda",
            "--batch-size",
            "8",
        ],
        calibration_factory=lambda value: ("calibration", value),
        source_probe=lambda value: ("source", value),
        setup_loader=lambda value: ("setup", value),
        gate_evaluator=lambda **kwargs: SimpleNamespace(
            status=CalibrationGateStatus.READY,
            calibration_path=calibration,
            calibration_artifact_id="c" * 64,
            calibration_revision=2,
            validation_mode=CalibrationValidationMode.EXACT_SOURCE,
            capture_setup_id="setup-a",
            setup_fingerprint="s" * 64,
            reasons=(),
        ),
        config_loader=lambda value: ("config", value),
        feature_factory=lambda **kwargs: ("features", kwargs),
        evidence_factory=lambda: "evidence",
        session_factory=lambda path: ("session", path),
        detector_model_factory=lambda path: ("detector", path),
        vision_loader=lambda path, **kwargs: ("vision", path, kwargs),
        identity_loader=lambda path, **kwargs: ("identity", path, kwargs),
        runner=runner,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "artifact_id": "a" * 64,
        "calibration_gate": {
            "artifact_id": "c" * 64,
            "capture_setup_id": "setup-a",
            "revision": 2,
            "validation_mode": "exact_source",
        },
        "frame_count": 17,
        "output": str(Artifact.path),
        "pipeline": "shot-local-unified-product-v1",
        "scientific_use": "development_shot_local_not_locked_accuracy",
        "shot_count": 1,
        "status": "PASS",
    }
    assert seen["video"] == video
    assert seen["config"] == ("config", config)
    assert seen["vision_backend"][0] == "vision"
    assert seen["identity_classifier"][0] == "identity"


def test_main_returns_two_for_actionable_runtime_failure(tmp_path: Path, capsys) -> None:
    module = _module()

    exit_code = module.main(
        [
            "--video",
            str(tmp_path / "input.mp4"),
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--capture-setup",
            str(tmp_path / "capture-setup.json"),
            "--calibration-root",
            str(tmp_path / "calibrations"),
            "--detector-training-run",
            str(tmp_path / "detector"),
            "--identity-training-run",
            str(tmp_path / "identity"),
            "--dinov2-assets",
            str(tmp_path / "assets"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-root",
            str(tmp_path / "output"),
        ],
        calibration_factory=lambda value: (_ for _ in ()).throw(ValueError("bad calibration")),
        source_probe=lambda value: ("source", value),
        setup_loader=lambda value: ("setup", value),
        gate_evaluator=lambda **kwargs: SimpleNamespace(
            status=CalibrationGateStatus.READY,
            calibration_path=tmp_path / "calibration.json",
            calibration_artifact_id="c" * 64,
            calibration_revision=1,
            validation_mode=CalibrationValidationMode.EXACT_SOURCE,
            capture_setup_id="setup-a",
            setup_fingerprint="s" * 64,
            reasons=(),
        ),
    )

    assert exit_code == 2
    assert "bad calibration" in capsys.readouterr().err


def test_calibration_gate_blocks_before_any_model_factory(tmp_path: Path, capsys) -> None:
    module = _module()
    model_calls: list[str] = []

    exit_code = module.main(
        [
            "--video",
            str(tmp_path / "input.mp4"),
            "--capture-setup",
            str(tmp_path / "capture-setup.json"),
            "--calibration-root",
            str(tmp_path / "calibrations"),
            "--detector-training-run",
            str(tmp_path / "detector"),
            "--identity-training-run",
            str(tmp_path / "identity"),
            "--dinov2-assets",
            str(tmp_path / "assets"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-root",
            str(tmp_path / "output"),
        ],
        source_probe=lambda value: ("source", value),
        setup_loader=lambda value: ("setup", value),
        gate_evaluator=lambda **kwargs: SimpleNamespace(
            status=CalibrationGateStatus.CALIBRATION_REQUIRED,
            calibration_path=None,
            capture_setup_id="setup-a",
            setup_fingerprint="s" * 64,
            reasons=("calibration_not_registered",),
        ),
        feature_factory=lambda **kwargs: model_calls.append("features"),
        vision_loader=lambda *args, **kwargs: model_calls.append("vision"),
        identity_loader=lambda *args, **kwargs: model_calls.append("identity"),
    )

    assert exit_code == 3
    assert model_calls == []
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "CALIBRATION_REQUIRED"
    assert error["reasons"] == ["calibration_not_registered"]
