#!/usr/bin/env python3
"""Run one video through the development Mainline 1 shot-local pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateStatus,
    evaluate_file_calibration_gate,
)
from billiards_ai.lab.mainline1_capture import load_capture_setup, probe_file_source
from billiards_ai.lab.mainline1_runtime import (
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.mainline1_shot_local_config import (
    load_mainline1_shot_local_config,
)
from billiards_ai.lab.mainline1_shot_local_video import (
    run_mainline1_shot_local_video,
)
from billiards_ai.lab.training.identity_dinov2 import Dinov2IdentityFeatureBackend
from billiards_ai.lab.training.identity_visual_evidence import (
    OpenCvIdentityVisualEvidenceBackend,
)
from billiards_ai.shared.calibration.homography import TableCalibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--capture-setup", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-corner-drift-px", type=float, default=12.0)
    parser.add_argument("--detector-training-run", type=Path, required=True)
    parser.add_argument("--identity-training-run", type=Path, required=True)
    parser.add_argument("--dinov2-assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _calibration(path: Path) -> TableCalibration:
    return TableCalibration.model_validate_json(path.read_bytes())


def _detector(path: Path) -> object:
    from rfdetr import from_checkpoint

    return from_checkpoint(str(path))


def _session(path: Path) -> object:
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def main(
    argv: Sequence[str] | None = None,
    *,
    calibration_factory: Callable[[Path], Any] = _calibration,
    source_probe: Callable[[Path], Any] = probe_file_source,
    setup_loader: Callable[[Path], Any] = load_capture_setup,
    gate_evaluator: Callable[..., Any] = evaluate_file_calibration_gate,
    config_loader: Callable[[Path], Any] = load_mainline1_shot_local_config,
    feature_factory: Callable[..., Any] = Dinov2IdentityFeatureBackend,
    evidence_factory: Callable[[], Any] = OpenCvIdentityVisualEvidenceBackend,
    session_factory: Callable[[Path], Any] = _session,
    detector_model_factory: Callable[[Path], Any] = _detector,
    vision_loader: Callable[..., Any] = load_development_vision_backend,
    identity_loader: Callable[..., Any] = load_development_group_classifier,
    runner: Callable[..., Any] = run_mainline1_shot_local_video,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        source = source_probe(arguments.video)
        setup = setup_loader(arguments.capture_setup)
        gate = gate_evaluator(
            source=source,
            setup=setup,
            calibration_root=arguments.calibration_root,
            max_corner_drift_px=arguments.max_corner_drift_px,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Mainline 1 calibration gate failed: {exc}", file=sys.stderr)
        return 2
    if gate.status is not CalibrationGateStatus.READY:
        print(
            json.dumps(
                {
                    "action": "calibrate_or_validate_capture_setup",
                    "capture_setup_id": gate.capture_setup_id,
                    "reasons": list(gate.reasons),
                    "setup_fingerprint": gate.setup_fingerprint,
                    "status": gate.status.value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3
    try:
        selected_calibration = Path(gate.calibration_path)
        if arguments.calibration is not None and (
            arguments.calibration.resolve() != selected_calibration.resolve()
        ):
            raise ValueError("explicit calibration does not match the active setup calibration")
        calibration = calibration_factory(selected_calibration)
        config = config_loader(arguments.config)
        features = feature_factory(
            asset_root=arguments.dinov2_assets,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        evidence = evidence_factory()
        vision = vision_loader(
            arguments.detector_training_run,
            model_factory=detector_model_factory,
        )
        identity = identity_loader(
            arguments.identity_training_run,
            feature_backend=features,
            evidence_backend=evidence,
            session_factory=session_factory,
        )
        artifact = runner(
            video=arguments.video,
            calibration=calibration,
            vision_backend=vision,
            identity_classifier=identity,
            config=config,
            output_root=arguments.output_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Mainline 1 shot-local video failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "calibration_gate": {
                    "artifact_id": gate.calibration_artifact_id,
                    "capture_setup_id": gate.capture_setup_id,
                    "revision": gate.calibration_revision,
                    "validation_mode": gate.validation_mode.value,
                },
                "frame_count": artifact.frame_count,
                "output": str(artifact.path),
                "pipeline": "shot-local-unified-product-v1",
                "scientific_use": "development_shot_local_not_locked_accuracy",
                "shot_count": artifact.shot_count,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
