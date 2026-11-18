#!/usr/bin/env python3
"""Run the frozen shot-local product against a labeled multi-video suite."""

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
from billiards_ai.lab.shot_local.regression_suite import (
    RegressionCalibrationBlocked,
    RegressionProductRun,
    run_regression_suite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    return parser


def _product_runner(
    *,
    source_probe: Callable[[Path], Any] = probe_file_source,
    setup_loader: Callable[[Path], Any] = load_capture_setup,
    gate_evaluator: Callable[..., Any] = evaluate_file_calibration_gate,
    config_loader: Callable[[Path], Any] | None = None,
    cached_runner: Callable[..., Any] | None = None,
) -> Callable[..., RegressionProductRun]:
    if config_loader is None:
        from run_mainline1_shot_local_video import _config

        config_loader = _config

    if cached_runner is None:
        from billiards_ai.lab.mainline1_shot_local_video import (
            run_cached_mainline1_shot_local_video,
        )

        cached_runner = run_cached_mainline1_shot_local_video

    def run(
        *,
        video: Path,
        source_artifact: Path,
        capture_setup: Path,
        calibration_root: Path,
        config_path: Path,
        output_root: Path,
    ) -> RegressionProductRun:
        source = source_probe(video)
        setup = setup_loader(capture_setup)
        gate = gate_evaluator(
            source=source,
            setup=setup,
            calibration_root=calibration_root,
        )
        gate_payload = {
            "status": gate.status.value,
            "capture_setup_id": gate.capture_setup_id,
            "setup_fingerprint": gate.setup_fingerprint,
            "reasons": list(gate.reasons),
            "artifact_id": gate.calibration_artifact_id,
            "revision": gate.calibration_revision,
            "validation_mode": (
                None if gate.validation_mode is None else gate.validation_mode.value
            ),
            "maximum_corner_drift_px": gate.maximum_corner_drift_px,
        }
        if gate.status is not CalibrationGateStatus.READY:
            raise RegressionCalibrationBlocked(
                status=gate.status.value,
                calibration_gate=gate_payload,
            )
        artifact = cached_runner(
            video=video,
            source_artifact=source_artifact,
            config=config_loader(config_path),
            output_root=output_root,
        )
        return RegressionProductRun(path=artifact.path, calibration_gate=gate_payload)

    return run


def main(
    argv: Sequence[str] | None = None,
    *,
    suite_runner: Callable[..., Any] = run_regression_suite,
    product_runner: Callable[..., RegressionProductRun] | Any | None = None,
) -> int:
    args = _parser().parse_args(argv)
    calibration_root = args.calibration_root or (args.output_root / "calibrations")
    try:
        artifact = suite_runner(
            args.manifest,
            args.output_root,
            calibration_root,
            _product_runner() if product_runner is None else product_runner,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Shot-local regression suite failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "case_count": artifact.case_count,
                "output": str(artifact.path),
                "status": artifact.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if artifact.status == "PASS":
        return 0
    if artifact.status == "FAIL":
        return 1
    if artifact.status in {"CALIBRATION_REQUIRED", "CALIBRATION_DRIFTED"}:
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
