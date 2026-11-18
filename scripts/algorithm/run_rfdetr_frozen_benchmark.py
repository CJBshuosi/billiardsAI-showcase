#!/usr/bin/env python3
"""Run the local frozen three-way detector benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from billiards_ai.lab.adapters.classic_cv import ClassicDetectorBackend
from billiards_ai.lab.training.rfdetr_frozen_benchmark import (
    FrozenBenchmarkInputs,
    FrozenBenchmarkRun,
    run_frozen_benchmark,
)

Runner = Callable[[FrozenBenchmarkInputs], FrozenBenchmarkRun]


def _trained_model(checkpoint: Path) -> object:
    import rfdetr

    return rfdetr.from_checkpoint(str(checkpoint))


def _coco_model(weight: Path) -> object:
    from rfdetr import RFDETRSegSmall

    return RFDETRSegSmall(pretrain_weights=str(weight))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--bootstrap-dataset", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--trained-checkpoint-sha256", required=True)
    parser.add_argument("--base-weight", required=True, type=Path)
    parser.add_argument("--base-weight-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, runner: Runner = run_frozen_benchmark) -> int:
    arguments = _parser().parse_args(argv)
    try:
        run = runner(
            FrozenBenchmarkInputs(
                input_bundle=arguments.input_bundle,
                evaluation_manifest=arguments.evaluation_manifest,
                bootstrap_dataset=arguments.bootstrap_dataset,
                training_run=arguments.training_run,
                base_weight=arguments.base_weight,
                expected_base_weight_sha256=arguments.base_weight_sha256,
                expected_training_manifest_sha256=arguments.training_manifest_sha256,
                expected_trained_checkpoint_sha256=arguments.trained_checkpoint_sha256,
                output_root=arguments.output_root,
                trained_model_factory=_trained_model,
                coco_model_factory=_coco_model,
                classic_backend_factory=ClassicDetectorBackend,
            )
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"frozen benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": run.artifact_id,
                "exit_code": 0,
                "output": str(run.path),
                "status": "PASS",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
