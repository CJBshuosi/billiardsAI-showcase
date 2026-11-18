#!/usr/bin/env python3
"""Run the real RF-DETR 1.9.0 one-batch smoke against a synthetic COCO fixture."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from billiards_ai.lab.raytheon_handoff import FrozenInputs, verify_frozen_inputs
from billiards_ai.lab.training.detector_dataset import (
    DetectorTrainingDataError,
    create_synthetic_rfdetr_fixture,
)
from billiards_ai.lab.training.rfdetr_backend import (
    DetectorOutOfMemoryError,
    RfDetrTrainingBackend,
)
from billiards_ai.lab.training.rfdetr_train import (
    DetectorTrainingError,
    run_detector_training_smoke,
)

BackendFactory = Callable[..., Any]
FrozenInputsLoader = Callable[[Path, Path], FrozenInputs]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory = RfDetrTrainingBackend,
    frozen_inputs_loader: FrozenInputsLoader = verify_frozen_inputs,
) -> int:
    args = _parser().parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            frozen = frozen_inputs_loader(_PROJECT_ROOT, args.weight)
            fixture_path = args.workspace / "synthetic-rfdetr-dataset"
            if not fixture_path.exists():
                create_synthetic_rfdetr_fixture(fixture_path)
            backend = backend_factory(
                weight_path=args.weight,
                expected_weight_sha256=frozen.weight_sha256,
                expected_version="1.9.0",
                model_class=frozen.model_class,
                weight_asset_id=frozen.weight_asset_id,
                accelerator=args.accelerator,
                amp=args.amp,
            )
            backend.validate_runtime()
            run = run_detector_training_smoke(
                dataset_dir=fixture_path,
                config_path=frozen.config_path,
                runs_root=args.runs_root,
                backend=backend,
            )
    except DetectorOutOfMemoryError as exc:
        print(
            json.dumps(
                {
                    "detail": str(exc),
                    "exit_code": 21,
                    "schema_version": "1.0",
                    "seed": frozen.seed,
                    "stage": exc.stage,
                    "status": "OOM",
                    "synthetic_only": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 21
    except (DetectorTrainingError, DetectorTrainingDataError, OSError, ValueError) as exc:
        print(f"RF-DETR smoke failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "exit_code": 0,
        "config_hash": run.config_hash,
        "input_config_sha256": frozen.config_sha256,
        "data_hash": run.data_hash,
        "output": str(run.output),
        "real_backend": True,
        "run_hash": run.run_hash,
        "seed": frozen.seed,
        "synthetic_only": True,
    }
    runtime_evidence = getattr(backend, "runtime_evidence", None)
    if args.accelerator == "cuda" and callable(runtime_evidence):
        with redirect_stdout(sys.stderr):
            summary["runtime"] = runtime_evidence()
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
