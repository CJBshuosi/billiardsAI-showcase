#!/usr/bin/env python3
"""Run the approved local RF-DETR Seg-Small bootstrap short training."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from billiards_ai.lab.raytheon_handoff import FrozenInputs, verify_frozen_inputs
from billiards_ai.lab.training.rfdetr_backend import (
    DetectorOutOfMemoryError,
    RfDetrTrainingBackend,
)
from billiards_ai.lab.training.rfdetr_bootstrap_train import (
    BootstrapTrainingRun,
    run_bootstrap_short_training,
)
from billiards_ai.lab.training.rfdetr_train import DetectorTrainingError

BackendFactory = Callable[..., Any]
FrozenInputsLoader = Callable[[Path, Path], FrozenInputs]
Runner = Callable[..., BootstrapTrainingRun]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _PROJECT_ROOT / "configs" / "algorithm" / "training" / "rfdetr-seg-small-bootstrap-short.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    backend_factory: BackendFactory = RfDetrTrainingBackend,
    frozen_inputs_loader: FrozenInputsLoader = verify_frozen_inputs,
    runner: Runner = run_bootstrap_short_training,
) -> int:
    args = _parser().parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            frozen = frozen_inputs_loader(_PROJECT_ROOT, args.weight)
            backend = backend_factory(
                weight_path=args.weight,
                expected_weight_sha256=frozen.weight_sha256,
                expected_version="1.9.0",
                model_class=frozen.model_class,
                weight_asset_id=frozen.weight_asset_id,
                accelerator="cuda",
                amp=True,
            )
            backend.validate_runtime()
            run = runner(
                dataset_dir=args.dataset,
                config_path=_CONFIG_PATH,
                runs_root=args.runs_root,
                backend=backend,
            )
            runtime = backend.runtime_evidence()
    except DetectorOutOfMemoryError as exc:
        print(
            json.dumps(
                {
                    "detail": str(exc),
                    "exit_code": 21,
                    "schema_version": "1.0",
                    "stage": exc.stage,
                    "status": "OOM",
                    "synthetic_only": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 21
    except (DetectorTrainingError, OSError, TypeError, ValueError) as exc:
        print(f"RF-DETR bootstrap short training failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "PASS",
                "exit_code": 0,
                "dataset_id": run.dataset_id,
                "config_hash": run.config_hash,
                "run_hash": run.run_hash,
                "output": str(run.output),
                "runtime": runtime,
                "synthetic_only": False,
                "scientific_use": "development_short_train_not_locked_accuracy",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
