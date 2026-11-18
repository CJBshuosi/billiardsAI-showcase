#!/usr/bin/env python3
"""Train the local group-only identity head over frozen DINOv2 features."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from billiards_ai.lab.training.identity_dinov2 import Dinov2IdentityFeatureBackend
from billiards_ai.lab.training.identity_group_training_run import run_group_short_training
from billiards_ai.lab.training.identity_hierarchical_backend import (
    Dinov2HierarchicalGroupTrainingBackend,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _PROJECT_ROOT / "configs" / "algorithm" / "training" / "identity.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--verdict-artifact", type=Path, required=True)
    parser.add_argument("--correction-verdict-artifact", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    feature_factory: Callable[..., Any] = Dinov2IdentityFeatureBackend,
    backend_factory: Callable[..., Any] = Dinov2HierarchicalGroupTrainingBackend,
    runner: Callable[..., Any] = run_group_short_training,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
        training = config["group_head_training"]
        features = feature_factory(
            asset_root=args.asset_root,
            device=args.device,
            batch_size=args.batch_size,
        )
        backend = backend_factory(
            candidate_root=args.candidate_artifact,
            feature_backend=features,
            hidden_size=int(training["hidden_size"]),
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            augmentation_count=int(training["augmentation_count"]),
            confidence_min=float(training["unknown_rejection"]["confidence_min"]),
            margin_min=float(training["unknown_rejection"]["margin_min"]),
            known_probability_min=float(training["unknown_rejection"]["known_probability_min"]),
            calibration_min_unknown_support=int(
                training["unknown_rejection"]["calibration_min_unknown_support"]
            ),
            hard_negative_focal_gamma=float(training["hard_negative"]["focal_gamma"]),
            object_hard_negative_weight=float(training["hard_negative"]["object_weight"]),
            visual_evidence_weight=float(training["visual_evidence"]["weight"]),
            seed=int(config["seed"]),
        )
        run = runner(
            candidate_artifact=args.candidate_artifact,
            verdict_artifact=args.verdict_artifact,
            correction_verdict_artifact=args.correction_verdict_artifact,
            config_path=_CONFIG,
            output_root=args.output_root,
            backend=backend,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Group identity short training failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": run.status,
                "gate_status": run.gate_status,
                "output": str(run.output),
                "run_id": run.run_id,
                "trained_heads": ["group"],
                "exact_head": {"status": "deferred", "trained": False},
                "synthetic_only": False,
                "scientific_use": "development_short_train_not_locked_accuracy",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
