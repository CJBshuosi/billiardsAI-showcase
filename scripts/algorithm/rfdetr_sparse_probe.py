#!/usr/bin/env python3
"""Run the verified RF-DETR Seg-Small COCO sports-ball sparse probe."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from billiards_ai.lab.rfdetr_sparse_probe import SparseProbeInputs, run_sparse_probe


def _model_factory(weight_path: Path) -> object:
    from rfdetr import RFDETRSegSmall

    return RFDETRSegSmall(pretrain_weights=str(weight_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    arguments = parser.parse_args(argv)
    run = run_sparse_probe(
        SparseProbeInputs(
            review_root=arguments.review_root,
            weight_path=arguments.weight,
            expected_weight_sha256=arguments.weight_sha256,
            output_root=arguments.output_root,
            threshold=arguments.threshold,
            model_factory=_model_factory,
        )
    )
    print(json.dumps(run.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
