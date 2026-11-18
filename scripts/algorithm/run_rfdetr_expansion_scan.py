#!/usr/bin/env python3
"""Scan Det-SAM2 expansion frames with the reviewed trained RF-DETR checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from billiards_ai.lab.training.rfdetr_expansion_scan import (
    ExpansionScanInputs,
    run_expansion_scan,
)


def _model(checkpoint: Path) -> object:
    import rfdetr

    return rfdetr.from_checkpoint(str(checkpoint))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--low-confidence-below", type=float, default=0.6)
    parser.add_argument("--review-limit", type=int, default=200)
    arguments = parser.parse_args()
    try:
        run = run_expansion_scan(
            ExpansionScanInputs(
                split_manifest=arguments.split_manifest,
                review_root=arguments.review_root,
                checkpoint=arguments.checkpoint,
                expected_checkpoint_sha256=arguments.checkpoint_sha256,
                output_root=arguments.output_root,
                model_factory=_model,
                threshold=arguments.threshold,
                low_confidence_below=arguments.low_confidence_below,
                review_limit=arguments.review_limit,
            )
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"RF-DETR expansion scan failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "output": str(run.path),
                "frame_count": run.frame_count,
                "detection_count": run.detection_count,
                "hard_review_count": run.hard_review_count,
                "locked_test_opened": run.locked_test_opened,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
