#!/usr/bin/env python3
"""Use frozen local DINOv2 features to select a compact ball-identity review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from billiards_ai.lab.training.identity_crop_export import export_identity_crop_candidates
from billiards_ai.lab.training.identity_dinov2 import Dinov2IdentityFeatureBackend
from billiards_ai.lab.training.identity_representative_selection import (
    build_identity_representative_queue,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dataset", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    candidates = export_identity_crop_candidates(
        bootstrap_dataset=args.bootstrap_dataset,
        output_root=args.output_root / "all-candidates",
    )
    embedder = Dinov2IdentityFeatureBackend(
        asset_root=args.asset_root,
        device=args.device,
        batch_size=args.batch_size,
    )
    representatives = build_identity_representative_queue(
        candidate_artifact=candidates.path,
        output_root=args.output_root / "representative-candidates",
        embedder=embedder,
        target_count=args.target_count,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(representatives.path),
                "representative_count": representatives.crop_count,
                "source_crop_count": representatives.source_crop_count,
                "locked_test_opened": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
