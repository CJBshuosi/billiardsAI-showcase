#!/usr/bin/env python3
"""Merge cumulative identity corrections with the newest review taking priority."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from billiards_ai.lab.training.identity_review import (
    IdentityReviewError,
    merge_identity_correction_verdicts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument(
        "--base-correction-verdict-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overlay-correction-verdict-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = merge_identity_correction_verdicts(
            candidate_artifact=args.candidate_artifact,
            base_correction_verdict_artifact=args.base_correction_verdict_artifact,
            overlay_correction_verdict_artifact=args.overlay_correction_verdict_artifact,
            output_root=args.output_root,
        )
    except (IdentityReviewError, OSError, TypeError, ValueError) as exc:
        print(f"Identity correction merge failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": "PASS", "output": str(output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
