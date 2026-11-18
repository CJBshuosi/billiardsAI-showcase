#!/usr/bin/env python3
"""Launch local exact ball-number review for a representative candidate artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from billiards_ai.lab.training.identity_review_cli import run_existing_identity_review_cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    return run_existing_identity_review_cli(
        candidate_artifact=args.candidate_artifact,
        output_root=args.output_root,
        port=args.port,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
