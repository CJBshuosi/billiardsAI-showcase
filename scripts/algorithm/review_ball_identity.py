#!/usr/bin/env python3
"""Prepare ball crops from a reviewed bootstrap dataset and launch local identity review."""

from __future__ import annotations

import argparse
from pathlib import Path

from billiards_ai.lab.training.identity_review_cli import run_identity_review_cli


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    return run_identity_review_cli(
        bootstrap_dataset=args.bootstrap_dataset,
        output_root=args.output_root,
        port=args.port,
        open_browser=args.open_browser,
    )


if __name__ == "__main__":
    raise SystemExit(main())
