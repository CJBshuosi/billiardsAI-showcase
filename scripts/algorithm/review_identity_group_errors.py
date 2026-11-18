#!/usr/bin/env python3
"""Review only group-head mistakes on the frozen validation split."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.training.identity_group_error_review import (
    run_identity_group_error_review_cli,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--verdict-artifact", type=Path, required=True)
    parser.add_argument("--correction-verdict-artifact", type=Path)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., Any] = run_identity_group_error_review_cli,
) -> int:
    args = _parser().parse_args(argv)
    try:
        exit_code = int(
            runner(
                candidate_artifact=args.candidate_artifact,
                verdict_artifact=args.verdict_artifact,
                correction_verdict_artifact=args.correction_verdict_artifact,
                training_run=args.training_run,
                asset_root=args.asset_root,
                output_root=args.output_root,
                port=args.port,
                open_browser=args.open_browser,
                device=args.device,
                batch_size=args.batch_size,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Group identity error review failed: {exc}", file=sys.stderr)
        return 2
    if argv is not None:
        print(json.dumps({"status": "STOPPED", "exit_code": exit_code}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
