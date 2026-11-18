#!/usr/bin/env python3
"""Compare one shot-local event timeline with a manual truth table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from billiards_ai.lab.shot_local.evaluation import evaluate_event_timeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        result = evaluate_event_timeline(timeline, truth, tolerance_ms=args.tolerance_ms)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Shot-local timeline evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), **result["overall"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
