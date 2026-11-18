#!/usr/bin/env python3
"""Project a verified shot-local artifact into the lightweight realtime evidence chain."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.edge.realtime_chain.replay import run_replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., Any] = run_replay,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        artifact = runner(
            source_artifact=arguments.source_artifact,
            output_root=arguments.output_root,
            user_id=arguments.user_id,
            session_id=arguments.session_id,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Lightweight realtime-chain replay failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "output": str(artifact.path),
                "shot_count": artifact.shot_count,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
