#!/usr/bin/env python3
"""Re-observe a video from one verified shot-local detection cache."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.mainline1_shot_local_video import run_cached_mainline1_shot_local_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    config_loader: Callable[[Path], Any] = load_mainline1_shot_local_config,
    runner: Callable[..., Any] = run_cached_mainline1_shot_local_video,
) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = runner(
            video=args.video,
            source_artifact=args.source_artifact,
            config=config_loader(args.config),
            output_root=args.output_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Shot-local cached re-observation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "frame_count": artifact.frame_count,
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
