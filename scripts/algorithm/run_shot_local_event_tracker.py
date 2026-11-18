#!/usr/bin/env python3
"""Compatibility wrapper for verified-cache unified Mainline 1 analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.mainline1_shot_local_video import (
    run_cached_mainline1_shot_local_video,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--shot-local-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    config_loader: Callable[[Path], Any] = load_mainline1_shot_local_config,
    runner: Callable[..., Any] = run_cached_mainline1_shot_local_video,
) -> int:
    """Delegate the retired second stage to the canonical cached publisher."""

    arguments = _parser().parse_args(argv)
    print(
        "DEPRECATED: use run_mainline1_shot_local_video.py for new product runs; "
        "this command only replays verified cached detections.",
        file=sys.stderr,
    )
    try:
        artifact = runner(
            video=arguments.video,
            source_artifact=arguments.source_artifact,
            config=config_loader(arguments.shot_local_config),
            output_root=arguments.output_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Unified Mainline 1 cache replay failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "frame_count": artifact.frame_count,
                "output": str(artifact.path),
                "pipeline": "shot-local-unified-product-v1",
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
