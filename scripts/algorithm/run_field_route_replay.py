#!/usr/bin/env python3
"""Replay one cached Mainline 1 video through the field route-assistance loop."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from billiards_ai.edge.route_assistance.stability import StabilityConfig
from billiards_ai.lab.field_assistance.mainline1_replay import (
    load_mainline1_replay_inputs,
)
from billiards_ai.lab.field_assistance.replay import (
    FieldReplayConfig,
    run_field_route_replay,
)
from billiards_ai.lab.field_assistance.visual_motion import VisualMotionConfig
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.calibration.homography import TableCalibration


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"field route replay arguments invalid: {message}", file=sys.stderr)
        raise SystemExit(64)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    replay_input = parser.add_mutually_exclusive_group(required=True)
    replay_input.add_argument("--observations", type=Path)
    replay_input.add_argument("--mainline1-artifact", type=Path)
    parser.add_argument("--shot-windows", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--context-group", required=True, choices=("open", "solid", "stripe"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--motion-threshold-mm", type=float, default=8.0)
    parser.add_argument("--motion-confirm-frames", type=int, default=2)
    parser.add_argument("--stable-duration-ms", type=int, default=400)
    parser.add_argument("--visual-pixel-delta-threshold", type=int, default=18)
    parser.add_argument("--visual-changed-pixel-ratio", type=float, default=0.0005)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.mainline1_artifact is not None and (
        arguments.calibration is not None or arguments.shot_windows is not None
    ):
        parser.error("--mainline1-artifact cannot be combined with --calibration or --shot-windows")
    if arguments.mainline1_artifact is None and arguments.calibration is None:
        parser.error("--calibration is required with --observations")
    try:
        if arguments.mainline1_artifact is not None:
            inputs = load_mainline1_replay_inputs(
                artifact_path=arguments.mainline1_artifact,
                video_path=arguments.video,
            )
            observations_path = inputs.observations_path
            shot_windows_path = inputs.shot_windows_path
            events_path = inputs.events_path
            calibration = inputs.calibration
            source_mainline1_artifact_id = inputs.artifact_id
        else:
            observations_path = arguments.observations
            shot_windows_path = arguments.shot_windows
            events_path = None
            calibration = TableCalibration.model_validate_json(arguments.calibration.read_bytes())
            source_mainline1_artifact_id = None
        artifact = run_field_route_replay(
            video=arguments.video,
            observations_path=observations_path,
            shot_windows_path=shot_windows_path,
            events_path=events_path,
            calibration=calibration,
            context=EightBallContext(group=arguments.context_group),
            config=FieldReplayConfig(
                stability=StabilityConfig(
                    motion_threshold_mm=arguments.motion_threshold_mm,
                    motion_confirm_frames=arguments.motion_confirm_frames,
                    stable_duration_ms=arguments.stable_duration_ms,
                ),
                visual_motion=VisualMotionConfig(
                    pixel_delta_threshold=arguments.visual_pixel_delta_threshold,
                    changed_pixel_ratio=arguments.visual_changed_pixel_ratio,
                ),
            ),
            output_root=arguments.output_root,
            source_mainline1_artifact_id=source_mainline1_artifact_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"field route replay failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "frame_count": artifact.frame_count,
                "guidance_generation_count": artifact.guidance_generation_count,
                "output": str(artifact.path),
                "presentation": str(artifact.path / "presentation.mp4"),
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
