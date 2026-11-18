#!/usr/bin/env python3
"""Run the local Mainline 1 table-state to guidance-artifact workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2

from billiards_ai.lab.adapters.pooltool_physics import PooltoolPhysicsBackend
from billiards_ai.lab.guidance import ExplicitIntent, build_recommendation
from billiards_ai.lab.guidance_artifacts import publish_guidance_run
from billiards_ai.lab.guidance_render import (
    build_guidance_scene,
    mobile_scene_payload,
    render_scene_frame,
    render_scene_video,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.calibration.homography import TableCalibration
from billiards_ai.shared.models.guidance import PocketId
from billiards_ai.shared.models.table_state import CanonicalTableState


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"guidance arguments invalid: {message}", file=sys.stderr)
        raise SystemExit(64)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    parser.add_argument("--table-state", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--context-group", choices=("open", "solid", "stripe"))
    mode.add_argument("--target-track-id")
    parser.add_argument("--pocket", choices=tuple(item.value for item in PocketId))
    parser.add_argument("--physics", choices=("off", "pooltool"), default="off")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def _media_output(
    media: Path,
    calibration: TableCalibration,
    scene,
    output_root: Path,
) -> tuple[str, bytes]:
    if media.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        frame = cv2.imread(str(media))
        if frame is None:
            raise ValueError("image input cannot be decoded")
        rendered = render_scene_frame(frame, scene, calibration)
        ok, payload = cv2.imencode(".png", rendered)
        if not ok:
            raise ValueError("overlay image cannot be encoded")
        return "overlay.png", payload.tobytes()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".guidance-media-", dir=output_root) as directory:
        temporary = Path(directory) / "overlay.mp4"
        render_scene_video(media, temporary, scene, calibration)
        return "overlay.mp4", temporary.read_bytes()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.target_track_id and not arguments.pocket:
            raise ValueError("--pocket is required with --target-track-id")
        if arguments.media and not arguments.calibration:
            raise ValueError("--calibration is required with --media")
        table_state_bytes = arguments.table_state.read_bytes()
        state = CanonicalTableState.model_validate_json(table_state_bytes)
        context = (
            EightBallContext(group=arguments.context_group) if arguments.context_group else None
        )
        intent = (
            ExplicitIntent(
                target_track_id=arguments.target_track_id,
                pocket=PocketId(arguments.pocket),
            )
            if arguments.target_track_id
            else None
        )
        physics = PooltoolPhysicsBackend() if arguments.physics == "pooltool" else None
        recommendation = build_recommendation(
            state=state,
            context=context,
            intent=intent,
            physics=physics,
        )
        scene = build_guidance_scene(recommendation)
        outputs: dict[str, object | bytes] = {
            "recommendation.json": recommendation.model_dump(mode="json"),
            "guidance-scene.json": scene.model_dump(mode="json"),
            "mobile-scene.json": mobile_scene_payload(scene),
        }
        if arguments.media:
            calibration = TableCalibration.model_validate_json(arguments.calibration.read_bytes())
            name, payload = _media_output(
                arguments.media, calibration, scene, arguments.output_root
            )
            outputs[name] = payload
        artifact = publish_guidance_run(
            arguments.output_root,
            inputs={
                "table_state_sha256": hashlib.sha256(table_state_bytes).hexdigest(),
                "planning_mode": "context" if context else "intent",
                "physics": arguments.physics,
            },
            outputs=outputs,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"guidance run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.name,
                "backend_versions": list(recommendation.backend_versions),
                "candidate_count": len(recommendation.candidates),
                "output": str(artifact),
                "physics_scored_count": sum(
                    "cue_ball_control" in candidate.score_components
                    for candidate in recommendation.candidates
                ),
                "recommendation_status": recommendation.status.value,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
