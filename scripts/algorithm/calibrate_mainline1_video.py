#!/usr/bin/env python3
"""Create one local four-corner table calibration from a video frame."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.mainline1_calibration import (
    CalibrationInputError,
    load_calibration_reference_frame,
    serve_calibration,
)
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import load_capture_setup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture-setup", type=Path, required=True)
    parser.add_argument("--device-id")
    parser.add_argument("--port", type=int, default=8767)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    frame_loader: Callable[[Path], Any] = load_calibration_reference_frame,
    server_runner: Callable[..., Any] = serve_calibration,
    registry_runner: Callable[..., Any] = register_calibration,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        setup = load_capture_setup(arguments.capture_setup)
        if arguments.device_id is not None and arguments.device_id != setup.camera_device_id:
            raise ValueError("device ID does not match capture setup")
        device_id = setup.camera_device_id
        reference = frame_loader(arguments.video)
        artifact = server_runner(
            video=arguments.video,
            frame=reference.frame,
            frame_index=reference.frame_index,
            device_id=device_id,
            output_root=arguments.output_root,
            port=arguments.port,
            capture_setup=setup,
        )
        record = registry_runner(arguments.output_root.resolve(), artifact, setup)
    except (CalibrationInputError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Mainline 1 calibration failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "calibration": str(artifact.calibration_path),
                "capture_setup_fingerprint": record.setup_fingerprint,
                "capture_setup_id": setup.capture_setup_id,
                "output": str(artifact.path),
                "revision": record.revision,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
