#!/usr/bin/env python3
"""Fit and persist verified OpenCV distortion parameters from USB lens samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

from billiards_ai.lab.mainline1_lens_calibration import (
    LensCalibrationError,
    LensCalibrationResult,
    LensCalibrationSample,
    calibrate_opencv_lens,
)


def _load_samples(path: Path) -> tuple[int, int, tuple[LensCalibrationSample, ...]]:
    if path.is_symlink() or not path.is_file():
        raise LensCalibrationError("lens calibration sample file is missing or unsafe")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise LensCalibrationError("lens calibration sample payload must be an object")
    try:
        width = int(payload["width"])
        height = int(payload["height"])
        samples = tuple(
            LensCalibrationSample(
                object_points=tuple(tuple(point) for point in item["object_points"]),
                image_points=tuple(tuple(point) for point in item["image_points"]),
            )
            for item in payload["samples"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LensCalibrationError("lens calibration sample payload is invalid") from exc
    return width, height, samples


def _write_atomic(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.pending-{uuid4().hex}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(
    argv: Sequence[str] | None = None,
    *,
    calibrator: Callable[..., LensCalibrationResult] = calibrate_opencv_lens,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        width, height, samples = _load_samples(args.samples)
        result = calibrator(samples=samples, width=width, height=height)
        report = {
            "distortion": result.distortion.model_dump(mode="json"),
            "height": result.height,
            "rms_reprojection_error": result.rms_reprojection_error,
            "sample_count": result.sample_count,
            "schema_version": "1.0",
            "width": result.width,
        }
        _write_atomic(
            args.output,
            json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
        )
        print(
            json.dumps(
                {"output": str(args.output.resolve()), "status": "PASS", **report},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (json.JSONDecodeError, LensCalibrationError, OSError, TypeError, ValueError) as exc:
        print(f"Mainline 1 USB lens calibration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
