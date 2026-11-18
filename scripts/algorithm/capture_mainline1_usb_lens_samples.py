#!/usr/bin/env python3
"""Capture checkerboard samples from the exact Mainline 1 USB camera."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

import cv2

from billiards_ai.lab.mainline1_lens_calibration import LensCalibrationSample
from billiards_ai.lab.mainline1_lens_sample_capture import (
    CheckerboardSpec,
    LensSampleCaptureError,
    collect_usb_checkerboard_samples,
    select_checkerboard_roi,
)
from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity, enumerate_usb_devices

Collector = Callable[..., tuple[LensCalibrationSample, ...]]

_WIDTH = 1920
_HEIGHT = 1080
_FPS = 60.0
_COLUMNS = 9
_ROWS = 6
_SQUARE_SIZE_MM = 25.0


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
    enumerator: Callable[[], tuple[UsbDeviceIdentity, ...]] = enumerate_usb_devices,
    collector: Collector = collect_usb_checkerboard_samples,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-locator", required=True)
    parser.add_argument("--runtime-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=12)
    parser.add_argument("--checkerboard-columns", type=int, choices=range(2, 31), default=_COLUMNS)
    parser.add_argument("--checkerboard-rows", type=int, choices=range(2, 31), default=_ROWS)
    parser.add_argument(
        "--rer-profile",
        action="store_true",
        help="Diagnostic RER profile: DirectShow MJPG, manual -8 readback; no control writes",
    )
    args = parser.parse_args(argv)
    try:
        devices = enumerator()
        configured = next(
            (
                device
                for device in devices
                if device.device_locator.casefold() == args.device_locator.casefold()
            ),
            None,
        )
        if configured is None:
            raise LensSampleCaptureError("configured USB device locator is not present")
        if len(devices) != 1:
            raise LensSampleCaptureError(
                "fixed-device sampling requires exactly one present camera"
            )
        spec = CheckerboardSpec(
            columns=args.checkerboard_columns,
            rows=args.checkerboard_rows,
            square_size_mm=_SQUARE_SIZE_MM,
        )
        if args.output.is_symlink():
            raise LensSampleCaptureError("lens sample output path is unsafe")
        if args.rer_profile and args.output.exists():
            raise LensSampleCaptureError("use a new output path; existing samples are preserved")
        if not args.rer_profile:
            args.output.unlink(missing_ok=True)
        evidence = {}
        options = {}
        if args.rer_profile:
            options = {
                "capture_factory": lambda index: cv2.VideoCapture(index, cv2.CAP_DSHOW),
                "require_mjpg": True,
                "warmup_frames": 120,
                "capture_evidence": evidence,
                "roi_selector": select_checkerboard_roi,
                "evidence_directory": args.output.with_suffix(".frames"),
                "expected_controls": {
                    cv2.CAP_PROP_EXPOSURE: -8,
                    cv2.CAP_PROP_AUTOFOCUS: 2,
                    cv2.CAP_PROP_AUTO_WB: 0,
                    cv2.CAP_PROP_FOCUS: 100,
                    cv2.CAP_PROP_WB_TEMPERATURE: 4650,
                    cv2.CAP_PROP_ZOOM: 100,
                    cv2.CAP_PROP_GAIN: 140,
                },
            }
        samples = collector(
            runtime_index=args.runtime_index,
            spec=spec,
            target_samples=args.target_samples,
            width=_WIDTH,
            height=_HEIGHT,
            fps=_FPS,
            **options,
        )
        report = {
            "checkerboard": {
                "columns": spec.columns,
                "rows": spec.rows,
                "square_size_mm": spec.square_size_mm,
            },
            "device_fingerprint": configured.durable_fingerprint,
            "height": _HEIGHT,
            "runtime_index": args.runtime_index,
            "samples": [
                {
                    "image_points": sample.image_points,
                    "object_points": sample.object_points,
                }
                for sample in samples
            ],
            "schema_version": "1.0",
            "width": _WIDTH,
        }
        if args.rer_profile:
            report.update(
                capture_backend="DSHOW",
                capture_evidence=evidence,
                identity_binding="enumeration_and_operator_confirmation_only",
                acceptance="NOT_EVALUATED",
                production_ready=False,
            )
        _write_atomic(
            args.output,
            json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
        )
        print(
            json.dumps(
                {
                    "device_fingerprint": configured.durable_fingerprint,
                    "output": str(args.output.resolve()),
                    "sample_count": len(samples),
                    "schema_version": "1.0",
                    "status": "CAPTURED_NOT_CALIBRATED" if args.rer_profile else "PASS",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (LensSampleCaptureError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Mainline 1 USB lens sample capture failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
