#!/usr/bin/env python3
"""Capture a baseline image and inspect explicit USB camera image controls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2

from billiards_ai.lab.mainline1_usb_capture import UsbDeviceIdentity, enumerate_usb_devices
from billiards_ai.lab.mainline1_usb_image_controls import (
    UsbImageControlName,
    UsbImageControlSetting,
    UsbImageControlsReport,
    UsbImageControlsRequest,
    UsbImageControlsStatus,
    probe_usb_image_controls,
)


def _parse_control(raw: str) -> UsbImageControlSetting:
    name, separator, value = raw.partition("=")
    if not separator:
        raise ValueError("image control must use name=value")
    try:
        control_name = UsbImageControlName(name.strip())
    except ValueError as exc:
        raise ValueError(f"unknown image control: {name.strip()}") from exc
    try:
        return UsbImageControlSetting(name=control_name, value=float(value))
    except ValueError as exc:
        raise ValueError(f"invalid value for image control {control_name.value}") from exc


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_image(
    path: Path,
    frame: object,
    *,
    image_writer: Callable[[Path, object], bool],
) -> None:
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        if not image_writer(temporary, frame):
            raise RuntimeError("baseline image writer rejected the frame")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jpeg(path: Path, frame: object) -> bool:
    return bool(cv2.imwrite(str(path), frame))


def main(
    argv: Sequence[str] | None = None,
    *,
    enumerator: Callable[[], tuple[UsbDeviceIdentity, ...]] = enumerate_usb_devices,
    prober: Callable[
        [UsbImageControlsRequest], tuple[UsbImageControlsReport, object | None]
    ] = probe_usb_image_controls,
    image_writer: Callable[[Path, object], bool] = _write_jpeg,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-locator", required=True)
    parser.add_argument("--runtime-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--control", action="append", default=[])
    args = parser.parse_args(argv)

    output_created = False
    try:
        settings = tuple(_parse_control(raw) for raw in args.control)
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(f"output directory already exists: {args.output_dir}")
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
            raise ValueError("configured USB device locator is not present")
        if settings and len(devices) != 1:
            raise ValueError("explicit image-control writes require exactly one present camera")
        if settings and args.runtime_index != 0:
            raise ValueError(
                "explicit image-control writes require the sole camera at runtime index 0"
            )
        request = UsbImageControlsRequest(
            device=configured.model_copy(update={"runtime_index": args.runtime_index}),
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            settings=settings,
        )
        args.output_dir.mkdir(parents=True)
        output_created = True
        report, frame = prober(request)
        payload = report.model_dump(mode="json")
        if frame is not None:
            _atomic_image(
                args.output_dir / "baseline.jpg",
                frame,
                image_writer=image_writer,
            )
            payload["baseline_image"] = "baseline.jpg"
        _atomic_json(args.output_dir / "controls.json", payload)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "status": report.status.value,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0 if report.status is UsbImageControlsStatus.PASS else 2
    except (cv2.error, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if output_created:
            shutil.rmtree(args.output_dir, ignore_errors=True)
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
