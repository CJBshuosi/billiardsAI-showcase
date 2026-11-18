#!/usr/bin/env python3
"""Enumerate and quality-probe a USB camera for Mainline 1."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureErrorCode,
    UsbCaptureReport,
    UsbCaptureRequest,
    UsbDeviceIdentity,
    enumerate_usb_devices,
    probe_usb_capture,
)


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(
    argv: Sequence[str] | None = None,
    *,
    enumerator: Callable[[], tuple[UsbDeviceIdentity, ...]] = enumerate_usb_devices,
    prober: Callable[[UsbCaptureRequest], UsbCaptureReport] = probe_usb_capture,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device-locator")
    parser.add_argument("--runtime-index", type=int)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--sample-frames", type=int, default=120)
    args = parser.parse_args(argv)

    try:
        devices = enumerator()
        if args.list:
            _print(
                {
                    "devices": [
                        {
                            "device_fingerprint": device.durable_fingerprint,
                            "device_locator": device.device_locator,
                            "display_name": device.display_name,
                            "product_id": device.product_id,
                            "serial_number": device.serial_number,
                            "vendor_id": device.vendor_id,
                        }
                        for device in devices
                    ],
                    "schema_version": "1.0",
                    "status": "PASS",
                }
            )
            return 0
        if args.device_locator is None or args.runtime_index is None:
            parser.error("--device-locator and --runtime-index are required unless --list is used")
        configured = next(
            (
                device
                for device in devices
                if device.device_locator.casefold() == args.device_locator.casefold()
            ),
            None,
        )
        if configured is None:
            _print(
                {
                    "error": "configured USB device locator is not present",
                    "schema_version": "1.0",
                    "status": UsbCaptureErrorCode.CAMERA_NOT_FOUND.value,
                }
            )
            return 2
        request = UsbCaptureRequest(
            device=configured.model_copy(update={"runtime_index": args.runtime_index}),
            width=args.width,
            height=args.height,
            fps=args.fps,
            sample_frames=args.sample_frames,
        )
        report = prober(request)
        _print(report.model_dump(mode="json"))
        return 0 if report.status is UsbCaptureErrorCode.READY else 2
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        _print(
            {
                "error": str(exc),
                "schema_version": "1.0",
                "status": "probe_failed",
            }
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
