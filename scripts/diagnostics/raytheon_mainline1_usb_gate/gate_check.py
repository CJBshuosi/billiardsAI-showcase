"""Read-only RER U3 current-epoch capture and calibration gate check."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateStatus,
    evaluate_usb_calibration_gate,
)
from billiards_ai.lab.mainline1_capture import FrameGeometry, UsbSourceMetadata, load_capture_setup
from billiards_ai.lab.mainline1_table_validation import detect_table_validation_corners
from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureErrorCode,
    UsbCaptureRequest,
    enumerate_usb_devices,
    probe_usb_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-setup", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = arguments.output.parent / "evidence-frame.jpg"

    def save_evidence_frame(frame: object) -> None:
        encoded, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not encoded:
            raise OSError("failed to encode current-epoch evidence frame")
        evidence_path.write_bytes(payload.tobytes())

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "mainline1_usb_current_epoch_gate",
        "production_ready": False,
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }
    retained: list[object] = []
    exit_code = 2
    try:
        setup = load_capture_setup(arguments.capture_setup)
        devices = enumerate_usb_devices()
        if len(devices) != 1:
            raise ValueError("trusted Windows binding requires exactly one present camera")
        device = devices[0].model_copy(update={"runtime_index": 0})
        request = UsbCaptureRequest(
            device=device,
            width=setup.geometry.width,
            height=setup.geometry.height,
            fps=float(setup.requested_fps or 60.0),
            sample_frames=120,
            image_controls=setup.image_controls,
        )
        report = probe_usb_capture(
            request,
            corner_detector=detect_table_validation_corners,
            capture_sink=retained.append,
            evidence_frame_sink=save_evidence_frame,
        )
        payload["probe"] = report.model_dump(mode="json")
        if report.status is not UsbCaptureErrorCode.READY:
            payload["status"] = "CAPTURE_PROBE_FAILED"
        else:
            source = UsbSourceMetadata(
                device_fingerprint=report.device_fingerprint,
                geometry=FrameGeometry(
                    width=int(report.observed_width),
                    height=int(report.observed_height),
                    crop=(0, 0, int(report.observed_width), int(report.observed_height)),
                ),
                fps=float(report.observed_fps),
                source_epoch_id=str(report.source_epoch_id),
                evidence_frame_sha256=str(report.evidence_frame_sha256),
                validation_corners_px=report.validation_corners_px,
                image_controls=report.image_controls,
            )
            gate = evaluate_usb_calibration_gate(
                source=source,
                setup=setup,
                calibration_root=arguments.calibration_root,
            )
            payload["gate"] = gate.model_dump(mode="json")
            payload["status"] = gate.status.value
            exit_code = 0 if gate.status is CalibrationGateStatus.READY else 2
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        payload["status"] = "ERROR"
        payload["error"] = str(exc)
    finally:
        for handle in retained:
            release = getattr(handle, "release", None)
            if callable(release):
                release()
    arguments.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
