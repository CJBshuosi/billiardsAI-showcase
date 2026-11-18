#!/usr/bin/env python3
"""Run the canonical Mainline 1 pipeline from one gated USB camera."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateStatus,
    evaluate_usb_calibration_gate,
)
from billiards_ai.lab.mainline1_capture import (
    FrameGeometry,
    UsbSourceMetadata,
    load_capture_setup,
)
from billiards_ai.lab.mainline1_frame_source import UsbFrameSource
from billiards_ai.lab.mainline1_live_runtime import (
    Mainline1LiveRuntime,
    Mainline1LiveRuntimeConfig,
    ValidatedReconnectSource,
)
from billiards_ai.lab.mainline1_runtime import (
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.mainline1_shot_local_config import (
    load_mainline1_shot_local_config,
)
from billiards_ai.lab.mainline1_table_validation import detect_table_validation_corners
from billiards_ai.lab.mainline1_usb_capture import (
    UsbCaptureErrorCode,
    UsbCaptureRequest,
    UsbDeviceIdentity,
    enumerate_usb_devices,
    probe_usb_capture,
)
from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    collect_observation_frames,
)
from billiards_ai.lab.training.identity_dinov2 import Dinov2IdentityFeatureBackend
from billiards_ai.lab.training.identity_visual_evidence import (
    OpenCvIdentityVisualEvidenceBackend,
)
from billiards_ai.shared.calibration.homography import TableCalibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-locator", required=True)
    parser.add_argument("--runtime-index", type=int, required=True)
    parser.add_argument("--capture-setup", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--detector-training-run", type=Path, required=True)
    parser.add_argument("--identity-training-run", type=Path, required=True)
    parser.add_argument("--dinov2-assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--maximum-spool-frames", type=int, default=3600)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def _calibration(path: Path) -> TableCalibration:
    return TableCalibration.model_validate_json(path.read_bytes())


def _detector(path: Path) -> object:
    from rfdetr import from_checkpoint

    return from_checkpoint(str(path))


def _session(path: Path) -> object:
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _emit(payload: object, *, error: bool = False) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "artifact_id": event.artifact_id,
        "artifact_path": str(event.artifact_path) if event.artifact_path else None,
        "frame_id": event.frame_id,
        "kind": event.kind.value,
        "reason": event.reason,
        "shot_id": event.shot_id,
    }


def _detect_validation_corners(frame: Any) -> tuple[tuple[float, float], ...] | None:
    """Extract current table geometry without reusing setup-sidecar corners."""

    return detect_table_validation_corners(frame)


def _release_source(source: Any | None) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()


def main(
    argv: Sequence[str] | None = None,
    *,
    enumerator: Callable[[], tuple[UsbDeviceIdentity, ...]] = enumerate_usb_devices,
    prober: Callable[[UsbCaptureRequest], Any] = probe_usb_capture,
    setup_loader: Callable[[Path], Any] = load_capture_setup,
    gate_evaluator: Callable[..., Any] = evaluate_usb_calibration_gate,
    calibration_loader: Callable[[Path], Any] = _calibration,
    config_loader: Callable[[Path], Any] = load_mainline1_shot_local_config,
    feature_factory: Callable[..., Any] = Dinov2IdentityFeatureBackend,
    evidence_factory: Callable[[], Any] = OpenCvIdentityVisualEvidenceBackend,
    session_factory: Callable[[Path], Any] = _session,
    detector_model_factory: Callable[[Path], Any] = _detector,
    vision_loader: Callable[..., Any] = load_development_vision_backend,
    identity_loader: Callable[..., Any] = load_development_group_classifier,
    source_factory: Callable[..., Any] = UsbFrameSource,
    runtime_factory: Callable[..., Any] = Mainline1LiveRuntime,
    corner_detector: Callable[
        [Any], tuple[tuple[float, float], ...] | None
    ] = _detect_validation_corners,
    warmup_runner: Callable[..., Any] = collect_observation_frames,
) -> int:
    arguments = _parser().parse_args(argv)
    retained_capture: list[Any] = []
    evidence_frames: list[Any] = []
    source: Any | None = None
    try:
        setup = setup_loader(arguments.capture_setup)
        configured = next(
            (
                device
                for device in enumerator()
                if device.device_locator.casefold() == arguments.device_locator.casefold()
            ),
            None,
        )
        if configured is None:
            raise ValueError("configured USB device locator is not present")
        request = UsbCaptureRequest(
            device=configured.model_copy(update={"runtime_index": arguments.runtime_index}),
            width=1920,
            height=1080,
            fps=60,
            sample_frames=120,
            image_controls=setup.image_controls,
        )
        report = prober(
            request,
            capture_sink=retained_capture.append,
            corner_detector=corner_detector,
            evidence_frame_sink=evidence_frames.append,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _emit({"error": str(exc), "status": "capture_probe_failed"}, error=True)
        return 2
    if report.status is not UsbCaptureErrorCode.READY:
        _emit(report.model_dump(mode="json"), error=True)
        return 3

    try:
        source_metadata = UsbSourceMetadata(
            device_fingerprint=report.device_fingerprint,
            geometry=FrameGeometry(
                width=report.observed_width,
                height=report.observed_height,
                crop=(0, 0, report.observed_width, report.observed_height),
            ),
            fps=report.observed_fps,
            source_epoch_id=report.source_epoch_id,
            evidence_frame_sha256=report.evidence_frame_sha256,
            validation_corners_px=report.validation_corners_px,
            image_controls=report.image_controls,
        )
        gate = gate_evaluator(
            source=source_metadata,
            setup=setup,
            calibration_root=arguments.calibration_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if retained_capture:
            retained_capture.pop().release()
        _emit({"error": str(exc), "status": "calibration_gate_failed"}, error=True)
        return 2
    if gate.status is not CalibrationGateStatus.READY:
        if retained_capture:
            retained_capture.pop().release()
        _emit(
            {
                "capture_setup_id": gate.capture_setup_id,
                "reasons": list(gate.reasons),
                "setup_fingerprint": gate.setup_fingerprint,
                "status": gate.status.value,
            },
            error=True,
        )
        return 3

    try:
        if len(retained_capture) != 1:
            raise ValueError("ready USB probe did not retain its validated capture handle")
        if gate.calibration_path is None or gate.calibration_artifact_id is None:
            raise ValueError("ready calibration gate omitted its active artifact")
        if gate.calibration_revision is None:
            raise ValueError("ready calibration gate omitted its revision")
        calibration = calibration_loader(Path(gate.calibration_path))
        config = config_loader(arguments.config)
        features = feature_factory(
            asset_root=arguments.dinov2_assets,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        vision = vision_loader(
            arguments.detector_training_run,
            model_factory=detector_model_factory,
        )
        identity = identity_loader(
            arguments.identity_training_run,
            feature_backend=features,
            evidence_backend=evidence_factory(),
            session_factory=session_factory,
        )
        if len(evidence_frames) != 1:
            raise ValueError("ready USB probe did not retain its evidence frame")
        warmup_observations = tuple(
            warmup_runner(
                frames=(
                    DecodedTrackingFrame(
                        frame_id=0,
                        timestamp_ms=0,
                        image=evidence_frames.pop(),
                    ),
                ),
                vision_backend=vision,
                identity_classifier=identity,
                calibration=calibration,
            )
        )
        if len(warmup_observations) != 1:
            raise ValueError("Mainline 1 inference warmup did not produce one observation")

        def reconnect_source_factory() -> ValidatedReconnectSource:
            reconnect_capture: list[Any] = []
            reconnect_source: Any | None = None
            try:
                refreshed = next(
                    (
                        device
                        for device in enumerator()
                        if device.device_locator.casefold() == arguments.device_locator.casefold()
                    ),
                    None,
                )
                if refreshed is None:
                    raise ValueError("configured USB device locator is not present")
                reconnect_request = UsbCaptureRequest(
                    device=refreshed.model_copy(update={"runtime_index": arguments.runtime_index}),
                    width=1920,
                    height=1080,
                    fps=60,
                    sample_frames=120,
                    image_controls=setup.image_controls,
                )
                reconnect_report = prober(
                    reconnect_request,
                    capture_sink=reconnect_capture.append,
                    corner_detector=corner_detector,
                )
                if reconnect_report.status is not UsbCaptureErrorCode.READY:
                    raise RuntimeError(
                        f"USB reconnect probe failed: {reconnect_report.status.value}"
                    )
                if len(reconnect_capture) != 1:
                    raise ValueError("ready USB reconnect probe did not retain its capture handle")
                reconnect_metadata = UsbSourceMetadata(
                    device_fingerprint=reconnect_report.device_fingerprint,
                    geometry=FrameGeometry(
                        width=reconnect_report.observed_width,
                        height=reconnect_report.observed_height,
                        crop=(
                            0,
                            0,
                            reconnect_report.observed_width,
                            reconnect_report.observed_height,
                        ),
                    ),
                    fps=reconnect_report.observed_fps,
                    source_epoch_id=reconnect_report.source_epoch_id,
                    evidence_frame_sha256=reconnect_report.evidence_frame_sha256,
                    validation_corners_px=reconnect_report.validation_corners_px,
                    image_controls=reconnect_report.image_controls,
                )
                reconnect_gate = gate_evaluator(
                    source=reconnect_metadata,
                    setup=setup,
                    calibration_root=arguments.calibration_root,
                )
                if reconnect_gate.status is not CalibrationGateStatus.READY:
                    raise RuntimeError(f"USB reconnect gate failed: {reconnect_gate.status.value}")
                if (
                    reconnect_gate.calibration_artifact_id is None
                    or reconnect_gate.calibration_revision is None
                ):
                    raise ValueError("ready reconnect gate omitted calibration authority")
                reconnect_source = source_factory(
                    request=reconnect_request,
                    capture_handle=reconnect_capture.pop(),
                    max_frames=arguments.max_frames,
                )
                return ValidatedReconnectSource(
                    source=reconnect_source,
                    source_epoch_id=reconnect_metadata.source_epoch_id,
                    capture_setup_id=reconnect_gate.capture_setup_id,
                    setup_fingerprint=reconnect_gate.setup_fingerprint,
                    calibration_artifact_id=reconnect_gate.calibration_artifact_id,
                    calibration_revision=reconnect_gate.calibration_revision,
                )
            except BaseException:
                if reconnect_capture:
                    reconnect_capture.pop().release()
                _release_source(reconnect_source)
                raise

        source = source_factory(
            request=request,
            capture_handle=retained_capture.pop(),
            max_frames=arguments.max_frames,
        )
        runtime = runtime_factory(
            source=source,
            calibration=calibration,
            vision_backend=vision,
            identity_classifier=identity,
            shot_config=config,
            runtime_config=Mainline1LiveRuntimeConfig(
                spool_root=arguments.spool_root,
                output_root=arguments.output_root,
                queue_capacity=arguments.queue_capacity,
                maximum_spool_frames=arguments.maximum_spool_frames,
            ),
            capture_setup_id=gate.capture_setup_id,
            setup_fingerprint=gate.setup_fingerprint,
            calibration_artifact_id=gate.calibration_artifact_id,
            calibration_revision=gate.calibration_revision,
            source_epoch_id=source_metadata.source_epoch_id,
            reconnect_source_factory=reconnect_source_factory,
            maximum_reconnect_attempts=3,
        )
        _emit(
            {
                "calibration_artifact_id": gate.calibration_artifact_id,
                "capture_setup_id": gate.capture_setup_id,
                "source_epoch_id": source_metadata.source_epoch_id,
                "status": "LIVE_RUNTIME_READY",
            }
        )
        runtime.run(event_sink=lambda event: _emit(_event_payload(event)))
    except KeyboardInterrupt:
        _emit({"reason": "operator_interrupt", "status": "CAPTURE_STOPPED"})
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if retained_capture:
            retained_capture.pop().release()
        _release_source(source)
        _emit({"error": str(exc), "status": "live_runtime_failed"}, error=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
