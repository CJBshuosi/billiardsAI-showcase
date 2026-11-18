#!/usr/bin/env python3
"""Run fail-closed product multi-rate Mainline 1 from the native UVC helper."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from billiards_ai.lab.mainline1_calibration_registry import resolve_active_calibration
from billiards_ai.lab.mainline1_capture import (
    CaptureSourceKind,
    capture_setup_fingerprint,
    load_capture_setup,
)
from billiards_ai.lab.mainline1_identity_backends import (
    Dinov2IdentityFeatureBackend,
    OpenCvIdentityVisualEvidenceBackend,
)
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEvent,
    ProductRealtimeEventChannel,
    ProductRealtimeRuntime,
    ProductRealtimeRuntimeConfig,
)
from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
from billiards_ai.lab.mainline1_realtime_ring import MjpegSegmentRing
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorBall,
    AnchorObservation,
    RealtimeBallTracker,
    RealtimeTrackerConfig,
)
from billiards_ai.lab.mainline1_runtime import (
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.mainline1_runtime_support import (
    CalibrationDriftError as CalibrationDriftError,
)
from billiards_ai.lab.mainline1_runtime_support import (
    NativeCaptureSubprocessBackend as NativeCaptureSubprocessBackend,
)
from billiards_ai.lab.mainline1_runtime_support import (
    _capture as _capture,
)
from billiards_ai.lab.mainline1_runtime_support import (
    _detector as _detector,
)
from billiards_ai.lab.mainline1_runtime_support import (
    _preflight_capture as _preflight_capture,
)
from billiards_ai.lab.mainline1_runtime_support import (
    _select_table_corners as _select_table_corners,
)
from billiards_ai.lab.mainline1_runtime_support import (
    _session as _session,
)
from billiards_ai.lab.mainline1_runtime_support import (
    assert_no_drift as assert_no_drift,
)
from billiards_ai.lab.mainline1_runtime_support import (
    detect_table_validation_corners as detect_table_validation_corners,
)
from billiards_ai.lab.mainline1_runtime_support import (
    np as np,
)
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.lab.vision.contracts import make_vision_frame_input
from billiards_ai.shared.calibration.homography import (
    TableCalibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-helper", type=Path, required=True)
    parser.add_argument("--device-locator", required=True)
    parser.add_argument("--device-fingerprint", required=True)
    parser.add_argument("--capture-setup", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--detector-training-run", type=Path, required=True)
    parser.add_argument("--identity-training-run", type=Path, required=True)
    parser.add_argument("--dinov2-assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ring-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diagnostic-mode", action="store_true")
    parser.add_argument("--stop-after-terminal-shots", type=int)
    parser.add_argument("--diagnostic-duration-seconds", type=float)
    return parser


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.diagnostic_mode and (
        arguments.stop_after_terminal_shots is not None
        or arguments.diagnostic_duration_seconds is not None
    ):
        parser.error(
            "--stop-after-terminal-shots and --diagnostic-duration-seconds "
            "require --diagnostic-mode"
        )
    return arguments


def _emit(payload: object, *, error: bool = False) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def _event_payload(event: ProductRealtimeEvent) -> dict[str, object]:
    return {
        "frame_id": event.frame_id,
        "kind": event.kind,
        "reason": event.reason,
        "settled": None if event.settled is None else dataclasses.asdict(event.settled),
        "shot_id": event.shot_id,
    }


def _forward_runtime_events(
    channel: ProductRealtimeEventChannel,
    completed: threading.Event,
    failures: list[BaseException],
) -> None:
    try:
        while True:
            event = channel.receive(timeout_seconds=0.1)
            if event is None:
                if completed.is_set():
                    channel.require_healthy()
                    return
                continue
            _emit(_event_payload(event))
    except BaseException as exc:
        reason = f"event consumer failed: {type(exc).__name__}: {exc}"
        channel.fail(reason)
        failures.append(RuntimeError(reason))


def _raise_event_forwarding_failure(failures: list[BaseException]) -> None:
    if failures:
        error = failures[0]
        raise RuntimeError(f"product realtime event forwarding failed: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        setup = load_capture_setup(arguments.capture_setup)
        if (
            setup.source_kind is not CaptureSourceKind.USB
            or setup.geometry.width != 1920
            or setup.geometry.height != 1080
            or setup.requested_fps != 60
            or setup.device_locator_fingerprint != arguments.device_fingerprint
        ):
            raise ValueError("capture setup does not authorize this exact native camera mode")
        setup_fingerprint = capture_setup_fingerprint(setup)
        record = resolve_active_calibration(arguments.calibration_root, setup_fingerprint)
        if record is None:
            raise ValueError("capture setup has no active immutable calibration")
        calibration_path = arguments.calibration_root / record.artifact_path / "calibration.json"
        calibration = TableCalibration.model_validate_json(calibration_path.read_bytes())
        if calibration.device_id != setup.camera_device_id:
            raise ValueError("active calibration belongs to another camera")

        preflight = _capture(
            arguments,
            epoch=f"preflight-{uuid4().hex}",
            controls=setup.image_controls,
        )
        preflight_frames = _preflight_capture(preflight)
        _select_table_corners(
            tuple(frame.decode_bgr() for frame in preflight_frames),
            calibration,
        )
        first, second = preflight_frames[-2:]

        config = load_mainline1_shot_local_config(arguments.config)
        features = Dinov2IdentityFeatureBackend(
            asset_root=arguments.dinov2_assets,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        vision = load_development_vision_backend(
            arguments.detector_training_run,
            model_factory=lambda path: _detector(path, device=arguments.device),
        )
        identity = load_development_group_classifier(
            arguments.identity_training_run,
            feature_backend=features,
            evidence_backend=OpenCvIdentityVisualEvidenceBackend(),
            session_factory=_session,
        )

        warm_image = first.decode_bgr()
        warm_input = make_vision_frame_input(
            warm_image,
            frame_id=first.capture_frame_id,
            timestamp_ms=first.timestamp_ms,
        )
        warm_observation = vision.infer(warm_input)
        warm_identity = identity.classify(
            image=warm_image,
            detections=warm_observation.detections,
        )
        anchor = AnchorObservation(
            capture_frame_id=first.capture_frame_id,
            captured_at_ns=first.captured_at_ns,
            balls=tuple(
                AnchorBall(
                    track_id=f"warm-{index:04d}",
                    center_px=(detection.center.x, detection.center.y),
                    confidence=detection.confidence,
                    group=BallGroup(
                        str(warm_identity[detection.detection_id].get("group", "unknown"))
                    ),
                    bbox_xyxy_px=(
                        detection.bbox.left,
                        detection.bbox.top,
                        detection.bbox.right,
                        detection.bbox.bottom,
                    ),
                )
                for index, detection in enumerate(warm_observation.detections)
            ),
        )
        tracker = RealtimeBallTracker(calibration, RealtimeTrackerConfig())
        tracker.apply_anchor(anchor, first)
        tracker.track(second)

        smoke = MjpegSegmentRing(
            arguments.ring_root / f"preflight-{uuid4().hex}",
            expected_fps=2,
            pre_roll_frames=0,
            post_roll_frames=0,
            maximum_shot_frames=2,
            maximum_bytes=64 * 1024 * 1024,
        )
        smoke.append(first)
        smoke.begin_shot(first.capture_frame_id)
        smoke.append(second)
        smoke.settle_shot(second.capture_frame_id)
        tuple(smoke.freeze_shot().frames())
        smoke.release_shot()
        smoke.close()

        epoch = uuid4().hex
        session_id = f"session-{uuid4().hex}"
        source = _capture(arguments, epoch=epoch, controls=setup.image_controls)
        event_channel = ProductRealtimeEventChannel()
        runtime = ProductRealtimeRuntime(
            source=source,
            capture_config=RealtimeCaptureConfig(),
            calibration=calibration,
            detector=vision,
            identity_classifier=identity,
            shot_config=config,
            job_store=SQLiteShotAnalysisJobStore(arguments.output_root / "analysis-jobs.sqlite3"),
            job_media_root=arguments.output_root / "analysis-media" / session_id,
            quarantine_root=(arguments.output_root / "analysis-quarantine" / session_id),
            session_id=session_id,
            capture_setup_id=setup.capture_setup_id,
            calibration_artifact_id=record.artifact_id,
            config_sha256=hashlib.sha256(arguments.config.read_bytes()).hexdigest(),
            event_channel=event_channel,
            config=ProductRealtimeRuntimeConfig(
                ring_root=arguments.ring_root / epoch,
                stop_after_terminal_shots=arguments.stop_after_terminal_shots,
                diagnostic_duration_seconds=arguments.diagnostic_duration_seconds,
            ),
        )
        source.start()
        _emit(
            {
                "calibration_artifact_id": record.artifact_id,
                "capture_setup_id": setup.capture_setup_id,
                "session_id": session_id,
                "source_epoch_id": epoch,
                "status": "LIVE_RUNTIME_READY",
            }
        )
        event_forwarding_completed = threading.Event()
        event_forwarding_failures: list[BaseException] = []
        event_forwarder = threading.Thread(
            target=_forward_runtime_events,
            args=(event_channel, event_forwarding_completed, event_forwarding_failures),
            name="mainline1-product-event-forwarder",
            daemon=True,
        )
        event_forwarder.start()
        try:
            events = runtime.run()
        finally:
            event_forwarding_completed.set()
            event_forwarder.join(timeout=5.0)
        if event_forwarder.is_alive():
            raise RuntimeError("product realtime event forwarder did not stop")
        _raise_event_forwarding_failure(event_forwarding_failures)
        _emit(
            {
                "anchor_health": dataclasses.asdict(runtime.anchor_health),
                "anchor_replay_health": dataclasses.asdict(runtime.anchor_replay_health),
                "capture_health": dataclasses.asdict(runtime.capture_health),
                "diagnostics": dataclasses.asdict(runtime.diagnostics),
                "maximum_capture_queue_lag_frames": (runtime.maximum_capture_queue_lag_frames),
                "maximum_ring_queue_depth": runtime.maximum_ring_queue_depth,
                "ring_health": dataclasses.asdict(runtime.ring_health),
                "status": "PRODUCT_REALTIME_SUMMARY",
                "tracker_health": (
                    None
                    if runtime.tracker_health is None
                    else dataclasses.asdict(runtime.tracker_health)
                ),
            }
        )
        if any(event.kind == "CAPTURE_DEGRADED" for event in events):
            return 2
        return 0
    except KeyboardInterrupt:
        _emit({"reason": "operator_interrupt", "status": "CAPTURE_STOPPED"})
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _emit({"error": str(exc), "status": "PRODUCT_REALTIME_FAILED"}, error=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
