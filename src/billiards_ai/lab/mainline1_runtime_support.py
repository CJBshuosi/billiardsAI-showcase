"""Common runtime construction helpers; no product CLI or training execution imports."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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
from billiards_ai.lab.mainline1_native_capture import NativeCaptureSubprocessBackend
from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
from billiards_ai.lab.mainline1_runtime import (
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.mainline1_table_validation import detect_table_validation_corners
from billiards_ai.lab.vision.contracts import make_vision_frame_input
from billiards_ai.shared.calibration.homography import (
    CalibrationDriftError,
    TableCalibration,
    assert_no_drift,
)

__all__ = [
    "_detector",
    "_session",
    "_capture",
    "_preflight_capture",
    "_select_table_corners",
    "CaptureSourceKind",
    "load_capture_setup",
    "capture_setup_fingerprint",
    "resolve_active_calibration",
    "TableCalibration",
    "RealtimeCaptureConfig",
    "load_mainline1_shot_local_config",
    "make_vision_frame_input",
    "load_development_group_classifier",
    "load_development_vision_backend",
    "Dinov2IdentityFeatureBackend",
    "OpenCvIdentityVisualEvidenceBackend",
]


def _detector(path: Path, *, device: str = "cuda") -> object:
    from rfdetr import from_checkpoint

    model = from_checkpoint(str(path))
    if device == "cuda":
        import torch

        model.inference(  # type: ignore[attr-defined]
            compile=False,
            inplace=True,
            dtype=torch.float16,
        )
    return model


def _session(path: Path) -> object:
    import onnxruntime

    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _capture(
    arguments: argparse.Namespace,
    *,
    epoch: str,
    controls: tuple[tuple[str, float], ...],
) -> NativeCaptureSubprocessBackend:
    return NativeCaptureSubprocessBackend(
        helper=arguments.native_helper,
        device_locator=arguments.device_locator,
        expected_fingerprint=arguments.device_fingerprint,
        capture_config=RealtimeCaptureConfig(),
        source_epoch_id=epoch,
        image_controls=controls,
    )


def _preflight_capture(
    source: NativeCaptureSubprocessBackend,
    *,
    sample_count: int = 12,
) -> tuple[object, ...]:
    if sample_count < 2:
        raise ValueError("preflight capture requires at least two frames")
    source.start()
    stop = threading.Event()
    frames = source.frames(stop)
    try:
        captured = tuple(next(frames) for _ in range(sample_count))
    finally:
        stop.set()
        source.cancel()
        frames.close()
    return captured


def _select_table_corners(
    images: Sequence[object],
    calibration: TableCalibration,
    *,
    minimum_confirmations: int = 3,
    drift_threshold_px: float = 12.0,
    consensus_threshold_px: float = 5.0,
) -> tuple[tuple[float, float], ...]:
    """Require calibration-bound current-frame detections, tolerating abstentions."""
    if minimum_confirmations < 2 or len(images) < minimum_confirmations:
        raise ValueError("table preflight confirmation count is invalid")
    candidates: list[np.ndarray] = []
    for image in images:
        corners = detect_table_validation_corners(image)
        if corners is None:
            continue
        try:
            assert_no_drift(
                calibration,
                list(corners),
                threshold_px=drift_threshold_px,
            )
        except CalibrationDriftError:
            continue
        candidates.append(np.asarray(corners, dtype=np.float64))
    best_cluster: list[np.ndarray] = []
    for candidate in candidates:
        cluster = [
            other
            for other in candidates
            if float(np.linalg.norm(other - candidate, axis=1).max()) <= consensus_threshold_px
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < minimum_confirmations:
        raise ValueError(
            "actual table corners could not be confirmed consistently: "
            f"only {len(best_cluster)} of {len(images)} frames passed"
        )
    consensus = np.median(np.stack(best_cluster), axis=0)
    result = tuple((float(row[0]), float(row[1])) for row in consensus)
    assert_no_drift(calibration, list(result), threshold_px=drift_threshold_px)
    return result
