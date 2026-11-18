"""Fail-closed calibration decision for the canonical Mainline 1 entry."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from billiards_ai.lab.mainline1_calibration_registry import resolve_active_calibration
from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    FileSourceMetadata,
    UsbSourceMetadata,
    capture_setup_fingerprint,
)
from billiards_ai.shared.calibration.homography import (
    CalibrationDriftError,
    TableCalibration,
    assert_no_drift,
    corner_drift_px,
)
from billiards_ai.shared.models.calibration import DistortionModel


class CalibrationGateStatus(StrEnum):
    READY = "READY"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    CALIBRATION_DRIFTED = "CALIBRATION_DRIFTED"


class CalibrationValidationMode(StrEnum):
    EXACT_SOURCE = "exact_source"
    CORNER_DRIFT = "corner_drift"
    USB_QUICK_CHECK = "usb_quick_check"
    USB_FIXED_INSTALLATION = "usb_fixed_installation"


class CalibrationGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CalibrationGateStatus
    capture_setup_id: str
    setup_fingerprint: str
    reasons: tuple[str, ...] = ()
    calibration_path: Path | None = None
    calibration_artifact_id: str | None = None
    calibration_revision: int | None = None
    validation_mode: CalibrationValidationMode | None = None
    maximum_corner_drift_px: float | None = None


def _blocked(
    status: CalibrationGateStatus, setup: CaptureSetup, *reasons: str
) -> CalibrationGateResult:
    return CalibrationGateResult(
        status=status,
        capture_setup_id=setup.capture_setup_id,
        setup_fingerprint=capture_setup_fingerprint(setup),
        reasons=reasons,
    )


def evaluate_file_calibration_gate(
    *,
    source: FileSourceMetadata,
    setup: CaptureSetup,
    calibration_root: Path,
    max_corner_drift_px: float = 12.0,
) -> CalibrationGateResult:
    fingerprint = capture_setup_fingerprint(setup)
    if source.geometry != setup.geometry:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "frame_geometry_mismatch"
        )
    try:
        record = resolve_active_calibration(calibration_root, fingerprint)
    except (OSError, TypeError, ValueError):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_registry_invalid"
        )
    if record is None:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_not_registered"
        )
    artifact = calibration_root.resolve() / record.artifact_path
    calibration_path = artifact / "calibration.json"
    if calibration_path.is_symlink() or not calibration_path.is_file():
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_artifact_invalid"
        )
    try:
        calibration = TableCalibration.model_validate_json(calibration_path.read_bytes())
    except (OSError, TypeError, ValueError):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_artifact_invalid"
        )
    if calibration.device_id != setup.camera_device_id:
        return _blocked(CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "camera_device_mismatch")

    drift: float | None = None
    if source.source_sha256 == record.source_video_sha256:
        validation_mode = CalibrationValidationMode.EXACT_SOURCE
    elif setup.validation_corners_px is not None:
        current_corners = list(setup.validation_corners_px)
        drift = corner_drift_px(calibration, current_corners)
        try:
            assert_no_drift(calibration, current_corners, threshold_px=max_corner_drift_px)
        except CalibrationDriftError:
            return CalibrationGateResult(
                status=CalibrationGateStatus.CALIBRATION_DRIFTED,
                capture_setup_id=setup.capture_setup_id,
                setup_fingerprint=fingerprint,
                reasons=("corner_drift_exceeds_threshold",),
                maximum_corner_drift_px=drift,
            )
        validation_mode = CalibrationValidationMode.CORNER_DRIFT
    else:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "cross_file_validation_evidence_missing",
        )
    return CalibrationGateResult(
        status=CalibrationGateStatus.READY,
        capture_setup_id=setup.capture_setup_id,
        setup_fingerprint=fingerprint,
        calibration_path=calibration_path,
        calibration_artifact_id=record.artifact_id,
        calibration_revision=record.revision,
        validation_mode=validation_mode,
        maximum_corner_drift_px=drift,
    )


def evaluate_usb_calibration_gate(
    *,
    source: UsbSourceMetadata,
    setup: CaptureSetup,
    calibration_root: Path,
    max_corner_drift_px: float = 12.0,
    require_current_corner_evidence: bool = True,
) -> CalibrationGateResult:
    """Validate the exact USB device/mode and current fixed-mount evidence."""
    fingerprint = capture_setup_fingerprint(setup)
    if setup.source_kind.value != "usb":
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "capture_source_kind_mismatch",
        )
    if source.geometry != setup.geometry:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "frame_geometry_mismatch"
        )
    if setup.device_locator_fingerprint != source.device_fingerprint:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "device_locator_mismatch"
        )
    if (
        setup.requested_fps is None
        or setup.observed_fps is None
        or abs(setup.requested_fps - source.fps) > 0.5
        or abs(setup.observed_fps - source.fps) > 0.5
    ):
        return _blocked(CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "capture_fps_mismatch")
    if not setup.image_controls or dict(source.image_controls) != dict(setup.image_controls):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "image_control_profile_mismatch",
        )
    try:
        record = resolve_active_calibration(calibration_root, fingerprint)
    except (OSError, TypeError, ValueError):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_registry_invalid"
        )
    if record is None:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_not_registered"
        )
    artifact = calibration_root.resolve() / record.artifact_path
    calibration_path = artifact / "calibration.json"
    manifest_path = artifact / "manifest.json"
    if (
        calibration_path.is_symlink()
        or not calibration_path.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_artifact_invalid"
        )
    try:
        calibration = TableCalibration.model_validate_json(calibration_path.read_bytes())
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, TypeError, ValueError):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "calibration_artifact_invalid"
        )
    if calibration.device_id != setup.camera_device_id:
        return _blocked(CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "camera_device_mismatch")
    if (
        calibration.distortion.model is not DistortionModel.OPENCV
        or not calibration.distortion.verified
    ):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED, setup, "distortion_not_verified"
        )
    maximum_reference_error = manifest.get("maximum_reference_error_mm")
    if not manifest.get("reference_count") or maximum_reference_error is None:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "independent_reference_evidence_missing",
        )
    try:
        numeric_reference_error = float(maximum_reference_error)
        if not math.isfinite(numeric_reference_error):
            raise ValueError("reference error must be finite")
        if numeric_reference_error > 20.0:
            return _blocked(
                CalibrationGateStatus.CALIBRATION_REQUIRED,
                setup,
                "reference_accuracy_exceeds_threshold",
            )
    except (TypeError, ValueError):
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "independent_reference_evidence_missing",
        )
    if setup.validation_corners_px is None:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "usb_reference_validation_evidence_missing",
        )
    if not require_current_corner_evidence:
        return CalibrationGateResult(
            status=CalibrationGateStatus.READY,
            capture_setup_id=setup.capture_setup_id,
            setup_fingerprint=fingerprint,
            calibration_path=calibration_path,
            calibration_artifact_id=record.artifact_id,
            calibration_revision=record.revision,
            validation_mode=CalibrationValidationMode.USB_FIXED_INSTALLATION,
        )
    if source.validation_corners_px is None:
        return _blocked(
            CalibrationGateStatus.CALIBRATION_REQUIRED,
            setup,
            "usb_current_epoch_evidence_missing",
        )
    drift = max(
        math.hypot(current_x - reference_x, current_y - reference_y)
        for (current_x, current_y), (reference_x, reference_y) in zip(
            source.validation_corners_px,
            setup.validation_corners_px,
            strict=True,
        )
    )
    if drift > max_corner_drift_px:
        return CalibrationGateResult(
            status=CalibrationGateStatus.CALIBRATION_DRIFTED,
            capture_setup_id=setup.capture_setup_id,
            setup_fingerprint=fingerprint,
            reasons=("corner_drift_exceeds_threshold",),
            maximum_corner_drift_px=drift,
        )
    return CalibrationGateResult(
        status=CalibrationGateStatus.READY,
        capture_setup_id=setup.capture_setup_id,
        setup_fingerprint=fingerprint,
        calibration_path=calibration_path,
        calibration_artifact_id=record.artifact_id,
        calibration_revision=record.revision,
        validation_mode=CalibrationValidationMode.USB_QUICK_CHECK,
        maximum_corner_drift_px=drift,
    )
