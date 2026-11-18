from pathlib import Path

import numpy as np

from billiards_ai.lab.mainline1_calibration import create_calibration_artifact
from billiards_ai.lab.mainline1_calibration_gate import (
    CalibrationGateStatus,
    CalibrationValidationMode,
    evaluate_file_calibration_gate,
    evaluate_usb_calibration_gate,
)
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    CaptureSourceKind,
    FileSourceMetadata,
    FrameGeometry,
    UsbSourceMetadata,
)
from billiards_ai.shared.models.calibration import (
    CalibrationReference,
    DistortionModel,
    DistortionParameters,
)
from billiards_ai.shared.models.geometry import Point

CORNERS = ((10.0, 210.0), (410.0, 210.0), (410.0, 10.0), (10.0, 10.0))


def _setup(**updates: object) -> CaptureSetup:
    values: dict[str, object] = {
        "capture_setup_id": "setup-a",
        "table_id": "table-a",
        "camera_device_id": "camera-a",
        "lens_id": "lens-a",
        "mount_revision": "mount-v1",
        "distortion_revision": "distortion-v1",
        "source_kind": CaptureSourceKind.FILE,
        "geometry": FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    }
    values.update(updates)
    return CaptureSetup.model_validate(values)


def _source(path: Path, sha: str = "a" * 64, *, width: int = 420) -> FileSourceMetadata:
    path.write_bytes(b"source")
    return FileSourceMetadata(
        path=path.resolve(),
        source_sha256=sha,
        geometry=FrameGeometry(width=width, height=220, crop=(0, 0, width, 220)),
        frame_count=10,
        fps=30,
    )


def _registered(
    tmp_path: Path,
    *,
    setup: CaptureSetup,
    distortion: DistortionParameters | None = None,
    references: tuple[CalibrationReference, ...] = (),
) -> tuple[Path, str]:
    root = tmp_path / "calibrations"
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"reference-video")
    artifact = create_calibration_artifact(
        video=video,
        frame_index=0,
        frame=np.zeros((220, 420, 3), dtype=np.uint8),
        corners_px=CORNERS,
        device_id="camera-a",
        output_root=root,
        capture_setup=setup,
        distortion=distortion,
        references=references,
    )
    record = register_calibration(root, artifact, setup)
    return root, record.source_video_sha256


def test_missing_record_requires_calibration(tmp_path: Path) -> None:
    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "video.mp4"),
        setup=_setup(),
        calibration_root=tmp_path / "empty",
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert "calibration_not_registered" in result.reasons
    assert result.calibration_path is None


def test_different_frame_geometry_is_blocked(tmp_path: Path) -> None:
    setup = _setup()
    root, _ = _registered(tmp_path, setup=setup)

    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "video.mp4", width=1280),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert "frame_geometry_mismatch" in result.reasons


def test_exact_source_passes_without_live_corner_evidence(tmp_path: Path) -> None:
    setup = _setup()
    root, source_sha = _registered(tmp_path, setup=setup)

    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "reference-copy.mp4", source_sha),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.READY
    assert result.validation_mode is CalibrationValidationMode.EXACT_SOURCE
    assert result.calibration_path is not None


def test_same_setup_second_video_passes_with_current_corner_evidence(tmp_path: Path) -> None:
    setup = _setup(validation_corners_px=tuple((x + 2, y + 2) for x, y in CORNERS))
    root, _ = _registered(tmp_path, setup=setup)

    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "other.mp4", "b" * 64),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.READY
    assert result.validation_mode is CalibrationValidationMode.CORNER_DRIFT


def test_drifted_corners_block_analysis(tmp_path: Path) -> None:
    setup = _setup(validation_corners_px=tuple((x + 30, y) for x, y in CORNERS))
    root, _ = _registered(tmp_path, setup=setup)

    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "other.mp4", "b" * 64),
        setup=setup,
        calibration_root=root,
        max_corner_drift_px=12,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_DRIFTED
    assert "corner_drift_exceeds_threshold" in result.reasons


def test_cross_file_without_current_evidence_fails_closed(tmp_path: Path) -> None:
    setup = _setup()
    root, _ = _registered(tmp_path, setup=setup)

    result = evaluate_file_calibration_gate(
        source=_source(tmp_path / "other.mp4", "b" * 64),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert "cross_file_validation_evidence_missing" in result.reasons


def _usb_setup(device_fingerprint: str) -> CaptureSetup:
    return _setup(
        source_kind=CaptureSourceKind.USB,
        requested_fps=60,
        observed_fps=60,
        device_locator_fingerprint=device_fingerprint,
        validation_corners_px=CORNERS,
        image_controls=(("exposure", -6.0), ("auto_exposure", 0.25)),
    )


def _usb_source(
    device_fingerprint: str,
    *,
    fps: float = 60,
    corners: tuple[tuple[float, float], ...] | None = CORNERS,
    controls: tuple[tuple[str, float], ...] = (("exposure", -6.0), ("auto_exposure", 0.25)),
) -> UsbSourceMetadata:
    return UsbSourceMetadata(
        device_fingerprint=device_fingerprint,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
        fps=fps,
        source_epoch_id="epoch-1",
        evidence_frame_sha256="f" * 64,
        validation_corners_px=corners,
        image_controls=controls,
    )


def _distortion() -> DistortionParameters:
    return DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=True,
        camera_matrix=((400.0, 0.0, 210.0), (0.0, 400.0, 110.0), (0.0, 0.0, 1.0)),
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _references() -> tuple[CalibrationReference, ...]:
    return (CalibrationReference(pixel=(210.0, 110.0), table=Point(x_mm=1270, y_mm=635)),)


def test_usb_gate_rejects_wide_lens_without_verified_distortion(tmp_path: Path) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(tmp_path, setup=setup)

    result = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert result.reasons == ("distortion_not_verified",)


def test_usb_gate_requires_exact_device_geometry_fps_and_reference_evidence(
    tmp_path: Path,
) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )

    ready = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint),
        setup=setup,
        calibration_root=root,
    )
    wrong_fps = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint, fps=30),
        setup=setup,
        calibration_root=root,
    )
    wrong_device = evaluate_usb_calibration_gate(
        source=_usb_source("e" * 64),
        setup=setup,
        calibration_root=root,
    )

    assert ready.status is CalibrationGateStatus.READY
    assert ready.validation_mode is CalibrationValidationMode.USB_QUICK_CHECK
    assert wrong_fps.reasons == ("capture_fps_mismatch",)
    assert wrong_device.reasons == ("device_locator_mismatch",)


def test_usb_gate_rejects_setup_sidecar_corners_without_current_epoch_evidence(
    tmp_path: Path,
) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )

    result = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint, corners=None),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert result.reasons == ("usb_current_epoch_evidence_missing",)


def test_usb_gate_can_use_explicit_fixed_installation_contract_without_corner_redetection(
    tmp_path: Path,
) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )

    result = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint, corners=None),
        setup=setup,
        calibration_root=root,
        require_current_corner_evidence=False,
    )

    assert result.status is CalibrationGateStatus.READY
    assert result.validation_mode is CalibrationValidationMode.USB_FIXED_INSTALLATION
    assert result.maximum_corner_drift_px is None


def test_usb_gate_rejects_changed_image_controls(tmp_path: Path) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )

    result = evaluate_usb_calibration_gate(
        source=_usb_source(
            fingerprint,
            controls=(("exposure", -4.0), ("auto_exposure", 0.25)),
        ),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.CALIBRATION_REQUIRED
    assert result.reasons == ("image_control_profile_mismatch",)


def test_usb_gate_compares_current_validation_anchors_to_setup_anchors(
    tmp_path: Path,
) -> None:
    fingerprint = "d" * 64
    validation_anchors = (
        (100.0, 200.0),
        (400.0, 205.0),
        (395.0, 25.0),
        (105.0, 20.0),
    )
    setup = _usb_setup(fingerprint).model_copy(update={"validation_corners_px": validation_anchors})
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )
    current = tuple((x + 1.0, y - 1.0) for x, y in validation_anchors)

    result = evaluate_usb_calibration_gate(
        source=_usb_source(fingerprint, corners=current),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.READY
    assert result.maximum_corner_drift_px is not None
    assert result.maximum_corner_drift_px < 2.0


def test_usb_gate_treats_control_profile_order_as_non_semantic(tmp_path: Path) -> None:
    fingerprint = "d" * 64
    setup = _usb_setup(fingerprint)
    root, _ = _registered(
        tmp_path,
        setup=setup,
        distortion=_distortion(),
        references=_references(),
    )

    result = evaluate_usb_calibration_gate(
        source=_usb_source(
            fingerprint,
            controls=(("auto_exposure", 0.25), ("exposure", -6.0)),
        ),
        setup=setup,
        calibration_root=root,
    )

    assert result.status is CalibrationGateStatus.READY
