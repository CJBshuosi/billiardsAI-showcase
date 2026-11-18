import json
from pathlib import Path

import numpy as np
import pytest

from billiards_ai.lab.mainline1_calibration import (
    CalibrationInputError,
    create_calibration_artifact,
)
from billiards_ai.lab.mainline1_capture import CaptureSetup, CaptureSourceKind, FrameGeometry
from billiards_ai.shared.calibration.homography import (
    CalibrationAccuracyError,
    pixel_to_table,
)
from billiards_ai.shared.models.calibration import (
    CalibrationReference,
    DistortionModel,
    DistortionParameters,
)
from billiards_ai.shared.models.geometry import Point


def _verified_distortion() -> DistortionParameters:
    return DistortionParameters(
        model=DistortionModel.OPENCV,
        verified=True,
        camera_matrix=((400.0, 0.0, 210.0), (0.0, 400.0, 110.0), (0.0, 0.0, 1.0)),
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_calibration_artifact_maps_clicked_corners_in_required_order(tmp_path: Path) -> None:
    video = tmp_path / "video01.mp4"
    video.write_bytes(b"video")
    setup = CaptureSetup(
        capture_setup_id="setup-a",
        table_id="table-a",
        camera_device_id="detsam2-fixed-camera",
        lens_id="lens-a",
        mount_revision="mount-v1",
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.FILE,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    )
    artifact = create_calibration_artifact(
        video=video,
        frame_index=12,
        frame=np.zeros((220, 420, 3), dtype=np.uint8),
        corners_px=((10.0, 210.0), (410.0, 210.0), (410.0, 10.0), (10.0, 10.0)),
        device_id="detsam2-fixed-camera",
        output_root=tmp_path / "artifacts",
        capture_setup=setup,
    )

    calibration = artifact.calibration
    assert pixel_to_table(calibration, 10, 210).model_dump() == {"x_mm": 0, "y_mm": 0}
    assert pixel_to_table(calibration, 410, 210).model_dump() == {
        "x_mm": 2540,
        "y_mm": 0,
    }
    assert pixel_to_table(calibration, 410, 10).model_dump() == {
        "x_mm": 2540,
        "y_mm": 1270,
    }
    assert pixel_to_table(calibration, 10, 10).model_dump() == {
        "x_mm": 0,
        "y_mm": 1270,
    }
    assert set(item.name for item in artifact.path.iterdir()) == {
        "calibration.json",
        "manifest.json",
        "reference-frame.jpg",
    }
    stored = json.loads((artifact.path / "calibration.json").read_text(encoding="utf-8"))
    assert stored["device_id"] == "detsam2-fixed-camera"
    assert stored["corners_px"] == [
        [10.0, 210.0],
        [410.0, 210.0],
        [410.0, 10.0],
        [10.0, 10.0],
    ]
    manifest = json.loads((artifact.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture_setup_id"] == "setup-a"
    assert len(manifest["capture_setup_fingerprint"]) == 64
    assert manifest["source_video_sha256"] == __import__("hashlib").sha256(b"video").hexdigest()


@pytest.mark.parametrize(
    "corners",
    [
        ((10.0, 210.0), (10.0, 210.0), (410.0, 10.0), (10.0, 10.0)),
        ((10.0, 10.0), (110.0, 10.0), (210.0, 10.0), (310.0, 10.0)),
        ((-1.0, 210.0), (410.0, 210.0), (410.0, 10.0), (10.0, 10.0)),
    ],
)
def test_calibration_artifact_rejects_unsafe_corner_sets(
    tmp_path: Path, corners: tuple[tuple[float, float], ...]
) -> None:
    video = tmp_path / "video01.mp4"
    video.write_bytes(b"video")
    setup = CaptureSetup(
        capture_setup_id="setup-a",
        table_id="table-a",
        camera_device_id="camera",
        lens_id="lens-a",
        mount_revision="mount-v1",
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.FILE,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    )
    with pytest.raises(CalibrationInputError):
        create_calibration_artifact(
            video=video,
            frame_index=0,
            frame=np.zeros((220, 420, 3), dtype=np.uint8),
            corners_px=corners,
            device_id="camera",
            output_root=tmp_path / "artifacts",
            capture_setup=setup,
        )


def test_calibration_artifact_records_verified_distortion_and_reference_error(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    setup = CaptureSetup(
        capture_setup_id="usb-setup",
        table_id="table-a",
        camera_device_id="camera",
        lens_id="wide-120",
        mount_revision="mount-v1",
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.USB,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    )

    artifact = create_calibration_artifact(
        video=video,
        frame_index=0,
        frame=np.zeros((220, 420, 3), dtype=np.uint8),
        corners_px=((10.0, 210.0), (410.0, 210.0), (410.0, 10.0), (10.0, 10.0)),
        device_id="camera",
        output_root=tmp_path / "artifacts",
        capture_setup=setup,
        distortion=_verified_distortion(),
        references=(CalibrationReference(pixel=(210.0, 110.0), table=Point(x_mm=1270, y_mm=635)),),
    )

    manifest = json.loads((artifact.path / "manifest.json").read_text())
    assert artifact.calibration.distortion.model is DistortionModel.OPENCV
    assert manifest["reference_count"] == 1
    assert manifest["maximum_reference_error_mm"] <= 20
    assert manifest["distortion"]["model"] == "OPENCV"


def test_calibration_artifact_rejects_reference_error_over_20mm(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    setup = CaptureSetup(
        capture_setup_id="usb-setup",
        table_id="table-a",
        camera_device_id="camera",
        lens_id="wide-120",
        mount_revision="mount-v1",
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.USB,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    )

    with pytest.raises(CalibrationAccuracyError):
        create_calibration_artifact(
            video=video,
            frame_index=0,
            frame=np.zeros((220, 420, 3), dtype=np.uint8),
            corners_px=(
                (10.0, 210.0),
                (410.0, 210.0),
                (410.0, 10.0),
                (10.0, 10.0),
            ),
            device_id="camera",
            output_root=tmp_path / "artifacts",
            capture_setup=setup,
            distortion=_verified_distortion(),
            references=(CalibrationReference(pixel=(210.0, 110.0), table=Point(x_mm=0, y_mm=0)),),
        )
