from pathlib import Path

import numpy as np

from billiards_ai.lab.mainline1_calibration import create_calibration_artifact
from billiards_ai.lab.mainline1_calibration_registry import (
    register_calibration,
    resolve_active_calibration,
)
from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    CaptureSourceKind,
    FrameGeometry,
    capture_setup_fingerprint,
)


def _setup(mount_revision: str = "mount-v1") -> CaptureSetup:
    return CaptureSetup(
        capture_setup_id="table-a-camera-a",
        table_id="table-a",
        camera_device_id="camera-a",
        lens_id="lens-a",
        mount_revision=mount_revision,
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.FILE,
        geometry=FrameGeometry(width=420, height=220, crop=(0, 0, 420, 220)),
    )


def _artifact(tmp_path: Path, *, setup: CaptureSetup, offset: float = 0):
    video = tmp_path / f"video-{offset}.mp4"
    video.write_bytes(f"video-{offset}".encode())
    return create_calibration_artifact(
        video=video,
        frame_index=0,
        frame=np.zeros((220, 420, 3), dtype=np.uint8),
        corners_px=(
            (10.0 + offset, 210.0),
            (410.0 + offset, 210.0),
            (410.0 + offset, 10.0),
            (10.0 + offset, 10.0),
        ),
        device_id=setup.camera_device_id,
        output_root=tmp_path / "calibrations",
        capture_setup=setup,
    )


def test_registered_calibration_resolves_only_exact_setup(tmp_path: Path) -> None:
    setup = _setup()
    artifact = _artifact(tmp_path, setup=setup)
    register_calibration(tmp_path / "calibrations", artifact, setup)

    record = resolve_active_calibration(tmp_path / "calibrations", capture_setup_fingerprint(setup))

    assert record is not None
    assert record.artifact_id == artifact.artifact_id
    assert (
        resolve_active_calibration(
            tmp_path / "calibrations", capture_setup_fingerprint(_setup("mount-v2"))
        )
        is None
    )


def test_new_revision_supersedes_old_record_without_mutating_artifact(tmp_path: Path) -> None:
    setup = _setup()
    first = _artifact(tmp_path, setup=setup)
    first_manifest = (first.path / "manifest.json").read_bytes()
    second = _artifact(tmp_path, setup=setup, offset=2)

    first_record = register_calibration(tmp_path / "calibrations", first, setup)
    second_record = register_calibration(tmp_path / "calibrations", second, setup)

    assert first_record.revision == 1
    assert second_record.revision == 2
    assert (
        resolve_active_calibration(
            tmp_path / "calibrations", capture_setup_fingerprint(setup)
        ).artifact_id
        == second.artifact_id
    )
    assert (first.path / "manifest.json").read_bytes() == first_manifest
