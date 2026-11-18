from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    CaptureSourceKind,
    FrameGeometry,
    UsbSourceMetadata,
    capture_setup_fingerprint,
    load_capture_setup,
    probe_file_source,
)


def _setup(**updates: object) -> CaptureSetup:
    values: dict[str, object] = {
        "capture_setup_id": "table-a-camera-a",
        "table_id": "table-a",
        "camera_device_id": "camera-a",
        "lens_id": "lens-a",
        "mount_revision": "mount-v1",
        "distortion_revision": "distortion-v1",
        "source_kind": CaptureSourceKind.FILE,
        "geometry": FrameGeometry(width=64, height=32, crop=(0, 0, 64, 32)),
        "validation_corners_px": ((1.0, 31.0), (63.0, 31.0), (63.0, 1.0), (1.0, 1.0)),
    }
    values.update(updates)
    return CaptureSetup.model_validate(values)


def _write_video(path: Path, *, width: int = 64, height: int = 32, frames: int = 2) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path


def test_setup_fingerprint_ignores_validation_evidence_but_changes_for_mount() -> None:
    first = _setup()
    second = first.model_copy(update={"validation_corners_px": None})
    moved = first.model_copy(update={"mount_revision": "mount-v2"})

    assert capture_setup_fingerprint(first) == capture_setup_fingerprint(second)
    assert capture_setup_fingerprint(first) != capture_setup_fingerprint(moved)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_capture_models_reject_non_finite_validation_corners(bad_value: float) -> None:
    corners = ((bad_value, 31.0), (63.0, 31.0), (63.0, 1.0), (1.0, 1.0))

    with pytest.raises(ValueError, match="finite corners"):
        _setup(validation_corners_px=corners)
    with pytest.raises(ValueError, match="finite corners"):
        UsbSourceMetadata(
            device_fingerprint="d" * 64,
            geometry=FrameGeometry(width=64, height=32, crop=(0, 0, 64, 32)),
            fps=60,
            source_epoch_id="epoch-1",
            evidence_frame_sha256="e" * 64,
            validation_corners_px=corners,
            image_controls=(("exposure", -6.0),),
        )


def test_probe_file_source_reports_geometry_sha256_and_frame_count(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")

    metadata = probe_file_source(video)

    assert (metadata.geometry.width, metadata.geometry.height) == (64, 32)
    assert metadata.frame_count == 2
    assert len(metadata.source_sha256) == 64
    assert metadata.path == video.resolve()


def test_load_capture_setup_rejects_symlink(tmp_path: Path) -> None:
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(_setup().model_dump_json(), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(setup_path)

    with pytest.raises(ValueError, match="unsafe"):
        load_capture_setup(link)


def test_probe_file_source_rejects_missing_or_symlinked_video(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing or unsafe"):
        probe_file_source(tmp_path / "missing.mp4")

    video = _write_video(tmp_path / "clip.avi")
    link = tmp_path / "clip-link.avi"
    link.symlink_to(video)
    with pytest.raises(ValueError, match="missing or unsafe"):
        probe_file_source(link)
