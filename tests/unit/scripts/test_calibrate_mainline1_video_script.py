import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from billiards_ai.lab.mainline1_capture import CaptureSetup, CaptureSourceKind, FrameGeometry

SCRIPT = Path("scripts/algorithm/calibrate_mainline1_video.py")


def _module():
    spec = importlib.util.spec_from_file_location("calibrate_mainline1_video", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_prints_saved_calibration_path(tmp_path: Path, capsys) -> None:
    module = _module()
    video = tmp_path / "video01.mp4"
    video.write_bytes(b"video")
    setup_path = tmp_path / "capture-setup.json"
    setup = CaptureSetup(
        capture_setup_id="setup-a",
        table_id="table-a",
        camera_device_id="fixed-camera",
        lens_id="lens-a",
        mount_revision="mount-v1",
        distortion_revision="distortion-v1",
        source_kind=CaptureSourceKind.FILE,
        geometry=FrameGeometry(width=20, height=12, crop=(0, 0, 20, 12)),
    )
    setup_path.write_text(setup.model_dump_json(), encoding="utf-8")
    artifact = SimpleNamespace(
        artifact_id="calibration-123",
        path=tmp_path / "calibration-123",
        calibration_path=tmp_path / "calibration-123" / "calibration.json",
    )
    observed = {}
    registered = {}

    def server_runner(**kwargs):
        observed.update(kwargs)
        return artifact

    def registry_runner(root, selected_artifact, selected_setup):
        registered.update(root=root, artifact=selected_artifact, setup=selected_setup)
        return SimpleNamespace(revision=3, setup_fingerprint="a" * 64)

    status = module.main(
        [
            "--video",
            str(video),
            "--output-root",
            str(tmp_path / "outputs"),
            "--capture-setup",
            str(setup_path),
            "--port",
            "8767",
        ],
        frame_loader=lambda path: SimpleNamespace(
            frame=np.zeros((12, 20, 3), dtype=np.uint8), frame_index=7
        ),
        server_runner=server_runner,
        registry_runner=registry_runner,
    )

    assert status == 0
    assert observed["video"] == video
    assert observed["frame_index"] == 7
    assert observed["device_id"] == "fixed-camera"
    assert observed["port"] == 8767
    assert observed["capture_setup"] == setup
    assert registered == {
        "root": (tmp_path / "outputs").resolve(),
        "artifact": artifact,
        "setup": setup,
    }
    assert json.loads(capsys.readouterr().out) == {
        "artifact_id": "calibration-123",
        "calibration": str(artifact.calibration_path),
        "capture_setup_fingerprint": "a" * 64,
        "capture_setup_id": "setup-a",
        "output": str(artifact.path),
        "revision": 3,
        "status": "PASS",
    }


def test_cli_rejects_device_id_that_disagrees_with_capture_setup(tmp_path: Path, capsys) -> None:
    module = _module()
    video = tmp_path / "video01.mp4"
    video.write_bytes(b"video")
    setup_path = tmp_path / "capture-setup.json"
    setup_path.write_text(
        CaptureSetup(
            capture_setup_id="setup-a",
            table_id="table-a",
            camera_device_id="fixed-camera",
            lens_id="lens-a",
            mount_revision="mount-v1",
            distortion_revision="distortion-v1",
            source_kind=CaptureSourceKind.FILE,
            geometry=FrameGeometry(width=20, height=12, crop=(0, 0, 20, 12)),
        ).model_dump_json(),
        encoding="utf-8",
    )

    status = module.main(
        [
            "--video",
            str(video),
            "--output-root",
            str(tmp_path / "outputs"),
            "--capture-setup",
            str(setup_path),
            "--device-id",
            "different-camera",
        ],
        frame_loader=lambda _: (_ for _ in ()).throw(AssertionError("frame loaded")),
    )

    assert status == 2
    assert "does not match capture setup" in capsys.readouterr().err
