"""Managed disk recovery resolves original semantics without opening a camera."""

from pathlib import Path

import numpy as np
import pytest

from billiards_ai.lab.field_assistance.recording_recovery import prepare_recording_recovery_factory
from billiards_ai.lab.mainline1_calibration import create_calibration_artifact
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import CaptureSourceKind, FrameGeometry
from tests.unit.lab.test_mainline1_capture import _setup


@pytest.mark.parametrize("kind", [CaptureSourceKind.FILE, CaptureSourceKind.USB])
def test_recovery_preparation_uses_real_setup_and_registry_without_capture(
    tmp_path, monkeypatch, kind
):
    from billiards_ai.lab import mainline1_native_capture as native
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    def forbidden(*a, **kw):
        pytest.fail("disk recovery opened camera/video backend")

    monkeypatch.setattr(native, "NativeCaptureSubprocessBackend", forbidden)
    monkeypatch.setattr(scenario, "ScenarioVideoCaptureBackend", forbidden)
    setup = _setup(
        source_kind=kind,
        geometry=FrameGeometry(width=1920, height=1080, crop=(0, 0, 1920, 1080)),
        requested_fps=60,
        device_locator_fingerprint="a" * 64,
    )
    path = tmp_path / "setup.json"
    path.write_text(setup.model_dump_json())
    video = tmp_path / "source.fixture"
    video.write_bytes(b"calibration source identity")
    registry = tmp_path / "calibrations"
    artifact = create_calibration_artifact(
        video=video,
        frame_index=0,
        frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
        corners_px=((10.0, 1070.0), (1910.0, 1070.0), (1910.0, 10.0), (10.0, 10.0)),
        device_id=setup.camera_device_id,
        output_root=registry,
        capture_setup=setup,
    )
    register_calibration(registry, artifact, setup)
    config = Path(__file__).resolve().parents[4] / "configs/algorithm/shot-local.yaml"
    options = dict(
        capture_setup_path=path,
        calibration_root=registry,
        config_path=config,
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
        source_kind=kind,
    )
    factory = prepare_recording_recovery_factory(
        **options, device_fingerprint="a" * 64 if kind is CaptureSourceKind.USB else None
    )
    assert factory.session_policy == "recording-v1"
    assert callable(factory.recover_interrupted_recordings)
    assert callable(factory.recover_recording)
    assert not (tmp_path / "rings").exists()
    with pytest.raises(ValueError, match="setup changed"):
        prepare_recording_recovery_factory(**options, device_fingerprint="b" * 64)


def test_participating_writer_excludes_recovery_until_its_close(tmp_path):
    from billiards_ai.lab.mainline1_session_recording import (
        ContinuousSessionRecorder,
        LocalStorageProbe,
        SessionRecordingConfig,
    )

    class Probe(LocalStorageProbe):
        def benchmark_write_mbps(self, root):
            return 1024.0

    options = dict(
        root=tmp_path / "session-recording",
        session_id="owned",
        expected_fps=60,
        storage_probe=Probe(),
        guard_writer=True,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    writer = ContinuousSessionRecorder(**options)
    try:
        with pytest.raises(TimeoutError, match="pinned"):
            ContinuousSessionRecorder(**options)
    finally:
        writer.close()
    second = ContinuousSessionRecorder(**options)
    second.close()
