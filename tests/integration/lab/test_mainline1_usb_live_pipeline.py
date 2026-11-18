from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_frame_source import UsbFrameSource
from billiards_ai.lab.mainline1_live_runtime import (
    LiveRuntimeEventKind,
    Mainline1LiveRuntime,
    Mainline1LiveRuntimeConfig,
)
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.mainline1_usb_capture import UsbCaptureRequest, UsbDeviceIdentity
from billiards_ai.lab.shot_local.cue import CueTrackingConfig
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.fragments import FragmentTrackingConfig
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate


class _Capture:
    def __init__(self, count: int) -> None:
        self._frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(count)]
        self._properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 160.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 120.0,
            cv2.CAP_PROP_FPS: 10.0,
        }
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, property_id: int, value: float) -> bool:
        self._properties[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self._properties.get(property_id, 0.0)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class _Vision:
    package_identity = VisionPackageIdentity(package_id="a" * 64, manifest_sha256="b" * 64)

    def infer(self, frame: DecodedTrackingFrame) -> VisionFrameObservation:
        cue_x = (50, 50, 60, 70, 70)[frame.frame_id]
        detections = tuple(
            BallInstanceObservation(
                detection_id=f"{name}-{frame.frame_id}",
                bbox=PixelBox(left=x - 4, top=56, right=x + 4, bottom=64),
                center=PixelPoint(x=x, y=60),
                confidence=0.95,
                class_label="ball",
            )
            for name, x in (("cue", cue_x), ("target", 100))
        )
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            package=self.package_identity,
            threshold=0.25,
            detections=detections,
            runtime_ref="integration/usb",
        )


class _Identity:
    def classify(
        self, *, image: np.ndarray, detections: tuple[BallInstanceObservation, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        return {
            item.detection_id: {
                "group": "cue" if item.detection_id.startswith("cue") else "solid",
                "confidence": 0.95,
            }
            for item in detections
        }


def _config() -> Mainline1ShotLocalConfig:
    return Mainline1ShotLocalConfig(
        window=ShotWindowConfig(
            motion_threshold_mm=8,
            motion_confirm_frames=2,
            still_duration_ms=100,
        ),
        cue=CueTrackingConfig(ambiguity_margin=0.08, reacquire_confidence_min=0.55),
        fragment=FragmentTrackingConfig(
            motion_threshold_mm=5,
            continuation_distance_mm=40,
            ambiguity_mm=3,
        ),
        event=ShotLocalEventConfig(
            first_contact_distance_mm=35,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=120,
            cushion_distance_mm=10,
            pocket_distance_mm=15,
            pot_missing_frames=3,
            after_match_distance_mm=20,
            table_length_mm=160,
            table_width_mm=120,
        ),
    )


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [
        (5, (LiveRuntimeEventKind.SHOT_STARTED, LiveRuntimeEventKind.SHOT_PUBLISHED)),
        (4, (LiveRuntimeEventKind.SHOT_STARTED, LiveRuntimeEventKind.SHOT_ABSTAINED)),
    ],
)
def test_real_usb_adapter_feeds_complete_or_abstained_canonical_pipeline(
    tmp_path: Path,
    frame_count: int,
    expected: tuple[LiveRuntimeEventKind, LiveRuntimeEventKind],
) -> None:
    capture = _Capture(frame_count)
    request = UsbCaptureRequest(
        device=UsbDeviceIdentity(
            device_locator=r"USB\VID_32E4&PID_9230\FIT1044-A",
            display_name="DFRobot FIT1044",
            runtime_index=2,
        ),
        width=160,
        height=120,
        fps=10,
        sample_frames=frame_count,
        warmup_frames=0,
    )
    capture.device_fingerprint = request.device.durable_fingerprint
    clock_values = iter((0, 40_000_000, 80_000_000, 120_000_000, 240_000_000))
    source = UsbFrameSource(
        request=request,
        capture_factory=lambda _: capture,
        monotonic_ns=lambda: next(clock_values),
        max_frames=frame_count,
    )
    runtime = Mainline1LiveRuntime(
        source=source,
        calibration=calibrate(
            "camera",
            ((0, 120), (160, 120), (160, 0), (0, 0)),
            table_length_mm=160,
            table_width_mm=120,
        ),
        vision_backend=_Vision(),
        identity_classifier=_Identity(),
        shot_config=_config(),
        runtime_config=Mainline1LiveRuntimeConfig(
            spool_root=tmp_path / "spool",
            output_root=tmp_path / "artifacts",
            maximum_spool_frames=10,
        ),
        capture_setup_id="setup-live",
        setup_fingerprint="c" * 64,
        calibration_artifact_id="d" * 64,
        calibration_revision=1,
    )

    events = runtime.run()

    assert tuple(event.kind for event in events[:2]) == expected
    assert events[-1].kind is LiveRuntimeEventKind.CAPTURE_STOPPED
    assert capture.released is True
    activated = [event for event in events if event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED]
    if frame_count == 5:
        assert len(activated) == 1
        assert activated[0].artifact_path is not None
        assert (activated[0].artifact_path / "event-timeline.json").is_file()
    else:
        assert activated == []
        assert not list((tmp_path / "artifacts").glob("*"))
