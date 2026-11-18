from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import billiards_ai.lab.mainline1_live_runtime as live_runtime_module
from billiards_ai.lab.mainline1_capture import CaptureSourceKind
from billiards_ai.lab.mainline1_frame_source import FrameSourceHealth, FrameSourceMetadata
from billiards_ai.lab.mainline1_live_runtime import (
    LiveRuntimeEventKind,
    Mainline1LiveRuntime,
    Mainline1LiveRuntimeConfig,
)
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
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


class _FiniteSource:
    metadata = FrameSourceMetadata(CaptureSourceKind.USB, "f" * 64, 160, 120, 10.0)

    def __init__(self, count: int) -> None:
        self._count = count
        self.health = FrameSourceHealth(count, True)

    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        timestamps = (0, 40, 80, 120, 240, 280, 320, 360, 480)
        for frame_id in range(self._count):
            yield DecodedTrackingFrame(
                frame_id=frame_id,
                timestamp_ms=timestamps[frame_id],
                image=np.zeros((120, 160, 3), dtype=np.uint8),
            )


class _WhiteFiniteSource(_FiniteSource):
    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        timestamps = (0, 40, 80, 120, 240, 280, 320, 360, 480)
        for frame_id in range(self._count):
            yield DecodedTrackingFrame(
                frame_id=frame_id,
                timestamp_ms=timestamps[frame_id],
                image=np.full((120, 160, 3), 255, dtype=np.uint8),
            )


class _InterruptedSource(_FiniteSource):
    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        yield from super().__iter__()
        raise OSError("camera disconnected")


class _SlowStartingSource(_FiniteSource):
    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        time.sleep(0.04)
        yield from super().__iter__()


class _ClosableFiniteSource(_FiniteSource):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AwaitingPublicationSource(_FiniteSource):
    def __init__(self, published: threading.Event) -> None:
        super().__init__(5)
        self._published = published

    def __iter__(self) -> Iterator[DecodedTrackingFrame]:
        yield from super().__iter__()
        if not self._published.wait(timeout=1):
            raise RuntimeError("publication event was not streamed")


class _Backend:
    package_identity = VisionPackageIdentity(package_id="a" * 64, manifest_sha256="b" * 64)

    def infer(self, frame) -> VisionFrameObservation:
        cue_x = (50, 50, 60, 70, 70, 80, 90, 90, 90)[frame.frame_id]
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
            runtime_ref="fake/1",
        )


class _SlowBackend(_Backend):
    def infer(self, frame) -> VisionFrameObservation:
        time.sleep(0.08)
        return super().infer(frame)


class _Classifier:
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


def _shot_config() -> Mainline1ShotLocalConfig:
    return Mainline1ShotLocalConfig(
        window=ShotWindowConfig(8, 2, 100),
        cue=CueTrackingConfig(0.08, 0.55),
        fragment=FragmentTrackingConfig(5, 40, 3),
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


def _runtime(tmp_path: Path, count: int) -> Mainline1LiveRuntime:
    return Mainline1LiveRuntime(
        source=_FiniteSource(count),
        calibration=calibrate(
            "camera",
            ((0, 120), (160, 120), (160, 0), (0, 0)),
            table_length_mm=160,
            table_width_mm=120,
        ),
        vision_backend=_Backend(),
        identity_classifier=_Classifier(),
        shot_config=_shot_config(),
        runtime_config=Mainline1LiveRuntimeConfig(
            spool_root=tmp_path / "spool",
            output_root=tmp_path / "artifacts",
            maximum_spool_frames=10,
        ),
        capture_setup_id="setup-1",
        setup_fingerprint="c" * 64,
        calibration_artifact_id="d" * 64,
        calibration_revision=1,
    )


def test_finite_live_source_publishes_one_complete_canonical_shot(tmp_path: Path) -> None:
    events = _runtime(tmp_path, 5).run()

    assert [event.kind for event in events] == [
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_PUBLISHED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    published = events[1]
    assert published.artifact_path is not None
    assert (published.artifact_path / "world-events.json").is_file()
    assert (published.artifact_path / "table-state-after.json").is_file()


def test_source_stop_during_motion_abstains_and_activates_no_artifact(tmp_path: Path) -> None:
    events = _runtime(tmp_path, 4).run()

    assert [event.kind for event in events] == [
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_ABSTAINED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    assert not list((tmp_path / "artifacts").glob("*"))


def test_consecutive_shots_reuse_settled_frame_as_next_stable_baseline(
    tmp_path: Path,
) -> None:
    events = _runtime(tmp_path, 9).run()

    assert [event.kind for event in events] == [
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_PUBLISHED,
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_PUBLISHED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    published = [event for event in events if event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED]
    assert [event.shot_id for event in published] == ["shot-0001", "shot-0002"]
    assert len({event.artifact_id for event in published}) == 2


def test_new_live_epoch_with_different_frames_cannot_reuse_old_artifact(tmp_path: Path) -> None:
    first = _runtime(tmp_path / "first", 5)
    first_event = first.run()[1]

    second = _runtime(tmp_path / "second", 5)
    second._source = _WhiteFiniteSource(5)
    second._calibration = first._calibration
    second._runtime_config = replace(
        second._runtime_config,
        output_root=tmp_path / "first" / "artifacts",
    )
    second_event = second.run()[1]

    assert second_event.artifact_id != first_event.artifact_id
    assert second_event.artifact_path != first_event.artifact_path
    assert second_event.artifact_path is not None
    manifest = json.loads((second_event.artifact_path / "manifest.json").read_text())
    assert (
        manifest["source_epoch_id"]
        != json.loads((first_event.artifact_path / "manifest.json").read_text())["source_epoch_id"]
    )


def test_restart_reuses_spool_root_without_colliding_with_previous_shot(tmp_path: Path) -> None:
    first = _runtime(tmp_path, 5)
    first_event = first.run()[1]

    second = _runtime(tmp_path, 5)
    second._calibration = first._calibration
    second_event = second.run()[1]

    assert second_event.artifact_id != first_event.artifact_id


def test_spool_backpressure_emits_explicit_degradation_and_abstains(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, 4)
    runtime._runtime_config = replace(runtime._runtime_config, maximum_spool_frames=2)

    events = runtime.run()

    assert [event.kind for event in events][-2:] == [
        LiveRuntimeEventKind.CAPTURE_DEGRADED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    assert not any(event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED for event in events)
    assert events[-2].reason == "live spool capacity exhausted"


def test_disconnect_reconnects_through_a_new_validated_epoch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, 4)
    runtime._source = _InterruptedSource(4)
    runtime._reconnect_source_factory = lambda: live_runtime_module.ValidatedReconnectSource(
        source=_FiniteSource(5),
        source_epoch_id="epoch-2",
        capture_setup_id="setup-1",
        setup_fingerprint="c" * 64,
        calibration_artifact_id="d" * 64,
        calibration_revision=1,
    )
    runtime._maximum_reconnect_attempts = 1

    events = runtime.run()

    assert [event.kind for event in events] == [
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_ABSTAINED,
        LiveRuntimeEventKind.CAPTURE_DEGRADED,
        LiveRuntimeEventKind.CAPTURE_RECONNECTED,
        LiveRuntimeEventKind.SHOT_STARTED,
        LiveRuntimeEventKind.SHOT_PUBLISHED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    assert events[1].artifact_id is None
    assert events[5].artifact_id is not None


def test_rejected_reconnect_releases_its_validated_capture_source(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, 1)
    runtime._source = _InterruptedSource(1)
    replacement = _ClosableFiniteSource(1)
    runtime._reconnect_source_factory = lambda: live_runtime_module.ValidatedReconnectSource(
        source=replacement,
        source_epoch_id="epoch-2",
        capture_setup_id="wrong-setup",
        setup_fingerprint="c" * 64,
        calibration_artifact_id="d" * 64,
        calibration_revision=1,
    )
    runtime._maximum_reconnect_attempts = 1

    runtime.run()

    assert replacement.closed is True


def test_live_events_are_streamed_before_the_capture_epoch_stops(tmp_path: Path) -> None:
    published = threading.Event()
    runtime = _runtime(tmp_path, 5)
    runtime._source = _AwaitingPublicationSource(published)

    events = runtime.run(
        event_sink=lambda event: (
            published.set() if event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED else None
        )
    )

    assert published.is_set()
    assert not any(event.kind is LiveRuntimeEventKind.CAPTURE_DEGRADED for event in events)


def test_capture_watchdog_degrades_when_source_produces_no_queue_activity(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path, 1)
    runtime._runtime_config = replace(runtime._runtime_config, capture_watchdog_seconds=0.01)
    monkeypatch.setattr(
        live_runtime_module.queue.Queue,
        "get",
        lambda _self, *, timeout: (_ for _ in ()).throw(live_runtime_module.queue.Empty),
    )

    events = runtime.run()

    assert [event.kind for event in events] == [
        LiveRuntimeEventKind.CAPTURE_DEGRADED,
        LiveRuntimeEventKind.CAPTURE_STOPPED,
    ]
    assert events[0].reason == "capture watchdog expired"


def test_startup_watchdog_allows_bounded_source_initialization_before_first_frame(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, 1)
    runtime._source = _SlowStartingSource(1)
    settings = asdict(replace(runtime._runtime_config, capture_watchdog_seconds=0.01))
    settings["capture_startup_watchdog_seconds"] = 0.2
    runtime._runtime_config = SimpleNamespace(**settings)

    events = runtime.run()

    assert not any(event.kind is LiveRuntimeEventKind.CAPTURE_DEGRADED for event in events)
    assert events[-1].reason == "source_stopped"


def test_artifact_publication_failure_does_not_reconnect_camera(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path, 5)
    reconnects = 0

    def reconnect():
        nonlocal reconnects
        reconnects += 1
        raise AssertionError("artifact failures must not reconnect capture")

    runtime._reconnect_source_factory = reconnect
    runtime._maximum_reconnect_attempts = 1
    monkeypatch.setattr(
        live_runtime_module,
        "publish_live_shot",
        lambda **_: (_ for _ in ()).throw(ValueError("artifact write failed")),
    )

    events = runtime.run()

    assert reconnects == 0
    assert not any(event.kind is LiveRuntimeEventKind.CAPTURE_DEGRADED for event in events)
    assert events[-1].kind is LiveRuntimeEventKind.CAPTURE_STOPPED
    assert events[-1].reason == "artifact write failed"


def test_short_consumer_delay_does_not_trigger_capture_backpressure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, 3)
    runtime._vision_backend = _SlowBackend()
    runtime._runtime_config = replace(runtime._runtime_config, queue_capacity=1)

    events = runtime.run()

    assert not any(event.kind is LiveRuntimeEventKind.CAPTURE_DEGRADED for event in events)
