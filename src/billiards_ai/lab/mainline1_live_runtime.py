"""USB live runtime for the canonical Mainline 1 per-shot pipeline."""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from billiards_ai.lab.mainline1_frame_source import FrameSource
from billiards_ai.lab.mainline1_live_publisher import publish_live_shot
from billiards_ai.lab.mainline1_live_spool import (
    CaptureSpool,
    LiveBackpressureError,
    LiveSpoolContractError,
)
from billiards_ai.lab.mainline1_live_window import (
    IncrementalShotWindowCoordinator,
    LiveShotBoundaryKind,
)
from billiards_ai.lab.mainline1_shot_local_video import (
    Mainline1ShotLocalConfig,
    build_shot_evidence,
)
from billiards_ai.lab.mainline1_shot_product import build_mainline1_shot_product
from billiards_ai.lab.shot_local.active_tracks import ActiveTrackConfig
from billiards_ai.lab.shot_local.models import ObservationFrame
from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    GroupIdentityClassifier,
    collect_observation_frames,
)
from billiards_ai.lab.shot_local.product import (
    build_canonical_shot_analyses,
    shot_local_product_active_config,
)
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.shared.calibration.homography import TableCalibration


class LiveRuntimeEventKind(StrEnum):
    SHOT_STARTED = "SHOT_STARTED"
    SHOT_PUBLISHED = "SHOT_PUBLISHED"
    SHOT_ABSTAINED = "SHOT_ABSTAINED"
    CAPTURE_STOPPED = "CAPTURE_STOPPED"
    CAPTURE_DEGRADED = "CAPTURE_DEGRADED"
    CAPTURE_RECONNECTED = "CAPTURE_RECONNECTED"


@dataclass(frozen=True, slots=True)
class LiveRuntimeEvent:
    kind: LiveRuntimeEventKind
    shot_id: str | None = None
    frame_id: int | None = None
    artifact_id: str | None = None
    artifact_path: Path | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Mainline1LiveRuntimeConfig:
    spool_root: Path
    output_root: Path
    queue_capacity: int = 8
    maximum_spool_frames: int = 3600
    capture_watchdog_seconds: float = 5.0
    capture_startup_watchdog_seconds: float = 30.0
    queue_enqueue_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.queue_capacity <= 0:
            raise ValueError("live queue capacity must be positive")
        if self.maximum_spool_frames <= 0:
            raise ValueError("live spool capacity must be positive")
        if not math.isfinite(self.capture_watchdog_seconds) or self.capture_watchdog_seconds <= 0:
            raise ValueError("capture watchdog seconds must be finite and positive")
        if (
            not math.isfinite(self.capture_startup_watchdog_seconds)
            or self.capture_startup_watchdog_seconds <= 0
        ):
            raise ValueError("capture startup watchdog seconds must be finite and positive")
        if (
            not math.isfinite(self.queue_enqueue_timeout_seconds)
            or self.queue_enqueue_timeout_seconds <= 0
        ):
            raise ValueError("queue enqueue timeout seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class ValidatedReconnectSource:
    """A reopened source that already passed the exact USB calibration gate."""

    source: FrameSource
    source_epoch_id: str
    capture_setup_id: str
    setup_fingerprint: str
    calibration_artifact_id: str
    calibration_revision: int

    def __post_init__(self) -> None:
        if (
            not all(
                value.strip()
                for value in (
                    self.source_epoch_id,
                    self.capture_setup_id,
                    self.setup_fingerprint,
                    self.calibration_artifact_id,
                )
            )
            or self.calibration_revision <= 0
        ):
            raise ValueError("reconnect source authority is incomplete")


@dataclass(frozen=True, slots=True)
class _ProducerStopped:
    error: BaseException | None


class Mainline1LiveRuntime:
    """Capture in one bounded producer and analyze in one ordered consumer."""

    def __init__(
        self,
        *,
        source: FrameSource,
        calibration: TableCalibration,
        vision_backend: VisionBackend,
        identity_classifier: GroupIdentityClassifier,
        shot_config: Mainline1ShotLocalConfig,
        runtime_config: Mainline1LiveRuntimeConfig,
        capture_setup_id: str,
        setup_fingerprint: str,
        calibration_artifact_id: str,
        calibration_revision: int,
        source_epoch_id: str | None = None,
        reconnect_source_factory: Callable[[], ValidatedReconnectSource] | None = None,
        maximum_reconnect_attempts: int = 0,
        active_track_config: ActiveTrackConfig | None = None,
    ) -> None:
        self._source = source
        self._calibration = calibration
        self._vision_backend = vision_backend
        self._identity_classifier = identity_classifier
        self._shot_config = shot_config
        self._runtime_config = runtime_config
        self._capture_setup_id = capture_setup_id
        self._setup_fingerprint = setup_fingerprint
        self._calibration_artifact_id = calibration_artifact_id
        self._calibration_revision = calibration_revision
        self._source_epoch_id = source_epoch_id or uuid4().hex
        if maximum_reconnect_attempts < 0:
            raise ValueError("maximum reconnect attempts must not be negative")
        self._reconnect_source_factory = reconnect_source_factory
        self._maximum_reconnect_attempts = maximum_reconnect_attempts
        self._active_track_config = active_track_config or shot_local_product_active_config()

    def run(
        self,
        *,
        event_sink: Callable[[LiveRuntimeEvent], None] | None = None,
    ) -> tuple[LiveRuntimeEvent, ...]:
        events: list[LiveRuntimeEvent] = []
        reconnect_attempts = 0
        while True:
            epoch_events = list(self._run_once(event_sink=event_sink))
            degraded = any(
                event.kind is LiveRuntimeEventKind.CAPTURE_DEGRADED for event in epoch_events
            )
            can_reconnect = (
                degraded
                and self._reconnect_source_factory is not None
                and reconnect_attempts < self._maximum_reconnect_attempts
            )
            if not can_reconnect:
                events.extend(epoch_events)
                if (
                    event_sink is not None
                    and epoch_events
                    and epoch_events[-1].kind is LiveRuntimeEventKind.CAPTURE_STOPPED
                ):
                    event_sink(epoch_events[-1])
                break
            reconnect_attempts += 1
            replacement: ValidatedReconnectSource | None = None
            try:
                replacement = self._reconnect_source_factory()
                self._validate_reconnect(replacement)
            except (OSError, RuntimeError, TypeError, ValueError):
                if replacement is not None:
                    close = getattr(replacement.source, "close", None)
                    if callable(close):
                        close()
                events.extend(epoch_events)
                if (
                    event_sink is not None
                    and epoch_events
                    and epoch_events[-1].kind is LiveRuntimeEventKind.CAPTURE_STOPPED
                ):
                    event_sink(epoch_events[-1])
                break
            if epoch_events and epoch_events[-1].kind is LiveRuntimeEventKind.CAPTURE_STOPPED:
                epoch_events.pop()
            events.extend(epoch_events)
            self._source = replacement.source
            self._source_epoch_id = replacement.source_epoch_id
            self._record_event(
                events,
                LiveRuntimeEvent(
                    kind=LiveRuntimeEventKind.CAPTURE_RECONNECTED,
                    reason=replacement.source_epoch_id,
                ),
                event_sink,
            )
        return tuple(events)

    def _run_once(
        self,
        *,
        event_sink: Callable[[LiveRuntimeEvent], None] | None,
    ) -> tuple[LiveRuntimeEvent, ...]:
        frame_queue: queue.Queue[DecodedTrackingFrame | _ProducerStopped] = queue.Queue(
            maxsize=self._runtime_config.queue_capacity
        )
        stop = threading.Event()
        producer = threading.Thread(
            target=self._produce,
            args=(frame_queue, stop),
            name="mainline1-usb-capture",
            daemon=True,
        )
        spool = CaptureSpool(
            self._runtime_config.spool_root / self._source_epoch_id,
            maximum_frames=self._runtime_config.maximum_spool_frames,
        )
        coordinator = IncrementalShotWindowCoordinator(self._shot_config.window)
        observations: dict[int, ObservationFrame] = {}
        events: list[LiveRuntimeEvent] = []
        producer.start()
        waiting_for_first_item = True
        try:
            try:
                while True:
                    try:
                        watchdog_seconds = (
                            self._runtime_config.capture_startup_watchdog_seconds
                            if waiting_for_first_item
                            else self._runtime_config.capture_watchdog_seconds
                        )
                        item = frame_queue.get(timeout=watchdog_seconds)
                    except queue.Empty as exc:
                        raise LiveBackpressureError("capture watchdog expired") from exc
                    waiting_for_first_item = False
                    if isinstance(item, _ProducerStopped):
                        reason = (
                            "source_stopped" if item.error is None else type(item.error).__name__
                        )
                        self._record_interruption(
                            events=events,
                            coordinator=coordinator,
                            spool=spool,
                            reason=reason,
                            degraded=item.error is not None,
                            event_sink=event_sink,
                        )
                        break

                    spool.append(item)
                    observation = collect_observation_frames(
                        frames=(item,),
                        vision_backend=self._vision_backend,
                        identity_classifier=self._identity_classifier,
                        calibration=self._calibration,
                    )[0]
                    observations[observation.frame_id] = observation
                    for boundary in coordinator.push(observation):
                        if boundary.window is None:
                            continue
                        window = boundary.window
                        if boundary.kind is LiveShotBoundaryKind.SHOT_STARTED:
                            self._record_event(
                                events,
                                LiveRuntimeEvent(
                                    kind=LiveRuntimeEventKind.SHOT_STARTED,
                                    shot_id=window.shot_id,
                                    frame_id=item.frame_id,
                                ),
                                event_sink,
                            )
                            continue
                        if boundary.kind is not LiveShotBoundaryKind.SHOT_SETTLED:
                            continue
                        decoded = spool.frames_between(window.before_frame_id, window.end_frame_id)
                        shot_observations = tuple(observations[frame.frame_id] for frame in decoded)
                        results = build_shot_evidence(
                            observations=shot_observations,
                            config=self._shot_config,
                        )
                        if len(results) != 1 or not results[0].window.complete:
                            raise ValueError(
                                "settled live boundary did not reproduce one complete shot"
                            )
                        timing: dict[str, float] = {}
                        analyses = build_canonical_shot_analyses(
                            evidence_results=results,
                            observations=shot_observations,
                            images_by_frame={frame.frame_id: frame.image for frame in decoded},
                            event_config=self._shot_config.event,
                            active_config=self._active_track_config,
                            timing_ms=timing,
                        )
                        product = build_mainline1_shot_product(
                            decoded_frames=decoded,
                            observations=shot_observations,
                            results=results,
                            analyses=analyses,
                            calibration=self._calibration,
                            config=self._shot_config,
                            fps=self._source.metadata.fps,
                        )
                        segment = spool.seal_shot(
                            shot_id=window.shot_id,
                            start_frame_id=window.before_frame_id,
                            end_frame_id=window.end_frame_id,
                            complete=True,
                        )
                        artifact = publish_live_shot(
                            product=product,
                            identity=self._identity(window, len(decoded), segment.sha256),
                            output_root=self._runtime_config.output_root,
                        )
                        observations = {
                            frame_id: value
                            for frame_id, value in observations.items()
                            if frame_id >= window.end_frame_id
                        }
                        self._record_event(
                            events,
                            LiveRuntimeEvent(
                                kind=LiveRuntimeEventKind.SHOT_PUBLISHED,
                                shot_id=window.shot_id,
                                frame_id=window.end_frame_id,
                                artifact_id=artifact.artifact_id,
                                artifact_path=artifact.path,
                            ),
                            event_sink,
                        )
            except (LiveBackpressureError, LiveSpoolContractError) as exc:
                self._record_interruption(
                    events=events,
                    coordinator=coordinator,
                    spool=spool,
                    reason=str(exc) or type(exc).__name__,
                    degraded=True,
                    event_sink=event_sink,
                )
            except (RuntimeError, ValueError) as exc:
                self._record_interruption(
                    events=events,
                    coordinator=coordinator,
                    spool=spool,
                    reason=str(exc) or type(exc).__name__,
                    degraded=False,
                    event_sink=event_sink,
                )
        finally:
            stop.set()
            producer.join(timeout=5)
            spool.close()
        return tuple(events)

    def _validate_reconnect(self, replacement: ValidatedReconnectSource) -> None:
        if replacement.source_epoch_id == self._source_epoch_id:
            raise ValueError("reconnect must create a new source epoch")
        expected = (
            self._capture_setup_id,
            self._setup_fingerprint,
            self._calibration_artifact_id,
            self._calibration_revision,
        )
        actual = (
            replacement.capture_setup_id,
            replacement.setup_fingerprint,
            replacement.calibration_artifact_id,
            replacement.calibration_revision,
        )
        if actual != expected:
            raise ValueError("reconnect source authority does not match active calibration")

    def _record_interruption(
        self,
        *,
        events: list[LiveRuntimeEvent],
        coordinator: IncrementalShotWindowCoordinator,
        spool: CaptureSpool,
        reason: str,
        degraded: bool,
        event_sink: Callable[[LiveRuntimeEvent], None] | None,
    ) -> None:
        interrupted = coordinator.interrupt(reason)
        if interrupted.window is not None:
            window = interrupted.window
            try:
                spool.seal_shot(
                    shot_id=f"{window.shot_id}-abstained",
                    start_frame_id=window.before_frame_id,
                    end_frame_id=window.end_frame_id,
                    complete=False,
                )
            except (LiveBackpressureError, LiveSpoolContractError):
                pass
            self._record_event(
                events,
                LiveRuntimeEvent(
                    kind=LiveRuntimeEventKind.SHOT_ABSTAINED,
                    shot_id=window.shot_id,
                    frame_id=window.end_frame_id,
                    reason=reason,
                ),
                event_sink,
            )
        if degraded:
            self._record_event(
                events,
                LiveRuntimeEvent(kind=LiveRuntimeEventKind.CAPTURE_DEGRADED, reason=reason),
                event_sink,
            )
        self._record_event(
            events,
            LiveRuntimeEvent(kind=LiveRuntimeEventKind.CAPTURE_STOPPED, reason=reason),
            None,
        )

    @staticmethod
    def _record_event(
        events: list[LiveRuntimeEvent],
        event: LiveRuntimeEvent,
        event_sink: Callable[[LiveRuntimeEvent], None] | None,
    ) -> None:
        events.append(event)
        if event_sink is not None:
            event_sink(event)

    def _produce(
        self,
        target: queue.Queue[DecodedTrackingFrame | _ProducerStopped],
        stop: threading.Event,
    ) -> None:
        error: BaseException | None = None
        try:
            for frame in self._source:
                if stop.is_set():
                    return
                try:
                    target.put(
                        frame,
                        timeout=self._runtime_config.queue_enqueue_timeout_seconds,
                    )
                except queue.Full as exc:
                    raise LiveBackpressureError("live inference queue capacity exhausted") from exc
        except BaseException as exc:  # preserve exact producer failure for fail-closed shutdown
            error = exc
        while not stop.is_set():
            try:
                target.put(_ProducerStopped(error), timeout=0.1)
                return
            except queue.Full:
                continue

    def _identity(self, window, frame_count: int, source_segment_sha256: str) -> dict[str, Any]:
        metadata = self._source.metadata
        return {
            "schema_version": "1.0",
            "kind": "mainline1_shot_local_video",
            "pipeline_version": "shot-local-unified-product-v1",
            "source_kind": "usb_live",
            "source_ref": metadata.source_ref,
            "source_epoch_id": self._source_epoch_id,
            "source_segment_sha256": source_segment_sha256,
            "capture_setup_id": self._capture_setup_id,
            "setup_fingerprint": self._setup_fingerprint,
            "calibration_artifact_id": self._calibration_artifact_id,
            "calibration_revision": self._calibration_revision,
            "calibration": self._calibration.model_dump(mode="json"),
            "vision_package_id": self._vision_backend.package_identity.package_id,
            "observation_mode": "full_detection",
            "config": asdict(self._shot_config),
            "active_track_config": asdict(self._active_track_config),
            "event_policy": "stable_anchor_tracks_three_layer_contacts_v1",
            "frame_count": frame_count,
            "live_shot_id": window.shot_id,
            "live_start_frame_id": window.start_frame_id,
            "live_end_frame_id": window.end_frame_id,
        }
