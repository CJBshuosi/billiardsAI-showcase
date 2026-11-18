"""Asynchronous heavy-detector anchors and deterministic late-result replay."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureContractError,
    CaptureFrame,
)
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    RealtimeBallTracker,
    RealtimeTrackerConfig,
    TrackedObservation,
    TrackerDiagnosticSnapshot,
    TrackerHealth,
    TrackerHealthState,
)
from billiards_ai.lab.shot_local.observations import GroupIdentityClassifier
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.lab.vision.contracts import (
    VisionFrameObservation,
    make_vision_frame_input,
)
from billiards_ai.shared.calibration.homography import TableCalibration


@dataclass(frozen=True, slots=True)
class AnchorSchedule:
    source_fps: int = 60
    idle_fps: int = 4
    active_fps: int = 8

    def __post_init__(self) -> None:
        if (
            self.source_fps <= 0
            or self.idle_fps <= 0
            or self.active_fps <= 0
            or self.idle_fps > self.source_fps
            or self.active_fps > self.source_fps
        ):
            raise ValueError("anchor schedule rates are invalid")

    def due(self, capture_frame_id: int, *, active: bool) -> bool:
        if capture_frame_id < 0:
            raise ValueError("capture frame ID must be non-negative")
        target = self.active_fps if active else self.idle_fps
        phase = capture_frame_id % self.source_fps
        due_frames = {math.floor(index * self.source_fps / target + 0.5) for index in range(target)}
        return phase in due_frames


@dataclass(frozen=True, slots=True)
class HeavyAnchorResult:
    capture_frame: CaptureFrame
    observation: VisionFrameObservation
    inference_duration_ms: float
    active: bool
    identities: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AnchorWorkerHealth:
    scheduled_frames: int
    accepted_frames: int
    skipped_scheduled_frames: int
    completed_anchors: int
    failed_anchors: int
    identity_classified_anchors: int
    p50_inference_ms: float | None
    p95_inference_ms: float | None
    maximum_inference_ms: float | None
    high_speed_coverage: bool
    last_error: str | None


@dataclass(frozen=True, slots=True)
class _Work:
    frame: CaptureFrame
    active: bool


class HeavyAnchorWorker:
    """Own the heavyweight backend on a bounded worker thread."""

    def __init__(
        self,
        backend: VisionBackend,
        schedule: AnchorSchedule | None = None,
        *,
        maximum_pending: int = 1,
        identity_classifier: GroupIdentityClassifier | None = None,
        identity_refresh_interval_ns: int = 1_000_000_000,
    ) -> None:
        if maximum_pending != 1:
            raise ValueError("product anchor worker requires exactly one pending slot")
        if identity_refresh_interval_ns <= 0:
            raise ValueError("identity refresh interval must be positive")
        self._backend = backend
        self._identity_classifier = identity_classifier
        self._identity_refresh_interval_ns = identity_refresh_interval_ns
        self._last_identity_anchor_ns: int | None = None
        self._schedule = schedule or AnchorSchedule()
        self._pending: queue.Queue[_Work | None] = queue.Queue(maxsize=maximum_pending)
        self._results: queue.SimpleQueue[HeavyAnchorResult] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._scheduled = 0
        self._accepted = 0
        self._skipped = 0
        self._completed = 0
        self._failed = 0
        self._identity_classified = 0
        self._inference_durations_ms: list[float] = []
        self._last_error: str | None = None
        self._last_active_anchor_ns: int | None = None
        self._high_speed_coverage = True
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="mainline1-heavy-anchor",
            daemon=True,
        )
        self._thread.start()

    @property
    def health(self) -> AnchorWorkerHealth:
        with self._lock:
            durations = np.asarray(self._inference_durations_ms, dtype=np.float64)
            return AnchorWorkerHealth(
                scheduled_frames=self._scheduled,
                accepted_frames=self._accepted,
                skipped_scheduled_frames=self._skipped,
                completed_anchors=self._completed,
                failed_anchors=self._failed,
                identity_classified_anchors=self._identity_classified,
                p50_inference_ms=(
                    None if not len(durations) else float(np.percentile(durations, 50))
                ),
                p95_inference_ms=(
                    None if not len(durations) else float(np.percentile(durations, 95))
                ),
                maximum_inference_ms=(None if not len(durations) else float(durations.max())),
                high_speed_coverage=self._high_speed_coverage,
                last_error=self._last_error,
            )

    def submit(self, frame: CaptureFrame, *, active: bool) -> bool:
        if self._closed:
            raise RuntimeError("anchor worker is closed")
        if not self._schedule.due(frame.capture_frame_id, active=active):
            return False
        with self._lock:
            self._scheduled += 1
        try:
            self._pending.put_nowait(_Work(frame=frame, active=active))
        except queue.Full:
            try:
                self._pending.get_nowait()
                self._pending.put_nowait(_Work(frame=frame, active=active))
            except queue.Empty:
                # The worker consumed the pending frame between the full check and
                # replacement. Preserve non-blocking capture and let the caller's
                # next scheduled frame retry.
                pass
            with self._lock:
                self._skipped += 1
                if (
                    active
                    and self._last_active_anchor_ns is not None
                    and frame.captured_at_ns - self._last_active_anchor_ns > 250_000_000
                ):
                    self._high_speed_coverage = False
            return False
        with self._lock:
            self._accepted += 1
        return True

    def poll(self) -> tuple[HeavyAnchorResult, ...]:
        rows: list[HeavyAnchorResult] = []
        while True:
            try:
                rows.append(self._results.get_nowait())
            except queue.Empty:
                return tuple(rows)

    def close(self, *, timeout_seconds: float = 0.5) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                self._pending.put_nowait(None)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("anchor worker did not accept cancellation") from None
                self._thread.join(timeout=0.01)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("anchor worker did not stop within cancellation timeout")

    def _run(self) -> None:
        while True:
            work = self._pending.get()
            if work is None:
                return
            started_at = time.perf_counter_ns()
            try:
                image = work.frame.decode_bgr()
                source = make_vision_frame_input(
                    image,
                    frame_id=work.frame.capture_frame_id,
                    timestamp_ms=work.frame.timestamp_ms,
                )
                observation = self._backend.infer(source)
                identities: Mapping[str, Mapping[str, Any]] = {}
                refresh_identity = (
                    self._identity_classifier is not None
                    and observation.detections
                    and (
                        self._last_identity_anchor_ns is None
                        or work.frame.captured_at_ns - self._last_identity_anchor_ns
                        >= self._identity_refresh_interval_ns
                    )
                )
                if refresh_identity:
                    assert self._identity_classifier is not None
                    identities = self._identity_classifier.classify(
                        image=image,
                        detections=observation.detections,
                    )
                    self._last_identity_anchor_ns = work.frame.captured_at_ns
                duration_ms = (time.perf_counter_ns() - started_at) / 1_000_000
                self._results.put(
                    HeavyAnchorResult(
                        capture_frame=work.frame,
                        observation=observation,
                        inference_duration_ms=duration_ms,
                        active=work.active,
                        identities=identities,
                    )
                )
                with self._lock:
                    self._completed += 1
                    self._inference_durations_ms.append(duration_ms)
                    if refresh_identity:
                        self._identity_classified += 1
                    if work.active:
                        if (
                            self._last_active_anchor_ns is not None
                            and work.frame.captured_at_ns - self._last_active_anchor_ns
                            > 250_000_000
                        ):
                            self._high_speed_coverage = False
                        self._last_active_anchor_ns = work.frame.captured_at_ns
            except Exception as exc:  # worker failure is surfaced through health
                with self._lock:
                    self._failed += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True, slots=True)
class AnchorReplayHealth:
    accepted_anchors: int
    unreplayable_anchors: int
    skipped_anchors: int


@dataclass(frozen=True, slots=True)
class _RetainedFrame:
    source: CaptureFrame
    working_gray: np.ndarray


class AnchorReplayCoordinator:
    """Replay a late anchor through retained source frames before publishing state."""

    def __init__(
        self,
        calibration: TableCalibration,
        tracker_config: RealtimeTrackerConfig,
    ) -> None:
        self._calibration = calibration
        self._tracker_config = tracker_config
        self._frames: deque[_RetainedFrame] = deque(maxlen=tracker_config.history_frames)
        self._tracked: dict[int, TrackedObservation] = {}
        self._tracker: RealtimeBallTracker | None = None
        self._accepted = 0
        self._unreplayable = 0
        self._skipped = 0
        self._lock = threading.RLock()
        self._pending_anchors: queue.Queue[AnchorObservation | None] = queue.Queue(maxsize=1)
        self._accepted_results: queue.SimpleQueue[AnchorObservation] = queue.SimpleQueue()
        self._working_worker_count = tracker_config.maximum_source_lag_frames + 1
        self._working_slots = threading.BoundedSemaphore(self._working_worker_count)
        self._working_queue: queue.Queue[CaptureFrame | None] = queue.Queue(
            maxsize=self._working_worker_count
        )
        self._working_condition = threading.Condition(self._lock)
        self._working_results: dict[int, _RetainedFrame] = {}
        self._working_pending = 0
        self._working_next_frame_id: int | None = None
        self._working_last_submitted_frame_id: int | None = None
        self._working_committed_frame_id: int | None = None
        self._working_submitted_ids: set[int] = set()
        self._working_failure: BaseException | None = None
        self._working_workers_remaining = self._working_worker_count
        self._closed = False
        self._working_threads = tuple(
            threading.Thread(
                target=self._run_working_decode,
                name=f"mainline1-working-decode-{index}",
                daemon=True,
            )
            for index in range(self._working_worker_count)
        )
        for thread in self._working_threads:
            thread.start()
        self._working_commit_thread = threading.Thread(
            target=self._run_working_commit,
            name="mainline1-working-commit",
            daemon=True,
        )
        self._working_commit_thread.start()
        self._thread = threading.Thread(
            target=self._run_async,
            name="mainline1-anchor-replay",
            daemon=True,
        )
        self._thread.start()

    @property
    def current(self) -> TrackedObservation | None:
        with self._lock:
            return None if self._tracker is None else self._tracker.current

    @property
    def health(self) -> AnchorReplayHealth:
        with self._lock:
            return AnchorReplayHealth(
                accepted_anchors=self._accepted,
                unreplayable_anchors=self._unreplayable,
                skipped_anchors=self._skipped,
            )

    @property
    def preprocessing_lag_frames(self) -> int:
        with self._lock:
            return max(0, self._working_pending - 1)

    @property
    def tracked_history(self) -> tuple[TrackedObservation, ...]:
        with self._lock:
            return tuple(
                self._tracked[frame.source.capture_frame_id]
                for frame in self._frames
                if frame.source.capture_frame_id in self._tracked
            )

    @property
    def tracker_health(self) -> TrackerHealth | None:
        with self._lock:
            return None if self._tracker is None else self._tracker.health

    @property
    def tracker_diagnostic_snapshot(self) -> TrackerDiagnosticSnapshot | None:
        with self._lock:
            return None if self._tracker is None else self._tracker.diagnostic_snapshot

    def wait_until_preprocessed(
        self,
        frame_id: int,
        *,
        timeout_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._working_condition:
            while self._working_failure is None and (
                self._working_committed_frame_id is None
                or self._working_committed_frame_id < frame_id
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._working_condition.wait(timeout=remaining)
            if self._working_failure is not None:
                raise CaptureContractError(
                    "working-frame preprocessing failed"
                ) from self._working_failure
            return True

    def retain(self, frame: CaptureFrame) -> None:
        with self._lock:
            if self._working_failure is not None:
                raise CaptureContractError(
                    "working-frame preprocessing failed"
                ) from self._working_failure
            if self._closed:
                raise RuntimeError("anchor replay coordinator is closed")
            if (
                self._working_last_submitted_frame_id is not None
                and frame.capture_frame_id <= self._working_last_submitted_frame_id
            ):
                raise ValueError("retained capture frames must be strictly ordered")
        while not self._working_slots.acquire(timeout=0.01):
            with self._lock:
                if self._working_failure is not None:
                    raise CaptureContractError(
                        "working-frame preprocessing failed"
                    ) from self._working_failure
                if self._closed:
                    raise RuntimeError("anchor replay coordinator is closed")
        with self._lock:
            if self._working_next_frame_id is None:
                self._working_next_frame_id = frame.capture_frame_id
            self._working_last_submitted_frame_id = frame.capture_frame_id
            self._working_submitted_ids.add(frame.capture_frame_id)
            self._working_pending += 1
        self._working_queue.put_nowait(frame)

    def submit(self, anchor: AnchorObservation) -> bool:
        if self._closed:
            raise RuntimeError("anchor replay coordinator is closed")
        try:
            self._pending_anchors.put_nowait(anchor)
        except queue.Full:
            with self._lock:
                self._skipped += 1
            return False
        return True

    def poll_accepted(self) -> tuple[AnchorObservation, ...]:
        rows: list[AnchorObservation] = []
        while True:
            try:
                rows.append(self._accepted_results.get_nowait())
            except queue.Empty:
                return tuple(rows)

    def close(self, *, timeout_seconds: float = 0.5) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + timeout_seconds
        for _ in self._working_threads:
            while True:
                try:
                    self._working_queue.put_nowait(None)
                    break
                except queue.Full:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "working-frame preprocessing did not accept cancellation"
                        ) from None
                    time.sleep(0.005)
        for thread in self._working_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._working_threads):
            raise TimeoutError(
                "working-frame preprocessing did not stop within cancellation timeout"
            )
        self._working_commit_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._working_commit_thread.is_alive():
            raise TimeoutError("working-frame commit did not stop within cancellation timeout")
        while True:
            try:
                self._pending_anchors.put_nowait(None)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("anchor replay did not accept cancellation") from None
                self._thread.join(timeout=0.01)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("anchor replay did not stop within cancellation timeout")
        with self._lock:
            if self._working_failure is not None:
                raise CaptureContractError(
                    "working-frame preprocessing failed"
                ) from self._working_failure

    def _run_working_decode(self) -> None:
        try:
            while True:
                frame = self._working_queue.get()
                if frame is None:
                    return
                working_gray = frame.decode_working_gray(
                    width=self._tracker_config.working_width,
                    height=self._tracker_config.working_height,
                )
                # Replay only consumes frame identity/geometry plus the decoded
                # grayscale plane. Keeping every native JPEG as well would make
                # a detector-latency-sized history unnecessarily expensive.
                replay_source = replace(frame, mjpeg=b"\0")
                with self._working_condition:
                    self._working_results[frame.capture_frame_id] = _RetainedFrame(
                        replay_source,
                        working_gray,
                    )
                    self._working_condition.notify_all()
        except BaseException as exc:
            with self._working_condition:
                if self._working_failure is None:
                    self._working_failure = exc
                self._working_condition.notify_all()
        finally:
            with self._working_condition:
                self._working_workers_remaining -= 1
                self._working_condition.notify_all()

    def _run_working_commit(self) -> None:
        while True:
            with self._working_condition:
                self._working_condition.wait_for(
                    lambda: (
                        self._working_failure is not None
                        or (
                            self._working_next_frame_id is not None
                            and self._working_next_frame_id in self._working_results
                        )
                        or (
                            self._working_next_frame_id is not None
                            and self._working_next_frame_id not in self._working_submitted_ids
                            and bool(self._working_results)
                        )
                        or self._working_workers_remaining == 0
                    )
                )
                if self._working_failure is not None:
                    return
                next_frame_id = self._working_next_frame_id
                if next_frame_id is None:
                    if self._working_workers_remaining == 0:
                        return
                    continue
                retained = self._working_results.pop(next_frame_id, None)
                if (
                    retained is None
                    and next_frame_id not in self._working_submitted_ids
                    and self._working_results
                ):
                    next_frame_id = min(self._working_results)
                    retained = self._working_results.pop(next_frame_id)
                if retained is None:
                    if self._working_workers_remaining == 0:
                        if (
                            self._working_last_submitted_frame_id is not None
                            and next_frame_id <= self._working_last_submitted_frame_id
                        ):
                            self._working_failure = CaptureContractError(
                                "working-frame preprocessing ended before ordered commit"
                            )
                            self._working_condition.notify_all()
                        return
                    continue
                if len(self._frames) == self._frames.maxlen:
                    self._tracked.pop(self._frames[0].source.capture_frame_id, None)
                self._frames.append(retained)
                if (
                    self._tracker is not None
                    and self._tracker.current is not None
                    and self._tracker.health.state is TrackerHealthState.RUNNING
                    and retained.source.capture_frame_id > self._tracker.current.capture_frame_id
                ):
                    frame_step = (
                        retained.source.capture_frame_id - self._tracker.current.capture_frame_id
                    )
                    if (
                        self._tracker_config.allow_accounted_frame_gaps
                        and frame_step > self._tracker_config.maximum_accounted_frame_gap
                    ):
                        self._tracker = None
                    elif self._tracker_config.allow_accounted_frame_gaps or frame_step == 1:
                        row = self._tracker.track_working(
                            retained.source,
                            retained.working_gray,
                            latest_capture_frame_id=retained.source.capture_frame_id,
                        )
                        self._tracked[row.capture_frame_id] = row
                self._working_submitted_ids.remove(retained.source.capture_frame_id)
                self._working_committed_frame_id = retained.source.capture_frame_id
                self._working_next_frame_id = retained.source.capture_frame_id + 1
                self._working_pending -= 1
                self._working_slots.release()
                self._working_condition.notify_all()

    def _run_async(self) -> None:
        while True:
            anchor = self._pending_anchors.get()
            if anchor is None:
                return
            if self._accept_anchor(anchor):
                self._accepted_results.put(anchor)

    def accept(self, anchor: AnchorObservation) -> bool:
        return self._accept_anchor(anchor)

    def _new_tracker(self) -> RealtimeBallTracker:
        return RealtimeBallTracker(self._calibration, self._tracker_config)

    def _accept_anchor(self, anchor: AnchorObservation) -> bool:
        with self._working_condition:
            replay_through_frame_id = self._working_last_submitted_frame_id
            self._working_condition.wait_for(
                lambda: (
                    self._working_failure is not None
                    or replay_through_frame_id is None
                    or (
                        self._working_committed_frame_id is not None
                        and self._working_committed_frame_id >= replay_through_frame_id
                    )
                )
            )
            if self._working_failure is not None:
                self._unreplayable += 1
                return False
            rows = [
                frame
                for frame in self._frames
                if frame.source.capture_frame_id >= anchor.capture_frame_id
            ]
        if (
            not rows
            or rows[0].source.capture_frame_id != anchor.capture_frame_id
            or (
                not self._tracker_config.allow_accounted_frame_gaps
                and any(
                    current.source.capture_frame_id != previous.source.capture_frame_id + 1
                    for previous, current in zip(rows, rows[1:], strict=False)
                )
            )
        ):
            with self._lock:
                self._unreplayable += 1
            return False
        candidate = self._new_tracker()
        try:
            candidate.apply_anchor_working(
                anchor,
                rows[0].source,
                rows[0].working_gray,
            )
            for retained in rows[1:]:
                candidate.track_working(
                    retained.source,
                    retained.working_gray,
                    latest_capture_frame_id=retained.source.capture_frame_id,
                )
            while True:
                assert candidate.current is not None
                with self._lock:
                    catch_up = [
                        frame
                        for frame in self._frames
                        if frame.source.capture_frame_id > candidate.current.capture_frame_id
                    ]
                    if not catch_up:
                        if candidate.health.state is TrackerHealthState.DEGRADED:
                            self._unreplayable += 1
                            return False
                        self._tracker = candidate
                        retained_ids = {frame.source.capture_frame_id for frame in self._frames}
                        for row in candidate.observations:
                            if row.capture_frame_id in retained_ids:
                                self._tracked[row.capture_frame_id] = row
                        self._accepted += 1
                        return True
                for retained in catch_up:
                    candidate.track_working(
                        retained.source,
                        retained.working_gray,
                        latest_capture_frame_id=retained.source.capture_frame_id,
                    )
        except (ValueError, RuntimeError):
            with self._lock:
                self._unreplayable += 1
            return False
