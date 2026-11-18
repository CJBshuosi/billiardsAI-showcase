"""Single-owner live capture fan-out for the field product."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from time import monotonic, monotonic_ns
from typing import Protocol

from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureBackend,
    CaptureCadenceMonitor,
    CaptureContractError,
    CaptureFrame,
    CaptureHealth,
    CaptureHealthState,
    RealtimeCaptureConfig,
)


class CaptureRecorder(Protocol):
    def append(self, frame: CaptureFrame) -> None: ...

    def close(self) -> object: ...


class CaptureSubscription:
    """Nonblocking preview coalescing or explicitly activated bounded FIFO."""

    allow_accounted_frame_gaps = True

    def __init__(
        self,
        *,
        name: str,
        maximum_pending_frames: int,
        delivery_mode: str = "coalescing",
        maximum_pending_bytes: int | None = None,
    ) -> None:
        if not name.strip() or maximum_pending_frames <= 0:
            raise ValueError("capture subscription configuration is invalid")
        if delivery_mode not in {"coalescing", "fifo"} or (
            delivery_mode == "fifo"
            and (type(maximum_pending_bytes) is not int or maximum_pending_bytes <= 0)
        ):
            raise ValueError("capture subscription delivery configuration is invalid")
        self.name = name
        self._maximum_pending_frames = maximum_pending_frames
        self._maximum_pending_bytes = maximum_pending_bytes
        self._delivery_mode = delivery_mode
        self._active = delivery_mode == "coalescing"
        self._pending_bytes = 0
        self._frames: deque[CaptureFrame] = deque()
        self._condition = threading.Condition()
        self._cancelled = False
        self._failure: BaseException | None = None
        self._continuity_failed = False
        self._sealed = False
        self._last_admitted = None

    @property
    def pending_frame_count(self) -> int:
        with self._condition:
            return len(self._frames)

    @property
    def pending_bytes(self) -> int:
        with self._condition:
            return self._pending_bytes

    def activate(self) -> None:
        """Admit FIFO frames only after the analysis runtime is constructed."""
        with self._condition:
            if self._failure is not None:
                raise self._failure
            if self._cancelled:
                raise CaptureContractError("capture subscription is cancelled")
            self._active = True
            self._condition.notify_all()

    def frames(self, stop: threading.Event) -> Iterator[CaptureFrame]:
        try:
            while not stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            (self._active and bool(self._frames))
                            or self._failure is not None
                            or self._cancelled
                            or self._sealed
                            or stop.is_set()
                        ),
                        timeout=0.1,
                    )
                    if self._cancelled or stop.is_set():
                        return
                    # A FIFO that lost continuity must never drain a successful
                    # prefix before exposing its sticky failure.
                    if self._continuity_failed:
                        raise self._failure
                    if self._active and self._frames:
                        frame = self._frames.popleft()
                        self._pending_bytes -= len(frame.mjpeg)
                    else:
                        failure = self._failure
                        if failure is None and self._sealed:
                            return
                        if failure is None:
                            continue
                        raise failure
                yield frame
        finally:
            if self._delivery_mode == "fifo":
                self.cancel()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._frames.clear()
            self._pending_bytes = 0
            self._condition.notify_all()

    def _publish(self, frame: CaptureFrame) -> None:
        with self._condition:
            if self._cancelled or self._sealed or self._failure is not None:
                return
            if self._delivery_mode == "fifo":
                frame_bytes = len(frame.mjpeg)
                if frame_bytes > self._maximum_pending_bytes or (
                    self._active and self._pending_bytes + frame_bytes > self._maximum_pending_bytes
                ):
                    self._fail(
                        CaptureContractError("capture FIFO subscription byte capacity exceeded"),
                        discard_pending=True,
                    )
                    return
                if self._active and len(self._frames) >= self._maximum_pending_frames:
                    self._fail(
                        CaptureContractError("capture FIFO subscription frame capacity exceeded"),
                        discard_pending=True,
                    )
                    return
                if not self._active:
                    # No authority consumer can receive these startup frames.
                    # Retain only the latest real frame until explicit activation.
                    self._frames.clear()
                    self._pending_bytes = 0
            else:
                while len(self._frames) >= self._maximum_pending_frames:
                    self._pending_bytes -= len(self._frames.popleft().mjpeg)
            self._frames.append(frame)
            self._last_admitted = frame
            self._pending_bytes += len(frame.mjpeg)
            self._condition.notify()

    def _fail(self, failure: BaseException, *, discard_pending: bool = False) -> None:
        with self._condition:
            if self._cancelled or self._sealed or self._failure is not None:
                return
            self._failure = failure
            if discard_pending:
                self._continuity_failed = True
                self._frames.clear()
                self._pending_bytes = 0
            self._condition.notify_all()

    def _seal(self):
        """Stop admission while preserving every admitted FIFO frame."""
        with self._condition:
            if self._failure is not None:
                raise self._failure
            if self._cancelled or self._delivery_mode != "fifo" or not self._active:
                raise CaptureContractError("recording subscription is not active")
            self._sealed = True
            self._condition.notify_all()
            return self._last_admitted


class CaptureHub:
    """Own one capture backend and expose independent product consumers."""

    def __init__(
        self,
        *,
        source: CaptureBackend,
        capture_config: RealtimeCaptureConfig,
        recorder: CaptureRecorder | None = None,
    ) -> None:
        self._source = source
        self._capture_config = capture_config
        self._recorder = recorder
        self._monitor = CaptureCadenceMonitor(expected_fps=capture_config.fps)
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._recorder_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._subscriptions: dict[str, CaptureSubscription] = {}
        self._latest: CaptureFrame | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._cancel_attempted = False
        self._cancel_failure: BaseException | None = None
        self._recorder_closed = False
        self._retired_recorders: list[CaptureRecorder] = []
        self._monitor_closed = False
        self._failure_reason: str | None = None
        self._failure: BaseException | None = None
        self._recording_failure_reason: str | None = None
        self._startup_started_at_ns: int | None = None
        self._first_frame_at_ns: int | None = None
        self._source_ended = False
        self._source_stopped_at_ns: int | None = None
        self._cancel_requested_at_ns: int | None = None

    @property
    def startup_timeout_seconds(self) -> float:
        return self._capture_config.startup_timeout_seconds

    @property
    def startup_preparation_seconds(self) -> float | None:
        return getattr(self._source, "startup_preparation_seconds", None)

    @property
    def preparation_completed_at_ns(self) -> int | None:
        return getattr(self._source, "preparation_completed_at_ns", None)

    @property
    def startup_started_at_ns(self) -> int | None:
        with self._condition:
            return self._startup_started_at_ns

    @property
    def first_frame_at_ns(self) -> int | None:
        with self._condition:
            return self._first_frame_at_ns

    @property
    def failure_reason(self) -> str | None:
        with self._condition:
            return self._failure_reason

    @property
    def startup_diagnostics(self) -> dict[str, object]:
        """Bounded private observations, independent of frame identity and clocks."""
        with self._condition:
            snapshot: dict[str, object] = {
                "started_at_ns": self._startup_started_at_ns,
                "first_frame_at_ns": self._first_frame_at_ns,
                "source_stopped_at_ns": self._source_stopped_at_ns,
                "cancel_requested_at_ns": self._cancel_requested_at_ns,
                "frame_available": self._latest is not None,
                "thread_id": None if self._thread is None else self._thread.ident,
                "failure_type": None if self._failure is None else type(self._failure).__name__,
            }
        # Read the source outside the hub lock. The optional source contract is a
        # fixed snapshot, never a query of the decoder or other native resources.
        try:
            raw = getattr(self._source, "startup_diagnostics", None)
        except Exception as exc:
            snapshot["source_snapshot_failure_type"] = type(exc).__name__
            raw = None
        if isinstance(raw, dict):
            safe: dict[str, object] = {}
            stages = {
                "not_started",
                "source_validation",
                "video_open",
                "mode_validation",
                "first_read",
                "first_encode",
                "frame_ready",
                "video_release",
            }
            for key in ("stage", "failure_stage"):
                value = raw.get(key)
                if isinstance(value, str) and value in stages:
                    safe[key] = value
            status = raw.get("status")
            if isinstance(status, str) and status in {
                "not_started",
                "in_progress",
                "ready",
                "failed",
                "released",
            }:
                safe["status"] = status
            for key in (
                "stage_started_at_ns",
                "stage_finished_at_ns",
                "cancel_requested_at_ns",
                "released_at_ns",
                "thread_id",
            ):
                value = raw.get(key)
                if value is None or (type(value) is int and 0 <= value < 2**64):
                    safe[key] = value
            failure_type = raw.get("failure_type")
            if (
                isinstance(failure_type, str)
                and len(failure_type) <= 64
                and failure_type == snapshot["failure_type"]
            ):
                safe["failure_type"] = failure_type
            snapshot["source"] = safe
        return snapshot

    def subscribe(
        self,
        name: str,
        *,
        maximum_pending_frames: int,
        delivery_mode: str = "coalescing",
        maximum_pending_bytes: int | None = None,
    ) -> CaptureSubscription:
        subscription = CaptureSubscription(
            name=name,
            maximum_pending_frames=maximum_pending_frames,
            delivery_mode=delivery_mode,
            maximum_pending_bytes=maximum_pending_bytes,
        )
        with self._condition:
            if self._closed:
                raise CaptureContractError("capture hub is closed")
            if name in self._subscriptions:
                raise ValueError("capture subscription name is already registered")
            self._subscriptions[name] = subscription
        return subscription

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise CaptureContractError("capture hub is closed")
            if self._thread is not None:
                return
            self._startup_started_at_ns = monotonic_ns()
            self._thread = threading.Thread(
                target=self._run,
                name="mainline1-capture-hub",
                daemon=True,
            )
            self._thread.start()

    def latest_frame(self) -> CaptureFrame | None:
        with self._condition:
            return self._latest

    @property
    def capture_health(self) -> CaptureHealth:
        with self._condition:
            failure_reason = self._failure_reason
        health = self._monitor.health
        if failure_reason is None:
            return health
        return replace(
            health,
            state=CaptureHealthState.DEGRADED,
            reason=failure_reason,
        )

    @property
    def recording_failure_reason(self) -> str | None:
        with self._condition:
            return self._recording_failure_reason

    @property
    def recording_health(self) -> object | None:
        with self._recorder_lock:
            recorder = self._recorder
            return None if recorder is None else getattr(recorder, "health", None)

    @property
    def recording_writer_pending_count(self) -> int | None:
        with self._recorder_lock:
            return getattr(self._recorder, "pending_command_count", None)

    def replace_recorder(self, recorder: CaptureRecorder | None) -> None:
        """Atomically route future frames to a new match recorder."""
        with self._close_lock:
            with self._recorder_lock:
                with self._condition:
                    if self._closed:
                        raise CaptureContractError("capture hub is closed")
                    previous = self._recorder
                    if previous is not recorder:
                        self._recorder = recorder
                        if previous is not None:
                            self._retired_recorders.append(previous)
                        self._recording_failure_reason = None
                        self._condition.notify_all()
            failures = self._close_retired_recorders()
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("capture recorder handoff failed", failures)

    def seal_recording(self, subscription, recorder, *, timeout_seconds, on_sealed):
        """Fence publication and append together; never wait for the consumer here."""
        deadline = monotonic() + timeout_seconds
        if timeout_seconds <= 0 or not self._close_lock.acquire(timeout=timeout_seconds):
            raise CaptureContractError("recording seal timed out")
        try:
            remaining = deadline - monotonic()
            if remaining <= 0 or not self._recorder_lock.acquire(timeout=remaining):
                raise CaptureContractError("recording seal timed out")
            try:
                return self._seal_recording_locked(subscription, recorder, on_sealed)
            finally:
                self._recorder_lock.release()
        finally:
            self._close_lock.release()

    def seal_recording_input(self, recorder, *, timeout_seconds, on_sealed):
        """Fence the writer's admission independently of any analysis subscription."""
        return self.seal_recording(
            None,
            recorder,
            timeout_seconds=timeout_seconds,
            on_sealed=on_sealed,
        )

    def _seal_recording_locked(self, subscription, recorder, on_sealed):
        with self._condition:
            if (
                self._closed
                or self._source_ended
                or self._failure_reason is not None
                or self._recording_failure_reason is not None
                or self._recorder is not recorder
                or (
                    subscription is not None
                    and (self._subscriptions.get(subscription.name) is not subscription)
                )
            ):
                raise CaptureContractError("recording seal authority is unavailable")
            health = getattr(recorder, "health", None)
            if health is not None and getattr(health, "reason", None):
                raise CaptureContractError("recording writer is degraded")
            if subscription is None:
                # _recorder_lock excludes append: this is the actual last frame
                # admitted by this writer, never the latest preview frame.
                cutoff = recorder.last_admitted_frame
                on_sealed()
            else:
                # A sealed iterator can finish immediately, so mark the runtime
                # before making its end-of-input visible.
                with subscription._condition:
                    if (
                        subscription._failure is not None
                        or subscription._cancelled
                        or not subscription._active
                        or subscription._delivery_mode != "fifo"
                    ):
                        raise CaptureContractError("recording subscription failed before seal")
                    on_sealed()
                    cutoff = subscription._seal()
            self._recorder = None
            self._retired_recorders.append(recorder)
            return cutoff

    def _close_retired_recorders(self) -> list[BaseException]:
        """Collect original retirement work; caller owns _close_lock only."""
        pending = []
        failures = []
        for recorder in self._retired_recorders:
            try:
                close_input = getattr(recorder, "close_recording_input", None)
                if close_input is None:
                    recorder.close()
                else:
                    # A retired recording's disk detector still owns its evidence
                    # adapter. Retire capture input without revoking late reads.
                    close_input(timeout_seconds=5)
            except BaseException as exc:
                pending.append(recorder)
                failures.append(exc)
        self._retired_recorders = pending
        return failures

    def _uses_recorder(self, recorder: CaptureRecorder) -> bool:
        """Report current frame-routing identity, not cleanup-only ownership."""
        with self._recorder_lock:
            return self._recorder is recorder

    def _retire_unused_recorder(self, recorder: CaptureRecorder) -> None:
        """Own rejected allocation cleanup without reopening frame admission."""
        with self._close_lock:
            with self._recorder_lock:
                if self._recorder is recorder:
                    raise CaptureContractError("cannot retire the current capture recorder")
                if not any(item is recorder for item in self._retired_recorders):
                    self._retired_recorders.append(recorder)
            failures = self._close_retired_recorders()
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("unused capture recorder cleanup failed", failures)

    def wait_for_frame(
        self, *, after_frame_id: int | None, timeout_seconds: float
    ) -> CaptureFrame | None:
        if timeout_seconds < 0:
            raise ValueError("capture frame wait timeout must not be negative")
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    (
                        self._latest is not None
                        and (
                            after_frame_id is None or self._latest.capture_frame_id > after_frame_id
                        )
                    )
                    or self._failure_reason is not None
                    or self._closed
                ),
                timeout=timeout_seconds,
            )
            if not ready:
                return None
            if self._latest is not None and (
                after_frame_id is None or self._latest.capture_frame_id > after_frame_id
            ):
                return self._latest
            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            if self._cancel_requested_at_ns is None:
                self._cancel_requested_at_ns = monotonic_ns()
            self._stop.set()
            subscriptions = tuple(self._subscriptions.values())
            thread = self._thread
            self._condition.notify_all()
        for subscription in subscriptions:
            subscription.cancel()
        # Admission is already revoked. Serialize cleanup separately so neither
        # capture nor observers need a lock held by a caller joining the worker.
        with self._close_lock:
            if not self._cancel_attempted:
                self._cancel_attempted = True
                try:
                    self._source.cancel()
                except BaseException as exc:
                    self._cancel_failure = exc
            if thread is not None:
                thread.join(timeout=self._capture_config.cancel_timeout_seconds)
                if thread.is_alive():
                    raise CaptureContractError(
                        "capture hub did not stop within timeout"
                    ) from self._cancel_failure
            failures = self._close_retired_recorders()
            if not self._recorder_closed:
                with self._recorder_lock:
                    recorder = self._recorder
                try:
                    if recorder is not None:
                        recorder.close()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    self._recorder_closed = True
            if not self._monitor_closed:
                self._monitor.close()
                self._monitor_closed = True
            if failures:
                with self._condition:
                    if self._recording_failure_reason is None:
                        first = failures[0]
                        self._recording_failure_reason = f"{type(first).__name__}: {first}"
            if self._cancel_failure is not None:
                failures.insert(0, self._cancel_failure)
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("capture hub cleanup failed", failures)

    def _run(self) -> None:
        try:
            for frame in self._source.frames(self._stop):
                if self._stop.is_set():
                    break
                self._monitor.observe(frame)
                with self._recorder_lock:
                    with self._condition:
                        self._latest = frame
                        if self._first_frame_at_ns is None:
                            self._first_frame_at_ns = monotonic_ns()
                        subscriptions = tuple(self._subscriptions.values())
                        recording_healthy = self._recording_failure_reason is None
                        self._condition.notify_all()
                    for subscription in subscriptions:
                        subscription._publish(frame)
                    if recording_healthy:
                        recorder = self._recorder
                        if recorder is not None:
                            try:
                                recorder.append(frame)
                            except BaseException as exc:
                                with self._condition:
                                    self._recording_failure_reason = f"{type(exc).__name__}: {exc}"
                                    self._condition.notify_all()
            self._source_ended = True
            if not self._stop.is_set():
                raise CaptureContractError("capture source ended unexpectedly")
        except BaseException as exc:
            self._source_ended = True
            with self._recorder_lock:
                if not self._stop.is_set():
                    finish_input = getattr(self._recorder, "finish_capture_input", None)
                    if finish_input is not None:
                        try:
                            # Capture owns the tail; a stalled detector must not
                            # be needed to commit the final partial segment.
                            finish_input()
                        except BaseException as tail_failure:
                            with self._condition:
                                self._recording_failure_reason = (
                                    f"{type(tail_failure).__name__}: {tail_failure}"
                                )
                with self._condition:
                    if not self._stop.is_set():
                        self._failure = exc
                        self._failure_reason = f"{type(exc).__name__}: {exc}"
                        subscriptions = tuple(self._subscriptions.values())
                    else:
                        subscriptions = ()
                    self._condition.notify_all()
            for subscription in subscriptions:
                subscription._fail(exc)
        finally:
            with self._condition:
                self._source_stopped_at_ns = monotonic_ns()
                self._condition.notify_all()
