"""Optional pixel-snapshot guidance. Owns no camera, recorder, shot or rule state."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from billiards_ai.lab.field_assistance.observations import build_snapshot_table_state
from billiards_ai.lab.field_assistance.visual_motion import (
    RecordingVisualMotion,
    VisualMotionConfig,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.lab.planning.s1 import recommend_s1


@dataclass(frozen=True)
class SnapshotConfig:
    sample_ms: int = 100
    stable_ms: int = 800
    stale_seconds: float = 1.0
    task_seconds: float = 8.0
    preparation_seconds: float = 300.0
    hard_task_seconds: float = 30.0
    max_worker_restarts: int = 1

    def __post_init__(self):
        try:
            valid = all(
                type(value) in {int, float} and math.isfinite(value) and value > 0
                for value in (
                    self.sample_ms,
                    self.stable_ms,
                    self.stale_seconds,
                    self.task_seconds,
                    self.preparation_seconds,
                    self.hard_task_seconds,
                )
            )
        except OverflowError:
            valid = False
        if (
            not valid
            or self.stable_ms < self.sample_ms
            or self.hard_task_seconds < self.task_seconds
            or type(self.max_worker_restarts) is not int
            or self.max_worker_restarts < 0
        ):
            raise ValueError("invalid snapshot timing")


def _calculate(
    connection, factory, frame, calibration, context, generation, epoch, planner, *, cancelled=None
):
    """Spawn boundary: only one image and immutable metadata; no recorder handles."""

    def obsolete():
        if cancelled is not None and cancelled():
            connection.send({"cancelled": True})
            return True
        return False

    try:
        if obsolete():
            return
        observer = factory()
        if obsolete():
            return
        observation = observer.observe(frame, calibration)
        if obsolete():
            return
        if (
            observation.frame_id != frame.capture_frame_id
            or observation.timestamp_ms != frame.timestamp_ms
        ):
            raise ValueError("snapshot observation source mismatch")
        state = build_snapshot_table_state(
            observation,
            calibration_id=calibration.device_id,
            source_ref=f"pixels:{epoch}#{frame.capture_frame_id}",
            scene_generation=generation,
        )
        if obsolete():
            return
        result = planner(state, EightBallContext(group=context))
        if obsolete():
            return
        region = result.region
        connection.send(
            {
                "source_frame_id": result.source_frame_id,
                "calibration_id": result.calibration_id,
                "route": None if result.route is None else result.route.model_dump(mode="json"),
                "region": None
                if region is None
                else {
                    "center": region.center.model_dump(),
                    "radius_mm": region.radius_mm,
                    "next_target_track_id": region.next_target_track_id,
                    "next_pocket": region.next_pocket.value,
                    "evidence": region.evidence,
                },
                "reason": result.reason,
                "region_reason": result.region_reason,
                "actual_cue": (
                    state.cue_ball().center.model_dump()
                    if state.cue_ball().detection_confidence >= 0.85
                    and state.cue_ball().identity_confidence >= 0.85
                    else None
                ),
                "observed_balls": [ball.model_dump(mode="json") for ball in state.balls],
            }
        )
    except Exception as exc:
        connection.send({"error": f"{type(exc).__name__}:{exc}"})
    finally:
        connection.close()


class _TaskReply:
    def __init__(self, connection, request):
        self.connection, self.request = connection, request

    def send(self, value):
        self.connection.send(
            {
                "kind": "result",
                "ticket": self.request["ticket"],
                "generation": self.request["generation"],
                "epoch": self.request["epoch"],
                "value": value,
            }
        )

    def close(self):
        # A calculation owns its reply, not the reusable worker transport.
        pass


def _prepare_resources(factory, resource_preparer=None):
    """Prepare code before models; no table frames enter this phase.

    S1 imports the original Python physics functions under a temporary Numba
    setting. Finish and restore that setting before model imports begin. Both
    remain in the same owned child under the original preparation deadline.
    """
    started = time.monotonic()
    # Called in the owned S1 child (also by the owned component probe). S1 uses
    # built-in CPU/CUDA, so optional out-of-tree device discovery is unnecessary.
    # Do not set this in the parent, ordinary recording or machine environment.
    backend_autoload = "TORCH_DEVICE_BACKEND_AUTOLOAD"
    previous = os.environ.get(backend_autoload)
    os.environ[backend_autoload] = "0"
    try:
        if resource_preparer is not None:
            resource_preparer()
        model_started = time.monotonic()
        observer = factory()
        model_seconds = time.monotonic() - model_started
        return observer, {
            "model_preparation_seconds": model_seconds,
            "resource_preparation_seconds": time.monotonic() - started,
        }
    finally:
        if previous is None:
            os.environ.pop(backend_autoload, None)
        else:
            os.environ[backend_autoload] = previous


def _snapshot_worker(connection, factory, planner, resource_preparer=None, cancel_event=None):
    """Load once, then do no observation/planning until a stable request arrives."""
    try:
        observer, timings = _prepare_resources(factory, resource_preparer)
        connection.send({"kind": "ready", **timings})
        while True:
            request = connection.recv()
            if request["kind"] == "stop":
                return
            if request["kind"] != "calculate":
                raise ValueError("invalid snapshot worker command")
            _calculate(
                _TaskReply(connection, request),
                lambda: observer,
                request["frame"],
                request["calibration"],
                request["context"],
                request["generation"],
                request["epoch"],
                planner,
                cancelled=None if cancel_event is None else lambda: bool(cancel_event.value),
            )
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"kind": "failed", "error": f"{type(exc).__name__}:{exc}"})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class SnapshotGuidance:
    def __init__(
        self,
        *,
        hub,
        session_id,
        source_epoch_id,
        calibration,
        observer_factory,
        journal_path: Path,
        config=None,
        planner=recommend_s1,
        resource_preparer=None,
    ):
        self.hub, self.session_id, self.epoch = hub, session_id, source_epoch_id
        self.calibration, self.factory, self.config = (
            calibration,
            observer_factory,
            config or SnapshotConfig(),
        )
        self.planner = planner
        self.resource_preparer = resource_preparer
        self.path = Path(journal_path)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._process = self._connection = None
        self._busy = False
        self._ticket = None
        self._cancel_event = None
        self._inflight_generation = None
        self._inflight_review = None
        self._task_obsolete = False
        self._reply = None
        self._model_state = "not_started"
        self._model_started_at = 0.0
        self._model_preparation_seconds = None
        self._resource_preparation_seconds = None
        self._model_loads = 0
        self._preparation_failed = False
        self._context = None
        self._context_key = None
        self._motion = None
        self._last_ms = None
        self._last_seen = time.monotonic()
        self._stable_since = None
        self._attempted = False
        self._generation = 0
        self._phase = "waiting"
        self._recommendation = None
        self._history = []
        self._pending_review = None
        self._observed_frame_id = None
        self._feedback = None
        self._feedback_until = None
        self._after_feedback = None
        self._started_at = 0
        self._tasks_started = self._tasks_cancelled = self._tasks_completed = 0
        self._tasks_discarded = 0
        self._max_cancel_ms = 0.0
        self._error = None

    @property
    def active_pid(self):
        with self._lock:
            return (
                None
                if self._process is None or not self._busy or self._task_obsolete
                else self._process.pid
            )

    def set_context(self, context, key=None):
        with self._lock:
            if (context, key) != (self._context, self._context_key):
                self._invalidate()
                self._context, self._context_key = context, key

    def start(self):
        if self._thread is not None or self._stop.is_set():
            return
        self._thread = threading.Thread(target=self._run, name="snapshot-guidance", daemon=True)
        self._thread.start()

    def request_stop(self):
        """Revoke advice without making the capture-end receipt wait for cleanup."""
        self._stop.set()
        event = self._cancel_event
        if event is not None:
            event.value = 1

    @property
    def cleanup_complete(self):
        return (
            self._stop.is_set()
            and (self._thread is None or not self._thread.is_alive())
            and self._process is None
        )

    def _run(self):
        last_id = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    self._start_worker()
                    self._poll_preparation()
                    self._poll_task()
                if self.hub is None:
                    self._stop.wait(0.01)
                    continue
                frame = self.hub.wait_for_frame(after_frame_id=last_id, timeout_seconds=0.05)
                if frame is None:
                    with self._lock:
                        if time.monotonic() - self._last_seen > self.config.stale_seconds:
                            self._invalidate()
                            self._phase = "unknown"
                    # A failed/closed Hub may return None without waiting.
                    self._stop.wait(0.01)
                    continue
                last_id = frame.capture_frame_id
                self.observe(frame)
        except Exception as exc:
            with self._lock:
                self._invalidate()
                self._phase = "failed"
                self._error = f"{type(exc).__name__}:{exc}"
        finally:
            with self._lock:
                self._cancel()
                self._recommendation = None
                if self._stop.is_set():
                    self._phase = "ended"

    def _cancel(self):
        process, connection, busy = self._process, self._connection, self._busy
        if process is None:
            return
        started = time.monotonic()
        if process.is_alive():
            process.terminate()
            process.join(0.25)
            if process.is_alive():
                process.kill()
                process.join(0.5)
        else:
            process.join()
        if process.is_alive():
            # Keep exact ownership and ticket until shutdown can retry cleanup.
            raise RuntimeError("guidance child did not terminate")
        connection.close()
        process.close()
        self._process = self._connection = None
        self._busy = False
        self._ticket = None
        self._inflight_generation = None
        self._inflight_review = None
        self._reply = None
        self._cancel_event = None
        self._model_state = "not_started"
        if busy and not self._task_obsolete:
            self._tasks_cancelled += 1
        self._task_obsolete = False
        self._max_cancel_ms = max(self._max_cancel_ms, (time.monotonic() - started) * 1000)

    def _start_worker(self):
        if self._process is not None or self._preparation_failed or self._stop.is_set():
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        # Only the parent writes this byte; the child reads it. Unlike Event,
        # a terminated reader cannot leave a lock that blocks capture shutdown.
        cancel_event = context.RawValue("b", 0)
        process = context.Process(
            target=_snapshot_worker,
            args=(child, self.factory, self.planner, self.resource_preparer, cancel_event),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._process, self._connection = process, parent
        self._cancel_event = cancel_event
        self._model_state = "preparing"
        self._model_preparation_seconds = self._resource_preparation_seconds = None
        self._model_started_at = time.monotonic()
        self._model_loads += 1

    def _fail_preparation(self, error):
        self._recommendation = None
        self._observed_frame_id = None
        self._pending_review = None
        self._clear_feedback()
        self._cancel()
        self._preparation_failed = True
        self._model_state = "failed"
        self._error = error
        self._phase = "failed"

    def _poll_preparation(self):
        if self._process is None or self._busy:
            return
        if self._model_state == "ready":
            if not self._process.is_alive():
                self._worker_fault("guidance_idle_worker_exit")
            return
        if time.monotonic() - self._model_started_at > self.config.preparation_seconds:
            self._fail_preparation("guidance_preparation_timeout")
            return
        if self._connection.poll():
            try:
                message = self._connection.recv()
            except EOFError:
                message = {"kind": "failed", "error": "guidance_preparation_child_exit"}
            if message.get("kind") != "ready":
                self._fail_preparation(message.get("error", "guidance_preparation_protocol"))
                return
            timings = [
                message.get("model_preparation_seconds"),
                message.get("resource_preparation_seconds"),
            ]
            if not (
                all(
                    type(value) in {int, float} and 0 <= value <= self.config.preparation_seconds
                    for value in timings
                )
                and timings[0] <= timings[1]
            ):
                self._fail_preparation("guidance_preparation_protocol")
                return
            elapsed = time.monotonic() - self._model_started_at
            if elapsed > self.config.preparation_seconds:
                self._fail_preparation("guidance_preparation_timeout")
                return
            self._model_preparation_seconds = timings[0]
            self._resource_preparation_seconds = elapsed
            self._model_state = "ready"
        elif not self._process.is_alive():
            self._fail_preparation("guidance_preparation_child_exit")

    def _invalidate(self, *, preserve_review=False):
        self._recommendation = None
        self._observed_frame_id = None
        self._clear_feedback()
        if not preserve_review:
            self._pending_review = None
        self._attempted = False
        self._stable_since = None
        self._generation += 1
        if self._reply is not None:
            self._reply = None
            self._tasks_cancelled += 1
            self._tasks_discarded += 1
        self._obsolete_task()

    def _obsolete_task(self):
        if self._busy and not self._task_obsolete:
            self._task_obsolete = True
            self._tasks_cancelled += 1
            if self._cancel_event is not None:
                self._cancel_event.value = 1

    def _worker_fault(self, error):
        self._recommendation = None
        self._observed_frame_id = None
        self._pending_review = None
        self._clear_feedback()
        self._reply = None
        self._cancel()
        self._preparation_failed = self._model_loads > self.config.max_worker_restarts
        self._model_state = "failed" if self._preparation_failed else "not_started"
        self._phase, self._error = "failed", error

    def _expire_hot_task(self):
        if self._task_obsolete:
            return
        self._obsolete_task()
        if self._inflight_generation == self._generation:
            if self._inflight_review == self._pending_review:
                self._pending_review = None
            self._phase, self._error = "failed", "guidance_task_timeout"

    def _launch(self, frame):
        if self._model_state != "ready" or self._busy:
            return
        self._ticket = uuid4().hex
        self._inflight_generation = self._generation
        self._inflight_review = self._pending_review
        self._task_obsolete = False
        # The previous terminal reply has been consumed before this slot is reused.
        self._cancel_event.value = 0
        self._started_at = time.monotonic()
        self._connection.send(
            {
                "kind": "calculate",
                "ticket": self._ticket,
                "frame": frame,
                "calibration": self.calibration,
                "context": self._context,
                "generation": self._generation,
                "epoch": self.epoch,
            }
        )
        self._busy = True
        self._attempted = True
        self._tasks_started += 1
        self._phase = "analyzing"
        self._error = None

    def _poll_task(self):
        if self._process is None or not self._busy:
            return
        elapsed = time.monotonic() - self._started_at
        if elapsed > self.config.hard_task_seconds:
            self._worker_fault("guidance_hard_task_timeout")
            return
        if elapsed > self.config.task_seconds:
            self._expire_hot_task()
        if self._connection.poll():
            try:
                message = self._connection.recv()
            except EOFError:
                message = {}
            if time.monotonic() - self._started_at > self.config.task_seconds:
                self._expire_hot_task()
            if (
                not isinstance(message, dict)
                or message.get("kind") != "result"
                or message.get("ticket") != self._ticket
                or message.get("generation") != self._inflight_generation
                or message.get("epoch") != self.epoch
                or not isinstance(message.get("value"), dict)
            ):
                self._worker_fault("guidance_result_scope_mismatch")
                return
            self._busy = False
            self._ticket = None
            self._inflight_generation = None
            review_index = self._inflight_review
            self._inflight_review = None
            if (
                self._task_obsolete
                or self._stop.is_set()
                or message["generation"] != self._generation
                or time.monotonic() - self._last_seen > self.config.stale_seconds
                or message["value"].get("cancelled")
            ):
                self._tasks_discarded += 1
                self._task_obsolete = False
                return
            self._reply = (message["generation"], review_index, message["value"])
        elif not self._process.is_alive():
            self._worker_fault("guidance_child_exit")

    def _clear_feedback(self):
        self._feedback = self._feedback_until = self._after_feedback = None
        if self._phase == "reviewing":
            self._phase = "waiting"

    def _advance_review(self):
        # Called only after the current frame has passed motion, continuity and
        # context checks. A calculated next route is not issued during feedback.
        if (
            self._stop.is_set()
            or self._feedback_until is None
            or time.monotonic() < self._feedback_until
        ):
            return
        recommendation, phase = self._after_feedback
        self._clear_feedback()
        self._recommendation, self._phase = recommendation, phase

    def _collect(self, frame):
        self._poll_task()
        self._advance_review()
        if self._reply is not None:
            generation, review_index, result = self._reply
            self._reply = None
            if generation != self._generation or self._stop.is_set():
                self._tasks_discarded += 1
                return
            if "error" in result:
                self._pending_review = None
                self._observed_frame_id = None
                self._error = result["error"]
                self._phase = "failed"
                return
            self._tasks_completed += 1
            self._error = None
            self._observed_frame_id = result["source_frame_id"]
            feedback = None
            # Only post-motion observations can evaluate the pending prior target.
            if review_index is not None and review_index == self._pending_review:
                row = self._history[review_index]
                self._pending_review = None
                row["observed_frame_id"] = result["source_frame_id"]
                row["actual_cue"] = result["actual_cue"]
                region = row["recommendation"]["region"]
                if region and result["actual_cue"] is not None:
                    x, y = result["actual_cue"]["x_mm"], result["actual_cue"]["y_mm"]
                    cx, cy = region["center"]["x_mm"], region["center"]["y_mm"]
                    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    row["region_result"] = (
                        "uncertain"
                        if abs(distance - region["radius_mm"]) < 20
                        else "inside"
                        if distance < region["radius_mm"]
                        else "outside"
                    )
                if region:
                    feedback = row
            recommendation = result if result["route"] else None
            phase = "ready" if recommendation else "abstained"
            if feedback is not None:
                self._recommendation = None
                self._feedback = feedback
                self._feedback_until = time.monotonic() + 1.0
                self._after_feedback = (recommendation, phase)
                self._phase = "reviewing"
            else:
                self._recommendation, self._phase = recommendation, phase
            self._persist()

    def observe(self, frame, context=...):
        with self._lock:
            if self._stop.is_set():
                return
            if frame.source_epoch_id != self.epoch:
                self._invalidate()
                self._motion = None
                self._phase = "unknown"
                return
            if context is not ...:
                self.set_context(context)
            self._last_seen = time.monotonic()
            ms = frame.timestamp_ms
            if self._last_ms is not None and ms < self._last_ms:
                self._invalidate()
                self._motion = None
                self._phase = "unknown"
                return
            if self._last_ms is not None and ms - self._last_ms < self.config.sample_ms:
                return
            gap = self._last_ms is not None and ms - self._last_ms > min(
                self.config.stable_ms, self.config.stale_seconds * 1000
            )
            self._last_ms = ms
            if gap:
                self._invalidate()
                self._motion = None
                self._phase = "unknown"
            encoded = np.frombuffer(frame.mjpeg, dtype=np.uint8)
            reduction = cv2.IMREAD_REDUCED_COLOR_2 if frame.width >= 1000 else cv2.IMREAD_COLOR
            image = cv2.imdecode(encoded, reduction)
            if image is None:
                raise ValueError("snapshot motion pixels cannot decode")
            if self._motion is None:
                h, w = image.shape[:2]
                scaled = self.calibration.model_copy(
                    update={
                        "corners_px": [
                            (x * w / frame.width, y * h / frame.height)
                            for x, y in self.calibration.corners_px
                        ]
                    }
                )
                self._motion = RecordingVisualMotion(
                    scaled, VisualMotionConfig(), still_duration_ms=self.config.stable_ms
                )
            moving = self._motion.push(image, timestamp_ms=ms)
            if moving:
                preserve_review = self._phase == "moving"
                if self._recommendation is not None:
                    self._history.append(
                        {
                            "motion_frame_id": frame.capture_frame_id,
                            "recommendation": self._recommendation,
                            "observed_frame_id": None,
                            "actual_cue": None,
                            "region_result": "unknown",
                        }
                    )
                    self._pending_review = len(self._history) - 1
                    preserve_review = True
                self._invalidate(preserve_review=preserve_review)
                self._phase = "moving"
                self._persist()
                return
            if self._stable_since is None:
                self._stable_since = ms
            if self._context not in {"open", "solid", "stripe"}:
                self._phase = "waiting_for_context"
                return
            if ms - self._stable_since < self.config.stable_ms:
                return
            self._start_worker()
            self._poll_preparation()
            self._collect(frame)
            if not self._attempted:
                if self._model_state == "ready":
                    self._launch(frame)
                else:
                    self._phase = "failed" if self._preparation_failed else "preparing"

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": "snapshot-guidance/1",
            "session_id": self.session_id,
            "source_epoch_id": self.epoch,
            "reviews": self._history,
        }
        temporary = self.path.with_suffix(".pending")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def projection(self):
        with self._lock:
            return json.loads(
                json.dumps(
                    {
                        "phase": "ended" if self._stop.is_set() else self._phase,
                        "recommendation": None if self._stop.is_set() else self._recommendation,
                        "feedback": None if self._stop.is_set() else self._feedback,
                        "observed_frame_id": None
                        if self._stop.is_set()
                        else self._observed_frame_id,
                        "reviews": self._history,
                        "model_state": self._model_state,
                        "model_loads": self._model_loads,
                        "model_preparation_seconds": self._model_preparation_seconds,
                        "resource_preparation_seconds": self._resource_preparation_seconds,
                        "worker_pid": None if self._process is None else self._process.pid,
                        "tasks_started": self._tasks_started,
                        "tasks_cancelled": self._tasks_cancelled,
                        "tasks_completed": self._tasks_completed,
                        "tasks_discarded": self._tasks_discarded,
                        "worker_restarts": max(0, self._model_loads - 1),
                        "active_pid": self.active_pid,
                        "maximum_cancel_ms": self._max_cancel_ms,
                        "error": self._error,
                        "accuracy_status": "FIELD_UNVALIDATED",
                    }
                )
            )

    def close(self):
        self.request_stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(2)
        with self._lock:
            self._cancel()
            self._recommendation = None
            self._phase = "ended"


@dataclass(frozen=True)
class SnapshotGuidanceFactory:
    observer_factory: object
    config: SnapshotConfig = SnapshotConfig()
    planner: object = recommend_s1
    resource_preparer: object = None

    def __call__(self, *, hub, session_id, source_epoch_id, calibration, journal_path):
        return SnapshotGuidance(
            hub=hub,
            session_id=session_id,
            source_epoch_id=source_epoch_id,
            calibration=calibration,
            observer_factory=self.observer_factory,
            journal_path=journal_path,
            config=self.config,
            planner=self.planner,
            resource_preparer=self.resource_preparer,
        )
