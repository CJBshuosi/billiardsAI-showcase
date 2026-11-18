"""Concurrent USB authority adapter for the established field HTTP projection."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict
from typing import Protocol

from billiards_ai.edge.route_assistance.records import LiveShotMediaReference, RecordedMediaV1
from billiards_ai.edge.route_assistance.rule_facts import (
    RuleFactDeadlineExceeded,
    build_abstained_realtime_rule_facts_with_preimage,
    build_realtime_rule_facts_with_preimage,
)
from billiards_ai.edge.route_assistance.turn_confirmation import TurnConfirmationRef
from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
from billiards_ai.lab.field_assistance.live_state import ReplayProjection, ReplayTimelineFrame
from billiards_ai.lab.mainline1_native_capture import NativeCaptureProtocolError
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    LiveSettledShot,
    ProductRealtimeEvent,
    ProductRealtimeEventChannel,
)
from billiards_ai.shared.calibration.homography import table_to_pixel


def match_ready_for_shots(match):
    return bool(
        match.red_user_id
        and match.blue_user_id
        and match.breaker
        and match.active_role
        and (match.rule_profile_id is None or match.settings_frozen)
    )


def session_has_live_activity(store, session_id):
    """Persisted intent or any original start keeps an OPEN match active until end."""
    session = store.load_session(session_id)
    if session.status.value != "open":
        return False
    return match_ready_for_shots(session.match) or bool(store.list_shots(session_id))


class LiveProjectionSource(Protocol):
    @property
    def restart_required(self) -> bool: ...
    def current(self) -> ReplayProjection: ...
    def poll_capture_ready(self) -> bool: ...
    def apply_control(self, payload: dict[str, object]) -> None: ...
    def wait_until_ready(self, *, timeout_seconds: float) -> bool: ...
    def end_session(self) -> None: ...
    def close(self) -> None: ...
    def shutdown(self) -> None: ...


class ReplayProjectionSource:
    def __init__(self, clock, frames):
        self.clock = clock
        self.frames = frames

    def current(self):
        return self.clock.projection()

    @property
    def restart_required(self):
        return False

    def poll_capture_ready(self):
        return True

    def apply_control(self, payload):
        command = payload.get("command")
        if command not in {"start", "pause", "reset"}:
            raise ValueError("unsupported replay control")
        getattr(self.clock, command)()

    def wait_until_ready(self, *, timeout_seconds):
        return True

    def end_session(self):
        self.clock.pause()

    def close(self):
        self.frames.close()

    def shutdown(self):
        self.close()


def _persisted_shot_number(shot_id: object) -> int:
    if not isinstance(shot_id, str) or re.fullmatch(r"shot-[0-9]+", shot_id) is None:
        raise ValueError("persisted shot sequence cannot prove a restart offset")
    number = int(shot_id.removeprefix("shot-"))
    if number <= 0 or shot_id not in {f"shot-{number}", f"shot-{number:04d}"}:
        raise ValueError("persisted shot sequence cannot prove a restart offset")
    return number


def _live_shot_number(shot_id: str) -> int:
    match = re.fullmatch(r"shot-([0-9]+)", shot_id)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError("authoritative shot identity is invalid")
    return int(match.group(1))


def _restart_shot_offset(store, session_id: str, shots: tuple[object, ...]) -> int:
    retained_numbers = []
    for compact_sequence, shot in enumerate(shots, 1):
        if shot.sequence != compact_sequence:
            raise ValueError("persisted shot sequence cannot prove a restart offset")
        retained_numbers.append(_persisted_shot_number(shot.shot_id))
    if retained_numbers != sorted(set(retained_numbers)):
        raise ValueError("persisted shot sequence cannot prove a restart offset")

    accounted_numbers = set(retained_numbers)
    for audit in store.list_admin_audit(session_id):
        if audit.get("command_type") != "discard_incomplete_shot":
            continue
        payload = audit.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("persisted shot sequence cannot prove a restart offset")
        accounted_numbers.add(_persisted_shot_number(payload.get("shot_id")))
    if not accounted_numbers:
        return 0
    restart_offset = max(accounted_numbers)
    if accounted_numbers != set(range(1, restart_offset + 1)):
        raise ValueError("persisted shot sequence cannot prove a restart offset")
    return restart_offset


def confirmation_reference(payload: object) -> TurnConfirmationRef:
    if not isinstance(payload, dict):
        raise ValueError("exact confirmation_ref is required")
    try:
        return TurnConfirmationRef(**payload)
    except (TypeError, AttributeError) as exc:
        raise ValueError("invalid confirmation_ref") from exc


class USBLiveProjectionSource:
    """One epoch/session, one capture runner and an independently draining consumer.

    The factory must return a handle exposing run() and close(); close cancels
    capture, while run owns runtime cleanup. No analysis worker is started here.
    """

    def __init__(
        self, *, store, session_id, source_epoch_id, calibration_id, runtime_factory, match_lock
    ):
        self.store = store
        self.session_policy = store.get_session_policy(session_id)
        if getattr(runtime_factory, "session_policy", "legacy") != self.session_policy:
            raise ValueError("runtime session recording policy mismatch")
        self.session_id = session_id
        self.source_epoch_id = source_epoch_id
        self.lock = match_lock
        self._stopped = threading.Event()
        self.coordinator = LiveMatchCoordinator(
            store=store,
            session_id=session_id,
            source_epoch_id=source_epoch_id,
            calibration_id=calibration_id,
        )
        self.channel = ProductRealtimeEventChannel(
            shot_admission=self._match_ready_for_shots,
            recording_evidence_sink=(
                self._persist_recording_event if self.session_policy == "recording-v1" else None
            ),
        )
        shots = store.list_shots(session_id)
        restart_shot_offset = _restart_shot_offset(store, session_id, shots)
        if any(shot.settled_ms is None for shot in shots):
            raise ValueError("cannot resume an unsettled persisted shot")
        self._last_snapshot = self.coordinator.snapshot
        self._shot_count = restart_shot_offset
        self.runtime = runtime_factory(
            session_id, self.channel, restart_shot_offset, source_epoch_id
        )
        self._closed = False
        self._close_lock = threading.RLock()
        self._shutdown_complete = False
        self._runtime_failure = None
        self._runtime_failure_collected = False
        self._cancellation_failure = None
        self._ended = False
        self._recording_end_prepared = False
        self._recording_end_requested = False
        self._capture_end_receipt = None
        self._recording_finalizer = None
        self._failure = None
        self._snapshot_projection_failure = None
        self._restart_reason = None
        self._validated_ready = False
        self._runtime_exited = threading.Event()
        self._phase = "stable"
        self._position = 0
        self._generation = 0
        self._observations = ()
        self._active_shot_id = None
        self._active_start_frame_id = None
        self._last_capture_action = "等待击球"
        self._runner = threading.Thread(target=self._run, name="field-usb-runtime", daemon=True)
        self._consumer = threading.Thread(
            target=self._consume, name="field-usb-events", daemon=True
        )

    def _match_ready_for_shots(self):
        if self._stopped.is_set():
            return False
        match = self.store.load_session(self.session_id).match
        return match_ready_for_shots(match)

    def recording_media_capacity_available(self):
        """Only writer pressure gates ordinary background media, never match activity."""
        health = getattr(self.runtime, "recording_writer_health", None)
        pending = getattr(health, "pending_chunks", None)
        return type(pending) is int and pending == 0

    def media_encoding_idle(self):
        """Corroborate empty-match admission with current camera and writer health."""
        with self.lock:
            video = _component_health(
                getattr(self.runtime, "video_health", None), fallback_status="unavailable"
            )
            raw_recording = getattr(self.runtime, "recording_writer_health", None)
            if raw_recording is None:
                raw_recording = getattr(self.runtime, "recording_health", None)
            recording = _component_health(raw_recording, fallback_status="unavailable")
            pending = getattr(self.runtime, "recording_writer_pending_count", None)
            if pending is None:
                pending = getattr(raw_recording, "pending_chunks", None)
            return bool(
                not self._stopped.is_set()
                and not self.restart_required
                and video["status"] == "ready"
                and recording["status"] == "ready"
                and type(pending) is int
                and pending == 0
            )

    def start(self):
        self._consumer.start()
        self._runner.start()

    def wait_until_ready(self, *, timeout_seconds):
        wait = getattr(self.runtime, "wait_until_ready", None)
        if wait is None or not wait(timeout_seconds=timeout_seconds):
            return False
        with self.lock:
            self._collect_runtime_failure()
            ready = (
                not self._stopped.is_set()
                and not self._runtime_exited.is_set()
                and self._failure is None
                and self._restart_reason is None
            )
            if ready and not self._validated_ready:
                self._validated_ready = True
                self._generation += 1
            return ready

    def poll_capture_ready(self):
        with self.lock:
            self._collect_runtime_failure()
            if self._validated_ready:
                return True
            if (
                self._stopped.is_set()
                or self._runtime_exited.is_set()
                or self._failure is not None
                or self._restart_reason is not None
            ):
                return False
        return self.wait_until_ready(timeout_seconds=0.001)

    def _run(self):
        try:
            self.runtime.run()
        except NativeCaptureProtocolError as exc:
            if self._stopped.is_set():
                self._runtime_failure = exc
            else:
                reason = f"NativeCaptureProtocolError: {exc}"
                self.channel.publish(
                    ProductRealtimeEvent(
                        "CAPTURE_DEGRADED",
                        reason=reason,
                    )
                )
        except BaseException as exc:
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            # Never wait for the match lock here: end/rollover may own it
            # while joining this runner. The owner collects this original error.
            self._runtime_failure = exc
            self.channel.fail(reason)
        finally:
            self._runtime_exited.set()
            if not self._stopped.is_set() and not self._recording_end_requested:
                self.channel.fail("capture stopped unexpectedly")

    def _consume(self):
        while not self._stopped.is_set():
            # Timed locking lets close() join while an HTTP control holds the lock.
            if not self.lock.acquire(timeout=0.05):
                continue
            try:
                # A waiter may acquire after service revocation released the
                # match lock. Leave every arrived boundary for close's drain.
                if self._stopped.is_set():
                    return
                self._collect_runtime_failure()
                if self._failure is not None:
                    self._degrade(self._failure)
                    return
                event = self.channel.receive(timeout_seconds=0)
                if event is not None:
                    self._apply_event(event)
            except Exception as exc:
                # Persistence/I/O faults are terminal authority failures, just
                # like invalid event lineage. Never leave a dead consumer READY.
                self._degrade(f"live event failure: {type(exc).__name__}: {exc}")
            finally:
                self.lock.release()
            self._stopped.wait(0.01)

    def _degrade(self, reason):
        self._failure = reason
        self._validated_ready = False
        self._phase = "degraded"
        self._observations = ()
        self._generation += 1
        self.channel.fail(reason)
        try:
            self.runtime.close()
        except BaseException as exc:
            self._cancellation_failure = exc
            self._failure += f"; capture cancellation: {type(exc).__name__}: {exc}"
        finally:
            self._stopped.set()

    def fail(self, reason):
        """Revoke authority when the owning HTTP service cannot read its state."""
        with self.lock:
            if self._failure is None:
                self._degrade(reason)

    def _collect_runtime_failure(self):
        # Called under the match lock; the runner publishes without acquiring it.
        if self._runtime_failure is not None and not self._runtime_failure_collected:
            exc = self._runtime_failure
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            self._failure = reason if self._failure is None else f"{self._failure}; {reason}"
            self._runtime_failure_collected = True
            self._validated_ready = False
            self._phase = "degraded"

    @property
    def ready_to_end(self):
        with self.lock:
            self._collect_runtime_failure()
            if self.session_policy == "recording-v1" and self._capture_end_receipt is not None:
                return True
            return (
                (self.session_policy == "legacy" or self._recording_end_prepared)
                and self._closed
                and self._failure is None
                and self._restart_reason is None
            )

    @property
    def failure_reason(self):
        with self.lock:
            self._collect_runtime_failure()
            runtime_reason = getattr(self.runtime, "failure_reason", None)
            return self._failure or runtime_reason

    @property
    def restart_required(self):
        return self._restart_reason is not None

    @staticmethod
    def _is_recoverable_capture_interruption(event):
        # Only the native capture boundary is sufficiently specific to distinguish
        # a physical source loss from materialization, enqueue, or persistence I/O.
        return (
            event.kind in {"CAPTURE_DEGRADED", "SHOT_INCOMPLETE"}
            and isinstance(event.reason, str)
            and event.reason.startswith("NativeCaptureProtocolError:")
        )

    @staticmethod
    def _recoverable_incomplete_stage(event):
        if event.kind != "SHOT_INCOMPLETE" or not isinstance(event.reason, str):
            return None
        try:
            payload = json.loads(event.reason)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        stage = payload.get("stage")
        return (
            stage
            if stage
            in {
                "freeze",
                "live_settle",
                "materialization",
                "enqueue",
                "post_roll_superseded",
            }
            else None
        )

    def _settle_interrupted_shot(self, event, *, reason="capture_interrupted"):
        shots = self.store.list_shots(self.session_id)
        if not shots:
            if event.shot_id is not None:
                raise ValueError("incomplete shot has no persisted start")
            return False
        original = shots[-1]
        if event.shot_id is not None and event.shot_id != original.shot_id:
            raise ValueError("incomplete shot identity does not match latest shot")
        if original.settled_ms is not None:
            return False
        if self._active_shot_id != original.shot_id or self._active_start_frame_id is None:
            raise ValueError("capture interruption shot lineage is ambiguous")
        settled_ms = max(original.started_ms + 1, self._position + 1)
        settled_frame_id = max(
            self._active_start_frame_id,
            self._active_start_frame_id if event.frame_id is None else event.frame_id,
        )
        shot = LiveSettledShot(
            shot_id=original.shot_id,
            start_frame_id=self._active_start_frame_id,
            settled_frame_id=settled_frame_id,
            settled_at_ns=settled_ms * 1_000_000,
            observations=(),
            publishable=False,
            shot_abstention_reason=reason,
            fact_coverage={},
            source_epoch_id=self.source_epoch_id,
            job_id=f"abstained:{reason}:{self.source_epoch_id}:{original.shot_id}",
        )
        match = self.coordinator.snapshot.match
        if match.rule_profile_id is None:
            self.coordinator.settle(shot, facts=())
        else:
            if match.active_role is None:
                raise ValueError("capture recovery requires an active player")
            build = build_abstained_realtime_rule_facts_with_preimage(
                shot,
                session_id=self.session_id,
                actor_role=match.active_role,
                match_revision=match.revision,
                profile_id=match.rule_profile_id,
                reason=reason,
                required_cue_ball_placement=match.cue_ball_placement,
            )
            self.coordinator.settle_profiled(
                shot,
                build.facts,
                elapsed_ms=0.0,
                fact_preimage_json=build.normalization_preimage_json,
            )
        self._position = settled_ms
        self._active_shot_id = None
        self._active_start_frame_id = None
        return True

    def _request_restart(self, event):
        self._settle_interrupted_shot(event)
        self.runtime.close()
        self._validated_ready = False
        self._restart_reason = event.reason or event.kind
        self._phase = "degraded"
        self._observations = ()
        self._generation += 1
        self._stopped.set()

    def _discard_incomplete_and_restart(self, event, *, stage):
        if (
            self._active_shot_id is None
            or self._active_start_frame_id is None
            or event.shot_id != self._active_shot_id
        ):
            raise ValueError("incomplete shot lineage is ambiguous")
        shot_number = self._shot_count
        self.store.discard_latest_unsettled_shot(
            self.session_id,
            self._active_shot_id,
            reason=f"incomplete shot discarded after {stage}",
            payload={
                "source_epoch_id": self.source_epoch_id,
                "frame_id": event.frame_id,
                "failure": event.reason,
            },
        )
        self._active_shot_id = None
        self._active_start_frame_id = None
        self._phase = "stable"
        self._observations = ()
        self._last_capture_action = f"第{shot_number}杆不完整，录像已丢弃，采集正在重启"
        self.runtime.close()
        self._validated_ready = False
        self._restart_reason = f"incomplete_{stage}"
        self._generation += 1
        self._stopped.set()

    def _discard_incomplete_and_continue(self, event, *, stage):
        if (
            self._active_shot_id is None
            or self._active_start_frame_id is None
            or event.shot_id != self._active_shot_id
        ):
            raise ValueError("incomplete shot lineage is ambiguous")
        shot_number = self._shot_count
        self.store.discard_latest_unsettled_shot(
            self.session_id,
            self._active_shot_id,
            reason=f"incomplete shot discarded after {stage}",
            payload={
                "source_epoch_id": self.source_epoch_id,
                "frame_id": event.frame_id,
                "failure": event.reason,
            },
        )
        self._active_shot_id = None
        self._active_start_frame_id = None
        self._phase = "stable"
        self._observations = ()
        self._last_capture_action = f"第{shot_number}杆不完整，录像已丢弃，继续采集下一杆"

    def _persist_recording_event(self, event):
        """Persist arrived media evidence before the lossy UI delivery boundary."""
        if event.kind == "SHOT_STARTED":
            self.store.record_recording_start(
                self.session_id,
                event.shot_id,
                self.source_epoch_id,
                event.frame_id,
                event.started_ms,
            )
        elif event.kind == "SHOT_RECORDING_BOUNDARY":
            boundary = event.recording_boundary
            if boundary is None or (boundary.source_epoch_id, boundary.shot_id) != (
                self.source_epoch_id,
                event.shot_id,
            ):
                raise ValueError("recording boundary source mismatch")
            self.store.record_recording_boundary(self.session_id, boundary)
        elif event.kind == "SHOT_MEDIA_RECORDED":
            media = event.recorded_media
            if media is None or (media.boundary.source_epoch_id, media.boundary.shot_id) != (
                self.source_epoch_id,
                event.shot_id,
            ):
                raise ValueError("recording media source mismatch")
            try:
                self.store.record_recording_media(self.session_id, media)
            except ValueError:
                if media.reference_path is None:
                    raise
                self.store.record_recording_media(
                    self.session_id,
                    RecordedMediaV1(
                        media.boundary,
                        "FAILED",
                        failure_code="RECORDING_REFERENCE_INVALID",
                    ),
                )

    def _apply_event(self, event, *, draining=False):
        if self._failure is not None or self._restart_reason is not None:
            # These already-arrived boundaries establish the immutable receipt's
            # original ownership. Draining them never restores live readiness.
            retained_boundary = draining and event.kind in {"SHOT_STARTED", "SHOT_SETTLED"}
            if (
                event.kind
                not in {
                    "SHOT_ENQUEUED",
                    "SHOT_RECORDING_BOUNDARY",
                    "SHOT_MEDIA_RECORDED",
                }
                and not retained_boundary
            ):
                return
        if event.kind == "SHOT_RECORDING_BOUNDARY":
            boundary = event.recording_boundary
            if (
                self.session_policy != "recording-v1"
                or boundary is None
                or (
                    boundary.source_epoch_id != self.source_epoch_id
                    or boundary.shot_id != event.shot_id
                )
            ):
                raise ValueError("recording boundary source mismatch")
            self.store.record_recording_boundary(self.session_id, boundary)
            if self._active_shot_id == boundary.shot_id:
                self._active_shot_id = self._active_start_frame_id = None
                self._phase = "stable"
                self._position = boundary.ended_ms
            self._last_capture_action = "击球边界已记录，媒体等待持久化"
        elif event.kind == "SHOT_MEDIA_RECORDED":
            media = event.recorded_media
            if (
                self.session_policy != "recording-v1"
                or media is None
                or (
                    media.boundary.source_epoch_id != self.source_epoch_id
                    or media.boundary.shot_id != event.shot_id
                )
            ):
                raise ValueError("recording media source mismatch")
            self._persist_recording_event(event)
            actual = self.store.get_recording_evidence(self.session_id, event.shot_id)
            self._last_capture_action = (
                "录像区间已持久化，尚未调度编码"
                if actual["media_status"] == "RECORDED"
                else "录像区间不完整或写入失败"
            )
        elif event.kind in {"CAPTURE_DEGRADED", "SHOT_INCOMPLETE"}:
            if self._is_recoverable_capture_interruption(event):
                self._request_restart(event)
            elif (stage := self._recoverable_incomplete_stage(event)) is not None:
                if stage == "post_roll_superseded":
                    self._discard_incomplete_and_continue(event, stage=stage)
                elif self._active_shot_id is not None:
                    self._discard_incomplete_and_restart(event, stage=stage)
                else:
                    self._settle_interrupted_shot(event, reason=f"incomplete_{stage}")
                    self._phase = "stable"
                    self._observations = ()
                    self._last_capture_action = f"第{self._shot_count}杆录像未保存，识别结果已保留"
            else:
                self._degrade(event.reason or event.kind)
        elif event.kind == "SHOT_STARTED":
            if (
                not event.shot_id
                or type(event.frame_id) is not int
                or event.frame_id < 0
                or type(event.started_ms) is not int
                or event.started_ms < 0
            ):
                raise ValueError("authoritative shot start boundary is missing")
            if self.session_policy == "recording-v1":
                self.store.record_recording_start(
                    self.session_id,
                    event.shot_id,
                    self.source_epoch_id,
                    event.frame_id,
                    event.started_ms,
                )
            else:
                self.coordinator.shot_started(shot_id=event.shot_id, started_ms=event.started_ms)
            self._shot_count = _live_shot_number(event.shot_id)
            self._active_shot_id = event.shot_id
            self._active_start_frame_id = event.frame_id
            self._position = event.started_ms
            self._observations = ()
            self._phase = "moving"
            self._last_capture_action = f"已检测第{self._shot_count}杆击球开始"
        elif event.kind == "SHOT_SETTLED":
            if event.settled is None or event.settled.shot_id != event.shot_id:
                raise ValueError("settled event identity mismatch")
            match = self.coordinator.snapshot.match
            if match.rule_profile_id is None:
                self.coordinator.settle(event.settled, facts=())
            else:
                if match.active_role is None:
                    raise ValueError("profiled settlement has no active player")
                started_ns = time.monotonic_ns()
                try:
                    build = build_realtime_rule_facts_with_preimage(
                        event.settled,
                        session_id=self.session_id,
                        actor_role=match.active_role,
                        match_revision=match.revision,
                        profile_id=match.rule_profile_id,
                        required_cue_ball_placement=match.cue_ball_placement,
                        deadline_ns=started_ns + 2_000_000_000,
                        monotonic_ns=time.monotonic_ns,
                    )
                except RuleFactDeadlineExceeded:
                    build = build_abstained_realtime_rule_facts_with_preimage(
                        event.settled,
                        session_id=self.session_id,
                        actor_role=match.active_role,
                        match_revision=match.revision,
                        profile_id=match.rule_profile_id,
                        reason="rule_fact_deadline_exceeded",
                        required_cue_ball_placement=match.cue_ball_placement,
                    )
                projection = self.coordinator.settle_profiled(
                    event.settled,
                    build.facts,
                    elapsed_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                    fact_preimage_json=build.normalization_preimage_json,
                )
                if projection.rule_status == "SESSION_ENDED":
                    self._ended = True
                    self.runtime.close()
                    self._stopped.set()
            self._position = event.settled.settled_at_ns // 1_000_000
            self._observations = tuple(
                frame
                for frame in event.settled.observations
                if frame.frame_id == event.settled.settled_frame_id
            )
            self._phase = "stable"
            self._active_shot_id = None
            self._active_start_frame_id = None
            self._last_capture_action = f"第{self._shot_count}杆已停止，证据已保留"
        elif event.kind == "SHOT_ENQUEUED":
            job = event.job
            if job is None or (job.session_id, job.source_epoch_id, job.shot_id) != (
                self.session_id,
                self.source_epoch_id,
                event.shot_id,
            ):
                raise ValueError("enqueued job identity mismatch")
            ownership = self.store.get_shot_ownership(job.session_id, job.shot_id)
            if ownership is None:
                raise ValueError("enqueued job has no authoritative start")
            self.store.attach_live_shot_media(
                LiveShotMediaReference(
                    job.session_id,
                    job.shot_id,
                    job.source_epoch_id,
                    ownership.match_revision,
                    job.job_id,
                    str(job.media_root.resolve()),
                    job.media_sha256,
                )
            )
            match = self.coordinator.snapshot.match
            if match.rule_profile_id is not None and match.rack_phase.value == "ended":
                projection = self.coordinator.finalize_profiled_terminal(
                    shot_id=job.shot_id,
                    job_id=job.job_id,
                )
                if projection.rule_status != "SESSION_ENDED":
                    raise ValueError("terminal rule decision did not end the session")
                self._ended = True
                self.runtime.close()
                self._stopped.set()
            else:
                self._last_capture_action = f"第{self._shot_count}杆录像证据已保存"
        self._generation += 1

    def current(self):
        with self.lock:
            self._collect_runtime_failure()
            if not self._validated_ready:
                self.poll_capture_ready()
            if self._failure is None:
                try:
                    self._last_snapshot = self.coordinator.snapshot
                except Exception as exc:
                    self._degrade(f"live projection failure: {type(exc).__name__}: {exc}")
            # A terminal failure must not depend on a now-broken database to
            # project DEGRADED. Cached metadata can never authorize guidance.
            snapshot = self._last_snapshot
            validation_pending = (
                not self._validated_ready
                and self._failure is None
                and self._restart_reason is None
                and not self._stopped.is_set()
            )
            unavailable = (
                self._failure
                or self._restart_reason
                or (
                    "capture stopped before session end committed"
                    if self._stopped.is_set() and not self._ended
                    else None
                )
                or ("capture validation pending" if validation_pending else None)
            )
            current_observation = getattr(self.runtime, "current_observation", None)
            if (
                unavailable is None
                and not self._ended
                and self._phase == "stable"
                and current_observation is not None
            ):
                try:
                    snapshot = self.coordinator.observe_stable(
                        current_observation,
                        source_epoch_id=self.source_epoch_id,
                    )
                    self._last_snapshot = snapshot
                except ValueError as exc:
                    # A bad route input revokes only route eligibility. Camera,
                    # shot boundaries, recording, and later stable retries continue.
                    snapshot = self.coordinator.reject_stable_observation(str(exc))
                    self._last_snapshot = snapshot
            guidance = None if unavailable or self._ended else snapshot.guidance
            # Camera availability is independent from match/rules/analysis health.
            # The runtime handle exposes the CaptureHub's latest frame even after
            # its analysis subscription has stopped or the match has ended.
            camera_frame = getattr(self.runtime, "current_capture_frame", None)
            calibration = getattr(self.runtime, "calibration", None)
            guidance_result = self.coordinator.guidance_result if guidance is not None else None
            route_overlay = None
            if (
                camera_frame is not None
                and calibration is not None
                and guidance_result is not None
                and guidance_result.scene is not None
            ):
                route_overlay = {
                    "width": camera_frame.width,
                    "height": camera_frame.height,
                    "routes": [
                        {
                            "cue_path": [
                                list(table_to_pixel(calibration, point.x_mm, point.y_mm))
                                for point in route.cue_path
                            ],
                            "object_path": [
                                list(table_to_pixel(calibration, point.x_mm, point.y_mm))
                                for point in route.object_path
                            ],
                            "ghost_ball_center": (
                                None
                                if route.ghost_ball_center is None
                                else list(
                                    table_to_pixel(
                                        calibration,
                                        route.ghost_ball_center.x_mm,
                                        route.ghost_ball_center.y_mm,
                                    )
                                )
                            ),
                        }
                        for route in guidance_result.scene.routes
                    ],
                }
            frame = {
                "frame_id": self._generation,
                "media_frame_id": (None if camera_frame is None else camera_frame.capture_frame_id),
                "timestamp_ms": self._position,
                "phase": (
                    "recovering"
                    if validation_pending
                    else "degraded"
                    if unavailable
                    else self._phase
                ),
                "guidance_visible": guidance is not None,
                "recommendation_id": None if guidance is None else guidance.recommendation_id,
                "shot_sequence": self._shot_count,
                "effects": [],
                "source_kind": "usb_live",
                "source_epoch_id": self.source_epoch_id,
                "session_id": self.session_id,
                "match_mode": None if snapshot.match.mode is None else snapshot.match.mode.value,
                "rule_status": "DEGRADED" if unavailable else snapshot.rule_status,
                "guidance": None if guidance is None else asdict(guidance),
                "confirmation_ref": None
                if unavailable or self._ended or snapshot.confirmation_ref is None
                else asdict(snapshot.confirmation_ref),
                "rule_pending_reason": snapshot.rule_pending_reason,
                "degraded_reason": unavailable,
                "route_abstained_reason": snapshot.route_abstained_reason,
                "route_overlay": route_overlay,
                "capture_action": self._last_capture_action,
            }
            s1 = self._snapshot_guidance_payload(snapshot, unavailable, camera_frame, calibration)
            if s1 is not None:
                frame["snapshot_guidance"] = s1
                frame["guidance_visible"] = s1["recommendation"] is not None
            return ReplayProjection(
                self._position,
                not self._stopped.is_set(),
                ReplayTimelineFrame(self._generation, self._position, frame),
            )

    def _snapshot_guidance_payload(self, snapshot, unavailable, camera_frame, calibration):
        """Optional presentation faults must never revoke the recording authority."""
        guidance = getattr(self.runtime, "snapshot_guidance", None)
        if guidance is None:
            return None
        if self._snapshot_projection_failure is None:
            try:
                match = snapshot.match
                group = (
                    match.red_group
                    if match.active_role is not None and match.active_role.value == "red"
                    else match.blue_group
                )
                eligible = (
                    unavailable is None
                    and not self._ended
                    and match.active_role is not None
                    and (match.mode is None or match.mode.value == "assistance")
                )
                guidance.set_context(
                    ("open" if group is None else group.value) if eligible else None,
                    key=(match.revision, match.active_role),
                )
                s1 = guidance.projection()
                suggestion = s1["recommendation"]
                feedback = s1.get("feedback")
                if feedback is not None:
                    suggestion = feedback["recommendation"]
                if suggestion is not None and calibration is not None and camera_frame is not None:
                    route = suggestion["route"]
                    s1["overlay"] = {
                        "width": camera_frame.width,
                        "height": camera_frame.height,
                        "cue_path": []
                        if feedback is not None
                        else [
                            table_to_pixel(calibration, p["x_mm"], p["y_mm"])
                            for p in route["cue_path"]
                        ],
                        "object_path": []
                        if feedback is not None
                        else [
                            table_to_pixel(calibration, p["x_mm"], p["y_mm"])
                            for p in route["object_path"]
                        ],
                        "region": None,
                    }
                    if suggestion["region"] is not None:
                        from math import cos, pi, sin

                        region = suggestion["region"]
                        center = region["center"]
                        s1["overlay"]["region"] = [
                            table_to_pixel(
                                calibration,
                                center["x_mm"] + region["radius_mm"] * cos(i * pi / 16),
                                center["y_mm"] + region["radius_mm"] * sin(i * pi / 16),
                            )
                            for i in range(32)
                        ]
                return s1
            except Exception as exc:
                self._snapshot_projection_failure = (
                    f"guidance_projection_failure:{type(exc).__name__}"
                )
                # Revoke immediately; runtime retains this owner and retries its
                # cleanup at end/shutdown. Never close/cancel the ordinary reader.
                try:
                    guidance.request_stop()
                except Exception as stop_error:
                    self._snapshot_projection_failure += f"; stop:{type(stop_error).__name__}"
        return {
            "phase": "ended" if self._ended else "failed",
            "recommendation": None,
            "overlay": None,
            "error": self._snapshot_projection_failure,
        }

    def apply_control(self, payload):
        with self.lock:
            if self._stopped.is_set() or self._ended:
                raise ValueError("live authority is unavailable")
            command = payload.get("command")
            if self.coordinator.snapshot.match.rule_profile_id is not None:
                raise ValueError("profiled match turn is controlled by the rules engine")
            if command not in {"continue_turn", "switch_turn"}:
                raise ValueError("unsupported live control")
            command_id = payload.get("command_id")
            if not isinstance(command_id, str) or not command_id.strip():
                raise ValueError("confirmation command_id is required")
            self.coordinator.confirm_turn(
                command_id, command, confirmation_reference(payload.get("confirmation_ref"))
            )
            self._generation += 1

    def end_session(self):
        if self.session_policy == "recording-v1":
            if (
                self.store.get_recording_capture_end(self.session_id) is None
                and self.store.get_recording_end_manifest(self.session_id) is None
            ):
                raise ValueError("RECORDING_CAPTURE_STOP_REQUIRED")
        else:
            self.close()
        self._ended = True
        self._observations = ()
        self._generation += 1

    def prepare_session_end(self):
        """Freeze capture and persist every arrived boundary before an end commit."""
        if self.session_policy == "recording-v1":
            return self._prepare_recording_end()
        try:
            self.close()
            if any(shot.settled_ms is None for shot in self.store.list_shots(self.session_id)):
                self._degrade("session end rejected: unsettled physical shot")
        except Exception as exc:
            self._degrade(f"session end preparation: {type(exc).__name__}: {exc}")

    def _prepare_recording_end(self):
        if not self._close_lock.acquire(blocking=False):
            raise ValueError("RECORDING_CLEANUP_IN_PROGRESS")
        try:
            if self._capture_end_receipt is None:
                self._recording_end_requested = True
                deadline = time.monotonic() + 1

                def persist(cutoff, path, first_frame_id):
                    return self.store.record_recording_capture_end(
                        self.session_id,
                        self.source_epoch_id,
                        None
                        if cutoff is None
                        else dict(
                            frame_id=cutoff.capture_frame_id,
                            captured_at_ns=cutoff.captured_at_ns,
                        ),
                        path,
                        first_frame_id=first_frame_id,
                        width=0 if cutoff is None else cutoff.width,
                        height=0 if cutoff is None else cutoff.height,
                        last_frame_sha256=(
                            None if cutoff is None else hashlib.sha256(cutoff.mjpeg).hexdigest()
                        ),
                        deadline=deadline,
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("RECORDING_CAPTURE_STOP_TIMEOUT")
                self._capture_end_receipt = self.runtime.request_capture_end(
                    persist,
                    timeout_seconds=remaining,
                )
            self._recording_end_prepared = True
            self._last_capture_action = "本场采集已停止，录像与片段正在后台处理"
            if self._recording_finalizer is None:
                thread = threading.Thread(
                    target=self._finish_recording_end,
                    name="recording-end-finalizer",
                    daemon=True,
                )
                thread.start()
                self._recording_finalizer = thread
        finally:
            self._close_lock.release()

    def _finish_recording_end(self):
        try:
            tail_published = False
            while True:
                if self._stopped.wait(0.02):
                    if self._failure is not None:
                        raise ValueError("RECORDING_PROCESSING_FAILED")
                    return
                if not self.runtime.recording_tail_finished:
                    continue
                if self.runtime.recording_tail_failure is not None:
                    raise self.runtime.recording_tail_failure
                if not tail_published:
                    self.store.set_recording_capture_progress(self.session_id, tail_status="SAVED")
                    tail_published = True
                if not self._runtime_exited.is_set():
                    continue
                if self.store.load_session(self.session_id).status.value != "ended":
                    continue
                if not self.lock.acquire(timeout=0.05):
                    continue
                try:
                    self._collect_runtime_failure()
                    while self.channel.pending_events:
                        self._apply_event(self.channel.receive(timeout_seconds=0), draining=True)
                    if self._failure is not None or self._restart_reason is not None:
                        raise ValueError("RECORDING_PROCESSING_FAILED")
                finally:
                    self.lock.release()
                path, segments = self.runtime.finalize_recording(timeout_seconds=6)
                self.store.prepare_recording_end(
                    self.session_id,
                    self.source_epoch_id,
                    self._capture_end_receipt["cutoff"],
                    path,
                    segments,
                    background=True,
                )
                self._stopped.set()
                self._closed = True
                return
        except BaseException as exc:
            failure_code = f"PROCESSING_{type(exc).__name__}"

            def publish_failure():
                self.store.set_recording_capture_progress(
                    self.session_id,
                    tail_status=(
                        "FAILED"
                        if self.runtime.recording_tail_failure
                        else "SAVED"
                        if self.runtime.recording_tail_finished
                        else None
                    ),
                    processing_status="FAILED",
                    failure_code=failure_code,
                )

            try:
                publish_failure()
                if not self.runtime.recording_tail_finished:
                    self.runtime.wait_for_recording_tail(timeout_seconds=6)
                    publish_failure()
            except BaseException:
                pass
            with self.lock:
                self._failure = self._failure or (
                    f"recording background processing: {type(exc).__name__}: {exc}"
                )

    def refresh_confirmed_turn(self, reference):
        with self.lock:
            current = self.coordinator.snapshot
            # A retried cloud acknowledgement can outlive its shot. Preserve the
            # ledger result without reapplying an old route to a later boundary.
            if current.shot_id != reference.shot_id or current.rule_status != "READY":
                return
            self.coordinator.refresh_confirmed_turn(reference)
            self._generation += 1

    def _revoke(self):
        """Revoke live authority without waiting on any owned worker."""
        with self.lock:
            if not self._stopped.is_set():
                self._generation += 1
            self._stopped.set()
            self._validated_ready = False

    def close(self):
        # An end/rollover caller can already own self.lock. Never wait for a
        # competing close that might be waiting for that same match lock.
        if not self._close_lock.acquire(blocking=False):
            raise RuntimeError("live runtime cleanup already in progress")
        try:
            if self._closed:
                return
            self._revoke()
            failures = []
            try:
                self.runtime.close()
            except BaseException as exc:
                failures.append(exc)
                with self.lock:
                    self._failure = f"capture cancellation: {type(exc).__name__}: {exc}"
                    self._phase = "degraded"
            for thread in (self._runner, self._consumer):
                if thread.ident is not None:
                    thread.join(timeout=6)
                    if thread.is_alive():
                        failures.append(RuntimeError("live runtime shutdown timed out"))
            with self.lock:
                self._collect_runtime_failure()
                if not any(thread.is_alive() for thread in (self._runner, self._consumer)):
                    # Cancellation can finish materialization. Drain each immutable
                    # receipt once, into this source's original session/epoch.
                    while self.channel.pending_events:
                        try:
                            self._apply_event(
                                self.channel.receive(timeout_seconds=0),
                                draining=True,
                            )
                        except Exception as exc:
                            self._degrade(f"live event drain failure: {type(exc).__name__}: {exc}")
                if self._cancellation_failure is not None:
                    failures.append(self._cancellation_failure)
                    self._cancellation_failure = None
                if not failures:
                    self._closed = True
            _raise_cleanup_failures("live analysis cleanup failed", failures)
        finally:
            self._close_lock.release()

    def shutdown(self):
        """Analysis close retains a shared hub; only service shutdown closes it."""
        if not self._close_lock.acquire(blocking=False):
            raise RuntimeError("live runtime cleanup already in progress")
        try:
            if self._shutdown_complete:
                return
            failures = []
            try:
                self.close()
            except BaseException as exc:
                failures.append(exc)
            if self._recording_finalizer is not None:
                self._recording_finalizer.join(timeout=6)
                if self._recording_finalizer.is_alive():
                    failures.append(RuntimeError("recording finalizer did not stop"))
            shutdown = getattr(self.runtime, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except BaseException as exc:
                    failures.append(exc)
            _raise_cleanup_failures("live source cleanup failed", failures)
            self._shutdown_complete = True
        finally:
            self._close_lock.release()

    def jpeg(self, frame_id):
        with self.lock:
            if frame_id < 0:
                raise ValueError("live camera frame is invalid")
            camera_frame = getattr(self.runtime, "current_capture_frame", None)
            if camera_frame is None:
                raise ValueError("live camera frame is unavailable")
            return camera_frame.mjpeg

    def mjpeg_snapshot(self):
        """Return the latest camera JPEG without projecting database-backed state."""

        with self.lock:
            camera_frame = getattr(self.runtime, "current_capture_frame", None)
            if camera_frame is None:
                return None
            return camera_frame.capture_frame_id, camera_frame.mjpeg

    def subsystem_health(self):
        """Report independent product health without revoking a healthy camera."""
        with self.lock:
            camera_frame = getattr(self.runtime, "current_capture_frame", None)
            video = _component_health(
                getattr(self.runtime, "video_health", None),
                fallback_status="ready" if camera_frame is not None else "unavailable",
            )
            recording_reason = getattr(self.runtime, "recording_failure_reason", None)
            recording = _component_health(
                getattr(self.runtime, "recording_health", None),
                fallback_status=("degraded" if recording_reason is not None else "unavailable"),
                fallback_reason=recording_reason,
            )
            raw_audit_health = getattr(self.runtime, "segment_audit_health", None)
            segment_audit = _component_health(
                raw_audit_health,
                fallback_status="unavailable",
            )
            if raw_audit_health is not None:
                segment_audit.update(
                    {
                        "audited_segments": getattr(raw_audit_health, "audited_segments", 0),
                        "recovered_boundaries": getattr(
                            raw_audit_health, "recovered_boundaries", 0
                        ),
                    }
                )
            audit_reason = (
                segment_audit["reason"] if segment_audit["status"] == "degraded" else None
            )
            downstream_reason = self._failure or self._restart_reason or audit_reason
            analysis = {
                "status": "degraded" if downstream_reason else "ready",
                "reason": downstream_reason,
                "segment_audit": segment_audit,
            }
            current_observation = getattr(self.runtime, "current_observation", None)
            tracker = _component_health(
                getattr(self.runtime, "tracker_health", None),
                fallback_status="unavailable",
            )
            anchor_worker = _anchor_worker_health(getattr(self.runtime, "anchor_health", None))
            anchor_replay = _anchor_replay_health(
                getattr(self.runtime, "anchor_replay_health", None)
            )
            route_input_missing = current_observation is None
            route_reason = (
                downstream_reason
                or tracker["reason"]
                or anchor_worker["reason"]
                or anchor_replay["reason"]
                or ("trusted stable observation unavailable" if route_input_missing else None)
                or self.coordinator.snapshot.route_abstained_reason
            )
            route_status = (
                "degraded"
                if downstream_reason or tracker["status"] == "degraded"
                else "unavailable"
                if route_input_missing
                else "abstained"
                if self.coordinator.snapshot.route_abstained_reason
                else "ready"
            )
            route = {
                "status": route_status,
                "reason": route_reason,
                "input_eligible": current_observation is not None and downstream_reason is None,
                "tracker": tracker,
            }
            diagnostics = getattr(self.runtime, "diagnostics", None)
            analysis["tracker"] = tracker
            analysis["anchor_worker"] = anchor_worker
            analysis["anchor_replay"] = anchor_replay
            analysis["realtime_diagnostics"] = (
                None
                if diagnostics is None
                else asdict(diagnostics)
                if hasattr(diagnostics, "__dataclass_fields__")
                else None
            )
            health = {
                "video_health": video,
                "recording_health": recording,
                "analysis_health": analysis,
                "route_health": route,
            }
            capture_diagnostics = getattr(self.runtime, "capture_diagnostics", None)
            if capture_diagnostics is not None:
                health["capture_diagnostics"] = capture_diagnostics
            return health


def _anchor_worker_health(health: object | None) -> dict[str, object]:
    fields = (
        "scheduled_frames",
        "accepted_frames",
        "skipped_scheduled_frames",
        "completed_anchors",
        "failed_anchors",
        "identity_classified_anchors",
        "p50_inference_ms",
        "p95_inference_ms",
        "maximum_inference_ms",
        "high_speed_coverage",
    )
    if health is None:
        return {"status": "unavailable", "reason": "anchor worker unavailable"}
    reason = getattr(health, "last_error", None)
    return {
        "status": "degraded" if reason else "ready",
        "reason": reason,
        **{field: getattr(health, field, None) for field in fields},
    }


def _anchor_replay_health(health: object | None) -> dict[str, object]:
    if health is None:
        return {"status": "unavailable", "reason": "anchor replay unavailable"}
    accepted = int(getattr(health, "accepted_anchors", 0))
    reason = None if accepted else "no detector anchor accepted"
    return {
        "status": "ready" if accepted else "unavailable",
        "reason": reason,
        "accepted_anchors": accepted,
        "unreplayable_anchors": int(getattr(health, "unreplayable_anchors", 0)),
        "skipped_anchors": int(getattr(health, "skipped_anchors", 0)),
    }


def _component_health(
    health: object | None,
    *,
    fallback_status: str,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    if health is None:
        return {"status": fallback_status, "reason": fallback_reason}
    state = getattr(health, "state", getattr(health, "status", None))
    status = getattr(state, "value", state)
    if status == "running":
        status = "ready"
    if not isinstance(status, str) or not status:
        status = fallback_status
    reason = getattr(health, "reason", None) or fallback_reason
    if reason and status == "ready":
        status = "degraded"
    return {"status": status, "reason": reason}


def _raise_cleanup_failures(message, failures):
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(message, failures)
