"""Assemble the shared live capture, recording, and realtime analysis path."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from billiards_ai.edge.route_assistance.records import require_session_policy
from billiards_ai.lab.field_assistance.visual_motion import VisualMotionConfig
from billiards_ai.lab.mainline1_capture_hub import CaptureHub
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEventChannel,
    ProductRealtimeRuntime,
    ProductRealtimeRuntimeConfig,
)
from billiards_ai.lab.mainline1_realtime_capture import (
    CaptureBackend,
    CaptureContractError,
    RealtimeCaptureConfig,
)
from billiards_ai.lab.mainline1_realtime_ring import AsyncMjpegSegmentRing, MjpegSegmentRing
from billiards_ai.lab.mainline1_recording_processing import (
    RecordingProcessingJournal,
    recording_processing_config_sha256,
)
from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore, _require_sha256
from billiards_ai.lab.mainline1_shot_local_video import Mainline1ShotLocalConfig
from billiards_ai.lab.shot_local.observations import GroupIdentityClassifier
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.shared.calibration.homography import TableCalibration

_ANALYSIS_FIFO_MAXIMUM_PENDING_FRAMES = 180
_ANALYSIS_FIFO_MAXIMUM_PENDING_BYTES = 64 * 1024 * 1024


class RuntimeHandle:
    def __init__(
        self,
        source,
        runtime_factory,
        *,
        capture_starter=None,
        startup_validator=None,
        startup_progress=None,
        capture_config=None,
        capture_hub=None,
        start_capture=False,
        segment_audit_worker=None,
        capture_diagnostics=None,
        recording_store=None,
        recording_source=None,
        snapshot_guidance=None,
    ):
        if capture_config is not None and capture_hub is not None:
            raise ValueError("runtime handle accepts one capture hub source")
        self.source = source
        self.runtime_factory = runtime_factory
        self.capture_starter = capture_starter
        self.startup_validator = startup_validator
        self._startup_progress = startup_progress or (lambda stage: None)
        self.capture_hub = capture_hub
        self._start_capture = start_capture or capture_config is not None
        self.segment_audit_worker = segment_audit_worker
        self._capture_diagnostics = (
            None
            if capture_diagnostics is None
            else json.dumps(capture_diagnostics, allow_nan=False)
        )
        if capture_config is not None:
            self.capture_hub = CaptureHub(source=source, capture_config=capture_config)
        self._analysis_source = recording_source or (
            None
            if self.capture_hub is None
            else self.capture_hub.subscribe(
                f"analysis-{uuid4().hex}",
                maximum_pending_frames=_ANALYSIS_FIFO_MAXIMUM_PENDING_FRAMES,
                maximum_pending_bytes=_ANALYSIS_FIFO_MAXIMUM_PENDING_BYTES,
                delivery_mode="fifo",
            )
        )
        self._cancelled = threading.Event()
        self._start_lock = threading.Lock()
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._runtime = None
        self._failure_reason = None
        self._recording_store = recording_store
        self._recording_source = recording_source
        self._recording_cutoff = None
        self._recording_sealed = False
        self._recording_tail_committed = False
        self._recording_tail_done = threading.Event()
        self._recording_tail_thread = None
        self._recording_tail_failure = None
        self._recording_capture_metadata = None
        self.snapshot_guidance = snapshot_guidance
        self._snapshot_guidance_failure = None

    def run(self):
        startup_failure = None
        try:
            with self._start_lock:
                if self._cancelled.is_set():
                    return
                if self.capture_hub is None or self._start_capture:
                    if self.capture_starter is not None:
                        self._startup_progress("live_capture_starting")
                        self.capture_starter(self.source)
                        self._startup_progress("live_capture_started")
                    if self.startup_validator is not None:
                        self._startup_progress("live_capture_validation_started")
                        self.startup_validator(self.source)
                        self._startup_progress("live_capture_validation_ready")
                if self.capture_hub is not None:
                    deadline = time.monotonic() + self.capture_hub.startup_timeout_seconds
                    self._startup_progress("capture_hub_starting")
                    self.capture_hub.start()
                    ready = (
                        self._wait_for_startup_frame(deadline)
                        if self.capture_hub.startup_preparation_seconds is None
                        else self._wait_for_prepared_startup_frame()
                    )
                    if not ready:
                        return
                    self._startup_progress("capture_hub_ready")
                    if self._cancelled.is_set():
                        return
                    runtime = self.runtime_factory(self._analysis_source)
                else:
                    if self._cancelled.is_set():
                        return
                    runtime = self.runtime_factory()
                self._runtime = runtime
                if self.snapshot_guidance is not None:
                    # Optional guidance owns no capture or recording resources.
                    try:
                        self.snapshot_guidance.start()
                    except Exception:
                        self._request_snapshot_guidance_stop()
                        self._close_snapshot_guidance()
                if self._recording_sealed and self._recording_source is not None:
                    # A user can stop capture while detector construction lags.
                    runtime.request_recording_end()
                if (
                    self.capture_hub is not None
                    and self.capture_hub.failure_reason is not None
                    and not (self._recording_sealed and self._recording_source is not None)
                ):
                    startup_failure = self._startup_frame_failure()
                if (
                    self._analysis_source is not None
                    and not self._cancelled.is_set()
                    and startup_failure is None
                ):
                    try:
                        activate = getattr(self._analysis_source, "activate", None)
                        if activate is not None:
                            activate()
                    except BaseException as exc:
                        startup_failure = exc
                if self._cancelled.is_set() or startup_failure is not None:
                    # ProductRealtimeRuntime.run owns worker/ring cleanup. Revoke
                    # its input before entering that existing cleanup path.
                    self._cancel_analysis_source()
                else:
                    self._ready.set()
            try:
                runtime.run()
            except BaseException as exc:
                if startup_failure is None:
                    raise
                startup_failure.add_note(f"runtime cleanup failed: {type(exc).__name__}: {exc}")
                raise startup_failure from startup_failure.__cause__
            if startup_failure is not None:
                raise startup_failure
        except BaseException as exc:
            self._failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._close_snapshot_guidance()
            self._finished.set()
            self._ready.clear()
            self._cancel_analysis_source()

    def _wait_for_prepared_startup_frame(self):
        """An explicit file-only allowance; neither progress nor polling resets it."""
        hub = self.capture_hub
        seconds = hub.startup_preparation_seconds
        started = hub.startup_started_at_ns
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
            or type(started) is not int
        ):
            raise self._startup_frame_failure()
        preparation_deadline = started / 1_000_000_000 + seconds
        while not self._cancelled.is_set():
            if hub.failure_reason is not None:
                raise self._startup_frame_failure()
            completed = hub.preparation_completed_at_ns
            now = time.monotonic()
            if completed is not None:
                if (
                    type(completed) is not int
                    or completed < started
                    or completed / 1_000_000_000 > min(now, preparation_deadline)
                ):
                    raise self._startup_frame_failure()
                break
            remaining = preparation_deadline - now
            if remaining <= 0 or hub.first_frame_at_ns is not None:
                # Completion may have been published after the empty read.
                # Revalidate its original receipt; never renew either deadline.
                if hub.preparation_completed_at_ns is not None:
                    continue
                raise self._startup_frame_failure()
            hub.wait_for_frame(after_frame_id=None, timeout_seconds=min(0.05, remaining))
        else:
            return False
        deadline = completed / 1_000_000_000 + hub.startup_timeout_seconds
        while not self._cancelled.is_set():
            if hub.failure_reason is not None or hub.preparation_completed_at_ns != completed:
                raise self._startup_frame_failure()
            first = hub.first_frame_at_ns
            if first is not None:
                if first < completed or first / 1_000_000_000 > deadline:
                    raise self._startup_frame_failure()
                # A later session may reuse this same already-started hub. Its
                # original frame receipt proves startup; never restart its clock.
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # A timely first frame can race the earlier empty read too.
                if hub.first_frame_at_ns is not None:
                    continue
                raise self._startup_frame_failure()
            hub.wait_for_frame(after_frame_id=None, timeout_seconds=min(0.05, remaining))
        return False

    def _wait_for_startup_frame(self, deadline):
        hub = self.capture_hub
        while not self._cancelled.is_set():
            failure_reason = hub.failure_reason
            remaining = deadline - time.monotonic()
            if failure_reason is not None or remaining <= 0:
                raise self._startup_frame_failure()
            frame = hub.wait_for_frame(
                after_frame_id=None,
                timeout_seconds=min(0.05, remaining),
            )
            if self._cancelled.is_set():
                return False
            if frame is not None and hub.failure_reason is None and time.monotonic() <= deadline:
                return True
        return False

    def _startup_frame_failure(self):
        hub = self.capture_hub
        failure_reason = hub.failure_reason
        prefix = "capture hub did not publish a startup frame"
        if failure_reason is not None:
            prefix += f": source failed: {failure_reason}"
        failure = RuntimeError(
            f"{prefix}; startup_diagnostics={json.dumps(hub.startup_diagnostics)}"
        )
        failure.__cause__ = hub._failure
        return failure

    def _cancel_analysis_source(self):
        if self._analysis_source is not None:
            self._analysis_source.cancel()
        else:
            self.source.cancel()

    def wait_until_ready(self, *, timeout_seconds):
        if timeout_seconds <= 0:
            raise ValueError("capture readiness timeout must be positive")
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._cancelled.is_set() or self._finished.is_set():
                return False
            if self.capture_hub is not None and self.capture_hub.failure_reason is not None:
                return False
            if self._ready.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._ready.wait(min(0.05, remaining))

    def close(self):
        self._cancelled.set()
        self._ready.clear()
        with self._start_lock:
            self._cancel_analysis_source()
        self._close_snapshot_guidance()

    def _request_snapshot_guidance_stop(self):
        if self.snapshot_guidance is not None:
            try:
                self.snapshot_guidance.request_stop()
            except Exception as exc:
                self._snapshot_guidance_failure = f"{type(exc).__name__}: {exc}"

    def _close_snapshot_guidance(self):
        if self.snapshot_guidance is not None:
            try:
                self.snapshot_guidance.close()
                self._snapshot_guidance_failure = None
            except Exception as exc:
                # Optional planning cleanup cannot abort recording finalization.
                # Retain its owner so service shutdown can retry cleanup.
                self._snapshot_guidance_failure = f"{type(exc).__name__}: {exc}"

    @property
    def snapshot_guidance_cleanup_complete(self):
        return self.snapshot_guidance is None or self.snapshot_guidance.cleanup_complete

    def seal_recording(self, *, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        if not self._recording_sealed:
            if self._recording_store is None or not self._ready.is_set():
                raise CaptureContractError("recording runtime is not ready for end")
            if self._recording_source is not None:
                cutoff = self.capture_hub.seal_recording_input(
                    self._recording_store,
                    timeout_seconds=timeout_seconds,
                    on_sealed=self._mark_recording_input_sealed,
                )
            else:
                cutoff = self.capture_hub.seal_recording(
                    self._analysis_source,
                    self._recording_store,
                    timeout_seconds=timeout_seconds,
                    on_sealed=self._runtime.request_recording_end,
                )
            # Admission is already detached. Keep this immutable result even if
            # tail persistence times out or the detector finishes before retry.
            self._recording_cutoff = cutoff
            self._recording_sealed = True
        if self._recording_source is not None and not self._recording_tail_committed:
            self._recording_source.seal(self._recording_cutoff)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CaptureContractError("recording tail commit timed out")
            self._recording_store.commit_recording_tail(timeout_seconds=remaining)
            self._recording_tail_committed = True
        return self._recording_cutoff

    def _mark_recording_input_sealed(self):
        # Called inside the Hub's writer-admission fence, before any later
        # preview failure can be published. The original recorder owns this
        # frame; never use the shared preview's latest frame as its cutoff.
        cutoff = self._recording_store.last_admitted_frame
        self._recording_source.seal(cutoff)
        self._recording_cutoff = cutoff
        self._recording_sealed = True
        if self._runtime is not None:
            self._runtime.request_recording_end()

    def request_capture_end(self, persist_cutoff, *, timeout_seconds):
        """Fence this writer, persist its immutable stop, and save the tail separately."""
        if self._recording_store is None or self._recording_source is None:
            raise CaptureContractError("background end requires disk recording")
        if not self._recording_sealed:
            cutoff = self.capture_hub.seal_recording_input(
                self._recording_store,
                timeout_seconds=timeout_seconds,
                on_sealed=self._mark_recording_input_sealed,
            )
            self._recording_cutoff = cutoff
            self._recording_sealed = True
            if self._runtime is not None:
                self._runtime.request_recording_end()
            self._recording_source.seal(cutoff)
        # Outside the Hub's writer-admission fence: no model/child wait here.
        self._request_snapshot_guidance_stop()
        # Snapshot admission metadata before the tail closer takes its append
        # lock. The end acknowledgement must never queue behind tail saving.
        if self._recording_capture_metadata is None:
            self._recording_capture_metadata = (
                self._recording_store.recording_manifest_path,
                self._recording_store.first_recorded_frame_id,
            )
        manifest_path, first_frame_id = self._recording_capture_metadata
        if self._recording_tail_thread is None:
            thread = threading.Thread(
                target=self._save_recording_tail,
                name="recording-tail-save",
                daemon=True,
            )
            thread.start()
            self._recording_tail_thread = thread
        cutoff = self._recording_cutoff
        return persist_cutoff(cutoff, manifest_path, first_frame_id)

    def _save_recording_tail(self):
        try:
            self._recording_store.close_recording_input(timeout_seconds=5)
            self._recording_tail_committed = True
        except BaseException as exc:
            self._recording_tail_failure = exc
        finally:
            self._recording_tail_done.set()

    @property
    def recording_tail_finished(self):
        return self._recording_tail_done.is_set()

    @property
    def recording_tail_failure(self):
        return self._recording_tail_failure

    def wait_for_recording_tail(self, *, timeout_seconds):
        return self._recording_tail_done.wait(timeout_seconds)

    def finalize_recording(self, *, timeout_seconds):
        if not self._recording_sealed or not self._finished.is_set() or self._failure_reason:
            raise CaptureContractError("recording drain is incomplete")
        latest = self._runtime.current_capture_frame
        if latest != self._recording_cutoff:
            raise CaptureContractError("recording drain did not reach its cutoff")
        self._recording_store.close(timeout_seconds=timeout_seconds)
        return (
            self._recording_store.finalized_manifest_path,
            self._recording_store.finalized_segment_receipts,
        )

    def shutdown(self):
        self.close()
        if self._recording_tail_thread is not None:
            self._recording_tail_thread.join(timeout=6)
            if self._recording_tail_thread.is_alive():
                raise CaptureContractError("recording tail saver did not stop")
        if self.capture_hub is not None:
            self.capture_hub.close()
        if not self.snapshot_guidance_cleanup_complete:
            raise CaptureContractError(
                f"snapshot guidance cleanup incomplete: {self._snapshot_guidance_failure}"
            )

    @property
    def failure_reason(self):
        return self._failure_reason

    @property
    def current_capture_frame(self):
        if self.capture_hub is not None:
            return self.capture_hub.latest_frame()
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "current_capture_frame", None)

    @property
    def calibration(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "calibration", None)

    @property
    def current_tracking(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "current_tracking", None)

    @property
    def current_observation(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "current_observation", None)

    @property
    def tracker_health(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "tracker_health", None)

    @property
    def anchor_health(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "anchor_health", None)

    @property
    def anchor_replay_health(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "anchor_replay_health", None)

    @property
    def diagnostics(self):
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "diagnostics", None)

    @property
    def video_health(self):
        return None if self.capture_hub is None else self.capture_hub.capture_health

    @property
    def capture_diagnostics(self):
        return None if self._capture_diagnostics is None else json.loads(self._capture_diagnostics)

    @property
    def recording_health(self):
        hub_health = None if self.capture_hub is None else self.capture_hub.recording_health
        if hub_health is not None:
            return hub_health
        runtime = self._runtime
        return None if runtime is None else getattr(runtime, "ring_health", None)

    @property
    def recording_writer_health(self):
        return getattr(self._recording_store, "writer_health", None)

    @property
    def recording_writer_pending_count(self):
        pending = getattr(self.capture_hub, "recording_writer_pending_count", None)
        if pending is not None:
            return pending
        pending = getattr(self._recording_store, "pending_command_count", None)
        if pending is not None:
            return pending
        writer = self.recording_writer_health
        if writer is not None:
            return writer.pending_chunks
        return None

    @property
    def recording_read_progress(self):
        source = self._recording_source
        return None if source is None else source.progress

    @property
    def recording_failure_reason(self):
        return None if self.capture_hub is None else self.capture_hub.recording_failure_reason

    @property
    def segment_audit_health(self):
        worker = self.segment_audit_worker
        return None if worker is None else worker.health


class _ExternallyFedEvidenceStore:
    """Let the capture hub own frames while analysis owns shot boundaries."""

    def __init__(self, delegate):
        self.delegate = delegate

    @property
    def health(self):
        return self.delegate.health

    @property
    def maximum_queue_depth(self):
        return self.delegate.maximum_queue_depth

    def append(self, frame):
        del frame

    def begin_shot(self, frame_id):
        return self.delegate.begin_shot(frame_id)

    def settle_shot(self, frame_id):
        return self.delegate.settle_shot(frame_id)

    def freeze_shot(self):
        return self.delegate.freeze_shot()

    def freeze_window(self, **kwargs):
        return self.delegate.freeze_window(**kwargs)

    def release_shot(self):
        return self.delegate.release_shot()

    def close(self):
        # The capture hub closes or replaces the physical-frame writer.
        return None


@dataclass(frozen=True, slots=True)
class _RecorderHandoff:
    session_id: str
    source_epoch_id: str
    boundary_ledger: ShotBoundaryLedger
    evidence_store: AsyncMjpegSegmentRing
    runtime_config: ProductRealtimeRuntimeConfig


def build_live_runtime_factory(
    *,
    capture_factory: Callable[[str], CaptureBackend],
    capture_config: RealtimeCaptureConfig,
    calibration: TableCalibration,
    detector: VisionBackend,
    identity_classifier: GroupIdentityClassifier,
    shot_config: Mainline1ShotLocalConfig,
    capture_setup_id: str,
    calibration_artifact_id: str,
    config_path: Path,
    ring_root: Path,
    jobs_root: Path,
    config_sha256: str | None = None,
    identity_training_package_id: str | None = None,
    identity_training_package_sha256: str | None = None,
    capture_starter: Callable[[CaptureBackend], None] | None = None,
    startup_validator: Callable[[CaptureBackend], object] | None = None,
    startup_progress: Callable[[str], None] | None = None,
    capture_diagnostics_factory: Callable[[CaptureBackend, str], dict[str, object]] | None = None,
    session_policy: str = "legacy",
    snapshot_guidance_factory=None,
) -> Callable[[str, ProductRealtimeEventChannel, int, str], RuntimeHandle]:
    """Share one backend per epoch; give each session its own recorder and ledger.

    Native sources may inject startup hooks. Analysis owns only a subscription;
    service shutdown retains responsibility for closing the capture hub.
    """
    if (identity_training_package_id is None) != (identity_training_package_sha256 is None):
        raise ValueError("identity training package identity must be complete")
    if identity_training_package_sha256 is not None:
        if not isinstance(identity_training_package_id, str) or not identity_training_package_id:
            raise ValueError("identity training package ID is invalid")
        _require_sha256(identity_training_package_sha256, "identity training package")
    require_session_policy(session_policy)
    jobs = (
        None
        if session_policy == "recording-v1"
        else SQLiteShotAnalysisJobStore(jobs_root / "analysis-jobs.sqlite3")
    )
    shared_lock = threading.Lock()
    shared_source = None
    shared_hub = None
    shared_boundary_ledger = None
    shared_evidence_store = None
    shared_session_id = None
    physical_source_epoch_id = None
    pending_handoff = None

    def create_product_runtime(
        session_id, channel, runtime_config, analysis_source, evidence_store, boundary_ledger
    ):
        return ProductRealtimeRuntime(
            source=analysis_source,
            capture_config=capture_config,
            calibration=calibration,
            detector=None if session_policy == "recording-v1" else detector,
            identity_classifier=(None if session_policy == "recording-v1" else identity_classifier),
            shot_config=shot_config,
            job_store=jobs,
            job_media_root=jobs_root / "media" / session_id,
            quarantine_root=jobs_root / "quarantine" / session_id,
            session_id=session_id,
            capture_setup_id=capture_setup_id,
            calibration_artifact_id=calibration_artifact_id,
            identity_training_package_id=identity_training_package_id,
            identity_training_package_sha256=identity_training_package_sha256,
            config_sha256=(
                config_sha256
                if config_sha256 is not None
                else hashlib.sha256(config_path.read_bytes()).hexdigest()
            ),
            event_channel=channel,
            evidence_store=_ExternallyFedEvidenceStore(evidence_store),
            boundary_ledger=boundary_ledger,
            visual_motion_config=VisualMotionConfig(),
            config=runtime_config,
            session_policy=session_policy,
        )

    def factory(
        session_id: str,
        channel: ProductRealtimeEventChannel,
        initial_shot_number: int,
        source_epoch_id: str,
    ) -> RuntimeHandle:
        nonlocal shared_source
        nonlocal shared_hub
        nonlocal shared_boundary_ledger
        nonlocal shared_evidence_store
        nonlocal shared_session_id
        nonlocal physical_source_epoch_id
        nonlocal pending_handoff
        runtime_config = ProductRealtimeRuntimeConfig(
            ring_root=ring_root / "sessions" / session_id / source_epoch_id / "shot-ring",
            initial_shot_number=initial_shot_number,
        )

        def new_evidence_store():
            if session_policy == "recording-v1":
                # Bind the original processing semantics before capture starts,
                # even if detector construction has not happened when it stops.
                RecordingProcessingJournal(
                    path=runtime_config.ring_root.parent / "shot-boundaries.sqlite3",
                    session_id=session_id,
                    source_epoch_id=source_epoch_id,
                    config_sha256=recording_processing_config_sha256(
                        capture_config=capture_config,
                        calibration=calibration,
                        window=shot_config.window,
                        motion=VisualMotionConfig(),
                        runtime_config=runtime_config,
                        config_sha256=(
                            config_sha256
                            if config_sha256 is not None
                            else hashlib.sha256(config_path.read_bytes()).hexdigest()
                        ),
                    ),
                    initial_shot_number=initial_shot_number,
                )
                return SessionRecordingEvidenceStore(
                    recorder=ContinuousSessionRecorder(
                        root=runtime_config.ring_root.parent / "session-recording",
                        session_id=session_id,
                        expected_fps=round(capture_config.fps),
                        guard_writer=True,
                    ),
                    session_id=session_id,
                    pre_roll_frames=runtime_config.pre_roll_frames,
                    post_roll_frames=runtime_config.post_roll_frames,
                    maximum_shot_frames=runtime_config.maximum_shot_frames,
                )
            return AsyncMjpegSegmentRing(
                MjpegSegmentRing(
                    runtime_config.ring_root,
                    expected_fps=round(capture_config.fps),
                    pre_roll_frames=runtime_config.pre_roll_frames,
                    post_roll_frames=runtime_config.post_roll_frames,
                    maximum_shot_frames=runtime_config.maximum_shot_frames,
                    maximum_bytes=runtime_config.maximum_ring_bytes,
                    idle_onset_lookback_frames=runtime_config.maximum_pending_ring_frames,
                    persist_idle_segments=False,
                ),
                maximum_pending_frames=(
                    runtime_config.pre_roll_frames
                    + runtime_config.maximum_shot_frames
                    + runtime_config.post_roll_frames
                ),
            )

        with shared_lock:
            if pending_handoff is not None and (
                pending_handoff.session_id != session_id
                or pending_handoff.source_epoch_id != source_epoch_id
                or pending_handoff.runtime_config.initial_shot_number != initial_shot_number
            ):
                raise CaptureContractError("another capture recorder handoff is pending")
            physical_open = shared_hub is None or physical_source_epoch_id != source_epoch_id
            if physical_open:
                if shared_hub is not None:
                    shared_hub.close()
                shared_source = capture_factory(source_epoch_id)
                shared_boundary_ledger = ShotBoundaryLedger(
                    ring_root
                    / "sessions"
                    / session_id
                    / source_epoch_id
                    / "shot-boundaries.sqlite3",
                    session_id=session_id,
                    source_epoch_id=source_epoch_id,
                )
                shared_evidence_store = new_evidence_store()
                shared_hub = CaptureHub(
                    source=shared_source,
                    capture_config=capture_config,
                    recorder=shared_evidence_store,
                )
                shared_session_id = session_id
                physical_source_epoch_id = source_epoch_id
            elif pending_handoff is not None or shared_session_id != session_id:
                if pending_handoff is None:
                    ledger = ShotBoundaryLedger(
                        ring_root
                        / "sessions"
                        / session_id
                        / source_epoch_id
                        / "shot-boundaries.sqlite3",
                        session_id=session_id,
                        source_epoch_id=source_epoch_id,
                    )
                    pending_handoff = _RecorderHandoff(
                        session_id,
                        source_epoch_id,
                        ledger,
                        new_evidence_store(),
                        runtime_config,
                    )
                target = pending_handoff
                try:
                    shared_hub.replace_recorder(target.evidence_store)
                except BaseException as handoff_failure:
                    if shared_hub._uses_recorder(target.evidence_store):
                        # Adoption can succeed before retirement times out. Keep
                        # the matching session context even while retry is pending.
                        shared_boundary_ledger = target.boundary_ledger
                        shared_evidence_store = target.evidence_store
                        shared_session_id = target.session_id
                        handoff_failure._pending_recorder_handoff_target = (
                            target.session_id,
                            target.source_epoch_id,
                            target.runtime_config.initial_shot_number,
                        )
                    else:
                        # A rejected allocation still needs an executable cleanup
                        # path through the existing hub's explicit shutdown.
                        try:
                            shared_hub._retire_unused_recorder(target.evidence_store)
                        except BaseException as cleanup_failure:
                            raise BaseExceptionGroup(
                                "capture recorder handoff and cleanup failed",
                                [handoff_failure, cleanup_failure],
                            ) from None
                        pending_handoff = None
                    raise
                shared_boundary_ledger = target.boundary_ledger
                shared_evidence_store = target.evidence_store
                shared_session_id = target.session_id
                runtime_config = target.runtime_config
                pending_handoff = None
            source = shared_source
            hub = shared_hub
            boundary_ledger = shared_boundary_ledger
            evidence_store = shared_evidence_store
        assert source is not None
        assert hub is not None
        assert boundary_ledger is not None
        assert evidence_store is not None

        def create_runtime(analysis_source):
            return create_product_runtime(
                session_id,
                channel,
                runtime_config,
                analysis_source,
                evidence_store,
                boundary_ledger,
            )

        guidance = None
        if snapshot_guidance_factory is not None and session_policy == "recording-v1":
            try:
                guidance = snapshot_guidance_factory(
                    hub=hub,
                    session_id=session_id,
                    source_epoch_id=source_epoch_id,
                    calibration=calibration,
                    journal_path=runtime_config.ring_root.parent / "snapshot-guidance.json",
                )
            except Exception:
                # Failure to construct optional guidance leaves ordinary recording usable.
                guidance = None
        return RuntimeHandle(
            source,
            create_runtime,
            capture_hub=hub,
            recording_store=evidence_store if session_policy == "recording-v1" else None,
            snapshot_guidance=guidance,
            recording_source=(
                evidence_store.committed_source(
                    source_epoch_id=source_epoch_id,
                    capture_failure=lambda: hub.failure_reason,
                )
                if session_policy == "recording-v1"
                else None
            ),
            start_capture=physical_open,
            segment_audit_worker=None,
            capture_starter=capture_starter if physical_open else None,
            startup_validator=startup_validator if physical_open else None,
            startup_progress=startup_progress,
            capture_diagnostics=(
                None
                if capture_diagnostics_factory is None
                else capture_diagnostics_factory(source, source_epoch_id)
            ),
        )

    def recover_recording(capture, channel, *, on_tail_saved):
        if session_policy != "recording-v1":
            raise CaptureContractError("disk recovery requires ordinary recording")
        session_id, epoch = capture["session_id"], capture["source_epoch_id"]
        context_root = ring_root / "sessions" / session_id / epoch
        path = context_root / "session-recording/session-recording.json"
        if path != path.resolve() or str(path) != capture["raw_manifest_path"]:
            raise CaptureContractError("recording recovery root conflicts with original capture")
        raw = ContinuousSessionRecorder(
            root=path.parent,
            session_id=session_id,
            expected_fps=round(capture_config.fps),
            read_only=True,
        )
        snapshot = raw.close()
        if (
            capture["contract"] == "recording-capture-end/v2"
            and hashlib.sha256(path.read_bytes()).hexdigest() != capture["raw_manifest_sha256"]
        ):
            raise CaptureContractError("committed interruption manifest changed")
        cutoff = capture["cutoff"]
        last = None
        if cutoff is None:
            if snapshot.segments:
                raise CaptureContractError("empty capture contains unexpected original frames")
        else:
            if (
                not snapshot.segments
                or snapshot.source_epoch_id != epoch
                or snapshot.segments[0].first_frame_id != capture["first_frame_id"]
                or snapshot.segments[-1].last_frame_id != cutoff["frame_id"]
                or (snapshot.width, snapshot.height) != (capture["width"], capture["height"])
            ):
                raise CaptureContractError("saved original does not reach capture cutoff")
            last = raw.read_segment(snapshot.segments[-1])[-1]
            if (
                last.captured_at_ns != cutoff["captured_at_ns"]
                or hashlib.sha256(last.mjpeg).hexdigest() != capture["last_frame_sha256"]
            ):
                raise CaptureContractError("saved original cutoff bytes changed")
        runtime_config = ProductRealtimeRuntimeConfig(
            ring_root=context_root / "shot-ring",
            initial_shot_number=0,
        )
        on_tail_saved()
        RecordingProcessingJournal(
            path=context_root / "shot-boundaries.sqlite3",
            session_id=session_id,
            source_epoch_id=epoch,
            initial_shot_number=0,
            require_existing=True,
            config_sha256=recording_processing_config_sha256(
                capture_config=capture_config,
                calibration=calibration,
                window=shot_config.window,
                motion=VisualMotionConfig(),
                runtime_config=runtime_config,
                config_sha256=(
                    config_sha256
                    if config_sha256 is not None
                    else hashlib.sha256(config_path.read_bytes()).hexdigest()
                ),
            ),
        )
        evidence = SessionRecordingEvidenceStore(
            recorder=raw,
            session_id=session_id,
            pre_roll_frames=runtime_config.pre_roll_frames,
            post_roll_frames=runtime_config.post_roll_frames,
            maximum_shot_frames=runtime_config.maximum_shot_frames,
            close_recorder=False,
        )
        source = evidence.committed_source(source_epoch_id=epoch)
        source.seal(last)
        runtime = create_product_runtime(
            session_id,
            channel,
            runtime_config,
            source,
            evidence,
            ShotBoundaryLedger(
                context_root / "shot-boundaries.sqlite3",
                session_id=session_id,
                source_epoch_id=epoch,
            ),
        )
        runtime.request_recording_end(reason=capture["end_reason"])
        return runtime, evidence

    def recover_interrupted_recordings(store, artifact_id):
        from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

        for session_id, epoch in store.recording_interruption_candidates(artifact_id):
            context = ring_root / "sessions" / session_id / epoch
            if context != context.resolve(strict=True):
                raise CaptureContractError("interrupted recording root is unsafe")
            # Only writers participating in this lifetime exclusion protocol are
            # eligible. An older producer without the lock stays fail-closed.
            lock_file = (
                context
                / ".session-media-locks"
                / (hashlib.sha256(b"recording-writer").hexdigest() + ".lock")
            )
            if not lock_file.is_file() or lock_file.is_symlink():
                raise CaptureContractError("interrupted recording writer ownership is unproven")
            with (
                session_media_lock(context, "recording-writer", exclusive=True),
                session_media_lock(context, "recording-detector", exclusive=True),
            ):
                raw_root = context / "session-recording"
                raw_path = raw_root / "session-recording.json"
                if raw_root != raw_root.resolve(strict=True) or raw_path != raw_path.resolve(
                    strict=True
                ):
                    raise CaptureContractError("interrupted recording raw path is unsafe")
                manifest_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                raw = ContinuousSessionRecorder(
                    root=raw_root,
                    session_id=session_id,
                    expected_fps=round(capture_config.fps),
                    read_only=True,
                )
                snapshot = raw.close()
                if not snapshot.segments or snapshot.source_epoch_id != epoch:
                    raise CaptureContractError(
                        "interrupted recording has no verified committed prefix"
                    )
                config = ProductRealtimeRuntimeConfig(
                    ring_root=context / "shot-ring", initial_shot_number=0
                )
                fingerprint = recording_processing_config_sha256(
                    capture_config=capture_config,
                    calibration=calibration,
                    window=shot_config.window,
                    motion=VisualMotionConfig(),
                    runtime_config=config,
                    config_sha256=config_sha256
                    if config_sha256 is not None
                    else hashlib.sha256(config_path.read_bytes()).hexdigest(),
                )
                journal = RecordingProcessingJournal.read_state(
                    context / "shot-boundaries.sqlite3", session_id, epoch
                )
                if journal["config_sha256"] != fingerprint:
                    raise CaptureContractError("interrupted recording processing config changed")
                source = SessionRecordingEvidenceStore(
                    recorder=raw,
                    session_id=session_id,
                    pre_roll_frames=config.pre_roll_frames,
                    post_roll_frames=config.post_roll_frames,
                    maximum_shot_frames=config.maximum_shot_frames,
                    close_recorder=False,
                ).committed_source(source_epoch_id=epoch)
                for key in ("input_prefix", "checkpoint"):
                    if journal[key] is not None:
                        source.validate_input_prefix(journal[key])
                last = None
                for segment in snapshot.segments:
                    last = raw.read_segment(segment)[-1]
                if hashlib.sha256(raw_path.read_bytes()).hexdigest() != manifest_sha:
                    raise CaptureContractError(
                        "interrupted recording manifest changed during validation"
                    )
                store.record_recording_capture_end(
                    session_id,
                    epoch,
                    {"frame_id": last.capture_frame_id, "captured_at_ns": last.captured_at_ns},
                    raw.finalized_manifest_path,
                    first_frame_id=snapshot.segments[0].first_frame_id,
                    width=snapshot.width,
                    height=snapshot.height,
                    last_frame_sha256=hashlib.sha256(last.mjpeg).hexdigest(),
                    interruption_manifest_sha256=manifest_sha,
                )

    factory.recover_interrupted_recordings = recover_interrupted_recordings
    factory.recover_recording = recover_recording
    factory.session_policy = session_policy
    # The field service may open this same queue for the independent, lightweight
    # postgame media product.  This path contains no model/config dependency.
    factory.jobs_db_path = jobs_root / "analysis-jobs.sqlite3"
    return factory
