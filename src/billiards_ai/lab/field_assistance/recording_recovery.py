"""Resume ended ordinary recordings without opening their physical camera again."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeEventChannel


class _DiskEventChannel(ProductRealtimeEventChannel):
    def publish(self, event):
        # Persistence is synchronous in record_evidence. There is no live UI
        # projection for an old scene, so consume its delivery immediately.
        super().publish(event)
        self.receive(timeout_seconds=0)


class RecordingRecovery:
    def __init__(self, *, store, capture, runtime_factory):
        self.store, self.capture = store, capture
        self._factory = runtime_factory
        self._stopped = threading.Event()
        self._runtime = None
        self._thread = threading.Thread(
            target=self._run,
            name="field-recording-recovery",
            daemon=True,
        )
        self.failure = None

    def start(self):
        self._thread.start()

    def close(self):
        self._stopped.set()
        runtime = self._runtime
        if runtime is not None:
            runtime._source.cancel()

    def shutdown(self):
        self.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise TimeoutError("recording recovery did not stop")

    def _run(self):
        session = self.capture["session_id"]
        tail_saved = False
        try:
            root = self.store.artifact_path_for_session(session)
            with session_media_lock(root, session, exclusive=False):
                context = SimpleNamespace(
                    store=self.store,
                    session_id=session,
                    source_epoch_id=self.capture["source_epoch_id"],
                )
                channel = _DiskEventChannel(
                    recording_evidence_sink=lambda event: (
                        USBLiveProjectionSource._persist_recording_event(context, event)
                    ),
                )

                def on_tail_saved():
                    nonlocal tail_saved
                    tail_saved = True
                    self.store.set_recording_capture_progress(
                        session,
                        tail_status="SAVED",
                        processing_status="PROCESSING",
                    )

                runtime, evidence = self._factory.recover_recording(
                    self.capture,
                    channel,
                    on_tail_saved=on_tail_saved,
                )
                self._runtime = runtime
                if self._stopped.is_set():
                    runtime._source.cancel()
                events = runtime.run()
                if self._stopped.is_set():
                    return
                for event in events:
                    if event.kind == "CAPTURE_DEGRADED":
                        raise ValueError(f"recording recovery processing failed: {event.reason}")
                cutoff = self.capture["cutoff"]
                frame = runtime.current_capture_frame
                if cutoff is not None and (
                    frame is None
                    or frame.capture_frame_id != cutoff["frame_id"]
                    or frame.captured_at_ns != cutoff["captured_at_ns"]
                ):
                    raise ValueError("recording recovery did not process its original cutoff")
                self.store.prepare_recording_end(
                    session,
                    self.capture["source_epoch_id"],
                    cutoff,
                    evidence.finalized_manifest_path,
                    evidence.finalized_segment_receipts,
                    background=True,
                )
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            if not self._stopped.is_set() and "session media is pinned" not in str(exc):
                if self.store.get_recording_end_manifest(session) is not None:
                    return
                self.store.set_recording_capture_progress(
                    session,
                    tail_status="SAVED" if tail_saved else "FAILED",
                    processing_status="FAILED",
                    failure_code=f"RECOVERY_{type(exc).__name__}",
                )


def prepare_recording_recovery_factory(
    *,
    capture_setup_path,
    calibration_root,
    config_path,
    ring_root,
    jobs_root,
    source_kind,
    device_fingerprint=None,
):
    """Prepare only immutable recording semantics, never a camera/model backend.

    Managed admission still constructs its live runtime after signed startup.
    This disk-only factory verifies the prior scene before that admission.
    """
    from billiards_ai.lab.mainline1_calibration_registry import resolve_active_calibration
    from billiards_ai.lab.mainline1_capture import capture_setup_fingerprint, load_capture_setup
    from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory
    from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
    from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
    from billiards_ai.shared.calibration.homography import TableCalibration

    setup = load_capture_setup(capture_setup_path)
    if (
        setup.source_kind != source_kind
        or (setup.geometry.width, setup.geometry.height, setup.requested_fps) != (1920, 1080, 60)
        or (
            device_fingerprint is not None
            and setup.device_locator_fingerprint != device_fingerprint
        )
    ):
        raise ValueError("recording recovery capture setup changed")
    record = resolve_active_calibration(calibration_root, capture_setup_fingerprint(setup))
    if record is None:
        raise ValueError("recording recovery calibration unavailable")
    calibration = TableCalibration.model_validate_json(
        (calibration_root / record.artifact_path / "calibration.json").read_bytes()
    )
    if calibration.device_id != setup.camera_device_id:
        raise ValueError("recording recovery calibration identity changed")

    def no_capture(_epoch):
        raise RuntimeError("disk recovery must not open capture")

    return build_live_runtime_factory(
        capture_factory=no_capture,
        capture_config=RealtimeCaptureConfig(),
        calibration=calibration,
        detector=None,
        identity_classifier=None,
        shot_config=load_mainline1_shot_local_config(config_path),
        capture_setup_id=setup.capture_setup_id,
        calibration_artifact_id=record.artifact_id,
        config_path=config_path,
        ring_root=ring_root,
        jobs_root=jobs_root,
        session_policy="recording-v1",
    )
