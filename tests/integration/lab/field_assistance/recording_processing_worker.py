"""Synthetic detector worker for real process-loss/replay integration tests."""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab import mainline1_product_realtime_runtime as product
from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
from billiards_ai.lab.field_assistance.visual_motion import VisualMotionConfig
from billiards_ai.lab.mainline1_field_runtime import _ExternallyFedEvidenceStore
from billiards_ai.lab.mainline1_product_realtime_runtime import (
    ProductRealtimeEventChannel,
    ProductRealtimeRuntime,
    ProductRealtimeRuntimeConfig,
)
from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
from billiards_ai.lab.mainline1_session_audit import ShotBoundaryLedger
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.shared.calibration.homography import calibrate
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _SHOT_CONFIG_PATH,
)


def run(root, session, crash_frame, crash_after="frame", fault="exit", sink="real"):
    store = SQLiteFieldRecordStore(root / "field.sqlite3")
    raw = ContinuousSessionRecorder(
        root=root / "raw",
        session_id=session,
        expected_fps=60,
        read_only=True,
    )
    evidence = SessionRecordingEvidenceStore(
        recorder=raw,
        session_id=session,
        pre_roll_frames=0,
        post_roll_frames=60,
        maximum_shot_frames=2400,
        close_recorder=False,
    )
    context = SimpleNamespace(store=store, session_id=session, source_epoch_id="epoch")
    channel = ProductRealtimeEventChannel(
        recording_evidence_sink=lambda event: USBLiveProjectionSource._persist_recording_event(
            context,
            event,
        ),
    )
    if sink == "no_sink":
        channel = ProductRealtimeEventChannel()
    elif sink == "none":
        channel = None
    if fault == "materialize":

        def broken_reference(*args, **kwargs):
            raise OSError("controlled materialization I/O")

        product.materialize_recorded_interval_reference = broken_reference
    runtime = ProductRealtimeRuntime(
        source=evidence.committed_source(source_epoch_id="epoch"),
        capture_config=RealtimeCaptureConfig(width=160, height=120, fps=60),
        calibration=calibrate(
            "replay",
            [(0, 119), (159, 119), (159, 0), (0, 0)],
            table_length_mm=2540,
            table_width_mm=1270,
        ),
        detector=None,
        identity_classifier=None,
        shot_config=load_mainline1_shot_local_config(_SHOT_CONFIG_PATH),
        session_id="foreign-session" if fault == "foreign_session" else session,
        job_media_root=root / "media",
        evidence_store=_ExternallyFedEvidenceStore(evidence),
        boundary_ledger=ShotBoundaryLedger(
            root / "boundaries.sqlite3",
            session_id=session,
            source_epoch_id="foreign-epoch" if fault == "foreign_epoch" else "epoch",
        ),
        visual_motion_config=VisualMotionConfig(),
        event_channel=channel,
        config=ProductRealtimeRuntimeConfig(
            ring_root=root / "unused-ring",
            pre_roll_frames=0,
            post_roll_frames=60,
            initial_shot_number=len(store.list_shots(session)),
        ),
        session_policy="recording-v1",
    )

    def crash(frame):
        if frame.capture_frame_id == crash_frame:
            # Actual process death: no detector/source/finally cleanup can fake
            # an interrupted terminal boundary or rewrite the persistent state.
            assert runtime._recording_active is not None
            if fault == "hold":
                (root / "processing-held").write_text("held")
                deadline = time.monotonic() + 10
                while not (root / "release-processing").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("test holder release timed out")
                    time.sleep(0.01)
                os._exit(23)
            if fault == "exception":
                raise RuntimeError("controlled processing interruption")
            if fault == "cancel":
                runtime._source.cancel()
                return
            os._exit(23)

    if crash_after == "checkpoint":
        original = runtime._checkpoint_recording_segment

        def checkpoint(frame):
            original(frame)
            crash(frame)

        runtime._checkpoint_recording_segment = checkpoint
    else:
        original = runtime._handle_visual_boundary

        def process(frame, events):
            original(frame, events)
            crash(frame)

        runtime._handle_visual_boundary = process
    runtime.request_recording_end()
    events = runtime.run()
    if fault in {"exception", "materialize", "probe"}:
        print(json.dumps({"degraded": any(event.kind == "CAPTURE_DEGRADED" for event in events)}))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), *sys.argv[4:7])
