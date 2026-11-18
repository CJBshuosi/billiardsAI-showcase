"""Lose the actual field server after durable end, before its detector catches up."""

import json
import os
import sys
import threading
from pathlib import Path

from pytest import MonkeyPatch

from billiards_ai.lab import mainline1_product_realtime_runtime as product
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_background_end import _end
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


def run(
    root, crash_frame, stop_stage="ended", processing_start_delay_seconds=0, capture_pause_frame=380
):
    monkeypatch = MonkeyPatch()
    if stop_stage == "recover_interrupted":
        sid = json.loads((root / "interrupted-session.json").read_text())["session_id"]
        original = product.ProductRealtimeRuntime._handle_visual_boundary

        def lose_recovery(runtime, frame, events):
            original(runtime, frame, events)
            if runtime._session_id == sid and frame.capture_frame_id == 185:
                os._exit(24)

        monkeypatch.setattr(
            product.ProductRealtimeRuntime, "_handle_visual_boundary", lose_recovery
        )
        server, release, _ = _recording_server(
            root, monkeypatch, source_epoch_id="unused-new-camera"
        )
        threading.Event().wait(10)
        raise AssertionError("recovery did not reach the loss point")
    blocked, paused = threading.Event(), threading.Event()
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed(runtime, frame, events):
        if frame.capture_frame_id == 0 and processing_start_delay_seconds:
            threading.Event().wait(processing_start_delay_seconds)
        original(runtime, frame, events)
        if frame.capture_frame_id == crash_frame:
            blocked.set()
            threading.Event().wait(20)

    monkeypatch.setattr(product.ProductRealtimeRuntime, "_handle_visual_boundary", delayed)
    managed = None
    if stop_stage == "interrupted_managed":
        from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
        from tests.unit.lab.field_assistance.test_managed_startup import (
            REQUEST,
            Remote,
            coordinator,
        )

        original_init = SQLiteFieldRecordStore.__init__

        def fixed_birth(store, *args, **kwargs):
            kwargs.setdefault("managed_open_request_id_factory", lambda: REQUEST)
            original_init(store, *args, **kwargs)

        monkeypatch.setattr(SQLiteFieldRecordStore, "__init__", fixed_birth)
        managed = coordinator(Remote())
    server, release, _ = _recording_server(
        root,
        monkeypatch,
        managed_startup=managed,
        total_frames=900,
        pause_after=capture_pause_frame,
        paused=paused,
        resume=threading.Event(),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state = server.field_state
    _setup_players(state)
    release.set()
    # Disk detection may lag the live camera. Establish the capture boundary
    # before waiting for the detector's crash point; retain the same 7+4 budget.
    assert paused.wait(7) and blocked.wait(4)
    if stop_stage in {"interrupted", "interrupted_managed"}:
        # No end API or finally: the last partially admitted chunk is unknown.
        (root / "interrupted-session.json").write_text(
            json.dumps(
                {
                    "session_id": state.session_id,
                    "source_epoch_id": state.projection_source.source_epoch_id,
                }
            )
        )
        os._exit(23)
    if stop_stage == "intent":

        def fail_end(*args, **kwargs):
            raise OSError("controlled loss before end session transaction")

        monkeypatch.setattr(state.record_store, "end_session", fail_end)
    status, reply = _end(server, state.session_id)
    assert status == (400 if stop_stage == "intent" else 200), reply
    _until(
        lambda: (
            True
            if state.record_store.get_recording_capture_end(state.session_id)["tail_status"]
            == "SAVED"
            else None
        ),
        timeout_seconds=3,
        label="durable tail before actual process loss",
    )
    (root / "crashed-session.json").write_text(
        json.dumps(state.record_store.get_recording_capture_end(state.session_id))
    )
    os._exit(23)


if __name__ == "__main__":
    run(
        Path(sys.argv[1]),
        int(sys.argv[2]),
        sys.argv[3] if len(sys.argv) > 3 else "ended",
        float(sys.argv[4]) if len(sys.argv) > 4 else 0,
        int(sys.argv[5]) if len(sys.argv) > 5 else 380,
    )
