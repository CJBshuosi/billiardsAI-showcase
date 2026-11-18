"""Ordinary product detection must catch up from saved pixels after a long stall."""

import threading
from dataclasses import replace

import pytest

from billiards_ai.lab import mainline1_product_realtime_runtime as product
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)
from tests.integration.lab.field_assistance.test_recording_first_runtime import _recording_server


@pytest.mark.parametrize("clear_current_readiness", [False, True])
def test_product_detector_catches_up_after_capture_saves_more_than_180_frames(
    tmp_path, monkeypatch, clear_current_readiness
):
    blocked, continue_detection = threading.Event(), threading.Event()
    paused, resume_capture = threading.Event(), threading.Event()
    processed = []
    original = product.ProductRealtimeRuntime._handle_visual_boundary

    def delayed_detection(runtime, frame, events):
        if not processed:
            blocked.set()
            assert continue_detection.wait(10)
        original(runtime, frame, events)
        processed.append(frame.capture_frame_id)

    monkeypatch.setattr(
        product.ProductRealtimeRuntime,
        "_handle_visual_boundary",
        delayed_detection,
    )
    server, release, semantic_failure = _recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume_capture,
    )
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert blocked.wait(3)
        assert paused.wait(7)
        health = state.projection_source.runtime.recording_writer_health
        assert health.committed_frames == 360 and health.reason is None
        assert processed == []
        if clear_current_readiness:
            # These pixels were recorded while players were ready. Processing
            # them later must not consult a newer roster/readiness state.
            match = state.record_store.load_session(state.session_id).match
            state.record_store.save_match(
                state.session_id,
                replace(match, breaker=None, active_role=None, revision=match.revision + 1),
                expected_revision=match.revision,
            )
        continue_detection.set()
        shots = _until(
            lambda: (
                rows
                if len(rows := state.record_store.list_shots(state.session_id)) == 2
                and all(
                    state.record_store.get_recording_evidence(state.session_id, row.shot_id)[
                        "media_status"
                    ]
                    == "RECORDED"
                    for row in rows
                )
                else None
            ),
            timeout_seconds=3,
            label="disk detection catches up without FIFO loss",
        )
        assert [row.sequence for row in shots] == [1, 2]
        _until(
            lambda: True if len(processed) >= 360 else None,
            timeout_seconds=2,
            label="all committed pixels processed in original order",
        )
        assert processed == list(range(360))
        assert not semantic_failure.is_set()
        assert all(
            state.record_store.get_shot_ownership(state.session_id, row.shot_id).evidence_status
            == "unknown"
            for row in shots
        )
    finally:
        continue_detection.set()
        release.set()
        resume_capture.set()
        server.server_close()
