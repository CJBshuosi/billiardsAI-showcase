"""Slow pixel motion must survive ordinary recording without semantic models."""

import threading

import cv2
import numpy as np

from tests.integration.lab.field_assistance import test_recording_first_runtime as fixture
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _setup_players,
    _until,
)


def test_slow_ball_produces_complete_original_interval(tmp_path, monkeypatch):
    def slow_video(path, total_frames=450):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 120))
        assert writer.isOpened()
        try:
            for frame_id in range(total_frames):
                image = np.zeros((120, 160, 3), dtype=np.uint8)
                x = 30 + min(20, max(0, frame_id - 90) // 5)
                cv2.circle(image, (x, 60), 3, (230, 230, 230), -1)
                writer.write(image)
        finally:
            writer.release()

    monkeypatch.setattr(fixture, "_two_shots", slow_video)
    paused, resume = threading.Event(), threading.Event()
    server, release, semantic_failure = fixture._recording_server(
        tmp_path,
        monkeypatch,
        total_frames=900,
        pause_after=380,
        paused=paused,
        resume=resume,
    )
    try:
        state = server.field_state
        _setup_players(state)
        release.set()
        assert paused.wait(8)
        shots = _until(
            lambda: (
                rows if len(rows := state.record_store.list_shots(state.session_id)) == 1 else None
            ),
            timeout_seconds=2,
            label="slow-motion original shot",
        )
        evidence = state.record_store.get_recording_evidence(state.session_id, shots[0].shot_id)
        assert evidence["media_status"] == "RECORDED"
        assert evidence["media"]["start_frame_id"] <= 95
        assert evidence["media"]["end_frame_id"] >= 190
        assert not semantic_failure.is_set()
    finally:
        release.set()
        resume.set()
        server.server_close()
