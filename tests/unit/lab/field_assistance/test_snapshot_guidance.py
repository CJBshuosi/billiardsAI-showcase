"""Real child cancellation and pixel-trigger lifecycle, not injected shot events."""

import importlib.util
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.shared.calibration.homography import calibrate


@dataclass
class BlockingObserver:
    marker: Path | None = None
    release: Path | None = None

    def __call__(self):
        return self

    def observe(self, frame, calibration):
        if self.marker is not None:
            self.marker.write_text("entered opaque inference", encoding="utf-8")
        while self.release is None or not self.release.exists():
            threading.Event().wait(0.1)


def frame(number, moving=False):
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(image, (180 if moving else 80, 120), 12, (255, 255, 255), -1)
    ok, jpg = cv2.imencode(".jpg", image)
    assert ok
    return CaptureFrame("epoch1", number, number * 100_000_000, 320, 240, jpg.tobytes())


def observer(tmp_path, *, prepare=True, releasable=False):
    assert (
        importlib.util.find_spec("billiards_ai.lab.field_assistance.snapshot_guidance") is not None
    ), "snapshot runner missing"
    from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotConfig, SnapshotGuidance

    service = SnapshotGuidance(
        hub=None,
        session_id="s1",
        source_epoch_id="epoch1",
        calibration=calibrate("test", [(0, 239), (319, 239), (319, 0), (0, 0)]),
        observer_factory=BlockingObserver(
            tmp_path / "inference-started.txt", tmp_path / "release" if releasable else None
        ),
        journal_path=tmp_path / "guidance.json",
        config=SnapshotConfig(sample_ms=100, stable_ms=200, stale_seconds=2, task_seconds=3),
    )
    if prepare:
        service.start()
        deadline = time.monotonic() + 5
        while service.projection()["model_state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.projection()["model_state"] == "ready"
    return service


def test_moving_pixels_revoke_active_call_without_destroying_models_or_queueing_work(tmp_path):
    service = observer(tmp_path)
    try:
        for i in range(4):
            service.observe(frame(i), context="solid")
        assert service.projection()["tasks_started"] == 1
        child = service.active_pid
        assert child is not None
        deadline = time.monotonic() + 3
        while not (tmp_path / "inference-started.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "inference-started.txt").exists(), "must cancel in-flight inference"
        service.observe(frame(4, moving=True), context="solid")
        state = service.projection()
        assert state["phase"] == "moving"
        assert state["recommendation"] is None
        assert service.active_pid is None
        assert state["worker_pid"] == child and service._process.is_alive()
        assert state["tasks_cancelled"] == 1
        for i in range(5, 9):
            service.observe(frame(i, moving=i % 2 == 0), context="solid")
        assert service.projection()["tasks_started"] == 1
    finally:
        service.close()


def test_foreign_epoch_pixels_cannot_authorize_guidance(tmp_path):
    from dataclasses import replace

    service = observer(tmp_path)
    try:
        for i in range(4):
            service.observe(replace(frame(i), source_epoch_id="foreign"), context="solid")
        assert service.projection()["tasks_started"] == 0
    finally:
        service.close()


def test_context_change_and_stale_input_revoke_task_without_restarting(tmp_path):
    service = observer(tmp_path)
    try:
        for i in range(4):
            service.observe(frame(i), context="solid")
        service.observe(frame(4), context=None)
        assert service.active_pid is None
        assert service.projection()["recommendation"] is None
        assert service.projection()["tasks_started"] == 1
    finally:
        service.close()


def test_repeated_stable_frames_launch_one_job_until_motion(tmp_path):
    service = observer(tmp_path)
    try:
        for i in range(12):
            service.observe(frame(i), context="solid")
        assert service.projection()["tasks_started"] == 1
    finally:
        service.close()


def test_stop_request_hides_advice_without_waiting_for_cleanup(tmp_path):
    service = observer(tmp_path)
    try:
        service._recommendation = {"source_frame_id": 0, "route": {}}
        service.request_stop()
        assert service.projection()["recommendation"] is None
        assert service.projection()["phase"] == "ended"
        for i in range(4):
            service.observe(frame(i), context="solid")
        assert service.projection()["tasks_started"] == 0
    finally:
        service.close()


def test_gap_longer_than_stability_horizon_revokes_task_and_requires_new_stability(tmp_path):
    service = observer(tmp_path, releasable=True)
    try:
        for i in range(4):
            service.observe(frame(i), context="solid")
        assert service.active_pid is not None
        # 700ms gap exceeds this service's 200ms stability horizon, but is
        # shorter than its 2s stale deadline. Continuity cannot be inferred.
        service.observe(frame(10), context="solid")
        assert service.active_pid is None
        assert service.projection()["recommendation"] is None
        assert service.projection()["tasks_started"] == 1
        deadline = time.monotonic() + 5
        while service.projection()["model_state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.projection()["model_state"] == "ready"
        service.observe(frame(11), context="solid")
        assert service.projection()["tasks_started"] == 1
        service.observe(frame(12), context="solid")
        # Stable pixels cannot queue a new call while the old one still owns its slot.
        assert service.projection()["tasks_started"] == 1
        (tmp_path / "release").touch()
        deadline = time.monotonic() + 5
        while service._busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not service._busy
        service.observe(frame(13), context="solid")
        assert service.projection()["tasks_started"] == 2
    finally:
        service.close()


def finish(service, result, **scope):
    from types import SimpleNamespace

    service._process = SimpleNamespace(
        pid=0, join=lambda _=None: None, is_alive=lambda: False, close=lambda: None
    )
    service._busy = True
    service._ticket = "test-ticket"
    service._inflight_generation = service._generation
    service._inflight_review = service._pending_review
    service._started_at = time.monotonic()
    service._connection = SimpleNamespace(
        poll=lambda: True,
        recv=lambda: {
            "kind": "result",
            "ticket": "test-ticket",
            "generation": service._generation,
            "epoch": service.epoch,
            "value": result,
            **scope,
        },
        close=lambda: None,
    )
    service._collect(frame(50))


def settled_feedback(tmp_path, monkeypatch, *, actual=None, region=True, next_route=True):
    from billiards_ai.lab.field_assistance import snapshot_guidance as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    service = observer(tmp_path, prepare=False)
    service.observe(frame(0))
    service.set_context("solid")
    previous = {
        "source_frame_id": 0,
        "route": {"pocket": "top_middle"},
        "region": {"center": {"x_mm": 1000, "y_mm": 500}, "radius_mm": 100} if region else None,
    }
    service._history.append(
        dict(
            motion_frame_id=1,
            recommendation=previous,
            actual_cue=None,
            observed_frame_id=None,
            region_result="unknown",
        )
    )
    service._pending_review = 0
    following = dict(
        source_frame_id=50,
        route={"pocket": "bottom_right"} if next_route else None,
        region=None,
        actual_cue=actual,
    )
    finish(service, following)
    return service, clock, previous, following


@pytest.mark.parametrize(
    ("actual", "outcome"),
    [
        ({"x_mm": 1000, "y_mm": 510}, "inside"),
        ({"x_mm": 1250, "y_mm": 500}, "outside"),
        ({"x_mm": 1095, "y_mm": 500}, "uncertain"),
        (None, "unknown"),
    ],
)
def test_previous_region_and_actual_endpoint_show_for_one_second_before_next_advice(
    tmp_path, monkeypatch, actual, outcome
):
    service, clock, previous, following = settled_feedback(tmp_path, monkeypatch, actual=actual)
    try:
        value = service.projection()
        assert value["phase"] == "reviewing"
        assert value["recommendation"] is None, "next recommendation has not been issued yet"
        assert value["feedback"]["recommendation"] == previous
        assert value["feedback"]["actual_cue"] == actual
        assert value["feedback"]["region_result"] == outcome
        assert value["feedback"]["observed_frame_id"] == 50
        assert value["observed_frame_id"] == 50
        clock[0] += 0.999
        service._advance_review()
        assert service.projection()["phase"] == "reviewing"
        clock[0] += 0.001
        service._advance_review()
        assert service.projection()["recommendation"] == following
        assert service.projection()["phase"] == "ready"
        assert service.projection()["feedback"] is None
        assert service.projection()["observed_frame_id"] == 50
        assert service.projection()["tasks_completed"] == 1
        assert len(service.projection()["reviews"]) == 1
    finally:
        service.close()


@pytest.mark.parametrize("revoke", ["motion", "context", "gap", "epoch", "stop"])
def test_feedback_revocation_cannot_publish_hidden_next_advice_or_register_it_as_issued(
    tmp_path, monkeypatch, revoke
):
    from dataclasses import replace

    service, clock, previous, _ = settled_feedback(
        tmp_path, monkeypatch, actual={"x_mm": 1000, "y_mm": 510}
    )
    try:
        assert service.projection()["phase"] == "reviewing"
        if revoke == "motion":
            service.observe(frame(1, moving=True))
        elif revoke == "context":
            service.set_context(None, key="different player")
        elif revoke == "gap":
            service.observe(frame(10))
        elif revoke == "epoch":
            service.observe(replace(frame(1), source_epoch_id="different epoch"))
        else:
            service.request_stop()
        assert service.projection()["feedback"] is None
        assert service.projection()["observed_frame_id"] is None
        clock[0] += 2
        service._advance_review()
        assert service.projection()["recommendation"] is None
        rows = service.projection()["reviews"]
        assert len(rows) == 1 and rows[0]["recommendation"] == previous
    finally:
        service.close()


def test_no_prior_region_does_not_delay_next_recommendation(tmp_path, monkeypatch):
    service, _, _, following = settled_feedback(tmp_path, monkeypatch, region=False)
    try:
        assert service.projection()["recommendation"] == following
        assert service.projection().get("feedback") is None
    finally:
        service.close()


def test_feedback_still_precedes_honest_next_scene_abstention(tmp_path, monkeypatch):
    service, clock, _, _ = settled_feedback(tmp_path, monkeypatch, next_route=False)
    try:
        assert service.projection()["phase"] == "reviewing"
        clock[0] += 1
        service._advance_review()
        assert service.projection()["phase"] == "abstained"
        assert service.projection()["feedback"] is None
        assert service.projection()["recommendation"] is None
    finally:
        service.close()


def test_reply_cannot_attach_actual_position_to_a_different_pending_review(tmp_path):
    from types import SimpleNamespace

    service = observer(tmp_path, prepare=False)
    service._history = [
        {"recommendation": {"region": None}, "actual_cue": None, "observed_frame_id": None},
        {"recommendation": {"region": None}, "actual_cue": None, "observed_frame_id": None},
    ]
    service._pending_review = service._inflight_review = 0
    service._busy = True
    service._ticket = "bound-reply"
    service._inflight_generation = service._generation
    service._started_at = time.monotonic()
    service._process = SimpleNamespace(
        pid=0, join=lambda _=None: None, is_alive=lambda: False, close=lambda: None
    )

    def receive():
        service._pending_review = 1
        return {
            "kind": "result",
            "ticket": "bound-reply",
            "generation": service._generation,
            "epoch": service.epoch,
            "value": {"source_frame_id": 50, "route": None, "actual_cue": {"x_mm": 1, "y_mm": 2}},
        }

    service._connection = SimpleNamespace(poll=lambda: True, recv=receive, close=lambda: None)
    try:
        service._collect(frame(50))
        assert all(row["actual_cue"] is None for row in service._history)
        assert service._pending_review == 1
    finally:
        service.close()


def test_failed_first_settled_observation_cannot_use_later_shot_endpoint(tmp_path):
    service = observer(tmp_path, prepare=False)
    try:
        service.observe(frame(0))
        service._phase = "ready"
        service._recommendation = {"source_frame_id": 0, "region": None}
        service.observe(frame(1, moving=True))
        assert len(service.projection()["reviews"]) == 1
        finish(service, {"error": "first settled inference failed"})
        service.observe(frame(2))  # Another shot starts before trustworthy observation.
        finish(
            service,
            {"source_frame_id": 50, "route": None, "actual_cue": {"x_mm": 1800, "y_mm": 500}},
        )
        row = service.projection()["reviews"][0]
        assert row["actual_cue"] is None
        assert row["region_result"] == "unknown"
    finally:
        service.close()


def test_receive_cannot_publish_a_result_after_the_task_deadline(tmp_path):
    from types import SimpleNamespace

    service = observer(tmp_path, prepare=False)
    service._process = SimpleNamespace(
        pid=0, join=lambda _=None: None, is_alive=lambda: False, close=lambda: None
    )
    service._busy = True
    service._ticket = "late-ticket"
    service._inflight_generation = service._generation
    service._started_at = time.monotonic()

    def delayed_receive():
        service._started_at -= service.config.task_seconds + 1
        return {
            "kind": "result",
            "ticket": "late-ticket",
            "generation": service._generation,
            "epoch": service.epoch,
            "value": {"source_frame_id": 5, "route": None, "actual_cue": None},
        }

    service._connection = SimpleNamespace(
        poll=lambda: True, recv=delayed_receive, close=lambda: None
    )
    try:
        service._collect(frame(5))
        assert service.projection()["tasks_completed"] == 0
        assert service.projection()["error"] == "guidance_task_timeout"
        assert service.projection()["recommendation"] is None
    finally:
        service.close()


@pytest.mark.parametrize(
    "scope", [{"ticket": "old-ticket"}, {"generation": -1}, {"epoch": "other"}]
)
def test_old_worker_result_cannot_complete_a_different_request(tmp_path, scope):
    service = observer(tmp_path, prepare=False)
    try:
        finish(service, {"source_frame_id": 50, "route": None, "actual_cue": None}, **scope)
        assert service.projection()["tasks_completed"] == 0
        assert service.projection()["recommendation"] is None
        assert service.projection()["error"] == "guidance_result_scope_mismatch"
    finally:
        service.close()
