from __future__ import annotations

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorBall,
    AnchorObservation,
    RealtimeBallTracker,
    RealtimeTrackerConfig,
    TrackerHealthState,
)
from billiards_ai.lab.shot_local.models import BallGroup
from billiards_ai.shared.calibration.homography import calibrate


def _image(*, ball_x: int | None, second_x: int | None = None) -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    if ball_x is not None:
        cv2.circle(image, (ball_x, 60), 8, (240, 240, 240), -1)
        cv2.line(image, (ball_x - 5, 60), (ball_x + 5, 60), (10, 10, 10), 2)
        cv2.line(image, (ball_x, 55), (ball_x, 65), (10, 10, 10), 2)
    if second_x is not None:
        cv2.circle(image, (second_x, 60), 8, (180, 180, 180), -1)
        cv2.line(image, (second_x - 5, 55), (second_x + 5, 65), (20, 20, 20), 2)
    return image


def _capture(frame_id: int, image: np.ndarray) -> CaptureFrame:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    assert ok
    return CaptureFrame(
        source_epoch_id="epoch-1",
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 16_666_667,
        width=160,
        height=120,
        mjpeg=encoded.tobytes(),
    )


def _anchor(frame_id: int, *centers: tuple[float, float]) -> AnchorObservation:
    return AnchorObservation(
        capture_frame_id=frame_id,
        captured_at_ns=frame_id * 16_666_667,
        balls=tuple(
            AnchorBall(
                track_id=f"ball-{index}",
                center_px=center,
                confidence=0.96,
                group=BallGroup.SOLID,
                bbox_xyxy_px=(
                    center[0] - 10,
                    center[1] - 10,
                    center[0] + 10,
                    center[1] + 10,
                ),
            )
            for index, center in enumerate(centers)
        ),
    )


def _tracker(**changes: object) -> RealtimeBallTracker:
    calibration = calibrate(
        "camera",
        [(0, 119), (159, 119), (159, 0), (0, 0)],
        table_length_mm=1590,
        table_width_mm=1190,
    )
    values = {
        "working_width": 160,
        "working_height": 120,
        "minimum_features": 4,
        "ambiguity_distance_px": 12,
        **changes,
    }
    return RealtimeBallTracker(calibration, RealtimeTrackerConfig(**values))


def test_diagnostic_snapshot_reads_current_state_without_recomputing_history() -> None:
    tracker = _tracker()
    assert tracker.diagnostic_snapshot.last_computed_health is None
    tracker._durations_ms.extend((1.0, 3.0, 5.0))
    previous = tracker.health

    class HistoryMustNotBeRead:
        def __len__(self):
            return 1_000_000

        def __iter__(self):
            raise AssertionError("diagnostics must not scan tracker timing history")

    tracker._durations_ms = HistoryMustNotBeRead()
    tracker._maximum_lag = 7
    tracker._degrade("tracker source lag exceeds realtime budget")
    snapshot = tracker.diagnostic_snapshot
    assert snapshot.state is TrackerHealthState.DEGRADED
    assert snapshot.processed_frames == 1_000_000
    assert snapshot.maximum_source_lag_frames == 7
    assert snapshot.reason == "tracker source lag exceeds realtime budget"
    assert snapshot.last_computed_health is previous
    assert previous.processed_frames == 3 and previous.state is TrackerHealthState.RUNNING


def test_flow_tracker_emits_every_source_frame_and_follows_linear_ball_motion() -> None:
    tracker = _tracker()
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    rows = [
        tracker.track(_capture(frame_id, _image(ball_x=40 + frame_id * 2)))
        for frame_id in range(1, 7)
    ]

    assert [row.capture_frame_id for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row.balls[0].state for row in rows] == ["confirmed"] * 6
    assert abs((rows[-1].balls[0].center_px or (0, 0))[0] - 52.0) <= 1.5
    assert rows[-1].balls[0].provenance == "flow"
    assert rows[-1].balls[0].anchor_frame_id == 0


def test_flow_tracker_marks_loss_instead_of_reusing_last_center() -> None:
    tracker = _tracker()
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    lost = tracker.track(_capture(1, _image(ball_x=None)))

    assert lost.balls[0].state == "lost"
    assert lost.balls[0].center_px is None
    assert lost.balls[0].table_mm is None


@pytest.mark.parametrize("failure", ["forward", "backward", "support", "bounds"])
def test_lost_track_stays_unconfirmed_over_missing_frames_until_new_anchor(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tracker = _tracker(perf_counter_ns=lambda: 0)
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))
    calls = 0

    def flow(_previous, _current, points, _next, **kwargs):
        nonlocal calls
        calls += 1
        status = np.ones((len(points), 1), dtype=np.uint8)
        # Lose support on the first missing frame. Later background flow can
        # report perfect self-consistency; that cannot reacquire a lost ball.
        if (failure == "forward" and calls == 1) or (failure == "backward" and calls == 2):
            return None, None, None
        if failure == "support" and calls == 1:
            status[:] = 0
        delta = (
            200
            if failure == "bounds" and calls == 1
            else (-200 if failure == "bounds" and calls == 2 else 0)
        )
        return points + delta, status, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", flow)
    for frame_id in range(1, 5):
        row = tracker.track(_capture(frame_id, _image(ball_x=None)))
        ball = row.balls[0]
        assert ball.state == "lost"
        assert ball.center_px is None and ball.table_mm is None
        assert ball.confidence == 0
    visible = _capture(5, _image(ball_x=70))
    assert tracker.track(visible).balls[0].state == "lost"
    recovered = tracker.apply_anchor(_anchor(5, (70, 60)), visible)
    assert recovered.balls[0].state == "confirmed"
    assert recovered.balls[0].center_px == (70, 60)
    assert recovered.balls[0].anchor_frame_id == 5


def test_nearby_tracks_are_ambiguous_instead_of_silently_swapped() -> None:
    tracker = _tracker()
    tracker.apply_anchor(
        _anchor(0, (40, 60), (70, 60)),
        _capture(0, _image(ball_x=40, second_x=70)),
    )

    row = tracker.track(_capture(1, _image(ball_x=51, second_x=59)))

    assert {ball.state for ball in row.balls} == {"ambiguous"}
    assert all(ball.center_px is None for ball in row.balls)


def test_tracker_decodes_directly_to_working_grayscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _tracker(working_width=80, working_height=60)
    frame = _capture(0, _image(ball_x=40))

    def reject_full_resolution_decode(self: CaptureFrame) -> np.ndarray:
        raise AssertionError("tracker must not decode full-resolution BGR")

    monkeypatch.setattr(CaptureFrame, "decode_bgr", reject_full_resolution_decode)

    row = tracker.apply_anchor(_anchor(0, (40, 60)), frame)

    assert row.capture_frame_id == 0


def test_tracker_batches_optical_flow_for_all_balls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _tracker()
    tracker.apply_anchor(
        _anchor(0, (40, 60), (100, 60)),
        _capture(0, _image(ball_x=40, second_x=100)),
    )
    original = cv2.calcOpticalFlowPyrLK
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", counted)

    tracker.track(_capture(1, _image(ball_x=42, second_x=102)))

    assert calls == 2


def test_tracker_rejects_source_gap() -> None:
    tracker = _tracker()
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    with pytest.raises(ValueError, match="contiguous"):
        tracker.track(_capture(2, _image(ball_x=44)))
    assert tracker.health.state == TrackerHealthState.DEGRADED


def test_tracker_rejects_an_accounted_gap_beyond_the_safe_flow_window() -> None:
    tracker = _tracker(
        allow_accounted_frame_gaps=True,
        maximum_accounted_frame_gap=2,
    )
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    with pytest.raises(ValueError, match="safe flow window"):
        tracker.track(_capture(3, _image(ball_x=46)))

    assert tracker.health.state == TrackerHealthState.DEGRADED


class _Clock:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def test_processing_budget_and_capture_lag_fail_closed_with_fake_clock() -> None:
    tracker = _tracker(
        processing_budget_ms=16.7,
        perf_counter_ns=_Clock((0, 20_000_000)),
    )
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    tracker.track(
        _capture(1, _image(ball_x=42)),
        latest_capture_frame_id=4,
    )

    assert tracker.health.state == TrackerHealthState.DEGRADED
    assert tracker.health.p95_processing_ms == 20.0
    assert tracker.health.maximum_source_lag_frames == 3


def test_accounted_coalescing_source_reports_slow_tracking_without_disabling_results() -> None:
    tracker = _tracker(
        allow_accounted_frame_gaps=True,
        enforce_processing_budget=False,
        processing_budget_ms=16.7,
        perf_counter_ns=_Clock((0, 190_000_000)),
    )
    tracker.apply_anchor(_anchor(0, (40, 60)), _capture(0, _image(ball_x=40)))

    row = tracker.track(
        _capture(1, _image(ball_x=42)),
        latest_capture_frame_id=1,
    )

    assert tracker.health.state == TrackerHealthState.RUNNING
    assert tracker.health.p95_processing_ms == 190.0
    assert row.balls[0].state == "confirmed"
