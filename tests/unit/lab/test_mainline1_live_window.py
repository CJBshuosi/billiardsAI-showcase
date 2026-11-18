from __future__ import annotations

import pytest

from billiards_ai.lab.mainline1_live_window import (
    IncrementalShotWindowCoordinator,
    LiveShotBoundaryKind,
    VisualShotBoundaryDetector,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.lab.shot_local.window import ShotWindowConfig, segment_shot_windows


def _frame(frame_id: int, timestamp_ms: int, x_mm: int) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        observations=(
            CalibratedBallObservation(
                detection_id=f"d{frame_id}",
                x_mm=x_mm,
                y_mm=100,
                confidence=0.9,
                group=BallGroup.CUE,
                group_confidence=0.9,
            ),
        ),
    )


def _config() -> ShotWindowConfig:
    return ShotWindowConfig(8, 2, 400)


def _sequence() -> tuple[ObservationFrame, ...]:
    return (
        _frame(0, 0, 100),
        _frame(1, 40, 100),
        _frame(2, 80, 110),
        _frame(3, 120, 120),
        _frame(4, 120, 120),
        _frame(5, 520, 120),
    )


def test_incremental_completed_window_matches_batch_segmenter() -> None:
    coordinator = IncrementalShotWindowCoordinator(_config())
    boundaries = tuple(boundary for frame in _sequence() for boundary in coordinator.push(frame))

    assert [boundary.kind for boundary in boundaries] == [
        LiveShotBoundaryKind.SHOT_STARTED,
        LiveShotBoundaryKind.SHOT_SETTLED,
    ]
    assert boundaries[-1].window == segment_shot_windows(_sequence(), _config())[0]


def test_resumed_window_keeps_explicit_persisted_shot_number():
    coordinator = IncrementalShotWindowCoordinator(_config(), initial_shot_number=7)
    boundaries = tuple(boundary for frame in _sequence() for boundary in coordinator.push(frame))
    assert [boundary.window.shot_id for boundary in boundaries] == ["shot-0008", "shot-0008"]
    with pytest.raises(ValueError):
        IncrementalShotWindowCoordinator(_config(), initial_shot_number=-1)


@pytest.mark.parametrize("reason", ["camera_disconnected", "source_stopped", "session_ended"])
def test_interrupt_during_motion_never_emits_settled(reason: str) -> None:
    coordinator = IncrementalShotWindowCoordinator(_config())
    emitted = tuple(boundary for frame in _sequence()[:4] for boundary in coordinator.push(frame))

    interrupted = coordinator.interrupt(reason)

    assert emitted[-1].kind is LiveShotBoundaryKind.SHOT_STARTED
    assert interrupted.kind is LiveShotBoundaryKind.SHOT_INTERRUPTED
    assert interrupted.status == "abstained"
    assert interrupted.reason == reason
    assert interrupted.window is not None
    assert interrupted.window.complete is False


def test_interrupted_epoch_requires_new_stability_before_another_start() -> None:
    coordinator = IncrementalShotWindowCoordinator(_config())
    for frame in _sequence()[:4]:
        coordinator.push(frame)
    coordinator.interrupt("camera_disconnected")

    recovering_motion = (
        _frame(10, 600, 140),
        _frame(11, 640, 150),
        _frame(12, 680, 160),
    )
    assert not tuple(
        boundary for frame in recovering_motion for boundary in coordinator.push(frame)
    )

    stable_then_motion = (
        _frame(13, 720, 160),
        _frame(14, 1120, 160),
        _frame(15, 1160, 170),
        _frame(16, 1200, 180),
    )
    emitted = tuple(
        boundary for frame in stable_then_motion for boundary in coordinator.push(frame)
    )

    assert [boundary.kind for boundary in emitted] == [LiveShotBoundaryKind.SHOT_STARTED]
    assert emitted[0].window is not None
    assert emitted[0].window.shot_id == "shot-0002"


def test_incremental_input_rejects_out_of_order_frames() -> None:
    coordinator = IncrementalShotWindowCoordinator(_config())
    coordinator.push(_frame(2, 100, 100))

    with pytest.raises(ValueError, match="ordered"):
        coordinator.push(_frame(2, 120, 100))


def test_visual_boundary_detector_does_not_require_heavy_ball_inference() -> None:
    detector = VisualShotBoundaryDetector(_config())
    samples = (
        (0, 0, False),
        (1, 40, False),
        (2, 80, True),
        (3, 120, True),
        (4, 160, False),
        (5, 560, False),
    )

    boundaries = tuple(
        boundary
        for frame_id, timestamp_ms, moving in samples
        for boundary in detector.push(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            moving=moving,
        )
    )

    assert [boundary.kind for boundary in boundaries] == [
        LiveShotBoundaryKind.SHOT_STARTED,
        LiveShotBoundaryKind.SHOT_SETTLED,
    ]
    assert boundaries[-1].window is not None
    assert boundaries[-1].window.start_frame_id == 2
    assert boundaries[-1].window.end_frame_id == 5


@pytest.mark.parametrize("confirmation_window,started", [(0, False), (400, True)])
def test_only_ordinary_policy_can_confirm_distinct_slow_displacements(confirmation_window, started):
    detector = VisualShotBoundaryDetector(
        _config(), motion_confirmation_window_ms=confirmation_window
    )
    events = [
        event
        for index, (timestamp, moving) in enumerate(
            [(0, False), (10, True), (100, False), (200, True)]
        )
        for event in detector.push(frame_id=index, timestamp_ms=timestamp, moving=moving)
    ]
    assert bool(events) is started
    if started:
        assert events[0].window.start_ms == 10


def test_slow_displacement_confirmation_expires_at_stability_duration():
    detector = VisualShotBoundaryDetector(_config(), motion_confirmation_window_ms=400)
    events = [
        event
        for index, (timestamp, moving) in enumerate(
            [(0, False), (10, True), (100, False), (410, True)]
        )
        for event in detector.push(frame_id=index, timestamp_ms=timestamp, moving=moving)
    ]
    assert events == []
