from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance.models import MotionState, SceneObservation
from billiards_ai.edge.route_assistance.stability import (
    StabilityConfig,
    StabilitySnapshot,
    observe_scene,
)


def _scene(
    frame_id: int,
    timestamp_ms: int,
    *centers: tuple[int, int],
    visual_motion: bool | None = None,
) -> SceneObservation:
    return SceneObservation(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        centers_mm=tuple(centers),
        visual_motion=visual_motion,
    )


def _config() -> StabilityConfig:
    return StabilityConfig(
        motion_threshold_mm=8,
        motion_confirm_frames=2,
        stable_duration_ms=400,
    )


def _reduce(observations: tuple[SceneObservation, ...]) -> tuple[StabilitySnapshot, ...]:
    snapshot = StabilitySnapshot.initial()
    snapshots = []
    for observation in observations:
        snapshot = observe_scene(snapshot, observation, _config())
        snapshots.append(snapshot)
    return tuple(snapshots)


def test_gate_confirms_motion_then_settlement_without_ball_ids() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100)),
            _scene(1, 100, (100, 100)),
            _scene(2, 200, (112, 100)),
            _scene(3, 300, (124, 100)),
            _scene(4, 400, (124, 100)),
            _scene(5, 800, (124, 100)),
        )
    )

    assert snapshots[3].motion is MotionState.MOVING
    assert snapshots[5].motion is MotionState.STABLE
    assert snapshots[5].scene_generation == 1


def test_initial_stationary_scene_becomes_stable_without_incrementing_generation() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100)),
            _scene(1, 200, (100, 100)),
            _scene(2, 400, (100, 100)),
        )
    )

    assert snapshots[-1].motion is MotionState.STABLE
    assert snapshots[-1].scene_generation == 0


def test_center_order_does_not_create_false_motion() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100), (400, 300)),
            _scene(1, 400, (400, 300), (100, 100)),
        )
    )

    assert snapshots[-1].motion is MotionState.STABLE


def test_single_moving_transition_does_not_confirm_motion() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100)),
            _scene(1, 400, (100, 100)),
            _scene(2, 500, (112, 100)),
            _scene(3, 600, (112, 100)),
        )
    )

    assert snapshots[1].motion is MotionState.STABLE
    assert snapshots[-1].motion is MotionState.STABLE
    assert snapshots[-1].pending_motion_count == 0


def test_detection_count_changes_can_confirm_motion_without_identity() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100)),
            _scene(1, 400, (100, 100)),
            _scene(2, 500, (100, 100), (300, 300)),
            _scene(3, 600, (100, 100), (300, 300), (600, 500)),
        )
    )

    assert snapshots[-1].motion is MotionState.MOVING


def test_visual_stability_ignores_detector_count_jitter() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100), visual_motion=None),
            _scene(1, 200, (100, 100), (300, 300), visual_motion=False),
            _scene(2, 400, (100, 100), visual_motion=False),
        )
    )

    assert snapshots[-1].motion is MotionState.STABLE


def test_visual_motion_overrides_unchanged_detector_centers() -> None:
    snapshots = _reduce(
        (
            _scene(0, 0, (100, 100)),
            _scene(1, 100, (100, 100), visual_motion=True),
            _scene(2, 200, (100, 100), visual_motion=True),
        )
    )

    assert snapshots[-1].motion is MotionState.MOVING


def test_out_of_order_observations_are_rejected() -> None:
    first = observe_scene(StabilitySnapshot.initial(), _scene(2, 200), _config())

    with pytest.raises(ValueError, match="ordered"):
        observe_scene(first, _scene(1, 100), _config())


@pytest.mark.parametrize(
    ("field", "value"),
    (("motion_threshold_mm", 0), ("motion_confirm_frames", 0), ("stable_duration_ms", -1)),
)
def test_invalid_gate_configuration_is_rejected(field: str, value: int) -> None:
    values = {
        "motion_threshold_mm": 8,
        "motion_confirm_frames": 2,
        "stable_duration_ms": 400,
    }
    values[field] = value

    with pytest.raises(ValueError):
        StabilityConfig(**values)
