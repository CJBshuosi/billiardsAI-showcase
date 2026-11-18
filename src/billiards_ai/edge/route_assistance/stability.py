"""Identity-free online table stability reducer."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from billiards_ai.edge.route_assistance.models import MotionState, SceneObservation


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    motion_threshold_mm: float
    motion_confirm_frames: int
    stable_duration_ms: int

    def __post_init__(self) -> None:
        if self.motion_threshold_mm <= 0:
            raise ValueError("motion threshold must be positive")
        if self.motion_confirm_frames <= 0:
            raise ValueError("motion confirmation must be positive")
        if self.stable_duration_ms < 0:
            raise ValueError("stable duration must be non-negative")


@dataclass(frozen=True, slots=True)
class StabilitySnapshot:
    motion: MotionState = MotionState.UNKNOWN
    previous_observation: SceneObservation | None = None
    pending_motion_count: int = 0
    stable_since_ms: int | None = None
    scene_generation: int = 0

    def __post_init__(self) -> None:
        if self.pending_motion_count < 0 or self.scene_generation < 0:
            raise ValueError("stability counters must be non-negative")
        if self.stable_since_ms is not None and self.stable_since_ms < 0:
            raise ValueError("stable timestamp must be non-negative")

    @classmethod
    def initial(cls) -> StabilitySnapshot:
        return cls()


def _directed_distance(first: SceneObservation, second: SceneObservation) -> float:
    if not first.centers_mm:
        return 0.0
    if not second.centers_mm:
        return math.inf
    return max(
        min(math.hypot(x1 - x2, y1 - y2) for x2, y2 in second.centers_mm)
        for x1, y1 in first.centers_mm
    )


def scene_change_mm(first: SceneObservation, second: SceneObservation) -> float:
    """Return symmetric set displacement without consuming detection identities."""

    if len(first.centers_mm) != len(second.centers_mm):
        return math.inf
    if not first.centers_mm:
        return 0.0
    return max(_directed_distance(first, second), _directed_distance(second, first))


def _validate_order(previous: SceneObservation, current: SceneObservation) -> None:
    if current.frame_id <= previous.frame_id or current.timestamp_ms < previous.timestamp_ms:
        raise ValueError("scene observations must be ordered")


def observe_scene(
    snapshot: StabilitySnapshot,
    observation: SceneObservation,
    config: StabilityConfig,
) -> StabilitySnapshot:
    """Reduce one observation into a confirmed stable or moving state."""

    previous = snapshot.previous_observation
    if previous is None:
        return replace(
            snapshot,
            previous_observation=observation,
            stable_since_ms=observation.timestamp_ms,
        )
    _validate_order(previous, observation)
    moving_transition = (
        observation.visual_motion
        if observation.visual_motion is not None
        else scene_change_mm(previous, observation) >= config.motion_threshold_mm
    )

    if moving_transition:
        pending = snapshot.pending_motion_count + 1
        motion = MotionState.MOVING if pending >= config.motion_confirm_frames else snapshot.motion
        return replace(
            snapshot,
            motion=motion,
            previous_observation=observation,
            pending_motion_count=pending,
            stable_since_ms=None,
        )

    if snapshot.pending_motion_count and snapshot.motion is not MotionState.MOVING:
        return replace(
            snapshot,
            previous_observation=observation,
            pending_motion_count=0,
            stable_since_ms=observation.timestamp_ms,
        )

    stable_since = snapshot.stable_since_ms
    if stable_since is None:
        stable_since = previous.timestamp_ms
    stable_confirmed = observation.timestamp_ms - stable_since >= config.stable_duration_ms
    if not stable_confirmed:
        return replace(
            snapshot,
            previous_observation=observation,
            pending_motion_count=0,
            stable_since_ms=stable_since,
        )

    generation = snapshot.scene_generation
    if snapshot.motion is MotionState.MOVING:
        generation += 1
    return replace(
        snapshot,
        motion=MotionState.STABLE,
        previous_observation=observation,
        pending_motion_count=0,
        stable_since_ms=stable_since,
        scene_generation=generation,
    )
