"""Breakable non-cue motion fragments that abstain on association ambiguity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    CueSample,
    FragmentSample,
    LocalState,
    MotionFragment,
    ObservationFrame,
    ShotAnchor,
    ShotWindow,
)


@dataclass(frozen=True, slots=True)
class FragmentTrackingConfig:
    motion_threshold_mm: float
    continuation_distance_mm: float
    ambiguity_mm: float

    def __post_init__(self) -> None:
        if self.motion_threshold_mm <= 0:
            raise ValueError("fragment motion threshold must be positive")
        if self.continuation_distance_mm <= 0:
            raise ValueError("fragment continuation distance must be positive")
        if self.ambiguity_mm < 0:
            raise ValueError("fragment ambiguity distance must be non-negative")


@dataclass(slots=True)
class _ActiveFragment:
    fragment_id: str
    group: BallGroup
    samples: list[FragmentSample] = field(default_factory=list)
    terminal_state: LocalState | None = None


def _distance(
    first_x: int,
    first_y: int,
    second: CalibratedBallObservation,
) -> float:
    return math.hypot(first_x - second.x_mm, first_y - second.y_mm)


def _sample(
    *,
    window: ShotWindow,
    fragment_id: str,
    frame: ObservationFrame,
    observation: CalibratedBallObservation,
) -> FragmentSample:
    return FragmentSample(
        shot_id=window.shot_id,
        fragment_id=fragment_id,
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=LocalState.CONFIRMED,
        x_mm=observation.x_mm,
        y_mm=observation.y_mm,
        confidence=observation.confidence,
        source_detection_id=observation.detection_id,
    )


def _moved_from_anchor(
    observation: CalibratedBallObservation,
    anchors: Sequence[ShotAnchor],
    threshold_mm: float,
) -> bool:
    if not anchors:
        return False
    nearest = min(
        math.hypot(observation.x_mm - anchor.x_mm, observation.y_mm - anchor.y_mm)
        for anchor in anchors
    )
    return nearest >= threshold_mm


def track_motion_fragments(
    window: ShotWindow,
    frames: Sequence[ObservationFrame],
    anchors: Sequence[ShotAnchor],
    cue_samples: Sequence[CueSample],
    config: FragmentTrackingConfig,
) -> tuple[MotionFragment, ...]:
    """Track only unambiguous motion; a fragment is not a physical-ball identity."""

    if any(anchor.shot_id != window.shot_id for anchor in anchors) or any(
        sample.shot_id != window.shot_id for sample in cue_samples
    ):
        raise ValueError("fragment input violates shot scope")
    cue_detection_by_frame = {
        sample.frame_id: sample.source_detection_id
        for sample in cue_samples
        if sample.state in {LocalState.CONFIRMED, LocalState.REACQUIRED}
        and sample.source_detection_id is not None
    }
    selected_frames = sorted(
        (
            frame
            for frame in frames
            if window.start_frame_id <= frame.frame_id <= window.end_frame_id
        ),
        key=lambda frame: frame.frame_id,
    )
    active: list[_ActiveFragment] = []
    next_fragment_number = 1

    for frame in selected_frames:
        cue_detection_id = cue_detection_by_frame.get(frame.frame_id)
        available = {
            item.detection_id: item
            for item in frame.observations
            if item.detection_id != cue_detection_id and item.group is not BallGroup.CUE
        }
        blocked_ids: set[str] = set()

        for fragment in active:
            if fragment.terminal_state is not None:
                continue
            previous = fragment.samples[-1]
            ranked = sorted(
                (
                    (_distance(previous.x_mm, previous.y_mm, candidate), candidate)
                    for candidate in available.values()
                ),
                key=lambda item: (item[0], item[1].detection_id),
            )
            nearby = [item for item in ranked if item[0] <= config.continuation_distance_mm]
            if not nearby:
                fragment.terminal_state = LocalState.LOST
                continue
            if len(nearby) > 1 and nearby[1][0] - nearby[0][0] <= config.ambiguity_mm:
                fragment.terminal_state = LocalState.AMBIGUOUS
                blocked_ids.update((nearby[0][1].detection_id, nearby[1][1].detection_id))
                continue
            observation = nearby[0][1]
            fragment.samples.append(
                _sample(
                    window=window,
                    fragment_id=fragment.fragment_id,
                    frame=frame,
                    observation=observation,
                )
            )
            available.pop(observation.detection_id)

        for detection_id in blocked_ids:
            available.pop(detection_id, None)
        for observation in sorted(available.values(), key=lambda item: item.detection_id):
            if not _moved_from_anchor(observation, anchors, config.motion_threshold_mm):
                continue
            fragment_id = f"{window.shot_id}-fragment-{next_fragment_number:04d}"
            next_fragment_number += 1
            active.append(
                _ActiveFragment(
                    fragment_id=fragment_id,
                    group=observation.group,
                    samples=[
                        _sample(
                            window=window,
                            fragment_id=fragment_id,
                            frame=frame,
                            observation=observation,
                        )
                    ],
                )
            )

    default_terminal = LocalState.STOPPED if window.complete else LocalState.LOST
    return tuple(
        MotionFragment(
            shot_id=window.shot_id,
            fragment_id=fragment.fragment_id,
            group=fragment.group,
            samples=tuple(fragment.samples),
            terminal_state=fragment.terminal_state or default_terminal,
        )
        for fragment in active
    )
