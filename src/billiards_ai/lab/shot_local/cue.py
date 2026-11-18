"""Fail-closed cue-ball state machine for a single shot window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    CueSample,
    LocalState,
    ObservationFrame,
    ShotWindow,
)


@dataclass(frozen=True, slots=True)
class CueTrackingConfig:
    ambiguity_margin: float
    reacquire_confidence_min: float

    def __post_init__(self) -> None:
        if not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("cue ambiguity margin must be between zero and one")
        if not 0 <= self.reacquire_confidence_min <= 1:
            raise ValueError("cue confidence minimum must be between zero and one")


def _score(observation: CalibratedBallObservation) -> float:
    return observation.confidence * observation.group_confidence


def _missing_sample(
    *,
    window: ShotWindow,
    frame: ObservationFrame,
    state: LocalState,
    confidence: float = 0.0,
) -> CueSample:
    return CueSample(
        shot_id=window.shot_id,
        local_id=f"{window.shot_id}-cue",
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        state=state,
        x_mm=None,
        y_mm=None,
        confidence=confidence,
        source_detection_id=None,
    )


def track_cue(
    window: ShotWindow,
    frames: Sequence[ObservationFrame],
    config: CueTrackingConfig,
) -> tuple[CueSample, ...]:
    """Track only current-frame cue evidence; never reuse a prior center."""

    selected_frames = tuple(
        sorted(
            (
                frame
                for frame in frames
                if window.start_frame_id <= frame.frame_id <= window.end_frame_id
            ),
            key=lambda frame: frame.frame_id,
        )
    )
    if not selected_frames:
        raise ValueError("shot window contains no observation frames")
    if (
        selected_frames[0].frame_id != window.start_frame_id
        or selected_frames[-1].frame_id != window.end_frame_id
    ):
        raise ValueError("shot window observation boundary is incomplete")

    samples = []
    previous_state: LocalState | None = None
    for frame in selected_frames:
        candidates = sorted(
            (item for item in frame.observations if item.group is BallGroup.CUE),
            key=lambda item: (-_score(item), item.detection_id),
        )
        if not candidates or _score(candidates[0]) < config.reacquire_confidence_min:
            sample = _missing_sample(window=window, frame=frame, state=LocalState.LOST)
        elif (
            len(candidates) > 1
            and _score(candidates[0]) - _score(candidates[1]) <= config.ambiguity_margin
        ):
            sample = _missing_sample(
                window=window,
                frame=frame,
                state=LocalState.AMBIGUOUS,
                confidence=_score(candidates[0]),
            )
        else:
            candidate = candidates[0]
            state = (
                LocalState.REACQUIRED
                if previous_state in {LocalState.LOST, LocalState.AMBIGUOUS}
                else LocalState.CONFIRMED
            )
            sample = CueSample(
                shot_id=window.shot_id,
                local_id=f"{window.shot_id}-cue",
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                state=state,
                x_mm=candidate.x_mm,
                y_mm=candidate.y_mm,
                confidence=_score(candidate),
                source_detection_id=candidate.detection_id,
            )
        samples.append(sample)
        previous_state = sample.state
    return tuple(samples)
