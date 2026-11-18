"""Pure contracts for the local field route-assistance loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class AssistancePhase(StrEnum):
    WAITING_FOR_STABLE_TABLE = "WAITING_FOR_STABLE_TABLE"
    ANALYZING_TABLE = "ANALYZING_TABLE"
    SHOWING_GUIDANCE = "SHOWING_GUIDANCE"
    WAITING_FOR_SHOT = "WAITING_FOR_SHOT"
    BALLS_MOVING = "BALLS_MOVING"
    WAITING_FOR_SETTLE = "WAITING_FOR_SETTLE"
    COMPARING_RESULT = "COMPARING_RESULT"
    DEGRADED = "DEGRADED"


class MotionState(StrEnum):
    UNKNOWN = "unknown"
    STABLE = "stable"
    MOVING = "moving"


@dataclass(frozen=True, slots=True)
class SceneObservation:
    frame_id: int
    timestamp_ms: int
    centers_mm: tuple[tuple[int, int], ...]
    visual_motion: bool | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("scene frame and timestamp must be non-negative")


@dataclass(frozen=True, slots=True)
class PowerDisplay:
    band: Literal["light", "medium_light", "medium", "medium_firm", "firm"]
    approximate_percent: int
    personalized: Literal[False] = False

    def __post_init__(self) -> None:
        if not 1 <= self.approximate_percent <= 100:
            raise ValueError("power percentage must be between one and one hundred")
        if self.personalized is not False:
            raise ValueError("V0 power display must remain non-personalized")


@dataclass(frozen=True, slots=True)
class GuidanceDisplay:
    recommendation_id: str
    candidate_id: str
    scene_generation: int
    power: PowerDisplay

    def __post_init__(self) -> None:
        if not self.recommendation_id or not self.candidate_id:
            raise ValueError("guidance identifiers must not be empty")
        if self.scene_generation < 0:
            raise ValueError("scene generation must be non-negative")


@dataclass(frozen=True, slots=True)
class AssistanceSnapshot:
    phase: AssistancePhase = AssistancePhase.WAITING_FOR_STABLE_TABLE
    motion: MotionState = MotionState.UNKNOWN
    scene_generation: int = 0
    shot_sequence: int = 0
    guidance: GuidanceDisplay | None = None
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        if self.scene_generation < 0 or self.shot_sequence < 0:
            raise ValueError("generation and shot sequence must be non-negative")
        if self.guidance is not None and self.guidance.scene_generation != self.scene_generation:
            raise ValueError("guidance scene generation does not match snapshot")
        if self.phase is AssistancePhase.DEGRADED and not self.degraded_reason:
            raise ValueError("degraded snapshot requires a reason")


class AssistanceEffectKind(StrEnum):
    REQUEST_ANALYSIS = "REQUEST_ANALYSIS"
    SHOW_GUIDANCE = "SHOW_GUIDANCE"
    CLEAR_GUIDANCE = "CLEAR_GUIDANCE"
    SHOT_STARTED = "SHOT_STARTED"
    SHOT_SETTLED = "SHOT_SETTLED"


@dataclass(frozen=True, slots=True)
class AssistanceEffect:
    kind: AssistanceEffectKind
    scene_generation: int
    shot_sequence: int

    def __post_init__(self) -> None:
        if self.scene_generation < 0 or self.shot_sequence < 0:
            raise ValueError("effect generation and shot sequence must be non-negative")
