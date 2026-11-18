"""Small normalized contracts shared by tracking and event observers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EventStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    ABSTAINED = "abstained"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BallState:
    identity: str
    x_mm: int
    y_mm: int
    confidence: float
    track_id: str = ""
    source_detection_id: str | None = None


@dataclass(frozen=True)
class TrackedFrame:
    frame_id: int
    timestamp_ms: int
    balls: tuple[BallState, ...]

    def by_identity(self) -> dict[str, BallState]:
        return {ball.identity: ball for ball in self.balls}


@dataclass(frozen=True)
class EvidenceEvent:
    event_type: str
    status: EventStatus
    timestamp_ms: int | None
    actor_identity: str | None
    target_identity: str | None
    evidence_frame_ids: tuple[int, ...]
    confidence: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_frame_ids:
            raise ValueError("events require evidence frames")
        if not 0 <= self.confidence <= 1:
            raise ValueError("event confidence must be between zero and one")


@dataclass(frozen=True)
class ShotInterval:
    start_ms: int
    end_ms: int
    evidence_frame_ids: tuple[int, ...]
    complete: bool
