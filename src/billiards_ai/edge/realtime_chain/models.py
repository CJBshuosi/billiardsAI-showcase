"""Immutable evidence and profile contracts for one shot-local replay."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceStatus(StrEnum):
    ADMITTED = "admitted"
    PENDING = "pending"
    ABSTAINED = "abstained"
    UNSUPPORTED = "unsupported"


def _confidence(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be finite and between zero and one")


def _text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    artifact_id: str
    source_video_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact ID")
        if not _SHA256.fullmatch(self.source_video_sha256) or not _SHA256.fullmatch(
            self.manifest_sha256
        ):
            raise ValueError("source lineage requires lowercase SHA-256 values")


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    anchor_id: str
    group: str
    x_mm: int
    y_mm: int
    confidence: float

    def __post_init__(self) -> None:
        _text(self.anchor_id, "anchor ID")
        _text(self.group, "anchor group")
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CueEvidence:
    frame_id: int
    timestamp_ms: int
    state: str
    x_mm: int | None
    y_mm: int | None
    confidence: float

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("cue frame and timestamp must be non-negative")
        if (self.x_mm is None) != (self.y_mm is None):
            raise ValueError("cue coordinates must be both present or both absent")
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class EventEvidence:
    event_id: str
    event_type: str
    status: str
    timestamp_ms: int | None
    actor_id: str | None
    target_id: str | None
    evidence_frame_ids: tuple[int, ...]
    confidence: float
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.event_id, "event ID")
        _text(self.event_type, "event type")
        if self.status not in {"observed", "inferred", "abstained", "unsupported"}:
            raise ValueError("event status is invalid")
        if not self.evidence_frame_ids or any(frame < 0 for frame in self.evidence_frame_ids):
            raise ValueError("event requires non-negative evidence frames")
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class TableBallEvidence:
    group: str
    x_mm: int
    y_mm: int
    confidence: float

    def __post_init__(self) -> None:
        _text(self.group, "table ball group")
        _confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ShotReplayEvidence:
    session_id: str
    shot_id: str
    complete: bool
    before_frame_id: int
    after_frame_id: int | None
    anchors: tuple[AnchorEvidence, ...]
    cue_samples: tuple[CueEvidence, ...]
    events: tuple[EventEvidence, ...]
    after_balls: tuple[TableBallEvidence, ...]

    def __post_init__(self) -> None:
        _text(self.session_id, "session ID")
        if not self.shot_id.startswith("shot-"):
            raise ValueError("shot ID must use the shot namespace")
        if self.before_frame_id < 0 or (
            self.after_frame_id is not None and self.after_frame_id < 0
        ):
            raise ValueError("table frame IDs must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    status: EvidenceStatus
    reason: str | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EventDecision:
    event_id: str
    event_type: str
    status: EvidenceStatus
    confidence: float
    reason: str | None
    evidence_frame_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TableDifference:
    before_frame_id: int
    after_frame_id: int | None
    before_ball_count: int
    after_ball_count: int
    net_ball_count: int | None


@dataclass(frozen=True, slots=True)
class HabitFeature:
    code: str
    value: float | str
    unit: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.code, "feature code")
        _text(self.unit, "feature unit")
        if not self.evidence_ids:
            raise ValueError("habit feature requires evidence lineage")


@dataclass(frozen=True, slots=True)
class ProfileIncrement:
    user_id: str
    session_id: str
    shot_id: str
    status: EvidenceStatus
    source: EvidenceSource
    features: tuple[HabitFeature, ...]

    def __post_init__(self) -> None:
        _text(self.user_id, "user ID")
        if self.status is not EvidenceStatus.ADMITTED and self.features:
            raise ValueError("non-admitted profile increment cannot contain features")


@dataclass(frozen=True, slots=True)
class ShotEvidenceProjection:
    source: EvidenceSource
    session_id: str
    shot_id: str
    status: EvidenceStatus
    reason: str | None
    cue_start: EvidenceDecision
    event_decisions: tuple[EventDecision, ...]
    table_difference: TableDifference
    profile_increment: ProfileIncrement
