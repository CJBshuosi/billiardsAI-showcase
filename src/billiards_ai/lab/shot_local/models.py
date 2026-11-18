"""Immutable contracts for one-shot, fail-closed ball tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BallGroup(StrEnum):
    CUE = "cue"
    EIGHT = "eight"
    SOLID = "solid"
    STRIPE = "stripe"
    UNKNOWN = "unknown"


class LocalState(StrEnum):
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    LOST = "lost"
    REACQUIRED = "reacquired"
    AMBIGUOUS = "ambiguous"
    STOPPED = "stopped"
    POCKETED = "pocketed"


def _validate_confidence(value: float, *, name: str = "confidence") -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _validate_shot_id(shot_id: str) -> None:
    if not shot_id.startswith("shot-"):
        raise ValueError("shot_id must use the shot namespace")


def _validate_local_id(shot_id: str, local_id: str) -> None:
    _validate_shot_id(shot_id)
    if not local_id.startswith(f"{shot_id}-"):
        raise ValueError("local ID violates shot scope")


@dataclass(frozen=True, slots=True)
class CalibratedBallObservation:
    detection_id: str
    x_mm: int
    y_mm: int
    confidence: float
    group: BallGroup
    group_confidence: float
    bbox_xyxy_px: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.detection_id:
            raise ValueError("detection_id must not be empty")
        _validate_confidence(self.confidence)
        _validate_confidence(self.group_confidence, name="group confidence")
        if self.bbox_xyxy_px is not None:
            x1, y1, x2, y2 = self.bbox_xyxy_px
            if x2 <= x1 or y2 <= y1:
                raise ValueError("observation bounding box must have positive area")


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    frame_id: int
    timestamp_ms: int
    observations: tuple[CalibratedBallObservation, ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("frame and timestamp must be non-negative")
        detection_ids = [item.detection_id for item in self.observations]
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("frame detection IDs must be unique")

    def by_detection_id(self) -> dict[str, CalibratedBallObservation]:
        return {item.detection_id: item for item in self.observations}


@dataclass(frozen=True, slots=True)
class ShotWindow:
    shot_id: str
    start_frame_id: int
    end_frame_id: int
    before_frame_id: int
    after_frame_id: int | None
    start_ms: int
    end_ms: int
    complete: bool

    def __post_init__(self) -> None:
        _validate_shot_id(self.shot_id)
        if not 0 <= self.before_frame_id < self.start_frame_id <= self.end_frame_id:
            raise ValueError("shot window requires ordered frames")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("shot window requires ordered timestamps")
        if self.complete:
            if self.after_frame_id is None or self.after_frame_id < self.end_frame_id:
                raise ValueError("complete window requires an ordered after frame")
        elif self.after_frame_id is not None:
            raise ValueError("incomplete window cannot have an after frame")


@dataclass(frozen=True, slots=True)
class ShotAnchor:
    shot_id: str
    anchor_id: str
    source_detection_id: str
    group: BallGroup
    x_mm: int
    y_mm: int
    confidence: float

    def __post_init__(self) -> None:
        _validate_local_id(self.shot_id, self.anchor_id)
        if not self.source_detection_id:
            raise ValueError("anchor requires a source detection")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CueSample:
    shot_id: str
    local_id: str
    frame_id: int
    timestamp_ms: int
    state: LocalState
    x_mm: int | None
    y_mm: int | None
    confidence: float
    source_detection_id: str | None

    def __post_init__(self) -> None:
        _validate_local_id(self.shot_id, self.local_id)
        _validate_confidence(self.confidence)
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("sample frame and timestamp must be non-negative")
        if (self.x_mm is None) != (self.y_mm is None):
            raise ValueError("sample coordinates must be both present or both absent")
        if self.state is LocalState.LOST and (
            self.x_mm is not None or self.source_detection_id is not None
        ):
            raise ValueError("lost sample cannot carry coordinates or a detection")
        if self.state in {LocalState.CONFIRMED, LocalState.REACQUIRED, LocalState.STOPPED} and (
            self.x_mm is None or self.source_detection_id is None
        ):
            raise ValueError("observed sample requires current coordinates and detection")


@dataclass(frozen=True, slots=True)
class FragmentSample:
    shot_id: str
    fragment_id: str
    frame_id: int
    timestamp_ms: int
    state: LocalState
    x_mm: int
    y_mm: int
    confidence: float
    source_detection_id: str

    def __post_init__(self) -> None:
        _validate_local_id(self.shot_id, self.fragment_id)
        _validate_confidence(self.confidence)
        if self.frame_id < 0 or self.timestamp_ms < 0:
            raise ValueError("fragment frame and timestamp must be non-negative")
        if not self.source_detection_id:
            raise ValueError("fragment sample requires a source detection")


@dataclass(frozen=True, slots=True)
class MotionFragment:
    shot_id: str
    fragment_id: str
    group: BallGroup
    samples: tuple[FragmentSample, ...]
    terminal_state: LocalState

    def __post_init__(self) -> None:
        _validate_local_id(self.shot_id, self.fragment_id)
        if not self.samples:
            raise ValueError("motion fragment requires samples")
        if any(
            sample.shot_id != self.shot_id or sample.fragment_id != self.fragment_id
            for sample in self.samples
        ):
            raise ValueError("fragment sample violates shot scope")
        frame_ids = [sample.frame_id for sample in self.samples]
        if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("fragment samples require unique ordered frames")


@dataclass(frozen=True, slots=True)
class ShotLocalEvent:
    shot_id: str
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
        _validate_local_id(self.shot_id, self.event_id)
        for local_id in (self.actor_id, self.target_id):
            if local_id is not None:
                _validate_local_id(self.shot_id, local_id)
        if self.status not in {"observed", "inferred", "abstained", "unsupported"}:
            raise ValueError("unsupported event status")
        if not self.evidence_frame_ids:
            raise ValueError("event requires evidence frames")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ShotLocalResult:
    shot_id: str
    window: ShotWindow
    anchors: tuple[ShotAnchor, ...]
    cue_samples: tuple[CueSample, ...]
    fragments: tuple[MotionFragment, ...]
    events: tuple[ShotLocalEvent, ...]

    def __post_init__(self) -> None:
        _validate_shot_id(self.shot_id)
        if self.window.shot_id != self.shot_id:
            raise ValueError("result window violates shot scope")
        members = (*self.anchors, *self.cue_samples, *self.fragments, *self.events)
        if any(member.shot_id != self.shot_id for member in members):
            raise ValueError("result member violates shot scope")
