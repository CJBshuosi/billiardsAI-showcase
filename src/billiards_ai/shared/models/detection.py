"""Typed, reviewable per-frame ball observations at the AD-13e boundary."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.constants import BALL_IDS, CONFIDENCE_HIGH
from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.geometry import Point

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DetectionStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BallColor(StrEnum):
    WHITE = "WHITE"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    RED = "RED"
    PURPLE = "PURPLE"
    ORANGE = "ORANGE"
    GREEN = "GREEN"
    MAROON = "MAROON"
    BLACK = "BLACK"
    UNKNOWN = "UNKNOWN"


class DetectionSource(StrEnum):
    CLASSIC_CV_CIRCLE_COLOR = "CLASSIC_CV_CIRCLE_COLOR"


class FrameQuality(StrEnum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"


class RejectionReason(StrEnum):
    OCCLUDED = "OCCLUDED"
    OUTSIDE_TABLE = "OUTSIDE_TABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class BallCandidate(DetectionStrictFrozenModel):
    ball_id: int = Field(ge=min(BALL_IDS), le=max(BALL_IDS))
    color: BallColor
    confidence: float = Field(ge=0.0, le=1.0)


class PixelObservation(DetectionStrictFrozenModel):
    """Implementation-specific pixels retained only for review and debugging."""

    center_px: tuple[float, float]
    radius_px: float = Field(gt=0.0)
    circularity: float = Field(ge=0.0, le=1.0)
    color: BallColor

    @field_validator("center_px")
    @classmethod
    def center_is_finite(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("pixel center must be finite")
        return value

    @field_validator("radius_px")
    @classmethod
    def radius_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pixel radius must be finite")
        return value


class FrameDetection(DetectionStrictFrozenModel):
    candidates: tuple[BallCandidate, ...] = Field(min_length=1)
    position: Point
    confidence: float = Field(ge=0.0, le=1.0)
    source: DetectionSource
    raw_observation: PixelObservation

    @field_validator("candidates")
    @classmethod
    def candidates_are_unique_and_ranked(
        cls, value: tuple[BallCandidate, ...]
    ) -> tuple[BallCandidate, ...]:
        ball_ids = tuple(candidate.ball_id for candidate in value)
        if len(ball_ids) != len(set(ball_ids)):
            raise ValueError("ball candidate IDs must be unique")
        confidences = tuple(candidate.confidence for candidate in value)
        if confidences != tuple(sorted(confidences, reverse=True)):
            raise ValueError("ball candidates must be ranked by confidence")
        return value

    @field_validator("position")
    @classmethod
    def position_is_on_table(cls, value: Point) -> Point:
        if not value.is_on_table():
            raise ValueError("detection position must be on the table")
        return value


class RejectedDetection(DetectionStrictFrozenModel):
    candidates: tuple[BallCandidate, ...]
    reason: RejectionReason
    source: DetectionSource
    raw_observation: PixelObservation
    position: None = None


class FrameDetections(DetectionStrictFrozenModel):
    schema_version: str = "1.0"
    timestamp_ms: int = Field(ge=0)
    detections: tuple[FrameDetection, ...]
    missing_ball_ids: tuple[int, ...]
    frame_quality: FrameQuality
    detector_version: NonEmptyStr
    calibration: CalibrationLineage | None = None
    rejected: tuple[RejectedDetection, ...] = ()

    @field_validator("schema_version")
    @classmethod
    def schema_is_supported(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("unsupported detection schema version")
        return value

    @field_validator("missing_ball_ids")
    @classmethod
    def missing_ids_are_canonical(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(ball_id not in BALL_IDS for ball_id in value):
            raise ValueError("missing ball IDs must be between 0 and 15")
        if value != tuple(sorted(set(value))):
            raise ValueError("missing ball IDs must be unique and sorted")
        return value

    @model_validator(mode="after")
    def missing_ids_describe_located_positions(self) -> FrameDetections:
        """Keep identity hypotheses distinct from confirmed table locations.

        Ambiguous colour candidates intentionally remain missing: emitting them
        as located would let segmentation silently lose a ball track.
        """

        if set(self.missing_ball_ids).intersection(self.high_confidence_positions()):
            raise ValueError("a confirmed ball position cannot also be marked missing")
        return self

    @property
    def raw_observation_count(self) -> int:
        return len(self.detections) + len(self.rejected)

    def high_confidence_positions(
        self, *, minimum_confidence: float = CONFIDENCE_HIGH
    ) -> dict[int, Point]:
        """Return only unambiguous positions without mutating the source contract."""

        eligible = [
            detection
            for detection in self.detections
            if detection.confidence >= minimum_confidence
            and len(detection.candidates) == 1
            and detection.candidates[0].confidence >= minimum_confidence
        ]
        counts: dict[int, int] = {}
        for detection in eligible:
            ball_id = detection.candidates[0].ball_id
            counts[ball_id] = counts.get(ball_id, 0) + 1
        result = {
            detection.candidates[0].ball_id: detection.position.model_copy(deep=True)
            for detection in eligible
            if counts[detection.candidates[0].ball_id] == 1
        }
        return result

    def candidate_ball_ids(self) -> frozenset[int]:
        """Return every identity hypothesis, including unresolved candidates."""

        return frozenset(
            candidate.ball_id for detection in self.detections for candidate in detection.candidates
        )
