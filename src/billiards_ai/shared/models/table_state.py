"""Canonical, calibrated table state used by local planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.shared.models.geometry import Point


class BallKind(StrEnum):
    CUE = "cue"
    EIGHT = "eight"
    SOLID = "solid"
    STRIPE = "stripe"
    UNKNOWN = "unknown"


class MotionStatus(StrEnum):
    STATIONARY = "stationary"
    MOVING = "moving"
    UNKNOWN = "unknown"


def kind_for_identity(identity: str) -> BallKind:
    if identity == "cue":
        return BallKind.CUE
    if identity == "8":
        return BallKind.EIGHT
    if identity.isdigit() and 1 <= int(identity) <= 7:
        return BallKind.SOLID
    if identity.isdigit() and 9 <= int(identity) <= 15:
        return BallKind.STRIPE
    return BallKind.UNKNOWN


class CanonicalBall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str = Field(min_length=1)
    identity: str = Field(pattern=r"^(cue|unknown|[1-9]|1[0-5])$")
    kind: BallKind
    identity_confidence: float = Field(ge=0, le=1)
    center: Point
    radius_mm: float = Field(gt=0)
    detection_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_identity_kind(self) -> CanonicalBall:
        expected = kind_for_identity(self.identity)
        if self.identity != "unknown" and self.kind is not expected:
            raise ValueError("identity does not match ball kind")
        return self


class CanonicalTableState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    frame_id: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    calibration_id: str = Field(min_length=1)
    motion_status: MotionStatus
    balls: tuple[CanonicalBall, ...]
    source_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_tracks(self) -> CanonicalTableState:
        track_ids = [ball.track_id for ball in self.balls]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique")
        if any(not ref for ref in self.source_refs):
            raise ValueError("source references must be non-empty")
        return self

    def require_plannable(self) -> None:
        if self.motion_status is not MotionStatus.STATIONARY:
            raise ValueError("planning requires a stationary table state")
        if sum(ball.kind is BallKind.CUE for ball in self.balls) != 1:
            raise ValueError("planning requires exactly one cue ball")

    def ball_by_track(self, track_id: str) -> CanonicalBall:
        for ball in self.balls:
            if ball.track_id == track_id:
                return ball
        raise KeyError(track_id)

    def cue_ball(self) -> CanonicalBall:
        cue_balls = [ball for ball in self.balls if ball.kind is BallKind.CUE]
        if len(cue_balls) != 1:
            raise ValueError("planning requires exactly one cue ball")
        return cue_balls[0]
