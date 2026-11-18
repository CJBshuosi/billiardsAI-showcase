"""Frozen and explicitly separated scoring contracts for Story 3.2."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScoringStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ScoreStatus(StrEnum):
    SCORED = "SCORED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class DimensionScore(ScoringStrictFrozenModel):
    score: int | None = Field(default=None, ge=0, le=100)
    status: ScoreStatus

    @model_validator(mode="after")
    def score_matches_status(self) -> DimensionScore:
        if self.status is ScoreStatus.SCORED and self.score is None:
            raise ValueError("SCORED requires a numeric score")
        if self.status is not ScoreStatus.SCORED and self.score is not None:
            raise ValueError("non-SCORED status requires a null score")
        return self


class SessionPerformanceScores(ScoringStrictFrozenModel):
    """Evidence-bounded scores for one free-play session; not a level assessment."""

    potting: DimensionScore
    positioning: DimensionScore
    power_control: DimensionScore
    rotation_application: DimensionScore


class LevelAssessmentScores(ScoringStrictFrozenModel):
    """Separate future fixed-scenario assessment shape; never produced by session scoring."""

    object_ball_position: DimensionScore
    cue_ball_cushion_contact: DimensionScore
    cue_ball_path: DimensionScore
    cue_ball_stop_zone: DimensionScore
