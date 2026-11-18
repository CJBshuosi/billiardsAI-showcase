"""Frozen field and session quality contracts for final shot conclusions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QualityStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FinalFieldSource(StrEnum):
    CV = "CV"
    OPEN_SOURCE_MODEL = "OPEN_SOURCE_MODEL"
    VLM = "VLM"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


class FieldQuality(QualityStrictFrozenModel):
    source: FinalFieldSource
    confidence: float = Field(ge=0.0, le=1.0)


class ShotFieldQuality(QualityStrictFrozenModel):
    ball_positions_before: FieldQuality
    ball_positions_after: FieldQuality
    cue_ball_end: FieldQuality
    struck_ball: FieldQuality
    potted_balls: FieldQuality
    cushion_count: FieldQuality
    travel_distance_mm: FieldQuality
    is_foul: FieldQuality
    trajectory: FieldQuality = FieldQuality(
        source=FinalFieldSource.UNKNOWN,
        confidence=0.0,
    )

    def values(self) -> tuple[FieldQuality, ...]:
        return tuple(
            getattr(self, name) for name in type(self).model_fields if name != "trajectory"
        )


class ShotAnalysisStatus(StrEnum):
    PENDING = "pending"
    ANALYZED = "analyzed"
    OUT_OF_CURRENT_ANALYSIS_SCOPE = "超出当前分析范围"


class SessionQualityStatus(StrEnum):
    SUFFICIENT_COVERAGE = "SUFFICIENT_COVERAGE"
    LOW_COVERAGE = "LOW_COVERAGE"


class SessionQualitySummary(QualityStrictFrozenModel):
    shot_count: int = Field(ge=0)
    abstained_shot_count: int = Field(ge=0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    session_quality_status: SessionQualityStatus

    @model_validator(mode="after")
    def counts_rate_and_status_are_consistent(self) -> SessionQualitySummary:
        if self.abstained_shot_count > self.shot_count:
            raise ValueError("abstained shot count cannot exceed shot count")
        expected_rate = self.abstained_shot_count / self.shot_count if self.shot_count else 0.0
        if abs(self.abstention_rate - expected_rate) > 1e-12:
            raise ValueError("abstention rate must match the shot counts")
        expected_status = (
            SessionQualityStatus.LOW_COVERAGE
            if expected_rate > 0.5
            else SessionQualityStatus.SUFFICIENT_COVERAGE
        )
        if self.session_quality_status is not expected_status:
            raise ValueError("session quality status must match abstention rate")
        return self
