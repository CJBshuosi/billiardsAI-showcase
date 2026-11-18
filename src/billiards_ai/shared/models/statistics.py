"""Frozen Story 3.1 contract for deterministic session statistics."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.quality import SessionQualitySummary
from billiards_ai.shared.models.shot import TargetZoneSource


class StatisticsStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SessionAggregationStatus(StrEnum):
    READY = "READY"
    NO_VALID_SHOTS = "NO_VALID_SHOTS"


class CushionCountFrequency(StatisticsStrictFrozenModel):
    cushion_count: int = Field(ge=0)
    shot_count: int = Field(gt=0)


class CueBallEndFrequency(StatisticsStrictFrozenModel):
    position: Point
    shot_count: int = Field(gt=0)


class TargetZoneDeviationSummary(StatisticsStrictFrozenModel):
    target_zone_source: TargetZoneSource
    shot_count: int = Field(gt=0)
    average_deviation_mm: float = Field(ge=0.0)

    @model_validator(mode="after")
    def source_supports_a_positional_verdict(self) -> TargetZoneDeviationSummary:
        if self.target_zone_source is TargetZoneSource.NONE:
            raise ValueError("target zone deviation cannot use the NONE source")
        return self


class SessionStatistics(StatisticsStrictFrozenModel):
    """Statistics plus the explicit observed-shot denominator for every rate or mean."""

    total_shot_count: int = Field(gt=0)
    valid_shot_count: int = Field(gt=0)
    session_quality: SessionQualitySummary

    potted_ball_count: int = Field(ge=0)
    successful_potting_shot_count: int = Field(ge=0)
    potting_shot_count: int = Field(ge=0)
    pot_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    average_cue_ball_travel_distance_mm: float | None = Field(default=None, ge=0.0)
    cue_ball_travel_shot_count: int = Field(ge=0)

    cushion_distribution: tuple[CushionCountFrequency, ...]
    cushion_count_shot_count: int = Field(ge=0)

    cue_ball_end_distribution: tuple[CueBallEndFrequency, ...]
    cue_ball_end_shot_count: int = Field(ge=0)

    target_zone_deviation_by_source: tuple[TargetZoneDeviationSummary, ...]

    @model_validator(mode="after")
    def counts_and_denominators_are_consistent(self) -> SessionStatistics:
        if self.valid_shot_count > self.total_shot_count:
            raise ValueError("valid shot count cannot exceed total shot count")
        if self.session_quality.shot_count != self.total_shot_count:
            raise ValueError("session quality shot count must match total shot count")
        if self.session_quality.abstained_shot_count != (
            self.total_shot_count - self.valid_shot_count
        ):
            raise ValueError("session quality abstention count must match valid shot count")
        if self.successful_potting_shot_count > self.potting_shot_count:
            raise ValueError("successful potting shot count cannot exceed its denominator")
        expected_pot_rate = (
            self.successful_potting_shot_count / self.potting_shot_count
            if self.potting_shot_count
            else None
        )
        if self.pot_rate != expected_pot_rate:
            raise ValueError("pot rate must match the explicit potting shot denominator")
        if (self.average_cue_ball_travel_distance_mm is None) != (
            self.cue_ball_travel_shot_count == 0
        ):
            raise ValueError("travel average presence must match its denominator")
        if sum(item.shot_count for item in self.cushion_distribution) != (
            self.cushion_count_shot_count
        ):
            raise ValueError("cushion distribution must match its denominator")
        if sum(item.shot_count for item in self.cue_ball_end_distribution) != (
            self.cue_ball_end_shot_count
        ):
            raise ValueError("cue-ball end distribution must match its denominator")
        sources = [item.target_zone_source for item in self.target_zone_deviation_by_source]
        if len(sources) != len(set(sources)):
            raise ValueError("target zone sources must be unique")
        return self


class SessionStatisticsResult(StatisticsStrictFrozenModel):
    status: SessionAggregationStatus
    deliverable: bool
    statistics: SessionStatistics | None

    @model_validator(mode="after")
    def status_matches_payload(self) -> SessionStatisticsResult:
        ready = self.status is SessionAggregationStatus.READY
        if self.deliverable is not ready:
            raise ValueError("only READY session statistics are deliverable")
        if (self.statistics is not None) is not ready:
            raise ValueError("only READY results carry session statistics")
        return self
