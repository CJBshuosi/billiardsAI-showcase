"""Frozen Story 2.6 contracts for deterministic shot segmentation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from billiards_ai.shared.constants import BALL_IDS
from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.geometry import Point


class SegmentationStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        values = dict(vars(copied))
        if copied.__pydantic_extra__:
            values.update(copied.__pydantic_extra__)
        return type(self).model_validate(values)


class SegmentationFailureReason(StrEnum):
    LONG_DETECTION_GAP = "LONG_DETECTION_GAP"
    STARTED_IN_MOTION = "STARTED_IN_MOTION"
    ENDED_IN_MOTION = "ENDED_IN_MOTION"
    MAX_SHOT_DURATION_EXCEEDED = "MAX_SHOT_DURATION_EXCEEDED"
    NON_INCREASING_TIMESTAMPS = "NON_INCREASING_TIMESTAMPS"


class BallPosition(SegmentationStrictFrozenModel):
    ball_id: int = Field(ge=min(BALL_IDS), le=max(BALL_IDS))
    position: Point


class TableSnapshot(SegmentationStrictFrozenModel):
    timestamp_ms: int = Field(ge=0)
    positions: tuple[BallPosition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def positions_are_sorted_and_unique(self) -> TableSnapshot:
        ball_ids = tuple(item.ball_id for item in self.positions)
        if ball_ids != tuple(sorted(set(ball_ids))):
            raise ValueError("snapshot ball IDs must be sorted and unique")
        return self

    def as_dict(self) -> dict[int, Point]:
        """Return a detached mapping suitable for the later Shot finalizer."""

        return {item.ball_id: item.position.model_copy(deep=True) for item in self.positions}


class TrajectorySample(SegmentationStrictFrozenModel):
    timestamp_ms: StrictInt = Field(ge=0)
    position: Point

    @model_validator(mode="after")
    def position_is_strict_on_table_mm(self) -> TrajectorySample:
        if type(self.position.x_mm) is not int or type(self.position.y_mm) is not int:
            raise ValueError("trajectory coordinates must be integer millimetres")
        if not self.position.is_on_table():
            raise ValueError("trajectory position must be on the table")
        return self


class ShotSegment(SegmentationStrictFrozenModel):
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    ball_positions_before: TableSnapshot
    ball_positions_after: TableSnapshot
    cue_ball_path: tuple[TrajectorySample, ...] = ()
    calibration: CalibrationLineage | None = None

    @model_validator(mode="after")
    def interval_and_snapshots_are_consistent(self) -> ShotSegment:
        if self.t_end_ms <= self.t_start_ms:
            raise ValueError("segment end must be after start")
        if self.ball_positions_before.timestamp_ms >= self.t_start_ms:
            raise ValueError("before snapshot must precede segment start")
        if self.ball_positions_after.timestamp_ms != self.t_end_ms:
            raise ValueError("after snapshot timestamp must equal segment end")
        if self.cue_ball_path:
            timestamps = tuple(sample.timestamp_ms for sample in self.cue_ball_path)
            if timestamps != tuple(sorted(set(timestamps))):
                raise ValueError("trajectory timestamps must be strictly increasing")
            if timestamps[0] != self.ball_positions_before.timestamp_ms:
                raise ValueError("trajectory must start at the reviewed before snapshot")
            before_cue = self.ball_positions_before.as_dict().get(0)
            if before_cue is None or self.cue_ball_path[0].position != before_cue:
                raise ValueError("trajectory anchor must equal the before snapshot cue ball")
            if any(timestamp < self.t_start_ms for timestamp in timestamps[1:]):
                raise ValueError("trajectory samples after the anchor must be inside the shot")
            if timestamps[-1] > self.t_end_ms:
                raise ValueError("trajectory cannot extend beyond the shot")
            positions = tuple(sample.position for sample in self.cue_ball_path)
            if any(left == right for left, right in zip(positions, positions[1:], strict=False)):
                raise ValueError("trajectory must remove adjacent duplicate positions")
        return self

    @property
    def duration_ms(self) -> int:
        return self.t_end_ms - self.t_start_ms


class SegmentationFailure(SegmentationStrictFrozenModel):
    reason: SegmentationFailureReason
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    missing_ball_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def failure_is_canonical(self) -> SegmentationFailure:
        if self.t_end_ms < self.t_start_ms:
            raise ValueError("failure end must not precede start")
        if self.missing_ball_ids != tuple(sorted(set(self.missing_ball_ids))):
            raise ValueError("missing ball IDs must be sorted and unique")
        if any(ball_id not in BALL_IDS for ball_id in self.missing_ball_ids):
            raise ValueError("missing ball IDs must be between 0 and 15")
        return self


class ShotSegmentationResult(SegmentationStrictFrozenModel):
    schema_version: str = "1.0"
    segments: tuple[ShotSegment, ...] = ()
    failures: tuple[SegmentationFailure, ...] = ()

    @model_validator(mode="after")
    def output_segments_are_ordered_and_nonoverlapping(self) -> ShotSegmentationResult:
        if self.schema_version != "1.0":
            raise ValueError("unsupported segmentation schema version")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if current.t_start_ms < previous.t_end_ms:
                raise ValueError("segments must be strictly increasing and non-overlapping")
        return self
