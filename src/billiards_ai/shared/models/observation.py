"""Frozen objective-observation contracts at the AD-13e boundary.

The contract contains only facts derivable from a Story 2.6 ``ShotSegment``.
Candidate lists and evidence remain inside ``raw_observation`` for review;
the scalar verdict fields explicitly remain unknown when endpoint snapshots
cannot support a conclusion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.shared.constants import BALL_IDS, CUE_BALL
from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.segmentation import BallPosition
from billiards_ai.shared.models.shot import TargetZoneSource


class ObservationStrictFrozenModel(BaseModel):
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


class ObservationConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ObservationSource(StrEnum):
    SEGMENT_BEFORE_SNAPSHOT = "SEGMENT_BEFORE_SNAPSHOT"
    SEGMENT_AFTER_SNAPSHOT = "SEGMENT_AFTER_SNAPSHOT"
    SEGMENT_SNAPSHOT_DELTA = "SEGMENT_SNAPSHOT_DELTA"
    ENDPOINT_DISPLACEMENT = "ENDPOINT_DISPLACEMENT"
    SEGMENT_CUE_BALL_PATH = "SEGMENT_CUE_BALL_PATH"
    UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS = "UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS"


class UnsupportedReason(StrEnum):
    FIRST_CONTACT_NOT_OBSERVABLE = "FIRST_CONTACT_NOT_OBSERVABLE"
    NO_OBJECT_BALL_MOVEMENT_EVIDENCE = "NO_OBJECT_BALL_MOVEMENT_EVIDENCE"
    CUSHION_CONTACT_NOT_OBSERVABLE = "CUSHION_CONTACT_NOT_OBSERVABLE"
    FOUL_NOT_OBSERVABLE = "FOUL_NOT_OBSERVABLE"
    CUE_BALL_POTTED = "CUE_BALL_POTTED"
    CUE_BALL_NOT_OBSERVED = "CUE_BALL_NOT_OBSERVED"
    TRAJECTORY_NOT_OBSERVED = "TRAJECTORY_NOT_OBSERVED"
    TRAJECTORY_INSUFFICIENT_POINTS = "TRAJECTORY_INSUFFICIENT_POINTS"


class ObservationField[ObservationValue](ObservationStrictFrozenModel):
    """A value together with its field-level provenance and support status."""

    value: ObservationValue
    confidence: ObservationConfidence
    source: ObservationSource
    unsupported_reason: UnsupportedReason | None = None

    @model_validator(mode="after")
    def support_status_is_honest(self) -> ObservationField[ObservationValue]:
        if self.value is None and self.unsupported_reason is None:
            raise ValueError("an unknown value requires an unsupported reason")
        if self.value is not None and self.unsupported_reason is not None:
            raise ValueError("a supported value cannot carry an unsupported reason")
        if self.value is None and self.confidence is not ObservationConfidence.UNKNOWN:
            raise ValueError("an unknown value must have UNKNOWN confidence")
        return self


class BallMovement(ObservationStrictFrozenModel):
    ball_id: int = Field(ge=min(BALL_IDS), le=max(BALL_IDS))
    distance_mm: int = Field(ge=0)


class StruckBallCandidate(ObservationStrictFrozenModel):
    ball_id: int = Field(ge=1, le=max(BALL_IDS))
    confidence: ObservationConfidence
    source: ObservationSource
    unsupported_reason: UnsupportedReason


class FoulEvidenceKind(StrEnum):
    CUE_BALL_POTTED = "CUE_BALL_POTTED"


class FoulEvidence(ObservationStrictFrozenModel):
    kind: FoulEvidenceKind
    supports_foul: bool
    confidence: ObservationConfidence
    source: ObservationSource
    unsupported_reason: UnsupportedReason | None = None


class RawObservation(ObservationStrictFrozenModel):
    """Review-only candidates and evidence; never a business-logic input."""

    struck_ball_candidates: tuple[StruckBallCandidate, ...] = ()
    foul_evidence: tuple[FoulEvidence, ...] = ()

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> RawObservation:
        candidate_ids = tuple(item.ball_id for item in self.struck_ball_candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("struck-ball candidate IDs must be sorted and unique")
        evidence_kinds = tuple(item.kind for item in self.foul_evidence)
        if len(evidence_kinds) != len(set(evidence_kinds)):
            raise ValueError("foul evidence kinds must be unique")
        return self


class ObjectiveShotObservation(ObservationStrictFrozenModel):
    """Objective facts and honest unknowns derived from one shot segment."""

    schema_version: str = "1.0"
    t_start_ms: int = Field(ge=0)
    t_end_ms: int = Field(ge=0)
    ball_positions_before: ObservationField[tuple[BallPosition, ...]]
    ball_positions_after: ObservationField[tuple[BallPosition, ...]]
    potted_balls: ObservationField[tuple[int, ...]]
    cue_ball_end: ObservationField[Point | None]
    ball_movements: ObservationField[tuple[BallMovement, ...]]
    travel_distance_mm: ObservationField[int | None]
    trajectory: ObservationField[tuple[Point, ...] | None] = ObservationField(
        value=None,
        confidence=ObservationConfidence.UNKNOWN,
        source=ObservationSource.UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
        unsupported_reason=UnsupportedReason.TRAJECTORY_NOT_OBSERVED,
    )
    trajectory_calibration: CalibrationLineage | None = None
    cushion_count: ObservationField[int | None]
    struck_ball: ObservationField[int | None]
    is_foul: ObservationField[bool | None]
    target_zone_source: TargetZoneSource = TargetZoneSource.NONE
    deviation_mm: None = None
    raw_observation: RawObservation

    @model_validator(mode="after")
    def objective_facts_are_consistent(self) -> ObjectiveShotObservation:
        if self.schema_version != "1.0":
            raise ValueError("unsupported observation schema version")
        if self.t_end_ms <= self.t_start_ms:
            raise ValueError("observation end must be after start")
        if self.target_zone_source is not TargetZoneSource.NONE:
            raise ValueError("objective observations have no preset target zone")

        before_ids = tuple(item.ball_id for item in self.ball_positions_before.value)
        after_ids = tuple(item.ball_id for item in self.ball_positions_after.value)
        if before_ids != tuple(sorted(set(before_ids))):
            raise ValueError("before ball IDs must be sorted and unique")
        if after_ids != tuple(sorted(set(after_ids))):
            raise ValueError("after ball IDs must be sorted and unique")
        if not set(after_ids).issubset(before_ids):
            raise ValueError("after ball IDs must be a subset of before ball IDs")

        potted_ids = self.potted_balls.value
        if potted_ids != tuple(sorted(set(potted_ids))):
            raise ValueError("potted ball IDs must be sorted and unique")
        if potted_ids != tuple(ball_id for ball_id in before_ids if ball_id not in after_ids):
            raise ValueError("potted ball IDs must exactly explain missing after keys")

        movement_ids = tuple(item.ball_id for item in self.ball_movements.value)
        if movement_ids != after_ids:
            raise ValueError("movement IDs must match the shared before/after keys")

        before_by_id = {item.ball_id: item.position for item in self.ball_positions_before.value}
        after_by_id = {item.ball_id: item.position for item in self.ball_positions_after.value}
        for movement in self.ball_movements.value:
            expected_distance = before_by_id[movement.ball_id].distance_to(
                after_by_id[movement.ball_id]
            )
            if movement.distance_mm != expected_distance:
                raise ValueError("movement distance must match endpoint snapshots")
        cue_end = after_by_id.get(CUE_BALL)
        if self.cue_ball_end.value != cue_end:
            raise ValueError("cue-ball endpoint must match the after snapshot")
        if self.travel_distance_mm.value is None and cue_end is not None:
            raise ValueError("cue-ball displacement is required when both endpoints exist")
        if cue_end is not None:
            cue_distance = next(
                movement.distance_mm
                for movement in self.ball_movements.value
                if movement.ball_id == CUE_BALL
            )
            if self.travel_distance_mm.value != cue_distance:
                raise ValueError("cue-ball displacement must match endpoint snapshots")
        if self.trajectory.value is not None:
            for point in self.trajectory.value:
                if type(point.x_mm) is not int or type(point.y_mm) is not int:
                    raise ValueError("trajectory coordinates must be integer millimetres")
                if not point.is_on_table():
                    raise ValueError("trajectory points must stay on the table")
            if any(
                left == right
                for left, right in zip(
                    self.trajectory.value,
                    self.trajectory.value[1:],
                    strict=False,
                )
            ):
                raise ValueError("trajectory must remove adjacent duplicate points")
        return self

    @property
    def struck_ball_candidates(self) -> tuple[StruckBallCandidate, ...]:
        return self.raw_observation.struck_ball_candidates

    @property
    def foul_evidence(self) -> tuple[FoulEvidence, ...]:
        return self.raw_observation.foul_evidence
