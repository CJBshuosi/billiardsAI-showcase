"""Project-owned contracts for geometry guidance and optional physics."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.table_state import BallKind


class PocketId(StrEnum):
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_MIDDLE = "bottom_middle"
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    TOP_MIDDLE = "top_middle"
    TOP_RIGHT = "top_right"


class GuidanceStatus(StrEnum):
    PLANNED = "planned"
    GEOMETRY_ONLY = "geometry_only"
    SIMULATED = "simulated"
    ABSTAINED = "abstained"
    UNSUPPORTED = "unsupported"


class PhysicsStatus(StrEnum):
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class PowerBand(StrEnum):
    LIGHT = "light"
    MEDIUM = "medium"
    FIRM = "firm"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannedShot(_FrozenContract):
    candidate_id: str = Field(min_length=1)
    target_track_id: str = Field(min_length=1)
    target_identity: str = Field(min_length=1)
    target_kind: BallKind
    pocket: PocketId
    status: GuidanceStatus
    ghost_ball_center: Point | None = None
    cue_path: tuple[Point, ...] = ()
    object_path: tuple[Point, ...] = ()
    cut_angle_deg: float | None = Field(default=None, ge=0, le=180)
    cue_travel_mm: float | None = Field(default=None, ge=0)
    object_travel_mm: float | None = Field(default=None, ge=0)
    evidence_confidence: float = Field(ge=0, le=1)
    score_components: dict[str, float] = Field(default_factory=dict)
    score: float = Field(default=0, ge=0, le=1)
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @field_validator("score_components")
    @classmethod
    def validate_score_components(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not 0 <= component <= 1 for component in value.values()):
            raise ValueError("score components must be between zero and one")
        return value

    @model_validator(mode="after")
    def validate_paths(self) -> PlannedShot:
        if self.status is GuidanceStatus.PLANNED:
            incomplete = (
                self.ghost_ball_center is None
                or len(self.cue_path) < 2
                or len(self.object_path) < 2
            )
            if incomplete:
                raise ValueError("planned shots require complete project-owned paths")
        if any(not ref for ref in self.evidence_refs):
            raise ValueError("evidence references must be non-empty")
        return self


class PhysicsResult(_FrozenContract):
    candidate_id: str = Field(min_length=1)
    status: PhysicsStatus
    backend_version: str = Field(min_length=1)
    cue_path: tuple[Point, ...] = ()
    object_paths: dict[str, tuple[Point, ...]] = Field(default_factory=dict)
    events: tuple[str, ...] = ()
    cue_ball_control_score: float = Field(default=0, ge=0, le=1)
    reason: str | None = None

    @model_validator(mode="after")
    def require_reason_for_non_success(self) -> PhysicsResult:
        if self.status is not PhysicsStatus.SIMULATED and not self.reason:
            raise ValueError("non-simulated physics result requires a reason")
        return self


class Recommendation(_FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    status: GuidanceStatus
    candidates: tuple[PlannedShot, ...]
    calibration_id: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    planner_config_hash: str = Field(min_length=1)
    backend_versions: tuple[str, ...] = ()
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def validate_recommendation(self) -> Recommendation:
        if len(self.candidates) > 3:
            raise ValueError("recommendations contain at most three candidates")
        if any(not ref for ref in self.evidence_refs):
            raise ValueError("evidence references must be non-empty")
        return self


class GuidanceScene(_FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    coordinate_space: Literal["table_mm"] = "table_mm"
    recommendation_id: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    routes: tuple[PlannedShot, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
