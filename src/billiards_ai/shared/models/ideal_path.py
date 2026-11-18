"""Strict versioned business contracts for ideal billiards paths."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models.calibration import CalibrationStatus

SafeMetadataToken = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$",
    ),
]


class IdealPathStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Revalidate copies so callers cannot inject unchecked third-party data."""

        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_model_values(copied))


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class IdealPathPocket(StrEnum):
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    TOP_LEFT = "TOP_LEFT"
    TOP_CENTER = "TOP_CENTER"
    TOP_RIGHT = "TOP_RIGHT"


class IdealPathIntent(StrEnum):
    DIRECT_POT = "DIRECT_POT"


class IdealPathStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"
    SIMULATION_FAILED = "SIMULATION_FAILED"


class IdealPathReason(StrEnum):
    INTENT_NOT_EXPLICIT = "intent_not_explicit"
    INTENT_NOT_REVIEWED = "intent_not_reviewed"
    CALIBRATION_NOT_ACTIVE = "calibration_not_active"
    TARGET_BALL_MISSING = "target_ball_missing"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    ADAPTER_FAILURE = "adapter_failure"


class IdealPathEventKind(StrEnum):
    CUE_STRIKE = "CUE_STRIKE"
    BALL_CONTACT = "BALL_CONTACT"
    CUSHION_CONTACT = "CUSHION_CONTACT"
    POCKETED = "POCKETED"
    STOP = "STOP"


class IdealPathPoint(IdealPathStrictFrozenModel):
    x_mm: StrictInt
    y_mm: StrictInt

    @model_validator(mode="after")
    def is_on_versioned_table_plane(self) -> IdealPathPoint:
        if not (0 <= self.x_mm <= 2540 and 0 <= self.y_mm <= 1270):
            raise ValueError("point must be on the versioned table plane")
        return self


class IdealPathBall(IdealPathStrictFrozenModel):
    ball_id: StrictInt = Field(ge=0, le=15)
    position: IdealPathPoint


class IdealPathProvenance(IdealPathStrictFrozenModel):
    calibration_id: UUID4
    calibration_revision_id: UUID4
    calibration_revision_number: StrictInt = Field(ge=1)
    table_profile_id: str = Field(min_length=1)
    table_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    cue_ball: IdealPathBall
    object_balls: tuple[IdealPathBall, ...]
    target_ball_id: StrictInt = Field(ge=1, le=15)

    @model_validator(mode="after")
    def starting_state_and_target_are_canonical(self) -> IdealPathProvenance:
        cue = IdealPathBall.model_validate(_stored_model_values(self.cue_ball))
        objects = tuple(
            IdealPathBall.model_validate(_stored_model_values(ball)) for ball in self.object_balls
        )
        if cue.ball_id != 0:
            raise ValueError("ideal provenance cue ball id must be 0")
        object_ids = tuple(ball.ball_id for ball in objects)
        if object_ids != tuple(sorted(set(object_ids))) or 0 in object_ids:
            raise ValueError("ideal provenance object balls must be sorted and unique")
        if self.target_ball_id not in object_ids:
            raise ValueError("ideal provenance target ball must be present")
        return self


class IdealPathTableConfiguration(IdealPathStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    configuration_version: Literal["chinese-eight-ball-9ft-v1"]
    length_mm: StrictInt
    width_mm: StrictInt
    ball_diameter_mm: StrictInt
    corner_pocket_mouth_mm: StrictInt
    side_pocket_mouth_mm: StrictInt
    cushion_nose_width_mm: StrictInt
    cushion_nose_height_mm: StrictInt = 36

    @model_validator(mode="after")
    def is_supported_chinese_eight_ball_table(self) -> IdealPathTableConfiguration:
        expected = {
            "length_mm": 2540,
            "width_mm": 1270,
            "ball_diameter_mm": 57,
            "corner_pocket_mouth_mm": 86,
            "side_pocket_mouth_mm": 92,
            "cushion_nose_width_mm": 40,
            "cushion_nose_height_mm": 36,
        }
        if any(getattr(self, field_name) != value for field_name, value in expected.items()):
            raise ValueError("table configuration is not the supported Chinese eight-ball table")
        return self


class IdealPathRequest(IdealPathStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    cue_ball: IdealPathBall
    object_balls: tuple[IdealPathBall, ...]
    target_ball_id: StrictInt | None = Field(default=None, ge=0, le=15)
    target_pocket: IdealPathPocket | None = None
    intent: IdealPathIntent | None = None
    target_ball_reviewed: StrictBool = False
    target_pocket_reviewed: StrictBool = False
    intent_reviewed: StrictBool = False
    calibration_status: CalibrationStatus
    table: IdealPathTableConfiguration
    provenance: IdealPathProvenance | None = None

    @field_validator("target_pocket", mode="before")
    @classmethod
    def parse_target_pocket(cls, value: object) -> object:
        return IdealPathPocket(value) if isinstance(value, str) else value

    @field_validator("intent", mode="before")
    @classmethod
    def parse_intent(cls, value: object) -> object:
        return IdealPathIntent(value) if isinstance(value, str) else value

    @field_validator("calibration_status", mode="before")
    @classmethod
    def parse_calibration_status(cls, value: object) -> object:
        return CalibrationStatus(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def balls_use_unique_valid_table_positions(self) -> IdealPathRequest:
        cue = IdealPathBall.model_validate(_stored_model_values(self.cue_ball))
        objects = tuple(
            IdealPathBall.model_validate(_stored_model_values(ball)) for ball in self.object_balls
        )
        if cue.ball_id != 0:
            raise ValueError("cue ball id must be 0")
        object_ids = [ball.ball_id for ball in objects]
        if 0 in object_ids or len(set(object_ids)) != len(object_ids):
            raise ValueError("object ball ids must be unique and cannot use cue ball id")

        radius_mm = self.table.ball_diameter_mm / 2
        all_balls = (cue, *objects)
        for ball in all_balls:
            if not (
                radius_mm <= ball.position.x_mm <= self.table.length_mm - radius_mm
                and radius_mm <= ball.position.y_mm <= self.table.width_mm - radius_mm
            ):
                raise ValueError("ball center must be within the playable table")
        for index, ball in enumerate(all_balls):
            for other in all_balls[index + 1 :]:
                dx = ball.position.x_mm - other.position.x_mm
                dy = ball.position.y_mm - other.position.y_mm
                if dx * dx + dy * dy < self.table.ball_diameter_mm**2:
                    raise ValueError("ball positions must not overlap")
        if self.provenance is not None:
            provenance = IdealPathProvenance.model_validate(_stored_model_values(self.provenance))
            if (
                provenance.cue_ball != cue
                or provenance.object_balls != objects
                or provenance.target_ball_id != self.target_ball_id
            ):
                raise ValueError("ideal provenance must match the requested starting state")
        return self


class IdealPathEvent(IdealPathStrictFrozenModel):
    sequence: StrictInt = Field(ge=1)
    kind: IdealPathEventKind
    ball_id: StrictInt = Field(ge=0, le=15)
    other_ball_id: StrictInt | None = Field(default=None, ge=0, le=15)
    pocket: IdealPathPocket | None = None
    position: IdealPathPoint

    @field_validator("kind", mode="before")
    @classmethod
    def parse_kind(cls, value: object) -> object:
        return IdealPathEventKind(value) if isinstance(value, str) else value

    @field_validator("pocket", mode="before")
    @classmethod
    def parse_pocket(cls, value: object) -> object:
        return IdealPathPocket(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def relationship_fields_match_event_kind(self) -> IdealPathEvent:
        if self.kind is IdealPathEventKind.BALL_CONTACT:
            if self.other_ball_id is None or self.other_ball_id == self.ball_id:
                raise ValueError("ball contact requires two distinct ball ids")
        elif self.other_ball_id is not None:
            raise ValueError("other ball id is only valid for ball contact")

        if self.kind is IdealPathEventKind.POCKETED:
            if self.pocket is None:
                raise ValueError("pocketed event requires a pocket")
        elif self.pocket is not None:
            raise ValueError("pocket is only valid for a pocketed event")
        return self


class IdealPathResult(IdealPathStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: IdealPathStatus
    reason: IdealPathReason | None
    adapter_version: SafeMetadataToken
    engine_name: SafeMetadataToken
    engine_version: SafeMetadataToken
    unit: Literal["mm"] = "mm"
    coordinate_system: Literal["LEFT_BOTTOM_POCKET_X_LONG_Y_SHORT"] = (
        "LEFT_BOTTOM_POCKET_X_LONG_Y_SHORT"
    )
    source_configuration_version: Literal["chinese-eight-ball-9ft-v1"]
    provenance: IdealPathProvenance | None = None
    events: tuple[IdealPathEvent, ...] = ()

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> object:
        return IdealPathStatus(value) if isinstance(value, str) else value

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> object:
        return IdealPathReason(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def status_and_events_are_consistent(self) -> IdealPathResult:
        events = tuple(
            IdealPathEvent.model_validate(_stored_model_values(event)) for event in self.events
        )
        if self.status is IdealPathStatus.SUCCEEDED:
            if self.reason is not None or not events:
                raise ValueError("successful ideal path requires events and no failure reason")
        elif self.reason is None or events or self.provenance is not None:
            raise ValueError("failed ideal path requires a reason and no events")
        if [event.sequence for event in events] != list(range(1, len(events) + 1)):
            raise ValueError("events must have contiguous sequence numbers")
        return self
