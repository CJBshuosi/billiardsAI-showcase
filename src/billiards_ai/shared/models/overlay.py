"""Shared, table-coordinate-only primitives for realtime coaching overlays."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM
from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr


class CoachingOutcomeStatus(StrEnum):
    INFORMATIONAL = "INFORMATIONAL"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ABSTAINED = "ABSTAINED"
    UNSUPPORTED = "UNSUPPORTED"


class CoachingDeliveryStatus(StrEnum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class OverlayCoordinateSpace(StrEnum):
    TABLE_MM = "TABLE_MM"


class OverlayPrimitiveKind(StrEnum):
    POINT = "POINT"
    ZONE = "ZONE"


class OverlaySemanticRole(StrEnum):
    ACTUAL_CUE_BALL_END = "ACTUAL_CUE_BALL_END"
    STATUS = "STATUS"
    TARGET_ZONE = "TARGET_ZONE"


class OverlayLineStyle(StrEnum):
    SOLID = "SOLID"
    DASHED = "DASHED"
    DOTTED = "DOTTED"


class TablePointMmV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    x_mm: StrictInt
    y_mm: StrictInt

    @model_validator(mode="after")
    def point_is_on_authoritative_table(self) -> TablePointMmV1:
        if not (0 <= self.x_mm <= TABLE_LENGTH_MM and 0 <= self.y_mm <= TABLE_WIDTH_MM):
            raise ValueError("table point is outside authoritative millimetre bounds")
        return self


class TableZoneMmV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    minimum_x_mm: StrictInt
    maximum_x_mm: StrictInt
    minimum_y_mm: StrictInt
    maximum_y_mm: StrictInt

    @model_validator(mode="after")
    def zone_is_ordered_and_on_authoritative_table(self) -> TableZoneMmV1:
        if self.minimum_x_mm > self.maximum_x_mm or self.minimum_y_mm > self.maximum_y_mm:
            raise ValueError("table zone bounds must be ordered")
        if not (
            0 <= self.minimum_x_mm <= TABLE_LENGTH_MM
            and 0 <= self.maximum_x_mm <= TABLE_LENGTH_MM
            and 0 <= self.minimum_y_mm <= TABLE_WIDTH_MM
            and 0 <= self.maximum_y_mm <= TABLE_WIDTH_MM
        ):
            raise ValueError("table zone is outside authoritative millimetre bounds")
        return self


class OverlayPrimitiveV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: OverlayPrimitiveKind
    semantic_role: OverlaySemanticRole
    label_code: NonEmptyStr
    line_style: OverlayLineStyle
    point: TablePointMmV1 | None = None
    zone: TableZoneMmV1 | None = None

    @model_validator(mode="after")
    def kind_has_exact_geometry(self) -> OverlayPrimitiveV1:
        if self.kind is OverlayPrimitiveKind.POINT and (
            self.point is None or self.zone is not None
        ):
            raise ValueError("POINT overlay requires only point geometry")
        if self.kind is OverlayPrimitiveKind.ZONE and (self.zone is None or self.point is not None):
            raise ValueError("ZONE overlay requires only zone geometry")
        expected_kind = {
            OverlaySemanticRole.ACTUAL_CUE_BALL_END: OverlayPrimitiveKind.POINT,
            OverlaySemanticRole.TARGET_ZONE: OverlayPrimitiveKind.ZONE,
        }.get(self.semantic_role)
        if expected_kind is None:
            raise ValueError("STATUS has no authoritative geometry in overlay v1")
        if self.kind is not expected_kind:
            raise ValueError("overlay semantic role does not match geometry kind")
        return self


class OverlayPayloadV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    coordinate_space: Literal[OverlayCoordinateSpace.TABLE_MM] = OverlayCoordinateSpace.TABLE_MM
    outcome_status: CoachingOutcomeStatus
    delivery_status: CoachingDeliveryStatus
    primitives: tuple[OverlayPrimitiveV1, ...] = Field(default=())

    @model_validator(mode="after")
    def roles_are_unique_and_canonical(self) -> OverlayPayloadV1:
        roles = tuple(item.semantic_role.value for item in self.primitives)
        if len(roles) != len(set(roles)):
            raise ValueError("overlay semantic roles must be unique")
        if roles != tuple(sorted(roles)):
            raise ValueError("overlay primitives must use canonical semantic-role order")
        return self


__all__ = [
    "CoachingDeliveryStatus",
    "CoachingOutcomeStatus",
    "OverlayCoordinateSpace",
    "OverlayLineStyle",
    "OverlayPayloadV1",
    "OverlayPrimitiveKind",
    "OverlayPrimitiveV1",
    "OverlaySemanticRole",
    "TablePointMmV1",
    "TableZoneMmV1",
]
