"""Raw-free reviewed metric evidence for ApprovedEvidenceBatch v2."""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Literal

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr
from billiards_ai.shared.models.shot import TargetZoneSource


class GameCode(StrEnum):
    CHINESE_EIGHT_BALL = "CHINESE_EIGHT_BALL"


class PocketCode(StrEnum):
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    TOP_LEFT = "TOP_LEFT"
    TOP_CENTER = "TOP_CENTER"
    TOP_RIGHT = "TOP_RIGHT"


class FieldSource(StrEnum):
    CV = "CV"
    OPEN_SOURCE_MODEL = "OPEN_SOURCE_MODEL"
    VLM = "VLM"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"


class EvidenceApprovalKind(StrEnum):
    HUMAN_REVIEW = "HUMAN_REVIEW"
    AUTOMATED_POLICY = "AUTOMATED_POLICY"


class PotAttemptStatus(StrEnum):
    ATTEMPTED = "ATTEMPTED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class PotResult(StrEnum):
    LEGAL_SUCCESS = "LEGAL_SUCCESS"
    MISS = "MISS"
    WRONG_BALL = "WRONG_BALL"
    FOUL_INVALIDATED = "FOUL_INVALIDATED"


class FoulResult(StrEnum):
    LEGAL = "LEGAL"
    FOUL = "FOUL"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class TargetZoneShape(StrEnum):
    RECTANGLE = "RECTANGLE"
    POLYGON = "POLYGON"


class TargetZoneEvaluationStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class TargetZoneResult(StrEnum):
    INSIDE_OR_TOLERATED = "INSIDE_OR_TOLERATED"
    OUTSIDE = "OUTSIDE"


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    return value


def _finite_coordinate(value: float) -> float:
    value = _finite(value)
    return 0.0 if value == 0.0 else value


class ReviewedFieldQuality(CoreContractModel):
    source: FieldSource
    confidence: float = Field(ge=0.0, le=1.0)

    _confidence_finite = field_validator("confidence")(_finite)

    @model_validator(mode="after")
    def unknown_source_has_no_confidence(self) -> ReviewedFieldQuality:
        if (self.source is FieldSource.UNKNOWN) != (self.confidence == 0.0):
            raise ValueError("UNKNOWN source requires exactly zero confidence")
        return self


class MetricFieldQuality(CoreContractModel):
    geometry: ReviewedFieldQuality
    intent: ReviewedFieldQuality
    pot_attempt: ReviewedFieldQuality
    pot_result: ReviewedFieldQuality
    foul_result: ReviewedFieldQuality
    target_zone: ReviewedFieldQuality
    cue_ball_end: ReviewedFieldQuality


class ReviewedCalibrationBound(CoreContractModel):
    calibration_id: UUID4
    calibration_revision_id: UUID4
    calibration_revision_number: StrictInt = Field(ge=1)

    @property
    def calibration_key(self) -> tuple[UUID4, UUID4, int]:
        return (
            self.calibration_id,
            self.calibration_revision_id,
            self.calibration_revision_number,
        )


class ReviewedPoint(ReviewedCalibrationBound):
    x_mm: float = Field(ge=0.0, le=2540.0)
    y_mm: float = Field(ge=0.0, le=1270.0)

    _x_finite = field_validator("x_mm")(_finite_coordinate)
    _y_finite = field_validator("y_mm")(_finite_coordinate)


class ReviewedTableGeometry(ReviewedCalibrationBound):
    table_length_mm: Literal[2540]
    table_width_mm: Literal[1270]
    ball_diameter_mm: float
    corner_pocket_mouth_mm: Literal[86]
    side_pocket_mouth_mm: Literal[92]

    _diameter_finite = field_validator("ball_diameter_mm")(_finite)

    @model_validator(mode="after")
    def approved_dimensions_are_exact(self) -> ReviewedTableGeometry:
        if self.ball_diameter_mm != 57.15:
            raise ValueError("R&D v1 requires the approved 57.15 mm ball diameter")
        return self


def _orientation(a: ReviewedPoint, b: ReviewedPoint, c: ReviewedPoint) -> float:
    return (b.x_mm - a.x_mm) * (c.y_mm - a.y_mm) - (b.y_mm - a.y_mm) * (c.x_mm - a.x_mm)


def _point_on_segment(point: ReviewedPoint, start: ReviewedPoint, end: ReviewedPoint) -> bool:
    return (
        _orientation(start, end, point) == 0.0
        and min(start.x_mm, end.x_mm) <= point.x_mm <= max(start.x_mm, end.x_mm)
        and min(start.y_mm, end.y_mm) <= point.y_mm <= max(start.y_mm, end.y_mm)
    )


def _segments_intersect(
    a: ReviewedPoint,
    b: ReviewedPoint,
    c: ReviewedPoint,
    d: ReviewedPoint,
) -> bool:
    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0:
        return True
    return (
        (orientations[0] == 0.0 and _point_on_segment(c, a, b))
        or (orientations[1] == 0.0 and _point_on_segment(d, a, b))
        or (orientations[2] == 0.0 and _point_on_segment(a, c, d))
        or (orientations[3] == 0.0 and _point_on_segment(b, c, d))
    )


def _point_segment_distance(point: ReviewedPoint, a: ReviewedPoint, b: ReviewedPoint) -> float:
    dx = b.x_mm - a.x_mm
    dy = b.y_mm - a.y_mm
    if dx == 0.0 and dy == 0.0:
        return math.hypot(point.x_mm - a.x_mm, point.y_mm - a.y_mm)
    ratio = max(
        0.0,
        min(
            1.0,
            ((point.x_mm - a.x_mm) * dx + (point.y_mm - a.y_mm) * dy) / (dx * dx + dy * dy),
        ),
    )
    return math.hypot(point.x_mm - (a.x_mm + ratio * dx), point.y_mm - (a.y_mm + ratio * dy))


class ReviewedTargetZone(ReviewedCalibrationBound):
    shape: TargetZoneShape
    minimum_x_mm: float | None = Field(default=None, ge=0.0, le=2540.0)
    minimum_y_mm: float | None = Field(default=None, ge=0.0, le=1270.0)
    maximum_x_mm: float | None = Field(default=None, ge=0.0, le=2540.0)
    maximum_y_mm: float | None = Field(default=None, ge=0.0, le=1270.0)
    vertices: tuple[ReviewedPoint, ...] = ()

    _bounds_finite = field_validator(
        "minimum_x_mm", "minimum_y_mm", "maximum_x_mm", "maximum_y_mm"
    )(lambda value: None if value is None else _finite_coordinate(value))

    @model_validator(mode="after")
    def shape_is_canonical_and_valid(self) -> ReviewedTargetZone:
        bounds = (
            self.minimum_x_mm,
            self.minimum_y_mm,
            self.maximum_x_mm,
            self.maximum_y_mm,
        )
        if self.shape is TargetZoneShape.RECTANGLE:
            if (
                any(value is None for value in bounds)
                or self.vertices
                or self.minimum_x_mm >= self.maximum_x_mm  # type: ignore[operator]
                or self.minimum_y_mm >= self.maximum_y_mm  # type: ignore[operator]
            ):
                raise ValueError("rectangle target zone requires positive canonical bounds")
            return self
        if any(value is not None for value in bounds) or not 3 <= len(self.vertices) <= 16:
            raise ValueError("polygon target zone requires 3-16 vertices and no rectangle bounds")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("polygon vertices must be unique")
        if any(vertex.calibration_key != self.calibration_key for vertex in self.vertices):
            raise ValueError("polygon vertices must share target-zone calibration lineage")
        if self.vertices[0] != min(self.vertices, key=lambda item: (item.x_mm, item.y_mm)):
            raise ValueError("polygon vertices must use canonical first vertex")
        edges = tuple(zip(self.vertices, self.vertices[1:] + self.vertices[:1], strict=True))
        for index, (a, b) in enumerate(edges):
            for other_index, (c, d) in enumerate(edges):
                if abs(index - other_index) in {0, 1, len(edges) - 1}:
                    continue
                if _segments_intersect(a, b, c, d):
                    raise ValueError("polygon must not self-intersect")
        signed_area = sum(left.x_mm * right.y_mm - right.x_mm * left.y_mm for left, right in edges)
        if signed_area >= 0:
            raise ValueError("polygon vertices must use canonical clockwise order")
        return self

    def error_mm(self, point: ReviewedPoint) -> float:
        if point.calibration_key != self.calibration_key:
            raise ValueError("target zone and point calibration lineage must match")
        if self.shape is TargetZoneShape.RECTANGLE:
            dx = max(self.minimum_x_mm - point.x_mm, 0.0, point.x_mm - self.maximum_x_mm)  # type: ignore[operator]
            dy = max(self.minimum_y_mm - point.y_mm, 0.0, point.y_mm - self.maximum_y_mm)  # type: ignore[operator]
            error = math.hypot(dx, dy)
            return float(Decimal(str(error)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        inside = False
        for left, right in zip(self.vertices, self.vertices[1:] + self.vertices[:1], strict=True):
            if _point_segment_distance(point, left, right) <= 1e-12:
                return 0.0
            if (left.y_mm > point.y_mm) != (right.y_mm > point.y_mm):
                intersection = (right.x_mm - left.x_mm) * (point.y_mm - left.y_mm) / (
                    right.y_mm - left.y_mm
                ) + left.x_mm
                if point.x_mm < intersection:
                    inside = not inside
        if inside:
            return 0.0
        error = min(
            _point_segment_distance(point, left, right)
            for left, right in zip(
                self.vertices, self.vertices[1:] + self.vertices[:1], strict=True
            )
        )
        return float(Decimal(str(error)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


class ApprovedMetricShotEvidence(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    shot_id: UUID4
    session_id: UUID4
    sequence: StrictInt = Field(ge=1)
    review_revision: StrictInt = Field(ge=1)
    game_code: GameCode
    table_geometry: ReviewedTableGeometry
    cue_ball_start_mm: ReviewedPoint | None
    object_ball_start_mm: ReviewedPoint | None
    intended_object_ball: StrictInt | None = Field(default=None, ge=1, le=15)
    intended_target_pocket: PocketCode | None
    pot_attempt_status: PotAttemptStatus
    pot_result: PotResult | None
    foul_result: FoulResult
    target_zone_source: TargetZoneSource
    target_zone: ReviewedTargetZone | None
    cue_ball_end_mm: ReviewedPoint | None
    target_zone_evaluation_status: TargetZoneEvaluationStatus
    target_zone_result: TargetZoneResult | None
    field_quality: MetricFieldQuality

    @model_validator(mode="after")
    def reviewed_semantics_are_consistent(self) -> ApprovedMetricShotEvidence:
        calibration_key = self.table_geometry.calibration_key
        points = (
            self.cue_ball_start_mm,
            self.object_ball_start_mm,
            self.cue_ball_end_mm,
        )
        if any(item is not None and item.calibration_key != calibration_key for item in points) or (
            self.target_zone is not None and self.target_zone.calibration_key != calibration_key
        ):
            raise ValueError("all reviewed geometry must share table calibration lineage")
        if (
            self.cue_ball_start_mm is not None
            and self.object_ball_start_mm is not None
            and self.cue_ball_start_mm == self.object_ball_start_mm
        ):
            raise ValueError("cue and object ball centres must differ")
        attempted = self.pot_attempt_status is PotAttemptStatus.ATTEMPTED
        if attempted != (self.pot_result is not None):
            raise ValueError("pot attempt and pot result must agree")
        if attempted and (self.intended_object_ball is None or self.intended_target_pocket is None):
            raise ValueError("pot attempt requires reviewed target ball and pocket")
        if self.pot_result is PotResult.LEGAL_SUCCESS and self.foul_result is not FoulResult.LEGAL:
            raise ValueError("legal pot success requires a legal foul result")
        if (
            self.pot_result is PotResult.FOUL_INVALIDATED
            and self.foul_result is not FoulResult.FOUL
        ):
            raise ValueError("foul-invalidated pot result requires a foul")
        if not attempted and (
            self.intended_object_ball is not None or self.intended_target_pocket is not None
        ):
            raise ValueError("non-attempt cannot claim pot intent")

        eligible = self.target_zone_evaluation_status is TargetZoneEvaluationStatus.ELIGIBLE
        if eligible:
            if (
                self.target_zone_source is TargetZoneSource.NONE
                or self.target_zone is None
                or self.cue_ball_end_mm is None
                or self.target_zone_result is None
            ):
                raise ValueError("eligible target zone requires reviewed geometry and result")
            expected = (
                TargetZoneResult.INSIDE_OR_TOLERATED
                if self.target_zone.error_mm(self.cue_ball_end_mm) <= 20.0
                else TargetZoneResult.OUTSIDE
            )
            if self.target_zone_result is not expected:
                raise ValueError("target zone result must match approved R&D v1 geometry")
        elif self.target_zone_result is not None:
            raise ValueError("non-eligible target zone cannot carry a result")
        if self.target_zone_evaluation_status is TargetZoneEvaluationStatus.NOT_APPLICABLE:
            if (
                self.target_zone_source is not TargetZoneSource.NONE
                or self.target_zone is not None
                or self.cue_ball_end_mm is not None
            ):
                raise ValueError("not-applicable target zone cannot claim geometry")
        elif not eligible and self.target_zone_source is TargetZoneSource.NONE:
            raise ValueError("unsupported or abstained target zone requires its approved source")
        return self


class ApprovedMetricEvidenceAuthorityRecord(CoreContractModel):
    """Trusted review authority response for v2 metric evidence admission."""

    schema_version: Literal["1.0"] = "1.0"
    approval_kind: EvidenceApprovalKind
    approval_policy_version: NonEmptyStr
    metric_shots: tuple[ApprovedMetricShotEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def approval_policy_is_frozen(self) -> ApprovedMetricEvidenceAuthorityRecord:
        expected = (
            "human-review-v1"
            if self.approval_kind is EvidenceApprovalKind.HUMAN_REVIEW
            else "automated-policy-rd-v1"
        )
        if self.approval_policy_version != expected:
            raise ValueError("authority approval metadata must use an approved immutable policy")
        return self
