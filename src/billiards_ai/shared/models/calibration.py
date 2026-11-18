"""Versioned table-calibration contracts shared by both deployment variants."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

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

from billiards_ai.shared.models.geometry import Point

Corner = tuple[float, float]
Matrix3x3 = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("datetime is outside the supported UTC range") from exc


def _finite(values: tuple[float, ...], *, label: str) -> tuple[float, ...]:
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain only finite values")
    return values


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationStatus(StrEnum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    INVALID = "INVALID"


class CalibrationLineage(StrictFrozenModel):
    """Minimal immutable authority attached to produced millimetre coordinates."""

    calibration_id: UUID4
    revision_id: UUID4
    revision_number: StrictInt = Field(ge=1)
    status: CalibrationStatus
    table_profile_id: NonEmptyStr
    table_length_mm: StrictInt
    table_width_mm: StrictInt
    table_configuration_fingerprint: Sha256

    @model_validator(mode="after")
    def table_plane_is_supported(self) -> CalibrationLineage:
        if (self.table_length_mm, self.table_width_mm) != (2540, 1270):
            raise ValueError("trajectory calibration requires the supported 2540:1270 table")
        return self

    @classmethod
    def from_record(cls, record: CalibrationRecord) -> CalibrationLineage:
        return cls(
            calibration_id=record.calibration_id,
            revision_id=record.revision_id,
            revision_number=record.revision_number,
            status=record.status,
            table_profile_id=record.table_profile.table_profile_id,
            table_length_mm=record.table_profile.length_mm,
            table_width_mm=record.table_profile.width_mm,
            table_configuration_fingerprint=record.table_profile.configuration_fingerprint,
        )


class DistortionModel(StrEnum):
    NONE = "NONE"
    OPENCV = "OPENCV"


class CalibrationResolution(StrictFrozenModel):
    width_px: StrictInt = Field(gt=0)
    height_px: StrictInt = Field(gt=0)


class CameraProfile(StrictFrozenModel):
    camera_profile_id: NonEmptyStr
    position_fingerprint: Sha256
    lens_fingerprint: Sha256


class TableProfile(StrictFrozenModel):
    table_profile_id: NonEmptyStr
    length_mm: StrictInt = Field(gt=0)
    width_mm: StrictInt = Field(gt=0)
    configuration_fingerprint: Sha256


class DistortionParameters(StrictFrozenModel):
    model: DistortionModel = DistortionModel.NONE
    verified: StrictBool = True
    camera_matrix: Matrix3x3 | None = None
    coefficients: tuple[float, ...] = ()

    @field_validator("camera_matrix")
    @classmethod
    def camera_matrix_is_finite(cls, value: Matrix3x3 | None) -> Matrix3x3 | None:
        if value is not None:
            _finite(tuple(item for row in value for item in row), label="camera matrix")
        return value

    @field_validator("coefficients")
    @classmethod
    def coefficients_are_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _finite(value, label="distortion coefficients")

    @model_validator(mode="after")
    def parameters_match_model(self) -> DistortionParameters:
        if self.model is DistortionModel.NONE:
            if self.camera_matrix is not None or self.coefficients:
                raise ValueError("negligible distortion cannot contain correction parameters")
            if not self.verified:
                raise ValueError("negligible distortion must be verified")
        elif self.camera_matrix is None or len(self.coefficients) not in {4, 5, 8, 12, 14}:
            raise ValueError(
                "OpenCV distortion requires a camera matrix and supported coefficients"
            )
        return self


class CalibrationReference(StrictFrozenModel):
    pixel: Corner
    table: Point

    @field_validator("pixel")
    @classmethod
    def pixel_is_finite(cls, value: Corner) -> Corner:
        _finite(value, label="reference pixel")
        return value


class CalibrationRecord(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    calibration_id: UUID4
    revision_id: UUID4
    revision_number: StrictInt = Field(ge=1)
    supersedes_revision_id: UUID4 | None = None
    device_id: NonEmptyStr
    camera_profile: CameraProfile
    resolution: CalibrationResolution
    table_profile: TableProfile
    distortion: DistortionParameters
    corners_px: tuple[Corner, ...] = Field(min_length=4, max_length=4)
    homography: Matrix3x3
    validation_references: tuple[CalibrationReference, ...]
    source_image_sha256: Sha256
    status: CalibrationStatus
    validated_at: datetime
    valid_until: datetime | None = None
    max_reference_error_mm: float = Field(ge=0)
    invalidation_reason: NonEmptyStr | None = None
    created_at: datetime

    _validated_at_utc = field_validator("validated_at")(_utc)
    _valid_until_utc = field_validator("valid_until")(
        lambda value: None if value is None else _utc(value)
    )
    _created_at_utc = field_validator("created_at")(_utc)

    @field_validator("corners_px")
    @classmethod
    def corners_are_finite(cls, value: tuple[Corner, ...]) -> tuple[Corner, ...]:
        _finite(tuple(item for corner in value for item in corner), label="calibration corners")
        return value

    @field_validator("homography")
    @classmethod
    def homography_is_finite(cls, value: Matrix3x3) -> Matrix3x3:
        _finite(tuple(item for row in value for item in row), label="homography")
        determinant = (
            value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
            - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
            + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
        )
        if abs(determinant) <= 1e-12:
            raise ValueError("homography must not be singular")
        return value

    @field_validator("max_reference_error_mm")
    @classmethod
    def error_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reference error must be finite")
        return value

    @model_validator(mode="after")
    def revision_is_consistent(self) -> CalibrationRecord:
        if self.revision_number == 1 and self.supersedes_revision_id is not None:
            raise ValueError("first revision cannot identify a superseded revision")
        if self.revision_number > 1 and self.supersedes_revision_id is None:
            raise ValueError("later revision must identify the superseded revision")
        if self.validated_at > self.created_at:
            raise ValueError("validation cannot occur after revision creation")
        if self.valid_until is not None and self.valid_until <= self.validated_at:
            raise ValueError("validity end must follow validation")
        if self.status is CalibrationStatus.ACTIVE:
            if self.invalidation_reason is not None:
                raise ValueError("active calibration cannot have an invalidation reason")
            if self.max_reference_error_mm > 20.0:
                raise ValueError("active calibration exceeds the 20 mm accuracy limit")
            if not self.validation_references:
                raise ValueError("active calibration requires independent reference evidence")
            if self.distortion.model is not DistortionModel.NONE and not self.distortion.verified:
                raise ValueError("active calibration requires verified distortion correction")
        elif self.invalidation_reason is None:
            raise ValueError("inactive calibration requires an invalidation reason")
        return self
