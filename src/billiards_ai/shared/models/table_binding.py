"""Strict Story 6.1 contracts for table admission before service consent."""

from __future__ import annotations

import base64
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)


def _safe_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError("control characters are not allowed")
    return normalized


def _canonical_qr_token(value: str) -> str:
    try:
        raw = base64.b64decode(value + "=", altchars=b"-_", validate=True)
        canonical = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("QR token must be canonical base64url") from exc
    if len(raw) != 32 or canonical != value:
        raise ValueError("QR token must be canonical base64url")
    return value


def _aware_utc(value: object) -> datetime:
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise ValueError("datetime must be an ISO string or datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include a timezone")
        return value.astimezone(UTC)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("datetime cannot be normalized to UTC") from exc


SafeIdentifier = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128),
    AfterValidator(_safe_text),
]
SafeDisplayLabel = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128),
    AfterValidator(_safe_text),
]
CanonicalQrToken = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{43}$"),
    AfterValidator(_canonical_qr_token),
]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
PositiveStrictInt = Annotated[StrictInt, Field(ge=1)]


class _StrictTableBindingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class TableRegistrationStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class TableQrStatus(StrEnum):
    CURRENT = "CURRENT"
    ROTATED = "ROTATED"
    REVOKED = "REVOKED"


class TableBindingStatus(StrEnum):
    PENDING_CONSENT = "PENDING_CONSENT"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TableBindingNextAction(StrEnum):
    REVIEW_CONSENT = "REVIEW_CONSENT"


class TableCapacityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    ONE_ACTIVE = "ONE_ACTIVE"
    FULL = "FULL"


class TableBindingErrorCode(StrEnum):
    IDENTITY_REQUIRED = "IDENTITY_REQUIRED"
    QR_INVALID = "QR_INVALID"
    QR_VERSION_UNSUPPORTED = "QR_VERSION_UNSUPPORTED"
    QR_BINDING_INACTIVE = "QR_BINDING_INACTIVE"
    TABLE_BINDING_MISMATCH = "TABLE_BINDING_MISMATCH"
    TABLE_UNAVAILABLE = "TABLE_UNAVAILABLE"
    TABLE_CAPACITY_REACHED = "TABLE_CAPACITY_REACHED"
    USER_ALREADY_BOUND = "USER_ALREADY_BOUND"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    TABLE_BINDING_REPOSITORY_UNAVAILABLE = "TABLE_BINDING_REPOSITORY_UNAVAILABLE"


class JoinTableRequest(_StrictTableBindingModel):
    schema_version: Literal["1.0"]
    qr_token: CanonicalQrToken


class TableRegistration(_StrictTableBindingModel):
    schema_version: Literal["1.0"] = "1.0"
    table_registration_id: UUID4
    venue_id: SafeIdentifier
    table_id: SafeIdentifier
    device_id: SafeIdentifier
    venue_display_label: SafeDisplayLabel
    table_display_label: SafeDisplayLabel
    current_qr_version: PositiveStrictInt
    status: TableRegistrationStatus


class TableQrRegistration(_StrictTableBindingModel):
    schema_version: Literal["1.0"] = "1.0"
    qr_registration_id: UUID4
    token_digest: Sha256Digest
    table_registration_id: UUID4
    qr_version: PositiveStrictInt
    status: TableQrStatus


class TableBinding(_StrictTableBindingModel):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: UUID4
    user_id: UUID4
    table_registration_id: UUID4
    qr_registration_id: UUID4
    qr_version: PositiveStrictInt
    status: TableBindingStatus
    created_at: datetime
    activated_at: datetime | None = None
    released_at: datetime | None = None

    _created_at_utc = field_validator("created_at", mode="before")(_aware_utc)
    _activated_at_utc = field_validator("activated_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )
    _released_at_utc = field_validator("released_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )

    @model_validator(mode="after")
    def lifecycle_is_consistent(self) -> TableBinding:
        if self.status is TableBindingStatus.PENDING_CONSENT and (
            self.activated_at is not None or self.released_at is not None
        ):
            raise ValueError("pending binding cannot have an activation time")
        if self.status is TableBindingStatus.ACTIVE and (
            self.activated_at is None or self.released_at is not None
        ):
            raise ValueError("active binding requires an activation time")
        if self.status in {TableBindingStatus.COMPLETED, TableBindingStatus.CANCELLED} and (
            self.activated_at is None or self.released_at is None
        ):
            raise ValueError("released binding requires activation and release times")
        if self.activated_at is not None and self.activated_at < self.created_at:
            raise ValueError("binding cannot activate before creation")
        if self.released_at is not None and self.released_at < self.activated_at:
            raise ValueError("binding cannot release before activation")
        return self


class TableBindingView(_StrictTableBindingModel):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: UUID4
    venue_display_label: SafeDisplayLabel
    table_display_label: SafeDisplayLabel
    qr_version: PositiveStrictInt
    binding_status: TableBindingStatus
    capacity_status: TableCapacityStatus
    created_at: datetime
    next_action: Literal[TableBindingNextAction.REVIEW_CONSENT] = (
        TableBindingNextAction.REVIEW_CONSENT
    )

    _created_at_utc = field_validator("created_at", mode="before")(_aware_utc)
