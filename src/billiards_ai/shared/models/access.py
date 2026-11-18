"""Strict Phase 0 media-access grants, approvals, and audit contracts."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    AfterValidator,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

from .privacy import MaterialUsePurpose, Sha256, StrictFrozenModel
from .retention import MediaAssetKind


def _normalize_access_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("control characters are not allowed")
    return normalized


AccessText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_normalize_access_text),
]


def _access_utc(value: datetime) -> datetime:
    try:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("datetime must include a timezone")
        return value.astimezone(UTC)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("datetime cannot be normalized to UTC") from exc


class AccessAction(StrEnum):
    CONTROLLED_REVIEW_VIEW = "CONTROLLED_REVIEW_VIEW"
    EXPORT_OR_DOWNLOAD = "EXPORT_OR_DOWNLOAD"
    THIRD_PARTY_KEYFRAME_TRANSFER = "THIRD_PARTY_KEYFRAME_TRANSFER"


class AccessDecisionResult(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class AccessActorRole(StrEnum):
    REVIEWER = "REVIEWER"
    ACCESS_ADMIN = "ACCESS_ADMIN"
    SYSTEM = "SYSTEM"


class AccessReasonCode(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    IDENTITY_UNAVAILABLE = "IDENTITY_UNAVAILABLE"
    ROLE_NOT_ALLOWED = "ROLE_NOT_ALLOWED"
    GRANT_NOT_FOUND = "GRANT_NOT_FOUND"
    GRANT_REVOKED = "GRANT_REVOKED"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    INVALID_REASON = "INVALID_REASON"
    ACTION_DENIED = "ACTION_DENIED"
    MEDIA_UNAVAILABLE = "MEDIA_UNAVAILABLE"
    GATE_DENIED = "GATE_DENIED"
    THIRD_PARTY_APPROVAL_REQUIRED = "THIRD_PARTY_APPROVAL_REQUIRED"
    THIRD_PARTY_APPROVAL_MISMATCH = "THIRD_PARTY_APPROVAL_MISMATCH"
    COST_LIMIT_EXCEEDED = "COST_LIMIT_EXCEEDED"
    AUDIT_UNAVAILABLE = "AUDIT_UNAVAILABLE"
    REPOSITORY_UNAVAILABLE = "REPOSITORY_UNAVAILABLE"


class ReviewerIdentity(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    principal_id: AccessText
    roles: tuple[AccessActorRole, ...]
    credential_id: AccessText

    @field_validator("roles")
    @classmethod
    def roles_are_unique_and_non_empty(
        cls, value: tuple[AccessActorRole, ...]
    ) -> tuple[AccessActorRole, ...]:
        if not value:
            raise ValueError("at least one access role is required")
        return tuple(sorted(set(value), key=lambda role: role.value))


class SessionAccessGrant(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    grant_id: UUID4
    reviewer_id: AccessText
    session_id: UUID4
    reason: AccessText
    granted_by: AccessText
    granted_at: datetime
    revoked_at: datetime | None = None
    revocation_reason: AccessText | None = None

    _granted_at_utc = field_validator("granted_at")(_access_utc)
    _revoked_at_utc = field_validator("revoked_at")(
        lambda value: None if value is None else _access_utc(value)
    )

    @model_validator(mode="after")
    def revocation_is_consistent(self) -> SessionAccessGrant:
        if (self.revoked_at is None) != (self.revocation_reason is None):
            raise ValueError("revocation time and reason must be set together")
        if self.revoked_at is not None and self.revoked_at < self.granted_at:
            raise ValueError("grant cannot be revoked before it is created")
        return self

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class ThirdPartyProcessingApproval(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    approval_id: UUID4
    provider_id: AccessText
    endpoint_region: AccessText
    data_scope: Literal[MediaAssetKind.KEYFRAME]
    purpose: MaterialUsePurpose
    cross_border: StrictBool
    cross_border_assessment: AccessText
    max_cost_cny: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    approved_by: AccessText
    approved_at: datetime
    approval_evidence_sha256: Sha256
    revoked_at: datetime | None = None
    revocation_reason: AccessText | None = None

    _approved_at_utc = field_validator("approved_at")(_access_utc)
    _revoked_at_utc = field_validator("revoked_at")(
        lambda value: None if value is None else _access_utc(value)
    )

    @model_validator(mode="after")
    def revocation_is_consistent(self) -> ThirdPartyProcessingApproval:
        if (self.revoked_at is None) != (self.revocation_reason is None):
            raise ValueError("revocation time and reason must be set together")
        if self.revoked_at is not None and self.revoked_at < self.approved_at:
            raise ValueError("approval cannot be revoked before it is created")
        return self

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class AccessAuditEvent(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID4
    sequence: int = Field(ge=1)
    actor_id: AccessText
    actor_role: AccessActorRole
    occurred_at: datetime
    session_id: UUID4
    action: AccessAction
    reason: AccessText
    result: AccessDecisionResult
    reason_code: AccessReasonCode
    security_event: StrictBool

    _occurred_at_utc = field_validator("occurred_at")(_access_utc)

    @model_validator(mode="after")
    def result_is_consistent(self) -> AccessAuditEvent:
        if self.result is AccessDecisionResult.ALLOW:
            if self.reason_code is not AccessReasonCode.AUTHORIZED or self.security_event:
                raise ValueError("allowed access requires AUTHORIZED and is not a security event")
        elif self.reason_code is AccessReasonCode.AUTHORIZED or not self.security_event:
            raise ValueError("denied access requires a denial reason and security event")
        return self


class AccessDecision(StrictFrozenModel):
    event_id: UUID4
    session_id: UUID4
    action: AccessAction
    result: AccessDecisionResult
    reason_code: AccessReasonCode
