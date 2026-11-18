"""Fail-closed consumer contracts for final-v2 published-plan admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, final
from uuid import UUID

from billiards_ai.edge.coach_runtime.state_machine import VerifiedPlanAdmission
from billiards_ai.shared.plan_publication.contracts import (
    CapabilityManifest,
    DecodedPublishedPlanBundle,
    DecodedPublishedPlanBundleV3,
    PublicVerificationKey,
)


class PlanAdmissionFailureCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    MALFORMED_BUNDLE = "MALFORMED_BUNDLE"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    UNSUPPORTED_SIGNATURE_ALGORITHM = "UNSUPPORTED_SIGNATURE_ALGORITHM"
    CONTENT_HASH_MISMATCH = "CONTENT_HASH_MISMATCH"
    INCOMPLETE_DRILL_DEFINITIONS = "INCOMPLETE_DRILL_DEFINITIONS"
    INVALID_CAPABILITY_MANIFEST = "INVALID_CAPABILITY_MANIFEST"
    INVALID_VALIDITY_WINDOW = "INVALID_VALIDITY_WINDOW"
    NOT_YET_VALID = "NOT_YET_VALID"
    EXPIRED = "EXPIRED"
    KEY_NOT_TRUSTED = "KEY_NOT_TRUSTED"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    UNAUTHORIZED = "UNAUTHORIZED"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    ROLLBACK_REJECTED = "ROLLBACK_REJECTED"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"


class AntiRollbackDecision(StrEnum):
    ACCEPTED = "ACCEPTED"
    REPLAYED = "REPLAYED"
    REJECTED = "REJECTED"


@final
@dataclass(frozen=True, slots=True)
class UntrustedPlanBundle:
    """Opaque envelope bytes; its repr never exposes signed content."""

    envelope_bytes: bytes = field(repr=False)


@final
@dataclass(frozen=True, slots=True)
class AuthorizationClaim:
    authorization_id: UUID
    authorization_version: int
    user_id: UUID
    session_id: UUID
    target_device_id: str
    plan_id: UUID
    bundle_id: UUID


@final
@dataclass(frozen=True, slots=True)
class AntiRollbackClaim:
    bundle_id: UUID
    plan_id: UUID
    plan_version: int
    user_id: UUID
    session_id: UUID
    target_device_id: str
    authorization_id: UUID
    authorization_version: int
    publication_sequence: int
    bundle_content_sha256: str


@final
@dataclass(frozen=True, slots=True, repr=False)
class PlanAdmissionFailure:
    code: PlanAdmissionFailureCode
    stage: str

    def __post_init__(self) -> None:
        if type(self.code) is not PlanAdmissionFailureCode:
            raise TypeError("code must be PlanAdmissionFailureCode")
        if type(self.stage) is not str or not self.stage.strip():
            raise TypeError("stage must be a non-empty string")

    def __repr__(self) -> str:
        return "PlanAdmissionFailure(<redacted>)"


@final
@dataclass(frozen=True, slots=True, repr=False)
class PlanAdmissionResult:
    admission: VerifiedPlanAdmission | None = None
    failure: PlanAdmissionFailure | None = None

    def __post_init__(self) -> None:
        if (self.admission is None) == (self.failure is None):
            raise ValueError("result must contain exactly one admission or failure")
        if self.admission is not None and type(self.admission) is not VerifiedPlanAdmission:
            raise TypeError("admission must be VerifiedPlanAdmission")
        if self.failure is not None and type(self.failure) is not PlanAdmissionFailure:
            raise TypeError("failure must be PlanAdmissionFailure")

    @property
    def accepted(self) -> bool:
        return self.admission is not None

    def __repr__(self) -> str:
        branch = "accepted" if self.accepted else "rejected"
        return f"PlanAdmissionResult({branch})"


class ClockPort(Protocol):
    def now(self) -> datetime: ...


class PublishedPlanBundleCodecPort(Protocol):
    def decode(
        self, envelope_bytes: bytes
    ) -> DecodedPublishedPlanBundleV3 | DecodedPublishedPlanBundle: ...


class TrustedKeyringPort(Protocol):
    def resolve(
        self,
        *,
        algorithm: str,
        key_id: str,
        key_version: int,
        purpose: str,
        at: datetime,
    ) -> PublicVerificationKey | None: ...


class SignatureVerifierPort(Protocol):
    def verify(
        self,
        key: PublicVerificationKey,
        message: bytes,
        signature: bytes,
        *,
        at: datetime,
    ) -> None: ...


class PlanAuthorizationPort(Protocol):
    def authorize(self, claim: AuthorizationClaim) -> bool: ...


class CapabilityAuthorityPort(Protocol):
    def compatible(self, *, target_device_id: str, manifest: CapabilityManifest) -> bool: ...


class AntiRollbackPort(Protocol):
    def accept(self, claim: AntiRollbackClaim) -> AntiRollbackDecision: ...


__all__ = [
    "AntiRollbackClaim",
    "AntiRollbackDecision",
    "AntiRollbackPort",
    "AuthorizationClaim",
    "CapabilityAuthorityPort",
    "ClockPort",
    "PlanAdmissionFailure",
    "PlanAdmissionFailureCode",
    "PlanAdmissionResult",
    "PlanAuthorizationPort",
    "PublishedPlanBundleCodecPort",
    "SignatureVerifierPort",
    "TrustedKeyringPort",
    "UntrustedPlanBundle",
]
