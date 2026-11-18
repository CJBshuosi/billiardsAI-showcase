"""Strict Story 6.2 contracts for consent-backed cloud session admission."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from .privacy import MaterialUsePurpose
from .table_binding import SafeDisplayLabel, Sha256Digest


def _aware_utc(value: object) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("datetime cannot be normalized to UTC") from exc


def _high_entropy_key(value: str) -> str:
    try:
        raw = base64.b64decode(value + "=", altchars=b"-_", validate=True)
        canonical = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("idempotency key must be canonical base64url") from exc
    if len(raw) != 32 or canonical != value:
        raise ValueError("idempotency key must contain 256 bits")
    return value


NonEmptyText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2_048)
]
NoticeText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
IdempotencyKey = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{43}$"),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ConsentDecision(StrEnum):
    GRANTED = "GRANTED"
    DECLINED = "DECLINED"
    WITHDRAWN = "WITHDRAWN"


class CloudSessionStatus(StrEnum):
    OPEN = "OPEN"
    ENDED_AWAITING_UPLOAD = "ENDED_AWAITING_UPLOAD"
    MEDIA_BOUND = "MEDIA_BOUND"
    CANCELLED = "CANCELLED"


class SessionAdmissionNextAction(StrEnum):
    WAITING_FOR_UPLOAD = "WAITING_FOR_UPLOAD"
    REVIEW_CONSENT = "REVIEW_CONSENT"


class SessionAdmissionErrorCode(StrEnum):
    IDENTITY_REQUIRED = "IDENTITY_REQUIRED"
    CONSENT_REQUEST_INVALID = "CONSENT_REQUEST_INVALID"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    CONSENT_POLICY_CHANGED = "CONSENT_POLICY_CHANGED"
    CONSENT_SCOPE_MISMATCH = "CONSENT_SCOPE_MISMATCH"
    REAL_VIDEO_GATE_NOT_APPROVED = "REAL_VIDEO_GATE_NOT_APPROVED"
    TABLE_BINDING_MISMATCH = "TABLE_BINDING_MISMATCH"
    TABLE_UNAVAILABLE = "TABLE_UNAVAILABLE"
    TABLE_CAPACITY_REACHED = "TABLE_CAPACITY_REACHED"
    USER_ALREADY_BOUND = "USER_ALREADY_BOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    CONSENT_ALREADY_WITHDRAWN = "CONSENT_ALREADY_WITHDRAWN"
    SESSION_ADMISSION_REPOSITORY_UNAVAILABLE = "SESSION_ADMISSION_REPOSITORY_UNAVAILABLE"
    SESSION_ROSTER_FROZEN = "SESSION_ROSTER_FROZEN"


class ConsentNotice(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    purpose: MaterialUsePurpose
    version: NonEmptyText
    content: NoticeText
    content_sha256: Sha256Digest
    scope_fingerprint: Sha256Digest
    gate_revision_id: UUID4
    canonicalization_schema_version: Literal["raw-utf8-v1"] = "raw-utf8-v1"
    collection_scope: NonEmptyText
    use_description: NonEmptyText
    retention_description: NonEmptyText
    access_description: NonEmptyText
    deletion_description: NonEmptyText
    operator_name: NonEmptyText
    operator_contact: NonEmptyText

    @model_validator(mode="after")
    def notice_is_verifiable(self) -> ConsentNotice:
        if self.purpose not in {MaterialUsePurpose.SERVICE_ANALYSIS, MaterialUsePurpose.TRAINING}:
            raise ValueError("only user-facing consent purposes have notices")
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise ValueError("notice content does not match canonical bytes digest")
        return self


class ConsentPolicyView(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: UUID4
    policy_id: Sha256Digest
    gate_revision_id: UUID4
    scope_fingerprint: Sha256Digest
    canonicalization_schema_version: Literal["raw-utf8-v1"] = "raw-utf8-v1"
    service_notice: ConsentNotice
    training_notice: ConsentNotice

    @model_validator(mode="after")
    def notices_match_policy(self) -> ConsentPolicyView:
        if self.service_notice.purpose is not MaterialUsePurpose.SERVICE_ANALYSIS:
            raise ValueError("service notice purpose mismatch")
        if self.training_notice.purpose is not MaterialUsePurpose.TRAINING:
            raise ValueError("training notice purpose mismatch")
        for notice in (self.service_notice, self.training_notice):
            if (
                notice.gate_revision_id != self.gate_revision_id
                or notice.scope_fingerprint != self.scope_fingerprint
                or notice.canonicalization_schema_version != self.canonicalization_schema_version
            ):
                raise ValueError("notice authority does not match policy")
        return self


class ConsentSubmissionRequest(_StrictModel):
    schema_version: Literal["1.0"]
    binding_id: UUID4
    service_decision: ConsentDecision
    training_decision: ConsentDecision
    policy_id: Sha256Digest
    gate_revision_id: UUID4
    service_notice_version: NonEmptyText
    training_notice_version: NonEmptyText
    idempotency_key: IdempotencyKey

    _key = field_validator("idempotency_key")(_high_entropy_key)

    @model_validator(mode="after")
    def decisions_are_a_valid_submission(self) -> ConsentSubmissionRequest:
        allowed = {ConsentDecision.GRANTED, ConsentDecision.DECLINED}
        if self.service_decision not in allowed or self.training_decision not in allowed:
            raise ValueError("submission accepts only explicit grant or decline")
        if (
            self.service_decision is ConsentDecision.DECLINED
            and self.training_decision is not ConsentDecision.DECLINED
        ):
            raise ValueError("training cannot be granted when service is declined")
        return self


class ConsentWithdrawalRequest(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    participant_id: UUID4
    purpose: MaterialUsePurpose
    idempotency_key: IdempotencyKey

    _key = field_validator("idempotency_key")(_high_entropy_key)

    @field_validator("purpose")
    @classmethod
    def purpose_is_user_consent(cls, value: MaterialUsePurpose) -> MaterialUsePurpose:
        if value not in {MaterialUsePurpose.SERVICE_ANALYSIS, MaterialUsePurpose.TRAINING}:
            raise ValueError("only user consent purposes may be withdrawn")
        return value


class TrainingConsentWithdrawalRequest(ConsentWithdrawalRequest):
    purpose: Literal[MaterialUsePurpose.TRAINING]


class ConsentDecisionRecord(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    consent_id: UUID4
    user_id: UUID4
    binding_id: UUID4
    session_id: UUID4 | None
    participant_id: UUID4 | None
    purpose: MaterialUsePurpose
    decision: ConsentDecision
    notice_version: NonEmptyText
    notice_sha256: Sha256Digest
    gate_revision_id: UUID4
    gate_scope_fingerprint: Sha256Digest
    decided_at: datetime
    withdrawn_at: datetime | None = None
    supersedes: UUID4 | None = None

    _decided = field_validator("decided_at", mode="before")(_aware_utc)
    _withdrawn = field_validator("withdrawn_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )

    @model_validator(mode="after")
    def lifecycle_is_consistent(self) -> ConsentDecisionRecord:
        linked = self.session_id is not None and self.participant_id is not None
        if (self.session_id is None) != (self.participant_id is None):
            raise ValueError("session and participant links must be both present or absent")
        if self.decision in {ConsentDecision.GRANTED, ConsentDecision.WITHDRAWN} and not linked:
            raise ValueError("grant and withdrawal records require session and participant")
        if self.decision is ConsentDecision.WITHDRAWN:
            if self.withdrawn_at is None or self.supersedes is None:
                raise ValueError("withdrawal requires time and superseded consent")
            if self.withdrawn_at != self.decided_at:
                raise ValueError("withdrawal decision and withdrawal times must match")
        elif self.withdrawn_at is not None or self.supersedes is not None:
            raise ValueError("only withdrawal records may supersede a decision")
        if self.purpose not in {MaterialUsePurpose.SERVICE_ANALYSIS, MaterialUsePurpose.TRAINING}:
            raise ValueError("records store only explicit user consent purposes")
        return self


class SessionIntervalRecord(_StrictModel):
    revision: int
    start_at: datetime
    end_at: datetime | None = None
    provenance: Literal["SERVER_MARKED", "STAFF_ATTESTED"]
    operator_id: UUID4
    recorded_at: datetime
    reason: NonEmptyText | None = None

    _start = field_validator("start_at", mode="before")(_aware_utc)
    _end = field_validator("end_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )
    _recorded = field_validator("recorded_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def interval_is_consistent(self):
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("interval revision must be positive")
        if self.end_at is not None and self.end_at <= self.start_at:
            raise ValueError("interval end must follow start")
        if self.provenance == "STAFF_ATTESTED" and not self.reason:
            raise ValueError("attested interval requires a reason")
        return self


class CloudSession(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: UUID4
    table_registration_id: UUID4
    status: CloudSessionStatus = CloudSessionStatus.OPEN
    created_at: datetime
    revision: int = 1
    roster_frozen_at: datetime | None = None
    interval_history: tuple[SessionIntervalRecord, ...] = ()
    media_asset_id: UUID4 | None = None
    media_content_sha256: Sha256Digest | None = None

    _created = field_validator("created_at", mode="before")(_aware_utc)
    _frozen = field_validator("roster_frozen_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )

    @model_validator(mode="after")
    def lifecycle_is_consistent(self):
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("session revision must be positive")
        if [item.revision for item in self.interval_history] != list(
            range(1, len(self.interval_history) + 1)
        ):
            raise ValueError("interval history must be append-only")
        if self.interval_history and self.roster_frozen_at is None:
            raise ValueError("capture interval requires frozen roster")
        if self.status is CloudSessionStatus.OPEN and any(
            item.end_at is not None for item in self.interval_history
        ):
            raise ValueError("open session cannot contain ended interval")
        if self.status is CloudSessionStatus.ENDED_AWAITING_UPLOAD and (
            not self.interval_history or self.interval_history[-1].end_at is None
        ):
            raise ValueError("ended session requires a complete interval")
        if self.status is CloudSessionStatus.MEDIA_BOUND and (
            self.media_asset_id is None or self.media_content_sha256 is None
        ):
            raise ValueError("media-bound session requires deterministic binding")
        if (self.media_asset_id is None) != (self.media_content_sha256 is None):
            raise ValueError("media binding is incomplete")
        return self


class SessionParticipant(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    participant_id: UUID4
    session_id: UUID4
    user_id: UUID4
    binding_id: UUID4
    service_consent_id: UUID4
    training_consent_id: UUID4
    joined_at: datetime

    _joined = field_validator("joined_at", mode="before")(_aware_utc)


class SessionAdmissionView(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: UUID4
    session_id: UUID4 | None
    participant_id: UUID4 | None
    venue_display_label: SafeDisplayLabel
    table_display_label: SafeDisplayLabel
    service_decision: ConsentDecision
    training_decision: ConsentDecision
    service_notice_version: NonEmptyText
    training_notice_version: NonEmptyText
    session_status: CloudSessionStatus | None
    created_at: datetime | None
    activated_at: datetime | None
    next_action: SessionAdmissionNextAction

    _created = field_validator("created_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )
    _activated = field_validator("activated_at", mode="before")(
        lambda value: None if value is None else _aware_utc(value)
    )

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> SessionAdmissionView:
        identifiers = (self.session_id, self.participant_id, self.session_status, self.created_at)
        if self.service_decision is ConsentDecision.DECLINED:
            if any(value is not None for value in identifiers) or self.activated_at is not None:
                raise ValueError("declined service cannot expose a session")
            if self.next_action is not SessionAdmissionNextAction.REVIEW_CONSENT:
                raise ValueError("declined service must return to consent review")
        elif self.service_decision is ConsentDecision.WITHDRAWN:
            if any(value is None for value in identifiers) or self.activated_at is None:
                raise ValueError("withdrawn service must retain admission audit links")
            if self.next_action is not SessionAdmissionNextAction.REVIEW_CONSENT:
                raise ValueError("withdrawn service must block further upload")
        else:
            if any(value is None for value in identifiers) or self.activated_at is None:
                raise ValueError("granted service requires an admitted participant")
            if self.next_action is not SessionAdmissionNextAction.WAITING_FOR_UPLOAD:
                raise ValueError("granted service must wait for upload")
        return self


class SessionAdmissionIdempotencyRecord(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    record_id: Sha256Digest
    operation: Literal["ADMIT", "WITHDRAW"]
    user_id: UUID4
    binding_id: UUID4
    participant_id: UUID4 | None = None
    purpose: MaterialUsePurpose | None = None
    key_digest: Sha256Digest
    payload_digest: Sha256Digest
    result_consent_ids: tuple[UUID4, ...]
    result_session_id: UUID4 | None = None
    result_participant_id: UUID4 | None = None
    created_at: datetime

    _created = field_validator("created_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def operation_shape_is_consistent(self) -> SessionAdmissionIdempotencyRecord:
        if self.operation == "ADMIT" and (
            self.participant_id is not None or self.purpose is not None
        ):
            raise ValueError("admission idempotency must not identify withdrawal inputs")
        if self.operation == "ADMIT" and (
            len(self.result_consent_ids) != 2
            or (self.result_session_id is None) != (self.result_participant_id is None)
        ):
            raise ValueError("admission idempotency result is incomplete")
        if self.operation == "WITHDRAW" and (
            self.participant_id is None or self.purpose is None or len(self.result_consent_ids) != 1
        ):
            raise ValueError("withdrawal idempotency requires its target and one result")
        if self.operation == "WITHDRAW" and (
            self.result_session_id is None or self.result_participant_id is None
        ):
            raise ValueError("withdrawal idempotency result is incomplete")
        return self


def canonical_payload_digest(model: BaseModel) -> str:
    """Digest a strict request without persisting or logging its raw idempotency key."""

    payload = model.model_dump(mode="json", exclude={"idempotency_key"})
    import json

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def idempotency_key_digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()
