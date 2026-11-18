"""Strict Story 6.3 staff capture and upload contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from .session_consent import CloudSessionStatus, IdempotencyKey, _high_entropy_key


def _aware_utc(value: object) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be aware UTC")
    return value.astimezone(UTC)


NonEmptyText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2048)
]
SafeLabel = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128)
]
Sha256Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
PositiveStrictInt = Annotated[StrictInt, Field(ge=1)]
PositiveStrictFloat = Annotated[float, Field(gt=0, strict=True)]


class StrictCaptureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class SessionCaptureStatus(StrEnum):
    RECORDING = "RECORDING"


class CaptureIntervalProvenance(StrEnum):
    SERVER_MARKED = "SERVER_MARKED"
    STAFF_ATTESTED = "STAFF_ATTESTED"


class UploadAttemptStatus(StrEnum):
    STAGED = "STAGED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class SessionCaptureNextAction(StrEnum):
    MARK_START = "MARK_START"
    MARK_END = "MARK_END"
    PREPARE_UPLOAD = "PREPARE_UPLOAD"
    RESUME_UPLOAD = "RESUME_UPLOAD"
    NONE = "NONE"


class SessionCaptureErrorCode(StrEnum):
    STAFF_IDENTITY_REQUIRED = "STAFF_IDENTITY_REQUIRED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    SESSION_NOT_ACTIVE = "SESSION_NOT_ACTIVE"
    SESSION_ROSTER_FROZEN = "SESSION_ROSTER_FROZEN"
    SESSION_ALREADY_STARTED = "SESSION_ALREADY_STARTED"
    SESSION_NOT_STARTED = "SESSION_NOT_STARTED"
    SESSION_ALREADY_ENDED = "SESSION_ALREADY_ENDED"
    INVALID_TIME_RANGE = "INVALID_TIME_RANGE"
    INTERVAL_OVERLAP = "INTERVAL_OVERLAP"
    BACKFILL_WINDOW_EXPIRED = "BACKFILL_WINDOW_EXPIRED"
    REVISION_CONFLICT = "REVISION_CONFLICT"
    SERVICE_CONSENT_NOT_CURRENT = "SERVICE_CONSENT_NOT_CURRENT"
    REAL_VIDEO_GATE_NOT_APPROVED = "REAL_VIDEO_GATE_NOT_APPROVED"
    ANONYMIZATION_OR_CONSENT_REQUIRED = "ANONYMIZATION_OR_CONSENT_REQUIRED"
    UPLOAD_NOT_READY = "UPLOAD_NOT_READY"
    MEDIA_FORMAT_UNSUPPORTED = "MEDIA_FORMAT_UNSUPPORTED"
    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    MEDIA_INSPECTION_FAILED = "MEDIA_INSPECTION_FAILED"
    MEDIA_AUDIO_PRESENT = "MEDIA_AUDIO_PRESENT"
    CLIP_INTERVAL_MISMATCH = "CLIP_INTERVAL_MISMATCH"
    DUPLICATE_UPLOAD = "DUPLICATE_UPLOAD"
    SESSION_MEDIA_ALREADY_BOUND = "SESSION_MEDIA_ALREADY_BOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    REPOSITORY_UNAVAILABLE = "REPOSITORY_UNAVAILABLE"
    UPLOAD_COMMIT_STATE_UNKNOWN = "UPLOAD_COMMIT_STATE_UNKNOWN"


class CaptureOperationRequest(StrictCaptureModel):
    schema_version: Literal["1.0"]
    session_id: UUID4
    idempotency_key: IdempotencyKey
    _key = field_validator("idempotency_key")(_high_entropy_key)


class CorrectIntervalRequest(CaptureOperationRequest):
    expected_interval_revision: PositiveStrictInt
    start_at: datetime
    end_at: datetime
    reason: NonEmptyText
    _start = field_validator("start_at", mode="before")(_aware_utc)
    _end = field_validator("end_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_at <= self.start_at:
            raise ValueError("end must be after start")
        return self


class UploadPrepareRequest(CaptureOperationRequest):
    pass


class UploadFinalizeRequest(CaptureOperationRequest):
    attempt_id: UUID4
    expected_session_revision: PositiveStrictInt


class CaptureIntervalRevision(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    interval_id: UUID4
    revision: PositiveStrictInt
    start_at: datetime
    end_at: datetime
    provenance: CaptureIntervalProvenance
    recorded_by: UUID4
    recorded_at: datetime
    reason: NonEmptyText | None = None
    supersedes_revision: PositiveStrictInt | None = None
    _start = field_validator("start_at", mode="before")(_aware_utc)
    _end = field_validator("end_at", mode="before")(_aware_utc)
    _recorded = field_validator("recorded_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def consistent(self):
        if self.end_at <= self.start_at:
            raise ValueError("end must be after start")
        if self.provenance is CaptureIntervalProvenance.STAFF_ATTESTED and not self.reason:
            raise ValueError("staff attestation requires reason")
        if self.revision == 1 and self.supersedes_revision is not None:
            raise ValueError("first interval cannot supersede")
        if self.revision > 1 and self.supersedes_revision != self.revision - 1:
            raise ValueError("interval revision chain is broken")
        return self


class SessionCaptureLimits(StrictCaptureModel):
    allowed_containers: tuple[NonEmptyText, ...]
    allowed_video_codecs: tuple[NonEmptyText, ...]
    max_upload_bytes: PositiveStrictInt
    max_duration_seconds: PositiveStrictFloat
    probe_timeout_seconds: PositiveStrictFloat
    interval_tolerance_seconds: PositiveStrictFloat

    @model_validator(mode="after")
    def nonempty_formats(self):
        if not self.allowed_containers or not self.allowed_video_codecs:
            raise ValueError("allowed media formats are required")
        return self


class StaffSessionView(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: UUID4
    venue_display_label: SafeLabel
    table_display_label: SafeLabel
    participant_count: Annotated[StrictInt, Field(ge=1, le=2)]
    session_status: CloudSessionStatus
    capture_status: SessionCaptureStatus | None
    interval: CaptureIntervalRevision | None
    upload_status: UploadAttemptStatus | None
    next_action: SessionCaptureNextAction


class StagedMediaEvidence(StrictCaptureModel):
    attempt_id: UUID4
    asset_id: UUID4
    content_sha256: Sha256Digest
    size_bytes: PositiveStrictInt
    container: NonEmptyText
    video_codec: NonEmptyText
    duration_seconds: PositiveStrictFloat
    audio_present: bool
    storage_locator: NonEmptyText


class CaptureIdempotencyRecord(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    record_id: Sha256Digest
    operation: Literal["START", "END", "CORRECT", "CANCEL", "PREPARE", "FINALIZE"]
    staff_id: UUID4
    session_id: UUID4
    key_digest: Sha256Digest
    payload_digest: Sha256Digest
    result_session_revision: Annotated[StrictInt, Field(ge=0)]
    result_interval_revision: Annotated[StrictInt, Field(ge=0)] | None
    result_status: CloudSessionStatus
    created_at: datetime
    _created = field_validator("created_at", mode="before")(_aware_utc)


class CaptureAuditEvent(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID4
    sequence: PositiveStrictInt
    event_type: SafeLabel
    operator_id: UUID4 | Literal["SYSTEM_RECONCILER"]
    session_id: UUID4
    table_registration_id: UUID4
    occurred_at: datetime
    result: Literal["SUCCESS", "DENIED", "FAILED"]
    error_code: SessionCaptureErrorCode | None = None
    idempotency_digest: Sha256Digest | None = None
    reason: NonEmptyText | None = None
    attempt_id: UUID4 | None = None
    asset_id: UUID4 | None = None
    _occurred = field_validator("occurred_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def result_is_complete(self):
        if self.result == "SUCCESS" and self.error_code is not None:
            raise ValueError("successful audit event cannot contain an error")
        if self.result != "SUCCESS" and self.error_code is None:
            raise ValueError("non-success audit event requires a frozen error code")
        return self


class UploadAttemptRecord(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    attempt_id: UUID4
    session_id: UUID4
    staff_id: UUID4
    idempotency_digest: Sha256Digest
    status: UploadAttemptStatus
    expected_session_revision: PositiveStrictInt
    interval_revision: PositiveStrictInt
    created_at: datetime
    content_sha256: Sha256Digest | None = None
    asset_id: UUID4 | None = None
    size_bytes: PositiveStrictInt | None = None
    storage_locator: NonEmptyText | None = None
    gate_revision_id: UUID4 | None = None
    subject_user_ids: tuple[NonEmptyText, ...] = ()
    cleanup_required: bool = False
    object_staged: bool = False
    supersedes: UUID4 | None = None
    _created = field_validator("created_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def staged_facts_are_complete(self):
        facts = (self.content_sha256, self.asset_id, self.size_bytes, self.storage_locator)
        if self.object_staged and any(item is None for item in facts):
            raise ValueError("staged object requires complete media facts")
        if not self.object_staged and any(item is not None for item in facts):
            raise ValueError("attempt without object cannot claim media facts")
        if self.status is UploadAttemptStatus.COMPLETED and not self.object_staged:
            raise ValueError("completed attempt requires a staged object")
        if self.object_staged and (
            self.gate_revision_id is None
            or not 1 <= len(self.subject_user_ids) <= 2
            or tuple(sorted(set(self.subject_user_ids))) != self.subject_user_ids
        ):
            raise ValueError("staged object requires canonical authority evidence")
        return self


class SessionMediaBindingRecord(StrictCaptureModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: UUID4
    attempt_id: UUID4
    asset_id: UUID4
    content_sha256: Sha256Digest
    storage_locator: NonEmptyText
    subject_user_ids: tuple[NonEmptyText, ...]
    interval_revision: PositiveStrictInt
    accepted_at: datetime
    _accepted = field_validator("accepted_at", mode="before")(_aware_utc)

    @model_validator(mode="after")
    def subjects_are_canonical(self):
        if not 1 <= len(self.subject_user_ids) <= 2:
            raise ValueError("media binding requires one or two subjects")
        if tuple(sorted(set(self.subject_user_ids))) != self.subject_user_ids:
            raise ValueError("media binding subjects must be unique and sorted")
        return self
