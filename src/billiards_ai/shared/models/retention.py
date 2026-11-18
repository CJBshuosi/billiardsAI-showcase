"""Versioned media-retention and deletion contracts (AD-19, Story 1.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import PurePosixPath
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

from billiards_ai.shared.constants import CLIP_RETENTION_DAYS, RETENTION_CLOCK_SKEW_MINUTES

from .privacy import Sha256, StrictFrozenModel, _utc


def _reject_control_characters(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("control characters are not allowed")
    return value


NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_reject_control_characters),
]


class MediaAssetKind(StrEnum):
    VIDEO = "VIDEO"
    FRAME = "FRAME"
    KEYFRAME = "KEYFRAME"
    SCREENSHOT = "SCREENSHOT"
    TRACEABLE_DERIVATIVE = "TRACEABLE_DERIVATIVE"


class MediaDeletionStatus(StrEnum):
    PENDING_BINDING = "PENDING_BINDING"
    ACTIVE = "ACTIVE"
    PENDING_DELETION = "PENDING_DELETION"
    DELETED = "DELETED"
    DELETION_FAILED = "DELETION_FAILED"


class DeletionRequestStatus(StrEnum):
    RECEIVED = "RECEIVED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DeletionFailureCode(StrEnum):
    STORAGE_DELETE_FAILED = "STORAGE_DELETE_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REFERENCE_SCRUB_FAILED = "REFERENCE_SCRUB_FAILED"
    STORE_UNAVAILABLE = "STORE_UNAVAILABLE"
    INVALID_STATE = "INVALID_STATE"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"


class DeletionEventType(StrEnum):
    STARTED = "STARTED"
    RECLAIMED = "RECLAIMED"


class MediaOwnerScope(StrictFrozenModel):
    subject_user_ids: tuple[NonEmptyStr, ...]

    @field_validator("subject_user_ids")
    @classmethod
    def canonical_subjects(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(item.strip() for item in value))
        if not 1 <= len(normalized) <= 2 or len(set(normalized)) != len(normalized):
            raise ValueError("owner scope requires one or two unique subjects")
        return normalized

    @classmethod
    def from_legacy(cls, owner_user_id: str) -> MediaOwnerScope:
        return cls(subject_user_ids=(owner_user_id,))


class DeletionScope(StrictFrozenModel):
    all_user_media: StrictBool = False
    session_ids: tuple[UUID4, ...] = ()
    asset_ids: tuple[UUID4, ...] = ()

    @field_validator("session_ids", "asset_ids")
    @classmethod
    def canonical_ids(cls, value: tuple[UUID4, ...]) -> tuple[UUID4, ...]:
        return tuple(sorted(set(value), key=str))

    @model_validator(mode="after")
    def has_one_scope_form(self) -> DeletionScope:
        explicit = bool(self.session_ids or self.asset_ids)
        if self.all_user_media == explicit:
            raise ValueError("scope must be ALL_USER_MEDIA or an explicit non-empty selection")
        return self

    @property
    def canonical_key(self) -> str:
        if self.all_user_media:
            return "ALL_USER_MEDIA"
        sessions = ",".join(map(str, self.session_ids))
        assets = ",".join(map(str, self.asset_ids))
        return f"sessions={sessions};assets={assets}"

    @property
    def summary(self) -> str:
        if self.all_user_media:
            return "ALL_USER_MEDIA"
        return f"{len(self.session_ids)} session(s), {len(self.asset_ids)} asset(s)"


class RetainedMediaAssetCreate(StrictFrozenModel):
    """Caller-supplied immutable media facts; lifecycle state is server-owned."""

    asset_id: UUID4
    session_id: UUID4
    owner_scope: MediaOwnerScope
    kind: MediaAssetKind
    storage_locator: NonEmptyStr
    content_sha256: Sha256
    captured_at: datetime
    source_asset_ids: tuple[UUID4, ...] = ()

    _captured_at_utc = field_validator("captured_at")(_utc)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_owner(cls, value: object) -> object:
        if isinstance(value, dict) and "owner_scope" not in value and "owner_user_id" in value:
            value = dict(value)
            value["owner_scope"] = {"subject_user_ids": (value.pop("owner_user_id"),)}
        return value

    @property
    def owner_user_id(self) -> str:
        return self.owner_scope.subject_user_ids[0]

    @field_validator("storage_locator")
    @classmethod
    def locator_is_controlled_relative_path(cls, value: str) -> str:
        logical = PurePosixPath(value)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError("storage locator must be a controlled relative path")
        return value

    @field_validator("source_asset_ids")
    @classmethod
    def canonical_sources(cls, value: tuple[UUID4, ...]) -> tuple[UUID4, ...]:
        return tuple(sorted(set(value), key=str))

    @model_validator(mode="after")
    def provenance_is_consistent(self) -> RetainedMediaAssetCreate:
        if self.asset_id in self.source_asset_ids:
            raise ValueError("an asset cannot derive from itself")
        return self


class RetainedMediaAsset(StrictFrozenModel):
    schema_version: Literal["2.0"] = "2.0"
    asset_id: UUID4
    session_id: UUID4
    owner_scope: MediaOwnerScope
    kind: MediaAssetKind
    storage_locator: NonEmptyStr | None
    content_sha256: Sha256
    captured_at: datetime
    reviewed_at: datetime | None = None
    delivered_at: datetime | None = None
    retention_expires_at: datetime | None = None
    deletion_status: MediaDeletionStatus = MediaDeletionStatus.ACTIVE
    source_asset_ids: tuple[UUID4, ...] = ()
    created_at: datetime
    updated_at: datetime

    _captured_at_utc = field_validator("captured_at")(_utc)
    _reviewed_at_utc = field_validator("reviewed_at")(
        lambda value: None if value is None else _utc(value)
    )
    _delivered_at_utc = field_validator("delivered_at")(
        lambda value: None if value is None else _utc(value)
    )
    _expires_at_utc = field_validator("retention_expires_at")(
        lambda value: None if value is None else _utc(value)
    )
    _created_at_utc = field_validator("created_at")(_utc)
    _updated_at_utc = field_validator("updated_at")(_utc)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_owner(cls, value: object) -> object:
        if isinstance(value, dict) and "owner_scope" not in value and "owner_user_id" in value:
            value = dict(value)
            value.pop("schema_version", None)
            value["owner_scope"] = {"subject_user_ids": (value.pop("owner_user_id"),)}
        return value

    @property
    def owner_user_id(self) -> str:
        return self.owner_scope.subject_user_ids[0]

    @field_validator("storage_locator")
    @classmethod
    def locator_is_controlled_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        logical = PurePosixPath(value)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError("storage locator must be a controlled relative path")
        return value

    @field_validator("source_asset_ids")
    @classmethod
    def canonical_sources(cls, value: tuple[UUID4, ...]) -> tuple[UUID4, ...]:
        return tuple(sorted(set(value), key=str))

    @model_validator(mode="after")
    def timeline_and_deletion_are_consistent(self) -> RetainedMediaAsset:
        future_limit = self.created_at + timedelta(minutes=RETENTION_CLOCK_SKEW_MINUTES)
        if any(
            timestamp is not None and timestamp > future_limit
            for timestamp in (self.captured_at, self.reviewed_at, self.delivered_at)
        ):
            raise ValueError("media timestamps cannot be more than five minutes in the future")
        for milestone in (self.reviewed_at, self.delivered_at):
            if milestone is not None and milestone < self.captured_at:
                raise ValueError("review and delivery cannot predate capture")
        if (self.reviewed_at is None or self.delivered_at is None) and (
            self.retention_expires_at is not None
        ):
            raise ValueError("expiry requires review and delivery")
        if self.reviewed_at is not None and self.delivered_at is not None:
            if self.retention_expires_at is None:
                raise ValueError("reviewed and delivered assets require expiry")
            expected_expiry = max(self.reviewed_at, self.delivered_at) + timedelta(
                days=CLIP_RETENTION_DAYS
            )
            if self.retention_expires_at != expected_expiry:
                raise ValueError("expiry must be exactly 30 days after the retention anchor")
        if self.updated_at < self.created_at or self.created_at < self.captured_at:
            raise ValueError("asset audit timeline moves backwards")
        if self.asset_id in self.source_asset_ids:
            raise ValueError("an asset cannot derive from itself")
        if self.deletion_status is MediaDeletionStatus.DELETED and self.storage_locator is not None:
            raise ValueError("deleted assets cannot retain a storage locator")
        if self.deletion_status is not MediaDeletionStatus.DELETED and self.storage_locator is None:
            raise ValueError("non-deleted assets require a storage locator")
        return self


class DeletionAttempt(StrictFrozenModel):
    attempt_number: int = Field(ge=1)
    executor_id: NonEmptyStr
    started_at: datetime
    finished_at: datetime
    planned_asset_count: int = Field(ge=0)
    verified_deleted_count: int = Field(ge=0)
    result: Literal["COMPLETED", "FAILED"]
    failure_code: DeletionFailureCode | None = None
    retryable: StrictBool = False
    evidence_summary: NonEmptyStr

    _started_at_utc = field_validator("started_at")(_utc)
    _finished_at_utc = field_validator("finished_at")(_utc)

    @model_validator(mode="after")
    def attempt_is_consistent(self) -> DeletionAttempt:
        if self.finished_at < self.started_at:
            raise ValueError("attempt cannot finish before it starts")
        if self.verified_deleted_count > self.planned_asset_count:
            raise ValueError("verified count cannot exceed planned count")
        if self.result == "COMPLETED":
            if self.failure_code is not None or self.retryable:
                raise ValueError("completed attempt cannot contain a failure")
            if self.verified_deleted_count != self.planned_asset_count:
                raise ValueError("completed attempt must verify every asset")
        elif self.failure_code is None:
            raise ValueError("failed attempt requires a typed failure code")
        return self


class DeletionEvent(StrictFrozenModel):
    event_number: int = Field(ge=1)
    event_type: DeletionEventType
    executor_id: NonEmptyStr
    occurred_at: datetime
    evidence_summary: NonEmptyStr

    _occurred_at_utc = field_validator("occurred_at")(_utc)


class DeletionRequest(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: UUID4
    user_id: NonEmptyStr
    scope: DeletionScope
    idempotency_key: NonEmptyStr
    status: DeletionRequestStatus
    created_at: datetime
    updated_at: datetime
    frozen_asset_ids: tuple[UUID4, ...] = ()
    attempts: tuple[DeletionAttempt, ...] = ()
    events: tuple[DeletionEvent, ...] = ()
    completed_at: datetime | None = None
    failure_reason_code: DeletionFailureCode | None = None
    retryable: StrictBool = False
    completion_evidence_summary: NonEmptyStr | None = None

    _created_at_utc = field_validator("created_at")(_utc)
    _updated_at_utc = field_validator("updated_at")(_utc)
    _completed_at_utc = field_validator("completed_at")(
        lambda value: None if value is None else _utc(value)
    )

    @field_validator("frozen_asset_ids")
    @classmethod
    def canonical_assets(cls, value: tuple[UUID4, ...]) -> tuple[UUID4, ...]:
        return tuple(sorted(set(value), key=str))

    @model_validator(mode="after")
    def request_is_consistent(self) -> DeletionRequest:
        if self.updated_at < self.created_at:
            raise ValueError("request timeline moves backwards")
        if self.status is DeletionRequestStatus.COMPLETED:
            if self.completed_at is None or self.completion_evidence_summary is None:
                raise ValueError("completed request requires evidence and completion time")
            if self.failure_reason_code is not None or self.retryable:
                raise ValueError("completed request cannot contain an active failure")
        elif self.completed_at is not None or self.completion_evidence_summary is not None:
            raise ValueError("only a completed request may contain completion evidence")
        if self.status is DeletionRequestStatus.FAILED:
            if self.failure_reason_code is None:
                raise ValueError("failed request requires a typed failure reason")
        elif self.failure_reason_code is not None or self.retryable:
            raise ValueError("only a failed request may contain failure state")
        return self


class PublicDeletionStatus(StrictFrozenModel):
    request_id: UUID4
    status: DeletionRequestStatus
    updated_at: datetime
    scope_summary: NonEmptyStr
    next_action: Literal["WAIT", "NONE", "RETRY_OR_CONTACT_OPERATOR"]

    _updated_at_utc = field_validator("updated_at")(_utc)

    @classmethod
    def from_request(cls, request: DeletionRequest) -> PublicDeletionStatus:
        if request.status is DeletionRequestStatus.COMPLETED:
            action = "NONE"
        elif request.status is DeletionRequestStatus.FAILED:
            action = "RETRY_OR_CONTACT_OPERATOR"
        else:
            action = "WAIT"
        return cls(
            request_id=request.request_id,
            status=request.status,
            updated_at=request.updated_at.astimezone(UTC),
            scope_summary=request.scope.summary,
            next_action=action,
        )
