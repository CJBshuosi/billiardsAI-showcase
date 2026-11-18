"""Strict transport-neutral commands and safe responses for human review."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import (
    UUID4,
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models.events import CoreContractModel, require_utc
from billiards_ai.shared.models.review import (
    ReviewableField,
    ReviewEvidence,
    ReviewTaskStatus,
    ReviewValue,
    SessionRejectionReason,
    ShotReviewOutcome,
    UserNotificationStatus,
)

from .metric_authority import MetricShotReviewInput

IntentNote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


def _require_diverse_idempotency_key(value: str) -> str:
    if len(set(value)) < 8:
        raise ValueError("idempotency key has insufficient character diversity")
    return value


IdempotencyKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False,
        min_length=32,
        max_length=128,
        pattern=r"^[A-Za-z0-9._~-]+$",
    ),
    AfterValidator(_require_diverse_idempotency_key),
]
OpaqueCursor = Annotated[
    str, StringConstraints(strip_whitespace=False, min_length=1, max_length=512)
]


class TaskRef(CoreContractModel):
    task_id: UUID4
    expected_lock_version: StrictInt = Field(ge=0)
    source_report_id: UUID4
    source_report_revision: StrictInt = Field(ge=1)


class CorrectionIntent(CoreContractModel):
    field: ReviewableField
    correct_value: ReviewValue
    note: IntentNote

    @model_validator(mode="after")
    def value_matches_field(self) -> CorrectionIntent:
        value = self.correct_value
        if self.field is ReviewableField.POTTED_BALLS:
            if not isinstance(value, tuple) or any(
                type(item) is not int or item < 0 for item in value
            ):
                raise ValueError("potted_balls correction must be non-negative integers")
        elif self.field is ReviewableField.IS_FOUL:
            if value is not None and type(value) is not bool:
                raise ValueError("is_foul correction must be boolean or null")
        elif value is not None and (type(value) is not int or value < 0):
            raise ValueError("numeric correction must be a non-negative integer or null")
        return self


class ShotDecisionIntent(CoreContractModel):
    shot_id: UUID4
    outcome: ShotReviewOutcome
    trajectory_reviewed: StrictBool
    corrections: tuple[CorrectionIntent, ...] = ()

    @model_validator(mode="after")
    def corrections_match_outcome(self) -> ShotDecisionIntent:
        if (self.outcome is ShotReviewOutcome.CORRECTED) != bool(self.corrections):
            raise ValueError("corrected outcome requires corrections exclusively")
        fields = tuple(correction.field for correction in self.corrections)
        if len(fields) != len(set(fields)):
            raise ValueError("a field may be corrected only once per shot")
        return self


class ClaimReviewCommand(CoreContractModel):
    task: TaskRef
    idempotency_key: IdempotencyKey


class DecideReviewCommand(CoreContractModel):
    task: TaskRef
    decisions: tuple[ShotDecisionIntent, ...] = Field(min_length=1)
    metric_shots: tuple[MetricShotReviewInput, ...] = Field(min_length=1)


class FinalizeReviewCommand(DecideReviewCommand):
    idempotency_key: IdempotencyKey


class RejectReviewCommand(CoreContractModel):
    task: TaskRef
    reason: SessionRejectionReason
    idempotency_key: IdempotencyKey


class ReviewTaskSummary(CoreContractModel):
    task_id: UUID4
    session_id: UUID4
    source_report_id: UUID4
    source_report_revision: StrictInt = Field(ge=1)
    status: ReviewTaskStatus
    lock_version: StrictInt = Field(ge=0)
    created_at: datetime

    _created_at_utc = field_validator("created_at")(require_utc)


class ReviewTaskPage(CoreContractModel):
    items: tuple[ReviewTaskSummary, ...]
    next_cursor: OpaqueCursor | None = None


class ReviewTaskDetail(CoreContractModel):
    task: ReviewTaskSummary
    evidence: ReviewEvidence


class ShotDecisionPreview(CoreContractModel):
    shot_id: UUID4
    outcome: ShotReviewOutcome


class DecisionPreview(CoreContractModel):
    task_id: UUID4
    lock_version: StrictInt = Field(ge=0)
    decisions: tuple[ShotDecisionPreview, ...] = Field(min_length=1)


class ApprovedReviewResult(CoreContractModel):
    task_id: UUID4
    session_id: UUID4
    status: ReviewTaskStatus
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    batch_id: UUID4

    @model_validator(mode="after")
    def status_is_approved_or_corrected(self) -> ApprovedReviewResult:
        if self.status not in {ReviewTaskStatus.APPROVED, ReviewTaskStatus.CORRECTED}:
            raise ValueError("approved result requires approved or corrected status")
        return self


class RejectedReviewResult(CoreContractModel):
    task_id: UUID4
    session_id: UUID4
    status: ReviewTaskStatus
    reason: SessionRejectionReason
    notification_event_id: UUID4
    notification_status: UserNotificationStatus

    @model_validator(mode="after")
    def status_is_rejected(self) -> RejectedReviewResult:
        if self.status is not ReviewTaskStatus.REJECTED:
            raise ValueError("rejected result requires rejected status")
        return self
