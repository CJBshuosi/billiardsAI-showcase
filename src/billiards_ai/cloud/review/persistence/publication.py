"""Internal authority contracts for approved-evidence publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import UUID4, Field, field_validator, model_validator

from billiards_ai.shared.models.events import CoreContractModel, DomainEvent, require_utc
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
    EvidenceApprovalKind,
)
from billiards_ai.shared.models.review import ReviewTaskStatus

if TYPE_CHECKING:
    from .repository import StoredReviewTask


class ReviewSessionOwnership(CoreContractModel):
    """Pre-finalize trusted binding; finalize requests cannot supply a user id."""

    schema_version: Literal["1.0"] = "1.0"
    task_id: UUID4
    user_id: UUID4
    session_id: UUID4
    bound_at: datetime

    _bound_at_utc = field_validator("bound_at")(require_utc)


class ReviewPublicationAuthority(CoreContractModel):
    """Server-owned candidate identities and approved metric authority."""

    schema_version: Literal["1.0"] = "1.0"
    task_id: UUID4
    session_id: UUID4
    result_report_id: UUID4
    review_revision: int
    task_status: ReviewTaskStatus
    batch_id: UUID4
    event_id: UUID4
    approved_at: datetime
    metric_authority: ApprovedMetricEvidenceAuthorityRecord

    _approved_at_utc = field_validator("approved_at")(require_utc)

    @model_validator(mode="after")
    def human_review_is_the_only_production_policy(self) -> ReviewPublicationAuthority:
        if (
            self.metric_authority.approval_kind is not EvidenceApprovalKind.HUMAN_REVIEW
            or self.metric_authority.approval_policy_version != "human-review-v1"
        ):
            raise ValueError("quality-gate publication requires human-review-v1 authority")
        if self.review_revision < 1:
            raise ValueError("review revision must be positive")
        return self


class ReviewPublicationRecord(CoreContractModel):
    """Persisted exact link from reviewed result authority to one v2 publication."""

    schema_version: Literal["1.0"] = "1.0"
    task_id: UUID4
    user_id: UUID4
    session_id: UUID4
    result_report_id: UUID4
    review_revision: int = Field(ge=1)
    task_status: Literal["approved", "corrected"]
    batch_id: UUID4
    event_id: UUID4
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_kind: EvidenceApprovalKind
    approval_policy_version: Literal["human-review-v1"]
    calibration_id: UUID4
    calibration_revision_id: UUID4
    calibration_revision_number: int = Field(ge=1)
    approved_at: datetime

    _approved_at_utc = field_validator("approved_at")(require_utc)


@dataclass(frozen=True, slots=True)
class StoredApprovedEvidencePublication:
    task: StoredReviewTask
    batch: ApprovedEvidenceBatchV2
    event: DomainEvent
    replayed: bool
