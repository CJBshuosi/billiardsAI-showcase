"""Pure session-level rejection transition and safe query projection."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from billiards_ai.shared.models.access import AccessActorRole, ReviewerIdentity
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import (
    RejectedSessionOutcome,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejection,
    SessionRejectionReason,
    SessionRejectionResult,
    UserNotificationEvent,
    UserNotificationStatus,
    _stored_model_values,
)


def _require_utc_rejection_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("session rejection time must use UTC")
    return value.astimezone(UTC)


def reject_review_session(
    *,
    report: ReportDocument,
    review_task: ReviewTask,
    reviewer: ReviewerIdentity,
    reason: SessionRejectionReason,
    rejection_id: UUID,
    notification_event_id: UUID,
    rejected_at: datetime,
) -> SessionRejectionResult:
    """Reject the exact current draft revision without creating or returning a report."""

    report = ReportDocument.model_validate(_stored_model_values(report))
    review_task = ReviewTask.model_validate(_stored_model_values(review_task))
    reviewer = ReviewerIdentity.model_validate(_stored_model_values(reviewer))
    rejected_at = _require_utc_rejection_time(rejected_at)

    if not isinstance(reason, SessionRejectionReason):
        raise ValueError("session rejection requires a known reason code")
    if AccessActorRole.REVIEWER not in reviewer.roles:
        raise ValueError("session rejection requires the REVIEWER role")
    if review_task.status is not ReviewTaskStatus.IN_REVIEW:
        raise ValueError("only an in-review task may enter the rejected terminal state")
    if report.report_status is not ReportStatus.DRAFT_PENDING_REVIEW:
        raise ValueError("only the current draft report may be rejected")
    if (
        review_task.session_id != report.session_id
        or review_task.report_id != report.report_id
        or review_task.report_revision != report.revision
    ):
        raise ValueError("review task must bind the exact draft session, report, and revision")
    if rejected_at < report.generated_at:
        raise ValueError("session rejection cannot predate the draft report")

    rejected_task = ReviewTask(
        id=review_task.id,
        session_id=review_task.session_id,
        report_id=review_task.report_id,
        report_revision=review_task.report_revision,
        status=ReviewTaskStatus.REJECTED,
    )
    rejection = SessionRejection(
        rejection_id=rejection_id,
        review_task_id=rejected_task.id,
        session_id=rejected_task.session_id,
        report_id=rejected_task.report_id,
        report_revision=rejected_task.report_revision,
        reason_code=reason,
        detail=reason.detail,
        reviewer=reviewer,
        rejected_at=rejected_at,
    )
    notification = UserNotificationEvent(
        event_id=notification_event_id,
        session_id=rejected_task.session_id,
        reason_code=reason,
        message=reason.notification_message,
        created_at=rejected_at,
        status=UserNotificationStatus.PENDING,
    )
    return SessionRejectionResult(
        review_task=rejected_task,
        rejection=rejection,
        notification=notification,
    )


def project_rejected_session(result: SessionRejectionResult) -> RejectedSessionOutcome:
    """Project only explicit rejected state and its pending notification fact."""

    result = SessionRejectionResult.model_validate(_stored_model_values(result))
    return RejectedSessionOutcome(
        session_id=result.rejection.session_id,
        status=ReviewTaskStatus.REJECTED,
        reason_code=result.rejection.reason_code,
        notification_event_id=result.notification.event_id,
        notification_status=result.notification.status,
    )
