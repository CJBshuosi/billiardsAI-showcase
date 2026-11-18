"""Pure, fail-closed review gate for report revisions."""

from __future__ import annotations

from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import (
    ReportDeliveryDecision,
    ReportNotDeliverableReason,
    ReviewTask,
    ReviewTaskStatus,
)
from billiards_ai.shared.models.shot import ReviewStatus, Shot

_TASK_BLOCK_REASONS = {
    ReviewTaskStatus.PENDING_REVIEW: ReportNotDeliverableReason.REVIEW_PENDING,
    ReviewTaskStatus.IN_REVIEW: ReportNotDeliverableReason.REVIEW_IN_PROGRESS,
    ReviewTaskStatus.REJECTED: ReportNotDeliverableReason.REVIEW_REJECTED,
}
_DELIVERABLE_REPORT_STATUS = {
    ReviewTaskStatus.APPROVED: ReportStatus.APPROVED,
    ReviewTaskStatus.CORRECTED: ReportStatus.CORRECTED,
}
_PROCESSED_SHOT_STATUSES = {
    ReviewStatus.APPROVED,
    ReviewStatus.CORRECTED,
    ReviewStatus.REJECTED,
}


def assess_report_delivery(
    report: ReportDocument,
    review_task: ReviewTask,
    shots: tuple[Shot, ...],
) -> ReportDeliveryDecision:
    """Evaluate every authority required by AD-16; unknown combinations block."""

    if not isinstance(review_task.status, ReviewTaskStatus):
        return _blocked(ReportNotDeliverableReason.UNKNOWN_REVIEW_STATUS)
    if report.session_id != review_task.session_id or report.report_id != review_task.report_id:
        return _blocked(ReportNotDeliverableReason.SESSION_MISMATCH)
    if report.revision != review_task.report_revision:
        return _blocked(ReportNotDeliverableReason.REVISION_MISMATCH)
    if reason := _TASK_BLOCK_REASONS.get(review_task.status):
        return _blocked(reason)

    expected_report_status = _DELIVERABLE_REPORT_STATUS.get(review_task.status)
    if expected_report_status is None:
        return _blocked(ReportNotDeliverableReason.UNKNOWN_REVIEW_STATUS)
    if report.report_status is ReportStatus.DRAFT_PENDING_REVIEW:
        return _blocked(ReportNotDeliverableReason.REPORT_NOT_APPROVED)
    if report.report_status is not expected_report_status:
        return _blocked(ReportNotDeliverableReason.REPORT_STATUS_MISMATCH)

    expected_shot_count = report.session_statistics.total_shot_count
    if (
        len(shots) != expected_shot_count
        or len({shot.id for shot in shots}) != len(shots)
        or {shot.sequence for shot in shots} != set(range(1, expected_shot_count + 1))
        or any(shot.session_id != report.session_id for shot in shots)
    ):
        return _blocked(ReportNotDeliverableReason.SHOT_SET_INCOMPLETE)
    shots_by_id = {shot.id: shot for shot in shots}
    if report.schema_version == "1.1":
        ordered_shot_ids = tuple(shot.id for shot in sorted(shots, key=lambda shot: shot.sequence))
        if report.objective_statistics.shot_ids != ordered_shot_ids or any(
            card.shot_id not in shots_by_id or shots_by_id[card.shot_id].sequence != card.sequence
            for card in report.typical_shot_analysis.cards
        ):
            return _blocked(ReportNotDeliverableReason.SHOT_SET_INCOMPLETE)
    for shot in shots:
        if not isinstance(shot.review_status, ReviewStatus):
            return _blocked(ReportNotDeliverableReason.UNKNOWN_SHOT_REVIEW_STATUS)
        if shot.review_status not in _PROCESSED_SHOT_STATUSES:
            return _blocked(ReportNotDeliverableReason.UNPROCESSED_CRITICAL_SHOT)
    return ReportDeliveryDecision(deliverable=True)


def require_deliverable_report(
    report: ReportDocument,
    review_task: ReviewTask,
    shots: tuple[Shot, ...],
) -> ReportDocument:
    decision = assess_report_delivery(report, review_task, shots)
    if not decision.deliverable:
        assert decision.reason is not None
        raise ReportNotDeliverableError(decision.reason)
    return report


def _blocked(reason: ReportNotDeliverableReason) -> ReportDeliveryDecision:
    return ReportDeliveryDecision(deliverable=False, reason=reason)
