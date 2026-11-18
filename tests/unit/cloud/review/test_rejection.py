"""Pure Story 4.3 session rejection and safe query projection tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.review import (
    InMemoryApprovedDataRepository,
    ReportNotDeliverableError,
    project_rejected_session,
    reject_review_session,
    require_deliverable_report,
)
from billiards_ai.shared.models import (
    AccessActorRole,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReviewerIdentity,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejectionDetail,
    SessionRejectionReason,
    UserNotificationMessage,
    UserNotificationStatus,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
REJECTED_AT = datetime(2026, 7, 29, 12, tzinfo=UTC)
REASON_CASES = (
    (
        SessionRejectionReason.OCCLUDED,
        SessionRejectionDetail.CAMERA_VIEW_OBSTRUCTED,
        UserNotificationMessage.CAMERA_VIEW_OBSTRUCTED,
    ),
    (
        SessionRejectionReason.WRONG_TIME_WINDOW,
        SessionRejectionDetail.RECORDING_WINDOW_MISMATCH,
        UserNotificationMessage.RECORDING_WINDOW_MISMATCH,
    ),
    (
        SessionRejectionReason.INSUFFICIENT_MATERIAL_QUALITY,
        SessionRejectionDetail.RECORDING_QUALITY_INSUFFICIENT,
        UserNotificationMessage.RECORDING_QUALITY_INSUFFICIENT,
    ),
)


def make_report() -> ReportDocument:
    return ReportDocument.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def make_task(
    report: ReportDocument,
    *,
    status: ReviewTaskStatus = ReviewTaskStatus.IN_REVIEW,
) -> ReviewTask:
    return ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=status,
    )


def make_reviewer(*, allowed: bool = True) -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-43",
        roles=(AccessActorRole.REVIEWER if allowed else AccessActorRole.ACCESS_ADMIN,),
        credential_id="credential-43",
    )


def reject(
    *,
    report: ReportDocument | None = None,
    task: ReviewTask | None = None,
    reviewer: ReviewerIdentity | None = None,
    reason: SessionRejectionReason = SessionRejectionReason.OCCLUDED,
    rejected_at: datetime = REJECTED_AT,
):
    report = report or make_report()
    return reject_review_session(
        report=report,
        review_task=task or make_task(report),
        reviewer=reviewer or make_reviewer(),
        reason=reason,
        rejection_id=uuid4(),
        notification_event_id=uuid4(),
        rejected_at=rejected_at,
    )


@pytest.mark.parametrize(("reason", "detail", "message"), REASON_CASES)
def test_each_reason_creates_a_terminal_audit_and_pending_notification(reason, detail, message):
    report = make_report()
    task = make_task(report)

    result = reject(report=report, task=task, reason=reason)

    assert task.status is ReviewTaskStatus.IN_REVIEW
    assert report.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert result.review_task.status is ReviewTaskStatus.REJECTED
    assert result.review_task.id == task.id
    assert result.review_task.session_id == report.session_id
    assert result.review_task.report_id == report.report_id
    assert result.review_task.report_revision == report.revision
    assert result.rejection.review_task_id == task.id
    assert result.rejection.reason_code is reason
    assert result.rejection.detail is detail
    assert result.rejection.reviewer == make_reviewer()
    assert result.rejection.rejected_at == REJECTED_AT
    assert result.notification.session_id == report.session_id
    assert result.notification.reason_code is reason
    assert result.notification.message is message
    assert result.notification.created_at == REJECTED_AT
    assert result.notification.status is UserNotificationStatus.PENDING
    assert not hasattr(result, "report")


@pytest.mark.parametrize(
    "status",
    [
        ReviewTaskStatus.PENDING_REVIEW,
        ReviewTaskStatus.APPROVED,
        ReviewTaskStatus.CORRECTED,
        ReviewTaskStatus.REJECTED,
    ],
)
def test_only_in_review_task_can_transition_and_rejected_is_terminal(status):
    report = make_report()

    with pytest.raises(ValueError, match="in-review|terminal"):
        reject(report=report, task=make_task(report, status=status))


def test_rejection_requires_reviewer_role_and_exact_current_draft_binding():
    report = make_report()
    task = make_task(report)

    with pytest.raises(ValueError, match="REVIEWER"):
        reject(report=report, task=task, reviewer=make_reviewer(allowed=False))
    with pytest.raises(ValueError, match="session|bind"):
        reject(report=report, task=task.model_copy(update={"session_id": uuid4()}))
    with pytest.raises(ValueError, match="report|bind"):
        reject(report=report, task=task.model_copy(update={"report_id": uuid4()}))
    with pytest.raises(ValueError, match="revision|bind"):
        reject(report=report, task=task.model_copy(update={"report_revision": report.revision + 1}))
    with pytest.raises(ValidationError):
        reject(report=report, task=task.model_copy(update={"report_revision": True}))


def test_rejection_rejects_non_draft_or_tampered_report():
    report = make_report()
    approved = ReportDocument.model_validate(
        report.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    )

    with pytest.raises(ValueError, match="draft"):
        reject(report=approved, task=make_task(approved))
    with pytest.raises(ValidationError):
        reject(
            report=report.model_copy(update={"revision": True}),
            task=make_task(report),
        )


def test_rejection_time_must_be_utc_and_not_predate_the_draft():
    report = make_report()
    with pytest.raises(ValueError, match="UTC"):
        reject(rejected_at=REJECTED_AT.astimezone(timezone(timedelta(hours=2))))
    with pytest.raises(ValueError, match="predate"):
        reject(report=report, rejected_at=report.generated_at - timedelta(seconds=1))


def test_projection_revalidates_the_terminal_fact_and_returns_only_rejected_outcome():
    result = reject()

    outcome = project_rejected_session(result)
    serialized = outcome.model_dump_json()

    assert outcome.session_id == result.rejection.session_id
    assert outcome.status is ReviewTaskStatus.REJECTED
    assert outcome.reason_code is result.rejection.reason_code
    assert outcome.notification_event_id == result.notification.event_id
    assert outcome.notification_status is UserNotificationStatus.PENDING
    assert all(
        forbidden not in serialized.lower()
        for forbidden in (
            "report",
            "score",
            "shot",
            "raw",
            "media",
            "locator",
            "url",
            "draft",
            "pending_review",
        )
    )

    tampered = result.model_copy(
        update={"notification": result.notification.model_copy(update={"session_id": uuid4()})}
    )
    with pytest.raises(ValidationError):
        project_rejected_session(tampered)


def test_projection_rejects_model_copy_extra_on_the_terminal_aggregate():
    tampered = reject().model_copy(update={"draft_report": "must not survive"})

    with pytest.raises(ValidationError, match="Extra inputs"):
        project_rejected_session(tampered)


@pytest.mark.parametrize(
    ("input_name", "extra"),
    [
        ("report", {"raw_observation": "SECRET RAW OBSERVATION"}),
        ("task", {"media_locator": "https://media.example/private.mp4"}),
        ("reviewer", {"email": "customer@example.com"}),
    ],
)
def test_transition_rejects_model_copy_extra_on_every_input(input_name, extra):
    report = make_report()
    task = make_task(report)
    reviewer = make_reviewer()
    values = {"report": report, "task": task, "reviewer": reviewer}
    values[input_name] = values[input_name].model_copy(update=extra)

    with pytest.raises(ValidationError, match="Extra inputs"):
        reject(report=values["report"], task=values["task"], reviewer=values["reviewer"])


def test_rejected_task_cannot_deliver_or_enter_the_approved_repository():
    report = make_report()
    result = reject(report=report)
    repository = InMemoryApprovedDataRepository(
        reports=(report,),
        review_tasks=(result.review_task,),
        shots=(),
    )

    with pytest.raises(ReportNotDeliverableError) as gate_error:
        require_deliverable_report(report, result.review_task, ())
    with pytest.raises(ReportNotDeliverableError) as repository_error:
        repository.get_approved_session_data(
            report.session_id,
            report.report_id,
            report.revision,
        )

    assert gate_error.value.reason is ReportNotDeliverableReason.REVIEW_REJECTED
    assert repository_error.value.reason is ReportNotDeliverableReason.REVIEW_REJECTED


def test_rejection_service_has_no_free_form_or_delivery_escape_hatch():
    report = make_report()
    task = make_task(report)

    with pytest.raises(TypeError):
        reject_review_session(
            report=report,
            review_task=task,
            reviewer=make_reviewer(),
            reason=SessionRejectionReason.OCCLUDED,
            rejection_id=uuid4(),
            notification_event_id=uuid4(),
            rejected_at=REJECTED_AT,
            note="customer@example.com https://media.example/private.mp4",
        )

    payload = json.loads(reject(report=report, task=task).model_dump_json())
    assert "report" not in payload
    assert "shot" not in payload
