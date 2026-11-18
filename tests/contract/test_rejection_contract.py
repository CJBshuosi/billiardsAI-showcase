"""Public Story 4.3 rejection facts are strict, immutable, and safe."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    AccessActorRole,
    RejectedSessionOutcome,
    ReviewerIdentity,
    ReviewTask,
    ReviewTaskStatus,
    SessionRejection,
    SessionRejectionDetail,
    SessionRejectionReason,
    SessionRejectionResult,
    UserNotificationEvent,
    UserNotificationMessage,
    UserNotificationStatus,
)

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


def make_reviewer(*, reviewer: bool = True) -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-43",
        roles=(AccessActorRole.REVIEWER if reviewer else AccessActorRole.ACCESS_ADMIN,),
        credential_id="credential-43",
    )


def make_result(
    *,
    reason: SessionRejectionReason = SessionRejectionReason.OCCLUDED,
    detail: SessionRejectionDetail = SessionRejectionDetail.CAMERA_VIEW_OBSTRUCTED,
    message: UserNotificationMessage = UserNotificationMessage.CAMERA_VIEW_OBSTRUCTED,
) -> SessionRejectionResult:
    session_id = uuid4()
    report_id = uuid4()
    task = ReviewTask(
        id=uuid4(),
        session_id=session_id,
        report_id=report_id,
        report_revision=1,
        status=ReviewTaskStatus.REJECTED,
    )
    rejection = SessionRejection(
        rejection_id=uuid4(),
        review_task_id=task.id,
        session_id=session_id,
        report_id=report_id,
        report_revision=1,
        reason_code=reason,
        detail=detail,
        reviewer=make_reviewer(),
        rejected_at=REJECTED_AT,
    )
    notification = UserNotificationEvent(
        event_id=uuid4(),
        session_id=session_id,
        reason_code=reason,
        message=message,
        created_at=REJECTED_AT,
        status=UserNotificationStatus.PENDING,
    )
    return SessionRejectionResult(
        review_task=task,
        rejection=rejection,
        notification=notification,
    )


@pytest.mark.parametrize(("reason", "detail", "message"), REASON_CASES)
def test_each_closed_reason_builds_a_typed_consistent_terminal_fact(reason, detail, message):
    result = make_result(reason=reason, detail=detail, message=message)

    assert result.review_task.status is ReviewTaskStatus.REJECTED
    assert result.rejection.reason_code is reason
    assert result.rejection.detail is detail
    assert result.rejection.reviewer.roles == (AccessActorRole.REVIEWER,)
    assert result.rejection.rejected_at == REJECTED_AT
    assert result.notification.reason_code is reason
    assert result.notification.message is message
    assert result.notification.status is UserNotificationStatus.PENDING
    assert result.notification.created_at == REJECTED_AT


def test_rejection_models_are_frozen_extra_forbid_and_strict_in_python_mode():
    result = make_result()
    payload = result.rejection.model_dump(mode="json")

    with pytest.raises(ValidationError):
        SessionRejection.model_validate(payload)
    with pytest.raises(ValidationError, match="Extra inputs"):
        SessionRejection.model_validate(
            result.rejection.model_dump(mode="python") | {"raw_observation": "secret"}
        )
    with pytest.raises(ValidationError, match="frozen"):
        result.rejection.reason_code = SessionRejectionReason.WRONG_TIME_WINDOW


def test_rejection_models_accept_the_equivalent_strict_json_representation():
    rejection = make_result().rejection

    parsed = SessionRejection.model_validate_json(rejection.model_dump_json())

    assert parsed == rejection


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason_code", "unknown_reason"),
        ("detail", "raw frame: https://media.example/private.mp4"),
    ],
)
def test_rejection_rejects_unknown_or_free_form_public_reason_content(field, value):
    payload = make_result().rejection.model_dump(mode="json") | {field: value}

    with pytest.raises(ValidationError):
        SessionRejection.model_validate_json(json.dumps(payload))


def test_reason_detail_and_notification_message_must_match():
    with pytest.raises(ValidationError, match="reason|detail"):
        make_result(detail=SessionRejectionDetail.RECORDING_WINDOW_MISMATCH)
    with pytest.raises(ValidationError, match="reason|message"):
        make_result(message=UserNotificationMessage.RECORDING_WINDOW_MISMATCH)


def test_reviewer_and_utc_audit_are_required_by_direct_model_construction():
    rejection = make_result().rejection
    with pytest.raises(ValidationError, match="REVIEWER"):
        SessionRejection.model_validate(
            rejection.model_dump(mode="python") | {"reviewer": make_reviewer(reviewer=False)}
        )
    with pytest.raises(ValidationError, match="UTC"):
        SessionRejection.model_validate(
            rejection.model_dump(mode="python")
            | {"rejected_at": REJECTED_AT.astimezone(timezone(timedelta(hours=2)))}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "review_task": lambda result: result.review_task.model_copy(
                update={"status": "approved"}
            )
        },
        {
            "review_task": lambda result: result.review_task.model_copy(
                update={"session_id": uuid4()}
            )
        },
        {"rejection": lambda result: result.rejection.model_copy(update={"report_id": uuid4()})},
        {
            "notification": lambda result: result.notification.model_copy(
                update={"status": "delivered"}
            )
        },
        {
            "notification": lambda result: result.notification.model_copy(
                update={"created_at": REJECTED_AT + timedelta(seconds=1)}
            )
        },
    ],
)
def test_terminal_result_revalidates_status_and_all_cross_fact_bindings(mutation):
    result = make_result()
    field, mutate = next(iter(mutation.items()))

    with pytest.raises(ValidationError):
        SessionRejectionResult.model_validate(
            result.model_dump(mode="python") | {field: mutate(result)}
        )


@pytest.mark.parametrize(
    "nested_field",
    ["reviewer", "rejection", "notification"],
)
def test_terminal_result_rejects_model_copy_extra_in_nested_audit_facts(nested_field):
    result = make_result()
    if nested_field == "reviewer":
        tampered_reviewer = result.rejection.reviewer.model_copy(
            update={"email": "customer@example.com"}
        )
        tampered = result.rejection.model_copy(update={"reviewer": tampered_reviewer})
        update = {"rejection": tampered}
    elif nested_field == "rejection":
        update = {
            "rejection": result.rejection.model_copy(
                update={"raw_observation": "SECRET RAW OBSERVATION"}
            )
        }
    else:
        update = {
            "notification": result.notification.model_copy(
                update={"media_locator": "https://media.example/private.mp4"}
            )
        }

    with pytest.raises(ValidationError, match="Extra inputs"):
        SessionRejectionResult.model_validate(result.model_dump(mode="python") | update)


def test_rejected_outcome_is_explicit_and_contains_no_report_score_or_media_shape():
    result = make_result()
    outcome = RejectedSessionOutcome(
        session_id=result.rejection.session_id,
        status=ReviewTaskStatus.REJECTED,
        reason_code=result.rejection.reason_code,
        notification_event_id=result.notification.event_id,
        notification_status=result.notification.status,
    )
    payload = outcome.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert payload == {
        "session_id": str(result.rejection.session_id),
        "status": "rejected",
        "reason_code": "occluded",
        "notification_event_id": str(result.notification.event_id),
        "notification_status": "pending",
    }
    assert all(
        forbidden not in serialized
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
