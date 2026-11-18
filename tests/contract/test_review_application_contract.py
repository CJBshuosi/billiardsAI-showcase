"""Strict transport-neutral contracts for the controlled review application."""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.review.application_models import (
    ClaimReviewCommand,
    CorrectionIntent,
    DecideReviewCommand,
    FinalizeReviewCommand,
    RejectReviewCommand,
    ShotDecisionIntent,
    TaskRef,
)
from billiards_ai.cloud.review.identity import ResolvedSessionSubject
from billiards_ai.cloud.review.metric_authority import (
    MetricFieldReviewStates,
    MetricPointInput,
    MetricReviewState,
    MetricShotReviewInput,
)
from billiards_ai.shared.models.metric_evidence import (
    FoulResult,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    TargetZoneEvaluationStatus,
)
from billiards_ai.shared.models.review import (
    ReviewableField,
    SessionRejectionReason,
    ShotReviewOutcome,
)
from billiards_ai.shared.models.shot import TargetZoneSource


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _task() -> TaskRef:
    return TaskRef(
        task_id=_id(1),
        expected_lock_version=2,
        source_report_id=_id(2),
        source_report_revision=3,
    )


def _metric(shot_id: UUID) -> MetricShotReviewInput:
    confirmed = MetricReviewState.CONFIRMED
    return MetricShotReviewInput(
        shot_id=shot_id,
        cue_ball_start_mm=MetricPointInput(x_mm=100.0, y_mm=200.0),
        object_ball_start_mm=MetricPointInput(x_mm=500.0, y_mm=200.0),
        intended_object_ball=3,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.LEGAL_SUCCESS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        review_states=MetricFieldReviewStates(
            geometry=confirmed,
            intent=confirmed,
            pot_attempt=confirmed,
            pot_result=confirmed,
            foul_result=confirmed,
            target_zone=confirmed,
            cue_ball_end=MetricReviewState.UNSUPPORTED,
        ),
    )


def test_commands_accept_only_business_intent_and_pair_corrections() -> None:
    correction = CorrectionIntent(
        field=ReviewableField.CUSHION_COUNT,
        correct_value=2,
        note="visible second rail contact",
    )
    decision = ShotDecisionIntent(
        shot_id=_id(10),
        outcome=ShotReviewOutcome.CORRECTED,
        trajectory_reviewed=True,
        corrections=(correction,),
    )

    decide = DecideReviewCommand(
        task=_task(), decisions=(decision,), metric_shots=(_metric(_id(10)),)
    )
    finalize = FinalizeReviewCommand(
        task=_task(),
        decisions=(decision,),
        metric_shots=(_metric(_id(10)),),
        idempotency_key="review-finalize-0001-0123456789abcdef",
    )
    claim = ClaimReviewCommand(task=_task(), idempotency_key="review-claim-000001-0123456789abcdef")
    reject = RejectReviewCommand(
        task=_task(),
        reason=SessionRejectionReason.OCCLUDED,
        idempotency_key="review-reject-00001-0123456789abcdef",
    )

    assert decide.decisions == finalize.decisions == (decision,)
    assert claim.task == reject.task == _task()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (ReviewableField.POTTED_BALLS, 1),
        (ReviewableField.IS_FOUL, 1),
        (ReviewableField.CUSHION_COUNT, True),
        (ReviewableField.TRAVEL_DISTANCE_MM, -1),
    ],
)
def test_correction_intent_rejects_values_outside_the_existing_field_contract(
    field: ReviewableField, value: object
) -> None:
    with pytest.raises(ValidationError):
        CorrectionIntent(field=field, correct_value=value, note="reviewed")


def test_decision_intent_rejects_correction_bypass_and_duplicate_fields() -> None:
    correction = CorrectionIntent(
        field=ReviewableField.CUSHION_COUNT,
        correct_value=1,
        note="reviewed",
    )
    with pytest.raises(ValidationError):
        ShotDecisionIntent(
            shot_id=_id(10),
            outcome=ShotReviewOutcome.APPROVED,
            trajectory_reviewed=True,
            corrections=(correction,),
        )
    with pytest.raises(ValidationError):
        ShotDecisionIntent(
            shot_id=_id(10),
            outcome=ShotReviewOutcome.CORRECTED,
            trajectory_reviewed=True,
            corrections=(correction, correction),
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            TaskRef,
            {
                "task_id": _id(1),
                "expected_lock_version": 0,
                "source_report_id": _id(2),
                "source_report_revision": 1,
                "reviewer": "forged",
            },
        ),
        (
            CorrectionIntent,
            {"field": "cushion_count", "correct_value": 1, "note": "reviewed", "ai_value": 0},
        ),
        (MetricPointInput, {"x_mm": 1.0, "y_mm": 2.0, "calibration_id": _id(3)}),
    ],
)
def test_nested_contracts_reject_caller_authored_authority_fields(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_metric_input_rejects_approval_and_derived_target_result_fields() -> None:
    payload = _metric(_id(10)).model_dump(mode="python")
    payload["approval_kind"] = "HUMAN_REVIEW"
    with pytest.raises(ValidationError):
        MetricShotReviewInput.model_validate(payload)

    payload = _metric(_id(10)).model_dump(mode="python")
    payload["target_zone_result"] = "INSIDE_OR_TOLERATED"
    with pytest.raises(ValidationError):
        MetricShotReviewInput.model_validate(payload)


def test_mutation_keys_are_required_and_high_entropy_shaped() -> None:
    for invalid in (
        "short",
        "a" * 32,
        "abcd" * 8,
        "contains spaces but is now long enough",
    ):
        with pytest.raises(ValidationError):
            ClaimReviewCommand(task=_task(), idempotency_key=invalid)
    with pytest.raises(ValidationError):
        RejectReviewCommand(
            task=_task(),
            reason=SessionRejectionReason.OCCLUDED,
            idempotency_key="contains spaces but is now long enough",
        )


def test_resolved_subject_rejects_non_positive_or_boolean_interval_revision() -> None:
    values = {
        "session_id": _id(1),
        "user_id": _id(2),
        "attempt_id": _id(3),
        "accepted_at": "2026-08-10T16:00:00Z",
    }
    for invalid in (0, True):
        with pytest.raises(ValidationError):
            ResolvedSessionSubject.model_validate_json(
                json.dumps(values | {"interval_revision": invalid}, default=str)
            )
