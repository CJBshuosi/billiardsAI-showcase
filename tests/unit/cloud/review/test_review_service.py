"""Story 4.2 review evidence, per-shot decisions, and immutable revisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.engine import (
    compose_coaching_report,
    generate_recommendations,
    select_typical_shots,
)
from billiards_ai.cloud.review import (
    InMemoryApprovedDataRepository,
    ReportNotDeliverableError,
    apply_shot_reviews,
    require_deliverable_report,
    review_session_evidence,
    submit_review,
)
from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    RecommendationCatalogVersion,
    ReportDocument,
    ReportRevisionReplacements,
    ReportStatus,
    ReviewableField,
    ReviewCorrection,
    ReviewerIdentity,
    ReviewStatus,
    ReviewTask,
    ReviewTaskStatus,
    Shot,
    ShotReviewDecision,
    ShotReviewOutcome,
    TargetZoneSource,
    TypicalShotRuleVersion,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
V11_FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "v1.1" / "complete.json"
REVIEWED_AT = datetime(2026, 7, 29, 11, tzinfo=UTC)


def make_report() -> ReportDocument:
    return ReportDocument.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def make_v11_report() -> ReportDocument:
    return ReportDocument.model_validate_json(V11_FIXTURE.read_text(encoding="utf-8"))


def make_task(report: ReportDocument) -> ReviewTask:
    return ReviewTask(
        id=uuid4(),
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        status=ReviewTaskStatus.IN_REVIEW,
    )


def make_reviewer(*, allowed: bool = True) -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-42",
        roles=(AccessActorRole.REVIEWER if allowed else AccessActorRole.SYSTEM,),
        credential_id="credential-42",
    )


def make_shot(report: ReportDocument, sequence: int, *, abstain: bool = False) -> Shot:
    return Shot(
        id=uuid4(),
        session_id=report.session_id,
        sequence=sequence,
        t_start_ms=(sequence - 1) * 1000,
        t_end_ms=sequence * 1000,
        ball_positions_before=None,
        ball_positions_after=None,
        cue_ball_end=None,
        struck_ball=sequence,
        potted_balls=[sequence] if sequence == 1 else [],
        cushion_count=0,
        travel_distance_mm=500,
        is_foul=False,
        deviation_mm=None,
        confidence=Confidence.ABSTAIN if abstain else Confidence.HIGH,
        review_status=ReviewStatus.IN_REVIEW,
        trajectory=None,
        raw_observation={
            "provider_text": "must-not-leak",
            "media_locator": "oss://private/object",
        },
        created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
    )


def make_shots(report: ReportDocument, *, abstain_sequence: int | None = None) -> tuple[Shot, ...]:
    return tuple(
        make_shot(report, sequence, abstain=sequence == abstain_sequence)
        for sequence in range(1, report.session_statistics.total_shot_count + 1)
    )


def make_decision(
    shot: Shot,
    *,
    outcome: ShotReviewOutcome = ShotReviewOutcome.APPROVED,
    correction: ReviewCorrection | None = None,
    reviewer: ReviewerIdentity | None = None,
    reviewed_at: datetime = REVIEWED_AT,
) -> ShotReviewDecision:
    return ShotReviewDecision(
        shot_id=shot.id,
        outcome=outcome,
        reviewer=reviewer or make_reviewer(),
        reviewed_at=reviewed_at,
        corrections=() if correction is None else (correction,),
    )


def make_correction(
    shot: Shot,
    *,
    field: ReviewableField = ReviewableField.CUSHION_COUNT,
    ai_value: object = 0,
    correct_value: object = 1,
    reviewer: ReviewerIdentity | None = None,
    reviewed_at: datetime = REVIEWED_AT,
) -> ReviewCorrection:
    return ReviewCorrection(
        shot_id=shot.id,
        field=field,
        ai_value=ai_value,
        correct_value=correct_value,
        note="visible rail contact",
        reviewer=reviewer or make_reviewer(),
        reviewed_at=reviewed_at,
    )


def approve_all(shots: tuple[Shot, ...]) -> tuple[ShotReviewDecision, ...]:
    return tuple(make_decision(shot) for shot in shots)


def test_evidence_is_safe_and_keeps_immutable_ai_values():
    report = make_report()
    shots = make_shots(report)
    shots = (
        shots[0].model_copy(
            update={
                "deviation_mm": 400,
                "target_zone_source": TargetZoneSource.SCENE_PRESET,
            }
        ),
        *shots[1:],
    )

    evidence = review_session_evidence(session_id=report.session_id, shots=shots)
    payload = evidence.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert evidence.session_id == report.session_id
    assert evidence.shots[0].ai_values.cushion_count == 0
    assert evidence.shots[0].ai_values.potted_balls == (1,)
    assert evidence.shots[0].ai_values.target_zone_source is TargetZoneSource.SCENE_PRESET
    assert all(
        forbidden not in serialized
        for forbidden in ("raw_observation", "provider_text", "trajectory", "media_locator")
    )
    with pytest.raises(ValidationError, match="frozen"):
        evidence.shots[0].ai_values.cushion_count = 2


def test_no_corrections_approves_same_revision_and_passes_story_4_1_gate():
    report = make_report()
    task = make_task(report)
    shots = make_shots(report, abstain_sequence=4)
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=approve_all(shots),
    )

    submission = submit_review(
        report=report,
        review_task=task,
        application=application,
        completed_at=REVIEWED_AT,
    )

    assert submission.report.report_id == report.report_id
    assert submission.report.revision == report.revision
    assert submission.report.report_status is ReportStatus.APPROVED
    assert submission.review_task.status is ReviewTaskStatus.APPROVED
    assert submission.review_task.report_id == submission.report.report_id
    assert submission.review_task.report_revision == submission.report.revision
    assert submission.corrections == ()
    assert all(shot.review_status is ReviewStatus.APPROVED for shot in submission.shots)
    assert (
        require_deliverable_report(submission.report, submission.review_task, submission.shots)
        is submission.report
    )
    assert report.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert task.status is ReviewTaskStatus.IN_REVIEW


def test_correction_records_pair_and_creates_new_immutable_revision():
    report = make_report()
    task = make_task(report)
    shots = make_shots(report)
    correction = make_correction(shots[0])
    decisions = (
        make_decision(shots[0], outcome=ShotReviewOutcome.CORRECTED, correction=correction),
        *approve_all(shots[1:]),
    )
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=decisions,
    )

    submission = submit_review(
        report=report,
        review_task=task,
        application=application,
        completed_at=REVIEWED_AT,
        corrected_report_id=uuid4(),
    )

    assert submission.shots[0].cushion_count == 1
    assert submission.shots[0].review_status is ReviewStatus.CORRECTED
    assert submission.corrections == (correction,)
    assert submission.corrections[0].ai_value == 0
    assert submission.corrections[0].correct_value == 1
    assert submission.corrections[0].reviewer == make_reviewer()
    assert submission.corrections[0].reviewed_at == REVIEWED_AT
    assert submission.report.report_id != report.report_id
    assert submission.report.parent_revision_id == report.report_id
    assert submission.report.revision == report.revision + 1
    assert submission.report.report_status is ReportStatus.CORRECTED
    assert submission.review_task.status is ReviewTaskStatus.CORRECTED
    assert submission.review_task.report_id == submission.report.report_id
    assert submission.review_task.report_revision == submission.report.revision
    assert report.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert report.parent_revision_id is None
    assert report.session_statistics == submission.report.session_statistics
    assert submission.shots[1].model_dump() == (
        shots[1].model_dump() | {"review_status": ReviewStatus.APPROVED}
    )
    assert (
        require_deliverable_report(submission.report, submission.review_task, submission.shots)
        is submission.report
    )


def test_v11_shot_correction_creates_clean_v10_base_before_recomposition():
    report = make_v11_report()
    completed_at = report.generated_at + timedelta(hours=1)
    task = make_task(report)
    shots = tuple(
        shot.model_copy(update={"confidence": Confidence.LOW}) for shot in make_shots(report)
    )
    correction = make_correction(shots[0])
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=(
            make_decision(shots[0], outcome=ShotReviewOutcome.CORRECTED, correction=correction),
            *approve_all(shots[1:]),
        ),
    )

    submission = submit_review(
        report=report,
        review_task=task,
        application=application,
        completed_at=completed_at,
        corrected_report_id=uuid4(),
    )

    assert submission.report.schema_version == "1.0"
    assert submission.report.report_status is ReportStatus.CORRECTED
    assert submission.report.headline is None
    assert submission.report.trust_scope is None
    assert submission.report.objective_statistics is None
    assert submission.report.typical_shot_analysis is None
    assert submission.report.training_recommendation is None
    assert submission.report.quality_notice is None
    assert submission.report.content_sha256 is None
    assert (
        require_deliverable_report(submission.report, submission.review_task, submission.shots)
        is submission.report
    )

    repository = InMemoryApprovedDataRepository(
        reports=(submission.report,),
        review_tasks=(submission.review_task,),
        shots=submission.shots,
    )
    selection = select_typical_shots(
        repository,
        submission.report.session_id,
        submission.report.report_id,
        submission.report.revision,
        TypicalShotRuleVersion.V1,
    )
    recommendation = generate_recommendations(
        repository,
        submission.report.session_id,
        submission.report.report_id,
        submission.report.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )
    recomposed = compose_coaching_report(
        repository=repository,
        session_id=submission.report.session_id,
        parent_report_id=submission.report.report_id,
        parent_revision=submission.report.revision,
        child_report_id=uuid4(),
        child_review_task_id=uuid4(),
        generated_at=completed_at + timedelta(seconds=1),
        selection_result=selection,
        visualization_results=(),
        recommendation_result=recommendation,
    )
    assert recomposed.report.schema_version == "1.1"
    assert recomposed.report.parent_revision_id == submission.report.report_id


def test_parent_report_and_old_task_cannot_pass_gate_after_correction():
    report = make_report()
    task = make_task(report)
    shots = make_shots(report)
    correction = make_correction(shots[0])
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=(
            make_decision(shots[0], outcome=ShotReviewOutcome.CORRECTED, correction=correction),
            *approve_all(shots[1:]),
        ),
    )
    submission = submit_review(
        report=report,
        review_task=task,
        application=application,
        completed_at=REVIEWED_AT,
        corrected_report_id=uuid4(),
    )

    with pytest.raises(ReportNotDeliverableError):
        require_deliverable_report(report, task, shots)
    with pytest.raises(ReportNotDeliverableError):
        require_deliverable_report(report, submission.review_task, submission.shots)


@pytest.mark.parametrize("bad_decisions", ["duplicate", "missing", "wrong_shot"])
def test_every_session_shot_requires_exactly_one_explicit_decision(bad_decisions: str):
    report = make_report()
    shots = make_shots(report, abstain_sequence=4)
    decisions = list(approve_all(shots))
    if bad_decisions == "duplicate":
        decisions[-1] = decisions[0]
    elif bad_decisions == "missing":
        decisions.pop()
    else:
        decisions[-1] = ShotReviewDecision.model_validate(
            decisions[-1].model_dump() | {"shot_id": uuid4()}
        )

    with pytest.raises(ValueError, match="decision|shot"):
        apply_shot_reviews(
            session_id=report.session_id,
            shots=shots,
            decisions=tuple(decisions),
        )


def test_wrong_session_duplicate_shots_and_non_reviewer_fail_closed():
    report = make_report()
    shots = make_shots(report)
    with pytest.raises(ValueError, match="session"):
        review_session_evidence(session_id=uuid4(), shots=shots)
    with pytest.raises(ValueError, match="shot"):
        review_session_evidence(session_id=report.session_id, shots=(shots[0], shots[0]))

    with pytest.raises(ValidationError, match="REVIEWER"):
        make_decision(shots[0], reviewer=make_reviewer(allowed=False))


def test_correction_ai_value_must_equal_original_and_cannot_change_other_shot():
    report = make_report()
    shots = make_shots(report)
    wrong_original = make_correction(shots[0], ai_value=9)
    with pytest.raises(ValueError, match="ai_value"):
        apply_shot_reviews(
            session_id=report.session_id,
            shots=shots,
            decisions=(
                make_decision(
                    shots[0],
                    outcome=ShotReviewOutcome.CORRECTED,
                    correction=wrong_original,
                ),
                *approve_all(shots[1:]),
            ),
        )

    other_shot_correction = ReviewCorrection.model_validate(
        make_correction(shots[1]).model_dump() | {"shot_id": shots[0].id}
    )
    with pytest.raises(ValueError, match="shot"):
        ShotReviewDecision(
            shot_id=shots[1].id,
            outcome=ShotReviewOutcome.CORRECTED,
            reviewer=make_reviewer(),
            reviewed_at=REVIEWED_AT,
            corrections=(other_shot_correction,),
        )


@pytest.mark.parametrize(
    ("field", "ai_value", "correct_value"),
    [
        (ReviewableField.CUSHION_COUNT, False, 1),
        (ReviewableField.IS_FOUL, 0, True),
        (ReviewableField.POTTED_BALLS, (1,), (True,)),
        (ReviewableField.TRAVEL_DISTANCE_MM, 500, -1),
    ],
)
def test_field_values_are_strict_and_field_appropriate(field, ai_value, correct_value):
    report = make_report()
    shot = make_shots(report)[0]
    with pytest.raises((ValidationError, ValueError)):
        make_correction(
            shot,
            field=field,
            ai_value=ai_value,
            correct_value=correct_value,
        )


def test_non_utc_decision_and_invalid_completion_time_are_rejected():
    report = make_report()
    shots = make_shots(report)
    with pytest.raises(ValidationError, match="UTC"):
        make_decision(shots[0], reviewed_at=REVIEWED_AT.astimezone(timezone(timedelta(hours=2))))

    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=approve_all(shots),
    )
    with pytest.raises(ValueError, match="completion|UTC"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application,
            completed_at=datetime(2026, 7, 29, 11),
        )
    with pytest.raises(ValueError, match="completion"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application,
            completed_at=REVIEWED_AT - timedelta(hours=2),
        )


def test_corrected_submission_requires_new_report_id_and_valid_replacement():
    report = make_report()
    shots = make_shots(report)
    correction = make_correction(shots[0])
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=(
            make_decision(shots[0], outcome=ShotReviewOutcome.CORRECTED, correction=correction),
            *approve_all(shots[1:]),
        ),
    )

    with pytest.raises(ValueError, match="report id"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application,
            completed_at=REVIEWED_AT,
        )
    with pytest.raises(ValueError, match="report id"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application,
            completed_at=REVIEWED_AT,
            corrected_report_id=report.report_id,
        )


def test_invalid_explicit_report_replacement_is_rejected_by_full_document_validation():
    report = make_report()
    shots = make_shots(report)
    correction = make_correction(shots[0])
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=(
            make_decision(shots[0], outcome=ShotReviewOutcome.CORRECTED, correction=correction),
            *approve_all(shots[1:]),
        ),
    )
    incompatible_quality = report.quality_summary.model_copy(update={"shot_count": 999})

    with pytest.raises(ValidationError, match="shot counts"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application,
            completed_at=REVIEWED_AT,
            corrected_report_id=uuid4(),
            replacements=ReportRevisionReplacements(
                quality_summary=incompatible_quality,
            ),
        )


def test_tampered_application_cannot_bypass_decision_and_correction_consistency():
    report = make_report()
    shots = make_shots(report)
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=approve_all(shots),
    )
    tampered = application.__class__(
        session_id=application.session_id,
        source_shots=application.source_shots,
        shots=application.shots,
        decisions=application.decisions,
        corrections=(make_correction(shots[0]),),
    )

    with pytest.raises(ValueError, match="correction"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=tampered,
            completed_at=REVIEWED_AT,
        )


def test_tampered_application_cannot_change_unlisted_shot_field():
    report = make_report()
    shots = make_shots(report)
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=approve_all(shots),
    )
    changed = Shot.model_validate(application.shots[0].model_dump() | {"travel_distance_mm": 999})
    tampered = application.__class__(
        session_id=application.session_id,
        source_shots=application.source_shots,
        shots=(changed, *application.shots[1:]),
        decisions=application.decisions,
        corrections=application.corrections,
    )

    with pytest.raises(ValueError, match="field|result|decision"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=tampered,
            completed_at=REVIEWED_AT,
        )


def test_submit_rejects_task_report_or_application_mismatch_and_boolean_revision():
    report = make_report()
    shots = make_shots(report)
    application = apply_shot_reviews(
        session_id=report.session_id,
        shots=shots,
        decisions=approve_all(shots),
    )
    wrong_task = make_task(report).model_copy(update={"report_revision": True})

    with pytest.raises(ValueError, match="revision"):
        submit_review(
            report=report,
            review_task=wrong_task,
            application=application,
            completed_at=REVIEWED_AT,
        )
    with pytest.raises(ValueError, match="session"):
        submit_review(
            report=report,
            review_task=make_task(report),
            application=application.__class__(
                session_id=uuid4(),
                source_shots=application.source_shots,
                shots=application.shots,
                decisions=application.decisions,
                corrections=application.corrections,
            ),
            completed_at=REVIEWED_AT,
        )
