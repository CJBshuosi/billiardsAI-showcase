"""Pure Phase 0 review service for safe evidence and immutable report revisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from billiards_ai.cloud.engine.reporting import revise_report_document
from billiards_ai.shared.models.access import AccessActorRole
from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.quality import FinalFieldSource
from billiards_ai.shared.models.report import ReportDocument, ReportStatus
from billiards_ai.shared.models.review import (
    ReportRevisionReplacements,
    ReviewActualPath,
    ReviewActualPathUnavailableReason,
    ReviewCorrection,
    ReviewEvidence,
    ReviewShotValues,
    ReviewTask,
    ReviewTaskStatus,
    ShotReviewDecision,
    ShotReviewEvidence,
)
from billiards_ai.shared.models.shot import ReviewStatus, Shot


@dataclass(frozen=True, slots=True)
class ReviewApplication:
    session_id: UUID
    source_shots: tuple[Shot, ...]
    shots: tuple[Shot, ...]
    decisions: tuple[ShotReviewDecision, ...]
    corrections: tuple[ReviewCorrection, ...]
    evidence: ReviewEvidence | None = None


@dataclass(frozen=True, slots=True)
class ReviewSubmission:
    report: ReportDocument
    review_task: ReviewTask
    shots: tuple[Shot, ...]
    decisions: tuple[ShotReviewDecision, ...]
    corrections: tuple[ReviewCorrection, ...]
    evidence: ReviewEvidence
    completed_at: datetime
    calibration: CalibrationLineage | None = None


def _require_session_shots(session_id: UUID, shots: tuple[Shot, ...]) -> None:
    if not isinstance(session_id, UUID):
        raise ValueError("session id must be a UUID")
    if not shots:
        raise ValueError("review requires at least one shot")
    if any(shot.session_id != session_id for shot in shots):
        raise ValueError("every shot must belong to the review session")
    if len({shot.id for shot in shots}) != len(shots):
        raise ValueError("review shots must be unique")
    if {shot.sequence for shot in shots} != set(range(1, len(shots) + 1)):
        raise ValueError("review shots must have a complete contiguous sequence")


def review_session_evidence(*, session_id: UUID, shots: tuple[Shot, ...]) -> ReviewEvidence:
    """Project only safe business facts and fixed audit metadata for review."""

    _require_session_shots(session_id, shots)
    return ReviewEvidence(
        session_id=session_id,
        shots=tuple(
            ShotReviewEvidence(
                shot_id=shot.id,
                session_id=shot.session_id,
                sequence=shot.sequence,
                analysis_status=shot.analysis_status,
                review_status=shot.review_status,
                shot_created_at=shot.created_at,
                ai_values=ReviewShotValues(
                    struck_ball=shot.struck_ball,
                    potted_balls=tuple(shot.potted_balls or ()),
                    cushion_count=shot.cushion_count,
                    travel_distance_mm=shot.travel_distance_mm,
                    is_foul=shot.is_foul,
                    deviation_mm=shot.deviation_mm,
                    target_zone_source=shot.target_zone_source,
                    confidence=shot.confidence,
                ),
                actual_path=ReviewActualPath(
                    points=tuple(shot.trajectory or ()),
                    source=(
                        shot.field_quality.trajectory.source
                        if shot.field_quality is not None
                        else FinalFieldSource.UNKNOWN
                    ),
                    confidence=(
                        shot.field_quality.trajectory.confidence
                        if shot.field_quality is not None
                        else 0.0
                    ),
                    calibration=(
                        shot.trajectory_calibration if shot.trajectory is not None else None
                    ),
                    unavailable_reason=(
                        None
                        if shot.trajectory is not None or shot.trajectory_unavailable_reason is None
                        else ReviewActualPathUnavailableReason(
                            shot.trajectory_unavailable_reason.value
                        )
                    ),
                ),
            )
            for shot in sorted(shots, key=lambda item: item.sequence)
        ),
    )


def _shot_value(shot: Shot, correction: ReviewCorrection):
    value = getattr(shot, correction.field.value)
    if correction.field.value == "potted_balls":
        return tuple(value or ())
    return value


def _values_identical(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _reviewed_shot(shot: Shot, decision: ShotReviewDecision) -> Shot:
    payload = shot.model_dump(mode="python")
    for correction in decision.corrections:
        original = _shot_value(shot, correction)
        if not _values_identical(original, correction.ai_value):
            raise ValueError("correction ai_value must equal the immutable Shot original")
        value = correction.correct_value
        payload[correction.field.value] = list(value) if isinstance(value, tuple) else value
    payload["review_status"] = ReviewStatus(decision.outcome.value)
    return Shot.model_validate(payload)


def apply_shot_reviews(
    *,
    session_id: UUID,
    shots: tuple[Shot, ...],
    decisions: tuple[ShotReviewDecision, ...],
) -> ReviewApplication:
    """Apply exactly one explicit, typed decision to every session shot."""

    _require_session_shots(session_id, shots)
    validated_decisions = tuple(
        ShotReviewDecision.model_validate(decision.model_dump(mode="python"))
        for decision in decisions
    )
    shot_ids = {shot.id for shot in shots}
    decision_ids = [decision.shot_id for decision in validated_decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("review decisions must not duplicate a shot")
    if set(decision_ids) != shot_ids:
        raise ValueError("review decisions must cover the exact session shot set")
    if any(
        AccessActorRole.REVIEWER not in decision.reviewer.roles for decision in validated_decisions
    ):
        raise ValueError("every decision requires a reviewer with the REVIEWER role")

    by_shot = {decision.shot_id: decision for decision in validated_decisions}
    if any(
        shot.trajectory is not None and not by_shot[shot.id].trajectory_reviewed for shot in shots
    ):
        raise ValueError("a visible trajectory requires explicit trajectory review")
    evidence = review_session_evidence(session_id=session_id, shots=shots)
    reviewed_shots = tuple(_reviewed_shot(shot, by_shot[shot.id]) for shot in shots)
    corrections = tuple(
        correction for decision in validated_decisions for correction in decision.corrections
    )
    return ReviewApplication(
        session_id=session_id,
        source_shots=shots,
        shots=reviewed_shots,
        decisions=validated_decisions,
        corrections=corrections,
        evidence=evidence,
    )


def _require_utc_completion(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("review completion time must use UTC")
    return value.astimezone(UTC)


def _validate_submission_inputs(
    report: ReportDocument,
    review_task: ReviewTask,
    application: ReviewApplication,
    completed_at: datetime,
) -> None:
    if type(review_task.report_revision) is not int:
        raise ValueError("review task revision must be a real integer")
    if review_task.status is not ReviewTaskStatus.IN_REVIEW:
        raise ValueError("only an in-review task may be submitted")
    if report.report_status is not ReportStatus.DRAFT_PENDING_REVIEW:
        raise ValueError("only a draft report may be reviewed")
    if (
        review_task.session_id != report.session_id
        or review_task.report_id != report.report_id
        or review_task.report_revision != report.revision
    ):
        raise ValueError("review task must bind the exact report session and revision")
    if application.session_id != report.session_id:
        raise ValueError("review application session must match the report")
    _require_session_shots(application.session_id, application.source_shots)
    _require_session_shots(application.session_id, application.shots)
    if len(application.shots) != report.session_statistics.total_shot_count:
        raise ValueError("review application must contain the complete report shot set")
    decision_ids = [decision.shot_id for decision in application.decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != {
        shot.id for shot in application.shots
    }:
        raise ValueError("review application decisions must match its exact shot set")
    expected_corrections = tuple(
        correction for decision in application.decisions for correction in decision.corrections
    )
    if application.corrections != expected_corrections:
        raise ValueError("review application correction records are inconsistent")
    expected_status = {
        decision.shot_id: ReviewStatus(decision.outcome.value) for decision in application.decisions
    }
    if any(shot.review_status is not expected_status[shot.id] for shot in application.shots):
        raise ValueError("review application shot results are inconsistent with decisions")
    source_by_id = {shot.id: shot for shot in application.source_shots}
    if set(source_by_id) != set(expected_status):
        raise ValueError("review application source shots must match its decisions")
    decisions_by_id = {decision.shot_id: decision for decision in application.decisions}
    expected_shots = tuple(
        _reviewed_shot(source_by_id[shot.id], decisions_by_id[shot.id])
        for shot in application.shots
    )
    if any(
        actual.model_dump(mode="python") != expected.model_dump(mode="python")
        for actual, expected in zip(application.shots, expected_shots, strict=True)
    ):
        raise ValueError("review application contains a field change not declared by decisions")
    if any(
        AccessActorRole.REVIEWER not in decision.reviewer.roles
        for decision in application.decisions
    ):
        raise ValueError("review application requires REVIEWER-authorized decisions")
    expected_evidence = review_session_evidence(
        session_id=application.session_id,
        shots=application.source_shots,
    )
    if application.evidence != expected_evidence:
        raise ValueError("review application evidence must match the exact source shots")
    if completed_at < report.generated_at or any(
        completed_at < decision.reviewed_at for decision in application.decisions
    ):
        raise ValueError("review completion cannot predate report generation or a decision")


def _final_report(document: ReportDocument, status: ReportStatus) -> ReportDocument:
    return ReportDocument.model_validate(
        document.model_dump(mode="python") | {"report_status": status}
    )


def submit_review(
    *,
    report: ReportDocument,
    review_task: ReviewTask,
    application: ReviewApplication,
    completed_at: datetime,
    corrected_report_id: UUID | None = None,
    replacements: ReportRevisionReplacements | None = None,
) -> ReviewSubmission:
    """Bind an approved same revision or a corrected child revision to its task."""

    completed_at = _require_utc_completion(completed_at)
    _validate_submission_inputs(report, review_task, application, completed_at)
    trajectory_calibrations = {
        shot.trajectory_calibration
        for shot in application.source_shots
        if shot.trajectory is not None
    }
    if None in trajectory_calibrations or len(trajectory_calibrations) > 1:
        raise ValueError("review submission requires one exact trajectory calibration lineage")
    trajectory_calibration = next(iter(trajectory_calibrations), None)
    if (
        trajectory_calibration is not None
        and trajectory_calibration.status is not report.calibration_status
    ):
        raise ValueError("reviewed trajectory calibration must match the current report")
    if application.corrections:
        if not isinstance(corrected_report_id, UUID) or corrected_report_id == report.report_id:
            raise ValueError("a correction requires a new report id")
        replacement = replacements or ReportRevisionReplacements()
        child = revise_report_document(
            report,
            report_id=corrected_report_id,
            generated_at=completed_at,
            schema_version="1.0" if report.schema_version == "1.1" else None,
            quality_summary=replacement.quality_summary,
            session_statistics=replacement.session_statistics,
            dimension_scores=replacement.dimension_scores,
            typical_shots=replacement.typical_shots,
            coaching_recommendations=replacement.coaching_recommendations,
            headline=replacement.headline,
            trust_scope=replacement.trust_scope,
            objective_statistics=replacement.objective_statistics,
            typical_shot_analysis=replacement.typical_shot_analysis,
            training_recommendation=replacement.training_recommendation,
            quality_notice=replacement.quality_notice,
        )
        final_report = _final_report(child, ReportStatus.CORRECTED)
        task_status = ReviewTaskStatus.CORRECTED
    else:
        if corrected_report_id is not None or replacements is not None:
            raise ValueError("an approval cannot create or replace a report revision")
        final_report = _final_report(report, ReportStatus.APPROVED)
        task_status = ReviewTaskStatus.APPROVED

    final_task = ReviewTask(
        id=review_task.id,
        session_id=final_report.session_id,
        report_id=final_report.report_id,
        report_revision=final_report.revision,
        status=task_status,
    )
    return ReviewSubmission(
        report=final_report,
        review_task=final_task,
        shots=application.shots,
        decisions=application.decisions,
        corrections=application.corrections,
        evidence=application.evidence,
        completed_at=completed_at,
        calibration=trajectory_calibration,
    )
