"""Narrow approved visualization projection over the unified delivery gate."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.gate import require_deliverable_report
from billiards_ai.cloud.review.service import ReviewSubmission
from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.report import ReportDocument
from billiards_ai.shared.models.review import (
    ApprovedActualPath,
    ApprovedActualPathReason,
    ApprovedActualPathStatus,
    ApprovedVisualizationProjection,
    ReportNotDeliverableReason,
    ReviewTask,
    VisualizationReviewAuthorization,
)
from billiards_ai.shared.models.segmentation import BallPosition
from billiards_ai.shared.models.shot import ReviewStatus, Shot


class ApprovedVisualizationRepository(Protocol):
    def get_approved_visualization_projection(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
        shot_id: UUID,
    ) -> ApprovedVisualizationProjection: ...


class InMemoryApprovedVisualizationRepository:
    """Phase 0 projection; every lookup re-runs AD-16 authorization."""

    def __init__(
        self,
        *,
        reports: tuple[ReportDocument, ...],
        review_tasks: tuple[ReviewTask, ...],
        shots: tuple[Shot, ...],
        review_submissions: tuple[ReviewSubmission, ...],
    ):
        self._reports = reports
        self._review_tasks = review_tasks
        self._shots = shots
        self._review_submissions = tuple(
            ReviewSubmission(
                report=submission.report.model_copy(deep=True),
                review_task=submission.review_task.model_copy(deep=True),
                shots=tuple(
                    Shot.model_validate(shot.model_dump(mode="python")) for shot in submission.shots
                ),
                decisions=tuple(
                    decision.model_copy(deep=True) for decision in submission.decisions
                ),
                corrections=tuple(
                    correction.model_copy(deep=True) for correction in submission.corrections
                ),
                evidence=submission.evidence.model_copy(deep=True),
                completed_at=submission.completed_at,
                calibration=submission.calibration,
            )
            for submission in review_submissions
        )

    def _approved(
        self,
        report_id: UUID,
        revision: int,
    ) -> tuple[ReportDocument, tuple[Shot, ...]]:
        if type(revision) is not int or revision < 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_FOUND)
        reports = tuple(
            report
            for report in self._reports
            if report.report_id == report_id and report.revision == revision
        )
        if len(reports) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_FOUND)
        report = reports[0]
        session_reports = tuple(
            candidate for candidate in self._reports if candidate.session_id == report.session_id
        )
        current_revision = max(candidate.revision for candidate in session_reports)
        current_heads = tuple(
            candidate for candidate in session_reports if candidate.revision == current_revision
        )
        if len(current_heads) != 1 or current_heads[0].report_id != report.report_id:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        tasks = tuple(
            task
            for task in self._review_tasks
            if task.report_id == report_id and task.report_revision == revision
        )
        if len(tasks) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVIEW_TASK_NOT_FOUND)
        shots = tuple(shot for shot in self._shots if shot.session_id == report.session_id)
        require_deliverable_report(report, tasks[0], shots)
        return report, shots

    @staticmethod
    def _same_shot(left: Shot, right: Shot) -> bool:
        return left.model_dump(mode="json") == right.model_dump(mode="json")

    def _authorizations(
        self,
        *,
        report: ReportDocument,
        task: ReviewTask,
        shot: Shot,
        shots: tuple[Shot, ...],
    ) -> tuple[VisualizationReviewAuthorization, ...]:
        result: list[VisualizationReviewAuthorization] = []
        for submission in self._review_submissions:
            if submission.report != report or submission.review_task != task:
                continue
            if len(submission.shots) != len(shots) or any(
                not self._same_shot(current, submitted)
                for current, submitted in zip(shots, submission.shots, strict=True)
            ):
                continue
            evidence_items = tuple(
                item for item in submission.evidence.shots if item.shot_id == shot.id
            )
            decisions = tuple(
                decision for decision in submission.decisions if decision.shot_id == shot.id
            )
            if len(evidence_items) != 1 or len(decisions) != 1:
                continue
            evidence = evidence_items[0]
            decision = decisions[0]
            quality = shot.field_quality.trajectory if shot.field_quality is not None else None
            canonical_path = tuple(shot.trajectory or ())
            if (
                evidence.sequence != shot.sequence
                or evidence.shot_created_at != shot.created_at
                or evidence.actual_path.points != canonical_path
                or quality is None
                or evidence.actual_path.source is not quality.source
                or evidence.actual_path.confidence != quality.confidence
                or evidence.actual_path.calibration
                != (shot.trajectory_calibration if canonical_path else None)
                or evidence.actual_path.unavailable_reason is not None
                and evidence.actual_path.unavailable_reason.value
                != (
                    None
                    if canonical_path or shot.trajectory_unavailable_reason is None
                    else shot.trajectory_unavailable_reason.value
                )
                or evidence.actual_path.unavailable_reason is None
                and not canonical_path
                and shot.trajectory_unavailable_reason is not None
                or submission.calibration != shot.trajectory_calibration
                or submission.completed_at < decision.reviewed_at
            ):
                continue
            result.append(
                VisualizationReviewAuthorization(
                    review_task_id=task.id,
                    session_id=report.session_id,
                    report_id=report.report_id,
                    report_revision=report.revision,
                    shot_id=shot.id,
                    shot_sequence=shot.sequence,
                    shot_created_at=shot.created_at,
                    actual_path=evidence.actual_path,
                    decision=decision,
                )
            )
        return tuple(result)

    @staticmethod
    def _actual_path(
        shot: Shot,
        authorizations: tuple[VisualizationReviewAuthorization, ...],
    ) -> ApprovedActualPath:
        if len(authorizations) != 1 or shot.review_status not in {
            ReviewStatus.APPROVED,
            ReviewStatus.CORRECTED,
        }:
            return ApprovedActualPath(
                status=ApprovedActualPathStatus.NOT_REVIEWED,
                reason=ApprovedActualPathReason.NOT_REVIEWED,
            )
        if not shot.trajectory:
            reason = {
                "INSUFFICIENT_POINTS": ApprovedActualPathReason.INSUFFICIENT_POINTS,
                "LOW_CONFIDENCE": ApprovedActualPathReason.LOW_CONFIDENCE,
                "NOT_OBSERVED": ApprovedActualPathReason.NOT_OBSERVED,
            }.get(
                (
                    shot.trajectory_unavailable_reason.value
                    if shot.trajectory_unavailable_reason is not None
                    else "NOT_OBSERVED"
                ),
                ApprovedActualPathReason.NOT_OBSERVED,
            )
            return ApprovedActualPath(
                status=(
                    ApprovedActualPathStatus.LOW_CONFIDENCE
                    if reason is ApprovedActualPathReason.LOW_CONFIDENCE
                    else ApprovedActualPathStatus.UNAVAILABLE
                ),
                reason=reason,
            )
        if len(set(shot.trajectory)) < 2:
            return ApprovedActualPath(
                status=ApprovedActualPathStatus.UNAVAILABLE,
                reason=ApprovedActualPathReason.INSUFFICIENT_POINTS,
            )
        quality = shot.field_quality.trajectory if shot.field_quality is not None else None
        if quality is None or quality.confidence < CONFIDENCE_ABSTAIN_BELOW:
            return ApprovedActualPath(
                status=ApprovedActualPathStatus.LOW_CONFIDENCE,
                reason=ApprovedActualPathReason.LOW_CONFIDENCE,
            )
        return ApprovedActualPath(
            status=ApprovedActualPathStatus.AVAILABLE,
            points=tuple(shot.trajectory),
            source=quality.source,
            confidence=quality.confidence,
        )

    def get_approved_visualization_projection(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
        shot_id: UUID,
    ) -> ApprovedVisualizationProjection:
        report, shots = self._approved(report_id, revision)
        if report.session_id != session_id:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.SESSION_MISMATCH)
        selected = tuple(shot for shot in shots if shot.id == shot_id)
        if len(selected) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.SHOT_SET_INCOMPLETE)
        shot = selected[0]
        task = next(
            task
            for task in self._review_tasks
            if task.report_id == report_id and task.report_revision == revision
        )
        authorizations = self._authorizations(
            report=report,
            task=task,
            shot=shot,
            shots=shots,
        )
        calibration = shot.trajectory_calibration
        if (
            calibration is None
            or calibration.status is not report.calibration_status
            or calibration.status.value != "ACTIVE"
        ):
            authorizations = ()
        positions = shot.ball_positions_before or {}
        if calibration is None:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        return ApprovedVisualizationProjection(
            session_id=session_id,
            report_id=report_id,
            report_revision=revision,
            shot_id=shot.id,
            shot_sequence=shot.sequence,
            struck_ball=shot.struck_ball,
            ball_positions=tuple(
                BallPosition(ball_id=ball_id, position=point)
                for ball_id, point in sorted(positions.items())
            ),
            actual_path=self._actual_path(shot, authorizations),
            calibration_id=calibration.calibration_id,
            calibration_revision_id=calibration.revision_id,
            calibration_revision_number=calibration.revision_number,
            calibration_status=calibration.status,
            table_profile_id=calibration.table_profile_id,
            table_length_mm=calibration.table_length_mm,
            table_width_mm=calibration.table_width_mm,
            table_configuration_fingerprint=calibration.table_configuration_fingerprint,
        )
