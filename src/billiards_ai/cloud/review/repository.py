"""Approved-only repository boundary for downstream consumers."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.cloud.review.errors import ReportNotDeliverableError
from billiards_ai.cloud.review.gate import require_deliverable_report
from billiards_ai.shared.models.report import ReportDocument
from billiards_ai.shared.models.review import (
    ApprovedSessionData,
    ApprovedShotData,
    ReportNotDeliverableReason,
    ReviewTask,
)
from billiards_ai.shared.models.shot import Shot


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class ApprovedDataRepository(Protocol):
    """The only public data source for reports, profiles, and trends."""

    def get_approved_report(self, report_id: UUID, revision: int) -> ReportDocument: ...

    def get_approved_session_data(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> ApprovedSessionData: ...


class InMemoryApprovedDataRepository:
    """Pure Phase 0 implementation; every read re-evaluates the gate."""

    def __init__(
        self,
        *,
        reports: tuple[ReportDocument, ...],
        review_tasks: tuple[ReviewTask, ...],
        shots: tuple[Shot, ...],
    ):
        self._reports = reports
        self._review_tasks = review_tasks
        self._shots = shots

    def _validated_reports(self) -> tuple[ReportDocument, ...]:
        validated: list[ReportDocument] = []
        for report in self._reports:
            try:
                validated.append(
                    ReportDocument.model_validate(_stored_model_values(report), strict=True)
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ReportNotDeliverableError(
                    ReportNotDeliverableReason.REPORT_NOT_FOUND
                ) from exc
        return tuple(validated)

    def _validated_tasks(self) -> tuple[ReviewTask, ...]:
        validated: list[ReviewTask] = []
        for task in self._review_tasks:
            try:
                validated.append(ReviewTask.model_validate(_stored_model_values(task), strict=True))
            except (TypeError, ValueError, ValidationError) as exc:
                raise ReportNotDeliverableError(
                    ReportNotDeliverableReason.REVIEW_TASK_NOT_FOUND
                ) from exc
        return tuple(validated)

    def _validated_session_shots(self, session_id: UUID) -> tuple[Shot, ...]:
        validated: list[Shot] = []
        for shot in self._shots:
            if getattr(shot, "session_id", None) != session_id:
                continue
            try:
                validated.append(Shot.model_validate(_stored_model_values(shot), strict=True))
            except (TypeError, ValueError, ValidationError) as exc:
                raise ReportNotDeliverableError(
                    ReportNotDeliverableReason.SHOT_SET_INCOMPLETE
                ) from exc
        return tuple(validated)

    def _approved(
        self,
        report_id: UUID,
        revision: int,
        *,
        reports: tuple[ReportDocument, ...] | None = None,
    ) -> tuple[ReportDocument, tuple[Shot, ...]]:
        if type(revision) is not int or revision < 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_FOUND)
        validated_reports = reports if reports is not None else self._validated_reports()
        reports = tuple(
            report
            for report in validated_reports
            if report.report_id == report_id and report.revision == revision
        )
        if len(reports) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_FOUND)
        report = reports[0]
        tasks = tuple(
            task
            for task in self._validated_tasks()
            if task.report_id == report_id and task.report_revision == revision
        )
        if len(tasks) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVIEW_TASK_NOT_FOUND)
        shots = self._validated_session_shots(report.session_id)
        require_deliverable_report(report, tasks[0], shots)
        return report, shots

    def get_approved_report(self, report_id: UUID, revision: int) -> ReportDocument:
        report, _shots = self._approved(report_id, revision)
        return report

    def get_approved_session_data(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> ApprovedSessionData:
        reports = self._validated_reports()
        requested = tuple(
            report
            for report in reports
            if report.report_id == report_id and report.revision == revision
        )
        if len(requested) != 1:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REPORT_NOT_FOUND)
        report = requested[0]
        if report.session_id != session_id:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.SESSION_MISMATCH)
        session_reports = tuple(
            candidate for candidate in reports if candidate.session_id == session_id
        )
        highest_revision = max(candidate.revision for candidate in session_reports)
        head_identities = {
            (candidate.report_id, candidate.revision)
            for candidate in session_reports
            if candidate.revision == highest_revision
        }
        if head_identities != {(report_id, revision)}:
            raise ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
        report, shots = self._approved(report_id, revision, reports=reports)
        return ApprovedSessionData(
            session_id=session_id,
            report=report,
            shots=tuple(self._project_shot(shot) for shot in shots),
        )

    @staticmethod
    def _project_shot(shot: Shot) -> ApprovedShotData:
        return ApprovedShotData(
            shot_id=shot.id,
            session_id=shot.session_id,
            sequence=shot.sequence,
            struck_ball=shot.struck_ball,
            potted_balls=tuple(shot.potted_balls or ()),
            cushion_count=shot.cushion_count,
            travel_distance_mm=shot.travel_distance_mm,
            is_foul=shot.is_foul,
            deviation_mm=shot.deviation_mm,
            target_zone_source=shot.target_zone_source,
            confidence=shot.confidence,
            analysis_status=shot.analysis_status,
            review_status=shot.review_status,
        )
