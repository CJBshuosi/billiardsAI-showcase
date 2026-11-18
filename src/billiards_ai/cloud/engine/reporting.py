"""Pure report rendering and immutable revision helpers for Story 3.3."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.shared.models.calibration import CalibrationStatus
from billiards_ai.shared.models.quality import SessionQualitySummary
from billiards_ai.shared.models.recommendation import (
    RecommendationResult,
    RecommendationStatus,
)
from billiards_ai.shared.models.report import (
    AbstentionSummary,
    CalibratedReportSessionStatistics,
    CoachingRecommendation,
    ObjectiveStatisticsSection,
    ReportCompositionError,
    ReportCompositionReason,
    ReportDocument,
    ReportHeadline,
    ReportHeadlineCode,
    ReportQualityNotice,
    ReportSessionStatistics,
    ReportSourceLineage,
    ReportStatus,
    ReportTargetLineage,
    ReportTrustScope,
    ReportVisualizationAssetReference,
    ReportVisualizationResult,
    TrainingRecommendationSection,
    TypicalShotAnalysisSection,
    TypicalShotNoDataReason,
    TypicalShotReportCard,
    TypicalShotSectionStatus,
    TypicalShotSummary,
    objective_statistics_digest,
)
from billiards_ai.shared.models.review import (
    ApprovedSessionData,
    ApprovedShotData,
    ReviewTask,
    ReviewTaskStatus,
)
from billiards_ai.shared.models.scoring import SessionPerformanceScores
from billiards_ai.shared.models.statistics import SessionStatistics
from billiards_ai.shared.models.typical_shot import TypicalShotSelectionResult
from billiards_ai.shared.models.visualization import (
    VisualizationAsset,
    VisualizationBindingStatus,
    VisualizationFailureReason,
    VisualizationResult,
    VisualizationSourceLineage,
    VisualizationStatus,
)


class ReportCompositionResult(BaseModel):
    model_config = {"extra": "forbid", "frozen": True, "strict": True}

    report: ReportDocument
    review_task: ReviewTask


class _ApprovedSnapshotRepository:
    """Expose one gated projection to Story 5.4 without a second repository read."""

    def __init__(self, approved: ApprovedSessionData):
        self._approved = approved

    def get_approved_session_data(
        self, session_id: UUID, report_id: UUID, revision: int
    ) -> ApprovedSessionData:
        report = self._approved.report
        if (
            session_id != self._approved.session_id
            or report_id != report.report_id
            or revision != report.revision
        ):
            raise ValueError("snapshot lineage mismatch")
        return self._approved


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    if not isinstance(value, BaseModel):
        raise TypeError("composition inputs must be Pydantic models")
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _common_statistics(statistics: SessionStatistics) -> dict[str, object]:
    return {
        "total_shot_count": statistics.total_shot_count,
        "valid_shot_count": statistics.valid_shot_count,
        "potted_ball_count": statistics.potted_ball_count,
        "successful_potting_shot_count": statistics.successful_potting_shot_count,
        "potting_shot_count": statistics.potting_shot_count,
        "pot_rate": statistics.pot_rate,
        "cushion_distribution": statistics.cushion_distribution,
        "cushion_count_shot_count": statistics.cushion_count_shot_count,
    }


def _project_statistics(
    statistics: SessionStatistics,
    calibration_status: CalibrationStatus,
) -> ReportSessionStatistics:
    common = _common_statistics(statistics)
    if calibration_status is not CalibrationStatus.ACTIVE:
        return ReportSessionStatistics(**common)
    return CalibratedReportSessionStatistics(
        **common,
        average_cue_ball_travel_distance_mm=statistics.average_cue_ball_travel_distance_mm,
        cue_ball_travel_shot_count=statistics.cue_ball_travel_shot_count,
        cue_ball_end_distribution=statistics.cue_ball_end_distribution,
        cue_ball_end_shot_count=statistics.cue_ball_end_shot_count,
        target_zone_deviation_by_source=statistics.target_zone_deviation_by_source,
    )


def render_report_document(
    *,
    report_id: UUID,
    session_id: UUID,
    generated_at: datetime,
    analysis_version: str,
    calibration_status: CalibrationStatus,
    statistics: SessionStatistics,
    dimension_scores: SessionPerformanceScores,
) -> ReportDocument:
    """Render the first immutable report revision without I/O, clocks, or identifiers."""

    abstained = statistics.session_quality.abstained_shot_count
    return ReportDocument(
        report_id=report_id,
        revision=1,
        parent_revision_id=None,
        session_id=session_id,
        generated_at=generated_at,
        analysis_version=analysis_version,
        calibration_status=calibration_status,
        quality_summary=statistics.session_quality,
        session_statistics=_project_statistics(statistics, calibration_status),
        dimension_scores=dimension_scores,
        abstention_summary=AbstentionSummary(
            total_shot_count=statistics.total_shot_count,
            abstained_shot_count=abstained,
            display_text=f"{abstained} 杆超出当前分析范围",
        ),
    )


def revise_report_document(
    document: ReportDocument,
    *,
    report_id: UUID,
    generated_at: datetime,
    quality_summary: SessionQualitySummary | None = None,
    session_statistics: ReportSessionStatistics | None = None,
    dimension_scores: SessionPerformanceScores | None = None,
    typical_shots: tuple[TypicalShotSummary, ...] | None = None,
    coaching_recommendations: tuple[CoachingRecommendation, ...] | None = None,
    schema_version: str | None = None,
    headline: ReportHeadline | None = None,
    trust_scope: ReportTrustScope | None = None,
    objective_statistics: ObjectiveStatisticsSection | None = None,
    typical_shot_analysis: TypicalShotAnalysisSection | None = None,
    training_recommendation: TrainingRecommendationSection | None = None,
    quality_notice: ReportQualityNotice | None = None,
) -> ReportDocument:
    """Create a child revision while leaving the parent document untouched."""

    if report_id == document.report_id:
        raise ValueError("a new revision requires a new report id")
    if generated_at < document.generated_at:
        raise ValueError("a new revision cannot predate its parent")

    payload = document.model_dump(mode="python")
    payload.update(
        report_id=report_id,
        revision=document.revision + 1,
        parent_revision_id=document.report_id,
        generated_at=generated_at,
        report_status=ReportStatus.DRAFT_PENDING_REVIEW,
    )
    replacements = {
        "quality_summary": quality_summary,
        "session_statistics": session_statistics,
        "dimension_scores": dimension_scores,
        "typical_shots": typical_shots,
        "coaching_recommendations": coaching_recommendations,
        "schema_version": schema_version,
        "headline": headline,
        "trust_scope": trust_scope,
        "objective_statistics": objective_statistics,
        "typical_shot_analysis": typical_shot_analysis,
        "training_recommendation": training_recommendation,
        "quality_notice": quality_notice,
    }
    payload.update({name: value for name, value in replacements.items() if value is not None})
    if schema_version == "1.0" and document.schema_version == "1.1":
        payload.update(
            schema_version="1.0",
            typical_shots=(),
            coaching_recommendations=(),
            headline=None,
            trust_scope=None,
            objective_statistics=None,
            typical_shot_analysis=None,
            training_recommendation=None,
            quality_notice=None,
            content_sha256=None,
        )
    if payload["schema_version"] == "1.1":
        payload.pop("content_sha256", None)
        return ReportDocument.build_v11(**payload)
    return ReportDocument.model_validate(payload)


_CAPABILITY_FAILURES = {
    VisualizationStatus.CALIBRATION_NOT_ACTIVE: {VisualizationFailureReason.CALIBRATION_NOT_ACTIVE},
    VisualizationStatus.ACTUAL_PATH_UNAVAILABLE: {
        VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED,
        VisualizationFailureReason.ACTUAL_PATH_INSUFFICIENT_POINTS,
        VisualizationFailureReason.ACTUAL_PATH_LOW_CONFIDENCE,
        VisualizationFailureReason.ACTUAL_PATH_NOT_REVIEWED,
    },
    VisualizationStatus.IDEAL_PATH_UNSUPPORTED: {
        VisualizationFailureReason.IDEAL_PATH_NOT_SUCCEEDED,
        VisualizationFailureReason.IDEAL_INTENT_NOT_EXPLICIT,
        VisualizationFailureReason.IDEAL_INTENT_NOT_REVIEWED,
        VisualizationFailureReason.IDEAL_CALIBRATION_NOT_ACTIVE,
        VisualizationFailureReason.IDEAL_TARGET_BALL_MISSING,
        VisualizationFailureReason.IDEAL_ENGINE_UNAVAILABLE,
        VisualizationFailureReason.IDEAL_ADAPTER_FAILURE,
    },
}


def _normalize_approved_data(
    value: object,
    *,
    session_id: UUID,
    report_id: UUID,
    revision: int,
) -> ApprovedSessionData:
    if not isinstance(value, ApprovedSessionData):
        raise TypeError("repository must return ApprovedSessionData")
    ApprovedSessionData.model_validate(_stored_model_values(value))
    report = ReportDocument.model_validate(_stored_model_values(value.report))
    shots = tuple(
        ApprovedShotData.model_validate(_stored_model_values(shot)) for shot in value.shots
    )
    expected_count = report.session_statistics.total_shot_count
    if (
        value.session_id != session_id
        or report.session_id != session_id
        or report.report_id != report_id
        or report.revision != revision
        or len(shots) != expected_count
        or len({shot.shot_id for shot in shots}) != expected_count
        or {shot.sequence for shot in shots} != set(range(1, expected_count + 1))
        or any(shot.session_id != session_id for shot in shots)
    ):
        raise ValueError("approved projection does not match requested complete lineage")
    return ApprovedSessionData(
        session_id=session_id,
        report=report,
        shots=tuple(sorted(shots, key=lambda shot: (shot.sequence, str(shot.shot_id)))),
    )


def _validated_selection(value: object) -> TypicalShotSelectionResult:
    if not isinstance(value, TypicalShotSelectionResult):
        raise TypeError("selection must be a TypicalShotSelectionResult")
    return TypicalShotSelectionResult.model_validate(_stored_model_values(value))


def _validated_recommendation(value: object) -> RecommendationResult:
    if not isinstance(value, RecommendationResult):
        raise TypeError("recommendation must be a RecommendationResult")
    return RecommendationResult.model_validate(_stored_model_values(value))


def _validated_visualizations(
    values: object,
) -> tuple[ReportVisualizationResult, ...]:
    if not isinstance(values, tuple):
        raise TypeError("visualization results must be a tuple")
    normalized = []
    for value in values:
        item = ReportVisualizationResult.model_validate(_stored_model_values(value))
        result = VisualizationResult.model_validate(_stored_model_values(item.result))
        if result.asset is not None:
            asset = VisualizationAsset.model_validate(_stored_model_values(result.asset))
            result = VisualizationResult(status=result.status, reason=result.reason, asset=asset)
        normalized.append(
            ReportVisualizationResult(shot_id=item.shot_id, source=item.source, result=result)
        )
    return tuple(normalized)


def _unsupported_dimensions(scores: SessionPerformanceScores) -> tuple[str, ...]:
    return tuple(sorted(name for name, score in scores if score.status.value == "UNSUPPORTED"))


def compose_coaching_report(
    *,
    repository: object,
    session_id: object,
    parent_report_id: object,
    parent_revision: object,
    child_report_id: object,
    child_review_task_id: object,
    generated_at: object,
    selection_result: object,
    visualization_results: object,
    recommendation_result: object,
) -> ReportCompositionResult:
    """Validate all approved Story 5 inputs, then purely construct child report/task."""

    from billiards_ai.cloud.engine.rules import generate_recommendations
    from billiards_ai.cloud.engine.typical_shot_selector import _select_typical_shots
    from billiards_ai.cloud.engine.visualization import bind_visualization_asset
    from billiards_ai.cloud.review.errors import ReportNotDeliverableError

    try:
        get_approved = getattr(repository, "get_approved_session_data", None)
    except Exception as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
    if (
        not callable(get_approved)
        or not isinstance(session_id, UUID)
        or session_id.version != 4
        or not isinstance(parent_report_id, UUID)
        or parent_report_id.version != 4
        or type(parent_revision) is not int
        or parent_revision < 1
        or not isinstance(child_report_id, UUID)
        or child_report_id.version != 4
        or not isinstance(child_review_task_id, UUID)
        or child_review_task_id.version != 4
        or child_report_id == parent_report_id
        or not isinstance(generated_at, datetime)
        or generated_at.tzinfo is None
        or generated_at.utcoffset() != UTC.utcoffset(generated_at)
    ):
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT)
    try:
        projection = get_approved(session_id, parent_report_id, parent_revision)
    except ReportNotDeliverableError:
        raise
    except Exception as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
    try:
        approved = _normalize_approved_data(
            projection,
            session_id=session_id,
            report_id=parent_report_id,
            revision=parent_revision,
        )
        selection = _validated_selection(selection_result)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
    try:
        authoritative_selection = _select_typical_shots(approved, selection.rule_version)
    except ReportNotDeliverableError:
        raise
    except Exception as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
    if selection != authoritative_selection:
        raise ReportCompositionError(ReportCompositionReason.SELECTION_MISMATCH)

    try:
        recommendation = _validated_recommendation(recommendation_result)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ReportCompositionError(ReportCompositionReason.RECOMMENDATION_MISMATCH) from exc
    try:
        authoritative_recommendation = generate_recommendations(
            _ApprovedSnapshotRepository(approved),
            session_id,
            parent_report_id,
            parent_revision,
            authoritative_selection,
            recommendation.catalog_version,
        )
    except ReportNotDeliverableError:
        raise
    except Exception as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
    if authoritative_recommendation.status in {
        RecommendationStatus.SELECTION_MISMATCH,
        RecommendationStatus.INVALID_INPUT,
    }:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT)
    if recommendation != authoritative_recommendation:
        raise ReportCompositionError(ReportCompositionReason.RECOMMENDATION_MISMATCH)

    try:
        visualizations = _validated_visualizations(visualization_results)
    except Exception as exc:
        raise ReportCompositionError(ReportCompositionReason.ASSET_INTEGRITY_MISMATCH) from exc
    selected_ids = tuple(item.shot_id for item in authoritative_selection.selections)
    result_ids = tuple(item.shot_id for item in visualizations)
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(selected_ids):
        raise ReportCompositionError(ReportCompositionReason.RESULT_SET_MISMATCH)
    by_shot = {item.shot_id: item for item in visualizations}
    source_lineage = ReportSourceLineage(
        session_id=session_id,
        report_id=parent_report_id,
        report_revision=parent_revision,
    )
    target_lineage = ReportTargetLineage(
        session_id=session_id,
        report_id=child_report_id,
        report_revision=parent_revision + 1,
    )
    cards = []
    for selected in authoritative_selection.selections:
        visualization_item = by_shot[selected.shot_id]
        visualization = visualization_item.result
        expected_visualization_source = VisualizationSourceLineage(
            session_id=session_id,
            report_id=parent_report_id,
            report_revision=parent_revision,
            shot_id=selected.shot_id,
        )
        if visualization_item.source != expected_visualization_source:
            raise ReportCompositionError(ReportCompositionReason.ASSET_INTEGRITY_MISMATCH)
        reference = None
        if visualization.status is VisualizationStatus.SUCCEEDED:
            asset = visualization.asset
            if (
                asset is None
                or asset.source != expected_visualization_source
                or asset.selection.rule_version != selected.rule_version.value
                or asset.selection.selection_reason is not selected.selection_reason
            ):
                raise ReportCompositionError(ReportCompositionReason.ASSET_INTEGRITY_MISMATCH)
            binding_result = bind_visualization_asset(
                asset,
                session_id=session_id,
                report_id=parent_report_id,
                revision=parent_revision,
                shot_id=selected.shot_id,
            )
            if (
                binding_result.status is not VisualizationBindingStatus.SUCCEEDED
                or binding_result.binding is None
            ):
                raise ReportCompositionError(ReportCompositionReason.ASSET_INTEGRITY_MISMATCH)
            reference = ReportVisualizationAssetReference(
                asset_id=asset.asset_id,
                asset_content_sha256=asset.content_sha256,
                spec_version=asset.spec_version,
                text_summary=asset.text_summary,
                actual=asset.actual,
                ideal=asset.ideal,
                source_approved_lineage=source_lineage,
                target_report_lineage=target_lineage,
                exact_source_binding=binding_result.binding,
            )
        elif (
            visualization.status not in _CAPABILITY_FAILURES
            or visualization.reason not in _CAPABILITY_FAILURES[visualization.status]
            or (
                visualization.status is VisualizationStatus.CALIBRATION_NOT_ACTIVE
                and approved.report.calibration_status is CalibrationStatus.ACTIVE
            )
        ):
            raise ReportCompositionError(ReportCompositionReason.ASSET_INTEGRITY_MISMATCH)
        cards.append(
            TypicalShotReportCard(
                shot_id=selected.shot_id,
                sequence=selected.sequence,
                selection_rank=selected.selection_rank,
                selection_reason=selected.selection_reason,
                rule_version=selected.rule_version,
                visualization_status=visualization.status,
                visualization_reason=visualization.reason,
                asset_reference=reference,
            )
        )

    typical_status = (
        TypicalShotSectionStatus.NO_RELIABLE_DATA
        if not cards
        else TypicalShotSectionStatus.READY
        if all(card.asset_reference is not None for card in cards)
        else TypicalShotSectionStatus.PARTIAL
    )
    typical_section = TypicalShotAnalysisSection(
        status=typical_status,
        no_data_reason=(TypicalShotNoDataReason.NO_RELIABLE_SELECTION if not cards else None),
        cards=tuple(cards),
    )
    headline_code = (
        ReportHeadlineCode.NO_RELIABLE_COACHING
        if not cards and recommendation.status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION
        else ReportHeadlineCode.READY
        if typical_status is TypicalShotSectionStatus.READY
        and recommendation.status is RecommendationStatus.READY
        else ReportHeadlineCode.PARTIAL
    )
    unsupported = _unsupported_dimensions(approved.report.dimension_scores)
    shot_ids = tuple(shot.shot_id for shot in approved.shots)
    objective = ObjectiveStatisticsSection(
        source=source_lineage,
        analysis_version=approved.report.analysis_version,
        shot_ids=shot_ids,
        content_sha256=objective_statistics_digest(
            source=source_lineage,
            analysis_version=approved.report.analysis_version,
            shot_ids=shot_ids,
            quality_summary=approved.report.quality_summary,
            session_statistics=approved.report.session_statistics,
            dimension_scores=approved.report.dimension_scores,
            abstention_summary=approved.report.abstention_summary,
        ),
    )
    try:
        child = revise_report_document(
            approved.report,
            report_id=child_report_id,
            generated_at=generated_at,
            schema_version="1.1",
            headline=ReportHeadline(code=headline_code),
            trust_scope=ReportTrustScope(
                review_status=approved.report.report_status,
                total_shot_count=approved.report.session_statistics.total_shot_count,
                valid_shot_count=approved.report.session_statistics.valid_shot_count,
                abstained_shot_count=approved.report.abstention_summary.abstained_shot_count,
                calibration_status=approved.report.calibration_status,
                unsupported_dimensions=unsupported,
            ),
            objective_statistics=objective,
            typical_shot_analysis=typical_section,
            training_recommendation=TrainingRecommendationSection(result=recommendation),
            quality_notice=ReportQualityNotice(
                data_coverage_status=approved.report.quality_summary.session_quality_status,
                algorithm_boundaries=unsupported,
                current_revision=parent_revision + 1,
                source_parent_report_id=parent_report_id,
                source_parent_revision=parent_revision,
            ),
        )
        task = ReviewTask(
            id=child_review_task_id,
            session_id=session_id,
            report_id=child.report_id,
            report_revision=child.revision,
            status=ReviewTaskStatus.PENDING_REVIEW,
        )
        return ReportCompositionResult(report=child, review_task=task)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ReportCompositionError(ReportCompositionReason.INVALID_INPUT) from exc
