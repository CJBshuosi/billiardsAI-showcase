"""Synthetic-only rendering and revision tests for Story 3.3."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from billiards_ai.cloud.engine.reporting import render_report_document, revise_report_document
from billiards_ai.shared.models.calibration import CalibrationStatus
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.quality import SessionQualityStatus, SessionQualitySummary
from billiards_ai.shared.models.report import ReportStatus
from billiards_ai.shared.models.scoring import DimensionScore, ScoreStatus, SessionPerformanceScores
from billiards_ai.shared.models.shot import TargetZoneSource
from billiards_ai.shared.models.statistics import (
    CueBallEndFrequency,
    CushionCountFrequency,
    SessionStatistics,
    TargetZoneDeviationSummary,
)

SESSION_ID = UUID("10000000-0000-4000-8000-000000000010")
REPORT_ID = UUID("30000000-0000-4000-8000-000000000010")
REVISION_ID = UUID("30000000-0000-4000-8000-000000000011")
GENERATED_AT = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _statistics(*, abstained_shot_count: int = 0) -> SessionStatistics:
    total = 4
    valid = total - abstained_shot_count
    quality = SessionQualitySummary(
        shot_count=total,
        abstained_shot_count=abstained_shot_count,
        abstention_rate=abstained_shot_count / total,
        session_quality_status=(
            SessionQualityStatus.LOW_COVERAGE
            if abstained_shot_count > total / 2
            else SessionQualityStatus.SUFFICIENT_COVERAGE
        ),
    )
    return SessionStatistics(
        total_shot_count=total,
        valid_shot_count=valid,
        session_quality=quality,
        potted_ball_count=2,
        successful_potting_shot_count=2,
        potting_shot_count=valid,
        pot_rate=2 / valid,
        average_cue_ball_travel_distance_mm=650.0,
        cue_ball_travel_shot_count=valid,
        cushion_distribution=(CushionCountFrequency(cushion_count=1, shot_count=valid),),
        cushion_count_shot_count=valid,
        cue_ball_end_distribution=(
            CueBallEndFrequency(position=Point(x_mm=800, y_mm=500), shot_count=valid),
        ),
        cue_ball_end_shot_count=valid,
        target_zone_deviation_by_source=(
            TargetZoneDeviationSummary(
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                shot_count=valid,
                average_deviation_mm=120.0,
            ),
        ),
    )


def _scores() -> SessionPerformanceScores:
    unsupported = DimensionScore(score=None, status=ScoreStatus.UNSUPPORTED)
    return SessionPerformanceScores(
        potting=DimensionScore(score=50, status=ScoreStatus.SCORED),
        positioning=unsupported,
        power_control=unsupported,
        rotation_application=unsupported,
    )


def _render(calibration_status: CalibrationStatus, *, abstained_shot_count: int = 0):
    return render_report_document(
        report_id=REPORT_ID,
        session_id=SESSION_ID,
        generated_at=GENERATED_AT,
        analysis_version="phase0-3.3",
        calibration_status=calibration_status,
        statistics=_statistics(abstained_shot_count=abstained_shot_count),
        dimension_scores=_scores(),
    )


def test_rendered_report_defaults_to_pending_review_and_keeps_calibrated_distances():
    document = _render(CalibrationStatus.ACTIVE)
    payload = document.model_dump(mode="json")

    assert document.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert document.revision == 1
    assert document.parent_revision_id is None
    assert payload["session_statistics"]["average_cue_ball_travel_distance_mm"] == 650.0
    assert payload["session_statistics"]["cue_ball_end_distribution"][0]["position"] == {
        "x_mm": 800,
        "y_mm": 500,
    }
    assert (
        payload["session_statistics"]["target_zone_deviation_by_source"][0]["average_deviation_mm"]
        == 120.0
    )
    assert document.typical_shots == ()
    assert document.coaching_recommendations == ()


def test_rendering_stale_or_invalid_calibration_omits_every_distance_field():
    for calibration_status in (CalibrationStatus.STALE, CalibrationStatus.INVALID):
        payload = _render(calibration_status).model_dump(mode="json")

        assert set(payload["session_statistics"]) == {
            "total_shot_count",
            "valid_shot_count",
            "potted_ball_count",
            "successful_potting_shot_count",
            "potting_shot_count",
            "pot_rate",
            "cushion_distribution",
            "cushion_count_shot_count",
        }


def test_rendering_makes_abstained_shot_count_explicit_without_mutating_inputs():
    statistics = _statistics(abstained_shot_count=2)
    scores = _scores()
    before = statistics.model_dump(mode="python")

    document = render_report_document(
        report_id=REPORT_ID,
        session_id=SESSION_ID,
        generated_at=GENERATED_AT,
        analysis_version="phase0-3.3",
        calibration_status=CalibrationStatus.ACTIVE,
        statistics=statistics,
        dimension_scores=scores,
    )

    assert document.abstention_summary.abstained_shot_count == 2
    assert document.abstention_summary.display_text == "2 杆超出当前分析范围"
    assert statistics.model_dump(mode="python") == before


def test_revision_creates_new_immutable_document_with_parent_and_leaves_original_unchanged():
    original = _render(CalibrationStatus.ACTIVE)

    revised = revise_report_document(
        original,
        report_id=REVISION_ID,
        generated_at=datetime(2026, 7, 29, 13, 0, tzinfo=UTC),
        dimension_scores=SessionPerformanceScores(
            potting=DimensionScore(score=75, status=ScoreStatus.SCORED),
            positioning=original.dimension_scores.positioning,
            power_control=original.dimension_scores.power_control,
            rotation_application=original.dimension_scores.rotation_application,
        ),
    )

    assert revised.report_id == REVISION_ID
    assert revised.revision == 2
    assert revised.parent_revision_id == original.report_id
    assert revised.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert revised.dimension_scores.potting.score == 75
    assert original.revision == 1
    assert original.dimension_scores.potting.score == 50
