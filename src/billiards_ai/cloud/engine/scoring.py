"""Pure evidence-bounded session performance scoring for Story 3.2."""

from __future__ import annotations

from billiards_ai.shared.models.quality import SessionQualityStatus
from billiards_ai.shared.models.scoring import (
    DimensionScore,
    ScoreStatus,
    SessionPerformanceScores,
)
from billiards_ai.shared.models.statistics import SessionStatistics


def _potting_score(statistics: SessionStatistics) -> DimensionScore:
    if statistics.session_quality.session_quality_status is SessionQualityStatus.LOW_COVERAGE:
        return DimensionScore(score=None, status=ScoreStatus.ABSTAINED)
    if statistics.potting_shot_count == 0:
        return DimensionScore(score=None, status=ScoreStatus.INSUFFICIENT_EVIDENCE)

    numerator = statistics.successful_potting_shot_count * 100
    denominator = statistics.potting_shot_count
    percentage = (numerator + denominator // 2) // denominator
    return DimensionScore(score=percentage, status=ScoreStatus.SCORED)


def score_session_performance(statistics: SessionStatistics) -> SessionPerformanceScores:
    """Score only supported public statistics, without I/O, mutation, or proxy metrics."""

    unsupported = DimensionScore(score=None, status=ScoreStatus.UNSUPPORTED)
    return SessionPerformanceScores(
        potting=_potting_score(statistics),
        positioning=unsupported,
        power_control=unsupported,
        rotation_application=unsupported,
    )
