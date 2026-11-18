"""Deterministic aggregation from the deliberately narrow approved-shot projection."""

from __future__ import annotations

from collections import Counter, defaultdict

from billiards_ai.cloud.engine.scoring import score_session_performance
from billiards_ai.shared.models.quality import (
    SessionQualityStatus,
    SessionQualitySummary,
    ShotAnalysisStatus,
)
from billiards_ai.shared.models.review import ApprovedShotData
from billiards_ai.shared.models.scoring import SessionPerformanceScores
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource
from billiards_ai.shared.models.statistics import (
    CushionCountFrequency,
    SessionStatistics,
    TargetZoneDeviationSummary,
)


def project_approved_shots(
    shots: tuple[ApprovedShotData, ...],
) -> tuple[SessionStatistics, SessionPerformanceScores]:
    """Rebuild every safely supported fact without trajectory or raw observations."""

    if not shots:
        raise ValueError("approved shots cannot be empty")
    approved = {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
    if any(shot.review_status not in approved for shot in shots):
        raise ValueError("growth projection accepts only approved or corrected shots")
    sequences = tuple(shot.sequence for shot in shots)
    if sequences != tuple(sorted(set(sequences))):
        raise ValueError("approved shot sequences must be sorted and unique")
    shot_ids = tuple(shot.shot_id for shot in shots)
    if len(shot_ids) != len(set(shot_ids)):
        raise ValueError("approved shot ids must be unique")

    valid = tuple(
        shot
        for shot in shots
        if shot.analysis_status is ShotAnalysisStatus.ANALYZED
        and shot.confidence is not Confidence.ABSTAIN
    )
    if not valid:
        raise ValueError("approved projection requires at least one analyzed non-abstained shot")
    abstained_count = len(shots) - len(valid)
    abstention_rate = abstained_count / len(shots)
    quality = SessionQualitySummary(
        shot_count=len(shots),
        abstained_shot_count=abstained_count,
        abstention_rate=abstention_rate,
        session_quality_status=(
            SessionQualityStatus.LOW_COVERAGE
            if abstention_rate > 0.5
            else SessionQualityStatus.SUFFICIENT_COVERAGE
        ),
    )

    travel_distances = tuple(
        shot.travel_distance_mm for shot in valid if shot.travel_distance_mm is not None
    )
    cushion_counts = Counter(shot.cushion_count for shot in valid if shot.cushion_count is not None)
    deviations: dict[TargetZoneSource, list[int]] = defaultdict(list)
    for shot in valid:
        if shot.deviation_mm is not None and shot.target_zone_source is not TargetZoneSource.NONE:
            deviations[shot.target_zone_source].append(shot.deviation_mm)

    statistics = SessionStatistics(
        total_shot_count=len(shots),
        valid_shot_count=len(valid),
        session_quality=quality,
        # Non-empty tuples preserve known pots. None and [] both collapse to (), so a
        # safe attempt denominator still cannot be reconstructed.
        potted_ball_count=sum(len(shot.potted_balls) for shot in valid),
        successful_potting_shot_count=0,
        potting_shot_count=0,
        pot_rate=None,
        average_cue_ball_travel_distance_mm=(
            sum(travel_distances) / len(travel_distances) if travel_distances else None
        ),
        cue_ball_travel_shot_count=len(travel_distances),
        cushion_distribution=tuple(
            CushionCountFrequency(cushion_count=count, shot_count=frequency)
            for count, frequency in sorted(cushion_counts.items())
        ),
        cushion_count_shot_count=sum(cushion_counts.values()),
        cue_ball_end_distribution=(),
        cue_ball_end_shot_count=0,
        target_zone_deviation_by_source=tuple(
            TargetZoneDeviationSummary(
                target_zone_source=source,
                shot_count=len(values),
                average_deviation_mm=sum(values) / len(values),
            )
            for source in (TargetZoneSource.SCENE_PRESET, TargetZoneSource.POST_HOC_INFERRED)
            if (values := deviations.get(source))
        ),
    )
    return statistics, score_session_performance(statistics)
