"""Pure aggregation of final Shot public scalar fields for Story 3.1."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from billiards_ai.cloud.perception.finalizer import assess_session_quality
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.shot import Shot, TargetZoneSource
from billiards_ai.shared.models.statistics import (
    CueBallEndFrequency,
    CushionCountFrequency,
    SessionAggregationStatus,
    SessionStatistics,
    SessionStatisticsResult,
    TargetZoneDeviationSummary,
)


def _average(values: Sequence[int]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_session_statistics(shots: Sequence[Shot]) -> SessionStatisticsResult:
    """Aggregate final shots without mutation, I/O, providers, or media access."""

    valid_shots = tuple(shot for shot in shots if not shot.is_abstained)
    if not valid_shots:
        return SessionStatisticsResult(
            status=SessionAggregationStatus.NO_VALID_SHOTS,
            deliverable=False,
            statistics=None,
        )

    observed_potting = tuple(shot.potted_balls for shot in shots if shot.potted_balls is not None)
    successful_potting_shot_count = sum(bool(potted) for potted in observed_potting)
    potting_shot_count = len(observed_potting)

    travel_distances = tuple(
        shot.travel_distance_mm for shot in valid_shots if shot.travel_distance_mm is not None
    )
    cushion_counts = Counter(
        shot.cushion_count for shot in valid_shots if shot.cushion_count is not None
    )
    cue_ball_end_counts = Counter(
        (shot.cue_ball_end.x_mm, shot.cue_ball_end.y_mm)
        for shot in valid_shots
        if shot.cue_ball_end is not None
    )
    deviations_by_source: dict[TargetZoneSource, list[int]] = defaultdict(list)
    for shot in valid_shots:
        if shot.has_positional_verdict:
            deviations_by_source[shot.target_zone_source].append(shot.deviation_mm)

    return SessionStatisticsResult(
        status=SessionAggregationStatus.READY,
        deliverable=True,
        statistics=SessionStatistics(
            total_shot_count=len(shots),
            valid_shot_count=len(valid_shots),
            session_quality=assess_session_quality(tuple(shots)),
            potted_ball_count=sum(len(potted) for potted in observed_potting),
            successful_potting_shot_count=successful_potting_shot_count,
            potting_shot_count=potting_shot_count,
            pot_rate=(
                successful_potting_shot_count / potting_shot_count if potting_shot_count else None
            ),
            average_cue_ball_travel_distance_mm=_average(travel_distances),
            cue_ball_travel_shot_count=len(travel_distances),
            cushion_distribution=tuple(
                CushionCountFrequency(cushion_count=count, shot_count=frequency)
                for count, frequency in sorted(cushion_counts.items())
            ),
            cushion_count_shot_count=sum(cushion_counts.values()),
            cue_ball_end_distribution=tuple(
                CueBallEndFrequency(
                    position=Point(x_mm=x_mm, y_mm=y_mm),
                    shot_count=frequency,
                )
                for (x_mm, y_mm), frequency in sorted(cue_ball_end_counts.items())
            ),
            cue_ball_end_shot_count=sum(cue_ball_end_counts.values()),
            target_zone_deviation_by_source=tuple(
                TargetZoneDeviationSummary(
                    target_zone_source=source,
                    shot_count=len(deviations),
                    average_deviation_mm=_average(deviations),
                )
                for source in (
                    TargetZoneSource.SCENE_PRESET,
                    TargetZoneSource.POST_HOC_INFERRED,
                )
                if (deviations := deviations_by_source.get(source))
            ),
        ),
    )
