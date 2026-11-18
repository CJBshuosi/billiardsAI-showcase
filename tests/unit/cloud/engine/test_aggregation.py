"""Synthetic-only acceptance tests for Story 3.1."""

import ast
import inspect
from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.cloud.engine.aggregation import aggregate_session_statistics
from billiards_ai.shared.models.geometry import Point, Zone
from billiards_ai.shared.models.shot import Confidence, Shot, TargetZoneSource
from billiards_ai.shared.models.statistics import SessionAggregationStatus

SESSION_ID = UUID("10000000-0000-0000-0000-000000000000")


def _shot(sequence: int, **overrides: object) -> Shot:
    base = {
        "id": UUID(f"00000000-0000-0000-0000-{sequence:012d}"),
        "session_id": SESSION_ID,
        "sequence": sequence,
        "t_start_ms": (sequence - 1) * 1000,
        "t_end_ms": sequence * 1000,
        "ball_positions_before": {0: Point(x_mm=100, y_mm=100)},
        "ball_positions_after": {0: Point(x_mm=200, y_mm=100)},
        "cue_ball_end": Point(x_mm=200, y_mm=100),
        "potted_balls": [],
        "cushion_count": 0,
        "travel_distance_mm": 100,
        "confidence": Confidence.HIGH,
        "created_at": datetime(2026, 7, 29, tzinfo=UTC),
    }
    return Shot(**(base | overrides))


def _statistics(shots: tuple[Shot, ...]):
    result = aggregate_session_statistics(shots)
    assert result.status is SessionAggregationStatus.READY
    assert result.statistics is not None
    return result.statistics


def test_aggregates_basic_counts_rates_and_distributions():
    target = Zone(center=Point(x_mm=1000, y_mm=500), radius_mm=100)
    statistics = _statistics(
        (
            _shot(
                1,
                potted_balls=[3],
                travel_distance_mm=1000,
                cushion_count=0,
                cue_ball_end=Point(x_mm=100, y_mm=200),
                target_zone=target,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                deviation_mm=100,
            ),
            _shot(
                2,
                potted_balls=[],
                travel_distance_mm=2000,
                cushion_count=2,
                cue_ball_end=Point(x_mm=100, y_mm=200),
                target_zone=target,
                target_zone_source=TargetZoneSource.POST_HOC_INFERRED,
                deviation_mm=300,
            ),
            _shot(
                3,
                potted_balls=[4, 5],
                travel_distance_mm=3000,
                cushion_count=2,
                cue_ball_end=Point(x_mm=500, y_mm=600),
                target_zone=target,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                deviation_mm=0,
            ),
        )
    )

    assert statistics.total_shot_count == 3
    assert statistics.valid_shot_count == 3
    assert statistics.potted_ball_count == 3
    assert statistics.successful_potting_shot_count == 2
    assert statistics.potting_shot_count == 3
    assert statistics.pot_rate == pytest.approx(2 / 3)
    assert statistics.average_cue_ball_travel_distance_mm == pytest.approx(2000)
    assert statistics.cue_ball_travel_shot_count == 3
    assert [(item.cushion_count, item.shot_count) for item in statistics.cushion_distribution] == [
        (0, 1),
        (2, 2),
    ]
    assert statistics.cushion_count_shot_count == 3
    assert [
        (item.position.x_mm, item.position.y_mm, item.shot_count)
        for item in statistics.cue_ball_end_distribution
    ] == [(100, 200, 2), (500, 600, 1)]
    assert statistics.cue_ball_end_shot_count == 3
    assert [
        (item.target_zone_source, item.shot_count, item.average_deviation_mm)
        for item in statistics.target_zone_deviation_by_source
    ] == [
        (TargetZoneSource.SCENE_PRESET, 2, 50.0),
        (TargetZoneSource.POST_HOC_INFERRED, 1, 300.0),
    ]


def test_abstention_counts_in_total_and_potting_but_not_positional_denominators():
    target = Zone(center=Point(x_mm=1000, y_mm=500), radius_mm=100)
    statistics = _statistics(
        (
            _shot(
                1,
                potted_balls=[],
                travel_distance_mm=1000,
                cushion_count=1,
                cue_ball_end=Point(x_mm=300, y_mm=400),
                target_zone=target,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                deviation_mm=200,
            ),
            _shot(
                2,
                confidence=Confidence.ABSTAIN,
                potted_balls=[8],
                travel_distance_mm=9000,
                cushion_count=9,
                cue_ball_end=Point(x_mm=900, y_mm=900),
                target_zone=target,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                deviation_mm=0,
            ),
            _shot(
                3,
                potted_balls=None,
                travel_distance_mm=None,
                cushion_count=None,
                cue_ball_end=None,
            ),
        )
    )

    assert statistics.total_shot_count == 3
    assert statistics.valid_shot_count == 2
    assert statistics.potted_ball_count == 1
    assert statistics.successful_potting_shot_count == 1
    assert statistics.potting_shot_count == 2
    assert statistics.pot_rate == 0.5
    assert statistics.average_cue_ball_travel_distance_mm == 1000
    assert statistics.cue_ball_travel_shot_count == 1
    assert [(item.cushion_count, item.shot_count) for item in statistics.cushion_distribution] == [
        (1, 1)
    ]
    assert statistics.cushion_count_shot_count == 1
    assert statistics.cue_ball_end_shot_count == 1
    assert statistics.cue_ball_end_distribution[0].position == Point(x_mm=300, y_mm=400)
    assert len(statistics.target_zone_deviation_by_source) == 1
    assert statistics.target_zone_deviation_by_source[0].shot_count == 1
    assert statistics.target_zone_deviation_by_source[0].average_deviation_mm == 200


def test_mixed_session_preserves_low_coverage_quality_semantics():
    from billiards_ai.shared.models.quality import SessionQualityStatus

    statistics = _statistics(
        (
            _shot(1),
            _shot(2, confidence=Confidence.ABSTAIN),
            _shot(3, confidence=Confidence.ABSTAIN),
        )
    )

    assert statistics.valid_shot_count == 1
    assert statistics.session_quality.shot_count == 3
    assert statistics.session_quality.abstained_shot_count == 2
    assert statistics.session_quality.abstention_rate == pytest.approx(2 / 3)
    assert statistics.session_quality.session_quality_status is SessionQualityStatus.LOW_COVERAGE


@pytest.mark.parametrize("shots", [(), (_shot(1, confidence=Confidence.ABSTAIN),)])
def test_empty_or_all_abstained_input_returns_no_valid_shots(shots: tuple[Shot, ...]):
    result = aggregate_session_statistics(shots)

    assert result.status is SessionAggregationStatus.NO_VALID_SHOTS
    assert result.deliverable is False
    assert result.statistics is None


def test_aggregation_is_deterministic_and_does_not_mutate_shots():
    shots = (_shot(1, potted_balls=[1], cushion_count=2), _shot(2, cushion_count=1))
    before = tuple(shot.model_dump(mode="python") for shot in shots)

    first = aggregate_session_statistics(shots)
    second = aggregate_session_statistics(shots)

    assert first == second
    assert tuple(shot.model_dump(mode="python") for shot in shots) == before


def test_aggregation_source_has_no_restricted_shot_field_access():
    from billiards_ai.cloud.engine import aggregation

    tree = ast.parse(inspect.getsource(aggregation))
    identifiers = {
        name
        for node in ast.walk(tree)
        for name in (
            node.attr if isinstance(node, ast.Attribute) else None,
            node.id if isinstance(node, ast.Name) else None,
        )
        if name is not None
    }

    assert {"trajectory", "raw_observation"}.isdisjoint(identifiers)
