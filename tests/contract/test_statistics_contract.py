"""Frozen public contract for Story 3.1 session statistics."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.engine.aggregation import aggregate_session_statistics
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.shot import Confidence, Shot
from billiards_ai.shared.models.statistics import (
    SessionAggregationStatus,
    SessionStatistics,
)


def _shot() -> Shot:
    return Shot(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        session_id=UUID("10000000-0000-0000-0000-000000000000"),
        sequence=1,
        t_start_ms=0,
        t_end_ms=1000,
        ball_positions_before={0: Point(x_mm=100, y_mm=100)},
        ball_positions_after={0: Point(x_mm=300, y_mm=100)},
        cue_ball_end=Point(x_mm=300, y_mm=100),
        potted_balls=[],
        cushion_count=0,
        travel_distance_mm=200,
        confidence=Confidence.HIGH,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_statistics_contract_is_strict_frozen_and_explicit_about_denominators():
    result = aggregate_session_statistics((_shot(),))

    assert result.status is SessionAggregationStatus.READY
    assert result.deliverable is True
    assert result.statistics is not None
    assert result.statistics.total_shot_count == 1
    assert result.statistics.valid_shot_count == 1
    assert result.statistics.session_quality.shot_count == 1
    assert result.statistics.potting_shot_count == 1
    assert result.statistics.cue_ball_travel_shot_count == 1
    assert result.statistics.cushion_count_shot_count == 1
    assert result.statistics.cue_ball_end_shot_count == 1

    with pytest.raises(ValidationError, match="frozen"):
        result.statistics.total_shot_count = 2
    with pytest.raises(ValidationError, match="unexpected"):
        SessionStatistics.model_validate(result.statistics.model_dump() | {"unexpected": 1})


def test_no_valid_shots_is_a_typed_non_deliverable_result():
    result = aggregate_session_statistics(())

    assert result.status is SessionAggregationStatus.NO_VALID_SHOTS
    assert result.deliverable is False
    assert result.statistics is None
