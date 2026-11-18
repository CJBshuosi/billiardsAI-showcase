"""Synthetic-only acceptance tests for Story 3.2."""

import ast
import inspect
from datetime import UTC, datetime
from uuid import UUID

from billiards_ai.cloud.engine.aggregation import aggregate_session_statistics
from billiards_ai.cloud.engine.scoring import score_session_performance
from billiards_ai.shared.models.geometry import Point, Zone
from billiards_ai.shared.models.scoring import ScoreStatus
from billiards_ai.shared.models.shot import Confidence, Shot, TargetZoneSource

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


def _scores(shots: tuple[Shot, ...]):
    result = aggregate_session_statistics(shots)
    assert result.statistics is not None
    return score_session_performance(result.statistics)


def test_scores_observed_pot_rate_without_defaulting_other_dimensions_to_zero():
    scores = _scores(
        (
            _shot(1, potted_balls=[1]),
            _shot(2, potted_balls=[]),
            _shot(3, potted_balls=[2]),
        )
    )

    assert scores.potting.score == 67
    assert scores.potting.status is ScoreStatus.SCORED
    assert scores.positioning.score is None
    assert scores.positioning.status is ScoreStatus.UNSUPPORTED
    assert scores.power_control.score is None
    assert scores.power_control.status is ScoreStatus.UNSUPPORTED
    assert scores.rotation_application.score is None
    assert scores.rotation_application.status is ScoreStatus.UNSUPPORTED


def test_missing_potting_observations_are_insufficient_evidence_not_zero():
    scores = _scores((_shot(1, potted_balls=None), _shot(2, potted_balls=None)))

    assert scores.potting.score is None
    assert scores.potting.status is ScoreStatus.INSUFFICIENT_EVIDENCE


def test_low_coverage_session_marks_an_observable_dimension_abstained():
    scores = _scores(
        (
            _shot(1, potted_balls=[1]),
            _shot(2, confidence=Confidence.ABSTAIN, potted_balls=None),
            _shot(3, confidence=Confidence.ABSTAIN, potted_balls=None),
        )
    )

    assert scores.potting.score is None
    assert scores.potting.status is ScoreStatus.ABSTAINED


def test_rotation_never_uses_potting_distance_cushion_or_deviation_proxies():
    target = Zone(center=Point(x_mm=1000, y_mm=500), radius_mm=100)
    proxy_heavy = _scores(
        (
            _shot(
                1,
                potted_balls=[1, 2, 3],
                travel_distance_mm=9000,
                cushion_count=6,
                target_zone=target,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
                deviation_mm=0,
            ),
        )
    )
    proxy_empty = _scores(
        (
            _shot(
                1,
                potted_balls=[],
                travel_distance_mm=None,
                cushion_count=None,
                cue_ball_end=None,
            ),
        )
    )

    assert proxy_heavy.rotation_application == proxy_empty.rotation_application
    assert proxy_heavy.rotation_application.score is None
    assert proxy_heavy.rotation_application.status is ScoreStatus.UNSUPPORTED


def test_scoring_is_deterministic_and_does_not_mutate_statistics():
    result = aggregate_session_statistics((_shot(1, potted_balls=[1]), _shot(2)))
    assert result.statistics is not None
    before = result.statistics.model_dump(mode="python")

    first = score_session_performance(result.statistics)
    second = score_session_performance(result.statistics)

    assert first == second
    assert result.statistics.model_dump(mode="python") == before


def test_scoring_source_has_no_restricted_or_shot_level_access():
    from billiards_ai.cloud.engine import scoring

    tree = ast.parse(inspect.getsource(scoring))
    identifiers = {
        name
        for node in ast.walk(tree)
        for name in (
            node.attr if isinstance(node, ast.Attribute) else None,
            node.id if isinstance(node, ast.Name) else None,
        )
        if name is not None
    }

    assert {"Shot", "trajectory", "raw_observation"}.isdisjoint(identifiers)
