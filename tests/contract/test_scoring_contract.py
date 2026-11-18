"""Frozen public scoring contracts for Story 3.2."""

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.scoring import (
    DimensionScore,
    LevelAssessmentScores,
    ScoreStatus,
    SessionPerformanceScores,
)


def test_dimension_score_requires_numeric_value_only_when_scored():
    scored = DimensionScore(score=0, status=ScoreStatus.SCORED)

    assert scored.score == 0
    with pytest.raises(ValidationError, match="SCORED requires a numeric score"):
        DimensionScore(score=None, status=ScoreStatus.SCORED)
    with pytest.raises(ValidationError, match="non-SCORED status requires a null score"):
        DimensionScore(score=0, status=ScoreStatus.INSUFFICIENT_EVIDENCE)


def test_score_contracts_are_strict_and_frozen():
    unavailable = DimensionScore(score=None, status=ScoreStatus.UNSUPPORTED)
    scores = SessionPerformanceScores(
        potting=unavailable,
        positioning=unavailable,
        power_control=unavailable,
        rotation_application=unavailable,
    )

    with pytest.raises(ValidationError, match="frozen"):
        scores.potting = DimensionScore(score=100, status=ScoreStatus.SCORED)
    with pytest.raises(ValidationError, match="unexpected"):
        SessionPerformanceScores.model_validate(scores.model_dump() | {"unexpected": 1})


def test_level_assessment_contract_is_a_separate_shape():
    assert set(SessionPerformanceScores.model_fields) == {
        "potting",
        "positioning",
        "power_control",
        "rotation_application",
    }
    assert set(LevelAssessmentScores.model_fields) == {
        "object_ball_position",
        "cue_ball_cushion_contact",
        "cue_ball_path",
        "cue_ball_stop_zone",
    }
    assert not issubclass(LevelAssessmentScores, SessionPerformanceScores)
