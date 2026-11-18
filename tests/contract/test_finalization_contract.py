"""Executable quality contracts for Story 2.9 finalization."""

import pytest
from pydantic import ValidationError


def test_field_quality_is_strict_frozen_and_has_only_approved_sources():
    from billiards_ai.shared.models.quality import FieldQuality, FinalFieldSource

    assert {source.value for source in FinalFieldSource} == {
        "CV",
        "OPEN_SOURCE_MODEL",
        "VLM",
        "MANUAL",
        "UNKNOWN",
    }
    quality = FieldQuality(source=FinalFieldSource.CV, confidence=0.85)

    with pytest.raises(ValidationError):
        quality.confidence = 0.5
    with pytest.raises(ValidationError):
        FieldQuality(source=FinalFieldSource.CV, confidence=1.01)
    with pytest.raises(ValidationError):
        FieldQuality.model_validate({"source": "CV", "confidence": 0.85, "detail": "raw"})


def test_session_quality_contract_uses_typed_low_coverage_status():
    from billiards_ai.shared.models.quality import SessionQualityStatus, SessionQualitySummary

    summary = SessionQualitySummary(
        shot_count=3,
        abstained_shot_count=2,
        abstention_rate=2 / 3,
        session_quality_status=SessionQualityStatus.LOW_COVERAGE,
    )

    assert summary.session_quality_status is SessionQualityStatus.LOW_COVERAGE
    with pytest.raises(ValidationError):
        summary.shot_count = 4
    with pytest.raises(ValidationError):
        SessionQualitySummary(
            shot_count=3,
            abstained_shot_count=1,
            abstention_rate=2 / 3,
            session_quality_status=SessionQualityStatus.SUFFICIENT_COVERAGE,
        )


def test_finalized_shot_quality_metadata_is_complete_and_raw_fields_stay_forbidden():
    from billiards_ai.shared.models.quality import (
        FieldQuality,
        FinalFieldSource,
        ShotAnalysisStatus,
        ShotFieldQuality,
    )
    from billiards_ai.shared.models.shot import Shot

    quality = FieldQuality(source=FinalFieldSource.CV, confidence=0.85)
    field_quality = ShotFieldQuality(
        ball_positions_before=quality,
        ball_positions_after=quality,
        cue_ball_end=quality,
        struck_ball=quality,
        potted_balls=quality,
        cushion_count=quality,
        travel_distance_mm=quality,
        is_foul=quality,
    )

    assert set(field_quality.model_dump()) == {
        "ball_positions_before",
        "ball_positions_after",
        "cue_ball_end",
        "struck_ball",
        "potted_balls",
        "cushion_count",
        "travel_distance_mm",
        "is_foul",
        "trajectory",
    }
    assert Shot.model_fields["trajectory"].description.startswith("上层禁读")
    assert Shot.model_fields["raw_observation"].description.startswith("上层禁读")
    assert ShotAnalysisStatus.OUT_OF_CURRENT_ANALYSIS_SCOPE.value == "超出当前分析范围"
