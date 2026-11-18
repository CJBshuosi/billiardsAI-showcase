"""Story 4.2 public review evidence contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    AccessActorRole,
    Confidence,
    ReviewableField,
    ReviewCorrection,
    ReviewerIdentity,
    ReviewEvidence,
    ReviewShotValues,
    ReviewStatus,
    ShotAnalysisStatus,
    ShotReviewDecision,
    ShotReviewEvidence,
    ShotReviewOutcome,
    TargetZoneSource,
)


def reviewer() -> ReviewerIdentity:
    return ReviewerIdentity(
        principal_id="reviewer-42",
        roles=(AccessActorRole.REVIEWER,),
        credential_id="credential-42",
    )


def test_review_correction_and_decision_are_strict_frozen_contracts():
    shot_id = uuid4()
    reviewed_at = datetime(2026, 7, 29, 11, tzinfo=UTC)
    correction = ReviewCorrection(
        shot_id=shot_id,
        field=ReviewableField.CUSHION_COUNT,
        ai_value=0,
        correct_value=1,
        note="rail contact was visible",
        reviewer=reviewer(),
        reviewed_at=reviewed_at,
    )
    decision = ShotReviewDecision(
        shot_id=shot_id,
        outcome=ShotReviewOutcome.CORRECTED,
        reviewer=reviewer(),
        reviewed_at=reviewed_at,
        corrections=(correction,),
    )

    with pytest.raises(ValidationError, match="frozen"):
        correction.correct_value = 2
    with pytest.raises(ValidationError, match="ai_value"):
        ReviewCorrection.model_validate(correction.model_dump() | {"ai_value": "0"})
    with pytest.raises(ValidationError, match="unexpected"):
        ShotReviewDecision.model_validate(decision.model_dump() | {"unexpected": True})

    assert ReviewCorrection.model_config == {
        "extra": "forbid",
        "frozen": True,
        "strict": True,
    }


def test_typed_json_is_accepted_without_weakening_python_mode():
    payload = {
        "shot_id": str(uuid4()),
        "field": "is_foul",
        "ai_value": False,
        "correct_value": True,
        "note": "foul confirmed",
        "reviewer": reviewer().model_dump(mode="json"),
        "reviewed_at": "2026-07-29T11:00:00Z",
    }

    correction = ReviewCorrection.model_validate_json(json.dumps(payload))

    assert correction.field is ReviewableField.IS_FOUL
    assert correction.ai_value is False
    assert correction.reviewed_at.tzinfo is UTC


@pytest.mark.parametrize(
    "field",
    ["raw_observation", "trajectory", "media_locator", "media_url", "raw_video_url"],
)
def test_correction_rejects_raw_observation_and_media_fields(field: str):
    with pytest.raises(ValidationError, match="field"):
        ReviewCorrection(
            shot_id=uuid4(),
            field=field,
            ai_value=0,
            correct_value=1,
            note="not allowed",
            reviewer=reviewer(),
            reviewed_at=datetime(2026, 7, 29, 11, tzinfo=UTC),
        )


def test_review_evidence_schema_contains_only_safe_projection_fields():
    serialized = json.dumps(ReviewEvidence.model_json_schema())

    assert ReviewEvidence.model_config["strict"] is True
    assert ReviewEvidence.model_config["frozen"] is True
    assert ReviewEvidence.model_config["extra"] == "forbid"
    assert all(
        forbidden not in serialized
        for forbidden in (
            "raw_observation",
            "trajectory",
            "media_locator",
            "media_url",
            "raw_video_url",
            "ball_positions_before",
            "ball_positions_after",
        )
    )


def test_decision_requires_correction_pairing_and_utc_time():
    common = {
        "shot_id": uuid4(),
        "reviewer": reviewer(),
        "reviewed_at": datetime(2026, 7, 29, 11, tzinfo=UTC),
    }
    with pytest.raises(ValidationError, match="correction"):
        ShotReviewDecision(outcome=ShotReviewOutcome.CORRECTED, corrections=(), **common)
    with pytest.raises(ValidationError, match="correction"):
        ShotReviewDecision(
            outcome=ShotReviewOutcome.APPROVED,
            corrections=(
                ReviewCorrection(
                    shot_id=common["shot_id"],
                    field=ReviewableField.IS_FOUL,
                    ai_value=False,
                    correct_value=True,
                    note="foul",
                    reviewer=reviewer(),
                    reviewed_at=common["reviewed_at"],
                ),
            ),
            **common,
        )
    with pytest.raises(ValidationError, match="UTC"):
        ShotReviewDecision(
            outcome=ShotReviewOutcome.APPROVED,
            corrections=(),
            **(common | {"reviewed_at": datetime(2026, 7, 29, 11)}),
        )


def test_decision_requires_reviewer_role_and_correction_note():
    unauthorized = ReviewerIdentity(
        principal_id="system-42",
        roles=(AccessActorRole.SYSTEM,),
        credential_id="credential-42",
    )
    with pytest.raises(ValidationError, match="REVIEWER"):
        ShotReviewDecision(
            shot_id=uuid4(),
            outcome=ShotReviewOutcome.APPROVED,
            reviewer=unauthorized,
            reviewed_at=datetime(2026, 7, 29, 11, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="note"):
        ReviewCorrection(
            shot_id=uuid4(),
            field=ReviewableField.IS_FOUL,
            ai_value=False,
            correct_value=True,
            note="   ",
            reviewer=reviewer(),
            reviewed_at=datetime(2026, 7, 29, 11, tzinfo=UTC),
        )


def test_review_evidence_direct_construction_rejects_mismatched_shot_set():
    session_id = uuid4()
    item = ShotReviewEvidence(
        shot_id=uuid4(),
        session_id=uuid4(),
        sequence=2,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=ReviewStatus.IN_REVIEW,
        shot_created_at=datetime(2026, 7, 29, 10, tzinfo=UTC),
        ai_values=ReviewShotValues(
            struck_ball=1,
            potted_balls=(),
            cushion_count=0,
            travel_distance_mm=500,
            is_foul=False,
            deviation_mm=None,
            target_zone_source=TargetZoneSource.NONE,
            confidence=Confidence.HIGH,
        ),
    )

    with pytest.raises(ValidationError, match="session|sequence"):
        ReviewEvidence(session_id=session_id, shots=(item,))


def test_reviewed_scalar_evidence_carries_explicit_target_zone_source():
    values = ReviewShotValues(
        struck_ball=1,
        potted_balls=(),
        cushion_count=0,
        travel_distance_mm=500,
        is_foul=False,
        deviation_mm=400,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        confidence=Confidence.HIGH,
    )

    assert values.target_zone_source is TargetZoneSource.SCENE_PRESET
    payload = values.model_dump()
    payload.pop("target_zone_source")
    with pytest.raises(ValidationError, match="target_zone_source"):
        ReviewShotValues.model_validate(payload)
    with pytest.raises(ValidationError):
        ReviewShotValues.model_validate(values.model_dump() | {"target_zone_source": "future"})
