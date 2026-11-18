from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from billiards_ai.cloud.growth import project_approved_shots
from billiards_ai.shared.models import (
    ApprovedEvidenceBatch,
    ApprovedShotData,
    Confidence,
    ReviewStatus,
    ScoreStatus,
    ShotAnalysisStatus,
    TargetZoneSource,
)
from billiards_ai.shared.models.events import canonical_sha256

FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "core_intelligence"
    / "v1"
    / "approved-evidence-batch.json"
)


def _shot(**updates: object) -> ApprovedShotData:
    values: dict[str, object] = {
        "shot_id": UUID("00000000-0000-4000-8000-000000000006"),
        "session_id": UUID("00000000-0000-4000-8000-000000000002"),
        "sequence": 1,
        "struck_ball": 3,
        "potted_balls": (3,),
        "cushion_count": 0,
        "travel_distance_mm": 500,
        "is_foul": False,
        "deviation_mm": 80,
        "target_zone_source": TargetZoneSource.SCENE_PRESET,
        "confidence": Confidence.HIGH,
        "analysis_status": ShotAnalysisStatus.ANALYZED,
        "review_status": ReviewStatus.APPROVED,
    }
    return ApprovedShotData.model_validate(values | updates)


def test_projector_recomputes_only_statistics_supported_by_safe_shot_fields() -> None:
    statistics, scores = project_approved_shots((_shot(),))

    assert statistics.model_dump(mode="json") == {
        "total_shot_count": 1,
        "valid_shot_count": 1,
        "session_quality": {
            "shot_count": 1,
            "abstained_shot_count": 0,
            "abstention_rate": 0.0,
            "session_quality_status": "SUFFICIENT_COVERAGE",
        },
        "potted_ball_count": 1,
        "successful_potting_shot_count": 0,
        "potting_shot_count": 0,
        "pot_rate": None,
        "average_cue_ball_travel_distance_mm": 500.0,
        "cue_ball_travel_shot_count": 1,
        "cushion_distribution": [{"cushion_count": 0, "shot_count": 1}],
        "cushion_count_shot_count": 1,
        "cue_ball_end_distribution": [],
        "cue_ball_end_shot_count": 0,
        "target_zone_deviation_by_source": [
            {
                "target_zone_source": "scene_preset",
                "shot_count": 1,
                "average_deviation_mm": 80.0,
            }
        ],
    }
    assert scores.potting.score is None
    assert scores.potting.status is ScoreStatus.INSUFFICIENT_EVIDENCE
    assert scores.positioning.status is ScoreStatus.UNSUPPORTED
    assert scores.power_control.status is ScoreStatus.UNSUPPORTED
    assert scores.rotation_application.status is ScoreStatus.UNSUPPORTED


def test_projector_does_not_invent_potting_attempts_from_lossy_projection() -> None:
    statistics, scores = project_approved_shots((_shot(potted_balls=()),))

    assert statistics.potted_ball_count == 0
    assert statistics.successful_potting_shot_count == 0
    assert statistics.potting_shot_count == 0
    assert statistics.pot_rate is None
    assert scores.potting.score is None
    assert scores.potting.status is ScoreStatus.INSUFFICIENT_EVIDENCE


def test_projector_preserves_known_potted_ball_count_without_inventing_attempts() -> None:
    statistics, scores = project_approved_shots((_shot(potted_balls=(3, 6)),))

    assert statistics.potted_ball_count == 2
    assert statistics.successful_potting_shot_count == 0
    assert statistics.potting_shot_count == 0
    assert statistics.pot_rate is None
    assert scores.potting.score is None
    assert scores.potting.status is ScoreStatus.INSUFFICIENT_EVIDENCE


def test_projector_ignores_residual_fields_from_non_analyzed_shots() -> None:
    statistics, _scores = project_approved_shots(
        (
            _shot(),
            _shot(
                shot_id=UUID("00000000-0000-4000-8000-000000000007"),
                sequence=2,
                analysis_status=ShotAnalysisStatus.PENDING,
                potted_balls=(8, 9),
                cushion_count=7,
                travel_distance_mm=999,
                deviation_mm=999,
            ),
        )
    )

    assert statistics.valid_shot_count == 1
    assert statistics.potted_ball_count == 1
    assert statistics.average_cue_ball_travel_distance_mm == 500.0
    assert tuple(
        (item.cushion_count, item.shot_count) for item in statistics.cushion_distribution
    ) == ((0, 1),)
    assert statistics.target_zone_deviation_by_source[0].average_deviation_mm == 80.0


def test_projector_preserves_abstention_and_never_invents_unavailable_cue_positions() -> None:
    statistics, scores = project_approved_shots(
        (
            _shot(),
            _shot(
                shot_id=UUID("00000000-0000-4000-8000-000000000007"),
                sequence=2,
                confidence=Confidence.ABSTAIN,
                potted_balls=(),
                cushion_count=None,
                travel_distance_mm=None,
                deviation_mm=None,
                target_zone_source=TargetZoneSource.NONE,
            ),
            _shot(
                shot_id=UUID("00000000-0000-4000-8000-000000000008"),
                sequence=3,
                confidence=Confidence.ABSTAIN,
                potted_balls=(),
                cushion_count=None,
                travel_distance_mm=None,
                deviation_mm=None,
                target_zone_source=TargetZoneSource.NONE,
            ),
        )
    )

    assert statistics.valid_shot_count == 1
    assert statistics.session_quality.abstained_shot_count == 2
    assert statistics.cue_ball_end_distribution == ()
    assert statistics.cue_ball_end_shot_count == 0
    assert scores.potting.status is ScoreStatus.ABSTAINED
    assert scores.potting.score is None


@pytest.mark.parametrize(
    "review_status",
    [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW, ReviewStatus.REJECTED],
)
def test_projector_rejects_every_non_approved_shot(review_status: ReviewStatus) -> None:
    with pytest.raises(ValueError, match="approved or corrected"):
        project_approved_shots((_shot(review_status=review_status),))


def test_projector_rejects_duplicate_shot_lineage() -> None:
    duplicate = _shot(sequence=2)

    with pytest.raises(ValueError, match="shot ids"):
        project_approved_shots((_shot(), duplicate))


def test_c0_batch_fixture_matches_authoritative_projection_and_canonical_digest() -> None:
    batch = ApprovedEvidenceBatch.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    statistics, scores = project_approved_shots(batch.shots)

    assert batch.statistics == statistics
    assert batch.scores == scores
    assert batch.content_sha256 == canonical_sha256(
        batch.model_dump(mode="json", exclude={"content_sha256"})
    )
    assert "raw_observation" not in json.dumps(batch.model_dump(mode="json"))
