"""Test-only builders for Story 8.4 snapshot behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from billiards_ai.cloud.growth import project_approved_shots
from billiards_ai.cloud.growth.repository import InMemoryGrowthEvidenceRepository
from billiards_ai.shared.models import (
    ApprovedShotData,
    Confidence,
    ReviewStatus,
    ShotAnalysisStatus,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import ApprovedEvidenceBatchV2
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricEvidenceAuthorityRecord,
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    GameCode,
    MetricFieldQuality,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    ReviewedFieldQuality,
    ReviewedPoint,
    ReviewedTableGeometry,
    TargetZoneEvaluationStatus,
)
from billiards_ai.shared.models.shot import TargetZoneSource

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def _quality() -> ReviewedFieldQuality:
    return ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=1.0)


def _qualities() -> MetricFieldQuality:
    value = _quality()
    return MetricFieldQuality(
        geometry=value,
        intent=value,
        pot_attempt=value,
        pot_result=value,
        foul_result=value,
        target_zone=value,
        cue_ball_end=value,
    )


def _calibration() -> dict[str, object]:
    return {
        "calibration_id": uid(1),
        "calibration_revision_id": uid(2),
        "calibration_revision_number": 1,
    }


def _point(x_mm: float, y_mm: float) -> ReviewedPoint:
    return ReviewedPoint(**_calibration(), x_mm=x_mm, y_mm=y_mm)


def _table() -> ReviewedTableGeometry:
    return ReviewedTableGeometry(
        **_calibration(),
        table_length_mm=2540,
        table_width_mm=1270,
        ball_diameter_mm=57.15,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
    )


def make_batch(
    number: int,
    *,
    user_number: int = 900,
    session_number: int | None = None,
    review_revision: int = 1,
    approved_at: datetime = NOW,
    successful: bool = True,
) -> ApprovedEvidenceBatchV2:
    session_number = session_number if session_number is not None else 100 + number
    session_id = uid(session_number)
    shot_id = uid(10_000 + number * 10 + review_revision)
    approved_shot = ApprovedShotData(
        shot_id=shot_id,
        session_id=session_id,
        sequence=1,
        struck_ball=3,
        potted_balls=(3,) if successful else (),
        cushion_count=0,
        travel_distance_mm=400,
        is_foul=False,
        deviation_mm=None,
        target_zone_source=TargetZoneSource.NONE,
        confidence=Confidence.HIGH,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=ReviewStatus.APPROVED,
    )
    metric_shot = ApprovedMetricShotEvidence(
        shot_id=shot_id,
        session_id=session_id,
        sequence=1,
        review_revision=review_revision,
        game_code=GameCode.CHINESE_EIGHT_BALL,
        table_geometry=_table(),
        cue_ball_start_mm=_point(100.0, 300.0),
        object_ball_start_mm=_point(500.0, 300.0),
        intended_object_ball=3,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.LEGAL_SUCCESS if successful else PotResult.MISS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.NONE,
        target_zone=None,
        cue_ball_end_mm=None,
        target_zone_evaluation_status=TargetZoneEvaluationStatus.NOT_APPLICABLE,
        target_zone_result=None,
        field_quality=_qualities(),
    )
    statistics, scores = project_approved_shots((approved_shot,))
    payload = {
        "schema_version": "2.0",
        "batch_id": uid(20_000 + number * 10 + review_revision),
        "user_id": uid(user_number),
        "session_id": session_id,
        "report_id": uid(30_000 + number),
        "report_revision": review_revision,
        "review_revision": review_revision,
        "approved_at": approved_at,
        "approval_kind": EvidenceApprovalKind.HUMAN_REVIEW,
        "approval_policy_version": "human-review-v1",
        "shots": (approved_shot,),
        "metric_shots": (metric_shot,),
        "statistics": statistics,
        "scores": scores,
        "source_envelope_ids": (uid(40_000 + number * 10 + review_revision),),
    }
    return ApprovedEvidenceBatchV2.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}
    )


def admit(*batches: ApprovedEvidenceBatchV2):
    repository = InMemoryGrowthEvidenceRepository()
    values = iter(uid(800_000 + index) for index in range(len(batches) * 2))
    admitted = []
    for batch in batches:
        result = repository.append_approved_batch(
            batch,
            metric_authority=ApprovedMetricEvidenceAuthorityRecord(
                approval_kind=batch.approval_kind,
                approval_policy_version=batch.approval_policy_version,
                metric_shots=batch.metric_shots,
            ),
            clock=lambda: NOW,
            uuid_factory=lambda: next(values),
        )
        assert result.accepted_metric_input is not None
        admitted.append(result.accepted_metric_input)
    return tuple(admitted)
