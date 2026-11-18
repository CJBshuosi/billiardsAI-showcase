"""Trusted subject and human-reviewed metric authority assembly."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from billiards_ai.cloud.review.identity import (
    SessionSubjectAuthorityError,
    TableBindingSessionSubjectAuthority,
)
from billiards_ai.cloud.review.metric_authority import (
    HumanReviewMetricAuthorityAssembler,
    MetricCalibrationAuthorityError,
    MetricFieldReviewStates,
    MetricPointInput,
    MetricReviewState,
    MetricShotReviewInput,
    MetricTargetZoneInput,
)
from billiards_ai.shared.models.calibration import CalibrationLineage, CalibrationStatus
from billiards_ai.shared.models.metric_evidence import (
    EvidenceApprovalKind,
    FieldSource,
    FoulResult,
    PocketCode,
    PotAttemptStatus,
    PotResult,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
    TargetZoneShape,
)
from billiards_ai.shared.models.session_capture import SessionMediaBindingRecord
from billiards_ai.shared.models.shot import (
    Confidence,
    ReviewStatus,
    Shot,
    ShotAnalysisStatus,
    TargetZoneSource,
)


def _id(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


NOW = datetime(2026, 8, 10, 18, tzinfo=UTC)


class _BindingRepository:
    def __init__(self, binding: object) -> None:
        self.binding = binding

    def get_media_binding(self, *, session_id: UUID) -> object:
        return self.binding


def _binding(
    *, session_id: UUID | None = None, subjects: tuple[str, ...] | None = None
) -> SessionMediaBindingRecord:
    return SessionMediaBindingRecord(
        session_id=session_id or _id(1),
        attempt_id=_id(3),
        asset_id=_id(4),
        content_sha256="a" * 64,
        storage_locator="controlled://synthetic",
        subject_user_ids=subjects or (str(_id(2)),),
        interval_revision=7,
        accepted_at=NOW,
    )


def test_table_binding_adapter_returns_only_exact_one_uuid4_subject_authority() -> None:
    binding = _binding()
    authority = TableBindingSessionSubjectAuthority(_BindingRepository(binding)).resolve_subject(
        session_id=binding.session_id
    )

    assert authority.session_id == binding.session_id
    assert authority.user_id == _id(2)
    assert authority.attempt_id == binding.attempt_id
    assert authority.interval_revision == 7
    assert set(authority.model_dump()) == {
        "session_id",
        "user_id",
        "attempt_id",
        "interval_revision",
        "accepted_at",
    }


@pytest.mark.parametrize(
    "binding",
    [
        None,
        _binding(subjects=(str(_id(2)), str(_id(5)))),
        _binding(subjects=("not-a-uuid",)),
        _binding(session_id=_id(6)),
        {"session_id": str(_id(1)), "subject_user_ids": [str(_id(2))]},
    ],
)
def test_table_binding_adapter_fails_closed_for_missing_ambiguous_or_malformed_authority(
    binding: object,
) -> None:
    with pytest.raises(SessionSubjectAuthorityError):
        TableBindingSessionSubjectAuthority(_BindingRepository(binding)).resolve_subject(
            session_id=_id(1)
        )


class _CalibrationProvider:
    def __init__(self, calibration: CalibrationLineage) -> None:
        self.calibration = calibration

    def calibration_for(self, *, session_id: UUID) -> CalibrationLineage:
        return self.calibration


def _calibration(*, status: CalibrationStatus = CalibrationStatus.ACTIVE) -> CalibrationLineage:
    return CalibrationLineage(
        calibration_id=_id(30),
        revision_id=_id(31),
        revision_number=2,
        status=status,
        table_profile_id="cn-8-ball",
        table_length_mm=2540,
        table_width_mm=1270,
        table_configuration_fingerprint="b" * 64,
    )


def _shot(sequence: int, status: ReviewStatus) -> Shot:
    return Shot(
        id=_id(100 + sequence),
        session_id=_id(1),
        sequence=sequence,
        t_start_ms=(sequence - 1) * 1000,
        t_end_ms=sequence * 1000,
        ball_positions_before=None,
        ball_positions_after=None,
        cue_ball_end=None,
        struck_ball=sequence,
        potted_balls=[],
        cushion_count=0,
        travel_distance_mm=400,
        is_foul=False,
        deviation_mm=10,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        confidence=Confidence.HIGH,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=status,
        raw_observation={},
        created_at=NOW,
    )


def _states() -> MetricFieldReviewStates:
    confirmed = MetricReviewState.CONFIRMED
    return MetricFieldReviewStates(
        geometry=confirmed,
        intent=confirmed,
        pot_attempt=confirmed,
        pot_result=confirmed,
        foul_result=confirmed,
        target_zone=confirmed,
        cue_ball_end=confirmed,
    )


def _metric_input(shot_id: UUID) -> MetricShotReviewInput:
    return MetricShotReviewInput(
        shot_id=shot_id,
        cue_ball_start_mm=MetricPointInput(x_mm=100.0, y_mm=200.0),
        object_ball_start_mm=MetricPointInput(x_mm=500.0, y_mm=200.0),
        intended_object_ball=3,
        intended_target_pocket=PocketCode.BOTTOM_RIGHT,
        pot_attempt_status=PotAttemptStatus.ATTEMPTED,
        pot_result=PotResult.MISS,
        foul_result=FoulResult.LEGAL,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        target_zone=MetricTargetZoneInput(
            shape=TargetZoneShape.RECTANGLE,
            minimum_x_mm=1000.0,
            minimum_y_mm=500.0,
            maximum_x_mm=1100.0,
            maximum_y_mm=600.0,
        ),
        cue_ball_end_mm=MetricPointInput(x_mm=1120.0, y_mm=600.0),
        target_zone_evaluation_status=TargetZoneEvaluationStatus.ELIGIBLE,
        review_states=_states(),
    )


def test_metric_assembler_binds_server_authority_and_computes_target_zone_result() -> None:
    approved = _shot(1, ReviewStatus.APPROVED)
    rejected = _shot(2, ReviewStatus.REJECTED)
    assembler = HumanReviewMetricAuthorityAssembler(_CalibrationProvider(_calibration()))

    authority = assembler.assemble(
        session_id=_id(1),
        review_revision=4,
        result_shots=(approved, rejected),
        inputs=(_metric_input(approved.id),),
    )

    assert authority.approval_kind is EvidenceApprovalKind.HUMAN_REVIEW
    assert authority.approval_policy_version == "human-review-v1"
    assert len(authority.metric_shots) == 1
    metric = authority.metric_shots[0]
    assert (metric.session_id, metric.sequence, metric.review_revision) == (_id(1), 1, 4)
    assert metric.table_geometry.calibration_id == _id(30)
    assert metric.target_zone_result is TargetZoneResult.INSIDE_OR_TOLERATED
    assert metric.field_quality.geometry.source is FieldSource.MANUAL
    assert metric.field_quality.geometry.confidence == 1.0


def test_metric_assembler_maps_abstained_review_state_to_unknown_zero_quality() -> None:
    shot = _shot(1, ReviewStatus.APPROVED)
    metric_input = _metric_input(shot.id).model_copy(
        update={
            "foul_result": FoulResult.ABSTAINED,
            "review_states": _states().model_copy(
                update={"foul_result": MetricReviewState.ABSTAINED}
            ),
        }
    )
    metric = (
        HumanReviewMetricAuthorityAssembler(_CalibrationProvider(_calibration()))
        .assemble(
            session_id=shot.session_id,
            review_revision=2,
            result_shots=(shot,),
            inputs=(metric_input,),
        )
        .metric_shots[0]
    )

    assert metric.field_quality.foul_result.source is FieldSource.UNKNOWN
    assert metric.field_quality.foul_result.confidence == 0.0


def test_metric_input_rejects_unavailable_status_claimed_as_confirmed() -> None:
    shot = _shot(1, ReviewStatus.APPROVED)
    payload = _metric_input(shot.id).model_dump(mode="python")
    payload["foul_result"] = FoulResult.UNSUPPORTED
    with pytest.raises(ValueError):
        MetricShotReviewInput.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "inputs",
    [
        (),
        (_metric_input(_id(101)), _metric_input(_id(102))),
        (_metric_input(_id(102)),),
    ],
)
def test_metric_assembler_rejects_missing_extra_or_rejected_shot_inputs(inputs) -> None:
    shots = (_shot(1, ReviewStatus.APPROVED), _shot(2, ReviewStatus.REJECTED))
    with pytest.raises(MetricCalibrationAuthorityError):
        HumanReviewMetricAuthorityAssembler(_CalibrationProvider(_calibration())).assemble(
            session_id=_id(1),
            review_revision=1,
            result_shots=shots,
            inputs=inputs,
        )


def test_metric_assembler_rejects_non_active_calibration() -> None:
    shot = _shot(1, ReviewStatus.APPROVED)
    with pytest.raises(MetricCalibrationAuthorityError):
        HumanReviewMetricAuthorityAssembler(
            _CalibrationProvider(_calibration(status=CalibrationStatus.STALE))
        ).assemble(
            session_id=shot.session_id,
            review_revision=1,
            result_shots=(shot,),
            inputs=(_metric_input(shot.id),),
        )
