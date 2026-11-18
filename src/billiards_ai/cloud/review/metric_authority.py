"""Human-reviewed metric inputs and server-bound authority assembly."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, model_validator

from billiards_ai.shared.models.calibration import CalibrationLineage, CalibrationStatus
from billiards_ai.shared.models.events import CoreContractModel
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
    ReviewedTargetZone,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
    TargetZoneShape,
)
from billiards_ai.shared.models.shot import ReviewStatus, Shot, TargetZoneSource


class MetricReviewState(StrEnum):
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class MetricPointInput(CoreContractModel):
    x_mm: float = Field(ge=0.0, le=2540.0)
    y_mm: float = Field(ge=0.0, le=1270.0)


class MetricTargetZoneInput(CoreContractModel):
    shape: TargetZoneShape
    minimum_x_mm: float | None = Field(default=None, ge=0.0, le=2540.0)
    minimum_y_mm: float | None = Field(default=None, ge=0.0, le=1270.0)
    maximum_x_mm: float | None = Field(default=None, ge=0.0, le=2540.0)
    maximum_y_mm: float | None = Field(default=None, ge=0.0, le=1270.0)
    vertices: tuple[MetricPointInput, ...] = ()

    @model_validator(mode="after")
    def shape_fields_are_complete(self) -> MetricTargetZoneInput:
        bounds = (
            self.minimum_x_mm,
            self.minimum_y_mm,
            self.maximum_x_mm,
            self.maximum_y_mm,
        )
        if self.shape is TargetZoneShape.RECTANGLE:
            if (
                any(value is None for value in bounds)
                or self.vertices
                or self.minimum_x_mm >= self.maximum_x_mm  # type: ignore[operator]
                or self.minimum_y_mm >= self.maximum_y_mm  # type: ignore[operator]
            ):
                raise ValueError("rectangle target zone requires canonical bounds")
        elif any(value is not None for value in bounds) or not 3 <= len(self.vertices) <= 16:
            raise ValueError("polygon target zone requires only 3-16 vertices")
        return self


class MetricFieldReviewStates(CoreContractModel):
    geometry: MetricReviewState
    intent: MetricReviewState
    pot_attempt: MetricReviewState
    pot_result: MetricReviewState
    foul_result: MetricReviewState
    target_zone: MetricReviewState
    cue_ball_end: MetricReviewState


class MetricShotReviewInput(CoreContractModel):
    shot_id: UUID4
    cue_ball_start_mm: MetricPointInput | None
    object_ball_start_mm: MetricPointInput | None
    intended_object_ball: StrictInt | None = Field(default=None, ge=1, le=15)
    intended_target_pocket: PocketCode | None
    pot_attempt_status: PotAttemptStatus
    pot_result: PotResult | None
    foul_result: FoulResult
    target_zone_source: TargetZoneSource
    target_zone: MetricTargetZoneInput | None
    cue_ball_end_mm: MetricPointInput | None
    target_zone_evaluation_status: TargetZoneEvaluationStatus
    review_states: MetricFieldReviewStates

    @model_validator(mode="after")
    def unsupported_states_match_closed_statuses(self) -> MetricShotReviewInput:
        actual_attempt_state = {
            PotAttemptStatus.UNSUPPORTED: MetricReviewState.UNSUPPORTED,
            PotAttemptStatus.ABSTAINED: MetricReviewState.ABSTAINED,
        }.get(self.pot_attempt_status)
        if (
            actual_attempt_state is not None
            and self.review_states.pot_attempt is not actual_attempt_state
        ):
            raise ValueError("unavailable pot attempt status requires the matching review state")
        expected_attempt = {
            MetricReviewState.UNSUPPORTED: PotAttemptStatus.UNSUPPORTED,
            MetricReviewState.ABSTAINED: PotAttemptStatus.ABSTAINED,
        }.get(self.review_states.pot_attempt)
        if expected_attempt is not None and self.pot_attempt_status is not expected_attempt:
            raise ValueError("pot attempt status must match its unavailable review state")
        actual_foul_state = {
            FoulResult.UNSUPPORTED: MetricReviewState.UNSUPPORTED,
            FoulResult.ABSTAINED: MetricReviewState.ABSTAINED,
        }.get(self.foul_result)
        if (
            actual_foul_state is not None
            and self.review_states.foul_result is not actual_foul_state
        ):
            raise ValueError("unavailable foul status requires the matching review state")
        expected_foul = {
            MetricReviewState.UNSUPPORTED: FoulResult.UNSUPPORTED,
            MetricReviewState.ABSTAINED: FoulResult.ABSTAINED,
        }.get(self.review_states.foul_result)
        if expected_foul is not None and self.foul_result is not expected_foul:
            raise ValueError("foul status must match its unavailable review state")
        actual_zone_state = {
            TargetZoneEvaluationStatus.UNSUPPORTED: MetricReviewState.UNSUPPORTED,
            TargetZoneEvaluationStatus.ABSTAINED: MetricReviewState.ABSTAINED,
        }.get(self.target_zone_evaluation_status)
        if (
            actual_zone_state is not None
            and self.review_states.target_zone is not actual_zone_state
        ):
            raise ValueError("unavailable target-zone status requires the matching review state")
        expected_zone = {
            MetricReviewState.UNSUPPORTED: TargetZoneEvaluationStatus.UNSUPPORTED,
            MetricReviewState.ABSTAINED: TargetZoneEvaluationStatus.ABSTAINED,
        }.get(self.review_states.target_zone)
        if expected_zone is not None and self.target_zone_evaluation_status is not expected_zone:
            raise ValueError("target-zone status must match its unavailable review state")
        return self


class MetricCalibrationAuthorityProvider(Protocol):
    def calibration_for(self, *, session_id: UUID) -> CalibrationLineage: ...


class MetricCalibrationAuthorityError(RuntimeError):
    """Metric facts cannot be bound to one complete trusted authority."""


def _quality(state: MetricReviewState) -> ReviewedFieldQuality:
    if state in {MetricReviewState.CONFIRMED, MetricReviewState.CORRECTED}:
        return ReviewedFieldQuality(source=FieldSource.MANUAL, confidence=1.0)
    return ReviewedFieldQuality(source=FieldSource.UNKNOWN, confidence=0.0)


class HumanReviewMetricAuthorityAssembler:
    def __init__(self, calibration_provider: MetricCalibrationAuthorityProvider) -> None:
        self._calibration_provider = calibration_provider

    def assemble(
        self,
        *,
        session_id: UUID,
        review_revision: int,
        result_shots: tuple[Shot, ...],
        inputs: tuple[MetricShotReviewInput, ...],
    ) -> ApprovedMetricEvidenceAuthorityRecord:
        try:
            if type(review_revision) is not int or review_revision < 1:
                raise ValueError("review revision must be a positive integer")
            shots = tuple(
                Shot.model_validate(shot.model_dump(mode="python"), strict=True)
                for shot in result_shots
            )
            reviewed_inputs = tuple(
                MetricShotReviewInput.model_validate(item.model_dump(mode="python"), strict=True)
                for item in inputs
            )
            if not shots or any(shot.session_id != session_id for shot in shots):
                raise ValueError("result shots must belong to the requested session")
            promotable = tuple(
                shot
                for shot in sorted(shots, key=lambda item: (item.sequence, item.id))
                if shot.review_status in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
            )
            if tuple(item.shot_id for item in reviewed_inputs) != tuple(
                shot.id for shot in promotable
            ):
                raise ValueError("metric inputs must exactly follow promotable result shots")
            calibration = CalibrationLineage.model_validate(
                self._calibration_provider.calibration_for(session_id=session_id).model_dump(
                    mode="python"
                ),
                strict=True,
            )
            if (
                calibration.status is not CalibrationStatus.ACTIVE
                or calibration.table_length_mm != 2540
                or calibration.table_width_mm != 1270
            ):
                raise ValueError("metric calibration must be active and supported")
            metric_shots = tuple(
                self._assemble_shot(
                    shot=shot,
                    item=item,
                    review_revision=review_revision,
                    calibration=calibration,
                )
                for shot, item in zip(promotable, reviewed_inputs, strict=True)
            )
            return ApprovedMetricEvidenceAuthorityRecord(
                approval_kind=EvidenceApprovalKind.HUMAN_REVIEW,
                approval_policy_version="human-review-v1",
                metric_shots=metric_shots,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise MetricCalibrationAuthorityError("metric authority is invalid") from exc

    @staticmethod
    def _assemble_shot(
        *,
        shot: Shot,
        item: MetricShotReviewInput,
        review_revision: int,
        calibration: CalibrationLineage,
    ) -> ApprovedMetricShotEvidence:
        lineage = {
            "calibration_id": calibration.calibration_id,
            "calibration_revision_id": calibration.revision_id,
            "calibration_revision_number": calibration.revision_number,
        }

        def point(value: MetricPointInput | None) -> ReviewedPoint | None:
            if value is None:
                return None
            return ReviewedPoint(**lineage, x_mm=value.x_mm, y_mm=value.y_mm)

        zone = None
        if item.target_zone is not None:
            zone = ReviewedTargetZone(
                **lineage,
                shape=item.target_zone.shape,
                minimum_x_mm=item.target_zone.minimum_x_mm,
                minimum_y_mm=item.target_zone.minimum_y_mm,
                maximum_x_mm=item.target_zone.maximum_x_mm,
                maximum_y_mm=item.target_zone.maximum_y_mm,
                vertices=tuple(point(vertex) for vertex in item.target_zone.vertices),
            )
        cue_ball_end = point(item.cue_ball_end_mm)
        target_result = None
        if item.target_zone_evaluation_status is TargetZoneEvaluationStatus.ELIGIBLE:
            if zone is None or cue_ball_end is None:
                raise ValueError("eligible target zone requires zone and cue-ball end")
            target_result = (
                TargetZoneResult.INSIDE_OR_TOLERATED
                if zone.error_mm(cue_ball_end) <= 20.0
                else TargetZoneResult.OUTSIDE
            )
        states = item.review_states
        return ApprovedMetricShotEvidence(
            shot_id=shot.id,
            session_id=shot.session_id,
            sequence=shot.sequence,
            review_revision=review_revision,
            game_code=GameCode.CHINESE_EIGHT_BALL,
            table_geometry=ReviewedTableGeometry(
                **lineage,
                table_length_mm=2540,
                table_width_mm=1270,
                ball_diameter_mm=57.15,
                corner_pocket_mouth_mm=86,
                side_pocket_mouth_mm=92,
            ),
            cue_ball_start_mm=point(item.cue_ball_start_mm),
            object_ball_start_mm=point(item.object_ball_start_mm),
            intended_object_ball=item.intended_object_ball,
            intended_target_pocket=item.intended_target_pocket,
            pot_attempt_status=item.pot_attempt_status,
            pot_result=item.pot_result,
            foul_result=item.foul_result,
            target_zone_source=item.target_zone_source,
            target_zone=zone,
            cue_ball_end_mm=cue_ball_end,
            target_zone_evaluation_status=item.target_zone_evaluation_status,
            target_zone_result=target_result,
            field_quality=MetricFieldQuality(
                geometry=_quality(states.geometry),
                intent=_quality(states.intent),
                pot_attempt=_quality(states.pot_attempt),
                pot_result=_quality(states.pot_result),
                foul_result=_quality(states.foul_result),
                target_zone=_quality(states.target_zone),
                cue_ball_end=_quality(states.cue_ball_end),
            ),
        )
