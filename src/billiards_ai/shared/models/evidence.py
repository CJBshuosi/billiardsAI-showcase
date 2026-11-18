"""Production evidence contracts between perception, review, and growth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import UUID4, Field, PrivateAttr, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
)
from billiards_ai.shared.models.quality import ShotAnalysisStatus, ShotFieldQuality
from billiards_ai.shared.models.review import ApprovedShotData
from billiards_ai.shared.models.scoring import SessionPerformanceScores
from billiards_ai.shared.models.segmentation import BallPosition
from billiards_ai.shared.models.shot import (
    Confidence,
    ReviewStatus,
    Shot,
    TargetZoneSource,
)
from billiards_ai.shared.models.statistics import SessionStatistics


class EvidenceEventKind(StrEnum):
    FIRST_CONTACT = "FIRST_CONTACT"
    CUSHION_CONTACT = "CUSHION_CONTACT"
    POT = "POT"
    CUE_SCRATCH = "CUE_SCRATCH"


class EvidenceSupportStatus(StrEnum):
    OBSERVED = "OBSERVED"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class AtomicEventEvidence(CoreContractModel):
    event_id: UUID4
    shot_id: UUID4
    kind: EvidenceEventKind
    support_status: EvidenceSupportStatus
    occurred_at_ms: StrictInt | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: NonEmptyStr

    @model_validator(mode="after")
    def support_matches_observation(self) -> AtomicEventEvidence:
        if self.support_status is EvidenceSupportStatus.OBSERVED:
            if self.occurred_at_ms is None or self.confidence <= 0.0:
                raise ValueError("observed event requires time and positive confidence")
        elif self.occurred_at_ms is not None or self.confidence != 0.0:
            raise ValueError("unsupported or abstained event cannot claim an observation")
        return self


class PerceptionRunLineage(CoreContractModel):
    run_id: UUID4
    pipeline_version: NonEmptyStr
    model_versions: tuple[NonEmptyStr, ...]
    started_at: datetime
    completed_at: datetime
    input_content_sha256: Sha256

    _started_at_utc = field_validator("started_at")(require_utc)
    _completed_at_utc = field_validator("completed_at")(require_utc)

    @model_validator(mode="after")
    def completion_follows_start(self) -> PerceptionRunLineage:
        if self.completed_at < self.started_at:
            raise ValueError("perception completion must not precede start")
        return self


class FinalizedShotEvidence(CoreContractModel):
    """Immutable raw-free projection produced from the existing Shot contract."""

    shot_id: UUID4
    session_id: UUID4
    sequence: StrictInt = Field(ge=1)
    t_start_ms: StrictInt = Field(ge=0)
    t_end_ms: StrictInt = Field(ge=0)
    ball_positions_before: tuple[BallPosition, ...]
    ball_positions_after: tuple[BallPosition, ...]
    cue_ball_end: Point | None
    struck_ball: StrictInt | None
    potted_balls: tuple[StrictInt, ...]
    cushion_count: StrictInt | None = Field(default=None, ge=0)
    travel_distance_mm: StrictInt | None = Field(default=None, ge=0)
    is_foul: bool | None
    target_zone_source: TargetZoneSource
    deviation_mm: StrictInt | None = Field(default=None, ge=0)
    confidence: Confidence
    field_quality: ShotFieldQuality | None
    analysis_status: ShotAnalysisStatus
    review_status: ReviewStatus
    created_at: datetime

    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def interval_and_positions_are_canonical(self) -> FinalizedShotEvidence:
        if self.t_end_ms <= self.t_start_ms:
            raise ValueError("finalized shot interval must be positive")
        for positions in (self.ball_positions_before, self.ball_positions_after):
            ids = tuple(item.ball_id for item in positions)
            if ids != tuple(sorted(set(ids))):
                raise ValueError("ball positions must be sorted and unique")
        return self

    @classmethod
    def from_shot(cls, shot: Shot) -> FinalizedShotEvidence:
        def positions(values: dict[int, Point] | None) -> tuple[BallPosition, ...]:
            return tuple(
                BallPosition(ball_id=ball_id, position=point)
                for ball_id, point in sorted((values or {}).items())
            )

        return cls(
            shot_id=shot.id,
            session_id=shot.session_id,
            sequence=shot.sequence,
            t_start_ms=shot.t_start_ms,
            t_end_ms=shot.t_end_ms,
            ball_positions_before=positions(shot.ball_positions_before),
            ball_positions_after=positions(shot.ball_positions_after),
            cue_ball_end=shot.cue_ball_end,
            struck_ball=shot.struck_ball,
            potted_balls=tuple(shot.potted_balls or ()),
            cushion_count=shot.cushion_count,
            travel_distance_mm=shot.travel_distance_mm,
            is_foul=shot.is_foul,
            target_zone_source=shot.target_zone_source,
            deviation_mm=shot.deviation_mm,
            confidence=shot.confidence,
            field_quality=shot.field_quality,
            analysis_status=shot.analysis_status,
            review_status=shot.review_status,
            created_at=shot.created_at,
        )


class ShotEvidenceEnvelope(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    envelope_id: UUID4
    session_id: UUID4
    shot: FinalizedShotEvidence
    event_evidence: tuple[AtomicEventEvidence, ...]
    perception_run: PerceptionRunLineage
    calibration: CalibrationLineage
    media_manifest_sha256: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def lineage_identifies_one_shot(self) -> ShotEvidenceEnvelope:
        if self.shot.session_id != self.session_id:
            raise ValueError("envelope and shot session must match")
        if any(item.shot_id != self.shot.shot_id for item in self.event_evidence):
            raise ValueError("event evidence must identify the envelope shot")
        event_ids = tuple(item.event_id for item in self.event_evidence)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event evidence ids must be unique")
        return self


class ApprovedEvidenceBatch(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    batch_id: UUID4
    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    review_revision: StrictInt = Field(ge=1)
    approved_at: datetime
    shots: tuple[ApprovedShotData, ...] = Field(min_length=1)
    statistics: SessionStatistics
    scores: SessionPerformanceScores
    source_envelope_ids: tuple[UUID4, ...] = Field(min_length=1)
    content_sha256: Sha256

    _approved_at_utc = field_validator("approved_at")(require_utc)

    @model_validator(mode="after")
    def approved_projection_is_consistent(self) -> ApprovedEvidenceBatch:
        if any(shot.session_id != self.session_id for shot in self.shots):
            raise ValueError("approved shots must belong to the batch session")
        approved = {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
        if any(shot.review_status not in approved for shot in self.shots):
            raise ValueError("batch may contain only approved or corrected shots")
        sequences = tuple(shot.sequence for shot in self.shots)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("approved shot sequences must be sorted and unique")
        if self.statistics.total_shot_count != len(self.shots):
            raise ValueError("statistics total must match approved shots")
        if len(self.source_envelope_ids) != len(set(self.source_envelope_ids)):
            raise ValueError("source envelope ids must be unique")
        return self


class ApprovedEvidenceBatchV2(CoreContractModel):
    """Breaking v2 batch with reviewed raw-free metric evidence."""

    schema_version: Literal["2.0"] = "2.0"
    batch_id: UUID4
    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    review_revision: StrictInt = Field(ge=1)
    approved_at: datetime
    approval_kind: EvidenceApprovalKind
    approval_policy_version: NonEmptyStr
    shots: tuple[ApprovedShotData, ...] = Field(min_length=1)
    metric_shots: tuple[ApprovedMetricShotEvidence, ...] = Field(min_length=1)
    statistics: SessionStatistics
    scores: SessionPerformanceScores
    source_envelope_ids: tuple[UUID4, ...] = Field(min_length=1)
    content_sha256: Sha256

    _approved_at_utc = field_validator("approved_at")(require_utc)

    @model_validator(mode="after")
    def reviewed_metric_projection_is_consistent(self) -> ApprovedEvidenceBatchV2:
        approved_policy = (
            "human-review-v1"
            if self.approval_kind is EvidenceApprovalKind.HUMAN_REVIEW
            else "automated-policy-rd-v1"
        )
        if self.approval_policy_version != approved_policy:
            raise ValueError("approval kind requires its immutable approved policy version")
        if self.report_revision != self.review_revision:
            raise ValueError("report and review revision must match")
        approved = {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
        if any(
            shot.session_id != self.session_id or shot.review_status not in approved
            for shot in self.shots
        ):
            raise ValueError("v2 batch may contain only same-session approved shots")
        shot_keys = tuple((shot.shot_id, shot.session_id, shot.sequence) for shot in self.shots)
        metric_keys = tuple(
            (shot.shot_id, shot.session_id, shot.sequence) for shot in self.metric_shots
        )
        shot_sequences = tuple(shot.sequence for shot in self.shots)
        metric_sequences = tuple(shot.sequence for shot in self.metric_shots)
        if shot_sequences != tuple(sorted(set(shot_sequences))):
            raise ValueError("approved shot sequences must be sorted and unique")
        if metric_sequences != shot_sequences:
            raise ValueError("metric shot sequences must exactly match approved shots")
        if len(shot_keys) != len(set(shot_keys)):
            raise ValueError("approved shot lineage must be sorted and unique")
        if metric_keys != shot_keys:
            raise ValueError("metric shots must exactly match approved shot lineage")
        if any(shot.review_revision != self.review_revision for shot in self.metric_shots):
            raise ValueError("metric shot review revision must match batch")
        calibration_keys = {
            (
                shot.table_geometry.calibration_id,
                shot.table_geometry.calibration_revision_id,
                shot.table_geometry.calibration_revision_number,
            )
            for shot in self.metric_shots
        }
        if len(calibration_keys) != 1:
            raise ValueError("metric shots must share one reviewed calibration lineage")
        if self.statistics.total_shot_count != len(self.shots):
            raise ValueError("statistics total must match approved shots")
        if len(self.source_envelope_ids) != len(self.shots) or len(
            set(self.source_envelope_ids)
        ) != len(self.source_envelope_ids):
            raise ValueError("source envelope ids must be one-to-one and unique")
        return self


ApprovedEvidenceBatchAny = ApprovedEvidenceBatch | ApprovedEvidenceBatchV2

_TRUSTED_METRIC_ADMISSION_SEAL = object()


@dataclass(frozen=True, slots=True)
class _MetricAdmissionBinding:
    wrapper_schema_version: str
    admission_id: UUID
    admitted_at: datetime
    declared_batch_schema_version: str
    declared_batch_id: UUID
    declared_session_id: UUID
    declared_report_id: UUID
    declared_review_revision: int
    declared_batch_content_sha256: str
    declared_approval_kind: EvidenceApprovalKind
    declared_approval_policy_version: str
    batch_schema_version: str
    batch_id: UUID
    batch_session_id: UUID
    batch_report_id: UUID
    batch_review_revision: int
    batch_content_sha256: str
    batch_approval_kind: EvidenceApprovalKind
    batch_approval_policy_version: str
    public_payload_sha256: str


class AcceptedMetricEvidenceInput(CoreContractModel):
    """Versioned authority admission consumed by the pure metric projector."""

    schema_version: Literal["1.0"] = "1.0"
    admission_id: UUID4
    admitted_at: datetime
    batch_schema_version: Literal["2.0"]
    batch_id: UUID4
    session_id: UUID4
    report_id: UUID4
    review_revision: StrictInt = Field(ge=1)
    batch_content_sha256: Sha256
    approval_kind: EvidenceApprovalKind
    approval_policy_version: NonEmptyStr
    batch: ApprovedEvidenceBatchV2
    _admission_seal: object | None = PrivateAttr(default=None)
    _admission_binding: _MetricAdmissionBinding | None = PrivateAttr(default=None)
    _admission_instance_id: int | None = PrivateAttr(default=None)

    _admitted_at_utc = field_validator("admitted_at")(require_utc)

    @property
    def is_trusted_admission(self) -> bool:
        if (
            self._admission_seal is not _TRUSTED_METRIC_ADMISSION_SEAL
            or self._admission_binding is None
            or self._admission_instance_id != id(self)
        ):
            return False
        try:
            return self._admission_binding == _metric_admission_binding(self)
        except Exception:
            return False

    @model_validator(mode="after")
    def admission_binds_accepted_batch(self) -> AcceptedMetricEvidenceInput:
        batch = self.batch
        if (
            self.batch_schema_version != batch.schema_version
            or self.batch_id != batch.batch_id
            or self.session_id != batch.session_id
            or self.report_id != batch.report_id
            or self.review_revision != batch.review_revision
            or self.batch_content_sha256 != batch.content_sha256
            or self.approval_kind is not batch.approval_kind
            or self.approval_policy_version != batch.approval_policy_version
        ):
            raise ValueError("accepted admission must exactly bind batch and authority metadata")
        return self


def _metric_admission_binding(value: AcceptedMetricEvidenceInput) -> _MetricAdmissionBinding:
    batch = value.batch
    return _MetricAdmissionBinding(
        wrapper_schema_version=value.schema_version,
        admission_id=value.admission_id,
        admitted_at=value.admitted_at,
        declared_batch_schema_version=value.batch_schema_version,
        declared_batch_id=value.batch_id,
        declared_session_id=value.session_id,
        declared_report_id=value.report_id,
        declared_review_revision=value.review_revision,
        declared_batch_content_sha256=value.batch_content_sha256,
        declared_approval_kind=value.approval_kind,
        declared_approval_policy_version=value.approval_policy_version,
        batch_schema_version=batch.schema_version,
        batch_id=batch.batch_id,
        batch_session_id=batch.session_id,
        batch_report_id=batch.report_id,
        batch_review_revision=batch.review_revision,
        batch_content_sha256=batch.content_sha256,
        batch_approval_kind=batch.approval_kind,
        batch_approval_policy_version=batch.approval_policy_version,
        public_payload_sha256=canonical_sha256(value.model_dump(mode="json")),
    )


def _seal_accepted_metric_evidence_input(
    value: AcceptedMetricEvidenceInput,
) -> AcceptedMetricEvidenceInput:
    """Seal an admission created inside the trusted Story 8.2 repository boundary."""

    value._admission_binding = _metric_admission_binding(value)
    value._admission_instance_id = id(value)
    value._admission_seal = _TRUSTED_METRIC_ADMISSION_SEAL
    return value
