"""Evidence-bounded Coach Agent assessment and run audit contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import MetricSupportStatus, Reliability
from billiards_ai.shared.models.growth_metrics import MetricReliability


class CoachFindingCode(StrEnum):
    LARGE_ANGLE_POTTING = "LARGE_ANGLE_POTTING"
    CUE_BALL_STOP_BIAS = "CUE_BALL_STOP_BIAS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FindingSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    LIMITED = "LIMITED"
    UNSUPPORTED = "UNSUPPORTED"


class CoachFinding(CoreContractModel):
    finding_id: UUID4
    code: CoachFindingCode
    support_status: FindingSupportStatus
    reliability: Reliability
    evidence_ref_ids: tuple[UUID4, ...]
    summary: NonEmptyStr

    @model_validator(mode="after")
    def support_requires_evidence(self) -> CoachFinding:
        if self.support_status is FindingSupportStatus.UNSUPPORTED:
            if self.evidence_ref_ids:
                raise ValueError("unsupported finding cannot claim evidence")
        elif not self.evidence_ref_ids:
            raise ValueError("supported finding requires evidence references")
        return self


class CoachAssessment(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    assessment_id: UUID4
    user_id: UUID4
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    revision: StrictInt = Field(ge=1)
    agent_run_id: UUID4
    created_at: datetime
    snapshot_evidence_ref_ids: tuple[UUID4, ...]
    findings: tuple[CoachFinding, ...]
    primary_finding_id: UUID4 | None = None
    secondary_finding_id: UUID4 | None = None
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def findings_reference_snapshot_evidence(self) -> CoachAssessment:
        available = set(self.snapshot_evidence_ref_ids)
        if any(not set(item.evidence_ref_ids) <= available for item in self.findings):
            raise ValueError("finding references evidence absent from snapshot evidence")
        finding_ids = tuple(item.finding_id for item in self.findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding ids must be unique")
        for selected in (self.primary_finding_id, self.secondary_finding_id):
            if selected is not None and selected not in finding_ids:
                raise ValueError("selected finding must exist in assessment")
        if (
            self.primary_finding_id == self.secondary_finding_id
            and self.primary_finding_id is not None
        ):
            raise ValueError("primary and secondary findings must differ")
        return self


class FindingType(StrEnum):
    STRENGTH = "STRENGTH"
    WEAKNESS = "WEAKNESS"
    IMPROVEMENT = "IMPROVEMENT"
    DECLINE = "DECLINE"
    PLATEAU = "PLATEAU"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CoachAssessmentStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class TrendDirection(StrEnum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    UNKNOWN = "UNKNOWN"


class GrowthTrendLineageReferenceV1(CoreContractModel):
    """Raw-free seam for one exact D1 metric trend; it does not synthesize trends."""

    schema_version: Literal["1.0"] = "1.0"
    lineage_id: UUID4
    source_snapshot_schema_version: Literal["3.0"]
    metric_code: NonEmptyStr
    metric_identity_sha256: Sha256
    trend_direction: TrendDirection
    trend_support_status: MetricSupportStatus
    reliability: MetricReliability
    trend_policy_version: NonEmptyStr
    evidence_ref_ids: tuple[UUID4, ...]

    @model_validator(mode="after")
    def lineage_is_canonical_and_evidence_bounded(self) -> Self:
        if self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids))):
            raise ValueError("trend evidence reference IDs must be sorted and unique")
        if self.trend_direction is TrendDirection.UNKNOWN:
            if self.trend_support_status is MetricSupportStatus.SUPPORTED:
                raise ValueError("UNKNOWN trend cannot be supported")
            if self.reliability is not MetricReliability.UNKNOWN:
                raise ValueError("UNKNOWN trend lineage requires UNKNOWN reliability")
        elif self.trend_support_status is MetricSupportStatus.SUPPORTED:
            if not self.evidence_ref_ids:
                raise ValueError("supported trend requires evidence references")
            if self.reliability is MetricReliability.UNKNOWN:
                raise ValueError("supported trend cannot use UNKNOWN reliability")
        else:
            raise ValueError("substantive trend direction requires supported lineage")
        return self


_TREND_FINDING_TYPES = {
    FindingType.IMPROVEMENT,
    FindingType.DECLINE,
    FindingType.PLATEAU,
}


class CoachingFindingV2(CoreContractModel):
    schema_version: Literal["2.0"] = "2.0"
    finding_id: UUID4
    finding_type: FindingType
    finding_code: NonEmptyStr
    affected_skill_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    support_status: FindingSupportStatus
    reliability: MetricReliability
    evidence_ref_ids: tuple[UUID4, ...]
    trend_lineage_id: UUID4 | None = None
    summary: NonEmptyStr
    explanation: NonEmptyStr
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def finding_is_canonical_and_bounded(self) -> Self:
        if self.affected_skill_codes != tuple(sorted(set(self.affected_skill_codes))):
            raise ValueError("affected skill codes must be sorted and unique")
        if self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids))):
            raise ValueError("finding evidence reference IDs must be sorted and unique")
        if self.support_status is FindingSupportStatus.SUPPORTED and not self.evidence_ref_ids:
            raise ValueError("supported finding requires evidence references")
        if (
            self.support_status is FindingSupportStatus.SUPPORTED
            and self.reliability is MetricReliability.UNKNOWN
        ):
            raise ValueError("supported finding cannot use UNKNOWN reliability")
        if self.support_status is FindingSupportStatus.UNSUPPORTED and self.evidence_ref_ids:
            raise ValueError("unsupported finding cannot claim evidence")
        if (
            self.support_status is FindingSupportStatus.UNSUPPORTED
            and self.reliability is not MetricReliability.UNKNOWN
        ):
            raise ValueError("unsupported finding requires UNKNOWN reliability")
        if self.finding_type in _TREND_FINDING_TYPES:
            if self.trend_lineage_id is None:
                raise ValueError("trend finding requires D1 trend lineage")
        elif self.trend_lineage_id is not None:
            raise ValueError("non-trend finding cannot claim trend lineage")
        if self.finding_type is FindingType.INSUFFICIENT_EVIDENCE:
            if self.support_status is FindingSupportStatus.SUPPORTED:
                raise ValueError("insufficient-evidence finding cannot be supported")
            if not self.limitations:
                raise ValueError("insufficient-evidence finding requires limitations")
            if self.reliability is not MetricReliability.UNKNOWN:
                raise ValueError("insufficient-evidence finding requires UNKNOWN reliability")
        return self


class CoachAssessmentV2(CoreContractModel):
    schema_version: Literal["2.0"] = "2.0"
    assessment_id: UUID4
    user_id: UUID4
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    snapshot_schema_version: Literal["3.0"]
    snapshot_content_sha256: Sha256
    revision: StrictInt = Field(ge=1)
    agent_run_id: UUID4
    status: CoachAssessmentStatus
    created_at: datetime
    snapshot_evidence_ref_ids: tuple[UUID4, ...]
    snapshot_trend_lineage: tuple[GrowthTrendLineageReferenceV1, ...]
    findings: tuple[CoachingFindingV2, ...]
    primary_finding_id: UUID4 | None = None
    secondary_finding_id: UUID4 | None = None
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def assessment_is_canonical_hash_bound_and_lineage_closed(self) -> Self:
        evidence_ids = self.snapshot_evidence_ref_ids
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("snapshot evidence reference IDs must be sorted and unique")
        lineage = tuple(
            GrowthTrendLineageReferenceV1.model_validate(item.model_dump(mode="python"))
            for item in self.snapshot_trend_lineage
        )
        lineage_ids = tuple(item.lineage_id for item in lineage)
        if lineage_ids != tuple(sorted(set(lineage_ids))):
            raise ValueError("snapshot trend lineage must be sorted and unique")
        metric_identities = tuple(item.metric_identity_sha256 for item in lineage)
        if len(metric_identities) != len(set(metric_identities)):
            raise ValueError("snapshot trend lineage must bind unique metric identities")
        if any(not set(item.evidence_ref_ids) <= set(evidence_ids) for item in lineage):
            raise ValueError("trend lineage references evidence absent from snapshot evidence")
        findings = tuple(
            CoachingFindingV2.model_validate(item.model_dump(mode="python"))
            for item in self.findings
        )
        finding_ids = tuple(item.finding_id for item in findings)
        if finding_ids != tuple(sorted(set(finding_ids))):
            raise ValueError("findings must use sorted unique IDs")
        available_evidence = set(evidence_ids)
        available_lineage = {item.lineage_id: item for item in lineage}
        expected_directions = {
            FindingType.IMPROVEMENT: TrendDirection.IMPROVING,
            FindingType.DECLINE: TrendDirection.DECLINING,
            FindingType.PLATEAU: TrendDirection.STABLE,
        }
        for finding in findings:
            if not set(finding.evidence_ref_ids) <= available_evidence:
                raise ValueError("finding references evidence absent from snapshot evidence")
            if finding.finding_type in _TREND_FINDING_TYPES:
                source = available_lineage.get(finding.trend_lineage_id)
                if source is None:
                    raise ValueError("finding references trend lineage absent from exact snapshot")
                if (
                    source.trend_direction is not expected_directions[finding.finding_type]
                    or source.trend_support_status is not MetricSupportStatus.SUPPORTED
                    or source.reliability is not finding.reliability
                    or not set(finding.evidence_ref_ids) <= set(source.evidence_ref_ids)
                ):
                    raise ValueError(
                        "unknown or insufficient trend permits only limited or unsupported findings"
                    )
        for selected in (self.primary_finding_id, self.secondary_finding_id):
            if selected is not None and selected not in finding_ids:
                raise ValueError("selected finding must exist in assessment")
        if self.primary_finding_id is not None and (
            self.primary_finding_id == self.secondary_finding_id
        ):
            raise ValueError("primary and secondary findings must differ")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected_hash:
            raise ValueError("coach assessment content hash mismatch")
        return self


class AgentRunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    FALLBACK = "FALLBACK"


class AgentRun(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: UUID4
    task_type: NonEmptyStr
    prompt_version: NonEmptyStr
    prompt_sha256: Sha256
    context_sha256: Sha256
    provider_name: NonEmptyStr
    provider_model: NonEmptyStr
    output_sha256: Sha256 | None = None
    status: AgentRunStatus
    repair_count: StrictInt = Field(ge=0, le=1)
    started_at: datetime
    completed_at: datetime

    _started_at_utc = field_validator("started_at")(require_utc)
    _completed_at_utc = field_validator("completed_at")(require_utc)
