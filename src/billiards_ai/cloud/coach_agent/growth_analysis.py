"""Evidence-bounded Story 9.4 growth analysis draft workflow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import UUID4, TypeAdapter, ValidationError, model_validator

from billiards_ai.cloud.coach_agent.context import (
    CoachAgentContext,
    CoachSourceReadFenceV3,
    CoachTaskType,
)
from billiards_ai.cloud.coach_agent.prompts import get_growth_analysis_prompt
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelProvider,
    CoachModelProviderAuditPort,
    CoachModelRequest,
    CoachModelResponse,
    revalidate_coach_model_result,
)
from billiards_ai.cloud.growth.snapshot_consumer import growth_trend_lineage_payloads
from billiards_ai.shared.models.coaching import (
    AgentRun,
    AgentRunStatus,
    CoachAssessmentStatus,
    CoachAssessmentV2,
    CoachingFindingV2,
    FindingSupportStatus,
    FindingType,
    GrowthTrendLineageReferenceV1,
    TrendDirection,
)
from billiards_ai.shared.models.events import (
    CoreContractModel,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import GrowthEvidenceSnapshotV3
from billiards_ai.shared.models.growth_metrics import (
    MetricReliability,
    MetricSupportStatusV2,
)

_UUID4 = TypeAdapter(UUID4)
_TREND_TYPES = frozenset({FindingType.IMPROVEMENT, FindingType.DECLINE, FindingType.PLATEAU})
_EXPECTED_TREND = {
    FindingType.IMPROVEMENT: TrendDirection.IMPROVING,
    FindingType.DECLINE: TrendDirection.DECLINING,
    FindingType.PLATEAU: TrendDirection.STABLE,
}
_MAX_PROVIDER_METRICS = 64
_MAX_PROVIDER_PAYLOAD_BYTES = 32_768
_MAX_EVIDENCE_REFS_PER_METRIC = 256
_MAX_SNAPSHOT_EVIDENCE_REFS = 4_096
_MAX_AUDIT_LABEL_LENGTH = 128
_INVALID_CONTEXT_AUDIT_SHA256 = canonical_sha256(
    {"failure_identity": "growth-analysis-v1:INVALID_CONTEXT"}
)


class GrowthAnalysisCandidate(CoreContractModel):
    metric_identity_sha256: Sha256
    finding_type: FindingType


class GrowthAnalysisProviderOutput(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    primary: GrowthAnalysisCandidate | None
    secondary: GrowthAnalysisCandidate | None

    @model_validator(mode="after")
    def candidates_are_distinct(self) -> Self:
        if self.primary is not None and self.secondary is not None:
            if self.primary.metric_identity_sha256 == self.secondary.metric_identity_sha256:
                raise ValueError("primary and secondary candidates must use distinct metrics")
        return self


class GrowthAnalysisResultStatus(StrEnum):
    COMPLETED = "COMPLETED"
    NO_CONCLUSION = "NO_CONCLUSION"


class GrowthAnalysisFailureCode(StrEnum):
    INVALID_CONTEXT = "INVALID_CONTEXT"
    UNSUPPORTED_SNAPSHOT = "UNSUPPORTED_SNAPSHOT"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    INPUT_LIMIT_EXCEEDED = "INPUT_LIMIT_EXCEEDED"


class GrowthAnalysisCompleted(CoreContractModel):
    status: Literal[GrowthAnalysisResultStatus.COMPLETED] = GrowthAnalysisResultStatus.COMPLETED
    assessment: CoachAssessmentV2
    agent_run: AgentRun

    @model_validator(mode="after")
    def assessment_and_audit_are_closed(self) -> Self:
        try:
            assessment = CoachAssessmentV2.model_validate(dict(vars(self.assessment)), strict=True)
            audit = AgentRun.model_validate(dict(vars(self.agent_run)), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ValueError("invalid completed growth analysis result") from None
        if (
            audit.status is not AgentRunStatus.SUCCEEDED
            or assessment.agent_run_id != audit.run_id
            or audit.output_sha256 != assessment.content_sha256
            or audit.completed_at < audit.started_at
            or assessment.created_at != audit.completed_at
            or (
                assessment.primary_finding_id is None
                and assessment.secondary_finding_id is not None
            )
        ):
            raise ValueError("completed growth analysis result is not audit closed")
        return self


class GrowthAnalysisFailure(CoreContractModel):
    status: Literal[GrowthAnalysisResultStatus.NO_CONCLUSION] = (
        GrowthAnalysisResultStatus.NO_CONCLUSION
    )
    code: GrowthAnalysisFailureCode
    agent_run: AgentRun

    @model_validator(mode="after")
    def failed_audit_is_closed(self) -> Self:
        try:
            audit = AgentRun.model_validate(dict(vars(self.agent_run)), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ValueError("invalid failed growth analysis result") from None
        if (
            audit.status is not AgentRunStatus.FAILED
            or audit.output_sha256 is not None
            or audit.completed_at < audit.started_at
        ):
            raise ValueError("failed growth analysis result is not audit closed")
        return self


type GrowthAnalysisResult = GrowthAnalysisCompleted | GrowthAnalysisFailure


def _uuid4_from_payload(value: object) -> UUID:
    raw = bytearray(bytes.fromhex(canonical_sha256(value))[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _skill_codes(metric_assessment: object | None) -> tuple[str, ...]:
    if metric_assessment is None:
        return ("GENERAL",)
    taxonomy = metric_assessment.metric.taxonomy
    values = {taxonomy.ability_dimension.value}
    if taxonomy.technique_code is not None:
        values.add(taxonomy.technique_code.value)
    return tuple(sorted(values))


def _finding_id(
    *,
    snapshot: GrowthEvidenceSnapshotV3,
    finding_type: FindingType,
    metric_identity_sha256: str | None,
    evidence_ref_ids: tuple[UUID, ...],
) -> UUID:
    return _uuid4_from_payload(
        {
            "snapshot_content_sha256": snapshot.content_sha256,
            "finding_type": finding_type.value,
            "metric_identity_sha256": metric_identity_sha256,
            "evidence_ref_ids": tuple(str(item) for item in evidence_ref_ids),
        }
    )


def _restricted_finding(
    *,
    snapshot: GrowthEvidenceSnapshotV3,
    metric_assessment: object | None,
    requested_metric_identity_sha256: str | None = None,
) -> CoachingFindingV2:
    identity = (
        requested_metric_identity_sha256
        if metric_assessment is None
        else metric_assessment.metric_identity_sha256
    )
    evidence_ids = (
        ()
        if metric_assessment is None
        else tuple(sorted(item.evidence_ref_id for item in metric_assessment.evidence_refs))
    )
    support = FindingSupportStatus.LIMITED if evidence_ids else FindingSupportStatus.UNSUPPORTED
    metric_code = (
        "unknown metric"
        if metric_assessment is None
        else metric_assessment.metric.metric_code.value
    )
    return CoachingFindingV2(
        finding_id=_finding_id(
            snapshot=snapshot,
            finding_type=FindingType.INSUFFICIENT_EVIDENCE,
            metric_identity_sha256=identity,
            evidence_ref_ids=evidence_ids,
        ),
        finding_type=FindingType.INSUFFICIENT_EVIDENCE,
        finding_code=f"INSUFFICIENT_EVIDENCE:{metric_code}",
        affected_skill_codes=_skill_codes(metric_assessment),
        support_status=support,
        reliability=MetricReliability.UNKNOWN,
        evidence_ref_ids=evidence_ids,
        trend_lineage_id=None,
        summary=f"{metric_code} 当前证据不足，无法形成确定成长结论。",
        explanation=(
            "该结果只表示冻结快照尚不支持确定判断；不将 UNKNOWN 伪装为 STABLE，也不证明因果关系。"
        ),
        limitations=("需要更多可比且经过复核的结构化证据。",),
    )


def _substantive_finding(
    *,
    snapshot: GrowthEvidenceSnapshotV3,
    candidate: GrowthAnalysisCandidate,
    metric_assessment: object,
    lineage_by_metric: dict[str, GrowthTrendLineageReferenceV1],
) -> CoachingFindingV2 | None:
    metric = metric_assessment.metric
    metric_code = metric.metric_code.value
    trend_lineage: GrowthTrendLineageReferenceV1 | None = None
    if candidate.finding_type in _TREND_TYPES:
        trend_lineage = lineage_by_metric.get(candidate.metric_identity_sha256)
        if (
            trend_lineage is None
            or trend_lineage.trend_direction is not _EXPECTED_TREND[candidate.finding_type]
        ):
            return None
        evidence_ids = trend_lineage.evidence_ref_ids
        reliability = trend_lineage.reliability
    elif candidate.finding_type in {FindingType.STRENGTH, FindingType.WEAKNESS}:
        if (
            metric.support_status is not MetricSupportStatusV2.SUPPORTED
            or metric.reliability is MetricReliability.UNKNOWN
            or not metric_assessment.evidence_refs
        ):
            return None
        evidence_ids = tuple(
            sorted(item.evidence_ref_id for item in metric_assessment.evidence_refs)
        )
        reliability = metric.reliability
    else:
        return None
    return CoachingFindingV2(
        finding_id=_finding_id(
            snapshot=snapshot,
            finding_type=candidate.finding_type,
            metric_identity_sha256=candidate.metric_identity_sha256,
            evidence_ref_ids=evidence_ids,
        ),
        finding_type=candidate.finding_type,
        finding_code=f"{candidate.finding_type.value}:{metric_code}",
        affected_skill_codes=_skill_codes(metric_assessment),
        support_status=FindingSupportStatus.SUPPORTED,
        reliability=reliability,
        evidence_ref_ids=evidence_ids,
        trend_lineage_id=(None if trend_lineage is None else trend_lineage.lineage_id),
        summary=f"{metric_code} 的冻结证据与 {candidate.finding_type.value} 判断一致。",
        explanation=(
            "该判断仅解释输入 V3 快照已确定计算的指标、"
            "support、reliability 与趋势；不重算任何数值，也不证明因果关系。"
        ),
        limitations=("仅适用于该快照截止时间与冻结研发策略。",),
    )


def _provider_output(value: CoachModelResponse) -> GrowthAnalysisProviderOutput:
    try:
        encoded = json.dumps(
            value.structured_output,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return GrowthAnalysisProviderOutput.model_validate_json(encoded, strict=True)
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise ValueError("invalid growth analysis provider output") from None


def _audit_label(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_AUDIT_LABEL_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"provider {field_name} must be a bounded safe label")
    return value


def _allowed_finding_types(metric_assessment: object) -> list[str]:
    values = {FindingType.INSUFFICIENT_EVIDENCE.value}
    metric = metric_assessment.metric
    if (
        metric.support_status is MetricSupportStatusV2.SUPPORTED
        and metric.reliability is not MetricReliability.UNKNOWN
        and metric_assessment.evidence_refs
    ):
        values.update({FindingType.STRENGTH.value, FindingType.WEAKNESS.value})
    trend = metric_assessment.trend
    if (
        trend.support_status.value == "SUPPORTED"
        and trend.reliability is not MetricReliability.UNKNOWN
        and trend.direction is not TrendDirection.UNKNOWN
        and trend.current_evidence_ref_ids
        and trend.previous_evidence_ref_ids
    ):
        values.add(
            {
                TrendDirection.IMPROVING: FindingType.IMPROVEMENT,
                TrendDirection.DECLINING: FindingType.DECLINE,
                TrendDirection.STABLE: FindingType.PLATEAU,
            }[trend.direction].value
        )
    return sorted(values)


def _provider_payload(snapshot: GrowthEvidenceSnapshotV3) -> dict[str, object]:
    if len(snapshot.metric_assessments) > _MAX_PROVIDER_METRICS:
        raise ValueError("growth analysis metric limit exceeded")
    total_evidence_refs = 0
    candidates: list[dict[str, object]] = []
    for item in snapshot.metric_assessments:
        evidence_ref_count = len(item.evidence_refs)
        if evidence_ref_count > _MAX_EVIDENCE_REFS_PER_METRIC:
            raise ValueError("growth analysis per-metric evidence limit exceeded")
        total_evidence_refs += evidence_ref_count
        candidates.append(
            {
                "metric_identity_sha256": item.metric_identity_sha256,
                "metric_code": item.metric.metric_code.value,
                "metric_value_field_code": item.metric_value_field_code,
                "metric_value": getattr(item.metric, item.metric_value_field_code),
                "support_status": item.metric.support_status.value,
                "reliability": item.metric.reliability.value,
                "evidence_ref_count": evidence_ref_count,
                "trend_direction": item.trend.direction.value,
                "trend_support_status": item.trend.support_status.value,
                "trend_reliability": item.trend.reliability.value,
                "allowed_finding_types": _allowed_finding_types(item),
            }
        )
    if total_evidence_refs > _MAX_SNAPSHOT_EVIDENCE_REFS:
        raise ValueError("growth analysis total evidence limit exceeded")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "metric_candidates": candidates,
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_PROVIDER_PAYLOAD_BYTES:
        raise ValueError("growth analysis provider payload limit exceeded")
    return payload


def growth_analysis_provider_payload(snapshot: GrowthEvidenceSnapshotV3) -> dict[str, object]:
    """Return the bounded, lineage-free candidate payload used by generation and repair."""

    return _provider_payload(snapshot)


def parse_growth_analysis_provider_output(
    response: CoachModelResponse,
) -> GrowthAnalysisProviderOutput:
    """Parse the minimal provider selection without granting it fact ownership."""

    return _provider_output(response)


def assemble_growth_assessment_draft(
    *,
    snapshot: GrowthEvidenceSnapshotV3,
    provider_output: GrowthAnalysisProviderOutput,
    assessment_id: UUID,
    agent_run_id: UUID,
    created_at: datetime,
) -> CoachAssessmentV2:
    """Deterministically rebuild a DRAFT from exact V3 facts and a minimal selection."""

    assessments = {item.metric_identity_sha256: item for item in snapshot.metric_assessments}
    lineage = tuple(
        GrowthTrendLineageReferenceV1.model_validate_json(
            json.dumps(item, default=str, separators=(",", ":")),
            strict=True,
        )
        for item in growth_trend_lineage_payloads(snapshot)
    )
    lineage_by_metric = {item.metric_identity_sha256: item for item in lineage}
    findings: list[CoachingFindingV2] = []
    selected_ids: list[UUID | None] = []
    for candidate in (provider_output.primary, provider_output.secondary):
        if candidate is None:
            selected_ids.append(None)
            continue
        metric_assessment = assessments.get(candidate.metric_identity_sha256)
        substantive = (
            None
            if metric_assessment is None
            else _substantive_finding(
                snapshot=snapshot,
                candidate=candidate,
                metric_assessment=metric_assessment,
                lineage_by_metric=lineage_by_metric,
            )
        )
        finding = substantive or _restricted_finding(
            snapshot=snapshot,
            metric_assessment=metric_assessment,
            requested_metric_identity_sha256=candidate.metric_identity_sha256,
        )
        findings.append(finding)
        selected_ids.append(None if substantive is None else finding.finding_id)
    if not findings:
        findings.append(_restricted_finding(snapshot=snapshot, metric_assessment=None))
        selected_ids = [None, None]
    if selected_ids[0] is None:
        selected_ids = [selected_ids[1], None]
    ordered_findings = tuple(sorted(findings, key=lambda item: item.finding_id))
    evidence_ids = tuple(
        sorted(
            {
                reference.evidence_ref_id
                for item in snapshot.metric_assessments
                for reference in item.evidence_refs
            }
        )
    )
    assessment_payload: dict[str, object] = {
        "schema_version": "2.0",
        "assessment_id": assessment_id,
        "user_id": snapshot.user_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_version": snapshot.version,
        "snapshot_schema_version": "3.0",
        "snapshot_content_sha256": snapshot.content_sha256,
        "revision": 1,
        "agent_run_id": agent_run_id,
        "status": CoachAssessmentStatus.DRAFT,
        "created_at": created_at,
        "snapshot_evidence_ref_ids": evidence_ids,
        "snapshot_trend_lineage": lineage,
        "findings": ordered_findings,
        "primary_finding_id": selected_ids[0],
        "secondary_finding_id": selected_ids[1],
    }
    return CoachAssessmentV2(
        **assessment_payload,
        content_sha256=canonical_sha256(assessment_payload),
    )


class GrowthAnalysisWorkflow:
    """Generate one V3-bound assessment draft without validation, repair, or writes."""

    def __init__(
        self,
        *,
        provider: CoachModelProvider,
        uuid_factory: Callable[[], UUID],
        clock: Callable[[], datetime],
        prompt_version: str = "growth-analysis-v1",
        timeout_ms: int = 30_000,
    ):
        if (
            not isinstance(timeout_ms, int)
            or isinstance(timeout_ms, bool)
            or not 1 <= timeout_ms <= 600_000
        ):
            raise ValueError("timeout_ms must be a bounded strict integer")
        self._provider = provider
        audit_port = cast(CoachModelProviderAuditPort, provider)
        try:
            self._audit_provider_name = _audit_label(
                audit_port.audit_provider_name,
                "name",
            )
            self._audit_model_name = _audit_label(audit_port.audit_model_name, "model")
        except (AttributeError, RecursionError, TypeError, ValueError):
            raise ValueError("provider audit metadata is invalid") from None
        self._uuid_factory = uuid_factory
        self._clock = clock
        self._prompt = get_growth_analysis_prompt(prompt_version)
        self._timeout_ms = timeout_ms

    def _next_uuid(self) -> UUID:
        try:
            return _UUID4.validate_python(self._uuid_factory(), strict=True)
        except (TypeError, ValidationError, ValueError):
            raise ValueError("uuid_factory must return UUID4 values") from None

    def _next_time(self) -> datetime:
        try:
            return require_utc(self._clock())
        except (TypeError, ValueError):
            raise ValueError("clock must return UTC datetime values") from None

    def _audit(
        self,
        *,
        run_id: UUID,
        context_sha256: str,
        provider_name: str,
        provider_model: str,
        status: AgentRunStatus,
        started_at: datetime,
        completed_at: datetime,
        output_sha256: str | None = None,
    ) -> AgentRun:
        return AgentRun(
            run_id=run_id,
            task_type=CoachTaskType.GROWTH_ASSESSMENT.value,
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.content_sha256,
            context_sha256=context_sha256,
            provider_name=provider_name,
            provider_model=provider_model,
            output_sha256=output_sha256,
            status=status,
            repair_count=0,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _failure(
        self,
        *,
        code: GrowthAnalysisFailureCode,
        run_id: UUID,
        context_sha256: str,
        started_at: datetime,
        provider_name: str,
        provider_model: str,
    ) -> GrowthAnalysisFailure:
        return GrowthAnalysisFailure(
            code=code,
            agent_run=self._audit(
                run_id=run_id,
                context_sha256=context_sha256,
                provider_name=provider_name,
                provider_model=provider_model,
                status=AgentRunStatus.FAILED,
                started_at=started_at,
                completed_at=max(started_at, self._next_time()),
            ),
        )

    async def analyze(self, context: CoachAgentContext) -> GrowthAnalysisResult:
        started_at = self._next_time()
        run_id = self._next_uuid()
        try:
            exact_context = CoachAgentContext.model_validate(context, strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            return self._failure(
                code=GrowthAnalysisFailureCode.INVALID_CONTEXT,
                run_id=run_id,
                context_sha256=_INVALID_CONTEXT_AUDIT_SHA256,
                started_at=started_at,
                provider_name="NOT_CALLED",
                provider_model="NOT_CALLED",
            )
        if (
            exact_context.task_type is not CoachTaskType.GROWTH_ASSESSMENT
            or not isinstance(exact_context.growth_snapshot, GrowthEvidenceSnapshotV3)
            or not isinstance(exact_context.read_fence, CoachSourceReadFenceV3)
        ):
            return self._failure(
                code=GrowthAnalysisFailureCode.UNSUPPORTED_SNAPSHOT,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name="NOT_CALLED",
                provider_model="NOT_CALLED",
            )
        snapshot = exact_context.growth_snapshot
        try:
            provider_payload = growth_analysis_provider_payload(snapshot)
        except (AttributeError, RecursionError, TypeError, ValueError):
            return self._failure(
                code=GrowthAnalysisFailureCode.INPUT_LIMIT_EXCEEDED,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name="NOT_CALLED",
                provider_model="NOT_CALLED",
            )
        request = CoachModelRequest(
            task_type=CoachTaskType.GROWTH_ASSESSMENT,
            prompt_version=self._prompt.version,
            system_prompt=self._prompt.system_prompt,
            input_payload=provider_payload,
            output_schema=GrowthAnalysisProviderOutput.model_json_schema(),
            request_id=run_id,
            timeout_ms=self._timeout_ms,
        )
        try:
            raw_result = await self._provider.generate_structured(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._failure(
                code=GrowthAnalysisFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        try:
            result = revalidate_coach_model_result(raw_result)
        except ValueError:
            return self._failure(
                code=GrowthAnalysisFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        if isinstance(result, CoachModelFailure):
            return self._failure(
                code=GrowthAnalysisFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        try:
            provider_output = parse_growth_analysis_provider_output(result)
        except ValueError:
            return self._failure(
                code=GrowthAnalysisFailureCode.INVALID_OUTPUT,
                run_id=run_id,
                context_sha256=exact_context.context_sha256,
                started_at=started_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )

        assessment_id = self._next_uuid()
        completed_at = max(started_at, self._next_time())
        assessment = assemble_growth_assessment_draft(
            snapshot=snapshot,
            provider_output=provider_output,
            assessment_id=assessment_id,
            agent_run_id=run_id,
            created_at=completed_at,
        )
        audit = self._audit(
            run_id=run_id,
            context_sha256=exact_context.context_sha256,
            provider_name=result.provider,
            provider_model=result.model,
            status=AgentRunStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=completed_at,
            output_sha256=assessment.content_sha256,
        )
        return GrowthAnalysisCompleted(assessment=assessment, agent_run=audit)


__all__ = [
    "GrowthAnalysisCandidate",
    "GrowthAnalysisCompleted",
    "GrowthAnalysisFailure",
    "GrowthAnalysisFailureCode",
    "GrowthAnalysisProviderOutput",
    "GrowthAnalysisResult",
    "GrowthAnalysisResultStatus",
    "GrowthAnalysisWorkflow",
    "assemble_growth_assessment_draft",
    "growth_analysis_provider_payload",
    "parse_growth_analysis_provider_output",
]
