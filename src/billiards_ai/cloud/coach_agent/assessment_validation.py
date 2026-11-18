"""Deterministic, payload-free validation for Story 9.5 assessment drafts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    UUID4,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.coach_agent.context import (
    CoachAgentContext,
    CoachSourceReadFenceV3,
    CoachTaskType,
)
from billiards_ai.cloud.coach_agent.growth_analysis import (
    GrowthAnalysisCandidate,
    GrowthAnalysisProviderOutput,
    assemble_growth_assessment_draft,
)
from billiards_ai.cloud.coach_agent.prompts import get_growth_analysis_prompt
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
    NonEmptyStr,
    Sha256,
    canonical_sha256,
    require_utc,
)
from billiards_ai.shared.models.growth import GrowthEvidenceSnapshotV3
from billiards_ai.shared.models.growth_metrics import MetricReliability, MetricSupportStatusV2

ASSESSMENT_VALIDATION_POLICY_VERSION = "assessment-validation-rd-v1.0.0"
_MAX_PROVIDER_LABEL_LENGTH = 128
_NUMERIC_LITERAL = re.compile(
    r"[0-9０-９]|(?:百分之|[十百千万亿])|"
    r"[零〇一二三四五六七八九两几半]+(?=倍|次|个|位|成|分|秒|分钟|小时|球|杆)"
)
_CAUSAL_CLAIM = re.compile(
    r"(?:因为|由于|所以|导致|造成|证明了|源自|归因于|"
    r"\bbecause\b|\btherefore\b|\bcaused?\s+by\b|\bresults?\s+in\b)",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = re.compile(
    r"(?:https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|"
    r"(?<!\d)\d{17}[0-9Xx](?!\d)|\b(?:\d{1,3}\.){3}\d{1,3}\b|"
    r"(?:身份证|手机号|电子?邮箱|家庭?住址|真实姓名|微信号)|"
    r"\b(?:access[_ -]?token|api[_ -]?key|openid|system[_ -]?prompt|password|"
    r"client[_ -]?secret|authorization|bearer|session[_ -]?cookie|wechat)\b)",
    re.IGNORECASE,
)
_TREND_TYPES = frozenset({FindingType.IMPROVEMENT, FindingType.DECLINE, FindingType.PLATEAU})
_EXPECTED_DIRECTIONS = {
    FindingType.IMPROVEMENT: TrendDirection.IMPROVING,
    FindingType.DECLINE: TrendDirection.DECLINING,
    FindingType.PLATEAU: TrendDirection.STABLE,
}
_SCHEMA_FAILURE_ASSESSMENT_ID = UUID("00000000-0000-4000-8000-000000000001")
_SCHEMA_FAILURE_RUN_ID = UUID("00000000-0000-4000-8000-000000000002")
_SCHEMA_FAILURE_SHA256 = canonical_sha256({"failure": "INVALID_ASSESSMENT_SCHEMA"})
_UUID4_ADAPTER = TypeAdapter(UUID4)


class AssessmentViolationCode(StrEnum):
    SCHEMA = "SCHEMA"
    SNAPSHOT_LINEAGE = "SNAPSHOT_LINEAGE"
    CONTEXT_LINEAGE = "CONTEXT_LINEAGE"
    PROMPT_LINEAGE = "PROMPT_LINEAGE"
    PROVIDER_LINEAGE = "PROVIDER_LINEAGE"
    EVIDENCE_REFS = "EVIDENCE_REFS"
    NUMERIC_INTEGRITY = "NUMERIC_INTEGRITY"
    SUPPORT_STATUS = "SUPPORT_STATUS"
    RELIABILITY = "RELIABILITY"
    TREND_LINEAGE = "TREND_LINEAGE"
    FINDING_POLICY = "FINDING_POLICY"
    SELECTION_POLICY = "SELECTION_POLICY"


class AssessmentValidationStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class ValidationViolation(CoreContractModel):
    code: AssessmentViolationCode
    location: NonEmptyStr

    def sort_key(self) -> tuple[str, str]:
        return self.code.value, self.location


class AssessmentValidationReport(CoreContractModel):
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    assessment_content_sha256: Sha256
    context_sha256: Sha256
    agent_run_id: UUID4
    agent_run_sha256: Sha256
    repair_audit_sha256: Sha256 | None = None
    previous_report_sha256: Sha256 | None = None
    policy_version: Literal["assessment-validation-rd-v1.0.0"] = (
        ASSESSMENT_VALIDATION_POLICY_VERSION
    )
    status: AssessmentValidationStatus
    validated_at: datetime
    violations: tuple[ValidationViolation, ...]
    report_sha256: Sha256

    _validated_at_utc = field_validator("validated_at")(require_utc)

    @classmethod
    def build(
        cls,
        *,
        assessment_id: UUID,
        assessment_revision: int,
        assessment_content_sha256: str,
        context_sha256: str,
        agent_run_id: UUID,
        agent_run_sha256: str,
        validated_at: datetime,
        violations: tuple[ValidationViolation, ...],
        repair_audit_sha256: str | None = None,
        previous_report_sha256: str | None = None,
    ) -> AssessmentValidationReport:
        ordered = tuple(sorted(set(violations), key=lambda item: item.sort_key()))
        payload = {
            "assessment_id": assessment_id,
            "assessment_revision": assessment_revision,
            "assessment_content_sha256": assessment_content_sha256,
            "context_sha256": context_sha256,
            "agent_run_id": agent_run_id,
            "agent_run_sha256": agent_run_sha256,
            "repair_audit_sha256": repair_audit_sha256,
            "previous_report_sha256": previous_report_sha256,
            "policy_version": ASSESSMENT_VALIDATION_POLICY_VERSION,
            "status": (
                AssessmentValidationStatus.FAILED if ordered else AssessmentValidationStatus.PASSED
            ),
            "validated_at": require_utc(validated_at),
            "violations": ordered,
        }
        return cls(**payload, report_sha256=canonical_sha256(payload))

    @model_validator(mode="after")
    def report_is_canonical_and_hash_bound(self) -> Self:
        ordered = tuple(sorted(set(self.violations), key=lambda item: item.sort_key()))
        if self.violations != ordered:
            raise ValueError("validation violations must be canonical and unique")
        expected_status = (
            AssessmentValidationStatus.FAILED
            if self.violations
            else AssessmentValidationStatus.PASSED
        )
        if self.status is not expected_status:
            raise ValueError("validation status must match violations")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"report_sha256"}))
        if self.report_sha256 != expected_hash:
            raise ValueError("validation report hash mismatch")
        return self


def _violation(code: AssessmentViolationCode, location: str) -> ValidationViolation:
    return ValidationViolation(code=code, location=location)


def _stored_model(model: type[CoreContractModel], value: object) -> CoreContractModel:
    if not isinstance(value, model):
        raise ValueError("unexpected model type")
    try:
        return model.model_validate(dict(vars(value)), strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise ValueError("invalid stored model") from None


def _expected_skills(metric_assessment: object | None) -> tuple[str, ...]:
    if metric_assessment is None:
        return ("GENERAL",)
    taxonomy = metric_assessment.metric.taxonomy
    values = {taxonomy.ability_dimension.value}
    if taxonomy.technique_code is not None:
        values.add(taxonomy.technique_code.value)
    return tuple(sorted(values))


def _provider_label_is_safe(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and len(value) <= _MAX_PROVIDER_LABEL_LENGTH
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _safe_uuid(value: object, fallback: UUID) -> UUID:
    try:
        return _UUID4_ADAPTER.validate_python(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        return fallback


def _safe_sha256(value: object) -> str:
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return _SCHEMA_FAILURE_SHA256


def _safe_model_sha256(value: object) -> str:
    try:
        if isinstance(value, AgentRun):
            exact = _stored_model(AgentRun, value)
            return canonical_sha256(exact.model_dump(mode="json"))
    except (RecursionError, TypeError, ValueError):
        pass
    return _SCHEMA_FAILURE_SHA256


def _expected_finding_id(
    *,
    snapshot: GrowthEvidenceSnapshotV3,
    finding: CoachingFindingV2,
    metric_identity_sha256: str | None,
) -> UUID:
    raw = bytearray(
        bytes.fromhex(
            canonical_sha256(
                {
                    "snapshot_content_sha256": snapshot.content_sha256,
                    "finding_type": finding.finding_type.value,
                    "metric_identity_sha256": metric_identity_sha256,
                    "evidence_ref_ids": tuple(str(item) for item in finding.evidence_ref_ids),
                }
            )
        )[:16]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


class AssessmentValidator:
    """Run independent schema, lineage, evidence, numeric and policy checks."""

    def validate(
        self,
        *,
        assessment: object,
        context: object,
        agent_run: object,
        validated_at: datetime,
        repair_audit_sha256: str | None = None,
        previous_report_sha256: str | None = None,
    ) -> AssessmentValidationReport:
        require_utc(validated_at)
        try:
            exact_assessment = _stored_model(CoachAssessmentV2, assessment)
        except ValueError:
            return AssessmentValidationReport.build(
                assessment_id=_safe_uuid(
                    getattr(assessment, "assessment_id", None),
                    _SCHEMA_FAILURE_ASSESSMENT_ID,
                ),
                assessment_revision=(
                    assessment.revision
                    if isinstance(getattr(assessment, "revision", None), int)
                    and not isinstance(assessment.revision, bool)
                    and assessment.revision >= 1
                    else 1
                ),
                assessment_content_sha256=_safe_sha256(getattr(assessment, "content_sha256", None)),
                context_sha256=_safe_sha256(getattr(context, "context_sha256", None)),
                agent_run_id=_safe_uuid(getattr(agent_run, "run_id", None), _SCHEMA_FAILURE_RUN_ID),
                agent_run_sha256=_safe_model_sha256(agent_run),
                validated_at=validated_at,
                violations=(_violation(AssessmentViolationCode.SCHEMA, "assessment"),),
                repair_audit_sha256=repair_audit_sha256,
                previous_report_sha256=previous_report_sha256,
            )
        violations: list[ValidationViolation] = []
        try:
            exact_context = _stored_model(CoachAgentContext, context)
        except ValueError:
            exact_context = None
            violations.append(_violation(AssessmentViolationCode.SCHEMA, "context"))
        try:
            exact_run = _stored_model(AgentRun, agent_run)
        except ValueError:
            exact_run = None
            violations.append(_violation(AssessmentViolationCode.SCHEMA, "agent_run"))

        if exact_context is not None:
            self._validate_context_and_snapshot(exact_assessment, exact_context, violations)
        if exact_run is not None and exact_context is not None:
            self._validate_run(
                exact_assessment,
                exact_context,
                exact_run,
                violations,
                repaired=repair_audit_sha256 is not None,
            )
        if exact_context is not None and isinstance(
            exact_context.growth_snapshot, GrowthEvidenceSnapshotV3
        ):
            self._validate_findings(
                exact_assessment,
                exact_context.growth_snapshot,
                violations,
            )
        return AssessmentValidationReport.build(
            assessment_id=exact_assessment.assessment_id,
            assessment_revision=exact_assessment.revision,
            assessment_content_sha256=exact_assessment.content_sha256,
            context_sha256=(
                exact_context.context_sha256
                if exact_context is not None
                else _SCHEMA_FAILURE_SHA256
            ),
            agent_run_id=(exact_run.run_id if exact_run is not None else _SCHEMA_FAILURE_RUN_ID),
            agent_run_sha256=(
                canonical_sha256(exact_run.model_dump(mode="json"))
                if exact_run is not None
                else _SCHEMA_FAILURE_SHA256
            ),
            validated_at=validated_at,
            violations=tuple(violations),
            repair_audit_sha256=repair_audit_sha256,
            previous_report_sha256=previous_report_sha256,
        )

    @staticmethod
    def _validate_context_and_snapshot(
        assessment: CoachAssessmentV2,
        context: CoachAgentContext,
        violations: list[ValidationViolation],
    ) -> None:
        if (
            context.task_type is not CoachTaskType.GROWTH_ASSESSMENT
            or not isinstance(context.growth_snapshot, GrowthEvidenceSnapshotV3)
            or not isinstance(context.read_fence, CoachSourceReadFenceV3)
        ):
            violations.append(_violation(AssessmentViolationCode.CONTEXT_LINEAGE, "context"))
            return
        snapshot = context.growth_snapshot
        if (
            assessment.user_id != snapshot.user_id
            or assessment.snapshot_id != snapshot.snapshot_id
            or assessment.snapshot_version != snapshot.version
            or assessment.snapshot_schema_version != snapshot.schema_version
            or assessment.snapshot_content_sha256 != snapshot.content_sha256
        ):
            violations.append(
                _violation(AssessmentViolationCode.SNAPSHOT_LINEAGE, "assessment.snapshot")
            )
        if assessment.status is not CoachAssessmentStatus.DRAFT or assessment.revision != 1:
            violations.append(
                _violation(AssessmentViolationCode.FINDING_POLICY, "assessment.status")
            )

    @staticmethod
    def _validate_run(
        assessment: CoachAssessmentV2,
        context: CoachAgentContext,
        run: AgentRun,
        violations: list[ValidationViolation],
        *,
        repaired: bool,
    ) -> None:
        prompt = get_growth_analysis_prompt("growth-analysis-v1")
        if (
            run.task_type != CoachTaskType.GROWTH_ASSESSMENT.value
            or run.context_sha256 != context.context_sha256
        ):
            violations.append(_violation(AssessmentViolationCode.CONTEXT_LINEAGE, "agent_run"))
        if run.prompt_version != prompt.version or run.prompt_sha256 != prompt.content_sha256:
            violations.append(_violation(AssessmentViolationCode.PROMPT_LINEAGE, "agent_run"))
        if (
            assessment.agent_run_id != run.run_id
            or run.status is not AgentRunStatus.SUCCEEDED
            or (not repaired and run.output_sha256 != assessment.content_sha256)
            or run.completed_at < run.started_at
            or (not repaired and assessment.created_at != run.completed_at)
            or run.repair_count != 0
        ):
            violations.append(_violation(AssessmentViolationCode.PROVIDER_LINEAGE, "agent_run"))
        if not _provider_label_is_safe(run.provider_name) or not _provider_label_is_safe(
            run.provider_model
        ):
            violations.append(_violation(AssessmentViolationCode.PROVIDER_LINEAGE, "agent_run"))

    @staticmethod
    def _validate_findings(
        assessment: CoachAssessmentV2,
        snapshot: GrowthEvidenceSnapshotV3,
        violations: list[ValidationViolation],
    ) -> None:
        expected_evidence = tuple(
            sorted(
                reference.evidence_ref_id
                for metric in snapshot.metric_assessments
                for reference in metric.evidence_refs
            )
        )
        if assessment.snapshot_evidence_ref_ids != expected_evidence:
            violations.append(
                _violation(AssessmentViolationCode.EVIDENCE_REFS, "snapshot_evidence_ref_ids")
            )
        expected_lineage = tuple(
            GrowthTrendLineageReferenceV1.model_validate_json(
                json.dumps(item, default=str, separators=(",", ":")),
                strict=True,
            )
            for item in growth_trend_lineage_payloads(snapshot)
        )
        if assessment.snapshot_trend_lineage != expected_lineage:
            violations.append(
                _violation(AssessmentViolationCode.TREND_LINEAGE, "snapshot_trend_lineage")
            )
        lineage_by_id = {item.lineage_id: item for item in expected_lineage}
        metrics = tuple(snapshot.metric_assessments)
        matched_identities: list[str] = []
        substantive_ids: set[UUID] = set()
        for index, finding in enumerate(assessment.findings):
            location = f"findings.{index}"
            matched = AssessmentValidator._match_metric(finding, metrics, lineage_by_id)
            if matched is not None:
                matched_identities.append(matched.metric_identity_sha256)
            AssessmentValidator._validate_finding(
                finding,
                matched,
                lineage_by_id,
                location,
                violations,
            )
            AssessmentValidator._validate_exact_finding(
                assessment=assessment,
                snapshot=snapshot,
                finding=finding,
                metric=matched,
                location=location,
                violations=violations,
            )
            metric_identity = None if matched is None else matched.metric_identity_sha256
            if finding.finding_id != _expected_finding_id(
                snapshot=snapshot,
                finding=finding,
                metric_identity_sha256=metric_identity,
            ):
                violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
            if finding.support_status is FindingSupportStatus.SUPPORTED:
                substantive_ids.add(finding.finding_id)
        if len(matched_identities) != len(set(matched_identities)):
            violations.append(_violation(AssessmentViolationCode.SELECTION_POLICY, "findings"))
        if len(assessment.findings) > 2:
            violations.append(_violation(AssessmentViolationCode.SELECTION_POLICY, "findings"))
        if assessment.primary_finding_id is None and assessment.secondary_finding_id is not None:
            violations.append(
                _violation(AssessmentViolationCode.SELECTION_POLICY, "secondary_finding_id")
            )
        for field_name, selected in (
            ("primary_finding_id", assessment.primary_finding_id),
            ("secondary_finding_id", assessment.secondary_finding_id),
        ):
            if selected is not None and selected not in substantive_ids:
                violations.append(_violation(AssessmentViolationCode.SELECTION_POLICY, field_name))
        selected_ids = {
            selected
            for selected in (assessment.primary_finding_id, assessment.secondary_finding_id)
            if selected is not None
        }
        if selected_ids != substantive_ids:
            violations.append(_violation(AssessmentViolationCode.SELECTION_POLICY, "selection"))

    @staticmethod
    def _match_metric(finding, metrics, lineage_by_id):
        if finding.finding_type in _TREND_TYPES:
            lineage = lineage_by_id.get(finding.trend_lineage_id)
            if lineage is None:
                return None
            return next(
                (
                    item
                    for item in metrics
                    if item.metric_identity_sha256 == lineage.metric_identity_sha256
                ),
                None,
            )
        evidence = set(finding.evidence_ref_ids)
        matches = [
            item
            for item in metrics
            if evidence
            and evidence <= {reference.evidence_ref_id for reference in item.evidence_refs}
        ]
        if len(matches) == 1:
            return matches[0]
        code_prefix = f"{finding.finding_type.value}:"
        if finding.finding_code.startswith(code_prefix):
            metric_code = finding.finding_code.removeprefix(code_prefix)
            by_code = [item for item in metrics if item.metric.metric_code.value == metric_code]
            if len(by_code) == 1:
                return by_code[0]
        return None

    @staticmethod
    def _validate_finding(
        finding: CoachingFindingV2,
        metric,
        lineage_by_id: dict[UUID, GrowthTrendLineageReferenceV1],
        location: str,
        violations: list[ValidationViolation],
    ) -> None:
        texts = (finding.summary, finding.explanation, *finding.limitations)
        if any(_NUMERIC_LITERAL.search(text.replace("V3", "")) for text in texts):
            violations.append(_violation(AssessmentViolationCode.NUMERIC_INTEGRITY, location))
        if any(_CAUSAL_CLAIM.search(text) or _SENSITIVE_TEXT.search(text) for text in texts):
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
        if finding.finding_type is FindingType.INSUFFICIENT_EVIDENCE:
            expected_support = (
                FindingSupportStatus.LIMITED
                if finding.evidence_ref_ids
                else FindingSupportStatus.UNSUPPORTED
            )
            expected_code = (
                "INSUFFICIENT_EVIDENCE:unknown metric"
                if metric is None
                else f"INSUFFICIENT_EVIDENCE:{metric.metric.metric_code.value}"
            )
            if (
                finding.support_status is not expected_support
                or finding.reliability is not MetricReliability.UNKNOWN
                or finding.trend_lineage_id is not None
                or not finding.limitations
            ):
                violations.append(_violation(AssessmentViolationCode.SUPPORT_STATUS, location))
            if finding.finding_code != expected_code or finding.affected_skill_codes != (
                _expected_skills(metric)
            ):
                violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
            if metric is not None:
                expected_refs = tuple(
                    sorted(reference.evidence_ref_id for reference in metric.evidence_refs)
                )
                if finding.evidence_ref_ids != expected_refs:
                    violations.append(_violation(AssessmentViolationCode.EVIDENCE_REFS, location))
            return
        if metric is None:
            violations.append(_violation(AssessmentViolationCode.EVIDENCE_REFS, location))
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
            return
        if (
            finding.finding_code
            != f"{finding.finding_type.value}:{metric.metric.metric_code.value}"
        ):
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
        if finding.affected_skill_codes != _expected_skills(metric):
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
        if finding.support_status is not FindingSupportStatus.SUPPORTED:
            violations.append(_violation(AssessmentViolationCode.SUPPORT_STATUS, location))
        if finding.finding_type in _TREND_TYPES:
            lineage = lineage_by_id.get(finding.trend_lineage_id)
            expected_refs = () if lineage is None else tuple(sorted(lineage.evidence_ref_ids))
            if (
                lineage is None
                or lineage.trend_direction is not _EXPECTED_DIRECTIONS[finding.finding_type]
                or finding.evidence_ref_ids != expected_refs
            ):
                violations.append(_violation(AssessmentViolationCode.TREND_LINEAGE, location))
            if finding.evidence_ref_ids != expected_refs:
                violations.append(_violation(AssessmentViolationCode.EVIDENCE_REFS, location))
            if lineage is None or finding.reliability is not lineage.reliability:
                violations.append(_violation(AssessmentViolationCode.RELIABILITY, location))
        else:
            expected_refs = tuple(
                sorted(reference.evidence_ref_id for reference in metric.evidence_refs)
            )
            if finding.evidence_ref_ids != expected_refs:
                violations.append(_violation(AssessmentViolationCode.EVIDENCE_REFS, location))
            if (
                metric.metric.support_status is not MetricSupportStatusV2.SUPPORTED
                or metric.metric.reliability is MetricReliability.UNKNOWN
            ):
                violations.append(_violation(AssessmentViolationCode.SUPPORT_STATUS, location))
            if finding.reliability is not metric.metric.reliability:
                violations.append(_violation(AssessmentViolationCode.RELIABILITY, location))

    @staticmethod
    def _validate_exact_finding(
        *,
        assessment: CoachAssessmentV2,
        snapshot: GrowthEvidenceSnapshotV3,
        finding: CoachingFindingV2,
        metric: object | None,
        location: str,
        violations: list[ValidationViolation],
    ) -> None:
        if metric is None:
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
            return
        try:
            selection = GrowthAnalysisCandidate(
                metric_identity_sha256=metric.metric_identity_sha256,
                finding_type=finding.finding_type,
            )
            expected_assessment = assemble_growth_assessment_draft(
                snapshot=snapshot,
                provider_output=GrowthAnalysisProviderOutput(
                    primary=selection,
                    secondary=None,
                ),
                assessment_id=assessment.assessment_id,
                agent_run_id=assessment.agent_run_id,
                created_at=assessment.created_at,
            )
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))
            return
        if len(expected_assessment.findings) != 1 or finding != expected_assessment.findings[0]:
            violations.append(_violation(AssessmentViolationCode.FINDING_POLICY, location))


class SchemaValidator:
    """Standalone strict/deep schema gate used by the aggregate validator."""

    def validate(self, assessment: object) -> tuple[ValidationViolation, ...]:
        try:
            _stored_model(CoachAssessmentV2, assessment)
        except ValueError:
            return (_violation(AssessmentViolationCode.SCHEMA, "assessment"),)
        return ()


class _FindingComponentValidator:
    violation_codes: frozenset[AssessmentViolationCode]

    def validate(
        self,
        *,
        assessment: CoachAssessmentV2,
        snapshot: GrowthEvidenceSnapshotV3,
    ) -> tuple[ValidationViolation, ...]:
        violations: list[ValidationViolation] = []
        AssessmentValidator._validate_findings(assessment, snapshot, violations)
        return tuple(
            sorted(
                {item for item in violations if item.code in self.violation_codes},
                key=lambda item: item.sort_key(),
            )
        )


class EvidenceValidator(_FindingComponentValidator):
    violation_codes = frozenset(
        {AssessmentViolationCode.EVIDENCE_REFS, AssessmentViolationCode.TREND_LINEAGE}
    )


class NumericIntegrityValidator(_FindingComponentValidator):
    violation_codes = frozenset({AssessmentViolationCode.NUMERIC_INTEGRITY})


class SupportStatusValidator(_FindingComponentValidator):
    violation_codes = frozenset({AssessmentViolationCode.SUPPORT_STATUS})


class ReliabilityValidator(_FindingComponentValidator):
    violation_codes = frozenset({AssessmentViolationCode.RELIABILITY})


class FindingPolicyValidator(_FindingComponentValidator):
    violation_codes = frozenset(
        {
            AssessmentViolationCode.FINDING_POLICY,
            AssessmentViolationCode.SELECTION_POLICY,
        }
    )


__all__ = [
    "ASSESSMENT_VALIDATION_POLICY_VERSION",
    "AssessmentValidationReport",
    "AssessmentValidationStatus",
    "AssessmentValidator",
    "AssessmentViolationCode",
    "EvidenceValidator",
    "FindingPolicyValidator",
    "NumericIntegrityValidator",
    "ReliabilityValidator",
    "SchemaValidator",
    "SupportStatusValidator",
    "ValidationViolation",
]
