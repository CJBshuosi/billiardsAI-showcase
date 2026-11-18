"""Evidence-bound, provider-neutral training plan draft workflow for Story 9.6."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import (
    UUID4,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.coach_agent.context import (
    CoachAgentContext,
    CoachSourceReadFenceV3,
    CoachTaskType,
    _stored_values,
)
from billiards_ai.cloud.coach_agent.prompts import get_training_planning_prompt
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelProvider,
    CoachModelProviderAuditPort,
    CoachModelRequest,
    CoachModelResponse,
    revalidate_coach_model_result,
)
from billiards_ai.cloud.training.catalog import DrillCatalog, DrillDefinition, DrillReviewStatus
from billiards_ai.cloud.training.execution_policy import ExecutionPolicyCatalogV1
from billiards_ai.cloud.training.planning_policy import TrainingPlanningPolicyV1
from billiards_ai.shared.models.coaching import (
    AgentRun,
    AgentRunStatus,
    CoachAssessmentStatus,
    CoachAssessmentV2,
    CoachingFindingV2,
    FindingSupportStatus,
    FindingType,
)
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
)
from billiards_ai.shared.models.growth import GrowthEvidenceSnapshotV3
from billiards_ai.shared.models.growth_metrics import MetricReliability
from billiards_ai.shared.models.training_plan import (
    DrillReference,
    TrainingBlock,
    TrainingPlan,
)
from billiards_ai.shared.plan_publication.contracts import TrainingBlockV3

_INVALID_CONTEXT_SHA256 = canonical_sha256({"failure_identity": "training-plan-v1:INVALID_CONTEXT"})
_MAX_AUDIT_LABEL_LENGTH = 128


def _require_exact_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError("datetime must be an exact normalized UTC value")
    return value


class TrainingPlanningBlockCandidate(CoreContractModel):
    drill_id: NonEmptyStr
    drill_revision: StrictInt = Field(ge=1)
    target_finding_id: UUID4
    target_repetitions: StrictInt = Field(gt=0)
    difficulty_level: StrictInt = Field(ge=1)


class TrainingPlanningProviderOutput(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    primary_goal_finding_id: UUID4
    secondary_goal_finding_ids: tuple[UUID4, ...] = Field(max_length=1)
    blocks: tuple[TrainingPlanningBlockCandidate, ...] = Field(min_length=1, max_length=3)
    fallback_drill_id: NonEmptyStr
    fallback_drill_revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def selection_is_canonical(self) -> Self:
        if self.primary_goal_finding_id in self.secondary_goal_finding_ids:
            raise ValueError("primary and secondary goals must differ")
        if self.secondary_goal_finding_ids != tuple(sorted(set(self.secondary_goal_finding_ids))):
            raise ValueError("secondary goals must be sorted and unique")
        keys = tuple(
            (item.drill_id, item.drill_revision, item.target_finding_id) for item in self.blocks
        )
        if len(keys) != len(set(keys)):
            raise ValueError("planning blocks must be unique")
        return self


class TrainingPlanDraftStatus(StrEnum):
    DRAFT = "DRAFT"


class TrainingPlanGoal(CoreContractModel):
    primary_finding_id: UUID4
    secondary_finding_ids: tuple[UUID4, ...] = Field(max_length=1)

    @model_validator(mode="after")
    def goals_are_unique(self) -> Self:
        if (
            self.primary_finding_id in self.secondary_finding_ids
            or self.secondary_finding_ids != tuple(sorted(set(self.secondary_finding_ids)))
        ):
            raise ValueError("training plan goals must be unique and canonical")
        return self


class TrainingPlanGoalAuthority(CoreContractModel):
    finding_id: UUID4
    finding_type: FindingType
    affected_skill_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    support_status: Literal[FindingSupportStatus.SUPPORTED]
    reliability: MetricReliability
    evidence_ref_ids: tuple[UUID4, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def authority_is_substantive_and_canonical(self) -> Self:
        if (
            self.finding_type is FindingType.INSUFFICIENT_EVIDENCE
            or self.reliability is MetricReliability.UNKNOWN
            or self.affected_skill_codes != tuple(sorted(set(self.affected_skill_codes)))
            or self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids)))
        ):
            raise ValueError("goal authority must be substantive and canonical")
        return self


class TrainingPlanBlockAudit(CoreContractModel):
    block_id: UUID4
    order: StrictInt = Field(ge=1)
    drill_id: NonEmptyStr
    drill_revision: StrictInt = Field(ge=1)
    drill_content_sha256: Sha256
    target_finding_id: UUID4
    target_skill_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    difficulty_level: StrictInt = Field(ge=1)
    target_repetitions: StrictInt = Field(gt=0)
    progression_reason_code: NonEmptyStr
    progression_threshold: StrictInt = Field(gt=0)
    progression_level_delta: StrictInt = Field(gt=0)
    regression_reason_code: NonEmptyStr
    regression_threshold: StrictInt = Field(gt=0)
    regression_level_delta: StrictInt = Field(lt=0)
    evidence_ref_ids: tuple[UUID4, ...] = Field(min_length=1)
    rationale_code: Literal["TARGETS_APPROVED_FINDING"] = "TARGETS_APPROVED_FINDING"

    @model_validator(mode="after")
    def canonical_sets(self) -> Self:
        if self.target_skill_codes != tuple(sorted(set(self.target_skill_codes))):
            raise ValueError("target skill codes must be sorted and unique")
        if self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids))):
            raise ValueError("evidence refs must be sorted and unique")
        return self


class TrainingPlanCandidateV3(CoreContractModel):
    """Unsigned, unadmitted V3-ready authority candidate; never executable by itself."""

    schema_version: Literal["3.0"] = "3.0"
    status: Literal[TrainingPlanDraftStatus.DRAFT] = TrainingPlanDraftStatus.DRAFT
    plan_id: UUID4
    user_id: UUID4
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    version: StrictInt = Field(ge=1)
    created_at: datetime
    valid_until: datetime
    blocks: tuple[TrainingBlockV3, ...] = Field(min_length=1)
    fallback_drill: DrillReference
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at", mode="before")(_require_exact_utc)
    _valid_until_utc = field_validator("valid_until", mode="before")(_require_exact_utc)

    @model_validator(mode="after")
    def candidate_is_ordered_unsigned_and_hash_bound(self) -> Self:
        if self.valid_until <= self.created_at:
            raise ValueError("candidate validity must follow creation")
        if tuple(item.order for item in self.blocks) != tuple(range(1, len(self.blocks) + 1)):
            raise ValueError("candidate blocks must be contiguous from one")
        block_ids = tuple(item.block_id for item in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("candidate block IDs must be unique")
        if any(item.fallback_drill != self.fallback_drill for item in self.blocks):
            raise ValueError("candidate blocks must bind the exact plan fallback")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected:
            raise ValueError("V3-ready candidate content hash mismatch")
        return self


class TrainingPlanDraft(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal[TrainingPlanDraftStatus.DRAFT] = TrainingPlanDraftStatus.DRAFT
    v3_candidate: TrainingPlanCandidateV3
    execution_policy_catalog: ExecutionPolicyCatalogV1
    plan: TrainingPlan
    context_id: UUID4
    context_sha256: Sha256
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    assessment_content_sha256: Sha256
    catalog_version: NonEmptyStr
    drill_catalog_content_sha256: Sha256
    catalog_heads_sha256: Sha256
    prompt_version: Literal["training-plan-v1"]
    prompt_sha256: Sha256
    planning_policy_version: Literal["training-planning-policy-v1"]
    goals: TrainingPlanGoal
    goal_authorities: tuple[TrainingPlanGoalAuthority, ...] = Field(min_length=1)
    blocks: tuple[TrainingPlanBlockAudit, ...] = Field(min_length=1)
    drill_definitions: tuple[DrillDefinition, ...] = Field(min_length=1)
    fallback_drill_content_sha256: Sha256
    fallback_rationale_code: Literal["SAFE_APPROVED_CATALOG_FALLBACK"]
    estimated_duration_seconds: StrictInt = Field(gt=0)
    agent_run_id: UUID4
    created_at: datetime
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at", mode="before")(_require_exact_utc)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_values(copied), strict=True)

    @model_validator(mode="after")
    def wrapper_is_closed_and_hash_bound(self) -> Self:
        plan_payload = self.plan.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(plan_payload) != self.plan.content_sha256:
            raise ValueError("embedded training plan hash mismatch")
        if (
            self.plan.plan_id.version != 4
            or self.plan.user_id is None
            or self.plan.assessment_id != self.assessment_id
            or self.plan.assessment_revision != self.assessment_revision
            or self.plan.created_at != self.created_at
            or len(self.plan.blocks) != len(self.blocks)
        ):
            raise ValueError("draft and embedded plan lineage mismatch")
        candidate = self.v3_candidate
        if (
            candidate.status is not TrainingPlanDraftStatus.DRAFT
            or candidate.plan_id != self.plan.plan_id
            or candidate.user_id != self.plan.user_id
            or candidate.assessment_id != self.assessment_id
            or candidate.assessment_revision != self.assessment_revision
            or candidate.snapshot_id != self.snapshot_id
            or candidate.snapshot_version != self.snapshot_version
            or candidate.version != self.plan.version
            or candidate.created_at != self.created_at
            or candidate.valid_until != self.plan.valid_until
            or candidate.fallback_drill != self.plan.fallback_drill
            or len(candidate.blocks) != len(self.blocks)
        ):
            raise ValueError("V3-ready candidate and audit plan lineage mismatch")
        if (
            self.execution_policy_catalog.catalog_version != self.catalog_version
            or self.execution_policy_catalog.drill_catalog_content_sha256
            != self.drill_catalog_content_sha256
        ):
            raise ValueError("execution policy authority does not bind the draft catalog")
        policies = {
            (item.drill_id, item.drill_revision): item
            for item in self.execution_policy_catalog.definitions
        }
        definitions = {(item.drill_id, item.revision): item for item in self.drill_definitions}
        reference_keys = {
            (item.drill.drill_id, item.drill.drill_revision) for item in self.plan.blocks
        } | {
            (
                self.plan.fallback_drill.drill_id,
                self.plan.fallback_drill.drill_revision,
            )
        }
        if set(definitions) != reference_keys or set(policies) != reference_keys:
            raise ValueError("draft authorities must exactly cover executable references")
        if any(
            definition.review_status is not DrillReviewStatus.APPROVED
            or definition.content_sha256 != policies[key].drill_content_sha256
            or not policies[key].matches_drill_definition(definition)
            for key, definition in definitions.items()
        ):
            raise ValueError("drill and execution policy authorities do not match")
        authorities = {item.finding_id: item for item in self.goal_authorities}
        for plan_block, audit in zip(self.plan.blocks, self.blocks, strict=True):
            v3_block = candidate.blocks[audit.order - 1]
            execution = policies.get((audit.drill_id, audit.drill_revision))
            definition = definitions.get((audit.drill_id, audit.drill_revision))
            goal_authority = authorities.get(audit.target_finding_id)
            expected_skills = (
                ()
                if definition is None or goal_authority is None
                else tuple(
                    sorted(
                        set(definition.skill_codes).intersection(
                            goal_authority.affected_skill_codes
                        )
                    )
                )
            )
            if (
                plan_block.block_id != audit.block_id
                or plan_block.order != audit.order
                or plan_block.drill.drill_id != audit.drill_id
                or plan_block.drill.drill_revision != audit.drill_revision
                or plan_block.target_finding_id != audit.target_finding_id
                or plan_block.target_repetitions != audit.target_repetitions
                or plan_block.progress_after_successes != audit.progression_threshold
                or plan_block.regress_after_failures != audit.regression_threshold
                or execution is None
                or definition is None
                or goal_authority is None
                or audit.drill_content_sha256 != definition.content_sha256
                or audit.target_skill_codes != expected_skills
                or audit.evidence_ref_ids != goal_authority.evidence_ref_ids
                or v3_block.block_id != audit.block_id
                or v3_block.drill != plan_block.drill
                or v3_block.target_finding_id != audit.target_finding_id
                or v3_block.target_repetitions != audit.target_repetitions
                or v3_block.initial_difficulty != audit.difficulty_level
                or v3_block.progression_rule != execution.progression_rule
                or v3_block.regression_rule != execution.regression_rule
                or v3_block.abstention_rule != execution.abstention_rule
                or v3_block.success_policy != execution.success_policy
                or v3_block.replan_rule != execution.replan_rule
                or v3_block.fallback_drill != execution.fallback_drill
            ):
                raise ValueError("block audit does not match embedded plan")
        goal_ids = {self.goals.primary_finding_id, *self.goals.secondary_finding_ids}
        target_ids = {item.target_finding_id for item in self.blocks}
        if target_ids != goal_ids or set(authorities) != goal_ids:
            raise ValueError("every declared goal must have an authoritative block")
        references = tuple(item.drill for item in self.plan.blocks) + (self.plan.fallback_drill,)
        if any(item.catalog_version != self.catalog_version for item in references):
            raise ValueError("every drill reference must bind the draft catalog version")
        if (
            self.estimated_duration_seconds
            != sum(item.target_repetitions for item in self.blocks) * 45
            or (self.plan.valid_until - self.created_at).total_seconds() != 604800
        ):
            raise ValueError("draft workload or validity does not match planning policy v1")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected_hash:
            raise ValueError("training plan draft content hash mismatch")
        return self


class TrainingPlanningCommand(CoreContractModel):
    context: CoachAgentContext
    assessment: CoachAssessmentV2
    drill_catalog: DrillCatalog
    execution_policy_catalog: ExecutionPolicyCatalogV1
    started_at: datetime
    completed_at: datetime
    timeout_ms: StrictInt = Field(ge=1, le=600_000)

    _started_at_utc = field_validator("started_at", mode="before")(_require_exact_utc)
    _completed_at_utc = field_validator("completed_at", mode="before")(_require_exact_utc)

    @model_validator(mode="after")
    def run_time_is_monotonic(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("planning completion cannot precede start")
        return self


class TrainingPlanningResultStatus(StrEnum):
    COMPLETED = "COMPLETED"
    NO_RELIABLE_PLAN = "NO_RELIABLE_PLAN"


class TrainingPlanningFailureCode(StrEnum):
    INVALID_CONTEXT = "INVALID_CONTEXT"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    INVALID_SELECTION = "INVALID_SELECTION"
    WORKLOAD_EXCEEDED = "WORKLOAD_EXCEEDED"
    INPUT_LIMIT_EXCEEDED = "INPUT_LIMIT_EXCEEDED"


class TrainingPlanningCompleted(CoreContractModel):
    status: Literal[TrainingPlanningResultStatus.COMPLETED] = TrainingPlanningResultStatus.COMPLETED
    draft: TrainingPlanDraft
    agent_run: AgentRun

    @model_validator(mode="after")
    def result_is_audit_closed(self) -> Self:
        if (
            self.agent_run.status is not AgentRunStatus.SUCCEEDED
            or self.agent_run.output_sha256 != self.draft.content_sha256
            or self.agent_run.run_id != self.draft.agent_run_id
            or self.agent_run.completed_at != self.draft.created_at
        ):
            raise ValueError("completed planning result is not audit closed")
        return self


class TrainingPlanningFailure(CoreContractModel):
    status: Literal[TrainingPlanningResultStatus.NO_RELIABLE_PLAN] = (
        TrainingPlanningResultStatus.NO_RELIABLE_PLAN
    )
    code: TrainingPlanningFailureCode
    agent_run: AgentRun

    @model_validator(mode="after")
    def result_is_failed_and_payload_free(self) -> Self:
        if (
            self.agent_run.status is not AgentRunStatus.FAILED
            or self.agent_run.output_sha256 is not None
        ):
            raise ValueError("failed planning result is not audit closed")
        return self


type TrainingPlanningResult = TrainingPlanningCompleted | TrainingPlanningFailure


class _PlanningError(ValueError):
    def __init__(self, code: TrainingPlanningFailureCode):
        self.code = code
        super().__init__(code.value)


def _uuid4_from_payload(value: object) -> UUID:
    raw = bytearray(bytes.fromhex(canonical_sha256(value))[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _audit_label(value: object, fallback: str) -> str:
    if (
        type(value) is str
        and value == value.strip()
        and 0 < len(value) <= _MAX_AUDIT_LABEL_LENGTH
        and not any(character.isspace() and character != " " for character in value)
    ):
        return value
    return fallback


def _strict_command(value: object) -> TrainingPlanningCommand:
    try:
        if not isinstance(value, TrainingPlanningCommand):
            raise ValueError
        return TrainingPlanningCommand.model_validate(_stored_values(value), strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT) from None


def _exact_assessment_and_context(
    command: TrainingPlanningCommand,
) -> tuple[CoachAgentContext, CoachAssessmentV2, dict[UUID, CoachingFindingV2]]:
    try:
        context = CoachAgentContext.model_validate(command.context, strict=True)
        assessment = CoachAssessmentV2.model_validate(
            _stored_values(command.assessment), strict=True
        )
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT) from None
    if (
        context.task_type is not CoachTaskType.TRAINING_PLAN
        or not isinstance(context.growth_snapshot, GrowthEvidenceSnapshotV3)
        or not isinstance(context.read_fence, CoachSourceReadFenceV3)
        or assessment.status is not CoachAssessmentStatus.APPROVED
        or assessment.user_id != context.user_id
        or assessment.snapshot_id != context.snapshot_id
        or assessment.snapshot_version != context.snapshot_version
        or assessment.snapshot_content_sha256 != context.growth_snapshot.content_sha256
        or assessment.created_at > context.as_of
        or context.as_of > command.started_at
        or context.catalog_version is None
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    if not context.prior_assessments:
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    current = context.prior_assessments[-1]
    if (
        current.assessment_id != assessment.assessment_id
        or current.revision != assessment.revision
        or current.content_sha256 != assessment.content_sha256
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    findings = {item.finding_id: item for item in assessment.findings}
    if len(findings) != len(assessment.findings):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    return context, assessment, findings


def _exact_catalog_authorities(
    command: TrainingPlanningCommand,
    context: CoachAgentContext,
) -> tuple[
    DrillCatalog,
    ExecutionPolicyCatalogV1,
    dict[tuple[str, int], DrillDefinition],
    dict[tuple[str, int], object],
]:
    try:
        catalog = DrillCatalog.model_validate(_stored_values(command.drill_catalog), strict=True)
        execution_catalog = ExecutionPolicyCatalogV1.model_validate(
            _stored_values(command.execution_policy_catalog), strict=True
        )
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT) from None
    if (
        context.catalog_version is None
        or catalog.catalog_version != context.catalog_version
        or execution_catalog.catalog_version != catalog.catalog_version
        or execution_catalog.drill_catalog_content_sha256 != catalog.content_sha256
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    heads = {(item.drill_id, item.revision): item for item in catalog.drills}
    context_heads = {(item.drill_id, item.revision): item for item in context.available_drills}
    if any(heads.get(key) != value for key, value in context_heads.items()):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    policies = {
        (item.drill_id, item.drill_revision): item for item in execution_catalog.definitions
    }
    if set(policies) != set(heads):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    if any(not policies[key].matches_drill_definition(drill) for key, drill in heads.items()):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_CONTEXT)
    return catalog, execution_catalog, heads, policies


def _is_substantive(finding: CoachingFindingV2) -> bool:
    return (
        finding.finding_type is not FindingType.INSUFFICIENT_EVIDENCE
        and finding.support_status is FindingSupportStatus.SUPPORTED
        and finding.reliability is not MetricReliability.UNKNOWN
        and bool(finding.evidence_ref_ids)
    )


def _open_plan_count(context: CoachAgentContext) -> int:
    planned = {(item.plan_id, item.plan_version) for item in context.prior_plans}
    completed = {(item.plan_id, item.plan_version) for item in context.plan_outcomes}
    return len(planned - completed)


def _provider_payload(
    *,
    context: CoachAgentContext,
    assessment: CoachAssessmentV2,
    execution_policies: dict[tuple[str, int], object],
    policy: TrainingPlanningPolicyV1,
) -> dict[str, object]:
    selected_ids = {
        item
        for item in (assessment.primary_finding_id, assessment.secondary_finding_id)
        if item is not None
    }
    findings = [
        {
            "finding_id": str(item.finding_id),
            "finding_type": item.finding_type.value,
            "affected_skill_codes": list(item.affected_skill_codes),
            "support_status": item.support_status.value,
            "reliability": item.reliability.value,
            "is_selected": item.finding_id in selected_ids,
        }
        for item in assessment.findings
    ]
    drills: list[dict[str, object]] = []
    for item in context.available_drills:
        execution = execution_policies[(item.drill_id, item.revision)]
        drills.append(
            {
                "drill_id": item.drill_id,
                "drill_revision": item.revision,
                "skill_codes": list(item.skill_codes),
                "minimum_user_level": item.minimum_level,
                "maximum_user_level": item.maximum_level,
                "initial_runtime_difficulty": execution.initial_difficulty,
                "minimum_runtime_difficulty": execution.minimum_runtime_difficulty,
                "maximum_runtime_difficulty": execution.maximum_runtime_difficulty,
                "success_metrics": [
                    {
                        "metric_code": metric.metric_code,
                        "target_attempts": metric.target_attempts,
                        "minimum_successes": metric.minimum_successes,
                    }
                    for metric in execution.success_policy.metric_rules
                ],
                "progression_threshold": execution.progression_rule.consecutive_count,
                "progression_level_delta": execution.progression_rule.difficulty_delta,
                "regression_threshold": execution.regression_rule.consecutive_count,
                "regression_level_delta": execution.regression_rule.difficulty_delta,
            }
        )
    open_plan_count = _open_plan_count(context)
    allowed_blocks = policy.allowed_blocks_for_open_plans(open_plan_count=open_plan_count)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "assessment": {
            "assessment_id": str(assessment.assessment_id),
            "assessment_revision": assessment.revision,
            "findings": findings,
        },
        "eligible_drills": drills,
        "history_summary": {
            "prior_plan_count": len(context.prior_plans),
            "completed_outcome_count": len(context.plan_outcomes),
            "open_plan_count": open_plan_count,
            "prior_plan_hashes": [item.content_sha256 for item in context.prior_plans],
            "outcome_hashes": [item.content_sha256 for item in context.plan_outcomes],
        },
        "planning_limits": {
            "maximum_blocks": allowed_blocks,
            "maximum_total_repetitions": policy.maximum_total_repetitions,
            "maximum_estimated_duration_seconds": policy.maximum_estimated_duration_seconds,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) > policy.maximum_provider_payload_bytes:
        raise _PlanningError(TrainingPlanningFailureCode.INPUT_LIMIT_EXCEEDED)
    return payload


def _bounded_provider_output(value: object, policy: TrainingPlanningPolicyV1) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_OUTPUT)
    stack: list[tuple[object, int]] = [(value, 1)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > policy.maximum_provider_output_nodes:
            raise _PlanningError(TrainingPlanningFailureCode.INPUT_LIMIT_EXCEEDED)
        if depth > policy.maximum_provider_output_depth:
            raise _PlanningError(TrainingPlanningFailureCode.INPUT_LIMIT_EXCEEDED)
        if isinstance(current, (dict, list, tuple)):
            identity = id(current)
            if identity in seen:
                raise _PlanningError(TrainingPlanningFailureCode.INVALID_OUTPUT)
            seen.add(identity)
            children = current.values() if isinstance(current, dict) else current
            stack.extend((child, depth + 1) for child in children)
        elif current is not None and type(current) not in {str, int, float, bool}:
            raise _PlanningError(TrainingPlanningFailureCode.INVALID_OUTPUT)
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (RecursionError, TypeError, ValueError):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_OUTPUT) from None
    if len(encoded.encode()) > policy.maximum_provider_output_bytes:
        raise _PlanningError(TrainingPlanningFailureCode.INPUT_LIMIT_EXCEEDED)
    return value


def _parse_provider_output(
    value: object, policy: TrainingPlanningPolicyV1
) -> TrainingPlanningProviderOutput:
    try:
        bounded = _bounded_provider_output(value, policy)
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
        return TrainingPlanningProviderOutput.model_validate_json(encoded, strict=True)
    except _PlanningError:
        raise
    except (RecursionError, TypeError, ValidationError, ValueError):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_OUTPUT) from None


def _drill_reference(catalog_version: str, drill: DrillDefinition) -> DrillReference:
    return DrillReference(
        catalog_version=catalog_version,
        drill_id=drill.drill_id,
        drill_revision=drill.revision,
        required_capabilities=drill.required_capabilities,
    )


def _assemble_draft(
    *,
    context: CoachAgentContext,
    assessment: CoachAssessmentV2,
    findings: dict[UUID, CoachingFindingV2],
    drill_catalog: DrillCatalog,
    execution_catalog: ExecutionPolicyCatalogV1,
    catalog_heads: dict[tuple[str, int], DrillDefinition],
    execution_policies: dict[tuple[str, int], object],
    output: TrainingPlanningProviderOutput,
    policy: TrainingPlanningPolicyV1,
    prompt_version: str,
    prompt_sha256: str,
    agent_run_id: UUID,
    created_at: datetime,
) -> TrainingPlanDraft:
    if assessment.primary_finding_id is None or (
        output.primary_goal_finding_id != assessment.primary_finding_id
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
    expected_secondary = (
        () if assessment.secondary_finding_id is None else (assessment.secondary_finding_id,)
    )
    if output.secondary_goal_finding_ids != expected_secondary:
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
    selected_goal_ids = {output.primary_goal_finding_id, *output.secondary_goal_finding_ids}
    if any(
        item not in findings or not _is_substantive(findings[item]) for item in selected_goal_ids
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
    if {item.target_finding_id for item in output.blocks} != selected_goal_ids:
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
    allowed_blocks = policy.allowed_blocks_for_open_plans(open_plan_count=_open_plan_count(context))
    if len(output.blocks) > allowed_blocks:
        raise _PlanningError(TrainingPlanningFailureCode.WORKLOAD_EXCEEDED)
    drill_heads = {(item.drill_id, item.revision): item for item in context.available_drills}
    fallback = drill_heads.get((output.fallback_drill_id, output.fallback_drill_revision))
    fallback_key = (output.fallback_drill_id, output.fallback_drill_revision)
    fallback_execution = execution_policies.get(fallback_key)
    if (
        fallback is None
        or fallback.review_status is not DrillReviewStatus.APPROVED
        or catalog_heads.get(fallback_key) != fallback
        or fallback_execution is None
        or fallback_execution.fallback_drill is not None
    ):
        raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
    total_repetitions = sum(item.target_repetitions for item in output.blocks)
    estimated_seconds = total_repetitions * policy.seconds_per_repetition
    if (
        total_repetitions > policy.maximum_total_repetitions
        or estimated_seconds > policy.maximum_estimated_duration_seconds
    ):
        raise _PlanningError(TrainingPlanningFailureCode.WORKLOAD_EXCEEDED)
    plan_version = max((item.plan_version for item in context.prior_plans), default=0) + 1
    identity = {
        "context_sha256": context.context_sha256,
        "assessment_content_sha256": assessment.content_sha256,
        "catalog_version": context.catalog_version,
        "drill_catalog_content_sha256": drill_catalog.content_sha256,
        "execution_policy_catalog_sha256": execution_catalog.content_sha256,
        "provider_output": output.model_dump(mode="json"),
        "policy_version": policy.policy_version,
        "prompt_version": prompt_version,
        "created_at": created_at,
        "plan_version": plan_version,
    }
    plan_id = _uuid4_from_payload(identity | {"entity": "plan"})
    plan_blocks: list[TrainingBlock] = []
    audits: list[TrainingPlanBlockAudit] = []
    v3_blocks: list[TrainingBlockV3] = []
    selected_execution_policies: dict[tuple[str, int], object] = {fallback_key: fallback_execution}
    fallback_reference = _drill_reference(context.catalog_version, fallback)
    for order, candidate in enumerate(output.blocks, start=1):
        drill_key = (candidate.drill_id, candidate.drill_revision)
        drill = drill_heads.get(drill_key)
        execution = execution_policies.get(drill_key)
        finding = findings.get(candidate.target_finding_id)
        attempt_horizon = (
            None
            if execution is None
            else max(item.target_attempts for item in execution.success_policy.metric_rules)
        )
        target_skills = (
            ()
            if drill is None or finding is None
            else tuple(sorted(set(drill.skill_codes).intersection(finding.affected_skill_codes)))
        )
        if (
            drill is None
            or drill.review_status is not DrillReviewStatus.APPROVED
            or catalog_heads.get(drill_key) != drill
            or execution is None
            or not execution.matches_drill_definition(drill)
            or execution.fallback_drill != fallback_reference
            or finding is None
            or candidate.target_finding_id not in selected_goal_ids
            or not _is_substantive(finding)
            or not target_skills
            or candidate.difficulty_level != execution.initial_difficulty
            or candidate.target_repetitions != attempt_horizon
        ):
            raise _PlanningError(TrainingPlanningFailureCode.INVALID_SELECTION)
        block_id = _uuid4_from_payload(identity | {"entity": "block", "order": order})
        plan_blocks.append(
            TrainingBlock(
                block_id=block_id,
                order=order,
                drill=_drill_reference(context.catalog_version, drill),
                target_finding_id=finding.finding_id,
                target_repetitions=candidate.target_repetitions,
                progress_after_successes=execution.progression_rule.consecutive_count,
                regress_after_failures=execution.regression_rule.consecutive_count,
            )
        )
        v3_blocks.append(
            TrainingBlockV3(
                block_id=block_id,
                order=order,
                drill=_drill_reference(context.catalog_version, drill),
                target_finding_id=finding.finding_id,
                target_repetitions=candidate.target_repetitions,
                initial_difficulty=execution.initial_difficulty,
                minimum_runtime_difficulty=execution.minimum_runtime_difficulty,
                maximum_runtime_difficulty=execution.maximum_runtime_difficulty,
                progression_rule=execution.progression_rule,
                regression_rule=execution.regression_rule,
                abstention_rule=execution.abstention_rule,
                success_policy=execution.success_policy,
                replan_rule=execution.replan_rule,
                fallback_drill=fallback_reference,
            )
        )
        selected_execution_policies[drill_key] = execution
        audits.append(
            TrainingPlanBlockAudit(
                block_id=block_id,
                order=order,
                drill_id=drill.drill_id,
                drill_revision=drill.revision,
                drill_content_sha256=drill.content_sha256,
                target_finding_id=finding.finding_id,
                target_skill_codes=target_skills,
                difficulty_level=execution.initial_difficulty,
                target_repetitions=candidate.target_repetitions,
                progression_reason_code=drill.progression_rule.reason_code,
                progression_threshold=execution.progression_rule.consecutive_count,
                progression_level_delta=execution.progression_rule.difficulty_delta,
                regression_reason_code=drill.regression_rule.reason_code,
                regression_threshold=execution.regression_rule.consecutive_count,
                regression_level_delta=execution.regression_rule.difficulty_delta,
                evidence_ref_ids=finding.evidence_ref_ids,
            )
        )
    plan_payload = {
        "schema_version": "1.0",
        "plan_id": plan_id,
        "user_id": context.user_id,
        "assessment_id": assessment.assessment_id,
        "assessment_revision": assessment.revision,
        "version": plan_version,
        "created_at": created_at,
        "valid_until": created_at + timedelta(seconds=policy.valid_for_seconds),
        "blocks": tuple(plan_blocks),
        "fallback_drill": fallback_reference,
    }
    plan = TrainingPlan(
        **plan_payload,
        content_sha256=canonical_sha256(
            TrainingPlan.model_construct(**plan_payload, content_sha256="0" * 64).model_dump(
                mode="json", exclude={"content_sha256"}
            )
        ),
    )
    v3_payload = {
        "schema_version": "3.0",
        "status": TrainingPlanDraftStatus.DRAFT,
        "plan_id": plan.plan_id,
        "user_id": plan.user_id,
        "assessment_id": assessment.assessment_id,
        "assessment_revision": assessment.revision,
        "snapshot_id": context.snapshot_id,
        "snapshot_version": context.snapshot_version,
        "version": plan.version,
        "created_at": created_at,
        "valid_until": plan.valid_until,
        "blocks": tuple(v3_blocks),
        "fallback_drill": fallback_reference,
    }
    v3_candidate = TrainingPlanCandidateV3.model_validate(
        v3_payload
        | {
            "content_sha256": canonical_sha256(
                TrainingPlanCandidateV3.model_construct(
                    **v3_payload, content_sha256="0" * 64
                ).model_dump(mode="json", exclude={"content_sha256"})
            )
        },
        strict=True,
    )
    projected_execution_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=execution_catalog.catalog_version,
        drill_catalog_content_sha256=drill_catalog.content_sha256,
        definitions=tuple(selected_execution_policies.values()),
    )
    selected_drill_definitions = tuple(
        catalog_heads[key] for key in sorted(selected_execution_policies)
    )
    goal_authorities = tuple(
        TrainingPlanGoalAuthority(
            finding_id=findings[finding_id].finding_id,
            finding_type=findings[finding_id].finding_type,
            affected_skill_codes=findings[finding_id].affected_skill_codes,
            support_status=FindingSupportStatus.SUPPORTED,
            reliability=findings[finding_id].reliability,
            evidence_ref_ids=findings[finding_id].evidence_ref_ids,
        )
        for finding_id in sorted(selected_goal_ids)
    )
    draft_payload = {
        "schema_version": "1.0",
        "status": TrainingPlanDraftStatus.DRAFT,
        "v3_candidate": v3_candidate,
        "execution_policy_catalog": projected_execution_catalog,
        "plan": plan,
        "context_id": context.context_id,
        "context_sha256": context.context_sha256,
        "snapshot_id": context.snapshot_id,
        "snapshot_version": context.snapshot_version,
        "assessment_id": assessment.assessment_id,
        "assessment_revision": assessment.revision,
        "assessment_content_sha256": assessment.content_sha256,
        "catalog_version": context.catalog_version,
        "drill_catalog_content_sha256": drill_catalog.content_sha256,
        "catalog_heads_sha256": canonical_sha256(
            tuple(item.content_sha256 for item in context.available_drills)
        ),
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "planning_policy_version": policy.policy_version,
        "goals": TrainingPlanGoal(
            primary_finding_id=output.primary_goal_finding_id,
            secondary_finding_ids=output.secondary_goal_finding_ids,
        ),
        "goal_authorities": goal_authorities,
        "blocks": tuple(audits),
        "drill_definitions": selected_drill_definitions,
        "fallback_drill_content_sha256": fallback.content_sha256,
        "fallback_rationale_code": "SAFE_APPROVED_CATALOG_FALLBACK",
        "estimated_duration_seconds": estimated_seconds,
        "agent_run_id": agent_run_id,
        "created_at": created_at,
    }
    encoded = TrainingPlanDraft.model_construct(
        **draft_payload, content_sha256="0" * 64
    ).model_dump(mode="json", exclude={"content_sha256"})
    return TrainingPlanDraft.model_validate(
        draft_payload | {"content_sha256": canonical_sha256(encoded)}, strict=True
    )


class TrainingPlanningWorkflow:
    def __init__(
        self,
        *,
        provider: CoachModelProvider,
        audit_authority: CoachModelProviderAuditPort,
        policy: TrainingPlanningPolicyV1 | None = None,
    ) -> None:
        if not isinstance(audit_authority, CoachModelProviderAuditPort):
            raise TypeError("provider audit authority must implement the explicit audit port")
        self._provider = provider
        try:
            self._policy = TrainingPlanningPolicyV1.model_validate(
                _stored_values(policy or TrainingPlanningPolicyV1()), strict=True
            )
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ValueError("invalid training planning policy") from None
        self._prompt = get_training_planning_prompt("training-plan-v1")
        self._audit_provider_name = _audit_label(
            audit_authority.audit_provider_name, "INVALID_PROVIDER"
        )
        self._audit_model_name = _audit_label(audit_authority.audit_model_name, "INVALID_MODEL")
        if (
            self._audit_provider_name == "INVALID_PROVIDER"
            or self._audit_model_name == "INVALID_MODEL"
        ):
            raise ValueError("provider audit authority returned an invalid identity")

    def _run_id(self, command: object) -> UUID:
        try:
            context_hash = command.context.context_sha256  # type: ignore[attr-defined]
            assessment_hash = command.assessment.content_sha256  # type: ignore[attr-defined]
            execution_policy_hash = command.execution_policy_catalog.content_sha256  # type: ignore[attr-defined]
            started_at = command.started_at  # type: ignore[attr-defined]
            if (
                type(context_hash) is not str
                or len(context_hash) != 64
                or type(assessment_hash) is not str
                or len(assessment_hash) != 64
                or type(execution_policy_hash) is not str
                or len(execution_policy_hash) != 64
                or type(started_at) is not datetime
                or started_at.tzinfo is not UTC
            ):
                raise ValueError
        except (AttributeError, RecursionError, TypeError, ValueError):
            context_hash = _INVALID_CONTEXT_SHA256
            assessment_hash = _INVALID_CONTEXT_SHA256
            execution_policy_hash = _INVALID_CONTEXT_SHA256
            started_at = datetime(1970, 1, 1, tzinfo=UTC)
        try:
            return _uuid4_from_payload(
                {
                    "task_type": "TRAINING_PLAN",
                    "context_sha256": context_hash,
                    "assessment_content_sha256": assessment_hash,
                    "execution_policy_catalog_sha256": execution_policy_hash,
                    "prompt_sha256": self._prompt.content_sha256,
                    "started_at": started_at,
                }
            )
        except (AttributeError, RecursionError, TypeError, ValueError):
            return _uuid4_from_payload(
                {
                    "task_type": "TRAINING_PLAN",
                    "failure_identity": _INVALID_CONTEXT_SHA256,
                }
            )

    def _times(self, command: object) -> tuple[datetime, datetime]:
        try:
            started = command.started_at  # type: ignore[attr-defined]
            completed = command.completed_at  # type: ignore[attr-defined]
            if (
                type(started) is datetime
                and started.tzinfo is UTC
                and type(completed) is datetime
                and completed.tzinfo is UTC
                and completed >= started
            ):
                return started, completed
        except (AttributeError, RecursionError, TypeError):
            pass
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return epoch, epoch

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
            task_type="TRAINING_PLAN",
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.content_sha256,
            context_sha256=context_sha256,
            provider_name=_audit_label(provider_name, "INVALID_PROVIDER"),
            provider_model=_audit_label(provider_model, "INVALID_MODEL"),
            output_sha256=output_sha256,
            status=status,
            repair_count=0,
            started_at=started_at,
            completed_at=max(started_at, completed_at),
        )

    def _failure(
        self,
        *,
        code: TrainingPlanningFailureCode,
        run_id: UUID,
        context_sha256: str,
        started_at: datetime,
        completed_at: datetime,
        provider_name: str = "NOT_CALLED",
        provider_model: str = "NOT_CALLED",
    ) -> TrainingPlanningFailure:
        return TrainingPlanningFailure(
            code=code,
            agent_run=self._audit(
                run_id=run_id,
                context_sha256=context_sha256,
                provider_name=provider_name,
                provider_model=provider_model,
                status=AgentRunStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
            ),
        )

    async def plan(self, command: object) -> TrainingPlanningResult:
        run_id = self._run_id(command)
        started_at, completed_at = self._times(command)
        try:
            exact_command = _strict_command(command)
            context, assessment, findings = _exact_assessment_and_context(exact_command)
            drill_catalog, execution_catalog, catalog_heads, execution_policies = (
                _exact_catalog_authorities(exact_command, context)
            )
        except _PlanningError as error:
            return self._failure(
                code=error.code,
                run_id=run_id,
                context_sha256=_INVALID_CONTEXT_SHA256,
                started_at=started_at,
                completed_at=completed_at,
            )
        try:
            payload = _provider_payload(
                context=context,
                assessment=assessment,
                execution_policies=execution_policies,
                policy=self._policy,
            )
        except _PlanningError as error:
            return self._failure(
                code=error.code,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
            )
        request = CoachModelRequest(
            task_type=CoachTaskType.TRAINING_PLAN,
            prompt_version=self._prompt.version,
            system_prompt=self._prompt.system_prompt,
            input_payload=payload,
            output_schema=TrainingPlanningProviderOutput.model_json_schema(),
            request_id=run_id,
            timeout_ms=exact_command.timeout_ms,
        )
        try:
            async with asyncio.timeout(exact_command.timeout_ms / 1000):
                raw_result = await self._provider.generate_structured(request)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, Exception):
            return self._failure(
                code=TrainingPlanningFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        try:
            if isinstance(raw_result, CoachModelResponse):
                _bounded_provider_output(raw_result.structured_output, self._policy)
            result = revalidate_coach_model_result(raw_result)
        except _PlanningError as error:
            return self._failure(
                code=error.code,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        except ValueError:
            result = None
        if result is None or isinstance(result, CoachModelFailure):
            return self._failure(
                code=TrainingPlanningFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        if result.provider != self._audit_provider_name or result.model != self._audit_model_name:
            return self._failure(
                code=TrainingPlanningFailureCode.PROVIDER_FAILURE,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=self._audit_provider_name,
                provider_model=self._audit_model_name,
            )
        try:
            output = _parse_provider_output(result.structured_output, self._policy)
            draft = _assemble_draft(
                context=context,
                assessment=assessment,
                findings=findings,
                drill_catalog=drill_catalog,
                execution_catalog=execution_catalog,
                catalog_heads=catalog_heads,
                execution_policies=execution_policies,
                output=output,
                policy=self._policy,
                prompt_version=self._prompt.version,
                prompt_sha256=self._prompt.content_sha256,
                agent_run_id=run_id,
                created_at=completed_at,
            )
        except _PlanningError as error:
            return self._failure(
                code=error.code,
                run_id=run_id,
                context_sha256=context.context_sha256,
                started_at=started_at,
                completed_at=completed_at,
                provider_name=result.provider,
                provider_model=result.model,
            )
        audit = self._audit(
            run_id=run_id,
            context_sha256=context.context_sha256,
            provider_name=result.provider,
            provider_model=result.model,
            status=AgentRunStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=completed_at,
            output_sha256=draft.content_sha256,
        )
        return TrainingPlanningCompleted(draft=draft, agent_run=audit)


__all__ = [
    "TrainingPlanBlockAudit",
    "TrainingPlanCandidateV3",
    "TrainingPlanDraft",
    "TrainingPlanDraftStatus",
    "TrainingPlanGoal",
    "TrainingPlanGoalAuthority",
    "TrainingPlanningBlockCandidate",
    "TrainingPlanningCommand",
    "TrainingPlanningCompleted",
    "TrainingPlanningFailure",
    "TrainingPlanningFailureCode",
    "TrainingPlanningProviderOutput",
    "TrainingPlanningResult",
    "TrainingPlanningResultStatus",
    "TrainingPlanningWorkflow",
]
