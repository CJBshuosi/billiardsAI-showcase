from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from billiards_ai.cloud.coach_agent.context import (
    AssessmentHistoryPointer,
    CapabilityManifest,
    CoachContextSourceBundle,
    CoachSourceReadFenceV3,
    CoachTaskType,
    PlanHistoryPointer,
    PlanOutcomeHistoryPointer,
    build_coach_context,
    coach_history_content_sha256,
)
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelRequest,
    CoachModelResponse,
)
from billiards_ai.cloud.coach_agent.training_planning import (
    TrainingPlanDraft,
    TrainingPlanningCommand,
    TrainingPlanningWorkflow,
)
from billiards_ai.cloud.training.catalog import DrillCatalog, DrillDefinition
from billiards_ai.shared.models.coaching import CoachAssessmentStatus, CoachAssessmentV2
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.training_plan import DrillReference
from billiards_ai.shared.plan_publication.contracts import (
    AbstentionAction,
    AbstentionRuleV1,
    DrillExecutionPolicyDefinitionV1,
    ExecutionPolicyCatalogV1,
    MetricCombination,
    ProgressionExecutionRuleV1,
    RegressionExecutionRuleV1,
    ReplanContinueAction,
    ReplanRuleV1,
    SuccessMetricRuleV1,
    SuccessPolicyV1,
)
from tests.unit.cloud.coach_agent.test_assessment_validation import (
    _replace_assessment,
    _valid_bundle,
)

ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
STARTED = datetime(2026, 8, 17, 9, 0, tzinfo=UTC)
COMPLETED = STARTED + timedelta(seconds=2)


class ScriptedPlanningProvider:
    def __init__(self, result: CoachModelResponse | CoachModelFailure | BaseException):
        self.result = result
        self.requests: list[CoachModelRequest] = []

    @property
    def audit_provider_name(self) -> str:
        return "scripted"

    @property
    def audit_model_name(self) -> str:
        return "synthetic-planner-v1"

    async def generate_structured(
        self, request: CoachModelRequest
    ) -> CoachModelResponse | CoachModelFailure:
        self.requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def response(output: dict[str, Any]) -> CoachModelResponse:
    return CoachModelResponse(
        provider="scripted",
        model="synthetic-planner-v1",
        provider_request_id="planning-request-1",
        structured_output=output,
        finish_reason="stop",
        input_tokens=20,
        output_tokens=10,
        latency_ms=5,
    )


def provider_output(assessment: CoachAssessmentV2) -> dict[str, Any]:
    assert assessment.primary_finding_id is not None
    return {
        "schema_version": "1.0",
        "primary_goal_finding_id": str(assessment.primary_finding_id),
        "secondary_goal_finding_ids": [],
        "blocks": [
            {
                "drill_id": "large-angle-potting-01",
                "drill_revision": 1,
                "target_finding_id": str(assessment.primary_finding_id),
                "target_repetitions": 20,
                "difficulty_level": 2,
            }
        ],
        "fallback_drill_id": "straight-potting-01",
        "fallback_drill_revision": 1,
    }


def execution_policy_catalog(
    catalog: DrillCatalog,
    *,
    primary_initial_difficulty: int = 2,
    primary_minimum_runtime_difficulty: int = 1,
    primary_maximum_runtime_difficulty: int = 5,
) -> ExecutionPolicyCatalogV1:
    heads = {item.drill_id: item for item in catalog.drills}
    fallback = heads["straight-potting-01"]

    def definition(
        drill: DrillDefinition,
        *,
        initial_difficulty: int,
        minimum_runtime_difficulty: int,
        maximum_runtime_difficulty: int,
        fallback_drill: DrillDefinition | None,
    ) -> DrillExecutionPolicyDefinitionV1:
        terminal = fallback_drill is None
        return DrillExecutionPolicyDefinitionV1(
            drill_id=drill.drill_id,
            drill_revision=drill.revision,
            drill_content_sha256=drill.content_sha256,
            required_capabilities=drill.required_capabilities,
            initial_difficulty=initial_difficulty,
            minimum_runtime_difficulty=minimum_runtime_difficulty,
            maximum_runtime_difficulty=maximum_runtime_difficulty,
            progression_rule=ProgressionExecutionRuleV1(
                evaluator_result="SUCCESS",
                consecutive_count=drill.progression_rule.threshold,
                difficulty_delta=drill.progression_rule.level_delta,
            ),
            regression_rule=RegressionExecutionRuleV1(
                evaluator_result="FAILURE",
                consecutive_count=drill.regression_rule.threshold,
                difficulty_delta=drill.regression_rule.level_delta,
            ),
            abstention_rule=AbstentionRuleV1(
                consecutive_count=2,
                action=(
                    AbstentionAction.SAFE_PAUSE
                    if terminal
                    else AbstentionAction.USE_PUBLISHED_FALLBACK
                ),
                reset_on="OBSERVED_SUPPORTED",
            ),
            success_policy=SuccessPolicyV1(
                evaluator_policy_version="edge-success-v1",
                metric_rules=tuple(
                    SuccessMetricRuleV1(
                        metric_code=item.metric_code,
                        target_attempts=item.target_attempts,
                        minimum_successes=item.minimum_successes,
                        evaluator_code="POT_RESULT",
                        evaluator_version="1.0",
                        required_capabilities=drill.required_capabilities,
                    )
                    for item in drill.success_metrics
                ),
                combination=MetricCombination.ALL,
            ),
            replan_rule=ReplanRuleV1(
                reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
                continue_action=(
                    ReplanContinueAction.SAFE_PAUSE
                    if terminal
                    else ReplanContinueAction.USE_PUBLISHED_FALLBACK
                ),
            ),
            fallback_drill=(
                None
                if fallback_drill is None
                else DrillReference(
                    catalog_version=catalog.catalog_version,
                    drill_id=fallback_drill.drill_id,
                    drill_revision=fallback_drill.revision,
                    required_capabilities=fallback_drill.required_capabilities,
                )
            ),
        )

    definitions = tuple(
        definition(
            drill,
            initial_difficulty=(
                primary_initial_difficulty if drill.drill_id == "large-angle-potting-01" else 1
            ),
            minimum_runtime_difficulty=(
                primary_minimum_runtime_difficulty
                if drill.drill_id == "large-angle-potting-01"
                else 1
            ),
            maximum_runtime_difficulty=(
                primary_maximum_runtime_difficulty
                if drill.drill_id == "large-angle-potting-01"
                else 3
            ),
            fallback_drill=None if drill is fallback else fallback,
        )
        for drill in catalog.drills
    )
    return ExecutionPolicyCatalogV1.build(
        catalog_version=catalog.catalog_version,
        drill_catalog_content_sha256=catalog.content_sha256,
        definitions=definitions,
    )


def _rehash_assessment(assessment: CoachAssessmentV2) -> CoachAssessmentV2:
    payload = {key: value for key, value in vars(assessment).items() if key != "content_sha256"}
    encoded = CoachAssessmentV2.model_construct(**payload, content_sha256="0" * 64).model_dump(
        mode="json", exclude={"content_sha256"}
    )
    return CoachAssessmentV2.model_validate(
        payload | {"content_sha256": canonical_sha256(encoded)}, strict=True
    )


def _assessment_pointer(assessment: CoachAssessmentV2) -> AssessmentHistoryPointer:
    return AssessmentHistoryPointer(
        assessment_id=assessment.assessment_id,
        user_id=assessment.user_id,
        snapshot_id=assessment.snapshot_id,
        snapshot_version=assessment.snapshot_version,
        revision=assessment.revision,
        created_at=assessment.created_at,
        content_sha256=assessment.content_sha256,
    )


def _plan_pointer(draft: TrainingPlanDraft) -> PlanHistoryPointer:
    return PlanHistoryPointer(
        plan_id=draft.plan.plan_id,
        user_id=draft.plan.user_id,
        plan_version=draft.plan.version,
        snapshot_id=draft.snapshot_id,
        snapshot_version=draft.snapshot_version,
        assessment_id=draft.assessment_id,
        assessment_revision=draft.assessment_revision,
        created_at=draft.plan.created_at,
        content_sha256=draft.content_sha256,
    )


@dataclass(frozen=True)
class PlanningHarness:
    provider: ScriptedPlanningProvider
    workflow: TrainingPlanningWorkflow


async def approved_planning_input(
    *,
    provider: ScriptedPlanningProvider | None = None,
    assessment: CoachAssessmentV2 | None = None,
    prior_assessments: tuple[CoachAssessmentV2, ...] = (),
    prior_drafts: tuple[TrainingPlanDraft, ...] = (),
    prior_outcomes: tuple[PlanOutcomeHistoryPointer, ...] = (),
    catalog: DrillCatalog | None = None,
    policies: ExecutionPolicyCatalogV1 | None = None,
):
    growth_context, draft, _ = await _valid_bundle()
    if assessment is None and prior_assessments:
        assessment = prior_assessments[-1]
    elif assessment is None:
        assessment = _replace_assessment(
            draft,
            status=CoachAssessmentStatus.APPROVED,
            revision=2,
            created_at=datetime(2026, 8, 16, 10, tzinfo=UTC),
        )
    else:
        assessment = _rehash_assessment(assessment)
    assessment_by_revision = {
        (item.assessment_id, item.revision): item for item in (*prior_assessments, assessment)
    }
    all_assessments = tuple(
        sorted(
            assessment_by_revision.values(),
            key=lambda item: (item.created_at, str(item.assessment_id), item.revision),
        )
    )
    assessment_pointers = tuple(_assessment_pointer(item) for item in all_assessments)
    plan_pointers = tuple(_plan_pointer(item) for item in prior_drafts)
    catalog = catalog or DrillCatalog.model_validate_json((ROOT / "drill-catalog.json").read_text())
    policies = policies or execution_policy_catalog(catalog)
    manifest = CapabilityManifest(
        manifest_version="planning-capabilities-v1",
        user_id=assessment.user_id,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("CUE_BALL_TARGET_ZONE", "TABLE_MM_COORDINATES"),
        available_equipment_codes=("STANDARD_TABLE",),
    )
    history_hash = coach_history_content_sha256(
        CoachTaskType.TRAINING_PLAN,
        assessment_pointers,
        plan_pointers,
        prior_outcomes,
    )
    fence = CoachSourceReadFenceV3.build_v3(
        snapshot_read_fence=growth_context.growth_snapshot.read_fence,
        authority_revision="planning-aux-r1",
        history_revision=f"history-r{len(plan_pointers)}",
        history_content_sha256=history_hash,
        catalog_version=catalog.catalog_version,
        capability_manifest_version=manifest.manifest_version,
    )
    history_times = (
        *(item.created_at for item in prior_drafts),
        *(item.completed_at for item in prior_outcomes),
    )
    context_as_of = (
        STARTED - timedelta(minutes=1)
        if not history_times
        else max(history_times) + timedelta(seconds=1)
    )
    run_started = context_as_of + timedelta(minutes=1)
    run_completed = run_started + timedelta(seconds=2)
    context = build_coach_context(
        CoachContextSourceBundle(
            task_type=CoachTaskType.TRAINING_PLAN,
            user_id=assessment.user_id,
            as_of=context_as_of,
            read_fence=fence,
            growth_snapshot=growth_context.growth_snapshot,
            prior_assessments=assessment_pointers,
            prior_plans=plan_pointers,
            plan_outcomes=prior_outcomes,
            drill_catalog=catalog,
            drill_profile=manifest.eligibility_profile(),
            capability_manifest=manifest,
        )
    )
    provider = provider or ScriptedPlanningProvider(response(provider_output(assessment)))
    workflow = TrainingPlanningWorkflow(provider=provider, audit_authority=provider)
    command = TrainingPlanningCommand(
        context=context,
        assessment=assessment,
        drill_catalog=catalog,
        execution_policy_catalog=policies,
        started_at=run_started,
        completed_at=run_completed,
        timeout_ms=10_000,
    )
    return context, assessment, command, PlanningHarness(provider=provider, workflow=workflow)
