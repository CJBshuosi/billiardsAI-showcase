from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent import prompts
from billiards_ai.cloud.coach_agent.context import PlanOutcomeHistoryPointer
from billiards_ai.cloud.coach_agent.provider import CoachModelResponse
from billiards_ai.cloud.coach_agent.training_plan_repository import (
    InMemoryTrainingPlanRepository,
    TrainingPlanFailureSaveCommand,
    TrainingPlanRepositoryError,
    TrainingPlanRepositoryErrorCode,
    TrainingPlanSaveCommand,
)
from billiards_ai.cloud.coach_agent.training_planning import (
    TrainingPlanningCommand,
    TrainingPlanningCompleted,
    TrainingPlanningFailure,
    TrainingPlanningFailureCode,
    TrainingPlanningWorkflow,
)
from billiards_ai.cloud.training.catalog import DrillCatalog
from billiards_ai.cloud.training.planning_policy import TrainingPlanningPolicyV1
from billiards_ai.shared.plan_publication.contracts import ExecutionPolicyCatalogV1
from tests.unit.cloud.coach_agent.training_planning_test_support import (
    ROOT,
    ScriptedPlanningProvider,
    _rehash_assessment,
    approved_planning_input,
    execution_policy_catalog,
    provider_output,
    response,
)


def _catalog() -> DrillCatalog:
    return DrillCatalog.model_validate_json((ROOT / "drill-catalog.json").read_text())


def _with_authorities(
    command: TrainingPlanningCommand,
    *,
    catalog: DrillCatalog | None = None,
    policies: ExecutionPolicyCatalogV1 | None = None,
) -> TrainingPlanningCommand:
    catalog = catalog or _catalog()
    policies = policies or execution_policy_catalog(catalog)
    return command.model_copy(
        update={"drill_catalog": catalog, "execution_policy_catalog": policies}
    )


def _workflow(provider: ScriptedPlanningProvider, **updates: object) -> TrainingPlanningWorkflow:
    return TrainingPlanningWorkflow(
        provider=provider,
        audit_authority=provider,
        **updates,
    )


@pytest.mark.asyncio
async def test_v3_ready_candidate_uses_exact_execution_policy_authority() -> None:
    catalog = _catalog()
    policies = execution_policy_catalog(
        catalog,
        primary_initial_difficulty=7,
        primary_minimum_runtime_difficulty=6,
        primary_maximum_runtime_difficulty=8,
    )
    _, assessment, command, harness = await approved_planning_input()
    output = provider_output(assessment)
    output["blocks"][0].update(target_repetitions=20, difficulty_level=7)
    provider = ScriptedPlanningProvider(response(output))

    result = await _workflow(provider).plan(
        _with_authorities(command, catalog=catalog, policies=policies)
    )

    assert isinstance(result, TrainingPlanningCompleted)
    assert result.draft.execution_policy_catalog.content_sha256
    block = result.draft.v3_candidate.blocks[0]
    assert block.target_repetitions == 20
    assert block.initial_difficulty == 7
    assert block.minimum_runtime_difficulty == 6
    assert block.maximum_runtime_difficulty == 8
    exact_policy = result.draft.execution_policy_catalog.definitions[0]
    assert block.progression_rule == exact_policy.progression_rule
    assert block.regression_rule == exact_policy.regression_rule
    assert block.success_policy == exact_policy.success_policy
    assert result.draft.blocks[0].target_skill_codes == ("POTTING",)
    assert result.draft.plan.blocks[0].target_repetitions == 20
    assert harness.provider.requests == []


@pytest.mark.asyncio
async def test_invalid_command_identity_serialization_maps_to_invalid_context() -> None:
    _, _, command, harness = await approved_planning_input()
    unsafe = command.model_construct(**(dict(vars(command)) | {"started_at": object()}))

    result = await _workflow(harness.provider).plan(unsafe)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INVALID_CONTEXT


@pytest.mark.asyncio
async def test_workflow_enforces_timeout_and_preserves_cancellation() -> None:
    _, assessment, command, _ = await approved_planning_input()

    class SlowProvider(ScriptedPlanningProvider):
        async def generate_structured(self, request):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            return await super().generate_structured(request)

    output = provider_output(assessment)
    output["blocks"][0].update(target_repetitions=20)
    provider = SlowProvider(response(output))
    result = await _workflow(provider).plan(
        _with_authorities(command.model_copy(update={"timeout_ms": 1}))
    )
    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.PROVIDER_FAILURE

    cancelled = ScriptedPlanningProvider(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _workflow(cancelled).plan(_with_authorities(command))


def test_policy_is_strictly_revalidated_at_workflow_boundary() -> None:
    provider = ScriptedPlanningProvider(
        CoachModelResponse.model_construct(
            provider="scripted",
            model="synthetic-planner-v1",
            structured_output={},
        )
    )
    unsafe = TrainingPlanningPolicyV1.model_construct(maximum_blocks=99)
    with pytest.raises(ValueError, match="planning policy"):
        _workflow(provider, policy=unsafe)


@pytest.mark.asyncio
async def test_only_current_assessment_head_is_accepted_and_run_id_binds_hash() -> None:
    _, current, current_command, current_harness = await approved_planning_input()
    older = _rehash_assessment(
        current.model_copy(
            update={"revision": 1, "created_at": current.created_at - timedelta(seconds=1)}
        )
    )
    # The helper rehashes both revisions and places current last in canonical history.
    _, current, command, harness = await approved_planning_input(
        assessment=current,
        prior_assessments=(older,),
    )
    current_result = await _workflow(harness.provider).plan(_with_authorities(command))
    old_command = command.model_copy(update={"assessment": older})
    old_result = await _workflow(current_harness.provider).plan(_with_authorities(old_command))

    assert isinstance(current_result, TrainingPlanningCompleted)
    assert isinstance(old_result, TrainingPlanningFailure)
    assert old_result.code is TrainingPlanningFailureCode.INVALID_CONTEXT
    assert old_result.agent_run.run_id != current_result.agent_run.run_id


def test_provider_audit_authority_is_explicit_and_required() -> None:
    provider = ScriptedPlanningProvider(
        CoachModelResponse.model_construct(
            provider="scripted",
            model="synthetic-planner-v1",
            structured_output={},
        )
    )
    with pytest.raises(TypeError):
        TrainingPlanningWorkflow(provider=provider)


@pytest.mark.asyncio
async def test_every_declared_goal_has_one_closed_authoritative_block() -> None:
    _, assessment, command, _ = await approved_planning_input()
    output = provider_output(assessment)
    output["blocks"][0].update(target_repetitions=20)
    output["blocks"].append(dict(output["blocks"][0]))
    provider = ScriptedPlanningProvider(response(output))

    result = await _workflow(provider).plan(_with_authorities(command))

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code in {
        TrainingPlanningFailureCode.INVALID_OUTPUT,
        TrainingPlanningFailureCode.INVALID_SELECTION,
    }


@pytest.mark.asyncio
async def test_open_load_uses_plan_version_set_difference() -> None:
    _, assessment, first_command, first_harness = await approved_planning_input()
    first = await first_harness.workflow.plan(first_command)
    assert isinstance(first, TrainingPlanningCompleted)
    _, _, second_command, second_harness = await approved_planning_input(
        assessment=assessment,
        prior_assessments=(assessment,),
        prior_drafts=(first.draft,),
    )
    second = await second_harness.workflow.plan(second_command)
    assert isinstance(second, TrainingPlanningCompleted)
    outcomes = tuple(
        PlanOutcomeHistoryPointer(
            outcome_id=uuid4(),
            user_id=assessment.user_id,
            plan_id=first.draft.plan.plan_id,
            plan_version=first.draft.plan.version,
            completed_at=second.draft.created_at + timedelta(seconds=index),
            content_sha256=("a" if index == 1 else "b") * 64,
        )
        for index in (1, 2)
    )
    _, assessment, third_command, _ = await approved_planning_input(
        assessment=assessment,
        prior_assessments=(assessment,),
        prior_drafts=(first.draft, second.draft),
        prior_outcomes=outcomes,
    )
    output = provider_output(assessment)
    output["blocks"][0].update(target_repetitions=20)
    provider = ScriptedPlanningProvider(response(output))
    result = await _workflow(provider).plan(_with_authorities(third_command))

    assert isinstance(result, TrainingPlanningCompleted)
    assert provider.requests[0].input_payload["planning_limits"]["maximum_blocks"] == 1


@pytest.mark.asyncio
async def test_same_authority_with_new_context_can_create_explicit_next_version() -> None:
    _, assessment, first_command, first_harness = await approved_planning_input()
    first = await first_harness.workflow.plan(first_command)
    assert isinstance(first, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    repository.save(
        TrainingPlanSaveCommand(
            draft=first.draft,
            agent_run=first.agent_run,
            expected_current_plan_version=0,
            idempotency_key="same-authority:first",
        )
    )
    _, _, second_command, second_harness = await approved_planning_input(
        assessment=assessment,
        prior_assessments=(assessment,),
        prior_drafts=(first.draft,),
    )
    second = await second_harness.workflow.plan(second_command)
    assert isinstance(second, TrainingPlanningCompleted)

    stored = repository.save(
        TrainingPlanSaveCommand(
            draft=second.draft,
            agent_run=second.agent_run,
            expected_current_plan_version=1,
            idempotency_key="same-authority:second",
        )
    )
    assert stored.draft.plan.version == 2


@pytest.mark.asyncio
async def test_failed_run_is_atomically_persisted_and_replayed() -> None:
    _, _, command, harness = await approved_planning_input()
    failure = await _workflow(harness.provider).plan(command.model_copy(update={"assessment": []}))
    assert isinstance(failure, TrainingPlanningFailure)
    repository = InMemoryTrainingPlanRepository()
    save = TrainingPlanFailureSaveCommand(
        user_id=command.context.user_id,
        failure=failure,
        idempotency_key="failure:first",
    )
    stored = repository.save_failure(save)
    assert repository.save_failure(save) == stored
    assert repository.list_agent_runs(command.context.user_id) == (failure.agent_run,)
    assert repository.list_drafts(command.context.user_id) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_update",
    [
        {"task_type": "GROWTH_ASSESSMENT"},
        {"prompt_version": "training-plan-v2"},
        {"prompt_sha256": "a" * 64},
        {"context_sha256": "b" * 64},
        {"output_sha256": "c" * 64},
        {"repair_count": 1},
        {
            "started_at": datetime(2026, 8, 17, 9, 0, 3, tzinfo=UTC),
            "completed_at": datetime(2026, 8, 17, 9, 0, 2, tzinfo=UTC),
        },
    ],
)
async def test_repository_rejects_cross_lineage_agent_run(
    run_update: dict[str, object],
) -> None:
    _, _, command, harness = await approved_planning_input()
    result = await harness.workflow.plan(command)
    assert isinstance(result, TrainingPlanningCompleted)
    bad_run = result.agent_run.model_copy(update=run_update)
    repository = InMemoryTrainingPlanRepository()
    with pytest.raises(TrainingPlanRepositoryError) as error:
        repository.save(
            TrainingPlanSaveCommand(
                draft=result.draft,
                agent_run=bad_run,
                expected_current_plan_version=0,
                idempotency_key="lineage:bad",
            )
        )
    assert error.value.code is TrainingPlanRepositoryErrorCode.INVALID_RECORD


@pytest.mark.asyncio
async def test_execution_policy_must_bind_exact_drill_catalog_authority() -> None:
    catalog = _catalog()
    policies = execution_policy_catalog(catalog)
    bad = ExecutionPolicyCatalogV1.build(
        catalog_version=policies.catalog_version,
        drill_catalog_content_sha256="f" * 64,
        definitions=policies.definitions,
    )
    _, assessment, command, _ = await approved_planning_input()
    output = provider_output(assessment)
    output["blocks"][0].update(target_repetitions=20)
    provider = ScriptedPlanningProvider(response(output))

    result = await _workflow(provider).plan(
        _with_authorities(command, catalog=catalog, policies=bad)
    )
    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INVALID_CONTEXT


def test_training_prompt_v1_hash_is_a_frozen_literal() -> None:
    assert prompts.TRAINING_PLANNING_PROMPT_V1_SHA256 == (
        "a1f48ab055bb02c44e982925aa37f8621e6f878023d26781db1eb7ed90caf5dd"
    )
    source = inspect.getsource(prompts)
    assert "TRAINING_PLANNING_PROMPT_V1_SHA256 = (" in source


def test_command_requires_exact_normalized_utc_and_monotonic_times() -> None:
    _, _, command, _ = asyncio.run(approved_planning_input())
    offset_utc = timezone(timedelta(0), name="not-canonical-utc")
    with pytest.raises(ValidationError):
        TrainingPlanningCommand.model_validate(
            command.model_dump(mode="python")
            | {"started_at": datetime(2026, 8, 17, 9, tzinfo=offset_utc)},
            strict=True,
        )
    with pytest.raises(ValidationError):
        TrainingPlanningCommand.model_validate(
            command.model_dump(mode="python")
            | {
                "started_at": datetime(2026, 8, 17, 9, 0, 1, tzinfo=UTC),
                "completed_at": datetime(2026, 8, 17, 9, tzinfo=UTC),
            },
            strict=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "oversized",
    [
        {"padding": "x" * 40_000},
        {"nested": [[[[[[[[[[{"value": 1}]]]]]]]]]]},
    ],
)
async def test_provider_output_budget_precedes_deep_validation(oversized: object) -> None:
    _, _, command, _ = await approved_planning_input()
    raw = CoachModelResponse.model_construct(
        provider="scripted",
        model="synthetic-planner-v1",
        provider_request_id="budget",
        structured_output=oversized,
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
    )
    provider = ScriptedPlanningProvider(raw)
    result = await _workflow(provider).plan(_with_authorities(command))
    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INPUT_LIMIT_EXCEEDED
