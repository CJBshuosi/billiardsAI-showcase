from __future__ import annotations

import asyncio
import json

import pytest

from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
)
from billiards_ai.cloud.coach_agent.training_planning import (
    TrainingPlanDraftStatus,
    TrainingPlanningCompleted,
    TrainingPlanningFailure,
    TrainingPlanningFailureCode,
)
from billiards_ai.shared.models.coaching import AgentRunStatus, CoachAssessmentStatus
from billiards_ai.shared.plan_publication.contracts import FinalTrainingPlan
from tests.unit.cloud.coach_agent.training_planning_test_support import (
    ScriptedPlanningProvider,
    approved_planning_input,
    provider_output,
    response,
)


@pytest.mark.asyncio
async def test_approved_assessment_generates_deterministic_auditable_draft() -> None:
    context, assessment, command, harness = await approved_planning_input()

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningCompleted)
    draft = result.draft
    assert draft.status is TrainingPlanDraftStatus.DRAFT
    assert draft.assessment_id == assessment.assessment_id
    assert draft.assessment_revision == assessment.revision
    assert draft.context_id == context.context_id
    assert draft.context_sha256 == context.context_sha256
    assert draft.catalog_version == context.catalog_version
    assert draft.plan.version == 1
    assert draft.plan.created_at == command.completed_at
    assert (draft.plan.valid_until - draft.plan.created_at).total_seconds() == 604800
    assert draft.estimated_duration_seconds == 20 * 45
    assert len(draft.blocks) == len(draft.plan.blocks) == 1
    block = draft.blocks[0]
    assert block.difficulty_level == 2
    assert block.evidence_ref_ids == assessment.findings[0].evidence_ref_ids
    assert block.progression_threshold == 4
    assert block.regression_threshold == 3
    assert block.drill_content_sha256
    assert draft.plan.fallback_drill.drill_id == "straight-potting-01"
    assert result.agent_run.status is AgentRunStatus.SUCCEEDED
    assert result.agent_run.output_sha256 == draft.content_sha256
    assert result.agent_run.task_type == "TRAINING_PLAN"

    second = await harness.workflow.plan(command)
    assert second == result


@pytest.mark.asyncio
async def test_provider_receives_only_bounded_selection_context() -> None:
    _, _, command, harness = await approved_planning_input()

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningCompleted)
    request = harness.provider.requests[0]
    assert set(request.input_payload) == {
        "schema_version",
        "assessment",
        "eligible_drills",
        "history_summary",
        "planning_limits",
    }
    serialized = json.dumps(request.input_payload, ensure_ascii=False, default=str)
    for forbidden in (
        "evidence_ref",
        "batch_id",
        "session_id",
        "shot_id",
        "media",
        "video",
        "openid",
        "token",
        "prompt",
        "context_sha256",
    ):
        assert forbidden not in serialized.lower()
    assert len(serialized.encode()) <= 32768


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"drill_id": "missing-drill"},
        {"difficulty_level": 5},
        {"target_repetitions": 21},
        {"target_finding_id": "00000000-0000-4000-8000-000000009999"},
    ],
)
async def test_invalid_provider_selection_fails_closed(override: dict[str, object]) -> None:
    _, assessment, command, _ = await approved_planning_input()
    output = provider_output(assessment)
    block = output["blocks"][0] | override
    provider = ScriptedPlanningProvider(response(output | {"blocks": [block]}))
    _, _, command, harness = await approved_planning_input(provider=provider)

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INVALID_SELECTION
    assert result.agent_run.status is AgentRunStatus.FAILED
    assert result.agent_run.output_sha256 is None


@pytest.mark.asyncio
async def test_unapproved_assessment_is_rejected_before_provider_call() -> None:
    _, assessment, command, harness = await approved_planning_input()
    unsafe = assessment.model_copy(update={"status": CoachAssessmentStatus.REVIEW_REQUIRED})
    command = command.model_copy(update={"assessment": unsafe})

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INVALID_CONTEXT
    assert harness.provider.requests == []


@pytest.mark.asyncio
async def test_provider_failure_is_payload_free_and_cancellation_propagates() -> None:
    failure = CoachModelFailure(
        provider="scripted",
        model="synthetic-planner-v1",
        code=CoachModelFailureCode.TIMEOUT,
        retryable=True,
        latency_ms=3,
    )
    provider = ScriptedPlanningProvider(failure)
    _, _, command, harness = await approved_planning_input(provider=provider)

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.PROVIDER_FAILURE
    assert set(result.model_dump()) == {"status", "code", "agent_run"}

    cancelled = ScriptedPlanningProvider(asyncio.CancelledError())
    _, _, command, harness = await approved_planning_input(provider=cancelled)
    with pytest.raises(asyncio.CancelledError):
        await harness.workflow.plan(command)


@pytest.mark.asyncio
async def test_open_plan_load_reduces_allowed_blocks_to_one() -> None:
    first_context, first_assessment, first_command, first_harness = await approved_planning_input()
    first = await first_harness.workflow.plan(first_command)
    assert isinstance(first, TrainingPlanningCompleted)
    output = provider_output(first_assessment)
    output["blocks"] = [
        output["blocks"][0],
        output["blocks"][0] | {"drill_id": "straight-potting-01"},
    ]
    provider = ScriptedPlanningProvider(response(output))
    _, _, command, harness = await approved_planning_input(
        provider=provider,
        prior_drafts=(first.draft,),
        prior_assessments=(first_assessment,),
    )

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.WORKLOAD_EXCEEDED


@pytest.mark.asyncio
async def test_recursive_unsafe_command_fails_without_payload_leak() -> None:
    _, _, command, harness = await approved_planning_input()
    recursive: list[object] = []
    recursive.append(recursive)
    unsafe = command.model_construct(**(dict(vars(command)) | {"assessment": recursive}))

    result = await harness.workflow.plan(unsafe)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.INVALID_CONTEXT
    assert "recursive" not in result.model_dump_json().lower()


@pytest.mark.asyncio
async def test_draft_embedded_plan_is_compatible_with_story_97_final_seam() -> None:
    _, _, command, harness = await approved_planning_input()
    result = await harness.workflow.plan(command)
    assert isinstance(result, TrainingPlanningCompleted)

    final = FinalTrainingPlan.from_v1(
        result.draft.plan,
        snapshot_id=result.draft.snapshot_id,
        snapshot_version=result.draft.snapshot_version,
        admission_status="VALIDATED",
    )

    assert final.plan_id == result.draft.plan.plan_id
    assert final.assessment_id == result.draft.assessment_id
    assert final.admission_status == "VALIDATED"


@pytest.mark.asyncio
async def test_provider_identity_drift_is_rejected() -> None:
    _, assessment, _, _ = await approved_planning_input()
    drifted = response(provider_output(assessment)).model_copy(
        update={"provider": "other-provider"}
    )
    provider = ScriptedPlanningProvider(drifted)
    _, _, command, harness = await approved_planning_input(provider=provider)

    result = await harness.workflow.plan(command)

    assert isinstance(result, TrainingPlanningFailure)
    assert result.code is TrainingPlanningFailureCode.PROVIDER_FAILURE
