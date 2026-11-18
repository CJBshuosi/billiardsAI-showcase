from __future__ import annotations

from datetime import timedelta

import pytest

from billiards_ai.cloud.coach_agent.training_plan_repository import (
    InMemoryTrainingPlanRepository,
    TrainingPlanLifecycleStatus,
    TrainingPlanRepositoryError,
    TrainingPlanRepositoryErrorCode,
    TrainingPlanSaveCommand,
)
from billiards_ai.cloud.coach_agent.training_planning import TrainingPlanningCompleted
from billiards_ai.cloud.training.catalog import DrillCatalog
from tests.unit.cloud.coach_agent.training_planning_test_support import (
    approved_planning_input,
)


@pytest.mark.asyncio
async def test_save_is_atomic_append_only_and_exactly_replayable() -> None:
    _, _, command, harness = await approved_planning_input()
    result = await harness.workflow.plan(command)
    assert isinstance(result, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    save = TrainingPlanSaveCommand(
        draft=result.draft,
        agent_run=result.agent_run,
        expected_current_plan_version=0,
        idempotency_key="plan:first",
    )

    stored = repository.save(save)
    replay = repository.save(save)

    assert replay == stored
    assert repository.get_current(result.draft.plan.user_id) == result.draft
    assert len(repository.list_drafts(result.draft.plan.user_id)) == 1
    assert len(repository.list_agent_runs(result.draft.plan.user_id)) == 1
    records = repository.list_lifecycle(result.draft.plan.user_id)
    assert len(records) == 1
    assert records[0].status is TrainingPlanLifecycleStatus.DRAFT


@pytest.mark.asyncio
async def test_same_idempotency_key_with_different_command_conflicts() -> None:
    _, _, command, harness = await approved_planning_input()
    result = await harness.workflow.plan(command)
    assert isinstance(result, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    repository.save(
        TrainingPlanSaveCommand(
            draft=result.draft,
            agent_run=result.agent_run,
            expected_current_plan_version=0,
            idempotency_key="plan:first",
        )
    )
    changed = TrainingPlanSaveCommand(
        draft=result.draft,
        agent_run=result.agent_run,
        expected_current_plan_version=1,
        idempotency_key="plan:first",
    )

    with pytest.raises(TrainingPlanRepositoryError) as error:
        repository.save(changed)

    assert error.value.code is TrainingPlanRepositoryErrorCode.IDEMPOTENCY_CONFLICT
    assert len(repository.list_drafts(result.draft.plan.user_id)) == 1


@pytest.mark.asyncio
async def test_new_assessment_revision_supersedes_old_plan_without_overwrite() -> None:
    _, first_assessment, first_command, first_harness = await approved_planning_input()
    first = await first_harness.workflow.plan(first_command)
    assert isinstance(first, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    repository.save(
        TrainingPlanSaveCommand(
            draft=first.draft,
            agent_run=first.agent_run,
            expected_current_plan_version=0,
            idempotency_key="plan:first",
        )
    )
    newer = first_assessment.model_copy(
        update={
            "revision": first_assessment.revision + 1,
            "created_at": first_assessment.created_at + timedelta(seconds=1),
        }
    )
    # Recompute through the helper so the approved assessment hash and Context fence remain exact.
    _, newer, second_command, second_harness = await approved_planning_input(
        assessment=newer,
        prior_assessments=(first_assessment,),
        prior_drafts=(first.draft,),
    )
    second = await second_harness.workflow.plan(second_command)
    assert isinstance(second, TrainingPlanningCompleted)

    repository.save(
        TrainingPlanSaveCommand(
            draft=second.draft,
            agent_run=second.agent_run,
            expected_current_plan_version=1,
            idempotency_key="plan:second",
        )
    )

    assert repository.get_current(first.draft.plan.user_id) == second.draft
    assert len(repository.list_drafts(first.draft.plan.user_id)) == 2
    records = repository.list_lifecycle(first.draft.plan.user_id)
    assert [item.status for item in records] == [
        TrainingPlanLifecycleStatus.DRAFT,
        TrainingPlanLifecycleStatus.SUPERSEDED,
        TrainingPlanLifecycleStatus.DRAFT,
    ]
    assert records[1].plan_id == first.draft.plan.plan_id
    assert records[1].superseded_by_plan_id == second.draft.plan.plan_id
    assert records[1].recorded_at == second.draft.created_at


@pytest.mark.asyncio
async def test_new_catalog_version_supersedes_old_plan() -> None:
    _, assessment, first_command, first_harness = await approved_planning_input()
    first = await first_harness.workflow.plan(first_command)
    assert isinstance(first, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    repository.save(
        TrainingPlanSaveCommand(
            draft=first.draft,
            agent_run=first.agent_run,
            expected_current_plan_version=0,
            idempotency_key="catalog:first",
        )
    )
    from tests.unit.cloud.coach_agent.training_planning_test_support import ROOT

    old_catalog = DrillCatalog.model_validate_json((ROOT / "drill-catalog.json").read_text())
    new_catalog = DrillCatalog.build(
        catalog_version="2026.09",
        created_at=old_catalog.created_at,
        revision_history=old_catalog.revision_history,
    )
    _, _, second_command, second_harness = await approved_planning_input(
        assessment=assessment,
        prior_assessments=(assessment,),
        prior_drafts=(first.draft,),
        catalog=new_catalog,
    )
    second = await second_harness.workflow.plan(second_command)
    assert isinstance(second, TrainingPlanningCompleted)

    repository.save(
        TrainingPlanSaveCommand(
            draft=second.draft,
            agent_run=second.agent_run,
            expected_current_plan_version=1,
            idempotency_key="catalog:second",
        )
    )

    assert repository.get_current(assessment.user_id).catalog_version == "2026.09"
    assert repository.list_lifecycle(assessment.user_id)[-2].status is (
        TrainingPlanLifecycleStatus.SUPERSEDED
    )


@pytest.mark.asyncio
async def test_cas_or_same_authority_replacement_fails_without_partial_state() -> None:
    _, _, command, harness = await approved_planning_input()
    result = await harness.workflow.plan(command)
    assert isinstance(result, TrainingPlanningCompleted)
    repository = InMemoryTrainingPlanRepository()
    repository.save(
        TrainingPlanSaveCommand(
            draft=result.draft,
            agent_run=result.agent_run,
            expected_current_plan_version=0,
            idempotency_key="plan:first",
        )
    )

    for expected, code in (
        (0, TrainingPlanRepositoryErrorCode.VERSION_CONFLICT),
        (1, TrainingPlanRepositoryErrorCode.AUTHORITY_UNCHANGED),
    ):
        with pytest.raises(TrainingPlanRepositoryError) as error:
            repository.save(
                TrainingPlanSaveCommand(
                    draft=result.draft,
                    agent_run=result.agent_run,
                    expected_current_plan_version=expected,
                    idempotency_key=f"plan:conflict:{expected}",
                )
            )
        assert error.value.code is code
    assert len(repository.list_drafts(result.draft.plan.user_id)) == 1
