from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.training_planning import (
    TrainingPlanDraft,
    TrainingPlanDraftStatus,
    TrainingPlanningBlockCandidate,
    TrainingPlanningProviderOutput,
)
from billiards_ai.cloud.training.planning_policy import TrainingPlanningPolicyV1
from tests.unit.cloud.coach_agent.training_planning_test_support import approved_planning_input


def test_provider_contract_is_strict_frozen_and_selection_only() -> None:
    output = TrainingPlanningProviderOutput(
        primary_goal_finding_id=UUID("71b7abcd-4430-4d16-9180-bbc6ffed6cdd"),
        secondary_goal_finding_ids=(),
        blocks=(
            TrainingPlanningBlockCandidate(
                drill_id="large-angle-potting-01",
                drill_revision=1,
                target_finding_id=UUID("71b7abcd-4430-4d16-9180-bbc6ffed6cdd"),
                target_repetitions=12,
                difficulty_level=2,
            ),
        ),
        fallback_drill_id="straight-potting-01",
        fallback_drill_revision=1,
    )

    assert output.schema_version == "1.0"
    with pytest.raises(ValidationError):
        TrainingPlanningProviderOutput.model_validate(
            output.model_dump() | {"rationale": "provider text"}, strict=True
        )
    with pytest.raises(ValidationError):
        TrainingPlanningProviderOutput.model_validate(
            output.model_dump() | {"schema_version": "2.0"}, strict=True
        )
    with pytest.raises(ValidationError):
        output.blocks[0].difficulty_level = 3  # type: ignore[misc]


@pytest.mark.asyncio
async def test_draft_contract_is_hash_bound_and_not_publishable() -> None:
    context, assessment, command, provider = await approved_planning_input()
    result = await provider.workflow.plan(command)
    draft = result.draft

    assert isinstance(draft, TrainingPlanDraft)
    assert draft.status is TrainingPlanDraftStatus.DRAFT
    assert "publication" not in TrainingPlanDraft.model_fields
    assert "signature" not in TrainingPlanDraft.model_fields
    assert draft.plan.user_id == context.user_id == assessment.user_id
    with pytest.raises(ValidationError):
        TrainingPlanDraft.model_validate(
            draft.model_dump(mode="python") | {"content_sha256": "f" * 64}, strict=True
        )
    with pytest.raises(ValidationError):
        TrainingPlanDraft.model_validate(
            draft.model_dump(mode="python") | {"created_at": datetime.now(tz=UTC)},
            strict=True,
        )


def test_policy_v1_is_frozen_and_has_hard_caps() -> None:
    policy = TrainingPlanningPolicyV1()

    assert policy.policy_version == "training-planning-policy-v1"
    assert policy.maximum_blocks == 2
    assert policy.maximum_total_repetitions == 40
    assert policy.maximum_estimated_duration_seconds == 1800
    assert policy.valid_for_seconds == 604800
    with pytest.raises(ValidationError):
        TrainingPlanningPolicyV1(maximum_blocks=99)
