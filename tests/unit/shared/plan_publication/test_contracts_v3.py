import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.training_plan import DrillReference
from billiards_ai.shared.plan_publication.codec import canonical_json_bytes
from billiards_ai.shared.plan_publication.contracts import (
    AbstentionAction,
    AbstentionRuleV1,
    FinalTrainingPlan,
    FinalTrainingPlanV3,
    MetricCombination,
    ProgressionExecutionRuleV1,
    RegressionExecutionRuleV1,
    ReplanContinueAction,
    ReplanRuleV1,
    SuccessMetricRuleV1,
    SuccessPolicyV1,
    TrainingBlockV3,
)


def _reference(drill_id: str = "primary") -> DrillReference:
    return DrillReference(
        catalog_version="2026.08",
        drill_id=drill_id,
        drill_revision=1,
        required_capabilities=("TABLE_MM_COORDINATES",),
    )


def _block(**updates: object) -> TrainingBlockV3:
    values = {
        "block_id": UUID("00000000-0000-4000-8000-000000000021"),
        "order": 1,
        "drill": _reference(),
        "target_finding_id": UUID("00000000-0000-4000-8000-000000000022"),
        "target_repetitions": 20,
        "initial_difficulty": 4,
        "minimum_runtime_difficulty": 2,
        "maximum_runtime_difficulty": 6,
        "progression_rule": ProgressionExecutionRuleV1(
            evaluator_result="SUCCESS", consecutive_count=4, difficulty_delta=1
        ),
        "regression_rule": RegressionExecutionRuleV1(
            evaluator_result="FAILURE", consecutive_count=3, difficulty_delta=-1
        ),
        "abstention_rule": AbstentionRuleV1(
            consecutive_count=2,
            action=AbstentionAction.USE_PUBLISHED_FALLBACK,
            reset_on="OBSERVED_SUPPORTED",
        ),
        "success_policy": SuccessPolicyV1(
            evaluator_policy_version="edge-success-v1",
            metric_rules=(
                SuccessMetricRuleV1(
                    metric_code="POT_SUCCESS_COUNT",
                    target_attempts=20,
                    minimum_successes=12,
                    evaluator_code="POT_RESULT",
                    evaluator_version="1.0",
                    required_capabilities=("TABLE_MM_COORDINATES",),
                ),
            ),
            combination=MetricCombination.ALL,
        ),
        "replan_rule": ReplanRuleV1(
            reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
            continue_action=ReplanContinueAction.USE_PUBLISHED_FALLBACK,
        ),
        "fallback_drill": _reference("fallback"),
    }
    values.update(updates)
    return TrainingBlockV3.model_validate(values)


def _plan(**updates: object) -> FinalTrainingPlanV3:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    payload = {
        "schema_version": "3.0",
        "plan_id": UUID("00000000-0000-4000-8000-000000000023"),
        "user_id": UUID("00000000-0000-4000-8000-000000000001"),
        "assessment_id": UUID("00000000-0000-4000-8000-000000000024"),
        "assessment_revision": 1,
        "snapshot_id": UUID("00000000-0000-4000-8000-000000000025"),
        "snapshot_version": 3,
        "version": 1,
        "admission_status": "APPROVED",
        "created_at": now,
        "valid_until": now + timedelta(days=1),
        "blocks": (_block(),),
        "fallback_drill": _reference("fallback"),
    }
    payload.update(updates)
    return FinalTrainingPlanV3.model_validate(
        payload | {"content_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}
    )


def test_v3_execution_fields_are_required_and_hash_bound() -> None:
    plan = _plan()
    changed = plan.model_dump(mode="python", exclude={"content_sha256"})
    changed["blocks"] = (plan.blocks[0].model_copy(update={"initial_difficulty": 5}),)
    with pytest.raises(ValidationError, match="content hash"):
        FinalTrainingPlanV3.model_validate(changed | {"content_sha256": plan.content_sha256})
    missing = plan.model_dump(mode="python")
    block = dict(missing["blocks"][0])
    del block["abstention_rule"]
    missing["blocks"] = (block,)
    with pytest.raises(ValidationError):
        FinalTrainingPlanV3.model_validate(missing)


def test_runtime_bounds_are_independent_and_ordered() -> None:
    with pytest.raises(ValidationError, match="runtime difficulty"):
        _block(initial_difficulty=1)


def test_v2_cannot_be_silently_upgraded_to_v3() -> None:
    assert FinalTrainingPlan.model_fields["schema_version"].default == "2.0"
    v3 = _plan().model_dump(mode="python")
    v3["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        FinalTrainingPlanV3.model_validate(v3)
