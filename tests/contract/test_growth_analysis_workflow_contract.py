from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.growth_analysis import (
    GrowthAnalysisCandidate,
    GrowthAnalysisFailure,
    GrowthAnalysisFailureCode,
    GrowthAnalysisProviderOutput,
    GrowthAnalysisResultStatus,
)
from billiards_ai.cloud.coach_agent.prompts import (
    GROWTH_ANALYSIS_PROMPT_V1_SHA256,
    get_growth_analysis_prompt,
)
from billiards_ai.shared.models.coaching import AgentRun, AgentRunStatus, FindingType
from billiards_ai.shared.models.events import canonical_sha256


def _uid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def test_growth_prompt_is_versioned_hash_bound_and_contains_non_causality_rules() -> None:
    prompt = get_growth_analysis_prompt("growth-analysis-v1")

    assert prompt.version == "growth-analysis-v1"
    assert GROWTH_ANALYSIS_PROMPT_V1_SHA256 == (
        "64d1d3bc878f53565b9f2f06f72afa7895c6f6989c3c53377b5200c0280cd1e4"
    )
    assert prompt.content_sha256 == GROWTH_ANALYSIS_PROMPT_V1_SHA256
    assert prompt.content_sha256 == canonical_sha256(prompt.system_prompt)
    assert "不得改写" in prompt.system_prompt
    assert "不证明因果" in prompt.system_prompt
    assert get_growth_analysis_prompt(prompt.version) is prompt

    with pytest.raises(ValueError, match="unsupported growth analysis prompt version"):
        get_growth_analysis_prompt("growth-analysis-v2")


def test_provider_candidate_contract_is_strict_frozen_and_minimal() -> None:
    candidate = GrowthAnalysisCandidate(
        metric_identity_sha256="a" * 64,
        finding_type=FindingType.IMPROVEMENT,
    )
    output = GrowthAnalysisProviderOutput(primary=candidate, secondary=None)

    assert set(candidate.model_dump()) == {"metric_identity_sha256", "finding_type"}
    assert output.schema_version == "1.0"
    with pytest.raises(ValidationError):
        GrowthAnalysisCandidate.model_validate(
            candidate.model_dump() | {"evidence_ref_ids": (_uid(1),)}, strict=True
        )
    with pytest.raises(ValidationError):
        candidate.finding_type = FindingType.DECLINE
    with pytest.raises(ValidationError):
        GrowthAnalysisProviderOutput(
            primary=candidate,
            secondary=candidate,
        )


def test_failure_result_is_payload_free_and_uses_stable_taxonomy() -> None:
    audit = AgentRun(
        run_id=_uid(1),
        task_type="GROWTH_ASSESSMENT",
        prompt_version="growth-analysis-v1",
        prompt_sha256="a" * 64,
        context_sha256="b" * 64,
        provider_name="NOT_CALLED",
        provider_model="NOT_CALLED",
        status=AgentRunStatus.FAILED,
        repair_count=0,
        started_at=datetime(2026, 8, 16, 8, tzinfo=UTC),
        completed_at=datetime(2026, 8, 16, 8, 0, 1, tzinfo=UTC),
    )
    result = GrowthAnalysisFailure(
        status=GrowthAnalysisResultStatus.NO_CONCLUSION,
        code=GrowthAnalysisFailureCode.UNSUPPORTED_SNAPSHOT,
        agent_run=audit,
    )

    assert set(result.model_dump()) == {"status", "code", "agent_run"}
    with pytest.raises(ValidationError):
        GrowthAnalysisFailure.model_validate(
            result.model_dump() | {"detail": "secret prompt"}, strict=True
        )
    with pytest.raises(ValidationError):
        GrowthAnalysisFailure(
            code=GrowthAnalysisFailureCode.PROVIDER_FAILURE,
            agent_run=audit.model_copy(
                update={
                    "status": AgentRunStatus.SUCCEEDED,
                    "output_sha256": "c" * 64,
                }
            ),
        )
