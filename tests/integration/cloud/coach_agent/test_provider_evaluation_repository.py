from __future__ import annotations

import pytest

from billiards_ai.cloud.coach_agent.provider_evaluation import (
    FixtureClassification,
    InMemoryProviderEvaluationRepository,
    ProviderEvaluationRepositoryError,
    ProviderEvaluationRepositoryErrorCode,
    ProviderEvaluationResult,
)
from tests.unit.cloud.coach_agent.test_provider_evaluation import (
    NOW,
    ApprovalAuthority,
    ProviderEvaluationRunner,
    ScriptedProvider,
    command,
    manifest,
    output,
    receipt,
    response,
)


@pytest.mark.asyncio
async def test_repository_replays_exact_result_and_rejects_key_reuse() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(receipt(frozen)),
        utc_now=lambda: NOW,
    ).evaluate(command(frozen))
    repository = InMemoryProviderEvaluationRepository(utc_now=lambda: NOW)

    first = repository.save(command(frozen), result)
    replay = repository.save(command(frozen), result)
    assert replay == first
    assert repository.list_allowlist_candidates() == (result.allowlist_candidate,)

    conflicting = command(frozen).model_copy(
        update={"candidate": command(frozen).candidate.model_copy(update={"model": "other-model"})}
    )
    with pytest.raises(ProviderEvaluationRepositoryError) as error:
        repository.save(conflicting, result)
    assert error.value.code is ProviderEvaluationRepositoryErrorCode.IDEMPOTENCY_CONFLICT


def test_repository_rejects_tampered_report_or_decision() -> None:
    repository = InMemoryProviderEvaluationRepository()
    with pytest.raises(ProviderEvaluationRepositoryError) as error:
        repository.save("not-a-command", "not-a-result")  # type: ignore[arg-type]
    assert error.value.code is ProviderEvaluationRepositoryErrorCode.INVALID_RECORD


@pytest.mark.asyncio
async def test_repository_deeply_rejects_a_hash_tampered_result() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    command_value = command(frozen)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(receipt(frozen)),
    ).evaluate(command_value)
    tampered = ProviderEvaluationResult.model_construct(
        **(result.model_dump(mode="python") | {"content_sha256": "f" * 64})
    )
    with pytest.raises(ProviderEvaluationRepositoryError) as error:
        InMemoryProviderEvaluationRepository().save(command_value, tampered)
    assert error.value.code is ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
