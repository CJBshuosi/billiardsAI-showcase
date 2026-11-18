from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.provider_evaluation import (
    BenchmarkCase,
    BenchmarkHardGatePolicy,
    EligibilityStatus,
    EvaluationFailureReason,
    FixtureClassification,
    FrozenBenchmarkManifest,
    InMemoryProviderEvaluationRepository,
    ProviderEvaluationReport,
    ProviderEvaluationRepositoryError,
    ProviderEvaluationRepositoryErrorCode,
    ProviderEvaluationResult,
    ProviderEvaluationRunner,
    revalidate_provider_allowlist_candidate,
)
from billiards_ai.shared.models.events import canonical_sha256
from tests.unit.cloud.coach_agent.test_provider_evaluation import (
    NOW,
    ApprovalAuthority,
    ScriptedProvider,
    command,
    manifest,
    output,
    receipt,
    response,
)


class CountingProvider(ScriptedProvider):
    @property
    def calls(self) -> int:
        return len(self.requests)


class SlowCountingProvider(CountingProvider):
    async def generate_structured(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        return await super().generate_structured(request)


def _manifest_with_input(payload: dict[str, object]) -> FrozenBenchmarkManifest:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    source = frozen.cases[0]
    case = BenchmarkCase(
        case_id=source.case_id,
        request=source.request.model_copy(update={"input_payload": payload}),
        reference=source.reference,
        estimated_input_tokens=source.estimated_input_tokens,
        estimated_output_tokens=source.estimated_output_tokens,
    )
    values = frozen.model_dump(
        mode="python", exclude={"content_sha256", "reference_set_sha256", "cases"}
    )
    return FrozenBenchmarkManifest.build(**values, cases=(case,))


@pytest.mark.asyncio
async def test_production_privacy_gate_deeply_rejects_sensitive_values_before_provider() -> None:
    frozen = _manifest_with_input(
        {
            "fixture_classification": "PRODUCTION_APPROVED_REFERENCE",
            "context": {"safe_label": "api_key=do-not-send-this"},
        }
    )
    provider = CountingProvider([response(output()), response(output())])

    with pytest.raises(ValueError, match="privacy") as error:
        await ProviderEvaluationRunner(provider=provider).evaluate(command(frozen))

    assert provider.calls == 0
    assert "do-not-send-this" not in str(error.value)


@pytest.mark.asyncio
async def test_latency_gate_uses_trusted_monotonic_measurement_not_provider_report() -> None:
    ticks: Iterator[int] = iter((0, 600_000_000, 700_000_000, 1_300_000_000))
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider(
            [response(output(), latency_ms=1), response(output(), latency_ms=1)]
        ),
        approval_authority=ApprovalAuthority(receipt(frozen)),
        monotonic_ns=lambda: next(ticks),
        utc_now=lambda: NOW,
    ).evaluate(command(frozen))

    assert result.report.metrics.p95_latency_ms == 600
    assert all(attempt.latency_ms == 600 for attempt in result.report.attempts)
    assert all(attempt.provider_reported_latency_ms == 1 for attempt in result.report.attempts)
    assert EvaluationFailureReason.LATENCY_GATE_FAILED in result.decision.failure_reasons


@pytest.mark.asyncio
async def test_repository_rejects_canonically_rehashed_truncated_attempt_graph() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    exact_command = command(frozen)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(receipt(frozen)),
        utc_now=lambda: NOW,
    ).evaluate(exact_command)
    report_values = result.report.model_dump(mode="python", exclude={"content_sha256"})
    forged_report_payload = report_values | {"attempts": result.report.attempts[:1]}
    forged_report = ProviderEvaluationReport.model_construct(
        **(forged_report_payload | {"content_sha256": canonical_sha256(forged_report_payload)})
    )
    result_payload = result.model_dump(mode="python", exclude={"content_sha256"}) | {
        "report": forged_report
    }
    forged = ProviderEvaluationResult.model_construct(
        **(result_payload | {"content_sha256": canonical_sha256(result_payload)})
    )

    with pytest.raises(ProviderEvaluationRepositoryError) as error:
        InMemoryProviderEvaluationRepository().save(exact_command, forged)
    assert error.value.code is ProviderEvaluationRepositoryErrorCode.INVALID_RECORD


@pytest.mark.asyncio
async def test_repository_returns_canonical_independent_copies() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    exact_command = command(frozen)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(receipt(frozen)),
        utc_now=lambda: NOW,
    ).evaluate(exact_command)
    repository = InMemoryProviderEvaluationRepository(utc_now=lambda: NOW)

    first = repository.save(exact_command, result)
    replay = repository.save(exact_command, result)
    first_list = repository.list_allowlist_candidates()
    second_list = repository.list_allowlist_candidates()

    assert first == replay
    assert first is not replay
    assert first.report is not replay.report
    assert first.report.attempts[0] is not replay.report.attempts[0]
    assert first_list == second_list
    assert first_list[0] is not second_list[0]


@pytest.mark.asyncio
async def test_shared_repository_reserves_before_provider_across_services() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    exact_command = command(frozen)
    provider = SlowCountingProvider([response(output()), response(output())])
    repository = InMemoryProviderEvaluationRepository(utc_now=lambda: NOW)
    first = ProviderEvaluationRunner(
        provider=provider,
        approval_authority=ApprovalAuthority(receipt(frozen)),
        repository=repository,
        utc_now=lambda: NOW,
    )
    second = ProviderEvaluationRunner(
        provider=provider,
        approval_authority=ApprovalAuthority(receipt(frozen)),
        repository=repository,
        utc_now=lambda: NOW,
    )

    left, right = await asyncio.gather(
        first.evaluate(exact_command), second.evaluate(exact_command)
    )

    assert left == right
    assert left is not right
    assert provider.calls == frozen.repetition_count * len(frozen.cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("claims", [(), ("CUT_ANGLE_NEEDS_WORK", "EXTRA")])
async def test_claim_codes_require_exact_nonempty_correctness_and_coverage(
    claims: tuple[str, ...],
) -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider(
            [response(output(claims=claims)), response(output(claims=claims))]
        ),
        approval_authority=ApprovalAuthority(receipt(frozen)),
        utc_now=lambda: NOW,
    ).evaluate(command(frozen))

    assert result.decision.status is EligibilityStatus.NOT_ELIGIBLE
    assert EvaluationFailureReason.CLAIM_GATE_FAILED in result.decision.failure_reasons


def test_stability_and_unsupported_policies_are_unconditionally_zero_tolerance() -> None:
    values = manifest(FixtureClassification.SYNTHETIC_TEST).hard_gates.model_dump(mode="python")
    with pytest.raises(ValidationError):
        BenchmarkHardGatePolicy.model_validate(
            values | {"minimum_stability_score_bps": 9_999}, strict=True
        )
    with pytest.raises(ValidationError):
        BenchmarkHardGatePolicy.model_validate(
            values | {"maximum_unsupported_violations": 1}, strict=True
        )


@pytest.mark.asyncio
async def test_trusted_current_time_controls_receipt_and_candidate_expiry_admission() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    approval = receipt(frozen)
    expired = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(approval),
        utc_now=lambda: approval.expires_at,
    ).evaluate(command(frozen))
    assert expired.decision.failure_reasons == (EvaluationFailureReason.APPROVAL_INVALID,)

    eligible = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(approval),
        utc_now=lambda: NOW,
    ).evaluate(command(frozen))
    candidate = eligible.allowlist_candidate
    assert candidate is not None
    assert candidate.reference_approval_expires_at == approval.expires_at
    assert revalidate_provider_allowlist_candidate(candidate, at=NOW) == candidate
    with pytest.raises(ValueError, match="expired"):
        revalidate_provider_allowlist_candidate(candidate, at=approval.expires_at)
