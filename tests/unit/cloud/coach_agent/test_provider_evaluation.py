from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from billiards_ai.cloud.coach_agent.context import CoachTaskType
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
    CoachModelRequest,
    CoachModelResponse,
)
from billiards_ai.cloud.coach_agent.provider_evaluation import (
    BenchmarkCase,
    BenchmarkHardGatePolicy,
    BenchmarkReferenceLabel,
    CandidateIdentity,
    DrillReferenceLabel,
    EligibilityStatus,
    EvaluationFailureReason,
    FixtureClassification,
    FrozenBenchmarkManifest,
    NumericFactLabel,
    ProviderEvaluationCommand,
    ProviderEvaluationRunner,
    ProviderPricingPolicy,
    ReferenceApprovalReceipt,
    ReferenceApprovalStatus,
)
from billiards_ai.shared.models.events import canonical_sha256

NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)


class ScriptedProvider:
    def __init__(self, results: list[Any]) -> None:
        self.results = deque(results)
        self.requests: list[CoachModelRequest] = []

    async def generate_structured(self, request: CoachModelRequest):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        result = self.results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result


class ApprovalAuthority:
    def __init__(self, receipt: ReferenceApprovalReceipt | None) -> None:
        self.receipt = receipt
        self.calls = 0

    def verify_reference_set(self, manifest: FrozenBenchmarkManifest, *, at: datetime):
        self.calls += 1
        return self.receipt


class HangingProvider:
    async def generate_structured(self, request: CoachModelRequest):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1)
        raise AssertionError("timeout did not cancel the provider")


def output(
    *,
    claims: tuple[str, ...] = ("CUT_ANGLE_NEEDS_WORK",),
    value: int = 10,
    evidence_refs: tuple[str, ...] = ("metric-shot-001",),
    drill_id: str = "cut-angle-basic",
):
    return {
        "schema_version": "1.0",
        "evidence_refs": list(evidence_refs),
        "numeric_facts": [{"fact_code": "attempt_count", "value": value}],
        "claim_codes": list(claims),
        "drill_refs": [{"drill_id": drill_id, "drill_revision": 1}],
    }


def response(
    structured_output: dict[str, object],
    *,
    provider: str = "provider-a",
    model: str = "model-a",
    latency_ms: int = 120,
    include_tokens: bool = True,
) -> CoachModelResponse:
    return CoachModelResponse(
        provider=provider,
        model=model,
        provider_request_id="synthetic-request",
        structured_output=structured_output,
        finish_reason="stop",
        input_tokens=100 if include_tokens else None,
        output_tokens=50 if include_tokens else None,
        latency_ms=latency_ms,
    )


def pricing() -> ProviderPricingPolicy:
    return ProviderPricingPolicy(
        policy_version="pricing-v1",
        input_microunits_per_million_tokens=1000,
        output_microunits_per_million_tokens=2000,
    )


def manifest(
    classification: FixtureClassification, *, timeout_ms: int = 5000
) -> FrozenBenchmarkManifest:
    prompt = "Synthetic golden evaluation prompt."
    request = CoachModelRequest(
        task_type=CoachTaskType.GROWTH_ASSESSMENT,
        prompt_version="growth-analysis-v1",
        system_prompt=prompt,
        input_payload={"fixture_classification": classification.value, "context": "case-1"},
        output_schema={"type": "object"},
        request_id=UUID("10000000-0000-4000-8000-000000000001"),
        timeout_ms=timeout_ms,
    )
    case = BenchmarkCase(
        case_id="case-1",
        request=request,
        reference=BenchmarkReferenceLabel(
            expected_evidence_refs=("metric-shot-001",),
            expected_numeric_facts=(NumericFactLabel(fact_code="attempt_count", value=10),),
            expected_claim_codes=("CUT_ANGLE_NEEDS_WORK",),
            expected_drill_refs=(
                DrillReferenceLabel(drill_id="cut-angle-basic", drill_revision=1),
            ),
        ),
        estimated_input_tokens=100,
        estimated_output_tokens=50,
    )
    return FrozenBenchmarkManifest.build(
        benchmark_id="coach-golden-v1",
        benchmark_revision=1,
        fixture_classification=classification,
        prompt_version="growth-analysis-v1",
        prompt_sha256=canonical_sha256(prompt),
        drill_catalog_content_sha256=canonical_sha256({"catalog": "synthetic-v1"}),
        reference_set_revision=1,
        repetition_count=2,
        cases=(case,),
        hard_gates=BenchmarkHardGatePolicy(
            policy_version="provider-hard-gates-v1",
            minimum_evidence_score_bps=10_000,
            minimum_numeric_score_bps=10_000,
            minimum_drill_score_bps=10_000,
            maximum_unsupported_violations=0,
            minimum_stability_score_bps=10_000,
            maximum_observed_cost_microunits=100,
            maximum_estimated_cost_microunits=100,
            maximum_p95_latency_ms=500,
        ),
    )


def receipt(frozen: FrozenBenchmarkManifest) -> ReferenceApprovalReceipt:
    return ReferenceApprovalReceipt.build(
        receipt_id=UUID("20000000-0000-4000-8000-000000000001"),
        benchmark_id=frozen.benchmark_id,
        benchmark_revision=frozen.benchmark_revision,
        benchmark_content_sha256=frozen.content_sha256,
        reference_set_sha256=frozen.reference_set_sha256,
        authority_id="external-reference-authority",
        signature_key_id="reference-key-1",
        signature_sha256="1" * 64,
        status=ReferenceApprovalStatus.VERIFIED,
        approved_at=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=30),
    )


def command(frozen: FrozenBenchmarkManifest) -> ProviderEvaluationCommand:
    return ProviderEvaluationCommand(
        manifest=frozen,
        pricing=pricing(),
        candidate=CandidateIdentity.from_manifest(
            provider="provider-a",
            model="model-a",
            manifest=frozen,
            pricing=pricing(),
        ),
        evaluated_at=NOW,
        idempotency_key="eval-1",
    )


@pytest.mark.asyncio
async def test_synthetic_quality_pass_is_never_production_eligible() -> None:
    frozen = manifest(FixtureClassification.SYNTHETIC_TEST)
    authority = ApprovalAuthority(receipt(frozen))
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=authority,
    ).evaluate(command(frozen))

    assert result.report.all_quality_gates_passed is True
    assert result.decision.status is EligibilityStatus.NOT_ELIGIBLE
    assert result.decision.failure_reasons == (EvaluationFailureReason.SYNTHETIC_ONLY,)
    assert result.allowlist_candidate is None
    assert authority.calls == 0
    assert len(result.report.attempts) == 2
    assert all(attempt.agent_run.output_sha256 for attempt in result.report.attempts)


@pytest.mark.asyncio
async def test_exact_externally_approved_production_candidate_can_be_eligible() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    approval = receipt(frozen)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(approval),
        utc_now=lambda: NOW,
    ).evaluate(command(frozen))

    assert result.decision.status is EligibilityStatus.ELIGIBLE
    assert result.decision.approval_receipt == approval
    assert result.allowlist_candidate is not None
    assert result.allowlist_candidate.identity == command(frozen).candidate


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("results", "reason"),
    [
        (
            [response(output(value=11)), response(output(value=11))],
            EvaluationFailureReason.NUMERIC_GATE_FAILED,
        ),
        (
            [
                response(output(evidence_refs=("wrong-ref",))),
                response(output(evidence_refs=("wrong-ref",))),
            ],
            EvaluationFailureReason.EVIDENCE_GATE_FAILED,
        ),
        (
            [
                response(output(drill_id="unknown-drill")),
                response(output(drill_id="unknown-drill")),
            ],
            EvaluationFailureReason.DRILL_GATE_FAILED,
        ),
        (
            [
                response(output(claims=("UNSUPPORTED_CLAIM",))),
                response(output(claims=("UNSUPPORTED_CLAIM",))),
            ],
            EvaluationFailureReason.UNSUPPORTED_GATE_FAILED,
        ),
        (
            [response(output()), response(output(value=11))],
            EvaluationFailureReason.STABILITY_GATE_FAILED,
        ),
        (
            [
                response(output(), include_tokens=False),
                response(output(), include_tokens=False),
            ],
            EvaluationFailureReason.COST_GATE_FAILED,
        ),
        (
            [
                response(output(), provider="provider-drift"),
                response(output(), provider="provider-drift"),
            ],
            EvaluationFailureReason.IDENTITY_GATE_FAILED,
        ),
        (
            [
                CoachModelFailure(
                    provider="provider-a",
                    model="model-a",
                    code=CoachModelFailureCode.TIMEOUT,
                    retryable=True,
                    latency_ms=5000,
                )
            ]
            * 2,
            EvaluationFailureReason.INVOCATION_GATE_FAILED,
        ),
    ],
)
async def test_each_hard_gate_fails_closed(
    results: list[object], reason: EvaluationFailureReason
) -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    result = await ProviderEvaluationRunner(
        provider=ScriptedProvider(results), approval_authority=ApprovalAuthority(receipt(frozen))
    ).evaluate(command(frozen))
    assert result.decision.status is EligibilityStatus.NOT_ELIGIBLE
    assert reason in result.decision.failure_reasons
    assert result.allowlist_candidate is None


@pytest.mark.asyncio
async def test_missing_or_mismatched_reference_authority_fails_closed() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    missing = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=None,
    ).evaluate(command(frozen))
    assert missing.decision.failure_reasons == (EvaluationFailureReason.APPROVAL_REQUIRED,)

    valid_receipt = receipt(frozen)
    wrong = ReferenceApprovalReceipt.build(
        **(
            valid_receipt.model_dump(mode="python", exclude={"content_sha256"})
            | {"reference_set_sha256": "f" * 64}
        )
    )
    rejected = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())]),
        approval_authority=ApprovalAuthority(wrong),
    ).evaluate(command(frozen))
    assert rejected.decision.status is EligibilityStatus.NOT_ELIGIBLE
    assert EvaluationFailureReason.APPROVAL_INVALID in rejected.decision.failure_reasons


@pytest.mark.asyncio
async def test_provider_timeout_is_bounded_and_fails_closed() -> None:
    frozen = manifest(FixtureClassification.PRODUCTION_APPROVED_REFERENCE, timeout_ms=1)
    result = await ProviderEvaluationRunner(provider=HangingProvider()).evaluate(command(frozen))
    assert result.decision.status is EligibilityStatus.NOT_ELIGIBLE
    assert EvaluationFailureReason.INVOCATION_GATE_FAILED in result.decision.failure_reasons
    assert all(attempt.latency_ms is not None for attempt in result.report.attempts)


@pytest.mark.asyncio
async def test_evaluation_is_provider_neutral_for_same_benchmark() -> None:
    frozen = manifest(FixtureClassification.SYNTHETIC_TEST)
    first = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output()), response(output())])
    ).evaluate(command(frozen))
    other_command = command(frozen).model_copy(
        update={
            "candidate": CandidateIdentity.from_manifest(
                provider="provider-b",
                model="model-b",
                manifest=frozen,
                pricing=pricing(),
            )
        }
    )
    second = await ProviderEvaluationRunner(
        provider=ScriptedProvider([response(output(), provider="provider-b", model="model-b")] * 2)
    ).evaluate(other_command)
    assert first.report.metrics == second.report.metrics
