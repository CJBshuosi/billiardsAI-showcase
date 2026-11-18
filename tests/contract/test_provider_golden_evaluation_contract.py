from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.context import CoachTaskType
from billiards_ai.cloud.coach_agent.provider import CoachModelRequest
from billiards_ai.cloud.coach_agent.provider_evaluation import (
    BenchmarkCase,
    BenchmarkHardGatePolicy,
    BenchmarkReferenceLabel,
    CandidateIdentity,
    DrillReferenceLabel,
    FixtureClassification,
    FrozenBenchmarkManifest,
    NumericFactLabel,
    ProviderPricingPolicy,
    ReferenceApprovalReceipt,
    ReferenceApprovalStatus,
)
from billiards_ai.shared.models.events import canonical_sha256

NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)
FIXTURE = Path(
    "tests/fixtures/core_intelligence/v1/provider_golden_evaluation/synthetic-benchmark.json"
)


def request(classification: FixtureClassification) -> CoachModelRequest:
    return CoachModelRequest(
        task_type=CoachTaskType.GROWTH_ASSESSMENT,
        prompt_version="growth-analysis-v1",
        system_prompt="Synthetic golden evaluation prompt.",
        input_payload={"fixture_classification": classification.value, "context": "case-1"},
        output_schema={"type": "object"},
        request_id=UUID("10000000-0000-4000-8000-000000000001"),
        timeout_ms=5000,
    )


def reference() -> BenchmarkReferenceLabel:
    return BenchmarkReferenceLabel(
        expected_evidence_refs=("metric-shot-001",),
        expected_numeric_facts=(NumericFactLabel(fact_code="attempt_count", value=10),),
        expected_claim_codes=("CUT_ANGLE_NEEDS_WORK",),
        expected_drill_refs=(DrillReferenceLabel(drill_id="cut-angle-basic", drill_revision=1),),
    )


def pricing() -> ProviderPricingPolicy:
    return ProviderPricingPolicy(
        policy_version="pricing-v1",
        input_microunits_per_million_tokens=1000,
        output_microunits_per_million_tokens=2000,
    )


def manifest(*, classification: FixtureClassification) -> FrozenBenchmarkManifest:
    case = BenchmarkCase(
        case_id="case-1",
        request=request(classification),
        reference=reference(),
        estimated_input_tokens=100,
        estimated_output_tokens=50,
    )
    return FrozenBenchmarkManifest.build(
        benchmark_id="coach-golden-v1",
        benchmark_revision=1,
        fixture_classification=classification,
        prompt_version="growth-analysis-v1",
        prompt_sha256=canonical_sha256("Synthetic golden evaluation prompt."),
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


def test_manifest_is_versioned_hash_bound_and_strict() -> None:
    frozen = manifest(classification=FixtureClassification.SYNTHETIC_TEST)
    assert frozen.schema_version == "1.0"
    assert frozen.reference_set_sha256 == canonical_sha256(
        tuple(case.reference for case in frozen.cases)
    )
    assert frozen.content_sha256 == canonical_sha256(
        frozen.model_dump(mode="json", exclude={"content_sha256"})
    )
    with pytest.raises((TypeError, ValidationError, ValueError)):
        frozen.model_copy(update={"benchmark_revision": 2})
    with pytest.raises(ValidationError):
        FrozenBenchmarkManifest.model_validate(
            frozen.model_dump(mode="python") | {"unknown": True}, strict=True
        )


def test_receipt_is_exact_hash_bound_and_requires_verified_signature() -> None:
    frozen = manifest(classification=FixtureClassification.PRODUCTION_APPROVED_REFERENCE)
    payload = {
        "schema_version": "1.0",
        "receipt_id": UUID("20000000-0000-4000-8000-000000000001"),
        "benchmark_id": frozen.benchmark_id,
        "benchmark_revision": frozen.benchmark_revision,
        "benchmark_content_sha256": frozen.content_sha256,
        "reference_set_sha256": frozen.reference_set_sha256,
        "authority_id": "external-coach-reference-authority",
        "signature_key_id": "coach-reference-key-1",
        "signature_sha256": "1" * 64,
        "status": ReferenceApprovalStatus.VERIFIED,
        "approved_at": NOW,
        "expires_at": NOW + timedelta(days=30),
    }
    receipt = ReferenceApprovalReceipt.build(**payload)
    assert receipt.content_sha256 == canonical_sha256(
        receipt.model_dump(mode="json", exclude={"content_sha256"})
    )
    with pytest.raises(ValidationError):
        ReferenceApprovalReceipt.build(**(payload | {"status": "UNVERIFIED"}))


def test_candidate_identity_binds_exact_provider_model_prompt_and_benchmark() -> None:
    frozen = manifest(classification=FixtureClassification.SYNTHETIC_TEST)
    identity = CandidateIdentity.from_manifest(
        provider="provider-a", model="model-a", manifest=frozen, pricing=pricing()
    )
    assert identity.prompt_sha256 == frozen.prompt_sha256
    assert identity.benchmark_revision == 1
    assert identity.reference_set_sha256 == frozen.reference_set_sha256
    assert identity.drill_catalog_content_sha256 == frozen.drill_catalog_content_sha256
    assert identity.pricing_policy_sha256 == canonical_sha256(pricing())


def test_frozen_fixture_is_explicitly_synthetic_and_contains_no_user_or_secret_data() -> None:
    payload = json.loads(FIXTURE.read_text())
    assert payload["fixture_classification"] == "SYNTHETIC_TEST"
    assert "NOT HUMAN-COACH APPROVED" in payload["notice"]
    encoded = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        '"openid"',
        '"user_id"',
        '"token"',
        '"secret"',
        '"video_url"',
        '"media_url"',
    ):
        assert forbidden not in encoded


def test_manifest_cannot_reclassify_synthetic_cases_as_production_reference() -> None:
    synthetic = manifest(classification=FixtureClassification.SYNTHETIC_TEST)
    values = synthetic.model_dump(mode="python", exclude={"content_sha256"})
    with pytest.raises(ValidationError):
        FrozenBenchmarkManifest.build(
            **(
                values
                | {"fixture_classification": (FixtureClassification.PRODUCTION_APPROVED_REFERENCE)}
            )
        )
