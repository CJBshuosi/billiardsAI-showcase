"""Offline, provider-neutral golden evaluation and fail-closed eligibility gates."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Any, Literal, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.coach_agent.provider import (
    CoachModelProvider,
    CoachModelRequest,
    CoachModelResponse,
    revalidate_coach_model_request,
    revalidate_coach_model_result,
)
from billiards_ai.shared.models.coaching import AgentRun, AgentRunStatus
from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
)


def _stored_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _require_exact_utc(value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError("datetime must be an exact UTC value")
    return value


def _uuid4_from_payload(value: object) -> UUID:
    raw = bytearray(bytes.fromhex(canonical_sha256(value))[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _hash_payload(model: BaseModel) -> str:
    return canonical_sha256(model.model_dump(mode="json", exclude={"content_sha256"}))


def _canonical_copy[T: BaseModel](model: T, model_type: type[T]) -> T:
    """Rebuild a wholly independent model graph from canonical primitives."""

    return model_type.model_validate(
        model.model_dump(mode="python", round_trip=True, warnings="none"), strict=True
    )


_PRIVATE_KEY = re.compile(
    r"(?i)(?:^|_)(?:user_?id|subject_?id|openid|open_?id|name|full_?name|"
    r"phone|mobile|email|jwt|token|secret|password|passwd|api_?key|credential|"
    r"authorization|cookie|media|video|audio|image|frame|raw|trajectory|"
    r"path|url|uri|locator|storage)(?:$|_)"
)
_PRIVATE_VALUE = re.compile(
    r"(?i)(?:https?://|s3://|gs://|file://|(?:^|\s)(?:/|\.\./)|"
    r"\b(?:api[_ -]?key|client[_ -]?secret|password|passwd|openid|"
    r"authorization|bearer|credential|session[_ -]?cookie)\s*[:=]|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"\b1[3-9][0-9]{9}\b)"
)


def _assert_privacy_minimized(value: object, active: set[int] | None = None) -> None:
    """Reject identity, media-location and credential material without echoing it."""

    if isinstance(value, str):
        if _PRIVATE_VALUE.search(value):
            raise ValueError("provider evaluation request violates privacy gate")
        return
    if value is None or type(value) in {bool, int, float}:
        return
    if not isinstance(value, (dict, list)):
        raise ValueError("provider evaluation request violates privacy gate")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("provider evaluation request violates privacy gate")
    active.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str or _PRIVATE_KEY.search(key):
                    raise ValueError("provider evaluation request violates privacy gate")
                _assert_privacy_minimized(item, active)
        else:
            for item in value:
                _assert_privacy_minimized(item, active)
    finally:
        active.remove(identity)


def _provider_ready_request(request: object, *, production: bool) -> CoachModelRequest:
    """Deeply rebuild the exact primitive request immediately before invocation."""

    try:
        exact = revalidate_coach_model_request(request)
        primitives = json.loads(exact.model_dump_json())
        rebuilt = CoachModelRequest.model_validate_json(
            json.dumps(primitives, ensure_ascii=False, separators=(",", ":")),
            strict=True,
        )
        if production:
            _assert_privacy_minimized(
                {
                    "system_prompt": rebuilt.system_prompt,
                    "input_payload": rebuilt.input_payload,
                    "output_schema": rebuilt.output_schema,
                }
            )
        return rebuilt
    except ValueError as error:
        if str(error) == "provider evaluation request violates privacy gate":
            raise
        raise ValueError("invalid provider evaluation request") from None
    except (AttributeError, RecursionError, TypeError, ValidationError):
        raise ValueError("invalid provider evaluation request") from None


def _trusted_utc_now(clock: Callable[[], datetime]) -> datetime:
    try:
        return _require_exact_utc(clock())
    except Exception:
        raise ValueError("trusted UTC clock failed") from None


def _trusted_monotonic_ns(clock: Callable[[], int]) -> int:
    try:
        value = clock()
    except Exception:
        raise ValueError("trusted monotonic clock failed") from None
    if type(value) is not int or value < 0:
        raise ValueError("trusted monotonic clock failed")
    return value


def _microunits(tokens: int, rate: int) -> int:
    return (tokens * rate + 999_999) // 1_000_000


class EvaluationContractModel(CoreContractModel):
    """Strict/frozen model whose copy path cannot bypass nested validation."""

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_values(copied), strict=True)


class FixtureClassification(StrEnum):
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    PRODUCTION_APPROVED_REFERENCE = "PRODUCTION_APPROVED_REFERENCE"


class NumericFactLabel(EvaluationContractModel):
    fact_code: NonEmptyStr
    value: StrictInt


class DrillReferenceLabel(EvaluationContractModel):
    drill_id: NonEmptyStr
    drill_revision: StrictInt = Field(ge=1)


class BenchmarkReferenceLabel(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    expected_evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    expected_numeric_facts: tuple[NumericFactLabel, ...] = Field(min_length=1)
    expected_claim_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    expected_drill_refs: tuple[DrillReferenceLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def values_are_canonical(self) -> Self:
        if self.expected_evidence_refs != tuple(sorted(set(self.expected_evidence_refs))):
            raise ValueError("evidence references must be sorted and unique")
        if self.expected_claim_codes != tuple(sorted(set(self.expected_claim_codes))):
            raise ValueError("expected claim codes must be sorted and unique")
        numeric_keys = tuple(item.fact_code for item in self.expected_numeric_facts)
        if numeric_keys != tuple(sorted(set(numeric_keys))):
            raise ValueError("numeric facts must use sorted unique codes")
        drill_keys = tuple(
            (item.drill_id, item.drill_revision) for item in self.expected_drill_refs
        )
        if drill_keys != tuple(sorted(set(drill_keys))):
            raise ValueError("drill references must be sorted and unique")
        return self


class BenchmarkCase(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: NonEmptyStr
    request: CoachModelRequest
    reference: BenchmarkReferenceLabel
    estimated_input_tokens: StrictInt = Field(ge=0)
    estimated_output_tokens: StrictInt = Field(ge=0)


class ProviderPricingPolicy(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    policy_version: NonEmptyStr
    input_microunits_per_million_tokens: StrictInt = Field(ge=0)
    output_microunits_per_million_tokens: StrictInt = Field(ge=0)


class BenchmarkHardGatePolicy(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    policy_version: NonEmptyStr
    minimum_evidence_score_bps: StrictInt = Field(ge=0, le=10_000)
    minimum_numeric_score_bps: StrictInt = Field(ge=0, le=10_000)
    minimum_drill_score_bps: StrictInt = Field(ge=0, le=10_000)
    maximum_unsupported_violations: Literal[0] = 0
    minimum_stability_score_bps: Literal[10_000] = 10_000
    maximum_observed_cost_microunits: StrictInt = Field(ge=0)
    maximum_estimated_cost_microunits: StrictInt = Field(ge=0)
    maximum_p95_latency_ms: StrictInt = Field(ge=0)


class FrozenBenchmarkManifest(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: NonEmptyStr
    benchmark_revision: StrictInt = Field(ge=1)
    fixture_classification: FixtureClassification
    prompt_version: NonEmptyStr
    prompt_sha256: Sha256
    drill_catalog_content_sha256: Sha256
    reference_set_revision: StrictInt = Field(ge=1)
    reference_set_sha256: Sha256
    repetition_count: StrictInt = Field(ge=2, le=20)
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)
    hard_gates: BenchmarkHardGatePolicy
    content_sha256: Sha256

    @classmethod
    def build(cls, **values: object) -> FrozenBenchmarkManifest:
        payload = dict(values)
        payload.pop("content_sha256", None)
        payload.setdefault("schema_version", "1.0")
        cases = tuple(
            BenchmarkCase.model_validate(
                _stored_values(case) if isinstance(case, BaseModel) else case,
                strict=True,
            )
            for case in tuple(payload.get("cases", ()))
        )
        payload["cases"] = cases
        payload["reference_set_sha256"] = canonical_sha256(tuple(case.reference for case in cases))
        return cls.model_validate(
            payload | {"content_sha256": canonical_sha256(payload)}, strict=True
        )

    @model_validator(mode="after")
    def manifest_is_canonical_and_hash_bound(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("benchmark cases must be sorted and unique")
        if any(
            item.request.prompt_version != self.prompt_version
            or canonical_sha256(item.request.system_prompt) != self.prompt_sha256
            for item in self.cases
        ):
            raise ValueError("every case must bind the exact frozen prompt")
        if any(
            item.request.input_payload.get("fixture_classification")
            != self.fixture_classification.value
            for item in self.cases
        ):
            raise ValueError("every benchmark case must match the manifest classification")
        expected_reference = canonical_sha256(tuple(item.reference for item in self.cases))
        if self.reference_set_sha256 != expected_reference:
            raise ValueError("reference set hash mismatch")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("benchmark content hash mismatch")
        return self


class CandidateIdentity(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    provider: NonEmptyStr
    model: NonEmptyStr
    prompt_version: NonEmptyStr
    prompt_sha256: Sha256
    benchmark_id: NonEmptyStr
    benchmark_revision: StrictInt = Field(ge=1)
    benchmark_content_sha256: Sha256
    reference_set_sha256: Sha256
    reference_set_revision: StrictInt = Field(ge=1)
    drill_catalog_content_sha256: Sha256
    hard_gate_policy_version: NonEmptyStr
    pricing_policy_version: NonEmptyStr
    pricing_policy_sha256: Sha256

    @classmethod
    def from_manifest(
        cls,
        *,
        provider: str,
        model: str,
        manifest: FrozenBenchmarkManifest,
        pricing: ProviderPricingPolicy,
    ) -> CandidateIdentity:
        exact = FrozenBenchmarkManifest.model_validate(_stored_values(manifest), strict=True)
        exact_pricing = ProviderPricingPolicy.model_validate(_stored_values(pricing), strict=True)
        return cls(
            provider=provider,
            model=model,
            prompt_version=exact.prompt_version,
            prompt_sha256=exact.prompt_sha256,
            benchmark_id=exact.benchmark_id,
            benchmark_revision=exact.benchmark_revision,
            benchmark_content_sha256=exact.content_sha256,
            reference_set_sha256=exact.reference_set_sha256,
            reference_set_revision=exact.reference_set_revision,
            drill_catalog_content_sha256=exact.drill_catalog_content_sha256,
            hard_gate_policy_version=exact.hard_gates.policy_version,
            pricing_policy_version=exact_pricing.policy_version,
            pricing_policy_sha256=canonical_sha256(exact_pricing),
        )


class ProviderEvaluationOutput(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_refs: tuple[NonEmptyStr, ...]
    numeric_facts: tuple[NumericFactLabel, ...]
    claim_codes: tuple[NonEmptyStr, ...]
    drill_refs: tuple[DrillReferenceLabel, ...]

    @model_validator(mode="after")
    def output_is_canonical(self) -> Self:
        if self.evidence_refs != tuple(sorted(set(self.evidence_refs))):
            raise ValueError("evidence references must be sorted and unique")
        if self.claim_codes != tuple(sorted(set(self.claim_codes))):
            raise ValueError("claim codes must be sorted and unique")
        numeric = tuple(item.fact_code for item in self.numeric_facts)
        drills = tuple((item.drill_id, item.drill_revision) for item in self.drill_refs)
        if numeric != tuple(sorted(set(numeric))) or drills != tuple(sorted(set(drills))):
            raise ValueError("numeric and drill references must be canonical")
        return self


class ReferenceApprovalStatus(StrEnum):
    VERIFIED = "VERIFIED"


class ReferenceApprovalReceipt(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    receipt_id: UUID4
    benchmark_id: NonEmptyStr
    benchmark_revision: StrictInt = Field(ge=1)
    benchmark_content_sha256: Sha256
    reference_set_sha256: Sha256
    authority_id: NonEmptyStr
    signature_key_id: NonEmptyStr
    signature_sha256: Sha256
    status: Literal[ReferenceApprovalStatus.VERIFIED] = ReferenceApprovalStatus.VERIFIED
    approved_at: datetime
    expires_at: datetime
    content_sha256: Sha256

    _approved_utc = field_validator("approved_at", mode="before")(_require_exact_utc)
    _expires_utc = field_validator("expires_at", mode="before")(_require_exact_utc)

    @classmethod
    def build(cls, **values: object) -> ReferenceApprovalReceipt:
        payload = dict(values)
        payload.pop("content_sha256", None)
        payload.setdefault("schema_version", "1.0")
        payload.setdefault("status", ReferenceApprovalStatus.VERIFIED)
        return cls.model_validate(
            payload | {"content_sha256": canonical_sha256(payload)}, strict=True
        )

    @model_validator(mode="after")
    def receipt_is_currently_well_formed_and_hash_bound(self) -> Self:
        if self.expires_at <= self.approved_at:
            raise ValueError("reference approval must have a positive validity window")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("reference approval content hash mismatch")
        return self


@runtime_checkable
class ReferenceApprovalPort(Protocol):
    """Externally trusted verification seam; implementations verify approval/signature."""

    def verify_reference_set(
        self, manifest: FrozenBenchmarkManifest, *, at: datetime
    ) -> ReferenceApprovalReceipt | None: ...


class EvaluationAttemptStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class EvaluationAttempt(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    attempt_id: UUID4
    case_id: NonEmptyStr
    repetition: StrictInt = Field(ge=1)
    identity: CandidateIdentity
    request_sha256: Sha256
    output_sha256: Sha256 | None = None
    observed_provider: NonEmptyStr
    observed_model: NonEmptyStr
    status: EvaluationAttemptStatus
    invocation_succeeded: StrictBool
    identity_matched: StrictBool
    evidence_correct: StrictBool
    numeric_correct: StrictBool
    claims_correct: StrictBool
    drill_compliant: StrictBool
    unsupported_violation_count: StrictInt = Field(ge=0)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    observed_cost_microunits: StrictInt | None = Field(default=None, ge=0)
    latency_ms: StrictInt | None = Field(default=None, ge=0)
    provider_reported_latency_ms: StrictInt | None = Field(default=None, ge=0)
    agent_run: AgentRun
    content_sha256: Sha256

    @model_validator(mode="after")
    def attempt_is_audit_closed(self) -> Self:
        success = self.status is EvaluationAttemptStatus.SUCCEEDED
        if (
            success != self.invocation_succeeded
            or (self.output_sha256 is not None) != success
            or self.agent_run.run_id != self.attempt_id
            or self.agent_run.output_sha256 != self.output_sha256
            or (self.agent_run.status is AgentRunStatus.SUCCEEDED) != success
            or self.agent_run.provider_name != self.observed_provider
            or self.agent_run.provider_model != self.observed_model
            or self.agent_run.prompt_version != self.identity.prompt_version
            or self.agent_run.prompt_sha256 != self.identity.prompt_sha256
        ):
            raise ValueError("evaluation attempt lineage mismatch")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("evaluation attempt content hash mismatch")
        return self


class EvaluationMetrics(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    attempt_count: StrictInt = Field(ge=1)
    evidence_score_bps: StrictInt = Field(ge=0, le=10_000)
    numeric_score_bps: StrictInt = Field(ge=0, le=10_000)
    claim_score_bps: StrictInt = Field(ge=0, le=10_000)
    drill_score_bps: StrictInt = Field(ge=0, le=10_000)
    unsupported_violation_count: StrictInt = Field(ge=0)
    stability_score_bps: StrictInt = Field(ge=0, le=10_000)
    observed_cost_microunits: StrictInt | None = Field(default=None, ge=0)
    estimated_cost_microunits: StrictInt = Field(ge=0)
    p95_latency_ms: StrictInt | None = Field(default=None, ge=0)
    content_sha256: Sha256

    @model_validator(mode="after")
    def metrics_are_hash_bound(self) -> Self:
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("evaluation metrics content hash mismatch")
        return self


class EvaluationFailureReason(StrEnum):
    INVOCATION_GATE_FAILED = "INVOCATION_GATE_FAILED"
    IDENTITY_GATE_FAILED = "IDENTITY_GATE_FAILED"
    EVIDENCE_GATE_FAILED = "EVIDENCE_GATE_FAILED"
    NUMERIC_GATE_FAILED = "NUMERIC_GATE_FAILED"
    CLAIM_GATE_FAILED = "CLAIM_GATE_FAILED"
    DRILL_GATE_FAILED = "DRILL_GATE_FAILED"
    UNSUPPORTED_GATE_FAILED = "UNSUPPORTED_GATE_FAILED"
    STABILITY_GATE_FAILED = "STABILITY_GATE_FAILED"
    COST_GATE_FAILED = "COST_GATE_FAILED"
    LATENCY_GATE_FAILED = "LATENCY_GATE_FAILED"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_INVALID = "APPROVAL_INVALID"


class ProviderEvaluationReport(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluation_id: UUID4
    identity: CandidateIdentity
    fixture_classification: FixtureClassification
    manifest_content_sha256: Sha256
    attempts: tuple[EvaluationAttempt, ...] = Field(min_length=1)
    metrics: EvaluationMetrics
    quality_failure_reasons: tuple[EvaluationFailureReason, ...]
    all_quality_gates_passed: StrictBool
    evaluated_at: datetime
    content_sha256: Sha256

    _evaluated_utc = field_validator("evaluated_at", mode="before")(_require_exact_utc)

    @model_validator(mode="after")
    def report_is_canonical_and_hash_bound(self) -> Self:
        if self.all_quality_gates_passed == bool(self.quality_failure_reasons):
            raise ValueError("quality gate summary is inconsistent")
        attempt_ids = tuple(item.attempt_id for item in self.attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("attempt IDs must be unique")
        if any(item.identity != self.identity for item in self.attempts):
            raise ValueError("attempt identities must match report identity")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("evaluation report content hash mismatch")
        return self


class EligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class EligibilityDecision(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    decision_id: UUID4
    identity: CandidateIdentity
    report_content_sha256: Sha256
    status: EligibilityStatus
    failure_reasons: tuple[EvaluationFailureReason, ...]
    approval_receipt: ReferenceApprovalReceipt | None = None
    decided_at: datetime
    content_sha256: Sha256

    _decided_utc = field_validator("decided_at", mode="before")(_require_exact_utc)

    @model_validator(mode="after")
    def decision_is_closed_and_hash_bound(self) -> Self:
        if self.status is EligibilityStatus.ELIGIBLE:
            if self.failure_reasons or self.approval_receipt is None:
                raise ValueError("eligible decision requires approval and no failures")
        elif not self.failure_reasons or self.approval_receipt is not None:
            raise ValueError("not-eligible decision requires failures and no approval")
        if self.failure_reasons != tuple(dict.fromkeys(self.failure_reasons)):
            raise ValueError("failure reasons must be canonical and unique")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("eligibility decision content hash mismatch")
        return self


class ProviderAllowlistCandidate(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: UUID4
    identity: CandidateIdentity
    decision_content_sha256: Sha256
    reference_approval_content_sha256: Sha256
    reference_approval_expires_at: datetime
    created_at: datetime
    content_sha256: Sha256

    _created_utc = field_validator("created_at", mode="before")(_require_exact_utc)
    _expires_utc = field_validator("reference_approval_expires_at", mode="before")(
        _require_exact_utc
    )

    @model_validator(mode="after")
    def candidate_is_hash_bound(self) -> Self:
        if self.reference_approval_expires_at <= self.created_at:
            raise ValueError("allowlist candidate approval is expired")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("allowlist candidate content hash mismatch")
        return self


def revalidate_provider_allowlist_candidate(
    candidate: object, *, at: datetime
) -> ProviderAllowlistCandidate:
    """Admission boundary that fails closed once reference approval expires."""

    try:
        exact_at = _require_exact_utc(at)
        if not isinstance(candidate, ProviderAllowlistCandidate):
            raise ValueError
        exact = _canonical_copy(candidate, ProviderAllowlistCandidate)
        if exact_at >= exact.reference_approval_expires_at:
            raise ValueError("allowlist candidate approval is expired")
        return exact
    except ValueError as error:
        if str(error) == "allowlist candidate approval is expired":
            raise
        raise ValueError("invalid allowlist candidate") from None
    except (AttributeError, RecursionError, TypeError, ValidationError):
        raise ValueError("invalid allowlist candidate") from None


class ProviderEvaluationResult(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    report: ProviderEvaluationReport
    decision: EligibilityDecision
    allowlist_candidate: ProviderAllowlistCandidate | None = None
    content_sha256: Sha256

    @model_validator(mode="after")
    def result_is_closed_and_hash_bound(self) -> Self:
        if (
            self.decision.identity != self.report.identity
            or self.decision.report_content_sha256 != self.report.content_sha256
            or (self.decision.status is EligibilityStatus.ELIGIBLE)
            != (self.allowlist_candidate is not None)
        ):
            raise ValueError("evaluation result lineage mismatch")
        if self.allowlist_candidate is not None and (
            self.allowlist_candidate.identity != self.decision.identity
            or self.allowlist_candidate.decision_content_sha256 != self.decision.content_sha256
            or self.decision.approval_receipt is None
            or self.allowlist_candidate.reference_approval_content_sha256
            != self.decision.approval_receipt.content_sha256
            or self.allowlist_candidate.reference_approval_expires_at
            != self.decision.approval_receipt.expires_at
        ):
            raise ValueError("allowlist candidate lineage mismatch")
        if self.content_sha256 != _hash_payload(self):
            raise ValueError("evaluation result content hash mismatch")
        return self


class ProviderEvaluationCommand(EvaluationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    manifest: FrozenBenchmarkManifest
    pricing: ProviderPricingPolicy
    candidate: CandidateIdentity
    evaluated_at: datetime
    idempotency_key: NonEmptyStr

    _evaluated_utc = field_validator("evaluated_at", mode="before")(_require_exact_utc)

    @model_validator(mode="after")
    def candidate_matches_manifest(self) -> Self:
        expected = CandidateIdentity.from_manifest(
            provider=self.candidate.provider,
            model=self.candidate.model,
            manifest=self.manifest,
            pricing=self.pricing,
        )
        if self.candidate != expected:
            raise ValueError("candidate does not bind the exact benchmark")
        return self


def _hashed_model[T: EvaluationContractModel](model: type[T], payload: dict[str, object]) -> T:
    return model.model_validate(
        payload | {"content_sha256": canonical_sha256(payload)}, strict=True
    )


def _score(values: list[bool]) -> int:
    return sum(values) * 10_000 // len(values)


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[((95 * len(ordered) + 99) // 100) - 1]


def _reference_matches(
    output: ProviderEvaluationOutput, reference: BenchmarkReferenceLabel
) -> tuple[bool, bool, bool, bool, int]:
    evidence = output.evidence_refs == reference.expected_evidence_refs
    numeric = output.numeric_facts == reference.expected_numeric_facts
    claims = output.claim_codes == reference.expected_claim_codes
    drills = output.drill_refs == reference.expected_drill_refs
    unsupported = len(set(output.claim_codes) - set(reference.expected_claim_codes))
    return evidence, numeric, claims, drills, unsupported


class ProviderEvaluationRunner:
    def __init__(
        self,
        *,
        provider: CoachModelProvider,
        approval_authority: ReferenceApprovalPort | None = None,
        repository: InMemoryProviderEvaluationRepository | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not isinstance(provider, CoachModelProvider):
            raise TypeError("provider must implement CoachModelProvider")
        if approval_authority is not None and not isinstance(
            approval_authority, ReferenceApprovalPort
        ):
            raise TypeError("approval authority must implement ReferenceApprovalPort")
        self._provider = provider
        self._approval_authority = approval_authority
        if repository is not None and not isinstance(
            repository, InMemoryProviderEvaluationRepository
        ):
            raise TypeError("repository must be InMemoryProviderEvaluationRepository")
        if not callable(monotonic_ns) or not callable(utc_now):
            raise TypeError("trusted clocks must be callable")
        self._monotonic_ns = monotonic_ns
        self._utc_now = utc_now
        self._repository = repository or InMemoryProviderEvaluationRepository(utc_now=utc_now)

    async def evaluate(self, command: object) -> ProviderEvaluationResult:
        try:
            if not isinstance(command, ProviderEvaluationCommand):
                raise ValueError
            exact = ProviderEvaluationCommand.model_validate(_stored_values(command), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ValueError("invalid provider evaluation command") from None

        return await self._repository.run_once(exact, self._evaluate_exact)

    async def _evaluate_exact(self, exact: ProviderEvaluationCommand) -> ProviderEvaluationResult:

        manifest = exact.manifest
        pricing = exact.pricing
        identity = exact.candidate
        attempts: list[EvaluationAttempt] = []
        for case in manifest.cases:
            request = _provider_ready_request(
                case.request,
                production=(
                    manifest.fixture_classification
                    is FixtureClassification.PRODUCTION_APPROVED_REFERENCE
                ),
            )
            request_sha256 = canonical_sha256(request)
            for repetition in range(1, manifest.repetition_count + 1):
                attempt_id = _uuid4_from_payload(
                    {
                        "identity": identity,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "evaluated_at": exact.evaluated_at,
                    }
                )
                response: CoachModelResponse | None = None
                latency_ms: int | None = None
                provider_reported_latency_ms: int | None = None
                observed_provider = identity.provider
                observed_model = identity.model
                started_ns = _trusted_monotonic_ns(self._monotonic_ns)
                try:
                    async with asyncio.timeout(request.timeout_ms / 1000):
                        raw = await self._provider.generate_structured(request)
                    result = revalidate_coach_model_result(raw)
                    provider_reported_latency_ms = result.latency_ms
                    observed_provider = result.provider
                    observed_model = result.model
                    if isinstance(result, CoachModelResponse):
                        response = result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    response = None
                finally:
                    completed_ns = _trusted_monotonic_ns(self._monotonic_ns)
                    if completed_ns < started_ns:
                        raise ValueError("trusted monotonic clock moved backwards")
                    latency_ms = (completed_ns - started_ns + 999_999) // 1_000_000

                matched_identity = bool(
                    response is not None
                    and response.provider == identity.provider
                    and response.model == identity.model
                )
                parsed: ProviderEvaluationOutput | None = None
                if matched_identity and response is not None:
                    try:
                        parsed = ProviderEvaluationOutput.model_validate_json(
                            json.dumps(
                                dict(response.structured_output),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            strict=True,
                        )
                    except (RecursionError, TypeError, ValidationError, ValueError):
                        parsed = None
                invocation_succeeded = parsed is not None
                output_sha256 = None if parsed is None else canonical_sha256(parsed)
                evidence, numeric, claims, drill, unsupported = (
                    (False, False, False, False, 1)
                    if parsed is None
                    else _reference_matches(parsed, case.reference)
                )
                input_tokens = None if response is None else response.input_tokens
                output_tokens = None if response is None else response.output_tokens
                observed_cost = (
                    None
                    if input_tokens is None or output_tokens is None
                    else _microunits(
                        input_tokens,
                        pricing.input_microunits_per_million_tokens,
                    )
                    + _microunits(
                        output_tokens,
                        pricing.output_microunits_per_million_tokens,
                    )
                )
                completed_at = exact.evaluated_at + timedelta(milliseconds=latency_ms or 0)
                agent_run = AgentRun(
                    run_id=attempt_id,
                    task_type=case.request.task_type.value,
                    prompt_version=identity.prompt_version,
                    prompt_sha256=identity.prompt_sha256,
                    context_sha256=canonical_sha256(case.request.input_payload),
                    provider_name=observed_provider,
                    provider_model=observed_model,
                    output_sha256=output_sha256,
                    status=(
                        AgentRunStatus.SUCCEEDED if invocation_succeeded else AgentRunStatus.FAILED
                    ),
                    repair_count=0,
                    started_at=exact.evaluated_at,
                    completed_at=completed_at,
                )
                attempt_payload: dict[str, object] = {
                    "schema_version": "1.0",
                    "attempt_id": attempt_id,
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "identity": identity,
                    "request_sha256": request_sha256,
                    "output_sha256": output_sha256,
                    "observed_provider": observed_provider,
                    "observed_model": observed_model,
                    "status": (
                        EvaluationAttemptStatus.SUCCEEDED
                        if invocation_succeeded
                        else EvaluationAttemptStatus.FAILED
                    ),
                    "invocation_succeeded": invocation_succeeded,
                    "identity_matched": matched_identity,
                    "evidence_correct": evidence,
                    "numeric_correct": numeric,
                    "claims_correct": claims,
                    "drill_compliant": drill,
                    "unsupported_violation_count": unsupported,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "observed_cost_microunits": observed_cost,
                    "latency_ms": latency_ms,
                    "provider_reported_latency_ms": provider_reported_latency_ms,
                    "agent_run": agent_run,
                }
                attempts.append(_hashed_model(EvaluationAttempt, attempt_payload))

        exact_attempts = tuple(attempts)
        metrics = _recalculate_metrics(exact, exact_attempts)
        quality_failures = list(_quality_failures(exact_attempts, metrics, manifest.hard_gates))

        evaluation_id = _uuid4_from_payload(
            {"identity": identity, "evaluated_at": exact.evaluated_at}
        )
        report_payload: dict[str, object] = {
            "schema_version": "1.0",
            "evaluation_id": evaluation_id,
            "identity": identity,
            "fixture_classification": manifest.fixture_classification,
            "manifest_content_sha256": manifest.content_sha256,
            "attempts": exact_attempts,
            "metrics": metrics,
            "quality_failure_reasons": tuple(quality_failures),
            "all_quality_gates_passed": not quality_failures,
            "evaluated_at": exact.evaluated_at,
        }
        report = _hashed_model(ProviderEvaluationReport, report_payload)

        failures = list(quality_failures)
        approval: ReferenceApprovalReceipt | None = None
        if not failures:
            if manifest.fixture_classification is FixtureClassification.SYNTHETIC_TEST:
                failures.append(EvaluationFailureReason.SYNTHETIC_ONLY)
            elif self._approval_authority is None:
                failures.append(EvaluationFailureReason.APPROVAL_REQUIRED)
            else:
                try:
                    trusted_now = _trusted_utc_now(self._utc_now)
                    raw_receipt = self._approval_authority.verify_reference_set(
                        manifest, at=trusted_now
                    )
                    if not isinstance(raw_receipt, ReferenceApprovalReceipt):
                        raise ValueError
                    candidate_receipt = ReferenceApprovalReceipt.model_validate(
                        _stored_values(raw_receipt), strict=True
                    )
                    if (
                        candidate_receipt.benchmark_id != manifest.benchmark_id
                        or candidate_receipt.benchmark_revision != manifest.benchmark_revision
                        or candidate_receipt.benchmark_content_sha256 != manifest.content_sha256
                        or candidate_receipt.reference_set_sha256 != manifest.reference_set_sha256
                        or not candidate_receipt.approved_at
                        <= trusted_now
                        < candidate_receipt.expires_at
                    ):
                        raise ValueError
                    approval = candidate_receipt
                except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
                    failures.append(EvaluationFailureReason.APPROVAL_INVALID)
                except Exception:
                    failures.append(EvaluationFailureReason.APPROVAL_INVALID)

        status = EligibilityStatus.ELIGIBLE if not failures else EligibilityStatus.NOT_ELIGIBLE
        decision_id = _uuid4_from_payload(
            {"evaluation_id": evaluation_id, "report": report.content_sha256, "status": status}
        )
        decision_payload: dict[str, object] = {
            "schema_version": "1.0",
            "decision_id": decision_id,
            "identity": identity,
            "report_content_sha256": report.content_sha256,
            "status": status,
            "failure_reasons": tuple(failures),
            "approval_receipt": approval,
            "decided_at": exact.evaluated_at,
        }
        decision = _hashed_model(EligibilityDecision, decision_payload)
        allowlist: ProviderAllowlistCandidate | None = None
        if status is EligibilityStatus.ELIGIBLE and approval is not None:
            allowlist_payload: dict[str, object] = {
                "schema_version": "1.0",
                "candidate_id": _uuid4_from_payload(
                    {"identity": identity, "decision": decision.content_sha256}
                ),
                "identity": identity,
                "decision_content_sha256": decision.content_sha256,
                "reference_approval_content_sha256": approval.content_sha256,
                "reference_approval_expires_at": approval.expires_at,
                "created_at": exact.evaluated_at,
            }
            allowlist = _hashed_model(ProviderAllowlistCandidate, allowlist_payload)
        result_payload: dict[str, object] = {
            "schema_version": "1.0",
            "report": report,
            "decision": decision,
            "allowlist_candidate": allowlist,
        }
        return _hashed_model(ProviderEvaluationResult, result_payload)


class ProviderEvaluationRepositoryErrorCode(StrEnum):
    INVALID_RECORD = "INVALID_RECORD"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class ProviderEvaluationRepositoryError(RuntimeError):
    def __init__(self, code: ProviderEvaluationRepositoryErrorCode):
        self.code = code
        super().__init__(code.value)


def _recalculate_metrics(
    command: ProviderEvaluationCommand,
    attempts: tuple[EvaluationAttempt, ...],
) -> EvaluationMetrics:
    manifest = command.manifest
    expected_count = len(manifest.cases) * manifest.repetition_count
    if len(attempts) != expected_count:
        raise ValueError("evaluation attempt graph is incomplete")
    output_hashes: dict[str, list[str | None]] = defaultdict(list)
    index = 0
    for case in manifest.cases:
        request = _provider_ready_request(
            case.request,
            production=(
                manifest.fixture_classification
                is FixtureClassification.PRODUCTION_APPROVED_REFERENCE
            ),
        )
        request_sha256 = canonical_sha256(request)
        for repetition in range(1, manifest.repetition_count + 1):
            attempt = attempts[index]
            index += 1
            expected_attempt_id = _uuid4_from_payload(
                {
                    "identity": command.candidate,
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "evaluated_at": command.evaluated_at,
                }
            )
            expected_cost = (
                None
                if attempt.input_tokens is None or attempt.output_tokens is None
                else _microunits(
                    attempt.input_tokens,
                    command.pricing.input_microunits_per_million_tokens,
                )
                + _microunits(
                    attempt.output_tokens,
                    command.pricing.output_microunits_per_million_tokens,
                )
            )
            if (
                attempt.case_id != case.case_id
                or attempt.repetition != repetition
                or attempt.attempt_id != expected_attempt_id
                or attempt.identity != command.candidate
                or attempt.request_sha256 != request_sha256
                or attempt.observed_cost_microunits != expected_cost
                or attempt.latency_ms is None
                or attempt.agent_run.started_at != command.evaluated_at
                or attempt.agent_run.completed_at
                != command.evaluated_at + timedelta(milliseconds=attempt.latency_ms)
            ):
                raise ValueError("evaluation attempt graph is not canonical")
            output_hashes[case.case_id].append(attempt.output_sha256)

    observed_values = [
        item.observed_cost_microunits
        for item in attempts
        if item.observed_cost_microunits is not None
    ]
    latencies = [item.latency_ms for item in attempts if item.latency_ms is not None]
    stability_values = [
        bool(values) and None not in values and len(set(values)) == 1
        for values in output_hashes.values()
    ]
    estimated_cost = manifest.repetition_count * sum(
        _microunits(
            case.estimated_input_tokens,
            command.pricing.input_microunits_per_million_tokens,
        )
        + _microunits(
            case.estimated_output_tokens,
            command.pricing.output_microunits_per_million_tokens,
        )
        for case in manifest.cases
    )
    return _hashed_model(
        EvaluationMetrics,
        {
            "schema_version": "1.0",
            "attempt_count": len(attempts),
            "evidence_score_bps": _score([item.evidence_correct for item in attempts]),
            "numeric_score_bps": _score([item.numeric_correct for item in attempts]),
            "claim_score_bps": _score([item.claims_correct for item in attempts]),
            "drill_score_bps": _score([item.drill_compliant for item in attempts]),
            "unsupported_violation_count": sum(
                item.unsupported_violation_count for item in attempts
            ),
            "stability_score_bps": _score(stability_values),
            "observed_cost_microunits": (
                sum(observed_values) if len(observed_values) == len(attempts) else None
            ),
            "estimated_cost_microunits": estimated_cost,
            "p95_latency_ms": (_p95(latencies) if len(latencies) == len(attempts) else None),
        },
    )


def _quality_failures(
    attempts: tuple[EvaluationAttempt, ...],
    metrics: EvaluationMetrics,
    gates: BenchmarkHardGatePolicy,
) -> tuple[EvaluationFailureReason, ...]:
    failures: list[EvaluationFailureReason] = []
    if not all(item.invocation_succeeded for item in attempts):
        failures.append(EvaluationFailureReason.INVOCATION_GATE_FAILED)
    if not all(item.identity_matched for item in attempts):
        failures.append(EvaluationFailureReason.IDENTITY_GATE_FAILED)
    if metrics.evidence_score_bps < gates.minimum_evidence_score_bps:
        failures.append(EvaluationFailureReason.EVIDENCE_GATE_FAILED)
    if metrics.numeric_score_bps < gates.minimum_numeric_score_bps:
        failures.append(EvaluationFailureReason.NUMERIC_GATE_FAILED)
    if metrics.claim_score_bps < 10_000:
        failures.append(EvaluationFailureReason.CLAIM_GATE_FAILED)
    if metrics.drill_score_bps < gates.minimum_drill_score_bps:
        failures.append(EvaluationFailureReason.DRILL_GATE_FAILED)
    if metrics.unsupported_violation_count != 0:
        failures.append(EvaluationFailureReason.UNSUPPORTED_GATE_FAILED)
    if metrics.stability_score_bps != 10_000:
        failures.append(EvaluationFailureReason.STABILITY_GATE_FAILED)
    if (
        metrics.observed_cost_microunits is None
        or metrics.observed_cost_microunits > gates.maximum_observed_cost_microunits
        or metrics.estimated_cost_microunits > gates.maximum_estimated_cost_microunits
    ):
        failures.append(EvaluationFailureReason.COST_GATE_FAILED)
    if metrics.p95_latency_ms is None or metrics.p95_latency_ms > gates.maximum_p95_latency_ms:
        failures.append(EvaluationFailureReason.LATENCY_GATE_FAILED)
    return tuple(failures)


def _rebuild_result(
    command: ProviderEvaluationCommand,
    result: ProviderEvaluationResult,
    *,
    at: datetime,
) -> ProviderEvaluationResult:
    exact_command = _canonical_copy(command, ProviderEvaluationCommand)
    exact_result = _canonical_copy(result, ProviderEvaluationResult)
    attempts = exact_result.report.attempts
    metrics = _recalculate_metrics(exact_command, attempts)
    if exact_result.report.metrics != metrics:
        raise ValueError("evaluation metrics do not match attempts")
    quality_failures = _quality_failures(attempts, metrics, exact_command.manifest.hard_gates)
    evaluation_id = _uuid4_from_payload(
        {"identity": exact_command.candidate, "evaluated_at": exact_command.evaluated_at}
    )
    report = _hashed_model(
        ProviderEvaluationReport,
        {
            "schema_version": "1.0",
            "evaluation_id": evaluation_id,
            "identity": exact_command.candidate,
            "fixture_classification": exact_command.manifest.fixture_classification,
            "manifest_content_sha256": exact_command.manifest.content_sha256,
            "attempts": attempts,
            "metrics": metrics,
            "quality_failure_reasons": quality_failures,
            "all_quality_gates_passed": not quality_failures,
            "evaluated_at": exact_command.evaluated_at,
        },
    )
    if report != exact_result.report:
        raise ValueError("evaluation report is not canonical")

    approval = exact_result.decision.approval_receipt
    failures = list(quality_failures)
    if not failures:
        if exact_command.manifest.fixture_classification is FixtureClassification.SYNTHETIC_TEST:
            failures.append(EvaluationFailureReason.SYNTHETIC_ONLY)
            approval = None
        elif approval is None:
            if exact_result.decision.failure_reasons not in {
                (EvaluationFailureReason.APPROVAL_REQUIRED,),
                (EvaluationFailureReason.APPROVAL_INVALID,),
            }:
                raise ValueError("approval failure is not canonical")
            failures.extend(exact_result.decision.failure_reasons)
        else:
            approval = _canonical_copy(approval, ReferenceApprovalReceipt)
            if (
                approval.benchmark_id != exact_command.manifest.benchmark_id
                or approval.benchmark_revision != exact_command.manifest.benchmark_revision
                or approval.benchmark_content_sha256 != exact_command.manifest.content_sha256
                or approval.reference_set_sha256 != exact_command.manifest.reference_set_sha256
                or not approval.approved_at <= at < approval.expires_at
            ):
                raise ValueError("reference approval is not current and exact")
    elif approval is not None:
        raise ValueError("failed quality gates cannot carry approval")

    status = EligibilityStatus.ELIGIBLE if not failures else EligibilityStatus.NOT_ELIGIBLE
    decision_id = _uuid4_from_payload(
        {"evaluation_id": evaluation_id, "report": report.content_sha256, "status": status}
    )
    decision = _hashed_model(
        EligibilityDecision,
        {
            "schema_version": "1.0",
            "decision_id": decision_id,
            "identity": exact_command.candidate,
            "report_content_sha256": report.content_sha256,
            "status": status,
            "failure_reasons": tuple(failures),
            "approval_receipt": approval,
            "decided_at": exact_command.evaluated_at,
        },
    )
    if decision != exact_result.decision:
        raise ValueError("eligibility decision is not canonical")

    candidate: ProviderAllowlistCandidate | None = None
    if status is EligibilityStatus.ELIGIBLE and approval is not None:
        candidate = _hashed_model(
            ProviderAllowlistCandidate,
            {
                "schema_version": "1.0",
                "candidate_id": _uuid4_from_payload(
                    {"identity": exact_command.candidate, "decision": decision.content_sha256}
                ),
                "identity": exact_command.candidate,
                "decision_content_sha256": decision.content_sha256,
                "reference_approval_content_sha256": approval.content_sha256,
                "reference_approval_expires_at": approval.expires_at,
                "created_at": exact_command.evaluated_at,
            },
        )
    if candidate != exact_result.allowlist_candidate:
        raise ValueError("allowlist candidate is not canonical")
    rebuilt = _hashed_model(
        ProviderEvaluationResult,
        {
            "schema_version": "1.0",
            "report": report,
            "decision": decision,
            "allowlist_candidate": candidate,
        },
    )
    if rebuilt != exact_result:
        raise ValueError("evaluation result is not canonical")
    return rebuilt


class InMemoryProviderEvaluationRepository:
    """Atomic reference adapter; not a production persistence claim."""

    def __init__(self, *, utc_now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        if not callable(utc_now):
            raise TypeError("trusted clock must be callable")
        self._lock = RLock()
        self._utc_now = utc_now
        self._receipts: dict[tuple[str, str], tuple[str, dict[str, object], dict[str, object]]] = {}
        self._allowlist: dict[UUID, dict[str, object]] = {}
        self._inflight: dict[
            tuple[str, str], tuple[str, asyncio.Future[dict[str, object] | None]]
        ] = {}

    @staticmethod
    def _command(command: object) -> ProviderEvaluationCommand:
        try:
            if not isinstance(command, ProviderEvaluationCommand):
                raise ValueError
            return _canonical_copy(command, ProviderEvaluationCommand)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ProviderEvaluationRepositoryError(
                ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
            ) from None

    @staticmethod
    def _load_result(payload: dict[str, object]) -> ProviderEvaluationResult:
        try:
            return ProviderEvaluationResult.model_validate(copy.deepcopy(payload), strict=True)
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ProviderEvaluationRepositoryError(
                ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
            ) from None

    def _rebuild(
        self,
        command: ProviderEvaluationCommand,
        result: object,
    ) -> ProviderEvaluationResult:
        try:
            if not isinstance(result, ProviderEvaluationResult):
                raise ValueError
            return _rebuild_result(
                command,
                result,
                at=_trusted_utc_now(self._utc_now),
            )
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ProviderEvaluationRepositoryError(
                ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
            ) from None

    @staticmethod
    def _identity(
        command: ProviderEvaluationCommand,
    ) -> tuple[str, tuple[str, str]]:
        return canonical_sha256(command), (
            command.manifest.benchmark_id,
            command.idempotency_key,
        )

    def save(
        self, command: ProviderEvaluationCommand, result: ProviderEvaluationResult
    ) -> ProviderEvaluationResult:
        exact_command = self._command(command)
        fingerprint, key = self._identity(exact_command)
        with self._lock:
            receipt = self._receipts.get(key)
            if receipt is not None:
                if receipt[0] != fingerprint:
                    raise ProviderEvaluationRepositoryError(
                        ProviderEvaluationRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return self._load_result(receipt[2])
        exact_result = self._rebuild(exact_command, result)
        command_payload = exact_command.model_dump(mode="python", round_trip=True)
        result_payload = exact_result.model_dump(mode="python", round_trip=True)
        with self._lock:
            receipt = self._receipts.get(key)
            if receipt is not None:
                if receipt[0] != fingerprint:
                    raise ProviderEvaluationRepositoryError(
                        ProviderEvaluationRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return self._load_result(receipt[2])
            self._receipts[key] = (
                fingerprint,
                copy.deepcopy(command_payload),
                copy.deepcopy(result_payload),
            )
            if exact_result.allowlist_candidate is not None:
                self._allowlist[exact_result.allowlist_candidate.candidate_id] = (
                    exact_result.allowlist_candidate.model_dump(mode="python", round_trip=True)
                )
            return self._load_result(result_payload)

    async def run_once(
        self,
        command: ProviderEvaluationCommand,
        evaluator: Callable[[ProviderEvaluationCommand], Awaitable[ProviderEvaluationResult]],
    ) -> ProviderEvaluationResult:
        exact_command = self._command(command)
        fingerprint, key = self._identity(exact_command)
        owner = False
        with self._lock:
            receipt = self._receipts.get(key)
            if receipt is not None:
                if receipt[0] != fingerprint:
                    raise ProviderEvaluationRepositoryError(
                        ProviderEvaluationRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                return self._load_result(receipt[2])
            active = self._inflight.get(key)
            if active is not None:
                if active[0] != fingerprint:
                    raise ProviderEvaluationRepositoryError(
                        ProviderEvaluationRepositoryErrorCode.IDEMPOTENCY_CONFLICT
                    )
                future = active[1]
            else:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = (fingerprint, future)
                owner = True

        if not owner:
            payload = await asyncio.shield(future)
            if payload is None:
                raise ProviderEvaluationRepositoryError(
                    ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
                )
            return self._load_result(payload)

        try:
            raw_result = await evaluator(exact_command)
            exact_result = self._rebuild(exact_command, raw_result)
            command_payload = exact_command.model_dump(mode="python", round_trip=True)
            result_payload = exact_result.model_dump(mode="python", round_trip=True)
            with self._lock:
                self._receipts[key] = (
                    fingerprint,
                    copy.deepcopy(command_payload),
                    copy.deepcopy(result_payload),
                )
                if exact_result.allowlist_candidate is not None:
                    candidate = exact_result.allowlist_candidate
                    self._allowlist[candidate.candidate_id] = candidate.model_dump(
                        mode="python", round_trip=True
                    )
                self._inflight.pop(key, None)
                if not future.done():
                    future.set_result(copy.deepcopy(result_payload))
            return self._load_result(result_payload)
        except BaseException:
            with self._lock:
                self._inflight.pop(key, None)
                if not future.done():
                    future.set_result(None)
            raise

    def list_allowlist_candidates(self) -> tuple[ProviderAllowlistCandidate, ...]:
        with self._lock:
            payloads = tuple(self._allowlist[key] for key in sorted(self._allowlist))
        try:
            return tuple(
                ProviderAllowlistCandidate.model_validate(copy.deepcopy(payload), strict=True)
                for payload in payloads
            )
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            raise ProviderEvaluationRepositoryError(
                ProviderEvaluationRepositoryErrorCode.INVALID_RECORD
            ) from None


__all__ = [
    "BenchmarkCase",
    "BenchmarkHardGatePolicy",
    "BenchmarkReferenceLabel",
    "CandidateIdentity",
    "DrillReferenceLabel",
    "EligibilityDecision",
    "EligibilityStatus",
    "EvaluationAttempt",
    "EvaluationFailureReason",
    "EvaluationMetrics",
    "FixtureClassification",
    "FrozenBenchmarkManifest",
    "InMemoryProviderEvaluationRepository",
    "NumericFactLabel",
    "ProviderAllowlistCandidate",
    "ProviderEvaluationCommand",
    "ProviderEvaluationReport",
    "ProviderEvaluationRepositoryError",
    "ProviderEvaluationRepositoryErrorCode",
    "ProviderEvaluationResult",
    "ProviderEvaluationRunner",
    "ProviderPricingPolicy",
    "ReferenceApprovalPort",
    "ReferenceApprovalReceipt",
    "ReferenceApprovalStatus",
    "revalidate_provider_allowlist_candidate",
]
