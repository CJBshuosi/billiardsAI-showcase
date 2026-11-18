"""Policy-driven comparison of a candidate vision package with the active incumbent."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256


class BenchmarkIdentity(FrozenContract):
    suite_id: NonEmptyStr
    dataset_id: NonEmptyStr
    population_sha256: Sha256
    labels_sha256: Sha256
    evaluator_sha256: Sha256
    thresholds_sha256: Sha256


class BenchmarkSuiteResult(FrozenContract):
    identity: BenchmarkIdentity
    metrics: dict[str, float]
    absolute_gates_passed: bool
    downstream_regression_passed: bool

    @field_validator("metrics")
    @classmethod
    def metrics_are_finite_and_named(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not key.strip() for key in value):
            raise ValueError("benchmark metrics must be named and non-empty")
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("benchmark metrics must be finite")
        return dict(sorted(value.items()))


class VisionBenchmarkResult(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    package_id: Sha256
    policy_sha256: Sha256
    suites: tuple[BenchmarkSuiteResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def suite_ids_are_unique(self) -> Self:
        suite_ids = [suite.identity.suite_id for suite in self.suites]
        if len(suite_ids) != len(set(suite_ids)):
            raise ValueError("benchmark suite IDs must be unique")
        return self


class RelativeMetricRule(FrozenContract):
    suite_id: NonEmptyStr
    metric: NonEmptyStr
    comparison: Literal[
        "candidate_gte_incumbent_plus_margin",
        "candidate_lte_incumbent_plus_margin",
    ]
    margin: float = Field(allow_inf_nan=False)


class VisionPromotionPolicy(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    policy_sha256: Sha256
    required_absolute_suites: tuple[NonEmptyStr, ...] = Field(min_length=1)
    required_downstream_suites: tuple[NonEmptyStr, ...] = Field(min_length=1)
    relative_rules: tuple[RelativeMetricRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def policy_references_are_unique(self) -> Self:
        for values in (self.required_absolute_suites, self.required_downstream_suites):
            if len(values) != len(set(values)):
                raise ValueError("required benchmark suite IDs must be unique")
        rule_keys = [(rule.suite_id, rule.metric) for rule in self.relative_rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError("relative metric rules must be unique")
        return self


class MetricComparisonEvidence(FrozenContract):
    suite_id: NonEmptyStr
    metric: NonEmptyStr
    comparison: NonEmptyStr
    margin: float = Field(allow_inf_nan=False)
    incumbent_value: float = Field(allow_inf_nan=False)
    candidate_value: float = Field(allow_inf_nan=False)
    passed: bool


class VisionPromotionVerdict(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    policy_sha256: Sha256
    incumbent_package_id: Sha256
    candidate_package_id: Sha256
    passed_absolute_gates: bool
    beats_incumbent: bool
    passed_downstream_regressions: bool
    promotion_status: Literal["passed", "failed", "incomplete"]
    comparisons: tuple[MetricComparisonEvidence, ...]
    reasons: tuple[NonEmptyStr, ...]


def _suite_map(result: VisionBenchmarkResult) -> dict[str, BenchmarkSuiteResult]:
    return {suite.identity.suite_id: suite for suite in result.suites}


def _incomplete_verdict(
    policy: VisionPromotionPolicy,
    incumbent: VisionBenchmarkResult,
    candidate: VisionBenchmarkResult,
    reasons: list[str],
) -> VisionPromotionVerdict:
    return VisionPromotionVerdict(
        policy_sha256=policy.policy_sha256,
        incumbent_package_id=incumbent.package_id,
        candidate_package_id=candidate.package_id,
        passed_absolute_gates=False,
        beats_incumbent=False,
        passed_downstream_regressions=False,
        promotion_status="incomplete",
        comparisons=(),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def evaluate_promotion(
    policy: VisionPromotionPolicy,
    incumbent: VisionBenchmarkResult,
    candidate: VisionBenchmarkResult,
) -> VisionPromotionVerdict:
    reasons: list[str] = []
    if incumbent.package_id == candidate.package_id:
        reasons.append("candidate and incumbent packages must differ")
    if incumbent.policy_sha256 != policy.policy_sha256:
        reasons.append("incumbent policy identity mismatch")
    if candidate.policy_sha256 != policy.policy_sha256:
        reasons.append("candidate policy identity mismatch")

    incumbent_suites = _suite_map(incumbent)
    candidate_suites = _suite_map(candidate)
    required = set(policy.required_absolute_suites) | set(policy.required_downstream_suites)
    required.update(rule.suite_id for rule in policy.relative_rules)
    for suite_id in sorted(required):
        before = incumbent_suites.get(suite_id)
        after = candidate_suites.get(suite_id)
        if before is None or after is None:
            reasons.append(f"required benchmark suite is missing: {suite_id}")
            continue
        if before.identity.population_sha256 != after.identity.population_sha256:
            reasons.append("population identity mismatch")
        if before.identity != after.identity:
            reasons.append(f"benchmark identity mismatch: {suite_id}")
    if reasons:
        return _incomplete_verdict(policy, incumbent, candidate, reasons)

    comparisons: list[MetricComparisonEvidence] = []
    for rule in policy.relative_rules:
        before = incumbent_suites[rule.suite_id]
        after = candidate_suites[rule.suite_id]
        if rule.metric not in before.metrics or rule.metric not in after.metrics:
            reasons.append(f"required comparison metric is missing: {rule.metric}")
            continue
        incumbent_value = before.metrics[rule.metric]
        candidate_value = after.metrics[rule.metric]
        if rule.comparison == "candidate_gte_incumbent_plus_margin":
            passed = candidate_value >= incumbent_value + rule.margin
        else:
            passed = candidate_value <= incumbent_value + rule.margin
        comparisons.append(
            MetricComparisonEvidence(
                suite_id=rule.suite_id,
                metric=rule.metric,
                comparison=rule.comparison,
                margin=rule.margin,
                incumbent_value=incumbent_value,
                candidate_value=candidate_value,
                passed=passed,
            )
        )
    if reasons:
        return _incomplete_verdict(policy, incumbent, candidate, reasons)

    passed_absolute = all(
        candidate_suites[suite_id].absolute_gates_passed
        for suite_id in policy.required_absolute_suites
    )
    passed_downstream = all(
        candidate_suites[suite_id].downstream_regression_passed
        for suite_id in policy.required_downstream_suites
    )
    beats_incumbent = all(comparison.passed for comparison in comparisons)
    if not passed_absolute:
        reasons.append("candidate failed absolute gates")
    if not beats_incumbent:
        reasons.append("candidate did not beat incumbent")
    if not passed_downstream:
        reasons.append("candidate failed downstream regressions")
    return VisionPromotionVerdict(
        policy_sha256=policy.policy_sha256,
        incumbent_package_id=incumbent.package_id,
        candidate_package_id=candidate.package_id,
        passed_absolute_gates=passed_absolute,
        beats_incumbent=beats_incumbent,
        passed_downstream_regressions=passed_downstream,
        promotion_status="passed" if not reasons else "failed",
        comparisons=tuple(comparisons),
        reasons=tuple(reasons),
    )
