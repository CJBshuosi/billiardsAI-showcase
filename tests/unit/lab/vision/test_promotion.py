from __future__ import annotations

import pytest
from pydantic import ValidationError

from billiards_ai.lab.vision.promotion import (
    BenchmarkIdentity,
    BenchmarkSuiteResult,
    RelativeMetricRule,
    VisionBenchmarkResult,
    VisionPromotionPolicy,
    evaluate_promotion,
)


def _identity(
    suite_id: str,
    *,
    population: str = "1" * 64,
) -> BenchmarkIdentity:
    return BenchmarkIdentity(
        suite_id=suite_id,
        dataset_id=f"{suite_id}-dataset",
        population_sha256=population,
        labels_sha256="2" * 64,
        evaluator_sha256="3" * 64,
        thresholds_sha256="4" * 64,
    )


def _policy(
    *,
    comparison: str = "candidate_gte_incumbent_plus_margin",
    margin: float = 0.0,
) -> VisionPromotionPolicy:
    return VisionPromotionPolicy(
        policy_sha256="d" * 64,
        required_absolute_suites=("target-domain-v1",),
        required_downstream_suites=("tracking-events-v1",),
        relative_rules=(
            RelativeMetricRule(
                suite_id="target-domain-v1",
                metric="f1",
                comparison=comparison,
                margin=margin,
            ),
        ),
    )


def _result(
    *,
    package_id: str,
    f1: float,
    population: str = "1" * 64,
    policy_sha256: str = "d" * 64,
    absolute_passed: bool = True,
    downstream_passed: bool = True,
) -> VisionBenchmarkResult:
    return VisionBenchmarkResult(
        package_id=package_id,
        policy_sha256=policy_sha256,
        suites=(
            BenchmarkSuiteResult(
                identity=_identity("target-domain-v1", population=population),
                metrics={"f1": f1, "latency_ms": 20.0},
                absolute_gates_passed=absolute_passed,
                downstream_regression_passed=True,
            ),
            BenchmarkSuiteResult(
                identity=_identity("tracking-events-v1"),
                metrics={"f1": 0.9},
                absolute_gates_passed=True,
                downstream_regression_passed=downstream_passed,
            ),
        ),
    )


def test_candidate_passes_only_when_all_three_gate_families_pass() -> None:
    verdict = evaluate_promotion(
        policy=_policy(),
        incumbent=_result(package_id="a" * 64, f1=0.91),
        candidate=_result(package_id="b" * 64, f1=0.93),
    )

    assert verdict.promotion_status == "passed"
    assert verdict.passed_absolute_gates is True
    assert verdict.beats_incumbent is True
    assert verdict.passed_downstream_regressions is True


def test_mismatched_population_is_incomplete() -> None:
    verdict = evaluate_promotion(
        _policy(),
        _result(package_id="a" * 64, f1=0.91, population="a" * 64),
        _result(package_id="b" * 64, f1=0.93, population="b" * 64),
    )

    assert verdict.promotion_status == "incomplete"
    assert "population identity mismatch" in verdict.reasons


def test_absolute_pass_without_incumbent_margin_is_failed() -> None:
    verdict = evaluate_promotion(
        _policy(margin=0.03),
        _result(package_id="a" * 64, f1=0.95),
        _result(package_id="b" * 64, f1=0.97),
    )

    assert verdict.passed_absolute_gates is True
    assert verdict.beats_incumbent is False
    assert verdict.promotion_status == "failed"


def test_failed_downstream_regression_blocks_promotion() -> None:
    verdict = evaluate_promotion(
        _policy(),
        _result(package_id="a" * 64, f1=0.91),
        _result(package_id="b" * 64, f1=0.93, downstream_passed=False),
    )

    assert verdict.passed_downstream_regressions is False
    assert verdict.promotion_status == "failed"


def test_lower_is_better_rule_is_supported() -> None:
    policy = _policy(comparison="candidate_lte_incumbent_plus_margin", margin=0.0)
    policy = policy.model_copy(
        update={
            "relative_rules": (
                RelativeMetricRule(
                    suite_id="target-domain-v1",
                    metric="latency_ms",
                    comparison="candidate_lte_incumbent_plus_margin",
                    margin=0.0,
                ),
            )
        }
    )
    verdict = evaluate_promotion(
        policy,
        _result(package_id="a" * 64, f1=0.91),
        _result(package_id="b" * 64, f1=0.93),
    )

    assert verdict.beats_incumbent is True


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_metric_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        BenchmarkSuiteResult(
            identity=_identity("target-domain-v1"),
            metrics={"f1": value},
            absolute_gates_passed=True,
            downstream_regression_passed=True,
        )


def test_policy_hash_mismatch_and_self_comparison_are_incomplete() -> None:
    hash_mismatch = evaluate_promotion(
        _policy(),
        _result(package_id="a" * 64, f1=0.91),
        _result(package_id="b" * 64, f1=0.93, policy_sha256="e" * 64),
    )
    self_comparison = evaluate_promotion(
        _policy(),
        _result(package_id="a" * 64, f1=0.91),
        _result(package_id="a" * 64, f1=0.93),
    )

    assert hash_mismatch.promotion_status == "incomplete"
    assert self_comparison.promotion_status == "incomplete"
