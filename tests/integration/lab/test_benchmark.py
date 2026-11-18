from billiards_ai.lab.benchmark import MetricResult, run_benchmark
from billiards_ai.lab.failures import (
    FailureDimensions,
    FailureRecord,
    FailureSeverity,
    build_failure_matrix,
)

CONFIG = {
    "schema_version": "1.0",
    "revision": "test-policy-r1",
    "hard_gates": {
        "position_mm": {"median_max": 10, "p95_max": 20},
        "first_hit": {"top1_accuracy_min": 0.70, "coverage_min": 0.80},
        "unsupported_field_false_claims_max": 0,
        "deterministic_prediction_metric_hash_required": True,
    },
}


def _passing_metrics() -> list[MetricResult]:
    return [
        MetricResult(name="position_mm.median", value=8.0, denominator=4),
        MetricResult(name="position_mm.p95", value=18.0, denominator=4),
        MetricResult(name="first_hit.top1_accuracy", value=0.75, denominator=4),
        MetricResult(name="first_hit.coverage", value=1.0, denominator=4),
        MetricResult(name="unsupported_field_false_claims", value=0, denominator=4),
    ]


def test_same_inputs_produce_the_same_metric_hash() -> None:
    first = run_benchmark(
        metrics=_passing_metrics(),
        config=CONFIG,
        repeated_prediction_hashes=("a" * 64, "a" * 64),
    )
    second = run_benchmark(
        metrics=reversed(_passing_metrics()),
        config=CONFIG,
        repeated_prediction_hashes=("a" * 64, "a" * 64),
    )

    assert first.passed is True
    assert first.result_hash == second.result_hash


def test_missing_metric_fails_closed() -> None:
    result = run_benchmark(
        metrics=_passing_metrics()[:-1],
        config=CONFIG,
        repeated_prediction_hashes=("a" * 64, "a" * 64),
    )

    assert result.passed is False
    assert any(gate.metric_name == "unsupported_field_false_claims" for gate in result.gates)


def test_failure_matrix_prioritizes_high_confidence_wrong_outputs() -> None:
    dimensions = FailureDimensions(
        lighting="bright",
        occlusion="none",
        cluster="isolated",
        rail_pocket_proximity="open_table",
        speed_blur="slow_clear",
        identity_orientation="number_visible",
        camera_stability="stable",
    )
    records = [
        FailureRecord(
            shot_id="shot-abstain",
            stage="event",
            field="first_hit",
            severity=FailureSeverity.ABSTENTION,
            confidence=0.0,
            dimensions=dimensions,
        ),
        FailureRecord(
            shot_id="shot-wrong",
            stage="event",
            field="first_hit",
            severity=FailureSeverity.HIGH_CONFIDENCE_WRONG,
            confidence=0.95,
            dimensions=dimensions,
        ),
    ]

    matrix = build_failure_matrix(records)

    assert matrix[0].records[0].shot_id == "shot-wrong"
    assert matrix[0].count == 2
