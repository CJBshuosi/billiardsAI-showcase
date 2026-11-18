"""Generic hard-gate evaluation for deterministic benchmark outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | int | bool
    denominator: int
    errors: int = 0
    abstentions: int = 0
    coverage: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None


@dataclass(frozen=True)
class GateResult:
    metric_name: str
    operator: str
    threshold: float | int | bool
    actual: float | int | bool | None
    passed: bool


@dataclass(frozen=True)
class BenchmarkResult:
    policy_revision: str
    metrics: tuple[MetricResult, ...]
    gates: tuple[GateResult, ...]
    passed: bool
    result_hash: str


_NON_GATE_PARAMETERS = {"match_tolerance_ms"}


def _gate_specs(node: dict[str, Any], prefix: str = "") -> list[tuple[str, str, Any]]:
    specs: list[tuple[str, str, Any]] = []
    for key, value in sorted(node.items()):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            specs.extend(_gate_specs(value, path))
            continue
        if key in _NON_GATE_PARAMETERS:
            continue
        if key.endswith("_min"):
            specs.append((path[: -len("_min")], ">=", value))
        elif key.endswith("_max"):
            specs.append((path[: -len("_max")], "<=", value))
        elif key.endswith("_required"):
            specs.append((path[: -len("_required")], "==", value))
        else:
            specs.append((path, "==", value))
    return specs


def _passes(actual: float | int | bool | None, operator: str, threshold: Any) -> bool:
    if actual is None:
        return False
    if operator == ">=":
        return actual >= threshold
    if operator == "<=":
        return actual <= threshold
    return actual == threshold


def run_benchmark(
    *,
    metrics: Iterable[MetricResult],
    config: dict[str, Any],
    repeated_prediction_hashes: tuple[str, ...],
) -> BenchmarkResult:
    if config.get("schema_version") != "1.0" or not config.get("revision"):
        raise ValueError("benchmark config must declare schema and revision")
    metric_rows = tuple(metrics)
    metric_by_name = {metric.name: metric for metric in metric_rows}
    if len(metric_by_name) != len(metric_rows):
        raise ValueError("metric names must be unique")
    deterministic = (
        len(repeated_prediction_hashes) >= 2
        and len(set(repeated_prediction_hashes)) == 1
        and all(len(value) == 64 for value in repeated_prediction_hashes)
    )
    metric_by_name["deterministic_prediction_metric_hash"] = MetricResult(
        name="deterministic_prediction_metric_hash",
        value=deterministic,
        denominator=len(repeated_prediction_hashes),
        errors=0 if deterministic else 1,
    )
    ordered_metrics = tuple(sorted(metric_by_name.values(), key=lambda item: item.name))
    gates = tuple(
        GateResult(
            metric_name=name,
            operator=operator,
            threshold=threshold,
            actual=metric_by_name[name].value if name in metric_by_name else None,
            passed=_passes(
                metric_by_name[name].value if name in metric_by_name else None,
                operator,
                threshold,
            ),
        )
        for name, operator, threshold in _gate_specs(config["hard_gates"])
    )
    payload = {
        "policy_revision": config["revision"],
        "metrics": [asdict(metric) for metric in ordered_metrics],
        "gates": [asdict(gate) for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }
    result_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkResult(
        policy_revision=config["revision"],
        metrics=ordered_metrics,
        gates=gates,
        passed=payload["passed"],
        result_hash=result_hash,
    )
