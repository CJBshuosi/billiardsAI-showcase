"""Deterministic benchmark metrics with explicit denominators and abstentions."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionErrorSummary:
    median: float
    p95: float
    denominator: int


@dataclass(frozen=True)
class AccuracyCoverageSummary:
    accuracy: float
    coverage: float
    denominator: int
    errors: int
    abstentions: int
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class ClassificationSummary:
    macro_f1: float
    class_f1: dict[str, float]
    coverage: float
    denominator: int


@dataclass(frozen=True)
class TrackingSummary:
    idf1: float
    hota: float
    id_switch_per_ball_shot: float


@dataclass(frozen=True)
class PrfSummary:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class ShotBoundarySummary:
    precision: float
    recall: float
    f1: float
    matched: int
    boundary_median_error_ms: float
    boundary_p95_error_ms: float


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def position_error_metrics(errors_mm: list[float]) -> PositionErrorSummary:
    if not errors_mm or any(error < 0 for error in errors_mm):
        raise ValueError("position errors must be a non-empty list of non-negative values")
    return PositionErrorSummary(
        median=float(statistics.median(errors_mm)),
        p95=_nearest_rank(errors_mm, 0.95),
        denominator=len(errors_mm),
    )


def first_hit_metrics(
    *,
    truth: list[str],
    prediction: list[str | None],
    eligible: list[bool],
) -> AccuracyCoverageSummary:
    if not (len(truth) == len(prediction) == len(eligible)):
        raise ValueError("first-hit inputs must have equal lengths")
    eligible_rows = [
        (expected, actual)
        for expected, actual, is_eligible in zip(truth, prediction, eligible, strict=True)
        if is_eligible
    ]
    denominator = len(eligible_rows)
    if denominator == 0:
        raise ValueError("first-hit metrics require at least one eligible shot")
    correct = sum(expected == actual for expected, actual in eligible_rows if actual is not None)
    abstentions = sum(actual is None for _, actual in eligible_rows)
    covered = denominator - abstentions
    low, high = wilson_interval(correct, denominator)
    return AccuracyCoverageSummary(
        accuracy=correct / denominator,
        coverage=covered / denominator,
        denominator=denominator,
        errors=denominator - correct,
        abstentions=abstentions,
        ci95_low=low,
        ci95_high=high,
    )


def classification_metrics(
    *,
    truth: list[str],
    prediction: list[str | None],
    classes: list[str],
) -> ClassificationSummary:
    if len(truth) != len(prediction):
        raise ValueError("classification inputs must have equal lengths")
    if not truth or not classes or len(classes) != len(set(classes)):
        raise ValueError("classification requires rows and unique classes")
    class_f1: dict[str, float] = {}
    for label in classes:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(truth, prediction, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(truth, prediction, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(truth, prediction, strict=True)
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        class_f1[label] = _f1(precision, recall)
    covered = sum(actual is not None for actual in prediction)
    return ClassificationSummary(
        macro_f1=statistics.fmean(class_f1.values()),
        class_f1=class_f1,
        coverage=covered / len(truth),
        denominator=len(truth),
    )


def tracking_metrics(
    *,
    identity_true_positive: int,
    identity_false_positive: int,
    identity_false_negative: int,
    detection_accuracy: float,
    association_accuracy: float,
    id_switches: int,
    ball_shots: int,
) -> TrackingSummary:
    counts = (
        identity_true_positive,
        identity_false_positive,
        identity_false_negative,
        id_switches,
    )
    if any(count < 0 for count in counts) or ball_shots <= 0:
        raise ValueError("tracking counts must be non-negative and ball_shots positive")
    if not (0 <= detection_accuracy <= 1 and 0 <= association_accuracy <= 1):
        raise ValueError("tracking accuracies must be between zero and one")
    idf1 = _ratio(
        2 * identity_true_positive,
        2 * identity_true_positive + identity_false_positive + identity_false_negative,
    )
    return TrackingSummary(
        idf1=idf1,
        hota=math.sqrt(detection_accuracy * association_accuracy),
        id_switch_per_ball_shot=id_switches / ball_shots,
    )


def _prf(true_positive: int, false_positive: int, false_negative: int) -> PrfSummary:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return PrfSummary(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def shot_boundary_metrics(
    *,
    truth: list[tuple[int, int]],
    prediction: list[tuple[int, int]],
    tolerance_ms: int,
) -> ShotBoundarySummary:
    if tolerance_ms < 0:
        raise ValueError("shot tolerance must be non-negative")
    remaining = set(range(len(prediction)))
    boundary_errors: list[float] = []
    matched = 0
    for expected_start, expected_end in truth:
        candidates = [
            (
                abs(expected_start - prediction[index][0])
                + abs(expected_end - prediction[index][1]),
                index,
            )
            for index in remaining
            if abs(expected_start - prediction[index][0]) <= tolerance_ms
            and abs(expected_end - prediction[index][1]) <= tolerance_ms
        ]
        if not candidates:
            continue
        _, selected = min(candidates)
        remaining.remove(selected)
        actual_start, actual_end = prediction[selected]
        boundary_errors.extend([abs(expected_start - actual_start), abs(expected_end - actual_end)])
        matched += 1
    summary = _prf(matched, len(prediction) - matched, len(truth) - matched)
    median_error = float(statistics.median(boundary_errors)) if boundary_errors else math.inf
    p95_error = _nearest_rank(boundary_errors, 0.95) if boundary_errors else math.inf
    return ShotBoundarySummary(
        precision=summary.precision,
        recall=summary.recall,
        f1=summary.f1,
        matched=matched,
        boundary_median_error_ms=median_error,
        boundary_p95_error_ms=p95_error,
    )


def event_metrics[T](
    *,
    truth: list[tuple[T, int]],
    prediction: list[tuple[T, int]],
    tolerance_ms: int,
) -> PrfSummary:
    if tolerance_ms < 0:
        raise ValueError("event tolerance must be non-negative")
    remaining = set(range(len(prediction)))
    matched = 0
    for expected_identity, expected_time in truth:
        candidates = [
            (abs(expected_time - prediction[index][1]), index)
            for index in remaining
            if prediction[index][0] == expected_identity
            and abs(expected_time - prediction[index][1]) <= tolerance_ms
        ]
        if not candidates:
            continue
        _, selected = min(candidates)
        remaining.remove(selected)
        matched += 1
    return _prf(matched, len(prediction) - matched, len(truth) - matched)
