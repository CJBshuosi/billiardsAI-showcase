"""Deterministic, lineage-preserving aggregation of admitted profile increments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from billiards_ai.edge.realtime_chain.models import (
    EvidenceSource,
    EvidenceStatus,
    HabitFeature,
    ProfileIncrement,
)


@dataclass(frozen=True, slots=True)
class ProfileContribution:
    session_id: str
    shot_id: str
    source: EvidenceSource
    feature_code: str
    value: float | str
    unit: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    code: str
    unit: str
    sample_count: int
    mean: float | None
    total: float | None
    category_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("profile metric requires samples")
        numeric = self.mean is not None or self.total is not None
        categorical = bool(self.category_counts)
        if numeric == categorical:
            raise ValueError("profile metric must be numeric or categorical")


@dataclass(frozen=True, slots=True)
class ProfileAggregate:
    user_id: str
    distinct_session_count: int
    admitted_shot_count: int
    pending_shot_count: int
    abstained_shot_count: int
    unsupported_shot_count: int
    metrics: tuple[ProfileMetric, ...]
    contributions: tuple[ProfileContribution, ...]


def _metric(
    code: str,
    rows: tuple[ProfileContribution, ...],
) -> ProfileMetric:
    units = {row.unit for row in rows}
    if len(units) != 1:
        raise ValueError(f"profile metric has conflicting units: {code}")
    unit = next(iter(units))
    numeric_values = tuple(float(row.value) for row in rows if isinstance(row.value, (float, int)))
    category_values = tuple(str(row.value) for row in rows if isinstance(row.value, str))
    if numeric_values and category_values:
        raise ValueError(f"profile metric has conflicting value types: {code}")
    if numeric_values:
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError(f"profile metric contains non-finite values: {code}")
        total = sum(numeric_values)
        return ProfileMetric(
            code=code,
            unit=unit,
            sample_count=len(numeric_values),
            mean=total / len(numeric_values),
            total=total,
            category_counts=(),
        )
    counts = Counter(category_values)
    return ProfileMetric(
        code=code,
        unit=unit,
        sample_count=len(category_values),
        mean=None,
        total=None,
        category_counts=tuple(sorted(counts.items())),
    )


def _contribution(increment: ProfileIncrement, feature: HabitFeature) -> ProfileContribution:
    return ProfileContribution(
        session_id=increment.session_id,
        shot_id=increment.shot_id,
        source=increment.source,
        feature_code=feature.code,
        value=feature.value,
        unit=feature.unit,
        evidence_ids=feature.evidence_ids,
    )


def aggregate_profile(
    user_id: str,
    increments: tuple[ProfileIncrement, ...],
) -> ProfileAggregate:
    """Aggregate admitted increments without allowing pending evidence to leak values."""

    if not user_id.strip():
        raise ValueError("profile user ID must not be empty")
    if any(increment.user_id != user_id for increment in increments):
        raise ValueError("profile increments must belong to the same user")
    evidence_keys = tuple(
        (increment.source.artifact_id, increment.session_id, increment.shot_id)
        for increment in increments
    )
    if len(evidence_keys) != len(set(evidence_keys)):
        raise ValueError("duplicate profile evidence is not allowed")

    contributions = tuple(
        _contribution(increment, feature)
        for increment in increments
        if increment.status is EvidenceStatus.ADMITTED
        for feature in increment.features
    )
    grouped: dict[str, list[ProfileContribution]] = defaultdict(list)
    for contribution in contributions:
        grouped[contribution.feature_code].append(contribution)
    metrics = tuple(_metric(code, tuple(grouped[code])) for code in sorted(grouped))
    statuses = Counter(increment.status for increment in increments)
    return ProfileAggregate(
        user_id=user_id,
        distinct_session_count=len({increment.session_id for increment in increments}),
        admitted_shot_count=statuses[EvidenceStatus.ADMITTED],
        pending_shot_count=statuses[EvidenceStatus.PENDING],
        abstained_shot_count=statuses[EvidenceStatus.ABSTAINED],
        unsupported_shot_count=statuses[EvidenceStatus.UNSUPPORTED],
        metrics=metrics,
        contributions=contributions,
    )
