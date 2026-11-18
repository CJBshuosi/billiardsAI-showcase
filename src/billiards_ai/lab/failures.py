"""Stable failure records and matrix aggregation for algorithm experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby


class FailureSeverity(StrEnum):
    HIGH_CONFIDENCE_WRONG = "high_confidence_wrong"
    ABSTENTION = "abstention"
    LOW_CONFIDENCE_UNKNOWN = "low_confidence_unknown"


@dataclass(frozen=True)
class FailureDimensions:
    lighting: str
    occlusion: str
    cluster: str
    rail_pocket_proximity: str
    speed_blur: str
    identity_orientation: str
    camera_stability: str


@dataclass(frozen=True)
class FailureRecord:
    shot_id: str
    stage: str
    field: str
    severity: FailureSeverity
    confidence: float
    dimensions: FailureDimensions


@dataclass(frozen=True)
class FailureBucket:
    stage: str
    field: str
    dimensions: FailureDimensions
    count: int
    records: tuple[FailureRecord, ...]


_SEVERITY_PRIORITY = {
    FailureSeverity.HIGH_CONFIDENCE_WRONG: 0,
    FailureSeverity.ABSTENTION: 1,
    FailureSeverity.LOW_CONFIDENCE_UNKNOWN: 2,
}


def _bucket_key(record: FailureRecord) -> tuple[str, ...]:
    dimensions = record.dimensions
    return (
        record.stage,
        record.field,
        dimensions.lighting,
        dimensions.occlusion,
        dimensions.cluster,
        dimensions.rail_pocket_proximity,
        dimensions.speed_blur,
        dimensions.identity_orientation,
        dimensions.camera_stability,
    )


def build_failure_matrix(records: list[FailureRecord]) -> tuple[FailureBucket, ...]:
    ordered = sorted(records, key=_bucket_key)
    buckets: list[FailureBucket] = []
    for _, group in groupby(ordered, key=_bucket_key):
        grouped = list(group)
        ranked = tuple(
            sorted(
                grouped,
                key=lambda record: (
                    _SEVERITY_PRIORITY[record.severity],
                    -record.confidence,
                    record.shot_id,
                ),
            )
        )
        first = ranked[0]
        buckets.append(
            FailureBucket(
                stage=first.stage,
                field=first.field,
                dimensions=first.dimensions,
                count=len(ranked),
                records=ranked,
            )
        )
    return tuple(buckets)
