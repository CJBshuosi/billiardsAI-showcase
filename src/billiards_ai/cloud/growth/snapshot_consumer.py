"""Strict Story 8.4 seam for consumers that support growth snapshot v2."""

from __future__ import annotations

from pydantic import ValidationError

from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.shared.models.growth import (
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
    GrowthSnapshotReadFence,
    GrowthSnapshotReadFenceV3,
    uuid4_from_canonical_payload,
)
from billiards_ai.shared.models.growth_metrics import MetricReliability


class SnapshotConsumerCompatibilityError(ValueError):
    """A consumer input is not an exact, fenced v2 snapshot."""


def prepare_growth_snapshot_for_consumer(
    value: GrowthEvidenceSnapshotV2 | str | bytes,
    *,
    expected_fence: GrowthSnapshotReadFence,
) -> GrowthEvidenceSnapshotV2:
    """Strictly reconstruct v2 without fabricating a lossy v1 compatibility view."""

    try:
        if isinstance(value, (str, bytes)):
            snapshot = GrowthEvidenceSnapshotV2.model_validate_json(value, strict=True)
        else:
            snapshot = GrowthEvidenceSnapshotV2.model_validate(
                stored_model_values(value), strict=True
            )
        fence = GrowthSnapshotReadFence.model_validate(
            stored_model_values(expected_fence), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotConsumerCompatibilityError("consumer requires a strict v2 snapshot") from exc
    if fence != snapshot.read_fence:
        raise SnapshotConsumerCompatibilityError("consumer read fence does not match snapshot")
    return snapshot


def prepare_growth_snapshot_v3_for_consumer(
    value: GrowthEvidenceSnapshotV3 | str | bytes,
    *,
    expected_fence: GrowthSnapshotReadFenceV3,
) -> GrowthEvidenceSnapshotV3:
    """Strictly reconstruct an exact V3 snapshot without a V2 downgrade."""

    try:
        if isinstance(value, (str, bytes)):
            snapshot = GrowthEvidenceSnapshotV3.model_validate_json(value, strict=True)
        else:
            snapshot = GrowthEvidenceSnapshotV3.model_validate(
                stored_model_values(value), strict=True
            )
        fence = GrowthSnapshotReadFenceV3.model_validate(
            stored_model_values(expected_fence), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotConsumerCompatibilityError("consumer requires a strict v3 snapshot") from exc
    if fence != snapshot.read_fence:
        raise SnapshotConsumerCompatibilityError("consumer read fence does not match snapshot")
    return snapshot


def growth_trend_lineage_payloads(
    value: GrowthEvidenceSnapshotV3,
) -> tuple[dict[str, object], ...]:
    """Map exact D1 trends losslessly to the frozen C1-D2 raw-free seam."""

    snapshot = GrowthEvidenceSnapshotV3.model_validate(stored_model_values(value), strict=True)
    payloads = []
    for assessment in snapshot.metric_assessments:
        trend = assessment.trend
        # The frozen D2 Reliability enum has no UNKNOWN member. Omitting an
        # unrepresentable trend preserves D1 truth instead of fabricating LOW.
        if trend.reliability is MetricReliability.UNKNOWN:
            continue
        evidence_ids = tuple(
            sorted(set(trend.previous_evidence_ref_ids + trend.current_evidence_ref_ids))
        )
        body: dict[str, object] = {
            "schema_version": "1.0",
            "source_snapshot_schema_version": "3.0",
            "metric_code": assessment.metric.metric_code.value,
            "metric_identity_sha256": assessment.metric_identity_sha256,
            "trend_direction": trend.direction.value,
            "trend_support_status": trend.support_status.value,
            "reliability": trend.reliability.value,
            "trend_policy_version": trend.trend_policy_version,
            "evidence_ref_ids": evidence_ids,
        }
        payloads.append(body | {"lineage_id": uuid4_from_canonical_payload(body)})
    ordered = tuple(sorted(payloads, key=lambda item: item["lineage_id"]))
    if len({item["lineage_id"] for item in ordered}) != len(ordered):
        raise SnapshotConsumerCompatibilityError("trend lineage ids must be unique")
    return ordered


__all__ = [
    "SnapshotConsumerCompatibilityError",
    "growth_trend_lineage_payloads",
    "prepare_growth_snapshot_for_consumer",
    "prepare_growth_snapshot_v3_for_consumer",
]
