"""Deterministic long-term growth facts and execution history contracts."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr, Sha256, require_utc
from billiards_ai.shared.models.growth_metrics import (
    CALCULATION_POLICY_VERSION,
    TAXONOMY_VERSION,
    MetricCode,
    MetricProjectionResult,
    MetricProjectionStatus,
    MetricReliability,
    MetricSupportStatusV2,
    ProjectedAbilityMetric,
)

SNAPSHOT_POLICY_VERSION = "growth-snapshot-rd-v1.0.0"
SNAPSHOT_IDENTITY_POLICY_VERSION = "growth-snapshot-identity-rd-v1.0.0"
SNAPSHOT_POLICY_VERSION_V3 = "growth-snapshot-rd-v3.0.0"
SNAPSHOT_IDENTITY_POLICY_VERSION_V3 = "growth-snapshot-identity-rd-v3.0.0"
TREND_POLICY_VERSION = "growth-trend-rd-v1.0.0"


class MetricSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class Reliability(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AbilityTaxonomy(CoreContractModel):
    skill_code: NonEmptyStr
    angle_band: NonEmptyStr | None = None
    distance_band: NonEmptyStr | None = None
    rail_context: NonEmptyStr | None = None


class GrowthEvidenceReference(CoreContractModel):
    evidence_ref_id: UUID4
    batch_id: UUID4
    session_id: UUID4
    metric_code: NonEmptyStr
    shot_ids: tuple[UUID4, ...] = Field(min_length=1)


class AbilityMetric(CoreContractModel):
    metric_code: NonEmptyStr
    taxonomy: AbilityTaxonomy
    support_status: MetricSupportStatus
    sample_count: StrictInt = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    reliability: Reliability
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_p50_mm: float | None = Field(default=None, ge=0.0)
    error_p90_mm: float | None = Field(default=None, ge=0.0)
    stability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def numeric_values_require_supported_evidence(self) -> AbilityMetric:
        values = (self.success_rate, self.error_p50_mm, self.error_p90_mm, self.stability)
        if self.support_status is MetricSupportStatus.SUPPORTED:
            if self.sample_count <= 0 or not any(value is not None for value in values):
                raise ValueError("supported metric requires samples and a numeric value")
        elif any(value is not None for value in values):
            raise ValueError("non-supported metric cannot carry derived numeric values")
        if self.support_status in {
            MetricSupportStatus.UNSUPPORTED,
            MetricSupportStatus.ABSTAINED,
        } and (self.sample_count != 0 or self.coverage != 0.0):
            raise ValueError("unsupported or abstained metric cannot claim evidence coverage")
        if (
            self.error_p50_mm is not None
            and self.error_p90_mm is not None
            and self.error_p50_mm > self.error_p90_mm
        ):
            raise ValueError("p50 error cannot exceed p90 error")
        return self


class PlanHistoryEntry(CoreContractModel):
    plan_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    status: NonEmptyStr
    started_at: datetime | None = None
    completed_at: datetime | None = None

    _started_at_utc = field_validator("started_at")(
        lambda value: None if value is None else require_utc(value)
    )
    _completed_at_utc = field_validator("completed_at")(
        lambda value: None if value is None else require_utc(value)
    )


class GrowthEvidenceSnapshot(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: UUID4
    user_id: UUID4
    version: StrictInt = Field(ge=1)
    created_at: datetime
    source_batch_ids: tuple[UUID4, ...] = Field(min_length=1)
    metrics: tuple[AbilityMetric, ...]
    evidence_refs: tuple[GrowthEvidenceReference, ...]
    plan_history: tuple[PlanHistoryEntry, ...] = ()
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def snapshot_collections_are_unique_and_linked(self) -> GrowthEvidenceSnapshot:
        if len(self.source_batch_ids) != len(set(self.source_batch_ids)):
            raise ValueError("source batch ids must be unique")
        metric_codes = tuple(item.metric_code for item in self.metrics)
        if len(metric_codes) != len(set(metric_codes)):
            raise ValueError("metric codes must be unique")
        ref_ids = tuple(item.evidence_ref_id for item in self.evidence_refs)
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("evidence reference ids must be unique")
        if any(item.batch_id not in self.source_batch_ids for item in self.evidence_refs):
            raise ValueError("evidence reference batch must exist in snapshot sources")
        return self


class GrowthSnapshotSourceBatchRef(CoreContractModel):
    """Raw-free source lineage retained even when evidence is not projected."""

    admission_schema_version: Literal["1.0"] = "1.0"
    admission_id: UUID4
    admitted_at: datetime
    batch_schema_version: Literal["2.0"] = "2.0"
    batch_id: UUID4
    user_id: UUID4
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    review_revision: StrictInt = Field(ge=1)
    approved_at: datetime
    approval_kind: NonEmptyStr
    approval_policy_version: NonEmptyStr
    batch_content_sha256: Sha256
    source_envelope_ids: tuple[UUID4, ...] = Field(min_length=1)
    metric_shot_count: StrictInt = Field(ge=1)

    _admitted_at_utc = field_validator("admitted_at")(require_utc)
    _approved_at_utc = field_validator("approved_at")(require_utc)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> GrowthSnapshotSourceBatchRef:
        if len(self.source_envelope_ids) != len(set(self.source_envelope_ids)):
            raise ValueError("snapshot source envelope ids must be unique")
        return self


class MetricShotTraceBindingV1(CoreContractModel):
    """Exact raw-free binding used by one metric observation."""

    schema_version: Literal["1.0"] = "1.0"
    metric_identity_sha256: Sha256
    shot_id: UUID4
    source_envelope_id: UUID4
    calculation_field_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def fields_are_canonical(self) -> Self:
        if self.calculation_field_codes != tuple(sorted(set(self.calculation_field_codes))):
            raise ValueError("metric shot calculation field codes must be sorted and unique")
        return self


class GrowthSnapshotSourceBatchRefV3(GrowthSnapshotSourceBatchRef):
    """V3 raw-free binding from an admitted batch to its approved metric shots."""

    approved_metric_shot_ids: tuple[UUID4, ...] = Field(min_length=1)
    metric_shot_bindings: tuple[MetricShotTraceBindingV1, ...] = ()

    @model_validator(mode="after")
    def approved_shot_ids_are_canonical(self) -> Self:
        if self.approved_metric_shot_ids != tuple(sorted(set(self.approved_metric_shot_ids))):
            raise ValueError("approved metric shot ids must be sorted and unique")
        if len(self.approved_metric_shot_ids) != self.metric_shot_count:
            raise ValueError("approved metric shot ids must reconcile with metric shot count")
        expected = tuple(
            sorted(
                self.metric_shot_bindings,
                key=lambda item: (item.metric_identity_sha256, item.shot_id),
            )
        )
        if self.metric_shot_bindings != expected:
            raise ValueError("metric shot trace bindings must use canonical order")
        keys = tuple(
            (item.metric_identity_sha256, item.shot_id) for item in self.metric_shot_bindings
        )
        if len(keys) != len(set(keys)):
            raise ValueError("metric shot trace bindings must be unique")
        if any(
            item.shot_id not in self.approved_metric_shot_ids
            or item.source_envelope_id not in self.source_envelope_ids
            for item in self.metric_shot_bindings
        ):
            raise ValueError("metric shot trace binding must belong to its source batch")
        return self


def growth_snapshot_source_sort_key(
    value: GrowthSnapshotSourceBatchRef,
) -> tuple[datetime, str, int, str, str]:
    return (
        value.approved_at,
        str(value.session_id),
        value.review_revision,
        str(value.batch_id),
        str(value.admission_id),
    )


class GrowthSnapshotLineage(CoreContractModel):
    version: StrictInt = Field(ge=1)
    previous_snapshot_id: UUID4 | None
    snapshot_policy_version: Literal["growth-snapshot-rd-v1.0.0"] = SNAPSHOT_POLICY_VERSION

    @model_validator(mode="after")
    def predecessor_matches_version(self) -> GrowthSnapshotLineage:
        if (self.version == 1) != (self.previous_snapshot_id is None):
            raise ValueError("only snapshot version 1 may omit its predecessor")
        return self


class GrowthSnapshotLineageV3(CoreContractModel):
    version: StrictInt = Field(ge=1)
    previous_snapshot_id: UUID4 | None
    snapshot_policy_version: Literal["growth-snapshot-rd-v3.0.0"] = SNAPSHOT_POLICY_VERSION_V3

    @model_validator(mode="after")
    def predecessor_matches_version(self) -> Self:
        if (self.version == 1) != (self.previous_snapshot_id is None):
            raise ValueError("only snapshot version 1 may omit its predecessor")
        return self


class GrowthSnapshotReadFence(CoreContractModel):
    user_id: UUID4
    snapshot_id: UUID4
    version: StrictInt = Field(ge=1)
    content_sha256: Sha256
    evidence_sha256: Sha256
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    snapshot_policy_version: Literal["growth-snapshot-rd-v1.0.0"] = SNAPSHOT_POLICY_VERSION


class GrowthSnapshotReadFenceV3(CoreContractModel):
    schema_version: Literal["3.0"] = "3.0"
    user_id: UUID4
    snapshot_id: UUID4
    version: StrictInt = Field(ge=1)
    content_sha256: Sha256
    evidence_sha256: Sha256
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    snapshot_policy_version: Literal["growth-snapshot-rd-v3.0.0"] = SNAPSHOT_POLICY_VERSION_V3
    trend_policy_version: Literal["growth-trend-rd-v1.0.0"] = TREND_POLICY_VERSION


class TrendDirection(StrEnum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    UNKNOWN = "UNKNOWN"


class MetricEvidenceReferenceV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_ref_id: UUID4
    metric_identity_sha256: Sha256
    batch_id: UUID4
    session_id: UUID4
    shot_ids: tuple[UUID4, ...] = Field(min_length=1)
    source_envelope_ids: tuple[UUID4, ...] = Field(min_length=1)
    calculation_field_codes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    shot_trace_bindings: tuple[MetricShotTraceBindingV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def lineage_is_canonical_and_identity_derived(self) -> Self:
        for values, label in (
            (self.shot_ids, "shot ids"),
            (self.source_envelope_ids, "source envelope ids"),
            (self.calculation_field_codes, "calculation field codes"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
        if self.evidence_ref_id != uuid4_from_canonical_payload(
            self.model_dump(mode="json", exclude={"evidence_ref_id"})
        ):
            raise ValueError("evidence reference id must derive from canonical payload")
        expected_bindings = tuple(sorted(self.shot_trace_bindings, key=lambda item: item.shot_id))
        if self.shot_trace_bindings != expected_bindings:
            raise ValueError("evidence shot trace bindings must use canonical order")
        if any(
            item.metric_identity_sha256 != self.metric_identity_sha256
            for item in self.shot_trace_bindings
        ):
            raise ValueError("evidence shot trace binding metric identity mismatch")
        if (
            set(self.shot_ids) != {item.shot_id for item in self.shot_trace_bindings}
            or set(self.source_envelope_ids)
            != {item.source_envelope_id for item in self.shot_trace_bindings}
            or set(self.calculation_field_codes)
            != {code for item in self.shot_trace_bindings for code in item.calculation_field_codes}
        ):
            raise ValueError("evidence reference violates exact metric shot trace binding")
        return self


def _metric_identity(metric: ProjectedAbilityMetric) -> str:
    from billiards_ai.shared.models.events import canonical_sha256

    return canonical_sha256(
        {
            "metric_code": metric.metric_code.value,
            "taxonomy": metric.taxonomy.model_dump(mode="json"),
            "calculation_policy_version": metric.calculation_policy_version,
        }
    )


class MetricTrendWindowSummaryV1(CoreContractModel):
    """Frozen metrics and lineage required to revalidate one trend window."""

    schema_version: Literal["1.0"] = "1.0"
    window_start: datetime
    window_end: datetime
    metric_identity_sha256: Sha256
    metric_code: MetricCode
    comparable_metric: ProjectedAbilityMetric | None = None
    family_metrics: tuple[ProjectedAbilityMetric, ...]
    evidence_ref_ids: tuple[UUID4, ...]

    _window_start_utc = field_validator("window_start")(require_utc)
    _window_end_utc = field_validator("window_end")(require_utc)

    @model_validator(mode="after")
    def summary_is_canonical_and_identity_bound(self) -> Self:
        if self.window_end - self.window_start != timedelta(days=90):
            raise ValueError("trend summary window must be exactly 90 days")
        identities = tuple(_metric_identity(item) for item in self.family_metrics)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("trend family metrics must use canonical unique identities")
        if any(item.metric_code is not self.metric_code for item in self.family_metrics):
            raise ValueError("trend family metrics must share one metric code")
        expected = next(
            (
                item
                for item in self.family_metrics
                if _metric_identity(item) == self.metric_identity_sha256
            ),
            None,
        )
        if self.comparable_metric != expected:
            raise ValueError("trend comparable metric must match its exact identity")
        if self.evidence_ref_ids != tuple(sorted(set(self.evidence_ref_ids))):
            raise ValueError("trend summary evidence ids must be sorted and unique")
        return self


def _rounded(value: float, digits: int) -> float:
    quantum = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _window_support(*summaries: MetricTrendWindowSummaryV1) -> MetricSupportStatus:
    statuses = {metric.support_status for summary in summaries for metric in summary.family_metrics}
    if MetricSupportStatusV2.UNSUPPORTED in statuses:
        return MetricSupportStatus.UNSUPPORTED
    if MetricSupportStatusV2.ABSTAINED in statuses:
        return MetricSupportStatus.ABSTAINED
    return MetricSupportStatus.INSUFFICIENT_EVIDENCE


class MetricTrendV1(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    direction: TrendDirection
    support_status: MetricSupportStatus
    current_window_start: datetime
    current_window_end: datetime
    previous_window_start: datetime
    previous_window_end: datetime
    current_value: float | None = None
    previous_value: float | None = None
    delta: float | None = None
    reliability: MetricReliability
    current_evidence_ref_ids: tuple[UUID4, ...]
    previous_evidence_ref_ids: tuple[UUID4, ...]
    current_window_summary: MetricTrendWindowSummaryV1
    previous_window_summary: MetricTrendWindowSummaryV1
    trend_policy_version: Literal["growth-trend-rd-v1.0.0"] = TREND_POLICY_VERSION

    _current_start_utc = field_validator("current_window_start")(require_utc)
    _current_end_utc = field_validator("current_window_end")(require_utc)
    _previous_start_utc = field_validator("previous_window_start")(require_utc)
    _previous_end_utc = field_validator("previous_window_end")(require_utc)

    @model_validator(mode="after")
    def trend_is_canonical_and_policy_closed(self) -> Self:
        if not (
            self.previous_window_start
            < self.previous_window_end
            == self.current_window_start
            < self.current_window_end
        ):
            raise ValueError("trend policy mismatch: windows must be adjacent and increasing")
        if self.current_window_end - self.current_window_start != timedelta(
            days=90
        ) or self.previous_window_end - self.previous_window_start != timedelta(days=90):
            raise ValueError("trend policy mismatch: windows must be exactly 90 days")
        if (self.current_window_start, self.current_window_end) != (
            self.current_window_summary.window_start,
            self.current_window_summary.window_end,
        ) or (self.previous_window_start, self.previous_window_end) != (
            self.previous_window_summary.window_start,
            self.previous_window_summary.window_end,
        ):
            raise ValueError("trend policy mismatch: summary windows differ")
        for values in (self.current_evidence_ref_ids, self.previous_evidence_ref_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("trend evidence ids must be sorted and unique")
        if set(self.current_evidence_ref_ids) & set(self.previous_evidence_ref_ids):
            raise ValueError("trend evidence windows must be disjoint")
        if (
            self.current_evidence_ref_ids != self.current_window_summary.evidence_ref_ids
            or self.previous_evidence_ref_ids != self.previous_window_summary.evidence_ref_ids
        ):
            raise ValueError("trend policy mismatch: summary evidence differs")
        numerics = (self.current_value, self.previous_value, self.delta)
        if any(value is not None and not math.isfinite(value) for value in numerics):
            raise ValueError("trend numeric values must be finite")
        if self.direction is TrendDirection.UNKNOWN:
            if any(value is not None for value in numerics):
                raise ValueError("unknown trend cannot claim numeric values")
            if self.reliability is not MetricReliability.UNKNOWN:
                raise ValueError("unknown trend requires unknown reliability")
            if self.support_status is MetricSupportStatus.SUPPORTED:
                raise ValueError("unknown trend cannot claim supported status")
        elif (
            self.support_status is not MetricSupportStatus.SUPPORTED
            or self.reliability is MetricReliability.UNKNOWN
            or any(value is None for value in numerics)
            or not self.current_evidence_ref_ids
            or not self.previous_evidence_ref_ids
        ):
            raise ValueError("substantive trend requires supported numeric evidence")
        elif round(float(self.current_value) - float(self.previous_value), 9) != round(
            float(self.delta), 9
        ):
            raise ValueError("trend delta must equal current minus previous")
        return self

    def validate_frozen_policy(
        self,
        *,
        metric_identity_sha256: str,
        metric_value_field_code: str,
    ) -> None:
        summaries = (self.previous_window_summary, self.current_window_summary)
        if any(item.metric_identity_sha256 != metric_identity_sha256 for item in summaries):
            raise ValueError("trend policy mismatch: metric identity differs")
        previous = self.previous_window_summary.comparable_metric
        current = self.current_window_summary.comparable_metric
        qualified = all(
            item is not None
            and item.support_status is MetricSupportStatusV2.SUPPORTED
            and item.sample_count >= 5
            and item.distinct_session_count >= 2
            for item in (previous, current)
        )
        previous_value = None if previous is None else getattr(previous, metric_value_field_code)
        current_value = None if current is None else getattr(current, metric_value_field_code)
        qualified = qualified and previous_value is not None and current_value is not None
        if not qualified:
            if (
                self.direction is not TrendDirection.UNKNOWN
                or self.support_status != _window_support(*summaries)
                or self.reliability is not MetricReliability.UNKNOWN
                or any(
                    value is not None
                    for value in (self.current_value, self.previous_value, self.delta)
                )
            ):
                raise ValueError("trend policy mismatch: UNKNOWN trend derivation differs")
            return
        assert previous is not None and current is not None
        assert previous_value is not None and current_value is not None
        digits = 3 if metric_value_field_code == "error_p50_mm" else 6
        expected_previous = _rounded(float(previous_value), digits)
        expected_current = _rounded(float(current_value), digits)
        expected_delta = _rounded(float(current_value) - float(previous_value), digits)
        threshold = 25.0 if metric_value_field_code == "error_p50_mm" else 0.05
        if abs(expected_delta) < threshold:
            expected_direction = TrendDirection.STABLE
        elif metric_value_field_code == "error_p50_mm":
            expected_direction = (
                TrendDirection.IMPROVING
                if expected_delta <= -threshold
                else TrendDirection.DECLINING
            )
        else:
            expected_direction = (
                TrendDirection.IMPROVING
                if expected_delta >= threshold
                else TrendDirection.DECLINING
            )
        expected_reliability = min(
            (previous.reliability, current.reliability),
            key=lambda item: {
                MetricReliability.LOW: 0,
                MetricReliability.MEDIUM: 1,
                MetricReliability.HIGH: 2,
            }[item],
        )
        if (
            self.support_status is not MetricSupportStatus.SUPPORTED
            or self.direction is not expected_direction
            or self.reliability is not expected_reliability
            or self.previous_value != expected_previous
            or self.current_value != expected_current
            or self.delta != expected_delta
        ):
            raise ValueError("trend policy mismatch: numeric derivation differs")


class GrowthMetricAssessmentV3(CoreContractModel):
    schema_version: Literal["3.0"] = "3.0"
    metric_identity_sha256: Sha256
    metric_value_field_code: Literal["success_rate", "error_p50_mm"]
    metric: ProjectedAbilityMetric
    evidence_refs: tuple[MetricEvidenceReferenceV1, ...]
    trend: MetricTrendV1

    @model_validator(mode="after")
    def assessment_is_identity_bound(self) -> Self:
        from billiards_ai.shared.models.events import canonical_sha256

        expected = canonical_sha256(
            {
                "metric_code": self.metric.metric_code.value,
                "taxonomy": self.metric.taxonomy.model_dump(mode="json"),
                "calculation_policy_version": self.metric.calculation_policy_version,
            }
        )
        if self.metric_identity_sha256 != expected:
            raise ValueError("metric identity hash mismatch")
        expected_field = (
            "error_p50_mm"
            if self.metric.metric_code.value == "positioning.target_zone.error_mm"
            else "success_rate"
        )
        if self.metric_value_field_code != expected_field:
            raise ValueError("metric value field code does not match metric policy")
        if self.evidence_refs != tuple(
            sorted(self.evidence_refs, key=lambda item: item.evidence_ref_id)
        ):
            raise ValueError("metric evidence references must use canonical order")
        ids = tuple(item.evidence_ref_id for item in self.evidence_refs)
        if len(ids) != len(set(ids)):
            raise ValueError("metric evidence references must be unique")
        if any(item.metric_identity_sha256 != expected for item in self.evidence_refs):
            raise ValueError("metric evidence reference identity mismatch")
        if not set(
            self.trend.current_evidence_ref_ids + self.trend.previous_evidence_ref_ids
        ) <= set(ids):
            raise ValueError("trend references evidence absent from its metric assessment")
        refs_by_id = {item.evidence_ref_id: item for item in self.evidence_refs}
        for summary in (
            self.trend.previous_window_summary,
            self.trend.current_window_summary,
        ):
            selected = tuple(refs_by_id[item] for item in summary.evidence_ref_ids)
            if summary.comparable_metric is None:
                if selected:
                    raise ValueError(
                        "trend summary without a comparable metric cannot cite samples"
                    )
                continue
            if (
                sum(len(ref.shot_ids) for ref in selected) != summary.comparable_metric.sample_count
                or len({ref.session_id for ref in selected})
                != summary.comparable_metric.distinct_session_count
            ):
                raise ValueError("trend summary references must reconcile with its window metric")
        self.trend.validate_frozen_policy(
            metric_identity_sha256=self.metric_identity_sha256,
            metric_value_field_code=self.metric_value_field_code,
        )
        return self


class GrowthSnapshotMetricProjectionBinding(CoreContractModel):
    source_bundle_sha256: Sha256
    metric_projection_sha256: Sha256


def growth_snapshot_evidence_payload(value: GrowthEvidenceSnapshotV2) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "user_id": str(value.user_id),
        "evidence_cutoff": value.evidence_cutoff.isoformat(),
        "created_at": value.created_at.isoformat(),
        "taxonomy_version": value.taxonomy_version,
        "calculation_policy_version": value.calculation_policy_version,
        "snapshot_policy_version": value.snapshot_policy_version,
        "source_batches": [item.model_dump(mode="json") for item in value.source_batches],
        "source_bundle_sha256": value.source_bundle_sha256,
        "metric_projection": value.metric_projection.model_dump(mode="json"),
        "metric_projection_binding": value.metric_projection_binding.model_dump(mode="json"),
    }


def growth_snapshot_content_payload(value: GrowthEvidenceSnapshotV2) -> dict[str, object]:
    return growth_snapshot_evidence_payload(value) | {
        "version": value.version,
        "previous_snapshot_id": (
            str(value.previous_snapshot_id) if value.previous_snapshot_id is not None else None
        ),
        "identity_policy_version": value.identity_policy_version,
        "evidence_sha256": value.evidence_sha256,
    }


def growth_snapshot_id_from_content(content_sha256: str) -> UUID:
    raw = bytearray.fromhex(content_sha256[:32])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def uuid4_from_canonical_payload(value: object) -> UUID:
    """Derive a stable RFC-4122 UUID4-shaped identifier from canonical content."""

    from billiards_ai.shared.models.events import canonical_sha256

    return growth_snapshot_id_from_content(canonical_sha256(value))


class GrowthEvidenceSnapshotV2(CoreContractModel):
    """Immutable Story 8.4 fact snapshot; it contains no Coach conclusions."""

    schema_version: Literal["2.0"] = "2.0"
    snapshot_id: UUID4
    user_id: UUID4
    version: StrictInt = Field(ge=1)
    previous_snapshot_id: UUID4 | None
    evidence_cutoff: datetime
    created_at: datetime
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    snapshot_policy_version: Literal["growth-snapshot-rd-v1.0.0"] = SNAPSHOT_POLICY_VERSION
    identity_policy_version: Literal["growth-snapshot-identity-rd-v1.0.0"] = (
        SNAPSHOT_IDENTITY_POLICY_VERSION
    )
    source_batches: tuple[GrowthSnapshotSourceBatchRef, ...] = Field(min_length=1)
    source_bundle_sha256: Sha256
    metric_projection: MetricProjectionResult
    metric_projection_binding: GrowthSnapshotMetricProjectionBinding
    evidence_sha256: Sha256
    content_sha256: Sha256
    read_fence: GrowthSnapshotReadFence

    _cutoff_utc = field_validator("evidence_cutoff")(require_utc)
    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def snapshot_is_canonical_and_integral(self) -> GrowthEvidenceSnapshotV2:
        from billiards_ai.shared.models.events import canonical_sha256

        if self.created_at != self.evidence_cutoff:
            raise ValueError("deterministic snapshot creation time must equal its evidence cutoff")
        if (self.version == 1) != (self.previous_snapshot_id is None):
            raise ValueError("snapshot predecessor must match its version")
        if self.metric_projection.status is not MetricProjectionStatus.PROJECTED:
            raise ValueError("snapshot requires a successful metric projection")
        if self.metric_projection.evidence_cutoff != self.evidence_cutoff:
            raise ValueError("snapshot and metric projection cutoffs must match")
        if (
            self.metric_projection.taxonomy_version != self.taxonomy_version
            or self.metric_projection.calculation_policy_version != self.calculation_policy_version
        ):
            raise ValueError("snapshot and metric projection policies must match")
        if any(item.user_id != self.user_id for item in self.source_batches):
            raise ValueError("snapshot source batches must belong to one user")
        if self.source_batches != tuple(
            sorted(self.source_batches, key=growth_snapshot_source_sort_key)
        ):
            raise ValueError("snapshot source batches must use canonical order")
        expected_source_hash = canonical_sha256(
            [item.model_dump(mode="json") for item in self.source_batches]
        )
        if self.source_bundle_sha256 != expected_source_hash:
            raise ValueError("snapshot source bundle hash mismatch")
        if self.metric_projection_binding.source_bundle_sha256 != self.source_bundle_sha256:
            raise ValueError("metric projection is not bound to the snapshot source bundle")
        expected_projection_hash = canonical_sha256(self.metric_projection.model_dump(mode="json"))
        if self.metric_projection_binding.metric_projection_sha256 != expected_projection_hash:
            raise ValueError("metric projection binding does not match projection content")
        if self.metric_projection.input_batch_count != len(self.source_batches):
            raise ValueError("metric projection batch count does not match snapshot sources")
        if self.metric_projection.input_shot_count != sum(
            item.metric_shot_count for item in self.source_batches
        ):
            raise ValueError("metric projection shot count does not match snapshot sources")
        source_identities = {
            (item.admission_id, item.batch_id, item.session_id, item.review_revision)
            for item in self.source_batches
        }
        for reference in self.metric_projection.superseded_evidence:
            superseded = (
                reference.admission_id,
                reference.batch_id,
                reference.session_id,
                reference.review_revision,
            )
            replacement = (
                reference.superseded_by_admission_id,
                reference.superseded_by_batch_id,
                reference.session_id,
                reference.superseded_by_review_revision,
            )
            if superseded not in source_identities or replacement not in source_identities:
                raise ValueError("superseded metric audit is not present in snapshot sources")
        if self.evidence_sha256 != canonical_sha256(growth_snapshot_evidence_payload(self)):
            raise ValueError("snapshot evidence hash mismatch")
        if self.content_sha256 != canonical_sha256(growth_snapshot_content_payload(self)):
            raise ValueError("snapshot content hash mismatch")
        if self.snapshot_id != growth_snapshot_id_from_content(self.content_sha256):
            raise ValueError("snapshot identity does not match its content")
        expected_fence = GrowthSnapshotReadFence(
            user_id=self.user_id,
            snapshot_id=self.snapshot_id,
            version=self.version,
            content_sha256=self.content_sha256,
            evidence_sha256=self.evidence_sha256,
            taxonomy_version=self.taxonomy_version,
            calculation_policy_version=self.calculation_policy_version,
            snapshot_policy_version=self.snapshot_policy_version,
        )
        if self.read_fence != expected_fence:
            raise ValueError("snapshot read fence mismatch")
        return self


def growth_snapshot_v3_evidence_payload(value: GrowthEvidenceSnapshotV3) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "user_id": str(value.user_id),
        "evidence_cutoff": value.evidence_cutoff.isoformat(),
        "created_at": value.created_at.isoformat(),
        "taxonomy_version": value.taxonomy_version,
        "calculation_policy_version": value.calculation_policy_version,
        "snapshot_policy_version": value.snapshot_policy_version,
        "trend_policy_version": value.trend_policy_version,
        "source_batches": [item.model_dump(mode="json") for item in value.source_batches],
        "source_bundle_sha256": value.source_bundle_sha256,
        "metric_projection": value.metric_projection.model_dump(mode="json"),
        "metric_projection_binding": value.metric_projection_binding.model_dump(mode="json"),
        "metric_assessments": [item.model_dump(mode="json") for item in value.metric_assessments],
    }


def growth_snapshot_v3_content_payload(value: GrowthEvidenceSnapshotV3) -> dict[str, object]:
    return growth_snapshot_v3_evidence_payload(value) | {
        "version": value.version,
        "previous_snapshot_id": (
            str(value.previous_snapshot_id) if value.previous_snapshot_id is not None else None
        ),
        "identity_policy_version": value.identity_policy_version,
        "evidence_sha256": value.evidence_sha256,
    }


class GrowthEvidenceSnapshotV3(CoreContractModel):
    """Immutable, raw-free metric lineage and deterministic trend snapshot."""

    schema_version: Literal["3.0"] = "3.0"
    snapshot_id: UUID4
    user_id: UUID4
    version: StrictInt = Field(ge=1)
    previous_snapshot_id: UUID4 | None
    evidence_cutoff: datetime
    created_at: datetime
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    snapshot_policy_version: Literal["growth-snapshot-rd-v3.0.0"] = SNAPSHOT_POLICY_VERSION_V3
    identity_policy_version: Literal["growth-snapshot-identity-rd-v3.0.0"] = (
        SNAPSHOT_IDENTITY_POLICY_VERSION_V3
    )
    trend_policy_version: Literal["growth-trend-rd-v1.0.0"] = TREND_POLICY_VERSION
    source_batches: tuple[GrowthSnapshotSourceBatchRefV3, ...] = Field(min_length=1)
    source_bundle_sha256: Sha256
    metric_projection: MetricProjectionResult
    metric_projection_binding: GrowthSnapshotMetricProjectionBinding
    metric_assessments: tuple[GrowthMetricAssessmentV3, ...] = Field(min_length=1)
    evidence_sha256: Sha256
    content_sha256: Sha256
    read_fence: GrowthSnapshotReadFenceV3

    _cutoff_utc = field_validator("evidence_cutoff")(require_utc)
    _created_at_utc = field_validator("created_at")(require_utc)

    @model_validator(mode="after")
    def snapshot_is_canonical_and_integral(self) -> Self:
        from billiards_ai.shared.models.events import canonical_sha256

        if self.created_at != self.evidence_cutoff:
            raise ValueError("deterministic snapshot creation time must equal its evidence cutoff")
        if (self.version == 1) != (self.previous_snapshot_id is None):
            raise ValueError("snapshot predecessor must match its version")
        if self.metric_projection.status is not MetricProjectionStatus.PROJECTED:
            raise ValueError("snapshot requires a successful metric projection")
        if self.metric_projection.evidence_cutoff != self.evidence_cutoff:
            raise ValueError("snapshot and metric projection cutoffs must match")
        if any(item.user_id != self.user_id for item in self.source_batches):
            raise ValueError("snapshot source batches must belong to one user")
        if self.source_batches != tuple(
            sorted(self.source_batches, key=growth_snapshot_source_sort_key)
        ):
            raise ValueError("snapshot source batches must use canonical order")
        source_hash = canonical_sha256(
            [item.model_dump(mode="json") for item in self.source_batches]
        )
        if source_hash != self.source_bundle_sha256:
            raise ValueError("snapshot source bundle hash mismatch")
        if self.metric_projection_binding.source_bundle_sha256 != source_hash:
            raise ValueError("metric projection is not bound to snapshot sources")
        if self.metric_projection.input_batch_count != len(self.source_batches):
            raise ValueError("metric projection batch count does not match snapshot sources")
        if self.metric_projection.input_shot_count != sum(
            item.metric_shot_count for item in self.source_batches
        ):
            raise ValueError("metric projection shot count does not match snapshot sources")
        identities = tuple(item.metric_identity_sha256 for item in self.metric_assessments)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("metric assessments must use canonical unique identities")
        projection_identities = {
            canonical_sha256(
                {
                    "metric_code": metric.metric_code.value,
                    "taxonomy": metric.taxonomy.model_dump(mode="json"),
                    "calculation_policy_version": metric.calculation_policy_version,
                }
            )
            for metric in self.metric_projection.metrics
        }
        if set(identities) != projection_identities:
            raise ValueError("metric assessments must bind every projected metric exactly once")
        projection_by_identity = {
            _metric_identity(metric): metric for metric in self.metric_projection.metrics
        }
        source_identities = {
            (item.admission_id, item.batch_id, item.session_id, item.review_revision)
            for item in self.source_batches
        }
        for reference in self.metric_projection.superseded_evidence:
            superseded = (
                reference.admission_id,
                reference.batch_id,
                reference.session_id,
                reference.review_revision,
            )
            replacement = (
                reference.superseded_by_admission_id,
                reference.superseded_by_batch_id,
                reference.session_id,
                reference.superseded_by_review_revision,
            )
            if superseded not in source_identities or replacement not in source_identities:
                raise ValueError("superseded metric audit is not present in snapshot sources")
        if self.metric_projection_binding.metric_projection_sha256 != canonical_sha256(
            self.metric_projection.model_dump(mode="json")
        ):
            raise ValueError("metric projection binding mismatch")
        superseded_batch_ids = {
            item.batch_id for item in self.metric_projection.superseded_evidence
        }
        sources: dict[UUID, list[GrowthSnapshotSourceBatchRefV3]] = {}
        for source in self.source_batches:
            sources.setdefault(source.batch_id, []).append(source)
        all_ref_ids: list[UUID] = []
        for assessment in self.metric_assessments:
            GrowthMetricAssessmentV3.model_validate(
                assessment.model_dump(mode="python"), strict=True
            )
            if assessment.metric != projection_by_identity[assessment.metric_identity_sha256]:
                raise ValueError("assessment metric must equal projected metric")
            trend = assessment.trend
            if trend.current_window_end != self.evidence_cutoff:
                raise ValueError("trend policy mismatch: current window must end at cutoff")
            current_ids = set(trend.current_evidence_ref_ids)
            previous_ids = set(trend.previous_evidence_ref_ids)
            if current_ids & previous_ids:
                raise ValueError("trend evidence windows must be disjoint")
            approved_at_by_ref: dict[UUID, datetime] = {}
            for ref in assessment.evidence_refs:
                candidates = sources.get(ref.batch_id, [])
                source = next(
                    (item for item in candidates if item.session_id == ref.session_id),
                    None,
                )
                if source is None or ref.batch_id in superseded_batch_ids:
                    raise ValueError("metric evidence reference is not bound to its trusted source")
                if not (
                    self.evidence_cutoff - timedelta(days=365)
                    <= source.approved_at
                    <= self.evidence_cutoff
                ):
                    raise ValueError("metric evidence reference falls outside the 365-day window")
                bindings = tuple(
                    item
                    for item in source.metric_shot_bindings
                    if item.metric_identity_sha256 == assessment.metric_identity_sha256
                )
                if ref.shot_trace_bindings != bindings:
                    raise ValueError(
                        "metric evidence reference violates exact metric shot trace binding"
                    )
                if ref.evidence_ref_id in current_ids and not (
                    trend.current_window_start < source.approved_at <= trend.current_window_end
                ):
                    raise ValueError("current trend evidence is outside its approved_at window")
                if ref.evidence_ref_id in previous_ids and not (
                    trend.previous_window_start < source.approved_at <= trend.previous_window_end
                ):
                    raise ValueError("previous trend evidence is outside its approved_at window")
                approved_at_by_ref[ref.evidence_ref_id] = source.approved_at
                all_ref_ids.append(ref.evidence_ref_id)
            expected_current_ids = {
                ref_id
                for ref_id, approved_at in approved_at_by_ref.items()
                if trend.current_window_start < approved_at <= trend.current_window_end
            }
            expected_previous_ids = {
                ref_id
                for ref_id, approved_at in approved_at_by_ref.items()
                if trend.previous_window_start < approved_at <= trend.previous_window_end
            }
            if current_ids != expected_current_ids or previous_ids != expected_previous_ids:
                raise ValueError("trend evidence references must completely partition both windows")
            if (
                sum(len(ref.shot_ids) for ref in assessment.evidence_refs)
                != assessment.metric.sample_count
            ):
                raise ValueError("metric evidence references must reconcile with sample count")
            if (
                len({ref.session_id for ref in assessment.evidence_refs})
                != assessment.metric.distinct_session_count
            ):
                raise ValueError("metric evidence references must reconcile with session count")
        if len(all_ref_ids) != len(set(all_ref_ids)):
            raise ValueError("evidence reference ids cannot be reused across metrics")
        if self.evidence_sha256 != canonical_sha256(growth_snapshot_v3_evidence_payload(self)):
            raise ValueError("snapshot evidence hash mismatch")
        if self.content_sha256 != canonical_sha256(growth_snapshot_v3_content_payload(self)):
            raise ValueError("snapshot content hash mismatch")
        if self.snapshot_id != growth_snapshot_id_from_content(self.content_sha256):
            raise ValueError("snapshot identity does not match its content")
        expected_fence = GrowthSnapshotReadFenceV3(
            user_id=self.user_id,
            snapshot_id=self.snapshot_id,
            version=self.version,
            content_sha256=self.content_sha256,
            evidence_sha256=self.evidence_sha256,
            taxonomy_version=self.taxonomy_version,
            calculation_policy_version=self.calculation_policy_version,
            snapshot_policy_version=self.snapshot_policy_version,
            trend_policy_version=self.trend_policy_version,
        )
        if self.read_fence != expected_fence:
            raise ValueError("snapshot read fence mismatch")
        return self


GrowthEvidenceSnapshotAny = (
    GrowthEvidenceSnapshot | GrowthEvidenceSnapshotV2 | GrowthEvidenceSnapshotV3
)


class PlanOutcomeStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    ABORTED = "ABORTED"


class BlockOutcome(CoreContractModel):
    block_id: UUID4
    attempted_shots: StrictInt = Field(ge=0)
    completed_shots: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def completions_do_not_exceed_attempts(self) -> BlockOutcome:
        if self.completed_shots > self.attempted_shots:
            raise ValueError("completed shots cannot exceed attempted shots")
        return self


class PlanOutcome(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    outcome_id: UUID4
    plan_id: UUID4
    plan_version: StrictInt = Field(ge=1)
    user_id: UUID4
    session_id: UUID4
    device_id: NonEmptyStr
    started_at: datetime
    completed_at: datetime
    status: PlanOutcomeStatus
    blocks: tuple[BlockOutcome, ...]
    evidence_envelope_ids: tuple[UUID4, ...]
    direct_ability_update_allowed: Literal[False] = False
    content_sha256: Sha256

    _started_at_utc = field_validator("started_at")(require_utc)
    _completed_at_utc = field_validator("completed_at")(require_utc)

    @model_validator(mode="after")
    def execution_timeline_and_ids_are_consistent(self) -> PlanOutcome:
        if self.completed_at < self.started_at:
            raise ValueError("outcome completion must not precede start")
        block_ids = tuple(item.block_id for item in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block outcomes must be unique")
        if len(self.evidence_envelope_ids) != len(set(self.evidence_envelope_ids)):
            raise ValueError("evidence envelope ids must be unique")
        return self
