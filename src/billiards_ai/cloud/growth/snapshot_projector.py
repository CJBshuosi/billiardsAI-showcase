"""Pure deterministic GrowthEvidenceSnapshot v2 projection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID

from pydantic import ValidationError

from billiards_ai.cloud.growth.metric_projector import (
    MetricObservationTrace,
    project_ability_metrics,
    project_ability_metrics_with_trace,
)
from billiards_ai.cloud.growth.repository import stored_model_values
from billiards_ai.cloud.growth.taxonomy import round_half_up
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import AcceptedMetricEvidenceInput
from billiards_ai.shared.models.growth import (
    SNAPSHOT_IDENTITY_POLICY_VERSION_V3,
    SNAPSHOT_POLICY_VERSION_V3,
    TREND_POLICY_VERSION,
    GrowthEvidenceSnapshotV2,
    GrowthEvidenceSnapshotV3,
    GrowthMetricAssessmentV3,
    GrowthSnapshotLineage,
    GrowthSnapshotLineageV3,
    GrowthSnapshotMetricProjectionBinding,
    GrowthSnapshotReadFence,
    GrowthSnapshotReadFenceV3,
    GrowthSnapshotSourceBatchRef,
    GrowthSnapshotSourceBatchRefV3,
    MetricEvidenceReferenceV1,
    MetricShotTraceBindingV1,
    MetricSupportStatus,
    MetricTrendV1,
    MetricTrendWindowSummaryV1,
    TrendDirection,
    growth_snapshot_content_payload,
    growth_snapshot_evidence_payload,
    growth_snapshot_id_from_content,
    growth_snapshot_source_sort_key,
    growth_snapshot_v3_content_payload,
    growth_snapshot_v3_evidence_payload,
    uuid4_from_canonical_payload,
)
from billiards_ai.shared.models.growth_metrics import (
    MetricCode,
    MetricProjectionStatus,
    MetricReliability,
    MetricSupportStatusV2,
    ProjectedAbilityMetric,
)


class SnapshotProjectionErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    METRIC_PROJECTION_FAILED = "METRIC_PROJECTION_FAILED"


class SnapshotProjectionError(ValueError):
    def __init__(self, code: SnapshotProjectionErrorCode):
        self.code = code
        super().__init__(code.value)


def _strict_public_admission(
    value: AcceptedMetricEvidenceInput,
) -> AcceptedMetricEvidenceInput:
    if not isinstance(value, AcceptedMetricEvidenceInput) or not value.is_trusted_admission:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
    try:
        return AcceptedMetricEvidenceInput.model_validate(stored_model_values(value), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT) from exc


def _source_ref(value: AcceptedMetricEvidenceInput) -> GrowthSnapshotSourceBatchRef:
    batch = value.batch
    return GrowthSnapshotSourceBatchRef(
        admission_schema_version=value.schema_version,
        admission_id=value.admission_id,
        admitted_at=value.admitted_at,
        batch_schema_version=batch.schema_version,
        batch_id=batch.batch_id,
        user_id=batch.user_id,
        session_id=batch.session_id,
        report_id=batch.report_id,
        report_revision=batch.report_revision,
        review_revision=batch.review_revision,
        approved_at=batch.approved_at,
        approval_kind=batch.approval_kind.value,
        approval_policy_version=batch.approval_policy_version,
        batch_content_sha256=batch.content_sha256,
        source_envelope_ids=batch.source_envelope_ids,
        metric_shot_count=len(batch.metric_shots),
    )


def _source_ref_v3(
    value: AcceptedMetricEvidenceInput,
    traces: tuple[MetricObservationTrace, ...],
) -> GrowthSnapshotSourceBatchRefV3:
    bindings = tuple(
        sorted(
            (
                MetricShotTraceBindingV1(
                    metric_identity_sha256=_trace_identity(item),
                    shot_id=item.shot_id,
                    source_envelope_id=item.source_envelope_ids[0],
                    calculation_field_codes=item.calculation_field_codes,
                )
                for item in traces
                if item.batch_id == value.batch_id
            ),
            key=lambda item: (item.metric_identity_sha256, item.shot_id),
        )
    )
    return GrowthSnapshotSourceBatchRefV3(
        **_source_ref(value).model_dump(mode="python"),
        approved_metric_shot_ids=tuple(sorted(item.shot_id for item in value.batch.metric_shots)),
        metric_shot_bindings=bindings,
    )


def _metric_identity(metric: ProjectedAbilityMetric) -> str:
    return canonical_sha256(
        {
            "metric_code": metric.metric_code.value,
            "taxonomy": metric.taxonomy.model_dump(mode="json"),
            "calculation_policy_version": metric.calculation_policy_version,
        }
    )


def _trace_identity(trace: MetricObservationTrace) -> str:
    return canonical_sha256(
        {
            "metric_code": trace.metric_code.value,
            "taxonomy": trace.taxonomy.model_dump(mode="json"),
            "calculation_policy_version": "ability-metrics-rd-v1.0.0",
        }
    )


def _canonical_latest_admissions(
    values: tuple[AcceptedMetricEvidenceInput, ...],
) -> tuple[AcceptedMetricEvidenceInput, ...]:
    by_revision: dict[tuple[object, int], AcceptedMetricEvidenceInput] = {}
    by_session: dict[object, AcceptedMetricEvidenceInput] = {}
    for value in values:
        identity = (value.session_id, value.review_revision)
        existing = by_revision.get(identity)
        if existing is not None:
            if existing != value:
                raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
            continue
        by_revision[identity] = value
        current = by_session.get(value.session_id)
        if current is None or value.review_revision > current.review_revision:
            by_session[value.session_id] = value
    return tuple(
        sorted(
            by_session.values(),
            key=lambda item: (
                item.batch.approved_at,
                str(item.session_id),
                item.review_revision,
                str(item.batch_id),
                str(item.admission_id),
            ),
        )
    )


def _window_projection(
    values: tuple[AcceptedMetricEvidenceInput, ...],
    *,
    start: datetime,
    end: datetime,
):
    window = tuple(item for item in values if start < item.batch.approved_at <= end)
    if not window:
        return None
    result = project_ability_metrics_with_trace(window, evidence_cutoff=end)
    if result.projection.status is not MetricProjectionStatus.PROJECTED:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.METRIC_PROJECTION_FAILED)
    return result


def _references(
    traces: tuple[MetricObservationTrace, ...],
    metric_identity_sha256: str,
) -> tuple[MetricEvidenceReferenceV1, ...]:
    grouped: dict[tuple[object, object], list[MetricObservationTrace]] = {}
    for trace in traces:
        if _trace_identity(trace) == metric_identity_sha256:
            grouped.setdefault((trace.batch_id, trace.session_id), []).append(trace)
    references = []
    for (batch_id, session_id), values in grouped.items():
        payload = {
            "schema_version": "1.0",
            "metric_identity_sha256": metric_identity_sha256,
            "batch_id": batch_id,
            "session_id": session_id,
            "shot_ids": tuple(sorted({item.shot_id for item in values})),
            "source_envelope_ids": tuple(
                sorted({envelope for value in values for envelope in value.source_envelope_ids})
            ),
            "calculation_field_codes": tuple(
                sorted({code for value in values for code in value.calculation_field_codes})
            ),
            "shot_trace_bindings": tuple(
                sorted(
                    (
                        MetricShotTraceBindingV1(
                            metric_identity_sha256=metric_identity_sha256,
                            shot_id=item.shot_id,
                            source_envelope_id=item.source_envelope_ids[0],
                            calculation_field_codes=item.calculation_field_codes,
                        )
                        for item in values
                    ),
                    key=lambda item: item.shot_id,
                )
            ),
        }
        references.append(
            MetricEvidenceReferenceV1.model_validate(
                payload | {"evidence_ref_id": uuid4_from_canonical_payload(payload)}
            )
        )
    return tuple(sorted(references, key=lambda item: item.evidence_ref_id))


def _metric_by_identity(result, identity: str) -> ProjectedAbilityMetric | None:
    if result is None:
        return None
    return next(
        (item for item in result.projection.metrics if _metric_identity(item) == identity),
        None,
    )


def _unknown_support(
    current_family: tuple[ProjectedAbilityMetric, ...],
    previous_family: tuple[ProjectedAbilityMetric, ...],
) -> MetricSupportStatus:
    statuses = {item.support_status for item in current_family + previous_family}
    if MetricSupportStatusV2.UNSUPPORTED in statuses:
        return MetricSupportStatus.UNSUPPORTED
    if MetricSupportStatusV2.ABSTAINED in statuses:
        return MetricSupportStatus.ABSTAINED
    return MetricSupportStatus.INSUFFICIENT_EVIDENCE


def _family_metrics(result, metric_code: MetricCode) -> tuple[ProjectedAbilityMetric, ...]:
    if result is None:
        return ()
    return tuple(
        sorted(
            (item for item in result.projection.metrics if item.metric_code is metric_code),
            key=_metric_identity,
        )
    )


def _window_summary(
    *,
    result,
    metric: ProjectedAbilityMetric,
    identity: str,
    start: datetime,
    end: datetime,
    refs: tuple[MetricEvidenceReferenceV1, ...],
) -> MetricTrendWindowSummaryV1:
    return MetricTrendWindowSummaryV1(
        window_start=start,
        window_end=end,
        metric_identity_sha256=identity,
        metric_code=metric.metric_code,
        comparable_metric=_metric_by_identity(result, identity),
        family_metrics=_family_metrics(result, metric.metric_code),
        evidence_ref_ids=tuple(item.evidence_ref_id for item in refs),
    )


def _trend(
    metric: ProjectedAbilityMetric,
    identity: str,
    *,
    previous_result,
    current_result,
    previous_start: datetime,
    previous_end: datetime,
    current_start: datetime,
    current_end: datetime,
) -> MetricTrendV1:
    previous = _metric_by_identity(previous_result, identity)
    current = _metric_by_identity(current_result, identity)
    previous_refs = _references(
        () if previous_result is None else previous_result.observations, identity
    )
    current_refs = _references(
        () if current_result is None else current_result.observations, identity
    )
    previous_summary = _window_summary(
        result=previous_result,
        metric=metric,
        identity=identity,
        start=previous_start,
        end=previous_end,
        refs=previous_refs,
    )
    current_summary = _window_summary(
        result=current_result,
        metric=metric,
        identity=identity,
        start=current_start,
        end=current_end,
        refs=current_refs,
    )
    qualified = all(
        item is not None
        and item.support_status is MetricSupportStatusV2.SUPPORTED
        and item.sample_count >= 5
        and item.distinct_session_count >= 2
        for item in (previous, current)
    )
    field_code = (
        "error_p50_mm" if metric.metric_code is MetricCode.TARGET_ZONE_ERROR_MM else "success_rate"
    )
    previous_value = None if previous is None else getattr(previous, field_code)
    current_value = None if current is None else getattr(current, field_code)
    if not qualified or previous_value is None or current_value is None:
        return MetricTrendV1(
            direction=TrendDirection.UNKNOWN,
            support_status=_unknown_support(
                current_summary.family_metrics,
                previous_summary.family_metrics,
            ),
            current_window_start=current_start,
            current_window_end=current_end,
            previous_window_start=previous_start,
            previous_window_end=previous_end,
            reliability=MetricReliability.UNKNOWN,
            current_evidence_ref_ids=tuple(item.evidence_ref_id for item in current_refs),
            previous_evidence_ref_ids=tuple(item.evidence_ref_id for item in previous_refs),
            current_window_summary=current_summary,
            previous_window_summary=previous_summary,
        )
    digits = 3 if field_code == "error_p50_mm" else 6
    delta = round_half_up(float(current_value) - float(previous_value), digits)
    threshold = 25.0 if field_code == "error_p50_mm" else 0.05
    if abs(delta) < threshold:
        direction = TrendDirection.STABLE
    elif field_code == "error_p50_mm":
        direction = TrendDirection.IMPROVING if delta <= -threshold else TrendDirection.DECLINING
    else:
        direction = TrendDirection.IMPROVING if delta >= threshold else TrendDirection.DECLINING
    reliability = min(
        (previous.reliability, current.reliability),
        key=lambda item: {
            MetricReliability.LOW: 0,
            MetricReliability.MEDIUM: 1,
            MetricReliability.HIGH: 2,
        }[item],
    )
    return MetricTrendV1(
        direction=direction,
        support_status=MetricSupportStatus.SUPPORTED,
        current_window_start=current_start,
        current_window_end=current_end,
        previous_window_start=previous_start,
        previous_window_end=previous_end,
        current_value=round_half_up(float(current_value), digits),
        previous_value=round_half_up(float(previous_value), digits),
        delta=delta,
        reliability=reliability,
        current_evidence_ref_ids=tuple(item.evidence_ref_id for item in current_refs),
        previous_evidence_ref_ids=tuple(item.evidence_ref_id for item in previous_refs),
        current_window_summary=current_summary,
        previous_window_summary=previous_summary,
    )


def project_growth_snapshot(
    inputs: tuple[AcceptedMetricEvidenceInput, ...],
    *,
    evidence_cutoff: datetime,
    lineage: GrowthSnapshotLineage,
) -> GrowthEvidenceSnapshotV2:
    """Create a byte-stable snapshot without I/O, clock, or randomness."""

    if (
        evidence_cutoff.tzinfo is None
        or evidence_cutoff.utcoffset() != UTC.utcoffset(evidence_cutoff)
        or not inputs
    ):
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
    evidence_cutoff = evidence_cutoff.astimezone(UTC)
    try:
        lineage = GrowthSnapshotLineage.model_validate(stored_model_values(lineage), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT) from exc

    materialized_inputs = tuple(inputs)
    validated = tuple(_strict_public_admission(item) for item in materialized_inputs)
    user_ids = {item.batch.user_id for item in validated}
    if len(user_ids) != 1:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
    projection = project_ability_metrics(materialized_inputs, evidence_cutoff=evidence_cutoff)
    if projection.status is not MetricProjectionStatus.PROJECTED:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.METRIC_PROJECTION_FAILED)

    sources = tuple(
        sorted((_source_ref(item) for item in validated), key=growth_snapshot_source_sort_key)
    )
    source_hash = canonical_sha256([item.model_dump(mode="json") for item in sources])
    placeholder = GrowthEvidenceSnapshotV2.model_construct(
        schema_version="2.0",
        snapshot_id=UUID("00000000-0000-4000-8000-000000000000"),
        user_id=validated[0].batch.user_id,
        version=lineage.version,
        previous_snapshot_id=lineage.previous_snapshot_id,
        evidence_cutoff=evidence_cutoff,
        created_at=evidence_cutoff,
        taxonomy_version=projection.taxonomy_version,
        calculation_policy_version=projection.calculation_policy_version,
        snapshot_policy_version=lineage.snapshot_policy_version,
        identity_policy_version="growth-snapshot-identity-rd-v1.0.0",
        source_batches=sources,
        source_bundle_sha256=source_hash,
        metric_projection=projection,
        metric_projection_binding=GrowthSnapshotMetricProjectionBinding(
            source_bundle_sha256=source_hash,
            metric_projection_sha256=canonical_sha256(projection.model_dump(mode="json")),
        ),
        evidence_sha256="0" * 64,
        content_sha256="0" * 64,
        read_fence=None,
    )
    evidence_hash = canonical_sha256(growth_snapshot_evidence_payload(placeholder))
    placeholder = placeholder.model_copy(update={"evidence_sha256": evidence_hash})
    content_hash = canonical_sha256(growth_snapshot_content_payload(placeholder))
    snapshot_id = growth_snapshot_id_from_content(content_hash)
    fence = GrowthSnapshotReadFence(
        user_id=validated[0].batch.user_id,
        snapshot_id=snapshot_id,
        version=lineage.version,
        content_sha256=content_hash,
        evidence_sha256=evidence_hash,
        taxonomy_version=projection.taxonomy_version,
        calculation_policy_version=projection.calculation_policy_version,
        snapshot_policy_version=lineage.snapshot_policy_version,
    )
    return GrowthEvidenceSnapshotV2.model_validate(
        placeholder.model_dump(mode="python")
        | {
            "snapshot_id": snapshot_id,
            "evidence_sha256": evidence_hash,
            "content_sha256": content_hash,
            "read_fence": fence,
        },
        strict=True,
    )


def project_growth_snapshot_v3(
    inputs: tuple[AcceptedMetricEvidenceInput, ...],
    *,
    evidence_cutoff: datetime,
    lineage: GrowthSnapshotLineageV3,
) -> GrowthEvidenceSnapshotV3:
    """Create a byte-stable V3 snapshot with adjacent-window per-metric lineage."""

    if (
        evidence_cutoff.tzinfo is None
        or evidence_cutoff.utcoffset() != UTC.utcoffset(evidence_cutoff)
        or not inputs
    ):
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
    evidence_cutoff = evidence_cutoff.astimezone(UTC)
    try:
        lineage = GrowthSnapshotLineageV3.model_validate(stored_model_values(lineage), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT) from exc
    materialized_inputs = tuple(inputs)
    validated = tuple(_strict_public_admission(item) for item in materialized_inputs)
    if len({item.batch.user_id for item in validated}) != 1:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.INVALID_INPUT)
    full = project_ability_metrics_with_trace(materialized_inputs, evidence_cutoff=evidence_cutoff)
    if full.projection.status is not MetricProjectionStatus.PROJECTED:
        raise SnapshotProjectionError(SnapshotProjectionErrorCode.METRIC_PROJECTION_FAILED)

    latest = _canonical_latest_admissions(materialized_inputs)
    current_start = evidence_cutoff - timedelta(days=90)
    previous_start = evidence_cutoff - timedelta(days=180)
    previous_result = _window_projection(
        latest,
        start=previous_start,
        end=current_start,
    )
    current_result = _window_projection(
        latest,
        start=current_start,
        end=evidence_cutoff,
    )
    assessments = []
    for metric in full.projection.metrics:
        identity = _metric_identity(metric)
        trend = _trend(
            metric,
            identity,
            previous_result=previous_result,
            current_result=current_result,
            previous_start=previous_start,
            previous_end=current_start,
            current_start=current_start,
            current_end=evidence_cutoff,
        )
        refs = _references(full.observations, identity)
        assessments.append(
            GrowthMetricAssessmentV3(
                metric_identity_sha256=identity,
                metric_value_field_code=(
                    "error_p50_mm"
                    if metric.metric_code is MetricCode.TARGET_ZONE_ERROR_MM
                    else "success_rate"
                ),
                metric=metric,
                evidence_refs=refs,
                trend=trend,
            )
        )
    assessments = sorted(assessments, key=lambda item: item.metric_identity_sha256)
    sources = tuple(
        sorted(
            (_source_ref_v3(item, full.observations) for item in validated),
            key=growth_snapshot_source_sort_key,
        )
    )
    source_hash = canonical_sha256([item.model_dump(mode="json") for item in sources])
    placeholder = GrowthEvidenceSnapshotV3.model_construct(
        schema_version="3.0",
        snapshot_id=UUID("00000000-0000-4000-8000-000000000000"),
        user_id=validated[0].batch.user_id,
        version=lineage.version,
        previous_snapshot_id=lineage.previous_snapshot_id,
        evidence_cutoff=evidence_cutoff,
        created_at=evidence_cutoff,
        taxonomy_version=full.projection.taxonomy_version,
        calculation_policy_version=full.projection.calculation_policy_version,
        snapshot_policy_version=SNAPSHOT_POLICY_VERSION_V3,
        identity_policy_version=SNAPSHOT_IDENTITY_POLICY_VERSION_V3,
        trend_policy_version=TREND_POLICY_VERSION,
        source_batches=sources,
        source_bundle_sha256=source_hash,
        metric_projection=full.projection,
        metric_projection_binding=GrowthSnapshotMetricProjectionBinding(
            source_bundle_sha256=source_hash,
            metric_projection_sha256=canonical_sha256(full.projection.model_dump(mode="json")),
        ),
        metric_assessments=tuple(assessments),
        evidence_sha256="0" * 64,
        content_sha256="0" * 64,
        read_fence=None,
    )
    evidence_hash = canonical_sha256(growth_snapshot_v3_evidence_payload(placeholder))
    placeholder = placeholder.model_copy(update={"evidence_sha256": evidence_hash})
    content_hash = canonical_sha256(growth_snapshot_v3_content_payload(placeholder))
    snapshot_id = growth_snapshot_id_from_content(content_hash)
    fence = GrowthSnapshotReadFenceV3(
        user_id=placeholder.user_id,
        snapshot_id=snapshot_id,
        version=placeholder.version,
        content_sha256=content_hash,
        evidence_sha256=evidence_hash,
        taxonomy_version=placeholder.taxonomy_version,
        calculation_policy_version=placeholder.calculation_policy_version,
        snapshot_policy_version=placeholder.snapshot_policy_version,
        trend_policy_version=placeholder.trend_policy_version,
    )
    return GrowthEvidenceSnapshotV3.model_validate(
        placeholder.model_dump(mode="python")
        | {
            "snapshot_id": snapshot_id,
            "evidence_sha256": evidence_hash,
            "content_sha256": content_hash,
            "read_fence": fence,
        },
        strict=True,
    )


__all__ = [
    "SnapshotProjectionError",
    "SnapshotProjectionErrorCode",
    "project_growth_snapshot",
    "project_growth_snapshot_v3",
]
