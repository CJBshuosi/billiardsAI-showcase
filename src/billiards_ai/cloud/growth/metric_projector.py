"""Pure deterministic Story 8.3 ability metric projection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.cloud.growth.taxonomy import (
    CALCULATION_POLICY_VERSION,
    TAXONOMY_VERSION,
    classify_angle_band,
    classify_distance_band,
    classify_rail_relation,
    cut_angle_deg,
    round_half_up,
    shot_distance_mm,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.evidence import (
    AcceptedMetricEvidenceInput,
    ApprovedEvidenceBatch,
    ApprovedEvidenceBatchV2,
)
from billiards_ai.shared.models.growth_metrics import (
    AbilityDimension,
    AbilityTaxonomyV2,
    AngleBand,
    DistanceBand,
    MetricCode,
    MetricProjectionResult,
    MetricProjectionStatus,
    MetricReliability,
    MetricSupportStatusV2,
    ProjectedAbilityMetric,
    RailRelation,
    SupersededEvidenceReference,
    TaxonomyApplicability,
    TechniqueCode,
)
from billiards_ai.shared.models.metric_evidence import (
    ApprovedMetricShotEvidence,
    EvidenceApprovalKind,
    FoulResult,
    PotAttemptStatus,
    PotResult,
    TargetZoneEvaluationStatus,
    TargetZoneResult,
)
from billiards_ai.shared.models.shot import TargetZoneSource

_SECONDS_PER_DAY = 86_400.0
_RECENCY_HALF_LIFE_DAYS = 90.0
_MAX_AGE_DAYS = 365.0
_Z_95 = 1.959963984540054


def _stored_values(value: object) -> object:
    if isinstance(value, BaseModel):
        values = dict(vars(value))
        if value.__pydantic_extra__:
            values.update(value.__pydantic_extra__)
        return {key: _stored_values(item) for key, item in values.items()}
    if isinstance(value, tuple):
        return tuple(_stored_values(item) for item in value)
    if isinstance(value, list):
        return [_stored_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _stored_values(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class _Observation:
    session_id: UUID | str
    weight: float
    value: float
    batch_id: UUID | None = None
    shot_id: UUID | None = None
    source_envelope_ids: tuple[UUID, ...] = ()
    calculation_field_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricObservationTrace:
    """Raw-free lineage captured in the same pass that produces a metric value."""

    metric_code: MetricCode
    taxonomy: AbilityTaxonomyV2
    batch_id: UUID
    session_id: UUID
    shot_id: UUID
    source_envelope_ids: tuple[UUID, ...]
    calculation_field_codes: tuple[str, ...]


@dataclass(frozen=True)
class MetricProjectionWithTrace:
    projection: MetricProjectionResult
    observations: tuple[MetricObservationTrace, ...]


@dataclass
class _Family:
    eligible: int = 0
    supported: int = 0
    unsupported: int = 0
    abstained: int = 0
    groups: dict[AbilityTaxonomyV2, list[_Observation]] = field(
        default_factory=lambda: defaultdict(list)
    )
    source_counts: dict[TargetZoneSource, _Availability] = field(
        default_factory=lambda: defaultdict(_Availability)
    )


@dataclass
class _Availability:
    eligible: int = 0
    supported: int = 0
    unsupported: int = 0
    abstained: int = 0


def _taxonomy(
    dimension: AbilityDimension,
    technique: TechniqueCode,
    *,
    angle: AngleBand | None = None,
    distance: DistanceBand | None = None,
    rail: RailRelation | None = None,
    source: TargetZoneSource | None = None,
) -> AbilityTaxonomyV2:
    return AbilityTaxonomyV2(
        ability_dimension=dimension,
        angle_band=angle,
        angle_applicability=(
            TaxonomyApplicability.APPLICABLE
            if angle is not None
            else TaxonomyApplicability.NOT_APPLICABLE
        ),
        distance_band=distance,
        distance_applicability=(
            TaxonomyApplicability.APPLICABLE
            if distance is not None
            else TaxonomyApplicability.NOT_APPLICABLE
        ),
        rail_relation=rail,
        rail_applicability=(
            TaxonomyApplicability.APPLICABLE
            if rail is not None
            else TaxonomyApplicability.NOT_APPLICABLE
        ),
        target_zone_source=source,
        target_zone_applicability=(
            TaxonomyApplicability.APPLICABLE
            if source is not None
            else TaxonomyApplicability.NOT_APPLICABLE
        ),
        technique_code=technique,
        technique_applicability=TaxonomyApplicability.APPLICABLE,
    )


def _summary_taxonomy(
    code: MetricCode,
    status: MetricSupportStatusV2,
    *,
    source: TargetZoneSource | None = None,
) -> AbilityTaxonomyV2:
    if code is MetricCode.POTTING_SUCCESS_RATE:
        applicability = {
            MetricSupportStatusV2.UNSUPPORTED: TaxonomyApplicability.UNSUPPORTED,
            MetricSupportStatusV2.ABSTAINED: TaxonomyApplicability.ABSTAINED,
        }.get(status, TaxonomyApplicability.NOT_APPLICABLE)
        return AbilityTaxonomyV2(
            ability_dimension=AbilityDimension.POTTING,
            angle_applicability=applicability,
            distance_applicability=applicability,
            rail_applicability=applicability,
            target_zone_applicability=TaxonomyApplicability.NOT_APPLICABLE,
            technique_applicability=applicability,
        )
    if code in {MetricCode.TARGET_ZONE_SUCCESS_RATE, MetricCode.TARGET_ZONE_ERROR_MM}:
        return _taxonomy(
            AbilityDimension.POSITIONING,
            TechniqueCode.TARGET_ZONE_POSITION,
            source=source,
        )
    return _taxonomy(AbilityDimension.FOUL_CONTROL, TechniqueCode.FOUL_AVOIDANCE)


def _approval_weight(batch: ApprovedEvidenceBatchV2) -> float:
    if batch.approval_kind is EvidenceApprovalKind.HUMAN_REVIEW:
        if batch.approval_policy_version != "human-review-v1":
            raise ValueError("unknown human review policy")
        return 1.0
    if batch.approval_policy_version != "automated-policy-rd-v1":
        raise ValueError("unknown automated approval policy")
    return 0.8


def _age_days(batch: ApprovedEvidenceBatchV2, cutoff: datetime) -> float:
    return (cutoff - batch.approved_at).total_seconds() / _SECONDS_PER_DAY


def _weight(
    batch: ApprovedEvidenceBatchV2,
    cutoff: datetime,
    confidences: tuple[float, ...],
) -> float:
    field_confidence = min(confidences)
    recency = 2 ** (-_age_days(batch, cutoff) / _RECENCY_HALF_LIFE_DAYS)
    return _approval_weight(batch) * field_confidence * recency


def _source_envelope_index(batch: ApprovedEvidenceBatchV2) -> dict[UUID, UUID]:
    """Build the frozen one-to-one shot/envelope binding once per batch."""

    index = dict(
        zip(
            (item.shot_id for item in batch.metric_shots),
            batch.source_envelope_ids,
            strict=True,
        )
    )
    if len(index) != len(batch.metric_shots):
        raise ValueError("metric shots must bind unique source envelopes")
    return index


_BATCH_WEIGHT_FIELDS = (
    "batch.approval_kind",
    "batch.approval_policy_version",
    "batch.approved_at",
)


def _pot_taxonomy(shot: ApprovedMetricShotEvidence) -> AbilityTaxonomyV2 | None:
    if (
        shot.cue_ball_start_mm is None
        or shot.object_ball_start_mm is None
        or shot.intended_target_pocket is None
    ):
        return None
    try:
        angle = classify_angle_band(
            cut_angle_deg(
                shot.cue_ball_start_mm,
                shot.object_ball_start_mm,
                shot.intended_target_pocket,
            )
        )
        distance = classify_distance_band(
            shot_distance_mm(shot.cue_ball_start_mm, shot.object_ball_start_mm)
        )
        rail = classify_rail_relation(shot.object_ball_start_mm, shot.table_geometry)
    except ValueError:
        return None
    if rail is None:
        return None
    technique = TechniqueCode.DIRECT_POT if angle is AngleBand.STRAIGHT else TechniqueCode.CUT_POT
    return _taxonomy(
        AbilityDimension.POTTING,
        technique,
        angle=angle,
        distance=distance,
        rail=rail,
    )


def _add_pot(
    family: _Family,
    shot: ApprovedMetricShotEvidence,
    batch: ApprovedEvidenceBatchV2,
    cutoff: datetime,
    source_envelope_id: UUID,
) -> None:
    status = shot.pot_attempt_status
    if status is PotAttemptStatus.NOT_ATTEMPTED:
        return
    family.eligible += 1
    if status is PotAttemptStatus.UNSUPPORTED:
        family.unsupported += 1
        return
    qualities = shot.field_quality
    confidences = (
        qualities.geometry.confidence,
        qualities.intent.confidence,
        qualities.pot_attempt.confidence,
        qualities.pot_result.confidence,
    )
    if status is PotAttemptStatus.ABSTAINED or min(confidences) < 0.5:
        family.abstained += 1
        return
    taxonomy = _pot_taxonomy(shot)
    if taxonomy is None or shot.pot_result is None:
        family.unsupported += 1
        return
    family.supported += 1
    family.groups[taxonomy].append(
        _Observation(
            batch_id=batch.batch_id,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            source_envelope_ids=(source_envelope_id,),
            calculation_field_codes=tuple(
                sorted(
                    _BATCH_WEIGHT_FIELDS
                    + (
                        "shot.cue_ball_start_mm",
                        "shot.field_quality.geometry.confidence",
                        "shot.field_quality.intent.confidence",
                        "shot.field_quality.pot_attempt.confidence",
                        "shot.field_quality.pot_result.confidence",
                        "shot.intended_target_pocket",
                        "shot.object_ball_start_mm",
                        "shot.pot_attempt_status",
                        "shot.pot_result",
                        "shot.session_id",
                        "shot.table_geometry",
                    )
                )
            ),
            weight=_weight(batch, cutoff, confidences),
            value=float(shot.pot_result is PotResult.LEGAL_SUCCESS),
        )
    )


def _add_target_zone(
    success: _Family,
    error: _Family,
    shot: ApprovedMetricShotEvidence,
    batch: ApprovedEvidenceBatchV2,
    cutoff: datetime,
    source_envelope_id: UUID,
) -> None:
    status = shot.target_zone_evaluation_status
    if status is TargetZoneEvaluationStatus.NOT_APPLICABLE:
        return
    source = shot.target_zone_source
    if source is TargetZoneSource.NONE:
        return
    families = (success, error)
    for family in families:
        family.eligible += 1
        family.source_counts[source].eligible += 1
    if status is TargetZoneEvaluationStatus.UNSUPPORTED:
        for family in families:
            family.unsupported += 1
            family.source_counts[source].unsupported += 1
        return
    qualities = shot.field_quality
    confidences = (qualities.target_zone.confidence, qualities.cue_ball_end.confidence)
    if status is TargetZoneEvaluationStatus.ABSTAINED or min(confidences) < 0.5:
        for family in families:
            family.abstained += 1
            family.source_counts[source].abstained += 1
        return
    if shot.target_zone is None or shot.cue_ball_end_mm is None or shot.target_zone_result is None:
        for family in families:
            family.unsupported += 1
            family.source_counts[source].unsupported += 1
        return
    taxonomy = _taxonomy(
        AbilityDimension.POSITIONING,
        TechniqueCode.TARGET_ZONE_POSITION,
        source=source,
    )
    weight = _weight(batch, cutoff, confidences)
    for family in families:
        family.supported += 1
        family.source_counts[source].supported += 1
    success.groups[taxonomy].append(
        _Observation(
            batch_id=batch.batch_id,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            source_envelope_ids=(source_envelope_id,),
            calculation_field_codes=tuple(
                sorted(
                    _BATCH_WEIGHT_FIELDS
                    + (
                        "shot.cue_ball_end_mm",
                        "shot.field_quality.cue_ball_end.confidence",
                        "shot.field_quality.target_zone.confidence",
                        "shot.session_id",
                        "shot.target_zone",
                        "shot.target_zone_evaluation_status",
                        "shot.target_zone_result",
                        "shot.target_zone_source",
                    )
                )
            ),
            weight=weight,
            value=float(shot.target_zone_result is TargetZoneResult.INSIDE_OR_TOLERATED),
        )
    )
    error.groups[taxonomy].append(
        _Observation(
            batch_id=batch.batch_id,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            source_envelope_ids=(source_envelope_id,),
            calculation_field_codes=tuple(
                sorted(
                    _BATCH_WEIGHT_FIELDS
                    + (
                        "shot.cue_ball_end_mm",
                        "shot.field_quality.cue_ball_end.confidence",
                        "shot.field_quality.target_zone.confidence",
                        "shot.session_id",
                        "shot.target_zone",
                        "shot.target_zone_evaluation_status",
                        "shot.target_zone_result",
                        "shot.target_zone_source",
                    )
                )
            ),
            weight=weight,
            value=shot.target_zone.error_mm(shot.cue_ball_end_mm),
        )
    )


def _add_foul(
    family: _Family,
    shot: ApprovedMetricShotEvidence,
    batch: ApprovedEvidenceBatchV2,
    cutoff: datetime,
    source_envelope_id: UUID,
) -> None:
    family.eligible += 1
    if shot.foul_result is FoulResult.UNSUPPORTED:
        family.unsupported += 1
        return
    confidence = shot.field_quality.foul_result.confidence
    if shot.foul_result is FoulResult.ABSTAINED or confidence < 0.5:
        family.abstained += 1
        return
    family.supported += 1
    taxonomy = _taxonomy(AbilityDimension.FOUL_CONTROL, TechniqueCode.FOUL_AVOIDANCE)
    family.groups[taxonomy].append(
        _Observation(
            batch_id=batch.batch_id,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            source_envelope_ids=(source_envelope_id,),
            calculation_field_codes=tuple(
                sorted(
                    _BATCH_WEIGHT_FIELDS
                    + (
                        "shot.field_quality.foul_result.confidence",
                        "shot.foul_result",
                        "shot.session_id",
                    )
                )
            ),
            weight=_weight(batch, cutoff, (confidence,)),
            value=float(shot.foul_result is FoulResult.LEGAL),
        )
    )


def _effective_n(values: list[_Observation]) -> float:
    scale = max((item.weight for item in values), default=0.0)
    if scale == 0.0:
        return 0.0
    normalized = [item.weight / scale for item in values]
    total = math.fsum(normalized)
    squared = math.fsum(weight * weight for weight in normalized)
    return 0.0 if squared == 0.0 else total**2 / squared


def _weighted_quantile(values: list[_Observation], probability: float) -> float:
    aggregated: dict[float, float] = defaultdict(float)
    for item in values:
        aggregated[item.value] += item.weight
    total = sum(aggregated.values())
    threshold = probability * total
    cumulative = 0.0
    for value, weight in sorted(aggregated.items()):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return max(aggregated)


def _wilson_half_width(success_rate: float, effective_n: float) -> float:
    denominator = 1.0 + _Z_95**2 / effective_n
    return (
        _Z_95
        / denominator
        * math.sqrt(
            success_rate * (1.0 - success_rate) / effective_n + _Z_95**2 / (4.0 * effective_n**2)
        )
    )


def _reliability(
    *,
    effective_n: float,
    sessions: int,
    coverage: float,
    uncertainty: float,
    binary: bool,
) -> MetricReliability | None:
    high_cap = 0.15 if binary else 75.0
    medium_cap = 0.25 if binary else 150.0
    if effective_n >= 40.0 and sessions >= 5 and coverage >= 0.85 and uncertainty <= high_cap:
        return MetricReliability.HIGH
    if effective_n >= 15.0 and sessions >= 3 and coverage >= 0.70 and uncertainty <= medium_cap:
        return MetricReliability.MEDIUM
    if effective_n >= 5.0 and sessions >= 2 and coverage >= 0.50:
        return MetricReliability.LOW
    return None


def _empty_metric(
    code: MetricCode,
    taxonomy: AbilityTaxonomyV2,
    family: _Family,
    status: MetricSupportStatusV2,
) -> ProjectedAbilityMetric:
    coverage = family.supported / family.eligible if family.eligible else 0.0
    return ProjectedAbilityMetric(
        metric_code=code,
        taxonomy=taxonomy,
        support_status=status,
        reliability=MetricReliability.UNKNOWN,
        eligible_count=family.eligible,
        supported_count=family.supported,
        unsupported_count=family.unsupported,
        abstained_count=family.abstained,
        sample_count=0,
        distinct_session_count=0,
        effective_sample_count=0.0,
        coverage=round_half_up(coverage, 6),
    )


def _empty_status(family: _Family | _Availability) -> MetricSupportStatusV2:
    if family.eligible == 0:
        return MetricSupportStatusV2.NO_SAMPLE
    if family.supported == 0 and family.unsupported:
        return MetricSupportStatusV2.UNSUPPORTED
    if family.supported == 0:
        return MetricSupportStatusV2.ABSTAINED
    return MetricSupportStatusV2.INSUFFICIENT_EVIDENCE


def _build_metric(
    code: MetricCode,
    taxonomy: AbilityTaxonomyV2,
    family: _Family,
    values: list[_Observation],
) -> ProjectedAbilityMetric:
    coverage = family.supported / family.eligible if family.eligible else 0.0
    effective_n = _effective_n(values)
    sessions = len({item.session_id for item in values})
    common: dict[str, Any] = {
        "metric_code": code,
        "taxonomy": taxonomy,
        "eligible_count": family.eligible,
        "supported_count": family.supported,
        "unsupported_count": family.unsupported,
        "abstained_count": family.abstained,
        "sample_count": len(values),
        "distinct_session_count": sessions,
        "effective_sample_count": round_half_up(effective_n, 6),
        "coverage": round_half_up(coverage, 6),
    }
    binary = code is not MetricCode.TARGET_ZONE_ERROR_MM
    if binary:
        total_weight = sum(item.weight for item in values)
        rate = sum(item.weight * item.value for item in values) / total_weight
        uncertainty = _wilson_half_width(rate, effective_n)
        reliability = _reliability(
            effective_n=effective_n,
            sessions=sessions,
            coverage=coverage,
            uncertainty=uncertainty,
            binary=True,
        )
        numeric = {
            "success_rate": round_half_up(rate, 6),
            "wilson_half_width": round_half_up(uncertainty, 6),
        }
    else:
        p50 = _weighted_quantile(values, 0.50)
        p75 = _weighted_quantile(values, 0.75)
        p90 = _weighted_quantile(values, 0.90)
        deviations = [
            _Observation(
                session_id=item.session_id,
                weight=item.weight,
                value=abs(item.value - p50),
                batch_id=item.batch_id,
                shot_id=item.shot_id,
                source_envelope_ids=item.source_envelope_ids,
                calculation_field_codes=item.calculation_field_codes,
            )
            for item in values
        ]
        mad = _weighted_quantile(deviations, 0.50)
        reliability = _reliability(
            effective_n=effective_n,
            sessions=sessions,
            coverage=coverage,
            uncertainty=mad,
            binary=False,
        )
        numeric = {
            "error_p50_mm": round_half_up(p50, 3),
            "error_p75_mm": round_half_up(p75, 3),
            "error_p90_mm": round_half_up(p90, 3),
            "mad_mm": round_half_up(mad, 3),
            "stability": round_half_up(max(0.0, 1.0 - mad / 250.0), 6),
        }
    if reliability is None:
        return ProjectedAbilityMetric(
            **common,
            support_status=MetricSupportStatusV2.INSUFFICIENT_EVIDENCE,
            reliability=MetricReliability.UNKNOWN,
        )
    return ProjectedAbilityMetric(
        **common,
        support_status=MetricSupportStatusV2.SUPPORTED,
        reliability=reliability,
        **numeric,
    )


def _metric_sort_key(metric: ProjectedAbilityMetric) -> tuple[str, ...]:
    taxonomy = metric.taxonomy
    return (
        metric.metric_code.value,
        taxonomy.game_code,
        taxonomy.ability_dimension.value,
        taxonomy.angle_band.value if taxonomy.angle_band else "",
        taxonomy.distance_band.value if taxonomy.distance_band else "",
        taxonomy.rail_relation.value if taxonomy.rail_relation else "",
        taxonomy.target_zone_source.value if taxonomy.target_zone_source else "",
        taxonomy.technique_code.value if taxonomy.technique_code else "",
    )


def _invalid(cutoff: datetime | None, status: MetricProjectionStatus) -> MetricProjectionResult:
    return MetricProjectionResult(status=status, evidence_cutoff=cutoff, metrics=())


def project_ability_metrics(
    inputs: tuple[object, ...],
    *,
    evidence_cutoff: datetime,
    _trace_out: list[MetricObservationTrace] | None = None,
) -> MetricProjectionResult:
    """Project approved v2 batches without clock, random, I/O, or external state."""

    if evidence_cutoff.tzinfo is None or evidence_cutoff.utcoffset() != UTC.utcoffset(
        evidence_cutoff
    ):
        return _invalid(None, MetricProjectionStatus.INVALID_INPUT)
    evidence_cutoff = evidence_cutoff.astimezone(UTC)
    if not inputs:
        return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
    accepted: list[AcceptedMetricEvidenceInput] = []
    saw_v1 = False
    saw_unaccepted_v2 = False
    for candidate in inputs:
        try:
            values = _stored_values(candidate)
            if not isinstance(values, dict):
                raise TypeError("metric projection input must be an object")
            if isinstance(candidate, AcceptedMetricEvidenceInput):
                if not candidate.is_trusted_admission:
                    raise ValueError("metric admission was not issued by the ingest authority")
                accepted.append(AcceptedMetricEvidenceInput.model_validate(values, strict=True))
            elif values.get("schema_version") == "1.0":
                ApprovedEvidenceBatch.model_validate(values, strict=True)
                saw_v1 = True
            elif values.get("schema_version") == "2.0":
                ApprovedEvidenceBatchV2.model_validate(values, strict=True)
                saw_unaccepted_v2 = True
            else:
                raise ValueError("unsupported projection input schema")
        except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
            return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
    if saw_unaccepted_v2:
        return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
    if saw_v1:
        return _invalid(evidence_cutoff, MetricProjectionStatus.UNSUPPORTED_INPUT_CONTRACT)
    if not accepted:
        return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)

    validated = tuple(item.batch for item in accepted)
    if len({batch.user_id for batch in validated}) != 1:
        return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
    for batch in validated:
        if batch.approved_at > evidence_cutoff:
            return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
        if batch.content_sha256 != canonical_sha256(
            batch.model_dump(mode="json", exclude={"content_sha256"})
        ):
            return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
        try:
            _approval_weight(batch)
        except ValueError:
            return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)

    input_batch_count = len(accepted)
    input_shot_count = sum(len(item.batch.metric_shots) for item in accepted)
    unique_by_identity: dict[tuple[object, int], AcceptedMetricEvidenceInput] = {}
    identities_by_batch_id: dict[object, tuple[object, int]] = {}
    replayed_batch_count = 0
    replayed_shot_count = 0
    for item in accepted:
        batch = item.batch
        identity = (batch.session_id, batch.review_revision)
        existing = unique_by_identity.get(identity)
        if existing is not None:
            if existing != item:
                return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
            replayed_batch_count += 1
            replayed_shot_count += len(batch.metric_shots)
            continue
        prior_identity = identities_by_batch_id.get(batch.batch_id)
        if prior_identity is not None and prior_identity != identity:
            return _invalid(evidence_cutoff, MetricProjectionStatus.INVALID_INPUT)
        unique_by_identity[identity] = item
        identities_by_batch_id[batch.batch_id] = identity
    latest_by_session: dict[object, AcceptedMetricEvidenceInput] = {}
    superseded: list[AcceptedMetricEvidenceInput] = []
    for item in unique_by_identity.values():
        current = latest_by_session.get(item.session_id)
        if current is None or item.review_revision > current.review_revision:
            if current is not None:
                superseded.append(current)
            latest_by_session[item.session_id] = item
        else:
            superseded.append(item)
    accepted = list(latest_by_session.values())
    superseded_refs = tuple(
        SupersededEvidenceReference(
            admission_id=item.admission_id,
            batch_id=item.batch_id,
            session_id=item.session_id,
            review_revision=item.review_revision,
            batch_content_sha256=item.batch_content_sha256,
            superseded_by_admission_id=latest_by_session[item.session_id].admission_id,
            superseded_by_batch_id=latest_by_session[item.session_id].batch_id,
            superseded_by_review_revision=latest_by_session[item.session_id].review_revision,
        )
        for item in sorted(
            superseded,
            key=lambda value: (
                str(value.session_id),
                value.review_revision,
                str(value.batch_id),
                str(value.admission_id),
            ),
        )
    )
    included_batch_count = 0
    included_shot_count = 0
    outside_window_batch_count = 0
    outside_window_shot_count = 0

    pot = _Family()
    foul = _Family()
    target_success = _Family()
    target_error = _Family()
    for accepted_item in sorted(
        accepted,
        key=lambda item: (
            item.batch.approved_at,
            str(item.session_id),
            item.review_revision,
            str(item.batch_id),
            str(item.admission_id),
        ),
    ):
        batch = accepted_item.batch
        if _age_days(batch, evidence_cutoff) > _MAX_AGE_DAYS:
            outside_window_batch_count += 1
            outside_window_shot_count += len(batch.metric_shots)
            continue
        included_batch_count += 1
        included_shot_count += len(batch.metric_shots)
        source_envelopes = _source_envelope_index(batch)
        for shot in sorted(
            batch.metric_shots,
            key=lambda item: (str(item.session_id), item.sequence, str(item.shot_id)),
        ):
            source_envelope_id = source_envelopes[shot.shot_id]
            _add_pot(pot, shot, batch, evidence_cutoff, source_envelope_id)
            _add_target_zone(
                target_success,
                target_error,
                shot,
                batch,
                evidence_cutoff,
                source_envelope_id,
            )
            _add_foul(foul, shot, batch, evidence_cutoff, source_envelope_id)

    metrics: list[ProjectedAbilityMetric] = []
    families: list[tuple[MetricCode, _Family]] = [
        (MetricCode.POTTING_SUCCESS_RATE, pot),
        (MetricCode.TARGET_ZONE_SUCCESS_RATE, target_success),
        (MetricCode.TARGET_ZONE_ERROR_MM, target_error),
        (MetricCode.LEGAL_SHOT_RATE, foul),
    ]
    for code, family in families:
        for taxonomy, values in family.groups.items():
            metrics.append(_build_metric(code, taxonomy, family, values))
            if _trace_out is not None:
                _trace_out.extend(
                    MetricObservationTrace(
                        metric_code=code,
                        taxonomy=taxonomy,
                        batch_id=item.batch_id,
                        session_id=item.session_id,
                        shot_id=item.shot_id,
                        source_envelope_ids=item.source_envelope_ids,
                        calculation_field_codes=item.calculation_field_codes,
                    )
                    for item in values
                    if item.batch_id is not None and item.shot_id is not None
                )
        if code in {MetricCode.TARGET_ZONE_SUCCESS_RATE, MetricCode.TARGET_ZONE_ERROR_MM}:
            grouped_sources = {taxonomy.target_zone_source for taxonomy in family.groups}
            for source, availability in sorted(
                family.source_counts.items(), key=lambda item: item[0].value
            ):
                if source not in grouped_sources:
                    status = _empty_status(availability)
                    metrics.append(
                        _empty_metric(
                            code,
                            _summary_taxonomy(code, status, source=source),
                            family,
                            status,
                        )
                    )
            if not family.groups and not family.source_counts:
                status = _empty_status(family)
                metrics.append(_empty_metric(code, _summary_taxonomy(code, status), family, status))
        elif not family.groups:
            status = _empty_status(family)
            metrics.append(_empty_metric(code, _summary_taxonomy(code, status), family, status))
    return MetricProjectionResult(
        status=MetricProjectionStatus.PROJECTED,
        evidence_cutoff=evidence_cutoff,
        input_batch_count=input_batch_count,
        input_shot_count=input_shot_count,
        included_batch_count=included_batch_count,
        included_shot_count=included_shot_count,
        outside_window_batch_count=outside_window_batch_count,
        outside_window_shot_count=outside_window_shot_count,
        replayed_batch_count=replayed_batch_count,
        replayed_shot_count=replayed_shot_count,
        superseded_batch_count=len(superseded),
        superseded_shot_count=sum(len(item.batch.metric_shots) for item in superseded),
        superseded_evidence=superseded_refs,
        metrics=tuple(sorted(metrics, key=_metric_sort_key)),
    )


def project_ability_metrics_with_trace(
    inputs: tuple[object, ...],
    *,
    evidence_cutoff: datetime,
) -> MetricProjectionWithTrace:
    """Return the frozen 8.3 projection plus raw-free same-pass observation lineage."""

    traces: list[MetricObservationTrace] = []
    projection = project_ability_metrics(
        inputs,
        evidence_cutoff=evidence_cutoff,
        _trace_out=traces,
    )
    return MetricProjectionWithTrace(
        projection=projection,
        observations=tuple(
            sorted(
                traces,
                key=lambda item: (
                    item.metric_code.value,
                    canonical_sha256(item.taxonomy.model_dump(mode="json")),
                    str(item.batch_id),
                    str(item.session_id),
                    str(item.shot_id),
                ),
            )
        ),
    )


__all__ = [
    "CALCULATION_POLICY_VERSION",
    "MetricObservationTrace",
    "MetricProjectionWithTrace",
    "TAXONOMY_VERSION",
    "project_ability_metrics",
    "project_ability_metrics_with_trace",
]
