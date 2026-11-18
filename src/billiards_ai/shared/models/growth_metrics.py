"""Versioned Story 8.3 ability metric projection contracts."""

from __future__ import annotations

import math
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Literal

from pydantic import UUID4, Field, StrictInt, field_validator, model_validator

from billiards_ai.shared.models.events import CoreContractModel, Sha256, require_utc
from billiards_ai.shared.models.shot import TargetZoneSource

TAXONOMY_VERSION = "ability-taxonomy-rd-v1.0.0"
CALCULATION_POLICY_VERSION = "ability-metrics-rd-v1.0.0"


class AbilityDimension(StrEnum):
    POTTING = "POTTING"
    POSITIONING = "POSITIONING"
    POWER_CONTROL = "POWER_CONTROL"
    FOUL_CONTROL = "FOUL_CONTROL"
    STRATEGY_EXECUTION = "STRATEGY_EXECUTION"
    SPIN_APPLICATION = "SPIN_APPLICATION"


class AngleBand(StrEnum):
    STRAIGHT = "STRAIGHT"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    EXTREME = "EXTREME"


class DistanceBand(StrEnum):
    VERY_SHORT = "VERY_SHORT"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    VERY_LONG = "VERY_LONG"


class RailRelation(StrEnum):
    FROZEN_TO_RAIL = "FROZEN_TO_RAIL"
    NEAR_RAIL = "NEAR_RAIL"
    OPEN_TABLE = "OPEN_TABLE"


class TechniqueCode(StrEnum):
    DIRECT_POT = "DIRECT_POT"
    CUT_POT = "CUT_POT"
    TARGET_ZONE_POSITION = "TARGET_ZONE_POSITION"
    FOUL_AVOIDANCE = "FOUL_AVOIDANCE"
    UNCLASSIFIED = "UNCLASSIFIED"


class TaxonomyApplicability(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class MetricCode(StrEnum):
    POTTING_SUCCESS_RATE = "potting.success_rate"
    TARGET_ZONE_SUCCESS_RATE = "positioning.target_zone.success_rate"
    TARGET_ZONE_ERROR_MM = "positioning.target_zone.error_mm"
    LEGAL_SHOT_RATE = "foul_control.legal_shot_rate"


class MetricSupportStatusV2(StrEnum):
    SUPPORTED = "SUPPORTED"
    NO_SAMPLE = "NO_SAMPLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    ABSTAINED = "ABSTAINED"


class MetricReliability(StrEnum):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MetricProjectionStatus(StrEnum):
    PROJECTED = "PROJECTED"
    UNSUPPORTED_INPUT_CONTRACT = "UNSUPPORTED_INPUT_CONTRACT"
    INVALID_INPUT = "INVALID_INPUT"


class AbilityTaxonomyV2(CoreContractModel):
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    game_code: Literal["CHINESE_EIGHT_BALL"] = "CHINESE_EIGHT_BALL"
    ability_dimension: AbilityDimension
    angle_band: AngleBand | None = None
    angle_applicability: TaxonomyApplicability = TaxonomyApplicability.NOT_APPLICABLE
    distance_band: DistanceBand | None = None
    distance_applicability: TaxonomyApplicability = TaxonomyApplicability.NOT_APPLICABLE
    rail_relation: RailRelation | None = None
    rail_applicability: TaxonomyApplicability = TaxonomyApplicability.NOT_APPLICABLE
    target_zone_source: TargetZoneSource | None = None
    target_zone_applicability: TaxonomyApplicability = TaxonomyApplicability.NOT_APPLICABLE
    technique_code: TechniqueCode | None = None
    technique_applicability: TaxonomyApplicability = TaxonomyApplicability.NOT_APPLICABLE

    @model_validator(mode="after")
    def values_match_applicability(self) -> AbilityTaxonomyV2:
        pairs = (
            (self.angle_band, self.angle_applicability, "angle"),
            (self.distance_band, self.distance_applicability, "distance"),
            (self.rail_relation, self.rail_applicability, "rail"),
            (self.target_zone_source, self.target_zone_applicability, "target zone"),
            (self.technique_code, self.technique_applicability, "technique"),
        )
        for value, applicability, label in pairs:
            if (applicability is TaxonomyApplicability.APPLICABLE) != (value is not None):
                raise ValueError(f"{label} value must exactly match APPLICABLE taxonomy status")
        if self.target_zone_source is TargetZoneSource.NONE:
            raise ValueError("taxonomy target-zone source cannot use the evidence NONE sentinel")
        return self


def _finite_optional(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("metric numeric values must be finite")
    return value


class ProjectedAbilityMetric(CoreContractModel):
    metric_code: MetricCode
    taxonomy: AbilityTaxonomyV2
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    support_status: MetricSupportStatusV2
    reliability: MetricReliability
    eligible_count: StrictInt = Field(ge=0)
    supported_count: StrictInt = Field(ge=0)
    unsupported_count: StrictInt = Field(ge=0)
    abstained_count: StrictInt = Field(ge=0)
    sample_count: StrictInt = Field(ge=0)
    distinct_session_count: StrictInt = Field(ge=0)
    effective_sample_count: float = Field(ge=0.0)
    coverage: float = Field(ge=0.0, le=1.0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    wilson_half_width: float | None = Field(default=None, ge=0.0, le=1.0)
    error_p50_mm: float | None = Field(default=None, ge=0.0)
    error_p75_mm: float | None = Field(default=None, ge=0.0)
    error_p90_mm: float | None = Field(default=None, ge=0.0)
    mad_mm: float | None = Field(default=None, ge=0.0)
    stability: float | None = Field(default=None, ge=0.0, le=1.0)

    _finite_effective_n = field_validator("effective_sample_count")(_finite_optional)
    _finite_coverage = field_validator("coverage")(_finite_optional)
    _finite_derived = field_validator(
        "success_rate",
        "wilson_half_width",
        "error_p50_mm",
        "error_p75_mm",
        "error_p90_mm",
        "mad_mm",
        "stability",
    )(_finite_optional)

    @model_validator(mode="after")
    def counts_and_values_match_status(self) -> ProjectedAbilityMetric:
        if (
            self.supported_count + self.unsupported_count + self.abstained_count
            != self.eligible_count
        ):
            raise ValueError("family audit counts must reconcile")
        if self.supported_count > self.eligible_count:
            raise ValueError("supported count cannot exceed eligible count")
        if self.sample_count > self.supported_count:
            raise ValueError("sample count cannot exceed supported count")
        if self.distinct_session_count > self.sample_count:
            raise ValueError("session count cannot exceed sample count")
        if self.effective_sample_count > self.sample_count:
            raise ValueError("effective sample count cannot exceed sample count")
        expected_coverage = (
            Decimal(str(self.supported_count / self.eligible_count)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            if self.eligible_count
            else Decimal("0.000000")
        )
        if Decimal(str(self.coverage)) != expected_coverage:
            raise ValueError("coverage must equal rounded supported_count / eligible_count")
        taxonomy = self.taxonomy
        if self.metric_code is MetricCode.POTTING_SUCCESS_RATE:
            if (
                taxonomy.ability_dimension is not AbilityDimension.POTTING
                or taxonomy.target_zone_applicability is not TaxonomyApplicability.NOT_APPLICABLE
            ):
                raise ValueError("metric taxonomy is incompatible with potting metric")
            band_applicability = (
                taxonomy.angle_applicability,
                taxonomy.distance_applicability,
                taxonomy.rail_applicability,
            )
            if taxonomy.technique_applicability is TaxonomyApplicability.APPLICABLE:
                if (
                    any(item is not TaxonomyApplicability.APPLICABLE for item in band_applicability)
                    or taxonomy.technique_code
                    not in {
                        TechniqueCode.DIRECT_POT,
                        TechniqueCode.CUT_POT,
                        TechniqueCode.UNCLASSIFIED,
                    }
                    or (
                        taxonomy.technique_code is TechniqueCode.DIRECT_POT
                        and taxonomy.angle_band is not AngleBand.STRAIGHT
                    )
                    or (
                        taxonomy.technique_code is TechniqueCode.CUT_POT
                        and taxonomy.angle_band is AngleBand.STRAIGHT
                    )
                ):
                    raise ValueError("metric taxonomy is incompatible with potting metric")
            elif (
                taxonomy.technique_code is not None
                or len(set(band_applicability + (taxonomy.technique_applicability,))) != 1
            ):
                raise ValueError("metric taxonomy is incompatible with potting metric")
        elif self.metric_code in {
            MetricCode.TARGET_ZONE_SUCCESS_RATE,
            MetricCode.TARGET_ZONE_ERROR_MM,
        }:
            if (
                taxonomy.ability_dimension is not AbilityDimension.POSITIONING
                or taxonomy.technique_code is not TechniqueCode.TARGET_ZONE_POSITION
                or taxonomy.technique_applicability is not TaxonomyApplicability.APPLICABLE
                or any(
                    applicability is not TaxonomyApplicability.NOT_APPLICABLE
                    for applicability in (
                        taxonomy.angle_applicability,
                        taxonomy.distance_applicability,
                        taxonomy.rail_applicability,
                    )
                )
            ):
                raise ValueError("metric taxonomy is incompatible with target-zone metric")
        elif (
            taxonomy.ability_dimension is not AbilityDimension.FOUL_CONTROL
            or taxonomy.technique_code is not TechniqueCode.FOUL_AVOIDANCE
            or taxonomy.technique_applicability is not TaxonomyApplicability.APPLICABLE
            or any(
                applicability is not TaxonomyApplicability.NOT_APPLICABLE
                for applicability in (
                    taxonomy.angle_applicability,
                    taxonomy.distance_applicability,
                    taxonomy.rail_applicability,
                    taxonomy.target_zone_applicability,
                )
            )
        ):
            raise ValueError("metric taxonomy is incompatible with foul-control metric")
        numeric = (
            self.success_rate,
            self.wilson_half_width,
            self.error_p50_mm,
            self.error_p75_mm,
            self.error_p90_mm,
            self.mad_mm,
            self.stability,
        )
        if self.support_status is MetricSupportStatusV2.SUPPORTED:
            if self.reliability is MetricReliability.UNKNOWN or not any(
                item is not None for item in numeric
            ):
                raise ValueError("supported metric requires numeric evidence and reliability")
        elif self.reliability is not MetricReliability.UNKNOWN or any(
            item is not None for item in numeric
        ):
            raise ValueError("non-supported metric cannot claim values or reliability")
        binary_values = (self.success_rate, self.wilson_half_width)
        error_values = (
            self.error_p50_mm,
            self.error_p75_mm,
            self.error_p90_mm,
            self.mad_mm,
            self.stability,
        )
        if self.support_status is MetricSupportStatusV2.SUPPORTED:
            if self.metric_code is MetricCode.TARGET_ZONE_ERROR_MM:
                if any(item is None for item in error_values) or any(
                    item is not None for item in binary_values
                ):
                    raise ValueError("supported error metric requires only its error values")
            elif any(item is None for item in binary_values) or any(
                item is not None for item in error_values
            ):
                raise ValueError("supported binary metric requires only rate and Wilson values")
        if (
            self.error_p50_mm is not None
            and self.error_p75_mm is not None
            and self.error_p90_mm is not None
            and not self.error_p50_mm <= self.error_p75_mm <= self.error_p90_mm
        ):
            raise ValueError("error quantiles must be ordered")
        return self


class SupersededEvidenceReference(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    admission_id: UUID4
    batch_id: UUID4
    session_id: UUID4
    review_revision: StrictInt = Field(ge=1)
    batch_content_sha256: Sha256
    superseded_by_admission_id: UUID4
    superseded_by_batch_id: UUID4
    superseded_by_review_revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def replacement_is_newer_in_same_session(self) -> SupersededEvidenceReference:
        if self.superseded_by_review_revision <= self.review_revision:
            raise ValueError("superseding review revision must be newer")
        return self


class MetricProjectionResult(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    status: MetricProjectionStatus
    taxonomy_version: Literal["ability-taxonomy-rd-v1.0.0"] = TAXONOMY_VERSION
    calculation_policy_version: Literal["ability-metrics-rd-v1.0.0"] = CALCULATION_POLICY_VERSION
    evidence_cutoff: datetime | None
    input_batch_count: StrictInt = Field(default=0, ge=0)
    input_shot_count: StrictInt = Field(default=0, ge=0)
    included_batch_count: StrictInt = Field(default=0, ge=0)
    included_shot_count: StrictInt = Field(default=0, ge=0)
    outside_window_batch_count: StrictInt = Field(default=0, ge=0)
    outside_window_shot_count: StrictInt = Field(default=0, ge=0)
    replayed_batch_count: StrictInt = Field(default=0, ge=0)
    replayed_shot_count: StrictInt = Field(default=0, ge=0)
    superseded_batch_count: StrictInt = Field(default=0, ge=0)
    superseded_shot_count: StrictInt = Field(default=0, ge=0)
    superseded_evidence: tuple[SupersededEvidenceReference, ...] = ()
    metrics: tuple[ProjectedAbilityMetric, ...]

    _cutoff_utc = field_validator("evidence_cutoff")(
        lambda value: None if value is None else require_utc(value)
    )

    @model_validator(mode="after")
    def status_matches_metrics(self) -> MetricProjectionResult:
        if self.status is not MetricProjectionStatus.PROJECTED and self.metrics:
            raise ValueError("failed projection cannot carry partial metrics")
        if self.status is MetricProjectionStatus.PROJECTED and self.evidence_cutoff is None:
            raise ValueError("projected result requires a UTC evidence cutoff")
        keys = tuple((item.metric_code, item.taxonomy) for item in self.metrics)
        if len(keys) != len(set(keys)):
            raise ValueError("metric identities must be unique")
        if (
            self.included_batch_count
            + self.outside_window_batch_count
            + self.replayed_batch_count
            + self.superseded_batch_count
            != self.input_batch_count
        ):
            raise ValueError("batch audit counts must reconcile")
        if (
            self.included_shot_count
            + self.outside_window_shot_count
            + self.replayed_shot_count
            + self.superseded_shot_count
            != self.input_shot_count
        ):
            raise ValueError("shot audit counts must reconcile")
        if self.superseded_batch_count != len(self.superseded_evidence):
            raise ValueError("superseded references must match the superseded batch audit count")
        return self
