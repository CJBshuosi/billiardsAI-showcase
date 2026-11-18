"""Versioned deterministic coach fallback built from sufficient rule inputs."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import UUID4, Field, StrictInt, field_validator

from billiards_ai.cloud.plan_publication.publisher import (
    PlanPublicationCandidate,
    PlanPublisher,
    PublicationResult,
)
from billiards_ai.cloud.training.catalog import (
    DrillCatalog,
    DrillEligibilityProfile,
    list_eligible_drills,
)
from billiards_ai.shared.models.events import require_utc
from billiards_ai.shared.models.training_plan import DrillReference, TrainingBlock
from billiards_ai.shared.plan_publication.codec import canonical_json_bytes
from billiards_ai.shared.plan_publication.contracts import (
    CanonicalString,
    CapabilityManifest,
    FinalTrainingPlan,
    PublicationContractModel,
    _uuid4_from_digest,
)


class FallbackFailureReason(StrEnum):
    NO_RELIABLE_PLAN = "NO_RELIABLE_PLAN"


class PlanResolutionSource(StrEnum):
    PROVIDER = "PROVIDER"
    REPAIR = "REPAIR"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class FallbackRuleInput(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    user_id: UUID4
    session_id: UUID4
    authorization_id: UUID4
    authorization_version: StrictInt = Field(ge=1)
    target_device_id: CanonicalString
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    target_finding_id: UUID4
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    plan_version: StrictInt = Field(ge=1)
    user_level: StrictInt = Field(ge=1)
    acquired_skill_codes: tuple[str, ...]
    active_contraindication_codes: tuple[str, ...]
    catalog: DrillCatalog
    capability_manifest: CapabilityManifest
    created_at: datetime
    valid_until: datetime

    _created_at_utc = field_validator("created_at")(require_utc)
    _valid_until_utc = field_validator("valid_until")(require_utc)

    @field_validator("active_contraindication_codes")
    @classmethod
    def contraindications_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not item
            or any(character.isspace() for character in item)
            or unicodedata.normalize("NFC", item) != item
            for item in value
        ) or value != tuple(sorted(set(value))):
            raise ValueError("contraindications must be sorted unique canonical codes")
        return value


class DeterministicFallbackResult(PublicationContractModel):
    plan: FinalTrainingPlan | None = None
    reason: FallbackFailureReason | None = None


@dataclass(frozen=True, slots=True)
class PlanResolutionResult:
    source: PlanResolutionSource | None = None
    publication: PublicationResult | None = None
    reason: FallbackFailureReason | None = None


def deterministic_coach_fallback(source: FallbackRuleInput) -> DeterministicFallbackResult:
    if source.valid_until <= source.created_at:
        return DeterministicFallbackResult(reason=FallbackFailureReason.NO_RELIABLE_PLAN)
    profile = DrillEligibilityProfile(
        user_level=source.user_level,
        acquired_skill_codes=tuple(sorted(set(source.acquired_skill_codes))),
        available_capabilities=source.capability_manifest.capabilities,
        available_equipment_codes=source.capability_manifest.equipment_codes,
        active_contraindication_codes=tuple(sorted(set(source.active_contraindication_codes))),
    )
    eligible = list_eligible_drills(catalog=source.catalog, profile=profile)
    if not eligible:
        return DeterministicFallbackResult(reason=FallbackFailureReason.NO_RELIABLE_PLAN)
    primary = eligible[0]
    fallback = eligible[-1]
    seed = canonical_json_bytes(
        {
            "assessment_id": source.assessment_id,
            "assessment_revision": source.assessment_revision,
            "authorization_id": source.authorization_id,
            "authorization_version": source.authorization_version,
            "catalog": source.catalog.content_sha256,
            "capability_manifest": source.capability_manifest.content_sha256,
            "created_at": source.created_at,
            "plan_version": source.plan_version,
            "session_id": source.session_id,
            "snapshot_id": source.snapshot_id,
            "snapshot_version": source.snapshot_version,
            "target_device_id": source.target_device_id,
            "target_finding_id": source.target_finding_id,
            "user_id": source.user_id,
            "valid_until": source.valid_until,
        }
    )
    primary_reference = DrillReference(
        catalog_version=source.catalog.catalog_version,
        drill_id=primary.drill_id,
        drill_revision=primary.revision,
        required_capabilities=primary.required_capabilities,
    )
    fallback_reference = DrillReference(
        catalog_version=source.catalog.catalog_version,
        drill_id=fallback.drill_id,
        drill_revision=fallback.revision,
        required_capabilities=fallback.required_capabilities,
    )
    block = TrainingBlock(
        block_id=_uuid4_from_digest(seed + b":block:1"),
        order=1,
        drill=primary_reference,
        target_finding_id=source.target_finding_id,
        target_repetitions=primary.success_metrics[0].target_attempts,
        progress_after_successes=primary.progression_rule.threshold,
        regress_after_failures=abs(primary.regression_rule.threshold),
    )
    payload = {
        "schema_version": "2.0",
        "plan_id": _uuid4_from_digest(seed + b":plan"),
        "user_id": source.user_id,
        "assessment_id": source.assessment_id,
        "assessment_revision": source.assessment_revision,
        "snapshot_id": source.snapshot_id,
        "snapshot_version": source.snapshot_version,
        "version": source.plan_version,
        "admission_status": "VALIDATED",
        "created_at": source.created_at,
        "valid_until": source.valid_until,
        "blocks": (block,),
        "fallback_drill": fallback_reference,
    }
    content_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return DeterministicFallbackResult(
        plan=FinalTrainingPlan.model_validate(payload | {"content_sha256": content_sha256})
    )


def publish_generated_or_fallback(
    *,
    publisher: PlanPublisher,
    provider_candidate: PlanPublicationCandidate | None,
    repair_candidate: PlanPublicationCandidate | None,
    fallback_input: FallbackRuleInput,
    fallback_candidate_template: PlanPublicationCandidate,
    now: datetime,
) -> PlanResolutionResult:
    """Resolve the fixed state machine; any candidate gate rejection is terminal."""

    if provider_candidate is not None:
        return PlanResolutionResult(
            source=PlanResolutionSource.PROVIDER,
            publication=publisher.publish(provider_candidate, now=now),
        )
    if repair_candidate is not None:
        return PlanResolutionResult(
            source=PlanResolutionSource.REPAIR,
            publication=publisher.publish(repair_candidate, now=now),
        )
    source = fallback_input
    template = fallback_candidate_template
    authorization = template.authorization
    if (
        source.user_id != template.plan.user_id
        or source.user_id != authorization.user_id
        or source.session_id != authorization.session_id
        or source.authorization_id != authorization.authorization_id
        or source.authorization_version != authorization.authorization_version
        or source.target_device_id != authorization.target_device_id
        or source.target_device_id != source.capability_manifest.target_device_id
        or source.assessment_id != template.plan.assessment_id
        or source.assessment_revision != template.plan.assessment_revision
        or source.snapshot_id != template.plan.snapshot_id
        or source.snapshot_version != template.plan.snapshot_version
        or source.catalog.catalog_version != template.catalog.catalog_version
        or source.catalog.content_sha256 != template.catalog.content_sha256
        or source.capability_manifest.manifest_id != template.capability_manifest.manifest_id
        or source.capability_manifest.content_sha256 != template.capability_manifest.content_sha256
        or source.created_at != template.requested_not_before
        or source.valid_until != template.requested_expires_at
        or source.active_contraindication_codes != template.active_contraindication_codes
        or source.created_at != now
    ):
        return PlanResolutionResult(reason=FallbackFailureReason.NO_RELIABLE_PLAN)
    fallback = deterministic_coach_fallback(fallback_input)
    if fallback.plan is None:
        return PlanResolutionResult(reason=FallbackFailureReason.NO_RELIABLE_PLAN)
    candidate = fallback_candidate_template.model_copy(update={"plan": fallback.plan})
    return PlanResolutionResult(
        source=PlanResolutionSource.DETERMINISTIC_FALLBACK,
        publication=publisher.publish(candidate, now=now),
    )
