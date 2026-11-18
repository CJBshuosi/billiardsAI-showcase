"""REAL_VIDEO_GATE shared contracts (AD-17, Story 1.1).

These models deliberately fail closed.  They describe immutable evidence and
revisions; only the cloud privacy service is allowed to create an approved
revision or authorize a real-video use.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.constants import PRIVACY_MAX_FUTURE_CLOCK_SKEW

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("datetime is outside the supported UTC range") from exc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _leads_beyond_clock_skew(later: datetime, earlier: datetime) -> bool:
    return later > earlier and later - earlier > PRIVACY_MAX_FUTURE_CLOCK_SKEW


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RealVideoGateStatus(StrEnum):
    BLOCKED = "BLOCKED"
    APPROVED = "APPROVED"


class BystanderDisposition(StrEnum):
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
    ANONYMIZED = "ANONYMIZED"
    CONSENTED_FOR_SERVICE = "CONSENTED_FOR_SERVICE"
    CONSENTED_FOR_TRAINING = "CONSENTED_FOR_TRAINING"
    UNCONSENTED_IDENTIFIABLE = "UNCONSENTED_IDENTIFIABLE"


class MaterialUsePurpose(StrEnum):
    SERVICE_ANALYSIS = "SERVICE_ANALYSIS"
    NECESSARY_REVIEW = "NECESSARY_REVIEW"
    ANNOTATION = "ANNOTATION"
    TRAINING = "TRAINING"


class MaterialUseDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_ANONYMIZATION_OR_CONSENT = "REQUIRE_ANONYMIZATION_OR_CONSENT"
    DENY = "DENY"


class GateChecklist(StrictFrozenModel):
    table_view_only: StrictBool = False
    audio_disabled: StrictBool = False
    notice_posted: StrictBool = False
    operator_contact_available: StrictBool = False
    separate_service_consent: StrictBool = False
    separate_training_consent: StrictBool = False
    purpose_authorization_defined: StrictBool = False
    access_path_defined: StrictBool = False
    retention_path_defined: StrictBool = False
    deletion_path_defined: StrictBool = False
    marked_session_clips_only: StrictBool = False
    full_stream_stays_local: StrictBool = False

    @property
    def is_complete(self) -> bool:
        return all(value is True for value in self.model_dump().values())

    @property
    def missing_items(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.model_dump().items() if value is not True)


class EvidenceReference(StrictFrozenModel):
    evidence_ref: NonEmptyStr
    sha256: Sha256
    captured_at: datetime

    _captured_at_utc = field_validator("captured_at")(_utc)


class GateScope(StrictFrozenModel):
    venue_id: NonEmptyStr
    table_id: NonEmptyStr
    device_id: NonEmptyStr
    camera_config_id: NonEmptyStr
    checklist: GateChecklist
    operator_name: NonEmptyStr
    operator_contact: NonEmptyStr
    service_consent_version: NonEmptyStr
    service_consent_sha256: Sha256
    training_consent_version: NonEmptyStr
    training_consent_sha256: Sha256
    access_policy_ref: NonEmptyStr
    retention_policy_ref: NonEmptyStr
    deletion_policy_ref: NonEmptyStr
    approved_purposes: tuple[MaterialUsePurpose, ...]

    @field_validator("approved_purposes")
    @classmethod
    def purposes_are_unique_and_non_empty(
        cls, value: tuple[MaterialUsePurpose, ...]
    ) -> tuple[MaterialUsePurpose, ...]:
        if not value:
            raise ValueError("at least one approved purpose is required")
        if len(value) != len(set(value)):
            raise ValueError("approved purposes must be unique")
        return tuple(sorted(value, key=lambda item: item.value))

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()


class GateApproval(StrictFrozenModel):
    approver_id: NonEmptyStr
    credential_id: NonEmptyStr
    key_id: NonEmptyStr = "legacy-v1"
    mac_version: Literal["1.0", "2.0", "3.0"] = "1.0"
    approved_at: datetime
    approval_evidence_sha256: Sha256
    gate_fingerprint: Sha256
    approval_mac: Sha256

    _approved_at_utc = field_validator("approved_at")(_utc)

    @model_validator(mode="before")
    @classmethod
    def infer_historical_mac_version(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "key_id" not in normalized:
            normalized["key_id"] = "legacy-v1"
            normalized["mac_version"] = "1.0"
        else:
            normalized.setdefault("mac_version", "2.0")
        return normalized


class GateRevision(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    gate_id: UUID4
    revision_id: UUID4
    revision_number: int = Field(ge=1)
    supersedes_revision_id: UUID4 | None = None
    status: RealVideoGateStatus = RealVideoGateStatus.BLOCKED
    scope: GateScope
    inspector_id: NonEmptyStr
    inspected_at: datetime
    evidence: tuple[EvidenceReference, ...] = ()
    approval: GateApproval | None = None
    invalidation_reason: NonEmptyStr | None = None
    created_at: datetime

    _inspected_at_utc = field_validator("inspected_at")(_utc)
    _created_at_utc = field_validator("created_at")(_utc)

    @model_validator(mode="after")
    def revision_is_consistent(self) -> GateRevision:
        if self.revision_number == 1 and self.supersedes_revision_id is not None:
            raise ValueError("first revision cannot supersede another revision")
        if self.revision_number > 1 and self.supersedes_revision_id is None:
            raise ValueError("later revisions must identify the superseded revision")
        if _leads_beyond_clock_skew(self.inspected_at, self.created_at):
            raise ValueError("inspection cannot lead revision creation beyond clock skew")
        now = _utc_now()
        if _leads_beyond_clock_skew(self.inspected_at, now):
            raise ValueError("inspection time cannot be more than five minutes in the future")
        if _leads_beyond_clock_skew(self.created_at, now):
            raise ValueError("revision creation time cannot exceed the allowed future clock skew")
        if any(item.captured_at > self.inspected_at for item in self.evidence):
            raise ValueError("gate evidence cannot be captured after inspection")
        if self.status is RealVideoGateStatus.APPROVED:
            if self.revision_number == 1:
                raise ValueError("the first gate revision must be BLOCKED")
            if self.approval is None:
                raise ValueError("approved revision requires approval evidence")
            if not self.scope.checklist.is_complete or not self.evidence:
                raise ValueError("approved revision requires complete checks and evidence")
            if self.approval.gate_fingerprint != self.scope.fingerprint:
                raise ValueError("approval fingerprint does not match gate scope")
            if self.invalidation_reason:
                raise ValueError("approved revision cannot be invalidated")
            if self.approval.approved_at < self.inspected_at:
                raise ValueError("approval cannot predate inspection")
            if self.created_at != self.approval.approved_at:
                raise ValueError("approved revision creation time must match approval time")
        elif self.approval is not None:
            raise ValueError("blocked revision cannot contain an approval")
        return self


class ApproverIdentity(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    approver_id: NonEmptyStr
    credential_id: NonEmptyStr


class SubjectClassification(StrictFrozenModel):
    asset_id: NonEmptyStr
    segment_ref: NonEmptyStr
    subject_ref: NonEmptyStr
    disposition: BystanderDisposition
    consent_version: NonEmptyStr | None = None
    consent_evidence_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def consent_evidence_matches_disposition(self) -> SubjectClassification:
        consented = self.disposition in {
            BystanderDisposition.CONSENTED_FOR_SERVICE,
            BystanderDisposition.CONSENTED_FOR_TRAINING,
        }
        has_complete_evidence = (
            self.consent_version is not None and self.consent_evidence_sha256 is not None
        )
        if consented and not has_complete_evidence:
            raise ValueError("consented classifications require versioned consent evidence")
        if not consented and (
            self.consent_version is not None or self.consent_evidence_sha256 is not None
        ):
            raise ValueError("consent evidence is only valid for consented classifications")
        return self


class AnonymizationRecord(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    source_asset_id: NonEmptyStr
    source_asset_sha256: Sha256
    derived_asset_id: NonEmptyStr
    method: NonEmptyStr
    operator_id: NonEmptyStr
    processed_at: datetime
    output_sha256: Sha256
    visual_evidence_sha256: Sha256
    metadata_removal_evidence_sha256: Sha256

    _processed_at_utc = field_validator("processed_at")(_utc)

    @model_validator(mode="after")
    def derived_asset_is_new(self) -> AnonymizationRecord:
        if self.source_asset_id == self.derived_asset_id:
            raise ValueError("anonymization must create a new derived asset")
        if self.source_asset_sha256 == self.output_sha256:
            raise ValueError("anonymization must change the media digest")
        return self


class SourceAssetProvenance(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    asset_id: NonEmptyStr
    sha256: Sha256
    captured_at: datetime
    gate_revision_id: UUID4
    gate_scope_fingerprint: Sha256

    _captured_at_utc = field_validator("captured_at")(_utc)


class AssetContext(StrictFrozenModel):
    asset_id: NonEmptyStr
    asset_sha256: Sha256
    gate_scope_fingerprint: Sha256
    captured_at: datetime
    captured_under_revision_id: UUID4
    bystanders_present: StrictBool
    classifications: tuple[SubjectClassification, ...] = ()
    anonymization_records: tuple[AnonymizationRecord, ...] = ()
    source_asset_provenance: SourceAssetProvenance | None = None

    _captured_at_utc = field_validator("captured_at")(_utc)

    @model_validator(mode="after")
    def classifications_match_asset(self) -> AssetContext:
        if _leads_beyond_clock_skew(self.captured_at, _utc_now()):
            raise ValueError("asset capture time cannot exceed the allowed future clock skew")
        if self.bystanders_present and not self.classifications:
            raise ValueError("bystanders require subject-level classifications")
        if not self.bystanders_present and self.classifications:
            raise ValueError("classifications contradict bystanders_present=false")
        if any(item.asset_id != self.asset_id for item in self.classifications):
            raise ValueError("all classifications must reference the current asset")
        has_anonymized_classification = any(
            item.disposition is BystanderDisposition.ANONYMIZED for item in self.classifications
        )
        if has_anonymized_classification:
            matching = [
                record
                for record in self.anonymization_records
                if record.derived_asset_id == self.asset_id
            ]
            if not matching:
                raise ValueError("ANONYMIZED requires evidence for this derived asset")
            if not any(record.output_sha256 == self.asset_sha256 for record in matching):
                raise ValueError("anonymization output hash must match the current asset")
            if self.source_asset_provenance is None:
                raise ValueError("ANONYMIZED requires source asset provenance")
            for record in matching:
                if (
                    record.source_asset_id != self.source_asset_provenance.asset_id
                    or record.source_asset_sha256 != self.source_asset_provenance.sha256
                ):
                    raise ValueError("anonymization record must match source asset provenance")
                if not (
                    self.source_asset_provenance.captured_at
                    <= record.processed_at
                    <= self.captured_at
                ):
                    raise ValueError(
                        "anonymization must occur after source capture and before derived capture"
                    )
        elif self.source_asset_provenance is not None:
            raise ValueError("source asset provenance is only valid for ANONYMIZED assets")
        return self


class UseAuthorization(StrictFrozenModel):
    decision: MaterialUseDecision
    reason_code: NonEmptyStr
    gate_revision_id: UUID4 | None = None
    requested_purpose: MaterialUsePurpose
