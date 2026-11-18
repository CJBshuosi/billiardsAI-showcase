"""Fail-closed validation and publication of final training plans."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, final
from uuid import UUID

from pydantic import ConfigDict, field_validator

from billiards_ai.cloud.training.catalog import (
    DrillCatalog,
    DrillDefinition,
    DrillReviewStatus,
    validate_drill_reference,
)
from billiards_ai.cloud.training.execution_policy import ExecutionPolicyCatalogV1
from billiards_ai.shared.models.events import require_utc
from billiards_ai.shared.plan_publication.codec import (
    canonical_json_bytes,
    decode_bundle,
    decode_bundle_v3,
    encode_bundle,
    encode_bundle_v3,
)
from billiards_ai.shared.plan_publication.contracts import (
    AuthorizationBinding,
    CapabilityManifest,
    FinalTrainingPlan,
    FinalTrainingPlanV3,
    GrowthRuntimeViewV2,
    PublicationContractModel,
    PublicationPolicy,
    PublicVerificationKey,
    PublishedPlanEnvelope,
    PublishedPlanEnvelopeV3,
    PublishedPlanPayload,
    PublishedPlanPayloadV3,
    _uuid4_from_digest,
)
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Verifier,
    SignatureVerificationError,
)


class AdmissionStatus(StrEnum):
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"


class PublicationFailureCode(StrEnum):
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    ADMISSION_REJECTED = "ADMISSION_REJECTED"
    ASSESSMENT_SUPERSEDED = "ASSESSMENT_SUPERSEDED"
    PLAN_BINDING_MISMATCH = "PLAN_BINDING_MISMATCH"
    CATALOG_MISMATCH = "CATALOG_MISMATCH"
    DRILL_INCOMPLETE = "DRILL_INCOMPLETE"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    SAFETY_REJECTED = "SAFETY_REJECTED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_VALIDITY = "INVALID_VALIDITY"
    KEY_NOT_TRUSTED = "KEY_NOT_TRUSTED"
    ROLLBACK_REJECTED = "ROLLBACK_REJECTED"
    SIGNING_FAILED = "SIGNING_FAILED"


@final
@dataclass(frozen=True, slots=True)
class PlanAdmissionDecision:
    status: AdmissionStatus
    plan_id: UUID
    plan_version: int
    assessment_id: UUID
    assessment_revision: int
    snapshot_id: UUID
    snapshot_version: int
    authority_revision: str
    assessment_superseded: bool
    user_id: UUID
    session_id: UUID
    target_device_id: str
    authorization_id: UUID
    authorization_version: int
    catalog_version: str
    catalog_content_sha256: str
    capability_manifest_id: str
    capability_manifest_content_sha256: str
    validated_finding_ids: tuple[UUID, ...]
    execution_policy_catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not AdmissionStatus:
            raise TypeError("admission status is invalid")
        for value in (
            self.plan_id,
            self.assessment_id,
            self.snapshot_id,
            self.user_id,
            self.session_id,
            self.authorization_id,
        ):
            if type(value) is not UUID or value.version != 4:
                raise TypeError("admission identity must be UUID4")
        for value in (
            self.plan_version,
            self.assessment_revision,
            self.snapshot_version,
            self.authorization_version,
        ):
            if type(value) is not int or value < 1:
                raise TypeError("admission versions must be positive strict integers")
        for value in (
            self.authority_revision,
            self.target_device_id,
            self.catalog_version,
            self.capability_manifest_id,
        ):
            if type(value) is not str or not value.strip():
                raise TypeError("admission authority strings must be non-empty")
        for value in (
            self.catalog_content_sha256,
            self.capability_manifest_content_sha256,
        ):
            if type(value) is not str or len(value) != 64:
                raise TypeError("admission authority hashes must be sha256")
        if self.execution_policy_catalog_sha256 is not None and (
            type(self.execution_policy_catalog_sha256) is not str
            or len(self.execution_policy_catalog_sha256) != 64
        ):
            raise TypeError("execution policy authority hash must be sha256")
        if type(self.assessment_superseded) is not bool:
            raise TypeError("assessment superseded must be strict bool")
        if (
            not self.validated_finding_ids
            or any(
                type(value) is not UUID or value.version != 4
                for value in self.validated_finding_ids
            )
            or self.validated_finding_ids != tuple(sorted(set(self.validated_finding_ids)))
        ):
            raise TypeError("validated finding IDs must be non-empty sorted unique UUID4")


class PlanPublicationCandidate(PublicationContractModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    plan: FinalTrainingPlan | FinalTrainingPlanV3
    catalog: DrillCatalog
    execution_policy_catalog: ExecutionPolicyCatalogV1 | None = None
    growth_runtime_view: GrowthRuntimeViewV2
    capability_manifest: CapabilityManifest
    authorization: AuthorizationBinding
    requested_not_before: datetime
    requested_expires_at: datetime
    active_contraindication_codes: tuple[str, ...] = ()

    _requested_not_before_utc = field_validator("requested_not_before")(require_utc)
    _requested_expires_at_utc = field_validator("requested_expires_at")(require_utc)

    @field_validator("active_contraindication_codes")
    @classmethod
    def contraindications_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            not item
            or any(character.isspace() for character in item)
            or unicodedata.normalize("NFC", item) != item
            for item in value
        ):
            raise ValueError("contraindications must be non-empty whitespace-free NFC codes")
        if value != tuple(sorted(set(value))):
            raise ValueError("contraindications must be sorted and unique")
        return value

    @field_validator("execution_policy_catalog")
    @classmethod
    def execution_catalog_matches_plan_version(
        cls, value: ExecutionPolicyCatalogV1 | None, info
    ) -> ExecutionPolicyCatalogV1 | None:
        plan = info.data.get("plan")
        if type(plan) is FinalTrainingPlanV3 and value is None:
            raise ValueError("FinalTrainingPlan 3.0 requires an execution policy catalog")
        if type(plan) is FinalTrainingPlan and value is not None:
            raise ValueError("FinalTrainingPlan 2.0 cannot be default-upgraded with v3 policy")
        return value


@final
@dataclass(frozen=True, slots=True, repr=False)
class SigningMaterial:
    public_key: PublicVerificationKey
    private_key_handle: object = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.public_key) is not PublicVerificationKey or self.private_key_handle is None:
            raise TypeError("signing material is invalid")


class PlanAdmissionAuthorityPort(Protocol):
    def admit(self, candidate: PlanPublicationCandidate) -> PlanAdmissionDecision: ...


class PublicationSequenceConflict(RuntimeError):
    """The plan/version scope was already reserved for a different canonical request."""


class _PublicationBuildError(RuntimeError):
    pass


class PublicationSequencePort(Protocol):
    def reserve_or_replay(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        device_id: str,
        authorization_id: UUID,
        authorization_version: int,
        plan_id: UUID,
        plan_version: int,
        request_sha256: str,
        build: Callable[[int], bytes],
    ) -> bytes: ...


class SigningKeyAuthorityPort(Protocol):
    def resolve(self, *, at: datetime) -> SigningMaterial: ...

    def validate_rotation(self, key: PublicVerificationKey, *, at: datetime) -> bool: ...


class PlanBundleSignerPort(Protocol):
    def sign(self, handle: object, message: bytes) -> bytes: ...


@final
@dataclass(frozen=True, slots=True, repr=False)
class PublicationResult:
    bundle: PublishedPlanEnvelope | PublishedPlanEnvelopeV3 | None = None
    envelope_bytes: bytes | None = field(default=None, repr=False)
    failure_code: PublicationFailureCode | None = None

    def __post_init__(self) -> None:
        published = self.bundle is not None and self.envelope_bytes is not None
        if published == (self.failure_code is not None):
            raise ValueError("publication result must contain exactly one result branch")

    @property
    def published(self) -> bool:
        return self.bundle is not None

    def __repr__(self) -> str:
        return f"PublicationResult({'published' if self.published else 'rejected'})"


def _reject(code: PublicationFailureCode) -> PublicationResult:
    return PublicationResult(failure_code=code)


def _validate_v3_execution_policies(
    candidate: PlanPublicationCandidate,
    definitions: dict[tuple[str, int], DrillDefinition],
) -> PublicationFailureCode | None:
    plan = candidate.plan
    policy_catalog = candidate.execution_policy_catalog
    if type(plan) is not FinalTrainingPlanV3 or policy_catalog is None:
        return PublicationFailureCode.DRILL_INCOMPLETE
    if (
        policy_catalog.catalog_version != candidate.catalog.catalog_version
        or policy_catalog.drill_catalog_content_sha256 != candidate.catalog.content_sha256
    ):
        return PublicationFailureCode.CATALOG_MISMATCH
    policy_keys = tuple((item.drill_id, item.drill_revision) for item in policy_catalog.definitions)
    expected_policy_keys = tuple(
        sorted(
            {(block.drill.drill_id, block.drill.drill_revision) for block in plan.blocks}
            | {(plan.fallback_drill.drill_id, plan.fallback_drill.drill_revision)}
        )
    )
    if policy_keys != expected_policy_keys:
        return PublicationFailureCode.DRILL_INCOMPLETE
    policies = {(item.drill_id, item.drill_revision): item for item in policy_catalog.definitions}
    for key, policy in policies.items():
        drill = definitions.get(key)
        if drill is None or policy is None:
            return PublicationFailureCode.CATALOG_MISMATCH
        if not policy.matches_drill_definition(drill):
            return PublicationFailureCode.CATALOG_MISMATCH
        if any(
            not set(rule.required_capabilities).issubset(candidate.capability_manifest.capabilities)
            for rule in policy.success_policy.metric_rules
        ):
            return PublicationFailureCode.CAPABILITY_MISMATCH
    fallback_key = (plan.fallback_drill.drill_id, plan.fallback_drill.drill_revision)
    fallback_policy = policies[fallback_key]
    if fallback_policy.fallback_drill is not None:
        return PublicationFailureCode.DRILL_INCOMPLETE
    for block in plan.blocks:
        key = (block.drill.drill_id, block.drill.drill_revision)
        policy = policies[key]
        if (
            block.initial_difficulty != policy.initial_difficulty
            or block.minimum_runtime_difficulty != policy.minimum_runtime_difficulty
            or block.maximum_runtime_difficulty != policy.maximum_runtime_difficulty
            or block.progression_rule != policy.progression_rule
            or block.regression_rule != policy.regression_rule
            or block.abstention_rule != policy.abstention_rule
            or block.success_policy != policy.success_policy
            or block.replan_rule != policy.replan_rule
            or block.fallback_drill != policy.fallback_drill
            or block.fallback_drill != plan.fallback_drill
        ):
            return PublicationFailureCode.DRILL_INCOMPLETE
    return None


class PlanPublisher:
    def __init__(
        self,
        *,
        admission_authority: PlanAdmissionAuthorityPort,
        sequence_authority: PublicationSequencePort,
        signing_key_authority: SigningKeyAuthorityPort,
        signer: PlanBundleSignerPort,
    ) -> None:
        self._admission = admission_authority
        self._sequence = sequence_authority
        self._keys = signing_key_authority
        self._signer = signer

    def with_admission_authority(self, authority: PlanAdmissionAuthorityPort) -> PlanPublisher:
        return PlanPublisher(
            admission_authority=authority,
            sequence_authority=self._sequence,
            signing_key_authority=self._keys,
            signer=self._signer,
        )

    def with_sequence_authority(self, authority: PublicationSequencePort) -> PlanPublisher:
        return PlanPublisher(
            admission_authority=self._admission,
            sequence_authority=authority,
            signing_key_authority=self._keys,
            signer=self._signer,
        )

    def publish(self, candidate: PlanPublicationCandidate, *, now: datetime) -> PublicationResult:
        if type(now) is not datetime or now.tzinfo is not UTC:
            return _reject(PublicationFailureCode.INVALID_VALIDITY)
        try:
            candidate = PlanPublicationCandidate.model_validate(candidate.model_dump(mode="python"))
            decision = self._admission.admit(candidate)
        except Exception:
            return _reject(PublicationFailureCode.DEPENDENCY_UNAVAILABLE)
        if (
            type(decision) is not PlanAdmissionDecision
            or type(decision.status) is not AdmissionStatus
        ):
            return _reject(PublicationFailureCode.ADMISSION_REJECTED)
        if decision.status not in (AdmissionStatus.VALIDATED, AdmissionStatus.APPROVED):
            return _reject(PublicationFailureCode.ADMISSION_REJECTED)
        if decision.assessment_superseded:
            return _reject(PublicationFailureCode.ASSESSMENT_SUPERSEDED)
        plan = candidate.plan
        if (
            decision.plan_id != plan.plan_id
            or decision.plan_version != plan.version
            or decision.assessment_id != plan.assessment_id
            or decision.assessment_revision != plan.assessment_revision
            or decision.snapshot_id != plan.snapshot_id
            or decision.snapshot_version != plan.snapshot_version
            or decision.status.value != plan.admission_status
            or not {block.target_finding_id for block in plan.blocks}.issubset(
                decision.validated_finding_ids
            )
        ):
            return _reject(PublicationFailureCode.PLAN_BINDING_MISMATCH)
        authorization = candidate.authorization
        if (
            decision.user_id != plan.user_id
            or decision.session_id != authorization.session_id
            or decision.target_device_id != authorization.target_device_id
            or decision.authorization_id != authorization.authorization_id
            or decision.authorization_version != authorization.authorization_version
        ):
            return _reject(PublicationFailureCode.UNAUTHORIZED)
        if (
            decision.catalog_version != candidate.catalog.catalog_version
            or decision.catalog_content_sha256 != candidate.catalog.content_sha256
        ):
            return _reject(PublicationFailureCode.CATALOG_MISMATCH)
        if (
            decision.capability_manifest_id != candidate.capability_manifest.manifest_id
            or decision.capability_manifest_content_sha256
            != candidate.capability_manifest.content_sha256
        ):
            return _reject(PublicationFailureCode.CAPABILITY_MISMATCH)
        if (
            authorization.user_id != plan.user_id
            or authorization.target_device_id != candidate.capability_manifest.target_device_id
            or not authorization.issued_at <= now < authorization.expires_at
        ):
            return _reject(PublicationFailureCode.UNAUTHORIZED)
        if (
            candidate.requested_not_before != now
            or candidate.requested_expires_at <= candidate.requested_not_before
            or candidate.requested_expires_at > plan.valid_until
            or candidate.requested_expires_at > authorization.expires_at
        ):
            return _reject(PublicationFailureCode.INVALID_VALIDITY)
        if (
            candidate.growth_runtime_view.snapshot_id != plan.snapshot_id
            or candidate.growth_runtime_view.snapshot_version != plan.snapshot_version
        ):
            return _reject(PublicationFailureCode.PLAN_BINDING_MISMATCH)
        if type(plan) is FinalTrainingPlanV3:
            policy_catalog = candidate.execution_policy_catalog
            if (
                policy_catalog is None
                or decision.execution_policy_catalog_sha256 != policy_catalog.content_sha256
            ):
                return _reject(PublicationFailureCode.PLAN_BINDING_MISMATCH)

        references = tuple(block.drill for block in plan.blocks) + (plan.fallback_drill,)
        definitions = []
        for reference in references:
            validation = validate_drill_reference(catalog=candidate.catalog, reference=reference)
            if not validation.is_valid or validation.drill is None:
                return _reject(PublicationFailureCode.CATALOG_MISMATCH)
            drill = validation.drill
            if drill.review_status is not DrillReviewStatus.APPROVED or not drill.safety_codes:
                return _reject(PublicationFailureCode.SAFETY_REJECTED)
            if set(drill.contraindication_codes).intersection(
                candidate.active_contraindication_codes
            ):
                return _reject(PublicationFailureCode.SAFETY_REJECTED)
            if not set(drill.required_capabilities).issubset(
                candidate.capability_manifest.capabilities
            ) or not set(drill.required_equipment_codes).issubset(
                candidate.capability_manifest.equipment_codes
            ):
                return _reject(PublicationFailureCode.CAPABILITY_MISMATCH)
            definitions.append(drill)
        definitions = list({(item.drill_id, item.revision): item for item in definitions}.values())
        definitions.sort(key=lambda item: (item.drill_id, item.revision))
        if type(plan) is FinalTrainingPlanV3:
            failure = _validate_v3_execution_policies(
                candidate,
                {(item.drill_id, item.revision): item for item in definitions},
            )
            if failure is not None:
                return _reject(failure)

        try:
            material = self._keys.resolve(at=now)
        except Exception:
            return _reject(PublicationFailureCode.DEPENDENCY_UNAVAILABLE)
        if type(material) is not SigningMaterial:
            return _reject(PublicationFailureCode.KEY_NOT_TRUSTED)
        key = material.public_key
        if (
            key.algorithm != "ED25519"
            or key.purpose != "PLAN_BUNDLE_SIGNING"
            or now < key.not_before
            or now >= key.expires_at
            or (key.revoked_at is not None and now >= key.revoked_at)
        ):
            return _reject(PublicationFailureCode.KEY_NOT_TRUSTED)
        try:
            if self._keys.validate_rotation(key, at=now) is not True:
                return _reject(PublicationFailureCode.KEY_NOT_TRUSTED)
        except Exception:
            return _reject(PublicationFailureCode.KEY_NOT_TRUSTED)

        request_sha256 = hashlib.sha256(
            canonical_json_bytes({"candidate": candidate, "issued_at": now})
        ).hexdigest()

        def build(sequence: int) -> bytes:
            if type(sequence) is not int or sequence < 1:
                raise PublicationSequenceConflict("invalid reserved sequence")
            bundle_id = _uuid4_from_digest(
                canonical_json_bytes(
                    {
                        "authorization_id": authorization.authorization_id,
                        "authorization_version": authorization.authorization_version,
                        "device_id": authorization.target_device_id,
                        "plan_id": plan.plan_id,
                        "plan_version": plan.version,
                        "sequence": sequence,
                        "session_id": authorization.session_id,
                        "user_id": plan.user_id,
                    }
                )
            )
            try:
                common = {
                    "bundle_id": bundle_id,
                    "plan": plan,
                    "session_id": authorization.session_id,
                    "target_device_id": authorization.target_device_id,
                    "catalog_version": candidate.catalog.catalog_version,
                    "catalog_content_sha256": candidate.catalog.content_sha256,
                    "drill_definitions": tuple(definitions),
                    "growth_runtime_view": candidate.growth_runtime_view,
                    "capability_manifest": candidate.capability_manifest,
                    "authorization": authorization,
                    "publication_policy": PublicationPolicy(
                        publication_sequence=sequence,
                        issued_at=now,
                        not_before=now,
                        expires_at=candidate.requested_expires_at,
                        signing_key=key,
                    ),
                }
                if type(plan) is FinalTrainingPlanV3:
                    policy_catalog = candidate.execution_policy_catalog
                    if policy_catalog is None:
                        raise _PublicationBuildError
                    payload = PublishedPlanPayloadV3(
                        **common,
                        execution_policy_catalog=policy_catalog,
                    )
                else:
                    payload = PublishedPlanPayload(**common)
                signed_payload = canonical_json_bytes(payload)
                signature = self._signer.sign(material.private_key_handle, signed_payload)
                CryptographyEd25519Verifier().verify(key, signed_payload, signature, at=now)
                if type(payload) is PublishedPlanPayloadV3:
                    return encode_bundle_v3(payload, signature)
                return encode_bundle(payload, signature)
            except (PublicationSequenceConflict, SignatureVerificationError):
                raise
            except Exception as error:
                raise _PublicationBuildError from error

        try:
            envelope_bytes = self._sequence.reserve_or_replay(
                user_id=plan.user_id,
                session_id=authorization.session_id,
                device_id=authorization.target_device_id,
                authorization_id=authorization.authorization_id,
                authorization_version=authorization.authorization_version,
                plan_id=plan.plan_id,
                plan_version=plan.version,
                request_sha256=request_sha256,
                build=build,
            )
        except PublicationSequenceConflict:
            return _reject(PublicationFailureCode.ROLLBACK_REJECTED)
        except (_PublicationBuildError, SignatureVerificationError):
            return _reject(PublicationFailureCode.SIGNING_FAILED)
        except Exception:
            return _reject(PublicationFailureCode.DEPENDENCY_UNAVAILABLE)
        try:
            decoded = (
                decode_bundle_v3(envelope_bytes)
                if type(plan) is FinalTrainingPlanV3
                else decode_bundle(envelope_bytes)
            )
            payload = decoded.payload
            sequence = payload.publication_sequence
            expected_bundle_id = _uuid4_from_digest(
                canonical_json_bytes(
                    {
                        "authorization_id": authorization.authorization_id,
                        "authorization_version": authorization.authorization_version,
                        "device_id": authorization.target_device_id,
                        "plan_id": plan.plan_id,
                        "plan_version": plan.version,
                        "sequence": sequence,
                        "session_id": authorization.session_id,
                        "user_id": plan.user_id,
                    }
                )
            )
            if (
                type(sequence) is not int
                or sequence < 1
                or payload.bundle_id != expected_bundle_id
                or payload.plan != plan
                or payload.authorization != authorization
                or payload.target_device_id != authorization.target_device_id
                or payload.catalog_version != candidate.catalog.catalog_version
                or payload.catalog_content_sha256 != candidate.catalog.content_sha256
                or payload.drill_definitions != tuple(definitions)
                or payload.growth_runtime_view != candidate.growth_runtime_view
                or payload.capability_manifest != candidate.capability_manifest
                or payload.publication_policy.issued_at != now
                or payload.publication_policy.not_before != now
                or payload.publication_policy.expires_at != candidate.requested_expires_at
                or payload.publication_policy.signing_key != key
            ):
                return _reject(PublicationFailureCode.ROLLBACK_REJECTED)
            if type(plan) is FinalTrainingPlanV3:
                policy_catalog = candidate.execution_policy_catalog
                if (
                    type(payload) is not PublishedPlanPayloadV3
                    or policy_catalog is None
                    or payload.execution_policy_catalog != policy_catalog
                ):
                    return _reject(PublicationFailureCode.ROLLBACK_REJECTED)
            CryptographyEd25519Verifier().verify(
                key,
                decoded.signed_payload,
                decoded.signature_bytes,
                at=now,
            )
        except SignatureVerificationError:
            return _reject(PublicationFailureCode.SIGNING_FAILED)
        except Exception:
            return _reject(PublicationFailureCode.ROLLBACK_REJECTED)
        return PublicationResult(bundle=decoded.envelope, envelope_bytes=envelope_bytes)
