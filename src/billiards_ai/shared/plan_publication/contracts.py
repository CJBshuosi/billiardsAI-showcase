"""Strict final wire contracts for plan publication.

The legacy C0 v1 contracts intentionally remain in ``shared.models.training_plan``.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    UUID4,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models.drill_catalog import DrillDefinition
from billiards_ai.shared.models.events import CoreContractModel, Sha256, require_utc
from billiards_ai.shared.models.growth import Reliability
from billiards_ai.shared.models.training_plan import DrillReference, TrainingBlock, TrainingPlan

CanonicalString = Annotated[
    str,
    StringConstraints(strip_whitespace=False, min_length=1, pattern=r"^\S(?:.*\S)?$"),
]


def _canonical_hash(value: object) -> str:
    from billiards_ai.shared.plan_publication.codec import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uuid4_from_digest(material: bytes) -> UUID:
    raw = bytearray(hashlib.sha256(material).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


class PublicationContractModel(CoreContractModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class FinalTrainingPlan(PublicationContractModel):
    schema_version: Literal["2.0"] = "2.0"
    plan_id: UUID4
    user_id: UUID4
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    version: StrictInt = Field(ge=1)
    admission_status: Literal["VALIDATED", "APPROVED"]
    created_at: datetime
    valid_until: datetime
    blocks: tuple[TrainingBlock, ...] = Field(min_length=1)
    fallback_drill: DrillReference
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)
    _valid_until_utc = field_validator("valid_until")(require_utc)

    @classmethod
    def from_v1(
        cls,
        plan: TrainingPlan,
        *,
        snapshot_id: UUID,
        snapshot_version: int,
        admission_status: object,
    ) -> FinalTrainingPlan:
        status = getattr(admission_status, "value", admission_status)
        payload = {
            "schema_version": "2.0",
            "plan_id": plan.plan_id,
            "user_id": plan.user_id,
            "assessment_id": plan.assessment_id,
            "assessment_revision": plan.assessment_revision,
            "snapshot_id": snapshot_id,
            "snapshot_version": snapshot_version,
            "version": plan.version,
            "admission_status": status,
            "created_at": plan.created_at,
            "valid_until": plan.valid_until,
            "blocks": plan.blocks,
            "fallback_drill": plan.fallback_drill,
        }
        return cls.model_validate(payload | {"content_sha256": _canonical_hash(payload)})

    @model_validator(mode="after")
    def plan_is_ordered_and_hash_bound(self) -> Self:
        if self.valid_until <= self.created_at:
            raise ValueError("plan validity must follow creation")
        if tuple(block.order for block in self.blocks) != tuple(range(1, len(self.blocks) + 1)):
            raise ValueError("training block order must be contiguous from one")
        block_ids = tuple(block.block_id for block in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("training block IDs must be unique")
        expected = _canonical_hash(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected:
            raise ValueError("final plan content hash mismatch")
        return self


class GrowthRuntimeViewV2(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    focus_metric_codes: tuple[CanonicalString, ...]
    minimum_reliability: Reliability

    @field_validator("focus_metric_codes")
    @classmethod
    def metric_codes_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("focus metric codes must be sorted and unique")
        return value


class CapabilityManifest(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    manifest_id: CanonicalString
    manifest_version: StrictInt = Field(ge=1)
    target_device_id: CanonicalString
    capabilities: tuple[CanonicalString, ...]
    equipment_codes: tuple[CanonicalString, ...]
    content_sha256: Sha256

    @classmethod
    def build(cls, **values: object) -> CapabilityManifest:
        payload = {"schema_version": "1.0", **values}
        for field_name in ("capabilities", "equipment_codes"):
            value = payload.get(field_name)
            if type(value) is not tuple or not all(type(item) is str for item in value):
                raise TypeError(f"{field_name} must be a tuple of strings")
            payload[field_name] = tuple(sorted(set(value)))
        return cls.model_validate(payload | {"content_sha256": _canonical_hash(payload)})

    @model_validator(mode="after")
    def sets_and_hash_are_canonical(self) -> Self:
        for value in (self.capabilities, self.equipment_codes):
            if value != tuple(sorted(set(value))):
                raise ValueError("manifest code sets must be sorted and unique")
        expected = _canonical_hash(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected:
            raise ValueError("capability manifest content hash mismatch")
        return self


class AuthorizationBinding(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    authorization_id: UUID4
    authorization_version: StrictInt = Field(ge=1)
    user_id: UUID4
    session_id: UUID4
    target_device_id: CanonicalString
    issued_at: datetime
    expires_at: datetime

    _issued_at_utc = field_validator("issued_at")(require_utc)
    _expires_at_utc = field_validator("expires_at")(require_utc)

    @model_validator(mode="after")
    def window_is_ordered(self) -> Self:
        if self.expires_at <= self.issued_at:
            raise ValueError("authorization expiry must follow issue time")
        return self


class PublicVerificationKey(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["ED25519"] = "ED25519"
    key_id: CanonicalString
    key_version: StrictInt = Field(ge=1)
    purpose: Literal["PLAN_BUNDLE_SIGNING"] = "PLAN_BUNDLE_SIGNING"
    public_key_base64: CanonicalString
    issued_at: datetime
    not_before: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    rotation_predecessor_key_id: CanonicalString | None = None

    _issued_at_utc = field_validator("issued_at")(require_utc)
    _not_before_utc = field_validator("not_before")(require_utc)
    _expires_at_utc = field_validator("expires_at")(require_utc)
    _revoked_at_utc = field_validator("revoked_at")(
        lambda value: require_utc(value) if value else None
    )

    @model_validator(mode="after")
    def key_policy_and_bytes_are_valid(self) -> Self:
        if not self.issued_at <= self.not_before < self.expires_at:
            raise ValueError("key lifecycle window is invalid")
        if (
            self.revoked_at is not None
            and not self.not_before <= self.revoked_at <= self.expires_at
        ):
            raise ValueError("revocation must be within the active key window")
        if (self.key_version == 1) != (self.rotation_predecessor_key_id is None):
            raise ValueError("key rotation lineage must match key version")
        if self.rotation_predecessor_key_id == self.key_id:
            raise ValueError("key rotation lineage cannot contain a self-loop")
        try:
            raw = base64.b64decode(self.public_key_base64, validate=True)
        except Exception:
            raise ValueError("public key must use canonical base64") from None
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != self.public_key_base64:
            raise ValueError("public key must be 32 canonical Ed25519 bytes")
        return self


class PublicationPolicy(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    policy_version: Literal["1.0"] = "1.0"
    publication_sequence: StrictInt = Field(ge=1)
    issued_at: datetime
    not_before: datetime
    expires_at: datetime
    signing_key: PublicVerificationKey

    _issued_at_utc = field_validator("issued_at")(require_utc)
    _not_before_utc = field_validator("not_before")(require_utc)
    _expires_at_utc = field_validator("expires_at")(require_utc)


class PublishedPlanPayload(PublicationContractModel):
    schema_version: Literal["2.0"] = "2.0"
    bundle_id: UUID4
    plan: FinalTrainingPlan
    session_id: UUID4
    target_device_id: CanonicalString
    catalog_version: CanonicalString
    catalog_content_sha256: Sha256
    drill_definitions: tuple[DrillDefinition, ...] = Field(min_length=1)
    growth_runtime_view: GrowthRuntimeViewV2
    capability_manifest: CapabilityManifest
    authorization: AuthorizationBinding
    publication_policy: PublicationPolicy

    @property
    def publication_sequence(self) -> int:
        return self.publication_policy.publication_sequence

    @model_validator(mode="after")
    def nested_identity_and_order_are_exact(self) -> Self:
        if self.plan.user_id != self.authorization.user_id:
            raise ValueError("authorization user does not match plan")
        if self.session_id != self.authorization.session_id:
            raise ValueError("authorization session does not match bundle")
        if self.target_device_id != self.authorization.target_device_id:
            raise ValueError("authorization device does not match bundle")
        if (
            self.plan.snapshot_id != self.growth_runtime_view.snapshot_id
            or self.plan.snapshot_version != self.growth_runtime_view.snapshot_version
        ):
            raise ValueError("growth runtime view does not match plan snapshot")
        keys = tuple((item.drill_id, item.revision) for item in self.drill_definitions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("drill definitions must be sorted and unique")
        references = tuple(block.drill for block in self.plan.blocks) + (self.plan.fallback_drill,)
        reference_keys = {(item.drill_id, item.drill_revision) for item in references}
        if reference_keys != set(keys):
            raise ValueError("drill definitions must exactly cover all plan references")
        definitions = {(item.drill_id, item.revision): item for item in self.drill_definitions}
        for reference in references:
            definition = definitions[(reference.drill_id, reference.drill_revision)]
            if (
                reference.catalog_version != self.catalog_version
                or reference.required_capabilities != definition.required_capabilities
                or definition.review_status.value != "APPROVED"
            ):
                raise ValueError("drill definition does not match the approved plan reference")
            if not set(definition.required_capabilities).issubset(
                self.capability_manifest.capabilities
            ) or not set(definition.required_equipment_codes).issubset(
                self.capability_manifest.equipment_codes
            ):
                raise ValueError("bundle capability manifest cannot execute every drill")
        policy = self.publication_policy
        key = policy.signing_key
        if not (
            self.plan.created_at <= policy.issued_at <= policy.not_before < policy.expires_at
            and policy.expires_at <= self.plan.valid_until
            and self.authorization.issued_at <= policy.issued_at < self.authorization.expires_at
            and policy.expires_at <= self.authorization.expires_at
            and key.not_before <= policy.issued_at < key.expires_at
            and (key.revoked_at is None or policy.issued_at < key.revoked_at)
        ):
            raise ValueError("publication, authorization, plan, or key window is inconsistent")
        return self


class EvaluatorResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class AbstentionAction(StrEnum):
    RETRY = "RETRY"
    USE_PUBLISHED_FALLBACK = "USE_PUBLISHED_FALLBACK"
    SAFE_PAUSE = "SAFE_PAUSE"


class MetricCombination(StrEnum):
    ALL = "ALL"
    ANY = "ANY"


class ReplanContinueAction(StrEnum):
    USE_PUBLISHED_FALLBACK = "USE_PUBLISHED_FALLBACK"
    SAFE_PAUSE = "SAFE_PAUSE"


class ProgressionExecutionRuleV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluator_result: Literal[EvaluatorResult.SUCCESS] = EvaluatorResult.SUCCESS
    consecutive_count: StrictInt = Field(gt=0)
    difficulty_delta: StrictInt = Field(gt=0)


class RegressionExecutionRuleV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluator_result: Literal[EvaluatorResult.FAILURE] = EvaluatorResult.FAILURE
    consecutive_count: StrictInt = Field(gt=0)
    difficulty_delta: StrictInt = Field(lt=0)


class AbstentionRuleV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    consecutive_count: StrictInt = Field(gt=0)
    action: AbstentionAction
    reset_on: Literal["OBSERVED_SUPPORTED"] = "OBSERVED_SUPPORTED"


class SuccessMetricRuleV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    metric_code: CanonicalString
    target_attempts: StrictInt = Field(gt=0)
    minimum_successes: StrictInt = Field(ge=0)
    evaluator_code: CanonicalString
    evaluator_version: CanonicalString
    required_capabilities: tuple[CanonicalString, ...]

    @model_validator(mode="after")
    def metric_rule_is_canonical(self) -> Self:
        if self.minimum_successes > self.target_attempts:
            raise ValueError("minimum successes cannot exceed target attempts")
        if self.required_capabilities != tuple(sorted(set(self.required_capabilities))):
            raise ValueError("evaluator capabilities must be sorted and unique")
        return self


class SuccessPolicyV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    evaluator_policy_version: CanonicalString
    metric_rules: tuple[SuccessMetricRuleV1, ...] = Field(min_length=1)
    combination: MetricCombination

    @model_validator(mode="after")
    def metric_rules_are_ordered_and_unique(self) -> Self:
        keys = tuple(item.metric_code for item in self.metric_rules)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("success metric rules must be sorted and unique")
        return self


class ReplanRuleV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    reason_codes: tuple[CanonicalString, ...] = Field(min_length=1)
    continue_action: ReplanContinueAction

    @model_validator(mode="after")
    def reason_codes_are_canonical(self) -> Self:
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("replan reason codes must be sorted and unique")
        return self


class DrillExecutionPolicyDefinitionV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    drill_id: CanonicalString
    drill_revision: StrictInt = Field(ge=1)
    drill_content_sha256: Sha256
    required_capabilities: tuple[CanonicalString, ...]
    initial_difficulty: StrictInt
    minimum_runtime_difficulty: StrictInt
    maximum_runtime_difficulty: StrictInt
    progression_rule: ProgressionExecutionRuleV1
    regression_rule: RegressionExecutionRuleV1
    abstention_rule: AbstentionRuleV1
    success_policy: SuccessPolicyV1
    replan_rule: ReplanRuleV1
    fallback_drill: DrillReference | None

    @model_validator(mode="after")
    def policy_is_canonical_runtime_bounded_and_loop_free(self) -> Self:
        if self.required_capabilities != tuple(sorted(set(self.required_capabilities))):
            raise ValueError("policy capabilities must be sorted and unique")
        if not (
            self.minimum_runtime_difficulty
            <= self.initial_difficulty
            <= self.maximum_runtime_difficulty
        ):
            raise ValueError("initial runtime difficulty must be within runtime bounds")
        if any(
            not set(rule.required_capabilities) <= set(self.required_capabilities)
            for rule in self.success_policy.metric_rules
        ):
            raise ValueError("evaluator capabilities must be declared by the drill policy")
        own_key = (self.drill_id, self.drill_revision)
        if (
            self.fallback_drill is not None
            and (
                self.fallback_drill.drill_id,
                self.fallback_drill.drill_revision,
            )
            == own_key
        ):
            raise ValueError("execution policy fallback loop is forbidden")
        if self.fallback_drill is None and (
            self.abstention_rule.action is not AbstentionAction.SAFE_PAUSE
            or self.replan_rule.continue_action is not ReplanContinueAction.SAFE_PAUSE
        ):
            raise ValueError("terminal execution policy must use SAFE_PAUSE")
        return self

    def matches_drill_definition(self, drill: DrillDefinition) -> bool:
        metric_shape = tuple(
            (item.metric_code, item.target_attempts, item.minimum_successes)
            for item in drill.success_metrics
        )
        policy_metric_shape = tuple(
            (item.metric_code, item.target_attempts, item.minimum_successes)
            for item in self.success_policy.metric_rules
        )
        return (
            self.drill_id == drill.drill_id
            and self.drill_revision == drill.revision
            and self.drill_content_sha256 == drill.content_sha256
            and self.required_capabilities == drill.required_capabilities
            and metric_shape == policy_metric_shape
            and self.progression_rule.consecutive_count == drill.progression_rule.threshold
            and self.progression_rule.difficulty_delta == drill.progression_rule.level_delta
            and self.regression_rule.consecutive_count == drill.regression_rule.threshold
            and self.regression_rule.difficulty_delta == drill.regression_rule.level_delta
        )


class ExecutionPolicyCatalogV1(PublicationContractModel):
    schema_version: Literal["1.0"] = "1.0"
    catalog_version: CanonicalString
    drill_catalog_content_sha256: Sha256
    definitions: tuple[DrillExecutionPolicyDefinitionV1, ...] = Field(min_length=1)
    content_sha256: Sha256

    @classmethod
    def build(cls, **values: object) -> ExecutionPolicyCatalogV1:
        payload = {"schema_version": "1.0", **values}
        payload.pop("content_sha256", None)
        definitions = payload.get("definitions")
        if type(definitions) is not tuple:
            raise TypeError("execution policy definitions must be a tuple")
        payload["definitions"] = tuple(
            sorted(definitions, key=lambda item: (item.drill_id, item.drill_revision))
        )
        return cls.model_validate(payload | {"content_sha256": _canonical_hash(payload)})

    @model_validator(mode="after")
    def definitions_and_hash_are_canonical(self) -> Self:
        keys = tuple((item.drill_id, item.drill_revision) for item in self.definitions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("execution policy definitions must be sorted and unique")
        expected = _canonical_hash(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected:
            raise ValueError("execution policy catalog content hash mismatch")
        return self


class TrainingBlockV3(PublicationContractModel):
    schema_version: Literal["3.0"] = "3.0"
    block_id: UUID4
    order: StrictInt = Field(ge=1)
    drill: DrillReference
    target_finding_id: UUID4
    target_repetitions: StrictInt = Field(gt=0)
    initial_difficulty: StrictInt
    minimum_runtime_difficulty: StrictInt
    maximum_runtime_difficulty: StrictInt
    progression_rule: ProgressionExecutionRuleV1
    regression_rule: RegressionExecutionRuleV1
    abstention_rule: AbstentionRuleV1
    success_policy: SuccessPolicyV1
    replan_rule: ReplanRuleV1
    fallback_drill: DrillReference

    @model_validator(mode="after")
    def runtime_policy_is_bounded_and_consistent(self) -> Self:
        if not (
            self.minimum_runtime_difficulty
            <= self.initial_difficulty
            <= self.maximum_runtime_difficulty
        ):
            raise ValueError("initial runtime difficulty must be within runtime difficulty bounds")
        if self.target_repetitions != max(
            item.target_attempts for item in self.success_policy.metric_rules
        ):
            raise ValueError("target repetitions must match the success-policy attempt horizon")
        fallback_action = self.abstention_rule.action is AbstentionAction.USE_PUBLISHED_FALLBACK
        replan_fallback = (
            self.replan_rule.continue_action is ReplanContinueAction.USE_PUBLISHED_FALLBACK
        )
        if (fallback_action or replan_fallback) and self.fallback_drill == self.drill:
            raise ValueError("published fallback must differ from the primary drill")
        return self


class FinalTrainingPlanV3(PublicationContractModel):
    schema_version: Literal["3.0"] = "3.0"
    plan_id: UUID4
    user_id: UUID4
    assessment_id: UUID4
    assessment_revision: StrictInt = Field(ge=1)
    snapshot_id: UUID4
    snapshot_version: StrictInt = Field(ge=1)
    version: StrictInt = Field(ge=1)
    admission_status: Literal["VALIDATED", "APPROVED"]
    created_at: datetime
    valid_until: datetime
    blocks: tuple[TrainingBlockV3, ...] = Field(min_length=1)
    fallback_drill: DrillReference
    content_sha256: Sha256

    _created_at_utc = field_validator("created_at")(require_utc)
    _valid_until_utc = field_validator("valid_until")(require_utc)

    @model_validator(mode="after")
    def plan_is_ordered_fallback_bound_and_hash_bound(self) -> Self:
        if self.valid_until <= self.created_at:
            raise ValueError("plan validity must follow creation")
        if tuple(block.order for block in self.blocks) != tuple(range(1, len(self.blocks) + 1)):
            raise ValueError("training block order must be contiguous from one")
        block_ids = tuple(block.block_id for block in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("training block IDs must be unique")
        if any(block.fallback_drill != self.fallback_drill for block in self.blocks):
            raise ValueError("every block must bind the exact published plan fallback")
        expected = _canonical_hash(self.model_dump(mode="json", exclude={"content_sha256"}))
        if self.content_sha256 != expected:
            raise ValueError("final plan content hash mismatch")
        return self


class PublishedPlanPayloadV3(PublicationContractModel):
    schema_version: Literal["3.0"] = "3.0"
    bundle_id: UUID4
    plan: FinalTrainingPlanV3
    session_id: UUID4
    target_device_id: CanonicalString
    catalog_version: CanonicalString
    catalog_content_sha256: Sha256
    execution_policy_catalog: ExecutionPolicyCatalogV1
    drill_definitions: tuple[DrillDefinition, ...] = Field(min_length=1)
    growth_runtime_view: GrowthRuntimeViewV2
    capability_manifest: CapabilityManifest
    authorization: AuthorizationBinding
    publication_policy: PublicationPolicy

    @property
    def publication_sequence(self) -> int:
        return self.publication_policy.publication_sequence

    @model_validator(mode="after")
    def nested_identity_order_and_capabilities_are_exact(self) -> Self:
        if self.plan.user_id != self.authorization.user_id:
            raise ValueError("authorization user does not match plan")
        if self.session_id != self.authorization.session_id:
            raise ValueError("authorization session does not match bundle")
        if self.target_device_id != self.authorization.target_device_id:
            raise ValueError("authorization device does not match bundle")
        if (
            self.plan.snapshot_id != self.growth_runtime_view.snapshot_id
            or self.plan.snapshot_version != self.growth_runtime_view.snapshot_version
        ):
            raise ValueError("growth runtime view does not match plan snapshot")
        keys = tuple((item.drill_id, item.revision) for item in self.drill_definitions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("drill definitions must be sorted and unique")
        references = tuple(block.drill for block in self.plan.blocks) + (self.plan.fallback_drill,)
        if {(item.drill_id, item.drill_revision) for item in references} != set(keys):
            raise ValueError("drill definitions must exactly cover all plan references")
        definitions = {(item.drill_id, item.revision): item for item in self.drill_definitions}
        for reference in references:
            definition = definitions[(reference.drill_id, reference.drill_revision)]
            if (
                reference.catalog_version != self.catalog_version
                or reference.required_capabilities != definition.required_capabilities
                or definition.review_status.value != "APPROVED"
            ):
                raise ValueError("drill definition does not match approved plan reference")
            if not set(definition.required_capabilities).issubset(
                self.capability_manifest.capabilities
            ) or not set(definition.required_equipment_codes).issubset(
                self.capability_manifest.equipment_codes
            ):
                raise ValueError("bundle capability manifest cannot execute every drill")
        execution_catalog = self.execution_policy_catalog
        if (
            execution_catalog.catalog_version != self.catalog_version
            or execution_catalog.drill_catalog_content_sha256 != self.catalog_content_sha256
        ):
            raise ValueError("execution policy catalog does not bind the drill catalog")
        policy_keys = tuple(
            (item.drill_id, item.drill_revision) for item in execution_catalog.definitions
        )
        expected_policy_keys = tuple(sorted(set(keys)))
        if policy_keys != expected_policy_keys:
            raise ValueError("execution policies must exactly cover all executable drills")
        policies = {
            (item.drill_id, item.drill_revision): item for item in execution_catalog.definitions
        }
        for policy_key, execution_policy in policies.items():
            drill = definitions[policy_key]
            if not execution_policy.matches_drill_definition(drill):
                raise ValueError("execution policy does not match exact drill definition")
            if any(
                not set(rule.required_capabilities).issubset(self.capability_manifest.capabilities)
                for rule in execution_policy.success_policy.metric_rules
            ):
                raise ValueError("capability manifest cannot execute evaluator policy")
        fallback_key = (
            self.plan.fallback_drill.drill_id,
            self.plan.fallback_drill.drill_revision,
        )
        fallback_policy = policies[fallback_key]
        if fallback_policy.fallback_drill is not None:
            raise ValueError("plan fallback execution policy must terminate without a loop")
        for block in self.plan.blocks:
            block_policy = policies[(block.drill.drill_id, block.drill.drill_revision)]
            if (
                block.initial_difficulty != block_policy.initial_difficulty
                or block.minimum_runtime_difficulty != block_policy.minimum_runtime_difficulty
                or block.maximum_runtime_difficulty != block_policy.maximum_runtime_difficulty
                or block.progression_rule != block_policy.progression_rule
                or block.regression_rule != block_policy.regression_rule
                or block.abstention_rule != block_policy.abstention_rule
                or block.success_policy != block_policy.success_policy
                or block.replan_rule != block_policy.replan_rule
                or block.fallback_drill != block_policy.fallback_drill
            ):
                raise ValueError("training block does not match signed execution policy")
        policy = self.publication_policy
        key = policy.signing_key
        if not (
            self.plan.created_at <= policy.issued_at <= policy.not_before < policy.expires_at
            and policy.expires_at <= self.plan.valid_until
            and self.authorization.issued_at <= policy.issued_at < self.authorization.expires_at
            and policy.expires_at <= self.authorization.expires_at
            and key.not_before <= policy.issued_at < key.expires_at
            and (key.revoked_at is None or policy.issued_at < key.revoked_at)
        ):
            raise ValueError("publication, authorization, plan, or key window is inconsistent")
        return self


class PublishedPlanEnvelope(PublicationContractModel):
    payload: PublishedPlanPayload
    signature: CanonicalString

    @model_validator(mode="after")
    def signature_is_canonical_ed25519(self) -> Self:
        try:
            raw = base64.b64decode(self.signature, validate=True)
        except Exception:
            raise ValueError("signature must use canonical base64") from None
        if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != self.signature:
            raise ValueError("signature must contain 64 canonical Ed25519 bytes")
        return self


class DecodedPublishedPlanBundle(PublicationContractModel):
    envelope: PublishedPlanEnvelope
    signed_payload: bytes = Field(repr=False)
    signature_bytes: bytes = Field(repr=False)
    bundle_content_sha256: Sha256

    @model_validator(mode="after")
    def decoded_claims_match_exact_envelope(self) -> Self:
        from billiards_ai.shared.plan_publication.codec import canonical_json_bytes

        signed_payload = canonical_payload = canonical_json_bytes(self.envelope.payload)
        signature_bytes = base64.b64decode(self.envelope.signature, validate=True)
        if self.signed_payload != signed_payload:
            raise ValueError("decoded signed payload does not match envelope")
        if self.signature_bytes != signature_bytes:
            raise ValueError("decoded signature does not match envelope")
        if self.bundle_content_sha256 != hashlib.sha256(canonical_payload).hexdigest():
            raise ValueError("decoded bundle hash does not match signed payload")
        return self

    @property
    def payload(self) -> PublishedPlanPayload:
        return self.envelope.payload

    @property
    def planned_drill_keys(self) -> tuple[str, ...]:
        references = tuple(block.drill for block in self.payload.plan.blocks) + (
            self.payload.plan.fallback_drill,
        )
        return tuple(sorted({f"{item.drill_id}@{item.drill_revision}" for item in references}))


class PublishedPlanEnvelopeV3(PublicationContractModel):
    payload: PublishedPlanPayloadV3
    signature: CanonicalString

    @model_validator(mode="after")
    def signature_is_canonical_ed25519(self) -> Self:
        try:
            raw = base64.b64decode(self.signature, validate=True)
        except Exception:
            raise ValueError("signature must use canonical base64") from None
        if len(raw) != 64 or base64.b64encode(raw).decode("ascii") != self.signature:
            raise ValueError("signature must contain 64 canonical Ed25519 bytes")
        return self


class DecodedPublishedPlanBundleV3(PublicationContractModel):
    envelope: PublishedPlanEnvelopeV3
    signed_payload: bytes = Field(repr=False)
    signature_bytes: bytes = Field(repr=False)
    bundle_content_sha256: Sha256

    @model_validator(mode="after")
    def decoded_claims_match_exact_envelope(self) -> Self:
        from billiards_ai.shared.plan_publication.codec import canonical_json_bytes

        canonical_payload = canonical_json_bytes(self.envelope.payload)
        signature_bytes = base64.b64decode(self.envelope.signature, validate=True)
        if self.signed_payload != canonical_payload:
            raise ValueError("decoded signed payload does not match envelope")
        if self.signature_bytes != signature_bytes:
            raise ValueError("decoded signature does not match envelope")
        if self.bundle_content_sha256 != hashlib.sha256(canonical_payload).hexdigest():
            raise ValueError("decoded bundle hash does not match signed payload")
        return self

    @property
    def payload(self) -> PublishedPlanPayloadV3:
        return self.envelope.payload

    @property
    def planned_drill_keys(self) -> tuple[str, ...]:
        references = tuple(block.drill for block in self.payload.plan.blocks) + (
            self.payload.plan.fallback_drill,
        )
        return tuple(sorted({f"{item.drill_id}@{item.drill_revision}" for item in references}))


__all__ = [
    "AbstentionAction",
    "AbstentionRuleV1",
    "AuthorizationBinding",
    "CapabilityManifest",
    "DecodedPublishedPlanBundle",
    "DecodedPublishedPlanBundleV3",
    "DrillExecutionPolicyDefinitionV1",
    "EvaluatorResult",
    "ExecutionPolicyCatalogV1",
    "FinalTrainingPlan",
    "FinalTrainingPlanV3",
    "GrowthRuntimeViewV2",
    "MetricCombination",
    "ProgressionExecutionRuleV1",
    "PublicVerificationKey",
    "PublicationPolicy",
    "PublishedPlanEnvelope",
    "PublishedPlanEnvelopeV3",
    "PublishedPlanPayload",
    "PublishedPlanPayloadV3",
    "RegressionExecutionRuleV1",
    "ReplanContinueAction",
    "ReplanRuleV1",
    "SuccessMetricRuleV1",
    "SuccessPolicyV1",
    "TrainingBlockV3",
    "_uuid4_from_digest",
]
