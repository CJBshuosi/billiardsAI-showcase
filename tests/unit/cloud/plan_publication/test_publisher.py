from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from billiards_ai.cloud.plan_publication.publisher import (
    AdmissionStatus,
    PlanAdmissionDecision,
    PlanPublicationCandidate,
    PlanPublisher,
    PublicationFailureCode,
    PublicationSequenceConflict,
    SigningMaterial,
)
from billiards_ai.cloud.training.catalog import DrillCatalog
from billiards_ai.cloud.training.execution_policy import (
    DrillExecutionPolicyDefinitionV1,
    ExecutionPolicyCatalogV1,
)
from billiards_ai.shared.models.growth import Reliability
from billiards_ai.shared.models.training_plan import TrainingPlan
from billiards_ai.shared.plan_publication.admission import admission_view
from billiards_ai.shared.plan_publication.codec import (
    CanonicalCodecError,
    canonical_json_bytes,
    decode_bundle,
    decode_bundle_v3,
)
from billiards_ai.shared.plan_publication.contracts import (
    AbstentionAction,
    AbstentionRuleV1,
    AuthorizationBinding,
    CapabilityManifest,
    FinalTrainingPlan,
    FinalTrainingPlanV3,
    GrowthRuntimeViewV2,
    MetricCombination,
    ProgressionExecutionRuleV1,
    PublishedPlanPayloadV3,
    RegressionExecutionRuleV1,
    ReplanContinueAction,
    ReplanRuleV1,
    SuccessMetricRuleV1,
    SuccessPolicyV1,
    TrainingBlockV3,
)
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Signer,
    OpaqueEd25519PrivateKey,
    public_verification_key,
)

ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
NOW = datetime(2026, 8, 5, 9, tzinfo=UTC)
SESSION_ID = UUID("00000000-0000-4000-8000-000000000701")


class FakeAdmissionAuthority:
    def __init__(self, decision: PlanAdmissionDecision | Exception) -> None:
        self.decision = decision

    def admit(self, candidate: PlanPublicationCandidate) -> PlanAdmissionDecision:
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class FakeSequenceAuthority:
    def __init__(self, values: tuple[int, ...] = (1, 2)) -> None:
        self.values = list(values)
        self.entries: dict[tuple[object, ...], tuple[str, bytes]] = {}
        self.lock = Lock()

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
    ) -> bytes:
        key = (
            user_id,
            session_id,
            device_id,
            authorization_id,
            authorization_version,
            plan_id,
            plan_version,
        )
        with self.lock:
            existing = self.entries.get(key)
            if existing is not None:
                if existing[0] != request_sha256:
                    raise PublicationSequenceConflict("request conflict")
                return existing[1]
            sequence = self.values.pop(0)
            envelope = build(sequence)
            self.entries[key] = (request_sha256, envelope)
            return envelope


@dataclass(frozen=True)
class FakeSigningKeyAuthority:
    material: SigningMaterial

    def resolve(self, *, at: datetime) -> SigningMaterial:
        return self.material

    def validate_rotation(self, key, *, at: datetime) -> bool:
        return key == self.material.public_key


def _legacy_plan() -> TrainingPlan:
    return TrainingPlan.model_validate_json((ROOT / "training-plan.json").read_text())


def _catalog() -> DrillCatalog:
    return DrillCatalog.model_validate_json((ROOT / "drill-catalog.json").read_text())


def _candidate() -> PlanPublicationCandidate:
    legacy = _legacy_plan()
    plan = FinalTrainingPlan.from_v1(
        legacy,
        snapshot_id=UUID("00000000-0000-4000-8000-000000000008"),
        snapshot_version=1,
        admission_status=AdmissionStatus.APPROVED,
    )

    catalog = _catalog()
    manifest = CapabilityManifest.build(
        manifest_id="table-profile-1",
        manifest_version=1,
        target_device_id="dilan-table-01",
        capabilities=("CUE_BALL_TARGET_ZONE", "TABLE_MM_COORDINATES"),
        equipment_codes=("STANDARD_TABLE",),
    )
    return PlanPublicationCandidate(
        plan=plan,
        catalog=catalog,
        growth_runtime_view=GrowthRuntimeViewV2(
            snapshot_id=plan.snapshot_id,
            snapshot_version=1,
            focus_metric_codes=("large_angle_potting.success_rate",),
            minimum_reliability=Reliability.MEDIUM,
        ),
        capability_manifest=manifest,
        authorization=AuthorizationBinding(
            authorization_id=UUID("00000000-0000-4000-8000-000000000702"),
            authorization_version=1,
            user_id=plan.user_id,
            session_id=SESSION_ID,
            target_device_id="dilan-table-01",
            issued_at=NOW - timedelta(minutes=5),
            expires_at=NOW + timedelta(days=1),
        ),
        requested_not_before=NOW,
        requested_expires_at=NOW + timedelta(days=1),
    )


def _candidate_v3() -> PlanPublicationCandidate:
    base = _candidate()
    old_block = base.plan.blocks[0]
    primary = next(
        item for item in base.catalog.drills if item.drill_id == old_block.drill.drill_id
    )
    success_policy = SuccessPolicyV1(
        evaluator_policy_version="edge-success-v1",
        metric_rules=tuple(
            SuccessMetricRuleV1(
                metric_code=item.metric_code,
                target_attempts=item.target_attempts,
                minimum_successes=item.minimum_successes,
                evaluator_code="POT_RESULT",
                evaluator_version="1.0",
                required_capabilities=primary.required_capabilities,
            )
            for item in primary.success_metrics
        ),
        combination=MetricCombination.ALL,
    )
    progression = ProgressionExecutionRuleV1(
        evaluator_result="SUCCESS",
        consecutive_count=primary.progression_rule.threshold,
        difficulty_delta=primary.progression_rule.level_delta,
    )
    regression = RegressionExecutionRuleV1(
        evaluator_result="FAILURE",
        consecutive_count=primary.regression_rule.threshold,
        difficulty_delta=primary.regression_rule.level_delta,
    )
    abstention = AbstentionRuleV1(
        consecutive_count=2,
        action=AbstentionAction.USE_PUBLISHED_FALLBACK,
        reset_on="OBSERVED_SUPPORTED",
    )
    replan = ReplanRuleV1(
        reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
        continue_action=ReplanContinueAction.USE_PUBLISHED_FALLBACK,
    )
    block = TrainingBlockV3(
        block_id=old_block.block_id,
        order=old_block.order,
        drill=old_block.drill,
        target_finding_id=old_block.target_finding_id,
        target_repetitions=primary.success_metrics[0].target_attempts,
        initial_difficulty=7,
        minimum_runtime_difficulty=5,
        maximum_runtime_difficulty=9,
        progression_rule=progression,
        regression_rule=regression,
        abstention_rule=abstention,
        success_policy=success_policy,
        replan_rule=replan,
        fallback_drill=base.plan.fallback_drill,
    )
    plan_payload = base.plan.model_dump(
        mode="python", exclude={"schema_version", "blocks", "content_sha256"}
    ) | {"schema_version": "3.0", "blocks": (block,)}
    plan = FinalTrainingPlanV3.model_validate(
        plan_payload
        | {"content_sha256": hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest()}
    )
    policy = DrillExecutionPolicyDefinitionV1(
        drill_id=primary.drill_id,
        drill_revision=primary.revision,
        drill_content_sha256=primary.content_sha256,
        required_capabilities=primary.required_capabilities,
        initial_difficulty=block.initial_difficulty,
        minimum_runtime_difficulty=block.minimum_runtime_difficulty,
        maximum_runtime_difficulty=block.maximum_runtime_difficulty,
        progression_rule=progression,
        regression_rule=regression,
        abstention_rule=abstention,
        success_policy=success_policy,
        replan_rule=replan,
        fallback_drill=base.plan.fallback_drill,
    )
    fallback = next(
        item for item in base.catalog.drills if item.drill_id == base.plan.fallback_drill.drill_id
    )
    fallback_success = SuccessPolicyV1(
        evaluator_policy_version="edge-success-v1",
        metric_rules=tuple(
            SuccessMetricRuleV1(
                metric_code=item.metric_code,
                target_attempts=item.target_attempts,
                minimum_successes=item.minimum_successes,
                evaluator_code="POT_RESULT",
                evaluator_version="1.0",
                required_capabilities=fallback.required_capabilities,
            )
            for item in fallback.success_metrics
        ),
        combination=MetricCombination.ALL,
    )
    fallback_policy = DrillExecutionPolicyDefinitionV1(
        drill_id=fallback.drill_id,
        drill_revision=fallback.revision,
        drill_content_sha256=fallback.content_sha256,
        required_capabilities=fallback.required_capabilities,
        initial_difficulty=3,
        minimum_runtime_difficulty=1,
        maximum_runtime_difficulty=5,
        progression_rule=ProgressionExecutionRuleV1(
            evaluator_result="SUCCESS",
            consecutive_count=fallback.progression_rule.threshold,
            difficulty_delta=fallback.progression_rule.level_delta,
        ),
        regression_rule=RegressionExecutionRuleV1(
            evaluator_result="FAILURE",
            consecutive_count=fallback.regression_rule.threshold,
            difficulty_delta=fallback.regression_rule.level_delta,
        ),
        abstention_rule=AbstentionRuleV1(
            consecutive_count=2,
            action=AbstentionAction.SAFE_PAUSE,
            reset_on="OBSERVED_SUPPORTED",
        ),
        success_policy=fallback_success,
        replan_rule=ReplanRuleV1(
            reason_codes=("FALLBACK_UNAVAILABLE", "TARGET_OUT_OF_RANGE"),
            continue_action=ReplanContinueAction.SAFE_PAUSE,
        ),
        fallback_drill=None,
    )
    policy_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=base.catalog.catalog_version,
        drill_catalog_content_sha256=base.catalog.content_sha256,
        definitions=(policy, fallback_policy),
    )
    values = base.model_dump(mode="python")
    values.update(plan=plan, execution_policy_catalog=policy_catalog)
    return PlanPublicationCandidate.model_validate(values)


def _publisher(candidate: PlanPublicationCandidate) -> PlanPublisher:
    private = Ed25519PrivateKey.generate()
    key = public_verification_key(
        private.public_key(),
        key_id="plan-key-1",
        key_version=1,
        issued_at=NOW - timedelta(days=1),
        not_before=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=30),
    )
    decision = PlanAdmissionDecision(
        status=AdmissionStatus.APPROVED,
        plan_id=candidate.plan.plan_id,
        plan_version=candidate.plan.version,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        authority_revision="assessment-authority:7",
        assessment_superseded=False,
        user_id=candidate.plan.user_id,
        session_id=candidate.authorization.session_id,
        target_device_id=candidate.authorization.target_device_id,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        catalog_version=candidate.catalog.catalog_version,
        catalog_content_sha256=candidate.catalog.content_sha256,
        capability_manifest_id=candidate.capability_manifest.manifest_id,
        capability_manifest_content_sha256=candidate.capability_manifest.content_sha256,
        validated_finding_ids=tuple(
            sorted({block.target_finding_id for block in candidate.plan.blocks})
        ),
        execution_policy_catalog_sha256=(
            candidate.execution_policy_catalog.content_sha256
            if candidate.execution_policy_catalog is not None
            else None
        ),
    )
    return PlanPublisher(
        admission_authority=FakeAdmissionAuthority(decision),
        sequence_authority=FakeSequenceAuthority(),
        signing_key_authority=FakeSigningKeyAuthority(
            SigningMaterial(
                public_key=key,
                private_key_handle=OpaqueEd25519PrivateKey(private),
            )
        ),
        signer=CryptographyEd25519Signer(),
    )


def test_publisher_emits_signed_complete_bundle_with_monotonic_sequence() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)

    first = publisher.publish(candidate, now=NOW)
    second = publisher.publish(candidate, now=NOW)

    assert first.published and second.published
    assert first.bundle is not None and second.bundle is not None
    assert first.bundle.payload.publication_sequence == 1
    assert second.bundle.payload.publication_sequence == 1
    assert first.envelope_bytes == second.envelope_bytes
    assert {item.drill_id for item in first.bundle.payload.drill_definitions} == {
        "large-angle-potting-01",
        "straight-potting-01",
    }
    assert first.envelope_bytes is not None
    assert json.loads(first.envelope_bytes) == json.loads(first.envelope_bytes.decode())
    decoded = decode_bundle(first.envelope_bytes)
    view = admission_view(decoded)
    assert view.bundle_content_sha256 == decoded.bundle_content_sha256
    assert view.signature_algorithm == "ED25519"
    assert view.not_before == view.issued_at
    assert view.planned_drill_keys == (
        "large-angle-potting-01@1",
        "straight-potting-01@1",
    )
    assert tuple(item.drill_key for item in view.drill_definitions) == view.planned_drill_keys


def test_publisher_v3_signs_all_execution_fields_and_requires_explicit_decoder() -> None:
    candidate = _candidate_v3()

    result = _publisher(candidate).publish(candidate, now=NOW)

    assert result.published and result.envelope_bytes is not None
    decoded = decode_bundle_v3(result.envelope_bytes)
    block = decoded.payload.plan.blocks[0]
    assert decoded.payload.schema_version == "3.0"
    assert block.initial_difficulty == 7
    assert block.minimum_runtime_difficulty == 5
    assert block.maximum_runtime_difficulty == 9
    assert block.success_policy.metric_rules[0].evaluator_code == "POT_RESULT"
    assert decoded.payload.execution_policy_catalog == candidate.execution_policy_catalog
    assert {
        (item.drill_id, item.drill_revision)
        for item in decoded.payload.execution_policy_catalog.definitions
    } == {
        (candidate.plan.blocks[0].drill.drill_id, candidate.plan.blocks[0].drill.drill_revision),
        (candidate.plan.fallback_drill.drill_id, candidate.plan.fallback_drill.drill_revision),
    }
    assert b'"initial_difficulty":7' in decoded.signed_payload
    assert b'"combination":"ALL"' in decoded.signed_payload
    with pytest.raises(CanonicalCodecError):
        decode_bundle(result.envelope_bytes)


def test_publisher_v3_rejects_policy_catalog_missing_plan_fallback() -> None:
    candidate = _candidate_v3()
    assert candidate.execution_policy_catalog is not None
    primary_key = (
        candidate.plan.blocks[0].drill.drill_id,
        candidate.plan.blocks[0].drill.drill_revision,
    )
    primary_only = tuple(
        item
        for item in candidate.execution_policy_catalog.definitions
        if (item.drill_id, item.drill_revision) == primary_key
    )
    incomplete_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=candidate.catalog.catalog_version,
        drill_catalog_content_sha256=candidate.catalog.content_sha256,
        definitions=primary_only,
    )
    values = candidate.model_dump(mode="python")
    values["execution_policy_catalog"] = incomplete_catalog
    incomplete = PlanPublicationCandidate.model_validate(values)

    result = _publisher(incomplete).publish(incomplete, now=NOW)

    assert result.failure_code is PublicationFailureCode.DRILL_INCOMPLETE


def test_v3_signed_payload_embeds_complete_policy_catalog_not_hash_only() -> None:
    candidate = _candidate_v3()
    result = _publisher(candidate).publish(candidate, now=NOW)
    assert result.envelope_bytes is not None
    decoded = decode_bundle_v3(result.envelope_bytes)
    signed = json.loads(decoded.signed_payload)

    assert "execution_policy_catalog" in signed
    assert "execution_policy_catalog_sha256" not in signed

    hash_only = decoded.payload.model_dump(mode="python")
    embedded = hash_only.pop("execution_policy_catalog")
    hash_only["execution_policy_catalog_sha256"] = embedded["content_sha256"]
    with pytest.raises(ValidationError):
        PublishedPlanPayloadV3.model_validate(hash_only)


def test_publisher_v3_rejects_extra_execution_policy_definition() -> None:
    candidate = _candidate_v3()
    assert candidate.execution_policy_catalog is not None
    terminal = next(
        item
        for item in candidate.execution_policy_catalog.definitions
        if item.fallback_drill is None
    )
    extra_drill = next(
        item for item in candidate.catalog.drills if item.drill_id == "capability-target-zone-01"
    )
    extra_success = SuccessPolicyV1(
        evaluator_policy_version="edge-success-v1",
        metric_rules=tuple(
            SuccessMetricRuleV1(
                metric_code=item.metric_code,
                target_attempts=item.target_attempts,
                minimum_successes=item.minimum_successes,
                evaluator_code="TARGET_ZONE_RESULT",
                evaluator_version="1.0",
                required_capabilities=extra_drill.required_capabilities,
            )
            for item in extra_drill.success_metrics
        ),
        combination=MetricCombination.ALL,
    )
    extra = DrillExecutionPolicyDefinitionV1.model_validate(
        terminal.model_dump(mode="python")
        | {
            "drill_id": extra_drill.drill_id,
            "drill_revision": extra_drill.revision,
            "drill_content_sha256": extra_drill.content_sha256,
            "required_capabilities": extra_drill.required_capabilities,
            "progression_rule": ProgressionExecutionRuleV1(
                evaluator_result="SUCCESS",
                consecutive_count=extra_drill.progression_rule.threshold,
                difficulty_delta=extra_drill.progression_rule.level_delta,
            ),
            "regression_rule": RegressionExecutionRuleV1(
                evaluator_result="FAILURE",
                consecutive_count=extra_drill.regression_rule.threshold,
                difficulty_delta=extra_drill.regression_rule.level_delta,
            ),
            "success_policy": extra_success,
        }
    )
    extra_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=candidate.catalog.catalog_version,
        drill_catalog_content_sha256=candidate.catalog.content_sha256,
        definitions=(*candidate.execution_policy_catalog.definitions, extra),
    )
    values = candidate.model_dump(mode="python")
    values["execution_policy_catalog"] = extra_catalog
    extra_candidate = PlanPublicationCandidate.model_validate(values)

    result = _publisher(extra_candidate).publish(extra_candidate, now=NOW)

    assert result.failure_code is PublicationFailureCode.DRILL_INCOMPLETE


def test_publisher_v3_rejects_unauthorized_fallback_policy_tampering() -> None:
    candidate = _candidate_v3()
    assert candidate.execution_policy_catalog is not None
    fallback_key = (
        candidate.plan.fallback_drill.drill_id,
        candidate.plan.fallback_drill.drill_revision,
    )
    fallback_policy = next(
        item
        for item in candidate.execution_policy_catalog.definitions
        if (item.drill_id, item.drill_revision) == fallback_key
    )
    metric = fallback_policy.success_policy.metric_rules[0].model_copy(
        update={"evaluator_code": "TAMPERED_EVALUATOR"}
    )
    tampered_policy = DrillExecutionPolicyDefinitionV1.model_validate(
        fallback_policy.model_dump(mode="python")
        | {
            "success_policy": fallback_policy.success_policy.model_copy(
                update={"metric_rules": (metric,)}
            )
        }
    )
    tampered_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=candidate.catalog.catalog_version,
        drill_catalog_content_sha256=candidate.catalog.content_sha256,
        definitions=tuple(
            tampered_policy if (item.drill_id, item.drill_revision) == fallback_key else item
            for item in candidate.execution_policy_catalog.definitions
        ),
    )
    values = candidate.model_dump(mode="python")
    values["execution_policy_catalog"] = tampered_catalog
    tampered = PlanPublicationCandidate.model_validate(values)

    result = _publisher(candidate).publish(tampered, now=NOW)

    assert result.failure_code is PublicationFailureCode.PLAN_BINDING_MISMATCH


def test_fallback_execution_policy_rejects_self_loop() -> None:
    candidate = _candidate_v3()
    assert candidate.execution_policy_catalog is not None
    primary_policy = candidate.execution_policy_catalog.definitions[0]
    fallback = next(
        item
        for item in candidate.catalog.drills
        if item.drill_id == candidate.plan.fallback_drill.drill_id
    )
    fallback_success = primary_policy.success_policy.model_copy(
        update={
            "metric_rules": tuple(
                primary_policy.success_policy.metric_rules[0].model_copy(
                    update={
                        "metric_code": item.metric_code,
                        "target_attempts": item.target_attempts,
                        "minimum_successes": item.minimum_successes,
                        "required_capabilities": fallback.required_capabilities,
                    }
                )
                for item in fallback.success_metrics
            )
        }
    )
    values = primary_policy.model_dump(mode="python")
    values.update(
        drill_id=fallback.drill_id,
        drill_revision=fallback.revision,
        drill_content_sha256=fallback.content_sha256,
        required_capabilities=fallback.required_capabilities,
        success_policy=fallback_success,
        fallback_drill=candidate.plan.fallback_drill,
    )

    with pytest.raises(ValidationError, match="fallback loop"):
        DrillExecutionPolicyDefinitionV1.model_validate(values)


@pytest.mark.parametrize("mismatch", ["threshold", "evaluator", "fallback", "revision"])
def test_publisher_v3_fails_closed_on_execution_policy_mismatch(mismatch: str) -> None:
    candidate = _candidate_v3()
    assert candidate.execution_policy_catalog is not None
    definition = candidate.execution_policy_catalog.definitions[0]
    updates: dict[str, object]
    if mismatch == "threshold":
        updates = {
            "progression_rule": definition.progression_rule.model_copy(
                update={"consecutive_count": definition.progression_rule.consecutive_count + 1}
            )
        }
    elif mismatch == "evaluator":
        metric = definition.success_policy.metric_rules[0].model_copy(
            update={"evaluator_code": "DIFFERENT_EVALUATOR"}
        )
        updates = {
            "success_policy": definition.success_policy.model_copy(
                update={"metric_rules": (metric,)}
            )
        }
    elif mismatch == "fallback":
        updates = {
            "fallback_drill": candidate.plan.fallback_drill.model_copy(
                update={
                    "drill_id": "capability-target-zone-01",
                    "required_capabilities": (
                        "CUE_BALL_TARGET_ZONE",
                        "TABLE_MM_COORDINATES",
                    ),
                }
            )
        }
    else:
        updates = {"drill_content_sha256": "0" * 64}
    changed = DrillExecutionPolicyDefinitionV1.model_validate(
        definition.model_dump(mode="python") | updates
    )
    policy_catalog = ExecutionPolicyCatalogV1.build(
        catalog_version=candidate.catalog.catalog_version,
        drill_catalog_content_sha256=candidate.catalog.content_sha256,
        definitions=(changed, *candidate.execution_policy_catalog.definitions[1:]),
    )
    values = candidate.model_dump(mode="python")
    values["execution_policy_catalog"] = policy_catalog
    mismatched = PlanPublicationCandidate.model_validate(values)

    result = _publisher(mismatched).publish(mismatched, now=NOW)

    assert not result.published
    assert result.failure_code in {
        PublicationFailureCode.CATALOG_MISMATCH,
        PublicationFailureCode.DRILL_INCOMPLETE,
    }


def test_publisher_rejects_untrusted_superseded_or_mismatched_admission() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)
    bad = PlanAdmissionDecision(
        status=AdmissionStatus.APPROVED,
        plan_id=candidate.plan.plan_id,
        plan_version=candidate.plan.version,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        authority_revision="assessment-authority:8",
        assessment_superseded=True,
        user_id=candidate.plan.user_id,
        session_id=candidate.authorization.session_id,
        target_device_id=candidate.authorization.target_device_id,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        catalog_version=candidate.catalog.catalog_version,
        catalog_content_sha256=candidate.catalog.content_sha256,
        capability_manifest_id=candidate.capability_manifest.manifest_id,
        capability_manifest_content_sha256=candidate.capability_manifest.content_sha256,
        validated_finding_ids=tuple(
            sorted({block.target_finding_id for block in candidate.plan.blocks})
        ),
    )
    publisher = publisher.with_admission_authority(FakeAdmissionAuthority(bad))

    result = publisher.publish(candidate, now=NOW)

    assert result.failure_code is PublicationFailureCode.ASSESSMENT_SUPERSEDED
    assert result.bundle is None
    assert repr(result) == "PublicationResult(rejected)"


def test_publisher_rejects_capability_authorization_expiry_and_stale_drill() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)
    reduced = candidate.model_copy(
        update={
            "capability_manifest": CapabilityManifest.build(
                manifest_id="table-profile-1",
                manifest_version=1,
                target_device_id="dilan-table-01",
                capabilities=("CUE_BALL_TARGET_ZONE",),
                equipment_codes=("STANDARD_TABLE",),
            )
        }
    )

    assert (
        publisher.publish(reduced, now=NOW).failure_code
        is PublicationFailureCode.CAPABILITY_MISMATCH
    )
    expired = candidate.model_copy(update={"requested_expires_at": NOW - timedelta(seconds=1)})
    assert (
        publisher.publish(expired, now=NOW).failure_code is PublicationFailureCode.INVALID_VALIDITY
    )


def test_authority_exception_and_bad_sequence_are_redacted_and_fail_closed() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate).with_admission_authority(
        FakeAdmissionAuthority(RuntimeError("private_key=do-not-leak"))
    )

    unavailable = publisher.publish(candidate, now=NOW)

    assert unavailable.failure_code is PublicationFailureCode.DEPENDENCY_UNAVAILABLE
    assert "private" not in repr(unavailable).lower()

    valid = _publisher(candidate).with_sequence_authority(FakeSequenceAuthority((0,)))
    rollback = valid.publish(candidate, now=NOW)
    assert rollback.failure_code is PublicationFailureCode.ROLLBACK_REJECTED


def test_admission_must_bind_authorization_catalog_and_manifest_authorities() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)
    valid = publisher._admission.decision
    mismatched = replace(valid, session_id=UUID("00000000-0000-4000-8000-000000000799"))

    result = publisher.with_admission_authority(FakeAdmissionAuthority(mismatched)).publish(
        candidate, now=NOW
    )

    assert result.failure_code is PublicationFailureCode.UNAUTHORIZED


def test_non_utc_publication_clock_is_rejected() -> None:
    candidate = _candidate()

    result = _publisher(candidate).publish(
        candidate, now=NOW.astimezone(timezone(timedelta(hours=1)))
    )

    assert result.failure_code is PublicationFailureCode.INVALID_VALIDITY


def test_invalid_publication_clock_never_escapes_type_errors() -> None:
    candidate = _candidate()

    for invalid in (None, "2026-08-05T09:00:00Z", NOW.replace(tzinfo=None)):
        result = _publisher(candidate).publish(candidate, now=invalid)  # type: ignore[arg-type]
        assert result.failure_code is PublicationFailureCode.INVALID_VALIDITY


def test_future_not_before_is_rejected_until_consumer_seam_supports_it() -> None:
    candidate = _candidate().model_copy(update={"requested_not_before": NOW + timedelta(minutes=1)})

    assert (
        _publisher(candidate).publish(candidate, now=NOW).failure_code
        is PublicationFailureCode.INVALID_VALIDITY
    )


def test_contraindication_codes_are_strict_canonical_codes() -> None:
    candidate = _candidate()

    for invalid in ("", "SHOULDER_INJURY ", "SHOULDER INJURY", "e\u0301"):
        with pytest.raises(ValueError):
            PlanPublicationCandidate.model_validate(
                candidate.model_dump(mode="python") | {"active_contraindication_codes": (invalid,)}
            )


def test_declared_public_key_must_match_signing_private_key() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)
    other = Ed25519PrivateKey.generate()
    mismatched = SigningMaterial(
        public_key=publisher._keys.material.public_key,
        private_key_handle=OpaqueEd25519PrivateKey(other),
    )
    result = PlanPublisher(
        admission_authority=publisher._admission,
        sequence_authority=FakeSequenceAuthority(),
        signing_key_authority=FakeSigningKeyAuthority(mismatched),
        signer=CryptographyEd25519Signer(),
    ).publish(candidate, now=NOW)

    assert result.failure_code is PublicationFailureCode.SIGNING_FAILED


def test_bundle_identity_includes_device_and_authorization_scope() -> None:
    candidate = _candidate()
    other = candidate.model_copy(
        update={
            "capability_manifest": CapabilityManifest.build(
                manifest_id="table-profile-2",
                manifest_version=1,
                target_device_id="dilan-table-02",
                capabilities=candidate.capability_manifest.capabilities,
                equipment_codes=candidate.capability_manifest.equipment_codes,
            ),
            "authorization": candidate.authorization.model_copy(
                update={
                    "authorization_id": UUID("00000000-0000-4000-8000-000000000703"),
                    "target_device_id": "dilan-table-02",
                }
            ),
        }
    )
    first = _publisher(candidate).publish(candidate, now=NOW)
    second = _publisher(other).publish(other, now=NOW)

    assert first.bundle is not None and second.bundle is not None
    assert first.bundle.payload.bundle_id != second.bundle.payload.bundle_id


def test_reserve_or_replay_is_atomic_under_concurrency_and_rejects_conflict() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _: publisher.publish(candidate, now=NOW), range(16)))

    assert all(result.published for result in results)
    assert len({result.envelope_bytes for result in results}) == 1
    assert {
        result.bundle.payload.publication_sequence  # type: ignore[union-attr]
        for result in results
    } == {1}

    conflicting = candidate.model_copy(
        update={"requested_expires_at": candidate.requested_expires_at - timedelta(minutes=1)}
    )
    conflict = publisher.publish(conflicting, now=NOW)
    assert conflict.failure_code is PublicationFailureCode.ROLLBACK_REJECTED


def test_admission_revalidates_target_finding_ownership() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)
    decision = replace(
        publisher._admission.decision,
        validated_finding_ids=(UUID("00000000-0000-4000-8000-000000000799"),),
    )

    result = publisher.with_admission_authority(FakeAdmissionAuthority(decision)).publish(
        candidate, now=NOW
    )

    assert result.failure_code is PublicationFailureCode.PLAN_BINDING_MISMATCH


def test_rotation_lineage_must_be_accepted_by_key_authority() -> None:
    candidate = _candidate()
    publisher = _publisher(candidate)

    class RejectLineage(FakeSigningKeyAuthority):
        def validate_rotation(self, key, *, at: datetime) -> bool:
            return False

    result = PlanPublisher(
        admission_authority=publisher._admission,
        sequence_authority=FakeSequenceAuthority(),
        signing_key_authority=RejectLineage(publisher._keys.material),
        signer=CryptographyEd25519Signer(),
    ).publish(candidate, now=NOW)

    assert result.failure_code is PublicationFailureCode.KEY_NOT_TRUSTED
