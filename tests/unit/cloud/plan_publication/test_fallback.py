from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from billiards_ai.cloud.plan_publication.fallback import (
    FallbackFailureReason,
    FallbackRuleInput,
    PlanResolutionSource,
    deterministic_coach_fallback,
    publish_generated_or_fallback,
)
from billiards_ai.cloud.plan_publication.publisher import (
    AdmissionStatus,
    PlanAdmissionDecision,
    PlanPublicationCandidate,
    PlanPublisher,
    PublicationFailureCode,
    PublicationResult,
    SigningMaterial,
)
from billiards_ai.cloud.training.catalog import DrillCatalog
from billiards_ai.shared.models.growth import Reliability
from billiards_ai.shared.models.training_plan import TrainingPlan
from billiards_ai.shared.plan_publication.contracts import (
    AuthorizationBinding,
    CapabilityManifest,
    FinalTrainingPlan,
    GrowthRuntimeViewV2,
)
from billiards_ai.shared.plan_publication.crypto import (
    CryptographyEd25519Signer,
    OpaqueEd25519PrivateKey,
    public_verification_key,
)

ROOT = Path(__file__).parents[3] / "fixtures" / "core_intelligence" / "v1"
NOW = datetime(2026, 8, 5, 9, tzinfo=UTC)
SESSION_ID = UUID("00000000-0000-4000-8000-000000000701")


def _candidate() -> PlanPublicationCandidate:
    legacy = TrainingPlan.model_validate_json((ROOT / "training-plan.json").read_text())
    catalog = DrillCatalog.model_validate_json((ROOT / "drill-catalog.json").read_text())
    plan = FinalTrainingPlan.from_v1(
        legacy,
        snapshot_id=UUID("00000000-0000-4000-8000-000000000008"),
        snapshot_version=1,
        admission_status=AdmissionStatus.APPROVED,
    )
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
            issued_at=NOW,
            expires_at=plan.valid_until,
        ),
        requested_not_before=NOW,
        requested_expires_at=plan.valid_until,
    )


def test_fallback_is_versioned_deterministic_and_uses_eligible_catalog_heads() -> None:
    candidate = _candidate()
    rule_input = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )

    first = deterministic_coach_fallback(rule_input)
    second = deterministic_coach_fallback(rule_input)

    assert first.plan == second.plan
    assert first.plan is not None
    assert first.plan.schema_version == "2.0"
    assert first.reason is None
    assert first.plan.blocks[0].drill.drill_id == "capability-target-zone-01"


def test_fallback_returns_no_reliable_plan_when_rule_inputs_are_insufficient() -> None:
    candidate = _candidate()

    result = deterministic_coach_fallback(
        FallbackRuleInput(
            user_id=candidate.plan.user_id,
            session_id=SESSION_ID,
            authorization_id=candidate.authorization.authorization_id,
            authorization_version=candidate.authorization.authorization_version,
            target_device_id=candidate.authorization.target_device_id,
            assessment_id=candidate.plan.assessment_id,
            assessment_revision=candidate.plan.assessment_revision,
            target_finding_id=candidate.plan.blocks[0].target_finding_id,
            snapshot_id=candidate.plan.snapshot_id,
            snapshot_version=candidate.plan.snapshot_version,
            plan_version=candidate.plan.version,
            user_level=99,
            acquired_skill_codes=(),
            active_contraindication_codes=(),
            catalog=candidate.catalog,
            capability_manifest=candidate.capability_manifest,
            created_at=NOW,
            valid_until=candidate.plan.valid_until,
        )
    )

    assert result.plan is None
    assert result.reason is FallbackFailureReason.NO_RELIABLE_PLAN


class DynamicAdmissionAuthority:
    def admit(self, candidate: PlanPublicationCandidate) -> PlanAdmissionDecision:
        plan = candidate.plan
        return PlanAdmissionDecision(
            status=AdmissionStatus(plan.admission_status),
            plan_id=plan.plan_id,
            plan_version=plan.version,
            assessment_id=plan.assessment_id,
            assessment_revision=plan.assessment_revision,
            snapshot_id=plan.snapshot_id,
            snapshot_version=plan.snapshot_version,
            authority_revision="dynamic-authority:1",
            assessment_superseded=False,
            user_id=plan.user_id,
            session_id=candidate.authorization.session_id,
            target_device_id=candidate.authorization.target_device_id,
            authorization_id=candidate.authorization.authorization_id,
            authorization_version=candidate.authorization.authorization_version,
            catalog_version=candidate.catalog.catalog_version,
            catalog_content_sha256=candidate.catalog.content_sha256,
            capability_manifest_id=candidate.capability_manifest.manifest_id,
            capability_manifest_content_sha256=candidate.capability_manifest.content_sha256,
            validated_finding_ids=tuple(sorted({block.target_finding_id for block in plan.blocks})),
        )


class OneSequence:
    def reserve_or_replay(self, *, build: Callable[[int], bytes], **kwargs) -> bytes:
        return build(1)


@dataclass(frozen=True)
class FallbackKeyAuthority:
    material: SigningMaterial

    def resolve(self, *, at: datetime) -> SigningMaterial:
        return self.material

    def validate_rotation(self, key, *, at: datetime) -> bool:
        return key == self.material.public_key


def test_provider_and_repair_failure_publish_fallback_through_identical_gate() -> None:
    candidate = _candidate()
    source = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )
    private = Ed25519PrivateKey.generate()
    key = public_verification_key(
        private.public_key(),
        key_id="fallback-key-1",
        key_version=1,
        issued_at=NOW - timedelta(days=1),
        not_before=NOW - timedelta(days=1),
        expires_at=NOW + timedelta(days=1),
    )
    publisher = PlanPublisher(
        admission_authority=DynamicAdmissionAuthority(),
        sequence_authority=OneSequence(),
        signing_key_authority=FallbackKeyAuthority(
            SigningMaterial(
                public_key=key,
                private_key_handle=OpaqueEd25519PrivateKey(private),
            )
        ),
        signer=CryptographyEd25519Signer(),
    )

    resolved = publish_generated_or_fallback(
        publisher=publisher,
        provider_candidate=None,
        repair_candidate=None,
        fallback_input=source,
        fallback_candidate_template=candidate,
        now=NOW,
    )

    assert resolved.source is PlanResolutionSource.DETERMINISTIC_FALLBACK
    assert resolved.publication is not None and resolved.publication.published
    assert resolved.publication.bundle is not None
    assert resolved.publication.bundle.payload.plan.admission_status == "VALIDATED"
    assert (
        resolved.publication.bundle.payload.plan.blocks[0].target_finding_id
        == source.target_finding_id
    )


def test_fallback_schema_and_target_finding_claim_are_strict() -> None:
    candidate = _candidate()
    common = dict(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )

    with pytest.raises(ValidationError):
        FallbackRuleInput(**(common | {"schema_version": "2.0"}))
    with pytest.raises(ValidationError):
        FallbackRuleInput(**(common | {"target_finding_id": UUID(int=0)}))


def test_invalid_fallback_window_returns_no_reliable_plan() -> None:
    candidate = _candidate()
    source = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=NOW,
    )

    result = deterministic_coach_fallback(source)

    assert result.plan is None
    assert result.reason is FallbackFailureReason.NO_RELIABLE_PLAN


@dataclass
class RecordingPublisher:
    result: PublicationResult
    calls: list[PlanPublicationCandidate]

    def publish(self, candidate: PlanPublicationCandidate, *, now: datetime) -> PublicationResult:
        self.calls.append(candidate)
        return self.result


@pytest.mark.parametrize("failure_code", list(PublicationFailureCode))
def test_candidate_gate_rejection_is_terminal_for_every_security_code(failure_code) -> None:
    candidate = _candidate()
    publisher = RecordingPublisher(result=PublicationResult(failure_code=failure_code), calls=[])
    source = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )

    resolved = publish_generated_or_fallback(
        publisher=publisher,  # type: ignore[arg-type]
        provider_candidate=candidate,
        repair_candidate=candidate,
        fallback_input=source,
        fallback_candidate_template=candidate,
        now=NOW,
    )

    assert resolved.source is PlanResolutionSource.PROVIDER
    assert resolved.publication is publisher.result
    assert resolved.publication.failure_code is failure_code
    assert publisher.calls == [candidate]


def test_repair_gate_rejection_is_terminal_and_source_is_explicit() -> None:
    candidate = _candidate()
    publisher = RecordingPublisher(
        result=PublicationResult(failure_code=PublicationFailureCode.SAFETY_REJECTED),
        calls=[],
    )
    source = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )

    resolved = publish_generated_or_fallback(
        publisher=publisher,  # type: ignore[arg-type]
        provider_candidate=None,
        repair_candidate=candidate,
        fallback_input=source,
        fallback_candidate_template=candidate,
        now=NOW,
    )

    assert resolved.source is PlanResolutionSource.REPAIR
    assert resolved.publication is publisher.result
    assert publisher.calls == [candidate]


def test_fallback_rejects_any_template_authority_mismatch() -> None:
    candidate = _candidate()
    source = FallbackRuleInput(
        user_id=candidate.plan.user_id,
        session_id=SESSION_ID,
        authorization_id=candidate.authorization.authorization_id,
        authorization_version=candidate.authorization.authorization_version,
        target_device_id=candidate.authorization.target_device_id,
        assessment_id=candidate.plan.assessment_id,
        assessment_revision=candidate.plan.assessment_revision,
        target_finding_id=candidate.plan.blocks[0].target_finding_id,
        snapshot_id=candidate.plan.snapshot_id,
        snapshot_version=candidate.plan.snapshot_version,
        plan_version=candidate.plan.version,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        active_contraindication_codes=(),
        catalog=candidate.catalog,
        capability_manifest=candidate.capability_manifest,
        created_at=NOW,
        valid_until=candidate.plan.valid_until,
    )
    other_id = UUID("00000000-0000-4000-8000-000000000799")
    mismatches = (
        candidate.model_copy(
            update={
                "authorization": candidate.authorization.model_copy(update={"user_id": other_id})
            }
        ),
        candidate.model_copy(
            update={
                "authorization": candidate.authorization.model_copy(update={"session_id": other_id})
            }
        ),
        candidate.model_copy(
            update={
                "authorization": candidate.authorization.model_copy(
                    update={"authorization_id": other_id}
                )
            }
        ),
        candidate.model_copy(
            update={"plan": candidate.plan.model_copy(update={"assessment_id": other_id})}
        ),
        candidate.model_copy(
            update={"plan": candidate.plan.model_copy(update={"snapshot_id": other_id})}
        ),
        candidate.model_copy(
            update={
                "catalog": DrillCatalog.build(
                    catalog_version="other",
                    created_at=candidate.catalog.created_at,
                    revision_history=(
                        candidate.catalog.revision_history or candidate.catalog.drills
                    ),
                )
            }
        ),
        candidate.model_copy(
            update={
                "capability_manifest": candidate.capability_manifest.model_copy(
                    update={"manifest_id": "other"}
                )
            }
        ),
        candidate.model_copy(
            update={
                "authorization": candidate.authorization.model_copy(
                    update={"target_device_id": "other-device"}
                )
            }
        ),
        candidate.model_copy(update={"requested_not_before": NOW + timedelta(seconds=1)}),
    )
    publisher = RecordingPublisher(
        result=PublicationResult(failure_code=PublicationFailureCode.ADMISSION_REJECTED),
        calls=[],
    )

    for template in mismatches:
        resolved = publish_generated_or_fallback(
            publisher=publisher,  # type: ignore[arg-type]
            provider_candidate=None,
            repair_candidate=None,
            fallback_input=source,
            fallback_candidate_template=template,
            now=NOW,
        )
        assert resolved.reason is FallbackFailureReason.NO_RELIABLE_PLAN
    assert publisher.calls == []
