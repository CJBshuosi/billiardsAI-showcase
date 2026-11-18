from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.context import (
    AssessmentHistoryPointer,
    CapabilityManifest,
    CoachAgentContext,
    CoachContextError,
    CoachContextErrorCode,
    CoachContextSourceBundle,
    CoachSourceReadFence,
    CoachTaskType,
    PlanHistoryPointer,
    build_coach_context,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth import (
    AbilityMetric,
    AbilityTaxonomy,
    GrowthEvidenceReference,
    GrowthEvidenceSnapshot,
    MetricSupportStatus,
    PlanHistoryEntry,
    Reliability,
)

USER_ID = UUID("00000000-0000-4000-8000-000000009201")
OTHER_USER_ID = UUID("00000000-0000-4000-8000-000000009299")
SNAPSHOT_ID = UUID("00000000-0000-4000-8000-000000009202")
BATCH_ID = UUID("00000000-0000-4000-8000-000000009203")
SESSION_ID = UUID("00000000-0000-4000-8000-000000009204")
SHOT_ID = UUID("00000000-0000-4000-8000-000000009205")
OTHER_SHOT_ID = UUID("00000000-0000-4000-8000-000000009211")
EVIDENCE_REF_ID = UUID("00000000-0000-4000-8000-000000009206")
ASSESSMENT_ID = UUID("00000000-0000-4000-8000-000000009208")
PLAN_ID = UUID("00000000-0000-4000-8000-000000009209")
OTHER_PLAN_ID = UUID("00000000-0000-4000-8000-000000009212")
AS_OF = datetime(2026, 8, 5, 8, 30, tzinfo=UTC)


def _snapshot(
    *,
    user_id: UUID = USER_ID,
    content_sha256: str | None = None,
    metrics: tuple[AbilityMetric, ...] | None = None,
    evidence_refs: tuple[GrowthEvidenceReference, ...] | None = None,
    plan_history: tuple[PlanHistoryEntry, ...] = (),
) -> GrowthEvidenceSnapshot:
    supported = AbilityMetric(
        metric_code="aim.large_angle",
        taxonomy=AbilityTaxonomy(
            skill_code="POTTING",
            angle_band="LARGE",
            distance_band="MEDIUM",
            rail_context=None,
        ),
        support_status=MetricSupportStatus.SUPPORTED,
        sample_count=11,
        coverage=0.55,
        reliability=Reliability.MEDIUM,
        success_rate=0.375,
        error_p50_mm=7.25,
        error_p90_mm=13.5,
        stability=None,
    )
    unsupported = AbilityMetric(
        metric_code="spin.control",
        taxonomy=AbilityTaxonomy(skill_code="SPIN"),
        support_status=MetricSupportStatus.UNSUPPORTED,
        sample_count=0,
        coverage=0.0,
        reliability=Reliability.LOW,
        success_rate=None,
        error_p50_mm=None,
        error_p90_mm=None,
        stability=None,
    )
    ref = GrowthEvidenceReference(
        evidence_ref_id=EVIDENCE_REF_ID,
        batch_id=BATCH_ID,
        session_id=SESSION_ID,
        metric_code="aim.large_angle",
        shot_ids=(SHOT_ID,),
    )
    payload = {
        "schema_version": "1.0",
        "snapshot_id": SNAPSHOT_ID,
        "user_id": user_id,
        "version": 4,
        "created_at": datetime(2026, 8, 5, 8, tzinfo=UTC),
        "source_batch_ids": (BATCH_ID,),
        "metrics": metrics if metrics is not None else (supported, unsupported),
        "evidence_refs": evidence_refs if evidence_refs is not None else (ref,),
        "plan_history": plan_history,
    }
    return GrowthEvidenceSnapshot(
        **payload,
        content_sha256=content_sha256 or canonical_sha256(payload),
    )


def _manifest(*, user_id: UUID = USER_ID) -> CapabilityManifest:
    return CapabilityManifest(
        schema_version="1.0",
        manifest_version="capability-7",
        user_id=user_id,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("TABLE_MM_COORDINATES",),
        available_equipment_codes=("STANDARD_TABLE",),
        active_contraindication_codes=(),
    )


def _bundle(
    *,
    snapshot: GrowthEvidenceSnapshot | None = None,
    manifest: CapabilityManifest | None = None,
    as_of: datetime = AS_OF,
    prior_assessments: tuple[AssessmentHistoryPointer, ...] = (),
    prior_plans: tuple[PlanHistoryPointer, ...] = (),
) -> CoachContextSourceBundle:
    history_content_sha256 = canonical_sha256(
        {
            "prior_assessments": tuple(item.model_dump(mode="json") for item in prior_assessments),
            "prior_plans": (),
            "plan_outcomes": (),
        }
    )
    return CoachContextSourceBundle(
        schema_version="1.0",
        task_type=CoachTaskType.GROWTH_ASSESSMENT,
        user_id=USER_ID,
        as_of=as_of,
        read_fence=CoachSourceReadFence(
            schema_version="1.0",
            fence_id="atomic-read-9201",
            snapshot_id=SNAPSHOT_ID,
            snapshot_version=4,
            history_revision="history-0",
            history_content_sha256=history_content_sha256,
            catalog_version=None,
            capability_manifest_version="capability-7",
        ),
        growth_snapshot=snapshot or _snapshot(),
        prior_assessments=prior_assessments,
        prior_plans=prior_plans,
        plan_outcomes=(),
        drill_catalog=None,
        drill_profile=None,
        capability_manifest=manifest or _manifest(),
    )


def _uuid4_from_hash(content_sha256: str) -> UUID:
    raw = bytearray(bytes.fromhex(content_sha256)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def _rehash_context(context: CoachAgentContext, **updates: object) -> CoachAgentContext:
    payload = {
        field_name: getattr(context, field_name)
        for field_name in type(context).model_fields
        if field_name not in {"context_id", "context_sha256"}
    }
    payload.update(updates)
    content_sha256 = canonical_sha256(payload)
    return CoachAgentContext.model_construct(
        **payload,
        context_id=_uuid4_from_hash(content_sha256),
        context_sha256=content_sha256,
    )


def test_p0_context_contract_is_strict_frozen_round_trippable_and_deterministic() -> None:
    source = _bundle()

    first = build_coach_context(source)
    second = build_coach_context(source)

    assert CoachAgentContext.model_validate_json(first.model_dump_json()) == first
    assert first.model_dump_json() == second.model_dump_json()
    assert first.context_id == second.context_id
    assert first.context_sha256 == second.context_sha256
    assert first.schema_version == "1.0"
    assert first.snapshot_id == SNAPSHOT_ID
    assert first.snapshot_version == 4
    assert first.context_sha256 == canonical_sha256(
        first.model_dump(mode="json", exclude={"context_id", "context_sha256"})
    )
    with pytest.raises(ValidationError, match="frozen"):
        first.snapshot_version = 5
    with pytest.raises(ValidationError, match="extra"):
        CoachAgentContext.model_validate(first.model_dump() | {"unexpected": "forbidden"})
    with pytest.raises(ValidationError):
        CoachContextSourceBundle.model_validate(source.model_dump() | {"snapshot_version": True})
    with pytest.raises(ValidationError):
        CoachSourceReadFence.model_validate(
            source.read_fence.model_dump() | {"snapshot_version": True}
        )
    with pytest.raises(ValidationError, match="UTC"):
        _bundle(as_of=datetime(2026, 8, 5, 8, 30))


def test_p0_legacy_v1_context_json_keeps_its_original_fence_shape_and_identity() -> None:
    current = build_coach_context(_bundle())
    payload = current.model_dump(mode="json", exclude={"context_id", "context_sha256"})
    legacy_fence = dict(payload["read_fence"])
    legacy_fence.pop("growth_snapshot_read_fence", None)
    legacy_fence.pop("authority_revision", None)
    legacy_payload = payload | {"read_fence": legacy_fence}
    content_sha256 = canonical_sha256(legacy_payload)
    serialized = json.dumps(
        legacy_payload
        | {
            "context_id": str(_uuid4_from_hash(content_sha256)),
            "context_sha256": content_sha256,
        }
    )

    restored = CoachAgentContext.model_validate_json(serialized, strict=True)

    assert restored.context_sha256 == content_sha256
    assert restored.model_dump(mode="json")["read_fence"] == legacy_fence


def test_p0_builder_deeply_revalidates_construct_copy_hash_fence_and_order_bypasses() -> None:
    source = _bundle()
    bad_hash_snapshot = GrowthEvidenceSnapshot.model_construct(
        **(source.growth_snapshot.model_dump(mode="python") | {"content_sha256": "f" * 64})
    )
    bad_hash_source = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"growth_snapshot": bad_hash_snapshot})
    )
    with pytest.raises(CoachContextError) as exc:
        build_coach_context(bad_hash_source)
    assert exc.value.code is CoachContextErrorCode.INVALID_CONTENT_HASH
    assert "f" * 64 not in str(exc.value)

    reordered_payload = source.growth_snapshot.model_dump(
        mode="python", exclude={"content_sha256"}
    ) | {"metrics": tuple(reversed(source.growth_snapshot.metrics))}
    reordered_snapshot = GrowthEvidenceSnapshot(
        **reordered_payload,
        content_sha256=canonical_sha256(reordered_payload),
    )
    reordered_source = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"growth_snapshot": reordered_snapshot})
    )
    with pytest.raises(CoachContextError) as order_exc:
        build_coach_context(reordered_source)
    assert order_exc.value.code is CoachContextErrorCode.NON_CANONICAL_ORDER

    mismatched_fence = source.read_fence.model_construct(
        **(source.read_fence.model_dump(mode="python") | {"snapshot_version": 5})
    )
    fence_source = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"read_fence": mismatched_fence})
    )
    with pytest.raises(CoachContextError) as fence_exc:
        build_coach_context(fence_source)
    assert fence_exc.value.code is CoachContextErrorCode.READ_FENCE_MISMATCH

    invalid_source = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"user_id": str(USER_ID)})
    )
    with pytest.raises(CoachContextError) as invalid_exc:
        build_coach_context(invalid_source)
    assert invalid_exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE


def test_p0_builder_rejects_cross_user_sensitive_and_dangling_evidence_without_payload_leak() -> (
    None
):
    cross_user = CoachContextSourceBundle.model_construct(
        **(
            _bundle().model_dump(mode="python")
            | {"growth_snapshot": _snapshot(user_id=OTHER_USER_ID)}
        )
    )
    with pytest.raises(CoachContextError) as cross_user_exc:
        build_coach_context(cross_user)
    assert cross_user_exc.value.code is CoachContextErrorCode.CROSS_USER_LINEAGE
    assert str(OTHER_USER_ID) not in str(cross_user_exc.value)

    sensitive = CoachContextSourceBundle.model_construct(**_bundle().model_dump(mode="python"))
    object.__setattr__(sensitive, "openid", "wx-user-secret")
    with pytest.raises(CoachContextError) as sensitive_exc:
        build_coach_context(sensitive)
    assert sensitive_exc.value.code in {
        CoachContextErrorCode.INVALID_SOURCE_BUNDLE,
        CoachContextErrorCode.SENSITIVE_CONTENT,
    }
    assert "wx-user-secret" not in str(sensitive_exc.value)

    dangling_ref = GrowthEvidenceReference(
        evidence_ref_id=EVIDENCE_REF_ID,
        batch_id=BATCH_ID,
        session_id=SESSION_ID,
        metric_code="missing.metric",
        shot_ids=(SHOT_ID,),
    )
    snapshot = _snapshot(evidence_refs=(dangling_ref,))
    bypassed = CoachContextSourceBundle.model_construct(
        **(_bundle().model_dump(mode="python") | {"growth_snapshot": snapshot})
    )
    with pytest.raises(CoachContextError) as dangling_exc:
        build_coach_context(bypassed)
    assert dangling_exc.value.code is CoachContextErrorCode.DANGLING_EVIDENCE_REFERENCE


def test_review_p0_public_context_validation_rechecks_semantics_after_valid_rehash() -> None:
    context = build_coach_context(_bundle())
    mismatched_fence = context.read_fence.model_construct(
        **(
            context.read_fence.model_dump(mode="python")
            | {"snapshot_version": context.snapshot_version + 1}
        )
    )
    invalid = _rehash_context(context, read_fence=mismatched_fence)

    with pytest.raises(ValidationError, match="read fence"):
        CoachAgentContext.model_validate(invalid)
    with pytest.raises(ValidationError, match="read fence"):
        CoachAgentContext.model_validate_json(invalid.model_dump_json())


def test_review_p0_public_context_validation_rechecks_history_user_and_task_source() -> None:
    context = build_coach_context(_bundle())
    cross_user_assessment = AssessmentHistoryPointer(
        assessment_id=ASSESSMENT_ID,
        user_id=OTHER_USER_ID,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        revision=1,
        created_at=datetime(2026, 8, 5, 8, 5, tzinfo=UTC),
        content_sha256="a" * 64,
    )
    cross_user = _rehash_context(context, prior_assessments=(cross_user_assessment,))
    with pytest.raises(ValidationError, match="cross-user"):
        CoachAgentContext.model_validate(cross_user)

    plan = PlanHistoryPointer(
        plan_id=PLAN_ID,
        user_id=USER_ID,
        plan_version=1,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        assessment_id=ASSESSMENT_ID,
        assessment_revision=1,
        created_at=datetime(2026, 8, 5, 8, 10, tzinfo=UTC),
        content_sha256="b" * 64,
    )
    assessment = AssessmentHistoryPointer(
        assessment_id=ASSESSMENT_ID,
        user_id=USER_ID,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        revision=1,
        created_at=datetime(2026, 8, 5, 8, 5, tzinfo=UTC),
        content_sha256="a" * 64,
    )
    history_content_sha256 = canonical_sha256(
        {
            "prior_assessments": (assessment.model_dump(mode="json"),),
            "prior_plans": (),
            "plan_outcomes": (),
        }
    )
    fence = context.read_fence.model_copy(update={"history_content_sha256": history_content_sha256})
    wrong_task_source = _rehash_context(
        context,
        prior_assessments=(assessment,),
        prior_plans=(plan,),
        read_fence=fence,
    )
    with pytest.raises(ValidationError, match="growth assessment"):
        CoachAgentContext.model_validate(wrong_task_source)


def test_review_p1_public_context_validation_rechecks_snapshot_canonical_order() -> None:
    context = build_coach_context(_bundle())
    snapshot_payload = context.growth_snapshot.model_dump(
        mode="python", exclude={"content_sha256"}
    ) | {"metrics": tuple(reversed(context.growth_snapshot.metrics))}
    snapshot = GrowthEvidenceSnapshot(
        **snapshot_payload,
        content_sha256=canonical_sha256(snapshot_payload),
    )
    invalid = _rehash_context(context, growth_snapshot=snapshot)

    with pytest.raises(ValidationError, match="canonical"):
        CoachAgentContext.model_validate(invalid)


def test_review_p1_snapshot_plan_history_and_shot_ids_are_unique_and_canonical() -> None:
    duplicate_shot_ref = GrowthEvidenceReference(
        evidence_ref_id=EVIDENCE_REF_ID,
        batch_id=BATCH_ID,
        session_id=SESSION_ID,
        metric_code="aim.large_angle",
        shot_ids=(SHOT_ID, SHOT_ID),
    )
    with pytest.raises(CoachContextError) as shot_exc:
        build_coach_context(_bundle(snapshot=_snapshot(evidence_refs=(duplicate_shot_ref,))))
    assert shot_exc.value.code is CoachContextErrorCode.NON_CANONICAL_ORDER

    duplicate_plan_history = (
        PlanHistoryEntry(
            plan_id=PLAN_ID,
            plan_version=1,
            status="ACTIVE",
            started_at=datetime(2026, 8, 5, 7, tzinfo=UTC),
        ),
        PlanHistoryEntry(
            plan_id=PLAN_ID,
            plan_version=1,
            status="COMPLETED",
            started_at=datetime(2026, 8, 5, 7, tzinfo=UTC),
            completed_at=datetime(2026, 8, 5, 8, tzinfo=UTC),
        ),
    )
    with pytest.raises(CoachContextError) as plan_exc:
        build_coach_context(_bundle(snapshot=_snapshot(plan_history=duplicate_plan_history)))
    assert plan_exc.value.code is CoachContextErrorCode.NON_CANONICAL_ORDER

    noncanonical_plan_history = tuple(reversed(duplicate_plan_history))
    noncanonical_plan_history = (
        noncanonical_plan_history[0].model_copy(update={"plan_id": OTHER_PLAN_ID}),
        noncanonical_plan_history[1],
    )
    with pytest.raises(CoachContextError) as order_exc:
        build_coach_context(_bundle(snapshot=_snapshot(plan_history=noncanonical_plan_history)))
    assert order_exc.value.code is CoachContextErrorCode.NON_CANONICAL_ORDER


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_review_p0_non_finite_numbers_are_rejected_before_hash_or_json_round_trip(
    non_finite: float,
) -> None:
    source = _bundle()
    supported = source.growth_snapshot.metrics[0]
    non_finite_metric = AbilityMetric.model_construct(
        **(supported.model_dump(mode="python") | {"error_p90_mm": non_finite})
    )
    snapshot_payload = source.growth_snapshot.model_dump(
        mode="python", exclude={"content_sha256"}
    ) | {"metrics": (non_finite_metric, source.growth_snapshot.metrics[1])}
    snapshot = GrowthEvidenceSnapshot.model_construct(
        **snapshot_payload,
        content_sha256=canonical_sha256(snapshot_payload),
    )
    bypassed = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"growth_snapshot": snapshot})
    )

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(bypassed)
    assert exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE


@pytest.mark.parametrize("sensitive_key", ["trajectory", "raw_observation"])
def test_review_p1_raw_source_field_keys_remain_forbidden(sensitive_key: str) -> None:
    source = CoachContextSourceBundle.model_construct(**_bundle().model_dump(mode="python"))
    object.__setattr__(source, sensitive_key, "not-a-locator")

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(source)

    assert exc.value.code is CoachContextErrorCode.SENSITIVE_CONTENT


def test_review_p1_validation_error_mapping_has_no_sensitive_cause_chain() -> None:
    secret = "rejected-user-value-9201"
    invalid = CoachContextSourceBundle.model_construct(
        **(_bundle().model_dump(mode="python") | {"user_id": secret})
    )

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(invalid)

    assert exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE
    assert exc.value.__cause__ is None
    rendered = "".join(
        traceback.format_exception(type(exc.value), exc.value, exc.value.__traceback__)
    )
    assert secret not in rendered


def test_review_p1_cyclic_source_maps_to_stable_payload_free_error() -> None:
    source = CoachContextSourceBundle.model_construct(**_bundle().model_dump(mode="python"))
    cycle: list[object] = []
    cycle.append(cycle)
    object.__setattr__(source, "cycle", cycle)

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(source)

    assert exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE
    assert exc.value.__cause__ is None
    assert "cycle" not in str(exc.value)
