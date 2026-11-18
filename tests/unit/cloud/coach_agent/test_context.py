from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.coach_agent.context import (
    AssessmentHistoryPointer,
    CapabilityManifest,
    CoachContextError,
    CoachContextErrorCode,
    CoachContextSourceBundle,
    CoachSourceReadFence,
    CoachTaskType,
    PlanHistoryPointer,
    PlanOutcomeHistoryPointer,
    build_coach_context,
)
from billiards_ai.cloud.training.catalog import (
    BallPlacement,
    DrillCatalog,
    DrillDefinition,
    DrillEligibilityProfile,
    DrillReviewStatus,
    ProgressionRule,
    RegressionRule,
    ReviewLineage,
    SuccessMetricDefinition,
    TablePointMm,
    TableSetup,
)
from billiards_ai.shared.models.events import canonical_sha256
from billiards_ai.shared.models.growth import (
    AbilityMetric,
    AbilityTaxonomy,
    GrowthEvidenceReference,
    GrowthEvidenceSnapshot,
    MetricSupportStatus,
    Reliability,
)

USER_ID = UUID("00000000-0000-4000-8000-000000009201")
SNAPSHOT_ID = UUID("00000000-0000-4000-8000-000000009202")
BATCH_ID = UUID("00000000-0000-4000-8000-000000009203")
SESSION_ID = UUID("00000000-0000-4000-8000-000000009204")
SHOT_ID = UUID("00000000-0000-4000-8000-000000009205")
EVIDENCE_REF_ID = UUID("00000000-0000-4000-8000-000000009206")
REVIEWER_ID = UUID("00000000-0000-4000-8000-000000009207")
ASSESSMENT_ID = UUID("00000000-0000-4000-8000-000000009208")
PLAN_ID = UUID("00000000-0000-4000-8000-000000009209")
OUTCOME_ID = UUID("00000000-0000-4000-8000-000000009210")
AS_OF = datetime(2026, 8, 5, 8, 30, tzinfo=UTC)


def _snapshot(*, success_rate: float = 0.375) -> GrowthEvidenceSnapshot:
    metrics = (
        AbilityMetric(
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
            success_rate=success_rate,
            error_p50_mm=7.25,
            error_p90_mm=13.5,
            stability=None,
        ),
        AbilityMetric(
            metric_code="spin.control",
            taxonomy=AbilityTaxonomy(skill_code="SPIN"),
            support_status=MetricSupportStatus.ABSTAINED,
            sample_count=0,
            coverage=0.0,
            reliability=Reliability.LOW,
            success_rate=None,
            error_p50_mm=None,
            error_p90_mm=None,
            stability=None,
        ),
    )
    refs = (
        GrowthEvidenceReference(
            evidence_ref_id=EVIDENCE_REF_ID,
            batch_id=BATCH_ID,
            session_id=SESSION_ID,
            metric_code="aim.large_angle",
            shot_ids=(SHOT_ID,),
        ),
    )
    payload = {
        "schema_version": "1.0",
        "snapshot_id": SNAPSHOT_ID,
        "user_id": USER_ID,
        "version": 4,
        "created_at": datetime(2026, 8, 5, 8, tzinfo=UTC),
        "source_batch_ids": (BATCH_ID,),
        "metrics": metrics,
        "evidence_refs": refs,
        "plan_history": (),
    }
    return GrowthEvidenceSnapshot(**payload, content_sha256=canonical_sha256(payload))


def _definition(
    *,
    drill_id: str,
    status: DrillReviewStatus = DrillReviewStatus.APPROVED,
    capabilities: tuple[str, ...] = ("TABLE_MM_COORDINATES",),
    instructions: tuple[str, ...] = ("Align the balls.", "Pot into the target pocket."),
    reviewed_at: datetime = datetime(2026, 8, 5, 7, tzinfo=UTC),
) -> DrillDefinition:
    return DrillDefinition.build(
        drill_id=drill_id,
        revision=1,
        supersedes_revision=None,
        skill_codes=("POTTING",),
        minimum_level=1,
        maximum_level=3,
        prerequisite_skill_codes=(),
        setup=TableSetup(
            table_width_mm=1270,
            table_length_mm=2540,
            placements=(
                BallPlacement(
                    ball_code="CUE",
                    position=TablePointMm(x_mm=635, y_mm=635),
                ),
                BallPlacement(
                    ball_code="OBJECT_1",
                    position=TablePointMm(x_mm=635, y_mm=1270),
                ),
            ),
            target_pocket_code="TOP_CENTER",
            target_zone=None,
        ),
        instructions=instructions,
        success_metrics=(
            SuccessMetricDefinition(
                metric_code="POT_SUCCESS_COUNT",
                target_attempts=10,
                minimum_successes=7,
            ),
        ),
        progression_rule=ProgressionRule(reason_code="SUCCESS_STREAK", threshold=4, level_delta=1),
        regression_rule=RegressionRule(reason_code="FAILURE_STREAK", threshold=3, level_delta=-1),
        required_equipment_codes=("STANDARD_TABLE",),
        required_capabilities=capabilities,
        safety_codes=("KEEP_CUE_CLEAR_OF_BYSTANDERS",),
        contraindication_codes=(),
        review_status=status,
        review_lineage=ReviewLineage(
            reviewer_id=REVIEWER_ID,
            reviewed_at=reviewed_at,
            reason_code="SYNTHETIC_REVIEW",
        ),
    )


def _manifest() -> CapabilityManifest:
    return CapabilityManifest(
        schema_version="1.0",
        manifest_version="capability-7",
        user_id=USER_ID,
        user_level=2,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("TABLE_MM_COORDINATES",),
        available_equipment_codes=("STANDARD_TABLE",),
        active_contraindication_codes=(),
    )


def _bundle(
    *,
    task_type: CoachTaskType,
    snapshot: GrowthEvidenceSnapshot | None = None,
    catalog: DrillCatalog | None = None,
    profile: DrillEligibilityProfile | None = None,
    manifest: CapabilityManifest | None = None,
    prior_assessments: tuple[AssessmentHistoryPointer, ...] = (),
    prior_plans: tuple[PlanHistoryPointer, ...] = (),
    plan_outcomes: tuple[PlanOutcomeHistoryPointer, ...] = (),
) -> CoachContextSourceBundle:
    projected_plans = prior_plans if task_type is CoachTaskType.TRAINING_PLAN else ()
    projected_outcomes = plan_outcomes if task_type is CoachTaskType.TRAINING_PLAN else ()
    history_content_sha256 = canonical_sha256(
        {
            "prior_assessments": tuple(item.model_dump(mode="json") for item in prior_assessments),
            "prior_plans": tuple(item.model_dump(mode="json") for item in projected_plans),
            "plan_outcomes": tuple(item.model_dump(mode="json") for item in projected_outcomes),
        }
    )
    return CoachContextSourceBundle(
        schema_version="1.0",
        task_type=task_type,
        user_id=USER_ID,
        as_of=AS_OF,
        read_fence=CoachSourceReadFence(
            schema_version="1.0",
            fence_id="atomic-read-9201",
            snapshot_id=SNAPSHOT_ID,
            snapshot_version=4,
            history_revision=(
                f"history-{len(prior_assessments) + len(prior_plans) + len(plan_outcomes)}"
            ),
            history_content_sha256=history_content_sha256,
            catalog_version=(catalog.catalog_version if catalog is not None else None),
            capability_manifest_version="capability-7",
        ),
        growth_snapshot=snapshot or _snapshot(),
        prior_assessments=prior_assessments,
        prior_plans=prior_plans,
        plan_outcomes=plan_outcomes,
        drill_catalog=catalog,
        drill_profile=profile,
        capability_manifest=manifest or _manifest(),
    )


def test_p0_builder_preserves_metric_none_status_counts_reliability_and_refs_exactly() -> None:
    source = _bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT)

    context = build_coach_context(source)

    assert context.growth_snapshot.metrics == source.growth_snapshot.metrics
    assert context.growth_snapshot.evidence_refs == source.growth_snapshot.evidence_refs
    supported, abstained = context.growth_snapshot.metrics
    assert supported.success_rate == 0.375
    assert supported.error_p50_mm == 7.25
    assert supported.sample_count == 11
    assert supported.coverage == 0.55
    assert supported.reliability is Reliability.MEDIUM
    assert abstained.support_status is MetricSupportStatus.ABSTAINED
    assert abstained.success_rate is None
    assert abstained.error_p50_mm is None
    assert abstained.sample_count == 0
    assert abstained.coverage == 0.0
    dumped_keys = context.model_dump(mode="json").keys()
    assert "primary_weakness" not in dumped_keys
    assert "secondary_weakness" not in dumped_keys
    assert "bottleneck" not in dumped_keys


def test_p0_task_type_enforces_minimum_disclosure_and_explicit_empty_history() -> None:
    context = build_coach_context(_bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT))

    assert context.task_type is CoachTaskType.GROWTH_ASSESSMENT
    assert context.prior_assessments == ()
    assert context.prior_plans == ()
    assert context.plan_outcomes == ()
    assert context.available_drills == ()
    assert context.growth_snapshot.user_id == USER_ID


def test_p0_task_type_filters_nonempty_history_without_fabricating_records() -> None:
    assessment = AssessmentHistoryPointer(
        assessment_id=ASSESSMENT_ID,
        user_id=USER_ID,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        revision=1,
        created_at=datetime(2026, 8, 5, 7, 40, tzinfo=UTC),
        content_sha256="a" * 64,
    )
    plan = PlanHistoryPointer(
        plan_id=PLAN_ID,
        user_id=USER_ID,
        plan_version=1,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        assessment_id=ASSESSMENT_ID,
        assessment_revision=1,
        created_at=datetime(2026, 8, 5, 7, 45, tzinfo=UTC),
        content_sha256="b" * 64,
    )
    outcome = PlanOutcomeHistoryPointer(
        outcome_id=OUTCOME_ID,
        user_id=USER_ID,
        plan_id=PLAN_ID,
        plan_version=1,
        completed_at=datetime(2026, 8, 5, 7, 50, tzinfo=UTC),
        content_sha256="c" * 64,
    )
    history = {
        "prior_assessments": (assessment,),
        "prior_plans": (plan,),
        "plan_outcomes": (outcome,),
    }

    growth = build_coach_context(_bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT, **history))
    eligible = _definition(drill_id="a-eligible")
    catalog = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 5, 7, 30, tzinfo=UTC),
        drills=(eligible,),
    )
    profile = _manifest().eligibility_profile()
    training = build_coach_context(
        _bundle(
            task_type=CoachTaskType.TRAINING_PLAN,
            catalog=catalog,
            profile=profile,
            **history,
        )
    )

    assert growth.prior_assessments == (assessment,)
    assert growth.prior_plans == ()
    assert growth.plan_outcomes == ()
    assert training.prior_assessments == (assessment,)
    assert training.prior_plans == (plan,)
    assert training.plan_outcomes == (outcome,)


def test_p0_training_context_reuses_catalog_eligibility_and_rejects_forged_capability() -> None:
    eligible = _definition(drill_id="a-eligible")
    draft = _definition(drill_id="b-draft", status=DrillReviewStatus.DRAFT)
    missing_capability = _definition(
        drill_id="c-needs-spin-camera",
        capabilities=("SPIN_CAMERA", "TABLE_MM_COORDINATES"),
    )
    catalog = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 5, 7, 30, tzinfo=UTC),
        drills=(missing_capability, draft, eligible),
    )
    profile = DrillEligibilityProfile(
        user_level=2,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("TABLE_MM_COORDINATES",),
        available_equipment_codes=("STANDARD_TABLE",),
        active_contraindication_codes=(),
    )

    context = build_coach_context(
        _bundle(
            task_type=CoachTaskType.TRAINING_PLAN,
            catalog=catalog,
            profile=profile,
        )
    )

    assert tuple(item.drill_id for item in context.available_drills) == ("a-eligible",)
    assert all(
        item.review_status is DrillReviewStatus.APPROVED for item in context.available_drills
    )

    forged_profile = profile.model_construct(
        **(
            profile.model_dump(mode="python")
            | {"available_capabilities": ("SPIN_CAMERA", "TABLE_MM_COORDINATES")}
        )
    )
    with pytest.raises(CoachContextError) as exc:
        build_coach_context(
            _bundle(
                task_type=CoachTaskType.TRAINING_PLAN,
                catalog=catalog,
                profile=forged_profile,
            )
        )
    assert exc.value.code is CoachContextErrorCode.CAPABILITY_MISMATCH

    bad_catalog = DrillCatalog.model_construct(
        **(catalog.model_dump(mode="python") | {"content_sha256": "f" * 64})
    )
    valid_source = _bundle(
        task_type=CoachTaskType.TRAINING_PLAN,
        catalog=catalog,
        profile=profile,
    )
    bad_catalog_source = CoachContextSourceBundle.model_construct(
        **(valid_source.model_dump(mode="python") | {"drill_catalog": bad_catalog})
    )
    with pytest.raises(CoachContextError) as catalog_exc:
        build_coach_context(bad_catalog_source)
    assert catalog_exc.value.code is CoachContextErrorCode.INVALID_SOURCE_BUNDLE


def test_p1_builder_is_deterministic_hash_sensitive_and_context_remains_immutable() -> None:
    original_source = _bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT)
    original = build_coach_context(original_source)
    same = build_coach_context(original_source)
    changed = build_coach_context(
        _bundle(
            task_type=CoachTaskType.GROWTH_ASSESSMENT,
            snapshot=_snapshot(success_rate=0.5),
        )
    )

    assert original.model_dump_json() == same.model_dump_json()
    assert original.context_id == same.context_id
    assert original.context_sha256 == same.context_sha256
    assert changed.context_sha256 != original.context_sha256
    assert changed.context_id != original.context_id
    assert original.growth_snapshot.metrics[0].success_rate == 0.375
    assert original.model_dump_json() == same.model_dump_json()


def test_p1_builder_is_pure_and_exposes_future_agent_run_binding(monkeypatch) -> None:
    import builtins
    import socket

    def forbidden_side_effect(*args, **kwargs):
        raise AssertionError("pure context builder attempted an external side effect")

    monkeypatch.setattr(builtins, "open", forbidden_side_effect)
    monkeypatch.setattr(socket, "socket", forbidden_side_effect)

    context = build_coach_context(_bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT))

    assert context.context_id is not None
    assert len(context.context_sha256) == 64
    assert "context_id" in type(context).model_fields
    assert "context_sha256" in type(context).model_fields


def _history() -> tuple[AssessmentHistoryPointer, PlanHistoryPointer, PlanOutcomeHistoryPointer]:
    assessment = AssessmentHistoryPointer(
        assessment_id=ASSESSMENT_ID,
        user_id=USER_ID,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        revision=1,
        created_at=datetime(2026, 8, 5, 7, 40, tzinfo=UTC),
        content_sha256="a" * 64,
    )
    plan = PlanHistoryPointer(
        plan_id=PLAN_ID,
        user_id=USER_ID,
        plan_version=1,
        snapshot_id=SNAPSHOT_ID,
        snapshot_version=4,
        assessment_id=ASSESSMENT_ID,
        assessment_revision=1,
        created_at=datetime(2026, 8, 5, 7, 45, tzinfo=UTC),
        content_sha256="b" * 64,
    )
    outcome = PlanOutcomeHistoryPointer(
        outcome_id=OUTCOME_ID,
        user_id=USER_ID,
        plan_id=PLAN_ID,
        plan_version=1,
        completed_at=datetime(2026, 8, 5, 7, 50, tzinfo=UTC),
        content_sha256="c" * 64,
    )
    return assessment, plan, outcome


@pytest.mark.parametrize("kind", ["assessment", "plan", "outcome"])
def test_review_p0_history_entries_cannot_be_later_than_context_as_of(kind: str) -> None:
    assessment, plan, outcome = _history()
    if kind == "assessment":
        assessment = assessment.model_copy(
            update={"created_at": datetime(2026, 8, 5, 9, tzinfo=UTC)}
        )
    elif kind == "plan":
        plan = plan.model_copy(update={"created_at": datetime(2026, 8, 5, 9, tzinfo=UTC)})
    else:
        outcome = outcome.model_copy(update={"completed_at": datetime(2026, 8, 5, 9, tzinfo=UTC)})

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(
            _bundle(
                task_type=CoachTaskType.GROWTH_ASSESSMENT,
                prior_assessments=(assessment,),
                prior_plans=(plan,),
                plan_outcomes=(outcome,),
            )
        )
    assert exc.value.code is CoachContextErrorCode.SOURCE_FROM_FUTURE


def test_review_p0_history_requires_exact_parent_assessment_and_plan_references() -> None:
    assessment, plan, outcome = _history()
    dangling_plan = plan.model_copy(
        update={"assessment_id": UUID("00000000-0000-4000-8000-000000009299")}
    )
    with pytest.raises(CoachContextError) as plan_exc:
        build_coach_context(
            _bundle(
                task_type=CoachTaskType.GROWTH_ASSESSMENT,
                prior_assessments=(assessment,),
                prior_plans=(dangling_plan,),
            )
        )
    assert plan_exc.value.code is CoachContextErrorCode.READ_FENCE_MISMATCH

    dangling_outcome = outcome.model_copy(update={"plan_version": 2})
    with pytest.raises(CoachContextError) as outcome_exc:
        build_coach_context(
            _bundle(
                task_type=CoachTaskType.GROWTH_ASSESSMENT,
                prior_assessments=(assessment,),
                prior_plans=(plan,),
                plan_outcomes=(dangling_outcome,),
            )
        )
    assert outcome_exc.value.code is CoachContextErrorCode.READ_FENCE_MISMATCH


def test_review_p1_history_logical_keys_are_unique() -> None:
    assessment, plan, outcome = _history()
    conflicting_assessment = assessment.model_copy(update={"content_sha256": "d" * 64})
    conflicting_plan = plan.model_copy(update={"content_sha256": "e" * 64})
    conflicting_outcome = outcome.model_copy(update={"content_sha256": "f" * 64})

    for histories in (
        {"prior_assessments": (assessment, conflicting_assessment)},
        {
            "prior_assessments": (assessment,),
            "prior_plans": (plan, conflicting_plan),
        },
        {
            "prior_assessments": (assessment,),
            "prior_plans": (plan,),
            "plan_outcomes": (outcome, conflicting_outcome),
        },
    ):
        with pytest.raises(CoachContextError) as exc:
            build_coach_context(_bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT, **histories))
        assert exc.value.code is CoachContextErrorCode.NON_CANONICAL_ORDER


def test_review_p0_history_content_is_bound_to_read_fence() -> None:
    assessment, _, _ = _history()
    source = _bundle(
        task_type=CoachTaskType.GROWTH_ASSESSMENT,
        prior_assessments=(assessment,),
    )
    replaced = assessment.model_copy(update={"content_sha256": "d" * 64})
    tampered = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"prior_assessments": (replaced,)})
    )

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(tampered)
    assert exc.value.code is CoachContextErrorCode.READ_FENCE_MISMATCH


@pytest.mark.parametrize(
    "sensitive_value",
    [
        "+41 79 123 45 67",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI5MjAxIn0.signature9201",
        "postgresql://coach:secret@db.internal/growth",
        "password=hunter9201",
        "passwd: hunter9201",
        "api_key=key-9201-secret",
        "client_secret: secret-9201",
        "private_key=material-9201",
    ],
)
def test_review_p0_sensitive_values_are_rejected(sensitive_value: str) -> None:
    source = _bundle(task_type=CoachTaskType.GROWTH_ASSESSMENT)
    fence = source.read_fence.model_construct(
        **(source.read_fence.model_dump(mode="python") | {"fence_id": sensitive_value})
    )
    bypassed = CoachContextSourceBundle.model_construct(
        **(source.model_dump(mode="python") | {"read_fence": fence})
    )

    with pytest.raises(CoachContextError) as exc:
        build_coach_context(bypassed)
    assert exc.value.code is CoachContextErrorCode.SENSITIVE_CONTENT


def test_review_p1_trajectory_instruction_is_allowed_but_future_review_is_not() -> None:
    trajectory_drill = _definition(
        drill_id="trajectory-lesson",
        instructions=("Observe the cue-ball trajectory after contact.",),
    )
    catalog = DrillCatalog.build(
        catalog_version="2026.08-trajectory",
        created_at=datetime(2026, 8, 5, 7, 30, tzinfo=UTC),
        drills=(trajectory_drill,),
    )
    context = build_coach_context(
        _bundle(
            task_type=CoachTaskType.TRAINING_PLAN,
            catalog=catalog,
            profile=_manifest().eligibility_profile(),
        )
    )
    assert context.available_drills == (trajectory_drill,)

    future_drill = _definition(
        drill_id="future-review",
        reviewed_at=datetime(2026, 8, 5, 9, tzinfo=UTC),
    )
    future_catalog = DrillCatalog.build(
        catalog_version="2026.08-future",
        created_at=datetime(2026, 8, 5, 7, 30, tzinfo=UTC),
        drills=(future_drill,),
    )
    with pytest.raises(CoachContextError) as exc:
        build_coach_context(
            _bundle(
                task_type=CoachTaskType.TRAINING_PLAN,
                catalog=future_catalog,
                profile=_manifest().eligibility_profile(),
            )
        )
    assert exc.value.code is CoachContextErrorCode.SOURCE_FROM_FUTURE


def test_review_p0_public_context_validation_rejects_ineligible_drill_after_rehash() -> None:
    eligible = _definition(drill_id="eligible")
    catalog = DrillCatalog.build(
        catalog_version="2026.08-public",
        created_at=datetime(2026, 8, 5, 7, 30, tzinfo=UTC),
        drills=(eligible,),
    )
    context = build_coach_context(
        _bundle(
            task_type=CoachTaskType.TRAINING_PLAN,
            catalog=catalog,
            profile=_manifest().eligibility_profile(),
        )
    )
    draft = _definition(drill_id="draft", status=DrillReviewStatus.DRAFT)
    payload = {
        field_name: getattr(context, field_name)
        for field_name in type(context).model_fields
        if field_name not in {"context_id", "context_sha256"}
    } | {"available_drills": (draft,)}
    content_sha256 = canonical_sha256(payload)
    raw = bytearray(bytes.fromhex(content_sha256)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    invalid = type(context).model_construct(
        **payload,
        context_id=UUID(bytes=bytes(raw)),
        context_sha256=content_sha256,
    )

    with pytest.raises(ValidationError, match="eligible"):
        type(context).model_validate(invalid)
