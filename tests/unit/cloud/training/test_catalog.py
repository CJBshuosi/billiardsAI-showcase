from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.training.catalog import (
    BallPlacement,
    DrillCatalog,
    DrillDefinition,
    DrillEligibilityProfile,
    DrillReviewStatus,
    DrillViolationCode,
    ProgressionRule,
    RegressionRule,
    ReviewLineage,
    SuccessMetricDefinition,
    TablePointMm,
    TableSetup,
    TableTargetZoneMm,
    list_eligible_drills,
    review_drill,
    validate_drill_reference,
)
from billiards_ai.shared.models.training_plan import DrillReference


def _definition(
    *,
    drill_id: str = "straight-potting-01",
    status: DrillReviewStatus = DrillReviewStatus.APPROVED,
    minimum_level: int = 1,
    maximum_level: int = 3,
    prerequisites: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = ("TABLE_MM_COORDINATES",),
    contraindications: tuple[str, ...] = ("SHOULDER_INJURY",),
) -> DrillDefinition:
    return DrillDefinition.build(
        drill_id=drill_id,
        revision=1,
        supersedes_revision=None,
        skill_codes=("POTTING",),
        minimum_level=minimum_level,
        maximum_level=maximum_level,
        prerequisite_skill_codes=prerequisites,
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
        instructions=("Align the balls.", "Pot into the target pocket."),
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
        contraindication_codes=contraindications,
        review_status=status,
        review_lineage=ReviewLineage(
            reviewer_id=UUID("00000000-0000-4000-8000-000000000901"),
            reviewed_at=datetime(2026, 8, 4, 8, tzinfo=UTC),
            reason_code="INITIAL_SYNTHETIC_REVIEW",
        ),
    )


def _catalog(*drills: DrillDefinition) -> DrillCatalog:
    return DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=drills,
    )


def _reference(
    *,
    catalog_version: str = "2026.08",
    drill_id: str = "straight-potting-01",
    drill_revision: int = 1,
    required_capabilities: tuple[str, ...] = ("TABLE_MM_COORDINATES",),
) -> DrillReference:
    return DrillReference(
        catalog_version=catalog_version,
        drill_id=drill_id,
        drill_revision=drill_revision,
        required_capabilities=required_capabilities,
    )


def _profile(
    *,
    level: int = 2,
    skills: tuple[str, ...] = ("POTTING",),
    capabilities: tuple[str, ...] = ("TABLE_MM_COORDINATES",),
    equipment: tuple[str, ...] = ("STANDARD_TABLE",),
    contraindications: tuple[str, ...] = (),
) -> DrillEligibilityProfile:
    return DrillEligibilityProfile(
        user_level=level,
        acquired_skill_codes=skills,
        available_capabilities=capabilities,
        available_equipment_codes=equipment,
        active_contraindication_codes=contraindications,
    )


def test_review_creates_contiguous_revision_with_exact_supersedes_and_audit_lineage() -> None:
    original = _definition(status=DrillReviewStatus.DRAFT)

    approved = review_drill(
        current=original,
        new_status=DrillReviewStatus.APPROVED,
        reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
        reviewed_at=datetime(2026, 8, 4, 10, tzinfo=UTC),
        reason_code="COACH_APPROVED",
    )

    assert original.review_status is DrillReviewStatus.DRAFT
    assert approved.revision == 2
    assert approved.supersedes_revision == 1
    assert approved.supersedes_content_sha256 == original.content_sha256
    assert approved.review_lineage.reviewer_id == UUID("00000000-0000-4000-8000-000000000902")
    assert approved.review_lineage.reason_code == "COACH_APPROVED"
    assert approved.content_sha256 != original.content_sha256
    with pytest.raises(ValidationError):
        review_drill(
            current=approved,
            new_status=DrillReviewStatus.DISABLED,
            reviewer_id="reviewer@example.com",
            reviewed_at=datetime(2026, 8, 4, 11, tzinfo=UTC),
            reason_code="DISABLED_FOR_SAFETY",
        )
    with pytest.raises(ValidationError, match="UTC"):
        review_drill(
            current=approved,
            new_status=DrillReviewStatus.DISABLED,
            reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
            reviewed_at=datetime(2026, 8, 4, 11),
            reason_code="DISABLED_FOR_SAFETY",
        )
    with pytest.raises(ValidationError):
        review_drill(
            current=approved,
            new_status=DrillReviewStatus.DISABLED,
            reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
            reviewed_at=datetime(2026, 8, 4, 11, tzinfo=UTC),
            reason_code="",
        )


def test_definition_model_copy_rejects_any_public_update_even_with_coordinated_hash() -> None:
    draft = _definition(status=DrillReviewStatus.DRAFT)
    approved = DrillDefinition.build(
        **(
            draft.model_dump(mode="python", exclude={"content_sha256"})
            | {"review_status": DrillReviewStatus.APPROVED}
        )
    )

    with pytest.raises(TypeError, match="review_drill|update"):
        draft.model_copy(
            update={
                "review_status": DrillReviewStatus.APPROVED,
                "content_sha256": approved.content_sha256,
            }
        )
    assert draft.model_copy(deep=True) == draft


def test_non_initial_revision_requires_previous_content_hash_lineage() -> None:
    values = _definition().model_dump(mode="python", exclude={"content_sha256"})
    values.update(revision=2, supersedes_revision=1)

    with pytest.raises(ValidationError, match="supersedes content hash"):
        DrillDefinition.build(**values)


def test_catalog_serializes_and_validates_complete_revision_history() -> None:
    draft = _definition(status=DrillReviewStatus.DRAFT)
    approved = review_drill(
        current=draft,
        new_status=DrillReviewStatus.APPROVED,
        reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
        reviewed_at=datetime(2026, 8, 4, 10, tzinfo=UTC),
        reason_code="COACH_APPROVED",
    )

    catalog = DrillCatalog.build(
        catalog_version="2026.09",
        created_at=datetime(2026, 8, 4, 11, tzinfo=UTC),
        revision_history=(draft, approved),
    )

    assert catalog.revision_history == (draft, approved)
    assert catalog.drills == (approved,)
    assert DrillCatalog.model_validate_json(catalog.model_dump_json()) == catalog


def _review_revision(
    current: DrillDefinition,
    status: DrillReviewStatus,
    *,
    minute: int,
) -> DrillDefinition:
    return review_drill(
        current=current,
        new_status=status,
        reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
        reviewed_at=datetime(2026, 8, 4, 10, minute, tzinfo=UTC),
        reason_code=f"REVIEW_{status.value}",
    )


def test_catalog_rejects_forged_missing_branched_skipped_and_post_retired_history() -> None:
    draft = _definition(status=DrillReviewStatus.DRAFT)
    approved = _review_revision(draft, DrillReviewStatus.APPROVED, minute=1)
    disabled_branch = _review_revision(draft, DrillReviewStatus.DISABLED, minute=2)
    retired = _review_revision(approved, DrillReviewStatus.RETIRED, minute=3)
    post_retired_values = retired.model_dump(mode="python", exclude={"content_sha256"})
    post_retired_values.update(
        revision=retired.revision + 1,
        supersedes_revision=retired.revision,
        supersedes_content_sha256=retired.content_sha256,
        review_status=DrillReviewStatus.APPROVED,
        review_lineage=ReviewLineage(
            reviewer_id=UUID("00000000-0000-4000-8000-000000000903"),
            reviewed_at=datetime(2026, 8, 4, 10, 4, tzinfo=UTC),
            reason_code="ILLEGAL_RESTORE",
        ),
    )
    post_retired = DrillDefinition.build(**post_retired_values)
    forged_values = approved.model_dump(mode="python", exclude={"content_sha256"})
    forged_values["supersedes_content_sha256"] = "f" * 64
    forged = DrillDefinition.build(**forged_values)

    invalid_histories = (
        (approved,),
        (draft, retired),
        (draft, approved, disabled_branch),
        (draft, forged),
        (draft, approved, retired, post_retired),
    )
    for history in invalid_histories:
        with pytest.raises(ValidationError, match="revision|chain|branch|supersedes|RETIRED"):
            DrillCatalog.build(
                catalog_version="2026.09",
                created_at=datetime(2026, 8, 4, 11, tzinfo=UTC),
                revision_history=history,
            )


def test_target_zone_requires_explicit_runtime_capability() -> None:
    definition = _definition()
    payload = definition.model_dump(mode="python", exclude={"content_sha256"})
    payload["setup"] = TableSetup(
        table_width_mm=1270,
        table_length_mm=2540,
        placements=definition.setup.placements,
        target_pocket_code=None,
        target_zone=TableTargetZoneMm(
            zone_code="CUE_TARGET",
            minimum_x_mm=100,
            maximum_x_mm=300,
            minimum_y_mm=100,
            maximum_y_mm=300,
        ),
    )

    with pytest.raises(ValidationError, match="CUE_BALL_TARGET_ZONE"):
        DrillDefinition.build(**payload)

    payload["required_capabilities"] = (
        "CUE_BALL_TARGET_ZONE",
        "TABLE_MM_COORDINATES",
    )
    assert "CUE_BALL_TARGET_ZONE" in DrillDefinition.build(**payload).required_capabilities


def test_retired_is_an_irreversible_terminal_review_status() -> None:
    retired = review_drill(
        current=_definition(),
        new_status=DrillReviewStatus.RETIRED,
        reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
        reviewed_at=datetime(2026, 8, 4, 10, tzinfo=UTC),
        reason_code="CONTENT_RETIRED",
    )

    with pytest.raises(ValueError, match="terminal|RETIRED"):
        review_drill(
            current=retired,
            new_status=DrillReviewStatus.APPROVED,
            reviewer_id=UUID("00000000-0000-4000-8000-000000000902"),
            reviewed_at=datetime(2026, 8, 4, 11, tzinfo=UTC),
            reason_code="RESTORE_NOT_ALLOWED",
        )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (_reference(catalog_version="2026.09"), DrillViolationCode.CATALOG_VERSION_MISMATCH),
        (_reference(drill_id="missing"), DrillViolationCode.DRILL_NOT_FOUND),
        (_reference(drill_revision=2), DrillViolationCode.DRILL_REVISION_STALE),
        (
            _reference(required_capabilities=("TABLE_MM_COORDINATES", "SPIN_DETECTION")),
            DrillViolationCode.REQUIRED_CAPABILITIES_MISMATCH,
        ),
    ],
)
def test_exact_reference_validation_returns_stable_violation_codes(reference, expected) -> None:
    result = validate_drill_reference(catalog=_catalog(_definition()), reference=reference)

    assert result.is_valid is False
    assert result.drill is None
    assert result.violation_codes == (expected,)


def test_exact_reference_resolves_only_approved_current_head() -> None:
    approved = validate_drill_reference(catalog=_catalog(_definition()), reference=_reference())
    disabled = validate_drill_reference(
        catalog=_catalog(_definition(status=DrillReviewStatus.DISABLED)),
        reference=_reference(),
    )

    assert approved.is_valid is True
    assert approved.drill is not None
    assert disabled.violation_codes == (DrillViolationCode.DRILL_NOT_APPROVED,)


def test_reference_validation_catches_capability_forged_with_unvalidated_c0_model_copy() -> None:
    forged = _reference().model_copy(
        update={"required_capabilities": ("SPIN_DETECTION", "TABLE_MM_COORDINATES")}
    )

    result = validate_drill_reference(catalog=_catalog(_definition()), reference=forged)

    assert result.violation_codes == (DrillViolationCode.REQUIRED_CAPABILITIES_MISMATCH,)


@pytest.mark.parametrize("operation", ["validate_reference", "list_eligible"])
def test_services_revalidate_catalogs_created_with_pydantic_model_construct(operation) -> None:
    drill = _definition()
    forged_drill = DrillDefinition.model_construct(
        **(drill.model_dump(mode="python") | {"content_sha256": "0" * 64})
    )
    forged_catalog = DrillCatalog.model_construct(
        schema_version="1.0",
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(forged_drill,),
        content_sha256="0" * 64,
    )

    with pytest.raises(ValidationError, match="hash"):
        if operation == "validate_reference":
            validate_drill_reference(catalog=forged_catalog, reference=_reference())
        else:
            list_eligible_drills(catalog=forged_catalog, profile=_profile())


@pytest.mark.parametrize(
    ("drill", "profile", "expected"),
    [
        (
            _definition(status=DrillReviewStatus.DISABLED),
            _profile(),
            DrillViolationCode.DRILL_NOT_APPROVED,
        ),
        (_definition(), _profile(level=4), DrillViolationCode.USER_LEVEL_OUT_OF_RANGE),
        (
            _definition(prerequisites=("BASIC_STANCE",)),
            _profile(skills=("POTTING",)),
            DrillViolationCode.PREREQUISITE_SKILL_MISSING,
        ),
        (
            _definition(capabilities=("TABLE_MM_COORDINATES", "SPIN_DETECTION")),
            _profile(),
            DrillViolationCode.DEVICE_CAPABILITY_MISSING,
        ),
        (
            _definition(),
            _profile(equipment=()),
            DrillViolationCode.REQUIRED_EQUIPMENT_MISSING,
        ),
        (
            _definition(),
            _profile(contraindications=("SHOULDER_INJURY",)),
            DrillViolationCode.CONTRAINDICATION_PRESENT,
        ),
    ],
)
def test_eligibility_rejects_each_closed_gate(drill, profile, expected) -> None:
    result = list_eligible_drills(catalog=_catalog(drill), profile=profile)

    assert result == ()
    assert expected in drill.eligibility_violations(profile)


def test_candidate_filtering_returns_only_eligible_approved_heads_in_stable_order() -> None:
    catalog = _catalog(
        _definition(drill_id="z-drill"),
        _definition(drill_id="a-drill"),
        _definition(drill_id="disabled", status=DrillReviewStatus.DISABLED),
        _definition(drill_id="advanced", minimum_level=4, maximum_level=5),
    )

    first = list_eligible_drills(catalog=catalog, profile=_profile())
    repeated = list_eligible_drills(catalog=catalog, profile=_profile())

    assert tuple(item.drill_id for item in first) == ("a-drill", "z-drill")
    assert repeated == first
