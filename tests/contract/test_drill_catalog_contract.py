from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.training.catalog import (
    BallPlacement,
    DrillCatalog,
    DrillDefinition,
    DrillEligibilityProfile,
    DrillReferenceValidation,
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
    validate_drill_reference,
)
from billiards_ai.shared.models.training_plan import DrillReference, TrainingPlan

PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "core_intelligence" / "v1"
CATALOG_PATH = FIXTURE_ROOT / "drill-catalog.json"
FORBIDDEN_KEYS = {
    "openid",
    "token",
    "access_token",
    "prompt",
    "provider",
    "raw_observation",
    "video_url",
    "media_url",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | {
            nested for item in value.values() for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()


def _all_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(item for nested in value.values() for item in _all_strings(nested))
    if isinstance(value, list):
        return tuple(item for nested in value for item in _all_strings(nested))
    return ()


def _definition(*, drill_id: str = "straight-potting-01") -> DrillDefinition:
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
        instructions=("Align the cue ball and object ball.", "Pot into the target pocket."),
        success_metrics=(
            SuccessMetricDefinition(
                metric_code="POT_SUCCESS_COUNT",
                target_attempts=10,
                minimum_successes=7,
            ),
        ),
        progression_rule=ProgressionRule(
            reason_code="SUCCESS_STREAK",
            threshold=4,
            level_delta=1,
        ),
        regression_rule=RegressionRule(
            reason_code="FAILURE_STREAK",
            threshold=3,
            level_delta=-1,
        ),
        required_equipment_codes=("STANDARD_TABLE",),
        required_capabilities=("TABLE_MM_COORDINATES",),
        safety_codes=("KEEP_CUE_CLEAR_OF_BYSTANDERS",),
        contraindication_codes=("SHOULDER_INJURY",),
        review_status=DrillReviewStatus.APPROVED,
        review_lineage=ReviewLineage(
            reviewer_id=UUID("00000000-0000-4000-8000-000000000901"),
            reviewed_at=datetime(2026, 8, 4, 8, tzinfo=UTC),
            reason_code="INITIAL_SYNTHETIC_REVIEW",
        ),
    )


def test_contracts_are_strict_frozen_extra_forbid_round_trippable_and_hash_bound() -> None:
    definition = _definition()
    catalog = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(definition,),
    )

    assert DrillCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    with pytest.raises(ValidationError, match="frozen"):
        definition.revision = 2
    with pytest.raises(ValidationError, match="unexpected"):
        DrillDefinition.model_validate(definition.model_dump() | {"unexpected": "forbidden"})
    with pytest.raises(ValidationError, match="hash"):
        DrillDefinition.model_validate(definition.model_dump() | {"content_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="hash"):
        DrillCatalog.model_validate(catalog.model_dump() | {"content_sha256": "0" * 64})


def test_contracts_reject_unknown_schema_naive_time_and_model_copy_bypass() -> None:
    definition = _definition()
    catalog = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(definition,),
    )

    with pytest.raises(ValidationError):
        DrillCatalog.model_validate(catalog.model_dump() | {"schema_version": "2.0"})
    with pytest.raises(ValidationError, match="UTC"):
        DrillCatalog.build(
            catalog_version="2026.08",
            created_at=datetime(2026, 8, 4, 9),
            drills=(definition,),
        )
    with pytest.raises(TypeError, match="review_drill|update"):
        definition.model_copy(update={"revision": 0})
    draft = _definition()
    draft = DrillDefinition.build(
        **(
            draft.model_dump(mode="python", exclude={"content_sha256"})
            | {"review_status": DrillReviewStatus.DRAFT}
        )
    )
    with pytest.raises(TypeError, match="review_drill|update"):
        draft.model_copy(update={"review_status": DrillReviewStatus.APPROVED})
    with pytest.raises(ValidationError, match="hash"):
        catalog.model_copy(update={"catalog_version": "forged"})


def test_build_hashes_validated_normalized_payload_with_omitted_defaults() -> None:
    payload = _definition().model_dump(
        mode="python",
        exclude={"schema_version", "content_sha256", "contraindication_codes"},
    )
    payload["drill_id"] = "  normalized-drill  "

    definition = DrillDefinition.build(**payload)
    catalog = DrillCatalog.build(
        catalog_version="  2026.09  ",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(definition,),
    )

    assert definition.drill_id == "normalized-drill"
    assert definition.contraindication_codes == ()
    assert catalog.catalog_version == "2026.09"


@pytest.mark.parametrize(
    "field_name",
    [
        "skill_codes",
        "prerequisite_skill_codes",
        "required_equipment_codes",
        "required_capabilities",
        "safety_codes",
        "contraindication_codes",
    ],
)
@pytest.mark.parametrize("bad_container", ["POTTING", b"POTTING"])
def test_build_rejects_string_and_bytes_for_every_code_set(field_name, bad_container) -> None:
    payload = _definition().model_dump(mode="python", exclude={"content_sha256"})
    payload[field_name] = bad_container

    with pytest.raises(ValidationError, match="tuple"):
        DrillDefinition.build(**payload)


@pytest.mark.parametrize("mapping_input", [False, True])
def test_catalog_validates_and_normalizes_drills_before_stable_sort(mapping_input) -> None:
    first = _definition(drill_id="a-drill")
    last = _definition(drill_id="z-drill")
    unnormalized = last.model_dump(mode="python") | {"drill_id": "  z-drill  "}
    if not mapping_input:
        unnormalized = DrillDefinition.model_construct(**unnormalized)

    catalog = DrillCatalog.build(
        catalog_version="2026.09",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(unnormalized, first),
    )

    assert tuple(drill.drill_id for drill in catalog.drills) == ("a-drill", "z-drill")


def test_table_metric_level_code_and_result_validators_reject_invalid_contracts() -> None:
    setup = _definition().setup
    zone = TableTargetZoneMm(
        zone_code="TARGET",
        minimum_x_mm=100,
        maximum_x_mm=200,
        minimum_y_mm=100,
        maximum_y_mm=200,
    )

    invalid_setups = (
        setup.model_dump() | {"target_zone": zone},
        setup.model_dump() | {"placements": (setup.placements[0], setup.placements[0])},
        setup.model_dump()
        | {
            "placements": (
                BallPlacement(
                    ball_code="CUE",
                    position=TablePointMm(x_mm=setup.table_length_mm + 1, y_mm=0),
                ),
            )
        },
    )
    for payload in invalid_setups:
        with pytest.raises(ValidationError):
            TableSetup.model_validate(payload)

    with pytest.raises(ValidationError, match="target zone"):
        TableSetup.model_validate(
            setup.model_dump()
            | {
                "target_pocket_code": None,
                "target_zone": zone.model_copy(update={"maximum_y_mm": setup.table_width_mm + 1}),
            }
        )

    with pytest.raises(ValidationError, match="positive area"):
        TableTargetZoneMm(
            zone_code="TARGET",
            minimum_x_mm=200,
            maximum_x_mm=100,
            minimum_y_mm=100,
            maximum_y_mm=200,
        )
    with pytest.raises(ValidationError, match="successes"):
        SuccessMetricDefinition(
            metric_code="POT_SUCCESS_COUNT", target_attempts=5, minimum_successes=6
        )
    with pytest.raises(ValidationError):
        ProgressionRule(reason_code="BAD", threshold=1, level_delta=-1)
    with pytest.raises(ValidationError):
        RegressionRule(reason_code="BAD", threshold=1, level_delta=1)

    definition = _definition()
    for update in (
        {"minimum_level": 4, "maximum_level": 3},
        {"success_metrics": (definition.success_metrics[0], definition.success_metrics[0])},
        {"skill_codes": ("POTTING", "AIMING")},
        {"skill_codes": ("POTTING", "POTTING")},
    ):
        with pytest.raises(ValidationError):
            DrillDefinition.model_validate(definition.model_dump() | update)


def test_table_setup_uses_ad13_length_x_width_y_axes_for_points_and_zones() -> None:
    long_edge_point = TableSetup(
        table_width_mm=1270,
        table_length_mm=2540,
        placements=(
            BallPlacement(
                ball_code="CUE",
                position=TablePointMm(x_mm=2000, y_mm=1200),
            ),
        ),
        target_pocket_code="TOP_CENTER",
        target_zone=None,
    )

    assert long_edge_point.placements[0].position.x_mm == 2000
    with pytest.raises(ValidationError, match="table millimeter bounds"):
        TableSetup(
            table_width_mm=1270,
            table_length_mm=2540,
            placements=(
                BallPlacement(
                    ball_code="CUE",
                    position=TablePointMm(x_mm=1200, y_mm=2000),
                ),
            ),
            target_pocket_code="TOP_CENTER",
            target_zone=None,
        )

    long_edge_zone = TableSetup(
        table_width_mm=1270,
        table_length_mm=2540,
        placements=long_edge_point.placements,
        target_pocket_code=None,
        target_zone=TableTargetZoneMm(
            zone_code="LONG_EDGE_TARGET",
            minimum_x_mm=1800,
            maximum_x_mm=2200,
            minimum_y_mm=800,
            maximum_y_mm=1100,
        ),
    )
    assert long_edge_zone.target_zone.maximum_x_mm == 2200
    with pytest.raises(ValidationError, match="target zone"):
        TableSetup(
            table_width_mm=1270,
            table_length_mm=2540,
            placements=long_edge_point.placements,
            target_pocket_code=None,
            target_zone=TableTargetZoneMm(
                zone_code="TRANSPOSED_TARGET",
                minimum_x_mm=800,
                maximum_x_mm=1100,
                minimum_y_mm=1800,
                maximum_y_mm=2200,
            ),
        )


def test_table_setup_rejects_distinct_balls_at_identical_coordinates() -> None:
    shared_position = TablePointMm(x_mm=635, y_mm=635)

    with pytest.raises(ValidationError, match="coordinate|overlap"):
        TableSetup(
            table_width_mm=1270,
            table_length_mm=2540,
            placements=(
                BallPlacement(ball_code="CUE", position=shared_position),
                BallPlacement(ball_code="OBJECT_1", position=shared_position),
            ),
            target_pocket_code="TOP_CENTER",
            target_zone=None,
        )


def test_reference_validation_result_rejects_inconsistent_state() -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        DrillReferenceValidation(is_valid=True, drill=None, violation_codes=())


@pytest.mark.parametrize(
    ("is_valid", "drill", "violations"),
    [
        (False, None, ()),
        (False, _definition(), (DrillViolationCode.DRILL_NOT_APPROVED,)),
        (True, None, ()),
        (True, _definition(), (DrillViolationCode.DRILL_NOT_APPROVED,)),
    ],
)
def test_reference_validation_freezes_valid_and_invalid_result_matrix(
    is_valid, drill, violations
) -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        DrillReferenceValidation(
            is_valid=is_valid,
            drill=drill,
            violation_codes=violations,
        )


@pytest.mark.parametrize(
    "sensitive_value",
    [
        "https://example.invalid/training.mp4",
        "user-openid-value",
        "access_token=abc123",
        "prompt override",
        "provider_response",
        "raw_observation",
        "media_path=/tmp/session.mp4",
        "video_locator=s3/object",
        "Bearer abcdefghijklmnop",
        "sk-1234567890abcdef",
        "client_secret=abcdef123456",
    ],
)
def test_models_reject_sensitive_string_values_not_only_fixture_keys(sensitive_value) -> None:
    payload = _definition().model_dump(mode="python", exclude={"content_sha256"})
    payload["instructions"] = (sensitive_value,)

    with pytest.raises(ValidationError, match="sensitive"):
        DrillDefinition.build(**payload)


def test_sensitive_value_policy_allows_normal_billiards_domain_codes() -> None:
    definition = _definition(drill_id="target-zone-potting-01")

    assert definition.success_metrics[0].metric_code == "POT_SUCCESS_COUNT"
    assert definition.safety_codes == ("KEEP_CUE_CLEAR_OF_BYSTANDERS",)


@pytest.mark.parametrize(
    "skill_codes",
    [("POTTING", "AIMING"), ("POTTING", "POTTING")],
)
def test_eligibility_profile_rejects_noncanonical_code_tuples(skill_codes) -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        DrillEligibilityProfile(
            user_level=1,
            acquired_skill_codes=skill_codes,
            available_capabilities=(),
            available_equipment_codes=(),
            active_contraindication_codes=(),
        )


def test_catalog_rejects_duplicate_heads_and_builds_stable_order_and_hash() -> None:
    first = _definition(drill_id="z-drill")
    second = _definition(drill_id="a-drill")

    catalog = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(first, second),
    )
    repeated = DrillCatalog.build(
        catalog_version="2026.08",
        created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
        drills=(second, first),
    )

    assert tuple(item.drill_id for item in catalog.drills) == ("a-drill", "z-drill")
    assert catalog == repeated
    with pytest.raises(ValidationError, match="head|unique"):
        DrillCatalog.build(
            catalog_version="2026.08",
            created_at=datetime(2026, 8, 4, 9, tzinfo=UTC),
            drills=(first, first),
        )


def test_frozen_fixture_excludes_sensitive_fields_and_uses_only_mm_setup() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = DrillCatalog.model_validate_json(CATALOG_PATH.read_text(encoding="utf-8"))

    assert DrillCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    assert catalog.revision_history == catalog.drills
    assert _all_keys(payload).isdisjoint(FORBIDDEN_KEYS)
    lowered_values = tuple(value.lower() for value in _all_strings(payload))
    assert all("://" not in value for value in lowered_values)
    assert all(
        fragment not in value
        for value in lowered_values
        for fragment in (
            "openid",
            "token",
            "raw_observation",
            "prompt",
            "provider",
            "sk-",
        )
    )
    for drill in payload["drills"]:
        setup = drill["setup"]
        assert "placements" in setup
        assert "target_pocket_code" in setup or "target_zone" in setup
        for placement in setup["placements"]:
            assert set(placement["position"]) == {"x_mm", "y_mm"}
            assert all(isinstance(value, int) for value in placement["position"].values())


def test_fixture_resolves_every_c0_training_plan_reference_exactly() -> None:
    catalog = DrillCatalog.model_validate_json(CATALOG_PATH.read_text(encoding="utf-8"))
    plan = TrainingPlan.model_validate_json(
        (FIXTURE_ROOT / "training-plan.json").read_text(encoding="utf-8")
    )
    references: tuple[DrillReference, ...] = (
        *(block.drill for block in plan.blocks),
        plan.fallback_drill,
    )

    results = tuple(validate_drill_reference(catalog=catalog, reference=ref) for ref in references)

    assert all(result.is_valid for result in results)
    assert tuple(result.drill.drill_id for result in results if result.drill) == (
        "large-angle-potting-01",
        "straight-potting-01",
    )


def test_fixture_freezes_capability_insufficient_and_disabled_paths() -> None:
    catalog = DrillCatalog.model_validate_json(CATALOG_PATH.read_text(encoding="utf-8"))
    profile = DrillEligibilityProfile(
        user_level=3,
        acquired_skill_codes=("POTTING",),
        available_capabilities=("TABLE_MM_COORDINATES",),
        available_equipment_codes=("STANDARD_TABLE",),
        active_contraindication_codes=(),
    )

    eligible = list_eligible_drills(catalog=catalog, profile=profile)
    drills = {item.drill_id: item for item in catalog.drills}

    assert tuple(item.drill_id for item in eligible) == (
        "large-angle-potting-01",
        "straight-potting-01",
    )
    assert drills["capability-target-zone-01"].eligibility_violations(profile) == (
        DrillViolationCode.DEVICE_CAPABILITY_MISSING,
    )
    assert drills["disabled-bank-shot-01"].eligibility_violations(profile) == (
        DrillViolationCode.DRILL_NOT_APPROVED,
    )


def test_catalog_module_has_no_db_api_provider_or_recommendation_catalog_dependency() -> None:
    source = (PROJECT_ROOT / "src/billiards_ai/cloud/training/catalog.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "_CORE_PAIRS",
        "RecommendationCatalogVersion",
        "billiards_ai.shared.models.recommendation",
        "sqlalchemy",
        "fastapi",
        "openai",
        "anthropic",
    )

    assert all(token not in source for token in forbidden)
