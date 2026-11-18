"""Strict public contracts for Story 5.4 traceable recommendations."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import (
    Confidence,
    Recommendation,
    RecommendationCatalogVersion,
    RecommendationClaimCode,
    RecommendationDrill,
    RecommendationEvidence,
    RecommendationEvidenceKind,
    RecommendationPracticeCode,
    RecommendationReason,
    RecommendationResult,
    RecommendationRuleId,
    RecommendationRuleVersion,
    RecommendationScope,
    RecommendationSemanticCore,
    RecommendationSourceLineage,
    RecommendationStatus,
    ReviewStatus,
    ShotAnalysisStatus,
    TargetZoneSource,
    TypicalShotRuleVersion,
    TypicalShotSelectionReason,
)

SESSION_ID = UUID("10000000-0000-4000-8000-000000000001")
REPORT_ID = UUID("20000000-0000-4000-8000-000000000001")
SHOT_ID = UUID("30000000-0000-4000-8000-000000000001")


def _lineage() -> RecommendationSourceLineage:
    return RecommendationSourceLineage(
        session_id=SESSION_ID,
        report_id=REPORT_ID,
        report_revision=1,
        selection_rule_version=TypicalShotRuleVersion.V1,
    )


def _evidence(
    *,
    kind: RecommendationEvidenceKind = RecommendationEvidenceKind.FOUL,
) -> RecommendationEvidence:
    common = dict(
        evidence_kind=kind,
        shot_id=SHOT_ID,
        sequence=1,
        confidence=Confidence.HIGH,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=ReviewStatus.APPROVED,
        selection_rank=1,
        selection_reason=TypicalShotSelectionReason.SEVERE_FOUL,
        selection_rule_version=TypicalShotRuleVersion.V1,
    )
    if kind is RecommendationEvidenceKind.FOUL:
        return RecommendationEvidence(is_foul=True, **common)
    if kind is RecommendationEvidenceKind.MISSED_POT:
        common["selection_reason"] = TypicalShotSelectionReason.SEVERE_MISSED_POT
        return RecommendationEvidence(is_foul=False, potted_balls=(), **common)
    common["selection_reason"] = TypicalShotSelectionReason.SEVERE_DEVIATION
    return RecommendationEvidence(
        is_foul=False,
        deviation_mm=500,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
        **common,
    )


def _recommendation() -> Recommendation:
    core = RecommendationSemanticCore(
        rule_id=RecommendationRuleId.REDUCE_FOUL_RISK,
        rule_version=RecommendationRuleVersion.V1,
        scope=RecommendationScope.SESSION_PRIMARY,
        claim_code=RecommendationClaimCode.REDUCE_FOUL_RISK,
        practice_code=RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
    )
    return Recommendation(
        semantic_core=core,
        evidence=(_evidence(),),
        drill=RecommendationDrill(
            practice_code=RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL
        ),
    )


def _ready_result() -> RecommendationResult:
    return RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=RecommendationStatus.READY,
        source_lineage=_lineage(),
        recommendations=(_recommendation(),),
    )


def test_recommendation_contract_is_strict_frozen_round_trippable_and_deterministic():
    result = _ready_result()

    assert RecommendationResult.model_validate_json(result.model_dump_json()) == result
    assert (
        RecommendationResult.build(
            catalog_version=RecommendationCatalogVersion.V1,
            status=RecommendationStatus.READY,
            source_lineage=_lineage(),
            recommendations=(_recommendation(),),
        )
        == result
    )
    assert len(result.content_sha256) == 64
    canonical = json.dumps(
        result.model_dump(mode="json", exclude={"content_sha256"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert result.content_sha256 == hashlib.sha256(canonical).hexdigest()
    with pytest.raises(ValidationError, match="frozen"):
        result.status = RecommendationStatus.INVALID_INPUT
    with pytest.raises(ValidationError, match="unexpected"):
        RecommendationResult.model_validate(result.model_dump() | {"unexpected": "value"})


@pytest.mark.parametrize(
    "update",
    [
        {"report_revision": True},
        {"report_revision": "1"},
        {"selection_rule_version": "future"},
        {"media_url": "https://example.invalid/a.mp4"},
    ],
)
def test_source_lineage_rejects_bool_strings_unknown_versions_and_extras(update):
    with pytest.raises(ValidationError):
        RecommendationSourceLineage.model_validate(_lineage().model_dump() | update)


@pytest.mark.parametrize(
    ("status", "reason", "lineage", "recommendations", "fallback"),
    [
        (RecommendationStatus.READY, None, _lineage(), (_recommendation(),), None),
        (
            RecommendationStatus.NO_RELIABLE_RECOMMENDATION,
            RecommendationReason.NO_NEGATIVE_RULE_MATCH,
            _lineage(),
            (),
            "当前数据不足以给出训练建议",
        ),
        (
            RecommendationStatus.SELECTION_MISMATCH,
            RecommendationReason.SELECTION_RESULT_MISMATCH,
            _lineage(),
            (),
            None,
        ),
        (
            RecommendationStatus.INVALID_INPUT,
            RecommendationReason.INVALID_IDENTIFIER_OR_INPUT,
            None,
            (),
            None,
        ),
    ],
)
def test_result_status_matrix_is_exact(status, reason, lineage, recommendations, fallback):
    result = RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=status,
        reason=reason,
        source_lineage=lineage,
        recommendations=recommendations,
    )

    assert result.fallback == fallback
    assert result.recommendations == recommendations


@pytest.mark.parametrize(
    "reason",
    [
        RecommendationReason.NO_NEGATIVE_RULE_MATCH,
        RecommendationReason.INSUFFICIENT_EVIDENCE,
        RecommendationReason.UNSUPPORTED_EVIDENCE,
    ],
)
def test_no_reliable_status_accepts_only_closed_abstention_reasons(reason):
    result = RecommendationResult.build(
        catalog_version=RecommendationCatalogVersion.V1,
        status=RecommendationStatus.NO_RELIABLE_RECOMMENDATION,
        reason=reason,
        source_lineage=_lineage(),
        recommendations=(),
    )
    assert result.reason is reason


@pytest.mark.parametrize(
    "changes",
    [
        {
            "status": RecommendationStatus.READY,
            "reason": RecommendationReason.INSUFFICIENT_EVIDENCE,
        },
        {"status": RecommendationStatus.READY, "recommendations": ()},
        {
            "status": RecommendationStatus.NO_RELIABLE_RECOMMENDATION,
            "reason": RecommendationReason.NO_NEGATIVE_RULE_MATCH,
            "recommendations": (_recommendation(),),
        },
        {
            "status": RecommendationStatus.SELECTION_MISMATCH,
            "reason": RecommendationReason.SELECTION_RESULT_MISMATCH,
            "source_lineage": None,
            "recommendations": (),
        },
        {
            "status": RecommendationStatus.INVALID_INPUT,
            "reason": RecommendationReason.INVALID_IDENTIFIER_OR_INPUT,
            "source_lineage": _lineage(),
            "recommendations": (),
        },
    ],
)
def test_invalid_status_combinations_are_rejected(changes):
    payload = {
        "catalog_version": RecommendationCatalogVersion.V1,
        "status": RecommendationStatus.READY,
        "reason": None,
        "source_lineage": _lineage(),
        "recommendations": (_recommendation(),),
        "content_sha256": "0" * 64,
    }
    with pytest.raises(ValidationError):
        RecommendationResult.model_validate(payload | changes)


def test_recommendation_has_one_semantic_core_one_evidence_set_and_at_most_one_drill():
    recommendation = _recommendation()

    assert recommendation.title == "建议下次重点练：减少犯规"
    assert recommendation.primary_advice == "先把无犯规的基础击球稳定下来。"
    assert recommendation.drill.instruction == "固定同一组基础直线球位重复击球，逐杆记录是否犯规。"
    with pytest.raises(ValidationError):
        Recommendation.model_validate(
            recommendation.model_dump() | {"drill": (recommendation.drill, recommendation.drill)}
        )


def test_serialized_recommendation_contains_only_catalog_controlled_user_copy():
    result = _ready_result()
    payload = result.model_dump(mode="json")
    recommendation = payload["recommendations"][0]

    assert recommendation["title"] == "建议下次重点练：减少犯规"
    assert recommendation["primary_advice"] == "先把无犯规的基础击球稳定下来。"
    assert recommendation["drill"]["instruction"] == (
        "固定同一组基础直线球位重复击球，逐杆记录是否犯规。"
    )
    with pytest.raises(ValidationError, match="title|catalog|controlled"):
        RecommendationResult.model_validate(
            payload
            | {"recommendations": (recommendation | {"title": "user supplied recommendation"},)}
        )


def test_semantic_core_evidence_kind_mapping_is_frozen():
    recommendation = _recommendation()

    with pytest.raises(ValidationError, match="evidence|rule|catalog"):
        Recommendation(
            semantic_core=recommendation.semantic_core,
            evidence=(_evidence(kind=RecommendationEvidenceKind.MISSED_POT),),
            drill=recommendation.drill,
        )


@pytest.mark.parametrize(
    "update",
    [
        {"confidence": Confidence.LOW},
        {"confidence": Confidence.ABSTAIN},
        {"analysis_status": ShotAnalysisStatus.PENDING},
        {"review_status": ReviewStatus.PENDING},
        {"review_status": ReviewStatus.IN_REVIEW},
    ],
)
def test_recommendation_evidence_requires_selector_eligible_reviewed_shot(update):
    with pytest.raises(ValidationError, match="eligible|confidence|analysis|review"):
        RecommendationEvidence.model_validate(_evidence().model_dump() | update)


def test_drill_is_required_and_must_match_practice_code_bidirectionally():
    recommendation = _recommendation()

    with pytest.raises(ValidationError, match="drill|practice"):
        Recommendation(
            semantic_core=recommendation.semantic_core,
            evidence=recommendation.evidence,
            drill=None,
        )
    with pytest.raises(ValidationError, match="drill|practice"):
        Recommendation(
            semantic_core=recommendation.semantic_core,
            evidence=recommendation.evidence,
            drill=RecommendationDrill(
                practice_code=RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS
            ),
        )


def test_evidence_selection_ranks_are_unique_and_bounded_by_v1_selector():
    recommendation = _recommendation()
    second = _evidence().model_copy(
        update={
            "shot_id": UUID("30000000-0000-4000-8000-000000000002"),
            "sequence": 2,
        }
    )

    with pytest.raises(ValidationError, match="rank"):
        Recommendation(
            semantic_core=recommendation.semantic_core,
            evidence=(recommendation.evidence[0], second),
            drill=recommendation.drill,
        )
    with pytest.raises(ValidationError, match="selection_rank|less than or equal"):
        RecommendationEvidence.model_validate(
            recommendation.evidence[0].model_dump() | {"selection_rank": 6}
        )


@pytest.mark.parametrize(
    ("kind", "update"),
    [
        (RecommendationEvidenceKind.FOUL, {"is_foul": False}),
        (RecommendationEvidenceKind.FOUL, {"potted_balls": ()}),
        (RecommendationEvidenceKind.MISSED_POT, {"is_foul": True}),
        (RecommendationEvidenceKind.MISSED_POT, {"potted_balls": (3,)}),
        (RecommendationEvidenceKind.LARGE_DEVIATION, {"deviation_mm": None}),
        (RecommendationEvidenceKind.LARGE_DEVIATION, {"target_zone_source": TargetZoneSource.NONE}),
    ],
)
def test_evidence_kind_allows_only_its_minimal_canonical_scalar_snapshot(kind, update):
    evidence = _evidence(kind=kind)
    with pytest.raises(ValidationError):
        RecommendationEvidence.model_validate(evidence.model_dump() | update)


@pytest.mark.parametrize(
    "update",
    [
        {"sequence": True},
        {"selection_rank": "1"},
        {"raw_observation": {"text": "inject"}},
        {"trajectory": ((0, 0), (1, 1))},
        {"asset_path": "/tmp/asset.svg"},
        {"media_url": "https://example.invalid/a.mp4"},
        {"free_text": "x\u0000y"},
    ],
)
def test_evidence_rejects_coercion_raw_geometry_paths_urls_and_control_text(update):
    with pytest.raises(ValidationError):
        RecommendationEvidence.model_validate(_evidence().model_dump() | update)


@pytest.mark.parametrize(
    "update",
    [
        {"catalog_version": "future"},
        {"status": "future"},
        {"reason": "free-form"},
        {"content_sha256": "g" * 64},
        {"content_sha256": "0" * 64},
        {"recommendations": (_recommendation(), _recommendation())},
    ],
)
def test_result_model_copy_revalidates_unknown_codes_digest_and_cardinality(update):
    with pytest.raises(ValidationError):
        _ready_result().model_copy(update=update)


def test_catalog_codes_lock_all_three_v1_templates_and_semantic_pairs():
    expected = {
        RecommendationClaimCode.REDUCE_FOUL_RISK: (
            RecommendationRuleId.REDUCE_FOUL_RISK,
            RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
            "建议下次重点练：减少犯规",
        ),
        RecommendationClaimCode.STABILIZE_POTTING: (
            RecommendationRuleId.STABILIZE_POTTING,
            RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS,
            "建议下次重点练：基础进球",
        ),
        RecommendationClaimCode.CONTROL_CUE_BALL_POSITION: (
            RecommendationRuleId.CONTROL_CUE_BALL_POSITION,
            RecommendationPracticeCode.TRACK_TARGET_ZONE_FINISHES,
            "建议下次重点练：母球停位",
        ),
    }
    for claim, (rule_id, practice, title) in expected.items():
        core = RecommendationSemanticCore(
            rule_id=rule_id,
            rule_version=RecommendationRuleVersion.V1,
            scope=RecommendationScope.SESSION_PRIMARY,
            claim_code=claim,
            practice_code=practice,
        )
        assert core.title == title


def test_public_schema_and_serialization_contain_no_external_or_uncontrolled_fields():
    serialized = json.dumps(_ready_result().model_dump(mode="json"), ensure_ascii=False).lower()
    schema = json.dumps(RecommendationResult.model_json_schema()).lower()
    forbidden = (
        "url",
        "path",
        "media",
        "trajectory",
        "geometry",
        "raw_observation",
        "pooltool",
        "provider",
        "coordinate",
    )
    assert all(token not in serialized for token in forbidden)
    assert all(token not in schema for token in forbidden)


def test_nested_constructed_models_cannot_bypass_semantic_or_result_validation():
    recommendation = _recommendation()
    invalid_core = RecommendationSemanticCore.model_construct(
        rule_id=RecommendationRuleId.REDUCE_FOUL_RISK,
        rule_version=RecommendationRuleVersion.V1,
        scope=RecommendationScope.SESSION_PRIMARY,
        claim_code=RecommendationClaimCode.STABILIZE_POTTING,
        practice_code=RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
    )
    with pytest.raises(ValidationError, match="rule|claim|practice"):
        Recommendation(
            semantic_core=invalid_core,
            evidence=recommendation.evidence,
            drill=recommendation.drill,
        )

    invalid_lineage = RecommendationSourceLineage.model_construct(
        session_id=SESSION_ID,
        report_id=REPORT_ID,
        report_revision=True,
        selection_rule_version=TypicalShotRuleVersion.V1,
    )
    with pytest.raises(ValidationError, match="report_revision"):
        RecommendationResult.build(
            catalog_version=RecommendationCatalogVersion.V1,
            status=RecommendationStatus.READY,
            source_lineage=invalid_lineage,
            recommendations=(recommendation,),
        )

    invalid_recommendation = Recommendation.model_construct(
        semantic_core=recommendation.semantic_core,
        evidence=(),
        drill=recommendation.drill,
        wording_variant=recommendation.wording_variant,
    )
    with pytest.raises(ValidationError, match="evidence"):
        RecommendationResult.build(
            catalog_version=RecommendationCatalogVersion.V1,
            status=RecommendationStatus.READY,
            source_lineage=_lineage(),
            recommendations=(invalid_recommendation,),
        )
