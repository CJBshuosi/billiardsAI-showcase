"""Traceable deterministic recommendation rules for Story 5.4."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.engine import generate_recommendations, select_typical_shots
from billiards_ai.cloud.review import ReportNotDeliverableError
from billiards_ai.shared.models import (
    ApprovedSessionData,
    ApprovedShotData,
    Confidence,
    RecommendationCatalogVersion,
    RecommendationClaimCode,
    RecommendationEvidenceKind,
    RecommendationPracticeCode,
    RecommendationReason,
    RecommendationRuleId,
    RecommendationStatus,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReviewStatus,
    ShotAnalysisStatus,
    TargetZoneSource,
    TypicalShotRuleVersion,
    TypicalShotSelectionResult,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"


class _ApprovedRepository:
    def __init__(self, data: ApprovedSessionData):
        self.data = data
        self.calls: list[tuple[UUID, UUID, int]] = []

    def get_approved_session_data(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
    ) -> ApprovedSessionData:
        self.calls.append((session_id, report_id, revision))
        return self.data


class _BlockedRepository:
    def __init__(self, failure: ReportNotDeliverableError):
        self.failure = failure

    def get_approved_session_data(self, session_id, report_id, revision):
        raise self.failure


class _BrokenRepository:
    def get_approved_session_data(self, session_id, report_id, revision):
        raise RuntimeError("private repository implementation detail")


class _BrokenAttributeRepository:
    def __getattribute__(self, name):
        if name == "get_approved_session_data":
            raise RuntimeError("private repository attribute detail")
        return super().__getattribute__(name)


def _report(*, status: ReportStatus = ReportStatus.APPROVED) -> ReportDocument:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["report_status"] = status.value
    return ReportDocument.model_validate_json(json.dumps(payload))


def _shot(
    sequence: int,
    *,
    suffix: int | None = None,
    confidence: Confidence = Confidence.HIGH,
    analysis_status: ShotAnalysisStatus = ShotAnalysisStatus.ANALYZED,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
    is_foul: bool | None = False,
    potted_balls: tuple[int, ...] = (),
    deviation_mm: int | None = None,
    target_zone_source: TargetZoneSource = TargetZoneSource.NONE,
) -> ApprovedShotData:
    report = _report()
    return ApprovedShotData(
        shot_id=UUID(f"00000000-0000-4000-8000-{(suffix or sequence):012d}"),
        session_id=report.session_id,
        sequence=sequence,
        struck_ball=1,
        potted_balls=potted_balls,
        cushion_count=0,
        travel_distance_mm=500,
        is_foul=is_foul,
        deviation_mm=deviation_mm,
        target_zone_source=target_zone_source,
        confidence=confidence,
        analysis_status=analysis_status,
        review_status=review_status,
    )


def _approved(
    *shots: ApprovedShotData,
    report: ReportDocument | None = None,
    complete: bool = True,
):
    selected_report = report or _report()
    normalized = tuple(
        shot.model_copy(update={"session_id": selected_report.session_id}) for shot in shots
    )
    if complete:
        existing_sequences = {shot.sequence for shot in normalized}
        padding = tuple(
            _shot(
                sequence,
                suffix=900 + sequence,
                confidence=Confidence.MEDIUM,
                potted_balls=(sequence,),
            ).model_copy(update={"session_id": selected_report.session_id})
            for sequence in range(1, selected_report.session_statistics.total_shot_count + 1)
            if sequence not in existing_sequences
        )
        normalized += padding
    return ApprovedSessionData(
        session_id=selected_report.session_id,
        report=selected_report,
        shots=normalized,
    )


def _selection(repository: _ApprovedRepository) -> TypicalShotSelectionResult:
    data = repository.data
    return select_typical_shots(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        TypicalShotRuleVersion.V1,
    )


def _generate(data: ApprovedSessionData):
    repository = _ApprovedRepository(data)
    selection = _selection(repository)
    repository.calls.clear()
    result = generate_recommendations(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )
    return result, repository, selection


@pytest.mark.parametrize(
    ("shot", "rule_id", "claim", "practice", "kind", "title", "advice", "instruction"),
    [
        (
            _shot(1, is_foul=True),
            RecommendationRuleId.REDUCE_FOUL_RISK,
            RecommendationClaimCode.REDUCE_FOUL_RISK,
            RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
            RecommendationEvidenceKind.FOUL,
            "建议下次重点练：减少犯规",
            "先把无犯规的基础击球稳定下来。",
            "固定同一组基础直线球位重复击球，逐杆记录是否犯规。",
        ),
        (
            _shot(1),
            RecommendationRuleId.STABILIZE_POTTING,
            RecommendationClaimCode.STABILIZE_POTTING,
            RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS,
            RecommendationEvidenceKind.MISSED_POT,
            "建议下次重点练：基础进球",
            "先提高基础进球稳定性。",
            "固定同一组直线球位重复击球，逐杆记录是否进球。",
        ),
        (
            _shot(
                1,
                deviation_mm=500,
                target_zone_source=TargetZoneSource.SCENE_PRESET,
            ),
            RecommendationRuleId.CONTROL_CUE_BALL_POSITION,
            RecommendationClaimCode.CONTROL_CUE_BALL_POSITION,
            RecommendationPracticeCode.TRACK_TARGET_ZONE_FINISHES,
            RecommendationEvidenceKind.LARGE_DEVIATION,
            "建议下次重点练：母球停位",
            "先练母球停位控制。",
            "固定同一球位和目标区重复击球，逐杆记录母球是否进入目标区。",
        ),
    ],
)
def test_v1_catalog_generates_each_frozen_rule_with_traceable_evidence(
    shot, rule_id, claim, practice, kind, title, advice, instruction
):
    result, repository, selection = _generate(_approved(shot))

    assert repository.calls == [
        (result.source_lineage.session_id, result.source_lineage.report_id, 1)
    ]
    assert result.status is RecommendationStatus.READY
    assert result.reason is None
    assert result.fallback is None
    assert len(result.recommendations) == 1
    recommendation = result.recommendations[0]
    assert recommendation.semantic_core.rule_id is rule_id
    assert recommendation.semantic_core.claim_code is claim
    assert recommendation.semantic_core.practice_code is practice
    assert recommendation.title == title
    assert recommendation.primary_advice == advice
    assert recommendation.drill.instruction == instruction
    assert recommendation.evidence[0].evidence_kind is kind
    assert recommendation.evidence[0].shot_id == shot.shot_id
    assert recommendation.evidence[0].sequence == shot.sequence
    assert recommendation.evidence[0].selection_rank == selection.selections[0].selection_rank
    assert recommendation.evidence[0].selection_reason is selection.selections[0].selection_reason


def test_rank_order_selects_only_first_negative_candidate_and_deduplicates_output():
    shots = (
        _shot(1, is_foul=True),
        _shot(
            2,
            deviation_mm=700,
            target_zone_source=TargetZoneSource.POST_HOC_INFERRED,
        ),
        _shot(3),
        _shot(4, is_foul=True),
    )

    result, _repository, selection = _generate(_approved(*reversed(shots)))

    assert [item.selection_rank for item in selection.selections] == list(
        range(1, len(selection.selections) + 1)
    )
    assert len(result.recommendations) == 1
    assert result.recommendations[0].semantic_core.rule_id is RecommendationRuleId.REDUCE_FOUL_RISK
    assert [item.shot_id for item in result.recommendations[0].evidence] == [shots[0].shot_id]


def test_positive_only_returns_exact_honest_fallback_with_lineage():
    result, _repository, _selection_result = _generate(
        _approved(_shot(1, potted_balls=(3,), deviation_mm=10))
    )

    assert result.status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION
    assert result.reason is RecommendationReason.NO_NEGATIVE_RULE_MATCH
    assert result.fallback == "当前数据不足以给出训练建议"
    assert result.source_lineage is not None
    assert result.recommendations == ()


def test_low_quality_negative_only_returns_insufficient_evidence_without_padding():
    result, _repository, selection = _generate(
        _approved(_shot(1, is_foul=True, confidence=Confidence.LOW))
    )

    assert selection.selections == ()
    assert result.status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION
    assert result.reason is RecommendationReason.INSUFFICIENT_EVIDENCE
    assert result.recommendations == ()


def test_incomplete_approved_projection_is_invalid_input():
    result, _repository, selection = _generate(_approved(complete=False))

    assert selection.selections == ()
    assert result.status is RecommendationStatus.INVALID_INPUT
    assert result.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT
    assert result.recommendations == ()


def test_repository_projection_must_match_requested_lineage_exactly():
    requested = _report()
    returned = ReportDocument.model_validate(
        requested.model_dump(mode="python")
        | {
            "session_id": uuid4(),
            "report_id": uuid4(),
        }
    )
    data = _approved(_shot(1, is_foul=True), report=returned)
    repository = _ApprovedRepository(data)
    selection = _selection(repository)
    repository.calls.clear()

    result = generate_recommendations(
        repository,
        requested.session_id,
        requested.report_id,
        requested.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )

    assert result.status is RecommendationStatus.INVALID_INPUT
    assert result.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT
    assert result.source_lineage is None


@pytest.mark.parametrize(
    "data",
    [
        _approved(_shot(1, potted_balls=(1,), confidence=Confidence.MEDIUM), complete=False),
        _approved(
            _shot(1, suffix=31, potted_balls=(1,), confidence=Confidence.MEDIUM),
            _shot(2, suffix=31, potted_balls=(2,), confidence=Confidence.MEDIUM),
            _shot(3, potted_balls=(3,), confidence=Confidence.MEDIUM),
            _shot(4, potted_balls=(4,), confidence=Confidence.MEDIUM),
            complete=False,
        ),
    ],
)
def test_repository_projection_requires_complete_unique_shot_set(data):
    result, _repository, selection = _generate(data)

    assert selection.selections == ()
    assert result.status is RecommendationStatus.INVALID_INPUT
    assert result.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT
    assert result.recommendations == ()


@pytest.mark.parametrize(
    "tamper",
    [
        lambda result: result.model_copy(
            update={"session_id": UUID("40000000-0000-4000-8000-000000000004")}
        ),
        lambda result: result.model_copy(
            update={"report_id": UUID("50000000-0000-4000-8000-000000000004")}
        ),
        lambda result: result.model_copy(update={"report_revision": 2}),
        lambda result: result.model_copy(
            update={
                "selections": tuple(
                    item.model_copy(update={"selection_rank": rank})
                    for rank, item in enumerate(reversed(result.selections), start=1)
                )
            }
        ),
        lambda result: result.model_copy(
            update={
                "selections": (
                    result.selections[0].model_copy(
                        update={"shot_id": UUID("60000000-0000-4000-8000-000000000009")}
                    ),
                    result.selections[1],
                )
            }
        ),
        lambda result: result.model_copy(update={"selections": ()}),
    ],
)
def test_any_well_typed_selection_swap_returns_selection_mismatch(tamper):
    data = _approved(_shot(1, is_foul=True), _shot(2))
    repository = _ApprovedRepository(data)
    selection = _selection(repository)
    forged = tamper(selection)
    repository.calls.clear()

    result = generate_recommendations(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        forged,
        RecommendationCatalogVersion.V1,
    )

    assert result.status is RecommendationStatus.SELECTION_MISMATCH
    assert result.reason is RecommendationReason.SELECTION_RESULT_MISMATCH
    assert result.source_lineage.session_id == data.session_id
    assert result.source_lineage.report_id == data.report.report_id
    assert result.source_lineage.report_revision == data.report.revision
    assert result.recommendations == ()


def test_recomputed_selection_with_inconsistent_scalar_invariant_fails_as_mismatch():
    shot = _shot(
        1,
        is_foul=None,
        deviation_mm=500,
        target_zone_source=TargetZoneSource.SCENE_PRESET,
    )
    result, _repository, selection = _generate(_approved(shot))

    assert selection.selections
    assert result.status is RecommendationStatus.SELECTION_MISMATCH
    assert result.reason is RecommendationReason.SELECTION_RESULT_MISMATCH
    assert result.recommendations == ()


@pytest.mark.parametrize(
    "shots",
    [
        (_shot(1, deviation_mm=500, target_zone_source=TargetZoneSource.NONE),),
        (
            _shot(1, deviation_mm=500, target_zone_source=TargetZoneSource.SCENE_PRESET),
            _shot(2, deviation_mm=700, target_zone_source=TargetZoneSource.POST_HOC_INFERRED),
        ),
    ],
)
def test_unreliable_deviation_source_abstains_as_unsupported_when_no_later_negative(shots):
    result, _repository, selection = _generate(_approved(*shots))

    assert selection.selections
    assert result.status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION
    assert result.reason is RecommendationReason.UNSUPPORTED_EVIDENCE
    assert result.recommendations == ()


@pytest.mark.parametrize(
    "deviations",
    [
        (_shot(1, deviation_mm=500, target_zone_source=TargetZoneSource.NONE),),
        (
            _shot(1, deviation_mm=500, target_zone_source=TargetZoneSource.SCENE_PRESET),
            _shot(2, deviation_mm=700, target_zone_source=TargetZoneSource.POST_HOC_INFERRED),
        ),
    ],
)
def test_unreliable_deviation_is_skipped_for_next_reliable_negative(deviations):
    missed = _shot(3, is_foul=False, potted_balls=())

    result, _repository, selection = _generate(_approved(*deviations, missed))

    assert len(selection.selections) >= 2
    recommendation = result.recommendations[0]
    assert result.status is RecommendationStatus.READY
    assert recommendation.semantic_core.rule_id is RecommendationRuleId.STABILIZE_POTTING
    assert recommendation.evidence[0].shot_id == missed.shot_id


def test_single_reliable_target_zone_source_is_preserved():
    shots = (
        _shot(1, deviation_mm=500, target_zone_source=TargetZoneSource.SCENE_PRESET),
        _shot(2, deviation_mm=700, target_zone_source=TargetZoneSource.SCENE_PRESET),
    )

    result, _repository, _selection_result = _generate(_approved(*shots))

    evidence = result.recommendations[0].evidence
    assert len(evidence) == 1
    assert evidence[0].target_zone_source is TargetZoneSource.SCENE_PRESET


def test_same_normalized_inputs_are_byte_and_semantically_deterministic():
    shots = (
        _shot(1, is_foul=True),
        _shot(2),
        _shot(3, potted_balls=(5,)),
    )
    first, _repository, _selection_result = _generate(_approved(*shots))
    repeated, _repository, _selection_result = _generate(_approved(*reversed(shots)))

    assert first == repeated
    assert first.model_dump_json() == repeated.model_dump_json()
    assert first.content_sha256 == repeated.content_sha256


def test_repository_gate_error_propagates_unchanged():
    report = _report()
    failure = ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
    repository = _BlockedRepository(failure)
    selection = TypicalShotSelectionResult(
        session_id=report.session_id,
        report_id=report.report_id,
        report_revision=report.revision,
        rule_version=TypicalShotRuleVersion.V1,
        selections=(),
    )

    with pytest.raises(ReportNotDeliverableError) as raised:
        generate_recommendations(
            repository,
            report.session_id,
            report.report_id,
            report.revision,
            selection,
            RecommendationCatalogVersion.V1,
        )

    assert raised.value is failure


@pytest.mark.parametrize(
    "args",
    [
        (object(), UUID(int=1), UUID(int=2), 1, object(), RecommendationCatalogVersion.V1),
        (object(), "session", UUID(int=2), 1, object(), RecommendationCatalogVersion.V1),
        (object(), UUID(int=1), UUID(int=2), True, object(), RecommendationCatalogVersion.V1),
        (object(), UUID(int=1), UUID(int=2), 1, object(), "future"),
    ],
)
def test_non_protocol_or_non_model_malicious_input_returns_typed_invalid_input(args):
    result = generate_recommendations(*args)

    assert result.status is RecommendationStatus.INVALID_INPUT
    assert result.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT
    assert result.source_lineage is None
    assert result.recommendations == ()


@pytest.mark.parametrize("repository", [_BrokenRepository(), _BrokenAttributeRepository()])
def test_non_gate_repository_implementation_errors_do_not_escape(repository):
    data = _approved(_shot(1, is_foul=True))
    selection = TypicalShotSelectionResult(
        session_id=data.session_id,
        report_id=data.report.report_id,
        report_revision=data.report.revision,
        rule_version=TypicalShotRuleVersion.V1,
        selections=(),
    )

    result = generate_recommendations(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )

    assert result.status is RecommendationStatus.INVALID_INPUT
    assert result.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT


def test_direct_approved_projection_and_ideal_or_visualization_arguments_are_rejected():
    data = _approved(_shot(1, is_foul=True))
    selection = TypicalShotSelectionResult(
        session_id=data.session_id,
        report_id=data.report.report_id,
        report_revision=data.report.revision,
        rule_version=TypicalShotRuleVersion.V1,
        selections=(),
    )

    direct = generate_recommendations(
        data,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )
    assert direct.status is RecommendationStatus.INVALID_INPUT
    with pytest.raises(TypeError):
        generate_recommendations(
            _ApprovedRepository(data),
            data.session_id,
            data.report.report_id,
            data.report.revision,
            selection,
            RecommendationCatalogVersion.V1,
            ideal_path=object(),
        )


def test_result_contains_no_score_status_technique_rotation_or_ideal_claims():
    result, _repository, _selection_result = _generate(_approved(_shot(1, is_foul=True)))
    serialized = result.model_dump_json().lower()

    assert all(
        token not in serialized
        for token in (
            "score_status",
            "power_control",
            "rotation_application",
            "cue_technique",
            "ideal",
            "trajectory",
            "direction",
            "cushion",
        )
    )
