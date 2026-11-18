"""Story 5.5 approved-parent report composition matrix."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.engine import compose_coaching_report
from billiards_ai.cloud.review import ReportNotDeliverableError, require_deliverable_report
from billiards_ai.shared.models import (
    ApprovedSessionData,
    ApprovedShotData,
    Confidence,
    FinalFieldSource,
    RecommendationCatalogVersion,
    RecommendationResult,
    RecommendationStatus,
    ReportCompositionError,
    ReportCompositionReason,
    ReportDocument,
    ReportNotDeliverableReason,
    ReportStatus,
    ReportVisualizationResult,
    ReviewStatus,
    ReviewTaskStatus,
    Shot,
    ShotAnalysisStatus,
    TargetZoneSource,
    TypicalShotRuleVersion,
    TypicalShotSelectionReason,
    VisualizationActualProvenance,
    VisualizationAsset,
    VisualizationBall,
    VisualizationCalibrationProvenance,
    VisualizationFailureReason,
    VisualizationIdealProvenance,
    VisualizationLegendItem,
    VisualizationLineStyle,
    VisualizationPathKind,
    VisualizationPathSeries,
    VisualizationPathStyle,
    VisualizationPoint,
    VisualizationResult,
    VisualizationSelectionProvenance,
    VisualizationSemanticToken,
    VisualizationSourceLineage,
    VisualizationStatus,
    VisualizationTable,
)

FIXTURE = Path(__file__).parents[3] / "fixtures" / "reports" / "v1" / "normal.json"
PARENT_REPORT_ID = UUID("30000000-0000-4000-8000-000000000001")
CHILD_REPORT_ID = UUID("30000000-0000-4000-8000-000000000101")
CHILD_TASK_ID = UUID("40000000-0000-4000-8000-000000000101")
GENERATED_AT = datetime(2026, 7, 29, 14, 0, tzinfo=UTC)


class _Repository:
    def __init__(self, data: ApprovedSessionData):
        self.data = data
        self.calls: list[tuple[UUID, UUID, int]] = []

    def get_approved_session_data(self, session_id, report_id, revision):
        self.calls.append((session_id, report_id, revision))
        return self.data


class _BlockedRepository:
    def __init__(self, error: ReportNotDeliverableError):
        self.error = error

    def get_approved_session_data(self, session_id, report_id, revision):
        raise self.error


def _parent(total_shot_count: int = 4, *, calibration_active: bool = True) -> ReportDocument:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["report_status"] = ReportStatus.APPROVED.value
    if not calibration_active:
        payload["calibration_status"] = "INVALID"
        for field in (
            "average_cue_ball_travel_distance_mm",
            "cue_ball_travel_shot_count",
            "cue_ball_end_distribution",
            "cue_ball_end_shot_count",
            "target_zone_deviation_by_source",
        ):
            payload["session_statistics"].pop(field)
    if total_shot_count != 4:
        payload["quality_summary"] |= {
            "shot_count": total_shot_count,
            "abstained_shot_count": 0,
            "abstention_rate": 0.0,
        }
        payload["session_statistics"] |= {
            "total_shot_count": total_shot_count,
            "valid_shot_count": total_shot_count,
            "potted_ball_count": 0,
            "successful_potting_shot_count": 0,
            "potting_shot_count": total_shot_count,
            "pot_rate": 0.0,
            "cushion_distribution": [{"cushion_count": 0, "shot_count": total_shot_count}],
            "cushion_count_shot_count": total_shot_count,
            "cue_ball_travel_shot_count": total_shot_count,
            "cue_ball_end_distribution": [
                {
                    "position": {"x_mm": 800, "y_mm": 500},
                    "shot_count": total_shot_count,
                }
            ],
            "cue_ball_end_shot_count": total_shot_count,
        }
        payload["abstention_summary"] |= {
            "total_shot_count": total_shot_count,
            "display_text": "0 杆超出当前分析范围",
        }
    return ReportDocument.model_validate_json(json.dumps(payload))


def _shot(
    sequence: int,
    *,
    is_foul: bool | None = False,
    potted_balls: tuple[int, ...] = (),
    deviation_mm: int | None = None,
    target_zone_source: TargetZoneSource = TargetZoneSource.NONE,
    confidence: Confidence = Confidence.HIGH,
) -> ApprovedShotData:
    parent = _parent()
    return ApprovedShotData(
        shot_id=UUID(f"50000000-0000-4000-8000-{sequence:012d}"),
        session_id=parent.session_id,
        sequence=sequence,
        struck_ball=sequence,
        potted_balls=potted_balls,
        cushion_count=0,
        travel_distance_mm=500,
        is_foul=is_foul,
        deviation_mm=deviation_mm,
        target_zone_source=target_zone_source,
        confidence=confidence,
        analysis_status=ShotAnalysisStatus.ANALYZED,
        review_status=ReviewStatus.APPROVED,
    )


def _approved(kind: str = "complete") -> ApprovedSessionData:
    parent = _parent(
        7 if kind == "five" else 4,
        calibration_active=kind != "calibration_inactive",
    )
    if kind == "complete":
        shots = (
            _shot(1, is_foul=True),
            _shot(2, deviation_mm=500, target_zone_source=TargetZoneSource.SCENE_PRESET),
            _shot(3, confidence=Confidence.LOW),
            _shot(4, potted_balls=(4,), deviation_mm=10),
        )
    elif kind == "partial":
        shots = (
            _shot(1, is_foul=True),
            _shot(2, potted_balls=(2,), deviation_mm=10),
            _shot(3, confidence=Confidence.LOW),
            _shot(4, confidence=Confidence.LOW),
        )
    elif kind in {"one", "calibration_inactive"}:
        shots = (
            _shot(1, potted_balls=(1,), deviation_mm=10),
            *tuple(_shot(index, confidence=Confidence.LOW) for index in range(2, 5)),
        )
    elif kind == "five":
        shots = (
            *tuple(_shot(index, is_foul=True) for index in range(1, 4)),
            *tuple(
                _shot(
                    index,
                    deviation_mm=500,
                    target_zone_source=TargetZoneSource.SCENE_PRESET,
                )
                for index in range(4, 7)
            ),
            _shot(7, potted_balls=(7,), deviation_mm=10),
        )
    else:
        shots = tuple(_shot(index, confidence=Confidence.LOW) for index in range(1, 5))
    return ApprovedSessionData(session_id=parent.session_id, report=parent, shots=shots)


def _authoritative_inputs(data: ApprovedSessionData):
    from billiards_ai.cloud.engine import generate_recommendations, select_typical_shots

    repository = _Repository(data)
    selection = select_typical_shots(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        TypicalShotRuleVersion.V1,
    )
    recommendation = generate_recommendations(
        repository,
        data.session_id,
        data.report.report_id,
        data.report.revision,
        selection,
        RecommendationCatalogVersion.V1,
    )
    return selection, recommendation


def _asset(shot_id: UUID, sequence: int, reason) -> VisualizationAsset:
    actual_style = VisualizationPathStyle(
        semantic_token=VisualizationSemanticToken.ACTUAL,
        color="#1565C0",
        line_style=VisualizationLineStyle.SOLID,
        direct_label="实际母球路径",
    )
    ideal_style = VisualizationPathStyle(
        semantic_token=VisualizationSemanticToken.IDEAL,
        color="#EF6C00",
        line_style=VisualizationLineStyle.DASHED,
        direct_label="理想路径",
    )
    source = VisualizationSourceLineage(
        session_id=_parent().session_id,
        report_id=PARENT_REPORT_ID,
        report_revision=1,
        shot_id=shot_id,
    )
    values = {
        "schema_version": "1.0",
        "spec_version": "actual-ideal-overlay/1.0",
        "source": source,
        "selection": VisualizationSelectionProvenance(rule_version="1.0", selection_reason=reason),
        "table": VisualizationTable(width_mm=2540, height_mm=1270),
        "calibration": VisualizationCalibrationProvenance(
            calibration_id=UUID("60000000-0000-4000-8000-000000000001"),
            revision_id=UUID("60000000-0000-4000-8000-000000000002"),
            revision_number=1,
            table_configuration_fingerprint="b" * 64,
        ),
        "actual": VisualizationActualProvenance(source=FinalFieldSource.CV, confidence=0.9),
        "ideal": VisualizationIdealProvenance(
            adapter_version="pooltool-adapter-1.0",
            engine_name="pooltool",
            engine_version="0.6.0",
            configuration_version="chinese-eight-ball-9ft-v1",
        ),
        "balls": (
            VisualizationBall(ball_id=0, position=VisualizationPoint(x_mm=100, y_mm=100)),
            VisualizationBall(ball_id=sequence, position=VisualizationPoint(x_mm=900, y_mm=500)),
        ),
        "paths": (
            VisualizationPathSeries(
                kind=VisualizationPathKind.ACTUAL_CUE,
                ball_id=0,
                points=(
                    VisualizationPoint(x_mm=100, y_mm=100),
                    VisualizationPoint(x_mm=400, y_mm=220),
                ),
                style=actual_style,
                label="实际母球路径",
            ),
            VisualizationPathSeries(
                kind=VisualizationPathKind.IDEAL_CUE,
                ball_id=0,
                points=(
                    VisualizationPoint(x_mm=100, y_mm=100),
                    VisualizationPoint(x_mm=500, y_mm=300),
                ),
                style=ideal_style,
                label="理想母球路径",
            ),
            VisualizationPathSeries(
                kind=VisualizationPathKind.IDEAL_OBJECT,
                ball_id=sequence,
                points=(
                    VisualizationPoint(x_mm=900, y_mm=500),
                    VisualizationPoint(x_mm=2540, y_mm=1270),
                ),
                style=ideal_style,
                label="理想目标球路径",
            ),
        ),
        "legend": (
            VisualizationLegendItem(
                semantic_token=VisualizationSemanticToken.ACTUAL,
                color="#1565C0",
                line_style=VisualizationLineStyle.SOLID,
                label="实际路径",
            ),
            VisualizationLegendItem(
                semantic_token=VisualizationSemanticToken.IDEAL,
                color="#EF6C00",
                line_style=VisualizationLineStyle.DASHED,
                label="理想路径",
            ),
        ),
        "text_summary": "实际母球路径来源为 CV，置信度 90%；理想路径来源为 pooltool 0.6.0。",
    }
    normalized = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in values.items()
    }
    normalized = json.loads(
        json.dumps(normalized, default=lambda value: value.model_dump(mode="json"))
    )
    digest = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return VisualizationAsset(
        **values,
        asset_id=f"visualization-{digest}",
        content_sha256=digest,
    )


def _visualizations(selection, *, failed_ranks: set[int] = frozenset()):
    entries = []
    for item in reversed(selection.selections):
        if item.selection_rank in failed_ranks:
            result = VisualizationResult(
                status=VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
                reason=VisualizationFailureReason.IDEAL_INTENT_NOT_EXPLICIT,
            )
        else:
            result = VisualizationResult(
                status=VisualizationStatus.SUCCEEDED,
                asset=_asset(item.shot_id, item.sequence, item.selection_reason),
            )
        entries.append(
            ReportVisualizationResult(
                shot_id=item.shot_id,
                source=_visualization_source(item.shot_id),
                result=result,
            )
        )
    return tuple(entries)


def _visualization_source(shot_id: UUID) -> VisualizationSourceLineage:
    return VisualizationSourceLineage(
        session_id=_parent().session_id,
        report_id=PARENT_REPORT_ID,
        report_revision=1,
        shot_id=shot_id,
    )


def _compose(data, selection, recommendation, visualizations, **overrides):
    values = {
        "repository": _Repository(data),
        "session_id": data.session_id,
        "parent_report_id": data.report.report_id,
        "parent_revision": data.report.revision,
        "child_report_id": CHILD_REPORT_ID,
        "child_review_task_id": CHILD_TASK_ID,
        "generated_at": GENERATED_AT,
        "selection_result": selection,
        "visualization_results": visualizations,
        "recommendation_result": recommendation,
    }
    values.update(overrides)
    return compose_coaching_report(**values)


def test_complete_composition_creates_deterministic_v11_child_and_pending_review_task():
    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)
    visualizations = _visualizations(selection)
    snapshots = tuple(
        json.dumps(value.model_dump(mode="json"), sort_keys=True)
        for value in (data.report, selection, recommendation, *visualizations)
    )

    first = _compose(data, selection, recommendation, visualizations)
    second = _compose(data, selection, recommendation, tuple(reversed(visualizations)))

    assert first == second
    assert first.report.schema_version == "1.1"
    assert first.report.report_id == CHILD_REPORT_ID
    assert first.report.revision == 2
    assert first.report.parent_revision_id == PARENT_REPORT_ID
    assert first.report.report_status is ReportStatus.DRAFT_PENDING_REVIEW
    assert first.review_task.id == CHILD_TASK_ID
    assert first.review_task.status is ReviewTaskStatus.PENDING_REVIEW
    assert first.review_task.report_id == CHILD_REPORT_ID
    assert first.review_task.report_revision == 2
    assert first.report.typical_shots == ()
    assert first.report.coaching_recommendations == ()
    assert [card.shot_id for card in first.report.typical_shot_analysis.cards] == [
        item.shot_id for item in selection.selections
    ]
    assert first.report.typical_shot_analysis.status.value == "READY"
    assert first.report.training_recommendation.status is RecommendationStatus.READY
    assert all(
        card.asset_reference.source_approved_lineage.report_id == PARENT_REPORT_ID
        and card.asset_reference.target_report_lineage.report_id == CHILD_REPORT_ID
        for card in first.report.typical_shot_analysis.cards
    )
    assert first.report.content_sha256 == second.report.content_sha256
    assert (
        tuple(
            json.dumps(value.model_dump(mode="json"), sort_keys=True)
            for value in (data.report, selection, recommendation, *visualizations)
        )
        == snapshots
    )


@pytest.mark.parametrize(
    ("kind", "failed", "typical_status", "recommendation_status"),
    [
        ("partial", {2}, "PARTIAL", RecommendationStatus.READY),
        ("none", set(), "NO_RELIABLE_DATA", RecommendationStatus.NO_RELIABLE_RECOMMENDATION),
    ],
)
def test_normal_no_data_is_a_valid_child_not_an_integrity_error(
    kind, failed, typical_status, recommendation_status
):
    data = _approved(kind)
    selection, recommendation = _authoritative_inputs(data)
    result = _compose(
        data, selection, recommendation, _visualizations(selection, failed_ranks=failed)
    )

    assert result.report.typical_shot_analysis.status.value == typical_status
    assert result.report.training_recommendation.status is recommendation_status
    if not selection.selections:
        assert result.report.typical_shot_analysis.cards == ()
        assert result.report.training_recommendation.fallback == "当前数据不足以给出训练建议"


@pytest.mark.parametrize(("kind", "expected_count"), [("one", 1), ("five", 5)])
def test_composition_supports_one_and_maximum_five_selected_shots(kind, expected_count):
    data = _approved(kind)
    selection, recommendation = _authoritative_inputs(data)
    result = _compose(data, selection, recommendation, _visualizations(selection))

    assert len(selection.selections) == expected_count
    assert len(result.report.typical_shot_analysis.cards) == expected_count
    assert result.report.typical_shot_analysis.status.value == "READY"


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            VisualizationStatus.CALIBRATION_NOT_ACTIVE,
            VisualizationFailureReason.CALIBRATION_NOT_ACTIVE,
        ),
        (
            VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
            VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED,
        ),
        (
            VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
            VisualizationFailureReason.ACTUAL_PATH_INSUFFICIENT_POINTS,
        ),
        (
            VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
            VisualizationFailureReason.ACTUAL_PATH_LOW_CONFIDENCE,
        ),
        (
            VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
            VisualizationFailureReason.ACTUAL_PATH_NOT_REVIEWED,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_PATH_NOT_SUCCEEDED,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_INTENT_NOT_EXPLICIT,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_INTENT_NOT_REVIEWED,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_CALIBRATION_NOT_ACTIVE,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_TARGET_BALL_MISSING,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_ENGINE_UNAVAILABLE,
        ),
        (
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            VisualizationFailureReason.IDEAL_ADAPTER_FAILURE,
        ),
    ],
)
def test_all_visualization_capability_failures_remain_normal_partial_content(status, reason):
    data = _approved(
        "calibration_inactive" if status is VisualizationStatus.CALIBRATION_NOT_ACTIVE else "one"
    )
    selection, recommendation = _authoritative_inputs(data)
    visualization = ReportVisualizationResult(
        shot_id=selection.selections[0].shot_id,
        source=_visualization_source(selection.selections[0].shot_id),
        result=VisualizationResult(status=status, reason=reason),
    )

    result = _compose(data, selection, recommendation, (visualization,))

    assert result.report.typical_shot_analysis.status.value == "PARTIAL"
    assert result.report.typical_shot_analysis.cards[0].asset_reference is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_visualization_results_must_map_one_to_one_to_selection(mutation):
    data = _approved("partial")
    selection, recommendation = _authoritative_inputs(data)
    results = list(_visualizations(selection))
    if mutation == "missing":
        results.pop()
    elif mutation == "duplicate":
        results.append(results[0])
    else:
        results.append(
            ReportVisualizationResult(
                shot_id=UUID("50000000-0000-4000-8000-000000000099"),
                source=_visualization_source(UUID("50000000-0000-4000-8000-000000000099")),
                result=VisualizationResult(
                    status=VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
                    reason=VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED,
                ),
            )
        )

    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, recommendation, tuple(results))
    assert raised.value.reason is ReportCompositionReason.RESULT_SET_MISMATCH


def test_selection_and_recommendation_are_authoritatively_recomputed_and_compared():
    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)
    swapped = selection.model_copy(
        update={
            "selections": selection.selections[:1]
            + (selection.selections[1].model_copy(update={"sequence": 3}),)
            + selection.selections[2:]
        }
    )
    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, swapped, recommendation, _visualizations(selection))
    assert raised.value.reason is ReportCompositionReason.SELECTION_MISMATCH

    bad_recommendation = RecommendationResult.model_construct(
        **(recommendation.model_dump(mode="python") | {"content_sha256": "0" * 64})
    )
    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, bad_recommendation, _visualizations(selection))
    assert raised.value.reason is ReportCompositionReason.RECOMMENDATION_MISMATCH


def test_success_asset_requires_recursive_hash_and_exact_parent_binding():
    data = _approved("partial")
    selection, recommendation = _authoritative_inputs(data)
    entries = list(_visualizations(selection))
    asset = entries[0].result.asset
    tampered_asset = asset.model_construct(**(dict(vars(asset)) | {"content_sha256": "0" * 64}))
    entries[0] = ReportVisualizationResult.model_construct(
        shot_id=entries[0].shot_id,
        source=entries[0].source,
        result=VisualizationResult.model_construct(
            status=VisualizationStatus.SUCCEEDED,
            reason=None,
            asset=tampered_asset,
            schema_version="1.0",
        ),
    )

    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, recommendation, tuple(entries))
    assert raised.value.reason is ReportCompositionReason.ASSET_INTEGRITY_MISMATCH


def test_visualization_result_requires_explicit_approved_source_lineage():
    data = _approved("one")
    selection, _recommendation = _authoritative_inputs(data)
    selected = selection.selections[0]

    with pytest.raises(ValidationError):
        ReportVisualizationResult(
            shot_id=selected.shot_id,
            result=VisualizationResult(
                status=VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
                reason=VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED,
            ),
        )


def test_success_asset_selection_provenance_must_match_authoritative_selection():
    data = _approved("partial")
    selection, recommendation = _authoritative_inputs(data)
    entries = list(_visualizations(selection))
    selected = selection.selections[0]
    wrong_reason = (
        TypicalShotSelectionReason.POSITIVE_POTTING
        if selected.selection_reason is not TypicalShotSelectionReason.POSITIVE_POTTING
        else TypicalShotSelectionReason.SEVERE_FOUL
    )
    entry_index = next(
        index for index, entry in enumerate(entries) if entry.shot_id == selected.shot_id
    )
    entries[entry_index] = ReportVisualizationResult(
        shot_id=selected.shot_id,
        source=_visualization_source(selected.shot_id),
        result=VisualizationResult(
            status=VisualizationStatus.SUCCEEDED,
            asset=_asset(selected.shot_id, selected.sequence, wrong_reason),
        ),
    )

    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, recommendation, tuple(entries))
    assert raised.value.reason is ReportCompositionReason.ASSET_INTEGRITY_MISMATCH


def test_visualization_source_lineage_and_parent_calibration_are_authoritative():
    data = _approved("one")
    selection, recommendation = _authoritative_inputs(data)
    selected = selection.selections[0]
    wrong_source = VisualizationSourceLineage(
        session_id=data.session_id,
        report_id=UUID("30000000-0000-4000-8000-000000000099"),
        report_revision=data.report.revision,
        shot_id=selected.shot_id,
    )
    wrong_lineage = ReportVisualizationResult(
        shot_id=selected.shot_id,
        source=wrong_source,
        result=VisualizationResult(
            status=VisualizationStatus.ACTUAL_PATH_UNAVAILABLE,
            reason=VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED,
        ),
    )
    with pytest.raises(ReportCompositionError) as lineage_error:
        _compose(data, selection, recommendation, (wrong_lineage,))
    assert lineage_error.value.reason is ReportCompositionReason.ASSET_INTEGRITY_MISMATCH

    contradictory = ReportVisualizationResult(
        shot_id=selected.shot_id,
        source=VisualizationSourceLineage(
            session_id=data.session_id,
            report_id=data.report.report_id,
            report_revision=data.report.revision,
            shot_id=selected.shot_id,
        ),
        result=VisualizationResult(
            status=VisualizationStatus.CALIBRATION_NOT_ACTIVE,
            reason=VisualizationFailureReason.CALIBRATION_NOT_ACTIVE,
        ),
    )
    with pytest.raises(ReportCompositionError) as calibration_error:
        _compose(data, selection, recommendation, (contradictory,))
    assert calibration_error.value.reason is ReportCompositionReason.ASSET_INTEGRITY_MISMATCH


@pytest.mark.parametrize(
    "status,reason",
    [
        (VisualizationStatus.SOURCE_NOT_APPROVED, VisualizationFailureReason.SHOT_NOT_SELECTED),
        (
            VisualizationStatus.PROVENANCE_MISMATCH,
            VisualizationFailureReason.IDEAL_SOURCE_MISMATCH,
        ),
        (
            VisualizationStatus.REVISION_MISMATCH,
            VisualizationFailureReason.REPORT_REVISION_MISMATCH,
        ),
        (
            VisualizationStatus.SELECTION_MISMATCH,
            VisualizationFailureReason.SELECTION_LINEAGE_MISMATCH,
        ),
        (
            VisualizationStatus.INVALID_INPUT,
            VisualizationFailureReason.INVALID_IDENTIFIER_OR_REVISION,
        ),
    ],
)
def test_integrity_visualization_failures_cannot_be_downgraded_to_no_data(status, reason):
    data = _approved("partial")
    selection, recommendation = _authoritative_inputs(data)
    entries = list(_visualizations(selection))
    entries[0] = ReportVisualizationResult(
        shot_id=entries[0].shot_id,
        source=entries[0].source,
        result=VisualizationResult(status=status, reason=reason),
    )
    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, recommendation, tuple(entries))
    assert raised.value.reason is ReportCompositionReason.ASSET_INTEGRITY_MISMATCH


def test_repository_gate_error_is_propagated_and_validation_finishes_before_construction():
    from billiards_ai.shared.models import ReportNotDeliverableReason

    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)
    error = ReportNotDeliverableError(ReportNotDeliverableReason.REVISION_MISMATCH)
    with pytest.raises(ReportNotDeliverableError) as raised:
        _compose(
            data,
            selection,
            recommendation,
            _visualizations(selection),
            repository=_BlockedRepository(error),
        )
    assert raised.value is error


@pytest.mark.parametrize(
    ("module_name", "attribute", "error"),
    [
        (
            "billiards_ai.cloud.engine.typical_shot_selector",
            "_select_typical_shots",
            ValueError("selector boundary"),
        ),
        (
            "billiards_ai.cloud.engine.rules",
            "generate_recommendations",
            TypeError("recommendation boundary"),
        ),
    ],
)
def test_authoritative_recomputation_errors_use_closed_composition_contract(
    monkeypatch, module_name, attribute, error
):
    import importlib

    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(importlib.import_module(module_name), attribute, fail)
    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, selection, recommendation, _visualizations(selection))
    assert raised.value.reason is ReportCompositionReason.INVALID_INPUT


def test_composer_uses_one_gated_parent_snapshot_for_all_authoritative_recomputation():
    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)
    repository = _Repository(data)

    _compose(
        data,
        selection,
        recommendation,
        _visualizations(selection),
        repository=repository,
    )

    assert repository.calls == [(data.session_id, data.report.report_id, data.report.revision)]


def test_child_ids_time_and_nested_malicious_models_fail_closed():
    data = _approved("complete")
    selection, recommendation = _authoritative_inputs(data)
    with pytest.raises(ReportCompositionError) as raised:
        _compose(
            data,
            selection,
            recommendation,
            _visualizations(selection),
            child_report_id=PARENT_REPORT_ID,
        )
    assert raised.value.reason is ReportCompositionReason.INVALID_INPUT

    with pytest.raises(ReportCompositionError) as raised:
        _compose(
            data,
            selection,
            recommendation,
            _visualizations(selection),
            generated_at=data.report.generated_at - timedelta(seconds=1),
        )
    assert raised.value.reason is ReportCompositionReason.INVALID_INPUT

    malicious_item = selection.selections[0].model_construct(
        **(selection.selections[0].model_dump(mode="python") | {"sequence": "1"})
    )
    malicious = selection.model_construct(
        **(selection.model_dump(mode="python") | {"selections": (malicious_item,)})
    )
    with pytest.raises(ReportCompositionError) as raised:
        _compose(data, malicious, recommendation, _visualizations(selection))
    assert raised.value.reason is ReportCompositionReason.INVALID_INPUT


def test_report_digest_excludes_status_but_covers_content_and_lineage():
    data = _approved("none")
    selection, recommendation = _authoritative_inputs(data)
    report = _compose(data, selection, recommendation, ()).report
    approved = ReportDocument.model_validate(
        report.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    )
    assert approved.content_sha256 == report.content_sha256

    payload = report.model_dump(mode="python")
    payload["generated_at"] = report.generated_at + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="digest"):
        ReportDocument.model_validate(payload)


def test_child_report_uses_existing_epic4_delivery_gate_for_review_handoff():
    data = _approved("none")
    selection, recommendation = _authoritative_inputs(data)
    result = _compose(data, selection, recommendation, ())
    fake_shots = tuple(
        Shot(
            id=UUID(f"70000000-0000-4000-8000-{sequence:012d}"),
            session_id=data.session_id,
            sequence=sequence,
            t_start_ms=(sequence - 1) * 1000,
            t_end_ms=sequence * 1000,
            ball_positions_before=None,
            ball_positions_after=None,
            cue_ball_end=None,
            confidence=Confidence.HIGH,
            review_status=ReviewStatus.APPROVED,
            raw_observation={"source": "epic4-gate-only"},
            created_at=GENERATED_AT,
        )
        for sequence in range(1, data.report.session_statistics.total_shot_count + 1)
    )

    with pytest.raises(ReportNotDeliverableError) as pending:
        require_deliverable_report(result.report, result.review_task, fake_shots)
    assert pending.value.reason is ReportNotDeliverableReason.REVIEW_PENDING

    approved_report = ReportDocument.model_validate(
        result.report.model_dump(mode="python") | {"report_status": ReportStatus.APPROVED}
    )
    approved_task = result.review_task.model_copy(update={"status": ReviewTaskStatus.APPROVED})
    with pytest.raises(ReportNotDeliverableError) as mismatched_shots:
        require_deliverable_report(approved_report, approved_task, fake_shots)
    assert mismatched_shots.value.reason is ReportNotDeliverableReason.SHOT_SET_INCOMPLETE

    exact_shots = tuple(
        Shot(
            id=approved_shot.shot_id,
            session_id=data.session_id,
            sequence=approved_shot.sequence,
            t_start_ms=(approved_shot.sequence - 1) * 1000,
            t_end_ms=approved_shot.sequence * 1000,
            ball_positions_before=None,
            ball_positions_after=None,
            cue_ball_end=None,
            confidence=Confidence.HIGH,
            review_status=ReviewStatus.APPROVED,
            raw_observation={"source": "epic4-gate-only"},
            created_at=GENERATED_AT,
        )
        for approved_shot in data.shots
    )
    assert (
        require_deliverable_report(approved_report, approved_task, exact_shots) is approved_report
    )


def test_report_composer_static_boundary_excludes_raw_geometry_media_and_persistence():
    source = (
        Path(__file__).parents[4] / "src" / "billiards_ai" / "cloud" / "engine" / "reporting.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "raw_observation",
        ".trajectory",
        "ball_positions",
        "pooltool",
        "media_locator",
        "http://",
        "https://",
        ".save(",
        ".add(",
        "uuid4(",
        "datetime.now(",
    )
    assert not any(token in source for token in forbidden)
