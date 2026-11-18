"""Pure approved-projection to deterministic vector-overlay generation."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError

from billiards_ai.shared.models.calibration import CalibrationStatus
from billiards_ai.shared.models.ideal_path import (
    IdealPathEvent,
    IdealPathProvenance,
    IdealPathReason,
    IdealPathResult,
    IdealPathStatus,
)
from billiards_ai.shared.models.review import (
    ApprovedActualPathReason,
    ApprovedActualPathStatus,
    ApprovedVisualizationProjection,
)
from billiards_ai.shared.models.typical_shot import (
    TypicalShotSelection,
    TypicalShotSelectionResult,
)
from billiards_ai.shared.models.visualization import (
    VisualizationActualProvenance,
    VisualizationAsset,
    VisualizationAssetBinding,
    VisualizationAssetBindingResult,
    VisualizationBall,
    VisualizationBindingReason,
    VisualizationBindingStatus,
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


class VisualizationProjectionRepository(Protocol):
    def get_approved_visualization_projection(
        self,
        session_id: UUID,
        report_id: UUID,
        revision: int,
        shot_id: UUID,
    ) -> ApprovedVisualizationProjection: ...


class _MillimetrePoint(Protocol):
    x_mm: int
    y_mm: int


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    if not isinstance(value, BaseModel):
        raise TypeError("visualization inputs must be Pydantic models")
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _failure(
    status: VisualizationStatus,
    reason: VisualizationFailureReason,
) -> VisualizationResult:
    return VisualizationResult(status=status, reason=reason)


def _point(point: _MillimetrePoint) -> VisualizationPoint:
    return VisualizationPoint(
        x_mm=point.x_mm,
        y_mm=point.y_mm,
    )


def _deduplicate(points: tuple[VisualizationPoint, ...]) -> tuple[VisualizationPoint, ...]:
    result: list[VisualizationPoint] = []
    for point in points:
        if not result or result[-1] != point:
            result.append(point)
    return tuple(result)


def _ideal_series_points(
    provenance: IdealPathProvenance,
    events: tuple[IdealPathEvent, ...],
) -> tuple[tuple[VisualizationPoint, ...], tuple[VisualizationPoint, ...]]:
    target = next(
        ball for ball in provenance.object_balls if ball.ball_id == provenance.target_ball_id
    )
    cue_points = (_point(provenance.cue_ball.position),) + tuple(
        _point(event.position) for event in events if event.ball_id == 0
    )
    target_points = (_point(target.position),) + tuple(
        _point(event.position) for event in events if event.ball_id == provenance.target_ball_id
    )
    return _deduplicate(cue_points), _deduplicate(target_points)


def _provenance_matches(
    projection: ApprovedVisualizationProjection,
    ideal: IdealPathResult,
) -> bool:
    provenance = ideal.provenance
    if provenance is None:
        return False
    if (
        provenance.calibration_id != projection.calibration_id
        or provenance.calibration_revision_id != projection.calibration_revision_id
        or provenance.calibration_revision_number != projection.calibration_revision_number
        or provenance.table_profile_id != projection.table_profile_id
        or provenance.table_configuration_fingerprint != projection.table_configuration_fingerprint
        or provenance.target_ball_id != projection.struck_ball
    ):
        return False
    projected = {item.ball_id: item.position for item in projection.ball_positions}
    ideal_positions = {
        provenance.cue_ball.ball_id: provenance.cue_ball.position,
        **{ball.ball_id: ball.position for ball in provenance.object_balls},
    }
    if set(projected) != set(ideal_positions):
        return False
    return all(
        (projected[ball_id].x_mm, projected[ball_id].y_mm)
        == (ideal_positions[ball_id].x_mm, ideal_positions[ball_id].y_mm)
        for ball_id in projected
    )


def _content_payload(
    *,
    source: VisualizationSourceLineage,
    selection: VisualizationSelectionProvenance,
    table: VisualizationTable,
    calibration: VisualizationCalibrationProvenance,
    actual: VisualizationActualProvenance,
    ideal: VisualizationIdealProvenance,
    balls: tuple[VisualizationBall, ...],
    paths: tuple[VisualizationPathSeries, ...],
    legend: tuple[VisualizationLegendItem, ...],
    text_summary: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "spec_version": "actual-ideal-overlay/1.0",
        "source": source.model_dump(mode="json"),
        "selection": selection.model_dump(mode="json"),
        "table": table.model_dump(mode="json"),
        "calibration": calibration.model_dump(mode="json"),
        "actual": actual.model_dump(mode="json"),
        "ideal": ideal.model_dump(mode="json"),
        "balls": [item.model_dump(mode="json") for item in balls],
        "paths": [item.model_dump(mode="json") for item in paths],
        "legend": [item.model_dump(mode="json") for item in legend],
        "text_summary": text_summary,
    }


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_visualization_asset_bytes(asset: VisualizationAsset) -> bytes:
    validated = VisualizationAsset.model_validate(_stored_model_values(asset))
    return _canonical_bytes(validated.model_dump(mode="json"))


def _build_asset(
    projection: ApprovedVisualizationProjection,
    selected: TypicalShotSelection,
    ideal_result: IdealPathResult,
) -> VisualizationAsset | None:
    assert projection.actual_path.source is not None
    assert projection.actual_path.confidence is not None
    assert ideal_result.provenance is not None
    cue_points, target_points = _ideal_series_points(
        ideal_result.provenance,
        ideal_result.events,
    )
    if len(set(cue_points)) < 2 or len(set(target_points)) < 2:
        return None

    source = VisualizationSourceLineage(
        session_id=projection.session_id,
        report_id=projection.report_id,
        report_revision=projection.report_revision,
        shot_id=projection.shot_id,
    )
    selection = VisualizationSelectionProvenance(
        rule_version=selected.rule_version.value,
        selection_reason=selected.selection_reason,
    )
    table = VisualizationTable(
        width_mm=projection.table_length_mm,
        height_mm=projection.table_width_mm,
    )
    calibration = VisualizationCalibrationProvenance(
        calibration_id=projection.calibration_id,
        revision_id=projection.calibration_revision_id,
        revision_number=projection.calibration_revision_number,
        table_configuration_fingerprint=projection.table_configuration_fingerprint,
    )
    actual = VisualizationActualProvenance(
        source=projection.actual_path.source,
        confidence=projection.actual_path.confidence,
    )
    ideal = VisualizationIdealProvenance(
        adapter_version=ideal_result.adapter_version,
        engine_name=ideal_result.engine_name,
        engine_version=ideal_result.engine_version,
        configuration_version=ideal_result.source_configuration_version,
    )
    balls = tuple(
        VisualizationBall(ball_id=item.ball_id, position=_point(item.position))
        for item in projection.ball_positions
    )
    actual_style = VisualizationPathStyle(
        semantic_token=VisualizationSemanticToken.ACTUAL,
        color="#1565C0",
        line_style=VisualizationLineStyle.SOLID,
        direct_label="\u5b9e\u9645\u6bcd\u7403\u8def\u5f84",
    )
    ideal_style = VisualizationPathStyle(
        semantic_token=VisualizationSemanticToken.IDEAL,
        color="#C62828",
        line_style=VisualizationLineStyle.DASHED,
        direct_label="\u7406\u60f3\u8def\u5f84",
    )
    paths = (
        VisualizationPathSeries(
            kind=VisualizationPathKind.ACTUAL_CUE,
            ball_id=0,
            points=tuple(_point(point) for point in projection.actual_path.points),
            style=actual_style,
            label="\u5b9e\u9645\u6bcd\u7403\u8def\u5f84",
        ),
        VisualizationPathSeries(
            kind=VisualizationPathKind.IDEAL_CUE,
            ball_id=0,
            points=cue_points,
            style=ideal_style,
            label="\u7406\u60f3\u6bcd\u7403\u8def\u5f84",
        ),
        VisualizationPathSeries(
            kind=VisualizationPathKind.IDEAL_OBJECT,
            ball_id=ideal_result.provenance.target_ball_id,
            points=target_points,
            style=ideal_style,
            label="\u7406\u60f3\u76ee\u6807\u7403\u8def\u5f84",
        ),
    )
    legend = (
        VisualizationLegendItem(
            semantic_token=VisualizationSemanticToken.ACTUAL,
            color=actual_style.color,
            line_style=actual_style.line_style,
            label="\u5b9e\u9645\u8def\u5f84",
        ),
        VisualizationLegendItem(
            semantic_token=VisualizationSemanticToken.IDEAL,
            color=ideal_style.color,
            line_style=ideal_style.line_style,
            label="\u7406\u60f3\u8def\u5f84",
        ),
    )
    confidence_percent = int(round(actual.confidence * 100))
    text_summary = (
        f"实际母球路径来源为 {actual.source.value}，置信度 {confidence_percent}%；"
        f"理想路径来源为 {ideal.engine_name} {ideal.engine_version}。"
    )
    payload = _content_payload(
        source=source,
        selection=selection,
        table=table,
        calibration=calibration,
        actual=actual,
        ideal=ideal,
        balls=balls,
        paths=paths,
        legend=legend,
        text_summary=text_summary,
    )
    content_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return VisualizationAsset(
        asset_id=f"visualization-{content_sha256}",
        content_sha256=content_sha256,
        source=source,
        selection=selection,
        table=table,
        calibration=calibration,
        actual=actual,
        ideal=ideal,
        balls=balls,
        paths=paths,
        legend=legend,
        text_summary=text_summary,
    )


def generate_visualization_asset(
    *,
    repository: VisualizationProjectionRepository,
    session_id: UUID,
    report_id: UUID,
    revision: int,
    shot_id: UUID,
    selection: TypicalShotSelectionResult,
    ideal_result: IdealPathResult,
) -> VisualizationResult:
    """Generate one asset from the repository's current approved projection only."""

    if (
        not isinstance(session_id, UUID)
        or not isinstance(report_id, UUID)
        or not isinstance(shot_id, UUID)
        or type(revision) is not int
        or revision < 1
    ):
        return _failure(
            VisualizationStatus.INVALID_INPUT,
            VisualizationFailureReason.INVALID_IDENTIFIER_OR_REVISION,
        )
    getter = getattr(repository, "get_approved_visualization_projection", None)
    if not callable(getter):
        return _failure(
            VisualizationStatus.INVALID_INPUT,
            VisualizationFailureReason.INVALID_IDENTIFIER_OR_REVISION,
        )
    try:
        validated_selection = TypicalShotSelectionResult.model_validate(
            _stored_model_values(selection)
        )
        validated_ideal = IdealPathResult.model_validate(_stored_model_values(ideal_result))
    except (TypeError, ValidationError, ValueError):
        return _failure(
            VisualizationStatus.INVALID_INPUT,
            VisualizationFailureReason.INVALID_IDENTIFIER_OR_REVISION,
        )

    raw_projection = getter(session_id, report_id, revision, shot_id)
    try:
        projection = ApprovedVisualizationProjection.model_validate(
            _stored_model_values(raw_projection)
        )
    except (TypeError, ValidationError, ValueError):
        return _failure(
            VisualizationStatus.INVALID_INPUT,
            VisualizationFailureReason.INVALID_IDENTIFIER_OR_REVISION,
        )
    if projection.report_revision != revision:
        return _failure(
            VisualizationStatus.REVISION_MISMATCH,
            VisualizationFailureReason.REPORT_REVISION_MISMATCH,
        )
    if (
        projection.session_id != session_id
        or projection.report_id != report_id
        or projection.shot_id != shot_id
    ):
        return _failure(
            VisualizationStatus.PROVENANCE_MISMATCH,
            VisualizationFailureReason.IDEAL_SOURCE_MISMATCH,
        )
    if projection.calibration_status is not CalibrationStatus.ACTIVE:
        return _failure(
            VisualizationStatus.CALIBRATION_NOT_ACTIVE,
            VisualizationFailureReason.CALIBRATION_NOT_ACTIVE,
        )
    if projection.actual_path.status is not ApprovedActualPathStatus.AVAILABLE:
        if projection.actual_path.status is ApprovedActualPathStatus.NOT_REVIEWED:
            return _failure(
                VisualizationStatus.SOURCE_NOT_APPROVED,
                VisualizationFailureReason.ACTUAL_PATH_NOT_REVIEWED,
            )
        reason = {
            ApprovedActualPathReason.LOW_CONFIDENCE: (
                VisualizationFailureReason.ACTUAL_PATH_LOW_CONFIDENCE
            ),
            ApprovedActualPathReason.INSUFFICIENT_POINTS: (
                VisualizationFailureReason.ACTUAL_PATH_INSUFFICIENT_POINTS
            ),
            ApprovedActualPathReason.NOT_OBSERVED: (
                VisualizationFailureReason.ACTUAL_PATH_NOT_OBSERVED
            ),
            ApprovedActualPathReason.NOT_REVIEWED: (
                VisualizationFailureReason.ACTUAL_PATH_NOT_REVIEWED
            ),
        }[projection.actual_path.reason]
        return _failure(VisualizationStatus.ACTUAL_PATH_UNAVAILABLE, reason)
    if (
        validated_selection.session_id != session_id
        or validated_selection.report_id != report_id
        or validated_selection.report_revision != revision
    ):
        return _failure(
            VisualizationStatus.SELECTION_MISMATCH,
            VisualizationFailureReason.SELECTION_LINEAGE_MISMATCH,
        )
    selected = next(
        (item for item in validated_selection.selections if item.shot_id == shot_id),
        None,
    )
    if selected is None or selected.sequence != projection.shot_sequence:
        return _failure(
            VisualizationStatus.SELECTION_MISMATCH,
            VisualizationFailureReason.SHOT_NOT_SELECTED,
        )
    if validated_ideal.status is not IdealPathStatus.SUCCEEDED:
        ideal_reason = {
            IdealPathReason.INTENT_NOT_EXPLICIT: (
                VisualizationFailureReason.IDEAL_INTENT_NOT_EXPLICIT
            ),
            IdealPathReason.INTENT_NOT_REVIEWED: (
                VisualizationFailureReason.IDEAL_INTENT_NOT_REVIEWED
            ),
            IdealPathReason.CALIBRATION_NOT_ACTIVE: (
                VisualizationFailureReason.IDEAL_CALIBRATION_NOT_ACTIVE
            ),
            IdealPathReason.TARGET_BALL_MISSING: (
                VisualizationFailureReason.IDEAL_TARGET_BALL_MISSING
            ),
            IdealPathReason.ENGINE_UNAVAILABLE: (
                VisualizationFailureReason.IDEAL_ENGINE_UNAVAILABLE
            ),
            IdealPathReason.ADAPTER_FAILURE: (VisualizationFailureReason.IDEAL_ADAPTER_FAILURE),
        }.get(
            validated_ideal.reason,
            VisualizationFailureReason.IDEAL_PATH_NOT_SUCCEEDED,
        )
        return _failure(
            VisualizationStatus.IDEAL_PATH_UNSUPPORTED,
            ideal_reason,
        )
    if not _provenance_matches(projection, validated_ideal):
        return _failure(
            VisualizationStatus.PROVENANCE_MISMATCH,
            VisualizationFailureReason.IDEAL_SOURCE_MISMATCH,
        )
    asset = _build_asset(projection, selected, validated_ideal)
    if asset is None:
        return _failure(
            VisualizationStatus.PROVENANCE_MISMATCH,
            VisualizationFailureReason.IDEAL_SOURCE_MISMATCH,
        )
    return VisualizationResult(status=VisualizationStatus.SUCCEEDED, asset=asset)


def bind_visualization_asset(
    asset: VisualizationAsset,
    *,
    session_id: UUID,
    report_id: UUID,
    revision: int,
    shot_id: UUID,
) -> VisualizationAssetBindingResult:
    """Purely bind an asset to the exact current report revision and shot."""

    if (
        not isinstance(session_id, UUID)
        or not isinstance(report_id, UUID)
        or not isinstance(shot_id, UUID)
        or type(revision) is not int
        or revision < 1
    ):
        return VisualizationAssetBindingResult(
            status=VisualizationBindingStatus.INVALID_INPUT,
            reason=VisualizationBindingReason.INVALID_IDENTIFIER_OR_REVISION,
        )
    try:
        validated = VisualizationAsset.model_validate(_stored_model_values(asset))
    except (TypeError, ValidationError, ValueError):
        return VisualizationAssetBindingResult(
            status=VisualizationBindingStatus.INVALID_INPUT,
            reason=VisualizationBindingReason.INVALID_IDENTIFIER_OR_REVISION,
        )
    if validated.source.report_revision != revision:
        return VisualizationAssetBindingResult(
            status=VisualizationBindingStatus.REVISION_MISMATCH,
            reason=VisualizationBindingReason.REPORT_REVISION_MISMATCH,
        )
    if (
        validated.source.session_id != session_id
        or validated.source.report_id != report_id
        or validated.source.shot_id != shot_id
    ):
        return VisualizationAssetBindingResult(
            status=VisualizationBindingStatus.SOURCE_MISMATCH,
            reason=VisualizationBindingReason.SOURCE_LINEAGE_MISMATCH,
        )
    binding = VisualizationAssetBinding(
        asset_id=validated.asset_id,
        asset_content_sha256=validated.content_sha256,
        source=validated.source,
        target_session_id=session_id,
        target_report_id=report_id,
        target_report_revision=revision,
        target_shot_id=shot_id,
    )
    return VisualizationAssetBindingResult(
        status=VisualizationBindingStatus.SUCCEEDED,
        binding=binding,
    )
