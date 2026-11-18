"""Strict deterministic vector-overlay contracts for Story 5.3."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from billiards_ai.shared.models.ideal_path import SafeMetadataToken
from billiards_ai.shared.models.quality import FinalFieldSource
from billiards_ai.shared.models.typical_shot import TypicalShotSelectionReason

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Color = Annotated[str, Field(pattern=r"^#[0-9A-F]{6}$")]


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class VisualizationStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_model_values(copied))


class VisualizationStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    CALIBRATION_NOT_ACTIVE = "CALIBRATION_NOT_ACTIVE"
    ACTUAL_PATH_UNAVAILABLE = "ACTUAL_PATH_UNAVAILABLE"
    IDEAL_PATH_UNSUPPORTED = "IDEAL_PATH_UNSUPPORTED"
    SOURCE_NOT_APPROVED = "SOURCE_NOT_APPROVED"
    SELECTION_MISMATCH = "SELECTION_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


class VisualizationFailureReason(StrEnum):
    CALIBRATION_NOT_ACTIVE = "calibration_not_active"
    ACTUAL_PATH_NOT_OBSERVED = "actual_path_not_observed"
    ACTUAL_PATH_INSUFFICIENT_POINTS = "actual_path_insufficient_points"
    ACTUAL_PATH_LOW_CONFIDENCE = "actual_path_low_confidence"
    ACTUAL_PATH_NOT_REVIEWED = "actual_path_not_reviewed"
    IDEAL_PATH_NOT_SUCCEEDED = "ideal_path_not_succeeded"
    IDEAL_INTENT_NOT_EXPLICIT = "ideal_intent_not_explicit"
    IDEAL_INTENT_NOT_REVIEWED = "ideal_intent_not_reviewed"
    IDEAL_CALIBRATION_NOT_ACTIVE = "ideal_calibration_not_active"
    IDEAL_TARGET_BALL_MISSING = "ideal_target_ball_missing"
    IDEAL_ENGINE_UNAVAILABLE = "ideal_engine_unavailable"
    IDEAL_ADAPTER_FAILURE = "ideal_adapter_failure"
    SELECTION_LINEAGE_MISMATCH = "selection_lineage_mismatch"
    SHOT_NOT_SELECTED = "shot_not_selected"
    IDEAL_SOURCE_MISMATCH = "ideal_source_mismatch"
    REPORT_REVISION_MISMATCH = "report_revision_mismatch"
    INVALID_IDENTIFIER_OR_REVISION = "invalid_identifier_or_revision"


class VisualizationPathKind(StrEnum):
    ACTUAL_CUE = "ACTUAL_CUE"
    IDEAL_CUE = "IDEAL_CUE"
    IDEAL_OBJECT = "IDEAL_OBJECT"


class VisualizationLineStyle(StrEnum):
    SOLID = "SOLID"
    DASHED = "DASHED"


class VisualizationSemanticToken(StrEnum):
    ACTUAL = "ACTUAL"
    IDEAL = "IDEAL"


class VisualizationPoint(VisualizationStrictFrozenModel):
    x_mm: StrictInt = Field(ge=0, le=2540)
    y_mm: StrictInt = Field(ge=0, le=1270)


class VisualizationBall(VisualizationStrictFrozenModel):
    ball_id: StrictInt = Field(ge=0, le=15)
    position: VisualizationPoint


class VisualizationPathStyle(VisualizationStrictFrozenModel):
    semantic_token: VisualizationSemanticToken
    color: Color
    line_style: VisualizationLineStyle
    direct_label: NonEmptyText

    @model_validator(mode="after")
    def redundant_encodings_are_consistent(self) -> VisualizationPathStyle:
        expected_line = {
            VisualizationSemanticToken.ACTUAL: VisualizationLineStyle.SOLID,
            VisualizationSemanticToken.IDEAL: VisualizationLineStyle.DASHED,
        }[self.semantic_token]
        if self.line_style is not expected_line:
            raise ValueError("semantic token and line style must be consistent")
        return self


class VisualizationPathSeries(VisualizationStrictFrozenModel):
    kind: VisualizationPathKind
    ball_id: StrictInt = Field(ge=0, le=15)
    points: tuple[VisualizationPoint, ...] = Field(min_length=2)
    style: VisualizationPathStyle
    label: NonEmptyText

    @model_validator(mode="after")
    def ball_role_points_and_style_are_consistent(self) -> VisualizationPathSeries:
        if len(set(self.points)) < 2:
            raise ValueError("path requires at least two distinct points")
        if self.kind in {
            VisualizationPathKind.ACTUAL_CUE,
            VisualizationPathKind.IDEAL_CUE,
        }:
            if self.ball_id != 0:
                raise ValueError("cue-ball path must use ball id 0")
        elif self.ball_id == 0:
            raise ValueError("object-ball path cannot use cue-ball id")
        expected_token = (
            VisualizationSemanticToken.ACTUAL
            if self.kind is VisualizationPathKind.ACTUAL_CUE
            else VisualizationSemanticToken.IDEAL
        )
        if self.style.semantic_token is not expected_token:
            raise ValueError("path kind and semantic style must match")
        expected_labels = {
            VisualizationPathKind.ACTUAL_CUE: ("实际母球路径", "实际母球路径"),
            VisualizationPathKind.IDEAL_CUE: ("理想母球路径", "理想路径"),
            VisualizationPathKind.IDEAL_OBJECT: ("理想目标球路径", "理想路径"),
        }
        expected_label, expected_direct_label = expected_labels[self.kind]
        if self.label != expected_label or self.style.direct_label != expected_direct_label:
            raise ValueError("path kind must use the canonical direct label")
        return self


class VisualizationLegendItem(VisualizationStrictFrozenModel):
    semantic_token: VisualizationSemanticToken
    color: Color
    line_style: VisualizationLineStyle
    label: NonEmptyText


class VisualizationTable(VisualizationStrictFrozenModel):
    width_mm: StrictInt
    height_mm: StrictInt
    unit: Literal["mm"] = "mm"
    coordinate_system: Literal["LEFT_BOTTOM_POCKET_X_LONG_Y_SHORT"] = (
        "LEFT_BOTTOM_POCKET_X_LONG_Y_SHORT"
    )

    @model_validator(mode="after")
    def is_supported_true_ratio_table(self) -> VisualizationTable:
        if (self.width_mm, self.height_mm) != (2540, 1270):
            raise ValueError("visualization table must be the supported 2540:1270 plane")
        return self


class VisualizationSourceLineage(VisualizationStrictFrozenModel):
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    shot_id: UUID4


class VisualizationSelectionProvenance(VisualizationStrictFrozenModel):
    rule_version: Literal["1.0"]
    selection_reason: TypicalShotSelectionReason

    @field_validator("selection_reason", mode="before")
    @classmethod
    def parse_selection_reason(cls, value: object) -> object:
        return TypicalShotSelectionReason(value) if isinstance(value, str) else value


class VisualizationCalibrationProvenance(VisualizationStrictFrozenModel):
    calibration_id: UUID4
    revision_id: UUID4
    revision_number: StrictInt = Field(ge=1)
    table_configuration_fingerprint: Sha256


class VisualizationActualProvenance(VisualizationStrictFrozenModel):
    source: FinalFieldSource
    confidence: StrictFloat = Field(ge=0.0, le=1.0)

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value: object) -> object:
        return FinalFieldSource(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def is_supported_source(self) -> VisualizationActualProvenance:
        if self.source is FinalFieldSource.UNKNOWN:
            raise ValueError("actual path must have a known source")
        return self


class VisualizationIdealProvenance(VisualizationStrictFrozenModel):
    adapter_version: SafeMetadataToken
    engine_name: SafeMetadataToken
    engine_version: SafeMetadataToken
    configuration_version: Literal["chinese-eight-ball-9ft-v1"]


class VisualizationAsset(VisualizationStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    spec_version: Literal["actual-ideal-overlay/1.0"] = "actual-ideal-overlay/1.0"
    asset_id: Annotated[str, Field(pattern=r"^visualization-[0-9a-f]{64}$")]
    content_sha256: Sha256
    source: VisualizationSourceLineage
    selection: VisualizationSelectionProvenance
    table: VisualizationTable
    calibration: VisualizationCalibrationProvenance
    actual: VisualizationActualProvenance
    ideal: VisualizationIdealProvenance
    balls: tuple[VisualizationBall, ...]
    paths: tuple[VisualizationPathSeries, ...]
    legend: tuple[VisualizationLegendItem, ...]
    text_summary: NonEmptyText

    @model_validator(mode="after")
    def content_is_canonical_and_accessible(self) -> VisualizationAsset:
        for nested in (
            self.source,
            self.selection,
            self.table,
            self.calibration,
            self.actual,
            self.ideal,
            *self.balls,
            *self.paths,
            *self.legend,
        ):
            type(nested).model_validate(_stored_model_values(nested))
        ball_ids = tuple(ball.ball_id for ball in self.balls)
        if ball_ids != tuple(sorted(set(ball_ids))):
            raise ValueError("balls must be sorted and unique")
        if 0 not in ball_ids:
            raise ValueError("asset must display the cue ball")
        expected_path_order = (
            VisualizationPathKind.ACTUAL_CUE,
            VisualizationPathKind.IDEAL_CUE,
            VisualizationPathKind.IDEAL_OBJECT,
        )
        if tuple(path.kind for path in self.paths) != expected_path_order:
            raise ValueError("path layer order must be actual cue, ideal cue, ideal object")
        target_path = self.paths[2]
        if target_path.ball_id not in ball_ids:
            raise ValueError("ideal object path must refer to a displayed ball")
        positions = {ball.ball_id: ball.position for ball in self.balls}
        if (
            self.paths[0].points[0] != positions[0]
            or self.paths[1].points[0] != positions[0]
            or target_path.points[0] != positions[target_path.ball_id]
        ):
            raise ValueError("every path must start at its displayed ball position")
        actual_style = self.paths[0].style
        ideal_styles = (self.paths[1].style, self.paths[2].style)
        if any(style != ideal_styles[0] for style in ideal_styles[1:]):
            raise ValueError("ideal path series must share a stable style")
        if actual_style.color == ideal_styles[0].color:
            raise ValueError("actual and ideal paths require different semantic colors")
        expected_legend_order = (
            VisualizationSemanticToken.ACTUAL,
            VisualizationSemanticToken.IDEAL,
        )
        if tuple(item.semantic_token for item in self.legend) != expected_legend_order:
            raise ValueError("legend order must be actual then ideal")
        for item, style in zip(
            self.legend,
            (actual_style, ideal_styles[0]),
            strict=True,
        ):
            if item.color != style.color or item.line_style is not style.line_style:
                raise ValueError("legend must be graphically equivalent to path styles")
        if tuple(item.label for item in self.legend) != ("实际路径", "理想路径"):
            raise ValueError("legend labels must match their semantic tokens")
        if not (
            all(
                token in self.text_summary
                for token in ("\u5b9e\u9645", "\u7406\u60f3", "\u7f6e\u4fe1\u5ea6")
            )
            and ("\u6765\u6e90" in self.text_summary or "\u6765\u81ea" in self.text_summary)
        ):
            raise ValueError("text summary must state actual, ideal, source, and confidence")
        confidence_percent = int(round(self.actual.confidence * 100))
        expected_summary = (
            f"实际母球路径来源为 {self.actual.source.value}，置信度 {confidence_percent}%；"
            f"理想路径来源为 {self.ideal.engine_name} {self.ideal.engine_version}。"
        )
        if self.text_summary != expected_summary:
            raise ValueError("text summary must be the controlled equivalent of asset provenance")
        content = self.model_dump(
            mode="json",
            exclude={"asset_id", "content_sha256"},
        )
        expected_hash = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            self.content_sha256 != expected_hash
            or self.asset_id != f"visualization-{expected_hash}"
        ):
            raise ValueError("asset id and content digest must match canonical business content")
        return self


class VisualizationResult(VisualizationStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: VisualizationStatus
    reason: VisualizationFailureReason | None = None
    asset: VisualizationAsset | None = None

    @model_validator(mode="after")
    def status_reason_and_asset_are_consistent(self) -> VisualizationResult:
        if self.status is VisualizationStatus.SUCCEEDED:
            if self.reason is not None or self.asset is None:
                raise ValueError("successful visualization requires an asset and no reason")
        elif self.reason is None or self.asset is not None:
            raise ValueError("failed visualization requires a reason and no asset")
        return self


class VisualizationAssetBinding(VisualizationStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    asset_id: Annotated[str, Field(pattern=r"^visualization-[0-9a-f]{64}$")]
    asset_content_sha256: Sha256
    source: VisualizationSourceLineage
    target_session_id: UUID4
    target_report_id: UUID4
    target_report_revision: StrictInt = Field(ge=1)
    target_shot_id: UUID4

    @model_validator(mode="after")
    def target_matches_exact_source_lineage(self) -> VisualizationAssetBinding:
        if self.asset_id != f"visualization-{self.asset_content_sha256}":
            raise ValueError("binding asset id must match its content hash")
        if (
            self.target_session_id != self.source.session_id
            or self.target_report_id != self.source.report_id
            or self.target_report_revision != self.source.report_revision
            or self.target_shot_id != self.source.shot_id
        ):
            raise ValueError("binding target must match exact source lineage")
        return self


class VisualizationBindingStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


class VisualizationBindingReason(StrEnum):
    SOURCE_LINEAGE_MISMATCH = "source_lineage_mismatch"
    REPORT_REVISION_MISMATCH = "report_revision_mismatch"
    INVALID_IDENTIFIER_OR_REVISION = "invalid_identifier_or_revision"


class VisualizationAssetBindingResult(VisualizationStrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: VisualizationBindingStatus
    reason: VisualizationBindingReason | None = None
    binding: VisualizationAssetBinding | None = None

    @model_validator(mode="after")
    def status_reason_and_binding_are_consistent(self) -> VisualizationAssetBindingResult:
        if self.status is VisualizationBindingStatus.SUCCEEDED:
            if self.reason is not None or self.binding is None:
                raise ValueError("successful binding requires a binding and no reason")
        elif self.reason is None or self.binding is not None:
            raise ValueError("failed binding requires a reason and no binding")
        return self
