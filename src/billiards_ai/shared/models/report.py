"""Strict, versioned public report contract for Story 3.3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from billiards_ai.shared.models.calibration import CalibrationStatus
from billiards_ai.shared.models.quality import SessionQualityStatus, SessionQualitySummary
from billiards_ai.shared.models.recommendation import (
    RecommendationResult,
    RecommendationStatus,
)
from billiards_ai.shared.models.scoring import ScoreStatus, SessionPerformanceScores
from billiards_ai.shared.models.statistics import (
    CueBallEndFrequency,
    CushionCountFrequency,
    TargetZoneDeviationSummary,
)
from billiards_ai.shared.models.typical_shot import (
    TypicalShotRuleVersion,
    TypicalShotSelectionReason,
)
from billiards_ai.shared.models.visualization import (
    VisualizationActualProvenance,
    VisualizationAssetBinding,
    VisualizationFailureReason,
    VisualizationIdealProvenance,
    VisualizationResult,
    VisualizationSourceLineage,
    VisualizationStatus,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC)


def _contains_url(value: object) -> bool:
    if isinstance(value, str):
        return "http://" in value.lower() or "https://" in value.lower()
    if isinstance(value, BaseModel):
        return _contains_url(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return any(_contains_url(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_url(item) for item in value)
    return False


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


def _contains_unsafe_text(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return (
            any(ord(character) < 32 for character in value)
            or "http://" in lowered
            or "https://" in lowered
            or "oss://" in lowered
            or value.startswith(("/", "./", "../", "~\\", "~/"))
            or "\\" in value
        )
    if isinstance(value, BaseModel):
        return _contains_unsafe_text(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return any(_contains_unsafe_text(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_unsafe_text(item) for item in value)
    return False


def _canonical_sha256(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ReportStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReportStatus(StrEnum):
    DRAFT_PENDING_REVIEW = "DRAFT_PENDING_REVIEW"
    APPROVED = "APPROVED"
    CORRECTED = "CORRECTED"


class ReportSessionStatistics(ReportStrictFrozenModel):
    """Non-distance statistics safe for every calibration state."""

    total_shot_count: StrictInt = Field(gt=0)
    valid_shot_count: StrictInt = Field(gt=0)
    potted_ball_count: StrictInt = Field(ge=0)
    successful_potting_shot_count: StrictInt = Field(ge=0)
    potting_shot_count: StrictInt = Field(ge=0)
    pot_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    cushion_distribution: tuple[CushionCountFrequency, ...]
    cushion_count_shot_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> ReportSessionStatistics:
        if self.valid_shot_count > self.total_shot_count:
            raise ValueError("valid shot count cannot exceed total shot count")
        if self.successful_potting_shot_count > self.potting_shot_count:
            raise ValueError("successful potting shot count cannot exceed its denominator")
        expected_rate = (
            self.successful_potting_shot_count / self.potting_shot_count
            if self.potting_shot_count
            else None
        )
        if self.pot_rate != expected_rate:
            raise ValueError("pot rate must match the explicit potting shot denominator")
        if sum(item.shot_count for item in self.cushion_distribution) != (
            self.cushion_count_shot_count
        ):
            raise ValueError("cushion distribution must match its denominator")
        return self


class CalibratedReportSessionStatistics(ReportSessionStatistics):
    """Distance statistics present only for an active calibration."""

    average_cue_ball_travel_distance_mm: float | None = Field(default=None, ge=0.0)
    cue_ball_travel_shot_count: StrictInt = Field(ge=0)
    cue_ball_end_distribution: tuple[CueBallEndFrequency, ...]
    cue_ball_end_shot_count: StrictInt = Field(ge=0)
    target_zone_deviation_by_source: tuple[TargetZoneDeviationSummary, ...]

    @model_validator(mode="after")
    def distance_denominators_are_consistent(self) -> CalibratedReportSessionStatistics:
        if (self.average_cue_ball_travel_distance_mm is None) != (
            self.cue_ball_travel_shot_count == 0
        ):
            raise ValueError("travel average presence must match its denominator")
        if sum(item.shot_count for item in self.cue_ball_end_distribution) != (
            self.cue_ball_end_shot_count
        ):
            raise ValueError("cue-ball end distribution must match its denominator")
        sources = [item.target_zone_source for item in self.target_zone_deviation_by_source]
        if len(sources) != len(set(sources)):
            raise ValueError("target zone sources must be unique")
        return self


class AbstentionSummary(ReportStrictFrozenModel):
    total_shot_count: StrictInt = Field(gt=0)
    abstained_shot_count: StrictInt = Field(ge=0)
    display_text: NonEmptyStr

    @model_validator(mode="after")
    def count_and_text_are_consistent(self) -> AbstentionSummary:
        if self.abstained_shot_count > self.total_shot_count:
            raise ValueError("abstained shot count cannot exceed total shot count")
        expected = f"{self.abstained_shot_count} 杆超出当前分析范围"
        if self.display_text != expected:
            raise ValueError("abstention display text must state the explicit count")
        return self


class TypicalShotSummary(ReportStrictFrozenModel):
    shot_id: UUID4
    summary: NonEmptyStr


class CoachingRecommendation(ReportStrictFrozenModel):
    title: NonEmptyStr
    recommendation: NonEmptyStr


class ReportCompositionReason(StrEnum):
    RESULT_SET_MISMATCH = "RESULT_SET_MISMATCH"
    SELECTION_MISMATCH = "SELECTION_MISMATCH"
    ASSET_INTEGRITY_MISMATCH = "ASSET_INTEGRITY_MISMATCH"
    RECOMMENDATION_MISMATCH = "RECOMMENDATION_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


class ReportCompositionError(ValueError):
    def __init__(self, reason: ReportCompositionReason):
        self.reason = reason
        super().__init__(f"report composition blocked: {reason.value}")


class ReportHeadlineCode(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    NO_RELIABLE_COACHING = "NO_RELIABLE_COACHING"

    @property
    def text(self) -> str:
        return {
            ReportHeadlineCode.READY: "本场已找到可复核的典型杆证据和下一步训练方向。",
            ReportHeadlineCode.PARTIAL: "本场部分教练内容可用，请结合可信范围查看。",
            ReportHeadlineCode.NO_RELIABLE_COACHING: (
                "本场保留客观统计，当前没有足够可靠的教练内容。"
            ),
        }[self]


class ReportHeadline(ReportStrictFrozenModel):
    code: ReportHeadlineCode
    text: NonEmptyStr | None = None

    @model_validator(mode="after")
    def text_matches_code(self) -> ReportHeadline:
        if self.text not in {None, self.code.text}:
            raise ValueError("headline text must match its controlled code")
        object.__setattr__(self, "text", self.code.text)
        return self


class ReportSourceLineage(ReportStrictFrozenModel):
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)


class ReportTargetLineage(ReportStrictFrozenModel):
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)


class ReportTrustScope(ReportStrictFrozenModel):
    review_status: Literal[ReportStatus.APPROVED, ReportStatus.CORRECTED]
    total_shot_count: StrictInt = Field(gt=0)
    valid_shot_count: StrictInt = Field(gt=0)
    abstained_shot_count: StrictInt = Field(ge=0)
    calibration_status: CalibrationStatus
    unsupported_dimensions: tuple[
        Literal["potting", "positioning", "power_control", "rotation_application"], ...
    ]
    default_expanded: Literal[True] = True

    @model_validator(mode="after")
    def counts_and_dimensions_are_canonical(self) -> ReportTrustScope:
        if self.valid_shot_count + self.abstained_shot_count != self.total_shot_count:
            raise ValueError("trust scope shot counts must be complete")
        if self.unsupported_dimensions != tuple(sorted(set(self.unsupported_dimensions))):
            raise ValueError("unsupported dimensions must be sorted and unique")
        return self


class ObjectiveStatisticsSection(ReportStrictFrozenModel):
    source: ReportSourceLineage
    analysis_version: NonEmptyStr
    shot_ids: tuple[UUID4, ...]
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def shot_ids_are_unique(self) -> ObjectiveStatisticsSection:
        if not self.shot_ids or len(set(self.shot_ids)) != len(self.shot_ids):
            raise ValueError("objective section requires every unique approved shot id")
        return self


_SELECTION_REASON_TEXT = {
    TypicalShotSelectionReason.SEVERE_FOUL: "严重犯规代表",
    TypicalShotSelectionReason.SEVERE_DEVIATION: "母球停位偏差代表",
    TypicalShotSelectionReason.SEVERE_MISSED_POT: "未进球代表",
    TypicalShotSelectionReason.REPEATED_FOUL: "重复犯规代表",
    TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION: "重复停位偏差代表",
    TypicalShotSelectionReason.REPEATED_MISSED_POT: "重复未进球代表",
    TypicalShotSelectionReason.POSITIVE_POTTING: "正向进球样本",
}


class ReportVisualizationAssetReference(ReportStrictFrozenModel):
    asset_id: Annotated[str, Field(pattern=r"^visualization-[0-9a-f]{64}$")]
    asset_content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    spec_version: Literal["actual-ideal-overlay/1.0"]
    text_summary: NonEmptyStr
    actual: VisualizationActualProvenance
    ideal: VisualizationIdealProvenance
    source_approved_lineage: ReportSourceLineage
    target_report_lineage: ReportTargetLineage
    exact_source_binding: VisualizationAssetBinding

    @model_validator(mode="after")
    def source_binding_is_exact_and_target_is_distinct(self) -> ReportVisualizationAssetReference:
        binding = VisualizationAssetBinding.model_validate(
            _stored_model_values(self.exact_source_binding)
        )
        if (
            self.asset_id != binding.asset_id
            or self.asset_content_sha256 != binding.asset_content_sha256
            or self.source_approved_lineage.session_id != binding.source.session_id
            or self.source_approved_lineage.report_id != binding.source.report_id
            or self.source_approved_lineage.report_revision != binding.source.report_revision
            or self.target_report_lineage.session_id != self.source_approved_lineage.session_id
            or self.target_report_lineage.report_id == self.source_approved_lineage.report_id
            or self.target_report_lineage.report_revision
            != self.source_approved_lineage.report_revision + 1
        ):
            raise ValueError(
                "report asset reference requires exact source and distinct child lineage"
            )
        return self


class TypicalShotReportCard(ReportStrictFrozenModel):
    shot_id: UUID4
    sequence: StrictInt = Field(ge=1)
    selection_rank: StrictInt = Field(ge=1, le=5)
    selection_reason: TypicalShotSelectionReason
    selection_reason_text: NonEmptyStr | None = None
    rule_version: TypicalShotRuleVersion
    visualization_status: VisualizationStatus
    visualization_reason: VisualizationFailureReason | None = None
    asset_reference: ReportVisualizationAssetReference | None = None

    @model_validator(mode="after")
    def selection_and_visualization_are_consistent(self) -> TypicalShotReportCard:
        expected = _SELECTION_REASON_TEXT[self.selection_reason]
        if self.selection_reason_text not in {None, expected}:
            raise ValueError("selection reason text must use the controlled mapping")
        object.__setattr__(self, "selection_reason_text", expected)
        if self.visualization_status is VisualizationStatus.SUCCEEDED:
            if self.visualization_reason is not None or self.asset_reference is None:
                raise ValueError("successful card requires an asset reference")
            if self.asset_reference.exact_source_binding.source.shot_id != self.shot_id:
                raise ValueError("asset source binding must identify the card shot")
        elif self.visualization_reason is None or self.asset_reference is not None:
            raise ValueError("unavailable card requires its original reason and no asset")
        return self


class TypicalShotSectionStatus(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    NO_RELIABLE_DATA = "NO_RELIABLE_DATA"


class TypicalShotNoDataReason(StrEnum):
    NO_RELIABLE_SELECTION = "当前没有可可靠展示的典型杆"


class TypicalShotAnalysisSection(ReportStrictFrozenModel):
    status: TypicalShotSectionStatus
    no_data_reason: TypicalShotNoDataReason | None = None
    cards: tuple[TypicalShotReportCard, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def status_matches_cards(self) -> TypicalShotAnalysisSection:
        cards = tuple(
            TypicalShotReportCard.model_validate(_stored_model_values(card)) for card in self.cards
        )
        if [card.selection_rank for card in cards] != list(range(1, len(cards) + 1)):
            raise ValueError("typical-shot cards must remain in selection rank order")
        if len({card.shot_id for card in cards}) != len(cards):
            raise ValueError("typical-shot card shot ids must be unique")
        succeeded = sum(card.asset_reference is not None for card in cards)
        expected = (
            TypicalShotSectionStatus.NO_RELIABLE_DATA
            if not cards
            else TypicalShotSectionStatus.READY
            if succeeded == len(cards)
            else TypicalShotSectionStatus.PARTIAL
        )
        if self.status is not expected:
            raise ValueError("typical-shot section status must match asset coverage")
        if (not cards) != (self.no_data_reason is not None):
            raise ValueError("only an empty typical-shot section carries the fixed reason")
        return self


class TrainingRecommendationSection(ReportStrictFrozenModel):
    result: RecommendationResult

    @model_validator(mode="after")
    def result_is_user_visible_not_integrity_failure(self) -> TrainingRecommendationSection:
        result = RecommendationResult.model_validate(_stored_model_values(self.result))
        if result.status not in {
            RecommendationStatus.READY,
            RecommendationStatus.NO_RELIABLE_RECOMMENDATION,
        }:
            raise ValueError("training section cannot expose an integrity failure")
        return self

    @property
    def status(self) -> RecommendationStatus:
        return self.result.status

    @property
    def fallback(self) -> str | None:
        return self.result.fallback


class ReportQualityNotice(ReportStrictFrozenModel):
    data_coverage_status: SessionQualityStatus
    algorithm_boundaries: tuple[
        Literal["potting", "positioning", "power_control", "rotation_application"], ...
    ]
    current_revision: StrictInt = Field(ge=2)
    source_parent_report_id: UUID4
    source_parent_revision: StrictInt = Field(ge=1)
    created_as_status: Literal[ReportStatus.DRAFT_PENDING_REVIEW] = (
        ReportStatus.DRAFT_PENDING_REVIEW
    )
    review_notice: Literal["本报告新增教练内容，须再次人工复核后方可交付。"] = (
        "本报告新增教练内容，须再次人工复核后方可交付。"
    )

    @model_validator(mode="after")
    def revision_and_boundaries_are_canonical(self) -> ReportQualityNotice:
        if self.current_revision != self.source_parent_revision + 1:
            raise ValueError("quality notice must identify adjacent parent and child revisions")
        if self.algorithm_boundaries != tuple(sorted(set(self.algorithm_boundaries))):
            raise ValueError("algorithm boundaries must be sorted and unique")
        return self


class ReportVisualizationResult(ReportStrictFrozenModel):
    """Caller result association required because failed 5.3 results have no lineage."""

    shot_id: UUID4
    source: VisualizationSourceLineage
    result: VisualizationResult

    @model_validator(mode="after")
    def nested_result_is_revalidated(self) -> ReportVisualizationResult:
        VisualizationSourceLineage.model_validate(_stored_model_values(self.source))
        VisualizationResult.model_validate(_stored_model_values(self.result))
        if self.source.shot_id != self.shot_id:
            raise ValueError("visualization result source must identify its selected shot")
        return self


class ReportDocument(ReportStrictFrozenModel):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    report_id: UUID4
    revision: StrictInt = Field(ge=1)
    parent_revision_id: UUID4 | None = None
    session_id: UUID4
    generated_at: datetime
    analysis_version: NonEmptyStr
    calibration_status: CalibrationStatus
    report_status: ReportStatus = ReportStatus.DRAFT_PENDING_REVIEW
    quality_summary: SessionQualitySummary
    session_statistics: CalibratedReportSessionStatistics | ReportSessionStatistics
    dimension_scores: SessionPerformanceScores
    abstention_summary: AbstentionSummary
    typical_shots: tuple[TypicalShotSummary, ...] = ()
    coaching_recommendations: tuple[CoachingRecommendation, ...] = ()
    headline: ReportHeadline | None = None
    trust_scope: ReportTrustScope | None = None
    objective_statistics: ObjectiveStatisticsSection | None = None
    typical_shot_analysis: TypicalShotAnalysisSection | None = None
    training_recommendation: TrainingRecommendationSection | None = None
    quality_notice: ReportQualityNotice | None = None
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None

    _generated_at_utc = field_validator("generated_at")(_utc)

    @model_validator(mode="after")
    def revision_quality_and_calibration_are_consistent(self) -> ReportDocument:
        if self.revision == 1 and self.parent_revision_id is not None:
            raise ValueError("first revision cannot identify a parent revision")
        if self.revision > 1 and self.parent_revision_id is None:
            raise ValueError("later revision must identify its parent revision")
        if self.parent_revision_id == self.report_id:
            raise ValueError("a revision cannot identify itself as its parent")

        calibrated_statistics = isinstance(
            self.session_statistics, CalibratedReportSessionStatistics
        )
        if self.calibration_status is CalibrationStatus.ACTIVE and not calibrated_statistics:
            raise ValueError("active calibration requires distance statistics")
        if self.calibration_status is not CalibrationStatus.ACTIVE and calibrated_statistics:
            raise ValueError("inactive calibration cannot expose distance statistics")

        if self.quality_summary.shot_count != self.session_statistics.total_shot_count:
            raise ValueError("quality and statistics shot counts must match")
        if self.quality_summary.abstained_shot_count != (
            self.session_statistics.total_shot_count - self.session_statistics.valid_shot_count
        ):
            raise ValueError("quality abstention count must match valid shot count")
        if self.abstention_summary.total_shot_count != self.quality_summary.shot_count:
            raise ValueError("abstention summary total must match quality summary")
        if (
            self.abstention_summary.abstained_shot_count
            != self.quality_summary.abstained_shot_count
        ):
            raise ValueError("abstention summary count must match quality summary")
        if _contains_url(self):
            raise ValueError("URLs are not allowed in ReportDocument")
        v11_fields = (
            self.headline,
            self.trust_scope,
            self.objective_statistics,
            self.typical_shot_analysis,
            self.training_recommendation,
            self.quality_notice,
            self.content_sha256,
        )
        if self.schema_version == "1.0":
            if any(value is not None for value in v11_fields):
                raise ValueError("schema 1.0 cannot contain 1.1 report content")
            return self
        if any(value is None for value in v11_fields):
            raise ValueError("schema 1.1 requires all six reading blocks and a digest")
        if self.typical_shots or self.coaching_recommendations:
            raise ValueError("schema 1.1 cannot populate legacy coaching placeholders")
        for nested in v11_fields[:-1]:
            type(nested).model_validate(_stored_model_values(nested))
        if _contains_unsafe_text(self):
            raise ValueError("unsafe report text is not allowed")
        source = self.objective_statistics.source
        if (
            source.session_id != self.session_id
            or source.report_id != self.parent_revision_id
            or source.report_revision != self.revision - 1
            or self.objective_statistics.analysis_version != self.analysis_version
            or len(self.objective_statistics.shot_ids) != self.session_statistics.total_shot_count
        ):
            raise ValueError("objective statistics must bind the exact approved parent")
        if (
            self.trust_scope.total_shot_count != self.session_statistics.total_shot_count
            or self.trust_scope.valid_shot_count != self.session_statistics.valid_shot_count
            or self.trust_scope.abstained_shot_count != self.abstention_summary.abstained_shot_count
            or self.trust_scope.calibration_status is not self.calibration_status
        ):
            raise ValueError("trust scope must match the approved objective facts")
        notice = self.quality_notice
        if (
            notice.current_revision != self.revision
            or notice.source_parent_report_id != self.parent_revision_id
            or notice.source_parent_revision != self.revision - 1
        ):
            raise ValueError("quality notice must bind the child and parent revisions")
        unsupported_dimensions = tuple(
            sorted(
                name
                for name, score in self.dimension_scores
                if score.status is ScoreStatus.UNSUPPORTED
            )
        )
        expected_headline = (
            ReportHeadlineCode.NO_RELIABLE_COACHING
            if self.typical_shot_analysis.status is TypicalShotSectionStatus.NO_RELIABLE_DATA
            and self.training_recommendation.status
            is RecommendationStatus.NO_RELIABLE_RECOMMENDATION
            else ReportHeadlineCode.READY
            if self.typical_shot_analysis.status is TypicalShotSectionStatus.READY
            and self.training_recommendation.status is RecommendationStatus.READY
            else ReportHeadlineCode.PARTIAL
        )
        if (
            self.headline.code is not expected_headline
            or notice.data_coverage_status is not self.quality_summary.session_quality_status
            or notice.algorithm_boundaries != unsupported_dimensions
            or self.trust_scope.unsupported_dimensions != unsupported_dimensions
        ):
            raise ValueError("derived report fields must match authoritative report facts")
        for card in self.typical_shot_analysis.cards:
            reference = card.asset_reference
            if reference is not None and (
                reference.source_approved_lineage != source
                or reference.target_report_lineage.session_id != self.session_id
                or reference.target_report_lineage.report_id != self.report_id
                or reference.target_report_lineage.report_revision != self.revision
            ):
                raise ValueError("every asset reference must bind this parent and child")
        recommendation_lineage = self.training_recommendation.result.source_lineage
        if recommendation_lineage is None or (
            recommendation_lineage.session_id != source.session_id
            or recommendation_lineage.report_id != source.report_id
            or recommendation_lineage.report_revision != source.report_revision
        ):
            raise ValueError("recommendation must bind the exact approved parent")
        expected_objective = objective_statistics_digest(
            source=source,
            analysis_version=self.analysis_version,
            shot_ids=self.objective_statistics.shot_ids,
            quality_summary=self.quality_summary,
            session_statistics=self.session_statistics,
            dimension_scores=self.dimension_scores,
            abstention_summary=self.abstention_summary,
        )
        if self.objective_statistics.content_sha256 != expected_objective:
            raise ValueError("objective statistics digest does not match approved facts")
        expected_report = _canonical_sha256(
            self.model_dump(mode="json", exclude={"content_sha256", "report_status"})
        )
        if self.content_sha256 != expected_report:
            raise ValueError("report content digest does not match canonical content")
        return self

    @classmethod
    def build_v11(cls, **values: object) -> ReportDocument:
        payload = dict(values) | {"schema_version": "1.1"}
        canonical_payload = to_jsonable_python(payload, by_alias=True)
        canonical_payload.pop("content_sha256", None)
        canonical_payload.pop("report_status", None)
        digest = _canonical_sha256(canonical_payload)
        return cls.model_validate(payload | {"content_sha256": digest})


def objective_statistics_digest(
    *,
    source: ReportSourceLineage,
    analysis_version: str,
    shot_ids: tuple[UUID4, ...],
    quality_summary: SessionQualitySummary,
    session_statistics: CalibratedReportSessionStatistics | ReportSessionStatistics,
    dimension_scores: SessionPerformanceScores,
    abstention_summary: AbstentionSummary,
) -> str:
    return _canonical_sha256(
        {
            "source": source.model_dump(mode="json"),
            "analysis_version": analysis_version,
            "shot_ids": [str(shot_id) for shot_id in shot_ids],
            "quality_summary": quality_summary.model_dump(mode="json"),
            "session_statistics": session_statistics.model_dump(mode="json"),
            "dimension_scores": dimension_scores.model_dump(mode="json"),
            "abstention_summary": abstention_summary.model_dump(mode="json"),
        }
    )


class UnsupportedReportSchemaVersionError(ValueError):
    """Raised before validation when a report uses an unsupported schema major."""

    def __init__(self, schema_version: object, supported_major_versions: tuple[int, ...]):
        self.schema_version = schema_version
        self.supported_major_versions = supported_major_versions
        super().__init__(
            f"unsupported report schema major in {schema_version!r}; "
            f"supported majors: {supported_major_versions}"
        )


def parse_report_document(payload: Mapping[str, object]) -> ReportDocument:
    """Validate a v1 report, rejecting unsupported majors without silent downgrade."""

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, str):
        major_text, separator, _minor = schema_version.partition(".")
        if separator and major_text.isdigit() and int(major_text) != 1:
            raise UnsupportedReportSchemaVersionError(schema_version, (1,))
        if schema_version not in {"1.0", "1.1"}:
            raise UnsupportedReportSchemaVersionError(schema_version, (1,))
    return ReportDocument.model_validate_json(json.dumps(payload))
