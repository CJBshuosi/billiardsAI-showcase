"""Strict traceable recommendation contracts for the frozen V1 rule catalog."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

from billiards_ai.shared.models.quality import ShotAnalysisStatus
from billiards_ai.shared.models.shot import Confidence, ReviewStatus, TargetZoneSource
from billiards_ai.shared.models.typical_shot import (
    TypicalShotRuleVersion,
    TypicalShotSelectionReason,
)

_FALLBACK = "当前数据不足以给出训练建议"


def _stored_model_values(value: BaseModel) -> dict[str, object]:
    values = dict(vars(value))
    if value.__pydantic_extra__:
        values.update(value.__pydantic_extra__)
    return values


class RecommendationStrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return type(self).model_validate(_stored_model_values(copied))


class RecommendationCatalogVersion(StrEnum):
    V1 = "1.0"


class RecommendationRuleVersion(StrEnum):
    V1 = "1.0"


class RecommendationRuleId(StrEnum):
    REDUCE_FOUL_RISK = "reduce-foul-risk"
    STABILIZE_POTTING = "stabilize-potting"
    CONTROL_CUE_BALL_POSITION = "control-cue-ball-position"


class RecommendationStatus(StrEnum):
    READY = "READY"
    NO_RELIABLE_RECOMMENDATION = "NO_RELIABLE_RECOMMENDATION"
    SELECTION_MISMATCH = "SELECTION_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"


class RecommendationReason(StrEnum):
    NO_NEGATIVE_RULE_MATCH = "NO_NEGATIVE_RULE_MATCH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED_EVIDENCE = "UNSUPPORTED_EVIDENCE"
    SELECTION_RESULT_MISMATCH = "SELECTION_RESULT_MISMATCH"
    INVALID_IDENTIFIER_OR_INPUT = "INVALID_IDENTIFIER_OR_INPUT"


class RecommendationEvidenceKind(StrEnum):
    FOUL = "FOUL"
    MISSED_POT = "MISSED_POT"
    LARGE_DEVIATION = "LARGE_DEVIATION"


class RecommendationScope(StrEnum):
    SESSION_PRIMARY = "SESSION_PRIMARY"


class RecommendationClaimCode(StrEnum):
    REDUCE_FOUL_RISK = "REDUCE_FOUL_RISK"
    STABILIZE_POTTING = "STABILIZE_POTTING"
    CONTROL_CUE_BALL_POSITION = "CONTROL_CUE_BALL_POSITION"


class RecommendationPracticeCode(StrEnum):
    TRACK_BASIC_SHOTS_WITHOUT_FOUL = "TRACK_BASIC_SHOTS_WITHOUT_FOUL"
    TRACK_BASIC_STRAIGHT_POTS = "TRACK_BASIC_STRAIGHT_POTS"
    TRACK_TARGET_ZONE_FINISHES = "TRACK_TARGET_ZONE_FINISHES"

    @property
    def instruction(self) -> str:
        return {
            RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL: (
                "固定同一组基础直线球位重复击球，逐杆记录是否犯规。"
            ),
            RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS: (
                "固定同一组直线球位重复击球，逐杆记录是否进球。"
            ),
            RecommendationPracticeCode.TRACK_TARGET_ZONE_FINISHES: (
                "固定同一球位和目标区重复击球，逐杆记录母球是否进入目标区。"
            ),
        }[self]


class RecommendationWordingVariant(StrEnum):
    RULE_ORIGINAL = "RULE_ORIGINAL"


_CORE_PAIRS = {
    RecommendationRuleId.REDUCE_FOUL_RISK: (
        RecommendationClaimCode.REDUCE_FOUL_RISK,
        RecommendationPracticeCode.TRACK_BASIC_SHOTS_WITHOUT_FOUL,
    ),
    RecommendationRuleId.STABILIZE_POTTING: (
        RecommendationClaimCode.STABILIZE_POTTING,
        RecommendationPracticeCode.TRACK_BASIC_STRAIGHT_POTS,
    ),
    RecommendationRuleId.CONTROL_CUE_BALL_POSITION: (
        RecommendationClaimCode.CONTROL_CUE_BALL_POSITION,
        RecommendationPracticeCode.TRACK_TARGET_ZONE_FINISHES,
    ),
}

_CORE_EVIDENCE_KINDS = {
    RecommendationRuleId.REDUCE_FOUL_RISK: RecommendationEvidenceKind.FOUL,
    RecommendationRuleId.STABILIZE_POTTING: RecommendationEvidenceKind.MISSED_POT,
    RecommendationRuleId.CONTROL_CUE_BALL_POSITION: RecommendationEvidenceKind.LARGE_DEVIATION,
}

RecommendationTitle = Literal[
    "建议下次重点练：减少犯规",
    "建议下次重点练：基础进球",
    "建议下次重点练：母球停位",
]
RecommendationAdvice = Literal[
    "先把无犯规的基础击球稳定下来。",
    "先提高基础进球稳定性。",
    "先练母球停位控制。",
]
RecommendationInstruction = Literal[
    "固定同一组基础直线球位重复击球，逐杆记录是否犯规。",
    "固定同一组直线球位重复击球，逐杆记录是否进球。",
    "固定同一球位和目标区重复击球，逐杆记录母球是否进入目标区。",
]


class RecommendationSourceLineage(RecommendationStrictFrozenModel):
    session_id: UUID4
    report_id: UUID4
    report_revision: StrictInt = Field(ge=1)
    selection_rule_version: TypicalShotRuleVersion


class RecommendationEvidence(RecommendationStrictFrozenModel):
    evidence_kind: RecommendationEvidenceKind
    shot_id: UUID4
    sequence: StrictInt = Field(ge=1)
    confidence: Confidence
    analysis_status: ShotAnalysisStatus
    review_status: ReviewStatus
    selection_rank: StrictInt = Field(ge=1, le=5)
    selection_reason: TypicalShotSelectionReason
    selection_rule_version: TypicalShotRuleVersion
    is_foul: StrictBool | None = None
    potted_balls: tuple[StrictInt, ...] | None = None
    deviation_mm: StrictInt | None = Field(default=None, ge=0)
    target_zone_source: TargetZoneSource | None = None

    @model_validator(mode="after")
    def scalar_snapshot_matches_evidence_kind(self) -> RecommendationEvidence:
        if (
            self.confidence not in {Confidence.HIGH, Confidence.MEDIUM}
            or self.analysis_status is not ShotAnalysisStatus.ANALYZED
            or self.review_status not in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}
        ):
            raise ValueError("recommendation evidence must be selector-eligible and reviewed")
        foul_reasons = {
            TypicalShotSelectionReason.SEVERE_FOUL,
            TypicalShotSelectionReason.REPEATED_FOUL,
        }
        missed_reasons = {
            TypicalShotSelectionReason.SEVERE_MISSED_POT,
            TypicalShotSelectionReason.REPEATED_MISSED_POT,
        }
        deviation_reasons = {
            TypicalShotSelectionReason.SEVERE_DEVIATION,
            TypicalShotSelectionReason.REPEATED_LARGE_DEVIATION,
        }
        if self.evidence_kind is RecommendationEvidenceKind.FOUL:
            valid = (
                self.selection_reason in foul_reasons
                and self.is_foul is True
                and self.potted_balls is None
                and self.deviation_mm is None
                and self.target_zone_source is None
            )
        elif self.evidence_kind is RecommendationEvidenceKind.MISSED_POT:
            valid = (
                self.selection_reason in missed_reasons
                and self.is_foul is False
                and self.potted_balls == ()
                and self.deviation_mm is None
                and self.target_zone_source is None
            )
        else:
            valid = (
                self.selection_reason in deviation_reasons
                and self.is_foul is False
                and self.potted_balls is None
                and self.deviation_mm is not None
                and self.target_zone_source
                in {TargetZoneSource.SCENE_PRESET, TargetZoneSource.POST_HOC_INFERRED}
            )
        if not valid:
            raise ValueError("evidence kind requires its exact canonical scalar snapshot")
        return self


class RecommendationSemanticCore(RecommendationStrictFrozenModel):
    rule_id: RecommendationRuleId
    rule_version: RecommendationRuleVersion
    scope: RecommendationScope
    claim_code: RecommendationClaimCode
    practice_code: RecommendationPracticeCode

    @model_validator(mode="after")
    def catalog_pair_is_frozen(self) -> RecommendationSemanticCore:
        if self.rule_version is not RecommendationRuleVersion.V1:
            raise ValueError("unsupported recommendation rule version")
        expected_claim, expected_practice = _CORE_PAIRS[self.rule_id]
        if self.claim_code is not expected_claim or self.practice_code is not expected_practice:
            raise ValueError("rule, claim, and practice codes must match the frozen catalog")
        return self

    @property
    def title(self) -> str:
        return {
            RecommendationClaimCode.REDUCE_FOUL_RISK: "建议下次重点练：减少犯规",
            RecommendationClaimCode.STABILIZE_POTTING: "建议下次重点练：基础进球",
            RecommendationClaimCode.CONTROL_CUE_BALL_POSITION: "建议下次重点练：母球停位",
        }[self.claim_code]

    @property
    def primary_advice(self) -> str:
        return {
            RecommendationClaimCode.REDUCE_FOUL_RISK: "先把无犯规的基础击球稳定下来。",
            RecommendationClaimCode.STABILIZE_POTTING: "先提高基础进球稳定性。",
            RecommendationClaimCode.CONTROL_CUE_BALL_POSITION: "先练母球停位控制。",
        }[self.claim_code]


class RecommendationDrill(RecommendationStrictFrozenModel):
    practice_code: RecommendationPracticeCode
    instruction: RecommendationInstruction | None = None

    @model_validator(mode="after")
    def instruction_matches_practice(self) -> RecommendationDrill:
        expected = self.practice_code.instruction
        if self.instruction not in {None, expected}:
            raise ValueError("drill instruction must match the controlled catalog")
        object.__setattr__(self, "instruction", expected)
        return self


class Recommendation(RecommendationStrictFrozenModel):
    semantic_core: RecommendationSemanticCore
    evidence: tuple[RecommendationEvidence, ...] = Field(min_length=1)
    drill: RecommendationDrill
    title: RecommendationTitle | None = None
    primary_advice: RecommendationAdvice | None = None
    wording_variant: RecommendationWordingVariant = RecommendationWordingVariant.RULE_ORIGINAL

    @model_validator(mode="after")
    def evidence_and_drill_match_core(self) -> Recommendation:
        evidence = tuple(
            RecommendationEvidence.model_validate(_stored_model_values(item))
            for item in self.evidence
        )
        if tuple(item.shot_id for item in evidence) != tuple(
            sorted({item.shot_id for item in evidence}, key=str)
        ):
            raise ValueError("recommendation evidence shot ids must be sorted and unique")
        ranks = tuple(item.selection_rank for item in evidence)
        if len(ranks) != len(set(ranks)):
            raise ValueError("recommendation evidence selection ranks must be unique")
        expected_kind = _CORE_EVIDENCE_KINDS[self.semantic_core.rule_id]
        if any(item.evidence_kind is not expected_kind for item in evidence):
            raise ValueError("recommendation evidence must match the frozen rule catalog")
        if self.drill.practice_code is not self.semantic_core.practice_code:
            raise ValueError("drill must match the recommendation semantic core")
        expected_title = self.semantic_core.title
        expected_advice = self.semantic_core.primary_advice
        if self.title not in {None, expected_title} or self.primary_advice not in {
            None,
            expected_advice,
        }:
            raise ValueError("title and primary advice must match the controlled catalog")
        object.__setattr__(self, "title", expected_title)
        object.__setattr__(self, "primary_advice", expected_advice)
        return self


class RecommendationResult(RecommendationStrictFrozenModel):
    catalog_version: RecommendationCatalogVersion
    status: RecommendationStatus
    reason: RecommendationReason | None = None
    source_lineage: RecommendationSourceLineage | None = None
    recommendations: tuple[Recommendation, ...] = Field(default=(), max_length=1)
    fallback: str | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def _canonical_payload(values: dict[str, object]) -> bytes:
        payload = {key: value for key, value in values.items() if key != "content_sha256"}
        normalized = RecommendationResult._json_ready(payload)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _json_ready(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, tuple):
            return [RecommendationResult._json_ready(item) for item in value]
        if isinstance(value, dict):
            return {key: RecommendationResult._json_ready(item) for key, item in value.items()}
        if isinstance(value, StrEnum):
            return value.value
        return value

    @classmethod
    def build(cls, **values: object) -> RecommendationResult:
        payload = dict(values)
        payload.setdefault("reason", None)
        payload.setdefault("source_lineage", None)
        payload.setdefault("recommendations", ())
        status = payload.get("status")
        if status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION:
            payload["fallback"] = _FALLBACK
        else:
            payload.setdefault("fallback", None)
        digest = hashlib.sha256(cls._canonical_payload(payload)).hexdigest()
        return cls.model_validate(payload | {"content_sha256": digest})

    @model_validator(mode="after")
    def state_matrix_and_digest_are_exact(self) -> RecommendationResult:
        if self.source_lineage is not None:
            RecommendationSourceLineage.model_validate(_stored_model_values(self.source_lineage))
        for recommendation in self.recommendations:
            Recommendation.model_validate(_stored_model_values(recommendation))
        if self.catalog_version is not RecommendationCatalogVersion.V1:
            raise ValueError("unsupported recommendation catalog version")
        if self.status is RecommendationStatus.READY:
            valid = (
                self.reason is None
                and self.source_lineage is not None
                and len(self.recommendations) == 1
                and self.fallback is None
            )
        elif self.status is RecommendationStatus.NO_RELIABLE_RECOMMENDATION:
            valid = (
                self.reason
                in {
                    RecommendationReason.NO_NEGATIVE_RULE_MATCH,
                    RecommendationReason.INSUFFICIENT_EVIDENCE,
                    RecommendationReason.UNSUPPORTED_EVIDENCE,
                }
                and self.source_lineage is not None
                and not self.recommendations
                and self.fallback == _FALLBACK
            )
        elif self.status is RecommendationStatus.SELECTION_MISMATCH:
            valid = (
                self.reason is RecommendationReason.SELECTION_RESULT_MISMATCH
                and self.source_lineage is not None
                and not self.recommendations
                and self.fallback is None
            )
        else:
            valid = (
                self.reason is RecommendationReason.INVALID_IDENTIFIER_OR_INPUT
                and self.source_lineage is None
                and not self.recommendations
                and self.fallback is None
            )
        if not valid:
            raise ValueError("recommendation result fields do not match its status")
        expected = hashlib.sha256(self._canonical_payload(_stored_model_values(self))).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("recommendation content digest does not match canonical content")
        return self
