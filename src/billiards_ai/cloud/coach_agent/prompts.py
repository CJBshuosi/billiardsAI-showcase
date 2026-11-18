"""Versioned, hash-bound prompts for Coach Agent workflows."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from billiards_ai.shared.models.events import (
    CoreContractModel,
    NonEmptyStr,
    Sha256,
    canonical_sha256,
)


class GrowthAnalysisPrompt(CoreContractModel):
    version: Literal["growth-analysis-v1"]
    system_prompt: NonEmptyStr = Field(repr=False)
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_is_hash_bound(self) -> Self:
        if self.content_sha256 != canonical_sha256(self.system_prompt):
            raise ValueError("growth analysis prompt hash mismatch")
        return self


_GROWTH_ANALYSIS_V1_TEXT = """\
你是受约束的台球成长分析候选选择器。
你只能从输入 metric_candidates 中选择一个主要候选和至多一个次要候选。
输出只包含 metric_identity_sha256 和 finding_type。
不得输出自由文本、数字、证据 ID、support、reliability 或 trend。
不得改写、重算或推测任何统计数字、support、reliability、trend 或 evidence lineage。
IMPROVEMENT、DECLINE、PLATEAU 只能选择已分别具有 IMPROVING、DECLINING、STABLE 冻结趋势的 metric。
对 UNKNOWN、NO_SAMPLE、INSUFFICIENT_EVIDENCE、UNSUPPORTED 或 ABSTAINED，
选择 INSUFFICIENT_EVIDENCE 或不选择。
只能描述证据中的相关关系；任何候选都不证明因果，不得做因果声称。
严格输出符合提供 JSON Schema 的单一 JSON object。
""".strip()

GROWTH_ANALYSIS_PROMPT_V1_SHA256 = (
    "64d1d3bc878f53565b9f2f06f72afa7895c6f6989c3c53377b5200c0280cd1e4"
)

GROWTH_ANALYSIS_PROMPT_V1 = GrowthAnalysisPrompt(
    version="growth-analysis-v1",
    system_prompt=_GROWTH_ANALYSIS_V1_TEXT,
    content_sha256=GROWTH_ANALYSIS_PROMPT_V1_SHA256,
)


def get_growth_analysis_prompt(version: str) -> GrowthAnalysisPrompt:
    if version != GROWTH_ANALYSIS_PROMPT_V1.version:
        raise ValueError("unsupported growth analysis prompt version")
    return GROWTH_ANALYSIS_PROMPT_V1


class TrainingPlanningPrompt(CoreContractModel):
    version: Literal["training-plan-v1"]
    system_prompt: NonEmptyStr = Field(repr=False)
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_is_hash_bound(self) -> Self:
        if self.content_sha256 != canonical_sha256(self.system_prompt):
            raise ValueError("training planning prompt hash mismatch")
        return self


_TRAINING_PLANNING_V1_TEXT = """\
你是受约束的台球训练规划候选选择器。
你只能从输入 assessment findings 与 eligible drills 中选择目标、训练 block 和 fallback。
不得发明 drill、finding、数字、证据、能力、阈值、时间、状态、ID、hash 或发布决定。
每个 block 只能输出 drill_id、drill_revision、target_finding_id、
target_repetitions 和 difficulty_level。
输出一个 primary goal、至多一个 secondary goal、1 至 3 个 block 和一个 fallback drill。
不得输出自由文本 rationale、签名、VALIDATED、APPROVED 或 PUBLISHED。
计划历史与 outcome 只表示训练负荷；不得由执行历史声称能力提升。
不确定或没有可执行选择时返回不符合 Schema 的空结果，让程序安全拒绝，不得编造。
严格输出符合提供 JSON Schema 的单一 JSON object。
""".strip()

TRAINING_PLANNING_PROMPT_V1_SHA256 = (
    "a1f48ab055bb02c44e982925aa37f8621e6f878023d26781db1eb7ed90caf5dd"
)

TRAINING_PLANNING_PROMPT_V1 = TrainingPlanningPrompt(
    version="training-plan-v1",
    system_prompt=_TRAINING_PLANNING_V1_TEXT,
    content_sha256=TRAINING_PLANNING_PROMPT_V1_SHA256,
)


def get_training_planning_prompt(version: str) -> TrainingPlanningPrompt:
    if version != TRAINING_PLANNING_PROMPT_V1.version:
        raise ValueError("unsupported training planning prompt version")
    return TRAINING_PLANNING_PROMPT_V1


__all__ = [
    "GROWTH_ANALYSIS_PROMPT_V1",
    "GROWTH_ANALYSIS_PROMPT_V1_SHA256",
    "GrowthAnalysisPrompt",
    "get_growth_analysis_prompt",
    "TRAINING_PLANNING_PROMPT_V1",
    "TRAINING_PLANNING_PROMPT_V1_SHA256",
    "TrainingPlanningPrompt",
    "get_training_planning_prompt",
]
