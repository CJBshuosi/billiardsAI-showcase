"""Static architecture boundaries for Story 5.4 rules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
RULES = ROOT / "src" / "billiards_ai" / "cloud" / "engine" / "rules.py"
STORY = ROOT / "_bmad-output" / "implementation-artifacts" / "5-4-可追踪规则建议.md"


def test_rules_module_has_no_raw_visualization_ideal_geometry_media_or_provider_boundary():
    tree = ast.parse(RULES.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    attributes = {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert not any(
        token in module
        for module in imported
        for token in (
            "visualization",
            "ideal_path",
            "geometry",
            "perception",
            "pooltool",
            "media",
            "provider",
            "http",
            "requests",
        )
    )
    assert attributes.isdisjoint(
        {"trajectory", "raw_observation", "ball_positions_before", "ball_positions_after"}
    )
    assert names.isdisjoint({"trajectory", "raw_observation", "pooltool", "llm", "provider"})


def test_story_records_selector_as_reused_unchanged_baseline_scope():
    story = STORY.read_text(encoding="utf-8")

    assert "复用 Story 5.1 基线已有的 package-internal 纯 selector" in story
    assert (
        "NO CHANGE `src/billiards_ai/cloud/engine/typical_shot_selector.py`"
        "（基线已存在可复用 package-internal 纯 selector，本 Story 不修改）"
    ) in story
