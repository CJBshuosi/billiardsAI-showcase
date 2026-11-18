"""Static AD-13 boundaries for Story 5.3."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src" / "billiards_ai"


def _attribute_reads(path: Path, attribute: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != attribute:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.keyword) and parent.arg == attribute:
            continue
        if isinstance(parent, ast.Assign) and node in parent.targets:
            continue
        reads.append(node.lineno)
    return reads


def test_trajectory_reads_are_confined_to_review_and_visualization_consumers():
    allowed = {
        Path("cloud/perception/finalizer.py"),
        Path("shared/models/observation.py"),
        Path("cloud/review/service.py"),
        Path("cloud/review/visualization_repository.py"),
        Path("cloud/engine/visualization.py"),
    }
    offenders = {}
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC)
        reads = _attribute_reads(path, "trajectory")
        if reads and relative not in allowed:
            offenders[str(relative)] = reads
    assert offenders == {}


def test_scoring_statistics_rules_profile_and_reporting_do_not_read_protected_fields():
    protected_consumers = (
        SRC / "cloud" / "engine" / "aggregation.py",
        SRC / "cloud" / "engine" / "scoring.py",
        SRC / "cloud" / "engine" / "reporting.py",
        SRC / "cloud" / "engine" / "typical_shot_selector.py",
    )
    for path in protected_consumers:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert {"trajectory", "raw_observation"}.isdisjoint(names), path


def test_visualization_boundary_never_reads_raw_observation():
    for relative in (
        Path("cloud/engine/visualization.py"),
        Path("cloud/review/visualization_repository.py"),
    ):
        path = SRC / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "raw_observation"
            for node in ast.walk(tree)
        )


def test_visualization_does_not_import_media_pooltool_reporting_or_persistence():
    path = SRC / "cloud" / "engine" / "visualization.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert all(
        forbidden not in module
        for module in imports
        for forbidden in ("pooltool", ".media", ".reporting", ".privacy", ".db")
    )
