"""Resolve current development links separately from dated client decisions."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
RESTART = ROOT / "PROJECT-RESTART.md"
SPINE = ROOT / (
    "_bmad-output/planning-artifacts/architecture/"
    "architecture-billiardsAI-2026-07-20/ARCHITECTURE-SPINE.md"
)
ENTRYPOINTS = (
    ROOT / "README.md",
    ROOT / "DEVELOPMENT.md",
    ROOT / "PROJECT-WORKFLOW.md",
    ROOT / "docs/README.md",
    ROOT / "_bmad-output/README.md",
    SPINE,
)


def current_entry(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        text = text.split("---", maxsplit=2)[2]
    return text.split("\n## 历史", maxsplit=1)[0]


def local_targets(path: Path, text: str) -> set[Path]:
    targets = set()
    for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        resolved = (path.parent / unquote(url.path)).resolve()
        assert resolved.is_relative_to(ROOT), (path, target)
        assert resolved.is_file(), (path, target)
        targets.add(resolved)
    return targets


def test_active_entrypoints_resolve_to_restart() -> None:
    historical_authorities = {
        ROOT / "docs/superpowers/specs/2026-08-01-algorithm-brain-product-roadmap-design.md",
        ROOT / "docs/superpowers/plans/2026-08-01-algorithm-brain-training.md",
    }
    for path in ENTRYPOINTS:
        targets = local_targets(path, current_entry(path))
        assert RESTART in targets, path
        assert not historical_authorities.intersection(targets), path


def test_restart_keeps_current_design_and_open_acceptance_links() -> None:
    targets = local_targets(RESTART, current_entry(RESTART))
    required = {
        ROOT / "AGENTS.md",
        ROOT / "docs/superpowers/specs/2026-11-06-recording-first-product-requirements-design.md",
        ROOT / "docs/superpowers/specs/2026-11-10-durable-recording-backend-design.md",
        ROOT / "docs/superpowers/plans/2026-11-06-recording-first-product-implementation.md",
        ROOT / "docs/operations/dev-integration-receipt-2026-11-13.md",
    }
    assert required <= targets


def test_architecture_and_development_separate_historical_scope() -> None:
    decision = "miniprogram-client-reentry-2026-08-26.md"
    for path in (ROOT / "DEVELOPMENT.md", SPINE):
        text = path.read_text(encoding="utf-8")
        assert "\n## 历史" in text, path
        assert decision not in current_entry(path), path
        assert decision in text.split("\n## 历史", maxsplit=1)[1], path


def test_current_validation_links_resolve() -> None:
    matrix = ROOT / "docs/engineering/development-verification-matrix.md"
    for path in (ROOT / "README.md", ROOT / "DEVELOPMENT.md"):
        assert matrix in local_targets(path, current_entry(path)), path
    targets = local_targets(matrix, matrix.read_text(encoding="utf-8"))
    assert RESTART in targets
    assert ROOT / ".github/workflows/pr-gate.yml" in targets


def test_current_workflow_delegates_commands_to_development() -> None:
    workflow = ROOT / "PROJECT-WORKFLOW.md"
    current = current_entry(workflow)
    targets = local_targets(workflow, current)
    assert ROOT / "DEVELOPMENT.md" in targets
    assert ROOT / "docs/engineering/development-verification-matrix.md" in targets
    assert not re.findall(r"^```(?:bash|sh|shell)\s*$", current, re.MULTILINE)
