from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WINDOWS = ROOT / "scripts" / "algorithm" / "windows"


def test_supported_windows_entrypoints_exist_without_research_handoff_inventory() -> None:
    entries = (
        "scripts/algorithm/windows/run-mainline1-shot-local-video.ps1",
        "scripts/algorithm/windows/run-shot-local-regression-suite.ps1",
        "scripts/algorithm/windows/replay-lightweight-realtime-chain.ps1",
        "scripts/algorithm/windows/reobserve-mainline1-shot-local-artifact.ps1",
        "scripts/algorithm/windows/evaluate-shot-local-timeline.ps1",
    )

    assert all((ROOT / entry).is_file() for entry in entries)
    assert not (WINDOWS / "lightweight-realtime-chain-handoff-files.txt").exists()


def test_raytheon_runbook_exposes_only_single_line_powershell_commands() -> None:
    runbook = ROOT / "docs" / "operations" / "lightweight-realtime-chain-raytheon.md"
    blocks = re.findall(
        r"```powershell\n(.*?)\n```",
        runbook.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )

    assert len(blocks) >= 3
    assert all(len(block.splitlines()) == 1 for block in blocks)
    assert any("run-mainline1-shot-local-video.ps1" in block for block in blocks)
    assert all("run-shot-local-event-tracker.ps1" not in block for block in blocks)
