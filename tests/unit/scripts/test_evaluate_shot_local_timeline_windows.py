from pathlib import Path


def test_windows_timeline_evaluator_is_offline_and_explicit() -> None:
    script = (
        Path("scripts/algorithm/windows/evaluate-shot-local-timeline.ps1")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "evaluate_shot_local_timeline.py" in script
    assert "timeline" in script
    assert "truth" in script
    assert "tolerance" in script
    assert "pip install" not in script
    assert "invoke-webrequest" not in script
