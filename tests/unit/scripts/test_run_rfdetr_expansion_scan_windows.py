from pathlib import Path


def test_windows_expansion_scan_is_offline_and_uses_frozen_split() -> None:
    script = (
        Path("scripts/algorithm/windows/run-rfdetr-expansion-scan.ps1")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "detsam2-training-split-v1.json" in script
    assert "run_rfdetr_expansion_scan.py" in script
    assert "checkpoint_best.pth" in script
    assert "get-filehash" in script
    assert "locked-test" not in script
    assert "pip install" not in script
    assert "invoke-webrequest" not in script
