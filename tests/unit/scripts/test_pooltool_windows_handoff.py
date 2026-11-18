from pathlib import Path

SCRIPT = Path("scripts/algorithm/windows/install-pooltool-smoke.ps1")


def test_windows_handoff_is_isolated_and_runs_only_synthetic_smoke() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "py -3.12" in text
    assert "--no-index" in text
    assert "--find-links" in text
    assert ".venv-pooltool" in text
    assert "_handoff\\wheelhouse" in text
    assert text.index("$env:PYTHONPATH") < text.index("pooltool_packet_handoff.py")
    assert "pooltool_real_smoke.py" in text
    assert "pooltool_native_smoke.py" in text
    assert "run_mainline1_guidance.py" in text
    assert "--context-group" in text
    assert '"solid"' in text
    assert '"--physics" "pooltool"' in text
    assert "[Parameter(Mandatory = $true)][string]$RfdetrPython" in text
    assert "$RfdetrPython = (Resolve-Path -LiteralPath $RfdetrPython).Path" in text
    assert 'Join-Path $Venv "Lib\\site-packages"' in text
    assert "CloudBase" not in text
    assert "real-video" not in text
