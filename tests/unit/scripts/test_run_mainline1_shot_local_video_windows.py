from pathlib import Path


def test_windows_shot_local_wrapper_is_local_and_requires_explicit_artifacts() -> None:
    script = (
        Path("scripts/algorithm/windows/run-mainline1-shot-local-video.ps1")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "run_mainline1_shot_local_video.py" in script
    assert "detectortrainingrun" in script
    assert "identitytrainingrun" in script
    assert "calibration" in script
    assert "capturesetup" in script
    assert "calibrationroot" in script
    assert "dinov2assets" in script
    assert "shotlocalconfig" in script
    assert "test-path -literalpath" in script
    assert "pip install" not in script
    assert "invoke-webrequest" not in script
    assert "cloud" not in script
    assert "get-childitem" not in script
    assert "lastwritetime" not in script
