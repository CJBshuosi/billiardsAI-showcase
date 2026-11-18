from pathlib import Path


def test_windows_calibration_wrapper_stays_local_and_requires_explicit_video() -> None:
    script = (
        Path("scripts/algorithm/windows/calibrate-mainline1-video.ps1")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert "calibrate_mainline1_video.py" in script
    assert "[parameter(mandatory = $true)] [string]$video" in script
    assert "[parameter(mandatory = $true)] [string]$capturesetup" in script
    assert "mainline1-calibrations" in script
    assert 'deviceid = ""' in script
    assert '"--device-id"' in script
    assert "detsam2-fixed-camera" not in script
    assert "pip install" not in script
    assert "invoke-webrequest" not in script
    assert "cloud" not in script
