from pathlib import Path


def test_windows_wrapper_forwards_samples_and_output() -> None:
    wrapper = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "windows"
        / "calibrate-mainline1-usb-lens.ps1"
    )
    source = wrapper.read_text(encoding="ascii")

    assert "calibrate_mainline1_usb_lens.py" in source
    assert '"--samples", $Samples' in source
    assert '"--output", $Output' in source
    assert "-ExecutionPolicy" not in source
