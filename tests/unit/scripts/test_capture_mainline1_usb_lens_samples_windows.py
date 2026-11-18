from pathlib import Path


def test_windows_wrapper_forwards_device_and_output() -> None:
    wrapper = (
        Path(__file__).parents[3]
        / "scripts"
        / "algorithm"
        / "windows"
        / "capture-mainline1-usb-lens-samples.ps1"
    )
    source = wrapper.read_text(encoding="ascii")

    assert "capture_mainline1_usb_lens_samples.py" in source
    assert '"--device-locator", $DeviceLocator' in source
    assert '"--runtime-index", $RuntimeIndex' in source
    assert '"--output", $Output' in source
    assert '"--target-samples", $TargetSamples' in source
