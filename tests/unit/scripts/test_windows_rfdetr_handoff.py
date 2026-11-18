from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[3]
WINDOWS = ROOT / "scripts" / "algorithm" / "windows"
INSTALLER = WINDOWS / "install-rfdetr.ps1"
RUNNER = WINDOWS / "run-rfdetr-synthetic-smoke.ps1"
UNINSTALLER = WINDOWS / "uninstall-rfdetr.ps1"
SELF_TEST = WINDOWS / "test-rfdetr-handoff.ps1"
REGISTRY = ROOT / "configs" / "algorithm" / "windows" / "raytheon-rfdetr-smoke.json"
GITATTRIBUTES = ROOT / ".gitattributes"


def _script(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_frozen_registry_is_the_only_package_weight_and_config_identity_source() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    installer = _script(INSTALLER)
    runner = _script(RUNNER)

    assert registry["packages"] == {
        "pypi_index": "https://pypi.org/simple",
        "pytorch_index": "https://download.pytorch.org/whl/cu126",
        "rfdetr": "rfdetr[train,onnx]==1.9.0",
        "torch": "torch==2.13.0+cu126",
        "torchvision": "torchvision==0.28.0+cu126",
    }
    assert registry["text_hash_mode"] == "sha256-normalized-utf8-lf"
    assert registry["weight"]["sha256"] == (
        "6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3"
    )
    assert registry["smoke"]["config_sha256"] == (
        "c3b2633fef80a9da6ebeac20b120f80561cf3fab760f2df058090decf12a57c5"
    )
    assert "package-plan" in installer
    assert "verify-inputs" in runner
    assert "WeightSha256" not in runner
    assert "--weight-sha256" not in runner
    assert "--config" not in runner


def test_venv_identity_is_verified_before_any_pip_mutation() -> None:
    installer = _script(INSTALLER)

    validate_position = installer.index('"validate-venv"')
    first_pip_position = installer.index('"pip", "install"')
    assert validate_position < first_pip_position
    assert "$venvProbeRaw = & $venvPython $helper interpreter-probe" in installer
    assert "$baseProbeRaw = & py -3.12 $helper interpreter-probe" in installer
    assert "$venvProbeRaw | Set-Content -LiteralPath $venvProbePath" in installer
    assert '"validate-venv", "--venv", $venvRoot, "--probe", $venvProbePath' in installer
    assert "$diagnosticsRaw = & $venvPython $helper runtime-diagnostics" in installer
    assert "$diagnostics = $diagnosticsRaw | ConvertFrom-Json" in installer
    assert "$diagnostics.torch -ne ($packagePlan.torch" in installer
    assert "ReparsePoint" in installer


def test_runner_classifies_json_not_stderr_and_uses_unique_directories() -> None:
    runner = _script(RUNNER)
    installer = _script(INSTALLER)

    assert "classify-smoke" in runner
    assert '"--capture-manifest", $CaptureManifestPath' in runner
    assert '"--native-exit-code", "22"' in runner
    assert "exit $classifiedExit" in runner
    assert "DetectorOutOfMemory" not in runner
    assert "-match" not in runner
    assert "create-artifact" in runner
    assert "create-artifact" in installer
    assert "yyyyMMdd-HHmmss" not in runner + installer
    assert "New-Item -ItemType Directory -Path $workspace -Force" not in runner
    assert "New-Item -ItemType Directory -Path $runsRoot -Force" not in runner


def test_classification_consumes_validated_capture_manifest() -> None:
    runner = _script(RUNNER)

    assert "[string]$CaptureManifestPath" in runner
    assert '"--capture-manifest", $CaptureManifestPath' in runner
    assert "& py -3.12 $helper @classificationArguments" in runner
    assert "read-capture-exit" in runner


def test_all_required_native_evidence_uses_python_raw_capture_boundary() -> None:
    runner = _script(RUNNER)
    installer = _script(INSTALLER)
    self_test = _script(SELF_TEST)

    assert "function Invoke-Captured" in runner
    assert "function Invoke-Captured" in installer
    assert "function Invoke-Captured" in self_test
    assert runner.count("Invoke-Captured ") >= 5
    assert installer.count("Invoke-Captured ") >= 4
    assert self_test.count("Invoke-Captured ") >= 4
    assert "cuda-before.capture.json" in runner
    assert "smoke.capture.json" in runner
    assert "vram-before.capture.json" in runner
    assert "vram-after.capture.json" in runner
    assert "pip-freeze.capture.json" in installer
    assert "pip-check.capture.json" in installer
    assert "nvidia-smi.capture.json" in installer
    assert "native-warning.capture.json" in self_test
    assert "native-exit.capture.json" in self_test
    assert "spaced-argument.capture.json" in self_test
    assert "Start-Process" not in runner + installer + self_test


def test_checked_uninstaller_refuses_ambiguous_target_before_remove() -> None:
    script = _script(UNINSTALLER)

    validation = script.index("validate-uninstall")
    removal = script.index("Remove-Item -LiteralPath $venvRoot")
    assert validation < removal
    assert "pyproject.toml" in script
    assert "pyvenv.cfg" in script
    assert ".venv-rfdetr" in script
    assert "ReparsePoint" in script


def test_raytheon_native_self_test_covers_required_powershell_boundaries() -> None:
    script = _script(SELF_TEST)

    assert '$quoted = "argument with spaces"' in script
    assert "[char]0x" not in script
    assert "Start-Transcript" in script
    assert '"echo-argument", "--value", $quoted' in script
    assert '"exit-with-code", "--code", "7"' in script
    assert "native-warning-probe" in script
    assert "Invoke-Captured" in script
    assert "nonfatal native stderr warning" in script
    assert 'status -ne "PASS"' in script
    assert '$ErrorActionPreference -ne "Stop"' in script
    assert "cleanup-sentinel.tmp" in script
    finally_position = script.index("finally")
    remove_position = script.index("Remove-Item -LiteralPath $cleanupSentinel", finally_position)
    cleanup_assertion = script.index("$cleanupStillExists", finally_position)
    stop_position = script.index("Stop-Transcript", finally_position)
    append_position = script.index("Add-Content -LiteralPath $transcriptPath", finally_position)
    assert remove_position < cleanup_assertion
    assert stop_position < append_position


def test_all_windows_powershell_sources_are_ascii_and_structurally_balanced() -> None:
    scripts = sorted(WINDOWS.glob("*.ps1"))

    assert set((INSTALLER, RUNNER, SELF_TEST, UNINSTALLER)).issubset(scripts)
    for path in scripts:
        payload = path.read_bytes()
        assert payload and all(byte < 128 for byte in payload), path
        assert not payload.startswith(b"\xef\xbb\xbf"), path
        source = payload.decode("ascii")
        assert source.count("@'") == source.count("'@"), path
        assert source.count("{") == source.count("}"), path
        assert source.count("(") == source.count(")"), path
        for line in source.splitlines():
            if "`" in line:
                assert line.endswith("`"), (path, line)


def test_no_windows_powershell_script_passes_python_source_with_dash_c() -> None:
    dash_c_option = re.compile(r"(?<![A-Za-z0-9_-])-c(?![A-Za-z0-9_-])")

    for path in sorted(WINDOWS.glob("*.ps1")):
        source = _script(path)
        assert dash_c_option.search(source) is None, path

    for path in (INSTALLER, RUNNER, UNINSTALLER):
        source = _script(path)
        assert "interpreter-probe" in source or path == SELF_TEST
    assert "runtime-diagnostics" in _script(INSTALLER)
    assert "cuda-preflight" in _script(RUNNER)


def test_rfdetr_capture_entrypoints_do_not_use_powershell_text_redirection() -> None:
    # This raw-evidence contract belongs to the four shipped RF-DETR entry points.
    # Unrelated recording wrappers have a separate human-readable parent-log contract.
    native_redirection = re.compile(r"(?<![A-Za-z0-9])[12]>")

    for path in (INSTALLER, RUNNER, SELF_TEST, UNINSTALLER):
        source = _script(path)
        assert native_redirection.search(source) is None, path
        assert "Out-File" not in source, path


def test_frozen_registry_and_config_are_forced_to_lf_by_git() -> None:
    attributes = _script(GITATTRIBUTES)

    assert "/configs/algorithm/windows/raytheon-rfdetr-smoke.json text eol=lf" in attributes
    assert "/configs/algorithm/training/rfdetr-single-batch-smoke.yaml text eol=lf" in attributes
