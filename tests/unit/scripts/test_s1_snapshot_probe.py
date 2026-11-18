"""The Windows readiness probe must fail closed without real model assets."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.lab.field_assistance.test_runtime_resources import (
    assets as runtime_assets,  # noqa: F401
)

SCRIPT = Path("scripts/development/s1_snapshot_probe.py")


def load_probe():
    assert SCRIPT.is_file(), "real-model component probe is missing"
    spec = importlib.util.spec_from_file_location("s1_snapshot_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_does_not_claim_full_acceptance_or_load_optional_models():
    assert SCRIPT.is_file(), "real-model component probe is missing"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert "component readiness" in result.stdout
    assert "not full functional acceptance" in result.stdout


def test_missing_real_assets_cannot_be_replaced_by_a_synthetic_observer(tmp_path, request):
    module = load_probe()
    assets = request.getfixturevalue("runtime_assets")

    # Use this repository's real resource inspector. A directory by itself
    # cannot satisfy detector/identity manifests and fixed weight inventories.
    assets.detector_training_run = tmp_path / "missing-real-detector"
    with pytest.raises((OSError, ValueError), match="missing|resource|directory"):
        module.inspect_and_probe(vars(assets) | {"scenario_video": str(tmp_path / "unused.avi")})


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "terminal_record_failed",
        "job_close_error",
        "source",
        "result_missing",
        "diagnostic_missing",
        "diagnostic_invalid",
        "diagnostic_overflow",
        None,
    ],
)
def test_controller_honors_each_independent_completion_boundary(tmp_path, monkeypatch, fault):
    module = load_probe()
    from scripts.development import blackbox_validation

    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({name: str(tmp_path / name) for name in module.PATHS}))
    output = tmp_path / "run"
    source_changed = False
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(module.sources, "pin", lambda _: {"bytes": 1, "sha256": "b" * 64})
    monkeypatch.setattr(
        module.sources, "verify", lambda _: {"commit": ("c" if source_changed else "a") * 40}
    )

    def owned(*args, **kwargs):
        nonlocal source_changed
        # This tests the controller result boundary only; no child/model runs.
        if fault != "result_missing":
            (output / "local-result.json").write_text(
                json.dumps({"model_component_completed": True, "within_calculation_budget": True})
            )
        result = {"status": "PASS", "cleanup_confirmed": True, "worker_exit_code": 0}
        from scripts.development.s1_probe_diagnostics import ProbeDiagnostics

        run_id = json.loads((output / "context.json").read_bytes())["run_id"]
        if fault != "diagnostic_missing":
            with ProbeDiagnostics(output, run_id) as trace:
                with trace.stage("result_write"):
                    pass
            if fault == "diagnostic_invalid":
                (output / "local-stages.json").write_text('{"private": "do not export"}')
            if fault == "diagnostic_overflow":
                receipt = json.loads((output / "local-stages.json").read_bytes())
                receipt["events"][0]["at_seconds"] = 10**400
                (output / "local-stages.json").write_text(json.dumps(receipt))
        if fault == "source":
            source_changed = True
        elif fault == "status":
            result["status"] = "FAIL"
        elif fault in {"terminal_record_failed", "job_close_error"}:
            result[fault] = True
        return result

    monkeypatch.setattr(
        blackbox_validation,
        "load_ownership",
        lambda _: SimpleNamespace(WindowsAPI=lambda: None, run_owned=owned),
    )
    code = module.control(
        SimpleNamespace(
            config=config,
            output_root=output,
            worker_python=tmp_path / "python.exe",
            ownership=tmp_path / "ownership.py",
        )
    )
    assert (code == 0) is (fault is None), "each boundary must independently reject failure"
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["full_functional_acceptance"] == "NOT_RUN"


def test_child_retains_work_failure_stage_and_both_source_checks(tmp_path, monkeypatch):
    import io

    from scripts.development.s1_probe_diagnostics import read_diagnostic_summary

    module = load_probe()
    run_id = "1" * 32
    (tmp_path / "context.json").write_text(
        json.dumps({"run_id": run_id, "inputs": {}, "source_lock": {"commit": "a" * 40}})
    )
    monkeypatch.setattr(
        module.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(("S1-PROBE-RELEASE " + run_id + "\n").encode())),
    )
    monkeypatch.setattr(module.sources, "verify", lambda _: {"commit": "a" * 40})

    def failing_inspection(config, *, trace):
        with trace.stage("model_factory"):
            raise TimeoutError("private path must stay local")

    monkeypatch.setattr(module, "inspect_and_probe", failing_inspection)
    assert module.child(tmp_path) == 2
    summary = read_diagnostic_summary(tmp_path, run_id)
    assert summary["status"] == "AVAILABLE", "child must retain its entered/failed work stage"
    assert [(row["stage"], row["event"]) for row in summary["events"]] == [
        ("source_before", "enter"),
        ("source_before", "exit"),
        ("dependency_metadata", "enter"),
        ("dependency_metadata", "exit"),
        ("model_factory", "enter"),
        ("model_factory", "error"),
        ("source_after", "enter"),
        ("source_after", "exit"),
        ("result_write", "enter"),
        ("result_write", "exit"),
    ]
    result = json.loads((tmp_path / "local-result.json").read_bytes())
    assert result["model_component_completed"] is False
    assert result["error_type"] == "TimeoutError"
    assert "private" not in json.dumps(summary)


def test_real_resource_failure_is_identified_before_any_model_load(tmp_path, request):
    from scripts.development.s1_probe_diagnostics import ProbeDiagnostics, read_diagnostic_summary

    module = load_probe()
    assets = request.getfixturevalue("runtime_assets")
    assets.detector_training_run = tmp_path / "missing-real-detector"
    with ProbeDiagnostics(tmp_path, "1" * 32) as trace:
        with pytest.raises((OSError, ValueError)):
            module.inspect_and_probe(
                vars(assets) | {"scenario_video": str(tmp_path / "unused.avi")}, trace=trace
            )
    summary = read_diagnostic_summary(tmp_path, "1" * 32)
    assert summary["last_stage"] == "resources_before"
    assert summary["last_event"] == "error"
    assert all(row["stage"] != "model_factory" for row in summary["events"])


@pytest.mark.parametrize("mode", ["cold", "prepared"])
def test_prepared_component_separates_model_preparation_without_reclassifying_cold_budget(
    monkeypatch, mode
):
    module = load_probe()
    clock = [0.0]
    observer = object()
    loads = []

    def factory():
        loads.append(True)
        clock[0] += 15.5
        return observer

    def calculate(sink, make, *args):
        assert make() is observer
        clock[0] += 6.0
        sink.send({"route": {"real-unit-boundary": True}, "region": {"unit": True}})
        sink.close()

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    result, timing = module.calculate_component(
        calculate, factory, object(), object(), "open", "epoch", object(), mode=mode
    )
    assert loads == [True]
    assert result["route"]
    assert timing["calculation_seconds"] == 21.5
    assert timing["within_calculation_budget"] is False
    if mode == "prepared":
        assert timing["model_preparation_seconds"] == 15.5
        assert timing["snapshot_request_seconds"] == 6.0
        assert timing["within_snapshot_budget"] is True
    else:
        assert timing["model_preparation_seconds"] is None
        assert timing["snapshot_request_seconds"] is None
        assert timing["within_snapshot_budget"] is False


@pytest.mark.parametrize("change", [None, "route", "region", "budget", "component", "mode"])
def test_positive_precheck_cannot_pass_by_abstaining_or_borrowing_the_wrong_timing(change):
    module = load_probe()
    summary = dict(
        probe_mode="prepared",
        model_component_completed=True,
        within_calculation_budget=False,
        within_snapshot_budget=True,
        route_available=True,
        region_available=True,
    )
    if change == "mode":
        summary["probe_mode"] = "cold"
    elif change:
        key = {
            "route": "route_available",
            "region": "region_available",
            "budget": "within_snapshot_budget",
            "component": "model_component_completed",
        }[change]
        summary[key] = False
    assert module.component_passes(summary) is (change is None)


@pytest.mark.parametrize("mode", ["cold", "prepared"])
def test_numerical_preparation_uses_shared_boundary_and_separate_request_timing(monkeypatch, mode):
    module = load_probe()
    clock, calls = [0.0], []
    observer = object()

    def factory():
        calls.append("models")
        clock[0] += 15.5
        return observer

    def prepare():
        calls.append("numerics")
        clock[0] += 18.0

    def calculate(sink, make, *args):
        assert make() is observer
        calls.append("calculate")
        clock[0] += 6.0
        sink.send({"route": True, "region": True})

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    _, timing = module.calculate_component(
        calculate,
        factory,
        None,
        None,
        "open",
        "epoch",
        None,
        mode=mode,
        resource_preparer=prepare,
    )
    assert calls == ["numerics", "models", "calculate"]
    assert timing["calculation_seconds"] == 39.5
    assert timing["within_calculation_budget"] is False
    if mode == "prepared":
        assert timing["model_preparation_seconds"] == 15.5
        assert timing["resource_preparation_seconds"] == 33.5
        assert timing["snapshot_request_seconds"] == 6.0
        assert timing["within_snapshot_budget"] is True
    else:
        assert timing["resource_preparation_seconds"] is None
        assert timing["snapshot_request_seconds"] is None


def test_failed_numerical_preparation_never_reaches_calculation():
    module = load_probe()

    def prepare():
        raise RuntimeError("controlled numerical initialization failure")

    with pytest.raises(RuntimeError, match="numerical initialization"):
        module.calculate_component(
            lambda *args: pytest.fail("failed preparation cannot reach calculation"),
            object,
            None,
            None,
            "open",
            "epoch",
            None,
            mode="prepared",
            resource_preparer=prepare,
        )


def test_owned_probe_environment_can_import_real_physics_with_windows_home_semantics(
    tmp_path, monkeypatch
):
    """Exercise actual launch env and real engine import; no model or simulation."""
    module = load_probe()
    from scripts.development import blackbox_validation

    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({name: str(tmp_path / name) for name in module.PATHS}))
    output = tmp_path / "run"
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    monkeypatch.setenv("PRIVATE_CREDENTIAL", "must-not-inherit")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=dict(os.environ)))
    monkeypatch.setattr(module.sources, "pin", lambda _: {"bytes": 1, "sha256": "b" * 64})
    monkeypatch.setattr(module.sources, "verify", lambda _: {"commit": "a" * 40})
    launches = []

    def spawn(*args, **kwargs):
        launches.append(kwargs)
        return object()

    def owned(api, launch, *args, **kwargs):
        assert kwargs["deadline"] == 60 and kwargs["cleanup_seconds"] == 10
        launch()
        return {"status": "PASS", "cleanup_confirmed": True, "worker_exit_code": 2}

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        blackbox_validation,
        "load_ownership",
        lambda _: SimpleNamespace(WindowsAPI=lambda: None, run_owned=owned),
    )
    assert (
        module.control(
            SimpleNamespace(
                config=config,
                output_root=output,
                worker_python=tmp_path / "python.exe",
                ownership=tmp_path / "ownership.py",
            )
        )
        == 2
    )
    environment = launches[0]["env"]
    assert "PRIVATE_CREDENTIAL" not in environment
    for key in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "MPLCONFIGDIR"):
        assert Path(environment[key]).is_relative_to(output)
        assert Path(environment[key]).is_dir()
    # Restore Popen before the actual cold import. Only home-resolution semantics
    # are Windows-specific; installed pooltool and adapter sources remain real.
    monkeypatch.undo()
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        """
import ntpath, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
def windows_home(cls):
    value = ntpath.expanduser('~')
    if value.startswith('~'):
        print('WINDOWS_HOME_UNRESOLVED', flush=True)
        raise RuntimeError('Could not determine home directory.')
    return cls(value)
Path.home = classmethod(windows_home)
from billiards_ai.cloud.engine.pooltool_adapter import PooltoolIdealPathAdapter, pt
PooltoolIdealPathAdapter()._require_engine()
assert pt is not None
from pooltool.config.paths import HOME, CONFIG_DIR
assert HOME == Path(os.environ['USERPROFILE'])
assert CONFIG_DIR.is_relative_to(HOME)
assert CONFIG_DIR.is_dir()
print('REAL_ENGINE_IMPORT_WITH_OWNED_WINDOWS_HOME')
""",
        str(Path("src").resolve()),
    ]
    # Hold all other dependencies and cache paths fixed. Removing only the
    # profile reproduces an unavailable real adapter in a fresh process.
    missing_profile = {
        k: v for k, v in environment.items() if k not in {"USERPROFILE", "HOMEDRIVE", "HOMEPATH"}
    }
    failed = subprocess.run(
        command,
        env=missing_profile,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode != 0
    assert "WINDOWS_HOME_UNRESOLVED" in failed.stdout
    assert "EngineUnavailableError: pooltool import unavailable" in failed.stderr
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "REAL_ENGINE_IMPORT_WITH_OWNED_WINDOWS_HOME"
