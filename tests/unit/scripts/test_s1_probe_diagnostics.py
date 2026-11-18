"""Keep evidence when a probe stalls, without running any model or Windows test."""

import importlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

RUN_ID = "1" * 32
HELPER = Path("scripts/development/s1_probe_diagnostics.py")


def diagnostics():
    assert HELPER.is_file(), "probe stage/stack diagnostics are missing"
    return importlib.import_module("scripts.development.s1_probe_diagnostics")


def test_killed_local_worker_retains_entered_stage_and_real_blocked_stack(tmp_path):
    diagnostics()
    code = """
import sys, threading, time
from pathlib import Path
from scripts.development.s1_probe_diagnostics import ProbeDiagnostics
with ProbeDiagnostics(Path(sys.argv[1]), '1' * 32, stack_at=time.monotonic() + .1) as trace:
    with trace.stage('model_factory'):
        threading.Event().wait()
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path)])
    try:
        stack = tmp_path / "local-stack.txt"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (not stack.exists() or stack.stat().st_size == 0):
            time.sleep(0.01)
        assert process.poll() is None, "the worker must still be blocked when evidence is read"
        assert stack.is_file() and "wait" in stack.read_text()
        state = json.loads((tmp_path / "local-stages.json").read_bytes())
        assert state["events"][-1]["stage"] == "model_factory"
        assert state["events"][-1]["event"] == "enter"
    finally:
        process.terminate()
        process.wait(timeout=5)
    summary = diagnostics().read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["status"] == "AVAILABLE"
    assert summary["last_stage"] == "model_factory"
    assert summary["last_event"] == "enter"
    assert summary["stack_capture_available"] is True
    assert str(tmp_path) not in json.dumps(summary)


def test_completed_worker_cancels_pending_stack_timer(tmp_path):
    diagnostics()
    code = """
import sys, threading, time
from pathlib import Path
from scripts.development.s1_probe_diagnostics import ProbeDiagnostics
with ProbeDiagnostics(Path(sys.argv[1]), '1' * 32, stack_at=time.monotonic() + .1) as trace:
    with trace.stage('result_write'):
        pass
threading.Event().wait(.2)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "local-stack.txt").read_bytes() == b""
    summary = diagnostics().read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["last_stage"] == "result_write"
    assert summary["last_event"] == "exit"
    assert summary["stack_capture_available"] is False


@pytest.mark.parametrize("original_fails", [False, True])
def test_failed_terminal_receipt_preserves_an_existing_backend_error(
    tmp_path, monkeypatch, original_fails
):
    module = diagnostics()
    failure = RuntimeError("original controlled inference failure")
    write = module.atomic_json

    def fail_terminal(path, value):
        if value["events"][-1]["event"] in {"error", "exit"}:
            raise OSError("controlled diagnostic disk error")
        return write(path, value)

    monkeypatch.setattr(module, "atomic_json", fail_terminal)
    expected = RuntimeError if original_fails else OSError
    with pytest.raises(expected) as raised:
        with module.ProbeDiagnostics(tmp_path, "a" * 32) as trace:
            with trace.stage("identity_inference"):
                if original_fails:
                    raise failure
    if original_fails:
        assert raised.value is failure
    assert trace._pending_exits == 0


def test_stage_error_is_retained_without_exception_text_or_private_paths(tmp_path):
    module = diagnostics()
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:
        with pytest.raises(ValueError, match="private"):
            with trace.stage("identity_inference"):
                raise ValueError("private C:/credential-and-model-path")
    summary = module.read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["last_event"] == "error"
    assert summary["last_stage"] == "identity_inference"
    assert summary["events"][-1]["error_type"] == "ValueError"
    assert "private" not in json.dumps(summary)
    assert "credential" not in (tmp_path / "local-stages.json").read_text()


@pytest.mark.parametrize(
    "fault", ["run", "stage", "event", "number", "big_integer", "extra", "oversized"]
)
def test_untrusted_diagnostic_receipts_are_rejected_not_exported(tmp_path, fault):
    module = diagnostics()
    receipt = {
        "contract": "s1-probe-stages/v1",
        "run_id": RUN_ID,
        "events": [{"stage": "model_factory", "event": "enter", "at_seconds": 0.0}],
    }
    event = receipt["events"][0]
    if fault == "run":
        receipt["run_id"] = "2" * 32
    elif fault == "stage":
        event["stage"] = "private-model-path"
    elif fault == "event":
        event["event"] = "assumed_complete"
    elif fault == "number":
        event["at_seconds"] = float("nan")
    elif fault == "big_integer":
        event["at_seconds"] = 10**400
    elif fault == "extra":
        event["private_config"] = "secret"
    path = tmp_path / "local-stages.json"
    path.write_text(json.dumps(receipt) if fault != "oversized" else " " * (32 * 1024 + 1))
    summary = module.read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary == {"status": "INVALID"}


def test_missing_diagnostics_are_unknown_and_stage_volume_is_bounded(tmp_path):
    module = diagnostics()
    assert module.read_diagnostic_summary(tmp_path, RUN_ID) == {"status": "NOT_AVAILABLE"}
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:
        with pytest.raises(ValueError, match="stage"):
            with trace.stage("C:/private-path"):
                pass
        for _ in range(32):
            with trace.stage("model_factory"):
                pass
        with pytest.raises(ValueError, match="budget"):
            with trace.stage("model_factory"):
                pass
    assert (tmp_path / "local-stages.json").stat().st_size <= 32 * 1024


def test_backend_wrapping_preserves_value_metadata_and_native_exception(tmp_path):
    module = diagnostics()

    class Backend:
        package_identity = "original identity"

        def infer(self, *, number):
            return {"original_result": number}

        def classify(self):
            raise RuntimeError("native failure")

    backend = Backend()
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:
        wrapped = module.CallStageProxy(backend, trace, "infer", "detector_inference")
        assert wrapped.package_identity == "original identity"
        assert wrapped.infer(number=7) == {"original_result": 7}
        wrapped_identity = module.CallStageProxy(backend, trace, "classify", "identity_inference")
        with pytest.raises(RuntimeError, match="native failure"):
            wrapped_identity.classify()
    events = module.read_diagnostic_summary(tmp_path, RUN_ID)["events"]
    assert [(row["stage"], row["event"]) for row in events] == [
        ("detector_inference", "enter"),
        ("detector_inference", "exit"),
        ("identity_inference", "enter"),
        ("identity_inference", "error"),
    ]


def test_nested_stage_exits_reserve_space_within_receipt_budget(tmp_path):
    from contextlib import ExitStack

    module = diagnostics()
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:
        with pytest.raises(ValueError, match="budget"):
            with ExitStack() as contexts:
                for _ in range(33):
                    contexts.enter_context(trace.stage("imports"))
    summary = module.read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["status"] == "AVAILABLE"
    assert len(summary["events"]) == 64
    assert summary["last_event"] == "error"


def test_parallel_model_and_physics_receipts_can_finish_in_either_order(tmp_path):
    module = diagnostics()
    model_entered, physics_entered, model_done = (threading.Event() for _ in range(3))
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:

        def models():
            with trace.stage("model_factory"):
                model_entered.set()
                assert physics_entered.wait(2)
            model_done.set()

        def physics():
            assert model_entered.wait(2)
            with trace.stage("physics_preparation"):
                physics_entered.set()
                assert model_done.wait(2)

        threads = [threading.Thread(target=target) for target in (models, physics)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
        with trace.stage("result_write"):
            pass
    summary = module.read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["status"] == "AVAILABLE"
    assert [(row["stage"], row["event"]) for row in summary["events"]] == [
        ("model_factory", "enter"),
        ("physics_preparation", "enter"),
        ("model_factory", "exit"),
        ("physics_preparation", "exit"),
        ("result_write", "enter"),
        ("result_write", "exit"),
    ]


def test_parallel_preparation_does_not_allow_unrelated_crossed_stages(tmp_path):
    receipt = {
        "contract": "s1-probe-stages/v1",
        "run_id": RUN_ID,
        "events": [
            {"stage": stage, "event": event, "at_seconds": index}
            for index, (stage, event) in enumerate(
                [
                    ("imports", "enter"),
                    ("planning", "enter"),
                    ("imports", "exit"),
                    ("planning", "exit"),
                ]
            )
        ],
    }
    (tmp_path / "local-stages.json").write_text(json.dumps(receipt))
    assert diagnostics().read_diagnostic_summary(tmp_path, RUN_ID) == {"status": "INVALID"}


def test_legacy_failed_model_retains_late_numeric_completion_during_source_check(tmp_path):
    # Historical concurrent receipts remain readable after sequential preparation.
    module = diagnostics()
    numeric_started, finish_numeric, numeric_done = (threading.Event() for _ in range(3))
    with module.ProbeDiagnostics(tmp_path, RUN_ID) as trace:

        def models():
            with trace.stage("model_factory"):
                assert numeric_started.wait(2)
                raise RuntimeError("controlled model preparation failure")

        def numeric():
            with trace.stage("physics_preparation"):
                numeric_started.set()
                assert finish_numeric.wait(2)
            numeric_done.set()

        thread = threading.Thread(target=numeric)
        thread.start()
        with pytest.raises(RuntimeError, match="controlled model preparation"):
            models()
        with trace.stage("source_after"):
            finish_numeric.set()
            assert numeric_done.wait(2)
        thread.join(2)
        assert not thread.is_alive()
        with trace.stage("result_write"):
            pass
    summary = module.read_diagnostic_summary(tmp_path, RUN_ID)
    assert summary["status"] == "AVAILABLE"
    assert summary["last_stage"] == "result_write"
    assert summary["last_event"] == "exit"
    assert any(r["stage"] == "model_factory" and r["event"] == "error" for r in summary["events"])


@pytest.mark.parametrize("failed", [False, True])
def test_only_failed_boot_allows_late_preparation_across_final_receipts(tmp_path, failed):
    pairs = [
        ("model_factory", "enter"),
        ("model_factory", "error" if failed else "exit"),
        ("source_after", "enter"),
        ("physics_preparation", "enter"),
        ("source_after", "exit"),
        ("result_write", "enter"),
        ("physics_preparation", "exit"),
        ("result_write", "exit"),
    ]
    events = []
    for index, (stage, event) in enumerate(pairs):
        row = {"stage": stage, "event": event, "at_seconds": index}
        if event == "error":
            row["error_type"] = "RuntimeError"
        events.append(row)
    (tmp_path / "local-stages.json").write_text(
        json.dumps(
            {
                "contract": "s1-probe-stages/v1",
                "run_id": RUN_ID,
                "events": events,
            }
        )
    )
    assert diagnostics().read_diagnostic_summary(tmp_path, RUN_ID)["status"] == (
        "AVAILABLE" if failed else "INVALID"
    )


def test_failed_preparation_does_not_allow_final_receipts_to_cross(tmp_path):
    pairs = [
        ("model_factory", "enter"),
        ("physics_preparation", "enter"),
        ("model_factory", "error"),
        ("source_after", "enter"),
        ("result_write", "enter"),
        ("source_after", "exit"),
        ("physics_preparation", "exit"),
        ("result_write", "exit"),
    ]
    events = []
    for index, (stage, event) in enumerate(pairs):
        row = {"stage": stage, "event": event, "at_seconds": index}
        if event == "error":
            row["error_type"] = "RuntimeError"
        events.append(row)
    (tmp_path / "local-stages.json").write_text(
        json.dumps(
            {
                "contract": "s1-probe-stages/v1",
                "run_id": RUN_ID,
                "events": events,
            }
        )
    )
    assert diagnostics().read_diagnostic_summary(tmp_path, RUN_ID) == {"status": "INVALID"}
