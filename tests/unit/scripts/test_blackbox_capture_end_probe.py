import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from billiards_ai.lab.mainline1_session_recording import CommittedRecordingSource
from scripts.development import blackbox_capture_end_probe as probe

BASE = Path(__file__).resolve().parents[3]
assert Path(probe.__file__).is_relative_to(BASE)
SOURCE = "a" * 40
INPUT = "b" * 64


def setup_state():
    receipt = {
        "contract": "recording-capture-end/v1",
        "session_id": "session",
        "source_epoch_id": "epoch",
        "capture_status": "STOPPED",
        "cutoff": {"frame_id": 15879},
        "accepted_at_utc": "fixed",
        "tail_status": "SAVED",
    }
    state = NS(
        session_id="session",
        projection_source=NS(
            source_epoch_id="epoch",
            runtime=NS(_analysis_source=object.__new__(CommittedRecordingSource)),
            _runtime_exited=NS(is_set=lambda: False),
        ),
        record_store=NS(
            get_recording_capture_end=lambda session: dict(receipt),
            load_session=lambda session: NS(status=NS(value="ended")),
        ),
    )
    report = {
        "emitted_frames": 15880,
        "capture_stop_seconds": 0.1,
        "sampled_detection_backlog_high_water": 1000,
    }
    return state, receipt, report


@pytest.mark.parametrize(
    "case",
    ["wrong_cutoff", "wrong_session", "not_disk", "not_ended", "slow", "nan", "changed_cutoff"],
)
def test_bad_end_receipt_or_state_never_passes(case):
    state, receipt, report = setup_state()
    if case == "wrong_cutoff":
        receipt["cutoff"]["frame_id"] = 20
    if case == "wrong_session":
        receipt["session_id"] = "wrong"
    if case == "not_disk":
        state.projection_source.runtime._analysis_source = object()
    if case == "not_ended":
        state.record_store.load_session = lambda session: NS(status=NS(value="open"))
    if case == "slow":
        report["capture_stop_seconds"] = 2
    if case == "nan":
        report["capture_stop_seconds"] = float("nan")
    if case == "changed_cutoff":
        state.record_store.get_recording_capture_end = lambda session: dict(
            receipt, cutoff={"frame_id": 0}
        )
    with pytest.raises(ValueError):
        probe.verify_end(state, receipt, report)


@pytest.mark.parametrize("fault", ["none", "cleanup", "other_failure", "wrong_source", "media"])
def test_scoped_completion_is_not_a_fullgame_pass(tmp_path, fault):
    state, receipt, report = setup_state()
    original = object()
    game = NS(_wait_for_background_completion=original)

    def main(argv):
        report.update(
            source_commit=SOURCE,
            source_sha256=INPUT,
            product_pass=False,
            cleanup_confirmed=fault != "cleanup",
            capture_stop=receipt,
        )
        try:
            game._wait_for_background_completion(
                None, state, receipt, deadline=time.monotonic() + 600, report=report
            )
        except Exception as exc:
            report["failure"] = {"type": type(exc).__name__}
        if fault == "other_failure":
            report["failure"] = {"type": "OSError"}
        if fault == "wrong_source":
            report["source_commit"] = "c" * 40
        if fault == "media":
            report["media"] = [{"private": "unexpected"}]
        (tmp_path / "recording").mkdir()
        (tmp_path / "recording/report.json").write_text(json.dumps(report))
        return 1

    game.main = main
    result = probe.run(game, [], tmp_path, SOURCE, INPUT)
    assert game._wait_for_background_completion is original
    assert result == (0 if fault == "none" else 1)
    value = json.loads((tmp_path / "capture-end-probe.json").read_text())
    safe = probe.export(value, tmp_path.name, SOURCE, INPUT)
    assert safe["postgame_checks"] == "NOT_RUN" and safe["fullgame_product_pass"] is False
    assert safe["capture_end_status"] == ("PASS" if fault == "none" else "FAIL")
    if fault == "none":
        assert safe["observation"]["backpressure_case_covered"] is True


def test_export_refuses_claim_without_required_evidence():
    value = {
        "contract": probe.CONTRACT,
        "run_id": "test",
        "source_commit": SOURCE,
        "input_sha256": INPUT,
        "scope": "CAPTURE_END_ONLY",
        "postgame_checks": "NOT_RUN",
        "fullgame_product_pass": False,
        "capture_end_status": "PASS",
        "cleanup_confirmed": True,
        "observation": {},
    }
    with pytest.raises(ValueError):
        probe.export(value, "test", SOURCE, INPUT)


@pytest.mark.parametrize("times,deadline", [([0, 0, 11], 600), ([0, 0, 2], 1)])
def test_slow_tail_read_cannot_turn_expired_budget_into_pass(
    tmp_path, monkeypatch, times, deadline
):
    state, receipt, report = setup_state()
    game = NS(_wait_for_background_completion=object())
    clock = iter(times)
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))

    def main(argv):
        report.update(
            source_commit=SOURCE,
            source_sha256=INPUT,
            product_pass=False,
            cleanup_confirmed=True,
            capture_stop=receipt,
        )
        try:
            game._wait_for_background_completion(
                None, state, receipt, deadline=deadline, report=report
            )
        except Exception as exc:
            report["failure"] = {"type": type(exc).__name__}
        (tmp_path / "recording").mkdir()
        (tmp_path / "recording/report.json").write_text(json.dumps(report))
        return 1

    game.main = main
    assert probe.run(game, [], tmp_path, SOURCE, INPUT) == 1
    assert report["failure"]["type"] == "TimeoutError"


@pytest.mark.parametrize(
    "controller",
    [
        {"status": "TIMEOUT", "exit_code": 3, "owned_cleanup": True},
        {"status": "ERROR", "exit_code": 5, "owned_cleanup": False},
        {"status": "PASS", "exit_code": 0, "owned_cleanup": False},
    ],
)
def test_earlier_ack_cannot_override_controller_failure(controller):
    assert not probe.controller_passed(controller, {"capture_end_status": "PASS"})
