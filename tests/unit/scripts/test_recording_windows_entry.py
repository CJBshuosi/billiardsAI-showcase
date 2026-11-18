import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.algorithm import recording_windows_entry as entry


def _complete_report():
    return dict(
        source_commit="a" * 40,
        source_sha256="b" * 64,
        windows_run=True,
        product_pass=True,
        model_pass=True,
        field_pass=True,
        cleanup_confirmed=True,
        emitted_frames=15880,
        capture_health=dict(received_frames=15880),
        consumer_health=dict(received_frames=15880),
        analysis_jobs=0,
        saved_outbox=0,
        raw_recording=dict(decoded_frames=15880, first_frame_id=0, last_frame_id=15879),
        media=[dict(shot_id=f"shot-{i:04d}") for i in range(1, 29)],
        differences=[],
        private_path="do not export",
        commands=["do not export"],
    )


def test_summary_preserves_separate_gates_and_rejects_inconsistent_pass():
    manifest = dict(source_commit="a" * 40, input_sha256="b" * 64)
    owned = dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)
    report = _complete_report()
    summary = entry.summarize(report, owned, manifest)
    assert summary["product_pass"]
    assert not summary["model_pass"] and not summary["field_pass"] and not summary["browser_pass"]
    assert "private_path" not in summary and "commands" not in summary
    report["raw_recording"]["private_path"] = "do not export"
    assert "private_path" not in entry.summarize(report, owned, manifest)["raw_recording"]
    report["emitted_frames"] = 1
    assert not entry.summarize(report, owned, manifest)["product_pass"]
    report = _complete_report()
    report["capture_health"]["reason"] = "private source failure"
    assert not entry.summarize(report, owned, manifest)["product_pass"]
    report["source_commit"] = "c" * 40
    assert entry.summarize(report, owned, manifest)["failure_type"] == "REPORT_IDENTITY_MISMATCH"


def test_worker_requires_release_before_any_root_inspection(monkeypatch):
    monkeypatch.setattr(entry.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"wrong\n")))
    monkeypatch.setattr(
        entry, "checked_root", lambda *a: pytest.fail("unreleased filesystem access")
    )
    with pytest.raises(ValueError, match="owned release"):
        entry.main(["worker", "/untrusted", "a" * 64])


def test_summary_retains_typed_encoder_failure_without_urls_or_owners():
    report = _complete_report()
    report.update(
        product_pass=False,
        media=[],
        review=dict(
            clips=[
                dict(
                    shot_id="shot-0001",
                    error_code="MEDIA_H264_ENCODER_UNAVAILABLE",
                    terminal_failure=True,
                    media_url="private url",
                    actor_role="private owner",
                )
            ]
        ),
    )
    result = entry.summarize(
        report,
        dict(status="FAIL", cleanup_confirmed=True),
        dict(source_commit="a" * 40, input_sha256="b" * 64),
    )
    assert result["media_errors"] == [
        dict(shot_id="shot-0001", error_code="MEDIA_H264_ENCODER_UNAVAILABLE")
    ]


def test_parent_environment_has_only_owned_cache_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-copy")
    result = entry.child_environment(tmp_path)
    assert "OPENAI_API_KEY" not in result
    for name in ("TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        assert result[name].startswith(str(tmp_path))


def test_fast_exit_cannot_bypass_log_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    monkeypatch.setattr(entry, "LOG_LIMIT", 16)
    (tmp_path / "recording").mkdir()
    (tmp_path / "recording/report.json").write_text(json.dumps(_complete_report()))

    def spawn(*args, **kwargs):
        kwargs["stdout"].write(b"x" * 17)
        return object()

    def owned(api, spawn, persist, metadata, release, **kwargs):
        spawn()  # Worker exits before the controller's next sleep checkpoint.
        return dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)

    monkeypatch.setattr(entry.subprocess, "Popen", spawn)
    monkeypatch.setitem(
        sys.modules, "ownership", SimpleNamespace(WindowsAPI=object, run_owned=owned)
    )
    manifest = dict(source_commit="a" * 40, input_sha256="b" * 64)
    assert entry.launch(tmp_path, "f" * 64, manifest) == 1
    summary = json.loads((tmp_path / "output/summary.json").read_bytes())
    assert not summary["product_pass"]
    assert summary["failure_type"] == "LOG_BUDGET_EXCEEDED"


def test_log_budget_does_not_interrupt_owned_cleanup(tmp_path, monkeypatch):
    log = tmp_path / "log.txt"
    log.write_bytes(b"oversized")
    budget = entry._LogBudget((log,), 1)
    monkeypatch.setattr(entry.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="log budget"):
        budget.sleep(0.01)
    terminated = []
    api = entry._CleanupAPI(SimpleNamespace(terminate=terminated.append), budget)
    api.terminate("owned-job")
    budget.sleep(0.01)
    assert terminated == ["owned-job"]


def test_summary_exports_only_numeric_async_processing_metrics():
    report = _complete_report() | dict(
        capture_stop_seconds=0.2,
        processing_seconds=14.5,
        sampled_detection_backlog_high_water=240,
        capture_stop={"raw_manifest_path": "private path", "actor": "private owner"},
    )
    result = entry.summarize(
        report,
        dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0),
        dict(source_commit="a" * 40, input_sha256="b" * 64),
    )
    assert result["capture_stop_seconds"] == 0.2
    assert result["processing_seconds"] == 14.5
    assert result["sampled_detection_backlog_high_water"] == 240
    assert "capture_stop" not in result
