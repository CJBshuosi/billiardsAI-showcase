import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import RealtimeRingError
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
)
from scripts.algorithm import run_recording_first_fullgame as runner


@pytest.mark.parametrize("component", [".project-archive", ".PROJECT-ARCHIVE", ".."])
def test_rejects_protected_or_unresolved_paths_before_io(tmp_path, component):
    with pytest.raises(ValueError, match="unsafe acceptance path"):
        runner._plain(tmp_path / component / "media.avi")


def test_symlink_input_is_not_adopted(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe acceptance path"):
        runner._plain(link / "input.avi")


@pytest.mark.parametrize("budget", [0, -1, 901, float("nan"), float("inf")])
def test_invalid_budget_never_creates_output(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(runner, "_validate_inputs", lambda _: (tmp_path, {}))
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="budget"):
        runner.run(
            SimpleNamespace(
                input_root=tmp_path,
                output_root=output,
                capture_seconds=budget,
                encoding_seconds=600,
            )
        )
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_validate_inputs", lambda _: (tmp_path, {}))
    sentinel = tmp_path / "original.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="fresh output"):
        runner.run(
            SimpleNamespace(
                input_root=tmp_path, output_root=tmp_path, capture_seconds=360, encoding_seconds=600
            )
        )
    assert sentinel.read_text() == "keep"


def test_omitted_consumed_calibration_is_rejected_before_opening_input(tmp_path):
    (tmp_path / "input-preflight.json").write_text(
        json.dumps(
            dict(
                contract="recording-first-r4-input-preflight/v1",
                selected_files={
                    name: dict(sha256=digest, bytes=1)
                    for name, digest in runner._INPUT_PINS.items()
                    if not name.startswith("calibration/calibration-registry/")
                },
            )
        )
    )
    with pytest.raises(ValueError, match="reviewed"):
        runner._validate_inputs(tmp_path)


def test_thread_start_failure_still_closes_and_reports(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(runner, "_validate_inputs", lambda _: (tmp_path, {}))
    monkeypatch.setattr(runner.blackbox, "_validate_paths", lambda _: None)
    prepared = SimpleNamespace(cancel=lambda: closed.append("source"))
    monkeypatch.setattr(
        runner.blackbox, "prepare_runtime", lambda *a, **kw: ("cal", None, prepared)
    )
    server = SimpleNamespace(
        field_state=SimpleNamespace(session_id="session"),
        serve_forever=lambda: None,
        server_close=lambda: closed.append("server"),
        shutdown=lambda: (_ for _ in ()).throw(AssertionError("server never started")),
    )
    monkeypatch.setattr(runner.service, "build_live_field_assistance_server", lambda **kw: server)

    def fail_start(_):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(runner.threading.Thread, "start", fail_start)
    output = tmp_path / "output"
    assert (
        runner.run(
            SimpleNamespace(
                input_root=tmp_path, output_root=output, capture_seconds=360, encoding_seconds=600
            )
        )
        == 1
    )
    report = json.loads((output / "report.json").read_bytes())
    assert report["failure"]["reason"] == "thread start failed"
    assert report["cleanup_confirmed"]
    assert closed == ["server", "source"]


def test_frozen_source_revision_checks_all_files_without_git(tmp_path, monkeypatch):
    module = tmp_path / "module.py"
    module.write_bytes(b"x = 1\n")
    revision = "a" * 40
    lock = dict(
        contract="recording-first-source-lock/v1",
        commit=revision,
        files={
            "module.py": dict(
                bytes=module.stat().st_size, sha256=hashlib.sha256(module.read_bytes()).hexdigest()
            )
        },
    )
    (tmp_path / "SOURCE-LOCK.json").write_text(json.dumps(lock))
    monkeypatch.setattr(runner, "_ROOT", tmp_path)
    assert runner._source_revision() == revision
    module.write_bytes(b"x = 2\n")
    with pytest.raises(ValueError, match="source"):
        runner._source_revision()
    module.write_bytes(b"x = 1\n")
    (tmp_path / "unlisted.py").write_bytes(b"x = 3\n")
    with pytest.raises(ValueError, match="source"):
        runner._source_revision()


def test_browser_hold_publishes_only_after_product_pass(tmp_path, monkeypatch):
    server = SimpleNamespace(server_address=("127.0.0.1", 12345))
    report = dict(product_pass=False)
    runner._hold_for_browser(server, report, tmp_path, 1)
    assert not (tmp_path / "browser-ready.json").exists()
    report.update(
        product_pass=True, session_id="session", review=dict(postgame_manifest_sha256="f" * 64)
    )
    times = iter([0, 2])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    runner._hold_for_browser(server, report, tmp_path, 1)
    ready = json.loads((tmp_path / "browser-ready.json").read_bytes())
    assert ready["url"] == "http://127.0.0.1:12345/screen"
    assert ready["session_id"] == "session"
    assert ready["postgame_manifest_sha256"] == "f" * 64
    assert "control_token" not in ready


def test_interval_comparator_reports_missing_merged_and_failed_shots():
    expected = [
        dict(motion_start_frame=180, last_motion_frame=558),
        dict(motion_start_frame=739, last_motion_frame=913),
    ]
    complete = [
        dict(media_status="RECORDED", media=dict(start_frame_id=60, end_frame_id=680)),
        dict(media_status="RECORDED", media=dict(start_frame_id=619, end_frame_id=1034)),
    ]
    for shot, truth in zip(complete, expected, strict=True):
        shot["boundary"] = dict(
            start_frame_id=truth["motion_start_frame"], end_frame_id=truth["last_motion_frame"]
        )
    assert runner.compare_intervals(complete, expected) == []
    assert "shot_count:1!=2" in runner.compare_intervals(complete[:1], expected)
    truncated = [
        dict(media_status="RECORDED", media=dict(start_frame_id=181, end_frame_id=680)),
        dict(media_status="FAILED", media={}),
    ]
    for shot, original in zip(truncated, complete, strict=True):
        shot["boundary"] = original["boundary"]
    assert runner.compare_intervals(truncated, expected) == [
        "shot_1:motion_not_contained",
        "shot_2:not_recorded",
    ]
    assert runner.compare_intervals(complete[1:], expected) == ["shot_count:1!=2", "shot_1:missing"]
    merged = dict(
        complete[0], shot_id="merged", boundary=dict(start_frame_id=180, end_frame_id=913)
    )
    assert "merged:merged_intervals" in runner.compare_intervals([merged, merged], expected)


def test_raw_audit_reads_original_frames_and_rejects_mutated_segment(tmp_path):
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    ok, image = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    for index in range(3):
        recorder.append(CaptureFrame("epoch", index, index * 16_666_667, 16, 16, image.tobytes()))
    recorder.close()
    path = recorder.finalized_manifest_path
    frozen = dict(
        raw_manifest_path=str(path),
        raw_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        session_id="session",
        source_epoch_id="epoch",
        first_frame_id=0,
        last_frame_id=2,
    )
    assert (
        runner.verify_raw_recording(frozen, tmp_path, frame_count=3, geometry=(16, 16))[
            "decoded_frames"
        ]
        == 3
    )
    with pytest.raises(ValueError, match="coverage"):
        runner.verify_raw_recording(
            dict(frozen, first_frame_id=1), tmp_path, frame_count=3, geometry=(16, 16)
        )
    segment = json.loads(path.read_bytes())["segments"][0]["name"]
    (path.parent / segment).write_bytes(b"corrupt")
    with pytest.raises(RealtimeRingError, match="hash"):
        runner.verify_raw_recording(frozen, tmp_path, frame_count=3, geometry=(16, 16))


def test_until_complete_background_passes_600_900_960_and_retains_bounded_samples(
    monkeypatch, capsys
):

    now = [0.0]
    receipt = dict(
        contract="recording-capture-end/v1",
        capture_status="STOPPED",
        session_id="session",
        source_epoch_id="epoch",
        cutoff={"frame_id": 3},
        accepted_at_utc="now",
    )
    current = dict(receipt, tail_status="SAVED", processing_status="PROCESSING", failure_code=None)
    frozen = dict(cutoff=receipt["cutoff"], source_epoch_id="epoch", shots=[{"shot_id": "shot"}])

    class Store:
        def get_recording_evidence(self, session, shot):
            return {"media_status": "RECORDED"}

        def get_recording_end_manifest(self, session):
            return frozen if now[0] >= 2000 else None

        def get_recording_capture_end(self, session):
            return dict(current, processing_status="COMPLETE" if now[0] >= 2000 else "PROCESSING")

    state = SimpleNamespace(
        session_id="session",
        record_store=Store(),
        projection_source=SimpleNamespace(source_epoch_id="epoch"),
        media_job_store=SimpleNamespace(
            session_media_progress=lambda _: dict(
                failed_count=0,
                encoding_count=0,
                retrying_count=0,
                generated_count=int(now[0] >= 2000),
                items=[{}],
            )
        ),
        _recording_processing_progress=lambda current: dict(status=current["processing_status"]),
    )

    def review(*a, **kw):
        assert 0 < kw["timeout_seconds"] <= 30
        return dict(
            review_session_id="session",
            phase="RECORDING_ENDED",
            processing={"status": "COMPLETE"},
            discovered_count=1,
            expected_count=1,
            available_count=int(now[0] >= 2000),
            clips=[dict(media_url="media" if now[0] >= 2000 else None, terminal_failure=False)],
        )

    monkeypatch.setattr(runner, "_request", review)
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner.time, "sleep", lambda _: now.__setitem__(0, now[0] + 1))
    report = {}
    result = runner._wait_for_background_completion(
        None, state, receipt, deadline=None, report=report
    )
    assert result[0] == frozen and now[0] >= 2000
    assert len(report["processing_samples"]) <= 128
    assert report["background_checkpoint_600"]["available_count"] == 0
    assert report["background_diagnostic"]["code"] == "NO_PROGRESS_CHECKPOINT"
    assert len(capsys.readouterr().out) < 16384


def test_observer_records_real_live_encoding_and_single_validation(tmp_path):
    import time

    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
    from scripts.algorithm.recording_fullgame_progress import FullgameProgress
    from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _ordinary_scene

    records, session, root = _ordinary_scene(tmp_path, live=True)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame" / session
    )
    worker.schedule(session)
    output = tmp_path / "recording"
    output.mkdir()
    report = {}
    progress = FullgameProgress(output, report, started=time.monotonic())
    with progress.observe():
        result = worker.process_one("real-observed", session, NOW)
    assert result is not None and records.get_recording_capture_end(session) is None
    row = report["clip_timeline"][0]
    assert row["status"] == "available" and row["attempt"] == 1
    assert row["encode_seconds"] > 0 and row["verify_seconds"] > 0
    assert row["done_seconds"] >= row["start_seconds"]
    heartbeat = json.loads((output / "observer-heartbeat.json").read_text())
    assert heartbeat["run_id"] == tmp_path.name and heartbeat["clips"] == [row]
    assert "source" not in heartbeat and "session_id" not in row


@pytest.mark.parametrize("fault", ["output", "snapshot", "heartbeat"])
def test_observer_fault_never_changes_actual_worker_result(tmp_path, monkeypatch, fault):
    import time

    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
    from scripts.algorithm import recording_fullgame_progress as progress_module
    from tests.unit.lab.test_mainline1_recording_media_worker import NOW, _ordinary_scene

    records, session, root = _ordinary_scene(tmp_path, live=True)
    jobs = SQLiteShotAnalysisJobStore(tmp_path / "jobs.sqlite3")
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=root / "postgame" / session
    )
    job = worker.schedule(session)[0]
    progress = progress_module.FullgameProgress(tmp_path, {}, started=time.monotonic())

    def broken(*a, **kw):
        raise OSError("controlled observer fault")

    if fault == "output":
        monkeypatch.setattr("builtins.print", broken)
    elif fault == "snapshot":
        monkeypatch.setattr(jobs, "snapshot", broken)
    else:
        monkeypatch.setattr(progress_module, "atomic_json", broken)
    with progress.observe():
        result = worker.process_one("observed", session, NOW)
    assert result is not None and result.job_id == job.job_id
    assert progress.report["observer_errors"]


def test_until_complete_terminal_source_without_job_fails_instead_of_waiting(monkeypatch):
    receipt = dict(
        contract="recording-capture-end/v1",
        capture_status="STOPPED",
        session_id="session",
        source_epoch_id="epoch",
        cutoff={"frame_id": 3},
        accepted_at_utc="now",
    )
    current = dict(receipt, tail_status="SAVED", processing_status="COMPLETE", failure_code=None)
    store = SimpleNamespace(
        get_recording_end_manifest=lambda _: dict(shots=[{"shot_id": "truncated"}]),
        get_recording_capture_end=lambda _: current,
        get_recording_evidence=lambda *a: {"media_status": "FAILED"},
    )
    state = SimpleNamespace(
        session_id="session",
        record_store=store,
        projection_source=SimpleNamespace(source_epoch_id="epoch"),
    )
    with pytest.raises(RuntimeError, match="terminal original recording interval"):
        runner._wait_for_background_completion(None, state, receipt, deadline=None, report={})


def test_observer_snapshot_failure_preserves_original_worker_exception(tmp_path, monkeypatch):
    import time

    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker
    from scripts.algorithm.recording_fullgame_progress import FullgameProgress

    sentinel = RuntimeError("original product exception")

    def product_failure(*a, **kw):
        raise sentinel

    def observer_failure(*a, **kw):
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(Mainline1RecordingMediaWorker, "_process_claim", product_failure)
    worker = object.__new__(Mainline1RecordingMediaWorker)
    worker._jobs = SimpleNamespace(snapshot=observer_failure)
    claim = SimpleNamespace(job=SimpleNamespace(job_id="job", shot_id="shot"), attempt_count=1)
    progress = FullgameProgress(tmp_path, {}, started=time.monotonic())
    with progress.observe(), pytest.raises(RuntimeError) as caught:
        worker._process_claim(claim, None)
    assert caught.value is sentinel
    assert progress.report["observer_errors"] == ["CLIP_OBSERVATION_UNAVAILABLE"]


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_heartbeat_retries_windows_sharing_only_at_replace(tmp_path, monkeypatch, winerror):
    import time

    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    target = tmp_path / "observer-heartbeat.json"
    reports.atomic_json(target, {"old": True})
    replace = reports.os.replace
    attempts = []

    def sharing_then_release(source, destination):
        attempts.append(source.read_bytes())
        if len(attempts) < 3:
            error = PermissionError("controlled Windows reader sharing")
            error.winerror = winerror
            raise error
        replace(source, destination)

    monkeypatch.setattr(reports.os, "replace", sharing_then_release)
    observer = progress.FullgameProgress(tmp_path, {}, started=time.monotonic())
    observer.heartbeat({"available_count": 4}, force=True)
    assert not observer.report.get("observer_errors")
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert json.loads(target.read_bytes())["latest"] == {"available_count": 4}


def test_heartbeat_replace_retry_has_time_and_attempt_limits(tmp_path, monkeypatch):
    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    calls = []

    def locked(*args):
        calls.append(True)
        error = PermissionError("controlled persistent reader")
        error.winerror = 5
        raise error

    monkeypatch.setattr(reports.os, "replace", locked)
    now = [0.0]
    with pytest.raises(PermissionError):
        progress._replace_heartbeat(
            tmp_path / "a",
            tmp_path / "b",
            clock=lambda: now[0],
            pause=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
    assert now[0] <= 0.251 and 1 < len(calls) <= 26
    calls.clear()
    with pytest.raises(PermissionError):
        progress._replace_heartbeat(
            tmp_path / "a", tmp_path / "b", clock=lambda: 0, pause=lambda seconds: None
        )
    assert len(calls) == 26


@pytest.mark.parametrize("winerror", [None, 2, 112])
def test_heartbeat_other_replace_errors_fail_without_retry(tmp_path, monkeypatch, winerror):
    import time

    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    calls = []

    def broken(*args):
        calls.append(True)
        error = OSError("controlled permanent failure")
        error.winerror = winerror
        raise error

    monkeypatch.setattr(reports.os, "replace", broken)
    observer = progress.FullgameProgress(tmp_path, {}, started=time.monotonic())
    observer.heartbeat(force=True)
    assert calls == [True]
    assert observer.report["observer_errors"] == ["HEARTBEAT_UNAVAILABLE"]


def test_heartbeat_persistent_sharing_failure_is_sticky_and_preserves_old_file(
    tmp_path, monkeypatch
):
    import time

    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    target = tmp_path / "observer-heartbeat.json"
    reports.atomic_json(target, {"old": True})
    before = target.read_bytes()
    replace = reports.os.replace
    calls = []

    def locked(*args):
        calls.append(True)
        error = PermissionError("controlled persistent reader")
        error.winerror = 32
        raise error

    monkeypatch.setattr(reports.os, "replace", locked)
    observer = progress.FullgameProgress(tmp_path, {}, started=time.monotonic())
    observer.heartbeat(force=True)
    assert 1 < len(calls) <= 26 and target.read_bytes() == before
    assert observer.report["observer_errors"] == ["HEARTBEAT_UNAVAILABLE"]
    monkeypatch.setattr(reports.os, "replace", replace)
    observer.heartbeat(force=True)
    assert json.loads(target.read_bytes())["contract"] == "recording-fullgame-progress/v1"
    assert observer.report["observer_errors"] == ["HEARTBEAT_UNAVAILABLE"]


def test_generic_atomic_reports_keep_immediate_replace_failure(tmp_path, monkeypatch):
    from scripts.development import blackbox_report as reports

    calls = []

    def locked(*args):
        calls.append(True)
        error = PermissionError("controlled generic report failure")
        error.winerror = 5
        raise error

    monkeypatch.setattr(reports.os, "replace", locked)
    with pytest.raises(PermissionError):
        reports.atomic_json(tmp_path / "summary.json", {})
    assert calls == [True]


@pytest.mark.parametrize("fault", ["serialize", "fsync"])
def test_heartbeat_never_retries_errors_before_replace(tmp_path, monkeypatch, fault):
    import time

    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    target = tmp_path / "observer-heartbeat.json"
    reports.atomic_json(target, {"old": True})
    before = target.read_bytes()
    replaces = []
    monkeypatch.setattr(reports.os, "replace", lambda *args: replaces.append(True))
    if fault == "fsync":

        def broken_fsync(*args):
            error = PermissionError("controlled fsync failure")
            error.winerror = 5
            raise error

        monkeypatch.setattr(reports.os, "fsync", broken_fsync)
    observer = progress.FullgameProgress(tmp_path, {}, started=time.monotonic())
    observer.heartbeat({"value": float("nan")} if fault == "serialize" else {}, force=True)
    assert observer.report["observer_errors"] == ["HEARTBEAT_UNAVAILABLE"]
    assert replaces == [] and target.read_bytes() == before


def test_heartbeat_does_not_retry_after_scheduler_oversleeps(tmp_path, monkeypatch):
    from scripts.algorithm import recording_fullgame_progress as progress
    from scripts.development import blackbox_report as reports

    calls = []
    error = PermissionError("controlled sharing failure")
    error.winerror = 5

    def delayed(*args):
        calls.append(True)
        if len(calls) == 1:
            raise error

    monkeypatch.setattr(reports.os, "replace", delayed)
    now = [0.0]
    with pytest.raises(PermissionError) as caught:
        progress._replace_heartbeat(
            tmp_path / "a",
            tmp_path / "b",
            clock=lambda: now[0],
            pause=lambda seconds: now.__setitem__(0, 1.0),
        )
    assert caught.value is error and calls == [True]
