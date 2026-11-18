"""A transient Windows manifest sharing failure must not end capture."""

import json
import threading
from types import SimpleNamespace

import pytest

from billiards_ai.lab import mainline1_session_recording as recording
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame


def _recorder(tmp_path):
    return recording.ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id="session",
        expected_fps=1,
        config=recording.SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )


def _error(code):
    error = PermissionError("controlled manifest access failure")
    error.winerror = code
    return error


def _platform(monkeypatch, replace, *, windows=True):
    # Patch this module only: pathlib must retain the real host filesystem.
    proxy = SimpleNamespace(**vars(recording.os))
    proxy.name = "nt" if windows else "posix"
    proxy.replace = replace
    monkeypatch.setattr(recording, "os", proxy)


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_manifest_conflict_keeps_durable_recording(tmp_path, monkeypatch, code):
    recorder = _recorder(tmp_path)
    original = recording.os.replace
    attempts = []

    def sharing_then_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise _error(code)
        original(source, target)

    try:
        with monkeypatch.context() as patch:
            _platform(patch, sharing_then_replace)
            recorder._write_manifest(recorder._snapshot_manifest())
        manifest = json.loads((tmp_path / "raw/session-recording.json").read_text())
        assert manifest["session_id"] == "session" and manifest["segments"] == []
        assert len({source for source, _ in attempts}) == 1
        assert not list((tmp_path / "raw").glob(".pending-manifest-*"))
    finally:
        recorder.close()


@pytest.mark.parametrize("windows,code", [(False, 5), (True, 2), (True, None)])
def test_other_manifest_errors_fail_immediately(tmp_path, monkeypatch, windows, code):
    recorder = _recorder(tmp_path)
    error = _error(code)
    sleeps = []

    def denied(source, target):
        raise error

    try:
        with monkeypatch.context() as patch:
            _platform(patch, denied, windows=windows)
            patch.setattr(
                recording,
                "time",
                SimpleNamespace(
                    monotonic=lambda: 0.0,
                    sleep=sleeps.append,
                ),
            )
            with pytest.raises(PermissionError) as caught:
                recorder._write_manifest(recorder._snapshot_manifest())
            assert caught.value is error and sleeps == []
        assert not (tmp_path / "raw/session-recording.json").exists()
    finally:
        recorder.close()


def test_persistent_manifest_denial_preserves_prior_manifest_and_original_error(
    tmp_path, monkeypatch
):
    recorder = _recorder(tmp_path)
    recorder._write_manifest(recorder._snapshot_manifest())
    destination = tmp_path / "raw/session-recording.json"
    before = destination.read_bytes()
    error = _error(5)
    elapsed = [0.0]
    calls = []

    def denied(source, target):
        calls.append(source)
        raise error

    def sleep(seconds):
        elapsed[0] += seconds

    try:
        with monkeypatch.context() as patch:
            _platform(patch, denied)
            patch.setattr(
                recording,
                "time",
                SimpleNamespace(
                    monotonic=lambda: elapsed[0],
                    sleep=sleep,
                ),
            )
            with pytest.raises(PermissionError) as caught:
                recorder._write_manifest(recorder._snapshot_manifest())
            assert caught.value is error
        assert 0 < elapsed[0] <= 0.25 and 1 < len(calls) <= 26
        assert destination.read_bytes() == before
        assert recorder.health.committed_frames == 0
    finally:
        recorder.close()


def test_manifest_conflict_does_not_publish_watermark_or_hold_capture_admission(
    tmp_path, monkeypatch
):
    recorder = _recorder(tmp_path)
    original = recording.os.replace
    entered, release = threading.Event(), threading.Event()
    failures = []
    denied_once = [False]

    def sharing_once(source, target):
        if target.name == "session-recording.json" and not denied_once[0]:
            denied_once[0] = True
            entered.set()
            assert release.wait(1)
            raise _error(5)
        original(source, target)

    def append_second():
        try:
            recorder.append(CaptureFrame("epoch", 1, 1_000_000_000, 16, 12, b"jpeg"))
        except BaseException as exc:
            failures.append(exc)

    try:
        with monkeypatch.context() as patch:
            _platform(patch, sharing_once)
            recorder.append(CaptureFrame("epoch", 0, 0, 16, 12, b"jpeg"))
            assert entered.wait(1)
            assert recorder.health.committed_frames == 0
            append = threading.Thread(target=append_second)
            append.start()
            append.join(0.1)
            assert not append.is_alive(), "manifest retry must not hold capture admission"
            release.set()
            assert recorder.wait_until_committed(1, timeout_seconds=1)
            assert failures == [] and recorder.health.committed_frames == 2
    finally:
        release.set()
        recorder.close()


def test_manifest_retries_do_not_reset_close_deadline(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    error = _error(5)
    elapsed = [0.0]

    def denied(source, target):
        raise error

    def sleep(seconds):
        elapsed[0] += seconds

    try:
        with monkeypatch.context() as patch:
            _platform(patch, denied)
            patch.setattr(
                recording,
                "time",
                SimpleNamespace(
                    monotonic=lambda: elapsed[0],
                    sleep=sleep,
                ),
            )
            with pytest.raises(PermissionError) as caught:
                recorder.close(timeout_seconds=0.02)
            assert caught.value is error
        assert 0 < elapsed[0] <= 0.02
        assert recorder._closed_manifest is None
    finally:
        recorder.close()


def test_manifest_cleanup_error_does_not_hide_original_replace_failure(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    error = _error(5)
    original_unlink = type(tmp_path).unlink

    def denied(source, target):
        raise error

    def cleanup_denied(path, *args, **kwargs):
        if path.name.startswith(".pending-manifest-"):
            raise OSError("controlled cleanup failure")
        return original_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            _platform(patch, denied, windows=False)
            patch.setattr(type(tmp_path), "unlink", cleanup_denied)
            with pytest.raises(PermissionError) as caught:
                recorder._write_manifest(recorder._snapshot_manifest())
            assert caught.value is error
        assert list((tmp_path / "raw").glob(".pending-manifest-*"))
    finally:
        for path in (tmp_path / "raw").glob(".pending-manifest-*"):
            path.unlink()
        recorder.close()


def test_manifest_fsync_cannot_start_replace_after_close_deadline(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    elapsed = [0.0]
    replacements = []
    original_replace, original_fsync = recording.os.replace, recording.os.fsync

    def replace(source, target):
        replacements.append(target)
        original_replace(source, target)

    def slow_fsync(descriptor):
        original_fsync(descriptor)
        elapsed[0] += 0.03

    try:
        with monkeypatch.context() as patch:
            _platform(patch, replace)
            patch.setattr(recording.os, "fsync", slow_fsync)
            patch.setattr(recording, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
            with pytest.raises(recording.RealtimeRingError, match="close exceeded timeout"):
                recorder.close(timeout_seconds=0.02)
        assert replacements == []
        assert recorder._closed_manifest is None
        assert not (tmp_path / "raw/session-recording.json").exists()
    finally:
        recorder.close()
