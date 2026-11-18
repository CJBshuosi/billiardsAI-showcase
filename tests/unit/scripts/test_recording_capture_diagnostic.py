import importlib
import importlib.util
import io
import json
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def diagnostic():
    name = "scripts.algorithm.recording_capture_diagnostic"
    assert importlib.util.find_spec(name) is not None, "Missing bounded capture profiler"
    return importlib.import_module(name)


@pytest.mark.parametrize("codec,encode_calls", [("MJPG", 0), ("FFV1", 6)])
def test_profile_preserves_real_capture_frames_and_measures_codec_and_validation(
    tmp_path, codec, encode_calls
):
    probe = diagnostic()
    from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
    from billiards_ai.lab.mainline1_scenario_capture import (
        ScenarioCaptureConfig,
        ScenarioVideoCaptureBackend,
    )

    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 60, (64, 64))
    assert writer.isOpened()
    for n in range(6):
        writer.write(np.full((64, 64, 3), n * 30, np.uint8))
    writer.release()
    config = ScenarioCaptureConfig(path, RealtimeCaptureConfig(width=64, height=64, fps=60))
    plain = list(
        ScenarioVideoCaptureBackend(config, source_epoch_id="test").frames(threading.Event())
    )
    source = ScenarioVideoCaptureBackend(config, source_epoch_id="test")
    measured = probe.CaptureProfiler()
    profiled = list(measured.frames(source.frames(threading.Event())))
    assert [(f.capture_frame_id, f.mjpeg) for f in profiled] == [
        (f.capture_frame_id, f.mjpeg) for f in plain
    ]
    result = measured.snapshot()
    assert result["complete"] and result["yielded_frames"] == 6
    assert result["operations"]["jpeg_encode"]["calls"] == encode_calls
    assert result["operations"]["video_decode"]["calls"] == 7  # Includes clean EOF.
    assert result["operations"]["source_validation"]["calls"] >= 6
    assert result["operations"]["pacing"]["calls"] > 0
    assert str(tmp_path) not in json.dumps(result)
    assert set(result["operations"]) == {
        "jpeg_encode",
        "video_decode",
        "source_validation",
        "pacing",
        "recorder_append",
        "subscription_publish",
    }


def test_profile_includes_downstream_time_between_yields_without_changing_cancellation():
    probe = diagnostic()
    measured = probe.CaptureProfiler()
    closed = []

    def frames():
        try:
            yield "unchanged"
            raise RuntimeError("original exception")
        finally:
            closed.append(True)

    iterator = measured.frames(frames())
    assert next(iterator) == "unchanged"
    assert not measured.snapshot()["complete"]
    time.sleep(0.01)  # Represents synchronous work by the source consumer on this thread.
    with pytest.raises(RuntimeError, match="original exception"):
        next(iterator)
    result = measured.snapshot()
    assert result["complete"] and result["yielded_frames"] == 1
    assert result["elapsed_seconds"] >= 0.01 and closed == [True]
    with pytest.raises(ValueError, match="single.use"):
        list(measured.frames(iter(())))


def healthy_profile():
    operations = {
        name: dict(calls=1200, self_seconds=1, cumulative_seconds=1)
        for name in (
            "jpeg_encode",
            "video_decode",
            "source_validation",
            "pacing",
            "recorder_append",
            "subscription_publish",
        )
    }
    buckets = {name: dict(value) for name, value in operations.items()}
    buckets.update(
        {
            name: dict(calls=0, self_seconds=0, cumulative_seconds=0)
            for name in (
                "source_dispatch",
                "between_source_yields",
                "observed_lock_calls",
                "path_metadata",
                "buffer_copy",
                "hub_recorder_acquire",
                "recorder_append_acquire",
                "condition_acquire",
            )
        }
    )
    buckets["between_source_yields"] = dict(calls=0, self_seconds=23, cumulative_seconds=23)
    for name in ("hub_recorder_acquire", "recorder_append_acquire", "condition_acquire"):
        buckets[name]["calls"] = 1200
    return dict(
        complete=True,
        yielded_frames=1200,
        elapsed_seconds=30,
        operations=operations,
        exclusive_buckets=buckets,
        accounting=dict(
            verified=True, direct_callback_seconds=1, accounted_seconds=29, callbacks=10000
        ),
    )


def report():
    return dict(
        source_commit="a" * 40,
        source_sha256="b" * 64,
        product_pass=False,
        failure=dict(type="TimeoutError", reason="whole input capture budget exhausted"),
        cleanup_confirmed=True,
        samples=[
            dict(seconds=1, frame=40, shots=0, pending_frames=0, pending_chunks=0),
            dict(seconds=30, frame=1200, shots=3, pending_frames=1, pending_chunks=0),
        ],
        private_owner="never export",
        session_id="never export",
        commands=["never export"],
    )


def test_short_probe_success_requires_expected_cutoff_identity_cleanup_and_completed_profile():
    probe = diagnostic()
    manifest = dict(source_commit="a" * 40, input_sha256="b" * 64)
    good = report()
    result = probe.summarize_probe(good, healthy_profile(), manifest)
    assert result["probe_completed"] and result["capture_budget_exhausted"]
    assert result["last_sample"] == dict(
        seconds=30, frame=1200, shots=3, pending_frames=1, pending_chunks=0
    )
    assert not result["product_pass"] and not result["browser_pass"]
    assert not result["model_pass"] and not result["field_pass"]
    assert "never export" not in json.dumps(result)
    for mutate in (
        lambda r: r.update(source_commit="c" * 40),
        lambda r: r.update(cleanup_confirmed=False),
        lambda r: r.update(failure=dict(type="TimeoutError", reason="different private reason")),
        lambda r: r.update(samples=[]),
    ):
        bad = report()
        mutate(bad)
        assert not probe.summarize_probe(bad, healthy_profile(), manifest)["probe_completed"]
    incomplete = healthy_profile()
    incomplete["complete"] = False
    assert not probe.summarize_probe(good, incomplete, manifest)["probe_completed"]


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "private text"])
def test_invalid_profile_measurement_cannot_be_exported_as_success(value):
    probe = diagnostic()
    profile = healthy_profile()
    profile["operations"]["jpeg_encode"]["self_seconds"] = value
    result = probe.summarize_probe(
        report(), profile, dict(source_commit="a" * 40, input_sha256="b" * 64)
    )
    assert not result["probe_completed"]
    assert "private text" not in json.dumps(result, allow_nan=False)


def test_worker_needs_owned_release_before_filesystem_or_product_imports(monkeypatch):
    probe = diagnostic()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"wrong\n")))
    monkeypatch.setattr(probe, "checked_root", lambda *a: pytest.fail("unreleased root read"))
    with pytest.raises(ValueError, match="owned release"):
        probe.main(["worker", "/untrusted", "a" * 64])


def test_parent_revalidates_worker_whitelist_and_never_exports_product_success(
    tmp_path, monkeypatch
):
    probe = diagnostic()
    from scripts.algorithm import recording_windows_entry as existing

    monkeypatch.setenv("SYSTEMROOT", "C:/Windows")
    payload = probe.summarize_probe(
        report(), healthy_profile(), dict(source_commit="a" * 40, input_sha256="b" * 64)
    )
    payload.update(private_owner="never export", product_pass=True, model_pass=True)

    def owned(api, spawn, persist, metadata, release, **budgets):
        (tmp_path / "output/probe.json").write_text(json.dumps(payload))
        return dict(status="PASS", cleanup_confirmed=True, worker_exit_code=0)

    monkeypatch.setitem(
        sys.modules, "ownership", SimpleNamespace(WindowsAPI=object, run_owned=owned)
    )
    monkeypatch.setattr(existing, "LOG_LIMIT", 16)
    assert (
        probe.launch(tmp_path, "f" * 64, dict(source_commit="a" * 40, input_sha256="b" * 64)) == 0
    )
    summary = json.loads((tmp_path / "output/summary.json").read_bytes())
    assert "never export" not in json.dumps(summary)
    assert not summary["product_pass"] and not summary["model_pass"]


def test_oversized_worker_integer_yields_incomplete_summary_instead_of_crashing():
    probe = diagnostic()
    payload = probe.summarize_probe(
        report(), healthy_profile(), dict(source_commit="a" * 40, input_sha256="b" * 64)
    )
    payload["yielded_frames"] = 10**400
    result = probe.sanitize_worker_output(
        payload, dict(source_commit="a" * 40, input_sha256="b" * 64)
    )
    assert not result["probe_completed"]
    assert result["yielded_frames"] is None


def test_missing_codec_profile_cannot_claim_diagnostic_completion():
    probe = diagnostic()
    profile = healthy_profile()
    profile["operations"]["jpeg_encode"]["calls"] = 0
    result = probe.summarize_probe(
        report(), profile, dict(source_commit="a" * 40, input_sha256="b" * 64)
    )
    assert not result["probe_completed"]


def test_bootstrap_import_does_not_load_profiler_or_product_native_modules():
    import subprocess
    from pathlib import Path

    path = Path(__file__).resolve().parents[3]
    code = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "from scripts.algorithm import recording_capture_diagnostic; "
        'print(json.dumps([name for name in ("_lsprof","cv2","numpy") if name in sys.modules]))'
    )
    result = subprocess.check_output(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(path)], text=True
    )
    assert json.loads(result) == []


def test_unrelated_thread_codec_is_not_counted_as_producer_work():
    probe = diagnostic()
    start, done = threading.Event(), threading.Event()
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    def unrelated():
        start.wait()
        for _ in range(7):
            assert cv2.imencode(".jpg", image)[0]
        done.set()

    thread = threading.Thread(target=unrelated)
    thread.start()
    profiler = probe.CaptureProfiler()

    def producer():
        start.set()
        assert done.wait(2)
        yield "no encoding on this thread"

    try:
        assert list(profiler.frames(producer())) == ["no encoding on this thread"]
    finally:
        start.set()
        thread.join(2)
    assert not thread.is_alive()
    result = profiler.snapshot()
    assert result["operations"]["jpeg_encode"]["calls"] == 0
    assert result["accounting"]["verified"]


def test_all_elapsed_time_is_accounted_even_for_unclassified_work_and_restores_hook():
    probe = diagnostic()
    profiler = probe.CaptureProfiler()
    previous = sys.getprofile()

    def producer():
        time.sleep(0.01)
        yield "original"

    assert list(profiler.frames(producer())) == ["original"]
    assert sys.getprofile() is previous
    result = profiler.snapshot()
    assert result["accounting"]["verified"]
    app = sum(row["self_seconds"] for row in result["exclusive_buckets"].values())
    overhead = result["accounting"]["direct_callback_seconds"]
    assert app > 0.005
    assert abs(app + overhead - result["elapsed_seconds"]) < 1e-6


def test_incomplete_or_inconsistent_elapsed_accounting_cannot_complete_probe():
    probe = diagnostic()
    manifest = dict(source_commit="a" * 40, input_sha256="b" * 64)
    value = healthy_profile()
    value.pop("accounting", None)
    value.pop("exclusive_buckets", None)
    assert not probe.summarize_probe(report(), value, manifest)["probe_completed"]


def test_profiling_never_replaces_an_existing_thread_hook():
    probe = diagnostic()
    previous = sys.getprofile()

    def sentinel(*args):
        return None

    sys.setprofile(sentinel)
    try:
        with pytest.raises(ValueError, match="hook"):
            list(probe.CaptureProfiler().frames(iter(("untouched",))))
        assert sys.getprofile() is sentinel
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize(
    "kind,bucket",
    [
        ("HubRecorderLock", "hub_recorder_acquire"),
        ("RecorderAppendLock", "recorder_append_acquire"),
    ],
)
def test_actual_context_manager_lock_contention_is_measured_without_changing_lock(kind, bucket):
    probe = diagnostic()
    proxy_type = getattr(probe, kind, None)
    assert proxy_type is not None, "Direct with-lock acquisition is not observed"
    inner = threading.Lock()
    held, release = threading.Event(), threading.Event()

    def holder():
        with inner:
            held.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(2)
    timer = threading.Timer(0.03, release.set)
    timer.start()
    proxy = proxy_type(inner)
    profiler = probe.CaptureProfiler()

    def producer():
        with proxy:
            yield "same underlying critical section"

    try:
        assert list(profiler.frames(producer())) == ["same underlying critical section"]
    finally:
        release.set()
        thread.join(2)
        timer.join(2)
    assert not inner.locked() and not thread.is_alive()
    result = profiler.snapshot()
    assert result["accounting"]["verified"]
    assert result["exclusive_buckets"][bucket]["calls"] == 1
    assert result["exclusive_buckets"][bucket]["cumulative_seconds"] >= 0.01


def test_observed_lock_preserves_original_nonblocking_and_release_protocol():
    probe = diagnostic()
    proxy_type = getattr(probe, "HubRecorderLock", None)
    assert proxy_type is not None
    inner = threading.Lock()
    proxy = proxy_type(inner)
    assert proxy.acquire(False) and inner.locked()
    assert not proxy.acquire(False)
    proxy.release()
    with proxy:
        assert inner.locked()
    assert not inner.locked()
