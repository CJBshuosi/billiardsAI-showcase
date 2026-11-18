"""Acceptance-driver assertions cannot stand in for actual model/display evidence."""

import importlib
import io
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


def tool():
    assert Path("scripts/development/s1_screen_acceptance.py").is_file()
    return importlib.import_module("scripts.development.s1_screen_acceptance")


def scenario_backend(tmp_path):
    import cv2
    import numpy as np

    from billiards_ai.lab.mainline1_realtime_capture import RealtimeCaptureConfig
    from billiards_ai.lab.mainline1_scenario_capture import (
        ScenarioCaptureConfig,
        ScenarioVideoCaptureBackend,
    )

    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
    writer.release()
    return ScenarioVideoCaptureBackend(
        ScenarioCaptureConfig(path, RealtimeCaptureConfig(width=32, height=24, fps=20)),
        source_epoch_id="original-pixels",
    )


def test_real_source_validation_receipt_preserves_pixels_and_original_methods(tmp_path):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    backend = scenario_backend(tmp_path)
    original = (scenario._source_stat, scenario._content_hash, backend._validate_source.__func__)
    stop = threading.Event()
    with tool().trace_scenario_validation(tmp_path / "validation"):
        stream = backend.frames(stop)
        first = next(stream)
        assert first.capture_frame_id == 0 and first.decode_bgr().shape == (24, 32, 3)
        stop.set()
        stream.close()
    assert original == (
        scenario._source_stat,
        scenario._content_hash,
        backend._validate_source.__func__,
    )
    receipts = list((tmp_path / "validation").glob("*.json"))
    assert len(receipts) == 1
    value = json.loads(receipts[0].read_bytes())
    assert value["bytes"] == (tmp_path / "source.avi").stat().st_size
    assert [(row["stage"], row["event"]) for row in value["events"]] == [
        ("validation", "ENTERED"),
        ("source_stat", "ENTERED"),
        ("source_stat", "RETURNED"),
        ("content_hash", "ENTERED"),
        ("content_hash", "RETURNED"),
        ("source_stat", "ENTERED"),
        ("source_stat", "RETURNED"),
        ("validation", "RETURNED"),
    ]
    assert "source.avi" not in receipts[0].read_text()


def test_blocked_original_hash_is_distinguished_from_source_stat_before_return(
    tmp_path, monkeypatch
):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    backend = scenario_backend(tmp_path)
    original = scenario._content_hash
    entered, release = threading.Event(), threading.Event()
    values, errors = [], []

    def blocked(path):
        entered.set()
        assert release.wait(3)
        return original(path)

    monkeypatch.setattr(scenario, "_content_hash", blocked)

    def capture():
        stream = backend.frames(threading.Event())
        try:
            values.append(next(stream))
        except Exception as exc:
            errors.append(exc)
        finally:
            stream.close()

    with tool().trace_scenario_validation(tmp_path / "validation"):
        thread = threading.Thread(target=capture)
        thread.start()
        try:
            assert entered.wait(2)
            receipt = next((tmp_path / "validation").glob("*.json"))
            value = json.loads(receipt.read_bytes())
            assert value["events"][-1]["stage"] == "content_hash"
            assert value["events"][-1]["event"] == "ENTERED"
            assert not values
        finally:
            release.set()
            thread.join(3)
        assert not thread.is_alive() and not errors and len(values) == 1
    assert scenario._content_hash is blocked


def test_original_source_validation_error_keeps_its_identity_and_restores_proxy(
    tmp_path, monkeypatch
):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    backend = scenario_backend(tmp_path)
    failure = RuntimeError("controlled original hash failure")

    def broken(path):
        raise failure

    monkeypatch.setattr(scenario, "_content_hash", broken)
    with pytest.raises(RuntimeError) as raised:
        with tool().trace_scenario_validation(tmp_path / "validation"):
            next(backend.frames(threading.Event()))
    assert raised.value is failure and scenario._content_hash is broken
    value = json.loads(next((tmp_path / "validation").glob("*.json")).read_bytes())
    assert [(row["stage"], row["event"]) for row in value["events"]][-2:] == [
        ("content_hash", "RAISED"),
        ("validation", "RAISED"),
    ]


def test_failed_validation_exception_receipt_cannot_replace_the_original_failure(
    tmp_path, monkeypatch
):
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    backend = scenario_backend(tmp_path)
    failure = RuntimeError("controlled original hash failure")

    def broken(path):
        raise failure

    module = tool()
    original_write = module.atomic_json

    def fail_exception_receipt(path, value):
        if value["events"] and value["events"][-1]["event"] == "RAISED":
            raise OSError("controlled diagnostic write failure")
        return original_write(path, value)

    monkeypatch.setattr(scenario, "_content_hash", broken)
    monkeypatch.setattr(module, "atomic_json", fail_exception_receipt)
    with pytest.raises(RuntimeError) as raised:
        with module.trace_scenario_validation(tmp_path / "validation"):
            next(backend.frames(threading.Event()))
    assert raised.value is failure and scenario._content_hash is broken


def test_runtime_arguments_exclude_oracle_and_any_prefilled_ball_or_route_data(tmp_path):
    module = tool()
    config = {key: str(tmp_path / key) for key in module.MODEL_PATHS}
    config.update(observed_balls=[{"fake": True}], route={"fake": True}, oracle="not-an-input")
    options = module.runtime_options(tmp_path / "input", tmp_path / "output", config)
    assert "--s1-guidance" in options and "--session-policy" in options
    assert "--scenario-video" in options
    assert not any("oracle" in value or "fake" in value for value in options)
    assert options[options.index("--device") + 1] == "cuda"


@pytest.mark.parametrize("mode", ["recommendation", "planner-failure"])
def test_positive_and_planner_failure_modes_cannot_pass_without_the_real_call_trace(mode):
    with pytest.raises(ValueError, match="trace"):
        tool().assess_guidance(mode, [], [], [], {"shots": []}, [])


def test_motion_time_observation_rejects_even_if_other_output_claims_success():
    trace = [{"event": "observe_enter", "frame_id": 12}, {"event": "observe_exit", "frame_id": 12}]
    oracle = {
        "shots": [
            {
                "samples": [
                    {"balls": {"0": [1, 1]}},
                    {"balls": {"0": [2, 1]}},
                    {"balls": {"0": [3, 1]}},
                ]
            }
        ]
    }
    intervals = [{"motion_start_frame": 10, "final_static_frame": 13}]
    with pytest.raises(ValueError, match="motion"):
        tool().assess_guidance("recommendation", [], [], trace, oracle, intervals)


def test_missing_model_control_is_not_a_positive_recommendation_result():
    samples = [{"guidance_phase": "failed", "model_state": "failed", "tasks_started": 0}]
    dom = [
        {
            "phase": "系统自动推荐 · 建议暂不可用",
            "has_dash": False,
            "has_region": False,
            "image_width": 1920,
        }
    ]
    result = tool().assess_guidance(
        "model-failure",
        samples,
        dom,
        [{"event": "factory_error", "frame_id": None}],
        {"shots": []},
        [],
    )
    assert result["control"] == "model-failure" and result["positive_recommendation"] is False


@pytest.mark.parametrize("change", [None, "reloaded", "pid_changed", "no_cold_timing"])
def test_ready_match_reuse_requires_real_worker_identity_and_separate_cold_timing(change):
    samples = [
        dict(
            model_state="ready",
            worker_pid=42,
            model_loads=1,
            worker_restarts=0,
            resource_preparation_seconds=90,
        )
        for _ in range(2)
    ]
    trace = [dict(event="factory_enter", pid=42)]
    if change == "reloaded":
        trace.append(dict(event="factory_enter", pid=43))
    elif change == "pid_changed":
        samples[-1]["worker_pid"] = 43
    elif change == "no_cold_timing":
        samples[-1]["resource_preparation_seconds"] = None
    if change:
        with pytest.raises(ValueError, match="worker reuse"):
            tool().assess_worker_reuse(samples, trace)
    else:
        result = tool().assess_worker_reuse(samples, trace)
        assert result == dict(worker_pid=42, model_loads=1, cold_preparation_seconds=90)


def test_native_playback_selects_clip_waits_for_metadata_then_uses_keyboard_controls(monkeypatch):
    module = tool()
    calls = []

    class Browser:
        def command(self, *args):
            calls.append(args)

    states = iter(
        [
            {"video_duration": None, "video_error": None},
            {"video_duration": 1.2, "video_error": None},
            {"video_time": 0.5, "video_error": None, "video_ended": False},
            {"video_time": 1.2, "video_error": None, "video_ended": True},
        ]
    )
    monkeypatch.setattr(module, "browser_state", lambda browser: next(states))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    result = module.play_visible_clip(Browser(), "回看 shot-0001")
    assert calls == [
        ("find", "role", "button", "click", "--name", "回看 shot-0001"),
        ("focus", "#postgame-player"),
        ("press", "Space"),
    ]
    assert result["observations"][-1]["video_ended"] is True


def test_one_second_recording_cannot_cover_a_full_real_case():
    with pytest.raises(ValueError, match="duration"):
        tool().assess_screen_duration({"seconds": 1.0}, 100.0, 220.0)


def test_recording_duration_must_cover_the_owned_capture_wall_clock():
    tool().assess_screen_duration({"seconds": 120.0}, 100.0, 220.0)


def test_postgame_requires_prior_and_actual_values_in_both_api_and_dom():
    with pytest.raises(ValueError, match="review"):
        tool().assess_postgame(
            {
                "case": "inside",
                "intervals": [{"motion_start_frame": 10, "final_static_frame": 20}],
                "frame_count": 100,
            },
            {"clips": [{"shot_id": "shot-0001"}]},
            [],
            {"shots": []},
        )


def test_initial_waiting_label_cannot_mask_stale_overlay_during_real_motion():
    dom = [
        dict(
            monotonic=100.0, observed_at=100.1, phase="等待停稳", has_dash=False, has_region=False
        ),
        dict(
            monotonic=101.7, observed_at=101.8, phase="系统自动推荐", has_dash=True, has_region=True
        ),
    ]
    with pytest.raises(ValueError, match="motion"):
        tool().assess_motion_display(dom, [(60, 120)], origin_seconds=100.0)


def test_missing_motion_dom_evidence_cannot_pass():
    with pytest.raises(ValueError, match="motion"):
        tool().assess_motion_display([], [(60, 120)], origin_seconds=100.0)


def test_uncaught_browser_errors_cannot_be_accepted():
    with pytest.raises(ValueError, match="browser"):
        tool().assess_browser_errors({"errors": ["uncaught TypeError"]})


def test_motion_hiding_is_tied_to_each_real_window():
    dom = [
        dict(monotonic=t, observed_at=t + 0.02, phase="等待停稳", has_dash=False, has_region=False)
        for t in (101.2, 101.8, 103.3)
    ]
    delays = tool().assess_motion_display(dom, [(60, 120), (180, 240)], origin_seconds=100.0)
    assert delays == pytest.approx([0.22, 0.32])


def test_postgame_uses_first_shot_prior_advice_separate_from_actual_measurement():
    manifest = dict(
        case="inside",
        intervals=[dict(motion_start_frame=60, final_static_frame=120)],
        frame_count=300,
    )
    value = dict(
        recommendation=dict(source_frame_id=5, region=dict(center=dict(x_mm=10, y_mm=20))),
        actual_cue=dict(x_mm=12, y_mm=22),
        motion_frame_id=65,
        observed_frame_id=200,
        region_result="inside",
    )
    review = dict(clips=[dict(shot_id="shot-0001", guidance_review=value)])
    dom = [
        dict(
            shot_rows=[
                dict(
                    shot_id="shot-0001",
                    text="击球前建议区中心：(10,20)；实际白球终点：(12,22) mm；已进入建议区",
                )
            ]
        )
    ]
    oracle = dict(shots=[dict(final={"0": [12, 22]})])
    result = tool().assess_postgame(manifest, review, dom, oracle)
    assert result["region_result"] == "inside"
    dom[0]["shot_rows"][0]["shot_id"] = "shot-0002"
    with pytest.raises(ValueError, match="DOM"):
        tool().assess_postgame(manifest, review, dom, oracle)


def test_slow_dom_query_cannot_backdate_late_motion_hiding():
    dom = [
        dict(monotonic=101.2, observed_at=102.2, phase="等待停稳", has_dash=False, has_region=False)
    ]
    with pytest.raises(ValueError, match="motion"):
        tool().assess_motion_display(dom, [(60, 180)], origin_seconds=100.0)


def test_windows_case_cannot_start_without_exact_owned_release():
    module = tool()
    run_id = "a" * 32
    with pytest.raises(ValueError, match="owned"):
        module.await_owned_release(None, platform="nt", stream=io.StringIO())
    with pytest.raises(ValueError, match="owned"):
        module.await_owned_release(run_id, platform="nt", stream=io.StringIO("wrong\n"))
    module.await_owned_release(
        run_id, platform="nt", stream=io.StringIO("S1-SCREEN-RELEASE " + run_id + "\n")
    )
    module.await_owned_release(None, platform="posix", stream=io.StringIO())


def test_killed_screen_model_factory_retains_its_real_stack_and_stage(tmp_path):
    code = """
import pathlib, threading, sys
from scripts.development import s1_screen_acceptance as screen
screen.MODEL_FACTORY_STACK_SECONDS = .05
def blocked_original_factory():
    threading.Event().wait()
screen.TracedFactory(blocked_original_factory, pathlib.Path(sys.argv[1])/'calls.jsonl')()
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path)])
    try:
        deadline = time.monotonic() + 5
        stacks = []
        while time.monotonic() < deadline:
            stacks = list(tmp_path.glob("model-preparation/*/local-stack.txt"))
            if stacks and stacks[0].stat().st_size:
                break
            time.sleep(0.01)
        assert process.poll() is None
        assert len(stacks) == 1 and "blocked_original_factory" in stacks[0].read_text()
    finally:
        process.terminate()
        process.wait(timeout=5)
    receipt = json.loads((stacks[0].parent / "local-stages.json").read_bytes())
    assert [(e["stage"], e["event"]) for e in receipt["events"]] == [("model_factory", "enter")]
    assert [
        json.loads(x)["event"] for x in (tmp_path / "calls.jsonl").read_bytes().splitlines()
    ] == ["factory_enter"]


def test_screen_factory_preserves_original_observer_and_exception(tmp_path):
    module = tool()
    original = object()
    assert module.TracedFactory(lambda: original, tmp_path / "calls.jsonl")().original is original
    failure = RuntimeError("private model failure")

    def fail():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        module.TracedFactory(fail, tmp_path / "calls.jsonl")()
    assert caught.value is failure
    receipts = list(tmp_path.glob("model-preparation/*/local-stages.json"))
    assert len(receipts) == 2
    assert {json.loads(p.read_bytes())["events"][-1]["event"] for p in receipts} == {
        "exit",
        "error",
    }
    assert all(p.read_bytes() == b"" for p in tmp_path.glob("model-preparation/*/local-stack.txt"))


def test_real_preparation_failure_is_persisted_before_positive_case_stops(tmp_path):
    module = tool()
    path = tmp_path / "local-guidance-samples.jsonl"
    sample = dict(model_state="failed", error="guidance_preparation_timeout", frame_id=3000)
    with pytest.raises(ValueError, match="guidance_preparation_timeout"):
        module.record_guidance_sample(path, sample, mode="recommendation")
    assert [json.loads(x) for x in path.read_bytes().splitlines()] == [sample]


def test_negative_planner_case_keeps_original_failure_sample_and_continues(tmp_path):
    module = tool()
    path = tmp_path / "local-guidance-samples.jsonl"
    sample = dict(model_state="ready", guidance_phase="failed", error="RuntimeError:planner")
    module.record_guidance_sample(path, sample, mode="planner-failure")
    assert json.loads(path.read_bytes()) == sample


def test_guidance_sample_file_has_a_fixed_local_budget(tmp_path):
    path = tmp_path / "local-guidance-samples.jsonl"
    path.write_bytes(b" " * (2 * 1024**2))
    with pytest.raises(ValueError, match="budget"):
        tool().record_guidance_sample(path, {"model_state": "ready"}, mode="recommendation")
    assert path.stat().st_size == 2 * 1024**2


def test_preparation_prefix_and_pickle_cannot_postpone_factory_stack_deadline(tmp_path):
    code = """
import pathlib, sys, time, pickle
from scripts.development import s1_screen_acceptance as screen
root = pathlib.Path(sys.argv[1])
screen.MODEL_FACTORY_STACK_SECONDS = 1.0
def original_factory():
    deadline = time.monotonic() + .25
    while time.monotonic() < deadline:
        stacks = list(root.glob('model-preparation/*/local-stack.txt'))
        if stacks and stacks[0].stat().st_size:
            return object()
        time.sleep(.005)
    raise RuntimeError('diagnostic deadline was restarted after preparation prefix')
factory = screen.TracedFactory(original_factory, root/'calls.jsonl')
factory = pickle.loads(pickle.dumps(factory))
time.sleep(1.05)
factory()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    stack = next(tmp_path.glob("model-preparation/*/local-stack.txt"))
    assert "original_factory" in stack.read_text()


def test_real_cancel_error_state_is_saved_and_stops_positive_case(tmp_path):
    module = tool()
    path = tmp_path / "local-guidance-samples.jsonl"
    sample = dict(
        model_state="not_started",
        guidance_phase="failed",
        error="RuntimeError:guidance child did not terminate",
    )
    with pytest.raises(ValueError, match="guidance child did not terminate"):
        module.record_guidance_sample(path, sample, mode="recommendation")
    assert json.loads(path.read_bytes()) == sample


def test_expired_factory_timer_cannot_be_consumed_by_slow_stage_receipt(tmp_path):
    code = """
import pathlib, sys, time
from scripts.development import s1_screen_acceptance as screen, s1_probe_diagnostics as diag
root = pathlib.Path(sys.argv[1])
write = diag.atomic_json
def slow_receipt(*args, **kwargs):
    time.sleep(.05)
    return write(*args, **kwargs)
diag.atomic_json = slow_receipt
screen.MODEL_FACTORY_STACK_SECONDS = .01
def actual_original_factory():
    time.sleep(.1)
    return object()
factory = screen.TracedFactory(actual_original_factory, root/'calls.jsonl')
time.sleep(.03)
factory()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    stack = next(tmp_path.glob("model-preparation/*/local-stack.txt"))
    assert "in actual_original_factory" in stack.read_text()


def _blocked_original_detector_worker(root):
    import faulthandler

    root = Path(root)
    from billiards_ai.lab import mainline1_runtime_support as runtime
    from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory
    from scripts.development.s1_screen_acceptance import TracedFactory

    def original_detector(path, *, device):
        (root / "detector-call.txt").write_text("once")
        threading.Event().wait()

    runtime.load_development_vision_backend = lambda path, model_factory: model_factory(path)
    runtime._detector = original_detector

    def forbidden_watchdog(*args, **kwargs):
        raise AssertionError("actual staged screen must not arm a native watchdog")

    faulthandler.dump_traceback_later = forbidden_watchdog
    original = ModelSnapshotObserverFactory(root, root, root)
    TracedFactory(original, root / "calls.jsonl", trace_model_stages=True)()


def test_real_spawn_keeps_nested_original_model_stage_when_terminated(tmp_path):
    process = multiprocessing.get_context("spawn").Process(
        target=_blocked_original_detector_worker, args=(str(tmp_path),)
    )
    process.start()
    receipt = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and process.is_alive():
            paths = list(tmp_path.glob("model-preparation/*/local-stages.json"))
            if paths:
                receipt = json.loads(paths[0].read_bytes())
                if receipt["events"][-1]["stage"] == "model_detector_initialization":
                    break
            time.sleep(0.01)
        assert process.is_alive()
        assert receipt["events"][-1]["event"] == "enter"
        assert receipt["events"][-1]["stage"] == "model_detector_initialization"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        assert not process.is_alive()
        process.close()
    assert (tmp_path / "detector-call.txt").read_text() == "once"
    assert not list(tmp_path.glob("model-preparation/*/local-stack.txt"))
    assert [(x["stage"], x["event"]) for x in receipt["events"]] == [
        ("model_factory", "enter"),
        ("model_runtime_import", "enter"),
        ("model_runtime_import", "exit"),
        ("model_vision_loader", "enter"),
        ("model_detector_initialization", "enter"),
    ]


def test_call_trace_records_actual_native_pid_and_wall_clock_for_external_binding(tmp_path):
    module = tool()
    before = time.time_ns()
    module.trace_event(tmp_path / "calls.jsonl", "factory_enter")
    after = time.time_ns()
    event = json.loads((tmp_path / "calls.jsonl").read_bytes())
    assert event["pid"] == os.getpid()
    assert before <= event["wall_time_ns"] <= after
    assert event["seconds"] <= time.perf_counter()


def test_storage_observer_keeps_real_original_fsync_result_and_restores_method(tmp_path):
    from billiards_ai.lab.mainline1_session_recording import LocalStorageProbe

    original = LocalStorageProbe.benchmark_write_mbps
    root = tmp_path / "recording"
    root.mkdir()
    calls = []
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is original.__code__:
            calls.append(frame.f_locals["root"])

    sys.setprofile(profile)
    try:
        with tool().trace_storage_preflight(tmp_path / "diagnostics"):
            speed = LocalStorageProbe().benchmark_write_mbps(root)
    finally:
        sys.setprofile(previous)
    assert calls == [root]
    assert LocalStorageProbe.benchmark_write_mbps is original
    assert not list(root.glob(".recording-preflight-*"))
    receipt = json.loads(next((tmp_path / "diagnostics").glob("*.json")).read_bytes())
    assert receipt["status"] == "RETURNED"
    assert receipt["measured_write_mbps"] == speed and speed > 0
    assert receipt["elapsed_seconds"] >= 0


def test_storage_observer_keeps_real_file_error_and_restores_method(tmp_path):
    from billiards_ai.lab.mainline1_session_recording import LocalStorageProbe

    original = LocalStorageProbe.benchmark_write_mbps
    with pytest.raises(FileNotFoundError):
        with tool().trace_storage_preflight(tmp_path / "diagnostics"):
            LocalStorageProbe().benchmark_write_mbps(tmp_path / "missing")
    assert LocalStorageProbe.benchmark_write_mbps is original
    receipt = json.loads(next((tmp_path / "diagnostics").glob("*.json")).read_bytes())
    assert receipt["status"] == "RAISED" and receipt["exception_type"] == "FileNotFoundError"


@pytest.mark.parametrize("session_fails", [False, True])
def test_original_model_factory_stage_proxies_restore_functions_and_preserve_values(
    tmp_path, monkeypatch, session_fails
):
    from billiards_ai.lab import mainline1_runtime_support as runtime
    from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory

    calls = []
    identity = object()
    vision = type(
        "Vision", (), {"prepare_execution": lambda self, **kwargs: calls.append("execution")}
    )()
    original_execution = vision.prepare_execution
    features = type("Features", (), {"prepare": lambda self: calls.append("weights")})()
    original_prepare = features.prepare
    failure = RuntimeError("original private session failure")

    def vision_loader(path, *, model_factory):
        calls.append("vision")
        return model_factory(path)

    def detector(path, *, device):
        calls.append("detector")
        assert path == tmp_path and device == "cpu"
        return vision

    def feature_factory(*, asset_root, device, batch_size):
        calls.append("features")
        assert (asset_root, device, batch_size) == (tmp_path, "cpu", 8)
        return features

    def group_loader(path, *, feature_backend, evidence_backend, session_factory):
        calls.append("group")
        assert feature_backend is features
        return session_factory(path)

    def session(path):
        calls.append("session")
        if session_fails:
            raise failure
        return identity

    functions = dict(
        load_development_vision_backend=vision_loader,
        _detector=detector,
        Dinov2IdentityFeatureBackend=feature_factory,
        load_development_group_classifier=group_loader,
        _session=session,
    )
    for name, function in functions.items():
        monkeypatch.setattr(runtime, name, function)
    original = ModelSnapshotObserverFactory(tmp_path, tmp_path, tmp_path)
    factory = tool().TracedFactory(original, tmp_path / "calls.jsonl", trace_model_stages=True)
    if session_fails:
        with pytest.raises(RuntimeError) as caught:
            factory()
        assert caught.value is failure
    else:
        observer = factory().original
        assert observer.vision is vision and observer.identity is identity
    assert calls == ["vision", "detector", "features", "weights", "group", "session"] + (
        [] if session_fails else ["execution"]
    )
    assert features.prepare == original_prepare
    assert vision.prepare_execution == original_execution
    assert all(getattr(runtime, name) is function for name, function in functions.items())
    receipt = json.loads(next(tmp_path.glob("model-preparation/*/local-stages.json")).read_bytes())
    assert receipt["events"][-1]["event"] == ("error" if session_fails else "exit")
    expected_stages = {
        "model_factory",
        "model_runtime_import",
        "model_vision_loader",
        "model_detector_initialization",
        "model_identity_features_setup",
        "model_identity_weights_preparation",
        "model_group_loader",
        "model_onnx_session_initialization",
    }
    if not session_fails:
        expected_stages.add("model_detector_execution_preparation")
    assert {x["stage"] for x in receipt["events"]} == expected_stages


def test_original_detector_preparation_records_actual_parameter_device_without_loading_models(
    tmp_path,
):
    from types import SimpleNamespace

    from scripts.development.s1_probe_diagnostics import ProbeDiagnostics

    parameter = SimpleNamespace(device="cuda:0", dtype="torch.float16")
    model = SimpleNamespace(
        model=SimpleNamespace(
            device="cuda:0",
            inference_model=SimpleNamespace(parameters=lambda: iter((parameter,))),
        )
    )
    with ProbeDiagnostics(tmp_path, "a" * 32) as trace:
        result = tool().model_stage_call(trace, "model_detector_initialization", lambda: model)
    assert result is model
    receipt = json.loads((tmp_path / "detector-runtime.json").read_bytes())
    assert receipt == dict(
        status="AVAILABLE",
        declared_device="cuda:0",
        parameter_device="cuda:0",
        parameter_dtype="torch.float16",
    )


@pytest.mark.parametrize("identity_fails", [False, True])
def test_staged_original_observer_separates_detection_and_identity_without_replacing_data(
    tmp_path, identity_fails
):
    from types import SimpleNamespace

    from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserver
    from tests.unit.lab.field_assistance.test_snapshot_guidance import frame

    calls = []
    failure = RuntimeError("controlled original identity inference error")

    class Vision:
        def infer(self, value):
            calls.append("detector")
            return SimpleNamespace(
                frame_id=value.frame_id, timestamp_ms=value.timestamp_ms, detections=()
            )

    class Identity:
        def classify(self, *, image, detections):
            calls.append("identity")
            assert image.shape == (240, 320, 3) and detections == ()
            if identity_fails:
                raise failure
            return {}

    original = ModelSnapshotObserver(Vision(), Identity())
    methods = original.vision.infer, original.identity.classify
    observer = tool().TracedObserver(original, tmp_path / "calls.jsonl", trace_model_stages=True)
    if identity_fails:
        with pytest.raises(RuntimeError) as raised:
            observer.observe(frame(5), "unused with no detections")
        assert raised.value is failure
    else:
        result = observer.observe(frame(5), "unused with no detections")
        assert result.frame_id == 5 and result.observations == ()
    assert (original.vision.infer, original.identity.classify) == methods
    assert calls == ["detector", "identity"]
    receipt = json.loads(next(tmp_path.glob("observation-stages/*/local-stages.json")).read_bytes())
    assert [(row["stage"], row["event"]) for row in receipt["events"]] == [
        ("detector_inference", "enter"),
        ("detector_inference", "exit"),
        ("identity_inference", "enter"),
        ("identity_inference", "error" if identity_fails else "exit"),
    ]
