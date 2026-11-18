"""Record the real local S1 screen and independent durable footage.

Development-only observer. Video pixels, calibration and normal settings are the
only table inputs. Test truth is inspected only after the service has ended.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.development import blackbox_source as sources  # noqa: E402
from scripts.development.blackbox_report import atomic_json  # noqa: E402
from scripts.development.s1_browser_capture import SCREEN_FPS, BrowserCapture  # noqa: E402
from scripts.development.s1_dom_observer import DOMObserver  # noqa: E402
from scripts.development.s1_probe_diagnostics import ProbeDiagnostics  # noqa: E402

MODEL_PATHS = ("detector_training_run", "identity_training_run", "dinov2_assets")
MODEL_FACTORY_STACK_SECONDS = 40
FILE_PREPARATION_SECONDS = 20
FEEDBACK_VISIBLE_SECONDS = (0.7, 1.3)
FEEDBACK_MAX_INTERVAL_WIDTH = 0.25
DOM_STATE = r"""(()=>{const get=id=>document.getElementById(id),svg=get('route-overlay'),
image=get('route-frame'),video=get('postgame-player');return {
phase:get('phase')?.textContent||'',note:get('snapshot-note')?.textContent||'',
image_width:image?.naturalWidth||0,image_height:image?.naturalHeight||0,
has_dash:Boolean(svg?.querySelector('polyline[stroke-dasharray]')),
has_region:Boolean(svg?.querySelector('polygon')),has_halo:Boolean(svg?.querySelector('circle')),
route_lines:Array.from(svg?.querySelectorAll('polyline')||[]).map(n=>n.getAttribute('points')),
region_points:svg?.querySelector('polygon')?.getAttribute('points')||null,
feedback_motion_frame_id:svg?.getAttribute('data-s1-feedback-motion')
  ?Number(svg.getAttribute('data-s1-feedback-motion')):null,
observed_frame_id:svg?.getAttribute('data-s1-observed-frame')
  ?Number(svg.getAttribute('data-s1-observed-frame')):null,
postgame_hidden:get('postgame-panel')?.hidden??true,
postgame_text:get('postgame-shot-list')?.textContent||'',
shot_rows:Array.from(document.querySelectorAll('#postgame-shot-list li'))
.map(row=>({shot_id:row.dataset.shotId,text:row.textContent})),
play_buttons:Array.from(document.querySelectorAll('#postgame-shot-list button'))
.filter(b=>!b.disabled&&b.textContent.startsWith('回看 ')).map(b=>b.textContent),
video_time:video?.currentTime||0,video_duration:Number.isFinite(video?.duration)?video.duration:null,
video_paused:video?.paused??true,video_ended:video?.ended??false,
video_error:video?.error?.code||null};})()"""


def capture_work_seconds(frame_count, authorized_seconds):
    if authorized_seconds is None:
        return frame_count / 60 + 60
    if type(authorized_seconds) is not int or authorized_seconds != 240:
        raise ValueError("fixed owned capture allowance must be 240 seconds")
    return authorized_seconds


def validate_capture_completion(deadline, completed_at):
    if type(completed_at) not in (int, float) or not math.isfinite(completed_at):
        raise ValueError("missing real capture completion receipt")
    if completed_at > deadline:
        raise TimeoutError("fixed input capture budget exhausted")


class FrameArrivalClock:
    """Development observation only: record before each original frame is yielded."""

    def __init__(self, frame_count):
        if type(frame_count) is not int or not 2 <= frame_count <= 12000:
            raise ValueError("invalid arrival frame budget")
        self.frame_count = frame_count
        self.times = []

    def record(self, frame_id, *, at=None):
        # Python 3.12 on Windows has coarse GetTickCount64 monotonic ticks.
        # Arrivals, DOM brackets and worker call traces share the real QPC clock.
        at = time.perf_counter() if at is None else at
        if (
            type(frame_id) is not int
            or frame_id != len(self.times)
            or frame_id >= self.frame_count
            or type(at) not in {float, int}
            or not math.isfinite(at)
            or at < 0
            or (self.times and at <= self.times[-1])
        ):
            raise ValueError("invalid or nonsequential frame arrival")
        self.times.append(at)

    def complete(self):
        if len(self.times) != self.frame_count:
            raise ValueError("incomplete frame arrival clock")
        return list(self.times)


def motion_wall_windows(motion, *, origin_seconds=None, arrival_times=None):
    if arrival_times is not None:
        if (
            not 2 <= len(arrival_times) <= 12000
            or any(type(t) not in {float, int} or not math.isfinite(t) for t in arrival_times)
            or any(b <= a for a, b in zip(arrival_times, arrival_times[1:], strict=False))
            or any(not 0 <= begin <= end < len(arrival_times) - 1 for begin, end in motion)
        ):
            raise ValueError("missing or nonmonotonic motion arrival evidence")
        # Last moving pixels remain current until the following delivered frame.
        return [(arrival_times[begin], arrival_times[end + 1]) for begin, end in motion]
    if origin_seconds is None:
        raise ValueError("motion display evidence requires the capture clock")
    # Compatibility for existing nominal-clock unit fixtures only. The actual
    # screen run always supplies the independently observed arrival clock.
    return [(origin_seconds + begin / 60, origin_seconds + end / 60) for begin, end in motion]


def trace_event(path, event, frame_id=None):
    path = sources.plain(path)
    if path.exists() and path.stat().st_size > 64 * 1024:
        raise ValueError("guidance call trace exceeds budget")
    data = (
        json.dumps(
            dict(
                event=event,
                frame_id=frame_id,
                seconds=time.perf_counter(),
                pid=os.getpid(),
                wall_time_ns=time.time_ns(),
                python_import_time=bool(sys._xoptions.get("importtime")),
            )
        )
        + "\n"
    ).encode()
    with path.open("ab", buffering=0) as stream:
        stream.write(data)
        os.fsync(stream.fileno())


def record_guidance_sample(path, sample, *, mode):
    """Keep actual state before owner termination; never infer a successful case."""
    path = sources.plain(path)
    raw = (json.dumps(sample) + "\n").encode()
    if len(raw) + (path.stat().st_size if path.exists() else 0) > 2 * 1024**2:
        raise ValueError("local guidance sample budget exceeded")
    with path.open("ab", buffering=0) as stream:
        stream.write(raw)
        os.fsync(stream.fileno())
    if mode == "recommendation" and (
        sample.get("model_state") == "failed" or sample.get("guidance_phase") == "failed"
    ):
        raise ValueError("real S1 recommendation failed: " + str(sample.get("error")))


def record_dom_observation(path, value):
    raw = (json.dumps(value) + "\n").encode()
    if len(raw) + (path.stat().st_size if path.exists() else 0) > 2 * 1024**2:
        raise ValueError("local DOM observation budget exceeded")
    with path.open("ab") as stream:
        stream.write(raw)


@contextmanager
def screen_stage(report, output, diagnostics, stage):
    """Persist local progress before and after opaque operations, never success."""
    path = output / "progress-report.json"
    report["active_stage"] = stage
    atomic_json(path, report)
    try:
        with diagnostics.stage(stage):
            yield
    except BaseException as exc:
        report["stage_failure"] = dict(stage=stage, type=type(exc).__name__)
        try:
            atomic_json(path, report)
        except Exception:
            pass
        raise
    report["active_stage"] = None
    report["last_completed_stage"] = stage
    atomic_json(path, report)


def cleanup_step(checkpoint, stage, operation):
    attempted = False
    try:
        with checkpoint(stage):
            attempted = True
            operation()
    finally:
        # Progress I/O may fail; it cannot prevent the original release/cancel.
        if not attempted:
            operation()


@contextmanager
def trace_scenario_validation(output):
    """Observe the original content check, including a call that never returns.

    Only full validations receive receipts; per-frame stat checks keep their
    original work. The capture startup deadline and source identity gate stay
    unchanged. These diagnostic writes have their own observation overhead.
    """
    from billiards_ai.lab import mainline1_scenario_capture as scenario

    output = sources.plain(output)
    output.mkdir()
    active = threading.local()
    allocation = threading.Lock()
    original = scenario.ScenarioVideoCaptureBackend._validate_source

    def invoke(stage, method, *args, **kwargs):
        receipt = getattr(active, "receipt", None)
        if receipt is None:
            return method(*args, **kwargs)
        path, row, started = receipt

        def record(event, error=None):
            row["events"].append(
                dict(
                    stage=stage,
                    event=event,
                    seconds=time.monotonic() - started,
                    error_type=None if error is None else type(error).__name__,
                )
            )
            atomic_json(path, row)

        record("ENTERED")
        try:
            value = method(*args, **kwargs)
        except BaseException as exc:
            try:
                record("RAISED", exc)
            except Exception:
                pass  # A diagnostic write cannot replace the original failure.
            raise
        record("RETURNED")
        return value

    def validate(backend, *, content=False):
        if not content:
            return original(backend, content=False)
        with allocation:
            if len(list(output.glob("*.json"))) >= 16:
                raise ValueError("scenario validation trace exceeds budget")
            path = output / (uuid4().hex + ".json")
            started = time.monotonic()
            row = dict(
                contract="s1-scenario-validation-trace/v1",
                pid=os.getpid(),
                thread_id=threading.get_ident(),
                started_at_ns=time.monotonic_ns(),
                bytes=backend._source_stat[2],
                events=[],
            )
            atomic_json(path, row)
        active.receipt = path, row, started
        try:
            return invoke("validation", original, backend, content=True)
        finally:
            active.receipt = None

    with ExitStack() as proxies:
        for name, stage in (("_source_stat", "source_stat"), ("_content_hash", "content_hash")):
            proxies.enter_context(
                patch.object(scenario, name, partial(invoke, stage, getattr(scenario, name)))
            )
        proxies.enter_context(
            patch.object(scenario.ScenarioVideoCaptureBackend, "_validate_source", validate)
        )
        yield


@contextmanager
def trace_storage_preflight(output):
    """Observe the original recording probe; its gate remains in the product."""
    from billiards_ai.lab.mainline1_session_recording import LocalStorageProbe

    output = sources.plain(output)
    output.mkdir()
    original = LocalStorageProbe.benchmark_write_mbps

    def measured(probe, root):
        if len(list(output.glob("*.json"))) >= 16:
            raise ValueError("storage diagnostic receipt budget exceeded")
        receipt = output / (uuid4().hex + ".json")
        value = dict(status="ENTERED", pid=os.getpid(), wall_time_ns=time.time_ns())
        atomic_json(receipt, value)
        started = time.monotonic()
        try:
            speed = original(probe, root)
        except BaseException as exc:
            value.update(
                status="RAISED",
                exception_type=type(exc).__name__,
                elapsed_seconds=time.monotonic() - started,
            )
            try:
                atomic_json(receipt, value)
            except Exception:
                pass  # Preserve the original probe failure even if diagnostics fail.
            raise
        value.update(
            status="RETURNED",
            measured_write_mbps=speed,
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(receipt, value)
        return speed

    with patch.object(LocalStorageProbe, "benchmark_write_mbps", measured):
        yield


@dataclass
class TracedObserver:
    original: object
    path: Path
    trace_model_stages: bool = False

    def observe(self, frame, calibration):
        trace_event(self.path, "observe_enter", frame.capture_frame_id)
        with ExitStack() as originals:
            if self.trace_model_stages:
                root = self.path.parent / "observation-stages"
                if root.exists() and len(list(root.iterdir())) >= 16:
                    raise ValueError("observation diagnostic budget exceeded")
                diagnostics = root / uuid4().hex
                diagnostics.mkdir(parents=True)
                trace = originals.enter_context(ProbeDiagnostics(diagnostics, diagnostics.name))
                for backend, method, stage in (
                    (self.original.vision, "infer", "detector_inference"),
                    (self.original.identity, "classify", "identity_inference"),
                ):
                    originals.enter_context(
                        patch.object(
                            backend,
                            method,
                            partial(model_stage_call, trace, stage, getattr(backend, method)),
                        )
                    )
            value = self.original.observe(frame, calibration)
        trace_event(self.path, "observe_exit", frame.capture_frame_id)
        return value


def model_stage_call(trace, stage, original, *args, **kwargs):
    with trace.stage(stage):
        result = original(*args, **kwargs)
        if stage == "model_detector_initialization":
            context = getattr(result, "model", None)
            network = getattr(context, "inference_model", None)
            if network is None:
                network = getattr(context, "model", None)
            parameter = next(network.parameters(), None) if network is not None else None
            value = dict(status="UNAVAILABLE")
            if parameter is not None:
                declared = str(getattr(context, "device", ""))
                device, dtype = str(parameter.device), str(parameter.dtype)
                if (
                    re.fullmatch(r"cpu|cuda(?::[0-9]+)?|mps", declared)
                    and re.fullmatch(r"cpu|cuda(?::[0-9]+)?|mps", device)
                    and dtype in {"torch.float16", "torch.float32", "torch.bfloat16"}
                ):
                    value = dict(
                        status="AVAILABLE",
                        declared_device=declared,
                        parameter_device=device,
                        parameter_dtype=dtype,
                    )
            atomic_json(trace.root / "detector-runtime.json", value)
        return result


def model_prepared_backend(
    trace, originals, setup_stage, prepare_method, prepare_stage, original, *args, **kwargs
):
    backend = model_stage_call(trace, setup_stage, original, *args, **kwargs)
    originals.enter_context(
        patch.object(
            backend,
            prepare_method,
            partial(model_stage_call, trace, prepare_stage, getattr(backend, prepare_method)),
        )
    )
    return backend


@dataclass(frozen=True)
class TracedFactory:
    original: object
    path: Path
    stack_at: float = field(default_factory=lambda: time.monotonic() + MODEL_FACTORY_STACK_SECONDS)
    trace_model_stages: bool = False

    def __call__(self):
        trace_event(self.path, "factory_enter")
        diagnostics = self.path.parent / "model-preparation" / uuid4().hex
        diagnostics.mkdir(parents=True)
        try:
            # Actual screen diagnostics use durable stages without a native
            # watchdog. Legacy stack diagnostics retain the parent's deadline.
            # Neither mode changes the original preparation gate or reap.
            with ProbeDiagnostics(
                diagnostics,
                diagnostics.name,
            ) as trace:
                with trace.stage("model_factory"):
                    with ExitStack() as originals:
                        if self.trace_model_stages:
                            # This is the original factory's first import,
                            # observed here in the same owned child call.
                            with trace.stage("model_runtime_import"):
                                from billiards_ai.lab import mainline1_runtime_support as runtime

                            for name, stage in (
                                ("_detector", "model_detector_initialization"),
                                ("load_development_group_classifier", "model_group_loader"),
                                ("_session", "model_onnx_session_initialization"),
                            ):
                                originals.enter_context(
                                    patch.object(
                                        runtime,
                                        name,
                                        partial(
                                            model_stage_call, trace, stage, getattr(runtime, name)
                                        ),
                                    )
                                )
                            for name, setup, method, preparation in (
                                (
                                    "load_development_vision_backend",
                                    "model_vision_loader",
                                    "prepare_execution",
                                    "model_detector_execution_preparation",
                                ),
                                (
                                    "Dinov2IdentityFeatureBackend",
                                    "model_identity_features_setup",
                                    "prepare",
                                    "model_identity_weights_preparation",
                                ),
                            ):
                                originals.enter_context(
                                    patch.object(
                                        runtime,
                                        name,
                                        partial(
                                            model_prepared_backend,
                                            trace,
                                            originals,
                                            setup,
                                            method,
                                            preparation,
                                            getattr(runtime, name),
                                        ),
                                    )
                                )
                        if not self.trace_model_stages:
                            trace.arm_stack(self.stack_at)
                        value = self.original()
        except Exception:
            trace_event(self.path, "factory_error")
            raise
        trace_event(self.path, "factory_exit")
        return TracedObserver(value, self.path, self.trace_model_stages)


@dataclass(frozen=True)
class TracedPlanner:
    original: object
    path: Path
    fail: bool = False

    def __call__(self, state, context):
        trace_event(self.path, "planner_enter", state.frame_id)
        try:
            if self.fail:
                raise RuntimeError("controlled_planner_failure")
            result = self.original(state, context)
        except Exception:
            trace_event(self.path, "planner_error", state.frame_id)
            raise
        trace_event(self.path, "planner_exit", state.frame_id)
        return result


def runtime_options(input_root, output, config):
    input_root, output = Path(input_root), Path(output)
    paths = {
        "scenario-video": input_root / "input.avi",
        "capture-setup": input_root / "capture-setup.json",
        "calibration-root": input_root / "calibration-registry",
        "config": Path(config.get("config", ROOT / "configs/algorithm/shot-local.yaml")),
        "state-db": output / "state.sqlite3",
        "ring-root": output / "ring",
        "jobs-root": output / "jobs",
        "artifact-root": output / "artifacts",
        **{name.replace("_", "-"): Path(config[name]) for name in MODEL_PATHS},
    }
    options = []
    for key, value in paths.items():
        options += ["--" + key, str(value)]
    return options + [
        "--session-policy",
        "recording-v1",
        "--s1-guidance",
        "--runtime-environment",
        "development",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--device",
        config.get("device", "cuda"),
        "--batch-size",
        "8",
    ]


def input_identity(root):
    import cv2

    root = sources.plain(root)
    manifest_path = root / "manifest.json"
    if manifest_path.stat().st_size > 128 * 1024:
        raise ValueError("input manifest too large")
    manifest = json.loads(manifest_path.read_bytes())
    if (
        manifest.get("contract") != "s1-physical-input/v1"
        or manifest.get("synthetic_physics_only") is not True
        or manifest.get("fps") != 60
        or type(manifest.get("frame_count")) is not int
        or not 2 <= manifest["frame_count"] <= 12000
        or not re.fullmatch("[0-9a-f]{64}", str(manifest.get("calibration_id")))
    ):
        raise ValueError("fixed physical input manifest invalid")
    video = root / "input.avi"
    if sources.pin(video) != manifest["video"] or manifest["video"]["bytes"] > 1024**3:
        raise ValueError("input video changed")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        ) != (1920, 1080, 60, manifest["frame_count"]):
            raise ValueError("physical input video contract mismatch")
    finally:
        capture.release()
    names = [
        "manifest.json",
        "input.avi",
        "capture-setup.json",
        "oracle.not-product-input.json",
        "calibration-registry/registry.json",
        *[
            f"calibration-registry/{manifest['calibration_id']}/{name}"
            for name in ("manifest.json", "calibration.json", "reference-frame.jpg")
        ],
    ]
    return manifest, {name: sources.pin(root / name) for name in names}


def source_identity(allow_dirty=False):
    if (ROOT / "SOURCE-LOCK.json").is_file():
        lock = sources.verify(ROOT)
        return dict(kind="locked", producer=lock["commit"], files=lock["files"])

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    dirty = git("status", "--porcelain").decode()
    if dirty and not allow_dirty:
        raise ValueError("acceptance source must be fixed and clean")
    names = (
        git(
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *sources.ROOTS,
            sources.CONFIG,
            ".gitattributes",
        )
        .decode()
        .split("\0")
    )
    return dict(
        kind="development" if dirty else "committed",
        producer=git("rev-parse", "HEAD").decode().strip(),
        files={
            name: sources.pin(ROOT / name)
            for name in sorted(set(names))
            if name and sources.allowed(name)
        },
    )


def assess_worker_reuse(samples, trace):
    ready = [row for row in samples if row.get("model_state") == "ready"]
    factories = [row for row in trace if row["event"] == "factory_enter"]
    if not ready or len(factories) != 1:
        raise ValueError("worker reuse evidence missing or reloaded")
    pid = factories[0].get("pid")
    if (
        type(pid) is not int
        or pid <= 0
        or any(
            row.get("worker_pid") != pid
            or row.get("model_loads") != 1
            or row.get("worker_restarts") != 0
            or type(row.get("resource_preparation_seconds")) not in {int, float}
            or not 0 <= row["resource_preparation_seconds"] <= 300
            for row in ready
        )
    ):
        raise ValueError("worker reuse identity or separate cold timing invalid")
    return dict(
        worker_pid=pid,
        model_loads=1,
        cold_preparation_seconds=ready[-1]["resource_preparation_seconds"],
    )


def assess_guidance(
    mode, samples, dom, trace, oracle, intervals, *, origin_seconds=None, arrival_times=None
):
    if not trace:
        raise ValueError("real guidance call trace missing")
    motion = []
    hiding_seconds_upper_bounds = None
    for shot, interval in zip(oracle["shots"], intervals, strict=True):
        poses = shot["samples"]
        changes = [i for i in range(1, len(poses)) if poses[i]["balls"] != poses[i - 1]["balls"]]
        if changes:
            motion.append(
                (interval["motion_start_frame"] + 1, interval["motion_start_frame"] + changes[-1])
            )
    wall_windows = (
        motion_wall_windows(motion, origin_seconds=origin_seconds, arrival_times=arrival_times)
        if origin_seconds is not None or arrival_times is not None
        else []
    )
    for event in trace:
        if event["event"] in {"observe_enter", "planner_enter"} and (
            any(begin <= event["frame_id"] <= end for begin, end in motion)
            or any(begin <= event["seconds"] <= end for begin, end in wall_windows)
        ):
            raise ValueError("observation/planning entered during physical translation motion")
    if not any(row.get("image_width") == 1920 for row in dom):
        raise ValueError("actual screen never displayed camera pixels")
    if mode in {"model-failure", "planner-failure"}:
        if any(d.get("has_dash") or d.get("has_region") or d.get("has_halo") for d in dom):
            raise ValueError("actual browser displayed advice in the controlled failure case")
        if not any("建议暂不可用" in d.get("phase", "") for d in dom):
            raise ValueError("actual browser did not present the controlled failure")
    if mode == "model-failure":
        if not any(e["event"] == "factory_error" for e in trace) or any(
            e["event"] == "observe_enter" for e in trace
        ):
            raise ValueError("missing-model control did not fail in real model preparation")
        if not any(
            s.get("guidance_phase") == "failed"
            and s.get("model_state") == "failed"
            and s.get("tasks_started") == 0
            for s in samples
        ):
            raise ValueError("model preparation failure not projected")
    elif mode == "planner-failure":
        if not any(e["event"] == "observe_exit" for e in trace) or not any(
            e["event"] == "planner_error" for e in trace
        ):
            raise ValueError("real observation/planner-failure trace missing")
        if not any(s.get("guidance_phase") == "failed" for s in samples):
            raise ValueError("planner failure not projected")
    else:
        if not any(e["event"] == "observe_exit" for e in trace) or not any(
            e["event"] == "planner_exit" for e in trace
        ):
            raise ValueError("real observation/planning trace missing")
        if not any(
            s.get("route")
            and s.get("region")
            and s.get("recommendation_frame", 12001) < intervals[0]["motion_start_frame"]
            for s in samples
        ):
            raise ValueError("no real initial S1 route and region before striking")
        if not any(d["has_dash"] and d["has_region"] and d["has_halo"] for d in dom):
            raise ValueError("actual browser never displayed route/halo/region")
        hiding_seconds_upper_bounds = assess_motion_display(
            dom, motion, origin_seconds=origin_seconds, arrival_times=arrival_times
        )
        if len({e["frame_id"] for e in trace if e["event"] == "observe_exit"}) < len(intervals) + 1:
            raise ValueError("fresh post-shot observation not established")
    return dict(
        control=mode,
        positive_recommendation=mode == "recommendation",
        trace_events=len(trace),
        physical_translation_windows=motion,
        physical_wall_windows=wall_windows,
        motion_hiding_seconds_upper_bounds=hiding_seconds_upper_bounds,
        worker_reuse=None if mode == "model-failure" else assess_worker_reuse(samples, trace),
    )


def assess_motion_display(dom, motion, *, origin_seconds=None, arrival_times=None):
    delays = []
    for start, end in motion_wall_windows(
        motion, origin_seconds=origin_seconds, arrival_times=arrival_times
    ):
        observations = [
            d for d in dom if start <= d.get("monotonic", -1) <= d.get("observed_at", -1) <= end
        ]
        hidden = [
            d
            for d in observations
            if "等待停稳" in d.get("phase", "")
            and not d["has_dash"]
            and not d["has_region"]
            and not d.get("has_halo", False)
        ]
        if not hidden or hidden[0]["observed_at"] - start > 0.5:
            raise ValueError("actual browser motion hiding missing or exceeds 500ms evidence limit")
        if any(
            d["has_dash"] or d["has_region"] or d.get("has_halo", False)
            for d in observations
            if d["monotonic"] >= hidden[0]["monotonic"]
        ):
            raise ValueError("actual browser displayed stale advice during motion")
        delays.append(hidden[0]["observed_at"] - start)
    return delays


def assess_browser_errors(value):
    if not isinstance(value, dict) or not isinstance(value.get("errors"), list) or value["errors"]:
        raise ValueError("actual browser has errors or invalid error evidence")


def _same_points(actual, expected):
    try:
        points = [[float(n) for n in p.split(",")] for p in actual.split()]
        return len(points) == len(expected) and all(
            len(a) == len(b) == 2
            and all(math.isfinite(n) for n in (*a, *b))
            and math.dist(a, b) < 0.001
            for a, b in zip(points, expected, strict=True)
        )
    except (AttributeError, ValueError, TypeError):
        return False


def assess_display_flow(report):
    dom = report["dom"]
    recommendations = report["recommendations"]
    if not recommendations or not report["feedbacks"]:
        raise ValueError("display route and feedback evidence missing")
    for frame_id, value in recommendations.items():
        overlay = value["overlay"]
        if not any(
            len(row.get("route_lines", [])) == 2
            and row.get("observed_frame_id") == int(frame_id)
            and row.get("has_halo")
            and _same_points(row["route_lines"][0], overlay["cue_path"])
            and _same_points(row["route_lines"][1], overlay["object_path"])
            for row in dom
        ):
            raise ValueError("display did not show both routes to the selected pocket")
    feedback_results = []
    labels = dict(
        inside="已进入建议区",
        outside="未进入建议区",
        uncertain="区域边界附近",
        unknown="落点未可靠确认",
    )
    for value in report["feedbacks"].values():
        review, overlay = value["review"], value["overlay"]
        label = labels[review["region_result"]]
        indices = [
            i
            for i, row in enumerate(dom)
            if row.get("feedback_motion_frame_id") == review["motion_frame_id"]
            and row.get("observed_frame_id") == review["observed_frame_id"]
        ]
        if (
            len(indices) < 2
            or indices[0] == 0
            or indices[-1] == len(dom) - 1
            or indices != list(range(indices[0], indices[-1] + 1))
            or any(
                label not in row.get("phase", "")
                or row.get("route_lines")
                or row.get("has_halo")
                or not _same_points(row.get("region_points"), overlay["region"])
                for row in (dom[i] for i in indices)
            )
        ):
            raise ValueError("display feedback is missing, stale or uses a different region")
        first, last = dom[indices[0]], dom[indices[-1]]
        previous, following = dom[indices[0] - 1], dom[indices[-1] + 1]
        # Read-only browser queries bracket, rather than precisely date, a DOM
        # transition. Report the measured bounds; never invent a display clock.
        lower = max(0, last["monotonic"] - first["observed_at"])
        upper = following["observed_at"] - previous["monotonic"]
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and FEEDBACK_VISIBLE_SECONDS[0] <= lower <= upper <= FEEDBACK_VISIBLE_SECONDS[1]
            and upper - lower <= FEEDBACK_MAX_INTERVAL_WIDTH
        ):
            raise ValueError("display feedback did not establish an approximately one-second hold")
        expected_next = recommendations.get(str(review["observed_frame_id"]))
        lines = following.get("route_lines", [])
        if following.get("observed_frame_id") != review["observed_frame_id"] or not (
            (
                expected_next is not None
                and len(lines) == 2
                and _same_points(lines[0], expected_next["overlay"]["cue_path"])
                and _same_points(lines[1], expected_next["overlay"]["object_path"])
            )
            or (
                expected_next is None
                and not lines
                and not following.get("has_halo")
                and "暂无可靠方案" in following.get("phase", "")
                and not following.get("has_region")
            )
        ):
            raise ValueError("display feedback handoff does not match this post-shot snapshot")
        feedback_results.append(
            dict(
                observed_frame_id=review["observed_frame_id"],
                region_result=review["region_result"],
                visible_seconds_bounds=[lower, upper],
                dom_observations=len(indices),
            )
        )
    return dict(
        selected_pocket_routes=len(recommendations),
        feedback=feedback_results,
        feedback_window_policy=dict(
            visible_seconds=FEEDBACK_VISIBLE_SECONDS,
            max_interval_width_seconds=FEEDBACK_MAX_INTERVAL_WIDTH,
        ),
    )


def assess_postgame(manifest, review, dom, oracle):
    first = review["clips"][0]
    value = first.get("guidance_review")
    if not value:
        raise ValueError("original shot review is missing")
    interval = manifest["intervals"][0]
    next_motion = (
        manifest["intervals"][1]["motion_start_frame"]
        if len(manifest["intervals"]) > 1
        else manifest["frame_count"]
    )
    rec, actual = value["recommendation"], value["actual_cue"]
    region = rec.get("region")
    if not (
        region
        and actual
        and rec["source_frame_id"] < interval["motion_start_frame"]
        and interval["motion_start_frame"]
        <= value["motion_frame_id"]
        <= interval["final_static_frame"]
        and interval["final_static_frame"] < value["observed_frame_id"] < next_motion
        and value["region_result"] == manifest["case"]
    ):
        raise ValueError("prior advice and actual result not bound to the original shot")
    truth = oracle["shots"][0]["final"]["0"]
    if math.hypot(actual["x_mm"] - truth[0], actual["y_mm"] - truth[1]) > 20:
        raise ValueError("actual observed cue does not match external physical truth")
    desired = f"击球前建议区中心：({region['center']['x_mm']},{region['center']['y_mm']})"
    measured = f"实际白球终点：({actual['x_mm']},{actual['y_mm']}) mm"
    verdict = "已进入建议区" if manifest["case"] == "inside" else "未进入建议区，已按新局面重算"
    if not any(
        row["shot_id"] == first["shot_id"]
        and all(text in row["text"] for text in (desired, measured, verdict))
        for observation in dom
        for row in observation.get("shot_rows", [])
    ):
        raise ValueError("actual postgame DOM does not distinguish advice and measurement")
    return dict(
        shot_id=first["shot_id"],
        region_result=value["region_result"],
        observed_frame_id=value["observed_frame_id"],
    )


def environment_identity():
    packages = {}
    names = set(
        (
            "numpy",
            "opencv-python",
            "torch",
            "torchvision",
            "rfdetr",
            "transformers",
            "onnxruntime",
            "onnxruntime-gpu",
            "pooltool-billiards",
            "panda3d",
        )
    )
    physics = json.loads(
        (Path(__file__).with_name("assets") / "s1-physics-windows.yaml").read_bytes()
    )
    names.update(wheel["name"] for wheel in physics["wheels"])
    for name in sorted(names):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        python=sys.version,
        platform=platform.platform(),
        executable=sources.pin(Path(sys.executable).resolve()),
        packages=packages,
    )


def assess_screen_duration(inventory, started_at, stopped_at):
    wall_seconds = stopped_at - started_at
    if wall_seconds <= 0 or abs(inventory["seconds"] - wall_seconds) > max(2, wall_seconds * 0.02):
        raise ValueError("screen video duration does not cover the real capture wall clock")


def browser_state(browser):
    started = time.perf_counter()
    result = browser.evaluate(DOM_STATE)
    if not isinstance(result, dict) or not isinstance(result.get("result"), dict):
        raise ValueError("browser did not return real DOM evidence")
    return dict(result["result"], monotonic=started, observed_at=time.perf_counter())


def play_visible_clip(browser, button):
    browser.command("find", "role", "button", "click", "--name", button)
    observations = []
    for _ in range(25):
        state = browser_state(browser)
        observations.append(state)
        if state["video_error"]:
            raise ValueError("actual browser playback error")
        if state["video_duration"]:
            break
        time.sleep(0.2)
    else:
        raise ValueError("actual browser video metadata unavailable")
    # Row selection intentionally does not autoplay. Use native video keyboard
    # controls, exactly as a user does; never inject play() into the page.
    browser.command("focus", "#postgame-player")
    browser.command("press", "Space")
    deadline = time.monotonic() + state["video_duration"] + 8
    while time.monotonic() < deadline:
        state = browser_state(browser)
        observations.append(state)
        if state["video_error"]:
            raise ValueError("actual browser playback error")
        if state["video_ended"] and state["video_time"] > 0:
            return dict(button=button, observations=observations)
        time.sleep(0.2)
    raise ValueError("actual browser video did not play to the end")


def screen_inventory(video):
    import cv2

    from billiards_ai.lab.mainline1_recording_media import _verify_mp4

    capture = cv2.VideoCapture(str(video))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    result = _verify_mp4(video, expected_frames=frames, width=1920, height=1080, fps=SCREEN_FPS)
    if frames < SCREEN_FPS or abs(fps - SCREEN_FPS) > 0.5:
        raise ValueError("screen capture cadence/duration invalid")
    return dict(**result, frames=frames, fps=fps, seconds=frames / fps)


def export_original(frozen, output, frame_count):
    # Compatibility only. Encoding now has one implementation and is invoked
    # after the online Job has cleaned up, under its own owned media stage.
    from scripts.development.s1_media_acceptance import export_original as original

    return original(frozen, output, frame_count)


def run(args):
    from billiards_ai.lab.field_assistance import server as service
    from billiards_ai.lab.mainline1_recording_media import verify_recording_playback_artifact
    from billiards_ai.lab.mainline1_scenario_capture import ScenarioVideoCaptureBackend
    from scripts.algorithm import run_recording_first_fullgame as existing
    from scripts.algorithm import serve_mainline1_blackbox_field as blackbox

    source_before = source_identity(args.allow_dirty_development)
    manifest, identity = input_identity(args.input_root)
    if args.config.stat().st_size > 1024**2:
        raise ValueError("private config exceeds budget")
    config = json.loads(args.config.read_bytes())
    output = sources.plain(args.output_root)
    output.mkdir()
    for name in ("ring", "jobs", "artifacts"):
        (output / name).mkdir()
    report = dict(
        contract="s1-online-screen-acceptance/v1",
        mode=args.mode,
        case_pass=False,
        online_pass=False,
        media_complete=False,
        full_goal_complete=False,
        field_accuracy="OPEN",
        owned_run_id=getattr(args, "owned_run_id", None),
        source=source_before,
        input=identity,
        input_manifest=manifest,
        evidence_status="DEVELOPMENT_ONLY"
        if source_before["kind"] == "development"
        else "CANDIDATE",
        samples=[],
        dom=[],
        media=[],
        recommendations={},
        feedbacks={},
        cleanup_confirmed=False,
        environment=environment_identity(),
        evidence_clock=dict(name="perf_counter", **vars(time.get_clock_info("perf_counter"))),
    )
    resource_spec = None
    server = thread = prepared = browser = None
    at_end, release_tail = threading.Event(), threading.Event()
    trace_path = output / "guidance-calls.jsonl"
    diagnostics = ExitStack()
    stage_root = output / "online-stages"
    stage_root.mkdir()
    stages = diagnostics.enter_context(ProbeDiagnostics(stage_root, uuid4().hex))
    checkpoint = partial(screen_stage, report, output, stages)
    arrivals = FrameArrivalClock(manifest["frame_count"])

    class HeldTail(ScenarioVideoCaptureBackend):
        def __init__(self, config, **kwargs):
            super().__init__(
                replace(config, startup_preparation_seconds=FILE_PREPARATION_SECONDS), **kwargs
            )

        def frames(self, stop):
            for frame in super().frames(stop):
                arrivals.record(frame.capture_frame_id)
                yield frame
                if frame.capture_frame_id == manifest["frame_count"] - 1:
                    report["capture_completed_at"] = time.perf_counter()
                    at_end.set()
                    while not self._stopped(stop) and not release_tail.wait(0.02):
                        pass

    original_factory = blackbox._snapshot_factory

    def traced_factory(options):
        factory = original_factory(options)
        return replace(
            factory,
            observer_factory=TracedFactory(
                factory.observer_factory, trace_path, trace_model_stages=True
            ),
            planner=TracedPlanner(factory.planner, trace_path, args.mode == "planner-failure"),
        )

    try:
        diagnostics.enter_context(trace_storage_preflight(output / "storage-preflight"))
        diagnostics.enter_context(trace_scenario_validation(output / "scenario-validation"))
        parsed = blackbox._parser().parse_args(runtime_options(args.input_root, output, config))
        blackbox._validate_paths(parsed)
        from billiards_ai.lab.field_assistance.runtime_resources import (
            inspect_managed_runtime_resources,
            verify_managed_runtime_resources,
        )

        if args.mode == "model-failure":
            missing = {name: str(sources.plain(config[name])) for name in MODEL_PATHS}
            if any(Path(path).exists() for path in missing.values()):
                raise ValueError("missing-model control requires actually absent assets")
            report["missing_model_assets"] = missing
        else:
            resource_spec = inspect_managed_runtime_resources(parsed, rule_profile_id=None)
            report["resources"] = dict(
                spec_sha256=resource_spec.spec_sha256,
                device=parsed.device,
                batch_size=parsed.batch_size,
                files=[
                    dict(role=f.role, bytes=f.byte_size, sha256=f.sha256)
                    for f in resource_spec.files
                ],
            )
        epoch = uuid4().hex
        with (
            patch.object(blackbox, "ScenarioVideoCaptureBackend", HeldTail),
            patch.object(blackbox, "_snapshot_factory", traced_factory),
        ):
            calibration, factory, prepared = blackbox.prepare_runtime(
                parsed, initial_source_epoch_id=epoch
            )
            token = secrets.token_urlsafe(32)
            server = service.build_live_field_assistance_server(
                artifact_path=output / "artifacts",
                state_db_path=output / "state.sqlite3",
                host="127.0.0.1",
                port=0,
                control_token=token,
                source_epoch_factory=lambda: epoch,
                calibration_id=calibration,
                runtime_factory=factory,
                fresh_session_on_start=True,
                session_policy="recording-v1",
            )
        state = server.field_state
        state.managed_blackbox = True
        state.monotonic = lambda: (
            0.0
        )  # Existing R4 review-window control, never capture/model time.
        report["session_id"] = state.session_id
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        browser = BrowserCapture(
            args.browser_cli, args.browser_exe, args.ffmpeg, output / "capture"
        )
        with checkpoint("screen_capture"):
            browser.start(f"http://127.0.0.1:{server.server_port}/screen")
            report["initial_snapshot"] = browser.snapshot()
            dom_observer = DOMObserver(browser, DOM_STATE)
            dom_observer.start()
            existing._setup_players(state)
            state.wait_until_capture_ready(timeout_seconds=30)
            hub = state.projection_source.runtime.capture_hub
            report["file_startup"] = dict(
                preparation_budget_seconds=hub.startup_preparation_seconds,
                first_frame_budget_seconds=hub.startup_timeout_seconds,
                started_at_ns=hub.startup_started_at_ns,
                preparation_completed_at_ns=hub.preparation_completed_at_ns,
                first_frame_at_ns=hub.first_frame_at_ns,
            )
            started = time.monotonic()
            report["capture_budget_seconds"] = capture_work_seconds(
                manifest["frame_count"], getattr(args, "capture_seconds", None)
            )
            deadline = time.perf_counter() + report["capture_budget_seconds"]
            report["capture_deadline"] = deadline
            print(json.dumps(dict(stage="capturing", session_id=state.session_id)), flush=True)
            while not at_end.wait(0.1):
                if time.perf_counter() > deadline and report.get("capture_completed_at") is None:
                    raise TimeoutError("fixed input capture budget exhausted")
                payload = existing._request(server, "/api/screen", timeout_seconds=3)
                frame = payload["frame"]
                s1 = frame.get("snapshot_guidance") or {}
                rec = s1.get("recommendation") or {}
                if rec:
                    report["recommendations"][str(rec["source_frame_id"])] = dict(
                        recommendation=rec, overlay=s1.get("overlay")
                    )
                feedback = s1.get("feedback")
                if feedback is not None:
                    report["feedbacks"][str(feedback["motion_frame_id"])] = dict(
                        review=feedback, overlay=s1.get("overlay")
                    )
                report["samples"].append(
                    dict(
                        seconds=time.monotonic() - started,
                        frame_id=frame.get("media_frame_id"),
                        guidance_phase=s1.get("phase"),
                        model_state=s1.get("model_state"),
                        worker_pid=s1.get("worker_pid"),
                        model_loads=s1.get("model_loads"),
                        worker_restarts=s1.get("worker_restarts"),
                        resource_preparation_seconds=s1.get("resource_preparation_seconds"),
                        tasks_started=s1.get("tasks_started"),
                        tasks_completed=s1.get("tasks_completed"),
                        tasks_cancelled=s1.get("tasks_cancelled"),
                        tasks_discarded=s1.get("tasks_discarded"),
                        active_pid=s1.get("active_pid"),
                        route=bool(rec.get("route")),
                        region=bool(rec.get("region")),
                        recommendation_frame=rec.get("source_frame_id"),
                    )
                )
                if len(report["samples"]) > 4096:
                    raise ValueError("timeline budget exceeded")
                report["samples"][-1]["error"] = s1.get("error")
                record_guidance_sample(
                    output / "local-guidance-samples.jsonl", report["samples"][-1], mode=args.mode
                )
            validate_capture_completion(deadline, report.get("capture_completed_at"))
            arrival_times = arrivals.complete()
            atomic_json(output / "frame-arrivals.private.json", arrival_times)
            report["arrival_clock"] = sources.pin(output / "frame-arrivals.private.json")
            observed = dom_observer.stop()
            atomic_json(output / "dom-observer.private.json", observed)
            report["dom_observer_pin"] = sources.pin(output / "dom-observer.private.json")
            report["dom_clock"] = observed["calibration"]
            report["dom"] = observed["dom"]
            for value in report["dom"]:
                record_dom_observation(output / "local-dom-observations.jsonl", value)
        runtime = state.projection_source.runtime
        report["guidance_before_end"] = runtime.snapshot_guidance.projection()
        with checkpoint("screen_end"):
            ended = existing._request(
                server,
                "/api/control",
                token=token,
                payload=dict(command="end_session", session_id=state.session_id),
            )
            receipt = ended["recording_end"]
            report["capture_end"] = receipt
            if receipt["cutoff"]["frame_id"] != manifest["frame_count"] - 1:
                raise ValueError("raw cutoff mismatch")
        with checkpoint("screen_background"):
            frozen, review = existing._wait_for_background_completion(
                server, state, receipt, deadline=time.monotonic() + 180, report=report
            )
            if review["available_count"] != len(manifest["intervals"]):
                raise ValueError("recorded shot count mismatch")
            report["review"] = review
            report["frozen"] = frozen
        with checkpoint("screen_playback"):
            for _ in range(30):
                visible = browser_state(browser)
                report["dom"].append(visible)
                record_dom_observation(output / "local-dom-observations.jsonl", visible)
                if (
                    not visible["postgame_hidden"]
                    and len(visible["play_buttons"]) == review["available_count"]
                ):
                    break
                threading.Event().wait(0.2)
            else:
                raise ValueError("actual browser did not present the postgame list")
            report["postgame_snapshot"] = browser.snapshot()
            for button in visible["play_buttons"]:
                report.setdefault("browser_playback", []).append(play_visible_clip(browser, button))
            browser.command("screenshot", str(output / "actual-postgame.png"))
            report["browser_errors"] = browser.command("errors")
            assess_browser_errors(report["browser_errors"])
        with checkpoint("screen_close"):
            browser.close()
        with checkpoint("screen_verify"):
            report["capture_tools"] = browser.identities
            report["screen"] = screen_inventory(browser.video)
            assess_screen_duration(report["screen"], browser.started_at, browser.stop_started_at)
            report["screen"]["wall_seconds"] = browser.stop_started_at - browser.started_at
        with checkpoint("screen_clips"):
            for job in state.media_job_store.list_session_media_jobs(state.session_id):
                snapshot = state.media_job_store.snapshot(job.job_id)
                if snapshot.artifact_id is None:
                    raise ValueError("original clip is not generated")
                artifact = verify_recording_playback_artifact(
                    output / "artifacts/postgame" / state.session_id / snapshot.artifact_id, job
                )
                clip = next(c for c in review["clips"] if c["shot_id"] == job.shot_id)
                digest, size = existing._media_hash(server, clip["media_url"])
                if (digest, size) != (artifact.content_sha256, artifact.byte_size):
                    raise ValueError("HTTP original changed")
                destination = output / (job.shot_id + ".mp4")
                shutil.copyfile(artifact.media_path, destination)
                if sources.pin(destination) != dict(bytes=size, sha256=digest):
                    raise ValueError("clip copy changed")
                report["media"].append(
                    dict(
                        shot_id=job.shot_id, frames=artifact.frame_count, bytes=size, sha256=digest
                    )
                )
        with checkpoint("screen_assess"):
            if (
                state.media_job_store.list_session_jobs(state.session_id)
                or state.record_store.list_saved_media_outbox()
            ):
                raise ValueError("unexpected semantic/save work")
            oracle = json.loads((args.input_root / "oracle.not-product-input.json").read_bytes())
            trace = [json.loads(line) for line in trace_path.read_bytes().splitlines()]
            report["guidance_assessment"] = assess_guidance(
                args.mode,
                report["samples"],
                report["dom"],
                trace,
                oracle,
                manifest["intervals"],
                arrival_times=arrival_times,
            )
            if args.mode == "recommendation":
                report["display_assessment"] = assess_display_flow(report)
                report["postgame_assessment"] = assess_postgame(
                    manifest, review, report["dom"], oracle
                )
        with checkpoint("screen_sources_after"):
            if resource_spec is not None:
                verify_managed_runtime_resources(resource_spec, purpose="analysis")
                report["resources_unchanged"] = True
            elif any(Path(path).exists() for path in report["missing_model_assets"].values()):
                raise ValueError("missing-model control assets changed")
            if environment_identity() != report["environment"]:
                raise ValueError("runtime environment changed")
            if input_identity(args.input_root)[1] != identity:
                raise ValueError("input changed during case")
        report["online_pass"] = True
    except Exception as exc:
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        failures = []
        if browser is not None and not browser.closed:
            try:
                browser.close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        if server is not None:
            try:
                cleanup_step(checkpoint, "screen_server_shutdown", server.shutdown)
            except Exception as exc:
                failures.append(type(exc).__name__)
            try:
                cleanup_step(checkpoint, "screen_server_close", server.server_close)
            except Exception as exc:
                failures.append(type(exc).__name__)
        if prepared is not None:
            try:
                cleanup_step(checkpoint, "screen_prepared_cancel", prepared.cancel)
            except Exception as exc:
                failures.append(type(exc).__name__)
        release_tail.set()
        if thread is not None:
            thread.join(5)
        if thread is not None and thread.is_alive():
            failures.append("ServerThreadStillAlive")
        diagnostics.close()
        try:
            if source_identity(args.allow_dirty_development) != source_before:
                failures.append("SourceChanged")
        except Exception as exc:
            failures.append("SourceVerification:" + type(exc).__name__)
        report["cleanup_errors"] = failures
        report["cleanup_confirmed"] = not failures
        report["online_pass"] = report["online_pass"] and not failures
        atomic_json(output / "local-report.json", report)
    print(
        json.dumps(
            dict(
                case_pass=False,
                online_pass=report["online_pass"],
                mode=args.mode,
                full_goal_complete=False,
                output=str(output),
            )
        ),
        flush=True,
    )
    return 0 if report["online_pass"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "output-root", "config", "browser-cli", "browser-exe", "ffmpeg"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("recommendation", "planner-failure", "model-failure"), required=True
    )
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--owned-run-id")
    parser.add_argument("--capture-seconds", type=int, choices=(240,))
    args = parser.parse_args()
    await_owned_release(args.owned_run_id)
    return run(args)


def await_owned_release(run_id, *, platform=os.name, stream=None):
    if run_id is None:
        if platform == "nt":
            raise ValueError("Windows screen case requires an owned release")
        return
    if not re.fullmatch("[0-9a-f]{32}", run_id) or (stream or sys.stdin).readline(80) != (
        "S1-SCREEN-RELEASE " + run_id + "\n"
    ):
        raise ValueError("exact owned release required before screen work")


if __name__ == "__main__":
    raise SystemExit(main())
