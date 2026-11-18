"""S1 real-model component readiness; not full functional acceptance.

Read one frame of an existing fixed FILE source using the real snapshot adapter
and planner. No video generation, substitute observer, recording or screen claim.
The Windows controller reuses the pinned Job owner: one 60s run + 10s cleanup.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.development import blackbox_report as reports  # noqa: E402
from scripts.development import blackbox_source as sources  # noqa: E402
from scripts.development.s1_probe_diagnostics import (  # noqa: E402
    CallStageProxy,
    ProbeDiagnostics,
    read_diagnostic_summary,
)

PATHS = (
    "capture_setup",
    "calibration_root",
    "config",
    "detector_training_run",
    "identity_training_run",
    "dinov2_assets",
    "artifact_root",
    "jobs_root",
    "ring_root",
    "scenario_video",
)


def phase(trace, name):
    return nullcontext() if trace is None else trace.stage(name)


def probe_mode(value):
    if value not in ("cold", "prepared"):
        raise ValueError("unknown component probe mode")
    return value


def calculate_component(
    calculate, factory, frame, calibration, group, epoch, planner, *, mode, resource_preparer=None
):
    """Measure a prepared component separately; retain the original cold total."""
    from billiards_ai.lab.field_assistance.snapshot_guidance import (
        SnapshotConfig,
        _prepare_resources,
    )

    probe_mode(mode)

    class ResultSink:
        value = None

        def send(self, value):
            self.value = value

        def close(self):
            pass

    sink = ResultSink()
    started = time.monotonic()
    preparation = None
    timings = {}

    if mode == "prepared":
        observer, timings = _prepare_resources(factory, resource_preparer)
        preparation = time.monotonic() - started
        make = lambda: observer  # noqa: E731 - matches the reusable worker's thin factory
    else:
        make = lambda: _prepare_resources(factory, resource_preparer)[0]  # noqa: E731
    task_started = time.monotonic()
    calculate(sink, make, frame, calibration, group, 0, epoch, planner)
    finished = time.monotonic()
    total = finished - started
    task = finished - task_started if mode == "prepared" else None
    budget = SnapshotConfig().task_seconds
    return sink.value, {
        "calculation_seconds": total,
        "within_calculation_budget": total <= budget,
        "model_preparation_seconds": timings.get("model_preparation_seconds"),
        "resource_preparation_seconds": preparation,
        "snapshot_request_seconds": task,
        "within_snapshot_budget": task is not None and task <= budget,
    }


def component_passes(summary):
    if summary.get("model_component_completed") is not True:
        return False
    if summary.get("probe_mode", "cold") == "cold":
        return summary.get("within_calculation_budget") is True
    return (
        summary.get("probe_mode") == "prepared"
        and summary.get("within_snapshot_budget") is True
        and summary.get("route_available") is True
        and summary.get("region_available") is True
    )


def inspect_and_probe(config, *, trace=None):
    mode = probe_mode(config.get("probe_mode", "cold"))
    with phase(trace, "imports"):
        from billiards_ai.lab.field_assistance.runtime_resources import (
            inspect_managed_runtime_resources,
            verify_managed_runtime_resources,
        )
        from billiards_ai.lab.field_assistance.snapshot_guidance import _calculate
        from billiards_ai.lab.field_assistance.snapshot_models import ModelSnapshotObserverFactory
        from billiards_ai.lab.mainline1_runtime_support import (
            CaptureSourceKind,
            RealtimeCaptureConfig,
            TableCalibration,
            load_capture_setup,
        )
        from billiards_ai.lab.mainline1_scenario_capture import (
            ScenarioCaptureConfig,
            ScenarioVideoCaptureBackend,
        )
        from billiards_ai.lab.planning.s1 import recommend_s1
        from billiards_ai.lab.planning.s1_validation import prepare_s1_physics

    args = SimpleNamespace(
        **{name: Path(config[name]) for name in PATHS},
        device=config.get("device", "cuda"),
        batch_size=config.get("batch_size", 8),
        fps=60,
    )
    with phase(trace, "resources_before"):
        spec = inspect_managed_runtime_resources(args, rule_profile_id=None)
    with phase(trace, "capture_setup"):
        setup = load_capture_setup(args.capture_setup)
        if setup.source_kind is not CaptureSourceKind.FILE or (
            setup.geometry.width,
            setup.geometry.height,
            setup.requested_fps,
        ) != (1920, 1080, 60):
            raise ValueError("probe requires the existing FILE 1920x1080/60 contract")
        group = config.get("group", "open")
        if group not in {"open", "solid", "stripe"}:
            raise ValueError("invalid match group")
    with phase(trace, "calibration"):
        calibration = TableCalibration.model_validate_json(
            (Path(spec.calibration_artifact_path) / "calibration.json").read_bytes()
        )
    epoch = uuid4().hex
    with phase(trace, "first_frame"):
        source = ScenarioVideoCaptureBackend(
            ScenarioCaptureConfig(args.scenario_video, RealtimeCaptureConfig()),
            source_epoch_id=epoch,
        )
        try:
            frame = next(source.frames(threading.Event()))
        finally:
            source.cancel()

    def make_observer():
        with phase(trace, "model_factory"):
            observer = ModelSnapshotObserverFactory(
                args.detector_training_run,
                args.identity_training_run,
                args.dinov2_assets,
                device=args.device,
                batch_size=args.batch_size,
            )()
        if trace is None:
            return observer
        # Only thin probe adapters; original model instances/results are unchanged.
        return replace(
            observer,
            vision=CallStageProxy(observer.vision, trace, "infer", "detector_inference"),
            identity=CallStageProxy(observer.identity, trace, "classify", "identity_inference"),
        )

    def plan(*args, **kwargs):
        with phase(trace, "planning"):
            return recommend_s1(*args, **kwargs)

    def prepare_numerics():
        with phase(trace, "physics_preparation"):
            prepare_s1_physics()

    value, timing = calculate_component(
        _calculate,
        make_observer,
        frame,
        calibration,
        group,
        epoch,
        plan,
        mode=mode,
        resource_preparer=prepare_numerics,
    )
    with phase(trace, "resources_after"):
        verify_managed_runtime_resources(spec, purpose="analysis")
    with phase(trace, "video_after"):
        if sources.pin(args.scenario_video)["sha256"] != source.identity.durable_fingerprint:
            raise ValueError("source changed during probe")
    if not isinstance(value, dict):
        raise ValueError("snapshot component did not return a result")
    return {
        "resource_spec_sha256": spec.spec_sha256,
        "video_sha256": source.identity.durable_fingerprint,
        "source_frame_id": frame.capture_frame_id,
        **timing,
        "model_component_completed": "error" not in value,
        "route_available": bool(value.get("route")),
        "region_available": bool(value.get("region")),
        "observation_count": len(value.get("observed_balls", [])),
        "error_type": str(value["error"]).split(":", 1)[0]
        if "error" in value and str(value["error"]).split(":", 1)[0].isidentifier()
        else None,
        "result": value,
    }


def child(root):
    context = json.loads((root / "context.json").read_bytes())
    expected = ("S1-PROBE-RELEASE " + context["run_id"] + "\n").encode()
    if sys.stdin.buffer.readline(256) != expected:
        raise ValueError("missing owned-process release")
    # No new work before the existing owner releases this exact child.
    with ProbeDiagnostics(
        root, context["run_id"], stack_at=context.get("diagnostic_stack_at")
    ) as trace:
        with trace.stage("source_before"):
            if sources.verify(ROOT) != context["source_lock"]:
                raise ValueError("probe source identity changed before worker release")
        versions = {}
        with trace.stage("dependency_metadata"):
            for name in (
                "torch",
                "rfdetr",
                "transformers",
                "onnxruntime",
                "pooltool-billiards",
                "opencv-python",
                "numpy",
            ):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "NOT_INSTALLED"
        try:
            result = inspect_and_probe(context["inputs"], trace=trace)
        except Exception as exc:
            result = {"model_component_completed": False, "error_type": type(exc).__name__}
        with trace.stage("source_after"):
            if sources.verify(ROOT) != context["source_lock"]:
                raise ValueError("probe source identity changed during calculation")
        result["dependency_versions"] = versions
        result["python_version"] = sys.version.split()[0]
        # Full observations/error strings remain local; only summary.json is exportable.
        with trace.stage("result_write"):
            reports.atomic_json(root / "local-result.json", result)
        return 0 if result.get("model_component_completed") else 2


def control(args):
    if os.name != "nt":
        raise ValueError("controller requires Windows and the existing pinned Job owner")
    from scripts.development.blackbox_validation import load_ownership

    lock = sources.verify(ROOT)
    config_path = sources.plain(args.config)
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("probe configuration too large")
    config = json.loads(config_path.read_bytes())
    # Never persist the caller's entire namespace/configuration.
    inputs = {name: str(sources.plain(config[name])) for name in PATHS}
    mode = probe_mode(config.get("probe_mode", "cold"))
    inputs.update(group=config.get("group", "open"), device="cuda", batch_size=8, probe_mode=mode)
    executable = sources.plain(args.worker_python)
    executable_pin = sources.pin(executable)
    owner = load_ownership(sources.plain(args.ownership))
    root = sources.plain(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    reports.atomic_json(
        root / "context.json",
        {
            "run_id": run_id,
            "inputs": inputs,
            "source_lock": lock,
            # Earlier than the owner's 60s deadline (which starts after release).
            "diagnostic_stack_at": time.monotonic() + 50,
        },
    )
    (root / "tmp").mkdir()
    logs = [root / "local-stdout.txt", root / "local-stderr.txt"]
    command = [str(executable), "-I", "-B", str(Path(__file__).resolve()), "--child", str(root)]
    from scripts.algorithm.recording_windows_entry import child_environment

    # Pooltool resolves Path.home() during import. Reuse the existing owned
    # Windows profile boundary rather than inheriting a user's mutable config.
    environment = child_environment(root)
    for name in ("numba", "mpl"):
        (root / "tmp" / name).mkdir()
    environment.update(
        TEMP=str(root / "tmp"),
        TMP=str(root / "tmp"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        NUMBA_CACHE_DIR=str(root / "tmp" / "numba"),
        MPLCONFIGDIR=str(root / "tmp" / "mpl"),
    )

    def bounded_sleep(seconds):
        time.sleep(seconds)
        if any(
            path.exists() and path.stat().st_size > 4 * 1024**2
            for path in (*logs, root / "local-stack.txt")
        ):
            raise RuntimeError("probe log budget exceeded")

    with logs[0].open("xb") as stdout, logs[1].open("xb") as stderr:
        owned = owner.run_owned(
            owner.WindowsAPI(),
            lambda: subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                cwd=root,
                env=environment,
                close_fds=True,
            ),
            lambda stage, state: reports.atomic_json(
                root / "ownership.json", {"stage": stage, **state}
            ),
            {"run_id": run_id, "purpose": "S1_MODEL_COMPONENT_ONLY"},
            ("S1-PROBE-RELEASE " + run_id + "\n").encode(),
            deadline=60,
            cleanup_seconds=10,
            sleep=bounded_sleep,
        )
    result_path = root / "local-result.json"
    try:
        result = (
            json.loads(result_path.read_bytes())
            if result_path.is_file() and result_path.stat().st_size <= 1024**2
            else {}
        )
        if not isinstance(result, dict):
            result = {}
    except (ValueError, OSError):
        result = {}
    try:
        source_unchanged = sources.verify(ROOT) == lock
    except (ValueError, OSError):
        source_unchanged = False
    summary = {
        "contract": "s1-model-component-probe/v1",
        "run_id": run_id,
        "source_commit": lock["commit"],
        "probe_mode": mode,
        "worker_python": executable_pin,
        "owned_cleanup": owned.get("cleanup_confirmed") is True,
        "worker_exit_code": owned.get("worker_exit_code"),
        "ownership_status": owned.get("status"),
        "ownership_terminal_record_failed": bool(owned.get("terminal_record_failed")),
        "ownership_job_close_error": bool(owned.get("job_close_error")),
        "source_unchanged": source_unchanged,
        "full_functional_acceptance": "NOT_RUN",
        "field_accuracy": "OPEN",
        "model_component_completed": False,
        "within_calculation_budget": False,
        "within_snapshot_budget": False,
        "diagnostic": read_diagnostic_summary(root, run_id),
        **{
            key: result[key]
            for key in (
                "resource_spec_sha256",
                "video_sha256",
                "source_frame_id",
                "calculation_seconds",
                "within_calculation_budget",
                "model_preparation_seconds",
                "resource_preparation_seconds",
                "snapshot_request_seconds",
                "within_snapshot_budget",
                "model_component_completed",
                "route_available",
                "region_available",
                "observation_count",
                "error_type",
                "dependency_versions",
                "python_version",
            )
            if key in result
        },
    }
    reports.atomic_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return (
        0
        if (
            summary["owned_cleanup"]
            and summary["worker_exit_code"] == 0
            and summary["ownership_status"] == "PASS"
            and not summary["ownership_terminal_record_failed"]
            and not summary["ownership_job_close_error"]
            and summary["source_unchanged"]
            and component_passes(summary)
            and summary["diagnostic"].get("status") == "AVAILABLE"
            and summary["diagnostic"].get("last_stage") == "result_write"
            and summary["diagnostic"].get("last_event") == "exit"
        )
        else 2
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--ownership", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args.child)
    if not all((args.config, args.worker_python, args.ownership, args.output_root)):
        parser.error("config, worker-python, ownership and a fresh output-root are required")
    return control(args)


if __name__ == "__main__":
    raise SystemExit(main())
