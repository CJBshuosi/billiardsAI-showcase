"""One-shot, 30-second synthetic capture profiling; never product acceptance."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.algorithm.recording_packet import plain, verify_packet, write_json

WORK_SECONDS = 90
CLEANUP_SECONDS = 10
CAPTURE_SECONDS = 30
REPORT_LIMIT = 256 * 1024
OPERATIONS = (
    "jpeg_encode",
    "video_decode",
    "source_validation",
    "pacing",
    "recorder_append",
    "subscription_publish",
)


def _operation(filename, name):
    filename = filename.replace("\\", "/")
    if filename == "~":
        return {
            "<imencode>": "jpeg_encode",
            "<method 'read' of 'cv2.VideoCapture' objects>": "video_decode",
        }.get(name)
    if filename.endswith("/mainline1_scenario_capture.py"):
        return {"_validate_source": "source_validation", "_wait": "pacing"}.get(name)
    if filename.endswith("/mainline1_session_recording.py") and name == "append":
        return "recorder_append"
    if filename.endswith("/mainline1_capture_hub.py") and name == "_publish":
        return "subscription_publish"
    return None


EXCLUSIVE_BUCKETS = (
    *OPERATIONS,
    "source_dispatch",
    "between_source_yields",
    "observed_lock_calls",
    "path_metadata",
    "buffer_copy",
    "hub_recorder_acquire",
    "recorder_append_acquire",
    "condition_acquire",
)


class _ObservedLock:
    """Forward every operation to the original lock, without changing its state."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __exit__(self, *args):
        return self._inner.__exit__(*args)


class HubRecorderLock(_ObservedLock):
    def __enter__(self):
        return self._inner.__enter__()


class RecorderAppendLock(_ObservedLock):
    def __enter__(self):
        return self._inner.__enter__()


class CaptureProfiler:
    """Thread-local phase timers with exhaustive wall-time accounting.

    sys.setprofile is installed only on the actual producer thread. Callback
    work is measured separately; no frames, locals, paths or arguments are kept.
    Named phases can nest, so cumulative times are not additive. Exclusive phase
    buckets plus callback overhead must explain the full observed interval.
    """

    def __init__(self):
        self._started = False
        self._done = threading.Event()
        self._frames = 0
        self._elapsed = 0.0
        self._owner = None
        self._stack = []
        self._active = {name: 0 for name in EXCLUSIVE_BUCKETS}
        self._metrics = {
            name: dict(calls=0, self_ns=0, cumulative_ns=0) for name in EXCLUSIVE_BUCKETS
        }
        self._overhead = self._pending_overhead = self._events = 0
        self._last = 0
        self._failed = False

    @staticmethod
    def _target(frame, event, arg):
        if event in ("call", "return"):
            code = frame.f_code
            tag = {
                "HubRecorderLock.__enter__": "hub_recorder_acquire",
                "RecorderAppendLock.__enter__": "recorder_append_acquire",
            }.get(code.co_qualname)
            if tag is None:
                tag = _operation(code.co_filename, code.co_name)
            if (
                tag is None
                and code.co_name == "__enter__"
                and code.co_filename.replace("\\", "/").rsplit("/", 1)[-1] == "threading.py"
            ):
                tag = "condition_acquire"
            if (
                tag is None
                and code.co_name == "frames"
                and code.co_filename.replace("\\", "/").endswith("/mainline1_scenario_capture.py")
            ):
                tag = "source_dispatch"
            return None if tag is None else (("py", id(frame)), tag, event == "call")
        if event not in ("c_call", "c_return", "c_exception"):
            return None
        name = getattr(arg, "__name__", "")
        owner = getattr(arg, "__self__", None)
        kind = type(owner).__name__
        module = type(owner).__module__
        tag = None
        if name == "imencode":
            tag = "jpeg_encode"
        elif name == "read" and kind == "VideoCapture":
            tag = "video_decode"
        elif name in ("acquire", "__enter__") and module == "_thread":
            tag = "observed_lock_calls"
        elif name == "stat" and getattr(arg, "__module__", "") in ("nt", "posix", "os"):
            tag = "path_metadata"
        elif name == "tobytes" and module == "numpy":
            tag = "buffer_copy"
        return None if tag is None else (("c", id(frame), name, kind), tag, event == "c_call")

    def _charge(self, now):
        elapsed = now - self._last - self._pending_overhead
        if elapsed < 0:
            self._failed = True
            elapsed = 0
        tag = self._stack[-1][1] if self._stack else "between_source_yields"
        self._metrics[tag]["self_ns"] += elapsed
        if not self._stack:
            self._metrics[tag]["cumulative_ns"] += elapsed
        for active, depth in self._active.items():
            if depth:
                self._metrics[active]["cumulative_ns"] += elapsed
        self._pending_overhead = 0
        self._last = now

    def _event(self, frame, event, arg):
        started = time.perf_counter_ns()
        try:
            self._events += 1
            if threading.get_ident() != self._owner:
                self._failed = True
                return
            target = self._target(frame, event, arg)
            if target is None:
                return
            self._charge(started)
            key, tag, entering = target
            if entering:
                if len(self._stack) >= 128:
                    self._failed = True
                    return
                self._metrics[tag]["calls"] += 1
                self._stack.append((key, tag))
                self._active[tag] += 1
            else:
                index = next(
                    (i for i in range(len(self._stack) - 1, -1, -1) if self._stack[i][0] == key),
                    None,
                )
                # A generator can have entered before observation began.
                if index is not None:
                    if index != len(self._stack) - 1:
                        self._failed = True
                    for _, removed in self._stack[index:]:
                        self._active[removed] -= 1
                    del self._stack[index:]
        except Exception:
            self._failed = True
        finally:
            overhead = time.perf_counter_ns() - started
            self._overhead += overhead
            self._pending_overhead += overhead

    def frames(self, iterator):
        if self._started:
            raise ValueError("capture profiler is single-use")
        if sys.getprofile() is not None:
            raise ValueError("existing producer profiling hook must not be replaced")
        self._started = True
        self._owner = threading.get_ident()
        started = time.perf_counter_ns()
        self._last = started
        sys.setprofile(self._event)
        try:
            for frame in iterator:
                self._frames += 1
                yield frame
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            finally:
                if sys.getprofile() != self._event:
                    self._failed = True
                else:
                    sys.setprofile(None)
                ended = time.perf_counter_ns()
                self._charge(ended)
                self._elapsed = (ended - started) / 1e9
                self._done.set()

    def snapshot(self):
        if not self._done.is_set():
            return dict(complete=False)
        buckets = {
            name: dict(
                calls=row["calls"],
                self_seconds=row["self_ns"] / 1e9,
                cumulative_seconds=row["cumulative_ns"] / 1e9,
            )
            for name, row in self._metrics.items()
        }
        accounted = sum(row["self_seconds"] for row in buckets.values())
        observer = self._overhead / 1e9
        verified = not self._failed and abs(accounted + observer - self._elapsed) < 1e-6
        return dict(
            complete=verified,
            yielded_frames=self._frames,
            elapsed_seconds=self._elapsed,
            operations={name: buckets[name] for name in OPERATIONS},
            exclusive_buckets=buckets,
            accounting=dict(
                verified=verified,
                direct_callback_seconds=observer,
                accounted_seconds=accounted,
                callbacks=self._events,
            ),
        )


def _number(value, *, integer=False):
    allowed = type(value) is int if integer else type(value) in (int, float)
    return value if allowed and 0 <= value < 2**53 and math.isfinite(value) else None


def _add_accounting(result, payload):
    result["accounting_verified"] = False
    raw = payload.get("exclusive_buckets") if isinstance(payload, dict) else None
    accounting = payload.get("accounting") if isinstance(payload, dict) else None
    if (
        not isinstance(raw, dict)
        or set(raw) != set(EXCLUSIVE_BUCKETS)
        or not isinstance(accounting, dict)
    ):
        return
    clean = {
        name: {
            key: _number(raw[name].get(key), integer=key == "calls")
            for key in ("calls", "self_seconds", "cumulative_seconds")
        }
        for name in EXCLUSIVE_BUCKETS
        if isinstance(raw[name], dict)
    }
    result["exclusive_buckets"] = clean
    observer = _number(accounting.get("direct_callback_seconds"))
    callbacks = _number(accounting.get("callbacks"), integer=True)
    declared = _number(accounting.get("accounted_seconds"))
    result["accounting"] = dict(
        verified=False,
        direct_callback_seconds=observer,
        accounted_seconds=declared,
        callbacks=callbacks,
    )
    if len(clean) != len(EXCLUSIVE_BUCKETS) or any(
        value is None for row in clean.values() for value in row.values()
    ):
        return
    total = sum(row["self_seconds"] for row in clean.values())
    elapsed = result.get("profile_elapsed_seconds")
    valid = bool(
        accounting.get("verified") is True
        and observer is not None
        and callbacks
        and declared is not None
        and elapsed is not None
        and abs(total - declared) < 1e-6
        and abs(total + observer - elapsed) < 1e-6
        and all(row["self_seconds"] <= row["cumulative_seconds"] + 1e-9 for row in clean.values())
        and all(clean[name] == result.get("operations", {}).get(name) for name in OPERATIONS)
    )
    result["accounting_verified"] = valid
    result["accounting"]["verified"] = valid


def _has_profile_coverage(result):
    operations = result.get("operations", {})
    return all(
        operations.get(name, {}).get("calls")
        for name in (
            "jpeg_encode",
            "video_decode",
            "source_validation",
            "recorder_append",
            "subscription_publish",
        )
    )


def summarize_probe(report, profile, manifest):
    """Fresh whitelist; even a completed probe always leaves product gates false."""
    result = dict(
        contract="recording-capture-diagnostic/v2",
        probe_completed=False,
        product_pass=False,
        model_pass=False,
        field_pass=False,
        browser_pass=False,
        capture_budget_exhausted=False,
        profile_complete=False,
        source_commit=manifest["source_commit"],
        input_sha256=manifest["input_sha256"],
    )
    if (
        not isinstance(report, dict)
        or report.get("source_commit") != manifest["source_commit"]
        or (report.get("source_sha256") != manifest["input_sha256"])
    ):
        return result
    failure = report.get("failure")
    result["capture_budget_exhausted"] = isinstance(failure, dict) and failure == dict(
        type="TimeoutError", reason="whole input capture budget exhausted"
    )
    samples = report.get("samples")
    if isinstance(samples, list) and 0 < len(samples) <= 64:
        for label, sample in (("first_sample", samples[0]), ("last_sample", samples[-1])):
            if isinstance(sample, dict):
                result[label] = {
                    key: _number(sample.get(key), integer=key != "seconds")
                    for key in ("seconds", "frame", "shots", "pending_frames", "pending_chunks")
                }
    if isinstance(profile, dict) and profile.get("complete") is True:
        operations = profile.get("operations")
        if isinstance(operations, dict) and set(operations) == set(OPERATIONS):
            result["operations"] = {
                name: {
                    key: _number(operations[name].get(key), integer=key == "calls")
                    for key in ("calls", "self_seconds", "cumulative_seconds")
                }
                for name in OPERATIONS
                if isinstance(operations[name], dict)
            }
            result["profile_complete"] = len(result["operations"]) == len(OPERATIONS) and all(
                value is not None for row in result["operations"].values() for value in row.values()
            )
        result["yielded_frames"] = _number(profile.get("yielded_frames"), integer=True)
        result["profile_elapsed_seconds"] = _number(profile.get("elapsed_seconds"))
    _add_accounting(result, profile)
    last = result.get("last_sample", {})
    result["probe_completed"] = bool(
        report.get("cleanup_confirmed") is True
        and result["capture_budget_exhausted"]
        and result["profile_complete"]
        and _has_profile_coverage(result)
        and result["accounting_verified"]
        and all(
            result.get("exclusive_buckets", {}).get(name, {}).get("calls")
            for name in ("hub_recorder_acquire", "recorder_append_acquire", "condition_acquire")
        )
        and result.get("yielded_frames")
        and result.get("profile_elapsed_seconds")
        and last.get("frame")
        and all(value is not None for value in last.values())
    )
    return result


def sanitize_worker_output(payload, manifest):
    result = dict(
        contract="recording-capture-diagnostic/v2",
        probe_completed=False,
        product_pass=False,
        model_pass=False,
        field_pass=False,
        browser_pass=False,
        source_commit=manifest["source_commit"],
        input_sha256=manifest["input_sha256"],
    )
    if not isinstance(payload, dict) or any(
        payload.get(key) != result[key] for key in ("contract", "source_commit", "input_sha256")
    ):
        return result
    for key in ("capture_budget_exhausted", "profile_complete"):
        result[key] = payload.get(key) is True
    for label in ("first_sample", "last_sample"):
        sample = payload.get(label)
        if isinstance(sample, dict):
            result[label] = {
                key: _number(sample.get(key), integer=key != "seconds")
                for key in ("seconds", "frame", "shots", "pending_frames", "pending_chunks")
            }
    for key in ("yielded_frames", "profile_elapsed_seconds"):
        result[key] = _number(payload.get(key), integer=key == "yielded_frames")
    operations = payload.get("operations")
    if isinstance(operations, dict):
        result["operations"] = {
            name: {
                key: _number(operations[name].get(key), integer=key == "calls")
                for key in ("calls", "self_seconds", "cumulative_seconds")
            }
            for name in OPERATIONS
            if isinstance(operations.get(name), dict)
        }
        result["profile_complete"] = result["profile_complete"] and (
            len(result["operations"]) == len(OPERATIONS)
            and all(
                value is not None for row in result["operations"].values() for value in row.values()
            )
        )
    else:
        result["profile_complete"] = False
    _add_accounting(result, payload)
    result["probe_completed"] = bool(
        payload.get("probe_completed") is True
        and result["capture_budget_exhausted"]
        and result["profile_complete"]
        and _has_profile_coverage(result)
        and result["accounting_verified"]
        and all(
            result.get("exclusive_buckets", {}).get(name, {}).get("calls")
            for name in ("hub_recorder_acquire", "recorder_append_acquire", "condition_acquire")
        )
        and result.get("yielded_frames")
        and result.get("profile_elapsed_seconds")
        and result.get("last_sample", {}).get("frame")
        and all(value is not None for value in result.get("last_sample", {}).values())
    )
    return result


def run_probe(root, manifest):
    # Product modules are loaded only in the owned, explicitly released worker.
    from unittest.mock import patch

    from billiards_ai.lab.mainline1_capture_hub import CaptureHub
    from billiards_ai.lab.mainline1_session_recording import ContinuousSessionRecorder
    from scripts.algorithm import run_recording_first_fullgame as runner

    profiler = CaptureProfiler()
    original_hub_init = CaptureHub.__init__
    original_recorder_init = ContinuousSessionRecorder.__init__

    def observed_hub_init(instance, *args, **kwargs):
        original_hub_init(instance, *args, **kwargs)
        instance._recorder_lock = HubRecorderLock(instance._recorder_lock)

    def observed_recorder_init(instance, *args, **kwargs):
        original_recorder_init(instance, *args, **kwargs)
        instance._append_lock = RecorderAppendLock(instance._append_lock)

    original = runner.ScenarioVideoCaptureBackend

    class ProfiledSource(original):
        def frames(self, stop):
            yield from profiler.frames(super().frames(stop))

    with (
        patch.object(runner, "ScenarioVideoCaptureBackend", ProfiledSource),
        patch.object(CaptureHub, "__init__", observed_hub_init),
        patch.object(ContinuousSessionRecorder, "__init__", observed_recorder_init),
    ):
        runner.main(
            [
                "--input-root",
                str(root / "packet/input"),
                "--output-root",
                str(root / "recording"),
                "--capture-seconds",
                str(CAPTURE_SECONDS),
                "--encoding-seconds",
                "1",
                "--browser-seconds",
                "0",
            ]
        )
    path = plain(root / "recording/report.json")
    report = None
    if path.is_file() and path.stat().st_size <= REPORT_LIMIT:
        report = json.loads(path.read_bytes())
    result = summarize_probe(report, profiler.snapshot(), manifest)
    write_json(root / "output/probe.json", result)
    return 0 if result["probe_completed"] else 1


def checked_root(root, digest):
    root = plain(root)
    if (
        os.name != "nt"
        or root.parent != Path("C:/Users/developer/Documents")
        or not re.fullmatch(
            r"billiardsAI-recording-profile-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", root.name
        )
    ):
        raise ValueError("fresh expected Windows profile root required")
    prepared = plain(root / "PREPARED.json")
    if (
        prepared.stat().st_size > 16384
        or json.loads(prepared.read_text(encoding="utf-8-sig")).get("manifest_sha256") != digest
    ):
        raise ValueError("prepared manifest mismatch")
    packet = root / "packet"
    if Path(__file__).resolve() != packet / "entry.py":
        raise ValueError("exact diagnostic entry required")
    manifest = verify_packet(packet, digest)
    if manifest.get("purpose") != "capture-throughput-diagnostic/v1":
        raise ValueError("diagnostic purpose required")
    return root, manifest


def launch(root, digest, manifest):
    # Reuse unchanged, manifest-pinned ownership/environment helpers, never old main().
    from ownership import WindowsAPI, run_owned

    from scripts.algorithm.recording_windows_entry import (
        LOG_LIMIT,
        _CleanupAPI,
        _LogBudget,
        child_environment,
    )

    if any(
        (root / name).exists()
        for name in (
            "once.json",
            "output",
            "recording",
            "profile",
            "temp",
            "worker-stdout.txt",
            "worker-stderr.txt",
        )
    ):
        raise ValueError("diagnostic single run consumed or root not fresh")
    write_json(
        root / "once.json", dict(contract="recording-profile-once/v1", manifest_sha256=digest)
    )
    output = root / "output"
    output.mkdir()
    environment = child_environment(root)
    paths = (root / "worker-stdout.txt", root / "worker-stderr.txt")
    budget = _LogBudget(paths, LOG_LIMIT)
    with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:

        def spawn():
            return subprocess.Popen(
                [
                    str(root / "packet/runtime/python.exe"),
                    "-I",
                    "-B",
                    str(root / "packet/entry.py"),
                    "worker",
                    str(root),
                    digest,
                ],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                cwd=root / "packet",
                close_fds=True,
            )

        owned = run_owned(
            _CleanupAPI(WindowsAPI(), budget),
            spawn,
            lambda stage, state: write_json(
                output / "ownership.json", dict(stage=stage, **state), replace=True
            ),
            dict(
                contract="recording-profile-owned/v1",
                manifest_sha256=digest,
                work_seconds=WORK_SECONDS,
                cleanup_seconds=CLEANUP_SECONDS,
            ),
            ("PROFILE-RELEASE " + digest + "\n").encode("ascii"),
            deadline=WORK_SECONDS,
            cleanup_seconds=CLEANUP_SECONDS,
            sleep=budget.sleep,
        )
    probe_path = plain(output / "probe.json")
    result = dict(
        contract="recording-capture-diagnostic/v2",
        probe_completed=False,
        product_pass=False,
        model_pass=False,
        field_pass=False,
        browser_pass=False,
    )
    if probe_path.is_file() and probe_path.stat().st_size <= 16384:
        try:
            result = sanitize_worker_output(json.loads(probe_path.read_bytes()), manifest)
        except (ValueError, OSError):
            pass
    result.update(
        cleanup_confirmed=owned.get("cleanup_confirmed") is True,
        ownership_status=owned.get("status"),
        worker_exit_code=owned.get("worker_exit_code"),
    )
    result["probe_completed"] = bool(
        result.get("probe_completed") is True
        and result["cleanup_confirmed"]
        and owned.get("status") == "PASS"
        and owned.get("worker_exit_code") == 0
        and not budget.exceeded()
    )
    write_json(output / "summary.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["probe_completed"] else 1


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] not in {"launch", "worker"}:
        raise ValueError("explicit role, fresh root and manifest SHA required")
    if args[0] == "worker" and sys.stdin.buffer.readline(200) != (
        "PROFILE-RELEASE " + args[2] + "\n"
    ).encode("ascii"):
        raise ValueError("worker did not receive exact owned release")
    root, manifest = checked_root(Path(args[1]), args[2])
    return launch(root, args[2], manifest) if args[0] == "launch" else run_probe(root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
