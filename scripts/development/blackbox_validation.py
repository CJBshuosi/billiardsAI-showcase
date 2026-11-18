"""Bounded Windows development runner for the existing synthetic full-game test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

# Before Job release this entry imports only trusted stdlib and stdlib-only helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.development import blackbox_report as reports  # noqa: E402
from scripts.development import blackbox_source as sources  # noqa: E402

OWNERSHIP_SHA = "76e90eb5490e7c899788a87040bf827bac7b5ae057efbcba59475efe9b011a8f"
SOURCE_SHA = "6a234abb4d9df0efea2c23f21bfe138025e5bd8c291e33229af9e48ed37a9453"
LOG_LIMIT = 4 * 1024**2
PHASES = {"setup", "worker"}
FAILURE_STAGES = {
    "CONFIG",
    "RUNTIME",
    "DEPENDENCIES",
    "SOURCE",
    "SETUP",
    "WORKER",
    "CODEC",
    "INPUT",
    "PREFLIGHT",
    "FULLGAME",
    "EXPORT",
    "CLEANUP",
    "LAUNCHER",
}
FAILURE_CODES = {
    "JSON_SIZE_BUDGET",
    "CONFIG_CONTRACT",
    "CONFIG_ABSOLUTE_PATH_REQUIRED",
    "CONFIG_PATH_MISSING",
    "RUNTIME_PINS_INCOMPLETE",
    "RUNTIME_PATH_INJECTION",
    "RUNTIME_IDENTITY_MISMATCH",
    "GIT_IDENTITY_MISMATCH",
    "BOOTSTRAP_IDENTITY_MISMATCH",
    "DEPENDENCY_PINS_REQUIRED",
    "DEPENDENCY_MEMBER_INVALID",
    "DEPENDENCY_IDENTITY_MISMATCH",
    "DEPENDENCY_SIZE_BUDGET",
    "DEPENDENCY_COPY_CHANGED",
    "TRANSPORT_COMMIT_INVALID",
    "TRANSPORT_COMMIT_MISMATCH",
    "SOURCE_CHECKOUT_DIRTY",
    "SOURCE_ORIGIN_MISMATCH",
    "SOURCE_LOCK_INVALID",
    "SOURCE_GIT_FAILED",
    "DISK_RESERVE",
    "CAPTURE_TIMEOUT",
    "H264_ENCODER_UNAVAILABLE",
    "H264_GEOMETRY",
    "H264_READBACK",
    "INPUT_READBACK",
    "DEPENDENCY_IMPORT_FAILED",
    "DEPENDENCY_IMPORT_OUTSIDE_SNAPSHOT",
    "PRODUCT_IMPORT_OUTSIDE_SNAPSHOT",
    "INPUT_IDENTITY_MISMATCH",
    "OWNERSHIP_IDENTITY_MISMATCH",
    "CANCELLED",
    "PRODUCT_FAILED",
    "PRODUCT_TIMEOUT",
    "GIT_TIMEOUT",
    "SETUP_VALIDATION_FAILED",
    "PREFLIGHT_FAILED",
    "FULLGAME_EXECUTION_FAILED",
    "SOURCE_IDENTITY_CHANGED",
    "SETUP_TIMEOUT",
    "WORKER_TIMEOUT",
    "PREFLIGHT_TIMEOUT",
    "FULLGAME_TIMEOUT",
    "LOG_BUDGET",
    "OUTPUT_BUDGET",
    "OWNERSHIP_FAILED",
    "WORKER_RESULT_MISSING",
    "REPORT_IDENTITY_MISMATCH",
    "REPORT_INVALID",
    "CHILD_RESULT_IDENTITY_MISMATCH",
    "CHILD_RESULT_INVALID",
    "REPORT_PUBLICATION_FAILED",
    "LOCK_RELEASE_FAILED",
    "RUNS_DIRECTORY_REQUIRED",
    "SETUP_IDENTITY_MISSING",
    "RUN_BUSY",
    "PATH_COLLISION",
    "LAUNCHER_FAILED",
}


class ValidationFailure(ValueError):
    def __init__(self, code, stage, exit_code=2):
        super().__init__(code)
        self.code, self.stage, self.exit_code = code, stage, exit_code


def utc_now():
    return datetime.now(UTC).isoformat()


def read_json(path, limit=2 * 1024**2):
    path = sources.plain(path)
    if path.stat().st_size > limit:
        raise ValidationFailure("JSON_SIZE_BUDGET", "CONFIG")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValidationFailure("JSON_SIZE_BUDGET", "CONFIG")
    return json.loads(data)


def absolute_machine_path(raw):
    # Lexical only: no stat/resolve/network metadata before the owned setup Job.
    if (
        not isinstance(raw, str)
        or not Path(raw).is_absolute()
        or "\n" in raw
        or "\r" in raw
        or raw.replace("\\", "/").startswith("//")
        or (os.name == "nt" and not re.match(r"^[A-Za-z]:[\\/]", raw))
    ):
        raise ValidationFailure("CONFIG_ABSOLUTE_PATH_REQUIRED", "CONFIG")
    return str(Path(raw))


def normalize_config(config):
    if not isinstance(config, dict) or config.get("contract") != "blackbox-dev-machine/v1":
        raise ValidationFailure("CONFIG_CONTRACT", "CONFIG")
    config = dict(config)
    for key in (
        "runtime",
        "dependencies",
        "fixtures",
        "codec",
        "ownership",
        "runs",
        "git_executable",
        "bootstrap_python",
    ):
        config[key] = absolute_machine_path(config.get(key))
    return config


def check_config(config):
    config = normalize_config(config)
    for key in (
        "runtime",
        "dependencies",
        "fixtures",
        "codec",
        "ownership",
        "runs",
        "git_executable",
        "bootstrap_python",
    ):
        config[key] = str(sources.plain(config[key]))
        if not Path(config[key]).exists():
            raise ValidationFailure("CONFIG_PATH_MISSING", "CONFIG")
    pins = config["runtime_pins"]
    if not {"python.exe", "python312.dll", "python312.zip", "_ctypes.pyd"} <= pins.keys():
        raise ValidationFailure("RUNTIME_PINS_INCOMPLETE", "RUNTIME")
    for name, identity in pins.items():
        sources.safe_name(name)
        if "/" in name or name.endswith((".pth", "._pth")):
            raise ValidationFailure("RUNTIME_PATH_INJECTION", "RUNTIME")
        if sources.pin(Path(config["runtime"]) / name) != identity:
            raise ValidationFailure("RUNTIME_IDENTITY_MISMATCH", "RUNTIME")
    if sources.pin(config["git_executable"])["sha256"] != config["git_sha256"]:
        raise ValidationFailure("GIT_IDENTITY_MISMATCH", "SOURCE")
    if sources.pin(config["bootstrap_python"])["sha256"] != config["bootstrap_python_sha256"]:
        raise ValidationFailure("BOOTSTRAP_IDENTITY_MISMATCH", "RUNTIME")
    if not config.get("dependency_pins"):
        raise ValidationFailure("DEPENDENCY_PINS_REQUIRED", "DEPENDENCIES")
    return config


def copy_pinned_tree(origin, target, pins):
    """Unlisted dependency files never enter the worker's import search path."""
    target.mkdir()
    folded, total = set(), 0
    for name, identity in pins.items():
        sources.safe_name(name)
        if (
            name.casefold() in folded
            or name.endswith((".pth", "._pth", ".pyc"))
            or Path(name).name in {"sitecustomize.py", "usercustomize.py"}
        ):
            raise ValidationFailure("DEPENDENCY_MEMBER_INVALID", "DEPENDENCIES")
        folded.add(name.casefold())
        path = sources.plain(Path(origin) / name)
        if sources.pin(path) != identity:
            raise ValidationFailure("DEPENDENCY_IDENTITY_MISMATCH", "DEPENDENCIES")
        total += identity["bytes"]
        if total > 1024**3:
            raise ValidationFailure("DEPENDENCY_SIZE_BUDGET", "DEPENDENCIES")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sources.pin(destination) != identity:
            raise ValidationFailure("DEPENDENCY_COPY_CHANGED", "DEPENDENCIES")


def freeze_checkout(repo, commit, target, *, git_executable="git", check_remote=False):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValidationFailure("TRANSPORT_COMMIT_INVALID", "SOURCE")

    def git(*args):
        return sources.git(repo, *args, executable=git_executable)

    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise ValidationFailure("TRANSPORT_COMMIT_MISMATCH", "SOURCE")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValidationFailure("SOURCE_CHECKOUT_DIRTY", "SOURCE")
    if check_remote:
        if git("remote", "get-url", "origin").decode().strip() != (
            "https://github.com/CJBshuosi/billiardsAI.git"
        ):
            raise ValidationFailure("SOURCE_ORIGIN_MISMATCH", "SOURCE")
        git(
            "merge-base", "--is-ancestor", commit, "refs/remotes/origin/codex/windows-blackbox-sync"
        )
    try:
        lock = sources.verify(repo, checkout=True)
    except (OSError, ValueError) as exc:
        raise ValidationFailure("SOURCE_LOCK_INVALID", "SOURCE") from exc
    target.mkdir()
    for name in (*lock["files"], "SOURCE-LOCK.json"):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    sources.verify(target)
    return lock


class Monitor:
    def __init__(self, root, summary, *, phase="worker", clock=time.monotonic, sleeper=time.sleep):
        self.root, self.summary, self.clock, self.sleeper = root, summary, clock, sleeper
        self.phase = phase
        self.start, self.offset, self.pending = clock(), 0, b""
        self.events = root / "export/events.jsonl"
        self.events.touch(exist_ok=True)
        self.last_persist = float("-inf")
        self.preflight_done = False
        self.capture_stopped = False
        self.phase_start = self.start
        self.failure_code = None
        self.last_storage_check = float("-inf")

    def observe(self):
        if self.failure_code:
            return
        logs = [
            self.root / f"local-logs/{self.phase}-{suffix}.txt" for suffix in ("stdout", "stderr")
        ]
        if any(p.exists() and p.stat().st_size > LOG_LIMIT for p in logs):
            self.fail("LOG_BUDGET")
        if logs[0].exists():
            with logs[0].open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(LOG_LIMIT)
                self.offset = stream.tell()
            self.pending += chunk
            if len(self.pending) > LOG_LIMIT:
                self.fail("LOG_BUDGET")
            rows = self.pending.split(b"\n")
            self.pending = rows.pop()
            for row in rows:
                try:
                    item = reports.event(json.loads(row))
                except (ValueError, UnicodeError):
                    continue
                if item:
                    item.update(
                        phase=self.phase, observed_seconds=round(self.clock() - self.start, 3)
                    )
                    with self.events.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(item, allow_nan=False) + "\n")
                    self.summary.update(stage=item["stage"], last_observed=item)
                    if item["stage"] == "capture_stopped":
                        self.capture_stopped = True
                    if item["stage"] == "preflight" and not self.preflight_done:
                        self.preflight_done = True
                        self.phase_start = self.clock()
        if self.summary.get("profile") == "FullgameComplete":
            cancel = self.root / "cancel.json"
            if cancel.exists():
                try:
                    requested = read_json(cancel, 1024)
                except (OSError, ValueError):
                    requested = None
                if requested == {"run_id": self.root.name}:
                    self.failure_code = "CANCELLED"
                    self.summary.update(failure_code="CANCELLED", failure_stage=self.phase.upper())
                    raise KeyboardInterrupt("explicit run cancellation")
            heartbeat = self.root / "recording/observer-heartbeat.json"
            if heartbeat.exists():
                try:
                    value = read_json(heartbeat, 256 * 1024)
                    if (
                        value.get("contract") == "recording-fullgame-progress/v1"
                        and value.get("run_id") == self.root.name
                    ):
                        latest = reports.event(dict(stage="background", **value.get("latest", {})))
                        self.summary["progress"] = latest
                        if reports.finite(value.get("seconds")):
                            self.summary["heartbeat_seconds"] = value["seconds"]
                        if isinstance(value.get("diagnostic"), dict):
                            self.summary["progress_diagnostic"] = "NO_PROGRESS_CHECKPOINT"
                except (OSError, ValueError, TypeError):
                    self.summary["progress_diagnostic"] = "HEARTBEAT_UNAVAILABLE"
        now = self.clock()
        complete_profile = self.summary.get("profile") == "FullgameComplete"
        if complete_profile and self.preflight_done:
            # Short capture watchdog ends permanently at the actual end ACK.
            # There is no cumulative background/product/worker deadline afterwards.
            if not self.capture_stopped and now - self.phase_start >= 390:
                self.fail("CAPTURE_TIMEOUT")
        else:
            limit = (
                (390 if self.summary.get("profile") == "CaptureEnd" else 900)
                if self.preflight_done
                else 60
            )
            if now - self.phase_start >= limit:
                self.fail(
                    "SETUP_TIMEOUT"
                    if self.phase == "setup"
                    else "FULLGAME_TIMEOUT"
                    if self.preflight_done
                    else "PREFLIGHT_TIMEOUT"
                )
        if now - self.last_storage_check >= 5:
            if shutil.disk_usage(self.root).free < 2 * 1024**3:
                self.fail("DISK_RESERVE")
            size = 0
            recording = self.root / "recording"
            if recording.exists():
                for directory, _, names in os.walk(recording, followlinks=False):
                    for name in names:
                        path = sources.plain(Path(directory) / name)
                        try:
                            size += path.stat().st_size
                        except FileNotFoundError:
                            continue
                if size > 5 * 1024**3:
                    self.fail("OUTPUT_BUDGET")
            self.last_storage_check = now
        if now - self.last_persist >= 1:
            reports.publish(self.root / "export", self.summary)
            self.last_persist = now

    def fail(self, code):
        self.failure_code = code
        self.summary.update(failure_code=code, failure_stage=self.phase.upper())
        raise RuntimeError(code)

    def sleep(self, seconds):
        self.observe()
        self.sleeper(seconds)


def child_runtime(root, config):
    runtime = root / "runtime"
    copy_pinned_tree(config["runtime"], runtime, config["runtime_pins"])
    lines = [
        "python312.zip",
        ".",
        str(root / "source/src"),
        str(root / "source"),
        str(root / "dependencies"),
    ]
    (runtime / "python312._pth").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runtime / "python.exe"


def child_environment(root, config, *, setup=False):
    system = os.environ["SYSTEMROOT"]
    env = {
        "SYSTEMROOT": system,
        "WINDIR": system,
        "OS": "Windows_NT",
        "PATH": str(Path(system) / "System32"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if not setup:
        from scripts.algorithm.recording_windows_entry import CODEC

        env["OPENH264_LIBRARY"] = str(root / "codec" / CODEC["filename"])
    for key, suffix in {
        "TEMP": "temp",
        "TMP": "temp",
        "USERPROFILE": "profile",
        "APPDATA": "profile/AppData/Roaming",
        "LOCALAPPDATA": "profile/AppData/Local",
    }.items():
        path = root / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def setup_worker(root, context):
    config = check_config(context["config"])
    lock = freeze_checkout(
        Path(context["repo"]),
        context["commit"],
        root / "source",
        git_executable=config["git_executable"],
        check_remote=True,
    )
    print(json.dumps({"stage": "setup"}), flush=True)
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise ValidationFailure("DISK_RESERVE", "SETUP")
    copy_pinned_tree(config["dependencies"], root / "dependencies", config["dependency_pins"])
    from scripts.algorithm.recording_windows_entry import CODEC

    codec_pins = {
        CODEC["filename"]: {"bytes": CODEC["bytes"], "sha256": CODEC["sha256"]},
        "BINARY_LICENSE.txt": {"bytes": CODEC["license_bytes"], "sha256": CODEC["license_sha256"]},
    }
    copy_pinned_tree(config["codec"], root / "codec", codec_pins)
    child_runtime(root, config)
    identity = {
        "runtime": config["runtime_pins"],
        "dependencies": config["dependency_pins"],
        "codec": codec_pins,
        "ownership": OWNERSHIP_SHA,
    }
    return {
        "source_commit": lock["commit"],
        "source_lock": sources.pin(root / "source/SOURCE-LOCK.json"),
        "environment_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    }


def media_preflight(root, fixtures):
    import cv2
    import numpy as np

    from billiards_ai.lab.mainline1_recording_media import _fsync_file

    probe = root / "encoder-preflight.mp4"
    writer = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*"avc1"), 60, (1920, 1080))
    capture = None
    try:
        if not writer.isOpened():
            raise ValidationFailure("H264_ENCODER_UNAVAILABLE", "CODEC")
        for _ in range(2):
            writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
        writer.release()
        _fsync_file(probe)
        capture = cv2.VideoCapture(str(probe))
        count = 0
        while count < 3:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (1080, 1920, 3):
                raise ValidationFailure("H264_GEOMETRY", "CODEC")
            count += 1
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower()
        if count != 2 or codec not in {"avc1", "h264", "x264"}:
            raise ValidationFailure("H264_READBACK", "CODEC")
    finally:
        writer.release()
        if capture is not None:
            capture.release()
    capture = cv2.VideoCapture(str(Path(fixtures) / "input.avi"))
    try:
        for _ in range(2):
            ok, frame = capture.read()
            if not ok or frame.shape != (1080, 1920, 3):
                raise ValidationFailure("INPUT_READBACK", "INPUT")
    finally:
        capture.release()


def worker(root, context):
    try:
        sources.verify(root / "source")
    except (OSError, ValueError) as exc:
        raise ValidationFailure("SOURCE_LOCK_INVALID", "SOURCE") from exc
    try:
        import cv2
        import numpy as np

        from billiards_ai.lab.mainline1_recording_media import _fsync_file
        from scripts.algorithm import run_recording_first_fullgame as fullgame
    except ImportError as exc:
        raise ValidationFailure("DEPENDENCY_IMPORT_FAILED", "DEPENDENCIES") from exc
    for module in (cv2, np):
        if not Path(module.__file__).resolve().is_relative_to((root / "dependencies").resolve()):
            raise ValidationFailure("DEPENDENCY_IMPORT_OUTSIDE_SNAPSHOT", "DEPENDENCIES")
    for module in (fullgame, sys.modules[_fsync_file.__module__]):
        if not Path(module.__file__).resolve().is_relative_to((root / "source").resolve()):
            raise ValidationFailure("PRODUCT_IMPORT_OUTSIDE_SNAPSHOT", "SOURCE")
    try:
        fullgame._validate_inputs(Path(context["config"]["fixtures"]))
    except (ValueError, OSError) as exc:
        raise ValidationFailure("INPUT_IDENTITY_MISMATCH", "INPUT") from exc
    print(json.dumps({"stage": "module_check"}), flush=True)
    media_preflight(root, context["config"]["fixtures"])
    print(json.dumps({"stage": "preflight", "decoded_frames": 2}), flush=True)
    if context["profile"] == "Preflight":
        return 0
    if context["profile"] == "CaptureEnd":
        from scripts.development.blackbox_capture_end_probe import run as probe

        result = probe(
            fullgame,
            [
                "--input-root",
                context["config"]["fixtures"],
                "--output-root",
                str(root / "recording"),
                "--capture-seconds",
                "360",
                "--encoding-seconds",
                "600",
                "--browser-seconds",
                "0",
            ],
            root,
            sources.verify(root / "source")["commit"],
            SOURCE_SHA,
        )
        try:
            sources.verify(root / "source")
        except (OSError, ValueError) as exc:
            raise ValidationFailure("SOURCE_IDENTITY_CHANGED", "SOURCE", 5) from exc
        return result
    try:
        result = fullgame.main(
            [
                "--input-root",
                context["config"]["fixtures"],
                "--output-root",
                str(root / "recording"),
                "--capture-seconds",
                "360",
                "--encoding-seconds",
                "600",
                "--browser-seconds",
                "90",
            ]
            + (["--until-complete"] if context["profile"] == "FullgameComplete" else [])
        )
    except TimeoutError as exc:
        raise ValidationFailure("PRODUCT_TIMEOUT", "FULLGAME", 3) from exc
    except Exception as exc:
        raise ValidationFailure("FULLGAME_EXECUTION_FAILED", "FULLGAME", 5) from exc
    try:
        sources.verify(root / "source")
    except (OSError, ValueError) as exc:
        raise ValidationFailure("SOURCE_IDENTITY_CHANGED", "SOURCE", 5) from exc
    return result


def load_ownership(path):
    if Path(path).stat().st_size != 8087 or sources.pin(path)["sha256"] != OWNERSHIP_SHA:
        raise ValidationFailure("OWNERSHIP_IDENTITY_MISMATCH", "SETUP")
    spec = importlib.util.spec_from_file_location("blackbox_owned_process", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_report(summary, report):
    return (
        isinstance(report, dict)
        and report.get("source_commit") == summary.get("source_commit")
        and report.get("source_sha256") == SOURCE_SHA
    )


def finish(summary, owned, report=None, result=None, *, phase="worker", observed_failure=None):
    summary["owned_cleanup"] = owned.get("cleanup_confirmed") is True
    rc, reason = owned.get("worker_exit_code"), owned.get("reason")
    code, status = 0, "PASS"
    failure_code, failure_stage = None, phase.upper()
    if (
        not summary["owned_cleanup"]
        or owned.get("terminal_record_failed")
        or owned.get("job_close_error")
    ):
        code, status, failure_code = 5, "ERROR", "OWNERSHIP_FAILED"
    elif reason == "deadline":
        code, status, failure_code = 3, "TIMEOUT", phase.upper() + "_TIMEOUT"
    elif owned.get("exception_type") == "KeyboardInterrupt":
        code, status, failure_code = 4, "CANCELLED", "CANCELLED"
    elif observed_failure:
        code, status, failure_code = (
            (3, "TIMEOUT", observed_failure)
            if observed_failure.endswith("TIMEOUT")
            else (5, "ERROR", observed_failure)
        )
    elif owned.get("status") != "PASS" and (rc == 0 or reason != "worker-failed"):
        code, status, failure_code = 5, "ERROR", "OWNERSHIP_FAILED"
    elif result is not None:
        code = result["exit_code"]
        status = {
            0: "PASS",
            1: "FAILED",
            2: "PRECONDITION_FAILED",
            3: "TIMEOUT",
            4: "CANCELLED",
            5: "ERROR",
        }[code]
        failure_code, failure_stage = result.get("failure_code"), result.get("failure_stage")
    elif rc != 0:
        code, status, failure_code = 1, "FAILED", "WORKER_RESULT_MISSING"
    else:
        code, status, failure_code = 5, "ERROR", "WORKER_RESULT_MISSING"
    if phase == "worker" and summary["profile"] in {"Fullgame", "FullgameComplete"}:
        summary["product_status"] = "REPORT_UNAVAILABLE"
        if valid_report(summary, report):
            from scripts.algorithm.recording_windows_entry import summarize

            safe = summarize(
                report,
                owned,
                {"source_commit": summary["source_commit"], "input_sha256": SOURCE_SHA},
            )
            if not isinstance(report.get("media"), list):
                safe["clip_count"] = None
            timeline = report.get("clip_timeline")
            if isinstance(timeline, list) and len(timeline) <= 384:
                summary["clip_timeline"] = [
                    reports.event(dict(stage="clip_done", **row))
                    for row in timeline
                    if isinstance(row, dict)
                ]
            if summary["profile"] == "FullgameComplete":
                summary["observation_status"] = (
                    "PASS"
                    if report.get("observation_status") == "PASS"
                    and isinstance(timeline, list)
                    and 28 <= len(timeline) <= 384
                    else "INCOMPLETE"
                )
                if code == 0 and summary["observation_status"] != "PASS":
                    code, status, failure_code = 5, "ERROR", "REPORT_INVALID"
            summary["completion_policy"] = (
                "UNTIL_COMPLETE" if summary["profile"] == "FullgameComplete" else "TIMED"
            )
            checkpoint = report.get("background_checkpoint_600")
            if isinstance(checkpoint, dict):
                summary["background_checkpoint_600"] = reports.event(
                    dict(stage="checkpoint_600", **checkpoint)
                )
            summary["product_evidence"] = safe
            summary["product_status"] = "PASS" if safe["product_pass"] else "FAIL"
            summary["deferred_cleanup"] = report.get("cleanup_confirmed") is not True
            if code in {0, 1} and safe.get("failure_type") == "TimeoutError":
                code, status, failure_code, failure_stage = (
                    3,
                    "TIMEOUT",
                    "PRODUCT_TIMEOUT",
                    "FULLGAME",
                )
        elif report is not None:
            summary["product_report_state"] = "IDENTITY_MISMATCH"
            if code not in {3, 4} and failure_code != "OWNERSHIP_FAILED":
                code, status, failure_code = 5, "ERROR", "REPORT_IDENTITY_MISMATCH"
        if summary["product_status"] != "PASS" and code == 0:
            code, status, failure_code = 1, "FAILED", "PRODUCT_FAILED"
    summary.update(
        status=status, exit_code=code, failure_code=failure_code, failure_stage=failure_stage
    )
    return summary


def read_result(root, phase, owned):
    path = root / f"{phase}-result.json"
    if not path.exists():
        return None
    value = read_json(path, 16384)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"contract", "run_id", "phase", "exit_code", "failure_code", "failure_stage", "identity"}
        or value.get("contract") != "blackbox-dev-child/v1"
        or value.get("run_id") != root.name
        or value.get("phase") != phase
        or type(value.get("exit_code")) is not int
        or value["exit_code"] not in range(6)
        or value["exit_code"] != owned.get("worker_exit_code")
    ):
        raise ValidationFailure("CHILD_RESULT_IDENTITY_MISMATCH", phase.upper(), 5)
    if (
        value.get("failure_code") not in FAILURE_CODES | {None}
        or value.get("failure_stage") not in FAILURE_STAGES
        or bool(value["exit_code"]) != (value["failure_code"] is not None)
    ):
        raise ValidationFailure("CHILD_RESULT_INVALID", phase.upper(), 5)
    if phase == "setup" and value["exit_code"] == 0:
        validate_identity(value["identity"])
    elif value["identity"] is not None:
        raise ValidationFailure("CHILD_RESULT_INVALID", phase.upper(), 5)
    return value


def validate_identity(identity):
    if (
        not isinstance(identity, dict)
        or set(identity) != {"source_commit", "source_lock", "environment_id"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("source_commit")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("environment_id")))
    ):
        raise ValidationFailure("SETUP_IDENTITY_MISSING", "SETUP", 5)
    pin = identity["source_lock"]
    if (
        not isinstance(pin, dict)
        or set(pin) != {"bytes", "sha256"}
        or type(pin["bytes"]) is not int
        or not 0 < pin["bytes"] <= 2 * 1024**2
        or not re.fullmatch(r"[0-9a-f]{64}", str(pin["sha256"]))
    ):
        raise ValidationFailure("SETUP_IDENTITY_MISSING", "SETUP", 5)
    return identity


def read_product_report(root, profile):
    # The fixed 28-shot source/evidence plus bounded timeline is 373KiB in Windows.
    # This is a local read ceiling, separate from the <=600KiB finite export batch.
    limit = 1024**2 if profile == "FullgameComplete" else 256 * 1024
    return read_json(root / "recording/report.json", limit)


def execute_phase(root, config, summary, ownership, phase):
    setup = phase == "setup"
    executable = Path(sys.executable) if setup else root / "runtime/python.exe"
    entry = (
        Path(__file__).resolve()
        if setup
        else root / "source/scripts/development/blackbox_validation.py"
    )
    command = [str(executable), "-I", "-B", str(entry), "--child", phase, str(root)]
    monitor = Monitor(root, summary, phase=phase)
    with (
        (root / f"local-logs/{phase}-stdout.txt").open("xb") as stdout,
        (root / f"local-logs/{phase}-stderr.txt").open("xb") as stderr,
    ):
        owned = ownership.run_owned(
            ownership.WindowsAPI(),
            lambda: subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=child_environment(root, config, setup=setup),
                cwd=root,
                close_fds=True,
            ),
            lambda stage, state: reports.atomic_json(
                root / f"{phase}-ownership.json", {"stage": stage, **state}
            ),
            {"run_id": root.name, "phase": phase},
            ("BLACKBOX-RELEASE " + root.name + "\n").encode(),
            deadline=(
                60
                if setup or summary["profile"] == "Preflight"
                else 450
                if summary["profile"] == "CaptureEnd"
                else float("inf")
                if summary["profile"] == "FullgameComplete"
                else 960
            ),
            cleanup_seconds=10,
            sleep=monitor.sleep,
        )
    summary["owned_cleanup"] = owned.get("cleanup_confirmed") is True
    try:
        monitor.observe()
    except RuntimeError:
        if not monitor.failure_code:
            raise
    # A deadline can stop a child after writing its result but before process exit.
    # In that case the Job's terminal state is authoritative, not its uncommitted result.
    result, report, invalid = None, None, None
    try:
        if type(owned.get("worker_exit_code")) is int:
            result = read_result(root, phase, owned)
    except (OSError, ValueError):
        invalid = "CHILD_RESULT_INVALID"
    report_path = root / "recording/report.json"
    if not setup:
        summary["product_report_state"] = "MISSING"
        try:
            if report_path.exists():
                report = read_product_report(root, summary["profile"])
                summary["product_report_state"] = "AVAILABLE"
        except (OSError, ValueError):
            invalid = invalid or "REPORT_INVALID"
            summary["product_report_state"] = "INVALID"
    finish(summary, owned, report, result, phase=phase, observed_failure=monitor.failure_code)
    if not setup and summary["profile"] == "CaptureEnd":
        from scripts.development.blackbox_capture_end_probe import controller_passed, export

        summary.update(validation_scope="CAPTURE_END_ONLY", capture_end_status="FAIL")
        try:
            raw_probe = read_json(root / "capture-end-probe.json", 16384)
            if raw_probe.get("original_report") != sources.pin(root / "recording/report.json"):
                raise ValueError("capture probe report hash mismatch")
            if raw_probe.get("capture_end_status") == "PASS" and (
                not valid_report(summary, report)
                or report.get("cleanup_confirmed") is not True
                or report.get("product_pass") is not False
                or report.get("failure", {}).get("type") != "CaptureEndProbeComplete"
            ):
                raise ValueError("capture probe report does not match scoped completion")
            probe = export(raw_probe, root.name, summary["source_commit"], SOURCE_SHA)
            summary["capture_end_probe"] = {
                key: value for key, value in probe.items() if key != "capture_end_status"
            }
            summary["capture_end_probe"]["observed_end_status"] = probe["capture_end_status"]
            summary["capture_end_status"] = "PASS" if controller_passed(summary, probe) else "FAIL"
        except (OSError, ValueError, TypeError):
            pass
        if summary["capture_end_status"] != "PASS" and summary["exit_code"] == 0:
            summary.update(
                status="FAILED",
                exit_code=1,
                failure_code="PRODUCT_FAILED",
                failure_stage="FULLGAME",
            )
    # A partial result can accompany forced termination. Preserve that terminal cause.
    if (
        invalid
        and summary["exit_code"] not in {3, 4}
        and summary["failure_code"] != "OWNERSHIP_FAILED"
    ):
        summary.update(
            status="ERROR", exit_code=5, failure_code=invalid, failure_stage=phase.upper()
        )
    return result


def finalize(root, summary, lock_path, acquired, started):
    summary.update(
        ended_at_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        loop_status="COMPLETE",
    )
    try:
        reports.publish(root / "export", summary)
    except Exception:
        if summary.get("profile") == "CaptureEnd":
            summary["capture_end_status"] = "FAIL"
        summary.update(
            status="ERROR",
            exit_code=5,
            failure_code="REPORT_PUBLICATION_FAILED",
            failure_stage="EXPORT",
            loop_status="REPORT_FAILED",
        )
        try:
            reports.atomic_json(root / "export/summary.json", summary)
        except Exception:
            pass  # Disk failure cannot be turned into a fabricated durable report.
    finally:
        if acquired:
            try:
                if sources.plain(lock_path).read_text(encoding="ascii") != root.name:
                    raise ValueError("lock identity changed")
                lock_path.unlink()
            except Exception:
                if summary.get("profile") == "CaptureEnd":
                    summary["capture_end_status"] = "FAIL"
                summary.update(
                    status="ERROR",
                    exit_code=5,
                    failure_code="LOCK_RELEASE_FAILED",
                    failure_stage="CLEANUP",
                    loop_status="INCOMPLETE",
                )
                try:
                    reports.publish(root / "export", summary)
                except Exception:
                    pass


def run(repo, commit, config_path, profile):
    if sys.platform != "win32":
        raise RuntimeError("Windows execution required; portable helpers can be tested separately")
    repo = sources.plain(repo)
    config = read_json(config_path)
    raw_runs = config.get("runs") if isinstance(config, dict) else None
    runs = sources.plain(absolute_machine_path(raw_runs))
    if not runs.is_dir():
        raise ValidationFailure("RUNS_DIRECTORY_REQUIRED", "CONFIG")
    root = runs / str(uuid.uuid4())
    root.mkdir()
    (root / "export").mkdir()
    (root / "export/events.jsonl").touch()
    (root / "local-logs").mkdir()
    started = time.monotonic()
    summary = {
        "contract": "blackbox-dev-validation/v1",
        "run_id": root.name,
        "source_commit": None,
        "transport_commit": commit,
        "wrapper_commit": commit,
        "input_sha256": SOURCE_SHA,
        "input_verified": False,
        "profile": profile,
        "source_lock": None,
        "environment_id": None,
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "elapsed_seconds": None,
        "status": "RUNNING",
        "stage": "SETUP",
        "exit_code": None,
        "failure_stage": None,
        "failure_code": None,
        "loop_status": "RUNNING",
        "product_status": "NOT_RUN",
        "browser_status": "NOT_RUN",
        "owned_cleanup": None,
        "deferred_cleanup": None,
        "model": "NOT_RUN",
        "referee": "NOT_RUN",
        "physical_camera": "NOT_RUN",
        "cloud": "NOT_RUN",
    }
    lock_path, acquired = runs / "validation.lock", False
    try:
        reports.publish(root / "export", summary)
        config = normalize_config(config)
        with lock_path.open("x", encoding="ascii") as stream:
            stream.write(root.name)
        acquired = True
        # Bootstrap is bounded in size. All expensive Git/hash/copy setup runs in its own Job.
        ownership = load_ownership(config["ownership"])
        reports.atomic_json(
            root / "context.json",
            {"config": config, "profile": profile, "repo": str(repo), "commit": commit},
        )
        result = execute_phase(root, config, summary, ownership, "setup")
        if summary["exit_code"] == 0:
            identity = result.get("identity", {}) if result else {}
            summary.update(validate_identity(identity))
            summary.update(status="RUNNING", stage="PREFLIGHT", exit_code=None)
            execute_phase(root, config, summary, ownership, "worker")
            if summary.get("last_observed", {}).get("stage") in {
                "preflight",
                "live_ready",
                "capture",
                "background",
                "capture_stopped",
                "ended",
                "encoded",
                "verified_media",
                "browser_ready",
                "finished",
            }:
                summary["input_verified"] = True
            if (
                profile in {"Fullgame", "FullgameComplete"}
                and (root / "recording/browser-ready.json").exists()
            ):
                summary["browser_status"] = "PENDING_MANUAL"
    except FileExistsError:
        summary.update(
            status="BUSY" if not acquired else "ERROR",
            exit_code=6 if not acquired else 5,
            failure_stage="SETUP",
            failure_code="RUN_BUSY" if not acquired else "PATH_COLLISION",
        )
    except ValidationFailure as exc:
        summary.update(
            status="PRECONDITION_FAILED" if exc.exit_code == 2 else "ERROR",
            exit_code=exc.exit_code,
            failure_stage=exc.stage,
            failure_code=exc.code,
        )
    except KeyboardInterrupt:
        summary.update(
            status="CANCELLED", exit_code=4, failure_stage="LAUNCHER", failure_code="CANCELLED"
        )
    except Exception:
        try:
            (root / "local-logs/launcher-error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        except OSError:
            pass
        summary.update(
            status="ERROR", exit_code=5, failure_stage="LAUNCHER", failure_code="LAUNCHER_FAILED"
        )
    finally:
        finalize(root, summary, lock_path, acquired, started)
    print(
        json.dumps(
            {
                "run_id": root.name,
                "status": summary["status"],
                "exit_code": summary["exit_code"],
                "failure_code": summary["failure_code"],
                "export": str(root / "export"),
            }
        ),
        flush=True,
    )
    return summary["exit_code"]


def child_main(root, phase):
    if (
        phase not in PHASES
        or sys.stdin.buffer.readline(200) != ("BLACKBOX-RELEASE " + root.name + "\n").encode()
    ):
        raise ValueError("owned release required")
    code, failure_code, stage, identity = 0, None, phase.upper(), None
    try:
        context = read_json(root / "context.json")
        if phase == "setup":
            identity = setup_worker(root, context)
        else:
            code = worker(root, context)
            stage = "PREFLIGHT" if context["profile"] == "Preflight" else "FULLGAME"
            if code:
                failure_code = "PRODUCT_FAILED"
                path = root / "recording/report.json"
                report = read_product_report(root, context["profile"]) if path.exists() else None
                setup = read_json(root / "setup-result.json")
                if valid_report(setup.get("identity", {}), report):
                    failure_value = report.get("failure")
                    failure = failure_value.get("type") if isinstance(failure_value, dict) else None
                    if failure == "TimeoutError":
                        code, failure_code = 3, "PRODUCT_TIMEOUT"
                    elif failure == "KeyboardInterrupt":
                        code, failure_code = 4, "CANCELLED"
    except ValidationFailure as exc:
        code, failure_code, stage = exc.exit_code, exc.code, exc.stage
    except subprocess.TimeoutExpired:
        code, failure_code = 3, "GIT_TIMEOUT"
    except subprocess.CalledProcessError:
        code, failure_code, stage = 2, "SOURCE_GIT_FAILED", "SOURCE"
    except KeyboardInterrupt:
        code, failure_code = 4, "CANCELLED"
    except Exception:
        (root / f"local-logs/{phase}-error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        code, failure_code = (
            (2, "SETUP_VALIDATION_FAILED") if phase == "setup" else (2, "PREFLIGHT_FAILED")
        )
    reports.atomic_json(
        root / f"{phase}-result.json",
        {
            "contract": "blackbox-dev-child/v1",
            "run_id": root.name,
            "phase": phase,
            "exit_code": code,
            "failure_code": failure_code,
            "failure_stage": stage,
            "identity": identity,
        },
    )
    return code


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        return child_main(Path(sys.argv[3]), sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("Preflight", "Fullgame", "FullgameComplete", "CaptureEnd"),
        required=True,
    )
    args = parser.parse_args()
    try:
        return run(args.repo, args.commit, args.config, args.profile)
    except (OSError, ValueError) as exc:
        # Invalid bootstrap config cannot name a trustworthy export directory.
        # Still return a bounded finite diagnostic instead of a raw traceback.
        print(
            json.dumps(
                {
                    "status": "PRECONDITION_FAILED",
                    "exit_code": 2,
                    "failure_code": exc.code
                    if isinstance(exc, ValidationFailure)
                    else "SETUP_VALIDATION_FAILED",
                    "failure_stage": "CONFIG",
                    "export": None,
                }
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
