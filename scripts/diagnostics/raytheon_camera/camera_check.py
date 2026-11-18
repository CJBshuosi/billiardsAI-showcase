"""Standalone, non-authoritative Windows camera diagnostics (no project imports).

OpenCV integer indices are OPERATOR-SELECTED, never trusted PnP identities.
Camera access lives in killable child processes. No business store is opened
through a project repository or migration; no camera image controls are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

BASELINE = "9a144955377b289ce47de71769d1305dd8585de4"
PACKET_VERSION = "20260905-v4"
CONTROLS = (
    "BRIGHTNESS",
    "CONTRAST",
    "SATURATION",
    "HUE",
    "GAIN",
    "EXPOSURE",
    "AUTO_EXPOSURE",
    "AUTO_WB",
    "WB_TEMPERATURE",
    "FOCUS",
    "AUTOFOCUS",
    "GAMMA",
    "SHARPNESS",
    "BACKLIGHT",
    "ZOOM",
)
SOURCE_FILES = (
    "scripts/algorithm/serve_field_route_shell.py",
    "scripts/algorithm/run_mainline1_usb_live.py",
    "scripts/algorithm/windows/serve-field-route-shell.ps1",
    "scripts/algorithm/windows/install-field-route-service.ps1",
    "src/billiards_ai/lab/field_assistance/server.py",
    "src/billiards_ai/lab/field_assistance/replay.py",
    "src/billiards_ai/lab/mainline1_usb_capture.py",
    "src/billiards_ai/lab/mainline1_frame_source.py",
    "src/billiards_ai/lab/mainline1_live_runtime.py",
    "src/billiards_ai/lab/mainline1_calibration_gate.py",
    "src/billiards_ai/edge/route_assistance/persistence.py",
    "release-manifest.json",
)


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cadence(timestamps, requested_fps):
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
    finite = all(math.isfinite(x) for x in timestamps)
    valid = bool(intervals) and finite and all(x >= 0 for x in intervals)
    elapsed = (timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0) if finite else None
    equal = sum(x == 0 for x in intervals)
    rate = len(intervals) / elapsed if valid and elapsed > 0 else None
    return {
        "frames": len(timestamps),
        "elapsed_seconds": elapsed,
        "measured_fps": rate,
        "monotonic": valid,
        "equal_timestamps": equal,
        "timestamp_reversals": sum(x < 0 for x in intervals),
        "maximum_gap_ms": max(intervals) * 1000 if intervals and finite else None,
        "gaps_over_1_5_frame_period": sum(x > 1.5 / requested_fps for x in intervals),
        "within_5_percent": rate is not None
        and equal == 0
        and abs(rate / requested_fps - 1) <= 0.05,
        "timestamp_source": "host_perf_counter_after_read_not_sensor_timestamp",
    }


def run_bounded(
    command,
    log_path,
    *,
    timeout,
    progress_path=None,
    heartbeat_path=None,
    stall_timeout=30,
):
    """Kill only the child created here, including its blocked capture thread."""
    started = time.monotonic()
    last_progress = started
    last_announcement = started
    heartbeat_version = None
    reason = None
    with Path(log_path).open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            announced = False
            while child.poll() is None:
                now = time.monotonic()
                if heartbeat_path and Path(heartbeat_path).exists():
                    version = Path(heartbeat_path).stat().st_mtime_ns
                    if version != heartbeat_version:
                        last_progress = now
                        heartbeat_version = version
                if heartbeat_path and now - last_progress >= stall_timeout:
                    reason = "WORKER_PROGRESS_STALLED"
                    raise subprocess.TimeoutExpired(command, timeout)
                if now - started >= timeout:
                    reason = "WORKER_TOTAL_TIMEOUT"
                    raise subprocess.TimeoutExpired(command, timeout)
                if now - last_announcement >= 30:
                    print(
                        f"{Path(log_path).stem}: running {int(now - started)} seconds.", flush=True
                    )
                    last_announcement = now
                if progress_path and not announced and Path(progress_path).exists():
                    marker = json.loads(Path(progress_path).read_text(encoding="utf-8"))
                    duration = marker.get("duration_seconds", "unknown")
                    print(
                        f"\aRECORDING NOW: strike once. Recording lasts {duration} seconds.",
                        flush=True,
                    )
                    announced = True
                try:
                    child.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    continue
            code = child.returncode
            status = "COMPLETED" if code == 0 else "ERROR"
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait(timeout=10)
            status = "TIMEOUT"
        except BaseException:
            child.kill()
            child.wait(timeout=10)
            raise
    return {
        "status": status,
        "reason": reason,
        "returncode": code,
        "elapsed_seconds": time.monotonic() - started,
    }


def inspect_state_db(path):
    path = Path(path)
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    tables = (
        "field_sessions",
        "field_shots",
        "field_applied_commands",
        "field_sync_jobs",
        "field_shot_ownership",
        "field_turn_confirmations",
    )
    try:
        # mode=ro forbids database creation and writes. Do not use FieldRecordStore.
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            existing = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            result = {
                "status": "READ_ONLY",
                "path": str(path),
                "schema_version": db.execute("PRAGMA user_version").fetchone()[0],
                "counts": {
                    name: db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    for name in tables
                    if name in existing
                },
            }
            if "field_sessions" in existing:
                rows = db.execute(
                    "SELECT session_id,status,created_at_utc FROM field_sessions "
                    "ORDER BY created_at_utc DESC LIMIT 5"
                ).fetchall()
                result["recent_sessions"] = [
                    dict(zip(("session_id", "status", "created_at_utc"), row, strict=True))
                    for row in rows
                ]
            return result
    except sqlite3.Error as error:
        return {"status": "UNAVAILABLE", "path": str(path), "reason": str(error)}


def pack_result(directory, summary):
    directory = Path(directory)
    target = directory.with_name("SEND_BACK-" + directory.name + ".zip")
    if target.exists():
        raise FileExistsError(target)
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("result contains a symlink")
        if path.is_file() and path.name != "summary.json":
            inventory[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(
        directory / "summary.json",
        {
            **summary,
            "acceptance": "NOT_EVALUATED",
            "inventory": inventory,
            "source_kind": "camera_diagnostic_only",
            "trusted_pnp_capture_binding": False,
            "production_ready": False,
        },
    )
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, directory.name + "/" + path.relative_to(directory).as_posix())
    return target


def read_controls(cv, capture):
    result = {}
    for name in CONTROLS:
        prop = getattr(cv, "CAP_PROP_" + name, None)
        try:
            value = float(capture.get(prop)) if prop is not None else None
            result[name.lower()] = value if value is not None and math.isfinite(value) else None
        except Exception:
            result[name.lower()] = None
    # OpenCV get() returns zero for some unsupported properties: never infer support.
    return result


def read_mode(cv, capture):
    result = {}
    for name in ("FRAME_WIDTH", "FRAME_HEIGHT", "FPS", "FOURCC"):
        value = float(capture.get(getattr(cv, "CAP_PROP_" + name)))
        result[name.lower()] = value if math.isfinite(value) else None
    fourcc = int(result["fourcc"] or 0)
    result["fourcc_text"] = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
    return result


def capture_run(
    cv,
    output,
    *,
    index,
    width,
    height,
    fps=60,
    seconds=10,
    sample_seconds=5,
    scan=False,
    backend="DSHOW",
    require_mode=False,
    expected_controls=None,
    capture_factory=None,
    maximum_video_bytes=512 * 1024 * 1024,
):
    """Record a diagnostic sample. No claim of trusted hardware/source identity.

    Cadence-only sampling precedes encoding. During recording, acquisition never
    waits for the writer; dropped archive frames and their IDs are recorded.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTING",
        "runtime_index": index,
        "backend_requested": backend,
        "packet_version": PACKET_VERSION,
        "clock_info": {
            "sample_clock": "perf_counter_ns",
            "implementation": time.get_clock_info("perf_counter").implementation,
            "resolution_seconds": time.get_clock_info("perf_counter").resolution,
            "legacy_monotonic_resolution_seconds": time.get_clock_info("monotonic").resolution,
        },
        "requested": {"width": width, "height": height, "fps": fps, "format": "MJPG"},
        "trusted_pnp_capture_binding": False,
        "image_control_writes": [],
        "control_support_and_lock": "NOT_VERIFIED",
        "expected_controls_readback": expected_controls,
        "media_encoding": "OpenCV_decoded_Bgr_to_MJPG_AVI_no_resize_no_overlays_no_audio",
        "media_timing": "AVI_nominal_fps_use_host_perf_counter_timestamps_for_actual_read_timing",
        "maximum_video_bytes": maximum_video_bytes,
        "sample_requested_seconds": sample_seconds,
        "record_requested_seconds": seconds,
        "sample_completed": False,
        "cadence_checkpoint_interval_seconds": 5,
        "cadence_checkpoint_io_included_in_measurement": True,
    }
    write_json(output / "report.json", report)
    capture = None
    writer_thread = None
    stop = threading.Event()
    frames = queue.Queue(maxsize=8)
    archive = {"saved_frames": 0, "queue_drops": 0, "writer_error": None}
    try:
        capture = (
            capture_factory(index)
            if capture_factory
            else cv.VideoCapture(index, getattr(cv, "CAP_" + backend))
        )
        if not capture.isOpened():
            raise RuntimeError("CAMERA_NOT_OPENED")
        report["controls_before_mode"] = read_controls(cv, capture)
        report["mode_before"] = read_mode(cv, capture)
        try:
            report["backend_observed"] = capture.getBackendName()
        except (AttributeError, RuntimeError):
            report["backend_observed"] = "UNAVAILABLE"
        if report["backend_observed"] not in (backend, "UNAVAILABLE"):
            raise RuntimeError("BACKEND_MISMATCH")
        if not scan:
            # FPS/geometry changes can rebuild the graph; request FOURCC last.
            report["negotiation_steps"] = []
            for name, value in (
                ("FRAME_WIDTH", width),
                ("FRAME_HEIGHT", height),
                ("FPS", fps),
                ("FOURCC", cv.VideoWriter_fourcc(*"MJPG")),
            ):
                accepted = bool(capture.set(getattr(cv, "CAP_PROP_" + name), value))
                report["negotiation_steps"].append(
                    {
                        "property": name,
                        "requested": value,
                        "setter_accepted": accepted,
                        "readback": read_mode(cv, capture),
                    }
                )
                write_json(output / "report.json", report)
        report["observed"] = read_mode(cv, capture)
        report["controls_after_mode"] = read_controls(cv, capture)
        write_json(output / "report.json", report)
        if expected_controls and any(
            report["controls_after_mode"].get(k) != v for k, v in expected_controls.items()
        ):
            raise RuntimeError("EXPECTED_CONTROLS_MISMATCH_BEFORE_CAPTURE")
        for _ in range(15):
            ok, first = capture.read()
            if not ok or first is None:
                raise RuntimeError("FRAME_READ_FAILED_DURING_WARMUP")
        actual_height, actual_width = first.shape[:2]
        report["decoded_geometry"] = [actual_width, actual_height]
        if not cv.imwrite(str(output / "baseline.png"), first):
            raise RuntimeError("BASELINE_WRITE_FAILED")
        if scan:
            report["status"] = "PREVIEW_AVAILABLE_UNVERIFIED_IDENTITY"
            return report
        if (actual_width, actual_height) != (width, height):
            raise RuntimeError("DECODED_GEOMETRY_MISMATCH")
        report["mode_matches_request"] = (
            report["observed"]["frame_width"] == width
            and report["observed"]["frame_height"] == height
            and report["observed"]["fps"] is not None
            and abs(report["observed"]["fps"] - fps) <= 0.5
            and report["observed"]["fourcc_text"] == "MJPG"
        )
        if require_mode and not report["mode_matches_request"]:
            raise RuntimeError("REQUESTED_MODE_NOT_CONFIRMED")
        pure_ns = []
        started = time.perf_counter_ns()
        checkpoint_ns = started
        saved = 0
        write_json(output / "progress.json", {"phase": "cadence", "frames": 0})
        with (output / "cadence-times.csv").open("w", newline="", encoding="utf-8") as stream:
            rows = csv.writer(stream)
            rows.writerow(["capture_index", "host_perf_counter_ns"])
            stream.flush()
            try:
                while time.perf_counter_ns() - started < sample_seconds * 1e9:
                    ok, image = capture.read()
                    if not ok or image is None:
                        raise RuntimeError("FRAME_READ_FAILED_IN_CADENCE_SAMPLE")
                    if image.shape[:2] != (height, width):
                        raise RuntimeError("GEOMETRY_CHANGED")
                    now_ns = time.perf_counter_ns()
                    pure_ns.append(now_ns)
                    if now_ns - checkpoint_ns >= 5e9:
                        rows.writerows(enumerate(pure_ns[saved:], start=saved))
                        stream.flush()
                        saved = len(pure_ns)
                        write_json(
                            output / "progress.json",
                            {
                                "phase": "cadence",
                                "frames": saved,
                                "elapsed_seconds": (now_ns - started) / 1e9,
                            },
                        )
                        checkpoint_ns = now_ns
                report["sample_completed"] = True
            finally:
                rows.writerows(enumerate(pure_ns[saved:], start=saved))
                report["cadence_without_recording"] = cadence([ns / 1e9 for ns in pure_ns], fps)
        write_json(output / "report.json", report)
        if seconds == 0:
            report["controls_after_capture"] = read_controls(cv, capture)
            report["observed_after_capture"] = read_mode(cv, capture)
            report["status"] = "READBACK_COLLECTED_REVIEW_REQUIRED"
            return report

        def write_frames():
            writer = None
            try:
                writer = cv.VideoWriter(
                    str(output / "camera.avi"), cv.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("VIDEO_WRITER_NOT_OPENED")
                with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
                    rows = csv.writer(stream)
                    rows.writerow(["saved_index", "capture_index", "host_perf_counter_seconds"])
                    while not stop.is_set() or not frames.empty():
                        try:
                            frame_index, timestamp, image = frames.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        writer.write(image)
                        rows.writerow([archive["saved_frames"], frame_index, f"{timestamp:.9f}"])
                        archive["saved_frames"] += 1
                        if (output / "camera.avi").stat().st_size > maximum_video_bytes:
                            raise RuntimeError("VIDEO_SIZE_LIMIT_EXCEEDED")
            except Exception as error:
                archive["writer_error"] = str(error)
            finally:
                if writer is not None:
                    writer.release()
                video_path = output / "camera.avi"
                if video_path.exists() and video_path.stat().st_size > maximum_video_bytes:
                    archive["writer_error"] = "VIDEO_SIZE_LIMIT_EXCEEDED"

        writer_thread = threading.Thread(target=write_frames, daemon=True)
        writer_thread.start()
        write_json(
            output / "recording-started.json",
            {"host_perf_counter_ns": time.perf_counter_ns(), "duration_seconds": seconds},
        )
        print("RECORDING NOW - strike the cue ball; keep hands clear after the shot.", flush=True)
        if sys.platform == "win32":
            import winsound

            winsound.MessageBeep()
        started = time.perf_counter_ns()
        checkpoint_ns = started
        write_json(output / "progress.json", {"phase": "recording", "frames": 0})
        times = []
        with (output / "capture-times.csv").open("w", newline="", encoding="utf-8") as stream:
            rows = csv.writer(stream)
            rows.writerow(["capture_index", "host_perf_counter_seconds", "queued_for_video"])
            while time.perf_counter_ns() - started < seconds * 1e9:
                if archive["writer_error"]:
                    raise RuntimeError(archive["writer_error"])
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError("FRAME_READ_FAILED_DURING_RECORDING")
                if image.shape[:2] != (height, width):
                    raise RuntimeError("GEOMETRY_CHANGED")
                stamp = time.perf_counter_ns() / 1e9
                index_in_run = len(times)
                times.append(stamp)
                queued = True
                try:
                    frames.put_nowait((index_in_run, stamp, image.copy()))
                except queue.Full:
                    archive["queue_drops"] += 1
                    queued = False
                rows.writerow([index_in_run, f"{stamp:.9f}", queued])
                now_ns = time.perf_counter_ns()
                if now_ns - checkpoint_ns >= 5e9:
                    stream.flush()
                    write_json(
                        output / "progress.json",
                        {
                            "phase": "recording",
                            "frames": len(times),
                            "elapsed_seconds": (now_ns - started) / 1e9,
                        },
                    )
                    checkpoint_ns = now_ns
        write_json(output / "progress.json", {"phase": "recording_finished", "frames": len(times)})
        stop.set()
        writer_thread.join(timeout=10)
        if writer_thread.is_alive():
            raise RuntimeError("VIDEO_WRITER_DID_NOT_STOP")
        report["cadence_with_recording"] = cadence(times, fps)
        report["controls_after_capture"] = read_controls(cv, capture)
        report["observed_after_capture"] = read_mode(cv, capture)
        report["controls_changed_during_run"] = (
            report["controls_after_mode"] != report["controls_after_capture"]
        )
        if archive["writer_error"]:
            raise RuntimeError(archive["writer_error"])
        report["status"] = "SAMPLE_COLLECTED_REVIEW_REQUIRED"
    except Exception as error:
        report["status"] = "ERROR"
        report["reason"] = str(error)
    finally:
        stop.set()
        if writer_thread is not None:
            writer_thread.join(timeout=2)
        report["archive"] = archive
        if "controls_after_capture" in report:
            report["controls_changed_during_run"] = (
                report["controls_after_mode"] != report["controls_after_capture"]
            )
            if expected_controls and any(
                report["controls_after_capture"].get(k) != v for k, v in expected_controls.items()
            ):
                report["status"] = "ERROR"
                report["reason"] = "EXPECTED_CONTROLS_MISMATCH_AFTER_CAPTURE"
        # Persist before release: if a driver hangs, the parent still has evidence.
        write_json(output / "report.json", report)
        if capture is not None:
            capture.release()
    return report


def safe_command(command):
    """Whitelist path flags only; do not export raw command lines or service tokens."""
    command = command or ""
    tokens = re.findall(r'"([^"\r\n]*)"|([^\s"]+)', command)
    tokens = [quoted if quoted else plain for quoted, plain in tokens]
    allowed = {
        "artifact",
        "state-db",
        "video",
        "mainline1-artifact",
        "capture-setup",
        "calibration-root",
        "detector-training-run",
        "identity-training-run",
        "dinov2-assets",
        "config",
        "output-root",
        "spool-root",
        "projectroot",
        "runtimeroot",
        "statedb",
        "pythonpath",
    }
    paths = {}
    for index, token in enumerate(tokens[:-1]):
        key = token.lstrip("-").casefold()
        value = tokens[index + 1]
        if token.startswith("-") and key in allowed and re.match(r"^[A-Za-z]:[\\/]", value):
            paths[key] = value
    return {
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "safe_arguments": paths,
        "entrypoints": [
            name
            for name in (
                "serve_field_route_shell.py",
                "run_mainline1_usb_live.py",
                "serve-field-route-shell.ps1",
                "start-field-route-kiosk.ps1",
            )
            if name.casefold() in command.casefold()
        ],
    }


def collect_windows_audit(project_root, runtime_root):
    # Raw command lines remain in memory only, and are whitelisted before persistence.
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$errors = @()
$pnp = @(); $drivers = @(); $processes = @(); $tasks = @(); $listeners = @()
try {
    $pnp = @(Get-CimInstance Win32_PnPEntity |
        Where-Object { $_.PNPClass -in @('Camera','Image') } |
        Select-Object Name,PNPDeviceID,Status,Service)
} catch { $errors += 'PNP_QUERY_FAILED' }
try {
    $drivers = @(Get-CimInstance Win32_PnPSignedDriver |
        Where-Object { $_.DeviceClass -in @('CAMERA','IMAGE') } |
        Select-Object DeviceName,DeviceID,DriverVersion,DriverDate,DriverProviderName)
} catch { $errors += 'DRIVER_QUERY_FAILED' }
try {
    $processes = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(amcap|WindowsCamera|obs64)\.exe$' -or
        $_.CommandLine -match
            'serve_field_route_shell\.py|run_mainline1_usb_live\.py|serve-field-route-shell\.ps1'
    } | Select-Object Name,ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine)
} catch { $errors += 'PROCESS_QUERY_FAILED' }
try {
    $tasks = @(Get-ScheduledTask | Where-Object {
        $_.TaskName -in @('billiardsAI-field-service','billiardsAI-field-kiosk')
    } | ForEach-Object {
        $task = $_
        $info = $task | Get-ScheduledTaskInfo
        [pscustomobject]@{
            name=$task.TaskName; state=[string]$task.State; last_result=$info.LastTaskResult
            actions=@($task.Actions | Select-Object Execute,Arguments)
        }
    })
} catch { $errors += 'TASK_QUERY_FAILED' }
try {
    $listeners = @(Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction Stop |
        Select-Object LocalAddress,LocalPort,OwningProcess)
} catch { $errors += 'PORT_8787_NOT_LISTENING_OR_UNAVAILABLE' }
[pscustomobject]@{
    inventory_pid=$PID
    pnp=$pnp; drivers=$drivers; processes=$processes
    tasks=$tasks; listeners=$listeners; errors=$errors
} | ConvertTo-Json -Depth 8 -Compress
"""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        timeout=35,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Windows inventory failed; raw command lines were not exported")
    raw = json.loads(completed.stdout.lstrip("\ufeff"))
    current_path = Path(runtime_root) / "current.json"
    current = {}
    if current_path.is_file():
        values = json.loads(current_path.read_text(encoding="utf-8-sig"))
        current = {
            key: values.get(key)
            for key in (
                "current_release",
                "previous_release",
                "artifact",
                "python_path",
            )
        }
    processes = []
    blockers = []
    for process in raw.get("processes", []):
        if process.get("ProcessId") == raw.get("inventory_pid"):
            continue
        command = process.get("CommandLine") or ""
        safe = {
            key: process.get(key)
            for key in (
                "Name",
                "ProcessId",
                "ParentProcessId",
                "CreationDate",
                "ExecutablePath",
            )
        }
        safe.update(safe_command(command))
        processes.append(safe)
        if (
            process.get("Name", "").casefold() in ("amcap.exe", "windowscamera.exe", "obs64.exe")
            or "run_mainline1_usb_live.py" in command
        ):
            blockers.append({"name": process.get("Name"), "pid": process.get("ProcessId")})
    tasks = []
    for task in raw.get("tasks", []):
        tasks.append(
            {
                "name": task.get("name"),
                "state": task.get("state"),
                "last_result": task.get("last_result"),
                "actions": [
                    {"execute": action.get("Execute"), **safe_command(action.get("Arguments"))}
                    for action in task.get("actions", [])
                ],
            }
        )
    result = {key: raw.get(key) for key in ("pnp", "drivers", "listeners", "errors")}
    result.update(
        {
            "project_root": str(project_root),
            "runtime_root": str(runtime_root),
            "current": current,
            "processes": processes,
            "tasks": tasks,
            "camera_apps_running": blockers,
            "default_state_db": str(Path(runtime_root) / "field-state.sqlite3"),
            "usb_link_speed": "NOT_MEASURED_do_not_infer_from_4K_preview",
            "trusted_pnp_capture_binding": False,
        }
    )
    try:
        git = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["project_head"] = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["project_head"] = None
    return result


def inspect_installation(audit):
    result = {"roots": {}, "state_databases": []}
    roots = {
        "project": audit.get("project_root"),
        "installed_release": audit.get("current", {}).get("current_release"),
    }
    for label, root in roots.items():
        if not root:
            continue
        root = Path(root)
        hashes = {}
        for name in SOURCE_FILES:
            path = root / name
            hashes[name] = sha256(path) if path.is_file() else None
        result["roots"][label] = {"path": str(root), "sha256": hashes}
    paths = {audit.get("default_state_db")}
    for process in audit.get("processes", []):
        paths.add(process.get("safe_arguments", {}).get("state-db"))
    for path in sorted(p for p in paths if p):
        result["state_databases"].append(inspect_state_db(path))
    artifact = audit.get("current", {}).get("artifact")
    if artifact:
        root = Path(artifact)
        result["field_artifact"] = {"path": str(root), "files": {}}
        for name in ("manifest.json", "summary.json"):
            path = root / name
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
                keys = (
                    "artifact_id",
                    "source_mainline1_artifact_id",
                    "kind",
                    "pipeline_version",
                    "schema_version",
                    "shot_count",
                    "status",
                    "source_kind",
                )
                result["field_artifact"]["files"][name] = {
                    "sha256": sha256(path),
                    "fields": {k: raw[k] for k in keys if k in raw},
                }
    return result


def run_camera_child(
    output,
    *,
    index,
    width=1920,
    height=1080,
    scan=False,
    record_seconds=10,
    backend="DSHOW",
    sample_seconds=5,
    require_mode=False,
    expected_controls=None,
):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--index",
        str(index),
        "--width",
        str(width),
        "--height",
        str(height),
        "--record-seconds",
        str(record_seconds),
        "--backend",
        backend,
        "--sample-seconds",
        str(sample_seconds),
    ]
    if scan:
        command.append("--scan")
    if require_mode:
        command.append("--require-mode")
    if expected_controls is not None:
        command.extend(["--expected-controls", json.dumps(expected_controls)])
    result = run_bounded(
        command,
        output.with_suffix(".log"),
        timeout=20 if scan else max(50, sample_seconds + record_seconds + 50),
        progress_path=output / "recording-started.json",
        heartbeat_path=None if scan else output / "progress.json",
        stall_timeout=30,
    )
    if (output / "report.json").is_file():
        result["report"] = json.loads((output / "report.json").read_text(encoding="utf-8"))
    write_json(output.with_suffix(".process.json"), result)
    return result


def select_camera(root, cv, summary, backend):
    """Operator confirmation only; never treat the index as a durable identity."""
    summary.setdefault("selection", {})
    summary.setdefault("scans", [])
    print(f"{backend}: scanning indices 0..3 (up to 20 sec each).", flush=True)
    candidates, tiles = [], []
    for index in range(4):
        location = root / f"{backend.lower()}-preview-{index}"
        run = run_camera_child(location, index=index, backend=backend, scan=True)
        summary["scans"].append({"backend": backend, "runtime_index": index, **run})
        if run["status"] == "TIMEOUT":
            raise RuntimeError("Camera scan timed out; return this result before retrying")
        if run["status"] == "COMPLETED" and (location / "baseline.png").is_file():
            candidates.append(index)
            frame = cv.resize(cv.imread(str(location / "baseline.png")), (480, 270))
            cv.putText(
                frame,
                f"{backend} CAMERA INDEX {index}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            tiles.append(frame)
    if not candidates:
        summary["selection"][backend] = {"status": "NO_PREVIEW_AVAILABLE"}
        return None
    montage = root / f"choose-camera-{backend.lower()}.jpg"
    if not cv.imwrite(str(montage), cv.vconcat(tiles)):
        raise RuntimeError("Could not write camera selection image")
    print(f"Open {montage} and confirm the NEW overhead table camera.", flush=True)
    try:
        os.startfile(montage)
    except (OSError, AttributeError):
        pass
    choice = input(f"{backend}: type camera index {candidates}, or q to stop: ").strip()
    if not choice.isdecimal() or int(choice) not in candidates:
        raise RuntimeError("Operator did not select a previewed camera")
    index = int(choice)
    summary["selection"][backend] = {
        "runtime_index": index,
        "basis": "operator_preview",
        "durable_identity_verified": False,
    }
    return index


def collect_exposure_comparison(root, cv, summary, project_root=None, runtime_root=None):
    """Operator changes exposure; every camera control remains read-only here."""
    summary["workflow"] = "dshow_1080p60_manual_exposure_comparison"
    reference_controls = None
    for exposure in (-7, -8):
        stage = root / f"exposure-{abs(exposure)}"
        stage.mkdir()
        input(
            f"Set MANUAL exposure {exposure} in AMCap; keep other controls/lights unchanged. "
            "Save screenshots, CLOSE AMCap and all camera apps, then ENTER: "
        )
        audit = collect_windows_audit(project_root, runtime_root)
        write_json(stage / "windows-audit.json", audit)
        if "PROCESS_QUERY_FAILED" in (audit.get("errors") or []) or audit.get(
            "camera_apps_running"
        ):
            raise RuntimeError(
                "Camera apps still running or inventory unavailable; capture skipped"
            )
        stage_summary = {"runs": []}
        summary.setdefault("exposure_stages", {})[str(exposure)] = stage_summary
        index = select_camera(stage, cv, stage_summary, "DSHOW")
        if index is None:
            raise RuntimeError("No DirectShow preview available")
        expected = dict(reference_controls or {"autofocus": 2, "auto_wb": 0})
        expected["exposure"] = exposure
        for name, sample_seconds, record_seconds in (("preflight", 2, 0), ("shot", 10, 30)):
            if name == "shot":
                print(f"Full-table still: {stage / 'preflight' / 'baseline.png'}", flush=True)
                input(
                    "Confirm full table. Prepare SAME fast shot; ENTER for 10s preparation "
                    "after camera setup, then 30s recording. WAIT for RECORDING NOW to strike: "
                )
            result = run_camera_child(
                stage / name,
                index=index,
                backend="DSHOW",
                require_mode=True,
                sample_seconds=sample_seconds,
                record_seconds=record_seconds,
                expected_controls=expected,
            )
            summary["runs"].append(
                {"mode": f"{stage.name}/{name}", "expected_exposure": exposure, **result}
            )
            stage_summary["runs"].append(summary["runs"][-1])
            if result["status"] != "COMPLETED":
                raise RuntimeError(f"{stage.name}/{name}: capture failed; subsequent opens skipped")
            controls = result.get("report", {}).get("controls_after_capture", {})
            if any(controls.get(k) != v for k, v in expected.items()):
                raise RuntimeError(
                    "EXPECTED_CONTROLS_MISMATCH; return evidence before changing settings"
                )
            if reference_controls is None:
                reference_controls = dict(controls)
            expected = dict(reference_controls, exposure=exposure)
            print(f"{stage.name}/{name}: collected for review.", flush=True)
    summary["status"] = "COLLECTED_REVIEW_REQUIRED"


def collect_motion_stability(root, cv, summary):
    index = select_camera(root, cv, summary, "DSHOW")
    if index is None:
        raise RuntimeError("No DirectShow preview available")
    summary["workflow"] = "dshow_1080p60_motion_then_single_handle_30min"
    initial_controls = None
    for name, sample_seconds, record_seconds in (
        ("preflight", 5, 0),
        ("motion-1", 2, 10),
        ("motion-2", 2, 10),
        ("motion-3", 2, 10),
        ("stability-30min", 1800, 0),
        ("reopen", 5, 0),
    ):
        if name.startswith("motion-"):
            input(
                f"{name}: prepare a fast shot; ENTER to arm, WAIT for RECORDING NOW, then strike: "
            )
        elif name == "stability-30min":
            input("ENTER for 30 minutes continuous capture (NO VIDEO). Keep USB/power connected: ")
        print(f"{name}: sampling {sample_seconds}s, recording {record_seconds}s.", flush=True)
        result = run_camera_child(
            root / name,
            index=index,
            backend="DSHOW",
            require_mode=True,
            sample_seconds=sample_seconds,
            record_seconds=record_seconds,
        )
        summary["runs"].append({"backend": "DSHOW", "mode": name, **result})
        if result["status"] != "COMPLETED":
            raise RuntimeError(f"{name}: capture failed; subsequent opens skipped")
        report = result.get("report", {})
        controls = report.get("controls_after_capture", {})
        changed_since_preflight = initial_controls is not None and controls != initial_controls
        summary["runs"][-1]["controls_changed_since_preflight"] = changed_since_preflight
        if initial_controls is None:
            initial_controls = controls
        if (
            any(
                controls.get(k) != v
                for k, v in {
                    "exposure": -7,
                    "autofocus": 2,
                    "auto_wb": 0,
                }.items()
            )
            or report.get("controls_changed_during_run")
            or changed_since_preflight
        ):
            raise RuntimeError(
                f"{name}: CONTROL_READBACK_MISMATCH; return evidence before changing settings"
            )
        rate = report.get("cadence_without_recording", {}).get("measured_fps")
        print(f"{name}: collected; cadence FPS={rate}. NOT acceptance PASS.", flush=True)
    summary["status"] = "COLLECTED_REVIEW_REQUIRED"


def collect_comparison(root, cv, summary):
    """Backend indices have separate namespaces: always ask for both previews."""
    complete = True
    for backend in ("MSMF", "DSHOW"):
        index = select_camera(root, cv, summary, backend)
        if index is None:
            complete = False
            continue
        print("No shot needed. Keep the table/lights fixed and all camera apps closed.", flush=True)
        for name, width, height, duration in (
            ("1080p60", 1920, 1080, 15),
            ("4k60", 3840, 2160, 15),
            ("1080p60-reopen", 1920, 1080, 5),
        ):
            print(f"{backend} {name}: {duration}s READ-ONLY IMAGE CONTROLS / NO VIDEO.", flush=True)
            result = run_camera_child(
                root / f"{backend.lower()}-{name}",
                index=index,
                backend=backend,
                width=width,
                height=height,
                record_seconds=0,
                sample_seconds=duration,
            )
            summary["runs"].append({"backend": backend, "mode": name, **result})
            if result["status"] == "TIMEOUT":
                raise RuntimeError("Camera capture timed out; further opens skipped for safety")
            if result["status"] != "COMPLETED":
                complete = False
            report = result.get("report", {})
            delivered = report.get("cadence_without_recording", {}).get("measured_fps")
            fmt = report.get("observed", {}).get("fourcc_text")
            print(
                f"{backend} {name}: {result['status']}; measured={delivered}; readback={fmt}",
                flush=True,
            )
    summary["status"] = "COLLECTED_REVIEW_REQUIRED" if complete else "PARTIAL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--require-mode", action="store_true")
    parser.add_argument(
        "--workflow",
        choices=("exposure-comparison", "motion-stability", "comparison"),
        default="exposure-comparison",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--record-seconds", type=int, choices=range(0, 31), default=10)
    parser.add_argument("--expected-controls", type=json.loads)
    parser.add_argument("--backend", choices=("MSMF", "DSHOW"), default="DSHOW")
    parser.add_argument("--sample-seconds", type=int, choices=range(1, 1801), default=5)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        import cv2

        result = capture_run(
            cv2,
            args.output,
            index=args.index,
            width=args.width,
            height=args.height,
            scan=args.scan,
            seconds=args.record_seconds,
            backend=args.backend,
            sample_seconds=args.sample_seconds,
            require_mode=args.require_mode,
            expected_controls=args.expected_controls,
        )
        return 0 if result["status"] != "ERROR" else 2
    if sys.platform != "win32":
        parser.error("The interactive capture workflow is Windows-only")
    root = args.output
    summary = {
        "baseline_commit": BASELINE,
        "packet_version": PACKET_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "status": "PARTIAL",
        "runs": [],
    }
    try:
        audit = collect_windows_audit(args.project_root, args.runtime_root)
        write_json(root / "windows-audit.json", audit)
        write_json(root / "installation.json", inspect_installation(audit))
        if "PROCESS_QUERY_FAILED" in (audit.get("errors") or []):
            raise RuntimeError("Process inventory unavailable; camera access skipped for safety")
        if audit.get("camera_apps_running"):
            raise RuntimeError("Close camera apps / live capture first and run a NEW diagnostic")
        import cv2

        summary["opencv_version"] = cv2.__version__
        write_json(
            root / "opencv.json", {"version": cv2.__version__, "build": cv2.getBuildInformation()}
        )
        if __import__("shutil").disk_usage(root).free < 5 * 1024**3:
            raise RuntimeError("At least 5 GiB free disk space is required")
        if args.workflow == "comparison":
            collect_comparison(root, cv2, summary)
        elif args.workflow == "motion-stability":
            collect_motion_stability(root, cv2, summary)
        else:
            collect_exposure_comparison(root, cv2, summary, args.project_root, args.runtime_root)
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = str(error) or "INTERRUPTED"
        print(f"Diagnostic stopped: {summary['error']}", flush=True)
    finally:
        target = pack_result(root, summary)
        print(f"\nSEND THIS FILE BACK (as a file, not a compressed WeChat video):\n{target}")
        print("Diagnostic collection only. No Mainline 1 / USB acceptance PASS is claimed.")
    return 0 if summary["status"] == "COLLECTED_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
