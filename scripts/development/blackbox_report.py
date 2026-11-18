"""Small allowlisted reports. Raw logs, database contents and URLs stay local."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

STAGES = {
    "capture",
    "background",
    "live_ready",
    "capture_stopped",
    "ended",
    "encoded",
    "verified_media",
    "browser_ready",
    "finished",
    "preflight",
    "module_check",
    "setup",
    "clip_started",
    "clip_done",
    "checkpoint_600",
}
NUMBERS = {
    "seconds",
    "start_seconds",
    "done_seconds",
    "encode_seconds",
    "verify_seconds",
    "attempt",
    "encoding_count",
    "retrying_count",
    "frame",
    "committed_frame_id",
    "delivered_frame_id",
    "processed_frame_id",
    "detection_backlog_frames",
    "buffered_frames",
    "pending_chunks",
    "shots",
    "peak_rss_bytes",
    "discovered_count",
    "expected_count",
    "available_count",
    "count",
    "emitted_frames",
    "cutoff",
    "available",
    "decoded_frames",
}


def atomic_json(path, value, *, replace=None):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    (os.replace if replace is None else replace)(tmp, path)


def finite(value):
    return type(value) in {int, float} and math.isfinite(value) and 0 <= value < 2**63


def event(raw):
    if not isinstance(raw, dict):
        return None
    stage = raw.get("stage", "capture" if "committed_frame_id" in raw else None)
    if stage not in STAGES:
        return None
    result = {"stage": stage}
    result.update({key: value for key, value in raw.items() if key in NUMBERS and finite(value)})
    processing = raw.get("processing")
    if isinstance(processing, dict):
        result["processing"] = {
            key: processing[key]
            for key in ("remaining_frames", "processed_frame_id", "committed_frame_id")
            if finite(processing.get(key))
        }
        if processing.get("status") in {"PENDING", "PROCESSING", "COMPLETE", "FAILED"}:
            result["processing"]["status"] = processing["status"]
    if stage in {"clip_started", "clip_done"}:
        import re

        for key in ("job_sha256", "shot_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(raw.get(key))):
                result[key] = raw[key]
        if raw.get("status") in {
            "pending",
            "running",
            "available",
            "failed",
            "failed_retrying",
            "RUNNING",
        }:
            result["status"] = raw["status"]
        if type(raw.get("terminal_failure")) is bool:
            result["terminal_failure"] = raw["terminal_failure"]
        code = raw.get("error_code")
        if isinstance(code, str) and re.fullmatch(r"MEDIA_[A-Z_]{1,64}", code):
            result["error_code"] = code
    return result


def publish(directory, summary):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    atomic_json(directory / "summary.json", summary)
    rows = [
        "# Windows blackbox development validation",
        "",
        "This report is not model, referee, physical-camera or cloud acceptance.",
        "",
    ]
    for key in (
        "run_id",
        "source_commit",
        "transport_commit",
        "wrapper_commit",
        "input_sha256",
        "started_at_utc",
        "ended_at_utc",
        "profile",
        "validation_scope",
        "capture_end_status",
        "status",
        "stage",
        "exit_code",
        "failure_stage",
        "failure_code",
        "loop_status",
        "product_status",
        "browser_status",
        "owned_cleanup",
        "deferred_cleanup",
        "last_observed",
        "elapsed_seconds",
    ):
        rows.append(f"- {key}: {json.dumps(summary.get(key), ensure_ascii=True)}")
    (directory / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    members = {}
    for name in ("summary.json", "summary.md", "events.jsonl"):
        path = directory / name
        if path.exists():
            data = path.read_bytes()
            members[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    local = {}
    for relative in (
        "setup-ownership.json",
        "worker-ownership.json",
        "setup-result.json",
        "worker-result.json",
        "recording/report.json",
        "recording/browser-ready.json",
        "local-logs/setup-stdout.txt",
        "local-logs/setup-stderr.txt",
        "local-logs/worker-stdout.txt",
        "local-logs/worker-stderr.txt",
    ):
        # Index only known local artifacts. Never export their contents or enumerate media.
        local[relative] = "PRESENT" if (directory.parent / relative).is_file() else "MISSING"
    atomic_json(
        directory / "evidence-manifest.json",
        {
            "contract": "blackbox-dev-export/v1",
            "run_id": summary["run_id"],
            "members": members,
            "local_artifacts": local,
        },
    )
