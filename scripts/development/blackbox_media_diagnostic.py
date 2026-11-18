"""One bounded component probe against a completed black-box run; never a product gate."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

RUN_ID = "3f7e2c1f-426c-472b-90c1-bdc0bd0f07ea"
SOURCE = "085f2916f120f67a66c27ed03bf53de356b422fb"
TRANSPORT = "812687445fdec93bec5542cb17edebbc5692221c"
ENVIRONMENT = "001345343f99ecb69e76d44e443b0e47f3e3b2c48dd23c0d0f6fa27c5ac2f43c"
CONTRACT = "blackbox-media-component-diagnostic/v1"

STAGE_NAMES = {
    "VALIDATION",
    "COMPONENTS",
    "GUARDS_100",
    "REVIEW",
    "PUBLISH_ONE",
    "MEASURE_TIMING",
    "ENCODE_INTERVAL",
    "VERIFY_MP4",
    "VERIFY_RECORDING_PLAYBACK_ARTIFACT",
}
EVENT_NAMES = {
    "INVENTORY",
    "PROGRESS",
    "REVIEW_COUNTS",
    "SELECTED_INTERVAL",
    "ARTIFACT_VERIFIED",
    "CODEC_STARTUP_ENV",
    "COMPLETE",
    "FAILED",
} | {name + suffix for name in STAGE_NAMES for suffix in ("_START", "_END")}
METRIC_NAMES = {
    "field_connect",
    "jobs_connect",
    "continue",
    "media_hash",
    "jpeg_decode",
    "writer_open",
    "writer_write",
    "writer_release",
}


def export_events(raw):
    def finite(value):
        return type(value) in {int, float} and math.isfinite(value) and 0 <= value < 2**63

    result = []
    for line in raw.splitlines()[:256]:
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("stage"), str)
            or row["stage"] not in EVENT_NAMES
        ):
            continue
        clean = {"stage": row["stage"]}
        if isinstance(row.get("active_stage"), str) and row["active_stage"] in STAGE_NAMES:
            clean["active_stage"] = row["active_stage"]
        for key in (
            "elapsed_seconds",
            "seconds",
            "frames",
            "output_bytes",
            "rows",
            "available",
            "start_frame_id",
            "end_frame_id",
        ):
            if finite(row.get(key)):
                clean[key] = row[key]
        for key in ("complete", "codec_path_matches"):
            if type(row.get(key)) is bool:
                clean[key] = row[key]
        if isinstance(row.get("exception_type"), str) and row["exception_type"] in {
            "TimeoutError",
            "ValueError",
            "RuntimeError",
            "OSError",
            "PermissionError",
            "RecordingPlaybackError",
            "error",
        }:
            clean["exception_type"] = row["exception_type"]
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            clean["metrics"] = {
                name: {
                    key: values[key]
                    for key in ("count", "seconds", "errors")
                    if finite(values.get(key))
                }
                for name, values in metrics.items()
                if name in METRIC_NAMES and isinstance(values, dict)
            }
        jobs = row.get("jobs")
        if isinstance(jobs, list) and len(jobs) <= 28:
            clean["jobs"] = [
                {
                    **{key: job[key] for key in ("index", "attempts") if finite(job.get(key))},
                    **{
                        key: job[key]
                        for key in ("terminal", "error_present")
                        if type(job.get(key)) is bool
                    },
                    "status": job.get("status")
                    if isinstance(job.get("status"), str)
                    and job["status"] in {"pending", "running", "available", "failed_retrying"}
                    else "OTHER",
                }
                for job in jobs
                if isinstance(job, dict)
            ]
        result.append(clean)
    return result


def expected_runtime_paths(run):
    return (
        "\n".join(
            [
                "python312.zip",
                ".",
                str(run / "source/src"),
                str(run / "source"),
                str(run / "dependencies"),
            ]
        )
        + "\n"
    ).encode()


def verify_runtime_paths(run):
    path = run / "runtime/python312._pth"
    if path.stat().st_size > 4096 or path.read_text(
        encoding="utf-8"
    ).encode() != expected_runtime_paths(run):
        raise ValueError("RUNTIME_PATH_IDENTITY")


class Profiler:
    def __init__(self, emit, *, seconds=60, clock=time.perf_counter):
        self.emit, self.seconds, self.clock = emit, seconds, clock
        self.started = clock()
        self.metrics = {}
        self.stage = "VALIDATION"

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise TimeoutError("DIAGNOSTIC_BUDGET")

    def wrap(self, name, function, *, check_budget=True):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            if check_budget:
                self.check()
            started = self.clock()
            metric = self.metrics.setdefault(name, {"count": 0, "seconds": 0.0, "errors": 0})
            metric["count"] += 1
            try:
                return function(*args, **kwargs)
            except BaseException:
                metric["errors"] += 1
                raise
            finally:
                metric["seconds"] += self.clock() - started
                if name == "continue" and metric["count"] % 100 == 0:
                    self.snapshot("PROGRESS")

        return measured

    def snapshot(self, event, **values):
        self.emit(
            {
                "stage": event,
                "active_stage": self.stage,
                "elapsed_seconds": self.clock() - self.started,
                "metrics": {k: dict(v) for k, v in self.metrics.items()},
                **values,
            }
        )

    def call(self, stage, function, *args, **kwargs):
        self.check()
        previous, self.stage = self.stage, stage
        self.snapshot(stage + "_START")
        started, complete = self.clock(), False
        try:
            result = function(*args, **kwargs)
            complete = True
            return result
        finally:
            self.snapshot(stage + "_END", seconds=self.clock() - started, complete=complete)
            self.stage = previous


def backup_database(source, target, check):
    """Read a stopped run's DB/WAL bytes; SQLite bookkeeping stays in the new snapshot."""
    if target.exists():
        raise FileExistsError(target)
    snapshot = target.with_suffix(".snapshot")
    snapshot.mkdir()
    members = [source]
    wal = source.with_name(source.name + "-wal")
    if wal.exists():
        members.append(wal)
    pins = {}
    for member in members:
        safe_under(member, source.parent)
        if member.stat().st_size > 32 * 1024**2:
            raise ValueError("DATABASE_SIZE")
        check()
        data = member.read_bytes()
        pins[member] = hashlib.sha256(data).hexdigest()
        (snapshot / member.name).write_bytes(data)
    # No source SQLite connection: even read-only WAL connections can update original -shm.
    for member, digest in pins.items():
        check()
        if hashlib.sha256(member.read_bytes()).hexdigest() != digest:
            raise ValueError("DATABASE_CHANGED")
    if wal.exists() != (wal in pins):
        raise ValueError("DATABASE_CHANGED")
    target.touch(exist_ok=False)
    with contextlib.closing(
        sqlite3.connect((snapshot / source.name).as_uri() + "?mode=ro", uri=True)
    ) as origin:
        with contextlib.closing(sqlite3.connect(target)) as destination:
            origin.backup(destination, pages=64, progress=lambda *_: check(), sleep=0.01)
    check()


def profile_components(recording, output, profile, *, verify_imports=False):
    # Runtime's frozen ._pth binds these to the tested source/environment, not this script's repo.
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab import mainline1_recording_media as media
    from billiards_ai.lab.field_assistance.server import _FieldMediaResourceGate
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker

    if verify_imports:
        import cv2
        import numpy

        run = recording.parent
        for module, expected in (
            (media, run / "source/src"),
            (cv2, run / "dependencies"),
            (numpy, run / "dependencies"),
        ):
            safe_under(Path(module.__file__), expected)
    for origin, target in (
        (recording / "state.sqlite3", output / "state.sqlite3"),
        (recording / "jobs/analysis-jobs.sqlite3", output / "jobs.sqlite3"),
    ):
        safe_under(origin, recording)
        backup_database(origin, target, profile.check)
    records = SQLiteFieldRecordStore(output / "state.sqlite3")
    jobs = SQLiteShotAnalysisJobStore(output / "jobs.sqlite3")
    # Preserve the production connection object and lifetime: no proxy or retained connection list.
    records._connect = profile.wrap("field_connect", records._connect)
    jobs._connect = profile.wrap("jobs_connect", jobs._connect)
    with contextlib.closing(sqlite3.connect(output / "jobs.sqlite3")) as connection:
        rows = connection.execute(
            "SELECT status,attempt_count,terminal_failure,error_code "
            "FROM shot_analysis_jobs WHERE job_kind='media' ORDER BY rowid LIMIT 29"
        ).fetchall()
        sessions = connection.execute(
            "SELECT DISTINCT json_extract(payload_json,'$.session_id') "
            "FROM shot_analysis_jobs WHERE job_kind='media' LIMIT 2"
        ).fetchall()
    if len(rows) != 28 or len(sessions) != 1:
        raise ValueError("INVENTORY")
    # Do not export arbitrary error strings, paths, payloads, lease tokens or ownership identities.
    profile.snapshot(
        "INVENTORY",
        jobs=[
            {
                "index": i,
                "status": r[0]
                if r[0] in {"pending", "running", "available", "failed_retrying"}
                else "OTHER",
                "attempts": r[1],
                "terminal": bool(r[2]),
                "error_present": r[3] is not None,
            }
            for i, r in enumerate(rows)
        ],
    )
    session_id = sessions[0][0]
    items = jobs.list_session_media_jobs(session_id)
    if len(items) != 28:
        raise ValueError("INVENTORY")
    job = items[0]
    if job.end_frame_id - job.start_frame_id + 1 > 1200:
        raise ValueError("FRAME_BUDGET")
    for item in items:
        safe_under(item.media_root, recording)
        safe_under(item.source.reference_path, recording)
        if item.source.reference_path.stat().st_size > 1024**2:
            raise ValueError("REFERENCE_SIZE")
        reference = json.loads(item.source.reference_path.read_bytes())
        safe_under(Path(reference["segment_root"]), recording)
    state = SimpleNamespace(
        match_lock=threading.RLock(),
        record_store=records,
        _media_stopped=threading.Event(),
        session_id=session_id,
        projection_lifecycle_closing=False,
        projection_lifecycle_closed=False,
    )
    worker = Mainline1RecordingMediaWorker(
        records=records,
        jobs=jobs,
        artifact_root=output / "artifacts",
        resource_gate=_FieldMediaResourceGate(state),
    )
    check = profile.wrap("continue", lambda: worker._check_continue(session_id, job.shot_id))
    profile.call("GUARDS_100", lambda: [check() for _ in range(100)])
    manifest = records.get_postgame_manifest(session_id)
    projection = profile.call("REVIEW", media.project_recording_playback, records, jobs, manifest)
    profile.snapshot(
        "REVIEW_COUNTS",
        rows=len(projection),
        available=sum(row.playback_clip is not None for row in projection),
    )
    # Preserve original calls, check cadence and integrity requirements.
    originals = {
        name: getattr(media, name)
        for name in (
            "_measure_timing",
            "_encode_interval",
            "_verify_mp4",
            "verify_recording_playback_artifact",
        )
    }
    for name, original in originals.items():
        setattr(media, name, functools.partial(profile.call, name.strip("_").upper(), original))
    original_hash = media._hash_media_file
    original_decode = media.cv2.imdecode
    media._hash_media_file = profile.wrap("media_hash", original_hash)
    media.cv2.imdecode = profile.wrap("jpeg_decode", original_decode)

    class MeasuredWriter:
        def __init__(self, *args):
            self.writer = profile.wrap("writer_open", media.cv2.VideoWriter)(*args)

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, image):
            return profile.wrap("writer_write", self.writer.write)(image)

        def release(self):
            return profile.wrap("writer_release", self.writer.release, check_budget=False)()

    profile.snapshot(
        "SELECTED_INTERVAL",
        start_frame_id=job.start_frame_id,
        end_frame_id=job.end_frame_id,
        frames=job.end_frame_id - job.start_frame_id + 1,
    )
    try:
        artifact = profile.call(
            "PUBLISH_ONE",
            media.publish_recording_playback,
            job,
            output / "artifacts",
            authority_check=worker._validate_authority,
            continue_check=check,
            writer_factory=MeasuredWriter,
        )
        profile.snapshot(
            "ARTIFACT_VERIFIED",
            frames=artifact.frame_count,
            output_bytes=(artifact.path / "original.mp4").stat().st_size,
        )
    finally:
        media._hash_media_file = original_hash
        media.cv2.imdecode = original_decode
        for name, original in originals.items():
            setattr(media, name, original)
    profile.snapshot("COMPLETE")


def safe_under(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not root.is_absolute() or ".." in path.parts or ".." in root.parts:
        raise ValueError("SOURCE_PATH")
    if not path.is_relative_to(root):
        raise ValueError("SOURCE_PATH")
    # Inspect root to leaf before resolve/stat can traverse a junction ancestor.
    for part in reversed((path, *path.parents)):
        if part.name.casefold() == ".project-archive" or part.is_symlink() or part.is_junction():
            raise ValueError("SOURCE_PATH")
    if path.resolve(strict=True) != path or root.resolve(strict=True) != root:
        raise ValueError("SOURCE_PATH")
    return path


def child(run, output):
    if sys.stdin.buffer.readline(200) != ("MEDIA-DIAGNOSTIC " + output.name + "\n").encode():
        raise ValueError("OWNED_RELEASE_REQUIRED")

    # The entry only imports stdlib until Job assignment and release.
    def emit(row):
        print(json.dumps(row, separators=(",", ":")), flush=True)

    profile = Profiler(emit, seconds=55)
    try:
        sys.path.insert(0, str(run / "source"))
        from scripts.development import blackbox_source as sources
        from scripts.development import blackbox_validation as validation

        if sources.pin(run / "export/summary.json") != {
            "bytes": 2620,
            "sha256": "df008a4b63fe818b7efbe8b6cbcb3c3c036f249f16011f84809a4060ee6016cd",
        }:
            raise ValueError("RUN_RECEIPT_IDENTITY")
        summary = validation.read_json(run / "export/summary.json")
        if (
            run.name,
            summary.get("source_commit"),
            summary.get("transport_commit"),
            summary.get("environment_id"),
            summary.get("status"),
            summary.get("owned_cleanup"),
        ) != (RUN_ID, SOURCE, TRANSPORT, ENVIRONMENT, "TIMEOUT", True):
            raise ValueError("RUN_IDENTITY")
        if sources.pin(run / "source/SOURCE-LOCK.json") != summary.get("source_lock"):
            raise ValueError("SOURCE_LOCK_IDENTITY")
        if sources.verify(run / "source")["commit"] != SOURCE:
            raise ValueError("SOURCE_IDENTITY")
        context = validation.read_json(run / "context.json")
        config = validation.check_config(context["config"])
        for folder, pins in (
            (run / "dependencies", config["dependency_pins"]),
            (run / "runtime", config["runtime_pins"]),
        ):
            for name, pin in pins.items():
                if sources.pin(folder / name) != pin:
                    raise ValueError("ENVIRONMENT_IDENTITY")
        verify_runtime_paths(run)
        if Path(sys.executable).resolve() != (run / "runtime/python.exe").resolve():
            raise ValueError("RUNTIME_IDENTITY")
        # The run's pinned codec and bounded environment are reused without installation.
        from scripts.algorithm.recording_windows_entry import CODEC

        if sources.pin(run / "codec" / CODEC["filename"])["sha256"] != CODEC["sha256"]:
            raise ValueError("CODEC_IDENTITY")
        profile.snapshot(
            "CODEC_STARTUP_ENV",
            codec_path_matches=(
                os.environ.get("OPENH264_LIBRARY") == str(run / "codec" / CODEC["filename"])
            ),
        )
        os.environ["OPENH264_LIBRARY"] = str(run / "codec" / CODEC["filename"])
        profile.call(
            "COMPONENTS",
            profile_components,
            run / "recording",
            output,
            profile,
            verify_imports=True,
        )
        return 0
    except BaseException as exc:
        profile.snapshot("FAILED", exception_type=type(exc).__name__)
        return 3 if isinstance(exc, TimeoutError) else 1


def public_owned(result):
    return {
        key: result.get(key)
        for key in (
            "status",
            "reason",
            "released",
            "job_assigned",
            "cleanup_confirmed",
            "worker_exit_code",
            "terminal_record_failed",
        )
    }


def diagnostic_environment(run, output, config, validation):
    """Bind the frozen codec before process creation; writable paths belong to output."""
    from scripts.algorithm.recording_windows_entry import CODEC

    env = validation.child_environment(output, config)
    env["OPENH264_LIBRARY"] = str(run / "codec" / CODEC["filename"])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows only; portable measurement helpers may be tested separately")
    run = args.run
    safe_under(run, run.parent)
    if (
        run.name != RUN_ID
        or run.parent.name != "runs"
        or run.parent.parent.name != "billiardsAI-dev-validation"
    ):
        raise ValueError("RUN_IDENTITY")
    if args.child_output:
        output = safe_under(args.child_output, run.parent.parent / "diagnostics")
        return child(run, output)
    sys.path.insert(0, str(run / "source"))
    from scripts.development import blackbox_report as reports
    from scripts.development import blackbox_validation as validation

    context = validation.read_json(run / "context.json")
    config = validation.normalize_config(context["config"])
    ownership = validation.load_ownership(config["ownership"])
    diagnostics = run.parent.parent / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    safe_under(diagnostics, run.parent.parent)
    output = diagnostics / ("media-components-" + str(uuid.uuid4()))
    output.mkdir()
    lock = run.parent / "validation.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(output.name)
    result = None
    try:
        command = [
            str(run / "runtime/python.exe"),
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--run",
            str(run),
            "--child-output",
            str(output),
        ]

        def sleep(seconds):
            if any(p.stat().st_size > 128 * 1024**2 for p in output.rglob("*") if p.is_file()):
                raise RuntimeError("OUTPUT_BUDGET")
            time.sleep(seconds)

        with (
            (output / "events.jsonl").open("xb") as stdout,
            (output / "local-stderr.txt").open("xb") as stderr,
        ):
            result = ownership.run_owned(
                ownership.WindowsAPI(),
                lambda: subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    env=diagnostic_environment(run, output, config, validation),
                    cwd=output,
                    close_fds=True,
                ),
                lambda stage, state: reports.atomic_json(
                    output / "ownership.json", {"stage": stage, **state}
                ),
                {"run_id": output.name, "phase": "media_components"},
                ("MEDIA-DIAGNOSTIC " + output.name + "\n").encode(),
                deadline=60,
                cleanup_seconds=10,
                sleep=sleep,
            )
    finally:
        lock_released = False
        try:
            if lock.read_text(encoding="ascii") == output.name:
                lock.unlink()
                lock_released = True
        except OSError:
            pass
    passed = bool(
        result
        and result.get("status") == "PASS"
        and result.get("worker_exit_code") == 0
        and result.get("cleanup_confirmed") is True
        and lock_released
        and not result.get("terminal_record_failed")
        and not result.get("job_close_error")
    )
    summary = {
        "contract": CONTRACT,
        "diagnostic_id": output.name,
        "input_run_id": RUN_ID,
        "source_commit": SOURCE,
        "environment_id": ENVIRONMENT,
        "input_transport_commit": TRANSPORT,
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "product_run": False,
        "product_pass": False,
        "browser_run": False,
        "status": "COMPLETE" if passed else "INCOMPLETE",
        "owned": public_owned(result or {}),
        "lock_released": lock_released,
        "limitation": (
            "Serial copied-database component costs; "
            "no HTTP/match-lock concurrency or claim/renew/complete coverage."
        ),
    }
    export = output / "export"
    export.mkdir()
    events = output / "events.jsonl"
    with events.open("rb") as stream:
        raw = stream.read(256 * 1024)
    filtered = export_events(raw)
    if events.stat().st_size > 256 * 1024 or not any(
        row["stage"] == "COMPLETE" for row in filtered
    ):
        passed = False
        summary["status"] = "INCOMPLETE"
    reports.atomic_json(export / "summary.json", summary)
    (export / "events.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in filtered),
        encoding="utf-8",
        newline="\n",
    )
    members = {}
    for name in ("summary.json", "events.jsonl"):
        raw = (export / name).read_bytes()
        members[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    reports.atomic_json(
        export / "evidence-manifest.json",
        {
            "contract": CONTRACT,
            "diagnostic_id": output.name,
            "input_run_id": RUN_ID,
            "members": members,
        },
    )
    print(json.dumps({"diagnostic_id": output.name, "status": summary["status"]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
