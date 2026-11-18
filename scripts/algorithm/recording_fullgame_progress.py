"""Bounded observation of the real media worker; never supplies product results."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from scripts.development.blackbox_report import atomic_json


def _replace_heartbeat(source, destination, *, clock=time.monotonic, pause=time.sleep):
    """Windows readers can briefly deny replacement; never hide persistent I/O failure."""
    deadline = clock() + 0.25
    for attempt in range(26):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            remaining = deadline - clock()
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 25 or remaining <= 0:
                raise
            pause(min(0.01, remaining))
            if clock() >= deadline:
                raise


class FullgameProgress:
    def __init__(self, output, report, *, started, clock=time.monotonic):
        self.output, self.report, self.started, self.clock = output, report, started, clock
        self.lock = threading.RLock()
        self.local = threading.local()
        self.rows = {}
        self.latest = {}
        self.last_write = float("-inf")
        self.report["clip_timeline"] = []

    def unavailable(self, code):
        with self.lock:
            errors = self.report.setdefault("observer_errors", [])
            if code not in errors and len(errors) < 16:
                errors.append(code)

    def emit(self, row):
        try:
            print(json.dumps(row), flush=True)
        except Exception:
            self.unavailable("PROGRESS_OUTPUT_UNAVAILABLE")

    def heartbeat(self, sample=None, *, force=False):
        with self.lock:
            if sample is not None:
                self.latest = dict(sample)
            now = self.clock()
            if not force and now - self.last_write < 2:
                return
            self.last_write = now
            try:
                atomic_json(
                    self.output / "observer-heartbeat.json",
                    dict(
                        contract="recording-fullgame-progress/v1",
                        run_id=self.output.parent.name,
                        seconds=round(now - self.started, 3),
                        latest=self.latest,
                        clips=list(self.rows.values()),
                        diagnostic=self.report.get("background_diagnostic"),
                    ),
                    replace=_replace_heartbeat,
                )
            except Exception:
                self.unavailable("HEARTBEAT_UNAVAILABLE")

    @contextmanager
    def observe(self):
        from billiards_ai.lab import mainline1_recording_media as media
        from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker

        process = Mainline1RecordingMediaWorker._process_claim
        observer = self

        def observed_process(worker, claim, now_utc):
            key = (claim.job.job_id, claim.attempt_count)
            with observer.lock:
                if key not in observer.rows and len(observer.rows) >= 384:
                    # A fixed-input observer must not grow with wall-clock waiting.
                    observer.report["timeline_overflow"] = True
                    observer.unavailable("TIMELINE_CAPACITY")
                    observer.local.row = None
                else:
                    observer.local.row = True
                if observer.local.row:
                    row = dict(
                        job_sha256=hashlib.sha256(claim.job.job_id.encode()).hexdigest(),
                        shot_sha256=hashlib.sha256(claim.job.shot_id.encode()).hexdigest(),
                        attempt=claim.attempt_count,
                        start_seconds=round(observer.clock() - observer.started, 3),
                        encode_seconds=0.0,
                        verify_seconds=0.0,
                        status="RUNNING",
                        error_code=None,
                    )
                    observer.rows[key] = row
                    observer.local.row = row
                    observer.emit(dict(stage="clip_started", **row))
            try:
                return process(worker, claim, now_utc)
            finally:
                try:
                    if getattr(observer.local, "row", None) is not None:
                        snapshot = worker._jobs.snapshot(claim.job.job_id)
                        with observer.lock:
                            row.update(
                                done_seconds=round(observer.clock() - observer.started, 3),
                                status=snapshot.status.value,
                                error_code=snapshot.error_code,
                                terminal_failure=snapshot.terminal_failure,
                            )
                            observer.report["clip_timeline"] = list(observer.rows.values())
                            observer.local.row = None
                            observer.emit(dict(stage="clip_done", **row))
                            observer.heartbeat(force=True)
                except Exception:
                    observer.unavailable("CLIP_OBSERVATION_UNAVAILABLE")
                finally:
                    observer.local.row = None

        def timed(function, field):
            def call(*args, **kwargs):
                start = observer.clock()
                try:
                    return function(*args, **kwargs)
                finally:
                    row = getattr(observer.local, "row", None)
                    if row is not None:
                        with observer.lock:
                            row[field] += round(observer.clock() - start, 3)

            return call

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(Mainline1RecordingMediaWorker, "_process_claim", observed_process)
            )
            stack.enter_context(
                patch.object(
                    media, "_encode_interval", timed(media._encode_interval, "encode_seconds")
                )
            )
            stack.enter_context(
                patch.object(media, "_verify_mp4", timed(media._verify_mp4, "verify_seconds"))
            )
            yield self
