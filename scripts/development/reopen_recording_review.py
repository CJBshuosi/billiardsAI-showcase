"""Owned, cancellable localhost review of an already completed recording.

This does not run capture/encoding or change the original validation verdict.
Original databases use fresh read-only connections; mutation routes are absent.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from scripts.development import blackbox_report as reports  # noqa: E402
from scripts.development import blackbox_source as sources  # noqa: E402
from scripts.development import blackbox_validation as validation  # noqa: E402


class ReadOnlyConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def readonly_connection(path):
    connection = sqlite3.connect(
        sources.plain(path).as_uri() + "?mode=ro", uri=True, timeout=5, factory=ReadOnlyConnection
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def readonly_stores(field_path, jobs_path):
    from billiards_ai.edge.route_assistance import persistence
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore

    class Records(persistence.SQLiteFieldRecordStore):
        def _initialize(self):
            with closing(self._connect()) as connection:
                if (
                    connection.execute("PRAGMA user_version").fetchone()[0]
                    != persistence._SCHEMA_VERSION
                ):
                    raise ValueError("review requires the current field schema")

        def _connect(self, *, deadline=None):
            return readonly_connection(self.path)

    class Jobs(SQLiteShotAnalysisJobStore):
        def __init__(self, path):
            self._database = sources.plain(path)
            with closing(self._connect()) as connection:
                connection.execute("SELECT job_id FROM shot_analysis_jobs LIMIT 1").fetchall()

        def _connect(self):
            return readonly_connection(self._database)

    return Records(sources.plain(field_path)), Jobs(jobs_path)


def review_page():
    from billiards_ai.lab.field_assistance import pages

    return (
        "<!doctype html><html lang=zh-CN><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>已结束场次回看</title><style>body{margin:0;background:#071009;"
        "color:#e5eee8;font:18px sans-serif}"
        + pages._POSTGAME_STYLE
        + "#postgame-panel{position:relative;left:auto;bottom:auto;margin:24px auto;"
        "max-height:none;"
        "width:min(1100px,calc(100vw - 64px))}#postgame-player{max-height:55vh}"
        "#postgame-finish,#postgame-panel>p:last-of-type{display:none}</style><main>"
        + pages._POSTGAME_PANEL
        + "</main>"
        + pages._POSTGAME_SCRIPT
        + "</html>"
    )


def build_review_server(records, jobs, session_id, original_root):
    from http import HTTPStatus
    from types import SimpleNamespace
    from urllib.parse import urlsplit

    from billiards_ai.lab.field_assistance import server as field

    session = records.load_session(session_id)
    if session.status.value != "ended" or records.has_managed_session_identity(session_id):
        raise ValueError("review requires the original unscoped ended session")
    if records.artifact_path_for_session(session_id) != original_root:
        raise ValueError("original review artifact binding changed")

    class State(field._ServerState):
        def maintain_service(self):
            pass  # No capture, encoder, publisher, sync, selection or cleanup worker.

        def close_projection(self):
            self.projection_lifecycle_closing = self.projection_lifecycle_closed = True

    class Handler(field._FieldAssistanceHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route in {"/", "/screen"}:
                self._send_bytes(
                    HTTPStatus.OK, review_page().encode(), content_type="text/html; charset=utf-8"
                )
            elif route == "/api/postgame-review":
                try:
                    self._send_json(
                        HTTPStatus.OK, self.server.field_state.postgame_review_payload()
                    )
                except (OSError, ValueError):
                    self._error(HTTPStatus.CONFLICT, "review is temporarily unavailable")
            elif field._POSTGAME_MEDIA_PATTERN.fullmatch(route):
                super().do_GET()
            else:
                self._error(HTTPStatus.NOT_FOUND, "review resource is unavailable")

        def do_POST(self):
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, "review is read-only")

    state = State(
        artifact_path=original_root,
        control_token=secrets.token_urlsafe(32),
        replay_clock=None,
        match=session.match,
        match_lock=threading.RLock(),
        frame_store=None,
        record_store=records,
        session_id=session_id,
        artifact_id=session.artifact_id,
        shot_index=None,
        recommendation_power={},
        duration_ms=0,
        projection_source=SimpleNamespace(),
        media_job_store=jobs,
    )
    server = field.FieldAssistanceHTTPServer(("127.0.0.1", 0), state)
    server.RequestHandlerClass = Handler
    return server


def close_requested(root):
    path = root / "close.json"
    if not path.exists():
        return False
    try:
        return validation.read_json(path, 1024) == {"run_id": root.name}
    except (OSError, ValueError):
        return False


def probe_http_close(records, jobs, session_id, artifact_root):
    """Actual idle browser-connection fault control; never supplies game facts."""
    import socket

    server = build_review_server(records, jobs, session_id, artifact_root)
    serving = threading.Thread(target=server.serve_forever, name="review-close-probe", daemon=True)
    clients, owned = [], []
    started = False
    try:
        serving.start()
        started = True
        for index in range(5):
            client = socket.create_connection(server.server_address, timeout=2)
            clients.append(client)
            if index % 2:
                client.sendall(b"GET /scr")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with server._request_lock:
                owned = list(server._request_connections.values())
            if len(owned) == 5:
                break
            time.sleep(0.01)
        if len(owned) != 5:
            raise ValueError("idle HTTP close probe did not own five requests")
        server.shutdown()
        server.server_close()
        serving.join(2)
        if serving.is_alive() or any(thread.is_alive() for thread in owned):
            raise ValueError("idle HTTP close probe leaked an owned thread")
        return dict(idle_connections=5, cleanup_confirmed=True)
    finally:
        for client in clients:
            client.close()
        if started:
            server.shutdown()
        server.server_close()
        if started:
            serving.join(2)


def serve_child(root):
    if sys.stdin.buffer.readline(128) != ("BLACKBOX-RELEASE " + root.name + "\n").encode():
        raise ValueError("review release identity mismatch")
    context = validation.read_json(root / "context.json")
    lock = sources.verify(root / "source")
    original = sources.plain(Path(context["original_run"]))
    report_path = original / "recording/report.json"
    if sources.pin(report_path)["sha256"] != context["original_report_sha256"]:
        raise ValueError("original report bytes changed")
    original_source = sources.verify(original / "source")
    report = validation.read_product_report(original, "FullgameComplete")
    if (
        original_source["commit"] != context["original_source"]
        or report.get("source_commit") != context["original_source"]
        or report.get("source_sha256") != validation.SOURCE_SHA
        or report.get("product_pass") is not True
        or len(report.get("media", [])) != 28
        or report.get("raw_recording", {}).get("decoded_frames") != 15880
    ):
        raise ValueError("original recording verification identity is unavailable")
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
    from billiards_ai.lab.mainline1_recording_media import verify_recording_playback_artifact

    records, jobs = readonly_stores(
        original / "recording/state.sqlite3", original / "recording/jobs/analysis-jobs.sqlite3"
    )
    session_id = report["session_id"]
    artifact_root = sources.plain(original / "recording/artifacts")
    media = {row["shot_id"]: row for row in report["media"]}
    snapshots = jobs.session_media_snapshots(session_id)
    if len(snapshots) != 28 or len(media) != 28:
        raise ValueError("original media inventory is not exactly 28")
    for count, snapshot in enumerate(snapshots, 1):
        if close_requested(root):
            return 4
        if snapshot.status.value != "available" or not snapshot.artifact_id:
            raise ValueError("original media task is not available")
        with session_media_lock(artifact_root, session_id, exclusive=False, timeout_seconds=0):
            artifact = verify_recording_playback_artifact(
                artifact_root / "postgame" / session_id / snapshot.artifact_id, snapshot.job
            )
            recorded = media.get(snapshot.job.shot_id, {})
            if (
                artifact.content_sha256 != recorded.get("sha256")
                or artifact.byte_size != recorded.get("bytes")
                or artifact.frame_count != recorded.get("frames")
            ):
                raise ValueError("original verified media content changed")
        reports.atomic_json(root / "progress.json", dict(run_id=root.name, verified_media=count))
    close_probe = probe_http_close(records, jobs, session_id, artifact_root)
    reports.atomic_json(root / "http-close-probe.json", close_probe)
    server = build_review_server(records, jobs, session_id, artifact_root)
    result = dict(
        scope="READONLY_REVIEW_ONLY",
        original_run_id=original.name,
        original_source=context["original_source"],
        viewer_source=lock["commit"],
        original_internal_cleanup=report.get("cleanup_confirmed"),
        cleanup_confirmed=False,
    )
    return serve_review(server, root, result, session_id)


def serve_review(server, root, result, session_id):
    serving = threading.Thread(
        target=server.serve_forever, name="readonly-review-http", daemon=True
    )
    before = set(threading.enumerate())
    primary = None
    try:
        serving.start()
        projection = server.field_state.postgame_review_payload()
        if projection["available_count"] != 28 or projection["expected_count"] != 28:
            raise ValueError("original ended review is incomplete")
        ready = dict(
            contract="recording-readonly-review-ready/v1",
            run_id=root.name,
            url=f"http://127.0.0.1:{server.server_address[1]}/screen",
            session_id=session_id,
            postgame_manifest_sha256=projection["postgame_manifest_sha256"],
        )
        reports.atomic_json(root / "ready.json", ready)
        print(json.dumps(dict(stage="review_ready", **ready)), flush=True)
        while not close_requested(root):
            time.sleep(0.25)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        if serving.ident is not None:
            try:
                server.shutdown()
            except BaseException as exc:
                failures.append(exc)
        try:
            server.server_close()
        except BaseException as exc:
            failures.append(exc)
        if serving.ident is not None:
            serving.join(2)
        result["remaining_threads"] = [
            t.name for t in threading.enumerate() if t not in before and t.is_alive()
        ]
        result["cleanup_confirmed"] = (
            not failures and not serving.is_alive() and not result["remaining_threads"]
        )
        reports.atomic_json(root / "review-result.json", result)
        if failures:
            if primary is not None:
                primary.add_note(
                    "review cleanup errors: " + ",".join(type(e).__name__ for e in failures)
                )
            else:
                raise BaseExceptionGroup("review cleanup failed", failures)
    return 0 if result["cleanup_confirmed"] else 1


class ReviewMonitor:
    def __init__(self, root, summary, *, clock=time.monotonic, sleeper=time.sleep):
        self.root, self.summary, self.clock, self.sleeper = root, summary, clock, sleeper
        self.stop_seen = None
        self.persisted = float("-inf")
        self.cleaning = False

    def sleep(self, seconds):
        if self.cleaning:
            self.sleeper(seconds)
            return
        try:
            now = self.clock()
            if close_requested(self.root):
                self.stop_seen = now if self.stop_seen is None else self.stop_seen
                if now - self.stop_seen > 10:
                    raise KeyboardInterrupt("review close did not finish within grace")
            for name in ("review-stdout.txt", "review-stderr.txt"):
                path = self.root / "local-logs" / name
                if path.exists() and path.stat().st_size > 1024**2:
                    raise RuntimeError("review log budget exceeded")
            if now - self.persisted >= 1:
                if (self.root / "ready.json").exists():
                    self.summary["stage"] = "READY"
                probe = self.root / "http-close-probe.json"
                if probe.exists():
                    self.summary["http_close_probe"] = validation.read_json(probe, 4096)
                progress = self.root / "progress.json"
                if progress.exists():
                    self.summary["verified_media"] = validation.read_json(progress, 4096).get(
                        "verified_media"
                    )
                reports.publish(self.root / "export", self.summary)
                self.persisted = now
        except BaseException:
            # run_owned reuses this callback during cleanup: no second exception.
            self.cleaning = True
            raise
        self.sleeper(seconds)


def run_parent(args):
    if sys.platform != "win32":
        raise ValueError("owned review entry requires Windows")
    config = validation.normalize_config(validation.read_json(args.config))
    base = sources.plain(args.review_base)
    base.mkdir(exist_ok=True)
    root = base / str(uuid.uuid4())
    root.mkdir()
    (root / "export").mkdir()
    (root / "local-logs").mkdir()
    context = dict(
        config=config,
        profile="Review",
        repo=str(sources.plain(args.repo)),
        commit=args.transport,
        original_run=str(sources.plain(args.original_run)),
        original_source=args.original_source,
        original_report_sha256=args.original_report_sha256,
    )
    reports.atomic_json(root / "context.json", context)
    summary = dict(
        contract="blackbox-readonly-review/v1",
        run_id=root.name,
        profile="Review",
        scope="READONLY_REVIEW_ONLY",
        original_run_id=args.original_run.name,
        original_source=args.original_source,
        status="STARTING",
        exit_code=None,
        source_commit=None,
        transport_commit=args.transport,
        wrapper_commit=args.transport,
        owned_cleanup=None,
    )
    ownership = validation.load_ownership(config["ownership"])
    result = validation.execute_phase(root, config, summary, ownership, "setup")
    if summary["exit_code"] != 0:
        reports.publish(root / "export", summary)
        return summary["exit_code"]
    summary.update(validation.validate_identity(result["identity"]))
    summary.update(status="RUNNING", exit_code=None, stage="VERIFYING_MEDIA")
    entry = root / "source/scripts/development/reopen_recording_review.py"
    monitor = ReviewMonitor(root, summary)
    with (
        (root / "local-logs/review-stdout.txt").open("xb") as stdout,
        (root / "local-logs/review-stderr.txt").open("xb") as stderr,
    ):
        owned = ownership.run_owned(
            ownership.WindowsAPI(),
            lambda: subprocess.Popen(
                [str(root / "runtime/python.exe"), "-I", "-B", str(entry), "--child", str(root)],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                env=validation.child_environment(root, config),
                cwd=root,
                close_fds=True,
            ),
            lambda stage, state: reports.atomic_json(
                root / "review-ownership.json", dict(stage=stage, **state)
            ),
            dict(run_id=root.name, phase="review", scope="READONLY_REVIEW_ONLY"),
            ("BLACKBOX-RELEASE " + root.name + "\n").encode(),
            deadline=float("inf"),
            cleanup_seconds=10,
            sleep=monitor.sleep,
        )
    cancelled = (
        owned.get("exception_type") == "KeyboardInterrupt" or owned.get("worker_exit_code") == 4
    )
    summary.update(
        status="CLOSED" if owned.get("status") == "PASS" else "CANCELLED" if cancelled else "ERROR",
        exit_code=owned.get("worker_exit_code"),
        owned_cleanup=owned.get("cleanup_confirmed") is True,
    )
    reports.publish(root / "export", summary)
    print(
        json.dumps(
            dict(run_id=root.name, status=summary["status"], owned_cleanup=summary["owned_cleanup"])
        ),
        flush=True,
    )
    return 0 if summary["status"] == "CLOSED" and summary["owned_cleanup"] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--child":
        return serve_child(sources.plain(Path(argv[1])))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "original-run", "review-base"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("transport", "original-source", "original-report-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    for value, size in (
        (args.transport, 40),
        (args.original_source, 40),
        (args.original_report_sha256, 64),
    ):
        if not re.fullmatch(f"[0-9a-f]{{{size}}}", value):
            parser.error("fixed source/report identity required")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
