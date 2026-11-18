#!/usr/bin/env python3
"""Run one local development episode using the genuine managed black-box CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import socket
import sqlite3
import sys
import threading
import time
from contextlib import ExitStack, closing, suppress
from dataclasses import replace
from functools import partial
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from billiards_ai.edge.route_assistance.retention_inventory import _job  # noqa: E402
from billiards_ai.lab.field_assistance.cloud_runtime import FieldCloudFactories  # noqa: E402
from scripts.algorithm import offline_managed_transport as protocol  # noqa: E402
from scripts.algorithm import serve_mainline1_blackbox_field as cli  # noqa: E402
from scripts.algorithm.offline_postgame_selection import (  # noqa: E402
    PostgameSelectionFailure,
    run_postgame_selection,
)
from scripts.algorithm.offline_realtime_diagnostics import (  # noqa: E402
    MAX_BYTES as FIRST_SHOT_DIAGNOSTIC_BYTES,
)
from scripts.algorithm.offline_realtime_diagnostics import (
    fit_first_shot_diagnostics,
    project_first_shot_diagnostics,
    validate_first_shot_diagnostics_export,
)

_ACTIVE = threading.Lock()
_RETAINED = None  # Keep owners reachable if CLI shutdown cannot prove termination.
_LIMITS = (60.0, 45.0, 10.0, 300.0)
_OUTPUTS = dict(
    state_db="state.sqlite3", ring_root="ring", jobs_root="jobs", artifact_root="artifacts"
)
_FORBIDDEN = {
    "--development-analysis-sync",
    "--sync-base-url",
    "--device-id",
    "--credential-target",
    "--development-cloudbase-environment-id",
}


def _write_first_shot_diagnostic(evidence, capture):
    """Best-effort independent metadata file, before optional episode fitting.

    The evidence directory belongs to the fresh episode. Exclusive creation
    preserves any existing file; storage failure cannot replace its primary result.
    """
    if "first_shot_diagnostics" not in capture:
        return False
    try:
        value = validate_first_shot_diagnostics_export(capture["first_shot_diagnostics"])
        raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > FIRST_SHOT_DIAGNOSTIC_BYTES:
            return False
        with (evidence / "first-shot-diagnostics.json").open("xb") as stream:
            stream.write(raw)
        return True
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        return False


def _write_episode_report(evidence, report):
    """Fit and write the same explicit UTF-8/LF encoding on every platform."""

    def render(value):
        return (json.dumps(value, indent=2) + "\n").encode("utf-8")

    fit_first_shot_diagnostics(
        report,
        owner=report["capture"],
        render=render,
        max_bytes=262144,
    )
    (evidence / "episode.json").write_bytes(render(report))


class _InactiveWorker:
    def __init__(self, **_):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _factories(remote):
    return FieldCloudFactories(
        credentials=remote.credentials_factory,
        transport=remote.transport_factory,
        owner_deletion_transport=remote.deny_transport_factory,
        live_fact_worker=_InactiveWorker,
        saved_media_worker=_InactiveWorker,
        upload_receipt_worker=_InactiveWorker,
        owner_deletion_worker=_InactiveWorker,
    )


def _local_command(builder, args, *, server_port, session_id, resource_context, purpose="analysis"):
    if resource_context is None or purpose not in {"media", "analysis"}:
        raise ValueError("original managed postgame context and purpose required")
    local = copy.copy(args)
    local.sync_base_url = None
    cmd = builder(
        local,
        server_port=server_port,
        session_id=session_id,
        resource_context=resource_context,
        purpose=purpose,
    )
    record = resource_context.spec_record
    expected = {
        "--session-id": session_id,
        "--state-db": str(args.state_db),
        "--jobs-db": str(Path(record.spec.paths["jobs_root"]) / "analysis-jobs.sqlite3"),
        "--artifact-root": str(Path(record.spec.paths["artifact_root"]) / "postgame" / session_id),
        "--purpose": purpose,
        "--managed-open-request-id": record.open_request_id,
        "--managed-resource-spec-sha256": record.spec_sha256,
        "--managed-environment-id": protocol.ENVIRONMENT_ID,
        "--managed-device-id": protocol.DEVICE_ID,
        "--managed-table-id": protocol.TABLE_ID,
        "--live-resource-url": f"http://127.0.0.1:{server_port}/api/screen",
    }
    model_args = {
        "--device": record.spec.device,
        "--batch-size": str(record.spec.batch_size),
        **{
            "--" + n.replace("_", "-"): str(record.spec.paths[n])
            for n in (
                "capture_setup",
                "calibration_root",
                "config",
                "detector_training_run",
                "identity_training_run",
                "dinov2_assets",
            )
        },
    }
    if purpose == "analysis":
        expected.update(model_args)
    elif any(arg.split("=", 1)[0] in model_args for arg in cmd):
        raise ValueError("offline media command must not load model assets")
    if any(arg.split("=", 1)[0] in _FORBIDDEN for arg in cmd) or any(
        cmd.count(flag) != 1 or cmd[cmd.index(flag) + 1 : cmd.index(flag) + 2] != [value]
        for flag, value in expected.items()
    ):
        raise ValueError("invalid offline postgame command")
    return cmd


def _validate(argv, root, budgets):
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= limit
        for v, limit in zip(budgets, _LIMITS, strict=True)
    ):
        raise ValueError("episode budgets must be positive finite values within development limits")
    args = cli._parser().parse_args(argv)
    scope = dict(
        host="127.0.0.1",
        runtime_environment="development",
        managed_sessions=True,
        development_model_acceptance=True,
        identity_gate_status="FAIL",
        rule_profile=cli.PROFILE_ID,
        observation_only=False,
        shutdown_stdin=False,
        sync_base_url=protocol.BASE_URL,
        device_id=protocol.DEVICE_ID,
        table_id=protocol.TABLE_ID,
        credential_target=protocol.CREDENTIAL_TARGET,
        cloudbase_environment_id=protocol.ENVIRONMENT_ID,
        expected_development_environment_id=protocol.ENVIRONMENT_ID,
        control_token=None,
        control_token_file=None,
    )
    if any(getattr(args, k) != v for k, v in scope.items()) or not (
        0 <= args.port <= 65535 and args.batch_size > 0
    ):
        raise ValueError("invalid offline development scope")
    paths = [root, *(v.absolute() for v in vars(args).values() if isinstance(v, Path))]
    # Reject protected spellings across the entire argv before probing any of its paths.
    if any(
        part.casefold() == ".project-archive" or part == ".."
        for path in paths
        for part in path.parts
    ):
        raise ValueError("unsafe episode path")
    for path in paths:
        if any(p.is_symlink() for p in (*reversed(path.parents), path)):
            raise ValueError("unsafe episode path")
    if root.exists() or not root.parent.is_dir():
        raise ValueError("episode needs a fresh root below an existing directory")
    if any(getattr(args, k) != root / "run" / v for k, v in _OUTPUTS.items()):
        raise ValueError("output paths must exactly match the fresh layout")
    for key in ("capture_setup", "config", "scenario_video", "join_qr_asset"):
        path = getattr(args, key)
        if path is not None and not path.is_file():
            raise ValueError("missing read-only input")
    for key in (
        "detector_training_run",
        "identity_training_run",
        "dinov2_assets",
        "calibration_root",
    ):
        path = getattr(args, key).absolute()
        if not path.is_dir() or any(path.samefile(parent) for parent in root.parents):
            raise ValueError("input directory must be separate from the run")
    setup = cli.realtime.load_capture_setup(args.capture_setup)
    if setup.source_kind is not cli.realtime.CaptureSourceKind.FILE or (
        setup.geometry.width,
        setup.geometry.height,
        setup.requested_fps,
    ) != (1920, 1080, 60):
        raise ValueError("black-box capture requires a FILE setup at 1920x1080/60")
    cli.resolve_acceptance_policy(args)
    cli.inspect_managed_runtime_resources(args, rule_profile_id=cli.PROFILE_ID)


class _EpisodeFailure(Exception):
    """Only locally selected public codes reach the evidence file."""


def _read_response(conn, path, token, *, media, payload=None, nonce=None):
    headers = {"X-Control-Token": token}
    if media:
        headers["Range"] = "bytes=0-31"
    if payload is None:
        conn.request("GET", path, headers=headers)
    else:
        headers.update({"Content-Type": "application/json", "X-Kiosk-Nonce": nonce})
        conn.request("POST", path, body=json.dumps(payload).encode(), headers=headers)
    with conn.getresponse() as response:
        if media == "denied":
            return response.status == 404
        if media:
            return (
                response.status == 206
                and response.getheader("Content-Type") == "video/mp4"
                and b"ftyp" in response.read(32)
            )
        if response.status != 200:
            if payload is not None:
                raise _EpisodeFailure("POSTGAME_POST_REJECTED")
            return None
        body = response.read(262145)
        if len(body) > 262144:
            raise _EpisodeFailure("LOOPBACK_SIZE")
        if response.length:
            raise HTTPException("incomplete loopback body")
        data = json.loads(body)
        if payload is not None and not isinstance(data, dict):
            raise _EpisodeFailure("POSTGAME_POST_INVALID")
        return data


def _job_matches(database, shot, media):
    # Opening the queue as a store would run schema initialization. Read just this row.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=0.5)) as db:
        db.row_factory = sqlite3.Row
        _, job = _job(db, media.job_id, shot.session_id, shot.shot_id)
    return (
        job["source_epoch_id"] == media.source_epoch_id
        and job["media_sha256"] == media.media_sha256
        and job["media_root"] == media.media_path
        and job["job_id"] == shot.live_job_id
    )


def _fullgame_job_id(identity):
    """Match ProductRealtimeRuntime._analysis_job's default ASCII JSON encoding."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "job-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fullgame_queue_job(db, job_id, session_id, shot_id):
    """Decode the producer contract without the legacy retention ASCII-ID restriction."""
    from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob
    from scripts.algorithm.offline_fullgame_observer import _integer, _json, _sha

    rows = db.execute(
        "SELECT job_id, substr(payload_json,1,1048577) AS payload_json, "
        "payload_sha256, status, artifact_id FROM shot_analysis_jobs WHERE job_id=? LIMIT 2",
        (job_id,),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("fullgame queue job missing or ambiguous")
    row = rows[0]
    raw = row["payload_json"]
    if (
        not isinstance(raw, str)
        or len(raw.encode("utf-8", errors="strict")) > 1024**2
        or not _sha(row["payload_sha256"])
        or hashlib.sha256(raw.encode("utf-8", errors="strict")).hexdigest() != row["payload_sha256"]
    ):
        raise ValueError("fullgame queue payload hash or size mismatch")
    payload = _json(raw)
    keys = {
        "schema_version",
        "job_id",
        "session_id",
        "shot_id",
        "source_epoch_id",
        "start_frame_id",
        "end_frame_id",
        "media_root",
        "media_sha256",
        "capture_setup_id",
        "calibration_artifact_id",
        "config_sha256",
        "model_package_id",
        "identity_training_package_id",
        "identity_training_package_sha256",
    }
    if (
        type(payload) is not dict
        or set(payload) != keys
        or payload["schema_version"] != "1.0"
        or (row["job_id"], payload["job_id"], payload["session_id"], payload["shot_id"])
        != (job_id, job_id, session_id, shot_id)
        or not all(_integer(payload[k]) for k in ("start_frame_id", "end_frame_id"))
        or payload["start_frame_id"] > payload["end_frame_id"]
        or not all(_sha(payload[k]) for k in ("media_sha256", "config_sha256"))
    ):
        raise ValueError("fullgame queue payload schema or lineage mismatch")
    for key in (
        "job_id",
        "session_id",
        "shot_id",
        "source_epoch_id",
        "media_root",
        "capture_setup_id",
        "calibration_artifact_id",
        "model_package_id",
    ):
        text = payload[key]
        if not isinstance(text, str) or not text or "\x00" in text:
            raise ValueError("fullgame queue identity invalid")
        text.encode("utf-8", errors="strict")
    package, package_sha = (
        payload["identity_training_package_id"],
        payload["identity_training_package_sha256"],
    )
    if (package is None) != (package_sha is None) or (
        package is not None
        and (
            not isinstance(package, str)
            or not package
            or "\x00" in package
            or not _sha(package_sha)
        )
    ):
        raise ValueError("fullgame queue identity package invalid")
    if package is not None:
        package.encode("utf-8", errors="strict")
    fields = {k: v for k, v in payload.items() if k != "schema_version"}
    fields["media_root"] = Path(fields["media_root"])
    if not fields["media_root"].is_absolute() or ".." in fields["media_root"].parts:
        raise ValueError("fullgame queue media root invalid")
    decoded = ShotAnalysisJob(**fields)
    if decoded.identity() != payload:
        raise ValueError("fullgame queue payload identity mismatch")
    return row, payload


def _fullgame_job(database, shot, media, *, rule, spec_record, epoch_record):
    """Read the actual canonical queue identity without opening a mutating job store."""
    from scripts.algorithm.offline_fullgame_observer import _digest, _integer, _json, _plain, _sha

    spec = spec_record.spec
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=0.5)) as db:
        db.row_factory = sqlite3.Row
        row, job = _fullgame_queue_job(db, media.job_id, shot.session_id, shot.shot_id)
    identity_keys = (
        "session_id",
        "shot_id",
        "source_epoch_id",
        "start_frame_id",
        "end_frame_id",
        "media_sha256",
        "capture_setup_id",
        "calibration_artifact_id",
        "config_sha256",
        "model_package_id",
    )
    if not (
        _fullgame_job_id({k: job[k] for k in identity_keys}) == media.job_id
        and job["source_epoch_id"] == media.source_epoch_id
        and job["media_sha256"] == media.media_sha256
        and job["media_root"] == media.media_path
        and job["job_id"] == shot.live_job_id
        and row["status"] in {"pending", "running", "available", "failed_retrying", "blocked"}
        and job["start_frame_id"]
        <= rule.start_frame_id
        < rule.settled_frame_id
        <= job["end_frame_id"]
        and epoch_record.session_id == shot.session_id
        and epoch_record.source_epoch_id == media.source_epoch_id
        and epoch_record.spec_sha256 == spec_record.spec_sha256
        and epoch_record.open_request_id == spec_record.open_request_id
        and job["media_root"]
        == str(Path(spec.paths["jobs_root"]) / "media" / shot.session_id / media.job_id)
        and all(
            job[key] == getattr(spec, key)
            for key in (
                "capture_setup_id",
                "calibration_artifact_id",
                "config_sha256",
                "model_package_id",
                "identity_training_package_id",
                "identity_training_package_sha256",
            )
        )
    ):
        raise ValueError("fullgame job lineage mismatch")
    interval_path = _plain(Path(job["media_root"]) / "interval.json")
    with interval_path.open("rb") as stream:
        interval = _json(stream.read(1024**2 + 1))
    if not (
        all(
            interval[key] == job[key]
            for key in (
                "schema_version",
                "source_epoch_id",
                "start_frame_id",
                "end_frame_id",
                "media_sha256",
            )
        )
        and all(
            _integer(interval[k])
            for k in ("start_frame_id", "end_frame_id", "frame_count", "width", "height")
        )
        and interval["frame_count"] == job["end_frame_id"] - job["start_frame_id"] + 1
        and (interval["width"], interval["height"]) == (1920, 1080)
        and type(interval["segment_sha256"]) is dict
        and interval["segment_sha256"]
        and all(
            re.fullmatch(r"[0-9]{20}-[0-9]{20}\.mjpgseg", name) and _sha(digest)
            for name, digest in interval["segment_sha256"].items()
        )
    ):
        raise ValueError("fullgame interval metadata mismatch")
    content = {
        k: interval[k]
        for k in (
            "schema_version",
            "source_epoch_id",
            "width",
            "height",
            "start_frame_id",
            "end_frame_id",
            "frame_count",
        )
    }
    content["segments"] = [{"name": k, "sha256": v} for k, v in interval["segment_sha256"].items()]
    if _digest(content) != media.media_sha256:
        raise ValueError("fullgame interval digest mismatch")
    return {
        key: job[key]
        for key in (
            "job_id",
            "session_id",
            "shot_id",
            "source_epoch_id",
            "media_sha256",
            "start_frame_id",
            "end_frame_id",
            "capture_setup_id",
            "calibration_artifact_id",
            "config_sha256",
            "model_package_id",
            "identity_training_package_id",
            "identity_training_package_sha256",
        )
    } | {
        "status": row["status"],
        "payload_sha256": row["payload_sha256"],
        "resource_spec_sha256": spec_record.spec_sha256,
    }


class _Actor:
    def __init__(self, remote, report, stop, budgets, database):
        self.remote, self.report, self.stop = remote, report, stop
        self.budgets, self.database = budgets, database
        self.serving = threading.Event()
        self._socket_lock = threading.Lock()
        self._active_socket = None
        self.fullgame_observer = None

    def fullgame_job(self, shot, media):
        from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope

        scope = ManagedSessionScope(protocol.ENVIRONMENT_ID, protocol.DEVICE_ID, protocol.TABLE_ID)
        spec = self.store.get_managed_resource_spec_for_session(scope, self.session)
        epoch = self.store.get_managed_capture_epoch(scope, self.session, self.epoch)
        rule = self.store.get_rule_decision(self.session, shot.shot_id)
        return _fullgame_job(
            self.database, shot, media, rule=rule, spec_record=spec, epoch_record=epoch
        )

    def cancel(self):
        with self._socket_lock:
            self.stop.set()
            if self._active_socket is not None:
                with suppress(OSError):
                    self._active_socket.shutdown(socket.SHUT_RDWR)

    def request(self, path, timeout, *, media=False, payload=None, nonce=None):
        if self.stop.is_set():
            raise _EpisodeFailure("CLI_CANCELLED")
        deadline = time.monotonic() + timeout
        # Loopback connection establishment has its own short bound. Response
        # verification can use the rest of the stage budget and is interruptible.
        with closing(
            HTTPConnection("127.0.0.1", self.server.server_port, timeout=min(timeout, 1.0))
        ) as conn:
            conn.connect()
            # HTTP/1.0 may detach the socket from conn before a body read. Keep
            # an owned handle to the same endpoint until the entire read exits.
            with conn.sock.dup() as interrupt:
                with self._socket_lock:
                    if self.stop.is_set():
                        raise _EpisodeFailure("CLI_CANCELLED")
                    self._active_socket = interrupt
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    conn.sock.settimeout(remaining)
                    return _read_response(
                        conn, path, self.token, media=media, payload=payload, nonce=nonce
                    )
                finally:
                    with self._socket_lock:
                        self._active_socket = None

    def wait(self, probe, deadline, timeout_code):
        while True:
            if self.stop.is_set():
                raise _EpisodeFailure("CLI_CANCELLED")
            if self.remote.snapshot()["failure"]:
                raise _EpisodeFailure("REMOTE_PROTOCOL_FAILURE")
            left = deadline - time.monotonic()
            if left <= 0:
                raise _EpisodeFailure(timeout_code)
            value = probe()
            left = deadline - time.monotonic()
            if left <= 0:
                raise _EpisodeFailure(timeout_code)
            if value:
                return value
            self.stop.wait(min(left, 0.05))

    def observe(self):
        while not self.serving.wait(0.05):
            if self.stop.is_set():
                return
        try:
            self.run()
        except _EpisodeFailure as exc:
            self.report["failure"] = exc.args[0]
            if self.fullgame_observer is not None:
                self.fullgame_observer.fail(exc.args[0])
        except Exception:
            self.report["failure"] = "CLI_CANCELLED" if self.stop.is_set() else "OBSERVER_ERROR"
            if self.fullgame_observer is not None:
                self.fullgame_observer.fail(self.report["failure"])
        finally:
            if not self.stop.is_set():
                # service_actions is called by the actual serve_forever loop.
                # Only command issuance additionally requires a healthy screen.
                self.server.shutdown()

    def health(self, *, ending=False):
        if self.state.session_id != self.session or self.state.projection_source is not self.source:
            raise _EpisodeFailure("SCENE_CHANGED")
        health = self.source.subsystem_health()
        capture = self.report["capture"]
        capture["session_status"] = self.store.load_session(self.session).status.value
        video = health["video_health"]
        capture["video_status"] = video["status"]
        diag = health["analysis_health"]["realtime_diagnostics"] or {}
        if "fullgame" in self.report:
            first = capture.get("first_shot_diagnostics")
            if first is None or first["status"] == "NOT_CAPTURED":
                capture["first_shot_diagnostics"] = project_first_shot_diagnostics(
                    diag.get("first_shot_diagnostics")
                )
        capture["inference"] = {
            k: diag[k]
            for k in (
                "accepted_anchor_frames",
                "anchor_frames_with_balls",
                "tracked_observation_frames",
                "tracked_frames_with_confirmed_balls",
                "motion_threshold_crossings",
                "maximum_frame_change_mm",
            )
            if k in diag
        }
        reason = self.source.failure_reason or video.get("reason") or ""
        if "capture source ended unexpectedly" in reason or "EOF" in reason:
            raise _EpisodeFailure("CAPTURE_EOF")
        if (
            self.source.failure_reason
            or self.source.restart_required
            or (video["status"] != "ready" and (not ending or video["status"] == "degraded"))
        ):
            raise _EpisodeFailure("CAPTURE_DEGRADED")

    def settled(self):
        self.health()
        shots = self.store.list_shots(self.session)
        unsettled = sum(s.settled_ms is None for s in shots)
        self.report["capture"]["unsettled_count"] = unsettled
        if not shots or unsettled:
            return False
        shot = shots[0]
        key = self.session, shot.shot_id
        owner = self.store.get_shot_ownership(*key)
        fact = self.store.get_live_shot_fact(*key)
        rule = self.store.get_rule_decision(*key)
        if not (owner and fact and rule):
            return False
        if not (
            all(
                row.session_id == self.session and row.shot_id == shot.shot_id
                for row in (shot, owner, fact, rule)
            )
            and owner.source_epoch_id == rule.source_epoch_id == self.epoch
            and owner.evidence_status == "bound_at_start"
            and owner.actor_user_id in ("offline-red", "offline-blue")
            and owner.actor_user_id == fact.actor_user_id
            and owner.actor_role == fact.actor_role
            and owner.match_revision == fact.rule_revision
            and rule.start_frame_id == fact.start_frame_id
            and rule.settled_frame_id == fact.settle_frame_id
        ):
            raise _EpisodeFailure("BOUNDARY_MISMATCH")
        frame = self.source.current().frame.payload
        if not (
            frame["source_kind"] == "usb_live"
            and frame["source_epoch_id"] == self.epoch
            and frame["session_id"] == self.session
            and shot.media.source_epoch_id == self.epoch
            and frame["rule_status"] != "DEGRADED"
        ):
            raise _EpisodeFailure("PROJECTION_MISMATCH")
        self.report["boundary"] = dict(
            projection_phase=frame["phase"],
            projection_rule_status=frame["rule_status"],
            shot_id=shot.shot_id,
            owner=owner.actor_user_id,
            start_frame_id=fact.start_frame_id,
            settle_frame_id=fact.settle_frame_id,
            rule_status=rule.status,
            fact_digest=rule.fact_digest,
            decision_digest=rule.decision_digest,
            rule_reason=str(json.loads(rule.decision_json).get("reason_code", ""))[:80],
        )
        media = self.store.get_live_shot_media(*key)
        if not media:
            return False
        if (
            media.source_epoch_id != self.epoch
            or media.match_revision != owner.match_revision
            or media.session_id != self.session
            or media.shot_id != shot.shot_id
            or not _job_matches(self.database, shot, media)
        ):
            raise _EpisodeFailure("MEDIA_JOB_MISMATCH")
        self.report["media"] = dict(job_id=media.job_id, media_sha256=media.media_sha256)
        return True

    def command(self, role, kind, payload, deadline):
        with self.state.match_lock:
            self.health()
            revision = self.store.load_session(self.session).match.revision
            if self.stop.is_set() or self.remote.snapshot()["failure"]:
                raise _EpisodeFailure("COMMAND_CANCELLED")
            if time.monotonic() >= deadline:
                raise _EpisodeFailure("END_TIMEOUT" if kind == "end_session" else "SETUP_TIMEOUT")
            if kind == "end_session" and any(
                shot.settled_ms is None for shot in self.store.list_shots(self.session)
            ):
                raise _EpisodeFailure("SHOT_UNSETTLED")
            command_id = self.remote.issue(
                session_id=self.session,
                actor_user_id="offline-" + role,
                command_type=kind,
                payload=payload,
                expected_revision=revision,
            )
        entry = dict(command=kind, expected_revision=revision, outcome="pending")
        self.report["commands"].append(entry)
        if kind == "end_session":
            self.report["end"] = entry
        receipt = self.wait(
            lambda: self.remote.completion(command_id),
            deadline,
            "END_TIMEOUT" if kind == "end_session" else "SETUP_TIMEOUT",
        )
        entry.update(
            resulting_revision=receipt.resulting_revision,
            result_code=receipt.result_code,
            outcome=receipt.outcome,
        )
        if kind == "end_session":
            entry["session_status"] = self.store.load_session(self.session).status.value
        if receipt.outcome != "applied" or receipt.result_code != "APPLIED":
            raise _EpisodeFailure("COMMAND_REJECTED")

    def review(self):
        deadline = time.monotonic() + self.budgets[3]

        def remaining():
            if self.stop.is_set():
                raise _EpisodeFailure("CLI_CANCELLED")
            if self.remote.snapshot()["failure"]:
                raise _EpisodeFailure("REMOTE_PROTOCOL_FAILURE")
            left = deadline - time.monotonic()
            if left <= 0:
                raise _EpisodeFailure("POSTGAME_TIMEOUT")
            return left

        def read(path, *, media=False):
            try:
                result = self.request(path, remaining(), media=media)
            except (HTTPException, OSError):
                remaining()
                return None
            remaining()
            return result

        def post(path, payload, *, nonce):
            scope = {"session_id", "postgame_manifest_sha256"}
            fields = {
                "/api/postgame-review/save": scope | {"shot_id", "request_id"},
                "/api/postgame-review/complete": scope,
            }
            if (
                set(payload) != fields.get(path)
                or payload["session_id"] != self.session
                or not re.fullmatch(r"[a-f0-9]{64}", payload["postgame_manifest_sha256"])
                or not isinstance(nonce, str)
                or not nonce
                or (
                    "shot_id" in payload
                    and payload["shot_id"] != self.report["boundary"]["shot_id"]
                )
            ):
                raise _EpisodeFailure("POSTGAME_INVALID_INTENT")
            try:
                result = self.request(path, remaining(), payload=payload, nonce=nonce)
            except (HTTPException, OSError, ValueError):
                remaining()
                raise _EpisodeFailure("POSTGAME_POST_UNCERTAIN") from None
            remaining()
            return result

        try:
            run_postgame_selection(
                session_id=self.session,
                shot_id=self.report["boundary"]["shot_id"],
                store=self.store,
                read=read,
                post=post,
                wait=lambda probe: self.wait(probe, deadline, "POSTGAME_TIMEOUT"),
                report=self.report.setdefault("postgame", {}),
            )
            remaining()
        except PostgameSelectionFailure as exc:
            remaining()
            raise _EpisodeFailure(exc.args[0]) from None
        except (HTTPException, OSError, ValueError):
            remaining()
            raise _EpisodeFailure("POSTGAME_READ_FAILED") from None

    def setup(self):
        deadline = self.ready_at + self.budgets[0]

        def screen():
            try:
                return self.request("/api/screen", 1.0)
            except (HTTPException, OSError):
                return None

        self.wait(screen, deadline, "SCREEN_TIMEOUT")
        self.state = self.server.field_state
        self.store, self.source = self.state.record_store, self.state.projection_source
        self.session, self.epoch = self.state.session_id, self.source.source_epoch_id
        self.report.update(session_id=self.session, source_epoch_id=self.epoch)
        session = self.request("/api/local/session", 1.0)
        if not session or session.get("session_id") != self.session:
            raise _EpisodeFailure("SESSION_READ_MISMATCH")
        self.health()
        capture = self.source.runtime.capture_diagnostics
        for key in ("source_kind", "source_epoch_id", "source_sha256", "width", "height", "fps"):
            self.report["capture"][key] = capture[key]
        acceptance = capture.get("acceptance", {})
        self.report["capture"]["identity_training_package_sha256"] = acceptance.get(
            "identity_training_package_sha256"
        )
        if (
            capture["source_epoch_id"] != self.epoch
            or capture["source_kind"] != "blackbox_video"
            or acceptance.get("identity_gate_status") != "FAIL"
            or self.gate.is_released()
            or self.source.runtime.current_capture_frame is None
            or self.remote.snapshot()["birth_count"] != 1
        ):
            raise _EpisodeFailure("READY_MISMATCH")
        self.report["setup"].update(
            held_on_ready=True,
            initial_revision=self.store.load_session(self.session).match.revision,
        )
        for role in ("red", "blue"):
            self.command(role, "claim_role", {"role": role}, deadline)
        settings = dict(
            role="red",
            mode="assistance",
            format="single",
            race_target=None,
            ranking_intent=False,
            settings_revision=1,
        )
        self.command("red", "set_breaker", settings, deadline)
        match = self.store.load_session(self.session).match
        if not (
            self.gate.is_released()
            and match.settings_frozen
            and (match.red_user_id, match.blue_user_id) == ("offline-red", "offline-blue")
            and match.settings_revision == 1
            and match.breaker.value == "red"
            and match.race_target is None
            and not match.ranking_intent
            and match.mode.value == "assistance"
            and match.match_format.value == "single"
        ):
            raise _EpisodeFailure("SETUP_MISMATCH")
        self.report["setup"].update(
            gate_released=True,
            settings_revision=match.settings_revision,
            red_actor=match.red_user_id,
            blue_actor=match.blue_user_id,
            breaker=match.breaker.value,
            mode=match.mode.value,
            format=match.match_format.value,
        )

    def run(self):
        self.setup()
        if self.fullgame_observer is not None:
            from scripts.algorithm.offline_fullgame_observer import FullgameFailure

            try:
                self.fullgame_observer.run()
            except FullgameFailure as exc:
                raise _EpisodeFailure(exc.args[0]) from None
            return
        self.wait(self.settled, time.monotonic() + self.budgets[1], "SHOT_TIMEOUT")
        self.command("red", "end_session", {}, time.monotonic() + self.budgets[2])
        self.health(ending=True)
        if self.report["end"]["session_status"] != "ended":
            raise _EpisodeFailure("END_NOT_OBSERVED")
        self.review()


def run_episode(
    argv: list[str],
    *,
    output_root: Path,
    setup_seconds=60.0,
    shot_seconds=45.0,
    end_seconds=10.0,
    postgame_seconds=300.0,
    observer_mode="single-shot-v1",
    input_oracle=None,
    input_oracle_sha256=None,
    match_seconds=600.0,
) -> int:
    if threading.current_thread() is not threading.main_thread() or not _ACTIVE.acquire(
        blocking=False
    ):
        raise RuntimeError("episode needs exclusive main-thread ownership")
    global _RETAINED
    release_owner = True
    try:
        budgets = setup_seconds, shot_seconds, end_seconds, postgame_seconds
        root = Path(output_root).absolute()
        _validate(argv, root, budgets)
        if observer_mode not in {"single-shot-v1", "fullgame-v1"}:
            raise ValueError("unknown offline observer mode")
        oracle = None
        if observer_mode == "fullgame-v1":
            from scripts.algorithm.offline_fullgame_observer import (
                MAX_MATCH_SECONDS,
                load_oracle,
            )

            if (
                type(match_seconds) not in (int, float)
                or not math.isfinite(match_seconds)
                or not 0 < match_seconds <= MAX_MATCH_SECONDS
            ):
                raise ValueError("invalid fullgame match budget")
            oracle = load_oracle(
                input_oracle, input_oracle_sha256, cli._parser().parse_args(argv).scenario_video
            )
        elif input_oracle is not None or input_oracle_sha256 is not None:
            raise ValueError("input oracle requires fullgame-v1 observer")
        root.mkdir(mode=0o700)
        run = root / "run"
        run.mkdir()
        for path in ("ring", "jobs", "artifacts"):
            (run / path).mkdir()
        evidence = root / "evidence"
        evidence.mkdir()
        remote = protocol.OfflineManagedRemote(observer_mode=observer_mode)
        report = dict(
            synthetic_actors=["offline-red", "offline-blue"],
            synthetic_remote=True,
            formal_model_qualified=False,
            formal_field_acceptance="NOT_ASSERTED",
            identity_gate_status="FAIL",
            complete_game_acceptance=False,
            provider_delivery="NOT_ASSERTED",
            setup={},
            capture={},
            boundary={},
            media={},
            end={},
            postgame={},
            commands=[],
            failure=None,
        )
        stop = threading.Event()
        actor = _Actor(remote, report, stop, budgets, run / "jobs/analysis-jobs.sqlite3")
        if oracle is not None:
            from scripts.algorithm.offline_fullgame_observer import FullgameObserver

            actor.fullgame_observer = FullgameObserver(actor, oracle, match_seconds=match_seconds)
        observer = threading.Thread(target=actor.observe, name="offline-managed-observer")
        prepare, build, command = (
            cli.prepare_cloud_runtime,
            cli.build_live_field_assistance_server,
            cli._postgame_worker_command,
        )
        exit_code = runtime = server = None

        def local_command(args, session_id, **kwargs):
            if stop.is_set():
                raise RuntimeError("offline postgame invocation has ended")
            return _local_command(command, args, session_id=session_id, **kwargs)

        def prepare_offline(args):
            nonlocal runtime
            factories = _factories(remote)

            def postgame_worker(**kwargs):
                # The CLI's callback looks up a module global at call time. A resolver
                # may finish after scope restoration. Bind this worker to its own run.
                kwargs["command_factory"] = partial(
                    local_command, args, server_port=server.server_port
                )
                return factories.postgame_worker(**kwargs)

            runtime = prepare(args, factories=replace(factories, postgame_worker=postgame_worker))
            return runtime

        with ExitStack() as scope:

            def forwarding_builder(**kwargs):
                nonlocal server
                server = build(**kwargs)
                actor.server, actor.token, actor.gate = (
                    server,
                    kwargs["control_token"],
                    kwargs["scenario_playback_gate"],
                )
                original = server.service_actions

                def service_actions():
                    original()
                    if not actor.serving.is_set():
                        actor.ready_at = time.monotonic()
                        actor.serving.set()

                scope.enter_context(patch.object(server, "service_actions", service_actions))
                return server

            scope.enter_context(
                patch.object(
                    cli,
                    "prepare_cloud_runtime",
                    prepare_offline,
                )
            )
            scope.enter_context(
                patch.object(cli, "build_live_field_assistance_server", forwarding_builder)
            )
            scope.enter_context(
                patch.object(
                    cli,
                    "_postgame_worker_command",
                    local_command,
                )
            )
            observer.start()
            release_owner = False
            try:
                exit_code = cli.main(argv)
            finally:
                actor.cancel()
                observer.join(timeout=10.0)
                if observer.is_alive():
                    report["failure"] = "OBSERVER_CLEANUP_TIMEOUT"
                report["remote"] = remote.snapshot()
                # stop() removes a real worker only after its stop returns. The
                # postgame stop includes the supervisor join and child process wait.
                workers_stopped = runtime is not None and runtime._stopped and not runtime._workers
                server_closed = (
                    server is not None
                    and getattr(server, "fileno", lambda: None)() == -1
                    and server.field_state.projection_lifecycle_closed
                )
                release_owner = (
                    exit_code == 0 and not observer.is_alive() and workers_stopped and server_closed
                )
                if not release_owner:
                    _RETAINED = runtime, server, observer
                report["cleanup"] = dict(
                    observer_joined=not observer.is_alive(),
                    cli_exit=exit_code,
                    workers_stopped=workers_stopped,
                    server_closed=server_closed,
                )

                if not report["failure"]:
                    if report["remote"]["failure"]:
                        report["failure"] = "REMOTE_PROTOCOL_FAILURE"
                    elif not release_owner or report["postgame"].get("selection", {}).get(
                        "cleanup"
                    ) not in {"VERIFIED", "NOT_EXERCISED"}:
                        report["failure"] = "CLI_INCOMPLETE"
                if oracle is not None and report["failure"] is not None:
                    actor.fullgame_observer.report["status"] = "FAILED"
                    actor.fullgame_observer.fail(report["failure"])
                _write_first_shot_diagnostic(evidence, report["capture"])
                _write_episode_report(evidence, report)
        return 0 if not report["failure"] else (exit_code or 2)
    finally:
        if release_owner:
            _ACTIVE.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--observer-mode", choices=("single-shot-v1", "fullgame-v1"), default="single-shot-v1"
    )
    parser.add_argument("--input-oracle", type=Path)
    parser.add_argument("--input-oracle-sha256")
    parser.add_argument("--match-seconds", type=float, default=600.0)
    args, rest = parser.parse_known_args(argv)
    try:
        return run_episode(
            rest,
            output_root=args.output_root,
            observer_mode=args.observer_mode,
            input_oracle=args.input_oracle,
            input_oracle_sha256=args.input_oracle_sha256,
            match_seconds=args.match_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
