#!/usr/bin/env python3
"""Consume one ended USB session through the isolated Mainline 1 worker."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    UrllibSyncHttpTransport,
    WindowsCredentialManagerProvider,
)
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    FieldCloudGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.cloud_runtime import validate_development_cloud_scope
from billiards_ai.lab.field_assistance.live_source import session_has_live_activity
from billiards_ai.lab.field_assistance.managed_resource_recovery import (
    resolve_managed_session_resources,
)
from billiards_ai.lab.field_assistance.runtime_resources import verify_managed_runtime_resources
from billiards_ai.lab.mainline1_calibration_registry import (
    resolve_active_calibration,
    resolve_calibration_record,
)
from billiards_ai.lab.mainline1_capture import capture_setup_fingerprint, load_capture_setup
from billiards_ai.lab.mainline1_identity_backends import (
    Dinov2IdentityFeatureBackend,
    OpenCvIdentityVisualEvidenceBackend,
)
from billiards_ai.lab.mainline1_runtime import (
    load_development_group_classifier,
    load_development_vision_backend,
)
from billiards_ai.lab.mainline1_session_analysis import (
    DevelopmentSessionAnalysisPublisher,
    SQLiteSessionAnalysisOutbox,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJobStatus,
    SQLiteShotAnalysisJobStore,
)
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.lab.mainline1_shot_worker import (
    LiveResourceGate,
    Mainline1RecordingMediaWorker,
    Mainline1ShotWorker,
    Mainline1ShotWorkerDependencies,
    MediaEncodingResourceGate,
    SQLiteLiveResourceGate,
)
from billiards_ai.shared.calibration.homography import TableCalibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--jobs-db", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--purpose", choices=("analysis", "media"), default="analysis")
    parser.add_argument("--capture-setup", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--detector-training-run", type=Path)
    parser.add_argument("--identity-training-run", type=Path)
    parser.add_argument("--dinov2-assets", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-empty-polls", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--server-exit-receipt", type=Path)
    parser.add_argument("--live-resource-url")
    parser.add_argument("--live-resource-audit", type=Path)
    parser.add_argument("--live-server-pid", type=int)
    parser.add_argument("--development-analysis-sync", action="store_true")
    parser.add_argument("--sync-base-url")
    parser.add_argument("--device-id")
    parser.add_argument("--credential-target")
    parser.add_argument("--development-cloudbase-environment-id")
    for name in (
        "environment-id",
        "device-id",
        "table-id",
        "open-request-id",
        "resource-spec-sha256",
    ):
        parser.add_argument("--managed-" + name)
    return parser


def _development_analysis_gateway(args: argparse.Namespace) -> FieldCloudGateway | None:
    values = (args.sync_base_url, args.device_id, args.credential_target)
    configured = sum(bool(value) for value in values)
    if configured not in {0, 3}:
        raise ValueError("development analysis sync requires the complete sync configuration")
    if not args.development_analysis_sync:
        if configured or args.development_cloudbase_environment_id:
            raise ValueError("analysis sync configuration requires explicit development opt-in")
        return None
    if configured != 3 or not args.development_cloudbase_environment_id:
        raise ValueError("development analysis sync requires the complete sync configuration")
    gateway = DevelopmentAttestedFieldCloudGateway(
        delegate=SignedFieldCloudGateway(
            credentials=WindowsCredentialManagerProvider(
                device_id=args.device_id,
                target_name=args.credential_target,
            ),
            transport=UrllibSyncHttpTransport(args.sync_base_url, allow_redirects=False),
        ),
        expected_cloudbase_environment_id=args.development_cloudbase_environment_id,
    )
    # Resolve the authenticated read-only assertion before constructing a worker
    # that can make any session-analysis mutation.
    gateway.runtime_attestation()
    return gateway


def _development_field_sync_client(
    args: argparse.Namespace, gateway: FieldCloudGateway | None
) -> FieldSyncClient | None:
    if gateway is None:
        return None
    credentials = WindowsCredentialManagerProvider(
        device_id=args.device_id,
        target_name=args.credential_target,
    )
    return FieldSyncClient(
        credentials=credentials,
        transport=UrllibSyncHttpTransport(args.sync_base_url, allow_redirects=False),
        before_mutation=gateway.runtime_attestation,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"development server receipt {name} is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"development server receipt {name} has no timezone")
    return parsed.astimezone(UTC)


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Query a Windows process handle without sending any signal to it."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # Access denied is treated as alive. Only invalid/nonexistent PIDs are absent.
        return ctypes.get_last_error() not in {87, 1168}
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _probe_live_screen(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=0.75) as response:
        if response.status != 200:
            raise OSError(f"live resource endpoint returned HTTP {response.status}")
        payload = json.loads(response.read(1024 * 1024))
    if not isinstance(payload, dict):
        raise ValueError("live resource endpoint returned a non-object")
    return payload


class AuditedLivePriorityGate(LiveResourceGate):
    """Allow ended-session work only while the live service proves idle state."""

    def __init__(
        self,
        *,
        records: SQLiteFieldRecordStore,
        target_session_id: str,
        live_url: str,
        server_pid: int,
        audit_path: Path,
        probe: Callable[[str], dict[str, Any]] = _probe_live_screen,
        pid_is_alive: Callable[[int], bool] = _pid_is_alive,
        now_utc: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        parsed = urllib.parse.urlsplit(live_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path != "/api/screen"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("live resource URL must be the loopback /api/screen endpoint")
        if server_pid <= 0:
            raise ValueError("live server PID is invalid")
        if not target_session_id.strip():
            raise ValueError("live resource target session is missing")
        if audit_path.is_symlink() or audit_path.name != (
            f"live-resource-{target_session_id}.jsonl"
        ):
            raise ValueError("live resource audit path is unsafe")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = records
        self.target_session_id = target_session_id
        self.live_url = live_url
        self.server_pid = server_pid
        self.audit_path = audit_path
        self.probe = probe
        self.pid_is_alive = pid_is_alive
        self.now_utc = now_utc

    def _record(self, *, allowed: bool, reason: str, live_session_id: str | None) -> bool:
        payload = {
            "schema_version": "1.0",
            "observed_at_utc": self.now_utc().astimezone(UTC).isoformat(),
            "target_session_id": self.target_session_id,
            "live_session_id": live_session_id,
            "server_pid": self.server_pid,
            "allowed": allowed,
            "reason": reason,
        }
        try:
            with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            return False
        return allowed

    def may_run_heavy_analysis(self) -> bool:
        try:
            target = self.records.load_session(self.target_session_id)
        except (OSError, RuntimeError, ValueError):
            return self._record(
                allowed=False,
                reason="target_session_unverifiable",
                live_session_id=None,
            )
        if (
            target.session_id != self.target_session_id
            or target.status is not FieldSessionStatus.ENDED
        ):
            return self._record(
                allowed=False,
                reason="target_session_not_ended",
                live_session_id=None,
            )
        if not self.pid_is_alive(self.server_pid):
            return self._record(
                allowed=False,
                reason="live_server_not_running",
                live_session_id=None,
            )
        try:
            payload = self.probe(self.live_url)
            session = payload.get("session")
            playback = payload.get("playback")
            frame = payload.get("frame")
            display = payload.get("display")
            if not all(isinstance(item, dict) for item in (session, playback, frame, display)):
                raise ValueError("live resource payload is incomplete")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            return self._record(
                allowed=False,
                reason="live_resource_probe_failed",
                live_session_id=None,
            )
        live_session_id = session.get("session_id")
        if live_session_id == self.target_session_id and session.get("status") == "ended":
            # Managed sessions stay on their original screen after ending. They
            # must not need a new match merely to produce this match's clips.
            # A persisted new match takes priority even if this screen is stale.
            try:
                no_live_match = self.records.has_open_session() is False
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                no_live_match = False
            idle = (
                no_live_match
                and frame.get("session_id") == self.target_session_id
                and frame.get("source_kind") == "usb_live"
                and playback.get("playing") is False
                and frame.get("phase") == "stable"
                and frame.get("rule_status") == "SESSION_ENDED"
                and display.get("phase") == "SESSION_ENDED"
                and frame.get("guidance_visible") is False
            )
            return self._record(
                allowed=idle,
                reason="ended_session_idle" if idle else "ended_session_not_idle",
                live_session_id=live_session_id,
            )
        if (
            not isinstance(live_session_id, str)
            or not live_session_id.strip()
            or live_session_id == self.target_session_id
            or session.get("status") != "open"
            or frame.get("session_id") != live_session_id
            or frame.get("source_kind") != "usb_live"
        ):
            return self._record(
                allowed=False,
                reason="live_session_identity_invalid",
                live_session_id=(live_session_id if isinstance(live_session_id, str) else None),
            )
        if (
            playback.get("playing") is not True
            or frame.get("rule_status") == "DEGRADED"
            or display.get("phase") == "DEGRADED"
        ):
            return self._record(
                allowed=False,
                reason="live_capture_not_healthy",
                live_session_id=live_session_id,
            )
        if (
            frame.get("phase") != "stable"
            or frame.get("guidance_visible") is not False
            or display.get("phase") != "WAITING_FOR_SHOT"
        ):
            return self._record(
                allowed=False,
                reason="live_route_or_capture_busy",
                live_session_id=live_session_id,
            )
        return self._record(
            allowed=True,
            reason="live_capture_healthy_and_idle",
            live_session_id=live_session_id,
        )


class AuditedMediaPriorityGate(AuditedLivePriorityGate):
    """Media-only admission; an empty rollover is idle, ready/started play is not."""

    def may_run_heavy_analysis(self) -> bool:
        try:
            target = self.records.load_session(self.target_session_id)
            if target.status is not FieldSessionStatus.ENDED or not self.pid_is_alive(
                self.server_pid
            ):
                return False
            payload = self.probe(self.live_url)
            session = payload.get("session", {})
            live_session_id = session.get("session_id")
            if live_session_id == self.target_session_id:
                return super().may_run_heavy_analysis()
            frame = payload.get("frame", {})
            current = self.records.load_session(live_session_id)
            allowed = bool(
                session.get("status") == "open"
                and current.status is FieldSessionStatus.OPEN
                and self.records.get_session_policy(live_session_id) == "recording-v1"
                and not session_has_live_activity(self.records, live_session_id)
                and payload.get("media_encoding_idle") is True
                and payload.get("playback", {}).get("playing") is True
                and frame.get("session_id") == live_session_id
                and frame.get("source_kind") == "usb_live"
                and frame.get("phase") == "stable"
                and frame.get("rule_status") != "DEGRADED"
                and frame.get("guidance_visible") is False
                and payload.get("display", {}).get("phase") != "DEGRADED"
            )
            return self._record(
                allowed=allowed,
                reason=(
                    "recording_rollover_idle" if allowed else "recording_match_active_or_unhealthy"
                ),
                live_session_id=live_session_id,
            )
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError, sqlite3.Error):
            return False


def _development_offline_gate(
    records: SQLiteFieldRecordStore,
    state_db: Path,
    session_id: str,
    receipt_path: Path,
) -> LiveResourceGate:
    """Verify a stopped packet-owned server before bypassing only the live-open probe."""

    if receipt_path.is_symlink() or receipt_path.name != "server-exit.json":
        raise ValueError("development server exit receipt path is unsafe")
    receipt = receipt_path.resolve(strict=True)
    work = receipt.parent
    result = work.parent
    if work.name != "work" or result == work:
        raise ValueError("development server exit receipt is outside a result work root")
    run_state_path = work / "run-state.json"
    if run_state_path.is_symlink() or not run_state_path.is_file():
        raise ValueError("development run state is missing or unsafe")
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    exit_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    resolved_state_db = state_db.resolve(strict=True)
    expected_result = str(result.resolve())
    expected_database = str(resolved_state_db)
    for payload in (run_state, exit_receipt):
        if payload.get("result_root") != expected_result:
            raise ValueError("development receipt result root conflicts with the packet run")
        if payload.get("state_db") != expected_database:
            raise ValueError("development receipt state DB conflicts with the worker")
    pid = exit_receipt.get("server_pid")
    if (
        not isinstance(pid, int)
        or pid <= 0
        or run_state.get("server_pid") != pid
        or _pid_is_alive(pid)
    ):
        raise ValueError("development field server is not verifiably stopped")
    if exit_receipt.get("ready_observed") is not True or exit_receipt.get("exit_code") != 0:
        raise ValueError("development field server exit was not clean and ready")
    if _timestamp(exit_receipt.get("stopped_at_utc"), "stop time") <= _timestamp(
        run_state.get("started_at_utc"), "start time"
    ):
        raise ValueError("development field server exit predates the packet run")
    session = records.load_session(session_id)
    if session.session_id != session_id or session.status is not FieldSessionStatus.ENDED:
        raise ValueError("development offline worker requires the exact ended session")
    return LiveResourceGate()


def _wait_for_initial_resource_gate(
    gate: LiveResourceGate,
    *,
    once: bool,
    max_empty_polls: int,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Keep model construction behind the same audited live-priority gate as inference."""

    denied_polls = 0
    while not gate.may_run_heavy_analysis():
        denied_polls += 1
        if once or denied_polls >= max_empty_polls:
            return False
        sleep(poll_seconds)
    return True


def _managed_context(args):
    """Refuse incomplete or substituted context before config/models/providers/jobs init."""
    names = ("environment_id", "device_id", "table_id", "open_request_id", "resource_spec_sha256")
    values = tuple(getattr(args, "managed_" + name) for name in names)
    if any(value is not None for value in values) and not all(values):
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    # The field database must already exist; this is not a bootstrap entrypoint.
    if not args.state_db.is_file() or args.state_db.is_symlink():
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    records = SQLiteFieldRecordStore(args.state_db)
    if not all(values):
        if records.has_managed_session_identity(args.session_id):
            raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
        return records, None
    scope = ManagedSessionScope(*values[:3])
    # Admit the complete supplied context using strict persisted metadata only.
    # Full analysis resolution also parses config/model assets, so it must wait
    # until request/spec, original arguments and ephemeral scope are checked.
    record = records.get_managed_resource_spec_for_session(scope, args.session_id)
    spec = record.spec
    if (record.open_request_id, record.spec_sha256) != values[3:]:
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    expected = dict(
        jobs_db=Path(spec.paths["jobs_root"]) / "analysis-jobs.sqlite3",
        artifact_root=Path(spec.paths["artifact_root"]) / "postgame" / args.session_id,
    )
    if args.purpose == "analysis":
        expected.update(
            {
                name: Path(spec.paths[name])
                for name in (
                    "capture_setup",
                    "calibration_root",
                    "config",
                    "detector_training_run",
                    "identity_training_run",
                    "dinov2_assets",
                )
            }
        )
    if any(getattr(args, name) != value for name, value in expected.items()) or (
        args.purpose == "analysis"
        and (args.device, args.batch_size, args.fps) != (spec.device, spec.batch_size, spec.fps)
    ):
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    if args.purpose == "media" and (
        args.development_analysis_sync
        or any(
            (
                args.sync_base_url,
                args.device_id,
                args.credential_target,
                args.development_cloudbase_environment_id,
            )
        )
    ):
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    if args.development_analysis_sync and (
        args.device_id != scope.device_id
        or args.development_cloudbase_environment_id != scope.environment_id
    ):
        raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
    if args.development_analysis_sync or any(
        (
            args.sync_base_url,
            args.device_id,
            args.credential_target,
            args.development_cloudbase_environment_id,
        )
    ):
        if not args.development_analysis_sync:
            raise ValueError("POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE")
        validate_development_cloud_scope(
            SimpleNamespace(
                managed_sessions=True,
                table_id=scope.table_id,
                runtime_environment="development",
                sync_base_url=args.sync_base_url,
                device_id=args.device_id,
                credential_target=args.credential_target,
                cloudbase_environment_id=args.development_cloudbase_environment_id,
                expected_development_environment_id=scope.environment_id,
            )
        )
    resolved = resolve_managed_session_resources(
        records,
        scope,
        args.session_id,
        purpose="cleanup" if records.is_postgame_work_fenced(args.session_id) else args.purpose,
    )
    return records, resolved


def _resource_gate(args, records):
    gate: LiveResourceGate = SQLiteLiveResourceGate(records)
    live_gate_values = (
        args.live_resource_url,
        args.live_resource_audit,
        args.live_server_pid,
    )
    if args.server_exit_receipt is not None and any(
        value is not None for value in live_gate_values
    ):
        raise ValueError("offline and live resource gates are mutually exclusive")
    if any(value is not None for value in live_gate_values):
        if not all(value is not None for value in live_gate_values):
            raise ValueError("live resource gate requires URL, audit path, and server PID")
        gate_type = AuditedMediaPriorityGate if args.purpose == "media" else AuditedLivePriorityGate
        gate = gate_type(
            records=records,
            target_session_id=args.session_id,
            live_url=args.live_resource_url,
            server_pid=args.live_server_pid,
            audit_path=args.live_resource_audit,
        )
    if args.server_exit_receipt is not None:
        gate = _development_offline_gate(
            records,
            args.state_db,
            args.session_id,
            args.server_exit_receipt,
        )
    return gate


class _MediaResourceGate(MediaEncodingResourceGate):
    def __init__(self, gate: LiveResourceGate) -> None:
        self._gate = gate

    def may_run_media_encoding(self, session_id: str) -> bool:
        return self._gate.may_run_heavy_analysis()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis_paths = (
            args.capture_setup,
            args.calibration_root,
            args.config,
            args.detector_training_run,
            args.identity_training_run,
            args.dinov2_assets,
            args.fps,
        )
        if args.purpose == "analysis" and any(value is None for value in analysis_paths):
            raise ValueError("analysis worker requires complete model resources")
        records, managed = _managed_context(args)
        fenced = getattr(records, "is_postgame_work_fenced", None)
        if callable(fenced) and fenced(args.session_id):
            print("[]")
            return 0
        spec = None if managed is None else managed.spec_record.spec
        resource_gate = _resource_gate(args, records)
        if not _wait_for_initial_resource_gate(
            resource_gate,
            once=args.once,
            max_empty_polls=args.max_empty_polls,
            poll_seconds=args.poll_seconds,
        ):
            print("[]")
            return 3 if args.purpose == "media" else 0
        if args.purpose == "media":
            jobs = SQLiteShotAnalysisJobStore(args.jobs_db)
            worker = Mainline1RecordingMediaWorker(
                records=records,
                jobs=jobs,
                artifact_root=args.artifact_root,
                resource_gate=_MediaResourceGate(resource_gate),
            )
            results = []
            empty_polls = 0
            while True:
                result = worker.process_one(args.worker_id, args.session_id, datetime.now(UTC))
                if result is not None:
                    results.append(result)
                    empty_polls = 0
                else:
                    empty_polls += 1
                if args.once or empty_polls >= args.max_empty_polls:
                    break
                time.sleep(args.poll_seconds)
            print(
                json.dumps(
                    [
                        {"job_id": result.job_id, "artifact_id": result.artifact_id}
                        for result in results
                    ],
                    sort_keys=True,
                )
            )
            scheduled = worker.schedule(args.session_id)
            complete = all(
                (snapshot := jobs.snapshot(job.job_id)).status is ShotAnalysisJobStatus.AVAILABLE
                or snapshot.terminal_failure
                for job in scheduled
            )
            return 0 if complete else 3
        config = load_mainline1_shot_local_config(
            args.config, **({} if spec is None else {"expected_sha256": spec.config_sha256})
        )
        capture_setup = load_capture_setup(args.capture_setup)
        registry = (
            resolve_active_calibration(
                args.calibration_root, capture_setup_fingerprint(capture_setup)
            )
            if spec is None
            else resolve_calibration_record(
                args.calibration_root,
                setup_fingerprint=spec.setup_fingerprint,
                artifact_id=spec.calibration_artifact_id,
                revision=spec.calibration_revision,
            )
        )
        if registry is None:
            raise ValueError("calibration registry has no active setup record")
        calibration_path = (
            args.calibration_root.resolve() / registry.artifact_path / "calibration.json"
        )
        if calibration_path.is_symlink():
            raise ValueError("registered calibration artifact is unsafe")
        calibration = TableCalibration.model_validate_json(calibration_path.read_bytes())
        if spec is not None:
            verify_managed_runtime_resources(spec, purpose="analysis")

        import onnxruntime
        from rfdetr import from_checkpoint

        vision = load_development_vision_backend(
            args.detector_training_run, model_factory=lambda path: from_checkpoint(str(path))
        )
        identity = load_development_group_classifier(
            args.identity_training_run,
            feature_backend=Dinov2IdentityFeatureBackend(
                asset_root=args.dinov2_assets, device=args.device, batch_size=args.batch_size
            ),
            evidence_backend=OpenCvIdentityVisualEvidenceBackend(),
            session_factory=lambda path: onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            ),
        )
        jobs = SQLiteShotAnalysisJobStore(args.jobs_db)
        analysis_gateway = _development_analysis_gateway(args)
        analysis_publisher = DevelopmentSessionAnalysisPublisher(
            records=records,
            jobs=jobs,
            outbox=SQLiteSessionAnalysisOutbox(args.jobs_db),
            gateway=analysis_gateway,
            sync_client=_development_field_sync_client(args, analysis_gateway),
            artifact_root=args.artifact_root,
        )
        worker = Mainline1ShotWorker(
            records=records,
            jobs=jobs,
            artifact_root=args.artifact_root,
            session_analysis_publisher=analysis_publisher,
            dependencies=Mainline1ShotWorkerDependencies(
                calibration=calibration,
                config=config,
                vision_backend=vision,
                identity_classifier=identity,
                fps=args.fps,
                capture_setup_id=capture_setup.capture_setup_id,
                calibration_artifact_id=registry.artifact_id,
                config_sha256=_sha256(args.config) if spec is None else spec.config_sha256,
                model_package_id=vision.package_identity.package_id,
                identity_training_package_id=args.identity_training_run.resolve().name,
                identity_training_package_sha256=(
                    spec.identity_training_package_sha256
                    if spec is not None
                    else _sha256(args.identity_training_run / "manifest.json")
                ),
                resource_gate=resource_gate,
            ),
        )
        results = []
        empty_polls = 0
        while True:
            result = worker.process_one(args.worker_id, args.session_id, datetime.now(UTC))
            if result is not None:
                results.append(result)
                empty_polls = 0
            else:
                empty_polls += 1
            if args.once or empty_polls >= args.max_empty_polls:
                break
            time.sleep(args.poll_seconds)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            [{"job_id": result.job_id, "artifact_id": result.artifact_id} for result in results],
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
