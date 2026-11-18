#!/usr/bin/env python3
"""Serve one verified route replay to a projector screen and control tablet."""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    UrllibSyncHttpTransport,
    WindowsCredentialManagerProvider,
)
from billiards_ai.lab.field_assistance.cloud_gateway import SignedFieldCloudGateway
from billiards_ai.lab.field_assistance.mainline1_replay import load_mainline1_replay_inputs
from billiards_ai.lab.field_assistance.server import (
    _BackgroundFieldCommandWorker,
    build_field_assistance_server,
)


class _BackgroundSyncWorker:
    def __init__(
        self,
        *,
        store,
        client: FieldSyncClient,
        wake: threading.Event,
        recover_missing_jobs: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._client = client
        self._wake = wake
        self._recover_missing_jobs = recover_missing_jobs or (lambda: None)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="field-sync")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stopped.is_set():
            job = None
            try:
                self._recover_missing_jobs()
                now = datetime.now(UTC)
                job = self._store.claim_due_sync_job(now)
                if job is None:
                    self._wake.wait(timeout=10.0)
                    self._wake.clear()
                    continue
                result = self._client.run_once(job)
                if result.outcome == "complete":
                    self._store.mark_sync_complete(job.sync_job_id)
                elif result.outcome == "retry":
                    delay = min(300, 2 ** min(job.retry_count + 1, 8))
                    self._store.mark_sync_partial(
                        job.sync_job_id,
                        error_code=result.error_code or "SYNC_RETRY",
                        next_attempt_at=now + timedelta(seconds=delay),
                    )
                else:
                    self._store.mark_sync_blocked(
                        job.sync_job_id,
                        error_code=result.error_code or "SYNC_BLOCKED",
                    )
            except Exception:
                if job is not None:
                    delay = min(300, 2 ** min(job.retry_count + 1, 8))
                    try:
                        self._store.mark_sync_partial(
                            job.sync_job_id,
                            error_code="SYNC_WORKER_EXCEPTION",
                            next_attempt_at=now + timedelta(seconds=delay),
                        )
                    except Exception:
                        pass
                self._wake.wait(timeout=0.25)
                self._wake.clear()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"field route shell arguments invalid: {message}", file=sys.stderr)
        raise SystemExit(64)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--video", type=Path, help="Optional original video for shot ownership evidence"
    )
    parser.add_argument(
        "--mainline1-artifact",
        type=Path,
        help="Optional explicit source artifact; pair with --video",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_port, default=8787)
    parser.add_argument("--display-host")
    parser.add_argument("--control-token")
    parser.add_argument("--state-db", required=True, type=Path)
    parser.add_argument("--sync-base-url")
    parser.add_argument("--device-id")
    parser.add_argument("--credential-target")
    parser.add_argument("--join-qr-asset", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _port(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return value


def _discover_display_host(bind_host: str) -> str:
    if bind_host not in {"0.0.0.0", "::"}:
        return bind_host
    try:
        candidate = socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"
    return candidate if candidate and candidate != "0.0.0.0" else "127.0.0.1"


def _validate_display_host(value: str) -> str:
    candidate = value.strip()
    if not candidate or "://" in candidate or "/" in candidate or "?" in candidate:
        raise ValueError("display host must be a host name or IPv4 address")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    token = arguments.control_token or secrets.token_urlsafe(32)
    server = None
    worker = None
    command_worker = None
    try:
        display_host = _validate_display_host(
            arguments.display_host or _discover_display_host(arguments.host)
        )
        cloud_gateway = None
        sync_client = None
        sync_wake = threading.Event()
        sync_values = (
            arguments.sync_base_url,
            arguments.device_id,
            arguments.credential_target,
        )
        if any(sync_values) and not all(sync_values):
            raise ValueError(
                "sync base URL, device ID, and credential target must be configured together"
            )
        if all(sync_values):
            credentials = WindowsCredentialManagerProvider(
                device_id=arguments.device_id,
                target_name=arguments.credential_target,
            )
            transport = UrllibSyncHttpTransport(arguments.sync_base_url)
            cloud_gateway = SignedFieldCloudGateway(
                credentials=credentials,
                transport=transport,
            )
            sync_client = FieldSyncClient(
                credentials=credentials,
                transport=transport,
            )
        source_inputs = None
        source_error = None
        if arguments.video is not None or arguments.mainline1_artifact is not None:
            if arguments.video is None or arguments.mainline1_artifact is None:
                source_error = "SOURCE_INPUTS_INCOMPLETE"
            else:
                try:
                    source_inputs = load_mainline1_replay_inputs(
                        artifact_path=arguments.mainline1_artifact,
                        video_path=arguments.video,
                    )
                except (OSError, KeyError, TypeError, ValueError):
                    # Optional attribution failure cannot bypass or replace the
                    # builder's mandatory verified-replay display gate below.
                    source_error = "SOURCE_CONTENT_INVALID"
        server = build_field_assistance_server(
            artifact_path=arguments.artifact,
            host=arguments.host,
            port=arguments.port,
            control_token=token,
            state_db_path=arguments.state_db,
            cloud_gateway=cloud_gateway,
            join_qr_asset_path=arguments.join_qr_asset,
            sync_wakeup=None if sync_client is None else sync_wake.set,
            mainline1_inputs=source_inputs,
        )
        if source_error is not None:
            server.field_state.source_validation_reason = source_error
        if server.field_state.source_validation_reason != "SOURCE_START_VERIFIED":
            print(
                "field ownership unavailable: " + server.field_state.source_validation_reason,
                file=sys.stderr,
            )
        if sync_client is not None:
            worker = _BackgroundSyncWorker(
                store=server.field_state.record_store,
                client=sync_client,
                wake=sync_wake,
                recover_missing_jobs=server.field_state._retry_unsynced_sessions,
            )
            worker.start()
            sync_wake.set()
        if cloud_gateway is not None:
            command_worker = _BackgroundFieldCommandWorker(
                gateway=cloud_gateway,
                store=server.field_state.record_store,
                match_lock=server.field_state.match_lock,
                before_application=server.field_state.before_field_command,
                match_changed=server.field_state.set_command_match,
                application_changed=server.field_state.handle_command_application,
                health_changed=server.field_state.set_cloud_control_status,
                maintenance=server.field_state.maintain_background,
            )
            command_worker.start()
        port = server.server_port
        ready = {
            "status": "READY",
            "screen_url": f"http://{display_host}:{port}/screen",
            "tablet_url": (f"http://{display_host}:{port}/tablet?token={quote(token, safe='')}"),
            "control_token": token,
            "artifact": str(arguments.artifact.resolve()),
            "state_db": str(arguments.state_db.resolve()),
            "session_id": server.field_state.session_id,
        }
        print(json.dumps(ready, ensure_ascii=False, sort_keys=True), flush=True)
        if arguments.check_only:
            return 0
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"field route shell failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if command_worker is not None:
            command_worker.stop()
        if worker is not None:
            worker.stop()
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
