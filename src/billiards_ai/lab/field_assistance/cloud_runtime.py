"""Shared development cloud preparation and field worker lifecycle.

Capture preparation and readiness remain the caller's responsibility. This module
never loads models or opens capture devices.
"""

from __future__ import annotations

import inspect
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from billiards_ai.edge.route_assistance.local_media_cleanup import (
    process_local_discard_once,
    process_local_owner_delete_once,
)
from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.local_media_schedule import LocalMediaCleanupScheduler
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.owner_device_delivery import (
    _ERRORS as OWNER_DEVICE_ERROR_CODES,
)
from billiards_ai.edge.route_assistance.owner_device_delivery import (
    OwnerDeletionReceiptOutbox,
)
from billiards_ai.edge.route_assistance.owner_device_transport import (
    OwnerDeviceMetadataHttpTransport,
)
from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask
from billiards_ai.edge.route_assistance.recording_cleanup import process_recording_discard_once
from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestLedger
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    UrllibSyncHttpTransport,
    WindowsCredentialManagerProvider,
)
from billiards_ai.edge.route_assistance.upload_receipt_delivery import (
    UPLOAD_RECEIPT_ERROR_CODES,
    UploadReceiptOutbox,
)
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    ExistingDevelopmentControlGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.managed_resource_recovery import (
    ManagedRuntimeResourceResolver,
    ManagedRuntimeResourceUnavailable,
)
from billiards_ai.lab.field_assistance.managed_startup import (
    BoundedStartupTransport,
    ManagedStartupCoordinator,
)
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

_EXISTING_DEVELOPMENT_BASE_URL = (
    "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com"
)
_EXISTING_DEVELOPMENT_ENVIRONMENT_ID = "cloud1-public-example"
_EXISTING_DEVELOPMENT_DEVICE_ID = "raytheon-01"
_EXISTING_DEVELOPMENT_CREDENTIAL_TARGET = "billiardsAI/development/raytheon-01"


def _note_unavailable_context(store, scope, diagnostics, session_id, code):
    if scope is not None:
        try:
            # Exclude foreign/unknown identities rather than expose them in this screen.
            store.get_managed_open_request_for_session(scope, session_id)
        except ValueError:
            return
    diagnostics[session_id] = code


def validate_development_cloud_scope(
    args,
    *,
    allow_existing_development_control_only=False,
    allow_local_development_environment=True,
):
    control_only = getattr(args, "existing_development_control_only", False)
    if getattr(args, "managed_sessions", False):
        if control_only or not args.sync_base_url:
            raise ValueError("managed sessions require full attested cloud configuration")
        ManagedSessionScope(
            args.cloudbase_environment_id, args.device_id, getattr(args, "table_id", None)
        )
    if control_only and not allow_existing_development_control_only:
        raise ValueError("existing development control-only mode is not supported by this source")
    sync_values = (args.sync_base_url, args.device_id, args.credential_target)
    scope_values = (
        args.runtime_environment,
        args.cloudbase_environment_id,
        args.expected_development_environment_id,
    )
    if any(sync_values) and not all(sync_values):
        raise ValueError("sync URL, device ID and credential target must be configured together")
    if not any(sync_values):
        if control_only:
            raise ValueError(
                "existing development control-only mode requires pinned control configuration"
            )
        local_scope = allow_local_development_environment and scope_values == (
            "development",
            None,
            None,
        )
        if any(scope_values) and not local_scope:
            raise ValueError(
                "development cloud scope cannot be asserted without sync configuration"
            )
        return
    if not all(scope_values):
        raise ValueError("development cloud scope is required before any cloud mutation")
    if args.runtime_environment != "development":
        raise ValueError("field test cloud scope must be development")
    if args.cloudbase_environment_id != args.expected_development_environment_id:
        raise ValueError("CloudBase environment is not the packet-approved development environment")
    expected_target = f"billiardsAI/development/{args.device_id}"
    if args.credential_target != expected_target:
        raise ValueError("credential target is not scoped to the exact development device")
    if control_only:
        expected = (
            _EXISTING_DEVELOPMENT_BASE_URL,
            _EXISTING_DEVELOPMENT_DEVICE_ID,
            _EXISTING_DEVELOPMENT_CREDENTIAL_TARGET,
            "development",
            _EXISTING_DEVELOPMENT_ENVIRONMENT_ID,
            _EXISTING_DEVELOPMENT_ENVIRONMENT_ID,
        )
        observed = (*sync_values, *scope_values)
        if not args.observation_only or observed != expected:
            raise ValueError(
                "existing development control-only mode requires ObservationOnly and "
                "the exact pinned development control configuration"
            )
        return
    endpoint = urlsplit(args.sync_base_url)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or args.cloudbase_environment_id not in f"{endpoint.hostname}{endpoint.path}"
    ):
        raise ValueError("sync endpoint is not bound to the approved development environment")


class _BackgroundLiveShotFactSyncWorker:
    """Drain facts-only shot rows without sharing capture or recorder threads."""

    def __init__(self, *, store, client, join_timeout_seconds=2 * 20.0 + 5.0):
        self._store = store
        self._client = client
        self._stopped = threading.Event()
        # One attestation plus one fact PUT, each with the transport's 20s timeout.
        self._join_timeout_seconds = join_timeout_seconds
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="field-live-shot-fact-sync",
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("live fact sync worker did not terminate")

    def _run(self):
        while not self._stopped.is_set():
            item = None
            try:
                item = self._store.claim_due_live_shot_fact(datetime.now(UTC))
                if item is None:
                    self._stopped.wait(0.5)
                    continue
                if self._stopped.is_set():
                    break
                result = self._client.run_live_shot_fact_once(item)
                if self._stopped.is_set():
                    break
                if result.outcome == "complete":
                    self._store.acknowledge_live_shot_fact(item.idempotency_key)
                else:
                    self._store.retry_live_shot_fact(item.idempotency_key)
                    self._stopped.wait(1.0)
            except Exception:
                if self._stopped.is_set():
                    break
                if item is not None:
                    try:
                        self._store.retry_live_shot_fact(item.idempotency_key)
                    except Exception:
                        pass
                self._stopped.wait(1.0)


class _BackgroundUploadReceiptSyncWorker:
    """Own an independent metadata lease; never hold a media pin during delivery."""

    def __init__(
        self,
        *,
        store,
        client,
        managed_scope,
        now_utc=lambda: datetime.now(UTC),
        poll_seconds=0.5,
        lease_seconds=60,
        join_timeout_seconds=2 * 20.0 + 5.0,
    ):
        self._outbox = UploadReceiptOutbox(store, managed_scope)
        self._client = client
        self._now_utc = now_utc
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._cursor = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="field-upload-receipt-sync"
        )
        self._started = False

    def start(self):
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("upload receipt sync worker did not terminate")

    def _retry(self, item, code):
        seconds = min(2 ** min(max(item.attempts - 1, 0), 9), 300)
        due = (
            (self._now_utc() + timedelta(seconds=seconds))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        self._outbox.retry(item, code, due)

    def run_once(self):
        if self._stopped.is_set():
            return False
        item = None
        try:
            batch = self._outbox.reconcile(after_request_id=self._cursor, limit=16)
            self._cursor = batch.next_request_id
            if self._stopped.is_set():
                return False
            item = self._outbox.claim_due(self._now_utc(), lease_seconds=self._lease_seconds)
            if item is None:
                return False
            if self._stopped.is_set():
                return True
            result = self._client.run_upload_receipt_once(item)
            if self._stopped.is_set():
                return True
            if result.outcome == "complete":
                self._outbox.acknowledge(item, result.receipt_ack)
            elif result.error_code in UPLOAD_RECEIPT_ERROR_CODES:
                if result.outcome == "blocked":
                    self._outbox.block(item, result.error_code)
                else:
                    self._retry(item, result.error_code)
            else:
                self._retry(item, "receipt_unavailable")
        except Exception:
            if item is not None and not self._stopped.is_set():
                try:
                    self._retry(item, "receipt_unavailable")
                except Exception:
                    pass  # Failed settlement leaves the independent lease recoverable.
        return item is not None

    def _run(self):
        while not self._stopped.is_set():
            self.run_once()
            self._stopped.wait(self._poll_seconds)


class _BackgroundSavedMediaSyncWorker:
    """Drain kiosk-selected clips from the durable outbox one at a time."""

    def __init__(
        self,
        *,
        store,
        client,
        trusted_postgame_root,
        wake_event,
        now_utc=lambda: datetime.now(UTC),
        poll_seconds=0.5,
        lease_seconds=180,
        # Create, authorize, upload, complete and four attestations: eight requests.
        join_timeout_seconds=8 * 20.0 + 5.0,
        managed_scope=None,
        recovery_diagnostics=None,
        exclude_managed=False,
    ):
        self._store = store
        self._client = client
        self._trusted_postgame_root = trusted_postgame_root
        self._managed_scope = managed_scope
        self._exclude_managed = exclude_managed
        self._recovery_diagnostics = (
            recovery_diagnostics if recovery_diagnostics is not None else {}
        )
        self._wake = wake_event
        self._now_utc = now_utc
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="field-saved-media-sync",
        )
        self._started = False

    @staticmethod
    def _retry_seconds(attempts):
        return min(2 ** max(int(attempts) - 1, 0), 300)

    def start(self):
        if self._started:
            return
        self._wake.set()
        self._thread.start()
        self._started = True

    def stop(self):
        self._stopped.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("saved media sync worker did not terminate")

    def _retry(self, item, error_code):
        try:
            self._store.retry_saved_media(
                item.clip_id,
                item.claim_token,
                error_code=error_code,
                next_attempt_at_utc=self._now_utc()
                + timedelta(seconds=self._retry_seconds(item.attempts)),
            )
        except Exception:
            # A reclaimed or expired lease belongs to the newer token owner.
            pass

    def run_once(self):
        now = self._now_utc()
        scope_arguments = (
            {} if self._managed_scope is None else {"managed_scope": self._managed_scope}
        )
        if self._exclude_managed:
            scope_arguments["exclude_managed"] = True
        try:
            item = self._store.claim_due_saved_media(
                now, lease_seconds=self._lease_seconds, **scope_arguments
            )
        except (OSError, ValueError):
            if self._managed_scope is None:
                raise
            self._recovery_diagnostics["saved"] = "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
            return False
        self._recovery_diagnostics.pop("saved", None)
        if item is None:
            return False
        if self._managed_scope is None and self._store.has_managed_session_identity(
            item.session_id
        ):
            self._recovery_diagnostics[item.session_id] = "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
            self._retry(item, "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE")
            return True
        root = self._trusted_postgame_root
        pin = nullcontext()
        if self._managed_scope is not None:
            try:
                request = self._store.get_managed_open_request_for_session(
                    self._managed_scope, item.session_id
                )
                self._store.validate_managed_local_session(
                    self._managed_scope, request.open_request_id
                )
                original_root = self._store.artifact_path_for_session(item.session_id)
                root = original_root / "postgame"
                pin = session_media_lock(original_root, item.session_id, exclusive=False)
            except (OSError, ValueError):
                self._recovery_diagnostics[item.session_id] = (
                    "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
                )
                self._retry(item, "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE")
                return True
        try:
            with pin:
                if not self._store.is_saved_media_claim_current(item.clip_id, item.claim_token):
                    return True
                request_arguments = (
                    {}
                    if self._managed_scope is None
                    else {
                        "upload_requests": SavedUploadRequestLedger(
                            self._store, self._managed_scope
                        )
                    }
                )
                result = self._client.run_saved_media_once(
                    item, trusted_postgame_root=root, **request_arguments
                )
        except Exception:
            if not self._stopped.is_set():
                if self._managed_scope is not None:
                    self._recovery_diagnostics[item.session_id] = (
                        "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
                    )
                self._retry(item, "SAVED_MEDIA_WORKER_ERROR")
            return True
        if self._stopped.is_set():
            return False
        if (
            self._managed_scope is not None
            and result.outcome == "blocked"
            and result.error_code == "SYNC_LOCAL_MEDIA_INVALID"
        ):
            self._recovery_diagnostics[item.session_id] = "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
            self._retry(item, "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE")
            return True
        try:
            if result.outcome == "complete":
                self._store.acknowledge_saved_media(item.clip_id, item.claim_token)
                if self._recovery_diagnostics.get(item.session_id) == (
                    "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
                ):
                    self._recovery_diagnostics.pop(item.session_id)
            elif result.outcome == "retry":
                self._retry(item, result.error_code or "SYNC_RETRY")
            elif result.outcome == "blocked":
                self._store.block_saved_media(
                    item.clip_id,
                    item.claim_token,
                    error_code=result.error_code or "SYNC_BLOCKED",
                )
            else:
                self._retry(item, "SYNC_INVALID_RESULT")
        except Exception:
            # Never let stale lease fencing terminate the long-lived worker.
            pass
        return True

    def _run(self):
        while not self._stopped.is_set():
            try:
                while not self._stopped.is_set() and self.run_once():
                    pass
                if self._stopped.is_set():
                    break
                # Clear first, then recheck to close the lost-wakeup window.
                self._wake.clear()
                if not self._stopped.is_set() and self.run_once():
                    continue
            except Exception:
                pass
            if self._stopped.is_set():
                break
            self._wake.wait(self._poll_seconds)


class _BackgroundPostgameAnalysisWorker:
    """Launch one isolated canonical analysis worker for each durable ended session."""

    def __init__(
        self,
        *,
        store,
        command_factory,
        process_factory=subprocess.Popen,
        poll_seconds=0.5,
        monotonic=time.monotonic,
        retry_seconds=30.0,
        resource_session_ids=None,
        recovery_diagnostics=None,
        managed_scope=None,
        exclude_managed=False,
        resource_resolver=None,
    ):
        self._resource_resolver = resource_resolver
        self._store = store
        self._command_factory = command_factory
        self._process_factory = process_factory
        self._poll_seconds = poll_seconds
        self._monotonic = monotonic
        self._retry_seconds = retry_seconds
        self._retry_not_before = {}
        try:
            parameters = inspect.signature(command_factory).parameters.values()
            self._purpose_aware = any(
                parameter.name == "purpose" or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            self._purpose_aware = False
        self._media_completed = set()
        self._resource_session_ids = resource_session_ids
        self._managed_scope = managed_scope
        self._exclude_managed = exclude_managed
        self._recovery_diagnostics = (
            recovery_diagnostics if recovery_diagnostics is not None else {}
        )
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="field-postgame-analysis-supervisor",
        )
        self._active_session_id = None
        self._active_purpose = None
        self._active_process = None
        self._started = False

    def start(self):
        self._thread.start()
        self._started = True

    def stop(self):
        self._stopped.set()
        # Finish any launcher already in flight before reading the child handle;
        # otherwise a child can appear after the initial process snapshot.
        if self._thread.is_alive():
            self._thread.join(timeout=6.0)
        if self._thread.is_alive():
            raise TimeoutError("postgame worker did not terminate")
        process = self._active_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    def run_once(self):
        process = self._active_process
        if process is not None:
            exit_code = process.poll()
            if exit_code is None:
                return
            if exit_code != 0 and self._active_session_id is not None:
                self._retry_not_before[(self._active_session_id, self._active_purpose)] = (
                    self._monotonic() + self._retry_seconds
                )
            if (
                exit_code == 0
                and self._active_purpose == "media"
                and self._active_session_id is not None
            ):
                self._media_completed.add(self._active_session_id)
            self._active_process = None
            self._active_session_id = None
            self._active_purpose = None
        if self._resource_resolver is not None:
            now = self._monotonic()
            candidates = self._store.list_ended_session_ids_requiring_postgame(
                managed_scope=self._managed_scope
            )
            for session_id in candidates:
                purpose = self._purpose(session_id)
                if purpose is None:
                    continue
                if self._retry_not_before.get((session_id, purpose), 0.0) > now:
                    continue
                try:
                    context = self._resource_resolver.resolve(session_id, purpose=purpose)
                    command = self._command_factory(
                        session_id, resource_context=context, purpose=purpose
                    )
                    self._launch(session_id, purpose, command)
                    return
                except (OSError, RuntimeError, ValueError):
                    self._retry_not_before[(session_id, purpose)] = now + self._retry_seconds
            return
        sessions = self._store.list_ended_sessions_requiring_postgame()
        if self._exclude_managed:
            sessions = tuple(
                s for s in sessions if not self._store.has_managed_session_identity(s.session_id)
            )
        if self._resource_session_ids is not None:
            for candidate in sessions:
                if candidate.session_id not in self._resource_session_ids:
                    _note_unavailable_context(
                        self._store,
                        self._managed_scope,
                        self._recovery_diagnostics,
                        candidate.session_id,
                        "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE",
                    )
            sessions = tuple(s for s in sessions if s.session_id in self._resource_session_ids)
        now = self._monotonic()
        for session in sessions:
            session_id = session.session_id
            purpose = self._purpose(session_id)
            if purpose is None:
                continue
            if self._retry_not_before.get((session_id, purpose), 0.0) > now:
                continue
            command = (
                self._command_factory(session_id, purpose=purpose)
                if self._purpose_aware
                else self._command_factory(session_id)
            )
            self._launch(session_id, purpose, command)
            return

    def _purpose(self, session_id):
        fenced = getattr(self._store, "is_postgame_work_fenced", None)
        if callable(fenced) and fenced(session_id):
            return None
        if not self._purpose_aware:
            return "analysis"
        if session_id not in self._media_completed:
            return "media"
        if self._store.get_session_policy(session_id) == "recording-v1":
            return None
        return "analysis"

    def _launch(self, session_id, purpose, command):
        self._active_session_id = session_id
        self._active_purpose = purpose
        self._active_process = self._process_factory(
            command,
            cwd=str(_PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
        )

    def _run(self):
        while not self._stopped.is_set():
            try:
                self.run_once()
            except (OSError, RuntimeError, ValueError):
                pass
            self._stopped.wait(self._poll_seconds)


class _BackgroundOwnerDeviceDeletionWorker:
    """One metadata operation per durable turn; never opens or pins media."""

    def __init__(
        self,
        *,
        store,
        client,
        managed_scope,
        now_utc=lambda: datetime.now(UTC),
        poll_seconds=0.5,
        join_timeout_seconds=60,
        stop_event=None,
    ):
        self._store, self._client, self._scope = store, client, managed_scope
        self._outbox = OwnerDeletionReceiptOutbox(store, managed_scope)
        self._now_utc, self._poll_seconds = now_utc, poll_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._stopped = stop_event if stop_event is not None else threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="field-owner-deletion")
        self._started = False
        self._poll_not_before = None
        self.last_error_code = None

    def start(self):
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("owner deletion worker did not terminate")
        self._client.close_owner_delete_transport()

    def _retry(self, item, code):
        seconds = min(2 ** min(max(item.attempts - 1, 0), 9), 300)
        self._outbox.retry(item, code, self._now_utc() + timedelta(seconds=seconds))

    def run_once(self):
        if self._stopped.is_set():
            return False
        item = None
        try:
            self._outbox.reconcile(limit=16)
            if self._stopped.is_set():
                return False
            turn = self._outbox.next_turn()
            if turn == "send":
                item = self._outbox.claim_due(self._now_utc(), lease_seconds=60)
            if self._stopped.is_set():
                return False
            if item is not None:
                result = self._client.complete_owner_delete_once(item)
                if self._stopped.is_set():
                    return True
                if result.outcome == "ack":
                    self._outbox.acknowledge(item, result.ack)
                else:
                    code = (
                        result.error_code
                        if result.error_code in OWNER_DEVICE_ERROR_CODES
                        else "invalid_response"
                    )
                    self.last_error_code = code
                    if result.outcome == "blocked":
                        self._outbox.block(item, code)
                    else:
                        self._retry(item, code)
                return True
            now = self._now_utc()
            if self._poll_not_before is not None and now < self._poll_not_before:
                return False
            result = self._client.poll_owner_delete_once(self._scope)
            if self._stopped.is_set():
                return True
            if result.outcome == "task":
                task = result.task
                if not isinstance(task, OwnerDeleteDeviceTask) or (
                    task.environment_id,
                    task.device_id,
                    task.table_id,
                ) != (self._scope.environment_id, self._scope.device_id, self._scope.table_id):
                    self.last_error_code = "scope_conflict"
                elif not self._stopped.is_set():
                    self._store.accept_owner_delete_task(task, managed_scope=self._scope)
            elif result.outcome != "idle":
                self.last_error_code = (
                    result.error_code
                    if result.error_code in OWNER_DEVICE_ERROR_CODES
                    else "invalid_response"
                )
                self._poll_not_before = now + timedelta(
                    seconds=5 if result.outcome == "retry" else 300
                )
            return True
        except Exception:
            self.last_error_code = "transport_unavailable"
            if item is not None and not self._stopped.is_set():
                try:
                    self._retry(item, "transport_unavailable")
                except Exception:
                    pass  # Original independent lease remains recoverable.
            return False

    def _run(self):
        while not self._stopped.is_set():
            self.run_once()
            self._stopped.wait(self._poll_seconds)


class _BackgroundLocalMediaCleanupWorker:
    """Own cancellable local disposal in online and offline capture runtimes."""

    def __init__(
        self,
        *,
        store,
        jobs_db,
        artifact_root,
        jobs_root,
        process_once=process_local_discard_once,
        owner_process_once=process_local_owner_delete_once,
        poll_seconds=0.5,
        join_timeout_seconds=5.0,
        session_ids=None,
        exclude_managed=False,
        managed_scope=None,
        recovery_diagnostics=None,
        resource_resolver=None,
        scheduler=None,
        ring_root=None,
    ):
        self._resource_resolver = resource_resolver
        self._scheduler = (
            scheduler
            if scheduler is not None
            else LocalMediaCleanupScheduler(store, managed_scope, exclude_managed)
        )
        self._store = store
        self._jobs_db, self._artifact_root, self._jobs_root = jobs_db, artifact_root, jobs_root
        self._ring_root = ring_root
        self._process_once = process_once
        self._owner_process_once = owner_process_once
        self._session_ids = session_ids
        self._exclude_managed = exclude_managed
        self._managed_scope = managed_scope
        self._recovery_diagnostics = (
            recovery_diagnostics if recovery_diagnostics is not None else {}
        )
        self._poll_seconds, self._join_timeout_seconds = poll_seconds, join_timeout_seconds
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="field-local-media-cleanup"
        )
        self._started = False

    def start(self):
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._join_timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("local media cleanup worker did not terminate")

    def run_once(self):
        # Scheduler commits cursor advancement BEFORE every resolver attempt.
        for _ in range(16):
            if self._stopped.is_set():
                return False
            candidate = self._scheduler.next_candidate(datetime.now(UTC))
            if candidate is None:
                return False
            sid = candidate.session_id
            if self._resource_resolver is not None:
                try:
                    context = self._resource_resolver.resolve(sid, purpose="cleanup")
                except ManagedRuntimeResourceUnavailable:
                    continue
                jobs_db, artifact_root, jobs_root = (
                    context.jobs_db,
                    context.artifact_root,
                    context.jobs_root,
                )
                ring_root = context.ring_root if candidate.kind == "recording" else None
            else:
                if candidate.kind == "owner" or (
                    self._session_ids is not None and sid not in self._session_ids
                ):
                    _note_unavailable_context(
                        self._store,
                        self._managed_scope,
                        self._recovery_diagnostics,
                        sid,
                        "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE",
                    )
                    continue
                jobs_db, artifact_root, jobs_root = (
                    self._jobs_db,
                    self._artifact_root,
                    self._jobs_root,
                )
                ring_root = self._ring_root
            if self._stopped.is_set():
                return False
            process = (
                process_recording_discard_once
                if candidate.kind == "recording"
                else self._owner_process_once
                if candidate.kind == "owner"
                else self._process_once
            )
            result = process(
                self._store,
                jobs_db,
                artifact_root,
                jobs_root,
                cancelled=self._stopped.is_set,
                session_ids=(sid,),
                **(
                    {"exclude_managed": True}
                    if self._exclude_managed and candidate.kind != "owner"
                    else {}
                ),
                **({"ring_root": ring_root} if candidate.kind == "recording" else {}),
            )
            # Even an idle/raced executor counts as this tick's single step.
            return result.outcome != "idle"
        return False

    def _run(self):
        while not self._stopped.is_set():
            try:
                # One claim per bounded batch; yield to capture between claims.
                self.run_once()
            except Exception:
                # Durable claim/lease remains recoverable after store/I/O errors.
                pass
            self._stopped.wait(self._poll_seconds)


@dataclass(frozen=True)
class FieldCloudFactories:
    """Explicit resource boundaries for offline lifecycle verification."""

    credentials: Callable = WindowsCredentialManagerProvider
    transport: Callable = UrllibSyncHttpTransport
    signed_gateway: Callable = SignedFieldCloudGateway
    attested_gateway: Callable = DevelopmentAttestedFieldCloudGateway
    existing_control_gateway: Callable = ExistingDevelopmentControlGateway
    sync_client: Callable = FieldSyncClient
    command_worker: Callable = _BackgroundFieldCommandWorker
    live_fact_worker: Callable = _BackgroundLiveShotFactSyncWorker
    postgame_worker: Callable = _BackgroundPostgameAnalysisWorker
    upload_receipt_worker: Callable = _BackgroundUploadReceiptSyncWorker
    saved_media_worker: Callable = _BackgroundSavedMediaSyncWorker
    local_cleanup_worker: Callable = _BackgroundLocalMediaCleanupWorker
    owner_deletion_worker: Callable = _BackgroundOwnerDeviceDeletionWorker
    owner_deletion_transport: Callable = OwnerDeviceMetadataHttpTransport


class FieldCloudShutdownError(RuntimeError):
    """Owned workers remain unconfirmed; callers must keep HTTP/capture open."""


class FieldCloudRuntime:
    """Own all constructed field workers, including a partially started worker."""

    def __init__(self, *, args, factories, gateway=None, credentials=None, transport=None):
        self.gateway = gateway
        self._args = args
        self._factories = factories
        self._credentials = credentials
        self._transport = transport
        self._wake = (
            threading.Event()
            if gateway is not None and not getattr(args, "existing_development_control_only", False)
            else None
        )
        self.sync_wakeup = None if self._wake is None else self._wake.set
        self._workers = {}
        self._start_attempted = False
        self._stopped = False

    def make_managed_startup_gateway(self, *, deadline, cancelled, monotonic=time.monotonic):
        """Construct a separately budgeted bootstrap gateway without starting workers."""
        if (
            self._stopped
            or self.gateway is None
            or self._credentials is None
            or getattr(self._args, "existing_development_control_only", False)
        ):
            raise ValueError("managed startup requires an active full attested cloud runtime")
        validate_development_cloud_scope(self._args)
        transport = BoundedStartupTransport(
            self._args.sync_base_url,
            deadline=deadline,
            cancelled=cancelled,
            monotonic=monotonic,
            transport_factory=self._factories.transport,
        )
        signed = self._factories.signed_gateway(credentials=self._credentials, transport=transport)
        gateway = self._factories.attested_gateway(
            delegate=signed,
            expected_cloudbase_environment_id=self._args.expected_development_environment_id,
        )
        gateway.runtime_attestation()
        return gateway

    def make_managed_startup(self, *, monotonic=time.monotonic):
        """Build the shared coordinator from explicit, fully validated configured scope."""
        if not getattr(self._args, "managed_sessions", False):
            return None
        validate_development_cloud_scope(self._args)
        if self._stopped or self.gateway is None or self._credentials is None:
            raise ValueError("managed startup requires an active full attested cloud runtime")
        return ManagedStartupCoordinator(
            scope=ManagedSessionScope(
                self._args.cloudbase_environment_id, self._args.device_id, self._args.table_id
            ),
            gateway_factory=lambda **budget: self.make_managed_startup_gateway(
                **budget, monotonic=monotonic
            ),
            monotonic=monotonic,
        )

    def _start_worker(self, name, factory, **kwargs):
        worker = factory(**kwargs)
        self._workers[name] = worker
        worker.start()

    def start(self, server, *, postgame_command_factory):
        """Start only after the caller has established capture readiness."""
        if self._stopped:
            raise RuntimeError("field cloud runtime is stopped")
        if self._start_attempted:
            return
        self._start_attempted = True
        try:
            state = server.field_state
            store = getattr(state, "record_store", None)
            resource_sessions = getattr(state, "resource_session_ids", None)
            recovery_diagnostics = getattr(state, "recovery_diagnostics", None)
            context_arguments = dict(exclude_managed=True)
            cleanup_arguments = dict(exclude_managed=True)
            saved_arguments = dict(exclude_managed=True)
            if resource_sessions is not None:
                context_arguments = dict(
                    resource_session_ids=resource_sessions,
                    managed_scope=state.managed_session_scope,
                    recovery_diagnostics=recovery_diagnostics,
                )
                cleanup_arguments = dict(
                    session_ids=resource_sessions,
                    managed_scope=state.managed_session_scope,
                    recovery_diagnostics=recovery_diagnostics,
                )
                saved_arguments = dict(
                    managed_scope=state.managed_session_scope,
                    recovery_diagnostics=recovery_diagnostics,
                )
                if getattr(state, "managed_resource_spec_record", None) is not None:
                    resolver = ManagedRuntimeResourceResolver(
                        store, state.managed_session_scope, recovery_diagnostics
                    )
                    context_arguments = dict(
                        resource_resolver=resolver,
                        managed_scope=state.managed_session_scope,
                        recovery_diagnostics=recovery_diagnostics,
                    )
                    cleanup_arguments = dict(context_arguments)
                else:
                    for session in store.list_ended_sessions_requiring_postgame():
                        if session.session_id not in resource_sessions:
                            _note_unavailable_context(
                                store,
                                state.managed_session_scope,
                                recovery_diagnostics,
                                session.session_id,
                                "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE",
                            )
            if store is not None:
                self._start_worker(
                    "postgame",
                    self._factories.postgame_worker,
                    store=store,
                    command_factory=postgame_command_factory,
                    **context_arguments,
                )
                self._start_worker(
                    "cleanup",
                    self._factories.local_cleanup_worker,
                    store=store,
                    jobs_db=self._args.jobs_root / "analysis-jobs.sqlite3",
                    artifact_root=self._args.artifact_root,
                    jobs_root=self._args.jobs_root,
                    **(
                        {"ring_root": self._args.ring_root}
                        if getattr(self._args, "ring_root", None) is not None
                        else {}
                    ),
                    **cleanup_arguments,
                )
            if store is not None and self._wake is not None:
                self._start_worker(
                    "saved",
                    self._factories.saved_media_worker,
                    store=store,
                    client=self._factories.sync_client(
                        credentials=self._credentials,
                        transport=self._transport,
                        before_mutation=self.gateway.runtime_attestation,
                    ),
                    trusted_postgame_root=self._args.artifact_root / "postgame",
                    wake_event=self._wake,
                    **saved_arguments,
                )
            if (
                store is not None
                and self._wake is not None
                and self._credentials is not None
                and getattr(self._args, "managed_sessions", False)
            ):
                scope = ManagedSessionScope(
                    self._args.cloudbase_environment_id, self._args.device_id, self._args.table_id
                )
                if getattr(state, "managed_session_scope", None) != scope:
                    raise ValueError(
                        "receipt worker requires the original configured managed scope"
                    )
                self._start_worker(
                    "receipts",
                    self._factories.upload_receipt_worker,
                    store=store,
                    managed_scope=scope,
                    client=self._factories.sync_client(
                        credentials=self._credentials,
                        transport=self._transport,
                        before_mutation=self.gateway.runtime_attestation,
                    ),
                )
                owner_transport = self._factories.owner_deletion_transport(self._args.sync_base_url)
                owner_signed = self._factories.signed_gateway(
                    credentials=self._credentials,
                    transport=owner_transport,
                )
                owner_gateway = self._factories.attested_gateway(
                    delegate=owner_signed,
                    expected_cloudbase_environment_id=self._args.expected_development_environment_id,
                )
                owner_stop = threading.Event()
                self._start_worker(
                    "owner_deletion",
                    self._factories.owner_deletion_worker,
                    store=store,
                    managed_scope=scope,
                    stop_event=owner_stop,
                    client=self._factories.sync_client(
                        credentials=self._credentials,
                        transport=owner_transport,
                        before_mutation=owner_gateway.runtime_attestation,
                        owner_cancelled=owner_stop.is_set,
                        owner_scope=scope,
                    ),
                )
            if self.gateway is not None:
                self._start_worker(
                    "commands",
                    self._factories.command_worker,
                    gateway=self.gateway,
                    store=state.record_store,
                    match_lock=state.match_lock,
                    before_application=state.before_field_command,
                    match_changed=state.set_command_match,
                    application_changed=state.handle_command_application,
                    health_changed=state.set_cloud_control_status,
                    maintenance=state.maintain_background,
                    confirmation_resolver=state.resolve_command_confirmation,
                    command_reducer=state.reduce_live_command,
                )
                if self._wake is not None:
                    self._start_worker(
                        "facts",
                        self._factories.live_fact_worker,
                        store=state.record_store,
                        client=self._factories.sync_client(
                            credentials=self._credentials,
                            transport=self._transport,
                        ),
                    )
        except BaseException as exc:
            try:
                self.stop()
            except Exception as cleanup_error:
                exc.add_note(f"field cloud cleanup also failed: {cleanup_error}")
            raise

    def stop(self):
        """Retain failed resources for retry; never report an active worker stopped."""
        self._stopped = True
        first_error = None
        for name in (
            "commands",
            "facts",
            "postgame",
            "saved",
            "receipts",
            "owner_deletion",
            "cleanup",
        ):
            if name == "cleanup" and "owner_deletion" in self._workers:
                continue  # Do not release cleanup ownership before network ownership.
            worker = self._workers.get(name)
            if worker is not None:
                try:
                    worker.stop()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    self._workers.pop(name)
        if first_error is not None:
            raise FieldCloudShutdownError(
                f"field cloud shutdown incomplete ({', '.join(self._workers)}): {first_error}"
            ) from first_error


def prepare_cloud_runtime(
    args,
    *,
    factories=None,
    allow_existing_development_control_only=False,
    allow_local_development_environment=True,
) -> FieldCloudRuntime:
    """Validate all scope before credentials/transport; attest before returning."""
    validate_development_cloud_scope(
        args,
        allow_existing_development_control_only=allow_existing_development_control_only,
        allow_local_development_environment=allow_local_development_environment,
    )
    factories = factories or FieldCloudFactories()
    if not args.sync_base_url:
        return FieldCloudRuntime(args=args, factories=factories)
    credentials = factories.credentials(
        device_id=args.device_id,
        target_name=args.credential_target,
    )
    transport = factories.transport(args.sync_base_url, allow_redirects=False)
    signed_gateway = factories.signed_gateway(credentials=credentials, transport=transport)
    if getattr(args, "existing_development_control_only", False):
        gateway = factories.existing_control_gateway(
            delegate=signed_gateway,
            expected_table_id=_EXISTING_DEVELOPMENT_DEVICE_ID,
        )
    else:
        gateway = factories.attested_gateway(
            delegate=signed_gateway,
            expected_cloudbase_environment_id=args.expected_development_environment_id,
        )
        gateway.runtime_attestation()
    return FieldCloudRuntime(
        args=args,
        factories=factories,
        gateway=gateway,
        credentials=credentials,
        transport=transport,
    )
