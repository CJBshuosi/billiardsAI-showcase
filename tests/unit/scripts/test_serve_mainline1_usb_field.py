import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.lab import mainline1_field_runtime as field_runtime
from billiards_ai.lab.field_assistance.cloud_gateway import FieldSessionLease
from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    prepare_cloud_runtime,
)
from billiards_ai.lab.mainline1_calibration_gate import CalibrationGateStatus
from billiards_ai.lab.mainline1_capture import FrameGeometry
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame, RealtimeCaptureConfig

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/algorithm/serve_mainline1_usb_field.py"


def arguments(root):
    result = []
    for flag in ("native-helper", "capture-setup", "config", "join-qr-asset"):
        path = root / flag
        path.write_bytes(b"fixture")
        result.extend(("--" + flag, str(path)))
    for flag in (
        "calibration-root",
        "detector-training-run",
        "identity-training-run",
        "dinov2-assets",
        "ring-root",
        "jobs-root",
        "artifact-root",
    ):
        path = root / flag
        path.mkdir(exist_ok=True)
        result.extend(("--" + flag, str(path)))
    return result + [
        "--device-locator",
        "usb:fixture",
        "--device-fingerprint",
        "a" * 64,
        "--state-db",
        str(root / "state.sqlite"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--control-token",
        "secret",
    ]


def load_script():
    assert SCRIPT.is_file(), "live service CLI is missing"
    spec = importlib.util.spec_from_file_location("live_field_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inject_cloud_factories(module, monkeypatch, **factories):
    dependencies = FieldCloudFactories(**factories)
    monkeypatch.setattr(
        module,
        "prepare_cloud_runtime",
        lambda args, **options: prepare_cloud_runtime(args, factories=dependencies, **options),
    )


def test_runtime_handle_forwards_realtime_perception_state() -> None:
    tracking = object()
    tracker_health = object()
    anchor_worker_health = object()
    anchor_replay_health = object()
    diagnostics = object()
    ring_health = object()
    runtime = SimpleNamespace(
        current_tracking=tracking,
        tracker_health=tracker_health,
        anchor_health=anchor_worker_health,
        anchor_replay_health=anchor_replay_health,
        diagnostics=diagnostics,
        ring_health=ring_health,
    )
    handle = field_runtime.RuntimeHandle(
        source=SimpleNamespace(cancel=lambda: None),
        runtime_factory=None,
    )
    handle._runtime = runtime

    assert handle.current_tracking is tracking
    assert handle.tracker_health is tracker_health
    assert handle.anchor_health is anchor_worker_health
    assert handle.anchor_replay_health is anchor_replay_health
    assert handle.diagnostics is diagnostics
    assert handle.recording_health is ring_health


def test_externally_fed_evidence_store_does_not_duplicate_analysis_frames() -> None:
    calls = []
    delegate = SimpleNamespace(
        health=object(),
        maximum_queue_depth=7,
        begin_shot=lambda frame_id: calls.append(("begin", frame_id)),
        settle_shot=lambda frame_id: calls.append(("settle", frame_id)),
        freeze_shot=lambda: calls.append(("freeze", None)) or "frozen",
        release_shot=lambda: calls.append(("release", None)),
    )
    store = field_runtime._ExternallyFedEvidenceStore(delegate)

    store.append(object())
    store.begin_shot(10)
    store.settle_shot(20)

    assert store.freeze_shot() == "frozen"
    store.release_shot()
    store.close()
    assert calls == [
        ("begin", 10),
        ("settle", 20),
        ("freeze", None),
        ("release", None),
    ]
    assert store.health is delegate.health
    assert store.maximum_queue_depth == 7


def test_postgame_supervisor_launches_one_worker_per_ended_session() -> None:
    module = load_script()
    ended = SimpleNamespace(session_id="session-ended")

    class Store:
        def list_ended_sessions_requiring_postgame(self):
            return (ended,)

    launched = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    supervisor = module._BackgroundPostgameAnalysisWorker(
        store=Store(),
        command_factory=lambda session_id: ["worker", session_id],
        process_factory=lambda command, **kwargs: launched.append((command, kwargs)) or Process(),
    )

    supervisor.run_once()
    supervisor.run_once()

    assert [command for command, _kwargs in launched] == [["worker", "session-ended"]]
    supervisor.stop()


def test_postgame_supervisor_backs_off_after_worker_failure() -> None:
    module = load_script()
    ended = SimpleNamespace(session_id="session-ended")
    now = [0.0]

    class Store:
        def list_ended_sessions_requiring_postgame(self):
            return (ended,)

    class FailedProcess:
        def poll(self):
            return 2

    launched = []
    supervisor = module._BackgroundPostgameAnalysisWorker(
        store=Store(),
        command_factory=lambda session_id: ["worker", session_id],
        process_factory=lambda command, **_kwargs: launched.append(command) or FailedProcess(),
        monotonic=lambda: now[0],
        retry_seconds=30.0,
    )

    supervisor.run_once()
    supervisor.run_once()
    assert launched == [["worker", "session-ended"]]

    now[0] = 31.0
    supervisor.run_once()
    assert launched == [["worker", "session-ended"], ["worker", "session-ended"]]


class _LegacySavedWorkerStore:
    def has_managed_session_identity(self, session_id):
        return False

    def is_saved_media_claim_current(self, clip_id, claim_token):
        return True


def test_saved_media_worker_maps_complete_retry_and_blocked_with_fenced_tokens(tmp_path) -> None:
    module = load_script()
    now = datetime(2026, 10, 28, tzinfo=UTC)
    items = [
        SimpleNamespace(
            session_id="legacy-session", clip_id="clip-complete", claim_token="token-1", attempts=1
        ),
        SimpleNamespace(
            session_id="legacy-session", clip_id="clip-retry", claim_token="token-2", attempts=3
        ),
        SimpleNamespace(
            session_id="legacy-session", clip_id="clip-block", claim_token="token-3", attempts=1
        ),
    ]
    calls = []

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            calls.append(("claim", claimed_at, lease_seconds))
            return items.pop(0) if items else None

        def acknowledge_saved_media(self, clip_id, token):
            calls.append(("ack", clip_id, token))

        def retry_saved_media(self, clip_id, token, *, error_code, next_attempt_at_utc):
            calls.append(("retry", clip_id, token, error_code, next_attempt_at_utc))

        def block_saved_media(self, clip_id, token, *, error_code):
            calls.append(("block", clip_id, token, error_code))

    outcomes = iter(
        (
            SimpleNamespace(outcome="complete", error_code=None),
            SimpleNamespace(outcome="retry", error_code="SYNC_PARTIAL"),
            SimpleNamespace(outcome="blocked", error_code="SYNC_CONFLICT"),
        )
    )

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            assert trusted_postgame_root == tmp_path / "postgame"
            calls.append(("attempt", item.clip_id))
            return next(outcomes)

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        now_utc=lambda: now,
    )

    assert worker.run_once() is True
    assert worker.run_once() is True
    assert worker.run_once() is True
    assert worker.run_once() is False
    assert ("ack", "clip-complete", "token-1") in calls
    assert (
        "retry",
        "clip-retry",
        "token-2",
        "SYNC_PARTIAL",
        now + timedelta(seconds=4),
    ) in calls
    assert ("block", "clip-block", "token-3", "SYNC_CONFLICT") in calls
    assert all(call[2] == 180 for call in calls if call[0] == "claim")
    assert worker._retry_seconds(1) == 1
    assert worker._retry_seconds(20) == 300


def test_saved_media_worker_retries_unexpected_error_without_acknowledging(tmp_path) -> None:
    module = load_script()
    now = datetime(2026, 10, 28, tzinfo=UTC)
    item = SimpleNamespace(
        session_id="legacy-session", clip_id="clip-1", claim_token="token-1", attempts=2
    )
    calls = []

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            return item

        def acknowledge_saved_media(self, clip_id, token):
            calls.append(("ack", clip_id, token))

        def retry_saved_media(self, clip_id, token, *, error_code, next_attempt_at_utc):
            calls.append(("retry", clip_id, token, error_code, next_attempt_at_utc))

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            raise RuntimeError("unexpected")

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        now_utc=lambda: now,
    )

    assert worker.run_once() is True
    assert calls == [
        ("retry", "clip-1", "token-1", "SAVED_MEDIA_WORKER_ERROR", now + timedelta(seconds=2))
    ]


@pytest.mark.parametrize(
    ("client_raises", "expected_error"),
    ((False, "SYNC_PARTIAL"), (True, "SAVED_MEDIA_WORKER_ERROR")),
)
def test_saved_media_worker_schedules_retry_from_attempt_completion(
    tmp_path, client_raises, expected_error
) -> None:
    module = load_script()
    started_at = datetime(2026, 10, 28, tzinfo=UTC)
    clock = [started_at]
    item = SimpleNamespace(
        session_id="legacy-session", clip_id="clip-1", claim_token="token-1", attempts=3
    )

    class Store(_LegacySavedWorkerStore):
        def __init__(self):
            self.due_at = started_at
            self.leased = False
            self.retry_deadline = None
            self.claim_count = 0

        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            if self.leased or claimed_at < self.due_at:
                return None
            self.leased = True
            self.claim_count += 1
            return item

        def retry_saved_media(self, clip_id, token, *, error_code, next_attempt_at_utc):
            assert (clip_id, token, error_code) == ("clip-1", "token-1", expected_error)
            self.leased = False
            self.due_at = next_attempt_at_utc
            self.retry_deadline = next_attempt_at_utc

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            clock[0] += timedelta(seconds=45)
            if client_raises:
                raise RuntimeError("attempt failed")
            return SimpleNamespace(outcome="retry", error_code="SYNC_PARTIAL")

    store = Store()
    worker = module._BackgroundSavedMediaSyncWorker(
        store=store,
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        now_utc=lambda: clock[0],
    )

    assert worker.run_once() is True
    assert store.retry_deadline == started_at + timedelta(seconds=49)
    assert worker.run_once() is False
    assert store.claim_count == 1


def test_saved_media_worker_ignores_stale_fencing_failure_and_keeps_running(tmp_path) -> None:
    module = load_script()
    item = SimpleNamespace(
        session_id="legacy-session", clip_id="clip-1", claim_token="stale", attempts=1
    )
    claims = [item, None]

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            return claims.pop(0)

        def acknowledge_saved_media(self, clip_id, token):
            raise ValueError("stale token")

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            return SimpleNamespace(outcome="complete", error_code=None)

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
    )

    assert worker.run_once() is True
    assert worker.run_once() is False


def test_saved_media_worker_wake_and_poll_both_discover_due_rows(tmp_path) -> None:
    module = load_script()
    wake = threading.Event()
    claimed = threading.Event()
    first_claim = threading.Event()
    rows = []

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            first_claim.set()
            if not rows:
                return None
            claimed.set()
            return rows.pop(0)

        def acknowledge_saved_media(self, clip_id, token):
            return None

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            return SimpleNamespace(outcome="complete", error_code=None)

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=wake,
        poll_seconds=0.02,
        join_timeout_seconds=0.2,
    )
    worker.start()
    assert first_claim.wait(1)

    rows.append(
        SimpleNamespace(
            session_id="legacy-session", clip_id="clip-wake", claim_token="token-1", attempts=1
        )
    )
    wake.set()
    assert claimed.wait(1)
    claimed.clear()

    rows.append(
        SimpleNamespace(
            session_id="legacy-session", clip_id="clip-poll", claim_token="token-2", attempts=1
        )
    )
    assert claimed.wait(1)
    worker.stop()
    assert not worker._thread.is_alive()


def test_saved_media_worker_stop_interrupts_idle_wait(tmp_path) -> None:
    module = load_script()

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            return None

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=object(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        poll_seconds=30.0,
        join_timeout_seconds=0.2,
    )
    worker.start()
    worker.stop()
    assert not worker._thread.is_alive()


def test_saved_media_worker_survives_unexpected_store_error(tmp_path) -> None:
    module = load_script()
    acknowledged = threading.Event()
    item = SimpleNamespace(
        session_id="legacy-session", clip_id="clip-1", claim_token="token-1", attempts=1
    )
    claims = [KeyError("database row"), item, None]

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            result = claims.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        def acknowledge_saved_media(self, clip_id, token):
            acknowledged.set()

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            return SimpleNamespace(outcome="complete", error_code=None)

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        poll_seconds=0.01,
        join_timeout_seconds=0.2,
    )
    worker.start()
    assert acknowledged.wait(1)
    worker.stop()
    assert not worker._thread.is_alive()


def test_saved_media_worker_timeout_reports_live_attempt_without_late_ack(tmp_path) -> None:
    module = load_script()
    entered = threading.Event()
    release = threading.Event()
    acknowledged = threading.Event()
    item = SimpleNamespace(
        session_id="legacy-session", clip_id="clip-1", claim_token="token-1", attempts=1
    )
    rows = [item]

    class Store(_LegacySavedWorkerStore):
        def claim_due_saved_media(self, claimed_at, *, lease_seconds):
            return rows.pop(0) if rows else None

        def acknowledge_saved_media(self, clip_id, token):
            acknowledged.set()

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root):
            entered.set()
            assert release.wait(1)
            return SimpleNamespace(outcome="complete", error_code=None)

    worker = module._BackgroundSavedMediaSyncWorker(
        store=Store(),
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        poll_seconds=30.0,
        join_timeout_seconds=0.01,
    )
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="saved media sync"):
            worker.stop()
        assert not acknowledged.is_set()
        assert worker._thread.is_alive()
    finally:
        release.set()
        worker._thread.join(1)
        worker.stop()
    assert not acknowledged.is_set()
    assert not worker._thread.is_alive()


def test_postgame_worker_command_reuses_installed_models_and_authenticated_cloud_scope(
    tmp_path,
) -> None:
    module = load_script()
    args = SimpleNamespace(
        state_db=tmp_path / "field.sqlite3",
        jobs_root=tmp_path / "jobs",
        artifact_root=tmp_path / "artifacts",
        capture_setup=tmp_path / "capture-setup.json",
        calibration_root=tmp_path / "calibrations",
        config=tmp_path / "shot-config.json",
        detector_training_run=tmp_path / "detector",
        identity_training_run=tmp_path / "identity-v4",
        dinov2_assets=tmp_path / "dinov2",
        device="cuda",
        batch_size=8,
        sync_base_url="https://development.example/sync-api",
        device_id="raytheon-01",
        credential_target="billiardsAI/development/raytheon-01",
        expected_development_environment_id="cloud-development",
        existing_development_control_only=False,
    )

    command = module._postgame_worker_command(
        args,
        server_port=8787,
        session_id="session-ended",
    )

    joined = " ".join(str(value) for value in command)
    assert "run_mainline1_shot_worker.py" in joined
    assert "--identity-training-run" in command
    assert str(args.identity_training_run) in command
    assert "--development-analysis-sync" in command
    assert "--live-resource-url" in command
    assert "http://127.0.0.1:8787/api/screen" in command


def test_help_and_reject_numeric_camera_outside_project(tmp_path):
    assert SCRIPT.is_file(), "live service CLI is missing"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for flag in (
        "--jobs-root",
        "--state-db",
        "--device-locator",
        "--device-fingerprint",
        "--join-qr-asset",
        "--rule-profile",
        "--runtime-environment",
        "--cloudbase-environment-id",
        "--expected-development-environment-id",
        "--identity-gate-status",
        "--observation-only",
        "--existing-development-control-only",
        "--control-token-file",
    ):
        assert flag in result.stdout
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments(tmp_path), "--camera-index", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "state.sqlite").exists()


def test_cloud_configuration_rejects_production_or_unknown_scope_before_server_build(
    tmp_path, monkeypatch
):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--sync-base-url",
        "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com",
        "--device-id",
        "raytheon-01",
        "--credential-target",
        "billiardsAI/development/raytheon-01",
        "--runtime-environment",
        "production",
        "--cloudbase-environment-id",
        "cloud1-public-example",
        "--expected-development-environment-id",
        "cloud1-public-example",
        "--identity-gate-status",
        "PASS",
    ]
    calls = []
    monkeypatch.setattr(module, "prepare_runtime", lambda args: calls.append("runtime"))
    monkeypatch.setattr(
        module, "build_live_field_assistance_server", lambda **kwargs: calls.append("server")
    )

    assert module.main(argv) == 2
    assert calls == []


def test_authenticated_cloud_attestation_blocks_mismatched_endpoint_before_runtime(
    tmp_path, monkeypatch
):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--sync-base-url",
        "https://cloud-development.invalid/sync-api",
        "--device-id",
        "raytheon-01",
        "--credential-target",
        "billiardsAI/development/raytheon-01",
        "--runtime-environment",
        "development",
        "--cloudbase-environment-id",
        "cloud-development",
        "--expected-development-environment-id",
        "cloud-development",
    ]
    calls = []

    class FakeSignedGateway:
        def __init__(self, **_kwargs):
            pass

        def runtime_attestation(self):
            return SimpleNamespace(
                runtime_environment="production",
                cloudbase_environment_id="cloud-production",
            )

    inject_cloud_factories(module, monkeypatch, signed_gateway=FakeSignedGateway)
    monkeypatch.setattr(module, "prepare_runtime", lambda args: calls.append("runtime"))

    assert module.main(argv) == 2
    assert calls == []


def test_existing_development_control_only_uses_legacy_endpoints_without_attestation(
    tmp_path, monkeypatch
):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--sync-base-url",
        "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com",
        "--device-id",
        "raytheon-01",
        "--credential-target",
        "billiardsAI/development/raytheon-01",
        "--runtime-environment",
        "development",
        "--cloudbase-environment-id",
        "cloud1-public-example",
        "--expected-development-environment-id",
        "cloud1-public-example",
        "--observation-only",
        "--existing-development-control-only",
    ]
    calls = []

    class Signed:
        def __init__(self, **kwargs):
            assert kwargs["transport"]._allow_redirects is False
            calls.append("signed")

        def open_field_session(self, session_id):
            calls.append(("open", session_id))
            return FieldSessionLease(session_id, "raytheon-01", "open")

        def runtime_attestation(self):
            raise AssertionError("compatibility mode must not call runtime attestation")

        def publish_session_analysis(self, payload):
            raise AssertionError("compatibility mode must not publish analysis")

    class Server:
        server_port = 1234
        field_state = SimpleNamespace(
            wait_until_capture_ready=lambda timeout_seconds: "epoch-validated",
            record_store=object(),
            match_lock=None,
            before_field_command=None,
            set_command_match=None,
            handle_command_application=None,
            set_cloud_control_status=None,
            maintain_background=None,
            resolve_command_confirmation=None,
            reduce_live_command=None,
        )

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def build(**kwargs):
        lease = kwargs["cloud_gateway"].open_field_session("session-1")
        assert lease.table_id == "raytheon-01"
        return Server()

    monkeypatch.setattr(
        module,
        "prepare_runtime",
        lambda args, *, initial_source_epoch_id: (
            "cal-1",
            lambda session_id, channel, count, epoch: None,
            object(),
        ),
    )
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)

    class Commands:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class Postgame:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    inject_cloud_factories(
        module,
        monkeypatch,
        signed_gateway=Signed,
        command_worker=Commands,
        postgame_worker=Postgame,
        saved_media_worker=lambda **_kwargs: pytest.fail(
            "control-only mode must not construct saved-media sync"
        ),
    )

    assert module.main(argv) == 130
    assert calls == ["signed", ("open", "session-1")]


@pytest.mark.parametrize("failed_start", [None, "postgame", "saved", "commands", "facts"])
def test_full_development_sync_owns_saved_media_worker_and_wake_lifecycle(
    tmp_path, monkeypatch, failed_start
) -> None:
    module = load_script()
    argv = arguments(tmp_path) + [
        "--sync-base-url",
        "https://cloud-development.example/cloud-development/sync-api",
        "--device-id",
        "raytheon-01",
        "--credential-target",
        "billiardsAI/development/raytheon-01",
        "--runtime-environment",
        "development",
        "--cloudbase-environment-id",
        "cloud-development",
        "--expected-development-environment-id",
        "cloud-development",
    ]
    lifecycle = []
    store = object()

    class Credentials:
        def __init__(self, **kwargs):
            lifecycle.append(("credentials", kwargs))

    class Transport:
        def __init__(self, base_url, *, allow_redirects):
            assert allow_redirects is False
            lifecycle.append(("transport", base_url))

    class Gateway:
        def __init__(self, **kwargs):
            lifecycle.append("gateway")

        def runtime_attestation(self):
            lifecycle.append("attest")
            return object()

    class Server:
        server_port = 1234
        field_state = SimpleNamespace(
            record_store=store,
            match_lock=object(),
            before_field_command=lambda: None,
            set_command_match=lambda value: None,
            handle_command_application=lambda value: None,
            set_cloud_control_status=lambda value: None,
            maintain_background=lambda: None,
            resolve_command_confirmation=lambda value: None,
            reduce_live_command=lambda value: None,
            wait_until_capture_ready=lambda timeout_seconds: (
                lifecycle.append("ready") or "epoch-ready"
            ),
        )

        def serve_forever(self):
            lifecycle.append("serve")
            raise KeyboardInterrupt

        def server_close(self):
            lifecycle.append("close")

    worker_instances = {}

    class Worker:
        def __init__(self, kind, **kwargs):
            self.kind = kind
            self.kwargs = kwargs
            worker_instances[kind] = self
            lifecycle.append((kind, "construct"))

        def start(self):
            lifecycle.append((self.kind, "start"))
            if self.kind == failed_start:
                raise RuntimeError("worker start failed")

        def stop(self):
            lifecycle.append((self.kind, "stop"))

    sync_clients = []

    def sync_client(**kwargs):
        value = SimpleNamespace(kind="saved" if "before_mutation" in kwargs else "facts")
        sync_clients.append((value, kwargs))
        return value

    observed_server = {}

    def build(**kwargs):
        observed_server.update(kwargs)
        return Server()

    monkeypatch.setattr(
        module,
        "prepare_runtime",
        lambda args, *, initial_source_epoch_id: (
            "cal-1",
            lambda session_id, channel, count, epoch: None,
            object(),
        ),
    )
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    inject_cloud_factories(
        module,
        monkeypatch,
        credentials=Credentials,
        transport=Transport,
        signed_gateway=lambda **kwargs: object(),
        attested_gateway=Gateway,
        sync_client=sync_client,
        postgame_worker=lambda **kwargs: Worker("postgame", **kwargs),
        command_worker=lambda **kwargs: Worker("commands", **kwargs),
        live_fact_worker=lambda **kwargs: Worker("facts", **kwargs),
        saved_media_worker=lambda **kwargs: Worker("saved", **kwargs),
    )

    assert module.main(argv) == (130 if failed_start is None else 2)
    assert lifecycle.count("close") == 1
    assert lifecycle[-1] == "close"
    assert [event for event in lifecycle if isinstance(event, tuple) and event[1] == "stop"] == [
        (name, "stop")
        for name in ("commands", "facts", "postgame", "saved")
        if name in worker_instances
    ]
    if failed_start is not None:
        assert "serve" not in lifecycle
        return

    wake = observed_server["sync_wakeup"]
    assert wake.__name__ == "set"
    assert isinstance(wake.__self__, threading.Event)
    saved = worker_instances["saved"]
    assert saved.kwargs["store"] is store
    assert saved.kwargs["trusted_postgame_root"] == tmp_path / "artifact-root" / "postgame"
    assert saved.kwargs["wake_event"] is wake.__self__
    assert saved.kwargs["client"].kind == "saved"
    saved_client_kwargs = next(kwargs for client, kwargs in sync_clients if client.kind == "saved")
    assert saved_client_kwargs["before_mutation"].__self__ is observed_server["cloud_gateway"]
    assert lifecycle.index("attest") < lifecycle.index("ready")
    assert lifecycle.index("ready") < lifecycle.index(("saved", "start"))
    assert lifecycle.index(("postgame", "stop")) < lifecycle.index(("saved", "stop"))
    assert lifecycle.index(("saved", "stop")) < lifecycle.index("close")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--sync-base-url", "https://cloud1-public-example.evil.invalid/sync-api"),
        ("--device-id", "another-device"),
        ("--credential-target", "billiardsAI/development/another-device"),
    ],
)
def test_existing_development_control_only_rejects_any_unpinned_scope(
    tmp_path, monkeypatch, flag, value
):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--sync-base-url",
        "https://cloud1-public-example-1461180277.ap-shanghai.app.tcloudbase.com",
        "--device-id",
        "raytheon-01",
        "--credential-target",
        "billiardsAI/development/raytheon-01",
        "--runtime-environment",
        "development",
        "--cloudbase-environment-id",
        "cloud1-public-example",
        "--expected-development-environment-id",
        "cloud1-public-example",
        "--observation-only",
        "--existing-development-control-only",
    ]
    argv[argv.index(flag) + 1] = value
    calls = []
    monkeypatch.setattr(module, "prepare_runtime", lambda args: calls.append("runtime"))

    assert module.main(argv) == 2
    assert calls == []


def test_failed_identity_cannot_receive_rule_authority(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--identity-gate-status",
        "FAIL",
        "--rule-profile",
        "cbsa-chinese-pool-2017-observable-v1",
    ]
    calls = []
    monkeypatch.setattr(module, "prepare_runtime", lambda args: calls.append("runtime"))

    assert module.main(argv) == 2
    assert calls == []


def test_missing_required_path_rejects_before_capture_or_state_creation(tmp_path):
    module = load_script()
    argv = arguments(tmp_path)
    (tmp_path / "native-helper").unlink()
    assert module.main(argv) == 2
    assert not (tmp_path / "state.sqlite").exists()


@pytest.mark.parametrize("failure", ["bind", "ready"])
def test_startup_failure_closes_capture_once_without_constructing_workers(
    tmp_path, monkeypatch, failure
):
    module = load_script()
    argv = arguments(tmp_path)
    lifecycle = []

    class Source:
        def cancel(self):
            lifecycle.append("cancel")

    source = Source()

    def prepare(args, *, initial_source_epoch_id):
        lifecycle.append("prepare")
        return "cal-1", object(), source

    def ready(*, timeout_seconds):
        lifecycle.append("ready")
        raise RuntimeError("capture not ready")

    class Server:
        field_state = SimpleNamespace(record_store=object(), wait_until_capture_ready=ready)

        def server_close(self):
            lifecycle.append("close")
            source.cancel()

    def build(**kwargs):
        lifecycle.append("bind")
        if failure == "bind":
            raise OSError("bind failed")
        return Server()

    monkeypatch.setattr(module, "prepare_runtime", prepare)
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    inject_cloud_factories(
        module,
        monkeypatch,
        postgame_worker=lambda **kwargs: pytest.fail("workers require capture readiness"),
    )

    assert module.main(argv) == 2
    assert lifecycle == (
        ["prepare", "bind", "cancel"]
        if failure == "bind"
        else ["prepare", "bind", "ready", "close", "cancel"]
    )


def test_service_owns_runtime_and_closes_on_interrupt(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path)
    lifecycle = []

    def prepare(args, *, initial_source_epoch_id):
        return "cal-1", lambda session_id, channel, count, epoch: None, object()

    class Server:
        server_port = 1234
        field_state = SimpleNamespace(
            wait_until_capture_ready=lambda timeout_seconds: (
                lifecycle.append(("ready", timeout_seconds)) or "epoch-validated"
            )
        )

        def serve_forever(self):
            lifecycle.append("serve")
            raise KeyboardInterrupt

        def server_close(self):
            lifecycle.append("close")

    def build(**kwargs):
        assert callable(kwargs["source_epoch_factory"])
        assert kwargs["state_db_path"] == tmp_path / "state.sqlite"
        assert kwargs["artifact_path"] == tmp_path / "artifact-root"
        assert kwargs["control_token"] == "secret"
        assert kwargs["rule_profile_id"] is None
        return Server()

    monkeypatch.setattr(module, "prepare_runtime", prepare)
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    assert module.main(argv) == 130
    assert lifecycle == [("ready", 30.0), "serve", "close"]


def test_local_service_can_omit_join_qr_without_forwarding_an_empty_asset(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path)
    qr_index = argv.index("--join-qr-asset")
    del argv[qr_index : qr_index + 2]
    observed = {}

    class Server:
        server_port = 1234
        field_state = SimpleNamespace(
            wait_until_capture_ready=lambda timeout_seconds: "epoch-validated"
        )

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(
        module,
        "prepare_runtime",
        lambda args, *, initial_source_epoch_id: (
            "cal-1",
            lambda session_id, channel, count, epoch: None,
            object(),
        ),
    )

    def build(**kwargs):
        observed.update(kwargs)
        return Server()

    monkeypatch.setattr(module, "build_live_field_assistance_server", build)

    assert module.main(argv) == 130
    assert "join_qr_asset_path" not in observed


def test_fixed_rule_profile_is_forwarded_explicitly(tmp_path, monkeypatch):
    module = load_script()
    argv = arguments(tmp_path) + [
        "--rule-profile",
        "cbsa-chinese-pool-2017-observable-v1",
        "--identity-gate-status",
        "PASS",
    ]
    observed = {}

    class Server:
        server_port = 1234
        field_state = SimpleNamespace(
            wait_until_capture_ready=lambda timeout_seconds: "epoch-validated"
        )

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(
        module,
        "prepare_runtime",
        lambda args, *, initial_source_epoch_id: (
            "cal-1",
            lambda session_id, channel, count, epoch: None,
            object(),
        ),
    )

    def build(**kwargs):
        observed.update(kwargs)
        return Server()

    monkeypatch.setattr(module, "build_live_field_assistance_server", build)

    assert module.main(argv) == 130
    assert observed["rule_profile_id"] == "cbsa-chinese-pool-2017-observable-v1"


def test_cancel_before_runner_start_never_launches_helper():
    calls = []

    class Source:
        def start(self):
            calls.append("start")

        def cancel(self):
            calls.append("cancel")

    class Runtime:
        def run(self):
            calls.append("run")

    handle = field_runtime.RuntimeHandle(
        Source(), lambda: Runtime(), capture_starter=lambda source: source.start()
    )
    handle.close()
    handle.run()
    assert "start" not in calls and "run" not in calls


def test_failed_helper_start_does_not_construct_runtime_workers():
    calls = []

    class Source:
        def start(self):
            raise OSError("device missing")

        def cancel(self):
            calls.append("cancel")

    def create_runtime():
        calls.append("workers")
        raise AssertionError("runtime workers must not be constructed")

    handle = field_runtime.RuntimeHandle(
        Source(), create_runtime, capture_starter=lambda source: source.start()
    )
    try:
        handle.run()
    except OSError:
        pass
    assert calls == ["cancel"]


def test_runtime_workers_are_created_only_after_successful_helper_start():
    calls = []

    class Source:
        def start(self):
            calls.append("start")

        def cancel(self):
            calls.append("cancel")

    class Runtime:
        def run(self):
            calls.append("run")

    def create_runtime():
        calls.append("workers")
        return Runtime()

    handle = field_runtime.RuntimeHandle(
        Source(), create_runtime, capture_starter=lambda source: source.start()
    )
    assert calls == []
    handle.run()
    assert calls == ["start", "workers", "run", "cancel"]


def test_runtime_handle_exposes_current_camera_frame_and_calibration():
    expected_frame = object()
    expected_calibration = object()

    class Source:
        def start(self):
            pass

        def cancel(self):
            pass

    class Runtime:
        current_capture_frame = expected_frame
        calibration = expected_calibration

        def run(self):
            pass

    handle = field_runtime.RuntimeHandle(
        Source(), Runtime, capture_starter=lambda source: source.start()
    )
    handle.run()

    assert handle.current_capture_frame is expected_frame
    assert handle.calibration is expected_calibration


def test_analysis_runtime_failure_does_not_cancel_capture_hub_camera():
    """Catches analysis shutdown being promoted to physical camera shutdown."""
    release_capture = threading.Event()

    class Source:
        def __init__(self):
            self.cancelled = False

        def start(self):
            pass

        def frames(self, stop):
            yield CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=0,
                captured_at_ns=0,
                width=16,
                height=12,
                mjpeg=b"healthy-camera",
            )
            while not stop.is_set() and not release_capture.wait(0.01):
                pass

        def cancel(self):
            self.cancelled = True
            release_capture.set()

    class Runtime:
        calibration = "calibration"

        def __init__(self, analysis_source):
            self.analysis_source = analysis_source

        def run(self):
            raise RuntimeError("analysis failed")

    source = Source()
    handle = field_runtime.RuntimeHandle(
        source,
        Runtime,
        capture_config=RealtimeCaptureConfig(width=16, height=12, fps=60),
    )

    with pytest.raises(RuntimeError, match="analysis failed"):
        handle.run()

    assert source.cancelled is False
    assert handle.current_capture_frame.mjpeg == b"healthy-camera"
    handle.shutdown()
    assert source.cancelled is True


def test_runtime_validates_the_started_source_before_creating_workers():
    calls = []

    class Source:
        def start(self):
            calls.append("start")

        def cancel(self):
            calls.append("cancel")

    class Runtime:
        def run(self):
            calls.append("run")

    def validate(source):
        assert isinstance(source, Source)
        calls.append("validate")

    def create_runtime():
        calls.append("workers")
        return Runtime()

    handle = field_runtime.RuntimeHandle(
        Source(),
        create_runtime,
        capture_starter=lambda source: source.start(),
        startup_validator=validate,
    )
    handle.run()

    assert calls == ["start", "validate", "workers", "run", "cancel"]


def test_runtime_ready_wait_requires_completed_startup_validation():
    validation_entered = threading.Event()
    allow_validation = threading.Event()
    runtime_entered = threading.Event()
    release_runtime = threading.Event()

    class Source:
        def start(self):
            pass

        def cancel(self):
            pass

    class Runtime:
        def run(self):
            runtime_entered.set()
            assert release_runtime.wait(1)

    def validate(_source):
        validation_entered.set()
        assert allow_validation.wait(1)

    handle = field_runtime.RuntimeHandle(
        Source(),
        Runtime,
        capture_starter=lambda source: source.start(),
        startup_validator=validate,
    )
    runner = threading.Thread(target=handle.run)
    runner.start()
    try:
        assert validation_entered.wait(1)
        assert handle.wait_until_ready(timeout_seconds=0.01) is False
        assert not runtime_entered.is_set()
        allow_validation.set()
        assert handle.wait_until_ready(timeout_seconds=1) is True
    finally:
        allow_validation.set()
        release_runtime.set()
        handle.close()
        runner.join(1)
    assert not runner.is_alive()
    assert runtime_entered.is_set()


def test_runtime_ready_wait_rejects_worker_construction_failure():
    class Source:
        def start(self):
            pass

        def cancel(self):
            pass

    def fail_worker_construction():
        raise ValueError("worker construction failed")

    handle = field_runtime.RuntimeHandle(
        Source(), fail_worker_construction, capture_starter=lambda source: source.start()
    )
    with pytest.raises(ValueError, match="worker construction failed"):
        handle.run()
    assert handle.wait_until_ready(timeout_seconds=1) is False
    assert handle.failure_reason == "ValueError: worker construction failed"


@pytest.mark.parametrize("development_acceptance", [False, True, "before", "during"])
def test_camera_lifecycle_closes_preflight_before_models_then_opens_live(
    tmp_path, monkeypatch, development_acceptance
):
    """Catches a camera epoch leaking across preflight, model loading, and live use."""
    module = load_script()
    lifecycle = []
    sources = []

    class Source:
        identity = SimpleNamespace(durable_fingerprint="d" * 64)

        def __init__(self, epoch):
            self.epoch = epoch
            self.cancelled = False

        def start(self):
            lifecycle.append((self.epoch, "start"))

        def frames(self, stop):
            index = 132
            while not stop.is_set() and not self.cancelled:
                yield CaptureFrame(
                    source_epoch_id=self.epoch,
                    capture_frame_id=index,
                    captured_at_ns=index * 16_666_667,
                    width=1920,
                    height=1080,
                    mjpeg=f"frame-{index}".encode(),
                )
                index += 1

        def cancel(self):
            self.cancelled = True
            lifecycle.append((self.epoch, "cancel"))

    geometry = FrameGeometry(width=1920, height=1080, crop=(0, 0, 1920, 1080))
    setup = SimpleNamespace(
        source_kind=module.realtime.CaptureSourceKind.USB,
        geometry=geometry,
        requested_fps=60,
        device_locator_fingerprint="d" * 64,
        camera_device_id="camera-1",
        image_controls=(),
        capture_setup_id="setup-1",
    )
    calibration = SimpleNamespace(device_id="camera-1")
    artifact_path = Path("calibration-1")
    calibration_file = tmp_path / artifact_path / "calibration.json"
    calibration_file.parent.mkdir()
    calibration_file.write_text("{}")
    record = SimpleNamespace(artifact_id="calibration-1", artifact_path=artifact_path)
    evidence = SimpleNamespace(
        capture_frame_id=131,
        timestamp_ms=2183.0,
        decode_bgr=lambda: object(),
    )
    gate = SimpleNamespace(
        status=CalibrationGateStatus.READY,
        calibration_artifact_id="calibration-1",
    )
    vision = SimpleNamespace(infer=lambda frame: SimpleNamespace(detections=()))
    identity = SimpleNamespace(classify=lambda **kwargs: None)

    monkeypatch.setattr(module.realtime, "load_capture_setup", lambda path: setup)
    monkeypatch.setattr(module.realtime, "capture_setup_fingerprint", lambda value: "fp")
    monkeypatch.setattr(module.realtime, "resolve_active_calibration", lambda *args: record)
    monkeypatch.setattr(
        module.realtime.TableCalibration,
        "model_validate_json",
        lambda payload: calibration,
    )

    def capture(capture_args, *, epoch, controls):
        assert capture_args is args
        assert controls is setup.image_controls
        source = Source(epoch)
        sources.append(source)
        lifecycle.append((epoch, "create"))
        return source

    monkeypatch.setattr(module.realtime, "_capture", capture)
    expected_setup, expected_calibration = setup, calibration

    def validate(source, *, setup, calibration, calibration_root):
        assert source in sources
        assert setup is expected_setup
        assert calibration is expected_calibration
        assert calibration_root == tmp_path
        return gate, (evidence,)

    monkeypatch.setattr(module, "_validate_started_usb_source", validate)
    monkeypatch.setattr(
        module.realtime,
        "load_mainline1_shot_local_config",
        lambda path: SimpleNamespace(window=object()),
    )
    monkeypatch.setattr(module.realtime, "Dinov2IdentityFeatureBackend", lambda **kwargs: object())

    def load_vision(*model_args, **kwargs):
        del model_args, kwargs
        lifecycle.append("model-load")
        if development_acceptance == "during":
            write_identity_assets(args.identity_training_run, gate_violations=["changed failure"])
        return vision

    monkeypatch.setattr(module.realtime, "load_development_vision_backend", load_vision)
    monkeypatch.setattr(
        module.realtime,
        "load_development_group_classifier",
        lambda *args, **kwargs: identity,
    )
    monkeypatch.setattr(
        module.realtime,
        "make_vision_frame_input",
        lambda *args, **kwargs: object(),
    )
    builder_options = []

    def build_factory(**kwargs):
        builder_options.append(kwargs)
        return lambda session_id, channel, initial_shot_number, source_epoch_id: kwargs[
            "capture_factory"
        ](source_epoch_id)

    monkeypatch.setattr(module, "build_live_runtime_factory", build_factory, raising=False)
    args = SimpleNamespace(
        capture_setup=tmp_path / "capture-setup.json",
        device_fingerprint="d" * 64,
        calibration_root=tmp_path,
        config=tmp_path / "shot-local.yaml",
        dinov2_assets=tmp_path,
        device="cuda",
        batch_size=4,
        detector_training_run=tmp_path,
        identity_training_run=tmp_path,
        jobs_root=tmp_path,
        ring_root=tmp_path,
    )

    expected_acceptance = None
    if development_acceptance:
        from billiards_ai.lab.field_assistance.development_acceptance import (
            resolve_acceptance_policy,
        )
        from tests.support.development_acceptance import write_identity_assets

        args.identity_training_run = tmp_path / "synthetic-identity"
        expected_acceptance = write_identity_assets(args.identity_training_run)
        args.development_model_acceptance = True
        args.runtime_environment = "development"
        args.identity_gate_status = "FAIL"
        args.rule_profile = module.PROFILE_ID
        args.observation_only = False
        args._development_acceptance_policy = resolve_acceptance_policy(args)
    if development_acceptance in ("before", "during"):
        if development_acceptance == "before":
            write_identity_assets(args.identity_training_run, gate_violations=["changed failure"])
        with pytest.raises(ValueError, match="changed"):
            module.prepare_runtime(args, initial_source_epoch_id="epoch-live")
        if development_acceptance == "before":
            assert sources == []
        else:
            assert len(sources) == 1 and sources[0].cancelled
        return
    _, runtime_factory, prepared_source = module.prepare_runtime(
        args, initial_source_epoch_id="epoch-live"
    )
    args.config.write_bytes(b"config")
    live_source = runtime_factory("session-1", object(), 1, "epoch-live")

    assert sources[0].cancelled is True
    assert len(sources) == 2
    assert sources[0].epoch.startswith("preflight-")
    assert live_source is sources[1]
    assert sources[1].epoch == "epoch-live"
    assert lifecycle.index((sources[0].epoch, "cancel")) < lifecycle.index("model-load")
    assert lifecycle.index("model-load") < lifecycle.index(("epoch-live", "create"))
    assert prepared_source is None
    options = builder_options[0]
    options["capture_starter"](live_source)
    assert lifecycle[-1] == ("epoch-live", "start")
    assert options["startup_validator"](live_source) == (gate, (evidence,))
    assert options["startup_progress"] is module._write_startup_progress
    assert options["detector"] is vision
    assert options["identity_classifier"] is identity
    assert options["calibration"] is calibration
    assert options["config_path"] == args.config
    assert options["ring_root"] == args.ring_root
    assert options["jobs_root"] == args.jobs_root
    assert options["capture_setup_id"] == "setup-1"
    assert options["calibration_artifact_id"] == "calibration-1"
    if development_acceptance:
        assert options["capture_diagnostics_factory"](live_source, "epoch-live") == {
            "acceptance": expected_acceptance,
        }
    else:
        assert "capture_diagnostics_factory" not in options


def test_started_usb_source_discards_camera_warmup_before_calibration_evidence() -> None:
    module = load_script()
    frames = tuple(SimpleNamespace(capture_frame_id=index) for index in range(132))

    class Source:
        def frames(self, stop):
            del stop
            return iter(frames)

    sampled = module._sample_started_source(Source())

    assert [frame.capture_frame_id for frame in sampled] == list(range(120, 132))


def test_usb_service_startup_progress_is_machine_readable(capsys) -> None:
    module = load_script()

    module._write_startup_progress("identity_warmup_started")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "USB_FIELD_STARTING",
        "stage": "identity_warmup_started",
    }


def test_started_usb_source_uses_fixed_installation_gate_without_corner_redetection(
    tmp_path, monkeypatch
):
    module = load_script()
    geometry = FrameGeometry(width=1920, height=1080, crop=(0, 0, 1920, 1080))
    controls = (("exposure", -7.0), ("auto_exposure", 0.0))
    setup = SimpleNamespace(geometry=geometry, image_controls=controls)
    frames = (
        SimpleNamespace(
            source_epoch_id="epoch-live-1",
            capture_frame_id=index,
            timestamp_ms=float(index),
            mjpeg=f"frame-{index}".encode(),
            decode_bgr=lambda: pytest.fail("fixed installation startup must not redetect corners"),
        )
        for index in range(132)
    )

    class Source:
        identity = SimpleNamespace(durable_fingerprint="d" * 64)

        def frames(self, stop):
            return iter(frames)

    observed = {}

    def evaluate(**kwargs):
        observed.update(gate_kwargs=kwargs)
        return SimpleNamespace(
            status=CalibrationGateStatus.READY,
            reasons=(),
            calibration_artifact_id="calibration-1",
        )

    monkeypatch.setattr(module, "evaluate_usb_calibration_gate", evaluate)
    result, sampled = module._validate_started_usb_source(
        Source(),
        setup=setup,
        calibration=object(),
        calibration_root=tmp_path,
    )

    assert result.calibration_artifact_id == "calibration-1"
    assert len(sampled) == 12
    assert observed["gate_kwargs"]["max_corner_drift_px"] == 12.0
    assert observed["gate_kwargs"]["require_current_corner_evidence"] is False
    metadata = observed["gate_kwargs"]["source"]
    assert metadata.source_epoch_id == "epoch-live-1"
    assert metadata.device_fingerprint == "d" * 64
    assert metadata.geometry == geometry
    assert metadata.image_controls == controls
    assert metadata.evidence_frame_sha256 == hashlib.sha256(b"frame-131").hexdigest()
    assert metadata.validation_corners_px is None


def test_started_usb_source_fails_closed_when_canonical_gate_is_not_ready(tmp_path, monkeypatch):
    module = load_script()
    geometry = FrameGeometry(width=1920, height=1080, crop=(0, 0, 1920, 1080))
    setup = SimpleNamespace(geometry=geometry, image_controls=(("exposure", -7.0),))
    frames = tuple(
        SimpleNamespace(
            source_epoch_id="epoch-live-2",
            capture_frame_id=index,
            timestamp_ms=float(index),
            mjpeg=b"frame",
            decode_bgr=lambda: object(),
        )
        for index in range(132)
    )

    class Source:
        identity = SimpleNamespace(durable_fingerprint="d" * 64)

        def frames(self, stop):
            return iter(frames)

    monkeypatch.setattr(
        module,
        "evaluate_usb_calibration_gate",
        lambda **kwargs: SimpleNamespace(
            status=CalibrationGateStatus.CALIBRATION_DRIFTED,
            reasons=("corner_drift_exceeds_threshold",),
        ),
    )

    with pytest.raises(ValueError, match="corner_drift_exceeds_threshold"):
        module._validate_started_usb_source(
            Source(),
            setup=setup,
            calibration=object(),
            calibration_root=tmp_path,
        )


@pytest.mark.parametrize(
    "conflict",
    ["environment", "nondev", "profile", "observation", "PASS", "UNDECLARED", "actual-PASS"],
)
def test_development_acceptance_rejected_before_cloud(tmp_path, monkeypatch, conflict):
    from tests.support.development_acceptance import write_identity_assets

    module = load_script()
    argv = arguments(tmp_path) + [
        "--development-model-acceptance",
        "--identity-gate-status",
        "FAIL",
    ]
    if conflict != "environment":
        argv += ["--runtime-environment", "production" if conflict == "nondev" else "development"]
    if conflict != "profile":
        argv += ["--rule-profile", module.PROFILE_ID]
    if conflict == "observation":
        argv += ["--observation-only"]
    if conflict in ("PASS", "UNDECLARED"):
        argv[argv.index("--identity-gate-status") + 1] = conflict
    write_identity_assets(
        tmp_path / "identity-training-run",
        gate_status="PASS" if conflict == "actual-PASS" else "FAIL",
    )
    monkeypatch.setattr(
        module, "prepare_cloud_runtime", lambda *a, **kw: pytest.fail("cloud prepared")
    )
    assert module.main(argv) == 2


def test_development_acceptance_ready_retains_full_rules(tmp_path, monkeypatch, capsys):
    from tests.support.development_acceptance import write_identity_assets

    module = load_script()
    argv = arguments(tmp_path) + [
        "--development-model-acceptance",
        "--runtime-environment",
        "development",
        "--identity-gate-status",
        "FAIL",
        "--rule-profile",
        module.PROFILE_ID,
    ]
    expected = write_identity_assets(tmp_path / "identity-training-run")
    observed = {}
    monkeypatch.setattr(
        module,
        "prepare_cloud_runtime",
        lambda *a, **kw: SimpleNamespace(
            gateway=None,
            sync_wakeup=None,
            start=lambda *a, **kw: None,
            stop=lambda: None,
        ),
    )

    def prepare(args, *, initial_source_epoch_id):
        assert args._development_acceptance_policy.as_payload() == expected
        return "cal", "factory", None

    def build(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            field_state=SimpleNamespace(wait_until_capture_ready=lambda **kw: "validated"),
            server_port=1234,
            serve_forever=lambda: None,
            server_close=lambda: None,
        )

    monkeypatch.setattr(module, "prepare_runtime", prepare)
    monkeypatch.setattr(module, "build_live_field_assistance_server", build)
    assert module.main(argv) == 0
    ready = [json.loads(line) for line in capsys.readouterr().out.splitlines()][-1]
    assert ready["acceptance"] == expected
    assert observed["rule_profile_id"] == module.PROFILE_ID
