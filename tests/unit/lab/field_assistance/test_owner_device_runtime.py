"""Metadata-only worker fairness, fencing, and owned runtime shutdown."""

# ruff: noqa: F811
from __future__ import annotations

import importlib
import threading
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.sync import (
    OwnerDeviceCompletionResult,
    OwnerDevicePollResult,
)
from tests.unit.edge.route_assistance.test_owner_device_delivery import ack, finish, queue
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    scene,  # noqa: F401
    task,
)


def worker(store, client, **kwargs):
    cls = importlib.import_module(
        "billiards_ai.lab.field_assistance.cloud_runtime"
    )._BackgroundOwnerDeviceDeletionWorker
    return cls(store=store, client=client, managed_scope=SCOPE, now_utc=lambda: NOW, **kwargs)


class Client:
    def __init__(self, store):
        self.store, self.calls = store, []

    def poll_owner_delete_once(self, scope):
        self.calls.append("poll")
        # Prove no transaction is kept across HTTP by acquiring a second writer.
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
        return OwnerDevicePollResult("task", task=task(self.store))

    def complete_owner_delete_once(self, item):
        self.calls.append("complete")
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
        return OwnerDeviceCompletionResult("ack", ack=ack(item))

    def close_owner_delete_transport(self):
        self.calls.append("close")


def test_durable_receive_send_turn_and_exact_ack_offline_completion(scene):
    finish(scene)
    client = Client(scene)
    first = worker(scene, client)
    assert first.run_once()
    assert client.calls == ["poll"]
    restarted = worker(scene, client)
    assert restarted.run_once()
    assert client.calls == ["poll", "complete"]
    assert queue(scene).claim_due(NOW + timedelta(days=1)) is None


def test_stop_during_poll_suppresses_late_local_acceptance(scene):
    entered, release = threading.Event(), threading.Event()
    client = Client(scene)

    def poll(scope):
        entered.set()
        assert release.wait(2)
        return OwnerDevicePollResult("task", task=task(scene))

    client.poll_owner_delete_once = poll
    w = worker(scene, client, join_timeout_seconds=0.01)
    w.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError):
            w.stop()
        assert w._thread.is_alive() and "close" not in client.calls
    finally:
        release.set()
        w._thread.join(2)
    w.stop()
    assert scene.get_owner_delete(task(scene).clip_id) is None
    assert client.calls == ["close"]


def test_stop_during_completion_keeps_original_lease_unsettled(scene):
    finish(scene)
    queue(scene).next_turn()  # send is now due
    c = Client(scene)
    w = worker(scene, c)

    def complete(item):
        w._stopped.set()
        return OwnerDeviceCompletionResult("ack", ack=ack(item))

    c.complete_owner_delete_once = complete
    assert w.run_once()
    with scene._connect() as db:
        row = db.execute("SELECT status,ack_json FROM field_owner_completion_outbox").fetchone()
    assert tuple(row) == ("leased", None)


def test_retry_uses_fixed_error_and_stale_ack_cannot_settle(scene):
    finish(scene)
    queue(scene).next_turn()
    c = Client(scene)
    w = worker(scene, c)

    def complete(item):
        newer = queue(scene).claim_due(NOW + timedelta(seconds=60))
        assert newer.claim_token != item.claim_token
        return OwnerDeviceCompletionResult("ack", ack=ack(item))

    c.complete_owner_delete_once = complete
    w.run_once()
    with scene._connect() as db:
        assert (
            db.execute("SELECT status FROM field_owner_completion_outbox").fetchone()[0] == "leased"
        )


def test_cleanup_all_branches_bound_resolver_attempts_and_one_step(scene, tmp_path):
    from billiards_ai.edge.route_assistance.local_media_schedule import LocalCleanupCandidate
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
    from billiards_ai.lab.field_assistance.managed_resource_recovery import (
        ManagedRuntimeResourceUnavailable,
    )

    attempts, steps = [], []

    class Scheduler:
        def next_candidate(self, now):
            return LocalCleanupCandidate("discard", str(len(attempts)))

    class Resolver:
        def resolve(self, sid, **kw):
            attempts.append(sid)
            if len(attempts) <= 34:
                raise ManagedRuntimeResourceUnavailable(
                    "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE"
                )
            return SimpleNamespace(
                jobs_db=tmp_path / "original.db",
                artifact_root=tmp_path / "original",
                jobs_root=tmp_path / "originaljobs",
            )

    w = _BackgroundLocalMediaCleanupWorker(
        store=scene,
        jobs_db=tmp_path / "current.db",
        artifact_root=tmp_path,
        jobs_root=tmp_path,
        managed_scope=SCOPE,
        resource_resolver=Resolver(),
        scheduler=Scheduler(),
        process_once=lambda *a, **kw: steps.append((a, kw)) or SimpleNamespace(outcome="idle"),
    )
    assert not w.run_once() and len(attempts) == 16
    assert not w.run_once() and len(attempts) == 32
    assert not w.run_once() and len(attempts) == 35 and len(steps) == 1
    assert steps[0][0][1] == tmp_path / "original.db"


@pytest.mark.parametrize("stage", ["start", "stop"])
def test_runtime_registered_owner_failure_retains_cleanup_until_confirmed(tmp_path, stage):
    from billiards_ai.lab.field_assistance.cloud_runtime import (
        FieldCloudShutdownError,
        prepare_cloud_runtime,
    )
    from tests.unit.lab.field_assistance.test_cloud_runtime import _args, _dependencies, _server

    factories, calls, workers = _dependencies(
        failure=("owner_deletion", "start") if stage == "start" else None,
        stop_failure="owner_deletion" if stage == "stop" else None,
    )
    runtime = prepare_cloud_runtime(
        _args(tmp_path, managed_sessions=True, table_id="table-1"), factories=factories
    )
    server = _server()
    server.field_state.managed_session_scope = type(SCOPE)(
        "cloud-development", "raytheon-01", "table-1"
    )
    if stage == "start":
        with pytest.raises(RuntimeError, match="start failed"):
            runtime.start(server, postgame_command_factory=lambda sid: [sid])
        assert runtime._workers == {}
        assert ("owner_deletion", "stop") in calls
    else:
        runtime.start(server, postgame_command_factory=lambda sid: [sid])
        with pytest.raises(FieldCloudShutdownError):
            runtime.stop()
        assert set(runtime._workers) == {"owner_deletion", "cleanup"}
        assert ("cleanup", "stop") not in calls
        workers["owner_deletion"].stop = lambda: calls.append(("owner_deletion", "confirmed"))
        runtime.stop()
        assert runtime._workers == {}
        assert calls[-2:] == [("owner_deletion", "confirmed"), ("cleanup", "stop")]


def test_real_scheduler_restart_crosses_missing_pages_without_current_root_fallback(
    scene, tmp_path, monkeypatch
):
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundLocalMediaCleanupWorker
    from billiards_ai.lab.field_assistance.managed_resource_recovery import (
        ManagedRuntimeResourceUnavailable,
    )
    from tests.unit.edge.route_assistance.test_local_media_schedule import inject
    from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import SESSION, accept

    accept(scene)
    for i in range(35):
        inject(scene, "owner", f"managed-{i:03d}")

    def unbounded(*a, **kw):
        raise AssertionError("unbounded discovery")

    monkeypatch.setattr(scene, "list_pending_owner_delete_session_ids", unbounded)
    monkeypatch.setattr(scene, "list_pending_local_discard_session_ids", unbounded)
    attempts, steps = [], []

    class Resolver:
        def resolve(self, sid, **kwargs):
            attempts.append(sid)
            if sid != SESSION:
                raise ManagedRuntimeResourceUnavailable(
                    "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE"
                )
            return SimpleNamespace(
                jobs_db=tmp_path / "old.db",
                artifact_root=tmp_path / "old",
                jobs_root=tmp_path / "oldjobs",
            )

    def build():
        return _BackgroundLocalMediaCleanupWorker(
            store=scene,
            jobs_db=tmp_path / "current.db",
            artifact_root=tmp_path,
            jobs_root=tmp_path,
            managed_scope=SCOPE,
            resource_resolver=Resolver(),
            owner_process_once=lambda *a, **kw: (
                steps.append((a, kw)) or SimpleNamespace(outcome="idle")
            ),
        )

    assert not build().run_once() and len(attempts) == 16
    assert not build().run_once() and len(attempts) == 32
    assert not build().run_once() and len(attempts) == 36
    assert attempts == [f"managed-{i:03d}" for i in range(35)] + [SESSION]
    assert len(steps) == 1 and steps[0][0][1] == tmp_path / "old.db"
    inject(scene, "discard", "managed-zz-new")  # arrival after runtime construction
    attempts.clear()
    build().run_once()
    assert attempts[0] == "managed-zz-new"  # durable kind turn prevents owner starvation


def test_send_runs_while_poll_is_offline_and_only_one_protocol_operation_per_tick(scene):
    finish(scene)
    c = Client(scene)

    def poll(scope):
        c.calls.append("poll")
        return OwnerDevicePollResult("retry", "transport_unavailable")

    c.poll_owner_delete_once = poll
    w = worker(scene, c)
    assert w.run_once() and c.calls == ["poll"]
    assert w.run_once() and c.calls == ["poll", "complete"]
    assert not w.run_once() and c.calls == ["poll", "complete"]


def test_failed_native_thread_start_closes_dedicated_transport(scene, monkeypatch):
    c = Client(scene)
    w = worker(scene, c)
    monkeypatch.setattr(w._thread, "start", lambda: (_ for _ in ()).throw(RuntimeError("start")))
    with pytest.raises(RuntimeError):
        w.start()
    w.stop()
    assert c.calls == ["close"]


def test_runtime_owner_uses_separate_bounded_attestation_client_original_scope(tmp_path):
    from billiards_ai.lab.field_assistance.cloud_runtime import prepare_cloud_runtime
    from tests.unit.lab.field_assistance.test_cloud_runtime import _args, _dependencies, _server

    factories, calls, workers = _dependencies()
    bounded = object()
    requested = []
    original_signed = factories.signed_gateway

    class Signed(original_signed):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.transport = kwargs["transport"]
            self.credentials = kwargs["credentials"]

    def transport(url):
        requested.append(url)
        return bounded

    runtime = prepare_cloud_runtime(
        _args(tmp_path, managed_sessions=True, table_id="table-1"),
        factories=replace(factories, signed_gateway=Signed, owner_deletion_transport=transport),
    )
    assert requested == []
    server = _server()
    scope = type(SCOPE)("cloud-development", "raytheon-01", "table-1")
    server.field_state.managed_session_scope = scope
    runtime.start(server, postgame_command_factory=lambda sid: [sid])
    owned = workers["owner_deletion"].kwargs
    c = owned["client"]
    assert requested == [runtime._args.sync_base_url]
    assert c.transport is bounded and c.transport is not runtime._transport
    assert c.credentials is runtime._credentials and c.owner_scope == scope
    assert c.before_mutation.__self__ is not runtime.gateway
    assert c.before_mutation.__self__.delegate.transport is bounded
    assert c.before_mutation.__self__.expected_cloudbase_environment_id == scope.environment_id
    assert c.owner_cancelled.__self__ is owned["stop_event"]
    assert owned["managed_scope"] == scope
    runtime.stop()
    assert calls.index(("owner_deletion", "stop")) < calls.index(("cleanup", "stop"))


def test_stop_during_actual_bounded_attestation_reaps_child_without_local_accept(scene):
    import time

    from billiards_ai.edge.route_assistance.owner_device_transport import (
        OwnerDeviceMetadataHttpTransport,
    )
    from billiards_ai.edge.route_assistance.sync import FieldSyncClient
    from billiards_ai.lab.field_assistance.cloud_gateway import (
        DevelopmentAttestedFieldCloudGateway,
        SignedFieldCloudGateway,
    )
    from tests.unit.edge.route_assistance.test_owner_device_sync import Credentials
    from tests.unit.edge.route_assistance.test_owner_device_transport import endpoint

    entered = threading.Event()
    stopped = threading.Event()

    def stall(handler, seen):
        entered.set()
        time.sleep(0.8)
        handler.send_response(200)
        handler.end_headers()
        handler.wfile.write(b"{}")

    with endpoint(stall) as (url, seen, _):
        tx = OwnerDeviceMetadataHttpTransport(url, timeout_seconds=0.45)
        gateway = DevelopmentAttestedFieldCloudGateway(
            SignedFieldCloudGateway(credentials=Credentials(), transport=tx), SCOPE.environment_id
        )
        c = FieldSyncClient(
            credentials=Credentials(),
            transport=tx,
            before_mutation=gateway.runtime_attestation,
            owner_scope=SCOPE,
            owner_cancelled=stopped.is_set,
        )
        w = worker(scene, c, stop_event=stopped)
        w.start()
        assert entered.wait(1)
        process = tx._active
        root = tx._temporary.name
        started = time.monotonic()
        w.stop()
        assert time.monotonic() - started < 0.8
        from pathlib import Path

        assert process.poll() is not None and not Path(root).exists()
        assert not w._thread.is_alive() and tx._active is None and len(seen) == 1
        assert scene.get_owner_delete(task(scene).clip_id) is None
