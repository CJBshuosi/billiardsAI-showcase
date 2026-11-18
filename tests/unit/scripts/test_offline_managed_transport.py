"""The offline remote is exercised through real signed gateways and local records."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import urllib.request
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandApplication
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.sync import SyncRequestToSign, sign_sync_request
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.cloud_runtime import (
    FieldCloudFactories,
    prepare_cloud_runtime,
)
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker
from scripts.algorithm.offline_managed_transport import OfflineManagedRemote

BASE_URL = "https://offline-development.invalid/sync-api"
SCOPE = ManagedSessionScope("offline-development", "offline-device", "offline-table")


def runtime(remote):
    args = SimpleNamespace(
        sync_base_url=BASE_URL,
        device_id=SCOPE.device_id,
        credential_target="billiardsAI/development/offline-device",
        runtime_environment="development",
        cloudbase_environment_id=SCOPE.environment_id,
        expected_development_environment_id=SCOPE.environment_id,
        table_id=SCOPE.table_id,
        managed_sessions=True,
        existing_development_control_only=False,
    )
    factories = replace(
        FieldCloudFactories(),
        credentials=remote.credentials_factory,
        transport=remote.transport_factory,
    )
    return prepare_cloud_runtime(args, factories=factories)


def signed(remote):
    return SignedFieldCloudGateway(credentials=remote.credentials, transport=remote)


def direct(remote, path, payload, *, changes=None):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    headers = {
        "content-type": "application/json",
        "x-device-id": "offline-device",
        "x-timestamp": "1000",
        "x-nonce": "unit-nonce",
        "x-body-sha256": hashlib.sha256(body).hexdigest(),
    }
    headers["x-signature"] = sign_sync_request(
        SyncRequestToSign(
            "POST", path, headers["x-timestamp"], headers["x-nonce"], headers["x-body-sha256"]
        ),
        secret=remote.credentials.secret(),
    ).signature
    headers.update(changes or {})
    return remote.post_json(path, headers, body)


def opened(remote):
    gateway = runtime(remote).gateway
    request_id = str(uuid4())
    binding = gateway.open_managed_session(
        open_request_id=request_id, expected_table_id=SCOPE.table_id
    )
    return gateway, request_id, binding


def test_real_attestation_startup_one_birth_and_scope():
    remote = OfflineManagedRemote()
    runtime_state = runtime(remote)
    assert (
        runtime_state.gateway.runtime_attestation().cloudbase_environment_id == SCOPE.environment_id
    )
    request = str(uuid4())
    first = runtime_state.gateway.open_managed_session(
        open_request_id=request, expected_table_id=SCOPE.table_id
    )
    assert (
        runtime_state.gateway.open_managed_session(
            open_request_id=request, expected_table_id=SCOPE.table_id
        )
        == first
    )
    assert first.session_id == f"managed-{first.birth_id}"
    assert runtime_state.gateway.refresh_join_requests(first.session_id) == ()
    with pytest.raises(OSError):
        runtime_state.gateway.open_managed_session(
            open_request_id=str(uuid4()), expected_table_id=SCOPE.table_id
        )
    assert remote.failure is not None
    assert remote.snapshot()["birth_count"] == 1


def test_actual_worker_applies_and_stale_rejects_in_sqlite(tmp_path):
    remote = OfflineManagedRemote()
    cloud = runtime(remote)
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    coordinator = cloud.make_managed_startup()
    session = coordinator.prepare(
        store, artifact_id="offline-artifact", rule_profile_id=None, policy="blackbox_fresh"
    )
    assert store.load_session(session.session_id).match.revision == 0
    worker = _BackgroundFieldCommandWorker(
        gateway=cloud.gateway, store=store, match_lock=threading.RLock()
    )
    first = remote.issue(
        session_id=session.session_id,
        actor_user_id="offline-red",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
    )
    assert worker.claim_and_apply_once(raise_errors=True)
    actual = remote.completion(first)
    assert actual == FieldCommandApplication(first, "applied", "APPLIED", 1)
    assert store.load_session(session.session_id).match.red_user_id == "offline-red"
    stale = remote.issue(
        session_id=session.session_id,
        actor_user_id="offline-blue",
        command_type="claim_role",
        payload={"role": "blue"},
        expected_revision=0,
    )
    assert worker.claim_and_apply_once(raise_errors=True)
    assert remote.completion(stale) == FieldCommandApplication(
        stale, "rejected", "STALE_REVISION", 1
    )
    assert store.load_session(session.session_id).match.blue_user_id is None
    assert remote.failure is None


def test_claim_close_prefix_and_duplicate_completion(tmp_path):
    remote = OfflineManagedRemote()
    cloud = runtime(remote)
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    session = cloud.make_managed_startup().prepare(
        store, artifact_id="offline-artifact", rule_profile_id=None, policy="blackbox_fresh"
    )
    gateway = cloud.gateway
    binding = store.get_active_managed_open_request(SCOPE).binding
    command_id = remote.issue(
        session_id=session.session_id,
        actor_user_id="offline-red",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
    )
    claimed = gateway.claim_field_command()
    assert claimed.command_id == command_id
    assert gateway.claim_field_command() == claimed
    close = gateway.close_managed_session(binding=binding)
    assert (close.coverage, close.closed_target, close.contiguous_ack_sequence) == ("closing", 1, 0)
    with pytest.raises(ValueError):
        remote.issue(
            session_id=binding.session_id,
            actor_user_id="offline-blue",
            command_type="claim_role",
            payload={"role": "blue"},
            expected_revision=0,
        )
    assert gateway.claim_field_command() == claimed
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )
    assert worker.claim_and_apply_once(raise_errors=True)
    completion = remote.completion(command_id)
    assert completion == store.get_field_command_application(claimed)
    assert gateway.complete_field_command(completion) == completion
    with pytest.raises(OSError, match="COMPLETION_CONFLICT"):
        direct(
            remote,
            f"/v1/field-commands/{command_id}/complete",
            {
                "status": "rejected",
                "public_result_code": "STALE_REVISION",
                "resulting_revision": completion.resulting_revision,
            },
        )
    assert remote.completion(command_id) == completion
    sealed = gateway.close_managed_session(binding=binding)
    assert (sealed.coverage, sealed.closed_target, sealed.contiguous_ack_sequence) == (
        "sealed",
        1,
        1,
    )
    assert remote.snapshot()["completed_prefix"] == 1


def test_four_real_commands_and_fifth_refused(tmp_path):
    remote = OfflineManagedRemote()
    cloud = runtime(remote)
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    session = cloud.make_managed_startup().prepare(
        store, artifact_id="offline-artifact", rule_profile_id=None, policy="blackbox_fresh"
    )
    worker = _BackgroundFieldCommandWorker(
        gateway=cloud.gateway, store=store, match_lock=threading.RLock()
    )
    command_specs = [
        ("offline-red", "claim_role", {"role": "red"}),
        ("offline-blue", "claim_role", {"role": "blue"}),
        (
            "offline-red",
            "set_breaker",
            {
                "role": "red",
                "mode": "assistance",
                "format": "single",
                "race_target": None,
                "ranking_intent": False,
                "settings_revision": 1,
            },
        ),
        ("offline-red", "end_session", {}),
    ]
    for actor, kind, payload in command_specs:
        revision = store.load_session(session.session_id).match.revision
        command_id = remote.issue(
            session_id=session.session_id,
            actor_user_id=actor,
            command_type=kind,
            payload=payload,
            expected_revision=revision,
        )
        claimed = cloud.gateway.claim_field_command()
        assert worker.claim_and_apply_once(raise_errors=True)
        assert remote.completion(command_id) == store.get_field_command_application(claimed)
        assert remote.completion(command_id).outcome == "applied"
    current = store.load_session(session.session_id)
    assert current.status.value == "ended"
    assert current.match.settings_frozen
    assert (current.match.red_user_id, current.match.blue_user_id) == (
        "offline-red",
        "offline-blue",
    )
    with pytest.raises(ValueError):
        remote.issue(
            session_id=session.session_id,
            actor_user_id="offline-red",
            command_type="end_session",
            payload={},
            expected_revision=current.match.revision,
        )
    assert remote.snapshot()["issued_count"] == 4


@pytest.mark.parametrize("bad", [True, -1, 9007199254740992, 1.5, "1"])
def test_invalid_issue_revision_does_not_advance(bad):
    remote = OfflineManagedRemote()
    _, _, binding = opened(remote)
    with pytest.raises(ValueError):
        remote.issue(
            session_id=binding.session_id,
            actor_user_id="offline-red",
            command_type="claim_role",
            payload={"role": "red"},
            expected_revision=bad,
        )
    assert remote.snapshot()["issued_count"] == 0
    assert remote.failure is not None


def test_conflicting_issue_completion_and_capacity_leave_prefix():
    remote = OfflineManagedRemote()
    gateway, _, binding = opened(remote)
    first = remote.issue(
        session_id=binding.session_id,
        actor_user_id="offline-red",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
    )
    with pytest.raises(ValueError):
        remote.issue(
            session_id=binding.session_id,
            actor_user_id="offline-blue",
            command_type="claim_role",
            payload={"role": "blue"},
            expected_revision=0,
        )
    assert gateway.claim_field_command().command_id == first
    with pytest.raises(OSError):
        gateway.complete_field_command(FieldCommandApplication(first, "applied", "BAD", 1))
    with pytest.raises(OSError):
        gateway.complete_field_command(FieldCommandApplication("unknown", "applied", "APPLIED", 1))
    assert gateway.claim_field_command().command_id == first
    assert remote.snapshot()["completed_prefix"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": "foreign"},
        {"actor_user_id": "other"},
        {"actor_user_id": "offline-blue", "payload": {"role": "red"}},
        {"command_type": "release_role"},
    ],
)
def test_issue_scope_rejection_keeps_binding_and_prefix(changes):
    remote = OfflineManagedRemote()
    _, request_id, binding = opened(remote)
    values = {
        "session_id": binding.session_id,
        "actor_user_id": "offline-red",
        "command_type": "claim_role",
        "payload": {"role": "red"},
        "expected_revision": 0,
    }
    with pytest.raises(ValueError):
        remote.issue(**(values | changes))
    assert remote.snapshot()["issued_count"] == 0
    assert remote.snapshot()["completed_prefix"] == 0
    assert (
        runtime(remote).gateway.open_managed_session(
            open_request_id=request_id, expected_table_id=SCOPE.table_id
        )
        == binding
    )


def test_completion_revision_is_strict_and_claim_survives():
    remote = OfflineManagedRemote()
    gateway, _, binding = opened(remote)
    command_id = remote.issue(
        session_id=binding.session_id,
        actor_user_id="offline-red",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
    )
    assert gateway.claim_field_command().command_id == command_id
    for revision in (True, -1, 9007199254740992):
        with pytest.raises(OSError):
            direct(
                remote,
                f"/v1/field-commands/{command_id}/complete",
                {
                    "status": "applied",
                    "public_result_code": "APPLIED",
                    "resulting_revision": revision,
                },
            )
    assert gateway.claim_field_command().command_id == command_id
    assert remote.snapshot()["completed_prefix"] == 0


def test_episode_command_order_is_bounded():
    remote = OfflineManagedRemote()
    _, _, binding = opened(remote)
    with pytest.raises(ValueError, match="ISSUE_ORDER"):
        remote.issue(
            session_id=binding.session_id,
            actor_user_id="offline-red",
            command_type="end_session",
            payload={},
            expected_revision=0,
        )
    assert remote.snapshot()["issued_count"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"x-device-id": "other"},
        {"x-signature": "0" * 64},
        {"x-body-sha256": "0" * 64},
        {"x-timestamp": "bad"},
    ],
)
def test_signed_request_rejects_header_mismatch(change):
    remote = OfflineManagedRemote()
    with pytest.raises(OSError):
        direct(remote, "/v1/runtime-attestation", {}, changes=change)
    assert remote.failure is not None
    assert remote.snapshot()["birth_count"] == 0


def test_route_body_and_deny_transport_fail_closed():
    remote = OfflineManagedRemote()
    for path, body in [
        ("/v1/runtime-attestation", {"unexpected": 1}),
        ("/v1/field-sessions/x/open", {}),
    ]:
        with pytest.raises(OSError):
            direct(remote, path, body)
    with pytest.raises(OSError):
        remote.deny_transport_factory(BASE_URL).post_json("/anything", {}, b"{}")
    with pytest.raises(OSError):
        remote.put_json("/v1/runtime-attestation", {}, b"{}")
    assert remote.snapshot()["birth_count"] == 0
    assert remote.failure is not None


def test_body_and_log_limits_are_bounded():
    remote = OfflineManagedRemote()
    with pytest.raises(OSError):
        direct(remote, "/v1/runtime-attestation", {"padding": "x" * 65_536})
    assert remote.snapshot()["call_count"] == 1
    assert remote.snapshot()["events"] == ("BODY_LIMIT",)


def test_protocol_call_limit_stops_without_unbounded_audit():
    remote = OfflineManagedRemote()
    for _ in range(16_384):
        assert direct(remote, "/v1/runtime-attestation", {}).status_code == 200
    with pytest.raises(OSError, match="CALL_LIMIT"):
        direct(remote, "/v1/runtime-attestation", {})
    observed = remote.snapshot()
    assert observed["call_count"] == 16_384
    assert len(observed["events"]) == 256


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (
            "credentials_factory",
            {"device_id": "wrong", "target_name": "billiardsAI/development/offline-device"},
        ),
        ("credentials_factory", {"device_id": "offline-device", "target_name": "wrong"}),
        ("transport_factory", {"base_url": "https://other.invalid/sync-api"}),
        ("transport_factory", {"base_url": BASE_URL, "allow_redirects": True}),
        ("transport_factory", {"base_url": BASE_URL, "timeout_seconds": 0}),
        ("transport_factory", {"base_url": BASE_URL, "timeout_seconds": True}),
        ("transport_factory", {"base_url": BASE_URL, "timeout_seconds": float("nan")}),
        ("transport_factory", {"base_url": BASE_URL, "timeout_seconds": float("inf")}),
    ],
)
def test_factories_reject_wrong_scope_without_network(monkeypatch, factory, kwargs):
    def forbidden_network(*_args, **_kwargs):
        pytest.fail("network attempted by offline factory")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    remote = OfflineManagedRemote()
    with pytest.raises(ValueError):
        getattr(remote, factory)(**kwargs)
    observed = remote.snapshot()
    assert observed["failure"] in {"CREDENTIAL_SCOPE", "TRANSPORT_SCOPE"}
    assert observed["call_count"] == 0
    assert observed["birth_count"] == 0
    assert observed["issued_count"] == 0


def test_worker_swallowed_transport_error_stays_sticky_in_remote(tmp_path):
    remote = OfflineManagedRemote()
    cloud = runtime(remote)
    store = SQLiteFieldRecordStore(tmp_path / "field.sqlite3")
    session = cloud.make_managed_startup().prepare(
        store, artifact_id="offline-artifact", rule_profile_id=None, policy="blackbox_fresh"
    )
    command_id = remote.issue(
        session_id=session.session_id,
        actor_user_id="offline-red",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
    )
    gateway = DevelopmentAttestedFieldCloudGateway(
        delegate=SignedFieldCloudGateway(
            credentials=remote.credentials,
            transport=remote.deny_transport_factory(BASE_URL),
        ),
        expected_cloudbase_environment_id=SCOPE.environment_id,
    )
    health = []
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway,
        store=store,
        match_lock=threading.RLock(),
        health_changed=health.append,
    )
    assert worker.claim_and_apply_once() is False  # The worker's default catches OSError.
    assert health == ["offline"]
    assert remote.snapshot()["failure"] == "DENIED_TRANSPORT"
    assert remote.failure == "DENIED_TRANSPORT"
    assert remote.snapshot()["completed_prefix"] == 0
    assert remote.completion(command_id) is None
    assert store.load_session(session.session_id).match.revision == 0
