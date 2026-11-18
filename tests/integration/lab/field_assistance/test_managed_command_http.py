"""Synthetic Python/Node close and drain acceptance, without cloud/provider access."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.control import (
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
from billiards_ai.edge.route_assistance.sync import UrllibSyncHttpTransport
from billiards_ai.lab.field_assistance.cloud_gateway import (
    FieldJoinCoordinator,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker, _ServerState
from tests.unit.edge.route_assistance.test_runtime_resources import payload as resource_payload

ROOT = Path(__file__).resolve().parents[4]
REQUEST = "11111111-1111-4111-8111-111111111111"
SCOPE = ManagedSessionScope("synthetic", "device-1", "table-1")


class Credentials:
    def device_id(self):
        return "device-1"

    def secret(self):
        return b"synthetic-only-secret"


@pytest.fixture
def bridge(request):
    # The helper intercepts SDK loading before sync-api's default init; no env credentials.
    process = subprocess.Popen(
        [
            "node",
            str(ROOT / "apps/miniprogram/tests/helpers/managed-command-http-bridge.js"),
            *(
                {
                    "writers": ["--registered-writers"],
                    "owner-erasure": ["--owner-erasure"],
                    "owner-device-delivery": ["--owner-device-delivery"],
                }.get(getattr(request, "param", None), [])
            ),
        ],
        cwd=ROOT,
        env={key: os.environ[key] for key in ("PATH", "NODE_PATH") if key in os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as readiness:
            readiness.register(process.stdout, selectors.EVENT_READ)
            assert readiness.select(5), "Node bridge startup timed out"
        port = process.stdout.readline(64).strip()
        assert port.isdigit(), process.stderr.read() if process.poll() is not None else port
        base = f"http://127.0.0.1:{port}"

        def control(op, **values):
            request = urllib.request.Request(
                f"{base}/__test",
                data=json.dumps({"op": op, **values}).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)

        yield base, control
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def gateway(base):
    return SignedFieldCloudGateway(
        credentials=Credentials(),
        transport=UrllibSyncHttpTransport(base, timeout_seconds=5),
    )


def synthetic_resources(root):
    """Metadata-only original model/config resources; no model or capture I/O."""
    raw = json.loads(
        json.dumps(resource_payload(), ensure_ascii=False).replace("/original/台球", str(root))
    )
    raw["paths"]["artifact_root"] = str(root)
    raw["artifact_id"] = "usb-field-" + hashlib.sha256(str(root).encode()).hexdigest()
    raw["rule_profile_id"] = None  # retain the original observation fixture's rule facts
    return ManagedRuntimeResourceSpec.from_payload(raw)


def materialize(tmp_path, cloud, *, resource_spec=None):
    store = SQLiteFieldRecordStore(
        tmp_path / "field.db",
        managed_open_request_id_factory=lambda: REQUEST,
        session_id_factory=lambda: pytest.fail("managed session cannot allocate random successor"),
    )
    store.reserve_managed_open(
        SCOPE,
        artifact_id="artifact-1" if resource_spec is None else resource_spec.artifact_id,
        rule_profile_id=None if resource_spec is None else resource_spec.rule_profile_id,
        resource_spec=resource_spec,
    )
    binding = cloud.open_managed_session(open_request_id=REQUEST, expected_table_id="table-1")
    store.record_managed_binding(SCOPE, REQUEST, binding)
    store.materialize_managed_session(SCOPE, REQUEST)
    return store, binding


def state_for(store, cloud, session_id, tmp_path):
    return _ServerState(
        artifact_path=tmp_path,
        control_token="synthetic",
        replay_clock=object(),
        match=store.load_session(session_id).match,
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id=session_id,
        artifact_id="artifact-1",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        projection_source=object(),
        join_coordinator=FieldJoinCoordinator(session_id=session_id, gateway=cloud),
        managed_session_scope=SCOPE,
        monotonic=lambda: 100,
    )


@pytest.mark.parametrize("loss", ["request-loss", "response-loss"])
def test_real_signed_worker_recovers_applied_ack_after_end_and_drains_frozen_prefix(
    bridge,
    tmp_path,
    loss,
):
    base, control = bridge
    cloud = gateway(base)
    store, binding = materialize(tmp_path, cloud)
    first = control("issue")
    second = control("issue")
    third = control("issue")
    control("fault", kind=loss)
    worker = _BackgroundFieldCommandWorker(
        gateway=cloud,
        store=store,
        match_lock=threading.RLock(),
    )
    assert worker.claim_and_apply_once() is False
    assert store.load_session(binding.session_id).match.revision == 1
    # The local applied claim survives end and a fresh SQLite store/worker instance.
    store.end_session(binding.session_id)
    restarted = SQLiteFieldRecordStore(store.path)
    state = state_for(restarted, cloud, binding.session_id, tmp_path)
    state._maintain_join(ended=True)
    snapshot = control("snapshot")
    head = snapshot["rows"]["session_command_authority"][binding.session_id]
    assert head["coverage"] == "closing" and head["closed_target"] == 3
    issued = snapshot["rows"]["field_session_commands"][first["command_id"]]
    original = restarted.apply_field_command(
        FieldCommandEnvelope(
            command_id=first["command_id"],
            session_id=binding.session_id,
            actor_user_id="alice",
            command_type=first["command_type"],
            payload=issued["validated_payload"],
            expected_revision=issued["expected_revision"],
            server_sequence=issued["server_sequence"],
        ),
        reduce_field_command,
    )
    assert original.result_code == "APPLIED" and original.resulting_revision == 1
    worker = _BackgroundFieldCommandWorker(
        gateway=cloud,
        store=restarted,
        match_lock=state.match_lock,
    )
    if loss == "request-loss":
        assert worker.claim_and_apply_once() is True
    else:
        # Exact completion replay uses the durable local outcome, not today's ended state.
        cloud.complete_field_command(original)
    assert worker.claim_and_apply_once() is True
    assert worker.claim_and_apply_once() is True
    snapshot = control("snapshot")
    rows = snapshot["rows"]
    assert rows["field_session_commands"][first["command_id"]]["public_result_code"] == "APPLIED"
    assert (
        rows["field_session_commands"][second["command_id"]]["public_result_code"]
        == "SESSION_ENDED"
    )
    assert (
        rows["field_session_commands"][third["command_id"]]["public_result_code"] == "SESSION_ENDED"
    )
    assert rows["session_command_authority"][binding.session_id]["coverage"] == "sealed"
    assert rows["session_command_authority"][binding.session_id]["contiguous_ack_sequence"] == 3
    assert len(rows["session_role_binding_history"]) == 1
    assert restarted.load_session(binding.session_id).match.revision == 1
    state.pending_cloud_closes[binding.session_id] = 0
    state._maintain_join(ended=True)
    assert restarted.list_pending_cloud_close_session_ids() == ()
    assert state.session_id == binding.session_id
    assert all(attempt["count"] <= 100 for attempt in snapshot["attempts"])
    assert any(
        request["path"].endswith("/close") and request["attempts"] == [2, 6]
        for request in snapshot["requests"]
    )


def test_actual_default_main_refuses_managed_activation(bridge):
    base, control = bridge
    with pytest.raises(OSError, match="SYNC_CONFIGURATION_REQUIRED"):
        gateway(base + "/default").open_managed_session(
            open_request_id=REQUEST,
            expected_table_id="table-1",
        )
    assert not control("snapshot")["rows"].get("managed_session_births")


def test_real_auth_rejects_nonce_replay_body_tampering_and_wrong_signature(bridge):
    base, control = bridge

    class RecordingTransport(UrllibSyncHttpTransport):
        def post_json(self, path, headers, body):
            self.request = path, dict(headers), body
            return super().post_json(path, headers, body)

    transport = RecordingTransport(base)
    cloud = SignedFieldCloudGateway(credentials=Credentials(), transport=transport)
    cloud.open_managed_session(open_request_id=REQUEST, expected_table_id="table-1")
    path, headers, body = transport.request
    for changed_headers, changed_body, code in [
        (headers, body, "SYNC_REPLAYED"),
        (headers, body + b" ", "SYNC_UNAUTHORIZED"),
        (headers | {"x-signature": "0" * 64}, body, "SYNC_UNAUTHORIZED"),
    ]:
        request = urllib.request.Request(
            base + path,
            data=changed_body,
            headers=changed_headers,
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 401
        assert json.load(error.value)["error"]["code"] == code
    assert len(control("snapshot")["rows"]["sync_nonces"]) == 1


def test_missing_issued_command_and_malformed_close_never_clear_durable_intent(bridge, tmp_path):
    base, control = bridge
    cloud = gateway(base)
    store, binding = materialize(tmp_path, cloud)
    first = control("issue")
    control("issue")
    control("remove-command", id=first["command_id"])
    store.end_session(binding.session_id)
    state = state_for(store, cloud, binding.session_id, tmp_path)
    control("fault", kind="malformed-close")
    state._maintain_join(ended=True)
    assert state.cloud_close_error
    assert store.list_pending_cloud_close_session_ids() == (binding.session_id,)
    worker = _BackgroundFieldCommandWorker(gateway=cloud, store=store, match_lock=state.match_lock)
    assert worker.claim_and_apply_once() is False
    state.pending_cloud_closes[binding.session_id] = 0
    state._maintain_join(ended=True)
    head = control("snapshot")["rows"]["session_command_authority"][binding.session_id]
    assert head["coverage"] == "closing" and head["contiguous_ack_sequence"] == 0
    assert head["closed_target"] == 2
    assert store.list_pending_cloud_close_session_ids() == (binding.session_id,)


def test_malformed_sealed_response_requires_valid_retry_before_intent_can_clear(bridge, tmp_path):
    base, control = bridge
    cloud = gateway(base)
    store, binding = materialize(tmp_path, cloud)
    store.end_session(binding.session_id)
    state = state_for(store, cloud, binding.session_id, tmp_path)
    control("fault", kind="malformed-close")
    state._maintain_join(ended=True)
    assert (
        control("snapshot")["rows"]["session_command_authority"][binding.session_id]["coverage"]
        == "sealed"
    )
    assert store.list_pending_cloud_close_session_ids() == (binding.session_id,)
    assert state.cloud_close_error
    state.pending_cloud_closes[binding.session_id] = 0
    state._maintain_join(ended=True)
    assert store.list_pending_cloud_close_session_ids() == ()


@pytest.mark.parametrize("terminal", [False, True])
def test_ranked_evidence_pending_does_not_deadlock_initial_close_or_drain(
    bridge, tmp_path, terminal
):
    base, control = bridge
    cloud = gateway(base)
    store, binding = materialize(tmp_path, cloud)
    issued = control("issue")
    match = replace(
        store.load_session(binding.session_id).match,
        revision=1,
        red_user_id="alice",
        blue_user_id="bob",
        mode=MatchMode.OBSERVATION,
        match_format=MatchFormat.SINGLE,
        ranking_intent=True,
        settings_revision=1,
        terminal_reason="legal_eight" if terminal else None,
        rack_phase=RackPhase.ENDED if terminal else RackPhase.OPEN,
        winner_role=WristbandRole.RED if terminal else None,
    )
    store.save_match(binding.session_id, match, expected_revision=0)
    store.end_session(binding.session_id)
    state = state_for(store, cloud, binding.session_id, tmp_path)
    state._maintain_join(ended=True)
    worker = _BackgroundFieldCommandWorker(gateway=cloud, store=store, match_lock=state.match_lock)
    assert worker.claim_and_apply_once() is True
    state.pending_cloud_closes[binding.session_id] = 0
    state._maintain_join(ended=True)
    rows = control("snapshot")["rows"]
    assert rows["session_command_authority"][binding.session_id]["coverage"] == "sealed"
    assert (
        rows["field_session_commands"][issued["command_id"]]["public_result_code"]
        == "SESSION_ENDED"
    )
    assert store.list_pending_cloud_close_session_ids() == (binding.session_id,)
    assert state.cloud_close_error
