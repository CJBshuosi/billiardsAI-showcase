"""Managed close uses persistent identity and drains the normal local ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, replace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.control import (
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionCloseState,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.cloud_gateway import FieldJoinCoordinator
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker, _ServerState

BIRTH = "22222222-2222-4222-8222-222222222222"
REQUEST = "11111111-1111-4111-8111-111111111111"
SCOPE = ManagedSessionScope("dev", "device-1", "table-1")
BINDING = ManagedSessionBinding(
    "1.0", f"managed-{BIRTH}", BIRTH, "managed_session_birth_v1", "table-1", "device-1", "open"
)
SESSION = BINDING.session_id


def command(sequence=1, **changes):
    values = dict(
        command_id=f"command-{sequence}",
        session_id=SESSION,
        actor_user_id="alice",
        command_type="claim_role",
        payload={"role": "red"},
        expected_revision=0,
        server_sequence=sequence,
    )
    return FieldCommandEnvelope(**(values | changes))


def materialized(path):
    store = SQLiteFieldRecordStore(
        path,
        managed_open_request_id_factory=lambda: REQUEST,
        session_id_factory=lambda: pytest.fail("random successor"),
    )
    store.reserve_managed_open(SCOPE, artifact_id="artifact-1")
    store.record_managed_binding(SCOPE, REQUEST, BINDING)
    store.materialize_managed_session(SCOPE, REQUEST)
    return store


def close_state(coverage="closing", ack=0, target=1):
    return ManagedSessionCloseState(
        **{k: v for k, v in asdict(BINDING).items() if k != "status"},
        coverage=coverage,
        closed_target=target,
        contiguous_ack_sequence=ack,
    )


class Gateway:
    def __init__(self, result=None):
        self.result = result
        self.closes = []
        self.acks = []
        self.results = []
        self.next_command = command()

    def close_managed_session(self, *, binding):
        self.closes.append(binding)
        return self.result or close_state()

    def open_field_session(self, session_id):
        pytest.fail("managed session cannot use legacy open")

    def close_field_session(self, session_id):
        pytest.fail("managed session cannot use legacy close")

    def claim_field_command(self):
        return self.next_command

    def complete_field_command(self, application):
        self.acks.append(application)
        return application

    def propose_match_result(self, **payload):
        self.results.append(payload)

    def refresh_join_requests(self, session_id):
        return ()


def state_for(store, gateway, tmp_path, scope=SCOPE):
    session = store.load_session(SESSION)
    return _ServerState(
        artifact_path=tmp_path,
        control_token="token",
        replay_clock=object(),
        match=session.match,
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id=SESSION,
        artifact_id="artifact-1",
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        projection_source=object(),
        join_coordinator=FieldJoinCoordinator(session_id=SESSION, gateway=gateway),
        managed_session_scope=scope,
        monotonic=lambda: 100,
        pending_cloud_closes={sid: 0 for sid in store.list_pending_cloud_close_session_ids()},
    )


@pytest.mark.parametrize("kind", ["end_session", "claim_role"])
def test_missing_managed_local_session_is_recoverable_without_ack_or_legacy_close(tmp_path, kind):
    store = SQLiteFieldRecordStore(tmp_path / "field.db")
    gateway = Gateway()
    gateway.next_command = command(
        command_type=kind, payload={} if kind == "end_session" else {"role": "red"}
    )
    health = []
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock(), health_changed=health.append
    )
    assert worker.claim_and_apply_once() is False
    assert worker.claim_and_apply_once() is False
    assert gateway.acks == [] and health == ["offline", "offline"]


@pytest.mark.parametrize("collision", [False, True])
def test_managed_transient_ledger_conflicts_are_never_acked(tmp_path, collision):
    store = materialized(tmp_path / "field.db")
    gateway = Gateway()
    if collision:
        store.apply_field_command(command(), reduce_field_command)
        gateway.next_command = command(payload={"role": "blue"})
    else:
        gateway.next_command = command(2)
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )
    assert worker.claim_and_apply_once() is False
    assert gateway.acks == []


def test_close_fences_then_drains_and_restart_replays_lost_ack_before_ended_rejection(tmp_path):
    store = materialized(tmp_path / "field.db")
    # The server has not received this local applied outcome when the match ends.
    applied = store.apply_field_command(command(), reduce_field_command)
    store.end_session(SESSION)
    gateway = Gateway(close_state("closing", 0, 2))
    state = state_for(store, gateway, tmp_path)
    state._maintain_join(ended=True)
    assert gateway.closes == [BINDING]
    assert store.list_pending_cloud_close_session_ids() == (SESSION,)
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=state.match_lock
    )
    assert worker.claim_and_apply_once() is True
    assert gateway.acks == [applied]
    # A restart recovers the same persisted close intent/binding, not a new UUID.
    restarted = SQLiteFieldRecordStore(store.path)
    state = state_for(restarted, gateway, tmp_path)
    state._maintain_join(ended=True)
    gateway.next_command = command(
        2, command_type="release_role", payload={"role": "red"}, expected_revision=1
    )
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=restarted, match_lock=state.match_lock
    )
    assert worker.claim_and_apply_once() is True
    assert gateway.acks[-1].result_code == "SESSION_ENDED"
    assert gateway.acks[-1].resulting_revision == 1
    gateway.result = close_state("sealed", 2, 2)
    state.pending_cloud_closes[SESSION] = 0
    state._maintain_join(ended=True)
    assert store.list_pending_cloud_close_session_ids() == ()
    assert not state.pending_cloud_closes
    assert gateway.closes == [BINDING] * 3
    assert restarted.load_session(SESSION).match.red_user_id == "alice"


@pytest.mark.parametrize("terminal", [False, True])
def test_ranking_business_gate_cannot_block_initial_fence_or_drain(tmp_path, terminal):
    store = materialized(tmp_path / "field.db")
    match = MatchControlSnapshot(
        red_user_id="alice",
        blue_user_id="bob",
        revision=1,
        mode=MatchMode.OBSERVATION,
        match_format=MatchFormat.SINGLE,
        ranking_intent=True,
        settings_revision=1,
        terminal_reason="legal_eight" if terminal else None,
        rack_phase=RackPhase.ENDED if terminal else RackPhase.OPEN,
        winner_role=WristbandRole.RED if terminal else None,
    )
    store.save_match(SESSION, match, expected_revision=0)
    store.end_session(SESSION)
    gateway = Gateway()
    state = state_for(store, gateway, tmp_path)
    state._maintain_join(ended=True)
    assert gateway.closes == [BINDING] and not gateway.results
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=state.match_lock
    )
    assert worker.claim_and_apply_once() is True
    assert gateway.acks[-1].result_code == "SESSION_ENDED"
    gateway.result = close_state("sealed", 1)
    state.pending_cloud_closes[SESSION] = 0
    state._maintain_join(ended=True)
    assert len(gateway.closes) == 2
    assert store.list_pending_cloud_close_session_ids() == (SESSION,)
    if terminal:
        store.is_session_sync_complete = lambda sid: True
        state.pending_cloud_closes[SESSION] = 0
        state._maintain_join(ended=True)
        assert len(gateway.results) == 1
        assert store.list_pending_cloud_close_session_ids() == ()


@pytest.mark.parametrize(
    "missing", ["scope", "binding", "local", "artifact", "profile", "wrong_scope"]
)
def test_close_missing_recovery_authority_retains_intent_without_network(tmp_path, missing):
    store = materialized(tmp_path / "field.db")
    store.end_session(SESSION)
    gateway = Gateway(close_state("sealed", 1))
    state = state_for(store, gateway, tmp_path, scope=None if missing == "scope" else SCOPE)
    if missing == "wrong_scope":
        state.managed_session_scope = replace(SCOPE, environment_id="other")
    if missing == "artifact":
        with sqlite3.connect(store.path) as db:
            db.execute(
                "UPDATE field_sessions SET artifact_id='other' WHERE session_id=?", (SESSION,)
            )
    if missing == "profile":
        with sqlite3.connect(store.path) as db:
            raw = db.execute("SELECT payload_json FROM field_match_state").fetchone()[0]
            payload = json.loads(raw)
            payload["rule_profile_id"] = "cbsa-chinese-pool-2017-observable-v1"
            db.execute("UPDATE field_match_state SET payload_json=?", (json.dumps(payload),))
    with sqlite3.connect(store.path) as db:
        if missing == "binding":
            db.execute("DELETE FROM field_managed_open_requests")
        if missing == "local":
            # Keep current legacy local state while the missing managed intent retries.
            state.session_id = "local-current"
            db.execute(
                "UPDATE field_sessions SET session_id='local-current' WHERE session_id=?",
                (SESSION,),
            )
            db.execute(
                "UPDATE field_match_state SET session_id='local-current' WHERE session_id=?",
                (SESSION,),
            )
    state._maintain_join(ended=True)
    assert not gateway.closes
    assert store.list_pending_cloud_close_session_ids() == (SESSION,)
    assert state.cloud_close_error


def test_close_network_releases_match_lock_and_closing_does_not_poll_until_due(tmp_path):
    store = materialized(tmp_path / "field.db")
    store.end_session(SESSION)
    gateway = Gateway()
    state = state_for(store, gateway, tmp_path)
    close = gateway.close_managed_session

    def checked_close(*, binding):
        acquired = []

        def acquire():
            with state.match_lock:
                acquired.append(True)

        thread = threading.Thread(target=acquire)
        thread.start()
        thread.join(timeout=1)
        assert acquired == [True], "close held match_lock while waiting for network"
        return close(binding=binding)

    gateway.close_managed_session = checked_close
    state._maintain_join(ended=True)
    state._maintain_join(ended=True)
    assert gateway.closes == [BINDING]
    assert not state.cloud_io_lock.locked()


def test_current_managed_open_does_not_legacy_open_and_ended_never_random_rollover(tmp_path):
    store = materialized(tmp_path / "field.db")
    gateway = Gateway(close_state("sealed", 0, 0))
    state = state_for(store, gateway, tmp_path)
    state._maintain_join(ended=False)
    store.end_session(SESSION)
    state.pending_cloud_closes[SESSION] = 0
    state._maintain_join(ended=True)
    state.session_ended_at_monotonic = 0
    store.enqueue_sync_job(session_id=SESSION, manifest_hash="a" * 64, manifest_json="{}")
    session, changed = state._rollover_ended_session(store.load_session(SESSION))
    assert not changed and session.session_id == state.session_id == SESSION
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT session_id FROM field_sessions").fetchall() == [(SESSION,)]
        assert db.execute("SELECT status FROM field_cloud_close_intents").fetchall() == [
            ("complete",)
        ]


def test_managed_local_end_after_start_reloads_durable_intent_without_legacy_open(tmp_path):
    store = materialized(tmp_path / "field.db")
    gateway = Gateway(close_state("sealed", 0, 0))
    state = state_for(store, gateway, tmp_path)
    assert state.pending_cloud_closes == {}
    store.end_session(SESSION)
    state._maintain_join(ended=True)
    assert gateway.closes == [BINDING]
    assert store.list_pending_cloud_close_session_ids() == ()


def test_worker_loop_fences_and_drains_even_when_evidence_preparation_fails(tmp_path, monkeypatch):
    store = materialized(tmp_path / "field.db")
    store.end_session(SESSION)
    gateway = Gateway()
    state = state_for(store, gateway, tmp_path)
    order = []
    completed = threading.Event()
    original_close = gateway.close_managed_session
    original_complete = gateway.complete_field_command

    def close(*, binding):
        order.append("close")
        return original_close(binding=binding)

    def evidence(self):
        order.append("evidence")
        raise ValueError("synthetic evidence pending")

    def complete(application):
        order.append("ack")
        gateway.next_command = None
        result = original_complete(application)
        completed.set()
        return result

    gateway.close_managed_session = close
    gateway.complete_field_command = complete
    monkeypatch.setattr(_ServerState, "_retry_unsynced_sessions", evidence)
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway,
        store=store,
        match_lock=state.match_lock,
        maintenance=state.maintain_background,
    )
    worker.start()
    try:
        assert completed.wait(timeout=2)
    finally:
        worker.stop()
    assert order[:3] == ["close", "evidence", "ack"]
    assert gateway.acks[0].result_code == "SESSION_ENDED"
    assert store.list_pending_cloud_close_session_ids() == (SESSION,)


def test_lost_ack_transport_response_replays_original_application_after_local_end(tmp_path):
    store = materialized(tmp_path / "field.db")
    gateway = Gateway()
    completed = gateway.complete_field_command

    def lose_response(application):
        completed(application)
        raise OSError("synthetic lost response")

    gateway.complete_field_command = lose_response
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=store, match_lock=threading.RLock()
    )
    assert worker.claim_and_apply_once() is False
    assert gateway.acks[0].result_code == "APPLIED"
    store.end_session(SESSION)
    restarted = SQLiteFieldRecordStore(store.path)
    gateway.complete_field_command = completed
    state = state_for(restarted, gateway, tmp_path)
    state._maintain_join(ended=True)
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway, store=restarted, match_lock=state.match_lock
    )
    assert worker.claim_and_apply_once() is True
    assert gateway.acks[0] == gateway.acks[1]
    assert restarted.load_session(SESSION).match.revision == 1
    assert restarted.list_pending_cloud_close_session_ids() == (SESSION,)


@pytest.mark.parametrize("retired", [False, True])
def test_historical_managed_close_uses_original_resources_after_current_scene_changes(
    tmp_path, retired
):
    store = materialized(tmp_path / "field.db")
    store.end_session(SESSION)
    gateway = Gateway(close_state("sealed", 0, 0))
    state = state_for(store, gateway, tmp_path)
    state.artifact_id = "new-current-artifact"
    state.match = replace(state.match, rule_profile_id="cbsa-chinese-pool-2017-observable-v1")
    expected_binding = BINDING
    if retired:
        previous = store.retire_managed_open(
            SCOPE, REQUEST, sealed_close=close_state("sealed", 0, 0)
        )
        expected_binding = previous.binding
        new_request = "33333333-3333-4333-8333-333333333333"
        new_birth = "44444444-4444-4444-8444-444444444444"
        store._managed_open_request_id_factory = lambda: new_request
        store.reserve_managed_open(
            SCOPE, artifact_id=state.artifact_id, rule_profile_id=state.match.rule_profile_id
        )
        store.record_managed_binding(
            SCOPE,
            new_request,
            replace(BINDING, session_id=f"managed-{new_birth}", birth_id=new_birth),
        )
        current = store.materialize_managed_session(SCOPE, new_request)
        state.session_id, state.match = current.session_id, current.match
        state.join_coordinator = FieldJoinCoordinator(
            session_id=current.session_id, gateway=gateway
        )
        state.cloud_session_status = "open"
    state._maintain_join(ended=not retired)
    assert gateway.closes == [expected_binding]
    assert store.list_pending_cloud_close_session_ids() == ()
