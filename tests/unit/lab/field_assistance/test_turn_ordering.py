"""Synthetic authoritative boundaries against the real clock/store/command worker."""

import threading
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
from billiards_ai.edge.route_assistance.recording import (
    ShotBoundary,
    ShotBoundaryKind,
    ShotTimelineIndex,
)
from billiards_ai.lab.field_assistance.live_state import (
    ReplayClock,
    ReplayTimeline,
    ReplayTimelineFrame,
)
from billiards_ai.lab.field_assistance.server import _BackgroundFieldCommandWorker, _ServerState
from tests.unit.edge.route_assistance.test_turn_ownership import make_store, match
from tests.unit.lab.field_assistance.test_live_state import FakeClock


def state_for(tmp_path, *, store=None, starts=(100, 300)):
    now = FakeClock()
    store = make_store(tmp_path) if store is None else store
    state = _ServerState(
        artifact_path=tmp_path,
        control_token="local",
        replay_clock=ReplayClock(
            ReplayTimeline((ReplayTimelineFrame(0, 0, {}),), 1000), monotonic=now
        ),
        match=store.load_session("session-1").match,
        match_lock=threading.RLock(),
        frame_store=None,
        record_store=store,
        session_id="session-1",
        artifact_id="replay-1",
        shot_index=ShotTimelineIndex(
            tuple(
                boundary
                for sequence, start in enumerate(starts, 1)
                for boundary in [
                    ShotBoundary(ShotBoundaryKind.STARTED, sequence, start, None),
                    ShotBoundary(ShotBoundaryKind.SETTLED, sequence, start + 50, None),
                ]
            )
        ),
        recommendation_power={},
        duration_ms=1000,
        monotonic=now,
        source_mainline1_artifact_id="a" * 64,
        verified_start_events=tuple(
            (i, f"shot-{i:04d}-event-shot-start") for i in range(1, len(starts) + 1)
        ),
    )
    if state.match.revision == 0:
        store.save_match("session-1", match(), expected_revision=0)
        state.match = match()
    return state, now


class LocalClaim:
    def __init__(self, envelope):
        self.envelope = envelope
        self.application = None

    def claim_field_command(self):
        return self.envelope

    def complete_field_command(self, application):
        self.application = application


def run_command(state, **changes):
    envelope = replace(
        FieldCommandEnvelope("cmd-1", "session-1", "user-red", "switch_turn", {}, 1, 1), **changes
    )
    gateway = LocalClaim(envelope)
    worker = _BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
    )
    assert worker.claim_and_apply_once()
    return gateway.application


@pytest.mark.parametrize(
    "at,owner,outcome",
    [
        (99, "user-blue", "applied"),
        (100, None, "rejected"),
        (101, "user-red", "rejected"),
        (151, "user-red", "applied"),
    ],
)
def test_start_before_after_same_clock_tick_as_switch(tmp_path, at, owner, outcome):
    state, now = state_for(tmp_path)
    state.replay_clock.start()
    now.advance(at / 1000)
    result = run_command(state)
    assert result.outcome == outcome
    now.advance((152 - at) / 1000)
    state.before_field_command(
        replace(FieldCommandEnvelope("next", "session-1", "user-red", "switch_turn", {}, 1, 2))
    )
    receipt = state.record_store.get_shot_ownership("session-1", "shot-0001")
    assert receipt.actor_user_id == owner
    assert state.record_store.get_turn_confirmation("session-1", "shot-0001").action == "abstain"


def test_multiple_arrived_starts_are_conservative_and_future_not_consumed(tmp_path):
    state, now = state_for(tmp_path, starts=(100, 300, 600))
    state.replay_clock.start()
    now.advance(0.351)
    run_command(state)
    assert len(state.record_store.list_shots("session-1")) == 2
    for shot in state.record_store.list_shots("session-1"):
        assert (
            state.record_store.get_shot_ownership("session-1", shot.shot_id).evidence_status
            == "unknown"
        )
    assert state.record_store.get_turn_confirmation("session-1", "shot-0001").status == "stale"


def test_pause_resume_duplicate_processing_preserves_frozen_receipt(tmp_path):
    state, now = state_for(tmp_path)
    state.replay_clock.start()
    now.advance(0.101)
    state.mutate({"command": "pause"})
    before = state.record_store.get_shot_ownership("session-1", "shot-0001")
    now.advance(5)
    assert run_command(state).result_code == "SHOT_UNSETTLED"
    state.mutate({"command": "start"})
    now.advance(0.05)
    run_command(state, command_id="cmd-2", server_sequence=2)
    state.mutate({"command": "reset"})
    state.mutate({"command": "start"})
    now.advance(0.151)
    state.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-red", "continue_turn", {}, 2, 3)
    )
    assert state.record_store.get_shot_ownership("session-1", "shot-0001") == before


def test_restart_cannot_bind_unprocessed_old_start_to_current_user(tmp_path):
    original, now = state_for(tmp_path)
    original.replay_clock.start()
    now.advance(0.099)
    assert run_command(original).outcome == "applied"
    recovered, restarted_clock = state_for(tmp_path, store=original.record_store)
    recovered.replay_clock.start()
    restarted_clock.advance(0.151)
    recovered.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-blue", "continue_turn", {}, 2, 2)
    )
    receipt = recovered.record_store.get_shot_ownership("session-1", "shot-0001")
    assert receipt.actor_user_id is None and receipt.evidence_status == "unknown"


def test_unstarted_clock_does_not_block_breaker_with_zero_timestamp_shot(tmp_path):
    state, now = state_for(tmp_path, starts=(0,))
    result = run_command(state, command_type="set_breaker", payload={"role": "red"})
    assert result.outcome == "applied"
    assert state.record_store.list_shots("session-1") == ()
    now.advance(0.01)
    state.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-red", "continue_turn", {}, 2, 2)
    )
    assert (
        state.record_store.get_shot_ownership("session-1", "shot-0001").evidence_status == "unknown"
    )


def test_legacy_join_and_admin_paths_flush_arrived_start_before_mutating(tmp_path):
    state, now = state_for(tmp_path)
    state.replay_clock.start()
    now.advance(0.101)
    from billiards_ai.edge.route_assistance.control import WristbandRole

    with pytest.raises(ValueError):
        state._save_participant(WristbandRole.RED, "late-user")
    assert (
        state.record_store.get_shot_ownership("session-1", "shot-0001").actor_user_id == "user-red"
    )
    with pytest.raises(ValueError):
        state.mutate(
            {
                "command": "emergency_set_breaker",
                "role": "blue",
                "reason": "manual emergency repair",
            }
        )
    assert state.match.active_role is WristbandRole.RED


def test_rollover_and_late_old_command_preserve_independent_receipts(tmp_path):
    state, now = state_for(tmp_path)
    state.replay_clock.start()
    now.advance(0.151)
    state.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-red", "continue_turn", {}, 1, 1)
    )
    original = state.record_store.get_shot_ownership("session-1", "shot-0001")
    ended = state.record_store.end_session("session-1")
    state.record_store.enqueue_sync_job(
        session_id="session-1",
        manifest_hash="a" * 64,
        manifest_json="{}",
    )
    state.session_ended_at_monotonic = now()
    state.record_store._session_id_factory = lambda: "session-2"
    now.advance(3.1)
    next_session, rolled = state._rollover_ended_session(ended)
    assert rolled and next_session.session_id == "session-2"
    assert run_command(state).result_code == "SESSION_ENDED"
    assert state.session_id == "session-2" and state.match.revision == 0
    state.record_store.save_match(
        "session-2", replace(match(), red_user_id="next-red"), expected_revision=0
    )
    state.match = state.record_store.load_session("session-2").match
    state.replay_clock.start()
    now.advance(0.151)
    state.before_field_command(
        FieldCommandEnvelope("new-probe", "session-2", "next-red", "continue_turn", {}, 1, 1)
    )
    assert (
        state.record_store.get_shot_ownership("session-2", "shot-0001").actor_user_id == "next-red"
    )
    assert state.record_store.get_shot_ownership("session-1", "shot-0001") == original
    old = state.record_store.get_turn_confirmation("session-1", "shot-0001")
    assert old.status == "stale" and old.resolved_command_id is None


def test_optional_source_added_on_restart_cannot_rewrite_or_block_existing_start(tmp_path):
    state, now = state_for(tmp_path)
    state.verified_start_events = ()
    state.replay_clock.start()
    now.advance(0.101)
    state.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-red", "continue_turn", {}, 1, 1)
    )
    original = state.record_store.get_shot_ownership("session-1", "shot-0001")
    assert original.source_start_event_id is None
    recovered, later = state_for(tmp_path, store=state.record_store)
    recovered.replay_clock.start()
    later.advance(0.151)
    recovered.before_field_command(
        FieldCommandEnvelope("probe", "session-1", "user-red", "continue_turn", {}, 1, 1)
    )
    assert recovered.record_store.list_shots("session-1")[0].settled_ms == 150
    assert recovered.record_store.get_shot_ownership("session-1", "shot-0001") == original
