from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
from billiards_ai.lab.mainline1_product_realtime_runtime import LiveSettledShot
from billiards_ai.lab.mainline1_realtime_refinement import FactCoverageVerdict
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)


@pytest.fixture
def store(tmp_path):
    return SQLiteFieldRecordStore(tmp_path / "field.sqlite")


@pytest.fixture
def coordinator(store):
    session = store.open_or_resume_session(artifact_id="usb-capture-1")
    store.save_match(
        session.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            active_role=WristbandRole.RED,
            red_group=BallGroupAssignment.SOLID,
            blue_group=BallGroupAssignment.STRIPE,
            revision=1,
        ),
        expected_revision=0,
    )
    return LiveMatchCoordinator(
        store=store,
        session_id=session.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )


@pytest.fixture
def settled_shot():
    frame = ObservationFrame(
        20,
        200,
        (
            CalibratedBallObservation("cue", 500, 500, 0.95, BallGroup.CUE, 0.95),
            CalibratedBallObservation("solid", 1500, 650, 0.95, BallGroup.SOLID, 0.95),
            CalibratedBallObservation("stripe", 1900, 900, 0.95, BallGroup.STRIPE, 0.95),
        ),
    )
    return LiveSettledShot(
        "shot-1",
        10,
        20,
        200_000_000,
        (frame,),
        True,
        None,
        {"pot": FactCoverageVerdict("observed")},
        "epoch-1",
        "job-1",
    )


def start(coordinator, shot_id="shot-1", started_ms=100):
    return coordinator.shot_started(shot_id=shot_id, started_ms=started_ms)


def test_missing_rule_fact_never_switches_turn(coordinator, settled_shot):
    before = coordinator.snapshot.match
    start(coordinator)
    projection = coordinator.settle(settled_shot, facts=())
    assert projection.rule_status == "RULES_PENDING"
    assert projection.guidance is None
    assert projection.match.active_role == before.active_role
    assert projection.confirmation_ref.match_revision == 1
    assert projection.rule_pending_reason == "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"


@pytest.mark.parametrize("coverage", ["observed", "abstained", "unsupported"])
def test_observed_pot_is_not_a_complete_turn_verdict(coordinator, settled_shot, coverage):
    start(coordinator)
    result = coordinator.settle(
        replace(settled_shot, fact_coverage={"pot": FactCoverageVerdict(coverage)}),
        facts=({"event_type": "pot"},),
    )
    assert result.rule_status == "RULES_PENDING"
    assert result.match.active_role is WristbandRole.RED
    assert result.guidance is None


def test_authoritative_turn_fact_advances_without_a_user_or_tablet_command(
    coordinator, settled_shot, store
):
    start(coordinator)
    decision_shot = replace(
        settled_shot,
        fact_coverage={"turn_decision": FactCoverageVerdict("observed")},
    )
    result_session_id = coordinator.snapshot.session_id

    result = coordinator.settle(
        decision_shot,
        facts=(
            {
                "event_type": "turn_decision",
                "action": "switch_turn",
                "session_id": result_session_id,
                "shot_id": "shot-1",
                "source_epoch_id": "epoch-1",
                "match_revision": 1,
            },
        ),
    )

    assert result.session_id == result_session_id
    assert result.rule_status == "READY"
    assert result.match.active_role is WristbandRole.BLUE
    assert result.guidance is not None
    receipt = store.get_turn_confirmation(result.session_id, "shot-1")
    assert receipt.status == "resolved"
    assert receipt.resolved_command_id.startswith("edge-auto-")


@pytest.mark.parametrize(
    "action,role,target",
    [
        ("continue_turn", WristbandRole.RED, "solid"),
        ("switch_turn", WristbandRole.BLUE, "stripe"),
    ],
)
def test_confirmed_turn_replans_for_authoritative_player(
    coordinator,
    settled_shot,
    action,
    role,
    target,
):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    result = coordinator.confirm_turn("cmd-1", action, pending.confirmation_ref)
    assert result.match.active_role is role
    assert result.rule_status == "READY"
    assert result.guidance is not None
    assert coordinator.guidance_result.recommendation.candidates[0].target_kind.value == target
    again = coordinator.confirm_turn("cmd-1", action, pending.confirmation_ref)
    assert again == result
    assert again.match.revision == 2
    with pytest.raises(ValueError, match="conflict"):
        coordinator.confirm_turn(
            "cmd-1",
            "switch_turn" if action == "continue_turn" else "continue_turn",
            pending.confirmation_ref,
        )


def test_start_clears_guidance_and_snapshots_actor(coordinator, settled_shot, store):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    coordinator.confirm_turn("cmd-1", "switch_turn", pending.confirmation_ref)
    result = start(coordinator, "shot-2", 300)
    assert result.guidance is None
    receipt = store.get_shot_ownership(result.session_id, "shot-2")
    assert (receipt.actor_user_id, receipt.actor_role, receipt.match_revision) == (
        "bob",
        WristbandRole.BLUE,
        2,
    )
    assert receipt.source_kind == "usb_live"
    assert receipt.source_epoch_id == "epoch-1"
    assert receipt.source_start_event_id


@pytest.mark.parametrize("change", ["session", "shot", "revision"])
def test_stale_confirmation_never_mutates_current_match(coordinator, settled_shot, store, change):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    if change == "session":
        coordinator.end_and_rollover()
    elif change == "shot":
        start(coordinator, "shot-2", 300)
    else:
        store.save_match(
            pending.session_id, replace(pending.match, revision=2), expected_revision=1
        )
    before = coordinator.snapshot.match
    with pytest.raises(ValueError, match="stale"):
        coordinator.confirm_turn("cmd-old", "switch_turn", pending.confirmation_ref)
    assert coordinator.snapshot.match == before
    assert coordinator.snapshot.guidance is None


def test_settle_requires_matching_start_and_epoch(coordinator, settled_shot):
    with pytest.raises(ValueError, match="start"):
        coordinator.settle(settled_shot, facts=())
    start(coordinator)
    with pytest.raises(ValueError, match="epoch"):
        coordinator.settle(replace(settled_shot, source_epoch_id="another"), facts=())


def test_abstained_observation_cannot_become_route_after_confirmation(coordinator, settled_shot):
    start(coordinator)
    pending = coordinator.settle(
        replace(settled_shot, publishable=False, shot_abstention_reason="tracking_uncertain"),
        facts=(),
    )
    result = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert result.rule_status == "READY"
    assert result.guidance is None
    assert result.route_abstained_reason == "tracking_uncertain"


def test_duplicate_settlement_does_not_reopen_resolved_turn(coordinator, settled_shot):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    confirmed = coordinator.confirm_turn("cmd-1", "switch_turn", pending.confirmation_ref)
    assert coordinator.settle(settled_shot, facts=()) == confirmed


def test_unknown_participant_remains_unknown(store):
    session = store.open_or_resume_session(artifact_id="usb-unknown")
    coordinator = LiveMatchCoordinator(
        store=store, session_id=session.session_id, source_epoch_id="epoch-1", calibration_id=None
    )
    start(coordinator)
    receipt = store.get_shot_ownership(session.session_id, "shot-1")
    assert receipt.evidence_status == "unknown"
    assert receipt.actor_user_id is None and receipt.actor_role is None


def test_postroll_observations_do_not_move_physical_settlement(coordinator, settled_shot, store):
    start(coordinator)
    postroll = replace(
        settled_shot.observations[-1], frame_id=30, timestamp_ms=300, observations=()
    )
    pending = coordinator.settle(
        replace(settled_shot, observations=settled_shot.observations + (postroll,)), facts=()
    )
    assert pending.rule_status == "RULES_PENDING"
    assert store.list_shots(pending.session_id)[0].settled_ms == 200
    result = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert result.guidance is not None


def test_missing_settled_observation_still_persists_pending_shot(coordinator, settled_shot, store):
    start(coordinator)
    pending = coordinator.settle(replace(settled_shot, observations=()), facts=())
    assert pending.rule_status == "RULES_PENDING"
    assert store.list_shots(pending.session_id)[0].settled_ms == 200
    result = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert result.guidance is None
    assert result.route_abstained_reason == "settled_observation_unavailable"


def test_resolved_confirmation_cannot_replan_after_later_revision(coordinator, settled_shot, store):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    confirmed = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    store.save_match(pending.session_id, replace(confirmed.match, revision=3), expected_revision=2)
    with pytest.raises(ValueError, match="stale"):
        coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert coordinator.snapshot.guidance is None
    assert coordinator.snapshot.rule_status == "RULES_PENDING"


def test_physical_settlement_survives_later_match_revision(coordinator, settled_shot, store):
    begun = start(coordinator)
    store.save_match(begun.session_id, replace(begun.match, revision=2), expected_revision=1)
    coordinator.settle(settled_shot, facts=())
    assert store.list_shots(begun.session_id)[0].settled_ms == 200
    assert store.get_live_shot_fact(begun.session_id, "shot-1").actor_user_id == "alice"


def test_session_end_clears_route_and_rejects_old_confirmed_command(
    coordinator, settled_shot, store
):
    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    store.end_session(pending.session_id)
    assert coordinator.snapshot.rule_status == "SESSION_ENDED"
    assert coordinator.snapshot.guidance is None
    with pytest.raises(ValueError, match="stale"):
        coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)


def test_settlement_fact_uses_start_owner_after_roster_changes(coordinator, settled_shot, store):
    start(coordinator)
    current = coordinator.snapshot
    store.save_match(
        current.session_id,
        replace(current.match, red_user_id="carol", revision=2),
        expected_revision=1,
    )
    resumed = LiveMatchCoordinator(
        store=store,
        session_id=current.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )
    resumed.settle(settled_shot, facts=())
    assert store.get_live_shot_fact(current.session_id, "shot-1").actor_user_id == "alice"
