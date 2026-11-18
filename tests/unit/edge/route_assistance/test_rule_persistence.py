from __future__ import annotations

from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    CueBallPlacement,
    MatchControlSnapshot,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import (
    PROFILE_ID,
    RuleOutcomeKind,
    decide_eight_ball,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    LiveShotMediaReference,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RealtimePotFact,
    RealtimeRuleFacts,
)
from billiards_ai.lab.shot_local.models import BallGroup


def _store(tmp_path) -> SQLiteFieldRecordStore:
    return SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        session_id_factory=lambda: "session-rules",
    )


def _match(*, phase: RackPhase = RackPhase.BREAK) -> MatchControlSnapshot:
    assigned = phase in {RackPhase.ASSIGNED, RackPhase.ON_EIGHT}
    return MatchControlSnapshot(
        red_user_id="user-red",
        blue_user_id="user-blue",
        breaker=WristbandRole.RED,
        active_role=WristbandRole.RED,
        red_group=BallGroupAssignment.SOLID if assigned else None,
        blue_group=BallGroupAssignment.STRIPE if assigned else None,
        revision=1,
        rule_profile_id=PROFILE_ID,
        rack_phase=phase,
    )


def _facts(*, revision: int = 1) -> RealtimeRuleFacts:
    return RealtimeRuleFacts(
        session_id="session-rules",
        shot_id="shot-0001",
        source_epoch_id="epoch-rules",
        match_revision_at_start=revision,
        actor_role=WristbandRole.RED,
        start_frame_id=100,
        settled_frame_id=160,
        settled_evidence_digest="e" * 64,
        first_contact_group=BallGroup.SOLID,
        pots=(),
        explicit_no_pot=FactTruth.TRUE,
        cue_ball_scratch=FactTruth.FALSE,
        post_contact_rail_or_pot=FactTruth.TRUE,
        distinct_break_object_rail_count=4,
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        after_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 7,
            BallGroup.STRIPE: 7,
        },
        object_ball_off_table=FactTruth.FALSE,
        evidence_frame_ids=(99, 100, 160, 170),
        abstention_reasons=(),
        digest="f" * 64,
        profile_id=PROFILE_ID,
    )


def _prepare_settled_shot(
    store: SQLiteFieldRecordStore,
    *,
    match: MatchControlSnapshot | None = None,
) -> None:
    session = store.open_or_resume_session(
        artifact_id="usb-rules",
        rule_profile_id=PROFILE_ID,
    )
    store.save_match(session.session_id, match or _match(), expected_revision=0)
    start = FieldShotRecord(
        session_id=session.session_id,
        shot_id="shot-0001",
        sequence=1,
        actor_role=WristbandRole.RED,
        started_ms=1000,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange("usb-rules", "usb:epoch-rules", "usb_live", 1000, 1001, "epoch-rules"),
    )
    store.record_shot_start(start, source_start_event_id="start-event-1")
    store.record_shot_settle(
        replace(
            start,
            settled_ms=2000,
            media=replace(start.media, end_ms=2000),
            live_evidence_digest="e" * 64,
            live_job_id="job-0001",
        ),
        source_epoch_id="epoch-rules",
    )


def test_rule_profile_is_pinned_only_when_a_new_session_is_created(tmp_path) -> None:
    store = _store(tmp_path)
    session = store.open_or_resume_session(
        artifact_id="usb-rules",
        rule_profile_id=PROFILE_ID,
    )

    assert session.match.rule_profile_id == PROFILE_ID
    assert session.match.rack_phase is RackPhase.BREAK

    resumed = store.open_or_resume_session(artifact_id="usb-rules")
    assert resumed.match.rule_profile_id == PROFILE_ID


def test_old_unprofiled_session_is_not_hot_upgraded(tmp_path) -> None:
    store = _store(tmp_path)
    original = store.open_or_resume_session(artifact_id="usb-rules")

    resumed = store.open_or_resume_session(
        artifact_id="usb-rules",
        rule_profile_id=PROFILE_ID,
    )

    assert original.session_id == resumed.session_id
    assert resumed.match.rule_profile_id is None


def test_usb_next_shot_can_collect_placement_evidence_after_scratch(tmp_path) -> None:
    store = _store(tmp_path)
    session = store.open_or_resume_session(
        artifact_id="usb-rules",
        rule_profile_id=PROFILE_ID,
    )
    match = replace(
        _match(phase=RackPhase.ASSIGNED),
        active_role=WristbandRole.BLUE,
        cue_ball_placement=CueBallPlacement.ANYWHERE,
        revision=2,
    )
    store.save_match(session.session_id, match, expected_revision=0)
    start = FieldShotRecord(
        session_id=session.session_id,
        shot_id="shot-0001",
        sequence=1,
        actor_role=WristbandRole.BLUE,
        started_ms=1000,
        settled_ms=None,
        recommendation_id=None,
        power_band=None,
        power_percent=None,
        media=ShotMediaRange("usb-rules", "usb:epoch-rules", "usb_live", 1000, 1001, "epoch-rules"),
    )

    store.record_shot_start(start, source_start_event_id="placement-start-event")

    assert store.list_shots(session.session_id) == (start,)


def test_rule_receipt_ledger_match_and_replay_are_one_idempotent_result(tmp_path) -> None:
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts = _facts()
    decision = decide_eight_ball(_match(), facts)

    receipt = store.apply_automatic_rule_decision(facts, decision, elapsed_ms=12.5)
    replay = store.apply_automatic_rule_decision(facts, decision, elapsed_ms=12.5)

    assert receipt == replay
    assert receipt.status == "applied"
    assert store.load_session("session-rules").match == replace(
        _match(),
        active_role=WristbandRole.BLUE,
        revision=2,
        rack_phase=RackPhase.OPEN,
    )
    ledger = store.list_command_ledger("session-rules")
    assert len(ledger) == 1
    assert ledger[0].origin == "rules_engine"
    assert ledger[0].principal_id == PROFILE_ID
    assert ledger[0].origin_sequence is None
    assert ledger[0].resulting_revision == 2


def test_abstention_is_a_durable_receipt_without_match_or_ledger_mutation(tmp_path) -> None:
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts = replace(_facts(), abstention_reasons=("first_contact_abstained",))
    decision = decide_eight_ball(_match(), facts)
    assert decision.kind is RuleOutcomeKind.ABSTAIN

    receipt = store.apply_automatic_rule_decision(facts, decision, elapsed_ms=15.0)

    assert receipt.status == "abstained"
    assert store.load_session("session-rules").match == _match()
    assert store.list_command_ledger("session-rules") == ()


def test_rule_apply_rejects_stale_revision_and_changed_facts(tmp_path) -> None:
    store = _store(tmp_path)
    _prepare_settled_shot(store)
    facts = _facts()
    decision = decide_eight_ball(_match(), facts)
    store.apply_automatic_rule_decision(facts, decision, elapsed_ms=9.0)

    changed = replace(facts, digest="a" * 64)
    changed_decision = decide_eight_ball(_match(), changed)

    try:
        store.apply_automatic_rule_decision(changed, changed_decision, elapsed_ms=10.0)
    except ValueError as exc:
        assert "conflict" in str(exc) or "stale" in str(exc)
    else:  # pragma: no cover - makes the safety assertion explicit
        raise AssertionError("changed rule facts must not overwrite the first receipt")


def test_terminal_rule_apply_ends_session_and_freezes_postgame_manifest(tmp_path) -> None:
    store = _store(tmp_path)
    match = replace(
        _match(phase=RackPhase.ON_EIGHT),
        red_group=BallGroupAssignment.SOLID,
        blue_group=BallGroupAssignment.STRIPE,
    )
    _prepare_settled_shot(store, match=match)
    store.record_cloud_session_open("session-rules")
    facts = replace(
        _facts(revision=1),
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 0,
            BallGroup.STRIPE: 7,
        },
        after_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: 1,
            BallGroup.SOLID: 0,
            BallGroup.STRIPE: 7,
        },
        first_contact_group=BallGroup.EIGHT,
        pots=(
            RealtimePotFact(
                track_id="eight-track",
                group=BallGroup.EIGHT,
                pocket_id="top-left",
                last_seen_frame_id=150,
                first_missing_frame_id=151,
                confirmed_through_frame_id=170,
                evidence_frame_ids=(150, 151, 170),
                confidence=0.99,
            ),
        ),
        explicit_no_pot=FactTruth.FALSE,
    )
    decision = decide_eight_ball(match, facts)
    assert decision.kind is RuleOutcomeKind.APPLY
    assert decision.action == "end"

    receipt = store.apply_automatic_rule_decision(facts, decision, elapsed_ms=11.0)

    session = store.load_session("session-rules")
    assert receipt.status == "applied"
    assert session.status.value == "open"
    assert session.match.winner_role is WristbandRole.RED
    assert store.get_postgame_manifest("session-rules") is None
    with pytest.raises(ValueError, match="durable USB media"):
        store.finalize_automatic_terminal_decision("session-rules", "shot-0001", "job-0001")

    store.attach_live_shot_media(
        LiveShotMediaReference(
            "session-rules",
            "shot-0001",
            "epoch-rules",
            1,
            "job-0001",
            str(tmp_path / "media"),
            "a" * 64,
        )
    )
    store.finalize_automatic_terminal_decision("session-rules", "shot-0001", "job-0001")

    session = store.load_session("session-rules")
    assert session.status.value == "ended"
    manifest = store.get_postgame_manifest("session-rules")
    assert manifest is not None
    assert manifest.session_id == "session-rules"
    assert store.list_pending_cloud_close_session_ids() == ("session-rules",)
