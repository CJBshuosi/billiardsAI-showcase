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
    apply_rule_decision,
    decide_eight_ball,
)
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RealtimePotFact,
    RealtimeRuleFacts,
)
from billiards_ai.lab.shot_local.models import BallGroup


def _pot(group: BallGroup, *, missing_frame: int = 120, ordinal: int = 1) -> RealtimePotFact:
    return RealtimePotFact(
        track_id=f"{group.value}:{ordinal}",
        group=group,
        pocket_id="corner-1",
        last_seen_frame_id=missing_frame - 1,
        first_missing_frame_id=missing_frame,
        confirmed_through_frame_id=140,
        evidence_frame_ids=(missing_frame - 1, missing_frame, 140),
        confidence=0.98,
    )


def _facts(
    *,
    profile_id: str = PROFILE_ID,
    actor: WristbandRole = WristbandRole.RED,
    revision: int = 4,
    first_contact: BallGroup | None = BallGroup.SOLID,
    pots: tuple[RealtimePotFact, ...] = (),
    explicit_no_pot: FactTruth | None = None,
    scratch: FactTruth | None = None,
    rail_or_pot: FactTruth = FactTruth.TRUE,
    break_rails: int | None = None,
    before_solid: int = 7,
    before_stripe: int = 7,
    before_eight: int = 1,
    after_solid: int | None = None,
    after_stripe: int | None = None,
    after_eight: int | None = None,
    off_table: FactTruth = FactTruth.FALSE,
    abstention_reasons: tuple[str, ...] = (),
    placement_observed: FactTruth = FactTruth.TRUE,
) -> RealtimeRuleFacts:
    potted_solid = sum(pot.group is BallGroup.SOLID for pot in pots)
    potted_stripe = sum(pot.group is BallGroup.STRIPE for pot in pots)
    potted_eight = any(pot.group is BallGroup.EIGHT for pot in pots)
    potted_cue = any(pot.group is BallGroup.CUE for pot in pots)
    return RealtimeRuleFacts(
        session_id="session-rules",
        shot_id="shot-0001",
        source_epoch_id="epoch-a",
        match_revision_at_start=revision,
        actor_role=actor,
        start_frame_id=100,
        settled_frame_id=140,
        settled_evidence_digest="e" * 64,
        first_contact_group=first_contact,
        pots=pots,
        explicit_no_pot=(
            explicit_no_pot
            if explicit_no_pot is not None
            else (FactTruth.FALSE if pots else FactTruth.TRUE)
        ),
        cue_ball_scratch=(
            scratch if scratch is not None else (FactTruth.TRUE if potted_cue else FactTruth.FALSE)
        ),
        post_contact_rail_or_pot=rail_or_pot,
        distinct_break_object_rail_count=break_rails,
        before_counts={
            BallGroup.CUE: 1,
            BallGroup.EIGHT: before_eight,
            BallGroup.SOLID: before_solid,
            BallGroup.STRIPE: before_stripe,
        },
        after_counts={
            BallGroup.CUE: 0 if potted_cue else 1,
            BallGroup.EIGHT: (
                after_eight if after_eight is not None else (0 if potted_eight else before_eight)
            ),
            BallGroup.SOLID: (
                after_solid if after_solid is not None else before_solid - potted_solid
            ),
            BallGroup.STRIPE: (
                after_stripe if after_stripe is not None else before_stripe - potted_stripe
            ),
        },
        object_ball_off_table=off_table,
        evidence_frame_ids=(100, 119, 120, 140),
        abstention_reasons=abstention_reasons,
        digest="f" * 64,
        profile_id=profile_id,
        cue_ball_placement_observed=placement_observed,
    )


def _match(
    phase: RackPhase,
    *,
    active: WristbandRole = WristbandRole.RED,
    revision: int = 4,
    profile: str | None = PROFILE_ID,
) -> MatchControlSnapshot:
    assigned = phase in {RackPhase.ASSIGNED, RackPhase.ON_EIGHT}
    return MatchControlSnapshot(
        red_user_id="user-red",
        blue_user_id="user-blue",
        breaker=WristbandRole.RED,
        active_role=active,
        red_group=BallGroupAssignment.SOLID if assigned else None,
        blue_group=BallGroupAssignment.STRIPE if assigned else None,
        revision=revision,
        rule_profile_id=profile,
        rack_phase=phase,
    )


@pytest.mark.parametrize(
    ("facts", "kind", "action", "placement"),
    [
        (
            _facts(pots=(_pot(BallGroup.SOLID),)),
            RuleOutcomeKind.APPLY,
            "continue_turn",
            CueBallPlacement.NONE,
        ),
        (
            _facts(break_rails=4),
            RuleOutcomeKind.APPLY,
            "switch_turn",
            CueBallPlacement.NONE,
        ),
        (
            _facts(break_rails=3),
            RuleOutcomeKind.MANUAL_CHOICE,
            None,
            CueBallPlacement.NONE,
        ),
        (
            _facts(pots=(_pot(BallGroup.EIGHT),)),
            RuleOutcomeKind.MANUAL_CHOICE,
            None,
            CueBallPlacement.NONE,
        ),
        (
            _facts(pots=(_pot(BallGroup.CUE),), break_rails=4),
            RuleOutcomeKind.APPLY,
            "switch_turn",
            CueBallPlacement.BEHIND_LINE,
        ),
        (
            _facts(pots=(_pot(BallGroup.CUE),), break_rails=3),
            RuleOutcomeKind.MANUAL_CHOICE,
            None,
            CueBallPlacement.NONE,
        ),
    ],
)
def test_break_matrix(
    facts: RealtimeRuleFacts,
    kind: RuleOutcomeKind,
    action: str | None,
    placement: CueBallPlacement,
) -> None:
    result = decide_eight_ball(_match(RackPhase.BREAK), facts)

    assert (result.kind, result.action, result.cue_ball_placement) == (
        kind,
        action,
        placement,
    )
    if kind is RuleOutcomeKind.APPLY:
        assert result.next_rack_phase is RackPhase.OPEN


def test_break_does_not_call_an_illegal_break_when_no_pot_is_unknown() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.BREAK),
        _facts(explicit_no_pot=FactTruth.UNKNOWN, break_rails=3),
    )

    assert decision.kind is RuleOutcomeKind.ABSTAIN


def test_pending_ball_in_hand_requires_observed_legal_placement() -> None:
    match = replace(
        _match(RackPhase.ASSIGNED),
        cue_ball_placement=CueBallPlacement.ANYWHERE,
    )

    unknown = decide_eight_ball(
        match,
        _facts(placement_observed=FactTruth.UNKNOWN),
    )
    valid = decide_eight_ball(
        match,
        _facts(placement_observed=FactTruth.TRUE),
    )

    assert unknown.kind is RuleOutcomeKind.ABSTAIN
    assert unknown.reason_code == "cue_ball_placement_unverified"
    assert valid.kind is RuleOutcomeKind.APPLY


def test_open_matching_first_contact_and_first_pot_assigns_and_continues_atomically() -> None:
    facts = _facts(
        first_contact=BallGroup.SOLID,
        pots=(_pot(BallGroup.SOLID),),
        after_solid=6,
    )

    decision = decide_eight_ball(_match(RackPhase.OPEN), facts)
    updated = apply_rule_decision(_match(RackPhase.OPEN), decision)

    assert decision.kind is RuleOutcomeKind.APPLY
    assert decision.action == "continue_turn"
    assert decision.red_group is BallGroupAssignment.SOLID
    assert updated.red_group is BallGroupAssignment.SOLID
    assert updated.blue_group is BallGroupAssignment.STRIPE
    assert updated.active_role is WristbandRole.RED
    assert updated.rack_phase is RackPhase.ASSIGNED
    assert updated.revision == 5


def test_open_only_opposite_group_pot_switches_without_assignment() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(first_contact=BallGroup.SOLID, pots=(_pot(BallGroup.STRIPE),)),
    )
    updated = apply_rule_decision(_match(RackPhase.OPEN), decision)

    assert decision.action == "switch_turn"
    assert updated.active_role is WristbandRole.BLUE
    assert updated.red_group is None
    assert updated.blue_group is None
    assert updated.rack_phase is RackPhase.OPEN


def test_open_uses_proven_pot_order_and_abstains_on_a_tie() -> None:
    ordered = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(
            first_contact=BallGroup.SOLID,
            pots=(
                _pot(BallGroup.SOLID, missing_frame=120),
                _pot(BallGroup.STRIPE, missing_frame=125),
            ),
        ),
    )
    tied = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(
            first_contact=BallGroup.SOLID,
            pots=(
                _pot(BallGroup.SOLID, missing_frame=120),
                _pot(BallGroup.STRIPE, missing_frame=120),
            ),
        ),
    )

    assert ordered.kind is RuleOutcomeKind.APPLY
    assert ordered.red_group is BallGroupAssignment.SOLID
    assert tied.kind is RuleOutcomeKind.ABSTAIN


def test_open_first_contact_with_eight_is_ball_in_hand_foul() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(first_contact=BallGroup.EIGHT),
    )

    assert decision.kind is RuleOutcomeKind.APPLY
    assert decision.action == "switch_turn"
    assert decision.cue_ball_placement is CueBallPlacement.ANYWHERE
    assert decision.next_rack_phase is RackPhase.OPEN


def test_open_assignment_enters_on_eight_when_the_assigned_inventory_is_empty() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(
            first_contact=BallGroup.SOLID,
            pots=(_pot(BallGroup.SOLID),),
            before_solid=1,
            after_solid=0,
        ),
    )

    assert decision.kind is RuleOutcomeKind.APPLY
    assert decision.next_rack_phase is RackPhase.ON_EIGHT


def test_open_assignment_is_symmetric_for_blue_player() -> None:
    match = _match(RackPhase.OPEN, active=WristbandRole.BLUE)
    facts = _facts(
        actor=WristbandRole.BLUE,
        first_contact=BallGroup.STRIPE,
        pots=(_pot(BallGroup.STRIPE),),
    )

    decision = decide_eight_ball(match, facts)
    updated = apply_rule_decision(match, decision)

    assert updated.red_group is BallGroupAssignment.SOLID
    assert updated.blue_group is BallGroupAssignment.STRIPE
    assert updated.active_role is WristbandRole.BLUE
    assert updated.revision == 5


@pytest.mark.parametrize(
    ("facts", "action", "placement", "next_phase"),
    [
        (
            _facts(pots=(_pot(BallGroup.SOLID),), after_solid=6),
            "continue_turn",
            CueBallPlacement.NONE,
            RackPhase.ASSIGNED,
        ),
        (
            _facts(rail_or_pot=FactTruth.TRUE),
            "switch_turn",
            CueBallPlacement.NONE,
            RackPhase.ASSIGNED,
        ),
        (
            _facts(first_contact=BallGroup.STRIPE),
            "switch_turn",
            CueBallPlacement.ANYWHERE,
            RackPhase.ASSIGNED,
        ),
        (
            _facts(scratch=FactTruth.TRUE),
            "switch_turn",
            CueBallPlacement.ANYWHERE,
            RackPhase.ASSIGNED,
        ),
        (
            _facts(rail_or_pot=FactTruth.FALSE),
            "switch_turn",
            CueBallPlacement.ANYWHERE,
            RackPhase.ASSIGNED,
        ),
        (
            _facts(
                pots=(_pot(BallGroup.SOLID),),
                before_solid=1,
                after_solid=0,
            ),
            "continue_turn",
            CueBallPlacement.NONE,
            RackPhase.ON_EIGHT,
        ),
    ],
)
def test_assigned_nonterminal_matrix(
    facts: RealtimeRuleFacts,
    action: str,
    placement: CueBallPlacement,
    next_phase: RackPhase,
) -> None:
    decision = decide_eight_ball(_match(RackPhase.ASSIGNED), facts)

    assert decision.kind is RuleOutcomeKind.APPLY
    assert (decision.action, decision.cue_ball_placement, decision.next_rack_phase) == (
        action,
        placement,
        next_phase,
    )


@pytest.mark.parametrize(
    "facts",
    [
        _facts(pots=(_pot(BallGroup.EIGHT),), before_solid=2),
        _facts(
            pots=(_pot(BallGroup.SOLID), _pot(BallGroup.EIGHT, ordinal=2)),
            before_solid=1,
            after_solid=0,
        ),
    ],
)
def test_early_or_same_shot_last_group_and_eight_loses(
    facts: RealtimeRuleFacts,
) -> None:
    decision = decide_eight_ball(_match(RackPhase.ASSIGNED), facts)
    updated = apply_rule_decision(_match(RackPhase.ASSIGNED), decision)

    assert decision.action == "end"
    assert decision.winner_role is WristbandRole.BLUE
    assert updated.rack_phase is RackPhase.ENDED
    assert updated.winner_role is WristbandRole.BLUE
    assert updated.revision == 5


def test_legal_eight_wins_for_current_player() -> None:
    facts = _facts(
        first_contact=BallGroup.EIGHT,
        pots=(_pot(BallGroup.EIGHT),),
        before_solid=0,
        after_solid=0,
    )

    decision = decide_eight_ball(_match(RackPhase.ON_EIGHT), facts)
    updated = apply_rule_decision(_match(RackPhase.ON_EIGHT), decision)

    assert decision.action == "end"
    assert decision.winner_role is WristbandRole.RED
    assert updated.rack_phase is RackPhase.ENDED
    assert updated.winner_role is WristbandRole.RED


def test_one_complete_single_rack_advances_break_group_turn_foul_and_result() -> None:
    """Exercise the product sequence as one match, not isolated reducer examples."""

    match = _match(RackPhase.BREAK)

    # A legal dry break opens the table and passes the turn.
    decision = decide_eight_ball(match, _facts(revision=match.revision, break_rails=4))
    match = apply_rule_decision(match, decision)
    assert (match.rack_phase, match.active_role) == (RackPhase.OPEN, WristbandRole.BLUE)

    # The first legal open-table pot assigns both groups and continues the inning.
    decision = decide_eight_ball(
        match,
        _facts(
            actor=WristbandRole.BLUE,
            revision=match.revision,
            first_contact=BallGroup.STRIPE,
            pots=(_pot(BallGroup.STRIPE),),
        ),
    )
    match = apply_rule_decision(match, decision)
    assert (match.red_group, match.blue_group, match.active_role) == (
        BallGroupAssignment.SOLID,
        BallGroupAssignment.STRIPE,
        WristbandRole.BLUE,
    )

    # A legal miss switches to red; a legal solid pot lets red continue.
    decision = decide_eight_ball(
        match,
        _facts(
            actor=WristbandRole.BLUE,
            revision=match.revision,
            first_contact=BallGroup.STRIPE,
        ),
    )
    match = apply_rule_decision(match, decision)
    assert match.active_role is WristbandRole.RED
    decision = decide_eight_ball(
        match,
        _facts(
            revision=match.revision,
            first_contact=BallGroup.SOLID,
            pots=(_pot(BallGroup.SOLID),),
        ),
    )
    match = apply_rule_decision(match, decision)
    assert match.active_role is WristbandRole.RED

    # A cue-ball scratch switches to blue and records ball in hand.
    decision = decide_eight_ball(
        match,
        _facts(
            revision=match.revision,
            first_contact=BallGroup.SOLID,
            pots=(_pot(BallGroup.CUE),),
        ),
    )
    match = apply_rule_decision(match, decision)
    assert (match.active_role, match.cue_ball_placement) == (
        WristbandRole.BLUE,
        CueBallPlacement.ANYWHERE,
    )

    # The observed legal placement clears ball in hand; blue clears its last
    # stripe, then legally pots the eight to produce the terminal result.
    decision = decide_eight_ball(
        match,
        _facts(
            actor=WristbandRole.BLUE,
            revision=match.revision,
            first_contact=BallGroup.STRIPE,
            pots=(_pot(BallGroup.STRIPE),),
            before_stripe=1,
            after_stripe=0,
            placement_observed=FactTruth.TRUE,
        ),
    )
    match = apply_rule_decision(match, decision)
    assert (match.rack_phase, match.cue_ball_placement) == (
        RackPhase.ON_EIGHT,
        CueBallPlacement.NONE,
    )
    decision = decide_eight_ball(
        match,
        _facts(
            actor=WristbandRole.BLUE,
            revision=match.revision,
            first_contact=BallGroup.EIGHT,
            pots=(_pot(BallGroup.EIGHT),),
            before_stripe=0,
            after_stripe=0,
        ),
    )
    match = apply_rule_decision(match, decision)

    assert match.rack_phase is RackPhase.ENDED
    assert match.winner_role is WristbandRole.BLUE
    assert match.terminal_reason == "legal_eight"
    assert match.revision == 11


@pytest.mark.parametrize(
    "facts",
    [
        _facts(
            first_contact=BallGroup.EIGHT,
            pots=(_pot(BallGroup.EIGHT), _pot(BallGroup.CUE, ordinal=2)),
            before_solid=0,
            after_solid=0,
        ),
        _facts(
            first_contact=BallGroup.STRIPE,
            pots=(_pot(BallGroup.EIGHT),),
            before_solid=0,
            after_solid=0,
        ),
    ],
)
def test_eight_with_scratch_or_wrong_first_contact_loses(
    facts: RealtimeRuleFacts,
) -> None:
    decision = decide_eight_ball(_match(RackPhase.ON_EIGHT), facts)

    assert decision.action == "end"
    assert decision.winner_role is WristbandRole.BLUE


def test_on_eight_legal_miss_switches_and_uses_opponent_inventory() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.ON_EIGHT),
        _facts(
            first_contact=BallGroup.EIGHT,
            before_solid=0,
            after_solid=0,
            before_stripe=1,
            after_stripe=1,
        ),
    )

    assert decision.action == "switch_turn"
    assert decision.cue_ball_placement is CueBallPlacement.NONE
    assert decision.next_rack_phase is RackPhase.ASSIGNED


@pytest.mark.parametrize("rail_or_pot", [FactTruth.TRUE, FactTruth.FALSE])
def test_unexpressed_split_hit_abstains_instead_of_guessing_an_automatic_foul(
    rail_or_pot: FactTruth,
) -> None:
    decision = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(first_contact=None, rail_or_pot=rail_or_pot),
    )

    # CBSA 2017 rule 10 favors the legal target on an observed split hit, but the
    # current facts contract has no explicit split-hit value. None therefore
    # remains unknown; it must not be reinterpreted as either a foul or legal hit.
    assert decision.kind is RuleOutcomeKind.ABSTAIN


@pytest.mark.parametrize(
    "facts",
    [
        _facts(scratch=FactTruth.UNKNOWN),
        _facts(rail_or_pot=FactTruth.UNKNOWN),
        _facts(first_contact=None),
        _facts(off_table=FactTruth.UNKNOWN),
        _facts(abstention_reasons=("source_frame_gap",)),
        _facts(actor=WristbandRole.BLUE),
        _facts(revision=3),
    ],
)
def test_required_unknown_or_lineage_conflict_abstains(
    facts: RealtimeRuleFacts,
) -> None:
    decision = decide_eight_ball(_match(RackPhase.ASSIGNED), facts)

    assert decision.kind is RuleOutcomeKind.ABSTAIN


def test_profile_mismatch_or_existing_winner_abstains() -> None:
    wrong_profile = decide_eight_ball(
        _match(RackPhase.OPEN, profile=None),
        _facts(),
    )
    ended = MatchControlSnapshot(
        active_role=WristbandRole.RED,
        revision=4,
        rule_profile_id=PROFILE_ID,
        rack_phase=RackPhase.ENDED,
        winner_role=WristbandRole.RED,
        terminal_reason="legal_eight",
    )
    existing_winner = decide_eight_ball(ended, _facts())

    assert wrong_profile.kind is RuleOutcomeKind.ABSTAIN
    assert existing_winner.kind is RuleOutcomeKind.ABSTAIN


def test_fact_profile_must_explicitly_match_the_pinned_match_profile() -> None:
    decision = decide_eight_ball(
        _match(RackPhase.OPEN),
        _facts(profile_id="different-profile"),
    )

    assert decision.kind is RuleOutcomeKind.ABSTAIN
    assert decision.reason_code == "profile_mismatch"


def test_apply_rejects_non_apply_and_stale_or_wrong_profile_decisions() -> None:
    match = _match(RackPhase.OPEN)
    abstained = decide_eight_ball(match, _facts(scratch=FactTruth.UNKNOWN))
    applied = decide_eight_ball(match, _facts(first_contact=BallGroup.EIGHT))

    with pytest.raises(ValueError, match="APPLY"):
        apply_rule_decision(match, abstained)
    with pytest.raises(ValueError, match="revision"):
        apply_rule_decision(replace(match, revision=5), applied)
    with pytest.raises(ValueError, match="profile"):
        apply_rule_decision(replace(match, rule_profile_id="other-profile"), applied)
