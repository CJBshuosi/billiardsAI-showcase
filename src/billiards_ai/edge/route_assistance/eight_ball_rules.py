"""Pure fail-closed reducer for the fixed observable Chinese eight-ball profile."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    CueBallPlacement,
    MatchControlSnapshot,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RealtimePotFact,
    RealtimeRuleFacts,
)
from billiards_ai.lab.shot_local.models import BallGroup

PROFILE_ID = "cbsa-chinese-pool-2017-observable-v1"
REFEREE_COMPLETENESS = "observable_subset"

RuleAction = Literal["continue_turn", "switch_turn", "end"]


class RuleOutcomeKind(StrEnum):
    APPLY = "apply"
    ABSTAIN = "abstain"
    MANUAL_CHOICE = "manual_choice"


@dataclass(frozen=True, slots=True)
class RuleDecision:
    session_id: str
    shot_id: str
    source_epoch_id: str
    expected_revision: int
    actor_role: WristbandRole
    profile_id: str
    fact_digest: str
    settled_evidence_digest: str
    kind: RuleOutcomeKind
    action: RuleAction | None
    red_group: BallGroupAssignment | None
    next_rack_phase: RackPhase | None
    cue_ball_placement: CueBallPlacement
    winner_role: WristbandRole | None
    terminal_reason: str | None
    reason_code: str
    evidence_frame_ids: tuple[int, ...]
    referee_completeness: str = REFEREE_COMPLETENESS
    decision_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.session_id,
                self.shot_id,
                self.source_epoch_id,
                self.profile_id,
                self.fact_digest,
                self.settled_evidence_digest,
                self.reason_code,
            )
        ):
            raise ValueError("rule decision lineage must be complete")
        if self.expected_revision < 0:
            raise ValueError("rule decision revision must be non-negative")
        if self.kind is RuleOutcomeKind.APPLY:
            if self.action is None or self.next_rack_phase is None:
                raise ValueError("APPLY decision requires an action and next rack phase")
        elif (
            any(
                value is not None
                for value in (
                    self.action,
                    self.red_group,
                    self.next_rack_phase,
                    self.winner_role,
                    self.terminal_reason,
                )
            )
            or self.cue_ball_placement is not CueBallPlacement.NONE
        ):
            raise ValueError("non-APPLY decision cannot carry match changes")
        if self.action == "end":
            if (
                self.next_rack_phase is not RackPhase.ENDED
                or self.winner_role is None
                or not self.terminal_reason
            ):
                raise ValueError("end decision requires winner and terminal reason")
        elif self.winner_role is not None or self.terminal_reason is not None:
            raise ValueError("non-terminal decision cannot carry a winner")
        payload = {
            "action": self.action,
            "actor_role": self.actor_role.value,
            "cue_ball_placement": self.cue_ball_placement.value,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "expected_revision": self.expected_revision,
            "fact_digest": self.fact_digest,
            "kind": self.kind.value,
            "next_rack_phase": (
                None if self.next_rack_phase is None else self.next_rack_phase.value
            ),
            "profile_id": self.profile_id,
            "reason_code": self.reason_code,
            "red_group": None if self.red_group is None else self.red_group.value,
            "referee_completeness": self.referee_completeness,
            "session_id": self.session_id,
            "settled_evidence_digest": self.settled_evidence_digest,
            "shot_id": self.shot_id,
            "source_epoch_id": self.source_epoch_id,
            "terminal_reason": self.terminal_reason,
            "winner_role": None if self.winner_role is None else self.winner_role.value,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        object.__setattr__(self, "decision_digest", sha256(encoded).hexdigest())


def _other(role: WristbandRole) -> WristbandRole:
    return WristbandRole.BLUE if role is WristbandRole.RED else WristbandRole.RED


def _decision(
    match: MatchControlSnapshot,
    facts: RealtimeRuleFacts,
    kind: RuleOutcomeKind,
    reason_code: str,
    *,
    action: RuleAction | None = None,
    red_group: BallGroupAssignment | None = None,
    next_phase: RackPhase | None = None,
    placement: CueBallPlacement = CueBallPlacement.NONE,
    winner: WristbandRole | None = None,
    terminal_reason: str | None = None,
) -> RuleDecision:
    return RuleDecision(
        session_id=facts.session_id,
        shot_id=facts.shot_id,
        source_epoch_id=facts.source_epoch_id,
        expected_revision=match.revision,
        actor_role=facts.actor_role,
        profile_id=PROFILE_ID,
        fact_digest=facts.digest,
        settled_evidence_digest=facts.settled_evidence_digest,
        kind=kind,
        action=action,
        red_group=red_group,
        next_rack_phase=next_phase,
        cue_ball_placement=placement,
        winner_role=winner,
        terminal_reason=terminal_reason,
        reason_code=reason_code,
        evidence_frame_ids=facts.evidence_frame_ids,
    )


def _abstain(
    match: MatchControlSnapshot, facts: RealtimeRuleFacts, reason_code: str
) -> RuleDecision:
    return _decision(match, facts, RuleOutcomeKind.ABSTAIN, reason_code)


def _manual(
    match: MatchControlSnapshot, facts: RealtimeRuleFacts, reason_code: str
) -> RuleDecision:
    return _decision(match, facts, RuleOutcomeKind.MANUAL_CHOICE, reason_code)


def _apply(
    match: MatchControlSnapshot,
    facts: RealtimeRuleFacts,
    reason_code: str,
    *,
    action: RuleAction,
    red_group: BallGroupAssignment | None = None,
    next_phase: RackPhase,
    placement: CueBallPlacement = CueBallPlacement.NONE,
    winner: WristbandRole | None = None,
    terminal_reason: str | None = None,
) -> RuleDecision:
    return _decision(
        match,
        facts,
        RuleOutcomeKind.APPLY,
        reason_code,
        action=action,
        red_group=red_group,
        next_phase=next_phase,
        placement=placement,
        winner=winner,
        terminal_reason=terminal_reason,
    )


def _count(facts: RealtimeRuleFacts, *, after: bool, group: BallGroup) -> int | None:
    counts = facts.after_counts if after else facts.before_counts
    value = counts.get(group)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _eight_potted(facts: RealtimeRuleFacts) -> bool | None:
    pots = tuple(pot for pot in facts.pots if pot.group is BallGroup.EIGHT)
    if len(pots) == 1:
        return True
    if pots:
        return None
    before = _count(facts, after=False, group=BallGroup.EIGHT)
    after = _count(facts, after=True, group=BallGroup.EIGHT)
    if before == after == 1:
        return False
    return None


def _normal_pots(facts: RealtimeRuleFacts) -> tuple[RealtimePotFact, ...]:
    return tuple(pot for pot in facts.pots if pot.group in {BallGroup.SOLID, BallGroup.STRIPE})


def _assignment(group: BallGroup) -> BallGroupAssignment:
    if group is BallGroup.SOLID:
        return BallGroupAssignment.SOLID
    if group is BallGroup.STRIPE:
        return BallGroupAssignment.STRIPE
    raise ValueError("group is not assignable")


def _red_assignment(actor: WristbandRole, group: BallGroup) -> BallGroupAssignment:
    actor_group = _assignment(group)
    if actor is WristbandRole.RED:
        return actor_group
    return (
        BallGroupAssignment.STRIPE
        if actor_group is BallGroupAssignment.SOLID
        else BallGroupAssignment.SOLID
    )


def _role_group(match: MatchControlSnapshot, role: WristbandRole) -> BallGroup | None:
    assignment = match.red_group if role is WristbandRole.RED else match.blue_group
    if assignment is BallGroupAssignment.SOLID:
        return BallGroup.SOLID
    if assignment is BallGroupAssignment.STRIPE:
        return BallGroup.STRIPE
    return None


def _phase_for_role(
    match: MatchControlSnapshot, facts: RealtimeRuleFacts, role: WristbandRole
) -> RackPhase | None:
    group = _role_group(match, role)
    if group is None:
        return None
    remaining = _count(facts, after=True, group=group)
    if remaining is None:
        return None
    return RackPhase.ON_EIGHT if remaining == 0 else RackPhase.ASSIGNED


def _switch(
    match: MatchControlSnapshot,
    facts: RealtimeRuleFacts,
    reason_code: str,
    *,
    placement: CueBallPlacement = CueBallPlacement.NONE,
    phase: RackPhase | None = None,
) -> RuleDecision:
    next_phase = phase or _phase_for_role(match, facts, _other(facts.actor_role))
    if next_phase is None:
        return _abstain(match, facts, "next_player_inventory_unknown")
    return _apply(
        match,
        facts,
        reason_code,
        action="switch_turn",
        next_phase=next_phase,
        placement=placement,
    )


def _end(
    match: MatchControlSnapshot,
    facts: RealtimeRuleFacts,
    winner: WristbandRole,
    reason_code: str,
) -> RuleDecision:
    return _apply(
        match,
        facts,
        reason_code,
        action="end",
        next_phase=RackPhase.ENDED,
        winner=winner,
        terminal_reason=reason_code,
    )


def _no_pot_end(
    match: MatchControlSnapshot,
    facts: RealtimeRuleFacts,
    *,
    legal_phase: RackPhase | None = None,
) -> RuleDecision:
    if facts.explicit_no_pot is not FactTruth.TRUE:
        return _abstain(match, facts, "no_pot_not_explicit")
    if facts.post_contact_rail_or_pot is FactTruth.UNKNOWN:
        return _abstain(match, facts, "post_contact_rail_unknown")
    if facts.post_contact_rail_or_pot is FactTruth.FALSE:
        return _switch(
            match,
            facts,
            "no_rail_foul",
            placement=CueBallPlacement.ANYWHERE,
            phase=legal_phase,
        )
    return _switch(match, facts, "legal_miss", phase=legal_phase)


def _decide_break(match: MatchControlSnapshot, facts: RealtimeRuleFacts) -> RuleDecision:
    eight = _eight_potted(facts)
    if eight is None:
        return _abstain(match, facts, "eight_status_unknown")
    if eight:
        return _manual(match, facts, "break_eight_requires_reset")
    if facts.cue_ball_scratch is FactTruth.UNKNOWN:
        return _abstain(match, facts, "scratch_unknown")
    if facts.first_contact_group not in {
        BallGroup.SOLID,
        BallGroup.STRIPE,
        BallGroup.EIGHT,
    }:
        return _abstain(match, facts, "break_first_contact_unknown")

    object_pots = _normal_pots(facts)
    if object_pots:
        if facts.cue_ball_scratch is FactTruth.TRUE:
            return _apply(
                match,
                facts,
                "break_scratch",
                action="switch_turn",
                next_phase=RackPhase.OPEN,
                placement=CueBallPlacement.BEHIND_LINE,
            )
        return _apply(
            match,
            facts,
            "legal_break_object_pot",
            action="continue_turn",
            next_phase=RackPhase.OPEN,
        )

    cue_pots = tuple(pot for pot in facts.pots if pot.group is BallGroup.CUE)
    if facts.cue_ball_scratch is FactTruth.TRUE:
        if len(cue_pots) != 1:
            return _abstain(match, facts, "break_scratch_evidence_inconsistent")
    elif cue_pots or facts.explicit_no_pot is not FactTruth.TRUE:
        return _abstain(match, facts, "break_no_object_pot_not_explicit")

    rails = facts.distinct_break_object_rail_count
    if isinstance(rails, bool) or not isinstance(rails, int) or rails < 0:
        return _abstain(match, facts, "break_rail_count_unknown")
    if rails < 4:
        return _manual(match, facts, "illegal_break_requires_choice")
    if facts.cue_ball_scratch is FactTruth.TRUE:
        return _apply(
            match,
            facts,
            "break_scratch",
            action="switch_turn",
            next_phase=RackPhase.OPEN,
            placement=CueBallPlacement.BEHIND_LINE,
        )
    return _apply(
        match,
        facts,
        "legal_dry_break",
        action="switch_turn",
        next_phase=RackPhase.OPEN,
    )


def _earliest_open_group(facts: RealtimeRuleFacts) -> BallGroup | None:
    first_by_group: dict[BallGroup, int] = {}
    for pot in _normal_pots(facts):
        previous = first_by_group.get(pot.group)
        if previous is None or pot.first_missing_frame_id < previous:
            first_by_group[pot.group] = pot.first_missing_frame_id
    if len(first_by_group) == 1:
        return next(iter(first_by_group))
    if len(first_by_group) != 2:
        return None
    if first_by_group[BallGroup.SOLID] == first_by_group[BallGroup.STRIPE]:
        return None
    return min(first_by_group, key=first_by_group.__getitem__)


def _decide_open(match: MatchControlSnapshot, facts: RealtimeRuleFacts) -> RuleDecision:
    eight = _eight_potted(facts)
    if eight is None:
        return _abstain(match, facts, "eight_status_unknown")
    if eight:
        return _end(match, facts, _other(facts.actor_role), "eight_potted_on_open_table")
    if facts.cue_ball_scratch is FactTruth.UNKNOWN:
        return _abstain(match, facts, "scratch_unknown")
    if facts.cue_ball_scratch is FactTruth.TRUE:
        return _switch(
            match,
            facts,
            "scratch_foul",
            placement=CueBallPlacement.ANYWHERE,
            phase=RackPhase.OPEN,
        )
    if facts.first_contact_group is BallGroup.EIGHT:
        return _switch(
            match,
            facts,
            "open_table_eight_first_foul",
            placement=CueBallPlacement.ANYWHERE,
            phase=RackPhase.OPEN,
        )
    if facts.first_contact_group not in {BallGroup.SOLID, BallGroup.STRIPE}:
        return _abstain(match, facts, "first_contact_unknown")

    pots = _normal_pots(facts)
    if not pots:
        return _no_pot_end(match, facts, legal_phase=RackPhase.OPEN)
    earliest = _earliest_open_group(facts)
    if earliest is None:
        return _abstain(match, facts, "open_table_pot_order_ambiguous")
    if earliest is not facts.first_contact_group:
        return _switch(match, facts, "opposite_group_potted", phase=RackPhase.OPEN)
    remaining = _count(facts, after=True, group=earliest)
    if remaining is None:
        return _abstain(match, facts, "assigned_group_inventory_unknown")
    return _apply(
        match,
        facts,
        "open_table_group_assigned",
        action="continue_turn",
        red_group=_red_assignment(facts.actor_role, earliest),
        next_phase=RackPhase.ON_EIGHT if remaining == 0 else RackPhase.ASSIGNED,
    )


def _decide_assigned(match: MatchControlSnapshot, facts: RealtimeRuleFacts) -> RuleDecision:
    own_group = _role_group(match, facts.actor_role)
    if own_group is None:
        return _abstain(match, facts, "group_assignment_missing")
    before_own = _count(facts, after=False, group=own_group)
    if before_own is None or before_own == 0:
        return _abstain(match, facts, "assigned_phase_inventory_mismatch")
    eight = _eight_potted(facts)
    if eight is None:
        return _abstain(match, facts, "eight_status_unknown")
    if eight:
        return _end(match, facts, _other(facts.actor_role), "early_eight")
    if facts.cue_ball_scratch is FactTruth.UNKNOWN:
        return _abstain(match, facts, "scratch_unknown")
    if facts.cue_ball_scratch is FactTruth.TRUE:
        return _switch(
            match,
            facts,
            "scratch_foul",
            placement=CueBallPlacement.ANYWHERE,
        )
    if facts.first_contact_group is None or facts.first_contact_group is BallGroup.UNKNOWN:
        return _abstain(match, facts, "first_contact_unknown")
    if facts.first_contact_group is not own_group:
        return _switch(
            match,
            facts,
            "wrong_first_contact_foul",
            placement=CueBallPlacement.ANYWHERE,
        )

    pots = _normal_pots(facts)
    if any(pot.group is own_group for pot in pots):
        remaining = _count(facts, after=True, group=own_group)
        if remaining is None:
            return _abstain(match, facts, "own_group_inventory_unknown")
        return _apply(
            match,
            facts,
            "own_group_potted",
            action="continue_turn",
            next_phase=RackPhase.ON_EIGHT if remaining == 0 else RackPhase.ASSIGNED,
        )
    if pots:
        return _switch(match, facts, "opponent_group_potted")
    return _no_pot_end(match, facts)


def _decide_on_eight(match: MatchControlSnapshot, facts: RealtimeRuleFacts) -> RuleDecision:
    own_group = _role_group(match, facts.actor_role)
    if own_group is None or _count(facts, after=False, group=own_group) != 0:
        return _abstain(match, facts, "on_eight_inventory_mismatch")
    eight = _eight_potted(facts)
    if eight is None:
        return _abstain(match, facts, "eight_status_unknown")
    if eight:
        if facts.cue_ball_scratch is FactTruth.TRUE:
            return _end(match, facts, _other(facts.actor_role), "eight_with_scratch")
        if facts.first_contact_group not in {BallGroup.EIGHT, None, BallGroup.UNKNOWN}:
            return _end(match, facts, _other(facts.actor_role), "eight_with_foul")
        if (
            facts.cue_ball_scratch is FactTruth.UNKNOWN
            or facts.first_contact_group is not BallGroup.EIGHT
        ):
            return _abstain(match, facts, "eight_legality_unknown")
        return _end(match, facts, facts.actor_role, "legal_eight")
    if facts.cue_ball_scratch is FactTruth.UNKNOWN:
        return _abstain(match, facts, "scratch_unknown")
    if facts.cue_ball_scratch is FactTruth.TRUE:
        return _switch(
            match,
            facts,
            "scratch_foul",
            placement=CueBallPlacement.ANYWHERE,
        )
    if facts.first_contact_group is None or facts.first_contact_group is BallGroup.UNKNOWN:
        return _abstain(match, facts, "first_contact_unknown")
    if facts.first_contact_group is not BallGroup.EIGHT:
        return _switch(
            match,
            facts,
            "wrong_first_contact_foul",
            placement=CueBallPlacement.ANYWHERE,
        )
    if _normal_pots(facts):
        return _switch(match, facts, "legal_eight_miss")
    return _no_pot_end(match, facts)


def decide_eight_ball(match: MatchControlSnapshot, facts: RealtimeRuleFacts) -> RuleDecision:
    """Return one deterministic decision without mutating the match."""

    if not (facts.profile_id == match.rule_profile_id == PROFILE_ID):
        return _abstain(match, facts, "profile_mismatch")
    if (
        match.cue_ball_placement is not CueBallPlacement.NONE
        and facts.cue_ball_placement_observed is not FactTruth.TRUE
    ):
        return _abstain(match, facts, "cue_ball_placement_unverified")
    if match.rack_phase is RackPhase.ENDED or match.winner_role is not None:
        return _abstain(match, facts, "rack_already_ended")
    if match.active_role is None or facts.actor_role is not match.active_role:
        return _abstain(match, facts, "actor_mismatch")
    if facts.match_revision_at_start != match.revision:
        return _abstain(match, facts, "revision_mismatch")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            facts.session_id,
            facts.shot_id,
            facts.source_epoch_id,
            facts.settled_evidence_digest,
            facts.digest,
        )
    ):
        return _abstain(match, facts, "lineage_incomplete")
    if facts.abstention_reasons:
        return _abstain(match, facts, "fact_adapter_abstained")
    if facts.object_ball_off_table is not FactTruth.FALSE:
        return _abstain(match, facts, "object_ball_off_table_unresolved")

    if match.rack_phase is RackPhase.BREAK:
        return _decide_break(match, facts)
    if match.rack_phase is RackPhase.OPEN:
        return _decide_open(match, facts)
    if match.rack_phase is RackPhase.ASSIGNED:
        return _decide_assigned(match, facts)
    if match.rack_phase is RackPhase.ON_EIGHT:
        return _decide_on_eight(match, facts)
    return _abstain(match, facts, "unsupported_rack_phase")


def apply_rule_decision(
    match: MatchControlSnapshot, decision: RuleDecision
) -> MatchControlSnapshot:
    """Atomically apply exactly one validated reducer decision and revision."""

    if decision.kind is not RuleOutcomeKind.APPLY:
        raise ValueError("only an APPLY rule decision can change a match")
    if match.rule_profile_id != PROFILE_ID or decision.profile_id != match.rule_profile_id:
        raise ValueError("rule decision profile mismatch")
    if decision.expected_revision != match.revision:
        raise ValueError("rule decision revision mismatch")
    if match.active_role is None or decision.actor_role is not match.active_role:
        raise ValueError("rule decision actor mismatch")
    if match.rack_phase is RackPhase.ENDED or match.winner_role is not None:
        raise ValueError("ended match cannot apply a rule decision")

    active_role = match.active_role
    if decision.action == "switch_turn":
        active_role = _other(active_role)
    elif decision.action not in {"continue_turn", "end"}:
        raise ValueError("rule decision action is invalid")

    red_group = match.red_group
    blue_group = match.blue_group
    if decision.red_group is not None:
        red_group = decision.red_group
        blue_group = (
            BallGroupAssignment.STRIPE
            if red_group is BallGroupAssignment.SOLID
            else BallGroupAssignment.SOLID
        )

    return replace(
        match,
        active_role=active_role,
        red_group=red_group,
        blue_group=blue_group,
        revision=match.revision + 1,
        rack_phase=decision.next_rack_phase,
        cue_ball_placement=decision.cue_ball_placement,
        winner_role=decision.winner_role,
        terminal_reason=decision.terminal_reason,
    )
