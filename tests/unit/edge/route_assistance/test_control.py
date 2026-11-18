from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance import control
from billiards_ai.edge.route_assistance.control import (
    AssignGroups,
    BallGroupAssignment,
    ContinueTurn,
    FreezeMatchSettings,
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    ReleaseParticipant,
    SetBreaker,
    SetParticipant,
    SwitchTurn,
    WristbandRole,
    apply_control,
)


def test_users_wristbands_and_ball_groups_remain_separate() -> None:
    snapshot = MatchControlSnapshot()
    snapshot = apply_control(
        snapshot,
        SetParticipant(WristbandRole.RED, "user-alice"),
    )
    snapshot = apply_control(
        snapshot,
        SetParticipant(WristbandRole.BLUE, "user-bob"),
    )
    snapshot = apply_control(snapshot, SetBreaker(WristbandRole.RED))
    snapshot = apply_control(
        snapshot,
        AssignGroups(red_group=BallGroupAssignment.STRIPE),
    )

    assert snapshot.red_user_id == "user-alice"
    assert snapshot.blue_user_id == "user-bob"
    assert snapshot.red_group is BallGroupAssignment.STRIPE
    assert snapshot.blue_group is BallGroupAssignment.SOLID
    assert snapshot.breaker is WristbandRole.RED
    assert snapshot.active_role is WristbandRole.RED
    assert snapshot.revision == 4


def test_same_long_term_user_cannot_occupy_both_wristbands() -> None:
    snapshot = apply_control(
        MatchControlSnapshot(),
        SetParticipant(WristbandRole.RED, "user-alice"),
    )

    with pytest.raises(ValueError, match="both wristbands"):
        apply_control(
            snapshot,
            SetParticipant(WristbandRole.BLUE, "user-alice"),
        )


def test_group_assignment_is_always_bijective() -> None:
    solid_red = apply_control(
        MatchControlSnapshot(),
        AssignGroups(red_group=BallGroupAssignment.SOLID),
    )
    stripe_red = apply_control(
        MatchControlSnapshot(),
        AssignGroups(red_group=BallGroupAssignment.STRIPE),
    )

    assert (solid_red.red_group, solid_red.blue_group) == (
        BallGroupAssignment.SOLID,
        BallGroupAssignment.STRIPE,
    )
    assert (stripe_red.red_group, stripe_red.blue_group) == (
        BallGroupAssignment.STRIPE,
        BallGroupAssignment.SOLID,
    )


def test_continue_and_switch_only_change_turn_state() -> None:
    snapshot = apply_control(MatchControlSnapshot(), SetBreaker(WristbandRole.BLUE))
    continued = apply_control(snapshot, ContinueTurn())
    switched = apply_control(continued, SwitchTurn())

    assert continued.active_role is WristbandRole.BLUE
    assert switched.active_role is WristbandRole.RED
    assert switched.breaker is WristbandRole.BLUE
    assert switched.revision == 3


def test_control_commands_validate_identity_and_require_an_active_turn() -> None:
    with pytest.raises(ValueError, match="user ID"):
        SetParticipant(WristbandRole.RED, " ")
    with pytest.raises(ValueError, match="active turn"):
        apply_control(MatchControlSnapshot(), ContinueTurn())
    with pytest.raises(ValueError, match="active turn"):
        apply_control(MatchControlSnapshot(), SwitchTurn())


def test_release_participant_clears_only_the_selected_role() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice",
        blue_user_id="user-bob",
    )

    released = apply_control(snapshot, ReleaseParticipant(WristbandRole.RED))

    assert released.red_user_id is None
    assert released.blue_user_id == "user-bob"
    assert released.revision == 1


def test_legacy_match_defaults_keep_automatic_rules_disabled() -> None:
    snapshot = MatchControlSnapshot()

    assert snapshot.rule_profile_id is None
    assert snapshot.rack_phase is control.RackPhase.BREAK
    assert snapshot.cue_ball_placement is control.CueBallPlacement.NONE
    assert snapshot.winner_role is None
    assert snapshot.terminal_reason is None


def test_winner_requires_an_ended_rack_and_terminal_reason() -> None:
    with pytest.raises(ValueError, match="ended rack"):
        MatchControlSnapshot(
            rack_phase=control.RackPhase.OPEN,
            winner_role=WristbandRole.RED,
            terminal_reason="legal_eight",
        )

    ended = MatchControlSnapshot(
        rack_phase=control.RackPhase.ENDED,
        winner_role=WristbandRole.BLUE,
        terminal_reason="early_eight",
    )

    assert ended.winner_role is WristbandRole.BLUE


def test_match_settings_freeze_both_players_mode_format_breaker_and_ranking() -> None:
    snapshot = MatchControlSnapshot(red_user_id="user-alice", blue_user_id="user-bob")

    frozen = apply_control(
        snapshot,
        FreezeMatchSettings(
            mode=MatchMode.OBSERVATION,
            match_format=MatchFormat.SINGLE,
            race_target=None,
            first_breaker=WristbandRole.BLUE,
            ranking_intent=True,
            settings_revision=4,
        ),
    )

    assert frozen.mode is MatchMode.OBSERVATION
    assert frozen.match_format is MatchFormat.SINGLE
    assert frozen.breaker is WristbandRole.BLUE
    assert frozen.active_role is WristbandRole.BLUE
    assert frozen.ranking_intent is True
    assert frozen.settings_revision == 4
    with pytest.raises(ValueError, match="frozen"):
        apply_control(
            frozen,
            FreezeMatchSettings(
                mode=MatchMode.ASSISTANCE,
                match_format=MatchFormat.SINGLE,
                race_target=None,
                first_breaker=WristbandRole.RED,
                ranking_intent=False,
                settings_revision=5,
            ),
        )


def test_assistance_settings_cannot_request_ranking() -> None:
    with pytest.raises(ValueError, match="ranking"):
        FreezeMatchSettings(
            mode=MatchMode.ASSISTANCE,
            match_format=MatchFormat.SINGLE,
            race_target=None,
            first_breaker=WristbandRole.RED,
            ranking_intent=True,
            settings_revision=1,
        )
