from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    MatchFormat,
    MatchMode,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID


def _envelope(command_type: str, payload: dict[str, object]) -> FieldCommandEnvelope:
    return FieldCommandEnvelope(
        command_id="command-1",
        session_id="session-1",
        actor_user_id="user-alice",
        command_type=command_type,
        payload=payload,
        expected_revision=0,
        server_sequence=1,
    )


def test_claim_role_uses_authenticated_actor_identity() -> None:
    reduced = reduce_field_command(
        MatchControlSnapshot(),
        _envelope("claim_role", {"role": "red"}),
        has_unsettled_shot=False,
    )

    assert reduced.match.red_user_id == "user-alice"
    assert reduced.match.revision == 1


def test_group_correction_proposal_is_recorded_without_mutating_facts() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice",
        blue_user_id="user-bob",
        red_group=BallGroupAssignment.STRIPE,
        blue_group=BallGroupAssignment.SOLID,
        revision=4,
    )
    reduced = reduce_field_command(
        snapshot,
        _envelope(
            "assign_groups",
            {"red_group": "solid", "correction_phase": "proposal"},
        ),
        has_unsettled_shot=False,
    )

    assert reduced.match == snapshot
    assert reduced.result_code == "GROUP_CORRECTION_PROPOSED"


def test_turn_or_end_command_rejects_while_a_shot_is_unsettled() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice", active_role=WristbandRole.RED, revision=1
    )

    with pytest.raises(ValueError, match="SHOT_UNSETTLED"):
        reduce_field_command(
            snapshot,
            _envelope("switch_turn", {}),
            has_unsettled_shot=True,
        )


@pytest.mark.parametrize(
    ("command_type", "payload"),
    [
        ("assign_groups", {"red_group": "solid"}),
        ("continue_turn", {}),
        ("switch_turn", {}),
    ],
)
def test_rules_managed_session_rejects_phone_rule_bypass(
    command_type: str,
    payload: dict[str, object],
) -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice",
        blue_user_id="user-bob",
        active_role=WristbandRole.RED,
        rule_profile_id=PROFILE_ID,
    )

    with pytest.raises(ValueError, match="COMMAND_FORBIDDEN"):
        reduce_field_command(
            snapshot,
            _envelope(command_type, payload),
            has_unsettled_shot=False,
        )


def test_rules_managed_session_still_allows_player_abort() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice",
        blue_user_id="user-bob",
        active_role=WristbandRole.RED,
        rule_profile_id=PROFILE_ID,
    )

    reduced = reduce_field_command(
        snapshot,
        _envelope("end_session", {}),
        has_unsettled_shot=False,
    )

    assert reduced.end_session is True
    assert reduced.match == snapshot


def test_non_active_joined_player_can_end_session() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-bob",
        blue_user_id="user-alice",
        active_role=WristbandRole.RED,
        rule_profile_id=PROFILE_ID,
    )

    reduced = reduce_field_command(
        snapshot,
        _envelope("end_session", {}),
        has_unsettled_shot=False,
    )

    assert reduced.end_session is True
    assert reduced.match == snapshot


def test_product_breaker_command_freezes_confirmed_match_settings_once() -> None:
    snapshot = MatchControlSnapshot(
        red_user_id="user-alice",
        blue_user_id="user-bob",
    )
    envelope = _envelope(
        "set_breaker",
        {
            "role": "red",
            "mode": "observation",
            "format": "single",
            "race_target": None,
            "ranking_intent": True,
            "settings_revision": 7,
        },
    )

    reduced = reduce_field_command(snapshot, envelope, has_unsettled_shot=False)

    assert reduced.match.mode is MatchMode.OBSERVATION
    assert reduced.match.match_format is MatchFormat.SINGLE
    assert reduced.match.breaker is WristbandRole.RED
    assert reduced.match.ranking_intent is True
    with pytest.raises(ValueError, match="COMMAND_FORBIDDEN"):
        reduce_field_command(reduced.match, envelope, has_unsettled_shot=False)


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"role": "red"}, None),
        ({}, "COMMAND_INVALID"),
        ({"role": "blue"}, "COMMAND_FORBIDDEN"),
        ({"role": "red", "extra": 1}, "COMMAND_INVALID"),
    ],
)
def test_managed_release_is_pinned_to_actual_actor_role(payload, code):
    from dataclasses import replace

    command = replace(
        _envelope("release_role", payload),
        session_id="managed-22222222-2222-4222-8222-222222222222",
    )
    snapshot = MatchControlSnapshot(red_user_id="user-alice")
    if code:
        with pytest.raises(ValueError, match=code):
            reduce_field_command(snapshot, command, has_unsettled_shot=False)
    else:
        reduced = reduce_field_command(snapshot, command, has_unsettled_shot=False)
        assert reduced.match.red_user_id is None


def test_legacy_release_keeps_empty_payload():
    assert (
        reduce_field_command(
            MatchControlSnapshot(red_user_id="user-alice"),
            _envelope("release_role", {}),
            has_unsettled_shot=False,
        ).match.red_user_id
        is None
    )
