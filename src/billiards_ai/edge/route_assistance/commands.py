"""Validated phone command envelopes and pure local reductions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

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

_COMMAND_TYPES = frozenset(
    {
        "claim_role",
        "release_role",
        "set_breaker",
        "assign_groups",
        "continue_turn",
        "switch_turn",
        "end_session",
    }
)


def _nonempty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128:
        raise ValueError(f"{label} is invalid")
    return normalized


@dataclass(frozen=True, slots=True)
class FieldCommandEnvelope:
    command_id: str
    session_id: str
    actor_user_id: str
    command_type: str
    payload: Mapping[str, object]
    expected_revision: int
    server_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_id", _nonempty(self.command_id, "command ID"))
        object.__setattr__(self, "session_id", _nonempty(self.session_id, "session ID"))
        object.__setattr__(self, "actor_user_id", _nonempty(self.actor_user_id, "actor user ID"))
        if self.command_type not in _COMMAND_TYPES:
            raise ValueError("command type is invalid")
        if self.expected_revision < 0:
            raise ValueError("expected revision must be non-negative")
        if self.server_sequence <= 0:
            raise ValueError("server sequence must be positive")
        if not isinstance(self.payload, Mapping):
            raise ValueError("command payload must be an object")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def payload_digest(self) -> str:
        intent = {
            "actor_user_id": self.actor_user_id,
            "command_type": self.command_type,
            "expected_revision": self.expected_revision,
            "payload": dict(self.payload),
            "server_sequence": self.server_sequence,
            "session_id": self.session_id,
        }
        encoded = json.dumps(
            intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldCommandReduction:
    match: MatchControlSnapshot
    result_code: str = "APPLIED"
    end_session: bool = False


@dataclass(frozen=True, slots=True)
class FieldCommandApplication:
    command_id: str
    outcome: str
    result_code: str
    resulting_revision: int

    def __post_init__(self) -> None:
        if self.outcome not in {"applied", "rejected"}:
            raise ValueError("command outcome is invalid")


def _exact_payload(envelope: FieldCommandEnvelope, allowed: set[str]) -> dict[str, object]:
    payload = dict(envelope.payload)
    if set(payload) != allowed:
        raise ValueError("COMMAND_INVALID")
    return payload


def _actor_role(snapshot: MatchControlSnapshot, actor_user_id: str) -> WristbandRole | None:
    if snapshot.red_user_id == actor_user_id:
        return WristbandRole.RED
    if snapshot.blue_user_id == actor_user_id:
        return WristbandRole.BLUE
    return None


def _role(value: object) -> WristbandRole:
    try:
        return WristbandRole(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("COMMAND_INVALID") from exc


def _group(value: object) -> BallGroupAssignment:
    try:
        return BallGroupAssignment(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("COMMAND_INVALID") from exc


def _match_mode(value: object) -> MatchMode:
    try:
        return MatchMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("COMMAND_INVALID") from exc


def _match_format(value: object) -> MatchFormat:
    try:
        return MatchFormat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("COMMAND_INVALID") from exc


def reduce_field_command(
    snapshot: MatchControlSnapshot,
    envelope: FieldCommandEnvelope,
    *,
    has_unsettled_shot: bool,
    has_started: bool = False,
) -> FieldCommandReduction:
    """Map one authenticated cloud intent onto the existing local controls."""

    actor_role = _actor_role(snapshot, envelope.actor_user_id)
    command_type = envelope.command_type
    if command_type == "claim_role":
        payload = _exact_payload(envelope, {"role"})
        role = _role(payload["role"])
        if has_started or snapshot.settings_frozen or actor_role is not None:
            raise ValueError("COMMAND_FORBIDDEN")
        occupied = snapshot.red_user_id if role is WristbandRole.RED else snapshot.blue_user_id
        if occupied is not None:
            raise ValueError("COMMAND_FORBIDDEN")
        match = apply_control(snapshot, SetParticipant(role, envelope.actor_user_id))
        return FieldCommandReduction(match)
    if actor_role is None:
        raise ValueError("COMMAND_FORBIDDEN")
    if command_type == "release_role":
        if envelope.session_id.startswith("managed-"):
            payload = _exact_payload(envelope, {"role"})
            if _role(payload["role"]) is not actor_role:
                raise ValueError("COMMAND_FORBIDDEN")
        else:
            _exact_payload(envelope, set())
        if has_started or snapshot.settings_frozen:
            raise ValueError("COMMAND_FORBIDDEN")
        return FieldCommandReduction(apply_control(snapshot, ReleaseParticipant(actor_role)))
    if command_type == "set_breaker":
        payload = dict(envelope.payload)
        product_keys = {
            "role",
            "mode",
            "format",
            "race_target",
            "ranking_intent",
            "settings_revision",
        }
        if set(payload) not in ({"role"}, product_keys):
            raise ValueError("COMMAND_INVALID")
        if (
            has_started
            or snapshot.settings_frozen
            or not snapshot.red_user_id
            or not snapshot.blue_user_id
        ):
            raise ValueError("COMMAND_FORBIDDEN")
        role = _role(payload["role"])
        if set(payload) == {"role"}:
            return FieldCommandReduction(apply_control(snapshot, SetBreaker(role)))
        ranking_intent = payload["ranking_intent"]
        settings_revision = payload["settings_revision"]
        race_target = payload["race_target"]
        if (
            not isinstance(ranking_intent, bool)
            or not isinstance(settings_revision, int)
            or isinstance(settings_revision, bool)
            or (
                race_target is not None
                and (not isinstance(race_target, int) or isinstance(race_target, bool))
            )
        ):
            raise ValueError("COMMAND_INVALID")
        try:
            command = FreezeMatchSettings(
                mode=_match_mode(payload["mode"]),
                match_format=_match_format(payload["format"]),
                race_target=race_target,
                first_breaker=role,
                ranking_intent=ranking_intent,
                settings_revision=settings_revision,
            )
            return FieldCommandReduction(apply_control(snapshot, command))
        except ValueError as exc:
            if str(exc) == "COMMAND_INVALID":
                raise
            raise ValueError("COMMAND_INVALID") from exc
    if snapshot.rule_profile_id is not None and command_type in {
        "assign_groups",
        "continue_turn",
        "switch_turn",
    }:
        # Once a rules profile owns the rack, phone commands cannot race or
        # override its atomic turn/group decisions.  Players may still end the
        # session as an explicit abort.
        raise ValueError("COMMAND_FORBIDDEN")
    if command_type == "assign_groups":
        payload = dict(envelope.payload)
        allowed = {"red_group"}
        correction_phase = payload.get("correction_phase")
        if correction_phase is not None:
            allowed.add("correction_phase")
        if "proposal_command_id" in payload:
            allowed.add("proposal_command_id")
        if set(payload) != allowed or not snapshot.red_user_id or not snapshot.blue_user_id:
            raise ValueError("COMMAND_INVALID")
        red_group = _group(payload["red_group"])
        if correction_phase == "proposal":
            if "proposal_command_id" in payload:
                raise ValueError("COMMAND_INVALID")
            return FieldCommandReduction(snapshot, "GROUP_CORRECTION_PROPOSED")
        if correction_phase == "confirmation":
            proposal_id = payload.get("proposal_command_id")
            if not isinstance(proposal_id, str) or not proposal_id.strip():
                raise ValueError("COMMAND_INVALID")
        elif correction_phase is not None:
            raise ValueError("COMMAND_INVALID")
        return FieldCommandReduction(apply_control(snapshot, AssignGroups(red_group)))
    if command_type == "end_session":
        _exact_payload(envelope, set())
        if has_unsettled_shot:
            raise ValueError("SHOT_UNSETTLED")
        return FieldCommandReduction(snapshot, end_session=True)
    if command_type in {"continue_turn", "switch_turn"}:
        _exact_payload(envelope, set())
        if has_unsettled_shot:
            raise ValueError("SHOT_UNSETTLED")
        if snapshot.active_role is None or actor_role is not snapshot.active_role:
            raise ValueError("COMMAND_FORBIDDEN")
        if command_type == "continue_turn":
            return FieldCommandReduction(apply_control(snapshot, ContinueTurn()))
        return FieldCommandReduction(apply_control(snapshot, SwitchTurn()))
    raise ValueError("COMMAND_INVALID")
