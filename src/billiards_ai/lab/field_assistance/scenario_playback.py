"""Authenticated command provenance for one managed scenario's raw playback latch."""

from typing import Protocol

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
)
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.records import FieldSessionRecord, FieldSessionStatus


class PlaybackGate(Protocol):
    def release(self) -> None: ...

    def is_released(self) -> bool: ...


class AuthenticatedScenarioPlayback:
    """Observe only the real worker's persisted applications under its match lock.

    Claims belong to the verified startup scene. Admin mutations invalidate the
    unreleased provenance, including edits away from and back to the same user.
    Old APPLIED retries cannot recertify a claim at a later product revision.
    """

    def __init__(self, session_id: str, gate: PlaybackGate) -> None:
        self._session_id = session_id
        self._gate = gate
        self._claims: dict[WristbandRole, str] = {}

    def invalidate(self) -> None:
        self._claims.clear()

    def observe(
        self,
        envelope: FieldCommandEnvelope,
        application: FieldCommandApplication,
        session: FieldSessionRecord,
    ) -> None:
        if (
            self._gate.is_released()
            or envelope.session_id != self._session_id
            or session.session_id != self._session_id
            or session.status is not FieldSessionStatus.OPEN
            or application.command_id != envelope.command_id
            or application.outcome != "applied"
            or application.result_code != "APPLIED"
            or application.resulting_revision != session.match.revision
        ):
            return
        match = session.match
        slots = {WristbandRole.RED: match.red_user_id, WristbandRole.BLUE: match.blue_user_id}
        self._claims = {role: actor for role, actor in self._claims.items() if slots[role] == actor}
        if envelope.command_type == "claim_role":
            role = WristbandRole(envelope.payload["role"])
            if slots[role] == envelope.actor_user_id:
                self._claims[role] = envelope.actor_user_id
        if envelope.command_type != "set_breaker":
            return
        # Only the full successful product settings application can release.
        # The legacy breaker command and arbitrary matching admin state cannot.
        if dict(envelope.payload) != {
            "role": match.breaker,
            "mode": match.mode,
            "format": match.match_format,
            "race_target": match.race_target,
            "ranking_intent": match.ranking_intent,
            "settings_revision": match.settings_revision,
        }:
            return
        if (
            match.settings_frozen
            and match.settings_revision > 0
            and match.breaker in slots
            and match.active_role is match.breaker
            and match.red_user_id
            and match.blue_user_id
            and match.red_user_id != match.blue_user_id
            and self._claims == slots
            and envelope.actor_user_id in slots.values()
        ):
            self._gate.release()
