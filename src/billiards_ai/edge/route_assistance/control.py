"""Pure tablet controls for player, wristband, group, and turn state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class WristbandRole(StrEnum):
    RED = "red"
    BLUE = "blue"


class BallGroupAssignment(StrEnum):
    SOLID = "solid"
    STRIPE = "stripe"


class RackPhase(StrEnum):
    BREAK = "break"
    OPEN = "open"
    ASSIGNED = "assigned"
    ON_EIGHT = "on_eight"
    ENDED = "ended"


class CueBallPlacement(StrEnum):
    NONE = "none"
    ANYWHERE = "anywhere"
    BEHIND_LINE = "behind_line"


class MatchMode(StrEnum):
    OBSERVATION = "observation"
    ASSISTANCE = "assistance"


class MatchFormat(StrEnum):
    SINGLE = "single"
    RACE = "race"


@dataclass(frozen=True, slots=True)
class MatchControlSnapshot:
    red_user_id: str | None = None
    blue_user_id: str | None = None
    breaker: WristbandRole | None = None
    active_role: WristbandRole | None = None
    red_group: BallGroupAssignment | None = None
    blue_group: BallGroupAssignment | None = None
    revision: int = 0
    rule_profile_id: str | None = None
    rack_phase: RackPhase = RackPhase.BREAK
    cue_ball_placement: CueBallPlacement = CueBallPlacement.NONE
    winner_role: WristbandRole | None = None
    terminal_reason: str | None = None
    mode: MatchMode | None = None
    match_format: MatchFormat | None = None
    race_target: int | None = None
    ranking_intent: bool = False
    settings_revision: int = 0

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("control revision must be non-negative")
        if (
            self.red_user_id is not None
            and self.blue_user_id is not None
            and self.red_user_id == self.blue_user_id
        ):
            raise ValueError("one user cannot occupy both wristbands")
        if (self.red_group is None) != (self.blue_group is None):
            raise ValueError("ball group assignment must be complete")
        if self.red_group is not None and self.red_group is self.blue_group:
            raise ValueError("ball group assignment must be bijective")
        if self.winner_role is not None and (
            self.rack_phase is not RackPhase.ENDED or not self.terminal_reason
        ):
            raise ValueError("winner requires an ended rack and terminal reason")
        if self.rack_phase is RackPhase.ENDED and (
            self.winner_role is None or not self.terminal_reason
        ):
            raise ValueError("ended rack requires a winner and terminal reason")
        if self.mode is None:
            if (
                self.match_format is not None
                or self.race_target is not None
                or self.ranking_intent
                or self.settings_revision != 0
            ):
                raise ValueError("match settings must be frozen together")
        else:
            if self.match_format is None or self.settings_revision <= 0:
                raise ValueError("frozen match settings require format and revision")
            if self.mode is MatchMode.ASSISTANCE and self.ranking_intent:
                raise ValueError("assistance mode cannot request ranking")
            if self.match_format is MatchFormat.SINGLE and self.race_target is not None:
                raise ValueError("single format cannot have a race target")
            if self.match_format is MatchFormat.RACE and (
                self.race_target is None or not 2 <= self.race_target <= 99
            ):
                raise ValueError("race format requires a target between 2 and 99")

    @property
    def settings_frozen(self) -> bool:
        return self.mode is not None


@dataclass(frozen=True, slots=True)
class SetParticipant:
    role: WristbandRole
    user_id: str

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("participant user ID must not be empty")


@dataclass(frozen=True, slots=True)
class ReleaseParticipant:
    role: WristbandRole


@dataclass(frozen=True, slots=True)
class SetBreaker:
    role: WristbandRole


@dataclass(frozen=True, slots=True)
class FreezeMatchSettings:
    mode: MatchMode
    match_format: MatchFormat
    race_target: int | None
    first_breaker: WristbandRole
    ranking_intent: bool
    settings_revision: int

    def __post_init__(self) -> None:
        if self.settings_revision <= 0:
            raise ValueError("match settings revision must be positive")
        if self.mode is MatchMode.ASSISTANCE and self.ranking_intent:
            raise ValueError("assistance mode cannot request ranking")
        if self.match_format is MatchFormat.SINGLE and self.race_target is not None:
            raise ValueError("single format cannot have a race target")
        if self.match_format is MatchFormat.RACE and (
            self.race_target is None or not 2 <= self.race_target <= 99
        ):
            raise ValueError("race format requires a target between 2 and 99")


@dataclass(frozen=True, slots=True)
class AssignGroups:
    red_group: BallGroupAssignment


@dataclass(frozen=True, slots=True)
class ContinueTurn:
    pass


@dataclass(frozen=True, slots=True)
class SwitchTurn:
    pass


ControlCommand = (
    SetParticipant
    | ReleaseParticipant
    | SetBreaker
    | FreezeMatchSettings
    | AssignGroups
    | ContinueTurn
    | SwitchTurn
)


def _next(snapshot: MatchControlSnapshot, **changes: object) -> MatchControlSnapshot:
    return replace(snapshot, revision=snapshot.revision + 1, **changes)


def apply_control(
    snapshot: MatchControlSnapshot,
    command: ControlCommand,
) -> MatchControlSnapshot:
    """Apply one validated tablet command without mixing identity and game role."""

    if type(command) is SetParticipant:
        if command.role is WristbandRole.RED:
            return _next(snapshot, red_user_id=command.user_id.strip())
        return _next(snapshot, blue_user_id=command.user_id.strip())
    if type(command) is ReleaseParticipant:
        if command.role is WristbandRole.RED:
            return _next(snapshot, red_user_id=None)
        return _next(snapshot, blue_user_id=None)
    if type(command) is SetBreaker:
        if snapshot.settings_frozen:
            raise ValueError("match settings are frozen")
        return _next(snapshot, breaker=command.role, active_role=command.role)
    if type(command) is FreezeMatchSettings:
        if snapshot.settings_frozen:
            raise ValueError("match settings are frozen")
        if snapshot.red_user_id is None or snapshot.blue_user_id is None:
            raise ValueError("match settings require two participants")
        return _next(
            snapshot,
            breaker=command.first_breaker,
            active_role=command.first_breaker,
            mode=command.mode,
            match_format=command.match_format,
            race_target=command.race_target,
            ranking_intent=command.ranking_intent,
            settings_revision=command.settings_revision,
        )
    if type(command) is AssignGroups:
        blue_group = (
            BallGroupAssignment.STRIPE
            if command.red_group is BallGroupAssignment.SOLID
            else BallGroupAssignment.SOLID
        )
        return _next(snapshot, red_group=command.red_group, blue_group=blue_group)
    if type(command) is ContinueTurn:
        if snapshot.active_role is None:
            raise ValueError("continue requires an active turn")
        return _next(snapshot)
    if type(command) is SwitchTurn:
        if snapshot.active_role is None:
            raise ValueError("switch requires an active turn")
        next_role = (
            WristbandRole.BLUE if snapshot.active_role is WristbandRole.RED else WristbandRole.RED
        )
        return _next(snapshot, active_role=next_role)
    raise TypeError("unsupported match control command")
