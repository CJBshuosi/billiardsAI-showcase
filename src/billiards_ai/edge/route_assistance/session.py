"""Pure deterministic reducer for one local route-assistance session."""

from __future__ import annotations

from dataclasses import dataclass, replace

from billiards_ai.edge.route_assistance.models import (
    AssistanceEffect,
    AssistanceEffectKind,
    AssistancePhase,
    AssistanceSnapshot,
    GuidanceDisplay,
    MotionState,
)


@dataclass(frozen=True, slots=True)
class AcceptMotion:
    motion: MotionState
    scene_generation: int

    def __post_init__(self) -> None:
        if self.scene_generation < 0:
            raise ValueError("scene generation must be non-negative")


@dataclass(frozen=True, slots=True)
class AcceptGuidance:
    guidance: GuidanceDisplay | None
    abstained_reason: str | None = None

    def __post_init__(self) -> None:
        if self.guidance is None and not self.abstained_reason:
            raise ValueError("abstained guidance requires a reason")
        if self.guidance is not None and self.abstained_reason is not None:
            raise ValueError("visible guidance cannot carry an abstention reason")


@dataclass(frozen=True, slots=True)
class DegradeSession:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("degradation reason must not be empty")


SessionCommand = AcceptMotion | AcceptGuidance | DegradeSession


@dataclass(frozen=True, slots=True)
class TransitionResult:
    snapshot: AssistanceSnapshot
    effects: tuple[AssistanceEffect, ...] = ()


def _effect(
    kind: AssistanceEffectKind,
    snapshot: AssistanceSnapshot,
    *,
    scene_generation: int | None = None,
    shot_sequence: int | None = None,
) -> AssistanceEffect:
    return AssistanceEffect(
        kind=kind,
        scene_generation=(
            snapshot.scene_generation if scene_generation is None else scene_generation
        ),
        shot_sequence=snapshot.shot_sequence if shot_sequence is None else shot_sequence,
    )


def _accept_motion(
    snapshot: AssistanceSnapshot,
    command: AcceptMotion,
) -> TransitionResult:
    if snapshot.phase is AssistancePhase.DEGRADED:
        raise ValueError("degraded session cannot accept motion")

    if command.motion is MotionState.MOVING:
        if command.scene_generation != snapshot.scene_generation:
            raise ValueError("moving scene generation does not match current scene generation")
        if snapshot.motion is MotionState.MOVING:
            return TransitionResult(snapshot)
        effects = []
        if snapshot.guidance is not None:
            effects.append(_effect(AssistanceEffectKind.CLEAR_GUIDANCE, snapshot))
        effects.append(_effect(AssistanceEffectKind.SHOT_STARTED, snapshot))
        return TransitionResult(
            replace(
                snapshot,
                phase=AssistancePhase.BALLS_MOVING,
                motion=MotionState.MOVING,
                guidance=None,
                degraded_reason=None,
            ),
            tuple(effects),
        )

    if command.motion is MotionState.STABLE:
        expected_generation = snapshot.scene_generation
        if snapshot.motion is MotionState.MOVING:
            expected_generation += 1
        if command.scene_generation != expected_generation:
            raise ValueError("stable scene generation does not match expected scene generation")
        if snapshot.motion is MotionState.STABLE:
            return TransitionResult(snapshot)
        if snapshot.motion is MotionState.MOVING:
            next_shot = snapshot.shot_sequence + 1
            updated = replace(
                snapshot,
                phase=AssistancePhase.ANALYZING_TABLE,
                motion=MotionState.STABLE,
                scene_generation=command.scene_generation,
                shot_sequence=next_shot,
                guidance=None,
                degraded_reason=None,
            )
            return TransitionResult(
                updated,
                (
                    _effect(
                        AssistanceEffectKind.SHOT_SETTLED,
                        snapshot,
                        scene_generation=command.scene_generation,
                        shot_sequence=next_shot,
                    ),
                    _effect(
                        AssistanceEffectKind.REQUEST_ANALYSIS,
                        snapshot,
                        scene_generation=command.scene_generation,
                        shot_sequence=next_shot,
                    ),
                ),
            )
        updated = replace(
            snapshot,
            phase=AssistancePhase.ANALYZING_TABLE,
            motion=MotionState.STABLE,
            degraded_reason=None,
        )
        return TransitionResult(
            updated,
            (_effect(AssistanceEffectKind.REQUEST_ANALYSIS, updated),),
        )

    if command.scene_generation != snapshot.scene_generation:
        raise ValueError("unknown motion scene generation does not match current scene generation")
    return TransitionResult(replace(snapshot, motion=MotionState.UNKNOWN))


def _accept_guidance(
    snapshot: AssistanceSnapshot,
    command: AcceptGuidance,
) -> TransitionResult:
    if snapshot.phase is not AssistancePhase.ANALYZING_TABLE:
        raise ValueError("guidance requires ANALYZING_TABLE phase")
    if command.guidance is None:
        return TransitionResult(
            replace(
                snapshot,
                guidance=None,
                degraded_reason=command.abstained_reason,
            )
        )
    if command.guidance.scene_generation != snapshot.scene_generation:
        raise ValueError("stale guidance scene generation")
    updated = replace(
        snapshot,
        phase=AssistancePhase.WAITING_FOR_SHOT,
        guidance=command.guidance,
        degraded_reason=None,
    )
    return TransitionResult(
        updated,
        (_effect(AssistanceEffectKind.SHOW_GUIDANCE, updated),),
    )


def _degrade(snapshot: AssistanceSnapshot, command: DegradeSession) -> TransitionResult:
    effects = (
        (_effect(AssistanceEffectKind.CLEAR_GUIDANCE, snapshot),)
        if snapshot.guidance is not None
        else ()
    )
    return TransitionResult(
        replace(
            snapshot,
            phase=AssistancePhase.DEGRADED,
            guidance=None,
            degraded_reason=command.reason,
        ),
        effects,
    )


def transition(snapshot: AssistanceSnapshot, command: SessionCommand) -> TransitionResult:
    """Apply one validated command without I/O or wall-clock reads."""

    if type(command) is AcceptMotion:
        return _accept_motion(snapshot, command)
    if type(command) is AcceptGuidance:
        return _accept_guidance(snapshot, command)
    if type(command) is DegradeSession:
        return _degrade(snapshot, command)
    raise TypeError("unsupported route-assistance command")
