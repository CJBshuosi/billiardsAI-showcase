"""Evidence-first observation of the cue ball's first object-ball contact."""

from __future__ import annotations

import math
from collections.abc import Sequence

from billiards_ai.lab.events.models import (
    BallState,
    EventStatus,
    EvidenceEvent,
    TrackedFrame,
)


def _track_key(ball: BallState) -> str:
    return ball.track_id or f"identity:{ball.identity}"


def _target_identity(ball: BallState) -> tuple[str, str | None]:
    if ball.identity == "cue":
        return f"unknown:{_track_key(ball)}", "duplicate_cue_label_downgraded_to_target"
    if ball.identity == "unknown":
        return f"unknown:{_track_key(ball)}", None
    return ball.identity, None


def _contact_event(
    *,
    status: EventStatus,
    timestamp_ms: int,
    actor: BallState,
    target: BallState,
    distance: float,
    contact_distance_mm: float,
    evidence_frame_ids: tuple[int, ...],
    confidence_scale: float,
    reason: str | None,
) -> EvidenceEvent:
    target_identity, downgrade_reason = _target_identity(target)
    confidence = (
        min(actor.confidence, target.confidence)
        * max(0.0, 1 - distance / max(contact_distance_mm * 2, 1))
        * confidence_scale
    )
    return EvidenceEvent(
        event_type="first_contact",
        status=status,
        timestamp_ms=timestamp_ms,
        actor_identity="cue",
        target_identity=target_identity,
        evidence_frame_ids=evidence_frame_ids,
        confidence=min(confidence, 1.0),
        reason=reason or downgrade_reason,
    )


def observe_first_contact(
    frames: Sequence[TrackedFrame],
    *,
    contact_distance_mm: float,
    ambiguity_mm: float = 3,
    cue_motion_min_mm: float = 5,
    bridge_frames: int = 3,
) -> EvidenceEvent:
    if len(frames) < 2:
        raise ValueError("first-contact observation requires at least two frames")
    if bridge_frames < 0:
        raise ValueError("bridge_frames must not be negative")
    cue_motion_samples: list[tuple[int, TrackedFrame, TrackedFrame, BallState, float]] = []
    for frame_index, (previous, current) in enumerate(
        zip(frames, frames[1:], strict=False), start=1
    ):
        before = {_track_key(ball): ball for ball in previous.balls}
        for actor in current.balls:
            actor_key = _track_key(actor)
            prior = before.get(actor_key)
            if actor.identity != "cue" or prior is None:
                continue
            cue_motion = math.hypot(actor.x_mm - prior.x_mm, actor.y_mm - prior.y_mm)
            if cue_motion < cue_motion_min_mm:
                continue
            cue_motion_samples.append((frame_index, previous, current, actor, cue_motion))

    moving_cue_samples: list[tuple[int, TrackedFrame, TrackedFrame, BallState]] = []
    if cue_motion_samples:
        first_moving = min(
            cue_motion_samples,
            key=lambda sample: (
                sample[2].timestamp_ms,
                -sample[4],
                _track_key(sample[3]),
            ),
        )
        actor_track_key = _track_key(first_moving[3])
        moving_cue_samples = [
            (frame_index, previous, current, actor)
            for frame_index, previous, current, actor, _ in cue_motion_samples
            if _track_key(actor) == actor_track_key
        ]

    for _, previous, current, actor in moving_cue_samples:
        actor_key = _track_key(actor)
        distances = sorted(
            (
                math.hypot(ball.x_mm - actor.x_mm, ball.y_mm - actor.y_mm),
                _track_key(ball),
                ball,
            )
            for ball in current.balls
            if _track_key(ball) != actor_key
        )
        contacts = [candidate for candidate in distances if candidate[0] <= contact_distance_mm]
        if not contacts:
            continue
        evidence = (previous.frame_id, current.frame_id)
        if len(contacts) > 1 and contacts[1][0] - contacts[0][0] <= ambiguity_mm:
            return EvidenceEvent(
                event_type="first_contact",
                status=EventStatus.ABSTAINED,
                timestamp_ms=current.timestamp_ms,
                actor_identity="cue",
                target_identity=None,
                evidence_frame_ids=evidence,
                confidence=0.25,
                reason="simultaneous_contact_ambiguous",
            )
        distance, _, target = contacts[0]
        return _contact_event(
            status=EventStatus.OBSERVED,
            timestamp_ms=current.timestamp_ms,
            actor=actor,
            target=target,
            distance=distance,
            contact_distance_mm=contact_distance_mm,
            evidence_frame_ids=evidence,
            confidence_scale=1.0,
            reason=None,
        )

    # During a collision or hand occlusion the detector can alternate between the
    # moving cue ball and its target.  Bridge only a bounded frame gap and publish
    # the result as inferred evidence, never as a directly observed contact.
    inferred_candidates: list[
        tuple[float, int, int, TrackedFrame, TrackedFrame, BallState, BallState]
    ] = []
    for frame_index, previous, _current, actor in moving_cue_samples:
        actor_key = _track_key(actor)
        lower = max(0, frame_index - bridge_frames)
        upper = min(len(frames), frame_index + bridge_frames + 1)
        for target_index in range(lower, upper):
            if target_index == frame_index:
                continue
            target_frame = frames[target_index]
            for target in target_frame.balls:
                if _track_key(target) == actor_key:
                    continue
                distance = math.hypot(target.x_mm - actor.x_mm, target.y_mm - actor.y_mm)
                if distance <= contact_distance_mm:
                    inferred_candidates.append(
                        (
                            distance,
                            abs(target_index - frame_index),
                            frame_index,
                            previous,
                            target_frame,
                            actor,
                            target,
                        )
                    )
    inferred_candidates.sort(key=lambda item: (item[2], item[1], item[0], _track_key(item[6])))
    if inferred_candidates:
        inferred_best = inferred_candidates[0]
        inferred_competing = [
            candidate
            for candidate in inferred_candidates[1:]
            if candidate[2] == inferred_best[2]
            and candidate[1] == inferred_best[1]
            and candidate[0] - inferred_best[0] <= ambiguity_mm
            and _track_key(candidate[6]) != _track_key(inferred_best[6])
        ]
        inferred_evidence = tuple(
            sorted(
                {
                    inferred_best[3].frame_id,
                    frames[inferred_best[2]].frame_id,
                    inferred_best[4].frame_id,
                }
            )
        )
        if inferred_competing:
            return EvidenceEvent(
                event_type="first_contact",
                status=EventStatus.ABSTAINED,
                timestamp_ms=max(
                    frames[inferred_best[2]].timestamp_ms,
                    inferred_best[4].timestamp_ms,
                ),
                actor_identity="cue",
                target_identity=None,
                evidence_frame_ids=inferred_evidence,
                confidence=0.2,
                reason="temporally_inferred_contact_ambiguous",
            )
        return _contact_event(
            status=EventStatus.INFERRED,
            timestamp_ms=max(
                frames[inferred_best[2]].timestamp_ms,
                inferred_best[4].timestamp_ms,
            ),
            actor=inferred_best[5],
            target=inferred_best[6],
            distance=inferred_best[0],
            contact_distance_mm=contact_distance_mm,
            evidence_frame_ids=inferred_evidence,
            confidence_scale=0.5,
            reason="temporally_inferred_contact",
        )

    # High-speed detections can fragment the cue ball into a new track on every
    # frame.  Recover only a tightly bounded kinematic transfer: two cue-labelled
    # fragments establish an incoming velocity, a separately tracked target was
    # stationary before the encounter, and that target begins moving where the
    # incoming cue fragment is predicted to arrive on the next frame.
    transfer_candidates: list[
        tuple[
            float,
            int,
            TrackedFrame,
            TrackedFrame,
            TrackedFrame,
            TrackedFrame,
            BallState,
            BallState,
        ]
    ] = []
    history_frames = bridge_frames * 2
    for current_index in range(1, len(frames) - 1):
        previous = frames[current_index - 1]
        current = frames[current_index]
        following = frames[current_index + 1]
        actor_dt = current.timestamp_ms - previous.timestamp_ms
        following_dt = following.timestamp_ms - current.timestamp_ms
        if actor_dt <= 0 or following_dt <= 0:
            continue
        ratio = following_dt / actor_dt
        for target in current.balls:
            target_key = _track_key(target)
            target_after = next(
                (ball for ball in following.balls if _track_key(ball) == target_key),
                None,
            )
            if target_after is None:
                continue
            target_motion = math.hypot(
                target_after.x_mm - target.x_mm,
                target_after.y_mm - target.y_mm,
            )
            if target_motion < cue_motion_min_mm:
                continue
            history_lower = max(0, current_index - history_frames)
            stationary_frame: TrackedFrame | None = None
            stationary_target: BallState | None = None
            for history_index in range(current_index - 1, history_lower - 1, -1):
                candidate = next(
                    (
                        ball
                        for ball in frames[history_index].balls
                        if _track_key(ball) == target_key
                    ),
                    None,
                )
                if candidate is not None:
                    stationary_frame = frames[history_index]
                    stationary_target = candidate
                    break
            if stationary_frame is None or stationary_target is None:
                continue
            target_prior_motion = math.hypot(
                target.x_mm - stationary_target.x_mm,
                target.y_mm - stationary_target.y_mm,
            )
            if target_prior_motion >= cue_motion_min_mm:
                continue
            for actor_before in previous.balls:
                if actor_before.identity != "cue":
                    continue
                for actor in current.balls:
                    if (
                        actor.identity != "cue"
                        or _track_key(actor) == _track_key(actor_before)
                        or _track_key(actor) == target_key
                    ):
                        continue
                    velocity_x = actor.x_mm - actor_before.x_mm
                    velocity_y = actor.y_mm - actor_before.y_mm
                    cue_motion = math.hypot(velocity_x, velocity_y)
                    if cue_motion < cue_motion_min_mm:
                        continue
                    predicted_x = actor.x_mm + velocity_x * ratio
                    predicted_y = actor.y_mm + velocity_y * ratio
                    predicted_distance = math.hypot(
                        target_after.x_mm - predicted_x,
                        target_after.y_mm - predicted_y,
                    )
                    if predicted_distance <= contact_distance_mm:
                        transfer_candidates.append(
                            (
                                predicted_distance,
                                current_index,
                                stationary_frame,
                                previous,
                                current,
                                following,
                                actor,
                                target,
                            )
                        )
    transfer_candidates.sort(key=lambda item: (item[1], item[0], _track_key(item[7])))
    if transfer_candidates:
        transfer_best = transfer_candidates[0]
        transfer_competing = [
            candidate
            for candidate in transfer_candidates[1:]
            if candidate[1] == transfer_best[1]
            and candidate[0] - transfer_best[0] <= ambiguity_mm
            and _track_key(candidate[7]) != _track_key(transfer_best[7])
        ]
        transfer_evidence = tuple(
            sorted(
                {
                    transfer_best[2].frame_id,
                    transfer_best[3].frame_id,
                    transfer_best[4].frame_id,
                    transfer_best[5].frame_id,
                }
            )
        )
        if transfer_competing:
            return EvidenceEvent(
                event_type="first_contact",
                status=EventStatus.ABSTAINED,
                timestamp_ms=transfer_best[5].timestamp_ms,
                actor_identity="cue",
                target_identity=None,
                evidence_frame_ids=transfer_evidence,
                confidence=0.15,
                reason="kinematic_contact_transfer_ambiguous",
            )
        return _contact_event(
            status=EventStatus.INFERRED,
            timestamp_ms=transfer_best[5].timestamp_ms,
            actor=transfer_best[6],
            target=transfer_best[7],
            distance=transfer_best[0],
            contact_distance_mm=contact_distance_mm,
            evidence_frame_ids=transfer_evidence,
            confidence_scale=0.4,
            reason="kinematic_contact_transfer",
        )
    reason = "no_contact_observed" if moving_cue_samples else "cue_ball_not_moving"
    return EvidenceEvent(
        event_type="first_contact",
        status=EventStatus.ABSTAINED,
        timestamp_ms=frames[-1].timestamp_ms,
        actor_identity="cue",
        target_identity=None,
        evidence_frame_ids=(frames[0].frame_id, frames[-1].frame_id),
        confidence=0.0,
        reason=reason,
    )
