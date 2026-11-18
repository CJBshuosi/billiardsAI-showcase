"""Evidence-first event observers for one shot-local result."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from billiards_ai.lab.shot_local.models import (
    CueSample,
    LocalState,
    MotionFragment,
    ObservationFrame,
    ShotAnchor,
    ShotLocalEvent,
    ShotWindow,
)


@dataclass(frozen=True, slots=True)
class ShotLocalEventConfig:
    first_contact_distance_mm: float
    first_contact_ambiguity_mm: float
    cue_motion_min_mm: float
    contact_time_tolerance_ms: int
    cushion_distance_mm: float
    pocket_distance_mm: float
    pot_missing_frames: int
    after_match_distance_mm: float
    table_length_mm: int
    table_width_mm: int
    cushion_normal_motion_min_mm: float = 3
    cushion_max_gap_frames: int = 3
    cushion_sparse_extra_mm: float = 40
    cushion_cooldown_frames: int = 6
    collision_distance_mm: float = 55
    collision_motion_min_mm: float = 8
    collision_max_gap_frames: int = 3
    collision_ambiguity_mm: float = 5
    collision_chain_confidence_min: float = 0.7

    def __post_init__(self) -> None:
        distances = (
            self.first_contact_distance_mm,
            self.cue_motion_min_mm,
            self.cushion_distance_mm,
            self.pocket_distance_mm,
            self.after_match_distance_mm,
        )
        if any(value <= 0 for value in distances):
            raise ValueError("event distances must be positive")
        if self.first_contact_ambiguity_mm < 0:
            raise ValueError("contact ambiguity must be non-negative")
        if self.contact_time_tolerance_ms < 0 or self.pot_missing_frames <= 0:
            raise ValueError("event time thresholds are invalid")
        if self.table_length_mm <= 0 or self.table_width_mm <= 0:
            raise ValueError("table dimensions must be positive")
        if self.cushion_normal_motion_min_mm <= 0 or self.cushion_sparse_extra_mm < 0:
            raise ValueError("cushion motion thresholds are invalid")
        if self.cushion_max_gap_frames <= 0 or self.cushion_cooldown_frames < 0:
            raise ValueError("cushion frame thresholds are invalid")
        if self.collision_distance_mm <= 0 or self.collision_motion_min_mm <= 0:
            raise ValueError("collision distances must be positive")
        if self.collision_max_gap_frames <= 0 or self.collision_ambiguity_mm < 0:
            raise ValueError("collision thresholds are invalid")
        if not 0 <= self.collision_chain_confidence_min <= 1:
            raise ValueError("collision chain confidence must be between zero and one")


def _point_distance(first_x: int, first_y: int, second_x: int, second_y: int) -> float:
    return math.hypot(first_x - second_x, first_y - second_y)


def fragment_anchor_lineage(
    fragments: Sequence[MotionFragment],
    anchors: Sequence[ShotAnchor],
    config: ShotLocalEventConfig,
) -> dict[str, str | None]:
    """Map fragments to unique shot-local stable anchors, never across shots."""

    shot_ids = {
        *(fragment.shot_id for fragment in fragments),
        *(anchor.shot_id for anchor in anchors),
    }
    if len(shot_ids) > 1:
        raise ValueError("fragment lineage evidence must belong to one shot")
    lineage: dict[str, str | None] = {}
    for fragment in fragments:
        first = fragment.samples[0]
        candidates = sorted(
            (
                (_point_distance(first.x_mm, first.y_mm, anchor.x_mm, anchor.y_mm), anchor)
                for anchor in anchors
            ),
            key=lambda item: (item[0], item[1].anchor_id),
        )
        if not candidates or candidates[0][0] > config.collision_distance_mm:
            lineage[fragment.fragment_id] = None
        elif (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= config.collision_ambiguity_mm
        ):
            lineage[fragment.fragment_id] = None
        else:
            lineage[fragment.fragment_id] = candidates[0][1].anchor_id
    return lineage


def observe_shot_boundaries(
    window: ShotWindow,
    cue_samples: Sequence[CueSample],
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ShotLocalEvent]:
    """Confirm existing shot boundaries without performing a second segmentation."""

    if any(sample.shot_id != window.shot_id for sample in cue_samples):
        raise ValueError("shot boundary evidence must belong to one shot")
    usable = tuple(
        sample
        for sample in cue_samples
        if sample.state in {LocalState.CONFIRMED, LocalState.REACQUIRED}
        and sample.x_mm is not None
        and sample.y_mm is not None
    )
    motion_pair = next(
        (
            (previous, current)
            for previous, current in pairwise(usable)
            if _point_distance(
                previous.x_mm,  # type: ignore[arg-type]
                previous.y_mm,  # type: ignore[arg-type]
                current.x_mm,  # type: ignore[arg-type]
                current.y_mm,  # type: ignore[arg-type]
            )
            >= config.cue_motion_min_mm
        ),
        None,
    )
    start = ShotLocalEvent(
        shot_id=window.shot_id,
        event_id=f"{window.shot_id}-event-shot-start",
        event_type="shot_start",
        status="observed" if motion_pair is not None else "abstained",
        timestamp_ms=None if motion_pair is None else motion_pair[1].timestamp_ms,
        actor_id=f"{window.shot_id}-cue",
        target_id=None,
        evidence_frame_ids=(
            (window.start_frame_id,)
            if motion_pair is None
            else (motion_pair[0].frame_id, motion_pair[1].frame_id)
        ),
        confidence=0.0 if motion_pair is None else min(item.confidence for item in motion_pair),
        reason=None if motion_pair is not None else "trusted_cue_motion_insufficient",
    )
    end_observed = window.complete and window.after_frame_id is not None
    end = ShotLocalEvent(
        shot_id=window.shot_id,
        event_id=f"{window.shot_id}-event-shot-end",
        event_type="shot_end",
        status="observed" if end_observed else "abstained",
        timestamp_ms=window.end_ms if end_observed else None,
        actor_id=None,
        target_id=None,
        evidence_frame_ids=(
            tuple(dict.fromkeys((window.end_frame_id, window.after_frame_id)))
            if end_observed
            else (window.end_frame_id,)
        ),
        confidence=1.0 if end_observed else 0.0,
        reason=None if end_observed else "shot_end_not_observed",
    )
    return start, end


def _collision_event(
    *,
    shot_id: str,
    number: int,
    status: str,
    frames: tuple[int, ...],
    actor_id: str | None = None,
    target_id: str | None = None,
    timestamp_ms: int | None = None,
    confidence: float = 0.0,
    reason: str | None = None,
) -> ShotLocalEvent:
    return ShotLocalEvent(
        shot_id=shot_id,
        event_id=f"{shot_id}-event-ball-collision-{number:04d}",
        event_type="ball_ball_collision",
        status=status,
        timestamp_ms=timestamp_ms,
        actor_id=actor_id,
        target_id=target_id,
        evidence_frame_ids=frames,
        confidence=confidence,
        reason=reason,
    )


def observe_ball_ball_collisions(
    fragments: Sequence[MotionFragment],
    anchors: Sequence[ShotAnchor],
    first_contact: ShotLocalEvent,
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ...]:
    """Observe unique short-lived motion transfers without joining fragment identities."""

    shot_id = first_contact.shot_id
    if (
        first_contact.event_type != "first_contact"
        or any(fragment.shot_id != shot_id for fragment in fragments)
        or any(anchor.shot_id != shot_id for anchor in anchors)
    ):
        raise ValueError("collision evidence must belong to one shot and one first contact")

    origins = fragment_anchor_lineage(fragments, anchors, config)
    fallback_frames = tuple(
        sorted(
            {
                *first_contact.evidence_frame_ids,
                *(sample.frame_id for fragment in fragments for sample in fragment.samples),
            }
        )
    ) or (0,)
    if (
        first_contact.status != "observed"
        or first_contact.timestamp_ms is None
        or first_contact.confidence < config.collision_chain_confidence_min
    ):
        return (
            _collision_event(
                shot_id=shot_id,
                number=1,
                status="abstained",
                frames=fallback_frames,
                reason="first_contact_unavailable_for_collision_chain",
            ),
        )

    activated_anchors = {first_contact.target_id}
    consumed_actor_anchors: set[str] = set()
    used_targets: set[str] = set()
    events: list[ShotLocalEvent] = []
    for actor in sorted(fragments, key=lambda item: item.samples[-1].frame_id):
        if len(actor.samples) < 2 or actor.samples[-1].timestamp_ms <= first_contact.timestamp_ms:
            continue
        previous, terminal = actor.samples[-2:]
        dx = terminal.x_mm - previous.x_mm
        dy = terminal.y_mm - previous.y_mm
        terminal_speed = math.hypot(dx, dy)
        if terminal_speed < config.collision_motion_min_mm:
            continue
        transfer_signal = actor.terminal_state in {LocalState.LOST, LocalState.AMBIGUOUS}
        if not transfer_signal and len(actor.samples) >= 3:
            before_previous = actor.samples[-3]
            prior_dx = previous.x_mm - before_previous.x_mm
            prior_dy = previous.y_mm - before_previous.y_mm
            prior_speed = math.hypot(prior_dx, prior_dy)
            if prior_speed >= config.collision_motion_min_mm:
                direction_cosine = (prior_dx * dx + prior_dy * dy) / (prior_speed * terminal_speed)
                transfer_signal = direction_cosine < 0.85 or terminal_speed < prior_speed * 0.6
        if not transfer_signal:
            continue
        actor_origin = origins[actor.fragment_id]
        if (
            actor_origin is None
            or actor_origin not in activated_anchors
            or actor_origin in consumed_actor_anchors
        ):
            continue
        frame_step = terminal.frame_id - previous.frame_id
        if frame_step <= 0:
            continue
        candidates: list[tuple[float, MotionFragment]] = []
        for target in fragments:
            if target.fragment_id == actor.fragment_id or target.fragment_id in used_targets:
                continue
            target_origin = origins[target.fragment_id]
            if (
                target_origin is None
                or target_origin == actor_origin
                or target_origin in activated_anchors
            ):
                continue
            if len(target.samples) < 2:
                continue
            target_first, target_next = target.samples[:2]
            gap = target_first.frame_id - terminal.frame_id
            if gap <= 0 or gap > config.collision_max_gap_frames:
                continue
            target_motion = _point_distance(
                target_first.x_mm, target_first.y_mm, target_next.x_mm, target_next.y_mm
            )
            if target_motion < config.collision_motion_min_mm:
                continue
            predicted_x = terminal.x_mm + dx * gap / frame_step
            predicted_y = terminal.y_mm + dy * gap / frame_step
            prediction_error = math.hypot(
                target_first.x_mm - predicted_x, target_first.y_mm - predicted_y
            )
            if prediction_error > config.collision_distance_mm:
                continue
            outgoing_x = target_next.x_mm - target_first.x_mm
            outgoing_y = target_next.y_mm - target_first.y_mm
            if outgoing_x * dx + outgoing_y * dy <= 0:
                continue
            candidates.append((prediction_error, target))
        candidates.sort(key=lambda item: (item[0], item[1].fragment_id))
        if not candidates:
            continue
        frames = (
            previous.frame_id,
            terminal.frame_id,
            candidates[0][1].samples[0].frame_id,
            candidates[0][1].samples[1].frame_id,
        )
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= config.collision_ambiguity_mm
        ):
            events.append(
                _collision_event(
                    shot_id=shot_id,
                    number=len(events) + 1,
                    status="abstained",
                    frames=frames,
                    actor_id=actor.fragment_id,
                    reason="ball_collision_target_ambiguous",
                )
            )
            consumed_actor_anchors.add(actor_origin)
            used_targets.update(target.fragment_id for _, target in candidates)
            continue
        error, target = candidates[0]
        target_first, target_next = target.samples[:2]
        gap = target_first.frame_id - terminal.frame_id
        status = "observed" if gap == 1 else "inferred"
        confidence = min(
            previous.confidence,
            terminal.confidence,
            target_first.confidence,
            target_next.confidence,
        ) * max(0.0, 1.0 - error / config.collision_distance_mm)
        events.append(
            _collision_event(
                shot_id=shot_id,
                number=len(events) + 1,
                status=status,
                frames=frames,
                actor_id=actor.fragment_id,
                target_id=target.fragment_id,
                timestamp_ms=target_first.timestamp_ms,
                confidence=confidence,
                reason=None if status == "observed" else "bounded_fragment_gap_motion_transfer",
            )
        )
        consumed_actor_anchors.add(actor_origin)
        used_targets.add(target.fragment_id)
        if status == "observed" and confidence >= config.collision_chain_confidence_min:
            target_origin = origins[target.fragment_id]
            if target_origin is not None:
                activated_anchors.add(target_origin)
    if events:
        return tuple(events)
    return (
        _collision_event(
            shot_id=shot_id,
            number=1,
            status="abstained",
            frames=fallback_frames,
            reason="ball_collision_evidence_insufficient",
        ),
    )


def _abstained_contact(
    *,
    shot_id: str,
    evidence_frame_ids: tuple[int, ...],
    reason: str,
) -> ShotLocalEvent:
    return ShotLocalEvent(
        shot_id=shot_id,
        event_id=f"{shot_id}-event-first-contact",
        event_type="first_contact",
        status="abstained",
        timestamp_ms=None,
        actor_id=f"{shot_id}-cue",
        target_id=None,
        evidence_frame_ids=evidence_frame_ids,
        confidence=0.0,
        reason=reason,
    )


def observe_first_anchor_contact(
    cue_samples: Sequence[CueSample],
    anchors: Sequence[ShotAnchor],
    fragments: Sequence[MotionFragment],
    config: ShotLocalEventConfig,
) -> ShotLocalEvent:
    """Observe first contact against the pre-shot anchor roster, or abstain."""

    shot_ids = {
        *(sample.shot_id for sample in cue_samples),
        *(anchor.shot_id for anchor in anchors),
        *(fragment.shot_id for fragment in fragments),
    }
    if len(shot_ids) != 1:
        raise ValueError("first-contact evidence must belong to one shot")
    shot_id = next(iter(shot_ids))
    usable_cue = tuple(
        sample
        for sample in cue_samples
        if sample.state in {LocalState.CONFIRMED, LocalState.REACQUIRED}
        and sample.x_mm is not None
        and sample.y_mm is not None
    )
    fallback_frames = tuple(sample.frame_id for sample in cue_samples) or (0,)

    for previous, current in pairwise(usable_cue):
        cue_motion = _point_distance(
            previous.x_mm,  # type: ignore[arg-type]
            previous.y_mm,  # type: ignore[arg-type]
            current.x_mm,  # type: ignore[arg-type]
            current.y_mm,  # type: ignore[arg-type]
        )
        if cue_motion < config.cue_motion_min_mm:
            continue
        candidates: list[tuple[float, ShotAnchor, int]] = []
        for anchor in anchors:
            cue_distance = _point_distance(
                current.x_mm,  # type: ignore[arg-type]
                current.y_mm,  # type: ignore[arg-type]
                anchor.x_mm,
                anchor.y_mm,
            )
            if cue_distance > config.first_contact_distance_mm:
                continue
            target_evidence = [
                sample
                for fragment in fragments
                for sample in fragment.samples
                if abs(sample.timestamp_ms - current.timestamp_ms)
                <= config.contact_time_tolerance_ms
                and _point_distance(sample.x_mm, sample.y_mm, anchor.x_mm, anchor.y_mm)
                <= config.first_contact_distance_mm
            ]
            if not target_evidence:
                continue
            target = min(
                target_evidence,
                key=lambda sample: (
                    abs(sample.timestamp_ms - current.timestamp_ms),
                    sample.frame_id,
                ),
            )
            target_distance = _point_distance(
                target.x_mm,
                target.y_mm,
                anchor.x_mm,
                anchor.y_mm,
            )
            candidates.append((cue_distance + target_distance, anchor, target.frame_id))
        candidates.sort(key=lambda item: (item[0], item[1].anchor_id))
        if not candidates:
            continue
        evidence = tuple(sorted({previous.frame_id, current.frame_id, candidates[0][2]}))
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= config.first_contact_ambiguity_mm
        ):
            return _abstained_contact(
                shot_id=shot_id,
                evidence_frame_ids=evidence,
                reason="first_contact_anchor_ambiguous",
            )
        _, anchor, _ = candidates[0]
        return ShotLocalEvent(
            shot_id=shot_id,
            event_id=f"{shot_id}-event-first-contact",
            event_type="first_contact",
            status="observed",
            timestamp_ms=current.timestamp_ms,
            actor_id=f"{shot_id}-cue",
            target_id=anchor.anchor_id,
            evidence_frame_ids=evidence,
            confidence=min(previous.confidence, current.confidence, anchor.confidence),
        )
    return _abstained_contact(
        shot_id=shot_id,
        evidence_frame_ids=fallback_frames,
        reason="first_contact_evidence_insufficient",
    )


@dataclass(frozen=True, slots=True)
class _CushionPoint:
    frame_id: int
    timestamp_ms: int
    x_mm: int
    y_mm: int
    confidence: float


def _rail_distance(
    point: _CushionPoint,
    rail: str,
    config: ShotLocalEventConfig,
) -> int:
    if rail == "left":
        return point.x_mm
    if rail == "right":
        return config.table_length_mm - point.x_mm
    if rail == "bottom":
        return point.y_mm
    return config.table_width_mm - point.y_mm


def _trajectory_cushion_candidates(
    *,
    actor_id: str,
    shot_id: str,
    points: Sequence[_CushionPoint],
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ...]:
    events = []
    last_event_by_rail: dict[str, int] = {}
    rails = ("left", "right", "bottom", "top")
    for index in range(1, len(points) - 1):
        first, middle, last = points[index - 1 : index + 2]
        first_gap = middle.frame_id - first.frame_id
        last_gap = last.frame_id - middle.frame_id
        if (
            first_gap <= 0
            or last_gap <= 0
            or first_gap > config.cushion_max_gap_frames
            or last_gap > config.cushion_max_gap_frames
        ):
            continue
        sparse = first_gap > 1 or last_gap > 1
        proximity = config.cushion_distance_mm + (config.cushion_sparse_extra_mm if sparse else 0)
        candidates = []
        for rail in rails:
            before = _rail_distance(first, rail, config)
            nearest = _rail_distance(middle, rail, config)
            after = _rail_distance(last, rail, config)
            if (
                nearest <= proximity
                and nearest - before <= -config.cushion_normal_motion_min_mm
                and after - nearest >= config.cushion_normal_motion_min_mm
            ):
                candidates.append((nearest, rail))
        candidates.sort()
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            continue
        _, rail = candidates[0]
        previous_event_frame = last_event_by_rail.get(rail)
        if (
            previous_event_frame is not None
            and middle.frame_id - previous_event_frame <= config.cushion_cooldown_frames
        ):
            continue
        last_event_by_rail[rail] = middle.frame_id
        events.append(
            ShotLocalEvent(
                shot_id=shot_id,
                event_id=f"{shot_id}-event-cushion-{len(events) + 1:04d}",
                event_type="cushion",
                status="observed",
                timestamp_ms=middle.timestamp_ms,
                actor_id=actor_id,
                target_id=None,
                evidence_frame_ids=(first.frame_id, middle.frame_id, last.frame_id),
                confidence=min(first.confidence, middle.confidence, last.confidence),
                reason=f"rail:{rail}",
            )
        )
    return tuple(events)


def observe_shot_cushions(
    cue_samples: Sequence[CueSample],
    fragments: Sequence[MotionFragment],
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ...]:
    """Observe debounced rail rebounds for the cue and every motion fragment."""

    shot_ids = {
        *(sample.shot_id for sample in cue_samples),
        *(fragment.shot_id for fragment in fragments),
    }
    if not shot_ids:
        return ()
    if len(shot_ids) != 1:
        raise ValueError("cushion evidence must belong to one shot")
    shot_id = next(iter(shot_ids))
    cue_points = tuple(
        _CushionPoint(
            frame_id=sample.frame_id,
            timestamp_ms=sample.timestamp_ms,
            x_mm=sample.x_mm,
            y_mm=sample.y_mm,
            confidence=sample.confidence,
        )
        for sample in cue_samples
        if sample.state in {LocalState.CONFIRMED, LocalState.REACQUIRED}
        and sample.x_mm is not None
        and sample.y_mm is not None
    )
    grouped: list[tuple[str, tuple[_CushionPoint, ...]]] = []
    if cue_points:
        grouped.append((f"{shot_id}-cue", cue_points))
    grouped.extend(
        (
            fragment.fragment_id,
            tuple(
                _CushionPoint(
                    frame_id=sample.frame_id,
                    timestamp_ms=sample.timestamp_ms,
                    x_mm=sample.x_mm,
                    y_mm=sample.y_mm,
                    confidence=sample.confidence,
                )
                for sample in fragment.samples
            ),
        )
        for fragment in fragments
    )
    events = []
    for actor_id, points in grouped:
        actor_events = _trajectory_cushion_candidates(
            actor_id=actor_id,
            shot_id=shot_id,
            points=points,
            config=config,
        )
        for event in actor_events:
            events.append(
                ShotLocalEvent(
                    shot_id=event.shot_id,
                    event_id=f"{shot_id}-event-cushion-{len(events) + 1:04d}",
                    event_type=event.event_type,
                    status=event.status,
                    timestamp_ms=event.timestamp_ms,
                    actor_id=event.actor_id,
                    target_id=event.target_id,
                    evidence_frame_ids=event.evidence_frame_ids,
                    confidence=event.confidence,
                    reason=event.reason,
                )
            )
    return tuple(events)


def observe_fragment_cushions(
    fragments: Sequence[MotionFragment],
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ...]:
    """Compatibility wrapper for fragment-only callers."""

    return observe_shot_cushions((), fragments, config)


def _nearest_pocket(x_mm: int, y_mm: int, config: ShotLocalEventConfig) -> tuple[str, float]:
    pockets = (
        ("left-bottom", 0, 0),
        ("middle-bottom", config.table_length_mm // 2, 0),
        ("right-bottom", config.table_length_mm, 0),
        ("left-top", 0, config.table_width_mm),
        ("middle-top", config.table_length_mm // 2, config.table_width_mm),
        ("right-top", config.table_length_mm, config.table_width_mm),
    )
    return min(
        (
            (label, _point_distance(x_mm, y_mm, pocket_x, pocket_y))
            for label, pocket_x, pocket_y in pockets
        ),
        key=lambda item: (item[1], item[0]),
    )


def observe_fragment_pots(
    fragments: Sequence[MotionFragment],
    after_frame: ObservationFrame,
    config: ShotLocalEventConfig,
) -> tuple[ShotLocalEvent, ...]:
    events = []
    for fragment in fragments:
        terminal = fragment.samples[-1]
        if fragment.terminal_state not in {LocalState.LOST, LocalState.AMBIGUOUS}:
            continue
        pocket, pocket_distance = _nearest_pocket(terminal.x_mm, terminal.y_mm, config)
        if pocket_distance > config.pocket_distance_mm:
            continue
        if after_frame.frame_id - terminal.frame_id < config.pot_missing_frames:
            continue
        still_present = any(
            _point_distance(terminal.x_mm, terminal.y_mm, item.x_mm, item.y_mm)
            <= config.after_match_distance_mm
            for item in after_frame.observations
        )
        if still_present:
            continue
        events.append(
            ShotLocalEvent(
                shot_id=fragment.shot_id,
                event_id=f"{fragment.shot_id}-event-pot-{len(events) + 1:04d}",
                event_type="pot",
                status="observed",
                timestamp_ms=terminal.timestamp_ms,
                actor_id=fragment.fragment_id,
                target_id=None,
                evidence_frame_ids=(terminal.frame_id, after_frame.frame_id),
                confidence=terminal.confidence,
                reason=f"pocket:{pocket}",
            )
        )
    return tuple(events)
