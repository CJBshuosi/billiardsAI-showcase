"""Project one shot-local replay into explicitly quality-gated evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from billiards_ai.edge.realtime_chain.models import (
    CueEvidence,
    EventDecision,
    EventEvidence,
    EvidenceDecision,
    EvidenceSource,
    EvidenceStatus,
    HabitFeature,
    ProfileIncrement,
    ShotEvidenceProjection,
    ShotReplayEvidence,
    TableDifference,
)


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    cue_confidence_min: float = 0.7
    cue_start_motion_min_mm: float = 20.0
    event_confidence_min: float = 0.7

    def __post_init__(self) -> None:
        if not 0 <= self.cue_confidence_min <= 1 or not 0 <= self.event_confidence_min <= 1:
            raise ValueError("projection confidence thresholds must be between zero and one")
        if self.cue_start_motion_min_mm <= 0:
            raise ValueError("cue start motion threshold must be positive")


def _distance(first: CueEvidence, second: CueEvidence) -> float:
    return math.hypot(
        second.x_mm - first.x_mm,  # type: ignore[operator]
        second.y_mm - first.y_mm,  # type: ignore[operator]
    )


def _usable_cue(shot: ShotReplayEvidence, config: ProjectionConfig) -> tuple[CueEvidence, ...]:
    return tuple(
        sample
        for sample in shot.cue_samples
        if sample.state in {"confirmed", "reacquired", "stopped"}
        and sample.x_mm is not None
        and sample.y_mm is not None
        and sample.confidence >= config.cue_confidence_min
    )


def _cue_gate(
    shot: ShotReplayEvidence,
    config: ProjectionConfig,
) -> tuple[EvidenceDecision, tuple[CueEvidence, ...], float, float]:
    samples = _usable_cue(shot, config)
    lineage = tuple(f"cue-frame:{sample.frame_id}" for sample in samples)
    if len(samples) < 2:
        return (
            EvidenceDecision(
                status=EvidenceStatus.ABSTAINED,
                reason="cue_start_evidence_insufficient",
                evidence_ids=lineage,
            ),
            samples,
            0.0,
            0.0,
        )
    distances = tuple(_distance(first, second) for first, second in pairwise(samples))
    travel = sum(distances)
    if travel < config.cue_start_motion_min_mm:
        return (
            EvidenceDecision(
                status=EvidenceStatus.ABSTAINED,
                reason="cue_start_motion_below_threshold",
                evidence_ids=lineage,
            ),
            samples,
            travel,
            0.0,
        )
    speeds = tuple(
        distance * 1000 / (second.timestamp_ms - first.timestamp_ms)
        for distance, (first, second) in zip(distances, pairwise(samples), strict=True)
        if second.timestamp_ms > first.timestamp_ms
    )
    if not speeds:
        return (
            EvidenceDecision(
                status=EvidenceStatus.ABSTAINED,
                reason="cue_timing_evidence_invalid",
                evidence_ids=lineage,
            ),
            samples,
            travel,
            0.0,
        )
    return (
        EvidenceDecision(EvidenceStatus.ADMITTED, None, lineage),
        samples,
        travel,
        max(speeds),
    )


def _event_decision(event: EventEvidence, config: ProjectionConfig) -> EventDecision:
    if event.status == "abstained":
        status = EvidenceStatus.ABSTAINED
    elif event.status == "unsupported":
        status = EvidenceStatus.UNSUPPORTED
    elif event.status == "inferred" or event.confidence < config.event_confidence_min:
        status = EvidenceStatus.PENDING
    else:
        status = EvidenceStatus.ADMITTED
    reason = event.reason
    if status is EvidenceStatus.PENDING and reason is None:
        reason = "event_confidence_below_threshold"
    return EventDecision(
        event_id=event.event_id,
        event_type=event.event_type,
        status=status,
        confidence=event.confidence,
        reason=reason,
        evidence_frame_ids=event.evidence_frame_ids,
    )


def _event_decisions(
    shot: ShotReplayEvidence,
    config: ProjectionConfig,
) -> tuple[EventDecision, ...]:
    decisions = list(_event_decision(event, config) for event in shot.events)
    boundary_frames = tuple(
        frame for frame in (shot.before_frame_id, shot.after_frame_id) if frame is not None
    )
    if not any(item.event_type == "ball_ball_collision" for item in decisions):
        decisions.append(
            EventDecision(
                event_id=f"{shot.shot_id}-decision-ball-ball-collision",
                event_type="ball_ball_collision",
                status=EvidenceStatus.UNSUPPORTED,
                confidence=0.0,
                reason="general_ball_collision_observer_unavailable",
                evidence_frame_ids=boundary_frames,
            )
        )
    if not any(item.event_type == "shot_end" for item in decisions):
        decisions.append(
            EventDecision(
                event_id=f"{shot.shot_id}-decision-shot-end",
                event_type="shot_end",
                status=(
                    EvidenceStatus.ADMITTED
                    if shot.complete and shot.after_frame_id is not None
                    else EvidenceStatus.PENDING
                ),
                confidence=1.0 if shot.complete and shot.after_frame_id is not None else 0.0,
                reason=(
                    None
                    if shot.complete and shot.after_frame_id is not None
                    else "shot_end_not_observed"
                ),
                evidence_frame_ids=(
                    (shot.after_frame_id,)
                    if shot.after_frame_id is not None
                    else (shot.before_frame_id,)
                ),
            )
        )
    return tuple(decisions)


def _event_features(
    shot: ShotReplayEvidence,
    decisions: tuple[EventDecision, ...],
) -> tuple[HabitFeature, ...]:
    by_id = {event.event_id: event for event in shot.events}
    admitted = tuple(item for item in decisions if item.status is EvidenceStatus.ADMITTED)
    anchor_groups = {anchor.anchor_id: anchor.group for anchor in shot.anchors}
    features: list[HabitFeature] = []
    contacts = tuple(item for item in admitted if item.event_type == "first_contact")
    if contacts:
        event = by_id[contacts[0].event_id]
        group = None if event.target_id is None else anchor_groups.get(event.target_id)
        if group is not None:
            features.append(
                HabitFeature(
                    code="first_contact_target_group",
                    value=group,
                    unit="category",
                    evidence_ids=(event.event_id,),
                )
            )
    cushions = tuple(item for item in admitted if item.event_type == "cushion")
    if cushions:
        features.append(
            HabitFeature(
                code="observed_cushion_count",
                value=float(len(cushions)),
                unit="count",
                evidence_ids=tuple(item.event_id for item in cushions),
            )
        )
    pots = tuple(item for item in admitted if item.event_type == "pot")
    if pots:
        features.append(
            HabitFeature(
                code="observed_pot_count",
                value=float(len(pots)),
                unit="count_not_success",
                evidence_ids=tuple(item.event_id for item in pots),
            )
        )
    return tuple(features)


def _empty_increment(
    *, user_id: str, source: EvidenceSource, shot: ShotReplayEvidence, status: EvidenceStatus
) -> ProfileIncrement:
    return ProfileIncrement(
        user_id=user_id,
        session_id=shot.session_id,
        shot_id=shot.shot_id,
        status=status,
        source=source,
        features=(),
    )


def project_shot_evidence(
    *,
    user_id: str,
    source: EvidenceSource,
    shot: ShotReplayEvidence,
    config: ProjectionConfig,
) -> ShotEvidenceProjection:
    """Admit only traceable evidence; never infer unsupported success or identity."""

    difference = TableDifference(
        before_frame_id=shot.before_frame_id,
        after_frame_id=shot.after_frame_id,
        before_ball_count=len(shot.anchors),
        after_ball_count=len(shot.after_balls),
        net_ball_count=(
            None if shot.after_frame_id is None else len(shot.after_balls) - len(shot.anchors)
        ),
    )
    event_decisions = _event_decisions(shot, config)
    if not shot.complete or shot.after_frame_id is None:
        cue_start = EvidenceDecision(EvidenceStatus.PENDING, "shot_incomplete", ())
        status = EvidenceStatus.PENDING
        return ShotEvidenceProjection(
            source=source,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            status=status,
            reason="shot_incomplete",
            cue_start=cue_start,
            event_decisions=event_decisions,
            table_difference=difference,
            profile_increment=_empty_increment(
                user_id=user_id, source=source, shot=shot, status=status
            ),
        )
    cue_start, samples, travel, peak_speed = _cue_gate(shot, config)
    if cue_start.status is not EvidenceStatus.ADMITTED:
        return ShotEvidenceProjection(
            source=source,
            session_id=shot.session_id,
            shot_id=shot.shot_id,
            status=EvidenceStatus.ABSTAINED,
            reason="cue_start_evidence_insufficient",
            cue_start=cue_start,
            event_decisions=event_decisions,
            table_difference=difference,
            profile_increment=_empty_increment(
                user_id=user_id,
                source=source,
                shot=shot,
                status=EvidenceStatus.ABSTAINED,
            ),
        )
    cue_lineage = cue_start.evidence_ids
    end = samples[-1]
    features = (
        HabitFeature(
            "cue_motion_proxy_peak_mm_s",
            round(peak_speed, 6),
            "motion_proxy_not_force",
            cue_lineage,
        ),
        HabitFeature("cue_travel_mm", round(travel, 6), "mm", cue_lineage),
        HabitFeature("cue_end_x_mm", float(end.x_mm), "mm", (f"cue-frame:{end.frame_id}",)),
        HabitFeature("cue_end_y_mm", float(end.y_mm), "mm", (f"cue-frame:{end.frame_id}",)),
        *_event_features(shot, event_decisions),
    )
    increment = ProfileIncrement(
        user_id=user_id,
        session_id=shot.session_id,
        shot_id=shot.shot_id,
        status=EvidenceStatus.ADMITTED,
        source=source,
        features=features,
    )
    return ShotEvidenceProjection(
        source=source,
        session_id=shot.session_id,
        shot_id=shot.shot_id,
        status=EvidenceStatus.ADMITTED,
        reason=None,
        cue_start=cue_start,
        event_decisions=event_decisions,
        table_difference=difference,
        profile_increment=increment,
    )
