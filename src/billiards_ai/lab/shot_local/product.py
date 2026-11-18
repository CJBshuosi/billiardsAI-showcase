"""Canonical product contract for one shot-local Mainline 1 analysis."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from billiards_ai.lab.shot_local.active_tracks import (
    ActiveBallTrack,
    ActiveTrackConfig,
    CushionContactDecision,
    DelayedCollisionDecision,
    extract_ball_appearance_signatures,
    observe_delayed_collision_chain,
    observe_three_layer_cushions,
    track_anchor_roster,
)
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig, fragment_anchor_lineage
from billiards_ai.lab.shot_local.models import ObservationFrame, ShotLocalEvent, ShotLocalResult


@dataclass(frozen=True, slots=True)
class CanonicalShotAnalysis:
    """All authoritative decisions for one shot, held in one in-memory object."""

    evidence: ShotLocalResult
    active_tracks: tuple[ActiveBallTrack, ...]
    collisions: tuple[DelayedCollisionDecision, ...]
    cushions: tuple[CushionContactDecision, ...]
    retained_events: tuple[ShotLocalEvent, ...]
    fragment_lineage: Mapping[str, str | None]

    def __post_init__(self) -> None:
        shot_id = self.evidence.shot_id
        if any(track.shot_id != shot_id for track in self.active_tracks):
            raise ValueError("canonical active track violates shot scope")
        if any(event.shot_id != shot_id for event in self.retained_events):
            raise ValueError("canonical retained event violates shot scope")


def shot_local_product_active_config() -> ActiveTrackConfig:
    """Return the reviewed milestone tracking and collision policy."""

    return ActiveTrackConfig(
        position_gate_mm=160,
        unmatched_cost=210,
        process_noise_mm=14,
        measurement_noise_mm=6,
        max_prediction_frames=5,
        appearance_weight=75,
        group_mismatch_cost=180,
        collision_distance_mm=72,
        moving_speed_mm_per_frame=6,
        stationary_speed_mm_per_frame=4,
        collision_lookahead_frames=5,
        collision_candidate_margin=0.08,
    )


def build_canonical_shot_analyses(
    *,
    evidence_results: Sequence[ShotLocalResult],
    observations: Sequence[ObservationFrame],
    images_by_frame: Mapping[int, np.ndarray],
    event_config: ShotLocalEventConfig,
    active_config: ActiveTrackConfig,
    timing_ms: dict[str, float] | None = None,
) -> tuple[CanonicalShotAnalysis, ...]:
    """Compose stable tracks and canonical contacts once for every shot."""

    tracking_started = time.perf_counter()
    signatures = extract_ball_appearance_signatures(images_by_frame, observations)
    tracking_seconds = time.perf_counter() - tracking_started
    event_seconds = 0.0
    analyses: list[CanonicalShotAnalysis] = []
    for evidence in evidence_results:
        frames = tuple(
            frame
            for frame in observations
            if evidence.window.before_frame_id <= frame.frame_id <= evidence.window.end_frame_id
        )
        tracking_started = time.perf_counter()
        tracks = track_anchor_roster(
            frames,
            evidence.anchors,
            active_config,
            appearance_by_detection_id=signatures,
        )
        tracking_seconds += time.perf_counter() - tracking_started
        event_started = time.perf_counter()
        first_contact = next(
            event for event in evidence.events if event.event_type == "first_contact"
        )
        collisions = observe_delayed_collision_chain(tracks, first_contact, active_config)
        cushions = observe_three_layer_cushions(
            tracks,
            table_length_mm=event_config.table_length_mm,
            table_width_mm=event_config.table_width_mm,
            contact_distance_mm=event_config.cushion_distance_mm,
            normal_motion_min_mm=event_config.cushion_normal_motion_min_mm,
            max_observation_gap_frames=event_config.cushion_max_gap_frames,
            sparse_extra_mm=event_config.cushion_sparse_extra_mm,
            ball_collisions=collisions,
        )
        retained_events = tuple(
            event
            for event in evidence.events
            if event.event_type in {"shot_start", "first_contact", "pot", "shot_end"}
        )
        analyses.append(
            CanonicalShotAnalysis(
                evidence=evidence,
                active_tracks=tracks,
                collisions=collisions,
                cushions=cushions,
                retained_events=retained_events,
                fragment_lineage=fragment_anchor_lineage(
                    evidence.fragments,
                    evidence.anchors,
                    event_config,
                ),
            )
        )
        event_seconds += time.perf_counter() - event_started
    if timing_ms is not None:
        timing_ms["stable_track_ms"] = round(tracking_seconds * 1000, 3)
        timing_ms["event_observation_ms"] = round(event_seconds * 1000, 3)
    return tuple(analyses)


def map_event_entity(
    local_id: str | None,
    fragment_lineage: Mapping[str, str | None],
) -> tuple[str | None, bool]:
    """Map fragment evidence to a stable shot-local entity, fail closed if ambiguous."""

    if local_id is None:
        return None, False
    if "-fragment-" not in local_id:
        return local_id, False
    mapped = fragment_lineage.get(local_id)
    return mapped, mapped is None


def _reason_identity(reason: str | None, prefix: str) -> str | None:
    if reason is None or not reason.startswith(prefix):
        return None
    value = reason.removeprefix(prefix)
    return value or None


def _retained_event_row(
    event: ShotLocalEvent,
    fragment_lineage: Mapping[str, str | None],
) -> dict[str, object]:
    actor_id, actor_ambiguous = map_event_entity(event.actor_id, fragment_lineage)
    target_id, target_ambiguous = map_event_entity(event.target_id, fragment_lineage)
    identity_ambiguous = actor_ambiguous or target_ambiguous
    return {
        "shot_id": event.shot_id,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "actor_id": None if identity_ambiguous else actor_id,
        "target_id": None if identity_ambiguous else target_id,
        "rail": _reason_identity(event.reason, "rail:"),
        "pocket": _reason_identity(event.reason, "pocket:"),
        "contact_frame_id": event.evidence_frame_ids[0],
        "decision_frame_id": event.evidence_frame_ids[-1],
        "timestamp_ms": event.timestamp_ms,
        "status": "abstained" if identity_ambiguous else event.status,
        "detection_layer": None,
        "contact_kind": event.event_type,
        "confidence": 0.0 if identity_ambiguous else event.confidence,
        "minimum_distance_mm": None,
        "evidence_frame_ids": list(event.evidence_frame_ids),
        "reason": "stable_track_mapping_ambiguous" if identity_ambiguous else event.reason,
    }


def _collision_row(
    shot_id: str,
    decision: DelayedCollisionDecision,
) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "event_id": (
            f"{shot_id}-event-ball-collision-"
            f"{decision.actor_id.rsplit('-', 1)[-1]}-"
            f"{(decision.target_id or 'unknown').rsplit('-', 1)[-1]}-"
            f"{decision.contact_frame_id:06d}"
        ),
        "event_type": "ball_collision",
        "actor_id": decision.actor_id,
        "target_id": decision.target_id,
        "rail": None,
        "pocket": None,
        "contact_frame_id": decision.contact_frame_id,
        "decision_frame_id": decision.decision_frame_id,
        "timestamp_ms": None,
        "status": decision.status,
        "detection_layer": decision.detection_layer,
        "contact_kind": decision.contact_kind,
        "confidence": decision.confidence,
        "minimum_distance_mm": decision.minimum_distance_mm,
        "evidence_frame_ids": list(decision.evidence_frame_ids),
        "reason": decision.reason,
    }


def _cushion_row(
    shot_id: str,
    decision: CushionContactDecision,
) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "event_id": (
            f"{shot_id}-event-cushion-{decision.actor_id.rsplit('-', 1)[-1]}-"
            f"{decision.rail.replace(':', '-')}-{decision.contact_frame_id:06d}"
        ),
        "event_type": "cushion",
        "actor_id": decision.actor_id,
        "target_id": None,
        "rail": decision.rail,
        "pocket": None,
        "contact_frame_id": decision.contact_frame_id,
        "decision_frame_id": decision.decision_frame_id,
        "timestamp_ms": None,
        "status": "observed",
        "detection_layer": decision.detection_layer,
        "contact_kind": decision.contact_kind,
        "confidence": decision.confidence,
        "minimum_distance_mm": decision.minimum_distance_mm,
        "evidence_frame_ids": list(decision.evidence_frame_ids or (decision.contact_frame_id,)),
        "reason": None,
    }


def canonical_world_event_rows(analysis: CanonicalShotAnalysis) -> list[dict[str, object]]:
    """Serialize every canonical decision using one identity-complete schema."""

    rows = [
        _retained_event_row(event, analysis.fragment_lineage) for event in analysis.retained_events
    ]
    rows.extend(_collision_row(analysis.evidence.shot_id, item) for item in analysis.collisions)
    rows.extend(_cushion_row(analysis.evidence.shot_id, item) for item in analysis.cushions)
    order = {
        "shot_start": 0,
        "first_contact": 1,
        "ball_collision": 2,
        "cushion": 3,
        "pot": 4,
        "shot_end": 5,
    }
    return sorted(
        rows,
        key=lambda row: (
            int(row["contact_frame_id"]),
            order.get(str(row["event_type"]), 99),
            str(row["actor_id"]),
            str(row["target_id"]),
        ),
    )


def canonical_event_key(event: Mapping[str, object]) -> tuple[object, ...]:
    """Return the identity-complete key shared by JSON and overlay markers."""

    return (
        event.get("event_id"),
        event.get("event_type"),
        event.get("actor_id"),
        event.get("target_id"),
        event.get("rail"),
        event.get("pocket"),
        event.get("contact_frame_id"),
        event.get("decision_frame_id"),
    )


def canonical_world_event_counts(
    events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize canonical events without treating abstentions as observations."""

    observed = [event for event in events if event.get("status") == "observed"]
    names = (
        "shot_start",
        "first_contact",
        "ball_collision",
        "cushion",
        "pot",
        "shot_end",
    )
    result: dict[str, object] = {
        f"{name}_count": sum(event.get("event_type") == name for event in observed)
        for name in names
    }
    result["event_count_by_layer"] = {
        layer: sum(event.get("detection_layer") == layer for event in observed)
        for layer in ("L1", "L2", "L3")
    }
    return result
