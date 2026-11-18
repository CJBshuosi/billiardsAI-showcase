"""Observation-only foul subset; full rules remain outside perception."""

from __future__ import annotations

from collections.abc import Sequence

from billiards_ai.lab.events.models import EventStatus, EvidenceEvent


def observe_supported_fouls(
    pot_events: Sequence[EvidenceEvent], *, requested_rule: str
) -> EvidenceEvent:
    evidence = tuple(
        dict.fromkeys(frame_id for event in pot_events for frame_id in event.evidence_frame_ids)
    )
    if not evidence:
        raise ValueError("foul observation requires event evidence")
    if requested_rule != "cue_ball_scratch":
        return EvidenceEvent(
            event_type=f"foul.{requested_rule}",
            status=EventStatus.UNSUPPORTED,
            timestamp_ms=None,
            actor_identity=None,
            target_identity=None,
            evidence_frame_ids=evidence,
            confidence=0.0,
            reason="rule_not_in_supported_observation_subset",
        )
    scratch = next(
        (
            event
            for event in pot_events
            if event.event_type == "pot" and event.actor_identity == "cue"
        ),
        None,
    )
    if scratch is None:
        return EvidenceEvent(
            event_type="foul.cue_ball_scratch",
            status=EventStatus.ABSTAINED,
            timestamp_ms=None,
            actor_identity="cue",
            target_identity=None,
            evidence_frame_ids=evidence,
            confidence=0.0,
            reason="cue_pot_not_observed",
        )
    return EvidenceEvent(
        event_type="foul.cue_ball_scratch",
        status=EventStatus.OBSERVED,
        timestamp_ms=scratch.timestamp_ms,
        actor_identity="cue",
        target_identity=None,
        evidence_frame_ids=scratch.evidence_frame_ids,
        confidence=scratch.confidence,
    )
