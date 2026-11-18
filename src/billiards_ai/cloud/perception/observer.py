"""Deterministic Story 2.7 observer over millimetre endpoint snapshots."""

from __future__ import annotations

from billiards_ai.shared.constants import CUE_BALL
from billiards_ai.shared.models.observation import (
    BallMovement,
    FoulEvidence,
    FoulEvidenceKind,
    ObjectiveShotObservation,
    ObservationConfidence,
    ObservationField,
    ObservationSource,
    RawObservation,
    StruckBallCandidate,
    UnsupportedReason,
)
from billiards_ai.shared.models.segmentation import ShotSegment
from billiards_ai.shared.models.shot import TargetZoneSource


def _supported(value, source: ObservationSource):
    return ObservationField(value=value, confidence=ObservationConfidence.HIGH, source=source)


def _unknown(reason: UnsupportedReason):
    return ObservationField(
        value=None,
        confidence=ObservationConfidence.UNKNOWN,
        source=ObservationSource.UNSUPPORTED_BY_ENDPOINT_SNAPSHOTS,
        unsupported_reason=reason,
    )


def observe_shot(segment: ShotSegment) -> ObjectiveShotObservation:
    """Derive only facts supported by a Story 2.6 segment's endpoint snapshots."""

    before = segment.ball_positions_before
    after = segment.ball_positions_after
    before_by_id = before.as_dict()
    after_by_id = after.as_dict()
    unexpected_after_ids = tuple(sorted(after_by_id.keys() - before_by_id.keys()))
    if unexpected_after_ids:
        raise ValueError(
            f"after snapshot contains ball IDs absent from before: {unexpected_after_ids}"
        )

    potted_ids = tuple(sorted(before_by_id.keys() - after_by_id.keys()))
    shared_ids = tuple(sorted(before_by_id.keys() & after_by_id.keys()))
    movements = tuple(
        BallMovement(
            ball_id=ball_id,
            distance_mm=before_by_id[ball_id].distance_to(after_by_id[ball_id]),
        )
        for ball_id in shared_ids
    )
    movement_by_id = {item.ball_id: item.distance_mm for item in movements}

    moved_object_ids = {
        ball_id for ball_id in shared_ids if ball_id != CUE_BALL and movement_by_id[ball_id] > 0
    }
    potted_object_ids = set(potted_ids) - {CUE_BALL}
    candidate_ids = tuple(sorted(moved_object_ids | potted_object_ids))
    candidates = tuple(
        StruckBallCandidate(
            ball_id=ball_id,
            confidence=ObservationConfidence.LOW,
            source=ObservationSource.SEGMENT_SNAPSHOT_DELTA,
            unsupported_reason=UnsupportedReason.FIRST_CONTACT_NOT_OBSERVABLE,
        )
        for ball_id in candidate_ids
    )
    struck_reason = (
        UnsupportedReason.FIRST_CONTACT_NOT_OBSERVABLE
        if candidates
        else UnsupportedReason.NO_OBJECT_BALL_MOVEMENT_EVIDENCE
    )

    cue_end = after_by_id.get(CUE_BALL)
    if cue_end is None:
        cue_reason = (
            UnsupportedReason.CUE_BALL_POTTED
            if CUE_BALL in before_by_id
            else UnsupportedReason.CUE_BALL_NOT_OBSERVED
        )
        cue_ball_end = _unknown(cue_reason)
        travel_distance = _unknown(cue_reason)
    else:
        cue_ball_end = _supported(cue_end, ObservationSource.SEGMENT_AFTER_SNAPSHOT)
        travel_distance = _supported(
            movement_by_id[CUE_BALL], ObservationSource.ENDPOINT_DISPLACEMENT
        )

    if CUE_BALL in potted_ids:
        foul_evidence = (
            FoulEvidence(
                kind=FoulEvidenceKind.CUE_BALL_POTTED,
                supports_foul=True,
                confidence=ObservationConfidence.HIGH,
                source=ObservationSource.SEGMENT_SNAPSHOT_DELTA,
            ),
        )
        is_foul = _supported(True, ObservationSource.SEGMENT_SNAPSHOT_DELTA)
    else:
        foul_evidence = ()
        is_foul = _unknown(UnsupportedReason.FOUL_NOT_OBSERVABLE)

    trajectory_points = tuple(sample.position for sample in segment.cue_ball_path)
    if len(set(trajectory_points)) >= 2:
        trajectory = _supported(
            trajectory_points,
            ObservationSource.SEGMENT_CUE_BALL_PATH,
        )
    else:
        trajectory = _unknown(
            UnsupportedReason.TRAJECTORY_INSUFFICIENT_POINTS
            if trajectory_points
            else UnsupportedReason.TRAJECTORY_NOT_OBSERVED
        )

    return ObjectiveShotObservation(
        t_start_ms=segment.t_start_ms,
        t_end_ms=segment.t_end_ms,
        ball_positions_before=_supported(
            before.positions, ObservationSource.SEGMENT_BEFORE_SNAPSHOT
        ),
        ball_positions_after=_supported(after.positions, ObservationSource.SEGMENT_AFTER_SNAPSHOT),
        potted_balls=_supported(potted_ids, ObservationSource.SEGMENT_SNAPSHOT_DELTA),
        cue_ball_end=cue_ball_end,
        ball_movements=_supported(movements, ObservationSource.ENDPOINT_DISPLACEMENT),
        travel_distance_mm=travel_distance,
        trajectory=trajectory,
        trajectory_calibration=segment.calibration,
        cushion_count=_unknown(UnsupportedReason.CUSHION_CONTACT_NOT_OBSERVABLE),
        struck_ball=_unknown(struck_reason),
        is_foul=is_foul,
        target_zone_source=TargetZoneSource.NONE,
        deviation_mm=None,
        raw_observation=RawObservation(
            struck_ball_candidates=candidates,
            foul_evidence=foul_evidence,
        ),
    )
