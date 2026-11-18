"""Deterministically segment millimetre-only FrameDetections into shots."""

from __future__ import annotations

from collections.abc import Sequence

from billiards_ai.shared.constants import (
    CUE_BALL,
    MAX_DETECTION_GAP_MS,
    MAX_SHOT_DURATION_MS,
    STILL_DURATION_MS,
    STILL_THRESHOLD_MM,
)
from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.detection import FrameDetections
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.segmentation import (
    BallPosition,
    SegmentationFailure,
    SegmentationFailureReason,
    ShotSegment,
    ShotSegmentationResult,
    TableSnapshot,
    TrajectorySample,
)


def _snapshot(timestamp_ms: int, positions: dict[int, Point]) -> TableSnapshot:
    return TableSnapshot(
        timestamp_ms=timestamp_ms,
        positions=tuple(
            BallPosition(ball_id=ball_id, position=position.model_copy(deep=True))
            for ball_id, position in sorted(positions.items())
        ),
    )


def _has_moved(anchor: dict[int, Point], current: dict[int, Point]) -> bool:
    return any(
        anchor[ball_id].distance_to(current[ball_id]) >= STILL_THRESHOLD_MM
        for ball_id in anchor.keys() & current.keys()
    )


def _failure(
    reason: SegmentationFailureReason,
    start: int,
    end: int,
    *,
    missing_ball_ids: tuple[int, ...] = (),
) -> SegmentationFailure:
    return SegmentationFailure(
        reason=reason,
        t_start_ms=start,
        t_end_ms=end,
        missing_ball_ids=missing_ball_ids,
    )


def _cue_ball_path(
    frames: Sequence[FrameDetections],
    *,
    shot_start_ms: int,
    shot_end_ms: int,
    before: TableSnapshot,
) -> tuple[TrajectorySample, ...]:
    before_cue = before.as_dict().get(CUE_BALL)
    if before_cue is None:
        return ()
    samples = [TrajectorySample(timestamp_ms=before.timestamp_ms, position=before_cue)]
    for frame in frames:
        if not shot_start_ms <= frame.timestamp_ms <= shot_end_ms:
            continue
        cue = frame.high_confidence_positions().get(CUE_BALL)
        if cue is None or cue == samples[-1].position:
            continue
        samples.append(TrajectorySample(timestamp_ms=frame.timestamp_ms, position=cue))
    return tuple(samples)


def _calibration_lineage(
    frames: Sequence[FrameDetections],
    *,
    shot_start_ms: int,
    shot_end_ms: int,
) -> CalibrationLineage | None:
    relevant = tuple(
        frame.calibration for frame in frames if shot_start_ms <= frame.timestamp_ms <= shot_end_ms
    )
    if not relevant or any(lineage is None for lineage in relevant):
        return None
    first = relevant[0]
    return first if all(lineage == first for lineage in relevant) else None


def segment_shots(frames: Sequence[FrameDetections]) -> ShotSegmentationResult:
    """Return reproducible boundaries, snapshots, and typed fail-closed outcomes.

    Short missing observations remain unknown: they neither imply movement nor
    contribute to a stillness window. Once a known ball reappears, only its real
    millimetre coordinate is compared; positions are never forward-filled.
    """

    if not frames:
        return ShotSegmentationResult()

    timestamps = tuple(frame.timestamp_ms for frame in frames)
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        if current <= previous:
            return ShotSegmentationResult(
                failures=(
                    _failure(
                        SegmentationFailureReason.NON_INCREASING_TIMESTAMPS,
                        previous,
                        current,
                    ),
                )
            )

    first_positions = frames[0].high_confidence_positions()
    # Keep identity hypotheses in the tracking set as well as confirmed
    # coordinates.  This prevents an ambiguous colour ball from disappearing
    # from the state machine simply because it has no safe position yet.
    known_ball_ids = set(frames[0].candidate_ball_ids())
    if CUE_BALL not in known_ball_ids:
        return ShotSegmentationResult(
            failures=(
                _failure(
                    SegmentationFailureReason.LONG_DETECTION_GAP,
                    frames[0].timestamp_ms,
                    frames[0].timestamp_ms,
                    missing_ball_ids=(CUE_BALL,),
                ),
            )
        )

    segments: list[ShotSegment] = []
    missing_since: dict[int, int] = {}
    initial_anchor = first_positions
    initial_started_at = frames[0].timestamp_ms
    initial_pending_change: int | None = None
    initial_stable = False

    idle_anchor = first_positions
    last_stable_snapshot = _snapshot(frames[0].timestamp_ms, first_positions)
    pending_shot_start: int | None = None
    pending_before_snapshot: TableSnapshot | None = None

    shot_start: int | None = None
    shot_before: TableSnapshot | None = None
    stability_anchor: dict[int, Point] | None = None
    stability_started_at: int | None = None
    stability_snapshot: TableSnapshot | None = None

    previous_timestamp = frames[0].timestamp_ms
    for frame in frames[1:]:
        timestamp = frame.timestamp_ms
        if timestamp - previous_timestamp > MAX_DETECTION_GAP_MS:
            return ShotSegmentationResult(
                segments=tuple(segments),
                failures=(
                    _failure(
                        SegmentationFailureReason.LONG_DETECTION_GAP,
                        previous_timestamp,
                        timestamp,
                        missing_ball_ids=tuple(sorted(known_ball_ids)),
                    ),
                ),
            )
        previous_timestamp = timestamp

        positions = frame.high_confidence_positions()
        known_ball_ids.update(frame.candidate_ball_ids())
        observed_known = known_ball_ids & positions.keys()
        absent = known_ball_ids - observed_known
        for ball_id in absent:
            missing_since.setdefault(ball_id, timestamp)
        for ball_id in observed_known:
            missing_since.pop(ball_id, None)
        overdue = tuple(
            sorted(
                ball_id
                for ball_id, start in missing_since.items()
                if timestamp - start > MAX_DETECTION_GAP_MS
            )
        )
        # Before a shot has begun, a prolonged missing track makes the input
        # unusable.  During an active shot, however, a ball absent through the
        # settled endpoint is an auditable terminal (potted) state rather than
        # an invented detection gap.
        if overdue and shot_start is None:
            gap_start = min(missing_since[ball_id] for ball_id in overdue)
            return ShotSegmentationResult(
                segments=tuple(segments),
                failures=(
                    _failure(
                        SegmentationFailureReason.LONG_DETECTION_GAP,
                        gap_start,
                        timestamp,
                        missing_ball_ids=overdue,
                    ),
                ),
            )

        complete = not absent
        if shot_start is None and not complete:
            continue

        if not initial_stable:
            if _has_moved(initial_anchor, positions):
                return ShotSegmentationResult(
                    failures=(
                        _failure(
                            SegmentationFailureReason.STARTED_IN_MOTION,
                            initial_pending_change or initial_started_at,
                            timestamp,
                        ),
                    )
                )
            if positions != initial_anchor and initial_pending_change is None:
                initial_pending_change = timestamp
            if timestamp - initial_started_at < STILL_DURATION_MS:
                continue
            initial_stable = True
            idle_anchor = positions
            last_stable_snapshot = _snapshot(timestamp, positions)
            initial_pending_change = None
            continue

        if shot_start is None:
            cue_distance = idle_anchor[CUE_BALL].distance_to(positions[CUE_BALL])
            cue_changed = idle_anchor[CUE_BALL] != positions[CUE_BALL]
            if cue_changed and pending_shot_start is None:
                pending_shot_start = timestamp
                pending_before_snapshot = last_stable_snapshot
            if cue_distance < STILL_THRESHOLD_MM:
                if not cue_changed:
                    pending_shot_start = None
                    pending_before_snapshot = None
                    idle_anchor = positions
                last_stable_snapshot = _snapshot(timestamp, positions)
                continue

            shot_start = pending_shot_start or timestamp
            shot_before = pending_before_snapshot or last_stable_snapshot
            stability_anchor = positions
            stability_started_at = None
            stability_snapshot = None
            pending_shot_start = None
            pending_before_snapshot = None
            continue

        assert stability_anchor is not None
        if not positions:
            stability_started_at = None
            stability_snapshot = None
            continue

        if _has_moved(stability_anchor, positions):
            if timestamp - shot_start > MAX_SHOT_DURATION_MS:
                return ShotSegmentationResult(
                    segments=tuple(segments),
                    failures=(
                        _failure(
                            SegmentationFailureReason.MAX_SHOT_DURATION_EXCEEDED,
                            shot_start,
                            timestamp,
                        ),
                    ),
                )
            stability_anchor = positions
            stability_started_at = None
            stability_snapshot = None
            continue

        if stability_started_at is None:
            stability_started_at = timestamp
            stability_snapshot = _snapshot(timestamp, positions)
        if timestamp - stability_started_at < STILL_DURATION_MS:
            continue

        assert shot_before is not None
        assert stability_snapshot is not None
        if stability_started_at - shot_start > MAX_SHOT_DURATION_MS:
            return ShotSegmentationResult(
                segments=tuple(segments),
                failures=(
                    _failure(
                        SegmentationFailureReason.MAX_SHOT_DURATION_EXCEEDED,
                        shot_start,
                        stability_started_at,
                    ),
                ),
            )
        segments.append(
            ShotSegment(
                t_start_ms=shot_start,
                t_end_ms=stability_started_at,
                ball_positions_before=shot_before,
                ball_positions_after=stability_snapshot,
                cue_ball_path=_cue_ball_path(
                    frames,
                    shot_start_ms=shot_start,
                    shot_end_ms=stability_started_at,
                    before=shot_before,
                ),
                calibration=_calibration_lineage(
                    frames,
                    shot_start_ms=shot_start,
                    shot_end_ms=stability_started_at,
                ),
            )
        )
        # Balls that were present at the start but absent at the confirmed
        # endpoint are terminally potted.  Remove them from later idle tracking
        # so the next shot is not spuriously rejected as a detection gap.
        terminally_missing = set(shot_before.as_dict()).difference(stability_snapshot.as_dict())
        known_ball_ids.difference_update(terminally_missing)
        for ball_id in terminally_missing:
            missing_since.pop(ball_id, None)
        shot_start = None
        shot_before = None
        idle_anchor = positions
        last_stable_snapshot = _snapshot(timestamp, positions)
        stability_anchor = None
        stability_started_at = None
        stability_snapshot = None

    if shot_start is not None:
        overdue = tuple(
            sorted(
                ball_id
                for ball_id, start in missing_since.items()
                if frames[-1].timestamp_ms - start > MAX_DETECTION_GAP_MS
            )
        )
        if overdue:
            gap_start = min(missing_since[ball_id] for ball_id in overdue)
            return ShotSegmentationResult(
                segments=tuple(segments),
                failures=(
                    _failure(
                        SegmentationFailureReason.LONG_DETECTION_GAP,
                        gap_start,
                        frames[-1].timestamp_ms,
                        missing_ball_ids=overdue,
                    ),
                ),
            )
        reason = (
            SegmentationFailureReason.MAX_SHOT_DURATION_EXCEEDED
            if frames[-1].timestamp_ms - shot_start > MAX_SHOT_DURATION_MS
            else SegmentationFailureReason.ENDED_IN_MOTION
        )
        return ShotSegmentationResult(
            segments=tuple(segments),
            failures=(
                _failure(
                    reason,
                    shot_start,
                    frames[-1].timestamp_ms,
                ),
            ),
        )

    return ShotSegmentationResult(segments=tuple(segments))
