"""Bounded metadata for diagnosing the first refined shot, never rule inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from billiards_ai.lab.events.models import TrackedFrame
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    RealtimeTrackerConfig,
    TrackedObservation,
    TrackerDiagnosticSnapshot,
)
from billiards_ai.lab.shot_local.models import BallGroup

MAX_DIAGNOSTIC_FRAMES = 4096
MAX_DIAGNOSTIC_BALLS = 128
_MAX_ANOMALIES = 8
_MAX_FRAME_ID = 2**63 - 1


def _checked_frame_id(value: int) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_FRAME_ID:
        raise ValueError("invalid diagnostic frame ID")
    return value


@dataclass(frozen=True, slots=True)
class FrameSequenceDiagnostics:
    count: int
    inspected_count: int
    truncated: bool
    first_frame_id: int | None
    last_frame_id: int | None
    inspected_order_sha256: str
    expected_count: int | None
    expected_range_truncated: bool
    missing_count: int | None
    first_missing_ids: tuple[int, ...]
    duplicate_count: int
    first_duplicate_indices: tuple[int, ...]
    reversed_count: int
    first_reversed_indices: tuple[int, ...]
    outside_interval_count: int
    first_outside_interval_indices: tuple[int, ...]


def diagnose_frame_sequence(
    ids: Sequence[int],
    *,
    start: int | None = None,
    end: int | None = None,
) -> FrameSequenceDiagnostics:
    """Inspect a bounded prefix without sorting/deduplicating away anomalies.

    Missing counts refer only to the inspected prefix and expected-range prefix.
    Either truncation flag prevents interpreting those as whole-sequence counts.
    Sparse anchors deliberately have no dense expected range.
    """
    if (start is None) != (end is None):
        raise ValueError("invalid diagnostic frame range")
    if start is not None and _checked_frame_id(end) < _checked_frame_id(start):
        raise ValueError("invalid diagnostic frame range")
    rows = tuple(_checked_frame_id(i) for i in ids[:MAX_DIAGNOSTIC_FRAMES])
    seen: set[int] = set()
    duplicates: list[int] = []
    reversed_rows: list[int] = []
    outside: list[int] = []
    duplicate_count = reversed_count = outside_count = 0
    for index, frame_id in enumerate(rows):
        if frame_id in seen:
            duplicate_count += 1
            if len(duplicates) < _MAX_ANOMALIES:
                duplicates.append(index)
        seen.add(frame_id)
        if index and frame_id < rows[index - 1]:
            reversed_count += 1
            if len(reversed_rows) < _MAX_ANOMALIES:
                reversed_rows.append(index)
        if start is not None and not start <= frame_id <= end:
            outside_count += 1
            if len(outside) < _MAX_ANOMALIES:
                outside.append(index)
    expected_count = None if start is None else end - start + 1
    missing = (
        ()
        if start is None
        else tuple(
            frame_id
            for frame_id in range(start, start + min(expected_count, MAX_DIAGNOSTIC_FRAMES))
            if frame_id not in seen
        )
    )
    return FrameSequenceDiagnostics(
        count=len(ids),
        inspected_count=len(rows),
        truncated=len(ids) > len(rows),
        first_frame_id=None if not ids else _checked_frame_id(ids[0]),
        last_frame_id=None if not ids else _checked_frame_id(ids[-1]),
        inspected_order_sha256=hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        expected_count=expected_count,
        expected_range_truncated=expected_count is not None
        and expected_count > MAX_DIAGNOSTIC_FRAMES,
        missing_count=None if start is None else len(missing),
        first_missing_ids=missing[:_MAX_ANOMALIES],
        duplicate_count=duplicate_count,
        first_duplicate_indices=tuple(duplicates),
        reversed_count=reversed_count,
        first_reversed_indices=tuple(reversed_rows),
        outside_interval_count=outside_count,
        first_outside_interval_indices=tuple(outside),
    )


@dataclass(frozen=True, slots=True)
class TrackerDiagnosticPolicy:
    allow_accounted_frame_gaps: bool
    maximum_accounted_frame_gap: int
    enforce_processing_budget: bool
    processing_budget_ms: float
    maximum_source_lag_frames: int
    history_frames: int


@dataclass(frozen=True, slots=True)
class RealtimeShotDiagnostics:
    status: str
    shot_id: str
    error_type: str | None = None
    interval_start: int | None = None
    interval_end: int | None = None
    declared_source_count: int | None = None
    shot_start: int | None = None
    shot_end: int | None = None
    source: FrameSequenceDiagnostics | None = None
    tracked: FrameSequenceDiagnostics | None = None
    rule: FrameSequenceDiagnostics | None = None
    active_rule: FrameSequenceDiagnostics | None = None
    anchors: FrameSequenceDiagnostics | None = None
    publishable: bool | None = None
    abstention_reason: str | None = None
    # A late anchor may replace the tracker; this is not its historical first cause.
    tracker_snapshot_at_refinement: TrackerDiagnosticSnapshot | None = None
    tracker_policy: TrackerDiagnosticPolicy | None = None
    # (frame ID, confirmed/on-table UNKNOWN count, all confirmed/on-table count).
    unknown_group_counts: tuple[tuple[int, int, int], ...] = ()
    unknown_counts_truncated: bool = False
    active_rule_input_truncated: bool = False


def diagnose_realtime_shot(
    *,
    shot_id: str,
    interval_start: int,
    interval_end: int,
    declared_source_count: int,
    shot_start: int,
    shot_end: int,
    source_ids: Sequence[int] | None,
    tracked: Sequence[TrackedObservation],
    rule_frames: Sequence[TrackedFrame],
    anchors: Sequence[AnchorObservation],
    publishable: bool,
    abstention_reason: str | None,
    tracker_snapshot: TrackerDiagnosticSnapshot | None,
    tracker_config: RealtimeTrackerConfig | None,
) -> RealtimeShotDiagnostics:
    """Reuse already-materialized metadata; never read/decode frames or infer facts."""
    tracked_rows = tracked[:MAX_DIAGNOSTIC_FRAMES]
    rule_prefix = tuple(row.frame_id for row in rule_frames[:MAX_DIAGNOSTIC_FRAMES])
    unknown_rows = []
    ball_limit = False
    for row in tracked_rows:
        ball_limit |= len(row.balls) > MAX_DIAGNOSTIC_BALLS
        confirmed = tuple(
            ball
            for ball in row.balls[:MAX_DIAGNOSTIC_BALLS]
            if ball.state == "confirmed" and ball.table_mm is not None
        )
        unknown_rows.append(
            (
                _checked_frame_id(row.capture_frame_id),
                sum(ball.group is BallGroup.UNKNOWN for ball in confirmed),
                len(confirmed),
            )
        )
    tracked_summary = diagnose_frame_sequence(
        tuple(row.capture_frame_id for row in tracked_rows),
        start=interval_start,
        end=interval_end,
    )
    anchor_summary = diagnose_frame_sequence(
        tuple(row.capture_frame_id for row in anchors[:MAX_DIAGNOSTIC_FRAMES])
    )
    rule_summary = diagnose_frame_sequence(rule_prefix, start=interval_start, end=interval_end)
    if len(tracked) > MAX_DIAGNOSTIC_FRAMES:
        tracked_summary = replace(
            tracked_summary,
            count=len(tracked),
            truncated=True,
            last_frame_id=_checked_frame_id(tracked[-1].capture_frame_id),
        )
    if len(anchors) > MAX_DIAGNOSTIC_FRAMES:
        anchor_summary = replace(
            anchor_summary,
            count=len(anchors),
            truncated=True,
            last_frame_id=_checked_frame_id(anchors[-1].capture_frame_id),
        )
    if len(rule_frames) > MAX_DIAGNOSTIC_FRAMES:
        rule_summary = replace(
            rule_summary,
            count=len(rule_frames),
            truncated=True,
            last_frame_id=_checked_frame_id(rule_frames[-1].frame_id),
        )
    truncated = (
        ball_limit
        or any(
            len(rows) > MAX_DIAGNOSTIC_FRAMES
            for rows in (
                tracked,
                rule_frames,
                anchors,
                () if source_ids is None else source_ids,
            )
        )
        or max(interval_end - interval_start, shot_end - shot_start) + 1 > MAX_DIAGNOSTIC_FRAMES
    )
    return RealtimeShotDiagnostics(
        status="PARTIAL_LIMIT" if truncated else "OBSERVED_METADATA",
        shot_id=shot_id,
        interval_start=interval_start,
        interval_end=interval_end,
        declared_source_count=declared_source_count,
        shot_start=shot_start,
        shot_end=shot_end,
        source=None
        if source_ids is None
        else diagnose_frame_sequence(
            source_ids,
            start=interval_start,
            end=interval_end,
        ),
        tracked=tracked_summary,
        rule=rule_summary,
        active_rule=diagnose_frame_sequence(
            tuple(i for i in rule_prefix if shot_start <= i <= shot_end),
            start=shot_start,
            end=shot_end,
        ),
        anchors=anchor_summary,
        publishable=publishable,
        abstention_reason=abstention_reason,
        tracker_snapshot_at_refinement=tracker_snapshot,
        tracker_policy=None
        if tracker_config is None
        else TrackerDiagnosticPolicy(
            allow_accounted_frame_gaps=tracker_config.allow_accounted_frame_gaps,
            maximum_accounted_frame_gap=tracker_config.maximum_accounted_frame_gap,
            enforce_processing_budget=tracker_config.enforce_processing_budget,
            processing_budget_ms=tracker_config.processing_budget_ms,
            maximum_source_lag_frames=tracker_config.maximum_source_lag_frames,
            history_frames=tracker_config.history_frames,
        ),
        unknown_group_counts=tuple(unknown_rows),
        unknown_counts_truncated=ball_limit or len(tracked) > MAX_DIAGNOSTIC_FRAMES,
        active_rule_input_truncated=len(rule_frames) > MAX_DIAGNOSTIC_FRAMES,
    )
