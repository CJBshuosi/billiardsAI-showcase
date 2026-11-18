"""Fail-closed coverage decisions for one retained realtime shot interval."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from billiards_ai.lab.events.models import BallState, TrackedFrame
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    TrackedObservation,
    TrackerHealth,
    TrackerHealthState,
)
from billiards_ai.lab.shot_local.models import CalibratedBallObservation, ObservationFrame


class RealtimeInterval(Protocol):
    start_frame_id: int
    end_frame_id: int
    frame_count: int
    segment_sha256: Mapping[str, str]

    def frames(self) -> Iterator[CaptureFrame]: ...


@dataclass(frozen=True, slots=True)
class FactCoverageVerdict:
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RealtimeEvidenceCoverage:
    source_frame_ids: tuple[int, ...]
    anchor_frame_ids: tuple[int, ...]
    maximum_anchor_gap_ms: float | None
    uncertain_frame_ids: tuple[int, ...]
    segment_sha256: dict[str, str]
    uncertain_track_states: tuple[tuple[int, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RefinedRealtimeShot:
    publishable: bool
    shot_abstention_reason: str | None
    fact_coverage: dict[str, FactCoverageVerdict]
    coverage: RealtimeEvidenceCoverage
    tracked: tuple[TrackedObservation, ...]

    def rule_frames(self) -> tuple[TrackedFrame, ...]:
        rows: list[TrackedFrame] = []
        for tracked_frame in self.tracked:
            balls = tuple(
                BallState(
                    identity=_rule_identity(ball.group, ball.track_id),
                    track_id=ball.track_id,
                    x_mm=ball.table_mm[0],
                    y_mm=ball.table_mm[1],
                    confidence=ball.confidence,
                )
                for ball in tracked_frame.balls
                if ball.state == "confirmed" and ball.table_mm is not None
            )
            rows.append(
                TrackedFrame(
                    frame_id=tracked_frame.capture_frame_id,
                    timestamp_ms=round(tracked_frame.captured_at_ns / 1_000_000),
                    balls=balls,
                )
            )
        return tuple(rows)

    def observations(self) -> tuple[ObservationFrame, ...]:
        rows: list[ObservationFrame] = []
        for tracked_frame in self.tracked:
            observations = tuple(
                CalibratedBallObservation(
                    detection_id=(f"realtime-{tracked_frame.capture_frame_id:08d}-{ball.track_id}"),
                    x_mm=ball.table_mm[0],
                    y_mm=ball.table_mm[1],
                    confidence=ball.confidence,
                    group=ball.group,
                    group_confidence=ball.confidence,
                )
                for ball in tracked_frame.balls
                if ball.state == "confirmed" and ball.table_mm is not None
            )
            rows.append(
                ObservationFrame(
                    frame_id=tracked_frame.capture_frame_id,
                    timestamp_ms=round(tracked_frame.captured_at_ns / 1_000_000),
                    observations=observations,
                )
            )
        return tuple(rows)

    def provenance(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "publishable": self.publishable,
            "shot_abstention_reason": self.shot_abstention_reason,
            "source_frame_ids": list(self.coverage.source_frame_ids),
            "anchor_frame_ids": list(self.coverage.anchor_frame_ids),
            "maximum_anchor_gap_ms": self.coverage.maximum_anchor_gap_ms,
            "uncertain_frame_ids": list(self.coverage.uncertain_frame_ids),
            "segment_sha256": dict(self.coverage.segment_sha256),
            "fact_coverage": {
                name: {"status": verdict.status, "reason": verdict.reason}
                for name, verdict in self.fact_coverage.items()
            },
        }


class RealtimeShotRefiner:
    def __init__(
        self,
        *,
        maximum_active_anchor_gap_ms: float = 250.0,
        maximum_shot_frames: int = 1800,
    ) -> None:
        if maximum_active_anchor_gap_ms <= 0 or maximum_shot_frames <= 0:
            raise ValueError("realtime refinement limits must be positive")
        self._maximum_anchor_gap_ms = maximum_active_anchor_gap_ms
        self._maximum_shot_frames = maximum_shot_frames

    def refine(
        self,
        *,
        interval: RealtimeInterval,
        tracked: Sequence[TrackedObservation],
        anchors: Sequence[AnchorObservation],
        tracker_health: TrackerHealth | None = None,
    ) -> RefinedRealtimeShot:
        try:
            source_rows = tuple(interval.frames())
        except (OSError, RuntimeError, ValueError):
            return self._shot_abstention(
                reason="ring_incomplete",
                interval=interval,
                tracked=tracked,
                anchors=anchors,
                source_ids=(),
            )
        source_ids = tuple(row.capture_frame_id for row in source_rows)
        expected_ids = tuple(range(interval.start_frame_id, interval.end_frame_id + 1))
        if source_ids != expected_ids or len(source_rows) != interval.frame_count:
            return self._shot_abstention(
                reason="source_frame_gap",
                interval=interval,
                tracked=tracked,
                anchors=anchors,
                source_ids=source_ids,
            )
        if any(
            current.captured_at_ns < previous.captured_at_ns
            for previous, current in zip(source_rows, source_rows[1:], strict=False)
        ):
            return self._shot_abstention(
                reason="source_frame_gap",
                interval=interval,
                tracked=tracked,
                anchors=anchors,
                source_ids=source_ids,
            )
        tracked_rows = tuple(tracked)
        if tuple(row.capture_frame_id for row in tracked_rows) != source_ids:
            return self._shot_abstention(
                reason="tracker_degraded",
                interval=interval,
                tracked=tracked_rows,
                anchors=anchors,
                source_ids=source_ids,
            )
        if tracker_health is not None and tracker_health.state is TrackerHealthState.DEGRADED:
            return self._shot_abstention(
                reason="tracker_degraded",
                interval=interval,
                tracked=tracked_rows,
                anchors=anchors,
                source_ids=source_ids,
            )
        if len(source_ids) > self._maximum_shot_frames:
            return self._shot_abstention(
                reason="shot_limit_exceeded",
                interval=interval,
                tracked=tracked_rows,
                anchors=anchors,
                source_ids=source_ids,
            )
        anchor_rows = tuple(sorted(anchors, key=lambda row: row.capture_frame_id))
        start_timestamp_ns = source_rows[0].captured_at_ns
        end_timestamp_ns = source_rows[-1].captured_at_ns
        if not anchor_rows or anchor_rows[0].captured_at_ns - start_timestamp_ns > 250_100_000:
            return self._shot_abstention(
                reason="trusted_before_missing",
                interval=interval,
                tracked=tracked_rows,
                anchors=anchor_rows,
                source_ids=source_ids,
            )
        if end_timestamp_ns - anchor_rows[-1].captured_at_ns > 250_100_000:
            return self._shot_abstention(
                reason="trusted_after_missing",
                interval=interval,
                tracked=tracked_rows,
                anchors=anchor_rows,
                source_ids=source_ids,
            )

        maximum_anchor_gap_ms = self._maximum_gap_ms(anchor_rows)
        uncertain_track_states = self._uncertain_tracks(tracked_rows)
        uncertain = tuple(sorted({frame_id for frame_id, _, _ in uncertain_track_states}))
        motion_reason = None
        if maximum_anchor_gap_ms is not None and (
            maximum_anchor_gap_ms > self._maximum_anchor_gap_ms + 0.1
        ):
            motion_reason = "anchor_gap_exceeded"
        elif uncertain:
            motion_reason = "tracker_uncertain"
        fact_coverage = {
            "table_state_before": FactCoverageVerdict(status="observed"),
            "table_state_after": FactCoverageVerdict(status="observed"),
            "first_contact": FactCoverageVerdict(
                status="observed" if motion_reason is None else "abstained",
                reason=motion_reason,
            ),
            "cushion_contacts": FactCoverageVerdict(
                status="observed" if motion_reason is None else "abstained",
                reason=motion_reason,
            ),
            "pots": FactCoverageVerdict(
                status="observed" if motion_reason != "anchor_gap_exceeded" else "abstained",
                reason=(motion_reason if motion_reason == "anchor_gap_exceeded" else None),
            ),
        }
        return RefinedRealtimeShot(
            publishable=True,
            shot_abstention_reason=None,
            fact_coverage=fact_coverage,
            coverage=self._coverage(
                interval,
                source_ids,
                anchor_rows,
                maximum_anchor_gap_ms,
                uncertain,
                uncertain_track_states,
            ),
            tracked=tracked_rows,
        )

    @staticmethod
    def _uncertain_tracks(
        tracked: Sequence[TrackedObservation],
    ) -> tuple[tuple[int, str, str], ...]:
        uncertain = {
            (row.capture_frame_id, ball.track_id, ball.state)
            for row in tracked
            for ball in row.balls
            if ball.state != "confirmed"
        }
        known_groups = {}
        previous_ids: set[str] = set()
        for index, row in enumerate(tracked):
            groups = {ball.track_id: ball.group for ball in row.balls}
            if index:
                # Late-anchor replay replaces the tracker. A missing ID can
                # therefore vanish without a tracker-emitted lost sample.
                uncertain.update(
                    (row.capture_frame_id, track_id, "lost")
                    for track_id in previous_ids - groups.keys()
                )
                uncertain.update(
                    (row.capture_frame_id, track_id, "ambiguous")
                    for track_id, group in groups.items()
                    if track_id not in known_groups or known_groups[track_id] != group
                )
            known_groups.update(groups)
            previous_ids = set(groups)
        # Keep changes in this retained interval, not permanent phantom balls:
        # a later recovery cannot erase a gap, nor does a pot poison later shots.
        return tuple(sorted(uncertain))

    @staticmethod
    def _maximum_gap_ms(anchors: Sequence[AnchorObservation]) -> float | None:
        if len(anchors) < 2:
            return None
        return max(
            (current.captured_at_ns - previous.captured_at_ns) / 1_000_000
            for previous, current in zip(anchors, anchors[1:], strict=False)
        )

    def _shot_abstention(
        self,
        *,
        reason: str,
        interval: RealtimeInterval,
        tracked: Sequence[TrackedObservation],
        anchors: Sequence[AnchorObservation],
        source_ids: tuple[int, ...],
    ) -> RefinedRealtimeShot:
        fact_coverage = {
            name: FactCoverageVerdict(status="abstained", reason=reason)
            for name in (
                "table_state_before",
                "table_state_after",
                "first_contact",
                "cushion_contacts",
                "pots",
            )
        }
        return RefinedRealtimeShot(
            publishable=False,
            shot_abstention_reason=reason,
            fact_coverage=fact_coverage,
            coverage=self._coverage(interval, source_ids, anchors, None, ()),
            tracked=tuple(tracked),
        )

    @staticmethod
    def _coverage(
        interval: RealtimeInterval,
        source_ids: tuple[int, ...],
        anchors: Sequence[AnchorObservation],
        maximum_anchor_gap_ms: float | None,
        uncertain: tuple[int, ...],
        uncertain_track_states: tuple[tuple[int, str, str], ...] = (),
    ) -> RealtimeEvidenceCoverage:
        return RealtimeEvidenceCoverage(
            source_frame_ids=source_ids,
            anchor_frame_ids=tuple(row.capture_frame_id for row in anchors),
            maximum_anchor_gap_ms=maximum_anchor_gap_ms,
            uncertain_frame_ids=uncertain,
            segment_sha256=dict(interval.segment_sha256),
            uncertain_track_states=uncertain_track_states,
        )


def _rule_identity(group: object, track_id: str) -> str:
    value = getattr(group, "value", str(group))
    if value == "cue":
        return "cue"
    if value == "eight":
        return "eight"
    if value in {"solid", "stripe"}:
        return f"{value}:{track_id}"
    return f"unknown:{track_id}"
