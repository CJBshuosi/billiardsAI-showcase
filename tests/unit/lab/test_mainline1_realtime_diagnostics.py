from dataclasses import asdict
from types import SimpleNamespace

import pytest

from billiards_ai.lab.events.models import TrackedFrame
from billiards_ai.lab.mainline1_realtime_diagnostics import (
    MAX_DIAGNOSTIC_FRAMES,
    diagnose_frame_sequence,
    diagnose_realtime_shot,
)
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    RealtimeTrackerConfig,
    TrackedBall,
    TrackedObservation,
    TrackerDiagnosticSnapshot,
    TrackerHealth,
    TrackerHealthState,
)
from billiards_ai.lab.shot_local.models import BallGroup


def test_sequence_distinguishes_missing_duplicate_and_reversed_ids():
    result = diagnose_frame_sequence((10, 12, 12, 11, 15), start=10, end=14)
    assert result.missing_count == 2
    assert result.first_missing_ids == (13, 14)
    assert result.duplicate_count == 1
    assert result.first_duplicate_indices == (2,)
    assert result.reversed_count == 1
    assert result.first_reversed_indices == (3,)
    assert result.outside_interval_count == 1
    assert result.first_outside_interval_indices == (4,)
    assert result.inspected_count == result.count == 5
    assert not result.truncated


def test_empty_sequence_and_sparse_anchor_have_different_expectations():
    empty = diagnose_frame_sequence((), start=4, end=6)
    assert empty.missing_count == 3 and empty.first_missing_ids == (4, 5, 6)
    sparse = diagnose_frame_sequence((4, 6))
    assert sparse.expected_count is None and sparse.missing_count is None
    assert sparse.first_frame_id == 4 and sparse.last_frame_id == 6


def test_large_diagnostics_are_explicitly_partial_and_anomalies_capped():
    result = diagnose_frame_sequence(
        (0,) * (MAX_DIAGNOSTIC_FRAMES + 7), start=0, end=MAX_DIAGNOSTIC_FRAMES + 20
    )
    assert result.count == MAX_DIAGNOSTIC_FRAMES + 7
    assert result.inspected_count == MAX_DIAGNOSTIC_FRAMES
    assert result.truncated and result.expected_range_truncated
    assert result.duplicate_count == MAX_DIAGNOSTIC_FRAMES - 1
    assert result.first_duplicate_indices == tuple(range(1, 9))
    # Counts are deliberately limited to inspected data/range; no full-coverage claim.
    assert result.missing_count == MAX_DIAGNOSTIC_FRAMES - 1
    assert result.first_missing_ids == tuple(range(1, 9))


def test_shot_reports_active_rule_gap_and_unknown_counts_without_track_labels():
    ball = TrackedBall(
        track_id="private-track-label",
        group=BallGroup.UNKNOWN,
        state="confirmed",
        provenance="anchor",
        center_px=(1, 2),
        table_mm=(10, 20),
        confidence=0.8,
        anchor_frame_id=10,
    )
    rows = tuple(TrackedObservation(i, i * 1000, (ball,)) for i in (10, 12))
    health = TrackerHealth(
        TrackerHealthState.DEGRADED,
        2,
        5.0,
        20.0,
        21.0,
        3,
        "tracker source lag exceeds realtime budget",
    )
    result = diagnose_realtime_shot(
        shot_id="shot-1",
        interval_start=9,
        interval_end=13,
        declared_source_count=5,
        shot_start=10,
        shot_end=12,
        source_ids=(9, 10, 11, 12, 13),
        tracked=rows,
        rule_frames=tuple(TrackedFrame(i, i, ()) for i in (10, 12)),
        anchors=(AnchorObservation(10, 10000, ()),),
        publishable=False,
        abstention_reason="tracker_degraded",
        tracker_snapshot=TrackerDiagnosticSnapshot(
            health.state,
            health.processed_frames,
            health.maximum_source_lag_frames,
            health.reason,
            health,
        ),
        tracker_config=RealtimeTrackerConfig(
            allow_accounted_frame_gaps=True,
            enforce_processing_budget=False,
        ),
    )
    assert result.source.missing_count == 0
    assert result.tracked.first_missing_ids == (9, 11, 13)
    assert result.active_rule.first_missing_ids == (11,)
    assert result.unknown_group_counts == ((10, 1, 1), (12, 1, 1))
    assert result.tracker_snapshot_at_refinement.last_computed_health is health
    assert result.tracker_policy.allow_accounted_frame_gaps is True
    assert result.tracker_policy.enforce_processing_budget is False
    assert result.status == "OBSERVED_METADATA"
    assert "private-track-label" not in str(asdict(result))
    assert result.anchors.missing_count is None


def test_missing_health_and_source_ids_are_not_reported_as_observed_empty_source():
    result = diagnose_realtime_shot(
        shot_id="shot-1",
        interval_start=1,
        interval_end=2,
        declared_source_count=2,
        shot_start=1,
        shot_end=2,
        source_ids=None,
        tracked=(),
        rule_frames=(),
        anchors=(),
        publishable=False,
        abstention_reason="ring_incomplete",
        tracker_snapshot=None,
        tracker_config=None,
    )
    assert result.source is None
    assert result.tracker_snapshot_at_refinement is None and result.tracker_policy is None
    assert result.unknown_group_counts == ()


def test_shot_limits_keep_total_counts_and_mark_partial_projections():
    size = MAX_DIAGNOSTIC_FRAMES + 1
    result = diagnose_realtime_shot(
        shot_id="shot-long",
        interval_start=0,
        interval_end=size - 1,
        declared_source_count=size,
        shot_start=0,
        shot_end=size - 1,
        source_ids=range(size),
        tracked=tuple(TrackedObservation(i, i, ()) for i in range(size)),
        rule_frames=tuple(TrackedFrame(i, i, ()) for i in range(size)),
        anchors=tuple(AnchorObservation(i, i, ()) for i in range(size)),
        publishable=False,
        abstention_reason="shot_limit_exceeded",
        tracker_snapshot=None,
        tracker_config=None,
    )
    assert result.status == "PARTIAL_LIMIT"
    for summary in (result.source, result.tracked, result.rule, result.anchors):
        assert summary.count == size and summary.truncated
        assert summary.inspected_count == MAX_DIAGNOSTIC_FRAMES
        assert summary.last_frame_id == size - 1
    assert result.active_rule_input_truncated and result.unknown_counts_truncated
    assert len(result.unknown_group_counts) == MAX_DIAGNOSTIC_FRAMES


@pytest.mark.parametrize("tail", ["private-label", True, -1, 2**63])
def test_uninspected_tail_frame_id_is_still_validated(tail):
    with pytest.raises(ValueError, match="invalid diagnostic frame ID"):
        diagnose_frame_sequence((0,) * MAX_DIAGNOSTIC_FRAMES + (tail,))


@pytest.mark.parametrize("stream", ["tracked", "rule_frames", "anchors"])
def test_projection_tail_metadata_cannot_bypass_frame_id_validation(stream):
    rows = {
        "tracked": (TrackedObservation(0, 0, ()),) * MAX_DIAGNOSTIC_FRAMES,
        "rule_frames": (TrackedFrame(0, 0, ()),) * MAX_DIAGNOSTIC_FRAMES,
        "anchors": (AnchorObservation(0, 0, ()),) * MAX_DIAGNOSTIC_FRAMES,
    }
    rows[stream] += (SimpleNamespace(capture_frame_id="private-label", frame_id="private-label"),)
    with pytest.raises(ValueError, match="invalid diagnostic frame ID"):
        diagnose_realtime_shot(
            shot_id="shot-1",
            interval_start=0,
            interval_end=1,
            declared_source_count=2,
            shot_start=0,
            shot_end=1,
            source_ids=(0, 1),
            **rows,
            publishable=False,
            abstention_reason="tracker_degraded",
            tracker_snapshot=None,
            tracker_config=None,
        )
