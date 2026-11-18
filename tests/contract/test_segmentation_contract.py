"""Executable Story 2.6 contracts for deterministic shot segmentation."""

import pytest
from pydantic import ValidationError


def _api():
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.segmentation import (
        BallPosition,
        SegmentationFailure,
        SegmentationFailureReason,
        ShotSegment,
        ShotSegmentationResult,
        TableSnapshot,
    )

    return {
        "BallPosition": BallPosition,
        "Point": Point,
        "SegmentationFailure": SegmentationFailure,
        "SegmentationFailureReason": SegmentationFailureReason,
        "ShotSegment": ShotSegment,
        "ShotSegmentationResult": ShotSegmentationResult,
        "TableSnapshot": TableSnapshot,
    }


def test_segment_snapshot_failure_and_result_are_strict_frozen_contracts():
    api = _api()
    positions = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=100, y_mm=200)),
        api["BallPosition"](ball_id=1, position=api["Point"](x_mm=500, y_mm=600)),
    )
    before = api["TableSnapshot"](timestamp_ms=400, positions=positions)
    after = api["TableSnapshot"](timestamp_ms=900, positions=positions)
    segment = api["ShotSegment"](
        t_start_ms=500,
        t_end_ms=900,
        ball_positions_before=before,
        ball_positions_after=after,
    )
    failure = api["SegmentationFailure"](
        reason=api["SegmentationFailureReason"].LONG_DETECTION_GAP,
        t_start_ms=1000,
        t_end_ms=1600,
        missing_ball_ids=(0,),
    )
    result = api["ShotSegmentationResult"](segments=(segment,), failures=(failure,))

    assert result.schema_version == "1.0"
    assert before.as_dict() == {
        0: api["Point"](x_mm=100, y_mm=200),
        1: api["Point"](x_mm=500, y_mm=600),
    }
    assert segment.duration_ms == 400

    with pytest.raises(ValidationError):
        before.timestamp_ms = 401
    with pytest.raises(ValidationError):
        segment.t_end_ms = 901
    with pytest.raises(ValidationError):
        failure.reason = api["SegmentationFailureReason"].ENDED_IN_MOTION
    with pytest.raises(ValidationError):
        api["ShotSegment"].model_validate(segment.model_dump() | {"pixel_x": 12})


def test_contract_rejects_noncanonical_snapshots_and_invalid_intervals():
    api = _api()
    duplicate_positions = (
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=100, y_mm=200)),
        api["BallPosition"](ball_id=0, position=api["Point"](x_mm=101, y_mm=200)),
    )

    with pytest.raises(ValidationError, match="sorted and unique"):
        api["TableSnapshot"](timestamp_ms=0, positions=duplicate_positions)

    point = api["BallPosition"](ball_id=0, position=api["Point"](x_mm=100, y_mm=200))
    snapshot = api["TableSnapshot"](timestamp_ms=500, positions=(point,))
    with pytest.raises(ValidationError, match="end must be after start"):
        api["ShotSegment"](
            t_start_ms=500,
            t_end_ms=500,
            ball_positions_before=snapshot,
            ball_positions_after=snapshot,
        )
