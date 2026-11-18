"""Executable AD-13e contracts for per-frame ball observations."""

import pytest
from pydantic import ValidationError


def _api():
    from billiards_ai.shared.models.detection import (
        BallCandidate,
        BallColor,
        DetectionSource,
        FrameDetection,
        FrameDetections,
        FrameQuality,
        PixelObservation,
    )
    from billiards_ai.shared.models.geometry import Point

    return {
        "BallCandidate": BallCandidate,
        "BallColor": BallColor,
        "DetectionSource": DetectionSource,
        "FrameDetection": FrameDetection,
        "FrameDetections": FrameDetections,
        "FrameQuality": FrameQuality,
        "PixelObservation": PixelObservation,
        "Point": Point,
    }


def _detection(api: dict, *, confidence: float = 0.93, candidates=None):
    return api["FrameDetection"](
        candidates=candidates
        or (
            api["BallCandidate"](
                ball_id=0,
                color=api["BallColor"].WHITE,
                confidence=0.98,
            ),
        ),
        position=api["Point"](x_mm=500, y_mm=600),
        confidence=confidence,
        source=api["DetectionSource"].CLASSIC_CV_CIRCLE_COLOR,
        raw_observation=api["PixelObservation"](
            center_px=(500.25, 600.5),
            radius_px=21.0,
            circularity=0.96,
            color=api["BallColor"].WHITE,
        ),
    )


def test_detection_contract_is_strict_frozen_and_keeps_candidates_and_source():
    api = _api()
    detection = _detection(
        api,
        candidates=(
            api["BallCandidate"](ball_id=1, color=api["BallColor"].YELLOW, confidence=0.52),
            api["BallCandidate"](ball_id=9, color=api["BallColor"].YELLOW, confidence=0.48),
        ),
    )
    frame = api["FrameDetections"](
        timestamp_ms=250,
        detections=(detection,),
        missing_ball_ids=tuple(ball_id for ball_id in range(16) if ball_id not in {1, 9}),
        frame_quality=api["FrameQuality"].GOOD,
        detector_version="opencv-ball-locator/1.0",
    )

    assert tuple(candidate.ball_id for candidate in frame.detections[0].candidates) == (1, 9)
    assert frame.detections[0].source is api["DetectionSource"].CLASSIC_CV_CIRCLE_COLOR
    assert frame.detections[0].position.is_on_table()
    assert frame.raw_observation_count == 1

    with pytest.raises(ValidationError):
        frame.timestamp_ms = 300
    with pytest.raises(ValidationError):
        frame.detections[0].position.x_mm = 900
    with pytest.raises(ValidationError):
        api["FrameDetections"].model_validate(frame.model_dump() | {"pixel_x": 12})


def test_contract_rejects_out_of_table_points_duplicate_candidates_and_bad_confidence():
    api = _api()

    with pytest.raises(ValidationError, match="on the table"):
        api["FrameDetection"].model_validate(
            _detection(api).model_dump() | {"position": {"x_mm": 2541, "y_mm": 500}}
        )

    with pytest.raises(ValidationError, match="unique"):
        _detection(
            api,
            candidates=(
                api["BallCandidate"](ball_id=2, color=api["BallColor"].BLUE, confidence=0.7),
                api["BallCandidate"](ball_id=2, color=api["BallColor"].BLUE, confidence=0.3),
            ),
        )

    with pytest.raises(ValidationError):
        api["BallCandidate"](ball_id=16, color=api["BallColor"].UNKNOWN, confidence=0.9)
    with pytest.raises(ValidationError):
        api["BallCandidate"](ball_id=1, color=api["BallColor"].YELLOW, confidence=1.01)
    with pytest.raises(ValidationError):
        api["BallCandidate"](ball_id="1", color=api["BallColor"].YELLOW, confidence=0.9)


def test_high_confidence_view_is_derived_without_mutating_or_flattening_contract():
    api = _api()
    certain = _detection(api)
    ambiguous = _detection(
        api,
        confidence=0.94,
        candidates=(
            api["BallCandidate"](ball_id=1, color=api["BallColor"].YELLOW, confidence=0.51),
            api["BallCandidate"](ball_id=9, color=api["BallColor"].YELLOW, confidence=0.49),
        ),
    )
    low = _detection(
        api,
        confidence=0.84,
        candidates=(
            api["BallCandidate"](ball_id=8, color=api["BallColor"].BLACK, confidence=0.99),
        ),
    )
    uncertain_identity = _detection(
        api,
        confidence=0.95,
        candidates=(
            api["BallCandidate"](ball_id=7, color=api["BallColor"].MAROON, confidence=0.7),
        ),
    )
    frame = api["FrameDetections"](
        timestamp_ms=0,
        detections=(certain, ambiguous, low, uncertain_identity),
        missing_ball_ids=tuple(range(2, 7)) + tuple(range(10, 16)),
        frame_quality=api["FrameQuality"].GOOD,
        detector_version="opencv-ball-locator/1.0",
    )

    view = frame.high_confidence_positions()

    assert view == {0: api["Point"](x_mm=500, y_mm=600)}
    assert len(frame.detections) == 4
    assert len(frame.detections[1].candidates) == 2


def test_ambiguous_candidate_is_not_falsely_reported_as_located():
    api = _api()
    ambiguous = _detection(
        api,
        confidence=0.94,
        candidates=(
            api["BallCandidate"](ball_id=1, color=api["BallColor"].YELLOW, confidence=0.51),
            api["BallCandidate"](ball_id=9, color=api["BallColor"].YELLOW, confidence=0.49),
        ),
    )

    frame = api["FrameDetections"](
        timestamp_ms=0,
        detections=(ambiguous,),
        missing_ball_ids=(1, 9),
        frame_quality=api["FrameQuality"].DEGRADED,
        detector_version="opencv-ball-locator/1.0",
    )

    assert frame.high_confidence_positions() == {}
