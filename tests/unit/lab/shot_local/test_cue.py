from __future__ import annotations

from billiards_ai.lab.shot_local.cue import CueTrackingConfig, track_cue
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    LocalState,
    ObservationFrame,
    ShotWindow,
)


def _window() -> ShotWindow:
    return ShotWindow(
        shot_id="shot-0001",
        before_frame_id=1,
        start_frame_id=2,
        end_frame_id=4,
        after_frame_id=4,
        start_ms=80,
        end_ms=160,
        complete=True,
    )


def _config() -> CueTrackingConfig:
    return CueTrackingConfig(ambiguity_margin=0.08, reacquire_confidence_min=0.55)


def _cue(
    detection_id: str,
    x_mm: int,
    *,
    detector_confidence: float = 0.9,
    group_confidence: float = 0.9,
) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=200,
        confidence=detector_confidence,
        group=BallGroup.CUE,
        group_confidence=group_confidence,
    )


def _solid(detection_id: str, x_mm: int) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=200,
        confidence=0.95,
        group=BallGroup.SOLID,
        group_confidence=0.95,
    )


def _frame(frame_id: int, *observations) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        observations=tuple(observations),
    )


def test_missing_cue_does_not_repeat_last_confirmed_position() -> None:
    frames = (
        _frame(2, _cue("cue-2", 100)),
        _frame(3, _solid("solid-3", 400)),
        _frame(4, _cue("cue-4", 300)),
    )

    samples = track_cue(_window(), frames, _config())

    missing = samples[1]
    assert missing.state is LocalState.LOST
    assert missing.x_mm is None
    assert missing.y_mm is None
    assert missing.source_detection_id is None


def test_unique_cue_after_loss_is_reacquired() -> None:
    frames = (
        _frame(2, _cue("cue-2", 100)),
        _frame(3),
        _frame(4, _cue("cue-4", 300)),
    )

    samples = track_cue(_window(), frames, _config())

    assert [sample.state for sample in samples] == [
        LocalState.CONFIRMED,
        LocalState.LOST,
        LocalState.REACQUIRED,
    ]
    assert all(sample.local_id == "shot-0001-cue" for sample in samples)


def test_two_equivalent_cue_candidates_are_ambiguous_without_coordinates() -> None:
    frames = (
        _frame(2, _cue("cue-2", 100)),
        _frame(
            3,
            _cue("cue-3a", 200, group_confidence=0.90),
            _cue("cue-3b", 500, group_confidence=0.86),
        ),
        _frame(4, _cue("cue-4", 300)),
    )

    sample = track_cue(_window(), frames, _config())[1]

    assert sample.state is LocalState.AMBIGUOUS
    assert sample.x_mm is None
    assert sample.y_mm is None
    assert sample.source_detection_id is None


def test_low_confidence_cue_candidate_fails_closed_as_lost() -> None:
    frames = (
        _frame(2, _cue("cue-2", 100)),
        _frame(3, _cue("cue-3", 200, detector_confidence=0.5, group_confidence=0.5)),
        _frame(4, _cue("cue-4", 300)),
    )

    sample = track_cue(_window(), frames, _config())[1]

    assert sample.state is LocalState.LOST
    assert sample.confidence == 0
