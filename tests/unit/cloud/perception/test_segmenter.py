"""Synthetic-only acceptance tests for Story 2.6 shot segmentation."""

from collections.abc import Iterable


def _frame(timestamp_ms: int, positions: dict[int, tuple[int, int]]):
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

    colors = {0: BallColor.WHITE, 1: BallColor.YELLOW}
    detections = tuple(
        FrameDetection(
            candidates=(BallCandidate(ball_id=ball_id, color=colors[ball_id], confidence=0.99),),
            position=Point(x_mm=x_mm, y_mm=y_mm),
            confidence=0.99,
            source=DetectionSource.CLASSIC_CV_CIRCLE_COLOR,
            raw_observation=PixelObservation(
                center_px=(float(x_mm), float(y_mm)),
                radius_px=10.0,
                circularity=0.99,
                color=colors[ball_id],
            ),
        )
        for ball_id, (x_mm, y_mm) in sorted(positions.items())
    )
    return FrameDetections(
        timestamp_ms=timestamp_ms,
        detections=detections,
        missing_ball_ids=tuple(ball_id for ball_id in range(16) if ball_id not in positions),
        frame_quality=FrameQuality.GOOD if len(positions) == 2 else FrameQuality.DEGRADED,
        detector_version="synthetic/1.0",
    )


def _frames(samples: Iterable[tuple[int, dict[int, tuple[int, int]]]]):
    return tuple(_frame(timestamp_ms, positions) for timestamp_ms, positions in samples)


def _segment(samples):
    from billiards_ai.cloud.perception.segmenter import segment_shots

    return segment_shots(_frames(samples))


REST = {0: (100, 100), 1: (500, 500)}


def test_segments_multiple_shots_into_increasing_nonoverlapping_half_open_intervals():
    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (101, 100), 1: (500, 500)}),
        (600, {0: (110, 100), 1: (500, 500)}),
        (700, {0: (130, 100), 1: (510, 500)}),
        (800, {0: (150, 100), 1: (530, 500)}),
    ]
    first_rest = {0: (150, 100), 1: (530, 500)}
    samples += [(time, first_rest) for time in range(900, 1600, 100)]
    samples += [
        (1600, {0: (151, 100), 1: (530, 500)}),
        (1700, {0: (160, 100), 1: (530, 500)}),
        (1800, {0: (180, 100), 1: (540, 500)}),
    ]
    second_rest = {0: (180, 100), 1: (540, 500)}
    samples += [(time, second_rest) for time in range(1900, 2400, 100)]

    result = _segment(samples)

    assert result.failures == ()
    assert tuple((item.t_start_ms, item.t_end_ms) for item in result.segments) == (
        (500, 900),
        (1600, 1900),
    )
    assert result.segments[0].ball_positions_before.timestamp_ms == 400
    assert result.segments[0].ball_positions_before.as_dict()[0].x_mm == 100
    assert result.segments[0].ball_positions_after.timestamp_ms == 900
    assert result.segments[0].ball_positions_after.as_dict()[1].x_mm == 530
    assert result.segments[1].ball_positions_before.timestamp_ms == 1500


def test_detects_slow_roll_by_displacement_across_the_stability_window():
    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (101, 100), 1: (500, 500)}),
        (600, {0: (102, 100), 1: (500, 500)}),
        (700, {0: (103, 100), 1: (500, 500)}),
        (800, {0: (104, 100), 1: (500, 500)}),
        (900, {0: (105, 100), 1: (500, 500)}),
    ]
    settled = {0: (105, 100), 1: (500, 500)}
    samples += [(time, settled) for time in range(1000, 1500, 100)]

    result = _segment(samples)

    assert tuple((item.t_start_ms, item.t_end_ms) for item in result.segments) == ((500, 800),)
    assert result.failures == ()


def test_short_occlusion_stays_unknown_and_preserves_deterministic_boundaries():
    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (101, 100), 1: (500, 500)}),
        (600, {0: (110, 100), 1: (500, 500)}),
        (700, {1: (500, 500)}),
        (800, {0: (130, 100), 1: (510, 500)}),
    ]
    settled = {0: (130, 100), 1: (510, 500)}
    samples += [(time, settled) for time in range(900, 1400, 100)]

    first = _segment(samples)
    second = _segment(samples)

    assert first == second
    assert tuple((item.t_start_ms, item.t_end_ms) for item in first.segments) == ((500, 900),)
    assert first.failures == ()


def test_ball_missing_after_motion_can_form_an_auditable_potted_terminal_snapshot():
    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (110, 100), 1: (500, 500)}),
        (600, {0: (130, 100)}),
        (700, {0: (150, 100)}),
        (800, {0: (150, 100)}),
        (900, {0: (150, 100)}),
        (1000, {0: (150, 100)}),
        (1100, {0: (150, 100)}),
        (1200, {0: (150, 100)}),
    ]

    result = _segment(samples)

    assert result.failures == ()
    assert len(result.segments) == 1
    assert result.segments[0].ball_positions_before.as_dict().keys() == {0, 1}
    assert result.segments[0].ball_positions_after.as_dict().keys() == {0}
    assert result.segments[0].t_end_ms == 800


def test_long_ball_detection_gap_returns_typed_failure_without_guessing_boundary():
    from billiards_ai.shared.constants import MAX_DETECTION_GAP_MS
    from billiards_ai.shared.models.segmentation import SegmentationFailureReason

    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (110, 100), 1: (500, 500)}),
        (600, {0: (130, 100)}),
        (600 + MAX_DETECTION_GAP_MS, {0: (150, 100)}),
        (601 + MAX_DETECTION_GAP_MS, {0: (150, 100)}),
    ]

    result = _segment(samples)

    assert result.segments == ()
    assert len(result.failures) == 1
    assert result.failures[0].reason is SegmentationFailureReason.LONG_DETECTION_GAP
    assert result.failures[0].missing_ball_ids == (1,)
    assert result.failures[0].t_start_ms == 600
    assert result.failures[0].t_end_ms == 601 + MAX_DETECTION_GAP_MS


def test_video_starting_in_motion_returns_typed_failure():
    from billiards_ai.shared.models.segmentation import SegmentationFailureReason

    result = _segment(
        [
            (0, REST),
            (100, {0: (101, 100), 1: (500, 500)}),
            (200, {0: (104, 100), 1: (500, 500)}),
            (300, {0: (110, 100), 1: (500, 500)}),
            (400, {0: (120, 100), 1: (500, 500)}),
        ]
    )

    assert result.segments == ()
    assert result.failures[0].reason is SegmentationFailureReason.STARTED_IN_MOTION


def test_video_ending_before_all_balls_stop_returns_typed_failure():
    from billiards_ai.shared.models.segmentation import SegmentationFailureReason

    samples = [(time, REST) for time in range(0, 500, 100)]
    samples += [
        (500, {0: (101, 100), 1: (500, 500)}),
        (600, {0: (110, 100), 1: (500, 500)}),
        (700, {0: (130, 100), 1: (510, 500)}),
    ]

    result = _segment(samples)

    assert result.segments == ()
    assert result.failures[0].reason is SegmentationFailureReason.ENDED_IN_MOTION
    assert result.failures[0].t_start_ms == 500
    assert result.failures[0].t_end_ms == 700


def test_shot_over_maximum_duration_is_failure_and_never_a_segment():
    from billiards_ai.shared.constants import MAX_SHOT_DURATION_MS
    from billiards_ai.shared.models.segmentation import SegmentationFailureReason

    samples = [(time, REST) for time in range(0, 500, 100)]
    timestamp_ms = 500
    while timestamp_ms <= 500 + MAX_SHOT_DURATION_MS + 100:
        samples.append((timestamp_ms, {0: (100 + timestamp_ms // 100, 100), 1: (500, 500)}))
        timestamp_ms += 100

    result = _segment(samples)

    assert result.segments == ()
    assert len(result.failures) == 1
    assert result.failures[0].reason is SegmentationFailureReason.MAX_SHOT_DURATION_EXCEEDED
    assert result.failures[0].t_start_ms == 500
    assert result.failures[0].t_end_ms == 500 + MAX_SHOT_DURATION_MS + 100


def test_stillness_confirmation_time_does_not_count_toward_shot_duration():
    from billiards_ai.shared.constants import MAX_SHOT_DURATION_MS

    samples = [(time, REST) for time in range(0, 500, 100)]
    stable_start = 500 + MAX_SHOT_DURATION_MS - 200
    timestamp_ms = 500
    cue_x = REST[0][0]
    while timestamp_ms < stable_start:
        cue_x += 4
        samples.append((timestamp_ms, {0: (cue_x, 100), 1: (500, 500)}))
        timestamp_ms += 100
    settled = {0: (cue_x, 100), 1: (500, 500)}
    while timestamp_ms <= stable_start + 400:
        samples.append((timestamp_ms, settled))
        timestamp_ms += 100

    result = _segment(samples)

    assert result.failures == ()
    assert len(result.segments) == 1
    assert result.segments[0].t_end_ms - result.segments[0].t_start_ms <= MAX_SHOT_DURATION_MS


def test_non_increasing_timestamps_return_typed_failure():
    from billiards_ai.shared.models.segmentation import SegmentationFailureReason

    result = _segment([(0, REST), (100, REST), (100, REST)])

    assert result.segments == ()
    assert result.failures[0].reason is SegmentationFailureReason.NON_INCREASING_TIMESTAMPS
