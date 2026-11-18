"""VFR sampling tests: timestamps come from PTS, never inferred frame numbers."""

import pytest


def test_milliseconds_use_exact_half_away_from_zero_rounding():
    from billiards_ai.shared.media_timing import pts_to_milliseconds
    from billiards_ai.shared.models.media import TimeBase

    half_millisecond = TimeBase(numerator=1, denominator=2_000)

    assert pts_to_milliseconds(1, half_millisecond) == 1
    assert pts_to_milliseconds(-1, half_millisecond) == -1
    assert pts_to_milliseconds(3, half_millisecond) == 2


def test_vfr_sampling_uses_first_valid_pts_as_zero_and_chooses_at_or_after_target():
    from billiards_ai.shared.media_timing import select_frame_samples
    from billiards_ai.shared.models.media import TimeBase

    samples = select_frame_samples(
        (None, 90_000, 93_000, 99_000, 108_000),
        time_base=TimeBase(numerator=1, denominator=90_000),
        interval_ms=50,
    )

    assert [sample.requested_timestamp_ms for sample in samples] == [0, 50, 100, 150, 200]
    assert [sample.source_pts for sample in samples] == [90_000, 99_000, 99_000, 108_000, 108_000]
    assert [sample.decode_index for sample in samples] == [1, 3, 3, 4, 4]


@pytest.mark.parametrize(
    ("pts", "interval_ms", "message"),
    [
        ((None, None), 100, "valid PTS"),
        ((10, 9), 100, "strictly increasing"),
        ((10, 11), 0, "positive"),
    ],
)
def test_invalid_sampling_inputs_fail_closed(pts, interval_ms, message):
    from billiards_ai.shared.media_timing import select_frame_samples
    from billiards_ai.shared.models.media import TimeBase

    with pytest.raises(ValueError, match=message):
        select_frame_samples(
            pts,
            time_base=TimeBase(numerator=1, denominator=1_000),
            interval_ms=interval_ms,
        )
