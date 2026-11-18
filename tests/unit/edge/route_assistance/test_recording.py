from __future__ import annotations

import pytest

from billiards_ai.edge.route_assistance.recording import (
    RecommendationPower,
    ShotBoundary,
    ShotBoundaryKind,
    ShotTimelineIndex,
    build_media_range,
    load_recommendation_power,
)
from billiards_ai.lab.field_assistance.live_state import (
    ReplayTimeline,
    ReplayTimelineFrame,
)


def _timeline(*payloads: dict[str, object], duration_ms: int = 10_000) -> ReplayTimeline:
    return ReplayTimeline(
        frames=tuple(
            ReplayTimelineFrame(
                frame_id=index,
                timestamp_ms=int(payload["timestamp_ms"]),
                payload={"frame_id": index, **payload},
            )
            for index, payload in enumerate(payloads)
        ),
        duration_ms=duration_ms,
    )


def _one_shot_timeline() -> ReplayTimeline:
    return _timeline(
        {
            "timestamp_ms": 0,
            "shot_sequence": 0,
            "recommendation_id": "recommendation-1",
            "effects": ["SHOW_GUIDANCE"],
        },
        {
            "timestamp_ms": 3_000,
            "shot_sequence": 0,
            "recommendation_id": None,
            "effects": ["CLEAR_GUIDANCE", "SHOT_STARTED"],
        },
        {
            "timestamp_ms": 5_000,
            "shot_sequence": 1,
            "recommendation_id": None,
            "effects": ["SHOT_SETTLED", "REQUEST_ANALYSIS"],
        },
    )


def test_indexes_one_shot_from_existing_authoritative_effects() -> None:
    index = ShotTimelineIndex.from_timeline(_one_shot_timeline())

    assert index.boundaries == (
        ShotBoundary(
            kind=ShotBoundaryKind.STARTED,
            sequence=1,
            timestamp_ms=3_000,
            recommendation_id="recommendation-1",
        ),
        ShotBoundary(
            kind=ShotBoundaryKind.SETTLED,
            sequence=1,
            timestamp_ms=5_000,
            recommendation_id="recommendation-1",
        ),
    )


def test_boundary_window_is_exclusive_then_inclusive() -> None:
    index = ShotTimelineIndex.from_timeline(_one_shot_timeline())

    assert index.boundaries_between(after_ms=-1, through_ms=3_000) == (index.boundaries[0],)
    assert index.boundaries_between(after_ms=3_000, through_ms=5_000) == (index.boundaries[1],)
    assert index.boundaries_between(after_ms=5_000, through_ms=5_000) == ()


@pytest.mark.parametrize(
    "timeline",
    [
        _timeline(
            {
                "timestamp_ms": 5_000,
                "shot_sequence": 1,
                "recommendation_id": None,
                "effects": ["SHOT_SETTLED"],
            }
        ),
        _timeline(
            {
                "timestamp_ms": 3_000,
                "shot_sequence": 0,
                "recommendation_id": "recommendation-1",
                "effects": ["SHOT_STARTED", "SHOT_STARTED"],
            },
            {
                "timestamp_ms": 5_000,
                "shot_sequence": 1,
                "recommendation_id": None,
                "effects": ["SHOT_SETTLED"],
            },
        ),
    ],
)
def test_rejects_orphan_or_duplicate_authoritative_effects(
    timeline: ReplayTimeline,
) -> None:
    with pytest.raises(ValueError, match="authoritative shot effects"):
        ShotTimelineIndex.from_timeline(timeline)


def test_drops_unsettled_trailing_shot_without_emitting_partial_boundaries() -> None:
    timeline = _timeline(
        {
            "timestamp_ms": 3_000,
            "shot_sequence": 0,
            "recommendation_id": "recommendation-1",
            "effects": ["SHOT_STARTED"],
        }
    )

    assert ShotTimelineIndex.from_timeline(timeline).boundaries == ()


def test_keeps_completed_shots_before_an_unsettled_trailing_shot() -> None:
    timeline = _timeline(
        {
            "timestamp_ms": 1_000,
            "shot_sequence": 0,
            "recommendation_id": "recommendation-1",
            "effects": ["SHOT_STARTED"],
        },
        {
            "timestamp_ms": 2_000,
            "shot_sequence": 1,
            "recommendation_id": None,
            "effects": ["SHOT_SETTLED"],
        },
        {
            "timestamp_ms": 3_000,
            "shot_sequence": 1,
            "recommendation_id": "recommendation-2",
            "effects": ["SHOT_STARTED"],
        },
    )

    assert ShotTimelineIndex.from_timeline(timeline).boundaries == (
        ShotBoundary(
            kind=ShotBoundaryKind.STARTED,
            sequence=1,
            timestamp_ms=1_000,
            recommendation_id="recommendation-1",
        ),
        ShotBoundary(
            kind=ShotBoundaryKind.SETTLED,
            sequence=1,
            timestamp_ms=2_000,
            recommendation_id="recommendation-1",
        ),
    )


def test_media_range_clamps_two_second_padding_to_video() -> None:
    media = build_media_range(
        artifact_id="artifact-1",
        started_ms=1_000,
        settled_ms=9_500,
        duration_ms=10_000,
    )

    assert media.start_ms == 0
    assert media.end_ms == 10_000
    assert media.source_locator == "presentation.mp4"
    assert media.source_kind == "development_replay"


def test_media_range_rejects_invalid_authoritative_times() -> None:
    with pytest.raises(ValueError, match="shot timing"):
        build_media_range(
            artifact_id="artifact-1",
            started_ms=5_000,
            settled_ms=5_000,
            duration_ms=10_000,
        )


def test_loads_generic_power_by_recommendation_id(tmp_path) -> None:
    path = tmp_path / "recommendations.json"
    path.write_text(
        """{"schema_version":"1.0","generations":[{"display":{"recommendation_id":"recommendation-1","power":{"band":"medium_light","approximate_percent":40,"personalized":false}}}]}""",
        encoding="utf-8",
    )

    assert load_recommendation_power(path) == {
        "recommendation-1": RecommendationPower(
            band="medium_light",
            approximate_percent=40,
        )
    }


def test_rejects_personalized_or_duplicate_recommendation_power(tmp_path) -> None:
    path = tmp_path / "recommendations.json"
    path.write_text(
        """{"schema_version":"1.0","generations":[{"display":{"recommendation_id":"recommendation-1","power":{"band":"medium","approximate_percent":50,"personalized":true}}}]}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recommendation power"):
        load_recommendation_power(path)
