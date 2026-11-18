from __future__ import annotations

from dataclasses import replace

import pytest

from billiards_ai.edge.realtime_chain.models import (
    EvidenceSource,
    EvidenceStatus,
    HabitFeature,
    ProfileIncrement,
)
from billiards_ai.edge.realtime_chain.profile import aggregate_profile


def _source(suffix: str) -> EvidenceSource:
    return EvidenceSource(
        artifact_id=f"artifact-{suffix}",
        source_video_sha256=suffix * 64,
        manifest_sha256=("f" if suffix != "f" else "e") * 64,
    )


def _increment(
    *,
    session_id: str = "session-1",
    shot_id: str = "shot-0001",
    status: EvidenceStatus = EvidenceStatus.ADMITTED,
    source: EvidenceSource | None = None,
) -> ProfileIncrement:
    features = (
        HabitFeature(
            code="cue_travel_mm",
            value=100.0,
            unit="mm",
            evidence_ids=(f"cue:{shot_id}",),
        ),
        HabitFeature(
            code="first_contact_target_group",
            value="solid",
            unit="category",
            evidence_ids=(f"contact:{shot_id}",),
        ),
        HabitFeature(
            code="observed_pot_count",
            value=1.0,
            unit="count_not_success",
            evidence_ids=(f"pot:{shot_id}",),
        ),
    )
    return ProfileIncrement(
        user_id="user-1",
        session_id=session_id,
        shot_id=shot_id,
        status=status,
        source=source or _source("a"),
        features=features if status is EvidenceStatus.ADMITTED else (),
    )


def _metrics(aggregate):
    return {metric.code: metric for metric in aggregate.metrics}


def test_aggregates_admitted_features_across_sessions_with_exact_lineage() -> None:
    second = replace(
        _increment(session_id="session-2", shot_id="shot-0002", source=_source("c")),
        features=(
            HabitFeature("cue_travel_mm", 300.0, "mm", ("cue:shot-0002",)),
            HabitFeature("first_contact_target_group", "stripe", "category", ("c2",)),
        ),
    )

    aggregate = aggregate_profile("user-1", (_increment(), second))
    metrics = _metrics(aggregate)

    assert aggregate.distinct_session_count == 2
    assert aggregate.admitted_shot_count == 2
    assert metrics["cue_travel_mm"].sample_count == 2
    assert metrics["cue_travel_mm"].mean == 200.0
    assert metrics["first_contact_target_group"].category_counts == (
        ("solid", 1),
        ("stripe", 1),
    )
    assert metrics["observed_pot_count"].total == 1.0
    assert len(aggregate.contributions) == 5
    assert {item.session_id for item in aggregate.contributions} == {"session-1", "session-2"}


def test_pending_and_abstained_shots_are_counted_but_never_contribute_metrics() -> None:
    aggregate = aggregate_profile(
        "user-1",
        (
            _increment(),
            _increment(
                shot_id="shot-0002",
                status=EvidenceStatus.PENDING,
                source=_source("c"),
            ),
            _increment(
                shot_id="shot-0003",
                status=EvidenceStatus.ABSTAINED,
                source=_source("d"),
            ),
        ),
    )

    assert aggregate.admitted_shot_count == 1
    assert aggregate.pending_shot_count == 1
    assert aggregate.abstained_shot_count == 1
    assert _metrics(aggregate)["cue_travel_mm"].sample_count == 1
    assert {item.shot_id for item in aggregate.contributions} == {"shot-0001"}


def test_duplicate_source_shot_or_foreign_user_fails_closed() -> None:
    increment = _increment()
    with pytest.raises(ValueError, match="duplicate profile evidence"):
        aggregate_profile("user-1", (increment, increment))
    with pytest.raises(ValueError, match="same user"):
        aggregate_profile("user-1", (replace(increment, user_id="user-2"),))


def test_same_metric_with_conflicting_units_fails_closed() -> None:
    first = _increment()
    conflicting = replace(
        _increment(shot_id="shot-0002", source=_source("c")),
        features=(HabitFeature("cue_travel_mm", 1.0, "meters", ("cue:2",)),),
    )

    with pytest.raises(ValueError, match="conflicting units"):
        aggregate_profile("user-1", (first, conflicting))
