from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from billiards_ai.lab.mainline1_realtime_refinement import RealtimeShotRefiner
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    TrackedBall,
    TrackedObservation,
)
from billiards_ai.lab.shot_local.models import BallGroup


@dataclass(frozen=True)
class _Frame:
    capture_frame_id: int
    captured_at_ns: int


class _Interval:
    def __init__(self, frame_ids: tuple[int, ...]) -> None:
        self.start_frame_id = frame_ids[0]
        self.end_frame_id = frame_ids[-1]
        self.frame_count = len(frame_ids)
        self.segment_sha256 = {"segment": "a" * 64}
        self._rows = tuple(_Frame(frame_id, frame_id * 16_666_667) for frame_id in frame_ids)

    def frames(self):
        return iter(self._rows)


def _tracked_rows(frame_ids: tuple[int, ...]) -> tuple[TrackedObservation, ...]:
    return tuple(
        TrackedObservation(
            capture_frame_id=frame_id,
            captured_at_ns=frame_id * 16_666_667,
            balls=(
                TrackedBall(
                    track_id="ball-0",
                    group=BallGroup.SOLID,
                    state="confirmed",
                    provenance="anchor" if frame_id in {frame_ids[0], frame_ids[-1]} else "flow",
                    center_px=(40.0 + frame_id, 60.0),
                    table_mm=(400 + frame_id * 10, 600),
                    confidence=0.9,
                    anchor_frame_id=frame_ids[0],
                ),
            ),
        )
        for frame_id in frame_ids
    )


def _anchors(frame_ids: tuple[int, ...]) -> tuple[AnchorObservation, ...]:
    return tuple(
        AnchorObservation(
            capture_frame_id=frame_id,
            captured_at_ns=frame_id * 16_666_667,
            balls=(),
        )
        for frame_id in frame_ids
    )


def test_missing_source_frame_abstains_entire_shot() -> None:
    result = RealtimeShotRefiner().refine(
        interval=_Interval((0, 1, 3, 4)),
        tracked=_tracked_rows((0, 1, 3, 4)),
        anchors=_anchors((0, 4)),
    )

    assert result.publishable is False
    assert result.shot_abstention_reason == "source_frame_gap"


def test_sparse_anchor_coverage_abstains_contact_but_keeps_table_states() -> None:
    frame_ids = tuple(range(39))
    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids),
        tracked=_tracked_rows(frame_ids),
        anchors=_anchors((0, 8, 30, 38)),
    )

    assert result.publishable is True
    assert result.fact_coverage["table_state_after"].status == "observed"
    assert result.fact_coverage["first_contact"].status == "abstained"
    assert result.fact_coverage["first_contact"].reason == "anchor_gap_exceeded"


def test_lost_samples_abstain_motion_facts_but_are_not_emitted_as_positions() -> None:
    frame_ids = tuple(range(5))
    rows = list(_tracked_rows(frame_ids))
    lost = rows[2].balls[0]
    rows[2] = TrackedObservation(
        capture_frame_id=2,
        captured_at_ns=2 * 16_666_667,
        balls=(
            TrackedBall(
                track_id=lost.track_id,
                group=lost.group,
                state="lost",
                provenance="flow",
                center_px=None,
                table_mm=None,
                confidence=0,
                anchor_frame_id=0,
            ),
        ),
    )

    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids),
        tracked=tuple(rows),
        anchors=_anchors((0, 4)),
    )

    assert result.publishable is True
    assert result.fact_coverage["first_contact"].reason == "tracker_uncertain"
    observations = result.observations()
    assert observations[2].observations == ()


def test_refinement_preserves_stable_track_identity_for_rules() -> None:
    frame_ids = tuple(range(100, 221))

    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids),
        tracked=_tracked_rows(frame_ids),
        anchors=_anchors(tuple(range(100, 221, 8))),
    )

    frames = result.rule_frames()
    assert frames[0].balls[0].identity == "solid:ball-0"
    assert frames[0].balls[0].track_id == "ball-0"
    assert frames[-1].balls[0].track_id == "ball-0"
    assert result.coverage.source_frame_ids == tuple(range(100, 221))


def test_potential_pocket_disappearance_is_not_globally_denied_by_tracker_loss() -> None:
    frame_ids = tuple(range(5))
    rows = list(_tracked_rows(frame_ids))
    seen = rows[1].balls[0]
    rows[1] = TrackedObservation(
        capture_frame_id=1,
        captured_at_ns=1 * 16_666_667,
        balls=(
            TrackedBall(
                track_id=seen.track_id,
                group=seen.group,
                state="confirmed",
                provenance="flow",
                center_px=(2.0, 2.0),
                table_mm=(20, 20),
                confidence=0.9,
                anchor_frame_id=0,
            ),
        ),
    )
    for frame_id in (2, 3, 4):
        rows[frame_id] = TrackedObservation(
            capture_frame_id=frame_id,
            captured_at_ns=frame_id * 16_666_667,
            balls=(
                TrackedBall(
                    track_id=seen.track_id,
                    group=seen.group,
                    state="lost",
                    provenance="flow",
                    center_px=None,
                    table_mm=None,
                    confidence=0,
                    anchor_frame_id=0,
                ),
            ),
        )

    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids),
        tracked=tuple(rows),
        anchors=_anchors((0, 4)),
    )

    assert result.fact_coverage["first_contact"].reason == "tracker_uncertain"
    assert result.fact_coverage["cushion_contacts"].reason == "tracker_uncertain"
    assert result.fact_coverage["pots"].status == "observed"
    assert result.coverage.uncertain_track_states == (
        (2, "ball-0", "lost"),
        (3, "ball-0", "lost"),
        (4, "ball-0", "lost"),
    )


def test_anchor_roster_drop_and_recovery_remain_uncertain_for_the_whole_interval() -> None:
    frame_ids = tuple(range(5))
    rows = list(_tracked_rows(frame_ids))
    rows[2] = replace(rows[2], balls=())

    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids), tracked=rows, anchors=_anchors((0, 2, 4))
    )

    assert result.publishable is True
    assert result.fact_coverage["first_contact"].reason == "tracker_uncertain"
    assert result.fact_coverage["cushion_contacts"].reason == "tracker_uncertain"
    assert result.coverage.uncertain_frame_ids == (2,)
    assert result.coverage.uncertain_track_states == ((2, "ball-0", "lost"),)
    # A trusted later position does not erase the missing observation.
    assert result.rule_frames()[2].balls == ()
    assert result.rule_frames()[3].balls[0].track_id == "ball-0"


@pytest.mark.parametrize("change", ["added", "replaced", "regrouped"])
def test_unmatched_anchor_identity_changes_abstain_motion_coverage(change: str) -> None:
    frame_ids = tuple(range(5))
    rows = list(_tracked_rows(frame_ids))
    for frame_id in range(2, 5):
        original = rows[frame_id].balls[0]
        changed = replace(
            original,
            track_id="new-ball" if change != "regrouped" else original.track_id,
            group=BallGroup.STRIPE,
        )
        rows[frame_id] = replace(
            rows[frame_id],
            balls=(original, changed) if change == "added" else (changed,),
        )

    result = RealtimeShotRefiner().refine(
        interval=_Interval(frame_ids), tracked=rows, anchors=_anchors((0, 2, 4))
    )

    ambiguous_id = "ball-0" if change == "regrouped" else "new-ball"
    assert (2, ambiguous_id, "ambiguous") in result.coverage.uncertain_track_states
    if change == "replaced":
        assert (2, "ball-0", "lost") in result.coverage.uncertain_track_states
    assert result.fact_coverage["first_contact"].reason == "tracker_uncertain"
    assert result.fact_coverage["cushion_contacts"].reason == "tracker_uncertain"


def test_past_roster_loss_does_not_poison_a_later_stable_interval() -> None:
    frame_ids = tuple(range(5))
    rows = list(_tracked_rows(frame_ids))
    for frame_id, row in enumerate(rows):
        survivor = replace(row.balls[0], track_id="survivor")
        rows[frame_id] = replace(row, balls=(*row.balls, survivor) if frame_id < 2 else (survivor,))
    refiner = RealtimeShotRefiner()
    earlier = refiner.refine(
        interval=_Interval(frame_ids), tracked=rows, anchors=_anchors((0, 2, 4))
    )
    later_ids = tuple(range(5, 10))
    later = refiner.refine(
        interval=_Interval(later_ids),
        tracked=tuple(
            replace(row, balls=(replace(row.balls[0], track_id="survivor"),))
            for row in _tracked_rows(later_ids)
        ),
        anchors=_anchors((5, 9)),
    )

    assert earlier.fact_coverage["first_contact"].reason == "tracker_uncertain"
    assert earlier.fact_coverage["pots"].status == "observed"
    assert later.coverage.uncertain_track_states == ()
    assert later.fact_coverage["first_contact"].status == "observed"
