from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance import rule_facts as fact_module
from billiards_ai.edge.route_assistance.control import (
    CueBallPlacement,
    MatchControlSnapshot,
    RackPhase,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.eight_ball_rules import (
    PROFILE_ID,
    RuleOutcomeKind,
    decide_eight_ball,
)
from billiards_ai.edge.route_assistance.rule_facts import (
    FactTruth,
    RuleFactDeadlineExceeded,
    build_realtime_rule_facts,
)
from billiards_ai.lab.events.models import BallState, TrackedFrame
from billiards_ai.lab.mainline1_realtime_refinement import (
    FactCoverageVerdict,
    RealtimeEvidenceCoverage,
    RealtimeShotRefiner,
)
from billiards_ai.lab.mainline1_realtime_tracker import (
    AnchorObservation,
    TrackedBall,
    TrackedObservation,
)
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import BallGroup


def _config() -> ShotLocalEventConfig:
    return ShotLocalEventConfig(
        first_contact_distance_mm=60,
        first_contact_ambiguity_mm=3,
        cue_motion_min_mm=5,
        contact_time_tolerance_ms=20,
        cushion_distance_mm=40,
        pocket_distance_mm=90,
        pot_missing_frames=2,
        after_match_distance_mm=50,
        table_length_mm=2540,
        table_width_mm=1270,
    )


def _config_digest(config: ShotLocalEventConfig) -> str:
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ball(identity: str, track_id: str, x: int, y: int) -> BallState:
    return BallState(identity, x, y, 0.95, track_id=track_id)


def _frame(frame_id: int, *balls: BallState) -> TrackedFrame:
    return TrackedFrame(frame_id, frame_id * 17, tuple(balls))


def _shot(
    frames: tuple[TrackedFrame, ...],
    *,
    uncertain: tuple[tuple[int, str, str], ...] = (),
    source_frame_ids: tuple[int, ...] | None = None,
    fact_coverage: dict[str, FactCoverageVerdict] | None = None,
    event_config: ShotLocalEventConfig | None = None,
    event_config_digest: str | None = None,
) -> SimpleNamespace:
    config = _config() if event_config is None else event_config
    source_ids = (
        tuple(frame.frame_id for frame in frames) if source_frame_ids is None else source_frame_ids
    )
    coverage = RealtimeEvidenceCoverage(
        source_frame_ids=source_ids,
        anchor_frame_ids=(source_ids[0], source_ids[-1]),
        maximum_anchor_gap_ms=100,
        uncertain_frame_ids=tuple(sorted({row[0] for row in uncertain})),
        segment_sha256={"segment-0": "a" * 64},
        uncertain_track_states=uncertain,
    )
    return SimpleNamespace(
        shot_id="shot-00000001",
        start_frame_id=1,
        settled_frame_id=4,
        settled_at_ns=68_000_000,
        observations=(),
        publishable=True,
        shot_abstention_reason=None,
        fact_coverage=fact_coverage
        or {
            name: FactCoverageVerdict("observed")
            for name in (
                "table_state_before",
                "table_state_after",
                "first_contact",
                "cushion_contacts",
                "pots",
            )
        },
        source_epoch_id="epoch-1",
        job_id="job-1",
        rule_frames=frames,
        rule_coverage=coverage,
        event_config=config,
        event_config_digest=(
            _config_digest(config) if event_config_digest is None else event_config_digest
        ),
    )


def _facts(shot: SimpleNamespace, *, monotonic_ns=lambda: 0, deadline_ns=10**12):
    return build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        deadline_ns=deadline_ns,
        monotonic_ns=monotonic_ns,
    )


def test_ball_in_hand_is_verified_from_the_trusted_before_frame() -> None:
    shot = _shot(
        (
            _frame(1, _ball("cue", "cue-1", 500, 635), _ball("solid:1", "1", 900, 635)),
            _frame(2, _ball("cue", "cue-1", 520, 635), _ball("solid:1", "1", 900, 635)),
            _frame(3, _ball("cue", "cue-1", 540, 635), _ball("solid:1", "1", 900, 635)),
            _frame(4, _ball("cue", "cue-1", 560, 635), _ball("solid:1", "1", 900, 635)),
        )
    )

    anywhere = build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        required_cue_ball_placement=CueBallPlacement.ANYWHERE,
        deadline_ns=10**12,
        monotonic_ns=lambda: 0,
    )
    behind_line = build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        required_cue_ball_placement=CueBallPlacement.BEHIND_LINE,
        deadline_ns=10**12,
        monotonic_ns=lambda: 0,
    )

    assert anywhere.cue_ball_placement_observed is FactTruth.TRUE
    assert behind_line.cue_ball_placement_observed is FactTruth.TRUE


def test_ball_in_hand_abstains_when_the_trusted_before_cue_is_missing() -> None:
    shot = _shot(
        (
            _frame(1, _ball("solid:1", "1", 900, 635)),
            _frame(2, _ball("solid:1", "1", 900, 635)),
            _frame(3, _ball("solid:1", "1", 900, 635)),
            _frame(4, _ball("solid:1", "1", 900, 635)),
        )
    )

    facts = build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        required_cue_ball_placement=CueBallPlacement.ANYWHERE,
        deadline_ns=10**12,
        monotonic_ns=lambda: 0,
    )

    assert facts.cue_ball_placement_observed is FactTruth.UNKNOWN
    assert "cue_ball_placement_unverified" in facts.abstention_reasons


def test_behind_line_placement_abstains_when_cue_is_beyond_head_string() -> None:
    shot = _shot(
        (
            _frame(1, _ball("cue", "cue-1", 900, 635), _ball("solid:1", "1", 1200, 635)),
            _frame(2, _ball("cue", "cue-1", 920, 635), _ball("solid:1", "1", 1200, 635)),
            _frame(3, _ball("cue", "cue-1", 940, 635), _ball("solid:1", "1", 1200, 635)),
            _frame(4, _ball("cue", "cue-1", 960, 635), _ball("solid:1", "1", 1200, 635)),
        )
    )

    facts = build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="cbsa-chinese-pool-2017-observable-v1",
        required_cue_ball_placement=CueBallPlacement.BEHIND_LINE,
        deadline_ns=10**12,
        monotonic_ns=lambda: 0,
    )

    assert facts.cue_ball_placement_observed is FactTruth.FALSE
    assert "cue_ball_placement_unverified" in facts.abstention_reasons


@pytest.mark.parametrize(
    ("identity", "track_id", "group"),
    [
        ("cue", "cue-track", BallGroup.CUE),
        ("eight", "eight-track", BallGroup.EIGHT),
        ("solid:track-3", "track-3", BallGroup.SOLID),
    ],
)
@pytest.mark.parametrize("loss_source", ["tracker", "anchor_roster"])
def test_near_pocket_persistent_disappearance_is_explicit_pot(
    identity: str,
    track_id: str,
    group: BallGroup,
    loss_source: str,
) -> None:
    frames = (
        _frame(0, _ball(identity, track_id, 50, 50)),
        _frame(1, _ball(identity, track_id, 20, 20)),
        _frame(2),
        _frame(3),
        _frame(4),
        _frame(5),
    )
    uncertain = tuple((frame_id, track_id, "lost") for frame_id in range(2, 6))

    shot = _shot(frames, uncertain=uncertain)
    if loss_source == "anchor_roster":
        source = tuple(
            SimpleNamespace(capture_frame_id=row.frame_id, captured_at_ns=row.timestamp_ms * 10**6)
            for row in frames
        )
        refined = RealtimeShotRefiner().refine(
            interval=SimpleNamespace(
                start_frame_id=0,
                end_frame_id=5,
                frame_count=6,
                segment_sha256={"unit-segment": "a" * 64},
                frames=lambda: iter(source),
            ),
            tracked=tuple(
                TrackedObservation(
                    capture_frame_id=row.frame_id,
                    captured_at_ns=row.timestamp_ms * 10**6,
                    balls=tuple(
                        TrackedBall(
                            track_id=ball.track_id,
                            group=group,
                            state="confirmed",
                            provenance="anchor",
                            center_px=(float(ball.x_mm), float(ball.y_mm)),
                            table_mm=(ball.x_mm, ball.y_mm),
                            confidence=ball.confidence,
                            anchor_frame_id=row.frame_id,
                        )
                        for ball in row.balls
                    ),
                )
                for row in frames
            ),
            anchors=(AnchorObservation(0, 0, ()), AnchorObservation(5, 85_000_000, ())),
        )
        assert refined.coverage.uncertain_track_states == ((2, track_id, "lost"),)
        assert refined.fact_coverage["first_contact"].reason == "tracker_uncertain"
        shot.rule_coverage = refined.coverage
        shot.fact_coverage = refined.fact_coverage

    facts = _facts(shot)

    assert [(pot.track_id, pot.group) for pot in facts.pots] == [(track_id, group)]
    assert facts.pots[0].pocket_id == "top-left"
    assert facts.pots[0].last_seen_frame_id == 1
    assert facts.pots[0].first_missing_frame_id == 2
    assert facts.pots[0].confirmed_through_frame_id == 5
    assert facts.pots[0].evidence_frame_ids == (1, 2, 3, 5)
    assert facts.explicit_no_pot is FactTruth.FALSE
    assert facts.cue_ball_scratch is (
        FactTruth.TRUE if group is BallGroup.CUE else FactTruth.UNKNOWN
    )


def test_explicit_no_pot_and_scratch_false_require_complete_track_conservation() -> None:
    cue = _ball("cue", "cue-track", 500, 500)
    solid = _ball("solid:solid-track", "solid-track", 800, 600)
    frames = tuple(_frame(frame_id, cue, solid) for frame_id in range(6))

    facts = _facts(_shot(frames))

    assert facts.pots == ()
    assert facts.explicit_no_pot is FactTruth.TRUE
    assert facts.cue_ball_scratch is FactTruth.FALSE
    assert facts.object_ball_off_table is FactTruth.FALSE
    assert facts.before_counts == {
        BallGroup.CUE: 1,
        BallGroup.EIGHT: 0,
        BallGroup.SOLID: 1,
        BallGroup.STRIPE: 0,
    }
    assert facts.after_counts == {
        BallGroup.CUE: 1,
        BallGroup.EIGHT: 0,
        BallGroup.SOLID: 1,
        BallGroup.STRIPE: 0,
    }


@pytest.mark.parametrize(
    ("frames", "uncertain"),
    [
        (
            (
                _frame(0, _ball("eight", "eight-track", 500, 500)),
                _frame(1, _ball("eight", "eight-track", 500, 500)),
                _frame(2),
                _frame(3),
                _frame(4),
                _frame(5),
            ),
            tuple((frame_id, "eight-track", "lost") for frame_id in range(2, 6)),
        ),
        (
            (
                _frame(0, _ball("eight", "eight-track", 50, 50)),
                _frame(1, _ball("eight", "eight-track", 20, 20)),
                _frame(2),
                _frame(3),
                _frame(4, _ball("eight", "eight-track", 35, 35)),
                _frame(5, _ball("eight", "eight-track", 35, 35)),
            ),
            ((2, "eight-track", "lost"), (3, "eight-track", "lost")),
        ),
        (
            (
                _frame(0, _ball("solid:old", "old", 50, 50)),
                _frame(1, _ball("solid:old", "old", 20, 20)),
                _frame(2),
                _frame(3),
                _frame(4, _ball("solid:new", "new", 25, 25)),
                _frame(5, _ball("solid:new", "new", 25, 25)),
            ),
            ((2, "old", "lost"), (3, "old", "lost")),
        ),
    ],
)
def test_ambiguous_disappearance_never_becomes_a_pot_or_no_pot(
    frames: tuple[TrackedFrame, ...],
    uncertain: tuple[tuple[int, str, str], ...],
) -> None:
    facts = _facts(_shot(frames, uncertain=uncertain))

    assert facts.pots == ()
    assert facts.explicit_no_pot is FactTruth.UNKNOWN
    assert facts.object_ball_off_table is FactTruth.UNKNOWN


def test_unrelated_short_uncertainty_does_not_erase_an_explicit_cue_pot() -> None:
    frames = (
        _frame(0, _ball("cue", "cue-track", 50, 50), _ball("solid:s", "s", 800, 600)),
        _frame(1, _ball("cue", "cue-track", 20, 20), _ball("solid:s", "s", 800, 600)),
        _frame(2, _ball("solid:s", "s", 800, 600)),
        _frame(3),
        _frame(4, _ball("solid:s", "s", 800, 600)),
        _frame(5, _ball("solid:s", "s", 800, 600)),
    )
    uncertain = (
        (2, "cue-track", "lost"),
        (3, "cue-track", "lost"),
        (4, "cue-track", "lost"),
        (5, "cue-track", "lost"),
        (3, "s", "ambiguous"),
    )

    facts = _facts(_shot(frames, uncertain=uncertain))

    assert [(pot.track_id, pot.group) for pot in facts.pots] == [("cue-track", BallGroup.CUE)]
    assert facts.cue_ball_scratch is FactTruth.TRUE
    assert facts.explicit_no_pot is FactTruth.FALSE
    assert facts.object_ball_off_table is FactTruth.UNKNOWN
    assert "tracker_uncertain:s" not in facts.abstention_reasons


def test_two_stable_tracks_potted_into_same_pocket_remain_two_explicit_pots() -> None:
    frames = (
        _frame(0, _ball("solid:a", "a", 50, 50), _ball("solid:b", "b", 60, 50)),
        _frame(1, _ball("solid:a", "a", 20, 20), _ball("solid:b", "b", 25, 20)),
        _frame(2),
        _frame(3),
        _frame(4),
        _frame(5),
    )
    uncertain = tuple(
        (frame_id, track_id, "lost") for frame_id in range(2, 6) for track_id in ("a", "b")
    )

    facts = _facts(_shot(frames, uncertain=uncertain))

    assert [(pot.track_id, pot.pocket_id) for pot in facts.pots] == [
        ("a", "top-left"),
        ("b", "top-left"),
    ]
    assert facts.explicit_no_pot is FactTruth.FALSE
    assert "pot_candidates_ambiguous" not in facts.abstention_reasons


def test_disappearance_without_same_track_lost_lineage_is_not_a_pot() -> None:
    frames = (
        _frame(0, _ball("eight", "eight-track", 50, 50)),
        _frame(1, _ball("eight", "eight-track", 20, 20)),
        _frame(2),
        _frame(3),
        _frame(4),
        _frame(5),
    )

    facts = _facts(_shot(frames))

    assert facts.pots == ()
    assert facts.explicit_no_pot is FactTruth.UNKNOWN


def test_stable_unknown_track_never_proves_complete_inventory() -> None:
    unknown = _ball("unknown:mystery", "mystery", 500, 500)
    frames = tuple(_frame(frame_id, unknown) for frame_id in range(6))

    facts = _facts(_shot(frames))

    assert facts.explicit_no_pot is FactTruth.UNKNOWN
    assert facts.object_ball_off_table is FactTruth.UNKNOWN
    assert "unknown_ball_identity:mystery" in facts.abstention_reasons


def test_missing_trusted_table_state_coverage_fails_closed() -> None:
    cue = _ball("cue", "cue-track", 500, 500)
    frames = tuple(_frame(frame_id, cue) for frame_id in range(6))
    coverage = {
        name: FactCoverageVerdict("observed")
        for name in ("first_contact", "cushion_contacts", "pots")
    }

    facts = _facts(_shot(frames, fact_coverage=coverage))

    assert facts.explicit_no_pot is FactTruth.UNKNOWN
    assert "table_state_before_coverage_missing" in facts.abstention_reasons
    assert "table_state_after_coverage_missing" in facts.abstention_reasons


def test_missing_before_or_post_settle_anchor_fails_closed() -> None:
    cue = _ball("cue", "cue-track", 500, 500)
    frames = tuple(_frame(frame_id, cue) for frame_id in range(6))
    shot = _shot(frames)
    shot.rule_coverage = RealtimeEvidenceCoverage(
        source_frame_ids=tuple(range(6)),
        anchor_frame_ids=(1, 4),
        maximum_anchor_gap_ms=100,
        uncertain_frame_ids=(),
        segment_sha256={"segment-0": "a" * 64},
    )

    facts = _facts(shot)

    assert facts.explicit_no_pot is FactTruth.UNKNOWN
    assert "trusted_before_anchor_missing" in facts.abstention_reasons
    assert "trusted_after_anchor_missing" in facts.abstention_reasons


def test_source_gap_and_event_config_digest_mismatch_fail_closed() -> None:
    cue = _ball("cue", "cue-track", 500, 500)
    frames = tuple(_frame(frame_id, cue) for frame_id in range(6))

    source_gap = _facts(_shot(frames, source_frame_ids=(0, 1, 3, 4, 5)))
    config_mismatch = _facts(_shot(frames, event_config_digest="f" * 64))

    assert source_gap.explicit_no_pot is FactTruth.UNKNOWN
    assert "source_frame_gap" in source_gap.abstention_reasons
    assert config_mismatch.explicit_no_pot is FactTruth.UNKNOWN
    assert "event_config_digest_mismatch" in config_mismatch.abstention_reasons


def test_only_observed_contact_and_cushion_evidence_enters_rule_facts() -> None:
    frames = (
        _frame(0, _ball("cue", "cue-track", 100, 100), _ball("solid:s", "s", 200, 100)),
        _frame(1, _ball("cue", "cue-track", 145, 100), _ball("solid:s", "s", 200, 100)),
        _frame(2, _ball("cue", "cue-track", 145, 100), _ball("solid:s", "s", 35, 100)),
        _frame(3, _ball("cue", "cue-track", 145, 100), _ball("solid:s", "s", 70, 100)),
        _frame(4, _ball("cue", "cue-track", 145, 100), _ball("solid:s", "s", 70, 100)),
        _frame(5, _ball("cue", "cue-track", 145, 100), _ball("solid:s", "s", 70, 100)),
    )

    facts = _facts(_shot(frames))

    assert facts.first_contact_group is BallGroup.SOLID
    assert facts.post_contact_rail_or_pot is FactTruth.TRUE
    assert facts.distinct_break_object_rail_count == 1
    assert {1, 2, 3}.issubset(facts.evidence_frame_ids)


def test_complete_conserved_window_proves_no_post_contact_rail_or_pot() -> None:
    frames = (
        _frame(0, _ball("cue", "cue-track", 100, 300), _ball("solid:s", "s", 200, 300)),
        _frame(1, _ball("cue", "cue-track", 145, 300), _ball("solid:s", "s", 200, 300)),
        _frame(2, _ball("cue", "cue-track", 145, 300), _ball("solid:s", "s", 240, 300)),
        _frame(3, _ball("cue", "cue-track", 145, 300), _ball("solid:s", "s", 260, 300)),
        _frame(4, _ball("cue", "cue-track", 145, 300), _ball("solid:s", "s", 260, 300)),
        _frame(5, _ball("cue", "cue-track", 145, 300), _ball("solid:s", "s", 260, 300)),
    )

    facts = _facts(_shot(frames))

    assert facts.first_contact_group is BallGroup.SOLID
    assert facts.explicit_no_pot is FactTruth.TRUE
    assert facts.post_contact_rail_or_pot is FactTruth.FALSE
    assert facts.distinct_break_object_rail_count == 0


def test_ball_reaching_right_cushion_then_stopping_does_not_prove_no_rail() -> None:
    frames = (
        _frame(
            0,
            _ball("cue", "cue-track", 100, 100),
            _ball("solid:s", "s", 200, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
        _frame(
            1,
            _ball("cue", "cue-track", 145, 100),
            _ball("solid:s", "s", 200, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
        _frame(
            2,
            _ball("cue", "cue-track", 145, 100),
            _ball("solid:s", "s", 2400, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
        _frame(
            3,
            _ball("cue", "cue-track", 145, 100),
            _ball("solid:s", "s", 2530, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
        _frame(
            4,
            _ball("cue", "cue-track", 145, 100),
            _ball("solid:s", "s", 2530, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
        _frame(
            5,
            _ball("cue", "cue-track", 145, 100),
            _ball("solid:s", "s", 2530, 100),
            _ball("eight", "eight-track", 1000, 600),
        ),
    )

    facts = _facts(_shot(frames))
    decision = decide_eight_ball(
        MatchControlSnapshot(
            breaker=WristbandRole.RED,
            active_role=WristbandRole.RED,
            revision=7,
            rule_profile_id=PROFILE_ID,
            rack_phase=RackPhase.OPEN,
        ),
        facts,
    )

    assert facts.post_contact_rail_or_pot is FactTruth.UNKNOWN
    assert facts.distinct_break_object_rail_count is None
    assert decision.kind is RuleOutcomeKind.ABSTAIN
    assert decision.reason_code == "post_contact_rail_unknown"


def test_four_distinct_observed_object_rails_prove_break_threshold() -> None:
    cue_before = _ball("cue", "cue-track", 100, 100)
    cue_contact = _ball("cue", "cue-track", 145, 100)
    object_before = (
        _ball("solid:s1", "s1", 200, 100),
        _ball("solid:s2", "s2", 100, 300),
        _ball("stripe:t1", "t1", 100, 500),
        _ball("stripe:t2", "t2", 100, 700),
    )
    object_at_rail = tuple(
        _ball(ball.identity, ball.track_id, 35, ball.y_mm) for ball in object_before
    )
    object_reversed = tuple(
        _ball(ball.identity, ball.track_id, 70, ball.y_mm) for ball in object_before
    )
    frames = (
        _frame(0, cue_before, *object_before),
        _frame(1, cue_contact, *object_before),
        _frame(2, cue_contact, *object_at_rail),
        _frame(3, cue_contact, *object_reversed),
        _frame(4, cue_contact, *object_reversed),
        _frame(5, cue_contact, *object_reversed),
    )

    facts = _facts(_shot(frames))

    assert facts.post_contact_rail_or_pot is FactTruth.TRUE
    assert facts.distinct_break_object_rail_count == 4


def test_inferred_contact_is_not_accepted_as_a_rule_fact() -> None:
    frames = (
        _frame(0, _ball("cue", "cue-track", 100, 100)),
        _frame(1, _ball("cue", "cue-track", 150, 100)),
        _frame(2, _ball("eight", "eight-track", 205, 100)),
        _frame(3, _ball("eight", "eight-track", 205, 100)),
        _frame(4, _ball("eight", "eight-track", 205, 100)),
        _frame(5, _ball("eight", "eight-track", 205, 100)),
    )

    facts = _facts(_shot(frames))

    assert facts.first_contact_group is None
    assert "first_contact_inferred" not in facts.abstention_reasons
    assert facts.post_contact_rail_or_pot is FactTruth.UNKNOWN
    assert facts.distinct_break_object_rail_count is None


def test_deadline_is_checked_during_frame_projection() -> None:
    frames = tuple(_frame(frame_id, _ball("cue", "cue-track", 500, 500)) for frame_id in range(6))
    ticks = iter((0, 1, 2, 3, 4, 5, 6, 7))

    with pytest.raises(RuleFactDeadlineExceeded):
        _facts(_shot(frames), monotonic_ns=lambda: next(ticks), deadline_ns=4)


def test_deadline_can_interrupt_inside_contact_observer_scan() -> None:
    frames = tuple(_frame(frame_id, _ball("cue", "cue-track", 500, 500)) for frame_id in range(120))
    shot = _shot(frames)
    shot.start_frame_id = 1
    shot.settled_frame_id = 118

    ticks = iter(range(10_000))
    with pytest.raises(RuleFactDeadlineExceeded, match="first_contact"):
        _facts(shot, monotonic_ns=lambda: next(ticks), deadline_ns=124)


def test_fact_digest_is_canonical_and_covers_profile_and_config() -> None:
    cue = _ball("cue", "cue-track", 500, 500)
    frames = tuple(_frame(frame_id, cue) for frame_id in range(6))
    shot = _shot(frames)

    first = _facts(shot)
    second = _facts(shot)
    other_profile = build_realtime_rule_facts(
        shot,
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id="different-profile",
        deadline_ns=10**12,
        monotonic_ns=lambda: 0,
    )
    changed_coverage = _shot(
        frames,
        fact_coverage={
            **shot.fact_coverage,
            "pots": FactCoverageVerdict("abstained", "anchor_gap_exceeded"),
        },
    )
    other_settled = _facts(changed_coverage)

    assert first == second
    assert first.profile_id == "cbsa-chinese-pool-2017-observable-v1"
    assert len(first.settled_evidence_digest) == 64
    assert len(first.digest) == 64
    assert first.digest != other_profile.digest
    assert first.settled_evidence_digest != other_settled.settled_evidence_digest


def _preimage_build(*, abstained: bool = False):
    frames = tuple(
        _frame(i, _ball("cue", "cue-track", 500 + i * 10, 500), _ball("solid:s", "s", 1500, 500))
        for i in range(6)
    )
    shot = _shot(frames)
    kwargs = dict(
        session_id="session-1",
        actor_role=WristbandRole.RED,
        match_revision=7,
        profile_id=PROFILE_ID,
    )
    if abstained:
        return fact_module.build_abstained_realtime_rule_facts_with_preimage(
            shot, reason="观测不足", **kwargs
        )
    return fact_module.build_realtime_rule_facts_with_preimage(
        shot, deadline_ns=10**12, monotonic_ns=lambda: 0, **kwargs
    )


def _preimage_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("abstained", [False, True])
def test_retained_preimage_preserves_exact_v1_facts_and_digest(abstained: bool) -> None:
    build = _preimage_build(abstained=abstained)
    facts = build.facts
    payload = json.loads(json.dumps(asdict(facts)))
    expected_payload_hash, expected_digest = (
        (
            "d7233462fabdb0beff8397b2f30b2326556bc050050e91963b795ea806649fc7",
            "b4879f97a7fdbc1a55c99b09bd6ef6d1e3bbfe02bcc468f4b36e9b8abf844429",
        )
        if abstained
        else (
            "c0e0370a50e4a9cbe1792cd482a60d7d0982271e9ed5d76c550488f3aa51ccd6",
            "8244ed5052a98f57a7c5ce2f61f26d89149b6ad0f7f7df5a6bb43f49b27a771a",
        )
    )
    stored = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(stored.encode()).hexdigest() == expected_payload_hash
    assert facts.digest == expected_digest
    assert hashlib.sha256(build.normalization_preimage_json.encode()).hexdigest() == facts.digest
    normalized = fact_module.validate_realtime_rule_fact_preimage(
        payload, build.normalization_preimage_json, expected_digest=facts.digest
    )
    assert normalized["event_config_digest"] == _config_digest(_config())
    assert normalized["required_cue_ball_placement"] == "none"
    assert set(normalized) == (set(payload) - {"digest"}) | {
        "event_config_digest",
        "required_cue_ball_placement",
    }
    if abstained:
        assert "\\u89c2" in build.normalization_preimage_json
    # Old public callers still receive the identical original dataclass.
    frames = tuple(
        _frame(i, _ball("cue", "cue-track", 500 + i * 10, 500), _ball("solid:s", "s", 1500, 500))
        for i in range(6)
    )
    if abstained:
        old = fact_module.build_abstained_realtime_rule_facts(
            _shot(frames),
            session_id="session-1",
            actor_role=WristbandRole.RED,
            match_revision=7,
            profile_id=PROFILE_ID,
            reason="观测不足",
        )
    else:
        old = _facts(_shot(frames))
    assert asdict(old) == asdict(facts)


@pytest.mark.parametrize(
    "mutation",
    [
        "actor",
        "count_bool_alias",
        "config_type",
        "config_hash",
        "placement_type",
        "placement_value",
        "missing",
        "extra",
        "unicode",
        "duplicate",
        "nonfinite",
    ],
)
def test_preimage_rejects_resealed_wrong_projection_and_noncanonical_input(mutation: str) -> None:
    build = _preimage_build(abstained=True)
    payload = json.loads(json.dumps(asdict(build.facts)))
    normalized = json.loads(build.normalization_preimage_json)
    if mutation == "actor":
        normalized["actor_role"] = "blue"
    elif mutation == "count_bool_alias":
        normalized["before_counts"]["cue"] = True
    elif mutation == "config_type":
        normalized["event_config_digest"] = 1
    elif mutation == "config_hash":
        normalized["event_config_digest"] = "not-a-hash"
    elif mutation == "placement_type":
        normalized["required_cue_ball_placement"] = False
    elif mutation == "placement_value":
        normalized["required_cue_ball_placement"] = "unrecognized"
    elif mutation == "missing":
        normalized.pop("event_config_digest")
    elif mutation == "extra":
        normalized["cue_object_contact"] = False
    elif mutation == "nonfinite":
        normalized["before_counts"]["cue"] = float("inf")
    raw = _preimage_canonical(normalized)
    if mutation == "unicode":
        raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    elif mutation == "duplicate":
        raw = '{"actor_role":"blue",' + raw[1:]
    # Resealing defeats a hash-only check; the actual fact projection must agree too.
    digest = hashlib.sha256(raw.encode()).hexdigest()
    payload["digest"] = digest
    with pytest.raises(ValueError, match="preimage"):
        fact_module.validate_realtime_rule_fact_preimage(payload, raw, expected_digest=digest)


def test_preimage_rejects_digest_and_payload_mismatch() -> None:
    build = _preimage_build()
    payload = json.loads(json.dumps(asdict(build.facts)))
    for bad_payload, digest in (
        (payload, "0" * 64),
        ({**payload, "digest": "0" * 64}, build.facts.digest),
    ):
        with pytest.raises(ValueError, match="preimage"):
            fact_module.validate_realtime_rule_fact_preimage(
                bad_payload, build.normalization_preimage_json, expected_digest=digest
            )
