from __future__ import annotations

from billiards_ai.lab.shot_local.events import (
    ShotLocalEventConfig,
    observe_ball_ball_collisions,
    observe_first_anchor_contact,
    observe_fragment_cushions,
    observe_fragment_pots,
    observe_shot_boundaries,
    observe_shot_cushions,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    CueSample,
    FragmentSample,
    LocalState,
    MotionFragment,
    ObservationFrame,
    ShotAnchor,
    ShotLocalEvent,
    ShotWindow,
)


def _config() -> ShotLocalEventConfig:
    return ShotLocalEventConfig(
        first_contact_distance_mm=35,
        first_contact_ambiguity_mm=5,
        cue_motion_min_mm=20,
        contact_time_tolerance_ms=80,
        cushion_distance_mm=25,
        pocket_distance_mm=60,
        pot_missing_frames=3,
        after_match_distance_mm=80,
        table_length_mm=1000,
        table_width_mm=500,
    )


def _anchor(number: int, x_mm: int, y_mm: int = 200) -> ShotAnchor:
    return ShotAnchor(
        shot_id="shot-0001",
        anchor_id=f"shot-0001-anchor-{number:02d}",
        source_detection_id=f"anchor-{number}",
        group=BallGroup.SOLID,
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.9,
    )


def _cue(frame_id: int, x_mm: int, y_mm: int = 200) -> CueSample:
    return CueSample(
        shot_id="shot-0001",
        local_id="shot-0001-cue",
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        state=LocalState.CONFIRMED,
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.9,
        source_detection_id=f"cue-{frame_id}",
    )


def _fragment(
    points: tuple[tuple[int, int, int], ...],
    *,
    terminal_state: LocalState = LocalState.LOST,
    number: int = 1,
) -> MotionFragment:
    fragment_id = f"shot-0001-fragment-{number:04d}"
    return MotionFragment(
        shot_id="shot-0001",
        fragment_id=fragment_id,
        group=BallGroup.SOLID,
        samples=tuple(
            FragmentSample(
                shot_id="shot-0001",
                fragment_id=fragment_id,
                frame_id=frame_id,
                timestamp_ms=frame_id * 40,
                state=LocalState.CONFIRMED,
                x_mm=x_mm,
                y_mm=y_mm,
                confidence=0.9,
                source_detection_id=f"fragment-{frame_id}",
            )
            for frame_id, x_mm, y_mm in points
        ),
        terminal_state=terminal_state,
    )


def _window(*, complete: bool = True) -> ShotWindow:
    return ShotWindow(
        shot_id="shot-0001",
        before_frame_id=0,
        start_frame_id=1,
        end_frame_id=9,
        after_frame_id=10 if complete else None,
        start_ms=40,
        end_ms=360,
        complete=complete,
    )


def test_boundaries_reuse_window_and_trusted_cue_motion() -> None:
    events = observe_shot_boundaries(_window(), (_cue(1, 100), _cue(2, 130)), _config())

    assert [(event.event_type, event.status) for event in events] == [
        ("shot_start", "observed"),
        ("shot_end", "observed"),
    ]
    assert events[1].evidence_frame_ids == (9, 10)


def test_incomplete_or_untrusted_boundaries_fail_closed() -> None:
    events = observe_shot_boundaries(_window(complete=False), (_cue(1, 100),), _config())

    assert events[0].status == "abstained"
    assert events[0].reason == "trusted_cue_motion_insufficient"
    assert events[1].status == "abstained"
    assert events[1].reason == "shot_end_not_observed"


def test_ball_collision_observes_unique_motion_transfer_after_first_contact() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=80,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1, 2),
        confidence=0.9,
    )
    actor = _fragment(((2, 100, 200), (3, 130, 200), (4, 158, 200)), number=1)
    target = _fragment(((5, 185, 200), (6, 215, 200), (7, 245, 200)), number=2)

    events = observe_ball_ball_collisions(
        (actor, target), (_anchor(1, 100), _anchor(2, 180)), first_contact, _config()
    )

    assert len(events) == 1
    assert events[0].status == "observed"
    assert events[0].actor_id == actor.fragment_id
    assert events[0].target_id == target.fragment_id
    assert events[0].evidence_frame_ids == (3, 4, 5, 6)


def test_ball_collision_marks_fragment_gap_inferred_not_observed() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor = _fragment(((2, 100, 200), (3, 130, 200), (4, 160, 200)), number=1)
    target = _fragment(((7, 218, 200), (8, 248, 200)), number=2)

    events = observe_ball_ball_collisions(
        (actor, target), (_anchor(1, 100), _anchor(2, 210)), first_contact, _config()
    )

    assert len(events) == 1
    assert events[0].status == "inferred"
    assert events[0].reason == "bounded_fragment_gap_motion_transfer"


def test_ball_collision_abstains_when_two_targets_are_equally_plausible() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor = _fragment(((2, 100, 200), (3, 130, 200), (4, 160, 200)), number=1)
    target_a = _fragment(((5, 188, 198), (6, 218, 198)), number=2)
    target_b = _fragment(((5, 188, 202), (6, 218, 202)), number=3)

    events = observe_ball_ball_collisions(
        (actor, target_a, target_b),
        (_anchor(1, 100), _anchor(2, 185)),
        first_contact,
        _config(),
    )

    assert len(events) == 1
    assert events[0].status == "abstained"
    assert events[0].reason == "ball_collision_target_ambiguous"


def test_ball_collision_requires_actor_transfer_signal() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor = _fragment(
        ((2, 100, 200), (3, 130, 200), (4, 160, 200)),
        number=1,
        terminal_state=LocalState.STOPPED,
    )
    target = _fragment(((5, 188, 200), (6, 218, 200)), number=2)

    events = observe_ball_ball_collisions(
        (actor, target), (_anchor(1, 100), _anchor(2, 185)), first_contact, _config()
    )

    assert len(events) == 1
    assert events[0].status == "abstained"
    assert events[0].reason == "ball_collision_evidence_insufficient"


def test_ball_collision_does_not_turn_same_anchor_refragmentation_into_contact() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor = _fragment(((2, 100, 200), (3, 125, 200), (4, 150, 200)), number=1)
    continuation = _fragment(((5, 175, 200), (6, 200, 200)), number=2)

    events = observe_ball_ball_collisions(
        (actor, continuation), (_anchor(1, 125),), first_contact, _config()
    )

    assert events[0].status == "abstained"
    assert events[0].reason == "ball_collision_evidence_insufficient"


def test_ball_collision_rejects_actor_not_reachable_from_first_contact_target() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    unrelated_actor = _fragment(((2, 100, 200), (3, 130, 200), (4, 160, 200)), number=1)
    unrelated_target = _fragment(((5, 188, 200), (6, 218, 200)), number=2)

    events = observe_ball_ball_collisions(
        (unrelated_actor, unrelated_target),
        (_anchor(1, 20), _anchor(2, 100), _anchor(3, 185)),
        first_contact,
        _config(),
    )

    assert events[0].status == "abstained"
    assert events[0].reason == "ball_collision_evidence_insufficient"


def test_low_confidence_first_contact_does_not_seed_collision_chain() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.69,
    )
    actor = _fragment(((2, 100, 200), (3, 130, 200), (4, 160, 200)), number=1)
    target = _fragment(((5, 188, 200), (6, 218, 200)), number=2)

    events = observe_ball_ball_collisions(
        (actor, target), (_anchor(1, 100), _anchor(2, 185)), first_contact, _config()
    )

    assert events[0].status == "abstained"
    assert events[0].reason == "first_contact_unavailable_for_collision_chain"


def test_ball_collision_builds_only_observed_reachable_chain_a_to_b_to_c() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor_a = _fragment(((2, 100, 200), (3, 130, 200), (4, 158, 200)), number=1)
    actor_b = _fragment(((5, 185, 200), (6, 215, 200), (7, 245, 200), (8, 275, 200)), number=2)
    target_c = _fragment(((9, 305, 200), (10, 335, 200)), number=3)

    events = observe_ball_ball_collisions(
        (actor_a, actor_b, target_c),
        (_anchor(1, 100), _anchor(2, 180), _anchor(3, 300)),
        first_contact,
        _config(),
    )

    observed = [event for event in events if event.status == "observed"]
    assert [(event.actor_id, event.target_id) for event in observed] == [
        (actor_a.fragment_id, actor_b.fragment_id),
        (actor_b.fragment_id, target_c.fragment_id),
    ]


def test_ball_collision_prevents_branch_cycle_and_reusing_one_actor_anchor() -> None:
    first_contact = ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=40,
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-01",
        evidence_frame_ids=(1,),
        confidence=0.9,
    )
    actor_a = _fragment(((2, 100, 200), (3, 130, 200), (4, 158, 200)), number=1)
    target_b = _fragment(((5, 185, 200), (6, 215, 200)), number=2)
    second_a_fragment = _fragment(((7, 105, 200), (8, 200, 200), (9, 230, 200)), number=3)
    branch_c = _fragment(((10, 260, 200), (11, 290, 200)), number=4)
    cycle_to_a = _fragment(((7, 105, 205), (8, 135, 205)), number=5)

    events = observe_ball_ball_collisions(
        (actor_a, target_b, second_a_fragment, branch_c, cycle_to_a),
        (_anchor(1, 100), _anchor(2, 180), _anchor(3, 260)),
        first_contact,
        _config(),
    )

    observed = [event for event in events if event.status == "observed"]
    assert [(event.actor_id, event.target_id) for event in observed] == [
        (actor_a.fragment_id, target_b.fragment_id)
    ]


def _after_frame(frame_id: int, *observations) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 40,
        observations=tuple(observations),
    )


def _observation(detection_id: str, x_mm: int, y_mm: int) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.9,
        group=BallGroup.SOLID,
        group_confidence=0.8,
    )


def test_first_contact_references_pre_shot_anchor() -> None:
    event = observe_first_anchor_contact(
        (_cue(2, 100), _cue(3, 170), _cue(4, 195)),
        (_anchor(1, 500), _anchor(2, 200)),
        (_fragment(((4, 210, 200),)),),
        _config(),
    )

    assert event.event_type == "first_contact"
    assert event.actor_id == "shot-0001-cue"
    assert event.target_id == "shot-0001-anchor-02"
    assert event.status == "observed"


def test_equal_anchor_candidates_abstain() -> None:
    event = observe_first_anchor_contact(
        (_cue(2, 100), _cue(3, 200)),
        (_anchor(1, 180), _anchor(2, 220)),
        (_fragment(((3, 200, 200),)),),
        _config(),
    )

    assert event.status == "abstained"
    assert event.target_id is None
    assert event.reason == "first_contact_anchor_ambiguous"


def test_cushion_requires_rail_proximity_and_direction_reversal() -> None:
    near_rail = _fragment(((2, 100, 40), (3, 80, 10), (4, 100, 40)))
    middle_table = _fragment(((2, 100, 240), (3, 80, 210), (4, 100, 240)))

    assert len(observe_fragment_cushions((near_rail,), _config())) == 1
    assert observe_fragment_cushions((middle_table,), _config()) == ()


def test_cue_ball_and_fragment_use_the_same_cushion_observer() -> None:
    cue = (_cue(2, 100, 40), _cue(3, 80, 10), _cue(4, 100, 40))
    fragment = _fragment(((2, 400, 460), (3, 420, 490), (4, 400, 460)))

    events = observe_shot_cushions(cue, (fragment,), _config())

    assert [(event.actor_id, event.reason) for event in events] == [
        ("shot-0001-cue", "rail:bottom"),
        ("shot-0001-fragment-0001", "rail:top"),
    ]


def test_near_rail_without_reversal_is_not_a_cushion() -> None:
    cue = (_cue(2, 100, 40), _cue(3, 90, 20), _cue(4, 80, 10))

    assert observe_shot_cushions(cue, (), _config()) == ()


def test_sparse_observation_can_bridge_a_missing_contact_frame() -> None:
    cue = (_cue(2, 100, 200), _cue(3, 60, 200), _cue(5, 75, 200))

    events = observe_shot_cushions(cue, (), _config())

    assert len(events) == 1
    assert events[0].actor_id == "shot-0001-cue"
    assert events[0].reason == "rail:left"


def test_pocket_disappearance_without_rebound_is_not_a_cushion() -> None:
    fragment = _fragment(((2, 100, 60), (3, 20, 20)), terminal_state=LocalState.LOST)

    assert observe_shot_cushions((), (fragment,), _config()) == ()


def test_one_physical_cushion_is_debounced() -> None:
    cue = (
        _cue(2, 100, 200),
        _cue(3, 20, 200),
        _cue(4, 100, 200),
        _cue(5, 20, 200),
        _cue(6, 100, 200),
    )

    events = observe_shot_cushions(cue, (), _config())

    assert len(events) == 1


def test_short_pocket_disappearance_is_not_a_pot() -> None:
    fragment = _fragment(((2, 30, 25),), terminal_state=LocalState.LOST)

    assert observe_fragment_pots((fragment,), _after_frame(4), _config()) == ()


def test_pot_requires_persistent_absence_from_after_snapshot() -> None:
    fragment = _fragment(((2, 30, 25),), terminal_state=LocalState.LOST)

    events = observe_fragment_pots((fragment,), _after_frame(6), _config())
    present = observe_fragment_pots(
        (fragment,),
        _after_frame(6, _observation("still-there", 35, 30)),
        _config(),
    )

    assert events[0].event_type == "pot"
    assert events[0].status == "observed"
    assert events[0].reason == "pocket:left-bottom"
    assert present == ()
