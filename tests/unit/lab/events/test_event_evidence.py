from billiards_ai.lab.events.contact import observe_first_contact
from billiards_ai.lab.events.cushion import observe_cushion_contacts
from billiards_ai.lab.events.foul import observe_supported_fouls
from billiards_ai.lab.events.models import BallState, EventStatus, TrackedFrame
from billiards_ai.lab.events.pot import observe_pots


def _frame(frame_id: int, timestamp_ms: int, **balls: tuple[int, int]) -> TrackedFrame:
    return TrackedFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        balls=tuple(
            BallState(identity=identity, x_mm=position[0], y_mm=position[1], confidence=0.95)
            for identity, position in sorted(balls.items())
        ),
    )


def _tracked_frame(
    frame_id: int,
    timestamp_ms: int,
    *balls: tuple[str, str, tuple[int, int]],
) -> TrackedFrame:
    return TrackedFrame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        balls=tuple(
            BallState(
                identity=identity,
                track_id=track_id,
                x_mm=position[0],
                y_mm=position[1],
                confidence=0.95,
            )
            for track_id, identity, position in balls
        ),
    )


def test_simultaneous_looking_first_contact_abstains_with_evidence() -> None:
    frames = (
        _frame(1, 0, cue=(500, 635), **{"1": (600, 620), "2": (600, 650)}),
        _frame(2, 40, cue=(550, 635), **{"1": (600, 620), "2": (600, 650)}),
        _frame(3, 80, cue=(575, 635), **{"1": (603, 618), "2": (603, 652)}),
    )

    event = observe_first_contact(frames, contact_distance_mm=35, ambiguity_mm=5)

    assert event.status is EventStatus.ABSTAINED
    assert event.reason == "simultaneous_contact_ambiguous"
    assert event.evidence_frame_ids
    assert event.confidence < 0.5


def test_stationary_cluster_is_not_reported_as_contact() -> None:
    frames = (
        _frame(1, 0, cue=(500, 500), **{"1": (525, 500), "2": (550, 500)}),
        _frame(2, 40, cue=(500, 500), **{"1": (525, 500), "2": (550, 500)}),
    )

    event = observe_first_contact(frames, contact_distance_mm=30)

    assert event.status is EventStatus.ABSTAINED
    assert event.reason == "cue_ball_not_moving"


def test_first_contact_uses_track_ids_when_two_tracks_are_labelled_cue() -> None:
    frames = (
        _tracked_frame(1, 0, ("white", "cue", (100, 100)), ("black", "cue", (205, 100))),
        _tracked_frame(2, 40, ("white", "cue", (145, 100)), ("black", "cue", (205, 100))),
    )

    event = observe_first_contact(frames, contact_distance_mm=65)

    assert event.status is EventStatus.OBSERVED
    assert event.actor_identity == "cue"
    assert event.target_identity == "unknown:black"
    assert event.reason == "duplicate_cue_label_downgraded_to_target"


def test_first_contact_uses_the_first_moving_cue_trajectory_as_actor() -> None:
    frames = (
        _tracked_frame(
            1,
            0,
            ("white", "cue", (100, 100)),
            ("black", "cue", (300, 100)),
            ("object", "solid:object", (400, 100)),
        ),
        _tracked_frame(
            2,
            40,
            ("white", "cue", (120, 100)),
            ("black", "cue", (300, 100)),
            ("object", "solid:object", (400, 100)),
        ),
        _tracked_frame(
            3,
            80,
            ("white", "cue", (140, 100)),
            ("black", "cue", (350, 100)),
            ("object", "solid:object", (400, 100)),
        ),
    )

    event = observe_first_contact(frames, contact_distance_mm=65)

    assert event.status is EventStatus.ABSTAINED
    assert event.reason == "no_contact_observed"


def test_first_contact_bridges_alternating_visibility_as_inferred_evidence() -> None:
    frames = (
        _tracked_frame(1, 0, ("white", "cue", (100, 100))),
        _tracked_frame(2, 40, ("white", "cue", (150, 100))),
        _tracked_frame(3, 80, ("black", "eight", (205, 100))),
    )

    event = observe_first_contact(frames, contact_distance_mm=65, bridge_frames=3)

    assert event.status is EventStatus.INFERRED
    assert event.target_identity == "eight"
    assert event.reason == "temporally_inferred_contact"
    assert event.evidence_frame_ids == (1, 2, 3)
    assert event.confidence < 0.5


def test_first_contact_does_not_bridge_visibility_gap_beyond_limit() -> None:
    frames = (
        _tracked_frame(1, 0, ("white", "cue", (100, 100))),
        _tracked_frame(2, 40, ("white", "cue", (150, 100))),
        _tracked_frame(3, 80),
        _tracked_frame(4, 120),
        _tracked_frame(5, 160),
        _tracked_frame(6, 200, ("black", "eight", (205, 100))),
    )

    event = observe_first_contact(frames, contact_distance_mm=65, bridge_frames=3)

    assert event.status is EventStatus.ABSTAINED
    assert event.reason == "no_contact_observed"


def test_first_contact_infers_kinematic_transfer_across_fragmented_cue_tracks() -> None:
    frames = (
        _tracked_frame(91, 3033, ("black", "cue", (258, 181))),
        _tracked_frame(92, 3067),
        _tracked_frame(93, 3100),
        _tracked_frame(94, 3133),
        _tracked_frame(95, 3167, ("white-1", "cue", (573, 307))),
        _tracked_frame(
            96,
            3200,
            ("black", "cue", (258, 181)),
            ("white-2", "cue", (443, 262)),
        ),
        _tracked_frame(97, 3233, ("black", "cue", (322, 220))),
    )

    event = observe_first_contact(frames, contact_distance_mm=65, bridge_frames=3)

    assert event.status is EventStatus.INFERRED
    assert event.timestamp_ms == 3233
    assert event.target_identity == "unknown:black"
    assert event.reason == "kinematic_contact_transfer"
    assert event.evidence_frame_ids == (91, 95, 96, 97)


def test_cushion_requires_near_rail_direction_reversal() -> None:
    frames = (
        _frame(1, 0, cue=(100, 200)),
        _frame(2, 40, cue=(35, 200)),
        _frame(3, 80, cue=(70, 200)),
    )

    events = observe_cushion_contacts(frames, rail_distance_mm=40)

    assert len(events) == 1
    assert events[0].actor_identity == "cue"
    assert events[0].evidence_frame_ids == (1, 2, 3)


def test_temporary_pocket_occlusion_is_not_a_pot_but_persistent_loss_is() -> None:
    temporary = (
        _frame(1, 0, **{"8": (30, 30)}),
        _frame(2, 40),
        _frame(3, 80, **{"8": (35, 35)}),
    )
    persistent = temporary[:2] + (_frame(3, 80), _frame(4, 120))

    assert observe_pots(temporary, persistence_frames=2) == ()
    events = observe_pots(persistent, persistence_frames=2)
    assert len(events) == 1
    assert events[0].actor_identity == "8"
    assert events[0].evidence_frame_ids == (1, 2, 3)
    assert events[0].reason == "pocket:top-left"


def test_disappearance_that_reappears_after_confirmation_is_not_a_pot() -> None:
    frames = (
        _tracked_frame(1, 0, ("eight-track", "eight", (30, 30))),
        _tracked_frame(2, 40),
        _tracked_frame(3, 80),
        _tracked_frame(4, 120, ("eight-track", "eight", (35, 35))),
    )

    assert observe_pots(frames, persistence_frames=2) == ()


def test_pot_observer_uses_track_identity_for_same_group_balls() -> None:
    frames = (
        _tracked_frame(
            1,
            0,
            ("solid-a", "solid:solid-a", (20, 20)),
            ("solid-b", "solid:solid-b", (600, 600)),
        ),
        _tracked_frame(2, 40, ("solid-b", "solid:solid-b", (600, 600))),
        _tracked_frame(3, 80, ("solid-b", "solid:solid-b", (600, 600))),
    )

    events = observe_pots(frames, persistence_frames=2)

    assert [event.actor_identity for event in events] == ["solid:solid-a"]
    assert events[0].reason == "pocket:top-left"


def test_foul_observer_limits_itself_to_supported_scratch() -> None:
    scratch = observe_pots(
        (_frame(1, 0, cue=(20, 20)), _frame(2, 40), _frame(3, 80)),
        persistence_frames=2,
    )[0]

    supported = observe_supported_fouls((scratch,), requested_rule="cue_ball_scratch")
    unsupported = observe_supported_fouls((scratch,), requested_rule="wrong_ball_first")

    assert supported.status is EventStatus.OBSERVED
    assert supported.event_type == "foul.cue_ball_scratch"
    assert unsupported.status is EventStatus.UNSUPPORTED
    assert unsupported.reason == "rule_not_in_supported_observation_subset"
