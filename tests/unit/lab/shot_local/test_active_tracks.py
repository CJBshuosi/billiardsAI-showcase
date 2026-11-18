from __future__ import annotations

import numpy as np

from billiards_ai.lab.shot_local.active_tracks import (
    ActiveBallTrack,
    ActiveTrackConfig,
    ActiveTrackSample,
    DelayedCollisionDecision,
    assign_calibrated_points,
    extract_ball_appearance_signatures,
    minimum_cost_assignment,
    observe_delayed_collision_chain,
    observe_three_layer_cushions,
    track_anchor_roster,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
    ShotAnchor,
    ShotLocalEvent,
)


def _config() -> ActiveTrackConfig:
    return ActiveTrackConfig(
        position_gate_mm=90,
        unmatched_cost=120,
        process_noise_mm=10,
        measurement_noise_mm=5,
        max_prediction_frames=3,
        appearance_weight=45,
        group_mismatch_cost=100,
        collision_distance_mm=58,
        moving_speed_mm_per_frame=8,
        stationary_speed_mm_per_frame=3,
        collision_lookahead_frames=5,
        collision_candidate_margin=0.08,
    )


def _anchor(number: int, x_mm: int, y_mm: int = 200) -> ShotAnchor:
    return ShotAnchor(
        shot_id="shot-0001",
        anchor_id=f"shot-0001-anchor-{number:02d}",
        source_detection_id=f"anchor-{number}",
        group=BallGroup.STRIPE,
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.95,
    )


def _observation(
    frame_id: int,
    number: int,
    x_mm: int,
    y_mm: int = 200,
) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=f"frame-{frame_id:04d}-detection-{number:02d}",
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.95,
        group=BallGroup.STRIPE,
        group_confidence=0.95,
    )


def _frame(frame_id: int, *observations: CalibratedBallObservation) -> ObservationFrame:
    return ObservationFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        observations=tuple(observations),
    )


def _first_contact(target_id: str) -> ShotLocalEvent:
    return ShotLocalEvent(
        shot_id="shot-0001",
        event_id="shot-0001-event-first-contact",
        event_type="first_contact",
        status="observed",
        timestamp_ms=66,
        actor_id="shot-0001-cue",
        target_id=target_id,
        evidence_frame_ids=(1, 2),
        confidence=0.95,
    )


def _track(anchor_id: str, points: tuple[tuple[int, int, int], ...]) -> ActiveBallTrack:
    return ActiveBallTrack(
        shot_id="shot-0001",
        track_id=anchor_id,
        group=BallGroup.STRIPE,
        samples=tuple(
            ActiveTrackSample(
                frame_id=frame_id,
                timestamp_ms=frame_id * 33,
                x_mm=float(x_mm),
                y_mm=float(y_mm),
                vx_mm_per_frame=0.0,
                vy_mm_per_frame=0.0,
                state="observed",
                confidence=0.95,
                source_detection_id=f"{anchor_id}-{frame_id}",
            )
            for frame_id, x_mm, y_mm in points
        ),
    )


def test_hungarian_assignment_avoids_order_dependent_greedy_swap() -> None:
    # A greedy detection-first matcher assigns the observation at 4 to track 2,
    # forcing the observation at 6 onto track 1 (total 7).  The global optimum is 5.
    assert minimum_cost_assignment(((4.0, 6.0), (1.0, 1.0))) == (0, 1)


def test_assign_calibrated_points_uses_global_minimum_not_greedy() -> None:
    matches = assign_calibrated_points(
        predicted=((0.0, 0.0), (3.0, 0.0)),
        observed=((2.0, 0.0), (4.0, 0.0)),
        maximum_distance_mm=4.0,
    )

    assert matches == ((0, 0), (1, 1))


def test_kalman_track_survives_two_missing_frames_and_reacquires_same_anchor() -> None:
    anchor = _anchor(1, 100)
    frames = (
        _frame(1, _observation(1, 1, 110)),
        _frame(2, _observation(2, 1, 125)),
        _frame(3),
        _frame(4),
        _frame(5, _observation(5, 1, 170)),
    )

    tracks = track_anchor_roster(frames, (anchor,), _config())

    assert len(tracks) == 1
    assert tracks[0].track_id == anchor.anchor_id
    assert [sample.state for sample in tracks[0].samples] == [
        "observed",
        "observed",
        "predicted",
        "predicted",
        "reacquired",
    ]
    assert tracks[0].samples[-1].source_detection_id == "frame-0005-detection-01"


def test_direct_collision_keeps_actor_target_roles_without_waiting_for_separation() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        (
            (1, 100, 200),
            (2, 125, 200),
            (3, 150, 200),
            (4, 175, 200),
            (5, 195, 200),
            (6, 197, 200),
            (7, 198, 200),
            (8, 198, 200),
            (9, 198, 200),
            (10, 198, 200),
        ),
    )
    target = _track(
        "shot-0001-anchor-02",
        (
            (1, 225, 200),
            (2, 225, 200),
            (3, 225, 200),
            (4, 225, 200),
            (5, 226, 200),
            (6, 244, 200),
            (7, 266, 200),
            (8, 288, 200),
            (9, 310, 200),
            (10, 332, 200),
        ),
    )

    decisions = observe_delayed_collision_chain(
        (actor, target),
        _first_contact(actor.track_id),
        _config(),
    )

    observed = [decision for decision in decisions if decision.status == "observed"]
    assert len(observed) == 1
    assert observed[0].actor_id == actor.track_id
    assert observed[0].target_id == target.track_id
    assert observed[0].contact_frame_id == 4
    assert observed[0].decision_frame_id == 4
    assert observed[0].evidence_frame_ids == (3, 4)
    assert observed[0].detection_layer == "L1"


def test_delayed_collision_does_not_call_continued_motion_a_collision() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        tuple((frame_id, 100 + frame_id * 20, 200) for frame_id in range(1, 11)),
    )
    stationary = _track(
        "shot-0001-anchor-02",
        tuple((frame_id, 500, 200) for frame_id in range(1, 11)),
    )

    decisions = observe_delayed_collision_chain(
        (actor, stationary),
        _first_contact(actor.track_id),
        _config(),
    )

    assert [decision for decision in decisions if decision.status == "observed"] == []


def test_l1_direct_contact_detects_slow_balls_without_post_contact_separation() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        ((1, 100, 200), (2, 104, 200), (3, 112, 200), (4, 112, 200), (5, 112, 200)),
    )
    target = _track(
        "shot-0001-anchor-02",
        ((1, 175, 200), (2, 174, 200), (3, 169, 200), (4, 169, 200), (5, 169, 200)),
    )

    decisions = observe_delayed_collision_chain(
        (actor, target), _first_contact(actor.track_id), _config()
    )

    assert len(decisions) == 1
    assert decisions[0].target_id == target.track_id
    assert decisions[0].contact_frame_id == 3
    assert decisions[0].decision_frame_id == 3
    assert decisions[0].detection_layer == "L1"
    assert decisions[0].contact_kind == "new_contact"


def test_l2_swept_contact_detects_contact_skipped_between_video_frames() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        ((1, 100, 200), (2, 100, 200), (3, 180, 200), (4, 190, 200)),
    )
    target = _track(
        "shot-0001-anchor-02",
        ((1, 180, 200), (2, 180, 200), (3, 100, 200), (4, 90, 200)),
    )

    decisions = observe_delayed_collision_chain(
        (actor, target), _first_contact(actor.track_id), _config()
    )

    assert len(decisions) == 1
    assert decisions[0].detection_layer == "L2"
    assert decisions[0].contact_kind == "swept_contact"
    assert decisions[0].minimum_distance_mm == 0


def test_l3_sustained_contact_detects_motion_propagation_without_new_entry() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        ((1, 100, 200), (2, 100, 200), (3, 100, 200), (4, 108, 200), (5, 116, 200)),
    )
    target = _track(
        "shot-0001-anchor-02",
        ((1, 157, 200), (2, 157, 200), (3, 157, 200), (4, 165, 200), (5, 173, 200)),
    )

    decisions = observe_delayed_collision_chain(
        (actor, target), _first_contact(actor.track_id), _config()
    )

    assert len(decisions) == 1
    assert decisions[0].detection_layer == "L3"
    assert decisions[0].contact_kind == "sustained_contact_impulse"
    assert decisions[0].decision_frame_id > decisions[0].contact_frame_id


def test_three_layers_deduplicate_one_pair_contact() -> None:
    actor = _track(
        "shot-0001-anchor-01",
        ((1, 100, 200), (2, 110, 200), (3, 145, 200), (4, 155, 200), (5, 160, 200)),
    )
    target = _track(
        "shot-0001-anchor-02",
        ((1, 220, 200), (2, 210, 200), (3, 200, 200), (4, 205, 200), (5, 210, 200)),
    )

    decisions = observe_delayed_collision_chain(
        (actor, target), _first_contact(actor.track_id), _config()
    )

    assert len(decisions) == 1
    assert decisions[0].detection_layer == "L1"


def test_l1_cushion_contact_uses_fixed_track_geometry() -> None:
    track = _track(
        "shot-0001-anchor-01",
        ((1, 80, 200), (2, 55, 200), (3, 28, 200), (4, 28, 200)),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=32
    )

    assert len(decisions) == 1
    assert decisions[0].actor_id == track.track_id
    assert decisions[0].rail == "left"
    assert decisions[0].detection_layer == "L1"


def test_l2_cushion_contact_uses_velocity_reversal_when_frame_skips_rail() -> None:
    track = ActiveBallTrack(
        shot_id="shot-0001",
        track_id="shot-0001-anchor-01",
        group=BallGroup.STRIPE,
        samples=(
            ActiveTrackSample(1, 33, 40, 200, -20, 0, "observed", 0.95, "d1"),
            ActiveTrackSample(2, 66, 40, 200, 20, 0, "observed", 0.95, "d2"),
        ),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=32
    )

    assert len(decisions) == 1
    assert decisions[0].rail == "left"
    assert decisions[0].detection_layer == "L2"


def test_l3_cushion_contact_handles_ball_already_touching_rail() -> None:
    track = ActiveBallTrack(
        shot_id="shot-0001",
        track_id="shot-0001-anchor-01",
        group=BallGroup.STRIPE,
        samples=(
            ActiveTrackSample(1, 33, 28, 200, 0, 0, "observed", 0.95, "d1"),
            ActiveTrackSample(2, 66, 28, 200, 0, 0, "observed", 0.95, "d2"),
            ActiveTrackSample(3, 99, 36, 200, 8, 0, "observed", 0.95, "d3"),
        ),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=32
    )

    assert len(decisions) == 1
    assert decisions[0].rail == "left"
    assert decisions[0].detection_layer == "L3"


def test_cushion_contact_uses_three_real_observations_across_predicted_frames() -> None:
    track = ActiveBallTrack(
        shot_id="shot-0001",
        track_id="shot-0001-anchor-01",
        group=BallGroup.STRIPE,
        samples=(
            ActiveTrackSample(1, 33, 100, 200, -15, 0, "observed", 0.95, "d1"),
            ActiveTrackSample(2, 66, 72, 200, -14, 0, "predicted", 0.80, None),
            ActiveTrackSample(3, 99, 55, 200, -10, 0, "observed", 0.95, "d3"),
            ActiveTrackSample(4, 132, 54, 200, -4, 0, "predicted", 0.80, None),
            ActiveTrackSample(5, 165, 68, 200, -1, 0, "reacquired", 0.90, "d5"),
        ),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=45
    )

    assert len(decisions) == 1
    assert decisions[0].rail == "left"
    assert decisions[0].contact_frame_id == 3
    assert decisions[0].decision_frame_id == 5
    assert decisions[0].detection_layer == "L2"
    assert decisions[0].minimum_distance_mm == 55


def test_cushion_rail_names_follow_left_bottom_table_origin() -> None:
    track = _track(
        "shot-0001-anchor-01",
        ((1, 500, 80), (2, 500, 55), (3, 500, 28), (4, 500, 28)),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=32
    )

    assert len(decisions) == 1
    assert decisions[0].rail == "bottom"


def test_cushion_contact_is_inferred_from_one_tracks_turn_at_pocket_jaw() -> None:
    track = _track(
        "shot-0001-anchor-10",
        (
            (1, 220, 220),
            (2, 150, 150),
            (7, 68, 58),
            (8, 110, 100),
            (9, 155, 145),
        ),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=45
    )

    assert len(decisions) == 1
    assert decisions[0].actor_id == track.track_id
    assert decisions[0].rail == "pocket-jaw:left-bottom"
    assert decisions[0].detection_layer == "L3"
    assert decisions[0].contact_frame_id == 7
    assert decisions[0].decision_frame_id == 8
    assert decisions[0].contact_kind == "trajectory_inferred_rebound"
    assert decisions[0].evidence_frame_ids == (2, 7, 8)


def test_trajectory_turn_far_from_table_boundary_is_not_a_cushion_contact() -> None:
    track = _track(
        "shot-0001-anchor-10",
        ((1, 400, 220), (2, 450, 220), (7, 470, 250), (8, 450, 280)),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=1000, table_width_mm=500, contact_distance_mm=45
    )

    assert decisions == ()


def test_trajectory_turn_near_ball_collision_is_not_misreported_as_cushion() -> None:
    track = _track(
        "shot-0001-anchor-10",
        ((1, 220, 220), (2, 150, 150), (7, 68, 58), (8, 110, 100)),
    )
    collision = DelayedCollisionDecision(
        actor_id=track.track_id,
        target_id="shot-0001-anchor-11",
        contact_frame_id=7,
        decision_frame_id=8,
        status="observed",
        confidence=0.9,
        evidence_frame_ids=(7, 8),
    )

    decisions = observe_three_layer_cushions(
        (track,),
        table_length_mm=1000,
        table_width_mm=500,
        contact_distance_mm=45,
        ball_collisions=(collision,),
    )

    assert decisions == ()


def test_cushion_contact_fits_inbound_and_outbound_segments_when_turn_spans_two_windows() -> None:
    track = _track(
        "shot-0001-anchor-10",
        (
            (130, 95, 80),
            (131, 83, 81),
            (132, 47, 95),
            (133, 46, 105),
            (134, 60, 116),
            (135, 77, 125),
        ),
    )

    decisions = observe_three_layer_cushions(
        (track,), table_length_mm=2540, table_width_mm=1270, contact_distance_mm=45
    )

    assert len(decisions) == 1
    assert decisions[0].actor_id == track.track_id
    assert decisions[0].rail == "left"
    assert decisions[0].detection_layer == "L3"
    assert decisions[0].contact_frame_id == 133
    assert decisions[0].decision_frame_id == 135
    assert decisions[0].contact_kind == "trajectory_fitted_rebound"
    assert decisions[0].evidence_frame_ids == (130, 131, 132, 133, 134, 135)


def test_ball_appearance_signatures_keep_colour_evidence_inside_each_box() -> None:
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[20:60, 20:60] = (20, 20, 230)
    image[20:60, 100:140] = (230, 20, 20)
    frame = _frame(
        1,
        CalibratedBallObservation(
            detection_id="red",
            x_mm=100,
            y_mm=100,
            confidence=0.95,
            group=BallGroup.STRIPE,
            group_confidence=0.95,
            bbox_xyxy_px=(20, 20, 60, 60),
        ),
        CalibratedBallObservation(
            detection_id="blue",
            x_mm=200,
            y_mm=100,
            confidence=0.95,
            group=BallGroup.STRIPE,
            group_confidence=0.95,
            bbox_xyxy_px=(100, 20, 140, 60),
        ),
    )

    signatures = extract_ball_appearance_signatures({1: image}, (frame,))

    assert set(signatures) == {"red", "blue"}
    assert np.isclose(np.linalg.norm(signatures["red"]), 1.0)
    assert np.isclose(np.linalg.norm(signatures["blue"]), 1.0)
    assert float(signatures["red"] @ signatures["blue"]) < 0.5
