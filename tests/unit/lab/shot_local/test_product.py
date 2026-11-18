from __future__ import annotations

from billiards_ai.lab.shot_local.active_tracks import (
    CushionContactDecision,
    DelayedCollisionDecision,
)
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
    ShotAnchor,
    ShotLocalEvent,
    ShotLocalResult,
    ShotWindow,
)
from billiards_ai.lab.shot_local.product import (
    CanonicalShotAnalysis,
    build_canonical_shot_analyses,
    canonical_world_event_counts,
    canonical_world_event_rows,
    shot_local_product_active_config,
)


def _evidence(*events: ShotLocalEvent) -> ShotLocalResult:
    return ShotLocalResult(
        shot_id="shot-0001",
        window=ShotWindow(
            shot_id="shot-0001",
            start_frame_id=10,
            end_frame_id=80,
            before_frame_id=9,
            after_frame_id=81,
            start_ms=333,
            end_ms=2667,
            complete=True,
        ),
        anchors=(),
        cue_samples=(),
        fragments=(),
        events=events,
    )


def _event(
    event_type: str,
    *,
    actor_id: str | None,
    target_id: str | None = None,
    evidence_frame_ids: tuple[int, ...],
    reason: str | None = None,
) -> ShotLocalEvent:
    return ShotLocalEvent(
        shot_id="shot-0001",
        event_id=f"shot-0001-event-{event_type}",
        event_type=event_type,
        status="observed",
        timestamp_ms=1000,
        actor_id=actor_id,
        target_id=target_id,
        evidence_frame_ids=evidence_frame_ids,
        confidence=0.91,
        reason=reason,
    )


def _analysis(
    *events: ShotLocalEvent,
    collisions: tuple[DelayedCollisionDecision, ...] = (),
    cushions: tuple[CushionContactDecision, ...] = (),
    lineage: dict[str, str | None] | None = None,
) -> CanonicalShotAnalysis:
    evidence = _evidence(*events)
    return CanonicalShotAnalysis(
        evidence=evidence,
        active_tracks=(),
        collisions=collisions,
        cushions=cushions,
        retained_events=events,
        fragment_lineage={} if lineage is None else lineage,
    )


def test_product_active_config_preserves_the_reviewed_milestone_policy() -> None:
    config = shot_local_product_active_config()

    assert (
        config.position_gate_mm,
        config.unmatched_cost,
        config.process_noise_mm,
        config.measurement_noise_mm,
        config.max_prediction_frames,
        config.appearance_weight,
        config.group_mismatch_cost,
        config.collision_distance_mm,
        config.moving_speed_mm_per_frame,
        config.stationary_speed_mm_per_frame,
        config.collision_lookahead_frames,
        config.collision_candidate_margin,
    ) == (160, 210, 14, 6, 5, 75, 180, 72, 6, 4, 5, 0.08)


def test_fragment_pot_maps_to_the_same_stable_anchor_track() -> None:
    analysis = _analysis(
        _event(
            "pot",
            actor_id="shot-0001-fragment-0002",
            evidence_frame_ids=(70, 81),
            reason="pocket:left-bottom",
        ),
        lineage={"shot-0001-fragment-0002": "shot-0001-anchor-10"},
    )

    pot = canonical_world_event_rows(analysis)[0]

    assert pot["actor_id"] == "shot-0001-anchor-10"
    assert pot["pocket"] == "left-bottom"
    assert pot["status"] == "observed"


def test_unmapped_fragment_event_abstains_instead_of_claiming_an_identity() -> None:
    analysis = _analysis(
        _event(
            "pot",
            actor_id="shot-0001-fragment-0002",
            evidence_frame_ids=(70, 81),
            reason="pocket:left-bottom",
        ),
        lineage={"shot-0001-fragment-0002": None},
    )

    pot = canonical_world_event_rows(analysis)[0]

    assert pot["status"] == "abstained"
    assert pot["actor_id"] is None
    assert pot["reason"] == "stable_track_mapping_ambiguous"


def test_world_event_rows_use_one_identity_complete_schema() -> None:
    start = _event(
        "shot_start",
        actor_id="shot-0001-cue",
        evidence_frame_ids=(10, 11),
    )
    first_contact = _event(
        "first_contact",
        actor_id="shot-0001-cue",
        target_id="shot-0001-anchor-09",
        evidence_frame_ids=(20, 21),
    )
    collision = DelayedCollisionDecision(
        actor_id="shot-0001-anchor-09",
        target_id="shot-0001-anchor-10",
        contact_frame_id=30,
        decision_frame_id=33,
        status="observed",
        confidence=0.98,
        evidence_frame_ids=(29, 30, 33),
        detection_layer="L2",
        contact_kind="swept_contact",
        minimum_distance_mm=69.4,
    )
    cushion = CushionContactDecision(
        actor_id="shot-0001-anchor-10",
        rail="left",
        contact_frame_id=40,
        decision_frame_id=42,
        detection_layer="L3",
        confidence=0.82,
        minimum_distance_mm=47.3,
        contact_kind="trajectory_fitted_rebound",
        evidence_frame_ids=(38, 40, 42),
    )
    analysis = _analysis(start, first_contact, collisions=(collision,), cushions=(cushion,))

    rows = canonical_world_event_rows(analysis)

    assert [row["event_type"] for row in rows] == [
        "shot_start",
        "first_contact",
        "ball_collision",
        "cushion",
    ]
    assert rows[2]["target_id"] == "shot-0001-anchor-10"
    assert rows[2]["detection_layer"] == "L2"
    assert rows[3]["rail"] == "left"
    assert all({"actor_id", "target_id", "rail", "pocket"} <= row.keys() for row in rows)
    assert canonical_world_event_counts(rows) == {
        "shot_start_count": 1,
        "first_contact_count": 1,
        "ball_collision_count": 1,
        "cushion_count": 1,
        "pot_count": 0,
        "shot_end_count": 0,
        "event_count_by_layer": {"L1": 0, "L2": 1, "L3": 1},
    }


def test_canonical_analysis_builds_tracks_from_the_pre_shot_anchor_roster() -> None:
    anchor = ShotAnchor(
        shot_id="shot-0001",
        anchor_id="shot-0001-anchor-01",
        source_detection_id="ball-0",
        group=BallGroup.STRIPE,
        x_mm=100,
        y_mm=200,
        confidence=0.95,
    )
    first_contact = _event(
        "first_contact",
        actor_id="shot-0001-cue",
        target_id=anchor.anchor_id,
        evidence_frame_ids=(1, 2),
    )
    evidence = ShotLocalResult(
        shot_id="shot-0001",
        window=ShotWindow(
            shot_id="shot-0001",
            start_frame_id=1,
            end_frame_id=3,
            before_frame_id=0,
            after_frame_id=3,
            start_ms=33,
            end_ms=99,
            complete=True,
        ),
        anchors=(anchor,),
        cue_samples=(),
        fragments=(),
        events=(first_contact,),
    )
    frames = tuple(
        ObservationFrame(
            frame_id=frame_id,
            timestamp_ms=frame_id * 33,
            observations=(
                CalibratedBallObservation(
                    detection_id=f"ball-{frame_id}",
                    x_mm=100 + frame_id * 5,
                    y_mm=200,
                    confidence=0.95,
                    group=BallGroup.STRIPE,
                    group_confidence=0.95,
                ),
            ),
        )
        for frame_id in range(4)
    )

    analyses = build_canonical_shot_analyses(
        evidence_results=(evidence,),
        observations=frames,
        images_by_frame={},
        event_config=ShotLocalEventConfig(
            first_contact_distance_mm=35,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=120,
            cushion_distance_mm=10,
            pocket_distance_mm=15,
            pot_missing_frames=3,
            after_match_distance_mm=20,
            table_length_mm=1600,
            table_width_mm=800,
        ),
        active_config=shot_local_product_active_config(),
    )

    assert analyses[0].active_tracks[0].track_id == anchor.anchor_id
    assert [sample.frame_id for sample in analyses[0].active_tracks[0].samples] == [0, 1, 2, 3]
    assert analyses[0].retained_events == (first_contact,)
