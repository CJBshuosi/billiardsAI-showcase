from __future__ import annotations

from dataclasses import replace

import pytest

from billiards_ai.edge.realtime_chain.models import (
    AnchorEvidence,
    CueEvidence,
    EventEvidence,
    EvidenceSource,
    EvidenceStatus,
    ShotReplayEvidence,
    TableBallEvidence,
)
from billiards_ai.edge.realtime_chain.projection import (
    ProjectionConfig,
    project_shot_evidence,
)


def _source() -> EvidenceSource:
    return EvidenceSource(
        artifact_id="artifact-43",
        source_video_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )


def _shot() -> ShotReplayEvidence:
    return ShotReplayEvidence(
        session_id="session-43",
        shot_id="shot-0001",
        complete=True,
        before_frame_id=10,
        after_frame_id=20,
        anchors=(
            AnchorEvidence(
                anchor_id="shot-0001-anchor-01",
                group="cue",
                x_mm=100,
                y_mm=200,
                confidence=0.92,
            ),
            AnchorEvidence(
                anchor_id="shot-0001-anchor-02",
                group="solid",
                x_mm=400,
                y_mm=200,
                confidence=0.9,
            ),
        ),
        cue_samples=(
            CueEvidence(11, 1000, "confirmed", 100, 200, 0.9),
            CueEvidence(12, 1100, "confirmed", 130, 200, 0.88),
            CueEvidence(13, 1200, "confirmed", 170, 200, 0.86),
        ),
        events=(
            EventEvidence(
                event_id="shot-0001-event-first-contact",
                event_type="first_contact",
                status="observed",
                timestamp_ms=1200,
                actor_id="shot-0001-cue",
                target_id="shot-0001-anchor-02",
                evidence_frame_ids=(12, 13),
                confidence=0.84,
            ),
            EventEvidence(
                event_id="shot-0001-event-cushion-0001",
                event_type="cushion",
                status="observed",
                timestamp_ms=1300,
                actor_id="shot-0001-cue",
                target_id=None,
                evidence_frame_ids=(12, 13, 14),
                confidence=0.8,
            ),
            EventEvidence(
                event_id="shot-0001-event-pot-0001",
                event_type="pot",
                status="observed",
                timestamp_ms=1400,
                actor_id="shot-0001-fragment-0001",
                target_id=None,
                evidence_frame_ids=(14, 20),
                confidence=0.82,
            ),
        ),
        after_balls=(TableBallEvidence(group="cue", x_mm=170, y_mm=200, confidence=0.9),),
    )


def _features(projection) -> dict[str, object]:
    return {feature.code: feature.value for feature in projection.profile_increment.features}


def test_admits_traceable_cue_started_shot_and_projects_only_supported_habits() -> None:
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=_shot(),
        config=ProjectionConfig(),
    )

    assert projection.status is EvidenceStatus.ADMITTED
    assert projection.cue_start.status is EvidenceStatus.ADMITTED
    assert projection.table_difference.before_ball_count == 2
    assert projection.table_difference.after_ball_count == 1
    assert _features(projection) == {
        "cue_motion_proxy_peak_mm_s": 400.0,
        "cue_travel_mm": 70.0,
        "cue_end_x_mm": 170.0,
        "cue_end_y_mm": 200.0,
        "first_contact_target_group": "solid",
        "observed_cushion_count": 1,
        "observed_pot_count": 1,
    }
    assert projection.profile_increment.source == _source()
    assert all(feature.evidence_ids for feature in projection.profile_increment.features)
    assert "success" not in " ".join(_features(projection))


def test_incomplete_shot_is_pending_and_cannot_contribute_profile_values() -> None:
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=replace(_shot(), complete=False, after_frame_id=None, after_balls=()),
        config=ProjectionConfig(),
    )

    assert projection.status is EvidenceStatus.PENDING
    assert projection.reason == "shot_incomplete"
    assert projection.profile_increment.features == ()


def test_low_confidence_or_insufficient_cue_motion_abstains_without_profile_pollution() -> None:
    low_confidence = tuple(replace(sample, confidence=0.49) for sample in _shot().cue_samples)
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=replace(_shot(), cue_samples=low_confidence),
        config=ProjectionConfig(cue_confidence_min=0.7),
    )

    assert projection.status is EvidenceStatus.ABSTAINED
    assert projection.reason == "cue_start_evidence_insufficient"
    assert projection.profile_increment.features == ()


def test_ambiguous_event_is_retained_but_never_becomes_a_habit_feature() -> None:
    ambiguous = replace(
        _shot().events[0],
        status="abstained",
        target_id=None,
        confidence=0.0,
        reason="first_contact_anchor_ambiguous",
    )
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=replace(_shot(), events=(ambiguous, *_shot().events[1:])),
        config=ProjectionConfig(),
    )

    assert projection.status is EvidenceStatus.ADMITTED
    assert projection.event_decisions[0].status is EvidenceStatus.ABSTAINED
    assert "first_contact_target_group" not in _features(projection)


def test_missing_general_ball_collision_observer_is_explicitly_unsupported() -> None:
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=_shot(),
        config=ProjectionConfig(),
    )

    decisions = {decision.event_type: decision for decision in projection.event_decisions}
    assert decisions["ball_ball_collision"].status is EvidenceStatus.UNSUPPORTED
    assert decisions["ball_ball_collision"].reason == "general_ball_collision_observer_unavailable"


def test_shot_end_decision_is_explicit_and_requires_complete_after_table() -> None:
    complete = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=_shot(),
        config=ProjectionConfig(),
    )
    incomplete = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=replace(_shot(), complete=False, after_frame_id=None, after_balls=()),
        config=ProjectionConfig(),
    )

    complete_end = next(item for item in complete.event_decisions if item.event_type == "shot_end")
    incomplete_end = next(
        item for item in incomplete.event_decisions if item.event_type == "shot_end"
    )
    assert complete_end.status is EvidenceStatus.ADMITTED
    assert complete_end.evidence_frame_ids == (20,)
    assert incomplete_end.status is EvidenceStatus.PENDING
    assert incomplete_end.reason == "shot_end_not_observed"


def test_front_observer_collision_and_shot_end_keep_existing_quality_gate_semantics() -> None:
    collision = EventEvidence(
        event_id="shot-0001-event-ball-collision-0001",
        event_type="ball_ball_collision",
        status="inferred",
        timestamp_ms=1350,
        actor_id="shot-0001-fragment-0001",
        target_id="shot-0001-fragment-0002",
        evidence_frame_ids=(13, 14, 16, 17),
        confidence=0.8,
        reason="bounded_fragment_gap_motion_transfer",
    )
    shot_end = EventEvidence(
        event_id="shot-0001-event-shot-end",
        event_type="shot_end",
        status="observed",
        timestamp_ms=1500,
        actor_id=None,
        target_id=None,
        evidence_frame_ids=(19, 20),
        confidence=1.0,
    )
    projection = project_shot_evidence(
        user_id="user-1",
        source=_source(),
        shot=replace(_shot(), events=(*_shot().events, collision, shot_end)),
        config=ProjectionConfig(),
    )

    collision_decisions = [
        item for item in projection.event_decisions if item.event_type == "ball_ball_collision"
    ]
    shot_end_decisions = [
        item for item in projection.event_decisions if item.event_type == "shot_end"
    ]
    assert len(collision_decisions) == 1
    assert collision_decisions[0].status is EvidenceStatus.PENDING
    assert len(shot_end_decisions) == 1
    assert shot_end_decisions[0].status is EvidenceStatus.ADMITTED
    assert all("collision" not in feature.code for feature in projection.profile_increment.features)


def test_invalid_source_lineage_fails_closed_before_projection() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        project_shot_evidence(
            user_id="user-1",
            source=replace(_source(), manifest_sha256="not-a-hash"),
            shot=_shot(),
            config=ProjectionConfig(),
        )
