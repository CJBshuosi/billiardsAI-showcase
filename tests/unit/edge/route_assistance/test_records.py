from __future__ import annotations

from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.records import (
    FieldFactReport,
    FieldSessionRecord,
    FieldSessionStatus,
    FieldShotRecord,
    SessionLimitCode,
    ShotMediaRange,
)


def _match() -> MatchControlSnapshot:
    return MatchControlSnapshot(
        red_user_id="manual-red",
        blue_user_id="manual-blue",
        breaker=WristbandRole.RED,
        active_role=WristbandRole.RED,
        red_group=BallGroupAssignment.STRIPE,
        blue_group=BallGroupAssignment.SOLID,
        revision=4,
    )


def _media() -> ShotMediaRange:
    return ShotMediaRange(
        artifact_id="artifact-1",
        source_locator="presentation.mp4",
        source_kind="development_replay",
        start_ms=1_000,
        end_ms=7_000,
    )


def _shot() -> FieldShotRecord:
    return FieldShotRecord(
        session_id="session-1",
        shot_id="shot-0001",
        sequence=1,
        actor_role=WristbandRole.RED,
        started_ms=3_000,
        settled_ms=5_000,
        recommendation_id="recommendation-1",
        power_band="medium_light",
        power_percent=40,
        media=_media(),
        limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
    )


def test_shot_media_range_is_bounded_and_development_only() -> None:
    media = _media()

    assert media.duration_ms == 6_000
    with pytest.raises(ValueError, match="canonical source"):
        replace(media, source_kind="raw_camera")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="end after it starts"):
        replace(media, end_ms=1_000)


def test_session_keeps_roles_separate_from_users_and_groups() -> None:
    session = FieldSessionRecord.open(
        session_id="session-1",
        artifact_id="artifact-1",
        created_at_utc="2026-08-26T10:00:00Z",
    )

    assert session.match == MatchControlSnapshot()
    assert session.status is FieldSessionStatus.OPEN
    assert session.ended_at_utc is None

    bound = replace(session, match=_match())
    assert bound.match.red_user_id == "manual-red"
    assert bound.match.red_group is BallGroupAssignment.STRIPE
    assert bound.match.active_role is WristbandRole.RED


def test_session_status_and_end_timestamp_must_agree() -> None:
    session = FieldSessionRecord.open(
        session_id="session-1",
        artifact_id="artifact-1",
        created_at_utc="2026-08-26T10:00:00Z",
    )

    with pytest.raises(ValueError, match="ended timestamp"):
        replace(session, status=FieldSessionStatus.ENDED)
    with pytest.raises(ValueError, match="open session"):
        replace(session, ended_at_utc="2026-08-26T11:00:00Z")


def test_shot_requires_consistent_timing_power_and_limitations() -> None:
    shot = _shot()

    assert shot.media.start_ms <= shot.started_ms < shot.settled_ms <= shot.media.end_ms
    with pytest.raises(ValueError, match="settle after it starts"):
        replace(shot, settled_ms=shot.started_ms)
    with pytest.raises(ValueError, match="power band and percent"):
        replace(shot, power_percent=None)
    with pytest.raises(ValueError, match="unique"):
        replace(
            shot,
            limitations=(
                SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,
                SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,
            ),
        )


def test_facts_report_is_closed_to_coaching_fields() -> None:
    with pytest.raises(TypeError):
        FieldFactReport(
            session_id="session-1",
            artifact_id="artifact-1",
            created_at_utc="2026-08-26T10:00:00Z",
            ended_at_utc="2026-08-26T11:00:00Z",
            match=_match(),
            shots=(_shot(),),
            limitations=(SessionLimitCode.NO_GLOBAL_FIXED_BALL_ID,),
            training_plan="aim drills",  # type: ignore[call-arg]
        )
