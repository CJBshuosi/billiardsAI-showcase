from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_fact_report_payload,
    field_sync_envelope,
)
from billiards_ai.edge.route_assistance.records import (
    FieldSessionRecord,
    FieldSessionStatus,
    FieldShotRecord,
    SessionLimitCode,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.sync import SyncMediaItem
from billiards_ai.shared.models.miniprogram import canonical_sync_hash


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


def _session(*, ended: bool = True) -> FieldSessionRecord:
    return FieldSessionRecord(
        session_id="session-1",
        artifact_id="artifact-1",
        status=FieldSessionStatus.ENDED if ended else FieldSessionStatus.OPEN,
        created_at_utc="2026-08-26T10:00:00Z",
        ended_at_utc="2026-08-26T11:00:00Z" if ended else None,
        match=_match(),
    )


def _shot(*, sequence: int = 1, settled: bool = True) -> FieldShotRecord:
    return FieldShotRecord(
        session_id="session-1",
        shot_id=f"shot-{sequence:04d}",
        sequence=sequence,
        actor_role=WristbandRole.RED,
        started_ms=3_000 + sequence * 100,
        settled_ms=5_000 + sequence * 100 if settled else None,
        recommendation_id="recommendation-1",
        power_band="medium_light",
        power_percent=40,
        media=ShotMediaRange(
            artifact_id="artifact-1",
            source_locator="presentation.mp4",
            source_kind="development_replay",
            start_ms=1_000,
            end_ms=10_000,
        ),
        limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
    )


def test_report_contains_only_persisted_facts_and_explicit_limitations() -> None:
    report = compose_field_fact_report(_session(), (_shot(),))
    payload = field_fact_report_payload(report)

    assert payload["schema_version"] == "1.0"
    assert payload["report_type"] == "field_session_facts"
    assert payload["shot_count"] == 1
    assert payload["match"]["red_participant_ref"] == "manual-red"
    assert payload["shots"][0]["actor_role"] == "red"
    assert payload["shots"][0]["power"] == {
        "band": "medium_light",
        "approximate_percent": 40,
        "personalized": False,
    }
    assert payload["limitations"] == [
        "development_replay_media",
        "no_global_fixed_ball_id",
        "shot_local_fragment_and_cushion_limits",
        "no_automatic_foul_verdict",
    ]
    assert not ({"score", "weaknesses", "training_plan", "highlight"} & payload.keys())


def test_usb_only_report_does_not_claim_development_replay_media() -> None:
    """Catches carrying a replay-only limitation into immutable USB postgame facts."""
    shot = _shot()
    usb_media = ShotMediaRange(
        artifact_id="artifact-1",
        source_locator="usb:epoch-1",
        source_kind="usb_live",
        source_epoch_id="epoch-1",
        start_ms=shot.media.start_ms,
        end_ms=shot.media.end_ms,
    )

    report = compose_field_fact_report(_session(), (replace(shot, media=usb_media),))

    assert SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA not in report.limitations


def test_usb_report_accepts_fresh_epoch_per_shot_and_exports_each_lineage() -> None:
    shots = tuple(
        replace(
            _shot(sequence=sequence),
            media=ShotMediaRange(
                artifact_id="artifact-1",
                source_locator=f"usb:{epoch}",
                source_kind="usb_live",
                source_epoch_id=epoch,
                start_ms=3_000,
                end_ms=6_000,
            ),
            limitations=(),
        )
        for sequence, epoch in ((1, "epoch-1"), (2, "epoch-2"))
    )

    payload = field_fact_report_payload(compose_field_fact_report(_session(), shots))

    assert [shot["media"]["source_epoch_id"] for shot in payload["shots"]] == [
        "epoch-1",
        "epoch-2",
    ]


def test_report_rejects_open_or_unsettled_session() -> None:
    with pytest.raises(ValueError, match="ended session"):
        compose_field_fact_report(_session(ended=False), ())
    with pytest.raises(ValueError, match="settled shots"):
        compose_field_fact_report(_session(), (_shot(settled=False),))


def test_report_rejects_non_contiguous_or_foreign_shots() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        compose_field_fact_report(_session(), (_shot(sequence=2),))
    with pytest.raises(ValueError, match="same session"):
        compose_field_fact_report(
            _session(),
            (replace(_shot(), session_id="session-other"),),
        )


def test_report_preserves_unknown_actor_as_a_data_gap() -> None:
    report = compose_field_fact_report(
        _session(),
        (replace(_shot(), actor_role=None),),
    )

    assert field_fact_report_payload(report)["shots"][0]["actor_role"] is None


def test_report_builds_a_strict_cloud_sync_envelope() -> None:
    report = compose_field_fact_report(_session(), (_shot(),))
    envelope = field_sync_envelope(
        report,
        (
            SyncMediaItem(
                clip_id="clip-shot-0001",
                shot_id="shot-0001",
                local_path=Path("/absolute/clip-shot-0001.mp4"),
                content_type="video/mp4",
                content_sha256="a" * 64,
                byte_size=1024,
            ),
        ),
    )

    assert envelope.session.participants[0].user_id == "manual-red"
    assert envelope.session.evidence_source_kind == "development_replay"
    assert envelope.media[0].shot_id == "shot-0001"
    assert len(canonical_sync_hash(envelope)) == 64


def test_report_builds_a_facts_only_cloud_sync_envelope() -> None:
    """Catches forcing generated media into immutable ended-session fact sync."""
    shot = _shot()
    report = compose_field_fact_report(
        _session(),
        (
            replace(
                shot,
                media=ShotMediaRange(
                    artifact_id="artifact-1",
                    source_locator="usb:epoch-1",
                    source_kind="usb_live",
                    source_epoch_id="epoch-1",
                    start_ms=shot.media.start_ms,
                    end_ms=shot.media.end_ms,
                ),
                limitations=(),
                live_evidence_digest="a" * 64,
                live_job_id="job-1",
            ),
        ),
    )

    envelope = field_sync_envelope(report, (), postgame_manifest_sha256="b" * 64)

    assert envelope.session.session_id == "session-1"
    assert envelope.session.postgame_manifest_sha256 == "b" * 64
    assert envelope.session.evidence_source_kind == "usb_live"
    assert [shot.shot_id for shot in envelope.session.shots] == ["shot-0001"]
    assert envelope.media == ()
    assert len(canonical_sync_hash(envelope)) == 64


def test_report_rejects_a_nonempty_partial_media_envelope() -> None:
    """Catches silently treating a subset of generated clips as complete session media."""
    report = compose_field_fact_report(_session(), (_shot(), _shot(sequence=2)))
    one_clip = SyncMediaItem(
        clip_id="clip-shot-0001",
        shot_id="shot-0001",
        local_path=Path("/absolute/clip-shot-0001.mp4"),
        content_type="video/mp4",
        content_sha256="a" * 64,
        byte_size=1024,
    )

    with pytest.raises(ValueError, match="empty or contain exactly"):
        field_sync_envelope(report, (one_clip,))


def test_report_rejects_mixed_media_sources() -> None:
    usb = replace(
        _shot(sequence=2),
        media=ShotMediaRange(
            artifact_id="artifact-1",
            source_locator="usb:epoch-1",
            source_kind="usb_live",
            source_epoch_id="epoch-1",
            start_ms=1_000,
            end_ms=10_000,
        ),
        limitations=(),
    )

    with pytest.raises(ValueError, match="one media source"):
        compose_field_fact_report(_session(), (_shot(), usb))


def test_facts_only_envelope_exports_usb_job_and_explicit_source_lineage() -> None:
    shot = _shot()
    usb = replace(
        shot,
        media=ShotMediaRange(
            artifact_id="artifact-1",
            source_locator="usb:epoch-1",
            source_kind="usb_live",
            source_epoch_id="epoch-1",
            start_ms=shot.media.start_ms,
            end_ms=shot.media.end_ms,
        ),
        limitations=(),
        live_evidence_digest="a" * 64,
        live_job_id="job-1",
    )

    envelope = field_sync_envelope(
        compose_field_fact_report(_session(), (usb,)),
        (),
        postgame_manifest_sha256="b" * 64,
    )

    assert envelope.session.evidence_source_kind == "usb_live"
    assert envelope.session.postgame_manifest_sha256 == "b" * 64
    assert envelope.session.shots[0].live_job_id == "job-1"


def test_zero_shot_facts_only_envelope_requires_and_uses_explicit_usb_source() -> None:
    report = compose_field_fact_report(_session(), ())

    with pytest.raises(ValueError, match="explicit evidence source"):
        field_sync_envelope(report, (), postgame_manifest_sha256="b" * 64)

    envelope = field_sync_envelope(
        report,
        (),
        postgame_manifest_sha256="b" * 64,
        evidence_source_kind="usb_live",
    )

    assert envelope.session.evidence_source_kind == "usb_live"
    assert envelope.session.postgame_manifest_sha256 == "b" * 64
    assert envelope.session.shots == ()
    assert envelope.media == ()
