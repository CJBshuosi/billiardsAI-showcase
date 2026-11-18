"""Compose a local session report from persisted facts without coaching claims."""

from __future__ import annotations

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.records import (
    FieldFactReport,
    FieldSessionRecord,
    FieldSessionStatus,
    FieldShotRecord,
    SessionLimitCode,
)
from billiards_ai.edge.route_assistance.sync import SyncMediaItem
from billiards_ai.shared.models.miniprogram import (
    CanonicalMediaSource,
    EdgeSyncEnvelopeV1,
    MiniProgramClipV1,
    MiniProgramParticipantV1,
    MiniProgramPowerV1,
    MiniProgramSessionManifestV1,
    MiniProgramShotV1,
)

_BASE_REPORT_LIMITATIONS = (
    SessionLimitCode.NO_GLOBAL_FIXED_BALL_ID,
    SessionLimitCode.SHOT_LOCAL_FRAGMENT_AND_CUSHION_LIMITS,
    SessionLimitCode.NO_AUTOMATIC_FOUL_VERDICT,
)


def compose_field_fact_report(
    session: FieldSessionRecord,
    shots: tuple[FieldShotRecord, ...],
) -> FieldFactReport:
    """Validate and freeze one evidence-backed, facts-only field report."""

    if session.status is not FieldSessionStatus.ENDED or session.ended_at_utc is None:
        raise ValueError("field fact report requires an ended session")
    ordered = tuple(sorted(shots, key=lambda shot: shot.sequence))
    if tuple(shot.sequence for shot in ordered) != tuple(range(1, len(ordered) + 1)):
        raise ValueError("field fact report requires contiguous shot sequences")
    if any(shot.session_id != session.session_id for shot in ordered):
        raise ValueError("field fact report shots must belong to the same session")
    if any(shot.settled_ms is None for shot in ordered):
        raise ValueError("field fact report requires settled shots")
    if any(shot.media.artifact_id != session.artifact_id for shot in ordered):
        raise ValueError("field fact report media must belong to the session artifact")
    source_kinds = {shot.media.source_kind for shot in ordered}
    if len(source_kinds) > 1:
        raise ValueError("field fact report requires one media source")
    limitations = _BASE_REPORT_LIMITATIONS
    if any(shot.media.source_kind == "development_replay" for shot in ordered):
        limitations = (SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA, *limitations)
    return FieldFactReport(
        session_id=session.session_id,
        artifact_id=session.artifact_id,
        created_at_utc=session.created_at_utc,
        ended_at_utc=session.ended_at_utc,
        match=session.match,
        shots=ordered,
        limitations=limitations,
    )


def _match_payload(report: FieldFactReport) -> dict[str, object]:
    match = report.match
    return {
        "red_participant_ref": match.red_user_id,
        "blue_participant_ref": match.blue_user_id,
        "breaker": None if match.breaker is None else match.breaker.value,
        "active_role_at_end": (None if match.active_role is None else match.active_role.value),
        "red_group": None if match.red_group is None else match.red_group.value,
        "blue_group": None if match.blue_group is None else match.blue_group.value,
        "revision": match.revision,
    }


def _shot_payload(shot: FieldShotRecord) -> dict[str, object]:
    power = (
        None
        if shot.power_band is None
        else {
            "band": shot.power_band,
            "approximate_percent": shot.power_percent,
            "personalized": False,
        }
    )
    return {
        "shot_id": shot.shot_id,
        "sequence": shot.sequence,
        "actor_role": None if shot.actor_role is None else shot.actor_role.value,
        "started_ms": shot.started_ms,
        "settled_ms": shot.settled_ms,
        "recommendation_id": shot.recommendation_id,
        "power": power,
        "media": {
            "artifact_id": shot.media.artifact_id,
            "source_locator": shot.media.source_locator,
            "source_kind": shot.media.source_kind,
            "source_epoch_id": shot.media.source_epoch_id,
            "start_ms": shot.media.start_ms,
            "end_ms": shot.media.end_ms,
            "duration_ms": shot.media.duration_ms,
        },
        "limitations": [item.value for item in shot.limitations],
    }


def field_fact_report_payload(report: FieldFactReport) -> dict[str, object]:
    """Serialize the closed report contract for local inspection and future sync."""

    return {
        "schema_version": "1.0",
        "report_type": "field_session_facts",
        "session_id": report.session_id,
        "artifact_id": report.artifact_id,
        "created_at_utc": report.created_at_utc,
        "ended_at_utc": report.ended_at_utc,
        "match": _match_payload(report),
        "shot_count": len(report.shots),
        "shots": [_shot_payload(shot) for shot in report.shots],
        "limitations": [item.value for item in report.limitations],
    }


def field_sync_envelope(
    report: FieldFactReport,
    media: tuple[SyncMediaItem, ...],
    *,
    postgame_manifest_sha256: str | None = None,
    evidence_source_kind: CanonicalMediaSource | None = None,
) -> EdgeSyncEnvelopeV1:
    """Build the strict upload envelope from an ended local fact report."""

    by_shot = {item.shot_id: item for item in media}
    if media and (
        len(by_shot) != len(media) or set(by_shot) != {shot.shot_id for shot in report.shots}
    ):
        raise ValueError("sync media must be empty or contain exactly one file for every shot")
    source_kinds = {shot.media.source_kind for shot in report.shots}
    if len(source_kinds) > 1:
        raise ValueError("sync envelope requires one explicit evidence source")
    if source_kinds:
        inferred_source_kind = source_kinds.pop()
        if evidence_source_kind is not None and evidence_source_kind != inferred_source_kind:
            raise ValueError("explicit evidence source conflicts with report shots")
        evidence_source_kind = inferred_source_kind
    elif evidence_source_kind is None:
        raise ValueError("sync envelope requires one explicit evidence source")
    if (
        not media
        and evidence_source_kind == "usb_live"
        and (
            postgame_manifest_sha256 is None
            or any(shot.live_job_id is None for shot in report.shots)
        )
    ):
        raise ValueError("facts-only sync requires frozen postgame and live job lineage")
    match = report.match
    participant_values = (
        (WristbandRole.RED, match.red_user_id, match.red_group),
        (WristbandRole.BLUE, match.blue_user_id, match.blue_group),
    )
    participants = tuple(
        MiniProgramParticipantV1(
            user_id=user_id,
            role=role.value,
            group="open" if group is None else group.value,
        )
        for role, user_id, group in participant_values
        if user_id is not None
    )
    shots = tuple(
        MiniProgramShotV1(
            shot_id=shot.shot_id,
            sequence=shot.sequence,
            actor_role=None if shot.actor_role is None else shot.actor_role.value,
            started_ms=shot.started_ms,
            settled_ms=shot.settled_ms,
            recommendation_id=shot.recommendation_id,
            live_job_id=shot.live_job_id,
            power=(
                None
                if shot.power_band is None
                else MiniProgramPowerV1(
                    band=shot.power_band,
                    approximate_percent=shot.power_percent,
                    personalized=False,
                )
            ),
            limitations=tuple(item.value for item in shot.limitations),
        )
        for shot in report.shots
    )
    session = MiniProgramSessionManifestV1(
        session_id=report.session_id,
        artifact_id=report.artifact_id,
        status="ended",
        created_at_utc=report.created_at_utc,
        ended_at_utc=report.ended_at_utc,
        postgame_manifest_sha256=postgame_manifest_sha256,
        evidence_source_kind=evidence_source_kind,
        participants=participants,
        breaker=None if match.breaker is None else match.breaker.value,
        active_role_at_end=(None if match.active_role is None else match.active_role.value),
        shots=shots,
        limitations=tuple(item.value for item in report.limitations),
    )
    clips = (
        tuple(
            MiniProgramClipV1(
                clip_id=item.clip_id,
                session_id=report.session_id,
                shot_id=shot.shot_id,
                artifact_id=report.artifact_id,
                relative_path=f"clips/{item.clip_id}.mp4",
                source_kind=shot.media.source_kind,
                start_ms=shot.media.start_ms,
                end_ms=shot.media.end_ms,
                duration_ms=shot.media.duration_ms,
                content_sha256=item.content_sha256,
                byte_size=item.byte_size,
                media_type=item.content_type,
            )
            for shot in report.shots
            for item in (by_shot[shot.shot_id],)
        )
        if media
        else ()
    )
    return EdgeSyncEnvelopeV1(session=session, media=clips)
