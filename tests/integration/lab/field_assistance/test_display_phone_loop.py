from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_fact_report_payload,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldSessionStatus,
    FieldShotRecord,
    SessionLimitCode,
    ShotMediaRange,
)


def _command(
    command_id: str,
    *,
    sequence: int,
    revision: int,
    actor: str,
    command_type: str,
    payload: dict[str, str],
) -> FieldCommandEnvelope:
    return FieldCommandEnvelope(
        command_id=command_id,
        session_id="session-display-phone",
        actor_user_id=actor,
        command_type=command_type,
        payload=payload,
        expected_revision=revision,
        server_sequence=sequence,
    )


def test_file_replay_phone_loop_converges_without_duplicate_commands(
    tmp_path: Path,
) -> None:
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: datetime(2026, 8, 27, 10, 0, tzinfo=UTC),
        session_id_factory=lambda: "session-display-phone",
    )
    store.open_or_resume_session(artifact_id="video43-replay")
    commands = (
        _command(
            "cmd-red",
            sequence=1,
            revision=0,
            actor="user-red",
            command_type="claim_role",
            payload={"role": "red"},
        ),
        _command(
            "cmd-blue",
            sequence=2,
            revision=1,
            actor="user-blue",
            command_type="claim_role",
            payload={"role": "blue"},
        ),
        _command(
            "cmd-breaker",
            sequence=3,
            revision=2,
            actor="user-red",
            command_type="set_breaker",
            payload={"role": "red"},
        ),
        _command(
            "cmd-groups",
            sequence=4,
            revision=3,
            actor="user-red",
            command_type="assign_groups",
            payload={"red_group": "stripe"},
        ),
    )
    results = tuple(
        store.apply_field_command(command, reduce_field_command) for command in commands
    )
    duplicate = store.apply_field_command(commands[-1], reduce_field_command)

    shot = FieldShotRecord(
        session_id="session-display-phone",
        shot_id="shot-0001",
        sequence=1,
        actor_role=store.load_session("session-display-phone").match.active_role,
        started_ms=3_000,
        settled_ms=5_000,
        recommendation_id="route-video43-1",
        power_band="medium_light",
        power_percent=40,
        media=ShotMediaRange(
            artifact_id="video43-replay",
            source_locator="presentation.mp4",
            source_kind="development_replay",
            start_ms=1_000,
            end_ms=6_000,
        ),
        limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
    )
    store.record_shot_start(replace(shot, settled_ms=None))
    store.record_shot_settle(shot)
    end_result = store.apply_field_command(
        _command(
            "cmd-end",
            sequence=5,
            revision=4,
            actor="user-red",
            command_type="end_session",
            payload={},
        ),
        reduce_field_command,
    )
    session = store.load_session("session-display-phone")
    report = field_fact_report_payload(
        compose_field_fact_report(session, store.list_shots(session.session_id))
    )

    assert tuple(result.command_id for result in results) == (
        "cmd-red",
        "cmd-blue",
        "cmd-breaker",
        "cmd-groups",
    )
    assert all(result.outcome == "applied" for result in results)
    assert duplicate == results[-1]
    assert end_result.outcome == "applied"
    assert session.status is FieldSessionStatus.ENDED
    assert set(report["limitations"]) >= {
        "development_replay_media",
        "no_global_fixed_ball_id",
        "shot_local_fragment_and_cushion_limits",
    }
