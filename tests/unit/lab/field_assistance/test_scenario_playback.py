"""Isolated callback identity/settings guards; phone-flow proof lives in integration."""

from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandApplication,
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.records import FieldSessionRecord, FieldSessionStatus
from billiards_ai.lab.field_assistance.scenario_playback import AuthenticatedScenarioPlayback
from billiards_ai.lab.mainline1_scenario_capture import ScenarioPlaybackGate


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "foreign-envelope",
        "foreign-current",
        "ended",
        "result-id",
        "rejected",
        "result-code",
        "stale-revision",
        "active-role",
        "settings-revision",
        "settings-payload",
        "slot-change",
    ],
)
def test_callback_requires_exact_current_open_identity_and_successful_frozen_settings(invalid):
    gate = ScenarioPlaybackGate()
    session = FieldSessionRecord.open(
        session_id="managed-22222222-2222-4222-8222-222222222222",
        artifact_id="fixture",
        created_at_utc="2026-11-01T00:00:00Z",
    )
    observer = AuthenticatedScenarioPlayback(session.session_id, gate)
    for sequence, actor, kind, payload in (
        (1, "alice", "claim_role", {"role": "red"}),
        (2, "bob", "claim_role", {"role": "blue"}),
        (
            3,
            "alice",
            "set_breaker",
            dict(
                role="red",
                mode="assistance",
                format="single",
                race_target=None,
                ranking_intent=False,
                settings_revision=1,
            ),
        ),
    ):
        envelope = FieldCommandEnvelope(
            f"command-{sequence}",
            session.session_id,
            actor,
            kind,
            payload,
            session.match.revision,
            sequence,
        )
        reduction = reduce_field_command(session.match, envelope, has_unsettled_shot=False)
        session = replace(session, match=reduction.match)
        application = FieldCommandApplication(
            envelope.command_id,
            "applied",
            "APPLIED",
            session.match.revision,
        )
        if sequence < 3:
            observer.observe(envelope, application, session)
            assert not gate.is_released()
    if invalid == "foreign-envelope":
        envelope = replace(envelope, session_id="another-scene")
    elif invalid == "foreign-current":
        session = replace(session, session_id="another-scene")
    elif invalid == "ended":
        session = replace(
            session, status=FieldSessionStatus.ENDED, ended_at_utc="2026-11-01T01:00:00Z"
        )
    elif invalid == "result-id":
        application = replace(application, command_id="another-command")
    elif invalid == "rejected":
        application = replace(application, outcome="rejected")
    elif invalid == "result-code":
        application = replace(application, result_code="COMMAND_FORBIDDEN")
    elif invalid == "stale-revision":
        application = replace(application, resulting_revision=2)
    elif invalid == "active-role":
        session = replace(session, match=replace(session.match, active_role=WristbandRole.BLUE))
    elif invalid == "settings-revision":
        session = replace(session, match=replace(session.match, settings_revision=2))
    elif invalid == "slot-change":
        session = replace(session, match=replace(session.match, blue_user_id="charlie"))
    elif invalid == "settings-payload":
        envelope = replace(envelope, payload={"role": "red"})
    observer.observe(envelope, application, session)
    assert gate.is_released() is (invalid is None)


@pytest.mark.parametrize("invalid", ["legacy", "resume", "released"])
def test_builder_rejects_gate_outside_fresh_managed_scene_before_any_storage(tmp_path, invalid):
    from types import SimpleNamespace

    from billiards_ai.lab.field_assistance.server import build_live_field_assistance_server

    gate = ScenarioPlaybackGate()
    if invalid == "released":
        gate.release()
    with pytest.raises(ValueError, match="fresh managed blackbox"):
        build_live_field_assistance_server(
            artifact_path=tmp_path / "artifacts",
            state_db_path=tmp_path / "state.sqlite3",
            host="127.0.0.1",
            port=0,
            control_token="fixture",
            source_epoch_factory=lambda: pytest.fail("opened capture"),
            calibration_id="calibration",
            runtime_factory=lambda *_: pytest.fail("runtime"),
            managed_startup=None if invalid == "legacy" else SimpleNamespace(),
            fresh_session_on_start=invalid != "resume",
            scenario_playback_gate=gate,
        )
    assert list(tmp_path.iterdir()) == []
