"""Real raw pixels + signed/attested command worker; only remote HTTP is controlled."""

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.lab import mainline1_scenario_capture as scenario
from billiards_ai.lab.field_assistance import server as service
from billiards_ai.lab.field_assistance.cloud_gateway import (
    DevelopmentAttestedFieldCloudGateway,
    SignedFieldCloudGateway,
)
from billiards_ai.lab.mainline1_field_runtime import build_live_runtime_factory
from billiards_ai.lab.mainline1_shot_local_config import load_mainline1_shot_local_config
from billiards_ai.shared.calibration.homography import calibrate
from tests.integration.lab.field_assistance.test_blackbox_camera_product_path import (
    _CAPTURE_CONFIG,
    _SHOT_CONFIG_PATH,
    _EmptyClassifier,
    _EmptyDetector,
    _observe_runtime_publication,
    _ScenarioRender,
    _until,
    _write_scenario,
)
from tests.integration.lab.field_assistance.test_managed_live_startup import options_at, seed
from tests.unit.lab.field_assistance.test_managed_startup import (
    BINDING,
    Credentials,
    Remote,
    coordinator,
)

SETTINGS = dict(
    role="red",
    mode="assistance",
    format="single",
    race_target=None,
    ranking_intent=False,
    settings_revision=1,
)


@pytest.fixture
def scene(tmp_path):
    video = tmp_path / "scenario.avi"
    _write_scenario(video, _ScenarioRender(total_frames=480))
    gate = scenario.ScenarioPlaybackGate()
    options, _ = options_at(tmp_path)
    seed(options)  # Retain a real predecessor; fresh startup must retire it.
    runtime = build_live_runtime_factory(
        capture_factory=lambda epoch: scenario.ScenarioVideoCaptureBackend(
            scenario.ScenarioCaptureConfig(video, _CAPTURE_CONFIG),
            source_epoch_id=epoch,
            playback_gate=gate,
        ),
        capture_config=_CAPTURE_CONFIG,
        calibration=calibrate(
            "managed-blackbox",
            [(0, 119), (159, 119), (159, 0), (0, 0)],
            table_length_mm=2540,
            table_width_mm=1270,
        ),
        detector=_EmptyDetector(),
        identity_classifier=_EmptyClassifier(),
        shot_config=load_mainline1_shot_local_config(_SHOT_CONFIG_PATH),
        capture_setup_id="managed-file",
        calibration_artifact_id="cal-1",
        config_path=_SHOT_CONFIG_PATH,
        ring_root=tmp_path / "rings",
        jobs_root=tmp_path / "jobs",
    )
    options["runtime_factory"] = runtime
    options["rule_profile_id"] = PROFILE_ID
    remote = Remote()
    server = service.build_live_field_assistance_server(
        **options,
        managed_startup=coordinator(remote),
        fresh_session_on_start=True,
        scenario_playback_gate=gate,
    )
    state = server.field_state
    state.wait_until_capture_ready(timeout_seconds=3)
    gateway = DevelopmentAttestedFieldCloudGateway(
        SignedFieldCloudGateway(credentials=Credentials(), transport=remote), "dev"
    )
    worker = service._BackgroundFieldCommandWorker(
        gateway=gateway,
        store=state.record_store,
        match_lock=state.match_lock,
        before_application=state.before_field_command,
        match_changed=state.set_command_match,
        application_changed=state.handle_command_application,
        command_reducer=state.reduce_live_command,
    )
    sequence = 0

    def command(actor, kind, payload, *, envelope=None):
        nonlocal sequence
        if envelope is None:
            sequence += 1
        revision = state.record_store.load_session(state.session_id).match.revision
        envelope = envelope or FieldCommandEnvelope(
            f"command-{sequence}",
            state.session_id,
            actor,
            kind,
            payload,
            revision,
            sequence,
        )
        remote.commands.append(envelope)
        assert worker.claim_and_apply_once(raise_errors=True)
        return envelope, remote.acks[-1]

    try:
        yield state, gate, command, remote
    finally:
        worker.stop()
        server.server_close()


def _later_frame(state, after):
    return _until(
        lambda: (
            frame
            if (frame := state.projection_source.runtime.current_capture_frame) is not None
            and frame.capture_frame_id > after
            else None
        ),
        timeout_seconds=3,
        label="continued raw capture cadence",
    )


def _admin(state, role, user):
    state.mutate(
        dict(
            command="emergency_set_participant",
            role=role,
            user_id=user,
            reason="Correct the wristband at the table",
        )
    )


def test_signed_current_phone_claims_release_pixels_into_real_product_chain(scene, monkeypatch):
    state, gate, command, _ = scene
    events, unauthorized = _observe_runtime_publication(monkeypatch)
    first = _later_frame(state, 3)
    later = _later_frame(state, first.capture_frame_id + 12)
    assert later.captured_at_ns > first.captured_at_ns
    assert later.mjpeg == first.mjpeg
    assert not gate.is_released()  # READY and elapsed capture ticks never release.
    assert state.record_store.list_shots(state.session_id) == ()
    _, red = command("alice", "claim_role", {"role": "red"})
    assert red["public_result_code"] == "APPLIED" and not gate.is_released()
    _, blue = command("bob", "claim_role", {"role": "blue"})
    assert blue["public_result_code"] == "APPLIED" and not gate.is_released()
    assert state.payload()["display"]["phase"] == "WAITING_FOR_SETUP"
    _, settings = command("alice", "set_breaker", SETTINGS)
    assert settings["public_result_code"] == "APPLIED" and gate.is_released()
    assert state.payload()["display"]["phase"] == "WAITING_FOR_SHOT"
    moved = _until(
        lambda: (
            frame
            if (frame := state.projection_source.runtime.current_capture_frame) is not None
            and frame.mjpeg != first.mjpeg
            else None
        ),
        timeout_seconds=3,
        label="real source motion after authenticated settings",
    )
    assert moved.capture_frame_id > later.capture_frame_id
    shots = _until(
        lambda: (
            shots
            if (shots := state.record_store.list_shots(state.session_id))
            and shots[0].settled_ms is not None
            else None
        ),
        timeout_seconds=8,
        label="real pixel-driven shot settlement",
    )
    assert len(shots) == 1
    shot = shots[0]
    fact = state.record_store.get_live_shot_fact(state.session_id, shot.shot_id)
    ownership = state.record_store.get_shot_ownership(state.session_id, shot.shot_id)
    receipt = state.record_store.get_rule_decision(state.session_id, shot.shot_id)
    assert fact is not None and ownership is not None and receipt is not None
    assert fact.actor_user_id == ownership.actor_user_id == "alice"
    assert receipt.source_epoch_id == ownership.source_epoch_id == "fresh-epoch"
    assert receipt.status == "abstained"
    assert {event.kind for event in events} >= {"SHOT_STARTED", "SHOT_SETTLED"}
    assert unauthorized == []
    assert gate.is_released()


def test_admin_slots_and_breaker_never_supply_phone_claim_provenance(scene):
    state, gate, command, _ = scene
    _admin(state, "red", "alice")
    _admin(state, "blue", "bob")
    state.mutate(dict(command="emergency_set_breaker", role="red", reason="Set first breaker now"))
    assert state.match.mode is None
    assert state.match.match_format is None
    assert not state.match.settings_frozen
    assert state.payload()["display"]["phase"] == "WAITING_FOR_SETUP"
    assert not gate.is_released()
    _, rejected = command("bob", "claim_role", {"role": "blue"})
    assert rejected["public_result_code"] == "COMMAND_FORBIDDEN"
    _, applied = command("alice", "set_breaker", SETTINGS)
    assert applied["public_result_code"] == "APPLIED"
    assert state.match.settings_frozen
    assert not gate.is_released()
    first = _later_frame(state, 2)
    assert _later_frame(state, first.capture_frame_id + 4).mjpeg == first.mjpeg


def test_admin_role_change_and_old_applied_replays_do_not_recertify_claims(scene):
    state, gate, command, _ = scene
    red, _ = command("alice", "claim_role", {"role": "red"})
    blue, _ = command("bob", "claim_role", {"role": "blue"})
    _admin(state, "blue", "charlie")
    _admin(state, "blue", "bob")
    for old in (red, blue):
        _, replay = command(old.actor_user_id, old.command_type, old.payload, envelope=old)
        assert replay["public_result_code"] == "APPLIED"
    command("alice", "set_breaker", SETTINGS)
    assert not gate.is_released()


def test_release_reclaim_uses_current_distinct_actors_and_rejects_old_or_foreign_claims(scene):
    state, gate, command, remote = scene
    command("alice", "claim_role", {"role": "red"})
    command("bob", "claim_role", {"role": "blue"})
    command("bob", "release_role", {"role": "blue"})
    _, same_actor = command("alice", "claim_role", {"role": "blue"})
    assert same_actor["public_result_code"] == "COMMAND_FORBIDDEN"
    command("charlie", "claim_role", {"role": "blue"})
    # An actual application to another retained session cannot affect this latch.
    foreign = state.record_store.open_or_resume_session(artifact_id="foreign-fixture")
    command(
        "outsider",
        "claim_role",
        {"role": "red"},
        envelope=FieldCommandEnvelope(
            "foreign",
            foreign.session_id,
            "outsider",
            "claim_role",
            {"role": "red"},
            0,
            1,
        ),
    )
    assert not gate.is_released()
    _, old = command(
        "old-player",
        "claim_role",
        {"role": "blue"},
        envelope=FieldCommandEnvelope(
            "old-claim",
            BINDING.session_id,
            "old-player",
            "claim_role",
            {"role": "blue"},
            0,
            1,
        ),
    )
    assert old["public_result_code"] == "SESSION_ENDED"
    assert not gate.is_released()
    # Legacy breaker only does not freeze legal product settings.
    command("alice", "set_breaker", {"role": "red"})
    assert not gate.is_released()
    _, invalid_settings = command(
        "alice",
        "set_breaker",
        {**SETTINGS, "format": "race", "race_target": 1},
    )
    assert invalid_settings["public_result_code"] == "COMMAND_INVALID"
    assert not gate.is_released()
    command("alice", "set_breaker", SETTINGS)
    assert gate.is_released()
    assert state.match.blue_user_id == "charlie"
    assert remote.acks[-1]["public_result_code"] == "APPLIED"


@pytest.mark.parametrize("unavailable", ["ended", "closed"])
def test_terminal_or_closed_current_scene_cannot_release_from_worker_callback(scene, unavailable):
    state, gate, command, remote = scene
    command("alice", "claim_role", {"role": "red"})
    command("bob", "claim_role", {"role": "blue"})
    if unavailable == "closed":
        state.close_projection()
        revision = state.record_store.load_session(state.session_id).match.revision
        acknowledgements = len(remote.acks)
        # A new command is no longer applied/acknowledged after service closure.
        with pytest.raises(ValueError, match="LIVE_AUTHORITY_UNAVAILABLE"):
            command("alice", "set_breaker", SETTINGS)
        assert len(remote.acks) == acknowledgements
        assert state.record_store.load_session(state.session_id).match.revision == revision
    else:
        state.record_store.end_session(state.session_id)
        _, result = command("alice", "set_breaker", SETTINGS)
        assert result["public_result_code"] == "SESSION_ENDED"
    assert not gate.is_released()


def test_released_scene_latches_through_later_successful_admin_correction(scene):
    state, gate, command, _ = scene
    command("alice", "claim_role", {"role": "red"})
    command("bob", "claim_role", {"role": "blue"})
    command("alice", "set_breaker", SETTINGS)
    assert gate.is_released()
    _admin(state, "blue", "charlie")
    assert state.match.blue_user_id == "charlie"
    assert gate.is_released()
