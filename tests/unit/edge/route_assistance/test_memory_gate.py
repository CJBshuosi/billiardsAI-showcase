"""Actual SQLite Stage 2 receipt consumption; all players/media are synthetic."""

from dataclasses import replace

import pytest

from billiards_ai.cloud.growth import reviewed_memory
from billiards_ai.edge.route_assistance import memory_gate
from tests.unit.cloud.growth.test_evidence_ingest import USER_ID
from tests.unit.cloud.growth.test_reviewed_memory import event, setup
from tests.unit.edge.route_assistance.test_turn_ownership import make_store, match, shot
from tests.unit.lab.field_assistance.test_turn_sources import source_pair  # noqa: F401


def attempt(store, session_id="session-1"):
    batch, authority, repo, service = setup()
    with pytest.raises(memory_gate.FieldMemoryAdmissionError) as error:
        reviewed_memory.consume_field_reviewed_event(
            event(batch),
            field_store=store,
            field_session_id=session_id,
            authenticated_user_id=USER_ID,
            approved_repository=authority,
            ingest_service=service,
        )
    assert repo.list_batches(USER_ID) == ()
    assert repo.list_receipts() == ()
    return error.value.code


def test_bound_replay_user_never_updates_profile_even_with_a_valid_reviewed_event(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    start = shot()
    store.record_shot_start(
        start,
        source_mainline1_artifact_id="a" * 64,
        source_start_event_id="shot-0001-event-shot-start",
    )
    store.record_shot_settle(replace(start, settled_ms=200))
    store.end_session("session-1")
    assert store.get_shot_ownership("session-1", start.shot_id).actor_user_id == "user-red"
    assert attempt(store) == "DEVELOPMENT_REPLAY_EXCLUDED"


def test_unknown_receipt_never_backfills_from_current_roster(tmp_path):
    store = make_store(tmp_path)
    store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    store.save_match("session-1", match(), expected_revision=0)
    assert attempt(store) == "OWNERSHIP_UNKNOWN"
    assert store.get_shot_ownership("session-1", "shot-0001").actor_user_id is None


def test_missing_receipt_is_blocked_not_reconstructed(tmp_path):
    store = make_store(tmp_path)
    store.record_shot_start(shot())
    # The Stage 2 reader explicitly permits legacy missing receipts; no migration.
    store.get_shot_ownership = lambda session_id, shot_id: None
    assert attempt(store) == "OWNERSHIP_UNKNOWN"


def test_wrong_source_receipt_is_rejected_before_any_growth_write(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    receipt = store.get_shot_ownership("session-1", "shot-0001")
    store.get_shot_ownership = lambda session_id, shot_id: replace(
        receipt, source_artifact_id="foreign"
    )
    assert attempt(store) == "OWNERSHIP_LINEAGE_MISMATCH"


def test_window_only_or_missing_mainline_source_is_unknown_and_never_promoted(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(shot())  # Window-derived boundary without verified original source.
    assert attempt(store) == "OWNERSHIP_UNKNOWN"
    assert store.get_shot_ownership("session-1", "shot-0001").actor_user_id is None


def test_two_users_multiple_sessions_remain_isolated_and_never_accumulate_replay(tmp_path):
    from billiards_ai.edge.route_assistance.control import WristbandRole
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

    ids = iter(("session-1", "session-2"))
    store = SQLiteFieldRecordStore(tmp_path / "multi.sqlite3", session_id_factory=lambda: next(ids))
    expected = []
    for role in (WristbandRole.RED, WristbandRole.BLUE):
        session = store.open_or_resume_session(artifact_id="replay-1")
        store.save_match(session.session_id, match(role=role), expected_revision=0)
        start = replace(shot(role=role), session_id=session.session_id)
        store.record_shot_start(
            start,
            source_mainline1_artifact_id="a" * 64,
            source_start_event_id="shot-0001-event-shot-start",
        )
        store.record_shot_settle(replace(start, settled_ms=200))
        store.end_session(session.session_id)
        assert attempt(store, session.session_id) == "DEVELOPMENT_REPLAY_EXCLUDED"
        expected.append(store.get_shot_ownership(session.session_id, start.shot_id).actor_user_id)
        assert len(store.list_shots(session.session_id)) == 1
    assert expected == ["user-red", "user-blue"]
    assert store.get_shot_ownership("session-1", "shot-0001").actor_user_id == "user-red"


def test_legacy_m1_bound_receipt_without_verified_event_is_not_upgraded(tmp_path):
    import json
    import sqlite3

    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(
        shot(),
        source_mainline1_artifact_id="a" * 64,
        source_start_event_id="shot-0001-event-shot-start",
    )
    with sqlite3.connect(store.path) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
        )
        del payload["source_start_event_id"]
        del payload["evidence_reason"]
        raw = json.dumps(payload)
        db.execute("UPDATE field_shot_ownership SET payload_json=?", (raw,))
    assert attempt(store) == "OWNERSHIP_SOURCE_UNVERIFIED"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0] == raw


@pytest.mark.parametrize("source_mode", ["verified", "missing", "invalid"])
def test_real_stage2_builder_receipts_preserve_facts_but_cannot_enter_growth(
    source_pair,  # noqa: F811
    tmp_path,
    source_mode,
):
    """Local synthetic producer→Stage2 builder→SQLite→Stage3; not recorded-video E2E."""
    from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope
    from billiards_ai.lab.field_assistance.server import build_field_assistance_server
    from tests.unit.lab.field_assistance.test_live_state import FakeClock

    inputs, manifest, _ = source_pair
    if source_mode == "missing":
        inputs = None
    elif source_mode == "invalid":
        inputs = replace(inputs, artifact_path=tmp_path / "absent")
    now = FakeClock()
    server = build_field_assistance_server(
        artifact_path=tmp_path / "replay" / manifest["artifact_id"],
        host="127.0.0.1",
        port=0,
        control_token="synthetic-local-test",
        monotonic=now,
        duration_probe=lambda _: 500,
        mainline1_inputs=inputs,
    )
    try:
        state = server.field_state
        store = state.record_store
        store.save_match(state.session_id, match(), expected_revision=0)
        state.match = match()
        state.replay_clock.start()
        now.advance(0.401)
        state.before_field_command(
            FieldCommandEnvelope(
                "probe",
                state.session_id,
                "user-red",
                "continue_turn",
                {},
                1,
                1,
            )
        )
        before = store.get_shot_ownership(state.session_id, "shot-0001")
        assert attempt(store, state.session_id) == (
            "DEVELOPMENT_REPLAY_EXCLUDED" if source_mode == "verified" else "OWNERSHIP_UNKNOWN"
        )
        assert before == store.get_shot_ownership(state.session_id, "shot-0001")
        assert bool(before.source_start_event_id) == (source_mode == "verified")
        assert store.list_shots(state.session_id)[0].settled_ms == 400
        assert store.get_turn_confirmation(state.session_id, "shot-0001").status == "pending"
    finally:
        server.server_close()
