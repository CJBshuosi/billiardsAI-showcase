"""Local-only confirmation seam; these are not phone or reviewed-evidence tests."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.commands import FieldCommandEnvelope, reduce_field_command
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from tests.fixtures.persistence.legacy_field_schema import (
    assert_legacy_schema,
    create_legacy_database,
    legacy_rows,
)
from tests.unit.edge.route_assistance.test_turn_ownership import make_store, match, shot


def prepared(tmp_path):
    store = make_store(tmp_path)
    store.save_match("session-1", match(), expected_revision=0)
    store.record_shot_start(shot(), source_mainline1_artifact_id="a" * 64)
    store.record_shot_settle(replace(shot(), settled_ms=150))
    return store


def command(**changes):
    return replace(
        FieldCommandEnvelope("cmd-1", "session-1", "user-red", "switch_turn", {}, 1, 1), **changes
    )


def test_settle_creates_durable_abstention_not_automatic_switch(tmp_path):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    assert pending.status == "pending"
    assert pending.action == "abstain"
    assert pending.reason == "AUTHORITATIVE_RULE_FACTS_INCOMPLETE"
    assert pending.resolved_command_id is None
    assert store.load_session("session-1").match.revision == 1
    store.record_shot_settle(replace(shot(), settled_ms=150))
    assert (
        SQLiteFieldRecordStore(store.path).get_turn_confirmation("session-1", "shot-0001")
        == pending
    )


def test_explicit_ref_resolves_exact_pending_atomically_and_replays(tmp_path):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    result = store.apply_field_command(
        command(), reduce_field_command, confirmation_ref=pending.reference
    )
    assert result.outcome == "applied" and result.resulting_revision == 2
    resolved = store.get_turn_confirmation("session-1", "shot-0001")
    assert resolved.status == "resolved" and resolved.resolved_command_id == "cmd-1"
    assert resolved.action == "abstain"  # human match operation does not review algorithm facts
    assert (
        store.apply_field_command(
            command(), reduce_field_command, confirmation_ref=pending.reference
        )
        == result
    )
    for ref in [
        None,
        replace(pending.reference, shot_id="shot-0002"),
        replace(pending.reference, evidence_digest="b" * 64),
    ]:
        assert (
            store.apply_field_command(
                command(), reduce_field_command, confirmation_ref=ref
            ).result_code
            == "COMMAND_IDEMPOTENCY_CONFLICT"
        )
    assert store.get_turn_confirmation("session-1", "shot-0001") == resolved


def test_no_ref_command_cannot_resolve_or_gain_ref_on_retry(tmp_path):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    result = store.apply_field_command(command(), reduce_field_command)
    assert result.outcome == "applied"
    stale = store.get_turn_confirmation("session-1", "shot-0001")
    assert stale.status == "stale" and stale.stale_reason == "MATCH_REVISION_CHANGED"
    assert stale.resolved_command_id is None
    assert (
        store.apply_field_command(
            command(), reduce_field_command, confirmation_ref=pending.reference
        ).result_code
        == "COMMAND_IDEMPOTENCY_CONFLICT"
    )
    assert store.get_turn_confirmation("session-1", "shot-0001") == stale


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "old-session"),
        ("shot_id", "shot-0002"),
        ("evidence_digest", "f" * 64),
        ("match_revision", 0),
    ],
)
def test_invalid_ref_cannot_apply_command(tmp_path, field, value):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    ref = replace(pending.reference, **{field: value})
    result = store.apply_field_command(command(), reduce_field_command, confirmation_ref=ref)
    assert result.result_code == "TURN_CONFIRMATION_MISMATCH"
    assert store.load_session("session-1").match.revision == 1
    assert store.get_turn_confirmation("session-1", "shot-0001") == pending


def test_two_shots_same_revision_cannot_use_old_reference(tmp_path):
    store = prepared(tmp_path)
    old = store.get_turn_confirmation("session-1", "shot-0001")
    store.record_shot_start(shot(2), source_mainline1_artifact_id="a" * 64)
    store.record_shot_settle(replace(shot(2), settled_ms=250))
    assert store.get_turn_confirmation("session-1", "shot-0001").stale_reason == "NEW_SHOT_STARTED"
    result = store.apply_field_command(
        command(), reduce_field_command, confirmation_ref=old.reference
    )
    assert result.result_code == "TURN_CONFIRMATION_MISMATCH"
    latest = store.get_turn_confirmation("session-1", "shot-0002")
    assert latest.status == "pending"
    assert (
        store.apply_field_command(
            command(command_id="cmd-2", server_sequence=2),
            reduce_field_command,
            confirmation_ref=latest.reference,
        ).outcome
        == "applied"
    )
    assert store.get_turn_confirmation("session-1", "shot-0001").resolved_command_id is None


def test_pending_update_failure_rolls_back_match_and_original_ledger(tmp_path):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER fail_resolution BEFORE UPDATE ON field_turn_confirmations "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(ValueError):
        store.apply_field_command(
            command(), reduce_field_command, confirmation_ref=pending.reference
        )
    assert store.load_session("session-1").match.revision == 1
    assert store.get_turn_confirmation("session-1", "shot-0001") == pending
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_applied_commands").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["save", "end"])
def test_non_command_mutations_stale_unresolved_pending(tmp_path, operation):
    store = prepared(tmp_path)
    if operation == "save":
        store.save_match("session-1", match(2), expected_revision=1)
    else:
        store.end_session("session-1")
    stale = store.get_turn_confirmation("session-1", "shot-0001")
    assert stale.status == "stale" and stale.resolved_command_id is None


def test_ref_does_not_bypass_auth_or_order(tmp_path):
    store = prepared(tmp_path)
    pending = store.get_turn_confirmation("session-1", "shot-0001")
    assert (
        store.apply_field_command(
            command(server_sequence=2), reduce_field_command, confirmation_ref=pending.reference
        ).result_code
        == "COMMAND_OUT_OF_ORDER"
    )
    assert (
        store.apply_field_command(
            command(actor_user_id="intruder"),
            reduce_field_command,
            confirmation_ref=pending.reference,
        ).result_code
        == "COMMAND_FORBIDDEN"
    )
    assert store.get_turn_confirmation("session-1", "shot-0001") == pending


def test_competing_commands_resolve_once_and_duplicates_share_original_receipt(tmp_path):
    store = prepared(tmp_path)
    reference = store.get_turn_confirmation("session-1", "shot-0001").reference

    def apply(envelope):
        return store.apply_field_command(envelope, reduce_field_command, confirmation_ref=reference)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(apply, [command(), command(), command(command_id="competitor")]))
    winner = store.get_turn_confirmation("session-1", "shot-0001").resolved_command_id
    assert winner in {"cmd-1", "competitor"}
    assert store.load_session("session-1").match.revision == 2
    assert all(result.outcome == "applied" for result in results if result.command_id == winner)
    assert all(result.outcome == "rejected" for result in results if result.command_id != winner)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_applied_commands").fetchone()[0] == 1


def test_v5_upgrade_and_failure_preserve_legacy_rows(tmp_path, monkeypatch):
    from billiards_ai.edge.route_assistance import persistence

    store = prepared(tmp_path)
    session = store.load_session("session-1")
    actor = store.get_shot_ownership("session-1", "shot-0001")
    legacy_path = tmp_path / "field-v5.db"
    before = create_legacy_database(store.path, legacy_path, 5)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 5)
    schema = persistence._TURN_CONFIRMATION_SCHEMA
    monkeypatch.setattr(persistence, "_TURN_CONFIRMATION_SCHEMA", schema + "INVALID SQL;")
    with pytest.raises(ValueError):
        SQLiteFieldRecordStore(legacy_path)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 5)
        assert legacy_rows(db, ownership=True) == before
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE name='field_turn_confirmations'"
            ).fetchall()
            == []
        )
    monkeypatch.setattr(persistence, "_TURN_CONFIRMATION_SCHEMA", schema)
    upgraded = SQLiteFieldRecordStore(legacy_path)
    assert upgraded.get_turn_confirmation("session-1", "shot-0001") is None
    assert upgraded.load_session("session-1") == session
    assert upgraded.get_shot_ownership("session-1", "shot-0001") == actor
    with sqlite3.connect(legacy_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert legacy_rows(db, ownership=True) == before
    upgraded.record_shot_settle(replace(shot(), settled_ms=150))
    assert upgraded.get_turn_confirmation("session-1", "shot-0001") is None
    with sqlite3.connect(legacy_path) as db:
        assert legacy_rows(db, ownership=True) == before


def test_settle_and_pending_insert_rollback_together(tmp_path):
    store = make_store(tmp_path)
    store.record_shot_start(shot())
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER fail_pending BEFORE INSERT ON field_turn_confirmations "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    with pytest.raises(ValueError):
        store.record_shot_settle(replace(shot(), settled_ms=150))
    assert store.list_shots("session-1")[0].settled_ms is None
    assert store.get_turn_confirmation("session-1", "shot-0001") is None
