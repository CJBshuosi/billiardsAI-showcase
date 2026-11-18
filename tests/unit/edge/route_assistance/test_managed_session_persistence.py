from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot
from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore

NOW = datetime(2026, 10, 30, tzinfo=UTC)
REQUEST = "11111111-1111-4111-8111-111111111111"
BIRTH = "22222222-2222-4222-8222-222222222222"
SCOPE = ManagedSessionScope("dev", "device-1", "table-1")
BINDING = ManagedSessionBinding(
    "1.0", f"managed-{BIRTH}", BIRTH, "managed_session_birth_v1", "table-1", "device-1", "open"
)


def store(path: Path, **kwargs) -> SQLiteFieldRecordStore:
    return SQLiteFieldRecordStore(
        path, now_utc=lambda: NOW, managed_open_request_id_factory=lambda: REQUEST, **kwargs
    )


def fail_factory():
    pytest.fail("replay called an ID factory")


def reserve(s: SQLiteFieldRecordStore):
    return s.reserve_managed_open(SCOPE, artifact_id="artifact-1", rule_profile_id=PROFILE_ID)


def bound(s: SQLiteFieldRecordStore):
    reserve(s)
    return s.record_managed_binding(SCOPE, REQUEST, BINDING)


def snapshot(path: Path):
    with sqlite3.connect(path) as db:
        names = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return {
            name: db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
            for (name,) in names
        }


def test_reservation_is_durable_before_http_and_does_not_create_local_scene(tmp_path):
    path = tmp_path / "field.db"
    s = store(path)
    request = reserve(s)
    assert request.open_request_id == REQUEST and request.phase == "pending"
    assert request.binding is None
    assert s.find_open_session("artifact-1") is None
    before = snapshot(path)
    restarted = SQLiteFieldRecordStore(
        path,
        now_utc=lambda: NOW + timedelta(days=1),
        managed_open_request_id_factory=fail_factory,
        session_id_factory=fail_factory,
    )
    assert reserve(restarted) == request
    assert restarted.get_managed_open_request(SCOPE, REQUEST) == request
    assert snapshot(path) == before


def test_concurrent_reservers_commit_one_request_and_only_call_winning_factory(tmp_path):
    path = tmp_path / "field.db"
    store(path)
    calls = []

    def run(_):
        s = SQLiteFieldRecordStore(
            path, managed_open_request_id_factory=lambda: calls.append(REQUEST) or REQUEST
        )
        return reserve(s)

    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = list(pool.map(run, range(8)))
    assert len(set(requests)) == 1 and calls == [REQUEST]


@pytest.mark.parametrize(
    "scope,artifact,profile",
    [
        (SCOPE, "artifact-2", PROFILE_ID),
        (SCOPE, "artifact-1", None),
        (replace(SCOPE, environment_id="other"), "artifact-1", PROFILE_ID),
        (replace(SCOPE, device_id="other"), "artifact-1", PROFILE_ID),
        (replace(SCOPE, table_id="other"), "artifact-1", PROFILE_ID),
    ],
)
def test_reservation_refuses_scope_artifact_or_profile_drift(tmp_path, scope, artifact, profile):
    s = store(tmp_path / "field.db")
    reserve(s)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.reserve_managed_open(scope, artifact_id=artifact, rule_profile_id=profile)
    assert snapshot(s.path) == before


@pytest.mark.parametrize("ended,pending_close", [(False, False), (True, True)])
def test_legacy_or_pending_predecessor_cannot_be_adopted(tmp_path, ended, pending_close):
    s = store(tmp_path / "field.db", session_id_factory=lambda: "legacy")
    s.open_or_resume_session(artifact_id="artifact-1")
    if ended:
        s.end_session("legacy")
    if pending_close:
        s.enqueue_cloud_close_intent("legacy")
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        reserve(s)
    assert snapshot(s.path) == before


def test_binding_commit_survives_restart_and_exact_retry_has_no_writes(tmp_path):
    s = store(tmp_path / "field.db")
    request = bound(s)
    assert request.phase == "bound" and request.binding == BINDING
    assert s.find_open_session("artifact-1") is None
    before = snapshot(s.path)
    restarted = store(s.path)
    assert restarted.record_managed_binding(SCOPE, REQUEST, BINDING) == request
    assert snapshot(s.path) == before


def test_binding_is_immutable_and_ended_is_monotonic(tmp_path):
    s = store(tmp_path / "field.db")
    bound(s)
    other_birth = "33333333-3333-4333-8333-333333333333"
    with pytest.raises(ValueError):
        s.record_managed_binding(
            SCOPE,
            REQUEST,
            replace(BINDING, birth_id=other_birth, session_id=f"managed-{other_birth}"),
        )
    ended = replace(BINDING, status="ended")
    assert s.record_managed_binding(SCOPE, REQUEST, ended).binding == ended
    assert reserve(s).binding == ended
    with pytest.raises(ValueError):
        s.record_managed_binding(SCOPE, REQUEST, BINDING)
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)


def test_missing_request_and_wrong_binding_scope_never_create_rows(tmp_path):
    s = store(tmp_path / "field.db")
    for operation in (
        lambda: s.get_managed_open_request(SCOPE, REQUEST),
        lambda: s.record_managed_binding(SCOPE, REQUEST, BINDING),
        lambda: s.materialize_managed_session(SCOPE, REQUEST),
    ):
        with pytest.raises(ValueError):
            operation()
    reserve(s)
    with pytest.raises(ValueError):
        s.record_managed_binding(SCOPE, REQUEST, replace(BINDING, table_id="other"))
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)
    assert s.find_open_session("artifact-1") is None


def test_materialization_commits_four_rows_and_retry_returns_current_match(tmp_path):
    s = store(tmp_path / "field.db", session_id_factory=fail_factory)
    bound(s)
    result = s.materialize_managed_session(SCOPE, REQUEST)
    assert result.session_id == BINDING.session_id
    assert result.artifact_id == "artifact-1" and result.match.rule_profile_id == PROFILE_ID
    changed = replace(result.match, revision=1)
    s.save_match(result.session_id, changed, expected_revision=0)
    before = snapshot(s.path)
    restarted = store(s.path, session_id_factory=fail_factory)
    assert restarted.materialize_managed_session(SCOPE, REQUEST).match == changed
    assert restarted.get_managed_open_request(SCOPE, REQUEST).phase == "materialized"
    assert snapshot(s.path) == before
    assert before["field_cloud_close_intents"][0][1] == "open"


@pytest.mark.parametrize(
    "target,operation",
    [
        ("field_sessions", "INSERT"),
        ("field_match_state", "INSERT"),
        ("field_cloud_close_intents", "INSERT"),
        ("field_managed_open_requests", "UPDATE"),
    ],
)
def test_materialization_rolls_back_every_write_on_failure(tmp_path, target, operation):
    s = store(tmp_path / "field.db")
    bound(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        db.execute(
            f"CREATE TRIGGER fail_write BEFORE {operation} ON {target} "
            "BEGIN SELECT RAISE(ABORT, 'injected crash'); END"
        )
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TRIGGER fail_write")
    restarted = store(s.path)
    assert restarted.materialize_managed_session(SCOPE, REQUEST).session_id == BINDING.session_id


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_cloud_close_intents",
        "UPDATE field_cloud_close_intents SET status='pending'",
        "UPDATE field_cloud_close_intents SET status='complete'",
        "UPDATE field_cloud_close_intents SET updated_at_utc='bad'",
        "DELETE FROM field_match_state",
        "UPDATE field_match_state SET revision=3",
        "UPDATE field_match_state SET payload_json='{}'",
        "UPDATE field_sessions SET artifact_id='wrong'",
        "UPDATE field_sessions SET status='ended',ended_at_utc='2026-10-30T00:00:00Z'",
        "UPDATE field_sessions SET created_at_utc='bad'",
    ],
)
def test_materialized_retry_refuses_damage_without_repair(tmp_path, damage):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        store(s.path).materialize_managed_session(SCOPE, REQUEST)
    assert snapshot(s.path) == before


def test_materialized_binding_status_updates_keep_phase_and_reservation_forever(tmp_path):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    ended = s.record_managed_binding(SCOPE, REQUEST, replace(BINDING, status="ended"))
    assert ended.phase == "materialized" and reserve(s) == ended
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)
    with pytest.raises(ValueError):
        s.record_managed_binding(SCOPE, REQUEST, BINDING)


def test_same_name_unbound_local_scene_is_not_adopted(tmp_path):
    s = store(tmp_path / "field.db", session_id_factory=lambda: BINDING.session_id)
    bound(s)
    with pytest.raises(ValueError, match="managed"):
        s.open_or_resume_session(artifact_id="artifact-1", rule_profile_id=PROFILE_ID)
    # Startup now rejects this legacy escape earlier. Retain a synthetic preexisting
    # unbound scene to independently verify materialization still cannot adopt it.
    unbound = store(tmp_path / "unbound.db", session_id_factory=lambda: BINDING.session_id)
    unbound.open_or_resume_session(artifact_id="artifact-1", rule_profile_id=PROFILE_ID)
    with sqlite3.connect(unbound.path) as source, sqlite3.connect(s.path) as target:
        target.execute(
            "INSERT INTO field_sessions VALUES(?,?,?,?,?)",
            source.execute("SELECT * FROM field_sessions").fetchone(),
        )
        target.execute(
            "INSERT INTO field_match_state VALUES(?,?,?)",
            source.execute("SELECT * FROM field_match_state").fetchone(),
        )
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "unknown"),
        ("open_request_id", "bad"),
        ("created_at_utc", "bad"),
        ("updated_at_utc", "bad"),
        ("rule_profile_id", "bad"),
        ("artifact_id", "../bad"),
        ("session_id", "unexpected"),
        ("remote_status", "open"),
    ],
)
def test_damaged_pending_rows_fail_closed_for_reads_and_mutations(tmp_path, field, value):
    s = store(tmp_path / "field.db")
    reserve(s)
    with sqlite3.connect(s.path) as db:
        db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute(f"UPDATE field_managed_open_requests SET {field}=?", (value,))
    for operation in (
        lambda: s.get_managed_open_request(SCOPE, REQUEST),
        lambda: reserve(s),
        lambda: s.record_managed_binding(SCOPE, REQUEST, BINDING),
        lambda: s.materialize_managed_session(SCOPE, REQUEST),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("value", ["bad", "11111111-1111-4111-8111-11111111111A", None, 42])
def test_invalid_factory_uuid_is_not_committed(tmp_path, value):
    s = SQLiteFieldRecordStore(tmp_path / "field.db", managed_open_request_id_factory=lambda: value)
    with pytest.raises(ValueError):
        reserve(s)
    assert snapshot(s.path)["field_managed_open_requests"] == []


def test_schema18_adds_only_empty_managed_metadata_and_preserves_old_logical_rows(tmp_path):
    s = store(tmp_path / "field.db", session_id_factory=lambda: "legacy")
    s.open_or_resume_session(artifact_id="artifact-1")
    s.save_match("legacy", MatchControlSnapshot(revision=1), expected_revision=0)
    s.record_cloud_session_open("legacy")
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE field_owner_media_deletions")
        db.execute("DROP TABLE field_managed_capture_epochs")
        db.execute("DROP TABLE field_managed_resource_specs")
        db.execute("DROP TABLE field_managed_open_requests")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=18")
    before = snapshot(s.path)
    migrated = store(s.path)
    after = snapshot(s.path)
    assert after.pop("field_managed_open_requests") == []
    assert after.pop("field_managed_resource_specs") == []
    assert after.pop("field_managed_capture_epochs") == []
    assert after.pop("field_recording_ends") == []
    assert after.pop("field_recording_evidence") == []
    assert after.pop("field_session_recording_policy") == []
    assert after.pop("field_rule_fact_preimages") == []
    assert after.pop("field_owner_completion_outbox") == []
    assert after.pop("field_local_media_schedule_state") == []
    assert after.pop("field_upload_receipt_outbox") == []
    assert after.pop("field_saved_upload_requests") == []
    assert after.pop("field_owner_media_deletions") == []
    assert before == after
    assert migrated.load_session("legacy").match.revision == 1
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


@pytest.mark.parametrize("operation", ["INSERT", "UPDATE"])
def test_failed_reservation_or_binding_transaction_is_durable_on_reopen(tmp_path, operation):
    s = store(tmp_path / "field.db")
    if operation == "UPDATE":
        reserve(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        db.execute(
            f"CREATE TRIGGER fail_write BEFORE {operation} ON field_managed_open_requests "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(ValueError):
        if operation == "INSERT":
            reserve(s)
        else:
            s.record_managed_binding(SCOPE, REQUEST, BINDING)
    assert snapshot(store(s.path).path) == before


def test_concurrent_materializers_return_one_atomic_session(tmp_path):
    path = tmp_path / "field.db"
    bound(store(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(
            pool.map(
                lambda _: store(path, session_id_factory=fail_factory).materialize_managed_session(
                    SCOPE, REQUEST
                ),
                range(8),
            )
        )
    assert len(set(records)) == 1
    rows = snapshot(path)
    for name in (
        "field_sessions",
        "field_match_state",
        "field_cloud_close_intents",
        "field_managed_open_requests",
    ):
        assert len(rows[name]) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "UPDATE field_sessions SET created_at_utc='bad'",
        "UPDATE field_sessions SET ended_at_utc='2026-10-30T00:00:00Z'",
        "UPDATE field_match_state SET revision=5",
    ],
)
def test_find_open_session_refuses_damaged_persisted_scene(tmp_path, damage):
    s = store(tmp_path / "field.db")
    s.open_or_resume_session(artifact_id="artifact-1")
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    with pytest.raises(ValueError):
        s.find_open_session("artifact-1")


@pytest.mark.parametrize("revision", ["0", False, 0.0], ids=["string", "bool", "float"])
def test_find_open_session_refuses_coerced_payload_revision(tmp_path, revision):
    s = store(tmp_path / "field.db")
    s.open_or_resume_session(artifact_id="artifact-1")
    with sqlite3.connect(s.path) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM field_match_state").fetchone()[0])
        payload["revision"] = revision
        db.execute("UPDATE field_match_state SET payload_json=?", (json.dumps(payload),))
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.find_open_session("artifact-1")
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_sessions",
        "UPDATE field_match_state SET revision=5",
        "UPDATE field_match_state SET payload_json=json_set(payload_json,'$.rule_profile_id',NULL)",
        "UPDATE field_match_state SET payload_json=json_set(payload_json,'$.revision','0')",
    ],
)
def test_materialized_retry_refuses_missing_session_and_match_drift(tmp_path, damage):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("birth_id", "bad"),
        ("birth_contract", "wrong"),
        ("remote_status", "unknown"),
        ("birth_id", None),
    ],
)
def test_damaged_bound_identity_is_never_repaired(tmp_path, field, value):
    s = store(tmp_path / "field.db")
    bound(s)
    with sqlite3.connect(s.path) as db:
        db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute(f"UPDATE field_managed_open_requests SET {field}=?", (value,))
    before = snapshot(s.path)
    for operation in (
        lambda: reserve(s),
        lambda: s.get_managed_open_request(SCOPE, REQUEST),
        lambda: s.record_managed_binding(SCOPE, REQUEST, BINDING),
        lambda: s.materialize_managed_session(SCOPE, REQUEST),
    ):
        with pytest.raises(ValueError):
            operation()
    assert snapshot(s.path) == before


def test_issued_birth_cannot_bind_another_reservation_in_same_environment(tmp_path):
    s = store(tmp_path / "field.db")
    bound(s)
    other_scope = replace(SCOPE, device_id="device-2", table_id="table-2")
    s.reserve_managed_open(other_scope, artifact_id="artifact-2")
    with pytest.raises(ValueError):
        s.record_managed_binding(
            other_scope, REQUEST, replace(BINDING, device_id="device-2", table_id="table-2")
        )
    assert s.get_managed_open_request(other_scope, REQUEST).phase == "pending"


def test_local_end_keeps_current_reservation_and_cannot_rotate_or_rematerialize(tmp_path):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    s.end_session(BINDING.session_id)
    assert reserve(s).open_request_id == REQUEST
    with pytest.raises(ValueError):
        s.reserve_managed_open(SCOPE, artifact_id="artifact-2", rule_profile_id=PROFILE_ID)
    with pytest.raises(ValueError):
        s.materialize_managed_session(SCOPE, REQUEST)


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_match_state",
        "UPDATE field_match_state SET revision=5",
        "UPDATE field_match_state SET payload_json='{}'",
    ],
)
def test_damaged_ended_predecessor_blocks_new_reservation(tmp_path, damage):
    s = store(tmp_path / "field.db", session_id_factory=lambda: "legacy")
    s.open_or_resume_session(artifact_id="artifact-1")
    s.end_session("legacy")
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        reserve(s)
    assert snapshot(s.path) == before


@pytest.mark.parametrize("revision", ["0", False, 0.0], ids=["string", "bool", "float"])
def test_ended_predecessor_with_coerced_payload_revision_blocks_reservation(tmp_path, revision):
    s = store(tmp_path / "field.db", session_id_factory=lambda: "legacy")
    s.open_or_resume_session(artifact_id="artifact-1")
    s.end_session("legacy")
    with sqlite3.connect(s.path) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM field_match_state").fetchone()[0])
        payload["revision"] = revision
        db.execute("UPDATE field_match_state SET payload_json=?", (json.dumps(payload),))
    before = snapshot(s.path)
    assert before["field_cloud_close_intents"] == []
    with pytest.raises(ValueError):
        reserve(s)
    assert snapshot(s.path) == before


def test_schema1_migration_chain_reaches22_without_managed_backfill(tmp_path):
    s = store(tmp_path / "field.db", session_id_factory=lambda: "legacy")
    original = s.open_or_resume_session(artifact_id="artifact-1")
    with sqlite3.connect(s.path) as db:
        for (name,) in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            if name not in {"field_sessions", "field_match_state", "field_shots"}:
                db.execute(f'DROP TABLE "{name}"')
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=1")
    before = snapshot(s.path)
    migrated = store(s.path)
    assert migrated.load_session("legacy") == original
    after = snapshot(s.path)
    assert {name: after[name] for name in before} == before
    assert after["field_managed_open_requests"] == []
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


@pytest.mark.parametrize("phase", ["pending", "bound", "materialized"])
def test_materialized_lookup_is_readonly_and_requires_materialized_phase(tmp_path, phase):
    s = store(tmp_path / "field.db")
    request = reserve(s)
    if phase != "pending":
        request = s.record_managed_binding(SCOPE, REQUEST, BINDING)
    if phase == "materialized":
        s.materialize_managed_session(SCOPE, REQUEST)
        s.end_session(BINDING.session_id)
        request = s.get_managed_open_request(SCOPE, REQUEST)
    before = snapshot(s.path)
    if phase == "materialized":
        assert s.get_materialized_managed_open_request(SCOPE, BINDING.session_id) == request
    else:
        with pytest.raises(ValueError):
            s.get_materialized_managed_open_request(SCOPE, BINDING.session_id)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "changes", [dict(environment_id="other"), dict(device_id="other"), dict(table_id="other")]
)
def test_materialized_lookup_pins_complete_scope_without_repair(tmp_path, changes):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.get_materialized_managed_open_request(replace(SCOPE, **changes), BINDING.session_id)
    assert snapshot(s.path) == before


def test_materialized_lookup_rejects_corrupt_dto_without_mutation(tmp_path):
    s = store(tmp_path / "field.db")
    bound(s)
    s.materialize_managed_session(SCOPE, REQUEST)
    with sqlite3.connect(s.path) as db:
        db.execute("UPDATE field_managed_open_requests SET birth_contract='corrupt'")
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.get_materialized_managed_open_request(SCOPE, BINDING.session_id)
    assert snapshot(s.path) == before
