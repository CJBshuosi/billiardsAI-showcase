from __future__ import annotations

import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from billiards_ai.edge.route_assistance.eight_ball_rules import PROFILE_ID
from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionCloseState,
    ManagedSessionScope,
)
from billiards_ai.edge.route_assistance.persistence import (
    _MANAGED_OPEN_SCHEMA,
    SQLiteFieldRecordStore,
)

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SCOPE = ManagedSessionScope("dev", "device", "table")


def store(path, **kwargs):
    return SQLiteFieldRecordStore(path, **({"now_utc": lambda: NOW} | kwargs))


def snapshot(path):
    with sqlite3.connect(path) as db:
        return {
            name: db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
            for (name,) in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }


def scene(s, *, scope=SCOPE, artifact="artifact", ended=True):
    request = s.reserve_managed_open(scope, artifact_id=artifact, rule_profile_id=PROFILE_ID)
    birth = str(uuid.uuid4())
    binding = ManagedSessionBinding(
        "1.0",
        f"managed-{birth}",
        birth,
        "managed_session_birth_v1",
        scope.table_id,
        scope.device_id,
        "open",
    )
    s.record_managed_binding(scope, request.open_request_id, binding)
    s.materialize_managed_session(scope, request.open_request_id)
    if ended:
        s.end_session(binding.session_id)
    seal = ManagedSessionCloseState(
        binding.schema_version,
        binding.session_id,
        binding.birth_id,
        binding.birth_contract,
        binding.table_id,
        binding.device_id,
        "sealed",
        2,
        2,
    )
    return request.open_request_id, binding, seal


def retire(s, request_id, seal, scope=SCOPE):
    return s.retire_managed_open(scope, request_id, sealed_close=seal)


def fail_clock():
    pytest.fail("exact replay called clock or factory")


def test_retirement_releases_only_active_occupancy_and_preserves_all_business_rows(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    before = snapshot(s.path)
    retired = retire(s, request_id, seal)
    assert retired.lifecycle == "retired" and retired.binding == replace(binding, status="ended")
    assert retired.sealed_close == seal and retired.retired_at_utc == "2026-11-01T00:00:00Z"
    assert s.get_active_managed_open_request(SCOPE) is None
    assert s.get_latest_retired_managed_open_request(SCOPE) == retired
    assert s.get_managed_open_request_for_session(SCOPE, binding.session_id) == retired
    assert s.get_materialized_managed_open_request(SCOPE, binding.session_id) == retired
    after = snapshot(s.path)
    assert {key: value for key, value in after.items() if key != "field_managed_open_requests"} == {
        key: value for key, value in before.items() if key != "field_managed_open_requests"
    }
    replay = store(s.path, now_utc=fail_clock, managed_open_request_id_factory=fail_clock)
    assert retire(replay, request_id, seal) == retired
    assert replay.record_managed_binding(SCOPE, request_id, retired.binding) == retired
    assert snapshot(s.path) == after
    for operation in (
        lambda: replay.materialize_managed_session(SCOPE, request_id),
        lambda: replay.record_managed_binding(SCOPE, request_id, binding),
        lambda: retire(
            replay, request_id, replace(seal, closed_target=3, contiguous_ack_sequence=3)
        ),
    ):
        with pytest.raises(ValueError):
            operation()
    new = s.reserve_managed_open(SCOPE, artifact_id="artifact", rule_profile_id=PROFILE_ID)
    assert new.open_request_id != request_id and new.lifecycle == "active"
    assert s.get_active_managed_open_request(SCOPE) == new
    assert s.get_managed_open_request(SCOPE, request_id) == retired
    assert not hasattr(new, "reservation_order")
    assert "reservation_order" not in asdict(new)


@pytest.mark.parametrize("artifact", ["artifact", "changed-artifact"])
def test_unretired_ended_scope_cannot_rotate_request_or_artifact(tmp_path, artifact):
    s = store(tmp_path / "field.db")
    request_id, _, _ = scene(s)
    before = snapshot(s.path)
    if artifact == "artifact":
        assert (
            s.reserve_managed_open(
                SCOPE, artifact_id=artifact, rule_profile_id=PROFILE_ID
            ).open_request_id
            == request_id
        )
    else:
        with pytest.raises(ValueError):
            s.reserve_managed_open(SCOPE, artifact_id=artifact, rule_profile_id=PROFILE_ID)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_sessions",
        "DELETE FROM field_cloud_close_intents",
        "UPDATE field_cloud_close_intents SET status='open'",
        "UPDATE field_cloud_close_intents SET updated_at_utc='bad'",
        "UPDATE field_sessions SET artifact_id='wrong'",
        "UPDATE field_sessions SET ended_at_utc='bad'",
        "UPDATE field_sessions SET ended_at_utc='2026-10-31T00:00:00Z'",
        "UPDATE field_match_state SET revision=3",
        "UPDATE field_match_state SET payload_json='{}'",
        "UPDATE field_match_state SET payload_json=json_set(payload_json,'$.rule_profile_id',NULL)",
    ],
)
def test_retirement_refuses_missing_or_damaged_local_evidence_without_writes(tmp_path, damage):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        retire(s, request_id, seal)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "change",
    [
        {"coverage": "closing", "contiguous_ack_sequence": 1},
        {"device_id": "wrong"},
        {"table_id": "wrong"},
        {
            "birth_id": "11111111-1111-4111-8111-111111111111",
            "session_id": "managed-11111111-1111-4111-8111-111111111111",
        },
    ],
)
def test_retirement_requires_identity_matching_typed_sealed_close(tmp_path, change):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        retire(s, request_id, replace(seal, **change))
    for value in (True, asdict(seal), None):
        with pytest.raises(ValueError):
            retire(s, request_id, value)
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "changes", [{"environment_id": "wrong"}, {"device_id": "wrong"}, {"table_id": "wrong"}]
)
def test_all_retirement_lookups_require_full_scope(tmp_path, changes):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    wrong = replace(SCOPE, **changes)
    assert s.get_active_managed_open_request(wrong) is None
    assert s.get_latest_retired_managed_open_request(wrong) is None
    with pytest.raises(ValueError):
        s.get_managed_open_request_for_session(wrong, binding.session_id)
    with pytest.raises(ValueError):
        retire(s, request_id, seal, wrong)
    retire(s, request_id, seal)
    assert s.get_latest_retired_managed_open_request(wrong) is None


def test_retirement_needs_materialized_local_ended_scene_and_pending_or_complete_marker(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s, ended=False)
    with pytest.raises(ValueError):
        retire(s, request_id, seal)
    s.end_session(seal.session_id)
    s.complete_cloud_close_intent(seal.session_id)
    assert retire(s, request_id, seal).lifecycle == "retired"


def test_same_second_latest_retirement_uses_scoped_order_and_concurrent_reservation(tmp_path):
    s = store(tmp_path / "field.db")
    retired = []
    for _ in range(3):
        request_id, _, seal = scene(s)
        retired.append(retire(s, request_id, seal))
    other = replace(SCOPE, table_id="other")
    for _ in range(4):
        request_id, _, seal = scene(s, scope=other, artifact="other-artifact")
        retire(s, request_id, seal, other)
    assert s.get_latest_retired_managed_open_request(SCOPE) == retired[-1]
    calls = []

    def reserve(_):
        return store(
            s.path, managed_open_request_id_factory=lambda: calls.append(1) or str(uuid.uuid4())
        ).reserve_managed_open(SCOPE, artifact_id="artifact", rule_profile_id=PROFILE_ID)

    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = list(pool.map(reserve, range(8)))
    assert len(set(requests)) == 1 and len(calls) == 1
    with sqlite3.connect(s.path) as db:
        assert db.execute(
            "SELECT reservation_order FROM field_managed_open_requests WHERE table_id=? "
            "ORDER BY reservation_order",
            (SCOPE.table_id,),
        ).fetchall() == [(1,), (2,), (3,), (4,)]


def test_retirement_and_order_overflow_are_atomic(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        db.execute(
            "CREATE TRIGGER fail_retire BEFORE UPDATE ON field_managed_open_requests "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(ValueError):
        retire(s, request_id, seal)
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TRIGGER fail_retire")
    retire(s, request_id, seal)
    with sqlite3.connect(s.path) as db:
        db.execute("UPDATE field_managed_open_requests SET reservation_order=9223372036854775807")
    before = snapshot(s.path)
    s = store(s.path, now_utc=fail_clock, managed_open_request_id_factory=fail_clock)
    with pytest.raises(ValueError):
        s.reserve_managed_open(SCOPE, artifact_id="artifact")
    assert snapshot(s.path) == before


def downgrade19(s):
    # Only synthetic DBs; retain the original 13 columns and all unrelated tables.
    with sqlite3.connect(s.path) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(field_managed_open_requests)")][
            :13
        ]
        rows = db.execute(
            "SELECT " + ",".join(columns) + " FROM field_managed_open_requests"
        ).fetchall()
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE field_owner_media_deletions")
        db.execute("DROP TABLE field_managed_capture_epochs")
        db.execute("DROP TABLE field_managed_resource_specs")
        db.execute("DROP TABLE field_managed_open_requests")
        db.executescript(_MANAGED_OPEN_SCHEMA)
        db.executemany(
            "INSERT INTO field_managed_open_requests VALUES ("
            + ",".join("?" for _ in columns)
            + ")",
            rows,
        )
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=19")


@pytest.mark.parametrize("marker_complete", [False, True])
def test_migration19_preserves_exact_old_columns_and_business_bytes_as_active(
    tmp_path, marker_complete
):
    s = store(tmp_path / "field.db")
    request_id, binding, _ = scene(s)
    if marker_complete:
        s.complete_cloud_close_intent(binding.session_id)
    with sqlite3.connect(s.path) as db:
        # Noncanonical historical JSON and receipt bytes must not be reserialized.
        raw = db.execute("SELECT payload_json FROM field_match_state").fetchone()[0]
        db.execute(
            "UPDATE field_match_state SET payload_json=?",
            (json.dumps(json.loads(raw), ensure_ascii=False, indent=3),),
        )
        db.execute(
            "INSERT INTO field_applied_commands VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "old-command",
                binding.session_id,
                1,
                "old-user",
                "old-type",
                "old-digest",
                "applied",
                "old-result",
                0,
                "2026-11-01T00:00:00.000000Z",
            ),
        )
        db.execute(
            "INSERT INTO field_admin_audit VALUES (?,?,?,?,?,?,?)",
            (
                1,
                binding.session_id,
                "old-type",
                "old-reason",
                '  {"receipt": "保留"}  ',
                0,
                "2026-11-01T00:00:00.000000Z",
            ),
        )
        db.execute(
            "UPDATE field_managed_open_requests SET created_at_utc=?,updated_at_utc=?",
            (
                "2026-11-01T00:00:00.000000Z",
                "2026-11-01T00:00:00.000000Z",
            ),
        )
    downgrade19(s)
    before = snapshot(s.path)
    s = store(s.path)
    after = snapshot(s.path)
    rows = after.pop("field_managed_open_requests")
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
    assert [row[:13] for row in rows] == before.pop("field_managed_open_requests")
    assert before == after
    assert rows[0][13:] == ("active", None, None, 1)
    assert s.get_active_managed_open_request(SCOPE).open_request_id == request_id
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


@pytest.mark.parametrize(
    "damage",
    [
        "UPDATE field_managed_open_requests SET created_at_utc='bad'",
        "UPDATE field_managed_open_requests SET updated_at_utc='2026-10-31T00:00:00Z'",
        "UPDATE field_managed_open_requests SET birth_contract='bad'",
    ],
)
def test_invalid19_migration_rolls_back_original_table_and_rows(tmp_path, damage):
    s = store(tmp_path / "field.db")
    scene(s)
    downgrade19(s)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
        sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE name='field_managed_open_requests'"
        ).fetchone()
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        store(s.path)
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 19
        assert (
            db.execute(
                "SELECT sql FROM sqlite_master WHERE name='field_managed_open_requests'"
            ).fetchone()
            == sql
        )


def test_retired_json_is_canonical_and_damaged_metadata_is_readonly_rejected(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retire(s, request_id, seal)
    with sqlite3.connect(s.path) as db:
        raw = db.execute("SELECT sealed_close_json FROM field_managed_open_requests").fetchone()[0]
        assert raw == json.dumps(asdict(seal), sort_keys=True, separators=(",", ":"))
        db.execute("UPDATE field_managed_open_requests SET sealed_close_json=?", (raw + " ",))
    before = snapshot(s.path)
    for operation in (
        lambda: s.get_managed_open_request(SCOPE, request_id),
        lambda: s.get_latest_retired_managed_open_request(SCOPE),
        lambda: s.get_managed_open_request_for_session(SCOPE, binding.session_id),
        lambda: s.reserve_managed_open(SCOPE, artifact_id="changed"),
    ):
        with pytest.raises(ValueError):
            operation()
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"updated_at_utc": "2026-11-01T00:00:01Z"},
        {"retired_at_utc": "2026-10-31T23:59:59Z"},
        {"lifecycle": "active"},
        {"phase": "bound"},
        {"sealed_close": None},
    ],
)
def test_retired_dto_requires_immutable_retirement_timestamp_and_seal(tmp_path, changes):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s)
    retired = retire(s, request_id, seal)
    with pytest.raises(ValueError):
        replace(retired, **changes)


def test_retirement_refuses_clock_before_original_updated_time(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, _, seal = scene(s)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        retire(store(s.path, now_utc=lambda: NOW - timedelta(seconds=1)), request_id, seal)
    assert snapshot(s.path) == before


@pytest.mark.parametrize("step", ["copy", "index", "commit"])
def test_migration19_failure_at_copy_index_or_commit_preserves_original_db(tmp_path, step):
    s = store(tmp_path / "field.db")
    scene(s)
    downgrade19(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        original_schema = db.execute(
            "SELECT type,name,sql FROM sqlite_master ORDER BY name"
        ).fetchall()

    class FailingConnection(sqlite3.Connection):
        def executemany(self, sql, parameters):
            if step == "copy" and "INSERT INTO field_managed_open_requests20" in sql:
                super().executemany(sql, parameters)
                raise sqlite3.IntegrityError("injected copy failure")
            return super().executemany(sql, parameters)

        def execute(self, sql, parameters=()):
            if step == "index" and "CREATE UNIQUE INDEX field_managed_active_artifact" in sql:
                raise sqlite3.IntegrityError("injected index failure")
            return super().execute(sql, parameters)

        def commit(self):
            if step == "commit":
                raise sqlite3.OperationalError("injected commit failure")
            return super().commit()

    class FailingStore(SQLiteFieldRecordStore):
        def _connect(self):
            db = sqlite3.connect(self.path, factory=FailingConnection)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            return db

    with pytest.raises(ValueError):
        FailingStore(s.path)
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 19
        assert (
            db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
            == original_schema
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("reservation_order", 0),
        ("reservation_order", -1),
        ("reservation_order", 1.5),
        ("reservation_order", "bad"),
        ("lifecycle", "unknown"),
        ("retired_at_utc", "2026-11-01T00:00:00Z"),
        ("sealed_close_json", "{}"),
    ],
)
def test_schema20_rejects_invalid_active_metadata(tmp_path, field, value):
    s = store(tmp_path / "field.db")
    s.reserve_managed_open(SCOPE, artifact_id="artifact")
    with sqlite3.connect(s.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute(f"UPDATE field_managed_open_requests SET {field}=?", (value,))


@pytest.mark.parametrize("artifact", ["artifact", "changed-artifact"])
def test_new_materialization_keeps_retired_pending_work_and_original_history(tmp_path, artifact):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retired = retire(s, request_id, seal)
    before = snapshot(s.path)
    request_id2, _, _ = scene(s, artifact=artifact, ended=False)
    assert request_id2 != request_id
    assert s.get_managed_open_request_for_session(SCOPE, binding.session_id) == retired
    after = snapshot(s.path)
    for name, rows in before.items():
        assert all(row in after[name] for row in rows)
    assert binding.session_id in s.list_pending_cloud_close_session_ids()


def test_retired_history_getters_do_not_require_local_scene_or_repair_it(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retired = retire(s, request_id, seal)
    with sqlite3.connect(s.path) as db:
        db.execute("DELETE FROM field_sessions")
    before = snapshot(s.path)
    assert s.get_managed_open_request_for_session(SCOPE, binding.session_id) == retired
    assert s.get_materialized_managed_open_request(SCOPE, binding.session_id) == retired
    with pytest.raises(ValueError):
        s.reserve_managed_open(SCOPE, artifact_id="changed")
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM field_sessions",
        "UPDATE field_sessions SET status='open',ended_at_utc=NULL",
        "UPDATE field_sessions SET artifact_id='wrong'",
        "UPDATE field_sessions SET ended_at_utc='bad'",
        "DELETE FROM field_match_state",
        "UPDATE field_match_state SET revision=3",
        "UPDATE field_match_state SET payload_json=json_set(payload_json,'$.rule_profile_id',NULL)",
        "DELETE FROM field_cloud_close_intents",
        "UPDATE field_cloud_close_intents SET status='open'",
        "UPDATE field_cloud_close_intents SET updated_at_utc='bad'",
    ],
)
def test_retirement_replay_requires_retained_local_evidence_without_writes(tmp_path, damage):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retired = retire(s, request_id, seal)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    replay = store(s.path, now_utc=fail_clock, managed_open_request_id_factory=fail_clock)
    with pytest.raises(ValueError):
        retire(replay, request_id, seal)
    assert replay.get_managed_open_request_for_session(SCOPE, binding.session_id) == retired
    assert replay.get_materialized_managed_open_request(SCOPE, binding.session_id) == retired
    assert snapshot(s.path) == before


def test_retirement_replay_accepts_marker_completion_without_clock_or_writes(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retired = retire(s, request_id, seal)
    later = store(s.path, now_utc=lambda: NOW + timedelta(seconds=1))
    later.complete_cloud_close_intent(binding.session_id)
    before = snapshot(s.path)
    replay = store(s.path, now_utc=fail_clock, managed_open_request_id_factory=fail_clock)
    assert retire(replay, request_id, seal) == retired
    assert snapshot(s.path) == before


def test_concurrent_schema19_initializers_recheck_version_under_transaction(tmp_path, monkeypatch):
    s = store(tmp_path / "field.db")
    scene(s)
    downgrade19(s)
    barrier = Barrier(4)
    migrate = SQLiteFieldRecordStore._migrate_managed_requests20

    def synchronized_migration(self, connection):
        barrier.wait(timeout=5)
        return migrate(self, connection)

    monkeypatch.setattr(
        SQLiteFieldRecordStore, "_migrate_managed_requests20", synchronized_migration
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        stores = list(pool.map(lambda _: store(s.path), range(4)))
    assert all(instance.get_active_managed_open_request(SCOPE) is not None for instance in stores)


@pytest.mark.parametrize("peer_version", [30, 31])
def test_waiting_initializer_never_commits_an_older_version_after_peer_upgrade(
    tmp_path, monkeypatch, peer_version
):
    s = store(tmp_path / "field.db")
    scene(s)
    downgrade19(s)
    migrate = SQLiteFieldRecordStore._migrate_managed_requests20
    statements = []
    peer_finished = False

    def delayed(self, connection):
        nonlocal peer_finished
        if not peer_finished:
            peer_finished = True
            store(self.path)
            if peer_version == 31:
                with sqlite3.connect(self.path) as peer:
                    peer.execute("PRAGMA user_version=31")
            connection.set_trace_callback(statements.append)
        return migrate(self, connection)

    monkeypatch.setattr(SQLiteFieldRecordStore, "_migrate_managed_requests20", delayed)
    if peer_version == 31:
        with pytest.raises(ValueError):
            store(s.path)
    else:
        store(s.path)
    assert not any("user_version=" in sql.replace(" ", "") for sql in statements)
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == peer_version


def test_current_schema_request_lookup_does_not_default_missing_retirement_columns(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, _, _ = scene(s)
    downgrade19(s)
    with sqlite3.connect(s.path) as db:
        db.execute("PRAGMA user_version=28")
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        store(s.path).get_managed_open_request(SCOPE, request_id)
    assert snapshot(s.path) == before


def test_retired_request_and_birth_identity_remain_permanently_unique(tmp_path):
    s = store(tmp_path / "field.db")
    request_id, binding, seal = scene(s)
    retired = retire(s, request_id, seal)
    before = snapshot(s.path)
    duplicate = store(s.path, managed_open_request_id_factory=lambda: request_id)
    with pytest.raises(ValueError):
        duplicate.reserve_managed_open(SCOPE, artifact_id="next")
    assert snapshot(s.path) == before
    new = s.reserve_managed_open(SCOPE, artifact_id="next")
    with pytest.raises(ValueError):
        s.record_managed_binding(SCOPE, new.open_request_id, binding)
    assert s.get_managed_open_request(SCOPE, request_id) == retired
    assert s.get_managed_open_request(SCOPE, new.open_request_id).phase == "pending"
