"""Metadata-only resource intent, migration and consumed capture-epoch contracts."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
from tests.unit.edge.route_assistance.test_managed_session_persistence import (
    BINDING,
    REQUEST,
    SCOPE,
    fail_factory,
    snapshot,
    store,
)
from tests.unit.edge.route_assistance.test_runtime_resources import payload

TABLES = ("field_managed_capture_epochs", "field_managed_resource_specs")


def spec():
    return ManagedRuntimeResourceSpec.from_payload(payload())


def reserve(s, resource=None, scope=SCOPE):
    resource = spec() if resource is None else resource
    return s.reserve_managed_open(
        scope,
        artifact_id=resource.artifact_id,
        rule_profile_id=resource.rule_profile_id,
        resource_spec=resource,
    )


def materialize(s):
    reserve(s)
    s.record_managed_binding(SCOPE, REQUEST, BINDING)
    s.materialize_managed_session(SCOPE, REQUEST)


def downgrade20(s):
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE field_owner_media_deletions")
        for table in TABLES:
            db.execute(f"DROP TABLE IF EXISTS {table}")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=20")


def test_resource_intent_is_atomic_durable_and_exact_replay_has_no_clock_or_uuid(tmp_path):
    s = store(tmp_path / "state.db")
    request = reserve(s)
    record = s.get_managed_resource_spec(SCOPE, REQUEST)
    assert record.spec == spec() and record.spec_sha256 == spec().spec_sha256
    assert record.scope == SCOPE and record.open_request_id == REQUEST
    assert record.created_at_utc == request.created_at_utc
    before = snapshot(s.path)
    restarted = SQLiteFieldRecordStore(
        s.path,
        now_utc=fail_factory,
        managed_open_request_id_factory=fail_factory,
        session_id_factory=fail_factory,
    )
    assert reserve(restarted) == request
    assert restarted.get_managed_resource_spec(SCOPE, REQUEST) == record
    assert snapshot(s.path) == before
    assert s.find_open_session(spec().artifact_id) is None


@pytest.mark.parametrize("table", ["field_managed_open_requests", "field_managed_resource_specs"])
def test_reserve_and_spec_rollback_together_on_insert_failure(tmp_path, table):
    s = store(tmp_path / "state.db")
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        db.execute(
            f"CREATE TRIGGER fault BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected crash'); END"
        )
    with pytest.raises(ValueError):
        reserve(s)
    assert snapshot(store(s.path).path) == before
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TRIGGER fault")
    reserve(s)
    assert len(snapshot(s.path)["field_managed_resource_specs"]) == 1


def test_legacy_intent_is_not_promoted_and_optional_caller_remains_compatible(tmp_path):
    s = store(tmp_path / "state.db")
    resource = spec()
    old = s.reserve_managed_open(
        SCOPE, artifact_id=resource.artifact_id, rule_profile_id=resource.rule_profile_id
    )
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        reserve(s)
    with pytest.raises(ValueError):
        s.get_managed_resource_spec(SCOPE, REQUEST)
    assert (
        s.reserve_managed_open(
            SCOPE, artifact_id=resource.artifact_id, rule_profile_id=resource.rule_profile_id
        )
        == old
    )
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda r: replace(r, batch_size=1),
        lambda r: replace(r, device="cuda"),
        lambda r: replace(r, rule_profile_id=None),
    ],
)
def test_replay_requires_original_spec_and_never_rewrites_it(tmp_path, change):
    s = store(tmp_path / "state.db")
    reserve(s)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        reserve(s, change(spec()))
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "scope",
    [
        replace(SCOPE, environment_id="other"),
        replace(SCOPE, device_id="other"),
        replace(SCOPE, table_id="other"),
    ],
)
def test_scope_drift_never_returns_original_resource(tmp_path, scope):
    s = store(tmp_path / "state.db")
    materialize(s)
    before = snapshot(s.path)
    for operation in (
        lambda: reserve(s, scope=scope),
        lambda: s.get_managed_resource_spec(scope, REQUEST),
        lambda: s.get_managed_resource_spec_for_session(scope, BINDING.session_id),
        lambda: s.register_managed_capture_epoch(scope, REQUEST, "epoch-1"),
        lambda: s.get_managed_capture_epoch(scope, BINDING.session_id, "epoch-1"),
    ):
        with pytest.raises(ValueError):
            operation()
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "UPDATE field_managed_resource_specs SET spec_json=spec_json || ' '",
        "UPDATE field_managed_resource_specs SET spec_sha256='" + "b" * 64 + "'",
        "UPDATE field_managed_resource_specs SET spec_json='{}'",
        "UPDATE field_managed_resource_specs SET created_at_utc='bad'",
        "UPDATE field_managed_open_requests SET artifact_id='foreign'",
        "UPDATE field_managed_open_requests SET rule_profile_id=NULL",
    ],
)
def test_resource_corruption_refuses_reads_replay_and_epoch_creation(tmp_path, damage):
    s = store(tmp_path / "state.db")
    materialize(s)
    with sqlite3.connect(s.path) as db:
        db.execute(damage)
    before = snapshot(s.path)
    for operation in (
        lambda: reserve(s),
        lambda: s.get_managed_resource_spec(SCOPE, REQUEST),
        lambda: s.get_managed_resource_spec_for_session(SCOPE, BINDING.session_id),
        lambda: s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1"),
    ):
        with pytest.raises(ValueError):
            operation()
    assert snapshot(s.path) == before


def test_capture_epoch_is_consumed_after_commit_and_history_survives_end(tmp_path):
    s = store(tmp_path / "state.db")
    materialize(s)
    epoch = s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    assert epoch.scope == SCOPE and epoch.session_id == BINDING.session_id
    assert epoch.birth_id == BINDING.birth_id and epoch.open_request_id == REQUEST
    assert epoch.spec_sha256 == spec().spec_sha256
    before = snapshot(s.path)
    restarted = SQLiteFieldRecordStore(s.path, now_utc=fail_factory)
    with pytest.raises(ValueError):
        restarted.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    assert snapshot(s.path) == before
    assert restarted.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1") == epoch
    s.end_session(BINDING.session_id)
    s.enqueue_cloud_close_intent(BINDING.session_id)
    s.record_managed_binding(SCOPE, REQUEST, replace(BINDING, status="ended"))
    assert s.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1") == epoch
    assert s.get_managed_resource_spec_for_session(SCOPE, BINDING.session_id).spec == spec()
    with pytest.raises(ValueError):
        s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-2")


@pytest.mark.parametrize("phase", ["pending", "bound", "missing-spec", "ended"])
def test_epoch_registration_requires_materialized_open_scene_and_original_spec(tmp_path, phase):
    s = store(tmp_path / "state.db")
    reserve(s)
    if phase != "pending":
        s.record_managed_binding(SCOPE, REQUEST, BINDING)
    if phase in {"missing-spec", "ended"}:
        s.materialize_managed_session(SCOPE, REQUEST)
    if phase == "missing-spec":
        with sqlite3.connect(s.path) as db:
            db.execute("DELETE FROM field_managed_resource_specs")
    if phase == "ended":
        s.end_session(BINDING.session_id)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    assert snapshot(s.path) == before


def test_epoch_insert_failure_does_not_consume_until_commit(tmp_path):
    s = store(tmp_path / "state.db")
    materialize(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        db.execute(
            "CREATE TRIGGER fault BEFORE INSERT ON field_managed_capture_epochs "
            "BEGIN SELECT RAISE(ABORT, 'injected crash'); END"
        )
    with pytest.raises(ValueError):
        s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        db.execute("DROP TRIGGER fault")
    epoch = store(s.path).register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    assert epoch.source_epoch_id == "epoch-1"


def test_concurrent_epoch_registration_has_exactly_one_winner(tmp_path):
    s = store(tmp_path / "state.db")
    materialize(s)

    def run(_):
        try:
            return store(s.path).register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))
    assert sum(result is not None for result in results) == 1
    assert len(snapshot(s.path)["field_managed_capture_epochs"]) == 1


@pytest.mark.parametrize("step", ["second-table", "index", "commit"])
def test_schema20_migration_failure_rolls_back_all_tables_and_version(tmp_path, step):
    s = store(tmp_path / "state.db")
    reserve(s)
    downgrade20(s)
    before = snapshot(s.path)
    with sqlite3.connect(s.path) as db:
        schema = db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()

    class FaultConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if (
                step == "second-table"
                and "CREATE TABLE field_managed_capture_epochs" in sql
                or step == "index"
                and "CREATE INDEX field_managed_capture_epochs_session" in sql
            ):
                raise sqlite3.OperationalError("injected migration failure")
            return super().execute(sql, parameters)

        def commit(self):
            if step == "commit":
                raise sqlite3.OperationalError("injected commit failure")
            return super().commit()

    class FaultStore(SQLiteFieldRecordStore):
        def _connect(self):
            db = sqlite3.connect(self.path, factory=FaultConnection)
            db.row_factory = sqlite3.Row
            return db

    with pytest.raises(ValueError):
        FaultStore(s.path)
    assert snapshot(s.path) == before
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 20
        assert (
            db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
        )
    store(s.path)
    after = snapshot(s.path)
    assert after.pop("field_recording_ends") == []
    assert after.pop("field_recording_evidence") == []
    assert after.pop("field_session_recording_policy") == []
    assert after.pop("field_rule_fact_preimages") == []
    assert after.pop("field_owner_completion_outbox") == []
    assert after.pop("field_local_media_schedule_state") == []
    assert after.pop("field_upload_receipt_outbox") == []
    assert after.pop("field_saved_upload_requests") == []
    assert after.pop("field_owner_media_deletions") == []
    for table in TABLES:
        assert after.pop(table) == []
    assert after == before


def test_concurrent_schema20_initializers_recheck_after_lock(tmp_path, monkeypatch):
    s = store(tmp_path / "state.db")
    reserve(s)
    downgrade20(s)
    barrier = Barrier(4)
    migrate = SQLiteFieldRecordStore._migrate_managed_resources21

    def synchronized(self, connection):
        barrier.wait(timeout=5)
        return migrate(self, connection)

    monkeypatch.setattr(SQLiteFieldRecordStore, "_migrate_managed_resources21", synchronized)
    with ThreadPoolExecutor(max_workers=4) as pool:
        stores = list(pool.map(lambda _: store(s.path), range(4)))
    assert all(item.get_managed_open_request(SCOPE, REQUEST) for item in stores)
    with sqlite3.connect(s.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30


def test_retired_epoch_remains_readable_and_cannot_be_reused_by_next_scene(tmp_path):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionCloseState

    s = store(tmp_path / "state.db")
    materialize(s)
    original = s.get_managed_resource_spec_for_session(SCOPE, BINDING.session_id)
    epoch = s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    s.end_session(BINDING.session_id)
    s.enqueue_cloud_close_intent(BINDING.session_id)
    seal = ManagedSessionCloseState(
        "1.0",
        BINDING.session_id,
        BINDING.birth_id,
        BINDING.birth_contract,
        SCOPE.table_id,
        SCOPE.device_id,
        "sealed",
        0,
        0,
    )
    s.retire_managed_open(SCOPE, REQUEST, sealed_close=seal)
    with pytest.raises(ValueError):
        s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-2")
    next_request = "33333333-3333-4333-8333-333333333333"
    next_birth = "44444444-4444-4444-8444-444444444444"
    next_scope = replace(SCOPE, table_id="table-2")
    s = SQLiteFieldRecordStore(s.path, managed_open_request_id_factory=lambda: next_request)
    reserve(s, scope=next_scope)
    next_binding = replace(
        BINDING,
        session_id=f"managed-{next_birth}",
        birth_id=next_birth,
        table_id=next_scope.table_id,
    )
    s.record_managed_binding(next_scope, next_request, next_binding)
    s.materialize_managed_session(next_scope, next_request)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.register_managed_capture_epoch(next_scope, next_request, "epoch-1")
    assert snapshot(s.path) == before
    assert s.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1") == epoch
    assert s.get_managed_resource_spec_for_session(SCOPE, BINDING.session_id) == original
    with pytest.raises(ValueError):
        s.get_managed_capture_epoch(next_scope, next_binding.session_id, "epoch-1")
    assert s.register_managed_capture_epoch(next_scope, next_request, "epoch-2").source_epoch_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("table_id", "foreign"),
        ("open_request_id", "33333333-3333-4333-8333-333333333333"),
        ("session_id", "managed-44444444-4444-4444-8444-444444444444"),
        ("birth_id", "44444444-4444-4444-8444-444444444444"),
        ("spec_sha256", "b" * 64),
        ("created_at_utc", "bad"),
    ],
)
def test_epoch_read_refuses_corrupt_retained_bindings_without_repair(tmp_path, field, value):
    s = store(tmp_path / "state.db")
    materialize(s)
    s.register_managed_capture_epoch(SCOPE, REQUEST, "epoch-1")
    with sqlite3.connect(s.path) as db:
        db.execute(f"UPDATE field_managed_capture_epochs SET {field}=?", (value,))
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.get_managed_capture_epoch(SCOPE, BINDING.session_id, "epoch-1")
    assert snapshot(s.path) == before


@pytest.mark.parametrize(
    "artifact,profile,resource",
    [
        ("other", spec().rule_profile_id, spec()),
        (spec().artifact_id, None, spec()),
        (spec().artifact_id, spec().rule_profile_id, payload()),
    ],
)
def test_invalid_resource_reservation_refuses_before_clock_and_uuid(
    tmp_path, artifact, profile, resource
):
    s = SQLiteFieldRecordStore(
        tmp_path / "state.db", now_utc=fail_factory, managed_open_request_id_factory=fail_factory
    )
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.reserve_managed_open(
            SCOPE, artifact_id=artifact, rule_profile_id=profile, resource_spec=resource
        )
    assert snapshot(s.path) == before


def test_session_resource_lookup_requires_real_materialization(tmp_path):
    s = store(tmp_path / "state.db")
    reserve(s)
    s.record_managed_binding(SCOPE, REQUEST, BINDING)
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.get_managed_resource_spec_for_session(SCOPE, BINDING.session_id)
    assert snapshot(s.path) == before


def test_schema20_migration_preserves_noncanonical_json_and_hash_bytes(tmp_path):
    s = store(tmp_path / "state.db")
    materialize(s)
    with sqlite3.connect(s.path) as db:
        raw = db.execute("SELECT payload_json FROM field_match_state").fetchone()[0]
        db.execute("UPDATE field_match_state SET payload_json=?", ("\n " + raw + " \n",))
        db.execute(
            "INSERT INTO field_admin_audit VALUES(?,?,?,?,?,?,?)",
            (
                1,
                BINDING.session_id,
                "type",
                "reason",
                ' {"receipt":"保留"} ',
                0,
                "2026-10-30T00:00:00.000000Z",
            ),
        )
    downgrade20(s)
    before = snapshot(s.path)
    store(s.path)
    after = snapshot(s.path)
    assert after.pop("field_recording_ends") == []
    assert after.pop("field_recording_evidence") == []
    assert after.pop("field_session_recording_policy") == []
    assert after.pop("field_rule_fact_preimages") == []
    assert after.pop("field_owner_completion_outbox") == []
    assert after.pop("field_local_media_schedule_state") == []
    assert after.pop("field_upload_receipt_outbox") == []
    assert after.pop("field_saved_upload_requests") == []
    assert after.pop("field_owner_media_deletions") == []
    for table in TABLES:
        assert after.pop(table) == []
    assert after == before


def test_stale_schema19_initializer_accepts_another_initializer_reaching22(tmp_path, monkeypatch):
    from tests.unit.edge.route_assistance.test_managed_retirement import downgrade19

    s = store(tmp_path / "state.db")
    reserve(s)
    downgrade19(s)
    migrate = SQLiteFieldRecordStore._migrate_managed_requests20
    observed = []

    def delayed(self, connection):
        if not observed:
            observed.append("saw19")
            store(self.path)
            observed.append(connection.execute("PRAGMA user_version").fetchone()[0])
        return migrate(self, connection)

    monkeypatch.setattr(SQLiteFieldRecordStore, "_migrate_managed_requests20", delayed)
    initialized = store(s.path)
    assert observed == ["saw19", 30]
    assert initialized.get_managed_open_request(SCOPE, REQUEST).open_request_id == REQUEST
    assert snapshot(s.path)["field_managed_resource_specs"] == []


@pytest.mark.parametrize("raw", ["[]", "null", "{" + " " * (1024 * 1024)])
def test_invalid_or_oversized_resource_json_refuses_without_repair(tmp_path, raw):
    s = store(tmp_path / "state.db")
    reserve(s)
    with sqlite3.connect(s.path) as db:
        db.execute("UPDATE field_managed_resource_specs SET spec_json=?", (raw,))
    before = snapshot(s.path)
    with pytest.raises(ValueError):
        s.get_managed_resource_spec(SCOPE, REQUEST)
    assert snapshot(s.path) == before
