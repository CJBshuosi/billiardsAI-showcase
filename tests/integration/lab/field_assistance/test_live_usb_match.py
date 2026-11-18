# Imported fixtures are registered with pytest, not ordinary unused imports.
# ruff: noqa: F811
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.memory_gate import (
    FieldMemoryAdmissionError,
    require_personal_memory_admission,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import LiveShotMediaReference
from billiards_ai.lab.field_assistance.live_match import LiveMatchCoordinator
from tests.fixtures.persistence.legacy_field_schema import (
    assert_legacy_schema,
    create_legacy_database,
    legacy_rows,
)
from tests.unit.lab.field_assistance.test_live_match import (  # noqa: F401
    coordinator,
    settled_shot,
    start,
    store,
)


def test_usb_settle_confirmation_and_media_survive_restart(coordinator, settled_shot, store):
    started = start(coordinator)
    actor = store.get_shot_ownership(started.session_id, "shot-1")
    pending = coordinator.settle(settled_shot, facts=())
    media = LiveShotMediaReference(
        started.session_id, "shot-1", "epoch-1", 1, "job-1", "/capture/immutable/job-1", "a" * 64
    )
    store.attach_live_shot_media(media)
    store.attach_live_shot_media(media)
    reopened = SQLiteFieldRecordStore(store.path)
    assert reopened.get_live_shot_media(started.session_id, "shot-1") == media
    assert reopened.get_shot_ownership(started.session_id, "shot-1") == actor
    restored = LiveMatchCoordinator(
        store=reopened,
        session_id=started.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )
    result = restored.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert result.match.active_role == started.match.active_role
    assert result.guidance is None
    assert result.route_abstained_reason == "settled_observation_unavailable"
    assert reopened.list_shots(started.session_id)[0].media.source_kind == "usb_live"
    with pytest.raises(FieldMemoryAdmissionError):
        require_personal_memory_admission(reopened, started.session_id)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_epoch_id": "wrong"},
        {"shot_id": "shot-other"},
        {"match_revision": 2},
        {"session_id": "session-other"},
    ],
)
def test_media_cannot_attach_to_different_start(coordinator, store, changes):
    shot = start(coordinator)
    media = LiveShotMediaReference(
        shot.session_id, "shot-1", "epoch-1", 1, "job-1", "/capture/immutable/job-1", "a" * 64
    )
    with pytest.raises(ValueError, match="lineage"):
        store.attach_live_shot_media(replace(media, **changes))
    assert store.get_live_shot_media(shot.session_id, "shot-1") is None


def test_media_hash_is_immutable_and_does_not_rebind_actor(coordinator, store):
    shot = start(coordinator)
    media = LiveShotMediaReference(
        shot.session_id, "shot-1", "epoch-1", 1, "job-1", "/capture/immutable/job-1", "a" * 64
    )
    store.attach_live_shot_media(media)
    with pytest.raises(ValueError, match="conflict"):
        store.attach_live_shot_media(replace(media, media_sha256="b" * 64))
    actor = store.get_shot_ownership(shot.session_id, "shot-1")
    assert actor.actor_user_id == "alice"


def test_changed_settled_facts_are_rejected_after_restart(coordinator, settled_shot, store):
    started = start(coordinator)
    coordinator.settle(settled_shot, facts=())
    restored = LiveMatchCoordinator(
        store=SQLiteFieldRecordStore(store.path),
        session_id=started.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )
    with pytest.raises(ValueError, match="conflict"):
        restored.settle(replace(settled_shot, job_id="another-job"), facts=())


def test_media_attachment_rejects_job_different_from_settlement(coordinator, settled_shot, store):
    started = start(coordinator)
    coordinator.settle(settled_shot, facts=())
    media = LiveShotMediaReference(
        started.session_id,
        "shot-1",
        "epoch-1",
        1,
        "wrong-job",
        "/capture/immutable/wrong",
        "a" * 64,
    )
    with pytest.raises(ValueError, match="lineage"):
        store.attach_live_shot_media(media)


def test_delayed_media_preserves_actor_after_confirmed_switch(coordinator, settled_shot, store):
    started = start(coordinator)
    actor = store.get_shot_ownership(started.session_id, "shot-1")
    pending = coordinator.settle(settled_shot, facts=())
    coordinator.confirm_turn("cmd-1", "switch_turn", pending.confirmation_ref)
    media = LiveShotMediaReference(
        started.session_id, "shot-1", "epoch-1", 1, "job-1", "/capture/immutable/job-1", "a" * 64
    )
    store.attach_live_shot_media(media)
    assert store.get_shot_ownership(started.session_id, "shot-1") == actor


def test_v7_migration_preserves_start_and_rolls_back_failed_attachment_schema(
    coordinator,
    store,
    monkeypatch,
):
    import sqlite3

    from billiards_ai.edge.route_assistance import persistence

    started = start(coordinator)
    session = store.load_session(started.session_id)
    actor = store.get_shot_ownership(started.session_id, "shot-1")
    legacy_path = store.path.with_name("field-v7.db")
    before = create_legacy_database(store.path, legacy_path, 7)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 7)
    schema = persistence._LIVE_MEDIA_SCHEMA
    monkeypatch.setattr(persistence, "_LIVE_MEDIA_SCHEMA", schema + "INVALID SQL;")
    with pytest.raises(ValueError, match="database"):
        SQLiteFieldRecordStore(legacy_path)
    with sqlite3.connect(legacy_path) as db:
        assert_legacy_schema(db, 7)
        assert legacy_rows(db, ownership=True) == before
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE name='field_live_shot_media'"
            ).fetchall()
            == []
        )
    monkeypatch.setattr(persistence, "_LIVE_MEDIA_SCHEMA", schema)
    reopened = SQLiteFieldRecordStore(legacy_path)
    assert reopened.get_live_shot_media(started.session_id, "shot-1") is None
    assert reopened.load_session(started.session_id) == session
    assert reopened.get_shot_ownership(started.session_id, "shot-1") == actor
    with sqlite3.connect(legacy_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert legacy_rows(db, ownership=True) == before


def test_development_replay_remains_excluded_from_personal_memory(tmp_path):
    from tests.unit.edge.route_assistance.test_turn_ownership import make_store, match, shot

    replay = make_store(tmp_path)
    replay.save_match("session-1", match(), expected_revision=0)
    replay.record_shot_start(
        shot(), source_mainline1_artifact_id="a" * 64, source_start_event_id="replay-start"
    )
    with pytest.raises(FieldMemoryAdmissionError, match="DEVELOPMENT_REPLAY_EXCLUDED"):
        require_personal_memory_admission(replay, "session-1")


def test_restarted_epoch_cannot_settle_previous_epoch_start(coordinator, settled_shot, store):
    begun = start(coordinator)
    ownership = store.get_shot_ownership(begun.session_id, "shot-1")
    restarted = LiveMatchCoordinator(
        store=SQLiteFieldRecordStore(store.path),
        session_id=begun.session_id,
        source_epoch_id="epoch-2",
        calibration_id="cal-1",
    )
    with pytest.raises(ValueError, match="epoch"):
        restarted.settle(replace(settled_shot, source_epoch_id="epoch-2"), facts=())
    assert store.list_shots(begun.session_id)[0].settled_ms is None
    assert store.get_shot_ownership(begun.session_id, "shot-1") == ownership
    assert store.get_turn_confirmation(begun.session_id, "shot-1") is None
    assert restarted.snapshot.guidance is None


def test_other_coordinator_cannot_confirm_new_shot_with_old_geometry(
    coordinator,
    settled_shot,
    store,
):
    begun = start(coordinator)
    first = coordinator.settle(settled_shot, facts=())
    coordinator.confirm_turn("cmd-1", "continue_turn", first.confirmation_ref)
    other = LiveMatchCoordinator(
        store=store,
        session_id=begun.session_id,
        source_epoch_id="epoch-1",
        calibration_id="cal-1",
    )
    other.shot_started(shot_id="shot-2", started_ms=300)
    second_shot = replace(
        settled_shot,
        shot_id="shot-2",
        job_id="job-2",
        settled_at_ns=400_000_000,
        observations=(replace(settled_shot.observations[0], timestamp_ms=400),),
    )
    second = other.settle(second_shot, facts=())
    assert coordinator.snapshot.guidance is None
    result = coordinator.confirm_turn("cmd-2", "continue_turn", second.confirmation_ref)
    assert result.shot_id == "shot-2"
    assert result.match.revision == 3
    assert result.guidance is None
    assert result.route_abstained_reason == "settled_observation_lineage_mismatch"
    assert coordinator.snapshot.guidance is None
    assert coordinator.guidance_result is None
    current = other.confirm_turn("cmd-2", "continue_turn", second.confirmation_ref)
    assert current.guidance is not None


def test_settlement_rejects_ownership_epoch_different_from_start(coordinator, settled_shot, store):
    import json
    import sqlite3

    begun = start(coordinator)
    with sqlite3.connect(store.path) as db:
        raw = db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
        payload = json.loads(raw)
        payload["source_epoch_id"] = "epoch-corrupt"
        db.execute("UPDATE field_shot_ownership SET payload_json=?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="epoch"):
        coordinator.settle(settled_shot, facts=())
    assert store.list_shots(begun.session_id)[0].settled_ms is None
    assert store.get_turn_confirmation(begun.session_id, "shot-1") is None


def test_snapshot_clears_route_when_persisted_evidence_digest_changes(
    coordinator,
    settled_shot,
    store,
):
    import json
    import sqlite3

    start(coordinator)
    pending = coordinator.settle(settled_shot, facts=())
    confirmed = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert confirmed.guidance is not None
    with sqlite3.connect(store.path) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM field_shots").fetchone()[0])
        payload["live_evidence_digest"] = "f" * 64
        db.execute("UPDATE field_shots SET payload_json=?", (json.dumps(payload),))
    refreshed = coordinator.snapshot
    assert refreshed.guidance is None
    assert refreshed.route_abstained_reason == "settled_observation_lineage_mismatch"
    assert coordinator.guidance_result is None
    repeated = coordinator.confirm_turn("cmd-1", "continue_turn", pending.confirmation_ref)
    assert repeated.guidance is None
    assert repeated.route_abstained_reason == "settled_observation_lineage_mismatch"
