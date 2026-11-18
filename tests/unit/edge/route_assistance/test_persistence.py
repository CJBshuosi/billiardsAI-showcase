from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.control import (
    BallGroupAssignment,
    MatchControlSnapshot,
    WristbandRole,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import (
    MAX_JS_SAFE_INTEGER,
    MAX_SAVED_CLIP_BYTES,
    MAX_SAVED_CLIP_DURATION_MS,
    PostgameSessionManifest,
)
from billiards_ai.edge.route_assistance.records import (
    FieldSessionStatus,
    FieldShotRecord,
    LiveShotFact,
    LiveShotMediaReference,
    SessionLimitCode,
    ShotMediaRange,
)

NOW = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


def _store(path: Path, *, session_id: str = "session-1") -> SQLiteFieldRecordStore:
    return SQLiteFieldRecordStore(
        path,
        now_utc=lambda: datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        session_id_factory=lambda: session_id,
    )


def _match(revision: int, *, role: WristbandRole = WristbandRole.RED) -> MatchControlSnapshot:
    return MatchControlSnapshot(
        red_user_id="manual-red",
        blue_user_id="manual-blue",
        breaker=WristbandRole.RED,
        active_role=role,
        red_group=BallGroupAssignment.STRIPE,
        blue_group=BallGroupAssignment.SOLID,
        revision=revision,
    )


def _media(*, end_ms: int = 10_000) -> ShotMediaRange:
    return ShotMediaRange(
        artifact_id="artifact-1",
        source_locator="presentation.mp4",
        source_kind="development_replay",
        start_ms=1_000,
        end_ms=end_ms,
    )


def _shot(*, settled_ms: int | None = None) -> FieldShotRecord:
    return FieldShotRecord(
        session_id="session-1",
        shot_id="shot-0001",
        sequence=1,
        actor_role=WristbandRole.RED,
        started_ms=3_000,
        settled_ms=settled_ms,
        recommendation_id="recommendation-1",
        power_band="medium_light",
        power_percent=40,
        media=_media(end_ms=7_000 if settled_ms is not None else 10_000),
        limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
    )


def _command(
    command_id: str,
    *,
    command_type: str = "claim_role",
    payload: dict[str, str] | None = None,
    revision: int = 0,
    sequence: int = 1,
    actor_user_id: str = "user-alice",
) -> FieldCommandEnvelope:
    return FieldCommandEnvelope(
        command_id=command_id,
        session_id="session-1",
        actor_user_id=actor_user_id,
        command_type=command_type,
        payload={"role": "red"} if payload is None else payload,
        expected_revision=revision,
        server_sequence=sequence,
    )


def test_restart_resumes_open_session_and_match_revision(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    first = _store(path)
    opened = first.open_or_resume_session(artifact_id="artifact-1")
    first.save_match(opened.session_id, _match(3), expected_revision=0)

    resumed = _store(path, session_id="session-2").open_or_resume_session(artifact_id="artifact-1")

    assert resumed.session_id == opened.session_id
    assert resumed.match == _match(3)
    assert resumed.status is FieldSessionStatus.OPEN


def test_stale_match_revision_cannot_overwrite_newer_state(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    session = store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match(session.session_id, _match(1), expected_revision=0)

    with pytest.raises(ValueError, match="stale match revision"):
        store.save_match(
            session.session_id,
            _match(2, role=WristbandRole.BLUE),
            expected_revision=0,
        )


def test_duplicate_shot_start_is_idempotent_but_conflict_fails(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    shot = _shot()

    store.record_shot_start(shot)
    store.record_shot_start(shot)

    assert store.list_shots("session-1") == (shot,)
    with pytest.raises(ValueError, match="conflicting shot evidence"):
        store.record_shot_start(replace(shot, actor_role=WristbandRole.BLUE))


def test_settle_updates_open_shot_and_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.record_shot_start(_shot())
    settled = _shot(settled_ms=5_000)

    store.record_shot_settle(settled)
    store.record_shot_settle(settled)

    assert store.list_shots("session-1") == (settled,)


def test_live_shot_fact_outbox_is_atomic_restart_safe_and_contains_no_video(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.record_shot_start(_shot())
    settled = _shot(settled_ms=5_000)
    fact = LiveShotFact.create(
        session_id="session-1",
        shot_id="shot-0001",
        sequence=1,
        start_frame_id=100,
        settle_frame_id=220,
        mode="observation",
        actor_role=WristbandRole.RED,
        actor_user_id="manual-red",
        observed_outcome="unknown",
        rule_revision=0,
        recording_reference="session-recording:epoch-1:100-220",
    )

    store.record_shot_settle(settled, live_fact=fact)
    reopened = _store(path)
    leased = reopened.claim_due_live_shot_fact(datetime(2026, 8, 26, 10, 0, tzinfo=UTC))

    assert leased is not None
    assert leased.fact == fact
    assert leased.status == "leased"
    assert "video" not in leased.payload_json
    assert "media_bytes" not in leased.payload_json
    assert "local_path" not in leased.payload_json
    reopened.retry_live_shot_fact(fact.idempotency_key)
    retried = reopened.claim_due_live_shot_fact(datetime(2026, 8, 26, 10, 0, tzinfo=UTC))
    assert retried is not None and retried.idempotency_key == fact.idempotency_key
    reopened.acknowledge_live_shot_fact(fact.idempotency_key)
    reopened.acknowledge_live_shot_fact(fact.idempotency_key)
    assert reopened.claim_due_live_shot_fact(datetime(2026, 8, 26, 10, 0, tzinfo=UTC)) is None


def test_one_hundred_live_shot_facts_remain_small_and_do_not_enqueue_video(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")

    for sequence in range(1, 101):
        shot_id = f"shot-{sequence:04d}"
        started_ms = sequence * 10_000
        media = replace(_media(), start_ms=started_ms - 1_000, end_ms=started_ms + 6_000)
        started = replace(
            _shot(),
            shot_id=shot_id,
            sequence=sequence,
            actor_role=WristbandRole.RED if sequence % 2 else WristbandRole.BLUE,
            started_ms=started_ms,
            settled_ms=None,
            media=media,
        )
        settled = replace(
            started,
            settled_ms=started_ms + 5_000,
        )
        fact = LiveShotFact.create(
            session_id="session-1",
            shot_id=shot_id,
            sequence=sequence,
            start_frame_id=sequence * 100,
            settle_frame_id=sequence * 100 + 50,
            mode="observation",
            actor_role=WristbandRole.RED if sequence % 2 else WristbandRole.BLUE,
            actor_user_id="manual-red" if sequence % 2 else "manual-blue",
            observed_outcome="unknown",
            rule_revision=sequence,
            recording_reference=(
                f"session-recording:epoch-1:{sequence * 100}-{sequence * 100 + 50}"
            ),
        )
        store.record_shot_start(started)
        store.record_shot_settle(settled, live_fact=fact)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT f.payload_json FROM field_live_shot_fact_outbox o "
            "JOIN field_live_shot_facts f USING(idempotency_key) "
            "ORDER BY f.shot_id"
        ).fetchall()

    assert len(rows) == 100
    assert sum(len(row[0].encode("utf-8")) for row in rows) < 100_000
    assert all("video" not in row[0] and "local_path" not in row[0] for row in rows)


def test_legacy_persisted_presentation_source_is_mapped_on_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.record_shot_start(_shot())
    with sqlite3.connect(path) as connection:
        raw = connection.execute(
            "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
            ("session-1", "shot-0001"),
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["media"]["source_kind"] = "development_presentation"
        connection.execute(
            "UPDATE field_shots SET payload_json=? WHERE session_id=? AND shot_id=?",
            (json.dumps(payload), "session-1", "shot-0001"),
        )

    restored = store.list_shots("session-1")[0]

    assert restored.media.source_kind == "development_replay"
    with sqlite3.connect(path) as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
                ("session-1", "shot-0001"),
            ).fetchone()[0]
        )
    assert persisted["media"]["source_kind"] == "development_presentation"


def test_unknown_persisted_media_source_still_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.record_shot_start(_shot())
    with sqlite3.connect(path) as connection:
        raw = connection.execute(
            "SELECT payload_json FROM field_shots WHERE session_id=? AND shot_id=?",
            ("session-1", "shot-0001"),
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["media"]["source_kind"] = "file_replay"
        connection.execute(
            "UPDATE field_shots SET payload_json=? WHERE session_id=? AND shot_id=?",
            (json.dumps(payload), "session-1", "shot-0001"),
        )

    with pytest.raises(ValueError, match="persisted shot state is invalid"):
        store.list_shots("session-1")


def test_end_session_is_idempotent_and_next_open_is_new(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    first = _store(path)
    session = first.open_or_resume_session(artifact_id="artifact-1")

    ended = first.end_session(session.session_id)
    assert first.end_session(session.session_id) == ended
    assert ended.status is FieldSessionStatus.ENDED

    next_session = _store(path, session_id="session-2").open_or_resume_session(
        artifact_id="artifact-1"
    )
    assert next_session.session_id == "session-2"


def test_end_session_freezes_one_postgame_manifest_and_open_session_has_none(
    tmp_path: Path,
) -> None:
    """Catches a worker being able to infer work from a live, mutable session."""
    store = _store(tmp_path / "field.sqlite3")
    session = store.open_or_resume_session(artifact_id="artifact-1")

    assert store.get_postgame_manifest(session.session_id) is None
    store.end_session(session.session_id)
    manifest = store.get_postgame_manifest(session.session_id)

    assert isinstance(manifest, PostgameSessionManifest)
    assert manifest.session_id == session.session_id
    assert manifest.expected_jobs == ()
    assert manifest.deadline_at_utc - manifest.ended_at_utc == timedelta(seconds=120)


def test_postgame_work_remains_discoverable_until_local_analysis_and_sync_complete(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "field.sqlite3")
    session = store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session(session.session_id)
    queued = store.enqueue_sync_job(
        session_id=session.session_id,
        manifest_hash="a" * 64,
        manifest_json='{"schema_version":"1.0"}',
    )

    assert [row.session_id for row in store.list_ended_sessions_requiring_postgame()] == [
        session.session_id
    ]

    store.record_postgame_finalization(session.session_id, '{"clips":[],"report":{}}')
    assert [row.session_id for row in store.list_ended_sessions_requiring_postgame()] == [
        session.session_id
    ]

    claimed = store.claim_due_sync_job(datetime(2026, 8, 26, 10, 0, tzinfo=UTC))
    assert claimed is not None and claimed.sync_job_id == queued.sync_job_id
    store.mark_sync_complete(claimed.sync_job_id)
    assert store.list_ended_sessions_requiring_postgame() == ()


def test_incomplete_abstained_shot_is_not_frozen_as_a_video_analysis_job(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match("session-1", _match(1), expected_revision=0)
    started = replace(
        _shot(),
        media=ShotMediaRange(
            artifact_id="artifact-1",
            source_locator="usb:epoch-1",
            source_kind="usb_live",
            start_ms=3_000,
            end_ms=3_001,
            source_epoch_id="epoch-1",
        ),
        limitations=(),
    )
    store.record_shot_start(started, source_start_event_id="start-incomplete")
    store.record_shot_settle(
        replace(
            started,
            settled_ms=3_240,
            media=replace(started.media, end_ms=3_240),
            live_evidence_digest="e" * 64,
            live_job_id="abstained:incomplete_freeze:epoch-1:shot-0001",
        ),
        source_epoch_id="epoch-1",
    )

    store.end_session("session-1")

    assert store.get_postgame_manifest("session-1").expected_jobs == ()


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=99")

    with pytest.raises(ValueError, match="schema version"):
        _store(path)


def test_schema18_migration_preserves_every_legacy_row_receipt_and_v2_descriptor(tmp_path):
    from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
    from billiards_ai.lab.mainline1_shot_jobs import ShotAnalysisJob, SQLiteShotAnalysisJobStore

    path = tmp_path / "field.sqlite3"
    original = _postgame_store(path)
    _available_clip(original, tmp_path)
    _save_clip(original)
    original.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    descriptor = original.list_saved_media_outbox()[0].descriptor_json
    jobs_path = tmp_path / "jobs.sqlite3"
    jobs = SQLiteShotAnalysisJobStore(jobs_path)
    jobs.enqueue(
        ShotAnalysisJob(
            "job-1",
            "session-1",
            "shot-0001",
            "epoch-1",
            1,
            2,
            tmp_path / "jobs",
            "a" * 64,
            "setup-1",
            "calibration-1",
            "b" * 64,
            "model-1",
        )
    )
    with sqlite3.connect(jobs_path) as db:
        original_job = db.execute("SELECT * FROM shot_analysis_jobs").fetchall()
        identity = json.loads(original_job[0][1])
        assert set(identity) == {
            "schema_version",
            "job_id",
            "session_id",
            "shot_id",
            "source_epoch_id",
            "start_frame_id",
            "end_frame_id",
            "media_root",
            "media_sha256",
            "capture_setup_id",
            "calibration_artifact_id",
            "config_sha256",
            "model_package_id",
            "identity_training_package_id",
            "identity_training_package_sha256",
        }
    with sqlite3.connect(path) as db:
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
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        before = {
            name: db.execute(f"SELECT * FROM {name} ORDER BY rowid").fetchall() for name in tables
        }
        ownership = json.loads(
            db.execute("SELECT payload_json FROM field_shot_ownership").fetchone()[0]
        )
        assert set(ownership) == {
            "session_id",
            "shot_id",
            "sequence",
            "actor_user_id",
            "actor_role",
            "match_revision",
            "source_artifact_id",
            "source_mainline1_artifact_id",
            "source_started_ms",
            "recorded_at_utc",
            "evidence_status",
            "source_kind",
            "revision",
            "supersedes_revision",
            "source_start_event_id",
            "evidence_reason",
            "source_epoch_id",
        }
    migrated = _store(path)
    assert migrated.list_saved_media_outbox()[0].descriptor_json == descriptor
    assert json.loads(descriptor)["schema_version"] == "2.0"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert db.execute("SELECT * FROM field_managed_open_requests").fetchall() == []
    migrated.reserve_managed_open(
        ManagedSessionScope("dev", "device-1", "table-1"), artifact_id="new-artifact"
    )
    with sqlite3.connect(path) as db:
        assert {
            name: db.execute(f"SELECT * FROM {name} ORDER BY rowid").fetchall() for name in tables
        } == before
    with sqlite3.connect(jobs_path) as db:
        assert db.execute("SELECT * FROM shot_analysis_jobs").fetchall() == original_job


def _postgame_store(path: Path) -> SQLiteFieldRecordStore:
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match("session-1", _match(1), expected_revision=0)
    started = replace(
        _shot(),
        media=ShotMediaRange(
            artifact_id="artifact-1",
            source_locator="usb:epoch-1",
            source_kind="usb_live",
            start_ms=3_000,
            end_ms=3_001,
            source_epoch_id="epoch-1",
        ),
        limitations=(),
    )
    store.record_shot_start(started, source_start_event_id="start-1")
    store.record_shot_settle(
        replace(
            started,
            settled_ms=5_000,
            media=replace(started.media, end_ms=5_000),
            live_evidence_digest="e" * 64,
            live_job_id="job-1",
        ),
        source_epoch_id="epoch-1",
    )
    store.attach_live_shot_media(
        LiveShotMediaReference(
            "session-1",
            "shot-0001",
            "epoch-1",
            1,
            "job-1",
            str(path.parent / "usb"),
            "d" * 64,
        )
    )
    store.end_session("session-1")
    return store


def _save_clip(store: SQLiteFieldRecordStore, **overrides):
    return store.save_postgame_clip(
        **{
            "request_id": "save-1",
            "session_id": "session-1",
            "shot_id": "shot-0001",
            "postgame_manifest_sha256": store.get_postgame_manifest("session-1").manifest_sha256,
            **overrides,
        }
    )


def _available_clip(store: SQLiteFieldRecordStore, tmp_path: Path, **overrides):
    return store.record_postgame_clip_available(
        **{
            "session_id": "session-1",
            "shot_id": "shot-0001",
            "job_id": "job-1",
            "artifact_id": "c" * 64,
            "local_path": tmp_path / "clip.mp4",
            "content_sha256": "a" * 64,
            "byte_size": 100,
            "postgame_manifest_sha256": store.get_postgame_manifest("session-1").manifest_sha256,
            **overrides,
        }
    )


def test_ready_review_completion_discards_unsaved_clip_and_fences_late_writes(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    digest = store.get_postgame_manifest("session-1").manifest_sha256
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    _available_clip(store, tmp_path)
    store.complete_postgame_review("artifact-1", "session-1", digest)
    assert store.is_postgame_discarded("session-1", "shot-0001")
    assert store.active_postgame_review("artifact-1") is None
    with pytest.raises(ValueError, match="discard"):
        _save_clip(store)
    with pytest.raises(ValueError, match="discard"):
        _available_clip(store, tmp_path)


def test_pending_legacy_review_completion_queues_cleanup_and_fences_late_publication(tmp_path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    manifest = store.get_postgame_manifest("session-1")
    store.complete_postgame_review("artifact-1", "session-1", manifest.manifest_sha256)
    assert store.get_postgame_manifest("session-1") == manifest
    assert store.is_postgame_work_fenced("session-1")
    assert store.active_postgame_review("artifact-1") is None
    assert store.claim_due_local_discard("cleaner", NOW) is None
    with sqlite3.connect(store.path) as db:
        assert db.execute(
            "SELECT session_id,manifest_sha256,status FROM field_recording_cleanup"
        ).fetchall() == [("session-1", manifest.manifest_sha256, "pending")]
    with pytest.raises(ValueError, match="discard"):
        _available_clip(store, tmp_path)


@pytest.mark.parametrize("state", ["pending", "leased", "blocked", "acknowledged"])
def test_any_save_intent_protects_from_completion_discard(tmp_path: Path, state: str):
    store = _postgame_store(tmp_path / "field.sqlite3")
    original = _save_clip(store)
    _available_clip(store, tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE field_saved_media_outbox SET status=?", (state,))
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    digest = store.get_postgame_manifest("session-1").manifest_sha256
    store.complete_postgame_review("artifact-1", "session-1", digest)
    assert not store.is_postgame_discarded("session-1", "shot-0001")
    assert _save_clip(store) == original
    assert store.claim_due_local_discard("cleaner", NOW) is None


@pytest.mark.parametrize("terminal", ["completed", "blocked"])
def test_discard_claims_are_durable_and_stale_tokens_cannot_update(tmp_path: Path, terminal: str):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    store.complete_postgame_review(
        "artifact-1", "session-1", store.get_postgame_manifest("session-1").manifest_sha256
    )
    claim = store.claim_due_local_discard("first", NOW, lease_seconds=1)
    assert claim.session_id == "session-1" and claim.job_id == "job-1"
    assert claim.local_path == tmp_path / "clip.mp4"
    assert store.claim_due_local_discard("second", NOW) is None
    replacement = _store(store.path).claim_due_local_discard("second", NOW + timedelta(seconds=2))
    assert replacement.claim_token != claim.claim_token
    assert not store.complete_local_discard(claim)
    assert not store.update_local_discard_progress(claim, '{"stale":true}')
    assert store.update_local_discard_progress(replacement, '{"overlay":"registered"}')
    assert store.retry_local_discard(
        replacement, "BUSY", NOW + timedelta(seconds=3), progress_json='{"overlay":"detached"}'
    )
    retry = store.claim_due_local_discard("third", NOW + timedelta(seconds=3))
    assert retry.progress_json == '{"overlay":"detached"}'
    if terminal == "completed":
        assert store.complete_local_discard(retry)
    else:
        assert store.block_local_discard(retry, "UNSAFE")
    assert store.claim_due_local_discard("fourth", NOW + timedelta(days=1)) is None
    assert store.is_postgame_discarded("session-1", "shot-0001")


@pytest.mark.parametrize("operation", ["checkpoint", "complete"])
def test_discard_progress_budget_counts_utf8_bytes(tmp_path: Path, operation: str):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    store.complete_postgame_review(
        "artifact-1", "session-1", store.get_postgame_manifest("session-1").manifest_sha256
    )
    claim = store.claim_due_local_discard("cleaner", NOW)
    # Canonical overhead is eight bytes; a three-byte code point fills the rest.
    within = json.dumps({"p": "界" * 21842 + "aa"}, ensure_ascii=False, separators=(",", ":"))
    over = json.dumps({"p": "界" * 21843}, ensure_ascii=False, separators=(",", ":"))
    assert len(within.encode("utf-8")) == 64 * 1024
    assert len(over) < 64 * 1024 < len(over.encode("utf-8"))

    def apply_progress(progress):
        if operation == "checkpoint":
            return store.update_local_discard_progress(claim, progress)
        return store.complete_local_discard(claim, progress_json=progress)

    with pytest.raises(ValueError, match="progress"):
        apply_progress(over)
    assert apply_progress(within)
    with sqlite3.connect(store.path) as db:
        persisted = db.execute("SELECT progress_json FROM field_local_media_discards").fetchone()[0]
        assert persisted == within


def test_save_and_complete_compete_atomically_across_store_connections(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _available_clip(store, tmp_path)
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    digest = store.get_postgame_manifest("session-1").manifest_sha256
    barrier = Barrier(2)

    def save():
        other = _store(path)
        barrier.wait(timeout=5)
        try:
            return _save_clip(other)
        except ValueError as exc:
            assert "discard" in str(exc)
            return None

    def complete():
        other = _store(path)
        barrier.wait(timeout=5)
        other.complete_postgame_review("artifact-1", "session-1", digest)

    with ThreadPoolExecutor(max_workers=2) as pool:
        saved, completed = pool.submit(save), pool.submit(complete)
        receipt = saved.result(timeout=10)
        completed.result(timeout=10)
    assert store.is_postgame_discarded("session-1", "shot-0001") == (receipt is None)
    if receipt is not None:
        assert _save_clip(store) == receipt


@pytest.mark.parametrize("selected", [False, True])
def test_schema17_migration_preserves_history_without_destructive_backfill(
    tmp_path: Path, selected: bool
):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    original = _save_clip(store) if selected else None
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE field_postgame_reviews SET status='completed'")
        db.execute("DROP TABLE field_owner_completion_outbox")
        db.execute("DROP TABLE field_local_media_schedule_state")
        db.execute("DROP TABLE field_upload_receipt_outbox")
        db.execute("DROP TABLE field_saved_upload_requests")
        db.execute("DROP TABLE field_owner_media_deletions")
        db.execute("DROP TABLE field_managed_capture_epochs")
        db.execute("DROP TABLE field_managed_resource_specs")
        db.execute("DROP TABLE field_managed_open_requests")
        db.execute("DROP TABLE field_local_media_discards")
        db.execute("DROP TABLE IF EXISTS field_recording_ends")
        db.execute("DROP TABLE IF EXISTS field_recording_evidence")
        db.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        db.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        db.execute("PRAGMA user_version=17")
        tables = (
            "field_sessions",
            "field_shots",
            "field_postgame_manifests",
            "field_postgame_finalizations",
            "field_postgame_clip_availability",
            "field_clip_save_requests",
            "field_saved_media_outbox",
            "field_postgame_reviews",
        )
        before = {name: db.execute(f"SELECT * FROM {name}").fetchall() for name in tables}
    migrated = _store(store.path)
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 30
        assert {name: db.execute(f"SELECT * FROM {name}").fetchall() for name in tables} == before
    assert not migrated.is_postgame_discarded("session-1", "shot-0001")
    assert migrated.claim_due_local_discard("cleaner", NOW) is None
    if original is not None:
        assert _save_clip(migrated) == original


def test_empty_no_shot_review_can_complete_without_media_finalization(tmp_path: Path):
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session("session-1")
    manifest = store.get_postgame_manifest("session-1")
    assert manifest.expected_jobs == ()
    store.complete_postgame_review("artifact-1", "session-1", manifest.manifest_sha256)
    assert store.active_postgame_review("artifact-1") is None
    assert store.claim_due_local_discard("cleaner", NOW) is None


def test_clip_save_pending_survives_restart_and_availability_enqueues_once(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    before = store.list_shots("session-1")
    pending = _save_clip(store)
    assert pending.status == "SAVE_PENDING"
    assert store.list_saved_media_outbox() == ()
    restarted = _store(path)
    assert _save_clip(restarted) == pending
    assert restarted.list_postgame_clips("session-1")[0].save_status == "SAVE_PENDING"

    _available_clip(restarted, tmp_path)
    _available_clip(restarted, tmp_path)
    assert _save_clip(restarted) == pending  # exact request replay is immutable
    assert _save_clip(restarted, request_id="save-2").status == "SAVED"
    assert restarted.list_postgame_clips("session-1")[0].save_status == "SAVED"
    assert len(restarted.list_saved_media_outbox()) == 1
    assert restarted.list_shots("session-1") == before


def test_local_availability_does_not_save_and_finalization_digest_is_separate(
    tmp_path: Path,
) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")
    frozen_digest = store.get_postgame_manifest("session-1").manifest_sha256
    _available_clip(store, tmp_path)
    assert store.list_postgame_clips("session-1")[0].save_status == "NOT_SAVED"
    assert store.list_saved_media_outbox() == ()
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    with sqlite3.connect(store.path) as connection:
        final_digest = connection.execute(
            "SELECT manifest_sha256 FROM field_postgame_finalizations"
        ).fetchone()[0]
    assert frozen_digest != final_digest
    with pytest.raises(ValueError, match="manifest"):
        _save_clip(store, postgame_manifest_sha256=final_digest)
    assert _save_clip(store).status == "SAVED"


def test_clip_save_reused_request_id_conflicts_and_stale_or_unknown_identity_fails(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _save_clip(store)
    with pytest.raises(ValueError, match="request.*conflict"):
        _save_clip(store, shot_id="other-shot")
    for changes in (
        {"postgame_manifest_sha256": "f" * 64},
        {"shot_id": "unknown"},
        {"session_id": "unknown"},
        {"request_id": ""},
    ):
        with pytest.raises(ValueError):
            _save_clip(store, **{"request_id": "new-save", **changes})
    assert store.list_saved_media_outbox() == ()


@pytest.mark.parametrize(
    "request_id",
    ("contains space", "slash/not-allowed", "雪", "control\n", "a" * 129),
)
def test_clip_save_rejects_non_cloud_safe_request_id_before_persistence(
    tmp_path: Path, request_id: str
) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")

    with pytest.raises(ValueError, match="request id"):
        _save_clip(store, request_id=request_id)

    with sqlite3.connect(store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0]
        assert count == 0


def test_clip_save_accepts_cloud_safe_request_id_at_128_character_boundary(
    tmp_path: Path,
) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")
    receipt = _save_clip(store, request_id="a" * 128)
    assert receipt.request_id == "a" * 128


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": "other-job"},
        {"artifact_id": ""},
        {"artifact_id": "analysis-1"},
        {"shot_id": "unknown"},
        {"postgame_manifest_sha256": "f" * 64},
        {"local_path": Path("relative.mp4")},
        {"content_sha256": "bad"},
        {"byte_size": 0},
        {"content_type": "text/plain"},
    ],
)
def test_clip_availability_rejects_invalid_lineage_or_media(tmp_path: Path, changes):
    store = _postgame_store(tmp_path / "field.sqlite3")
    with pytest.raises(ValueError):
        _available_clip(store, tmp_path, **changes)
    assert store.list_saved_media_outbox() == ()
    assert store.list_postgame_clips("session-1")[0].local_path is None


def test_saved_clip_contract_exposes_and_enforces_256_mib_limit(tmp_path: Path) -> None:
    maximum = 268_435_456
    assert MAX_SAVED_CLIP_BYTES == maximum
    store = _postgame_store(tmp_path / "field.sqlite3")

    with pytest.raises(ValueError, match="metadata"):
        _available_clip(store, tmp_path, byte_size=maximum + 1)

    _available_clip(store, tmp_path, byte_size=maximum)
    _save_clip(store)
    item = store.claim_due_saved_media(datetime(2026, 8, 26, 10, tzinfo=UTC))
    assert item is not None and item.byte_size == maximum
    with pytest.raises(ValueError, match="identity"):
        replace(item, clip_id="clip-" + "0" * 64)
    with pytest.raises(ValueError, match="descriptor"):
        replace(item, byte_size=maximum + 1)


def test_saved_clip_descriptor_enforces_duration_and_js_safe_integer_bounds(
    tmp_path: Path,
) -> None:
    assert MAX_SAVED_CLIP_DURATION_MS == 300_000
    assert MAX_JS_SAFE_INTEGER == 9_007_199_254_740_991
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    item = store.claim_due_saved_media(datetime(2026, 8, 26, 10, tzinfo=UTC))
    assert item is not None

    with pytest.raises(ValueError, match="descriptor"):
        replace(item, end_ms=300_001, duration_ms=300_001)
    with pytest.raises(ValueError, match="descriptor"):
        replace(
            item,
            start_ms=MAX_JS_SAFE_INTEGER,
            end_ms=MAX_JS_SAFE_INTEGER + 1,
            duration_ms=1,
        )


def test_clip_availability_conflicts_cannot_replace_saved_media(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _save_clip(store)
    _available_clip(store, tmp_path)
    with pytest.raises(ValueError, match="availability.*conflict"):
        _available_clip(store, tmp_path, content_sha256="b" * 64)
    assert store.list_saved_media_outbox()[0].content_sha256 == "a" * 64


def test_saved_media_outbox_fences_stale_claim_and_replays_ack_without_duplicate_upload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    first = store.claim_due_saved_media(now, lease_seconds=10)
    assert first is not None
    assert store.claim_due_saved_media(now) is None
    restarted = _store(path)
    second = restarted.claim_due_saved_media(now + timedelta(seconds=10))
    assert second is not None
    assert second.idempotency_key == first.idempotency_key == first.clip_id
    assert second.claim_token != first.claim_token
    assert second.attempts == 2
    with pytest.raises(ValueError, match="lease"):
        restarted.acknowledge_saved_media(first.clip_id, first.claim_token)
    with pytest.raises(ValueError, match="lease"):
        restarted.retry_saved_media(first.clip_id, first.claim_token, error_code="offline")
    # A remote idempotent receiver commits once even when the first ack is lost.
    uploads = {}
    for item in (first, second):
        uploads.setdefault(item.idempotency_key, item.content_sha256)
    assert len(uploads) == 1
    restarted.acknowledge_saved_media(second.clip_id, second.claim_token)
    restarted.acknowledge_saved_media(second.clip_id, second.claim_token)
    _available_clip(restarted, tmp_path)
    _save_clip(restarted, request_id="save-again")
    assert restarted.claim_due_saved_media(now + timedelta(days=1)) is None
    assert len(restarted.list_saved_media_outbox()) == 1
    assert restarted.list_saved_media_outbox()[0].status == "acknowledged"


def test_saved_media_claim_derives_stable_descriptor_from_immutable_shot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)

    first = store.claim_due_saved_media(now, lease_seconds=10)
    restarted = _store(path).claim_due_saved_media(now + timedelta(seconds=10))

    assert first is not None and restarted is not None
    assert first.source_kind == restarted.source_kind == "usb_live"
    assert (first.start_ms, first.end_ms, first.duration_ms) == (3_000, 5_000, 2_000)
    assert first.descriptor_json == restarted.descriptor_json
    descriptor = json.loads(first.descriptor_json)
    assert descriptor["schema_version"] == "2.0"
    assert descriptor["owner_user_id"] == "manual-red"
    assert descriptor["ownership_receipt"]["source_start_event_id"] == "start-1"


def test_saved_media_binds_first_durable_save_request_and_exact_receipt_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _save_clip(store, request_id="save-z-first")
    _save_clip(store, request_id="save-a-later")
    _available_clip(store, tmp_path)

    with sqlite3.connect(path) as connection:
        receipt_json = connection.execute(
            "SELECT receipt_json FROM field_clip_save_requests WHERE request_id=?",
            ("save-z-first",),
        ).fetchone()[0]
    expected_receipt_sha256 = hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()

    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    first = store.claim_due_saved_media(now, lease_seconds=10)
    restarted = _store(path).claim_due_saved_media(now + timedelta(seconds=10))
    assert first is not None and restarted is not None
    assert first.save_request_id == restarted.save_request_id == "save-z-first"
    assert first.save_receipt_sha256 == restarted.save_receipt_sha256 == expected_receipt_sha256
    assert first.descriptor_json == restarted.descriptor_json

    _save_clip(_store(path), request_id="save-0-newer")
    assert _store(path).list_saved_media_outbox()[0].save_request_id == "save-z-first"


def test_saved_media_failure_is_visible_and_retry_is_due_and_idempotent(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    item = store.claim_due_saved_media(now)
    retry_at = now + timedelta(seconds=30)
    for _ in range(2):
        store.retry_saved_media(
            item.clip_id, item.claim_token, error_code="offline", next_attempt_at_utc=retry_at
        )
    assert store.list_postgame_clips("session-1")[0].save_status == "SAVE_FAILED_RETRYING"
    with pytest.raises(ValueError, match="lease"):
        store.retry_saved_media(
            item.clip_id, item.claim_token, error_code="changed", next_attempt_at_utc=retry_at
        )
    assert store.claim_due_saved_media(now + timedelta(seconds=29)) is None
    retry = store.claim_due_saved_media(retry_at)
    assert retry is not None and retry.attempts == 2
    assert store.list_postgame_clips("session-1")[0].save_status == "SAVED"


def test_saved_media_block_is_lease_fenced_idempotent_and_never_claimed_again(
    tmp_path: Path,
) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    item = store.claim_due_saved_media(now)
    assert item is not None and item.claim_token is not None

    store.block_saved_media(
        item.clip_id,
        item.claim_token,
        error_code="SYNC_IDENTITY_CONFLICT",
    )
    store.block_saved_media(
        item.clip_id,
        item.claim_token,
        error_code="SYNC_IDENTITY_CONFLICT",
    )

    blocked = store.list_saved_media_outbox()[0]
    assert blocked.status == "blocked"
    assert blocked.last_error_code == "SYNC_IDENTITY_CONFLICT"
    assert blocked.lease_until_utc is None
    assert store.claim_due_saved_media(now + timedelta(days=365)) is None
    with pytest.raises(ValueError, match="conflict"):
        store.block_saved_media(item.clip_id, item.claim_token, error_code="CHANGED")
    assert _save_clip(store).status == "SAVED"  # exact historical request replay is read-only
    with pytest.raises(ValueError, match="blocked"):
        _save_clip(store, request_id="save-new-after-block")
    with sqlite3.connect(store.path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM field_clip_save_requests").fetchone()[0]
        assert count == 1


@pytest.mark.parametrize(
    ("target", "mutate"),
    (
        (
            "receipt",
            lambda payload, _clip: {
                **payload,
                "status": "SAVE_FAILED_RETRYING",
            },
        ),
        (
            "receipt",
            lambda payload, clip: {
                **payload,
                "status": "SAVE_PENDING",
                "clip_id": clip,
            },
        ),
        (
            "receipt",
            lambda payload, _clip: {
                **payload,
                "status": "SAVED",
                "clip_id": "clip-" + "0" * 64,
            },
        ),
        ("shot", lambda payload, _clip: {**payload, "session_id": "other-session"}),
        ("shot", lambda payload, _clip: {**payload, "shot_id": "other-shot"}),
        ("shot", lambda payload, _clip: {**payload, "live_job_id": "other-job"}),
        (
            "shot",
            lambda payload, _clip: {
                **payload,
                "media": {
                    **payload["media"],
                    "source_kind": "development_replay",
                    "source_locator": "presentation.mp4",
                    "source_epoch_id": None,
                },
            },
        ),
    ),
)
def test_claim_quarantines_semantically_tampered_receipt_or_shot(
    tmp_path: Path, target: str, mutate
) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    with sqlite3.connect(store.path) as connection:
        clip_id, request_id = connection.execute(
            "SELECT clip_id,save_request_id FROM field_saved_media_outbox"
        ).fetchone()
        if target == "receipt":
            raw = connection.execute(
                "SELECT receipt_json FROM field_clip_save_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()[0]
            changed = mutate(json.loads(raw), clip_id)
            connection.execute(
                "UPDATE field_clip_save_requests SET receipt_json=? WHERE request_id=?",
                (json.dumps(changed, sort_keys=True, separators=(",", ":")), request_id),
            )
        else:
            raw = connection.execute(
                "SELECT payload_json FROM field_shots WHERE session_id='session-1' "
                "AND shot_id='shot-0001'"
            ).fetchone()[0]
            changed = mutate(json.loads(raw), clip_id)
            connection.execute(
                "UPDATE field_shots SET payload_json=? WHERE session_id='session-1' "
                "AND shot_id='shot-0001'",
                (json.dumps(changed, sort_keys=True, separators=(",", ":")),),
            )

    assert store.claim_due_saved_media(datetime(2026, 8, 26, 10, tzinfo=UTC)) is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status,last_error_code FROM field_saved_media_outbox WHERE clip_id=?",
            (clip_id,),
        ).fetchone() == ("blocked", "SYNC_PERSISTED_DESCRIPTOR_INVALID")
    assert store.list_postgame_clips("session-1")[0].save_status == "SAVE_BLOCKED"


def test_claim_quarantines_poison_row_and_continues_to_next_due_clip(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match("session-1", _match(1), expected_revision=0)
    for index in (1, 2):
        started_ms = 3_000 + (index - 1) * 4_000
        shot_id = f"shot-{index:04d}"
        epoch = f"epoch-{index}"
        started = replace(
            _shot(),
            shot_id=shot_id,
            sequence=index,
            started_ms=started_ms,
            media=ShotMediaRange(
                artifact_id="artifact-1",
                source_locator=f"usb:{epoch}",
                source_kind="usb_live",
                start_ms=started_ms,
                end_ms=started_ms + 1,
                source_epoch_id=epoch,
            ),
            limitations=(),
        )
        store.record_shot_start(started, source_start_event_id=f"start-{index}")
        store.record_shot_settle(
            replace(
                started,
                settled_ms=started_ms + 2_000,
                media=replace(started.media, end_ms=started_ms + 2_000),
                live_evidence_digest=str(index) * 64,
                live_job_id=f"job-{index}",
            ),
            source_epoch_id=epoch,
        )
        store.attach_live_shot_media(
            LiveShotMediaReference(
                "session-1",
                shot_id,
                epoch,
                1,
                f"job-{index}",
                str(tmp_path / f"usb-{index}"),
                "d" * 64,
            )
        )
    store.end_session("session-1")
    digest = store.get_postgame_manifest("session-1").manifest_sha256
    for index in (1, 2):
        shot_id = f"shot-{index:04d}"
        store.save_postgame_clip(
            request_id=f"save-{index}",
            session_id="session-1",
            shot_id=shot_id,
            postgame_manifest_sha256=digest,
        )
        store.record_postgame_clip_available(
            session_id="session-1",
            shot_id=shot_id,
            postgame_manifest_sha256=digest,
            job_id=f"job-{index}",
            artifact_id=str(index + 1) * 64,
            local_path=(tmp_path / f"clip-{index}.mp4").resolve(),
            content_sha256=str(index + 2) * 64,
            byte_size=100 + index,
        )
    with sqlite3.connect(path) as connection:
        poison_clip, poison_request = connection.execute(
            "SELECT clip_id,save_request_id FROM field_saved_media_outbox ORDER BY clip_id LIMIT 1"
        ).fetchone()
        connection.execute(
            "UPDATE field_clip_save_requests SET receipt_json='{}' WHERE request_id=?",
            (poison_request,),
        )

    claimed = store.claim_due_saved_media(datetime(2026, 8, 26, 10, tzinfo=UTC))
    assert claimed is not None and claimed.clip_id != poison_clip
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status,last_error_code FROM field_saved_media_outbox WHERE clip_id=?",
            (poison_clip,),
        ).fetchone() == ("blocked", "SYNC_PERSISTED_DESCRIPTOR_INVALID")


def test_saved_media_stale_owner_cannot_block_reclaimed_lease(tmp_path: Path) -> None:
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    first = store.claim_due_saved_media(now, lease_seconds=10)
    second = _store(store.path).claim_due_saved_media(now + timedelta(seconds=10))
    assert first is not None and second is not None
    assert first.claim_token != second.claim_token

    with pytest.raises(ValueError, match="lease"):
        store.block_saved_media(first.clip_id, first.claim_token, error_code="PERMANENT")
    _store(store.path).block_saved_media(
        second.clip_id,
        second.claim_token,
        error_code="PERMANENT",
    )
    assert _store(store.path).list_saved_media_outbox()[0].status == "blocked"


@pytest.mark.parametrize("state", ["pending", "leased", "acknowledged"])
def test_schema_15_migration_preserves_saved_media_and_adds_blocked_state(
    tmp_path: Path,
    state: str,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _available_clip(store, tmp_path)
    _save_clip(store)
    if state != "pending":
        claimed = store.claim_due_saved_media(datetime(2026, 8, 26, 10, tzinfo=UTC))
        assert claimed is not None and claimed.claim_token is not None
        if state == "acknowledged":
            store.acknowledge_saved_media(claimed.clip_id, claimed.claim_token)
    before = store.list_saved_media_outbox()[0]
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE field_owner_completion_outbox")
        connection.execute("DROP TABLE field_local_media_schedule_state")
        connection.execute("DROP TABLE field_upload_receipt_outbox")
        connection.execute("DROP TABLE field_saved_upload_requests")
        connection.execute("DROP TABLE field_owner_media_deletions")
        connection.execute("DROP TABLE field_managed_capture_epochs")
        connection.execute("DROP TABLE field_managed_resource_specs")
        connection.execute("DROP TABLE field_managed_open_requests")
        connection.execute("DROP TABLE field_local_media_discards")
        connection.execute("DROP TABLE IF EXISTS field_recording_ends")
        connection.execute("DROP TABLE IF EXISTS field_recording_evidence")
        connection.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        connection.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        connection.execute("PRAGMA user_version=15")

    migrated = _store(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 30
    after = migrated.list_saved_media_outbox()[0]
    assert (after.clip_id, after.status, after.attempts, after.claim_token) == (
        before.clip_id,
        state,
        before.attempts,
        before.claim_token,
    )
    assert (after.save_request_id, after.save_receipt_sha256) == (
        before.save_request_id,
        before.save_receipt_sha256,
    )


def test_schema_16_migration_binds_original_save_request_without_rewriting_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    _save_clip(store, request_id="save-z-first")
    _save_clip(store, request_id="save-a-later")
    _available_clip(store, tmp_path)
    with sqlite3.connect(path) as connection:
        receipt_json = connection.execute(
            "SELECT receipt_json FROM field_clip_save_requests WHERE request_id='save-z-first'"
        ).fetchone()[0]
        connection.executescript(
            """
            ALTER TABLE field_saved_media_outbox RENAME TO field_saved_media_outbox_v17;
            CREATE TABLE field_saved_media_outbox (
              clip_id TEXT PRIMARY KEY REFERENCES field_postgame_clip_availability(clip_id),
              status TEXT NOT NULL CHECK(status IN ('pending','leased','acknowledged','blocked')),
              attempts INTEGER NOT NULL CHECK(attempts >= 0),
              claim_token TEXT,
              lease_until_utc TEXT,
              next_attempt_at_utc TEXT NOT NULL,
              last_error_code TEXT
            );
            INSERT INTO field_saved_media_outbox(
              clip_id,status,attempts,claim_token,lease_until_utc,
              next_attempt_at_utc,last_error_code
            ) SELECT clip_id,status,attempts,claim_token,lease_until_utc,
              next_attempt_at_utc,last_error_code FROM field_saved_media_outbox_v17;
            DROP TABLE field_saved_media_outbox_v17;
            CREATE INDEX due_saved_media
              ON field_saved_media_outbox(status,next_attempt_at_utc);
            DROP TABLE field_owner_completion_outbox;
            DROP TABLE field_local_media_schedule_state;
            DROP TABLE field_upload_receipt_outbox;
            DROP TABLE field_saved_upload_requests;
            DROP TABLE field_owner_media_deletions;
            DROP TABLE field_managed_capture_epochs;
            DROP TABLE field_managed_resource_specs;
            DROP TABLE field_managed_open_requests;
            DROP TABLE field_local_media_discards;
            DROP TABLE IF EXISTS field_recording_ends;
            DROP TABLE IF EXISTS field_recording_evidence;
            DROP TABLE IF EXISTS field_session_recording_policy;
            DROP TABLE IF EXISTS field_rule_fact_preimages;
            PRAGMA user_version=16;
            """
        )

    migrated = _store(path)
    item = migrated.list_saved_media_outbox()[0]
    assert item.save_request_id == "save-z-first"
    assert item.save_receipt_sha256 == hashlib.sha256(receipt_json.encode()).hexdigest()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 30


def test_schema_13_migration_preserves_frozen_session_and_finalization_bytes(tmp_path: Path):
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    frozen_tables = (
        "field_sessions",
        "field_shots",
        "field_postgame_manifests",
        "field_postgame_finalizations",
        "local_postgame_outbox",
    )
    with sqlite3.connect(path) as connection:
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in frozen_tables
        }
        for table in (
            "field_postgame_reviews",
            "field_saved_media_outbox",
            "field_clip_save_requests",
            "field_postgame_clip_availability",
        ):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("DROP TABLE field_owner_completion_outbox")
        connection.execute("DROP TABLE field_local_media_schedule_state")
        connection.execute("DROP TABLE field_upload_receipt_outbox")
        connection.execute("DROP TABLE field_saved_upload_requests")
        connection.execute("DROP TABLE field_owner_media_deletions")
        connection.execute("DROP TABLE field_managed_capture_epochs")
        connection.execute("DROP TABLE field_managed_resource_specs")
        connection.execute("DROP TABLE field_managed_open_requests")
        connection.execute("DROP TABLE field_local_media_discards")
        connection.execute("DROP TABLE IF EXISTS field_recording_ends")
        connection.execute("DROP TABLE IF EXISTS field_recording_evidence")
        connection.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        connection.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        connection.execute("PRAGMA user_version=13")
    migrated = _store(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] > 13
        assert {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in frozen_tables
        } == before
    assert _save_clip(migrated).status == "SAVE_PENDING"


def test_review_queue_survives_restart_and_activates_next_ended_session(tmp_path: Path):
    path = tmp_path / "field.sqlite3"
    first = _postgame_store(path)
    manifest = first.get_postgame_manifest("session-1")
    assert first.active_postgame_review("artifact-1") == manifest
    first.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    _available_clip(first, tmp_path)
    second = _store(path, session_id="session-2")
    second.open_or_resume_session(artifact_id="artifact-1")
    second.end_session("session-2")
    restarted = _store(path, session_id="session-3")
    restarted.open_or_resume_session(artifact_id="artifact-1")
    assert restarted.active_postgame_review("artifact-1") == manifest
    with pytest.raises(ValueError, match="active"):
        restarted.complete_postgame_review("artifact-1", "session-2", manifest.manifest_sha256)
    with pytest.raises(ValueError, match="digest"):
        restarted.complete_postgame_review("artifact-1", "session-1", "a" * 64)
    restarted.complete_postgame_review("artifact-1", "session-1", manifest.manifest_sha256)
    next_manifest = restarted.active_postgame_review("artifact-1")
    assert next_manifest.session_id == "session-2"
    restarted.complete_postgame_review("artifact-1", "session-2", next_manifest.manifest_sha256)
    assert _store(path).active_postgame_review("artifact-1") is None
    assert restarted.get_postgame_manifest("session-1") == manifest


def test_schema_14_migration_backfills_review_without_rewriting_frozen_rows(tmp_path: Path):
    path = tmp_path / "field.sqlite3"
    store = _postgame_store(path)
    manifest = store.get_postgame_manifest("session-1")
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT * FROM field_sessions").fetchall()
        connection.execute("DROP TABLE IF EXISTS field_postgame_reviews")
        connection.execute("DROP TABLE field_owner_completion_outbox")
        connection.execute("DROP TABLE field_local_media_schedule_state")
        connection.execute("DROP TABLE field_upload_receipt_outbox")
        connection.execute("DROP TABLE field_saved_upload_requests")
        connection.execute("DROP TABLE field_owner_media_deletions")
        connection.execute("DROP TABLE field_managed_capture_epochs")
        connection.execute("DROP TABLE field_managed_resource_specs")
        connection.execute("DROP TABLE field_managed_open_requests")
        connection.execute("DROP TABLE field_local_media_discards")
        connection.execute("DROP TABLE IF EXISTS field_recording_ends")
        connection.execute("DROP TABLE IF EXISTS field_recording_evidence")
        connection.execute("DROP TABLE IF EXISTS field_session_recording_policy")
        connection.execute("DROP TABLE IF EXISTS field_rule_fact_preimages")
        connection.execute("PRAGMA user_version=14")
    migrated = _store(path)
    assert migrated.active_postgame_review("artifact-1") == manifest
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 30
        assert connection.execute("SELECT * FROM field_sessions").fetchall() == before


def test_review_queue_insert_failure_rolls_back_session_end_and_manifest(tmp_path: Path):
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_review BEFORE INSERT ON field_postgame_reviews "
            "BEGIN SELECT RAISE(ABORT, 'queue unavailable'); END"
        )
    with pytest.raises(ValueError, match="database"):
        store.end_session("session-1")
    assert store.load_session("session-1").status.value == "open"
    assert store.get_postgame_manifest("session-1") is None
    assert store.active_postgame_review("artifact-1") is None


def test_saved_media_expired_lease_cannot_ack_or_retry_without_reclaim(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    item = store.claim_due_saved_media(now, lease_seconds=10)
    expired = SQLiteFieldRecordStore(store.path, now_utc=lambda: now + timedelta(seconds=10))
    with pytest.raises(ValueError, match="lease"):
        expired.acknowledge_saved_media(item.clip_id, item.claim_token)
    with pytest.raises(ValueError, match="lease"):
        expired.retry_saved_media(item.clip_id, item.claim_token, error_code="offline")


def test_saved_media_retry_without_explicit_deadline_replays_after_clock_advances(tmp_path: Path):
    store = _postgame_store(tmp_path / "field.sqlite3")
    _available_clip(store, tmp_path)
    _save_clip(store)
    now = datetime(2026, 8, 26, 10, tzinfo=UTC)
    item = store.claim_due_saved_media(now)
    store.retry_saved_media(item.clip_id, item.claim_token, error_code="offline")
    later = SQLiteFieldRecordStore(store.path, now_utc=lambda: now + timedelta(seconds=1))
    later.retry_saved_media(item.clip_id, item.claim_token, error_code="offline")
    assert datetime.fromisoformat(later.list_saved_media_outbox()[0].next_attempt_at_utc) == now


def test_clip_save_and_availability_reject_open_session_and_unexpected_replay_shot(tmp_path: Path):
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.record_shot_start(_shot())
    store.record_shot_settle(_shot(settled_ms=5_000))
    with pytest.raises(ValueError, match="ended"):
        store.save_postgame_clip(
            request_id="save-open",
            session_id="session-1",
            shot_id="shot-0001",
            postgame_manifest_sha256="a" * 64,
        )
    store.end_session("session-1")
    with pytest.raises(ValueError, match="frozen"):
        _save_clip(store)
    with pytest.raises(ValueError, match="frozen"):
        _available_clip(store, tmp_path)
    assert store.list_saved_media_outbox() == ()


def test_duplicate_field_command_is_not_reduced_twice_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "field.sqlite3"
    store = _store(path)
    store.open_or_resume_session(artifact_id="artifact-1")

    first = store.apply_field_command(_command("cmd-1"), reduce_field_command)
    restarted = _store(path, session_id="unused")
    second = restarted.apply_field_command(_command("cmd-1"), reduce_field_command)

    assert second == first
    assert second.outcome == "applied"
    assert restarted.load_session("session-1").match.revision == 1


def test_stale_field_command_is_recorded_without_changing_revision(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    session = store.open_or_resume_session(artifact_id="artifact-1")
    store.save_match(session.session_id, _match(2), expected_revision=0)

    result = store.apply_field_command(_command("cmd-stale", revision=0), reduce_field_command)

    assert result.outcome == "rejected"
    assert result.result_code == "STALE_REVISION"
    assert result.resulting_revision == 2
    assert store.load_session("session-1").match == _match(2)


def test_field_commands_must_arrive_in_server_sequence_order(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")

    result = store.apply_field_command(_command("cmd-2", sequence=2), reduce_field_command)

    assert result.outcome == "rejected"
    assert result.result_code == "COMMAND_OUT_OF_ORDER"
    assert store.load_session("session-1").match.revision == 0


def test_command_id_reuse_with_changed_intent_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")
    store.apply_field_command(_command("cmd-1"), reduce_field_command)

    conflict = store.apply_field_command(
        _command(
            "cmd-1",
            command_type="set_breaker",
            payload={"role": "red"},
            revision=1,
        ),
        reduce_field_command,
    )

    assert conflict.outcome == "rejected"
    assert conflict.result_code == "COMMAND_IDEMPOTENCY_CONFLICT"
    assert store.load_session("session-1").match.revision == 1


def test_admin_override_persists_reason_with_match_mutation(tmp_path: Path) -> None:
    store = _store(tmp_path / "field.sqlite3")
    store.open_or_resume_session(artifact_id="artifact-1")

    session = store.apply_admin_override(
        "session-1",
        _match(1),
        expected_revision=0,
        command_type="emergency_set_breaker",
        reason="现场手机暂时无法联网",
        payload={"role": "red"},
    )

    assert session.match == _match(1)
    assert store.list_admin_audit("session-1") == (
        {
            "command_type": "emergency_set_breaker",
            "reason": "现场手机暂时无法联网",
            "payload": {"role": "red"},
            "resulting_revision": 1,
            "created_at_utc": "2026-08-26T10:00:00Z",
        },
    )
