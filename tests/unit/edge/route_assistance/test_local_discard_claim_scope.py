"""Real SQLite authority and physical disposal across configured resource roots."""

import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import LiveShotMediaReference, ShotMediaRange
from tests.unit.edge.route_assistance.test_local_media_cleanup import NOW, sha
from tests.unit.edge.route_assistance.test_local_media_cleanup import media as media
from tests.unit.edge.route_assistance.test_persistence import (
    _match,
    _postgame_store,
    _shot,
    _store,
)


def authorize(store, session, artifact, job, overlay):
    manifest = store.get_postgame_manifest(session)
    store.record_postgame_clip_available(
        session_id=session,
        shot_id="shot-0001",
        job_id=job,
        artifact_id="a" * 64,
        postgame_manifest_sha256=manifest.manifest_sha256,
        local_path=overlay,
        content_sha256=sha(overlay.read_bytes()),
        byte_size=overlay.stat().st_size,
        content_type="video/mp4",
    )
    store.record_postgame_finalization(session, '{"clips":[],"report":{}}')
    store.complete_postgame_review(artifact, session, manifest.manifest_sha256)


@pytest.fixture
def retained(media):
    _, db, root, jobs, overlay, raw = media
    store = _postgame_store(root.parent / "field.sqlite3")
    store.bind_artifact_path("artifact-1", root)
    authorize(store, "session-1", "artifact-1", "job-1", overlay)
    return store, db, root, jobs, overlay, raw


def row(store, session="session-1"):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(
            "SELECT * FROM field_local_media_discards WHERE session_id=?", (session,)
        ).fetchone()


def collect(retained, **kwargs):
    return process_local_discard_once(*retained[:4], now_utc=NOW, **kwargs)


def assert_removed(retained):
    assert not retained[4].exists()
    assert not list(retained[5].glob("*.mjpeg"))
    assert (retained[5] / "interval.json").is_file()
    assert (retained[4].parent / "summary.json").read_bytes() == b"{}"


@pytest.mark.parametrize("reopen_alias", [False, True])
def test_wrong_root_cannot_reclaim_expired_live_detach(retained, reopen_alias):
    store, db, root, jobs, _, raw = retained
    foreign = root.parent / "foreign"
    foreign.mkdir()
    sentinel = foreign / "saved.mp4"
    sentinel.write_bytes(b"foreign-saved-media")
    if reopen_alias:
        alias = store.path.with_name("field-alias.sqlite3")
        alias.symlink_to(store.path)
        competitor = SQLiteFieldRecordStore(alias)
        assert competitor.path == store.path
    else:
        competitor = store
    attempts, snapshots = [], []

    def compete(point, obj):
        if point == "after_detach" and not attempts:
            before = row(store)
            detached = next(root.rglob("*.detached"))
            detached_bytes = detached.read_bytes()
            raw_bytes = (raw / "segment-1.mjpeg").read_bytes()
            attempts.append(
                process_local_discard_once(
                    competitor, db, foreign, jobs, now_utc=NOW + timedelta(seconds=2)
                )
            )
            snapshots.append((before, row(store)))
            assert detached.read_bytes() == detached_bytes
            assert (raw / "segment-1.mjpeg").read_bytes() == raw_bytes

    original = collect(retained, lease_seconds=1, fault_hook=compete)
    assert attempts[0].outcome == "retry", (attempts, original, snapshots)
    assert attempts[0].error_code == "collector_control_unavailable"
    assert snapshots[0][0] == snapshots[0][1]
    assert original.outcome == "completed"
    assert_removed(retained)
    assert sentinel.read_bytes() == b"foreign-saved-media"
    assert list(foreign.iterdir()) == [sentinel]


@pytest.mark.parametrize("expired_lease", [False, True])
def test_idle_wrong_root_preserves_pending_or_expired_row(retained, expired_lease):
    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        if point == "after_detach":
            raise PowerLoss()

    store, db, root, jobs, *_ = retained
    foreign = root.parent / "foreign"
    foreign.mkdir()
    sentinel = foreign / "saved.mp4"
    sentinel.write_bytes(b"foreign-saved-media")
    if expired_lease:
        with pytest.raises(PowerLoss):
            collect(retained, lease_seconds=1, fault_hook=crash)
    before = row(store)
    result = process_local_discard_once(
        store, db, foreign, jobs, now_utc=NOW + timedelta(seconds=2)
    )
    assert result.outcome == "idle", result
    assert row(store) == before
    assert sentinel.read_bytes() == b"foreign-saved-media"
    result = process_local_discard_once(store, db, root, jobs, now_utc=NOW + timedelta(seconds=2))
    assert result.outcome == "completed"
    assert_removed(retained)


def test_missing_retained_root_is_not_claimed(media):
    _, db, root, jobs, overlay, _ = media
    store = _postgame_store(root.parent / "field.sqlite3")
    authorize(store, "session-1", "artifact-1", "job-1", overlay)
    before = row(store)
    assert process_local_discard_once(store, db, root, jobs, now_utc=NOW).outcome == "idle"
    assert row(store) == before
    assert overlay.read_bytes() == b"synthetic-overlay"
    store.bind_artifact_path("artifact-1", root)
    retained = (store, *media[1:])
    assert collect(retained).outcome == "completed"
    assert_removed(retained)


def test_earlier_foreign_root_does_not_starve_eligible_discard(retained):
    store, _, root, _, *_ = retained
    foreign = root.parent / "foreign"
    foreign.mkdir()
    overlay = foreign / "overlay.mp4"
    overlay.write_bytes(b"foreign-unsaved-media")
    earlier = _store(store.path, session_id="session-0")
    earlier.open_or_resume_session(artifact_id="artifact-0")
    earlier.save_match("session-0", _match(1), expected_revision=0)
    started = replace(
        _shot(),
        session_id="session-0",
        limitations=(),
        media=ShotMediaRange(
            artifact_id="artifact-0",
            source_locator="usb:epoch-0",
            source_kind="usb_live",
            start_ms=3_000,
            end_ms=3_001,
            source_epoch_id="epoch-0",
        ),
    )
    earlier.record_shot_start(started, source_start_event_id="start-0")
    earlier.record_shot_settle(
        replace(
            started,
            settled_ms=5_000,
            media=replace(started.media, end_ms=5_000),
            live_evidence_digest="e" * 64,
            live_job_id="job-0",
        ),
        source_epoch_id="epoch-0",
    )
    earlier.attach_live_shot_media(
        LiveShotMediaReference(
            "session-0", "shot-0001", "epoch-0", 1, "job-0", str(foreign), "d" * 64
        )
    )
    earlier.end_session("session-0")
    earlier.bind_artifact_path("artifact-0", foreign)
    authorize(earlier, "session-0", "artifact-0", "job-0", overlay)
    before = row(store, "session-0")
    result = collect(retained)
    assert result.outcome == "completed" and result.session_id == "session-1", result
    assert row(store, "session-0") == before
    assert overlay.read_bytes() == b"foreign-unsaved-media"
    assert_removed(retained)


def test_distinct_databases_in_same_parent_have_independent_claim_pins(retained):
    store, db, root, jobs, *_ = retained
    independent = SQLiteFieldRecordStore(store.path.with_name("independent.sqlite3"))
    attempts = []

    def compete(point, obj):
        if point == "after_detach" and not attempts:
            attempts.append(process_local_discard_once(independent, db, root, jobs, now_utc=NOW))

    assert collect(retained, fault_hook=compete).outcome == "completed"
    assert attempts[0].outcome == "idle"
    assert_removed(retained)
