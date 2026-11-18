"""Real bounded subprocess exclusion, independent of SQLite lease and media root."""
# Imported pytest fixtures are injected by their public names.
# ruff: noqa: F811

import multiprocessing
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestLedger
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import media  # noqa: F401
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import NOW, SCOPE
from tests.unit.edge.route_assistance.test_saved_upload_requests import (
    attempt,
    scene,  # noqa: F401
)


def compete(connection, database, item, foreign_root):
    """The connection is synchronization: one command/response, no sleep loop."""
    from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestError

    store = SQLiteFieldRecordStore(Path(database), now_utc=lambda: NOW)
    item = replace(item, local_path=Path(foreign_root) / "overlay.mp4")
    original = store._connect

    def forbidden_sql():
        raise AssertionError("contender reached SQL before exclusion")

    store._connect = forbidden_sql
    try:
        with SavedUploadRequestLedger(store, SCOPE).use(item):
            connection.send(("unexpected", False))
    except SavedUploadRequestError as exc:
        connection.send((exc.code, exc.retryable))
    finally:
        store._connect = original
        connection.close()


@pytest.mark.parametrize("alias", [False, True])
def test_subprocess_cannot_steal_expired_use_or_mutate_sql_through_different_root(
    scene, tmp_path, alias
):
    from datetime import timedelta

    store, item = scene
    path = store.path
    if alias:
        path = tmp_path / "alias.db"
        path.symlink_to(store.path)
    foreign = tmp_path / "foreign-media-root"
    foreign.mkdir()
    (foreign / "sentinel").write_bytes(b"untouched synthetic bytes")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = None
    try:
        with SavedUploadRequestLedger(store, SCOPE).use(item) as session:
            row = session.load_or_create("cloud-job-1")
            session.record_received(row.request_id, attempt())
            session.record_started(row.request_id)
            competitor = SQLiteFieldRecordStore(path, now_utc=lambda: NOW + timedelta(seconds=2))
            replacement = competitor.claim_due_saved_media(
                NOW + timedelta(seconds=2),
                managed_scope=SCOPE,
            )
            assert replacement.claim_token != item.claim_token
            with sqlite3.connect(store.path) as db:
                before = db.execute("SELECT * FROM field_saved_upload_requests").fetchall()
            process = context.Process(
                target=compete, args=(child, str(path), replacement, str(foreign))
            )
            process.start()
            child.close()
            assert parent.poll(10), "bounded child response missing"
            assert parent.recv() == ("control_unavailable", True)
            process.join(10)
            assert process.exitcode == 0
            with sqlite3.connect(store.path) as db:
                assert db.execute("SELECT * FROM field_saved_upload_requests").fetchall() == before
            success = session.record_succeeded(row.request_id, 200)
            assert success.state == "succeeded"
        with SavedUploadRequestLedger(competitor, SCOPE).use(replacement) as resumed:
            assert resumed.current() == success
        assert (foreign / "sentinel").read_bytes() == b"untouched synthetic bytes"
        assert not (foreign / ".saved-upload-request-control").exists()
        control = store.path.parent / ".saved-upload-request-control"
        assert len(list(control.rglob("*.lock"))) == 1
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()


def test_distinct_database_and_clip_independence_and_exception_release(scene, tmp_path):
    store, item = scene
    clone = tmp_path / "other.db"
    with sqlite3.connect(store.path) as db, sqlite3.connect(clone) as other:
        db.backup(other)
    other_store = SQLiteFieldRecordStore(clone, now_utc=lambda: NOW)
    other_clip = next(row for row in store.list_saved_media_outbox() if row.clip_id != item.clip_id)
    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        with SavedUploadRequestLedger(store, SCOPE).use(item):
            with SavedUploadRequestLedger(other_store, SCOPE).use(item) as independent:
                assert independent.current() is None
            # Historical reads remain valid under the original owner's deletion fence.
            with SavedUploadRequestLedger(store, SCOPE).use(other_clip) as independent:
                assert independent.current() is None
            raise RuntimeError("synthetic cancellation")
    with SavedUploadRequestLedger(store, SCOPE).use(item) as session:
        assert session.load_or_create("cloud-job-1").state == "requested"


@pytest.mark.parametrize("unsafe", ["symlink_control", "symlink_lock", "hardlink_lock"])
def test_unsafe_control_chain_or_file_has_no_unlocked_fallback(scene, tmp_path, unsafe):
    from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestError

    store, item = scene
    control = store.path.parent / ".saved-upload-request-control"
    if unsafe == "symlink_control":
        target = tmp_path / "foreign-control"
        target.mkdir()
        control.symlink_to(target, target_is_directory=True)
    else:
        with SavedUploadRequestLedger(store, SCOPE).use(item):
            pass
        lock = next(control.rglob("*.lock"))
        target = tmp_path / "foreign-lock"
        if unsafe == "symlink_lock":
            target.write_bytes(b"")
            lock.unlink()
            lock.symlink_to(target)
        else:
            import os

            os.link(lock, target)
    with pytest.raises(SavedUploadRequestError, match="control_unavailable") as error:
        with SavedUploadRequestLedger(store, SCOPE).use(item):
            pytest.fail("unsafe lock must fail closed")
    assert error.value.retryable
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM field_saved_upload_requests").fetchone()[0] == 0
