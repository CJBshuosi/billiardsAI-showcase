"""Owner tombstones close new review/upload work using temporary SQLite state."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.postgame import ClipSaveStatus
from billiards_ai.lab.field_assistance import cloud_runtime, server
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import (
    media as media,
)
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import (
    run as erase_owner,
)
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    SESSION,
    accept,
    payload,
)
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    scene as scene,
)


def state_for(store, root):
    return server._ServerState(
        artifact_path=root,
        control_token="token",
        replay_clock=object(),
        match=store.load_session(SESSION).match,
        match_lock=threading.RLock(),
        frame_store=object(),
        record_store=store,
        session_id=SESSION,
        artifact_id=store.load_session(SESSION).artifact_id,
        shot_index=object(),
        recommendation_power={},
        duration_ms=1,
        managed_session_scope=SCOPE,
    )


@pytest.mark.parametrize("status", ["pending", "leased", "retry", "blocked"])
def test_owner_tombstone_retains_history_but_closes_projection_and_reader(scene, tmp_path, status):
    # Catches projecting the fenced outbox as an ordinary playable saved clip.
    deletion = accept(scene)
    with scene._connect() as db:
        db.execute("UPDATE field_owner_media_deletions SET status=?", (status,))
    scene.bind_artifact_path("artifact-1", tmp_path)
    state = state_for(scene, tmp_path)
    clips = scene.list_postgame_clips(SESSION)
    assert len(clips) == 2
    assert clips[0].owner_delete_status == "pending"
    assert clips[0].local_path is None
    assert clips[0].can_save is False
    assert clips[0].save_status in ClipSaveStatus
    assert clips[1].owner_delete_status is None
    assert clips[1].can_save is True and clips[1].local_path is not None
    payload = state.postgame_review_payload(kiosk=True)
    assert (payload["expected_count"], payload["available_count"]) == (2, 2)
    saved = state.mutate_postgame_review(
        {
            "request_id": "save-opponent-after-owner-delete",
            "session_id": SESSION,
            "shot_id": "shot-0002",
            "postgame_manifest_sha256": deletion.postgame_manifest_sha256,
        },
        payload["kiosk_nonce"],
        complete=False,
    )
    assert saved["status"] == "SAVED"
    row = payload["clips"][0]
    assert row["owner_delete_status"] == "pending" and row["media_url"] is None
    assert "local_path" not in row and "receipt_json" not in row
    with session_media_lock(tmp_path, SESSION, exclusive=False):
        with pytest.raises(ValueError, match="unavailable"):
            state.review_clip(deletion.clip_id, deletion.postgame_manifest_sha256)
        assert (
            state.review_clip(clips[1].clip_id, deletion.postgame_manifest_sha256).shot_id
            == "shot-0002"
        )


def test_http_media_rechecks_owner_fence_after_acquiring_original_pin(scene, tmp_path, monkeypatch):
    scene.bind_artifact_path("artifact-1", tmp_path)
    state = state_for(scene, tmp_path)
    clip = scene.list_postgame_clips(SESSION)[0]

    @contextmanager
    def pin_then_delete(*args, **kwargs):
        with session_media_lock(*args, **kwargs):
            accept(scene)
            yield

    monkeypatch.setattr(server, "session_media_lock", pin_then_delete)
    monkeypatch.setattr(
        server,
        "_open_review_media",
        lambda *_: pytest.fail("owner tombstone bypassed before media open"),
    )
    handler = object.__new__(server._FieldAssistanceHandler)
    handler.server = SimpleNamespace(field_state=state)
    errors = []
    handler._error = lambda *args: errors.append(args)
    handler._stream_media = lambda *_: pytest.fail("owner-deleted media streamed")
    handler._serve_postgame_media(clip.clip_id, "manifest=" + clip.postgame_manifest_sha256)
    assert errors == [(HTTPStatus.NOT_FOUND, "review media is unavailable")]


def test_saved_upload_claim_waiting_for_pin_cannot_call_client_after_delete(
    scene, tmp_path, monkeypatch
):
    # The saved claim is real and is invalidated while the worker obtains its pin.
    scene.bind_artifact_path("artifact-1", tmp_path)
    claimed = [scene.claim_due_saved_media(NOW, managed_scope=SCOPE) for _ in range(2)]
    selected = next(row for row in claimed if row.shot_id == "shot-0001")
    monkeypatch.setattr(scene, "claim_due_saved_media", lambda *args, **kwargs: selected)
    calls = []

    class Client:
        def run_saved_media_once(self, item, **kwargs):
            calls.append(item.clip_id)
            return SimpleNamespace(outcome="complete")

    @contextmanager
    def pin_then_delete(*args, **kwargs):
        with session_media_lock(*args, **kwargs):
            accept(scene)
            yield

    monkeypatch.setattr(cloud_runtime, "session_media_lock", pin_then_delete)
    worker = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=scene,
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        now_utc=lambda: NOW,
        managed_scope=SCOPE,
    )
    assert worker.run_once() is True
    assert calls == []
    item = next(row for row in scene.list_saved_media_outbox() if row.shot_id == "shot-0001")
    assert (item.status, item.claim_token, item.last_error_code) == (
        "blocked",
        None,
        "OWNER_CLIP_DELETE_REQUESTED",
    )
    assert scene.list_postgame_clips(SESSION)[1].can_save is True


def test_pinned_upload_already_active_can_finish_but_stale_ack_does_not_revive(
    scene, tmp_path, monkeypatch
):
    scene.bind_artifact_path("artifact-1", tmp_path)
    claimed = [scene.claim_due_saved_media(NOW, managed_scope=SCOPE) for _ in range(2)]
    selected = next(row for row in claimed if row.shot_id == "shot-0001")
    monkeypatch.setattr(scene, "claim_due_saved_media", lambda *args, **kwargs: selected)
    calls = []

    class Client:
        def run_saved_media_once(self, item, **kwargs):
            with pytest.raises(TimeoutError):
                with session_media_lock(tmp_path, SESSION, exclusive=True, timeout_seconds=0):
                    pass
            accept(scene)
            calls.append(item.clip_id)
            return SimpleNamespace(outcome="complete")

    worker = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=scene,
        client=Client(),
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
        now_utc=lambda: NOW,
        managed_scope=SCOPE,
    )
    assert worker.run_once() is True
    assert len(calls) == 1
    assert (
        next(
            row for row in scene.list_saved_media_outbox() if row.shot_id == "shot-0001"
        ).last_error_code
        == "OWNER_CLIP_DELETE_REQUESTED"
    )


def test_physical_completion_projects_completed_without_exposing_erase_receipt(media):
    store, _, root, _, *_ = media
    assert erase_owner(media).outcome == "completed"
    state = state_for(store, root)
    clip = store.list_postgame_clips(SESSION)[0]
    assert clip.owner_delete_status == "completed"
    assert clip.local_path is None and clip.can_save is False
    row = state.postgame_review_payload(kiosk=True)["clips"][0]
    assert row["owner_delete_status"] == "completed" and row["media_url"] is None
    assert set(row) == {
        "shot_id",
        "sequence",
        "actor_role",
        "can_save",
        "ownership_unavailable_reason",
        "clip_id",
        "save_status",
        "owner_delete_status",
        "media_url",
        "playback_clip_id",
        "encoding_status",
        "terminal_failure",
        "error_code",
    }

    assert row["playback_clip_id"] == clip.clip_id
    assert row["encoding_status"] == "NOT_SCHEDULED"
    assert row["terminal_failure"] is False and row["error_code"] is None


def test_active_http_reader_can_finish_after_owner_acceptance_inside_stream(media):
    from billiards_ai.edge.route_assistance.owner_media_deletion import OwnerDeleteDeviceTask

    store, _, root, _, *_ = media
    state = state_for(store, root)
    clip = store.list_postgame_clips(SESSION)[1]
    handler = object.__new__(server._FieldAssistanceHandler)
    handler.server = SimpleNamespace(field_state=state)
    streamed = []

    def stream(source, size, *, continue_check):
        store.accept_owner_delete_task(
            OwnerDeleteDeviceTask.from_payload(payload(store, "shot-0002")), managed_scope=SCOPE
        )
        assert erase_owner(media).outcome == "retry"
        streamed.append(source.read())
        assert len(streamed[0]) == size

    handler._stream_media = stream
    handler._error = lambda *args: pytest.fail("active pinned reader refused")
    handler._serve_postgame_media(clip.clip_id, "manifest=" + clip.postgame_manifest_sha256)
    assert streamed == [b"synthetic-overlay-2"]
    with pytest.raises(ValueError, match="unavailable"):
        state.review_clip(clip.clip_id, clip.postgame_manifest_sha256)


def test_unscoped_saved_worker_refuses_managed_identity_without_media_or_client_access(
    scene, tmp_path
):
    calls = []

    class Client:
        def run_saved_media_once(self, item, **kwargs):
            calls.append(item.clip_id)
            return SimpleNamespace(outcome="complete")

    worker = cloud_runtime._BackgroundSavedMediaSyncWorker(
        store=scene,
        client=Client(),
        trusted_postgame_root=tmp_path / "wrong-current-root",
        wake_event=threading.Event(),
        now_utc=lambda: NOW,
    )
    assert worker.run_once() is True
    assert calls == []
    assert worker._recovery_diagnostics[SESSION] == "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
    assert any(
        row.last_error_code == "SAVED_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE"
        and row.status == "pending"
        for row in scene.list_saved_media_outbox()
    )
