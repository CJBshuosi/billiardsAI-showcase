from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.postgame import LocalDiscardClaim, session_scoped_clip_id
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    ShotAnalysisJob,
    ShotMediaEncodeJobV1,
    _media_job_payload,
)

NOW = datetime(2026, 10, 29, tzinfo=UTC)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Store:
    """Durable checkpoint contract double; media operations are real temporary files."""

    def __init__(self, root, claim):
        self.root, self.item = root, claim
        self.path = (root.parent / "field.sqlite3").resolve()
        self.status, self.progress, self.error = "pending", "{}", None
        self.checkpoints = []
        self.fence = True

    def artifact_path_for_session(self, session):
        assert session == self.item.session_id
        return self.root

    def claim_due_local_discard(self, worker_id, now, *, artifact_root=None, **kwargs):
        if artifact_root is not None and artifact_root.resolve() != self.root:
            return None
        if self.status in {"completed", "blocked"}:
            return None
        self.status = "leased"
        self.item = replace(self.item, progress_json=self.progress)
        return self.item

    def update_local_discard_progress(self, claim, progress_json):
        if not self.fence:
            return False
        self.progress = progress_json
        self.checkpoints.append(json.loads(progress_json))
        return True

    def complete_local_discard(self, claim, *, progress_json=None):
        return self.finish("completed", None, progress_json)

    def retry_local_discard(self, claim, error_code, next_attempt_at_utc, *, progress_json=None):
        return self.finish("pending", error_code, progress_json)

    def block_local_discard(self, claim, error_code, *, progress_json=None):
        return self.finish("blocked", error_code, progress_json)

    def finish(self, status, error, progress):
        if not self.fence:
            return False
        self.status, self.error = status, error
        if progress is not None:
            self.progress = progress
        return True


@pytest.fixture
def media(tmp_path):
    root, jobs = tmp_path / "artifacts", tmp_path / "jobs"
    session, job_id, shot, artifact = "session-1", "job-1", "shot-0001", "a" * 64
    output = root / "postgame" / session / artifact
    output.mkdir(parents=True)
    overlay = output / "overlay.mp4"
    overlay.write_bytes(b"synthetic-overlay")
    raw = jobs / "media" / session / job_id
    raw.mkdir(parents=True)
    segments = {"segment-1.mjpeg": b"raw-one", "segment-2.mjpeg": b"raw-two"}
    for name, data in segments.items():
        (raw / name).write_bytes(data)
    identity = dict(
        schema_version="1.0",
        source_epoch_id="epoch",
        width=640,
        height=480,
        start_frame_id=1,
        end_frame_id=2,
        frame_count=2,
    )
    identity["segments"] = [{"name": name, "sha256": sha(data)} for name, data in segments.items()]
    digest = sha(canonical(identity).encode())
    interval = {key: value for key, value in identity.items() if key != "segments"}
    interval.update(
        media_sha256=digest, segment_sha256={name: sha(data) for name, data in segments.items()}
    )
    (raw / "interval.json").write_text(canonical(interval))
    job = ShotAnalysisJob(
        job_id, session, shot, "epoch", 1, 2, raw, digest, "setup", "calibration", "b" * 64, "model"
    )
    payload = canonical(job.identity())
    db = jobs / "analysis-jobs.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE shot_analysis_jobs("
            "job_id, payload_json, payload_sha256, status, artifact_id)"
        )
        connection.execute(
            "INSERT INTO shot_analysis_jobs VALUES(?,?,?,?,?)",
            (job_id, payload, sha(payload.encode()), "available", artifact),
        )
    manifest = dict(job.identity())
    manifest.update(
        kind="mainline1_shot_local_video",
        artifact_role="usb_postgame_shot",
        source_kind="usb_live",
        live_shot_id=shot,
    )
    manifest.update(
        artifact_id=artifact,
        inventory={
            "overlay.mp4": {"bytes": overlay.stat().st_size, "sha256": sha(overlay.read_bytes())},
            "summary.json": {"bytes": 2, "sha256": sha(b"{}")},
        },
    )
    (output / "manifest.json").write_text(canonical(manifest))
    (output / "summary.json").write_text("{}")
    claim = LocalDiscardClaim(
        session,
        shot,
        "c" * 64,
        job_id,
        artifact,
        session_scoped_clip_id(session, shot, job_id, artifact),
        overlay,
        sha(overlay.read_bytes()),
        overlay.stat().st_size,
        "video/mp4",
        1,
        "worker:token",
        NOW.isoformat(),
        "{}",
    )
    store = Store(root, claim)
    return store, db, root, jobs, overlay, raw


def run(media, **kwargs):
    from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once

    store, db, root, jobs, *_ = media
    return process_local_discard_once(store, db, root, jobs, now_utc=NOW, **kwargs)


def test_removes_overlay_and_all_copied_segments_preserving_facts_and_other_media(media):
    store, _, root, _, overlay, raw = media
    sentinel = root / "selected.mp4"
    sentinel.write_bytes(b"selected")
    result = run(media)
    assert result.outcome == "completed"
    assert result.reclaimed_bytes == len(b"synthetic-overlayraw-oneraw-two")
    assert store.status == "completed"
    assert not overlay.exists()
    assert not list(raw.glob("*.mjpeg"))
    assert (raw / "interval.json").is_file()
    assert (overlay.parent / "summary.json").read_text() == "{}"
    assert sentinel.read_bytes() == b"selected"
    assert any(checkpoint.get("objects") for checkpoint in store.checkpoints)


@pytest.mark.parametrize("crash_target", [None, "pending", "published", "empty_pending"])
@pytest.mark.parametrize("crash_point", ["after_detach", "after_unlink"])
def test_removes_validated_typed_playback_sibling_before_discard_completion(
    media, crash_target, crash_point
):
    import cv2
    import numpy as np

    store, db, root, _, _, raw = media
    with sqlite3.connect(db) as connection:
        source_payload_sha256 = connection.execute(
            "SELECT payload_sha256 FROM shot_analysis_jobs WHERE job_id=?",
            (store.item.job_id,),
        ).fetchone()[0]
    job = ShotMediaEncodeJobV1.create(
        session_id=store.item.session_id,
        shot_id=store.item.shot_id,
        source_epoch_id="epoch",
        start_frame_id=1,
        end_frame_id=2,
        media_root=raw,
        media_sha256=json.loads((raw / "interval.json").read_text())["media_sha256"],
        source=LegacyFrozenJobMediaSourceV1(
            kind="legacy_frozen_job",
            postgame_manifest_sha256=store.item.postgame_manifest_sha256,
            source_job_id=store.item.job_id,
            source_job_payload_sha256=source_payload_sha256,
        ),
    )
    temporary_video = root / "fixture.mp4"
    writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"avc1"), 60, (640, 480))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    writer.release()
    video_bytes = temporary_video.read_bytes()
    source = dict(
        media_sha256=job.media_sha256,
        width=640,
        height=480,
        frame_count=2,
        first_captured_at_ns=0,
        last_captured_at_ns=16_666_667,
        fps=60,
    )
    core = dict(
        job=job.identity(),
        source_interval=source,
        output=dict(
            name="original.mp4",
            content_type="video/mp4",
            content_sha256=sha(video_bytes),
            byte_size=len(video_bytes),
            codec="avc1",
            frame_count=2,
            width=640,
            height=480,
            fps=60,
        ),
    )
    artifact_id = "recording-playback-" + sha(canonical(core).encode())
    output = root / "postgame" / store.item.session_id / artifact_id
    output.mkdir()
    original = output / "original.mp4"
    temporary_video.rename(original)
    (output / "manifest.json").write_text(
        canonical(
            dict(schema_version="recording-playback-artifact/v1", artifact_id=artifact_id, **core)
        )
    )
    payload = canonical(_media_job_payload(job))
    with sqlite3.connect(db) as connection:
        connection.execute("ALTER TABLE shot_analysis_jobs ADD COLUMN job_kind")
        connection.execute(
            "INSERT INTO shot_analysis_jobs("
            "job_id,payload_json,payload_sha256,status,artifact_id,job_kind) "
            "VALUES(?,?,?,?,?,?)",
            (job.job_id, payload, sha(payload.encode()), "available", artifact_id, "media"),
        )
    pending = (
        root
        / "postgame"
        / store.item.session_id
        / f".pending-recording-playback-{job.job_id}-{'a' * 32}"
    )
    pending.mkdir()
    (pending / "original.mp4").write_bytes(b"")
    (pending / "manifest.json").write_bytes(b"{")
    if crash_target == "empty_pending":
        (pending / "original.mp4").unlink()
        (pending / "manifest.json").unlink()

    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        path = Path(obj["path"])
        target = (
            path.parent == pending
            if crash_target == "pending"
            else path == media[4]
            if crash_target == "empty_pending"
            else path == original
        )
        if point == crash_point and target:
            raise PowerLoss()

    lost_bytes = 0
    if crash_target is not None:
        with pytest.raises(PowerLoss):
            run(media, fault_hook=crash)
        if crash_point == "after_unlink":
            lost_bytes = next(
                obj["size"]
                for obj in json.loads(store.progress)["objects"]
                if obj["state"] == "unlinking"
            )

    result = run(media)

    assert result.outcome == "completed"
    assert not original.exists()
    assert not (pending / "original.mp4").exists()
    assert not (pending / "manifest.json").exists()
    assert (output / "manifest.json").is_file()
    expected_bytes = len(b"synthetic-overlayraw-oneraw-two") + len(video_bytes)
    if crash_target != "empty_pending":
        expected_bytes += 1
    assert result.reclaimed_bytes == expected_bytes - lost_bytes


@pytest.mark.parametrize(
    "unsafe",
    [
        "unknown_raw",
        "unknown_output",
        "hardlink",
        "symlink",
        "corrupt",
        "shared",
        "missing",
        "job_digest",
    ],
)
def test_unsafe_inventory_blocks_before_any_media_is_removed(media, unsafe):
    store, db, _, _, overlay, raw = media
    if unsafe == "unknown_raw":
        (raw / "unregistered.mjpeg").write_bytes(b"unknown")
    elif unsafe == "unknown_output":
        (overlay.parent / "unknown.mp4").write_bytes(b"unknown")
    elif unsafe == "hardlink":
        os.link(raw / "segment-1.mjpeg", raw.parent / "aliased.mjpeg")
    elif unsafe == "symlink":
        segment = raw / "segment-1.mjpeg"
        segment.unlink()
        segment.symlink_to(overlay)
    elif unsafe == "corrupt":
        (raw / "segment-1.mjpeg").write_bytes(b"wrong")
    elif unsafe == "missing":
        (raw / "segment-1.mjpeg").unlink()
    elif unsafe == "job_digest":
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE shot_analysis_jobs SET payload_sha256=?", ("0" * 64,))
    else:
        path = raw / "interval.json"
        value = json.loads(path.read_text())
        value["storage_kind"] = "session_segment_reference"
        path.write_text(canonical(value))
    result = run(media)
    assert result.outcome == "blocked"
    assert store.status == "blocked"
    assert overlay.is_file()
    assert (raw / "segment-2.mjpeg").is_file()
    assert result.reclaimed_bytes == 0


def test_stale_checkpoint_stops_before_any_detach(media):
    media[0].fence = False
    assert run(media).outcome == "stale"
    assert media[4].is_file()
    assert len(list(media[5].glob("*.mjpeg"))) == 2


def test_shared_pin_retries_without_touching_media(media):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    store, _, root, _, overlay, _ = media
    with session_media_lock(root, store.item.session_id, exclusive=False):
        result = run(media)
    assert result.outcome == "retry"
    assert store.status == "pending"
    assert overlay.is_file()


@pytest.mark.parametrize("crash_point", ["after_detach", "after_unlink"])
def test_restart_recovers_durable_exact_object_progress(media, crash_point):
    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        if point == crash_point:
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash)
    progress = json.loads(media[0].progress)
    assert progress["objects"]
    result = run(media)
    assert result.outcome == "completed"
    assert result.reclaimed_bytes == (
        len(b"synthetic-overlayraw-oneraw-two")
        if crash_point == "after_detach"
        else len(b"raw-oneraw-two")
    )
    assert not media[4].exists()
    assert not list(media[5].glob("*.mjpeg"))


def test_changed_object_after_detach_is_preserved_and_blocks(media):
    def replace_detached(point, obj):
        if point == "after_detach":
            Path(obj["staged_path"]).write_bytes(b"replaced")

    result = run(media, fault_hook=replace_detached)
    assert result.outcome == "blocked"
    assert list(media[2].rglob("*.detached"))
    assert result.reclaimed_bytes == 0


def test_missing_jobs_database_is_not_initialized(media):
    media[1].unlink()
    assert run(media).outcome == "blocked"
    assert not media[1].exists()


def test_unsupported_platform_fails_closed(media):
    assert run(media, platform="other").outcome == "blocked"
    assert media[4].is_file()


def test_cancelled_attempt_retries_preserving_media(media):
    assert run(media, cancelled=lambda: True).outcome == "retry"
    assert media[4].is_file()


def test_unknown_staging_child_blocks_and_preserves_detached_object(media):
    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        if point == "after_detach":
            Path(obj["staged_path"]).with_name("unknown").write_bytes(b"private")
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash)
    assert run(media).outcome == "blocked"
    assert list(media[2].rglob("*.detached"))


def test_replacement_source_after_detach_blocks_recovery_without_removing_either(media):
    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        if point == "after_detach":
            Path(obj["path"]).write_bytes(b"replacement")
            raise PowerLoss()

    with pytest.raises(PowerLoss):
        run(media, fault_hook=crash)
    assert run(media).outcome == "blocked"
    assert media[4].read_bytes() == b"replacement"
    assert list(media[2].rglob("*.detached"))


def test_stale_owner_after_detach_retains_journal_for_next_owner(media):
    def lose_claim(point, obj):
        if point == "after_detach":
            media[0].fence = False

    assert run(media, fault_hook=lose_claim).outcome == "stale"
    assert list(media[2].rglob("*.detached"))
    media[0].fence = True
    assert run(media).outcome == "completed"


def test_corrupt_recovery_checkpoint_blocks_without_mutation(media):
    media[0].progress = '{"version":1,"objects":[null]}'
    assert run(media).outcome == "blocked"
    assert media[4].is_file()


def test_windows_port_selection_has_no_posix_fallback(monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    windows = object()
    monkeypatch.setattr(cleanup, "_WindowsFS", lambda: windows)
    assert cleanup._port("nt") is windows


def test_windows_missing_native_api_blocks_with_no_filesystem_fallback(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    def unavailable():
        raise AttributeError("native API unavailable")

    monkeypatch.setattr(cleanup, "_windows_api", unavailable)
    assert run(media, platform="nt").outcome == "blocked"
    assert media[4].is_file()


@pytest.mark.parametrize("operation", ["rename", "unlink"])
def test_windows_mutations_use_exact_open_handle_without_path_fallback(operation):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    calls = []

    class API:
        def SetFileInformationByHandle(self, handle, kind, data, length):
            calls.append((handle, kind, length))
            return False

        def GetLastError(self):
            return 5

    port = cleanup._WindowsFS(api=API(), handle_for_fd=lambda descriptor: 77)
    with pytest.raises(OSError):
        if operation == "rename":
            port.rename(1, Path("source"), 2, Path("destination"), 3)
        else:
            port.unlink(2, Path("destination"), 3)
    assert calls[0][:2] == (77, 3 if operation == "rename" else 4)


def test_windows_existing_media_open_never_creates_and_rejects_reparse():
    import ctypes

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    calls = []

    class API:
        def CreateFileW(self, *args):
            calls.append(args)
            return 77

        def GetFileInformationByHandleEx(self, handle, kind, data, size):
            ctypes.cast(data, ctypes.POINTER(ctypes.c_uint32))[0] = 0x400
            return True

        def CloseHandle(self, handle):
            calls.append(("close", handle))

    with pytest.raises(ValueError, match="reparse"):
        cleanup._windows_existing(API(), Path("C:/media.mp4"), directory=False)
    assert calls[0][4] == 3  # OPEN_EXISTING; never OPEN_ALWAYS.
    assert calls[0][5] & 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT.
    assert calls[0][2] == 3  # No FILE_SHARE_DELETE.
    assert calls[-1] == ("close", 77)


def test_staging_collision_never_overwrites_an_unknown_object(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original = cleanup._PosixFS.rename

    def collide(self, source, source_path, target, target_path, descriptor):
        target_path.write_bytes(b"unknown-owner")
        return original(self, source, source_path, target, target_path, descriptor)

    monkeypatch.setattr(cleanup._PosixFS, "rename", collide)
    result = run(media)
    assert result.outcome == "blocked"
    assert media[4].is_file()
    assert next(media[2].rglob("*.detached")).read_bytes() == b"unknown-owner"


@pytest.mark.parametrize("boundary", ["raw", "output", "root"])
def test_symlinked_ancestor_blocks_without_traversal(media, boundary):
    store, _, root, _, overlay, raw = media
    path = raw if boundary == "raw" else overlay.parent if boundary == "output" else root
    replacement = path.with_name(path.name + "-real")
    path.rename(replacement)
    path.symlink_to(replacement, target_is_directory=True)
    result = run(media)
    assert result.outcome == "blocked"
    assert overlay.exists()
    assert (raw / "segment-1.mjpeg").exists()


def test_journal_capacity_overflow_blocks_without_retry_loop(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original = cleanup._inventory

    def oversized(*args, **kwargs):
        result = original(*args, **kwargs)
        result["overflow"] = "x" * 65536
        return result

    monkeypatch.setattr(cleanup, "_inventory", oversized)
    assert run(media).outcome == "blocked"
    assert media[4].is_file()


def test_bounded_media_size_blocks_before_hash_or_mutation(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    monkeypatch.setattr(cleanup, "_MAX_MEDIA_BYTES", 1)
    assert run(media).outcome == "blocked"
    assert media[4].is_file()


def test_detach_io_failure_reports_retry_with_zero_reclaimed_bytes(media):
    import errno

    def interrupted(point, obj):
        if point == "after_detach":
            raise OSError(errno.EIO, "temporary device failure")

    result = run(media, fault_hook=interrupted)
    assert result.outcome == "retry"
    assert result.reclaimed_bytes == 0
    assert list(media[2].rglob("*.detached"))
    assert run(media).outcome == "completed"


@pytest.mark.parametrize("saved", [False, True])
def test_real_store_completed_review_and_reopened_journal(media, saved):
    from datetime import timedelta

    from billiards_ai.edge.route_assistance.local_media_cleanup import process_local_discard_once
    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from tests.unit.edge.route_assistance.test_persistence import _postgame_store

    _, db, root, jobs, overlay, raw = media
    store = _postgame_store(root.parent / "field.sqlite3")
    store.bind_artifact_path("artifact-1", root)
    manifest = store.get_postgame_manifest("session-1")
    store.record_postgame_clip_available(
        session_id="session-1",
        shot_id="shot-0001",
        job_id="job-1",
        artifact_id="a" * 64,
        postgame_manifest_sha256=manifest.manifest_sha256,
        local_path=overlay,
        content_sha256=sha(overlay.read_bytes()),
        byte_size=overlay.stat().st_size,
        content_type="video/mp4",
    )
    store.record_postgame_finalization("session-1", '{"clips":[],"report":{}}')
    if saved:
        store.save_postgame_clip(
            request_id="save-before-complete",
            session_id="session-1",
            shot_id="shot-0001",
            postgame_manifest_sha256=manifest.manifest_sha256,
        )
    store.complete_postgame_review("artifact-1", "session-1", manifest.manifest_sha256)

    class PowerLoss(BaseException):
        pass

    def crash(point, obj):
        if point == "after_detach":
            raise PowerLoss()

    if saved:
        assert process_local_discard_once(store, db, root, jobs, now_utc=NOW).outcome == "idle"
        assert overlay.is_file()
        assert len(list(raw.glob("*.mjpeg"))) == 2
        return
    with pytest.raises(PowerLoss):
        process_local_discard_once(
            store, db, root, jobs, now_utc=NOW, lease_seconds=1, fault_hook=crash
        )
    restarted = SQLiteFieldRecordStore(store.path)
    result = process_local_discard_once(
        restarted, db, root, jobs, now_utc=NOW + timedelta(seconds=2)
    )
    assert result.outcome == "completed"
    with sqlite3.connect(store.path) as connection:
        status, progress = connection.execute(
            "SELECT status, progress_json FROM field_local_media_discards"
        ).fetchone()
    assert status == "completed"
    assert all(obj["state"] == "deleted" for obj in json.loads(progress)["objects"])
    assert not overlay.exists()
    assert not list(raw.glob("*.mjpeg"))
    assert restarted.is_postgame_finalized("session-1")
    assert len(restarted.list_shots("session-1")) == 1


@pytest.mark.parametrize("acl", ["D:P(A;;FA;;;SY)(A;;FA;;;OW)", "D:(A;;FA;;;WD)"])
def test_windows_staging_acl_must_be_protected_owner_system_only(monkeypatch, acl):
    import ctypes
    from types import SimpleNamespace

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    calls = []
    text_buffer = ctypes.create_unicode_buffer(acl)

    class Call:
        def __init__(self, function):
            self.function = function

        def __call__(self, *args):
            return self.function(*args)

    def convert(sddl, revision, output, size):
        calls.append(("convert", sddl))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        return True

    def get_security(path, kind, flags, owner, group, dacl, sacl, output):
        calls.append(("inspect", path, kind, flags))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 456
        return 0

    def to_string(descriptor, revision, flags, output, size):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_wchar_p))[0] = ctypes.cast(
            text_buffer, ctypes.c_wchar_p
        )
        return True

    advapi = SimpleNamespace(
        ConvertStringSecurityDescriptorToSecurityDescriptorW=Call(convert),
        GetNamedSecurityInfoW=Call(get_security),
        ConvertSecurityDescriptorToStringSecurityDescriptorW=Call(to_string),
    )
    kernel = SimpleNamespace(
        CreateDirectoryW=Call(lambda path, attrs: calls.append(("create", path)) or True),
        LocalFree=Call(lambda handle: calls.append(("free",))),
    )
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **kwargs: advapi if name == "advapi32" else kernel,
        raising=False,
    )
    if acl.startswith("D:P"):
        cleanup._windows_private_directory(Path("C:/private"), create=True)
        assert calls[:3] == [("convert", acl), ("create", "C:/private"), ("free",)]
    else:
        with pytest.raises(ValueError, match="unsafe_staging_permissions"):
            cleanup._windows_private_directory(Path("C:/private"), create=False)
        assert not any(call[0] == "create" for call in calls)
    assert any(call[0] == "inspect" and call[-1] == 4 for call in calls)


def test_windows_handle_rename_uses_no_replace_and_utf16_destination():
    import ctypes

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    calls = []

    class API:
        def SetFileInformationByHandle(self, handle, kind, data, length):
            calls.append((handle, kind, ctypes.string_at(data, length)))
            return True

    fs = cleanup._WindowsFS(api=API(), handle_for_fd=lambda descriptor: 123)
    fs.rename(None, Path("source"), None, Path("C:/private/000.detached"), 9)
    fs.unlink(None, Path("C:/private/000.detached"), 9)
    assert calls[0][:2] == (123, 3)
    assert calls[0][2][0] == 0
    assert "C:/private/000.detached".encode("utf-16-le") in calls[0][2]
    assert calls[1] == (123, 4, b"\x01")


@pytest.mark.parametrize("sqlite_first", [False, True])
def test_windows_jobs_guard_and_sqlite_share_compatible_read_access(
    media, monkeypatch, sqlite_first
):
    import ctypes
    from contextlib import contextmanager

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    class SharingAPI:
        def __init__(self):
            self.handles = {}
            self.next_handle = 10
            self.opens = []

        def CreateFileW(self, path, access, share, *rest):
            bits = ((0x80000000, 1), (0x40000000, 2), (0x10000, 4))
            for previous_access, previous_share in self.handles.values():
                if any(
                    (access & desired and not previous_share & allowed)
                    or (previous_access & desired and not share & allowed)
                    for desired, allowed in bits
                ):
                    return ctypes.c_void_p(-1).value
            self.next_handle += 1
            self.handles[self.next_handle] = (access, share)
            self.opens.append((access, share))
            return self.next_handle

        def GetLastError(self):
            return 32

        def GetFileInformationByHandleEx(self, *args):
            return True

        def CloseHandle(self, handle):
            del self.handles[handle]

    api = SharingAPI()

    class GuardPort(cleanup._PosixFS):
        @contextmanager
        def file(self, parent, path, **kwargs):
            handle = cleanup._windows_existing(api, path, directory=False, **kwargs)
            try:
                with super().file(parent, path) as descriptor:
                    yield descriptor
            finally:
                api.CloseHandle(handle)

    connect = sqlite3.connect

    def sqlite_open(*args, **kwargs):
        # SQLite Windows VFS: generic read/write, no FILE_SHARE_DELETE.
        handle = api.CreateFileW(str(media[1]), 0xC0000000, 3, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise sqlite3.OperationalError("database sharing violation")
        return connect(*args, **kwargs)

    if sqlite_first:
        assert (
            api.CreateFileW(str(media[1]), 0xC0000000, 3, None, 3, 0, None)
            != ctypes.c_void_p(-1).value
        )
    monkeypatch.setattr(cleanup.sqlite3, "connect", sqlite_open)
    payload = cleanup._load_job(GuardPort(), media[1], media[0].item)
    assert payload["job_id"] == media[0].item.job_id
    assert (0x80000000, 3) in api.opens
    assert not any(access & 0x10000 for access, _ in api.opens)


def test_stolen_claim_during_final_hash_preserves_staged_overlay(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original = cleanup._verify
    stolen = []

    def steal_during_verify(descriptor, obj, cancelled):
        original(descriptor, obj, cancelled)
        if obj["state"] == "unlinking" and not stolen:
            media[0].fence = False
            stolen.append(True)

    monkeypatch.setattr(cleanup, "_verify", steal_during_verify)
    result = run(media)
    assert result.outcome == "stale"
    assert stolen
    assert next(media[2].rglob("*.detached")).read_bytes() == b"synthetic-overlay"
    assert (media[5] / "segment-1.mjpeg").is_file()


def test_second_collector_cannot_take_over_inside_final_verify_unlink_boundary(media, monkeypatch):
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    original_unlink = cleanup._PosixFS.unlink
    claim = media[0].claim_due_local_discard
    claimed = []
    attempts = []

    def count_claims(*args, **kwargs):
        claimed.append(True)
        return claim(*args, **kwargs)

    def compete_before_unlink(self, directory, path, descriptor):
        if not attempts:
            attempts.append(run(media))
        original_unlink(self, directory, path, descriptor)

    monkeypatch.setattr(media[0], "claim_due_local_discard", count_claims)
    monkeypatch.setattr(cleanup._PosixFS, "unlink", compete_before_unlink)
    assert run(media).outcome == "completed"
    assert attempts[0].outcome == "retry"
    assert len(claimed) == 1


def test_concurrent_collectors_serialize_before_claim_while_session_stays_exclusively_pinned(
    media, monkeypatch
):
    import threading

    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    entered, release = threading.Event(), threading.Event()
    original_verify = cleanup._verify
    original_claim = media[0].claim_due_local_discard
    claims, results = [], []

    def count_claims(*args, **kwargs):
        claims.append(True)
        return original_claim(*args, **kwargs)

    def blocked_verify(descriptor, obj, cancelled):
        original_verify(descriptor, obj, cancelled)
        if obj["state"] == "unlinking" and not entered.is_set():
            entered.set()
            assert release.wait(2)

    monkeypatch.setattr(media[0], "claim_due_local_discard", count_claims)
    monkeypatch.setattr(cleanup, "_verify", blocked_verify)
    first = threading.Thread(target=lambda: results.append(run(media)))
    first.start()
    try:
        assert entered.wait(2)
        second = run(media)
        assert second.outcome == "retry"
        assert second.error_code == "collector_control_unavailable"
        assert len(claims) == 1
        with pytest.raises(TimeoutError):
            with session_media_lock(media[2], media[0].item.session_id, exclusive=False):
                pytest.fail("session reader entered during physical cleanup")
    finally:
        release.set()
        first.join(2)
    assert not first.is_alive()
    assert results[0].outcome == "completed"


def test_claim_control_cannot_alias_a_real_session_pin_and_is_never_recreated(media):
    from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

    root = media[2]
    store_path = media[0].path
    claim_identity = "claims-v1:" + sha(str(store_path).encode())
    with (
        session_media_lock(root, claim_identity, exclusive=True),
        session_media_lock(store_path.parent, claim_identity, exclusive=True),
    ):
        assert run(media).outcome == "completed"
    controls = list((media[0].path.parent / ".local-media-cleanup-control").rglob("*.lock"))
    assert len(controls) == 1
    assert not (root / ".local-media-cleanup-control").exists()
    identity = controls[0].stat().st_ino
    assert run(media).outcome == "idle"
    assert controls[0].stat().st_ino == identity


def test_windows_media_mutation_handle_retains_delete_access():
    from billiards_ai.edge.route_assistance import local_media_cleanup as cleanup

    opens = []

    class API:
        def CreateFileW(self, path, access, share, *rest):
            opens.append((access, share))
            return 77

        def GetFileInformationByHandleEx(self, *args):
            return True

    assert (
        cleanup._windows_existing(API(), Path("C:/overlay.mp4"), directory=False, mutable=True)
        == 77
    )
    assert opens == [(0x80010000, 3)]


def test_real_ring_materialized_mjpgseg_copies_are_removed_without_touching_ring(media):
    from billiards_ai.lab.mainline1_shot_jobs import materialize_frozen_interval
    from tests.unit.lab.test_mainline1_shot_jobs import _populated_ring

    _, database, root, _, overlay, raw = media
    ring = _populated_ring(root.parent / "ring")
    frozen = ring.freeze_shot()
    original_segments = {name: (frozen.root / name).read_bytes() for name in frozen.segment_sha256}
    for name in ("interval.json", "segment-1.mjpeg", "segment-2.mjpeg"):
        (raw / name).unlink()
    raw.rmdir()
    copied = materialize_frozen_interval(frozen, raw)
    assert all(Path(name).suffix == ".mjpgseg" for name in copied.segment_sha256)
    assert [frame.capture_frame_id for frame in copied.frames()] == list(
        range(copied.start_frame_id, copied.end_frame_id + 1)
    )
    with sqlite3.connect(database) as connection:
        payload = json.loads(
            connection.execute("SELECT payload_json FROM shot_analysis_jobs").fetchone()[0]
        )
        payload.update(
            source_epoch_id=copied.source_epoch_id,
            start_frame_id=copied.start_frame_id,
            end_frame_id=copied.end_frame_id,
            media_sha256=copied.media_sha256,
        )
        encoded = canonical(payload)
        connection.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?,payload_sha256=?",
            (encoded, sha(encoded.encode())),
        )
    manifest_path = overlay.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(payload)
    manifest_path.write_text(canonical(manifest))
    expected_bytes = overlay.stat().st_size + sum(
        (raw / name).stat().st_size for name in copied.segment_sha256
    )

    result = run(media)

    assert result.outcome == "completed", result.error_code
    assert result.reclaimed_bytes == expected_bytes
    assert not overlay.exists()
    assert all(not (raw / name).exists() for name in copied.segment_sha256)
    assert (raw / "interval.json").is_file()
    assert (overlay.parent / "summary.json").read_text() == "{}"
    assert {
        name: (frozen.root / name).read_bytes() for name in original_segments
    } == original_segments
    ring.close()
