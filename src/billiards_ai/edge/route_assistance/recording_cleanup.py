"""Explicit ordinary selection disposal using the existing pinned erasure journal.

The ended recording and frozen shot evidence remain the media authority. This
queue contains only deletion intent, a verified inventory and recovery progress.
No model, cloud, jobs-store initializer or recursive file deletion is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from billiards_ai.edge.route_assistance.local_media_cleanup import (
    LocalDiscardAttempt,
    _absolute,
    _canonical_media_specs,
    _dispose,
    _hash,
    _identity_of,
    _inventory,
    _load_job,
    _name,
    _owner_retryable,
    _port,
    _read_json,
    _source_manifest,
    _specs,
    _stage_path,
    _typed_media_specs,
    _Unsafe,
    _validate_progress,
)
from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock
from billiards_ai.edge.route_assistance.owner_device_delivery import _time
from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    recording_playback_inventory,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    RecordingCaptureMediaSourceV1,
    RecordingEndMediaSourceV1,
    RecordingIntervalMediaSourceV1,
    ShotMediaEncodeJobV1,
)

_NAMESPACE = ".recording-media-cleanup"
_BATCH = 16
_MAX_TARGETS = 16384
_MAX_INVENTORY = 4 * 1024**2


def _json(value, limit=64 * 1024):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )
    if len(raw.encode()) > limit:
        raise _Unsafe("recording_cleanup_metadata_limit")
    return raw


def _digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


def _batch_claim(session, manifest_sha, cursor):
    return SimpleNamespace(
        clip_id="recording-discard-" + _digest(_json([session, manifest_sha, cursor])),
    )


def _require_completed_absent(fs, targets, count, roots, session, manifest_sha, cancelled):
    for index, (kind, path, _, _) in enumerate(targets[:count]):
        if cancelled():
            raise InterruptedError("cleanup_cancelled")
        path = _absolute(path)
        if kind not in roots or not path.is_relative_to(roots[kind]):
            raise _Unsafe("recording_cleanup_target_escape")
        claim = _batch_claim(session, manifest_sha, (index // _BATCH) * _BATCH)
        staged = _stage_path(roots[kind], claim, index % _BATCH, _NAMESPACE)
        for candidate in (path, staged):
            try:
                with fs.directory(candidate.parent) as parent, fs.file(parent, candidate):
                    raise _Unsafe("recording_media_reappeared")
            except FileNotFoundError:
                pass


def _authority(store, session, manifest_sha):
    with store._connect() as db:
        row = db.execute(
            "SELECT status FROM field_postgame_reviews WHERE session_id=?", (session,)
        ).fetchone()
        selected = db.execute(
            "SELECT DISTINCT shot_id FROM field_clip_save_requests WHERE session_id=?", (session,)
        ).fetchall()
    manifest = store.get_postgame_manifest(session)
    recording = store.get_recording_end_manifest(session)
    if (
        row is None
        or row[0] != "completed"
        or manifest is None
        or manifest.manifest_sha256 != manifest_sha
    ):
        raise _Unsafe("recording_selection_authority_changed")
    if store.get_session_policy(session) == "legacy":
        shots = {shot.live_job_id: shot.shot_id for shot in store.list_shots(session)}
        if any(job not in shots for job in manifest.expected_jobs):
            raise _Unsafe("legacy_selection_source_changed")
        return dict(
            policy="legacy",
            session_id=session,
            postgame_manifest_sha256=manifest_sha,
            jobs=[(job, shots[job]) for job in manifest.expected_jobs],
            selected_shots=[item[0] for item in selected],
        )
    if selected or recording is None or manifest.expected_jobs:
        raise _Unsafe("recording_selection_authority_changed")
    capture = store.get_recording_capture_end(session)
    if capture is not None and (
        capture["processing_status"] != "COMPLETE"
        or capture["cutoff"] != recording["cutoff"]
        or capture["source_epoch_id"] != recording["source_epoch_id"]
    ):
        raise _Unsafe("recording_processing_not_complete")
    return recording


def _expected_jobs(store, recording, jobs_root):
    result = {}
    capture = store.get_recording_capture_end(recording["session_id"])
    for frozen in recording["shots"]:
        evidence = store.get_recording_evidence(recording["session_id"], frozen["shot_id"])
        if _digest(_json(evidence)) != frozen["payload_sha256"]:
            raise _Unsafe("recording_evidence_changed")
        media, boundary = evidence["media"], evidence["boundary"]
        if evidence["media_status"] != "RECORDED":
            continue
        reference = _absolute(media["reference_path"])
        expected = jobs_root / "media" / recording["session_id"] / _name(media["evidence_ref_id"])
        if reference != expected / "interval.json":
            raise _Unsafe("recording_reference_root_mismatch")
        job = ShotMediaEncodeJobV1.create(
            session_id=recording["session_id"],
            shot_id=frozen["shot_id"],
            source_epoch_id=recording["source_epoch_id"],
            start_frame_id=media["start_frame_id"],
            end_frame_id=media["end_frame_id"],
            media_root=reference.parent,
            media_sha256=media["media_sha256"],
            source=(
                RecordingEndMediaSourceV1(
                    kind="recording_end",
                    recording_end_sha256=recording["manifest_sha256"],
                    shot_evidence_sha256=frozen["payload_sha256"],
                    reference_path=reference,
                    reference_sha256=media["reference_sha256"],
                )
                if capture is None
                else RecordingCaptureMediaSourceV1(
                    kind="recording_capture",
                    capture_end_sha256=capture["capture_end_sha256"],
                    shot_evidence_sha256=frozen["payload_sha256"],
                    reference_path=reference,
                    reference_sha256=media["reference_sha256"],
                )
            ),
        )
        if not boundary or not boundary["complete"]:
            raise _Unsafe("recording_media_boundary_conflict")
        result[job.job_id] = job
        # Live-born tasks retain their immutable interval identity after capture ends.
        interval_job = ShotMediaEncodeJobV1.create(
            session_id=job.session_id,
            shot_id=job.shot_id,
            source_epoch_id=job.source_epoch_id,
            start_frame_id=job.start_frame_id,
            end_frame_id=job.end_frame_id,
            media_root=job.media_root,
            media_sha256=job.media_sha256,
            source=RecordingIntervalMediaSourceV1(
                kind="recording_interval",
                shot_evidence_sha256=frozen["payload_sha256"],
                reference_path=reference,
                reference_sha256=media["reference_sha256"],
            ),
        )
        result[interval_job.job_id] = interval_job
    return result


def _legacy_targets(fs, store, selection, root, jobs_root, jobs_db, retained):
    if jobs_db != jobs_root / "analysis-jobs.sqlite3":
        raise _Unsafe("legacy_selection_jobs_database_mismatch")
    targets = []
    progress = (
        {
            "objects": [
                dict(root=kind, path=path, sha256=digest, size=size, state="planned")
                for kind, path, digest, size in retained
            ]
        }
        if retained
        else None
    )
    for job_id, shot_id in selection["jobs"]:
        if shot_id in selection["selected_shots"]:
            continue
        claim = SimpleNamespace(
            session_id=selection["session_id"],
            shot_id=shot_id,
            job_id=job_id,
            postgame_manifest_sha256=selection["postgame_manifest_sha256"],
        )
        status, artifact_id, payload = _load_job(
            fs,
            jobs_db,
            claim,
            require_artifact=False,
            include_state=True,
        )
        raw = jobs_root / "media" / selection["session_id"] / _name(job_id)
        live = store.get_live_shot_media(selection["session_id"], shot_id)
        if (
            live is None
            or live.job_id != job_id
            or live.media_path != str(raw)
            or payload["media_root"] != str(raw)
            or live.media_sha256 != payload["media_sha256"]
            or live.source_epoch_id != payload["source_epoch_id"]
        ):
            raise _Unsafe("legacy_selection_source_mismatch")
        if status == "available" and artifact_id is not None:
            path = root / "postgame" / selection["session_id"] / _name(artifact_id)
            manifest = _read_json(fs, path / "manifest.json")
            video = manifest["inventory"]["overlay.mp4"]
            claim.artifact_id = artifact_id
            claim.clip_id = session_scoped_clip_id(
                selection["session_id"],
                shot_id,
                job_id,
                artifact_id,
            )
            claim.local_path = path / "overlay.mp4"
            claim.content_sha256, claim.byte_size = video["sha256"], video["bytes"]
            claim.content_type = "video/mp4"
            specs, directories = _specs(fs, claim, jobs_db, root, jobs_root, progress=progress)
        else:
            segments = _source_manifest(_read_json(fs, raw / "interval.json"), payload)
            specs, directories = _typed_media_specs(
                fs,
                jobs_db,
                claim,
                root,
                payload,
                progress,
            )
            specs += [
                ("jobs", raw / name, digest, None) for name, digest in sorted(segments.items())
            ]
            canonical_specs, canonical_directories = _canonical_media_specs(
                fs, claim, root, payload, progress
            )
            specs += canonical_specs
            directories.update(canonical_directories)
            directories[raw] = {"interval.json", *segments}
        retained_paths = {item[1] for item in retained}
        for directory, expected in directories.items():
            with fs.directory(directory) as handle:
                actual = fs.names(handle, directory, limit=_MAX_TARGETS)
            missing = {name for name in expected if str(directory / name) in retained_paths}
            if actual - expected or expected - actual - missing:
                raise _Unsafe("legacy_selection_unregistered_objects")
        targets += [[kind, str(path), digest, size] for kind, path, digest, size in specs]
    return targets


def _targets(
    fs,
    store,
    recording,
    root,
    jobs_root,
    ring_root,
    cancelled,
    *,
    jobs_db=None,
    retained=(),
):
    if recording.get("policy") == "legacy":
        return _legacy_targets(fs, store, recording, root, jobs_root, jobs_db, retained)
    if ring_root is None:
        raise _Unsafe("original_recording_root_unavailable")
    raw_path = _absolute(recording["raw_manifest_path"])
    raw_root = (
        ring_root
        / "sessions"
        / _name(recording["session_id"])
        / _name(recording["source_epoch_id"])
        / "session-recording"
    )
    if raw_path != raw_root / "session-recording.json":
        raise _Unsafe("recording_raw_root_mismatch")
    with fs.directory(raw_root) as directory, fs.file(directory, raw_path) as descriptor:
        if os.fstat(descriptor).st_size > 2 * 1024**2:
            raise _Unsafe("recording_manifest_limit")
        if _hash(descriptor, cancelled) != recording["raw_manifest_sha256"]:
            raise _Unsafe("recording_manifest_changed")
        names = fs.names(directory, raw_root, limit=_MAX_TARGETS)
    allowed = {_name(row["name"]) for row in recording["raw_segments"]}
    if names - (allowed | {raw_path.name}):
        raise _Unsafe("unregistered_recording_objects")
    targets = [
        ["raw", str(raw_root / _name(row["name"])), row["sha256"], row["size"]]
        for row in recording["raw_segments"]
    ]
    expected = _expected_jobs(store, recording, jobs_root)
    parent = root / "postgame" / recording["session_id"]
    try:
        with fs.directory(parent) as directory:
            names = fs.names(directory, parent, limit=_MAX_TARGETS)
    except FileNotFoundError:
        names = set()
    for name in sorted(names):
        if cancelled():
            raise InterruptedError("cleanup_cancelled")
        path = parent / _name(name)
        if re.fullmatch(r"recording-playback-[0-9a-f]{64}", name):
            metadata = _read_json(fs, path / "manifest.json")
            identity = metadata.get("job", {})
            job = expected.get(identity.get("job_id"))
            if job is None:
                raise _Unsafe("unregistered_recording_playback")
            # max_attempts controls retries, not content identity. Retain the
            # actual frozen policy when checking the manifest's complete shape.
            from dataclasses import replace

            job = replace(job, max_attempts=identity.get("max_attempts"))
            artifact = recording_playback_inventory(path, job)
            with fs.directory(path) as directory:
                if fs.names(directory, path) - {"manifest.json", "original.mp4"}:
                    raise _Unsafe("unexpected_recording_playback_objects")
            targets.append(
                [
                    "artifacts",
                    str(artifact.media_path),
                    artifact.content_sha256,
                    artifact.byte_size,
                ]
            )
        else:
            matches = [
                job
                for job in expected.values()
                if re.fullmatch(
                    re.escape(f".pending-recording-playback-{job.job_id}-") + r"[0-9a-f]{32}", name
                )
            ]
            if len(matches) != 1:
                raise _Unsafe("unregistered_recording_playback")
            with fs.directory(path) as directory:
                children = fs.names(directory, path)
                if children - {"manifest.json", "original.mp4"}:
                    raise _Unsafe("unexpected_recording_temporary_objects")
                media = path / "original.mp4"
                previous = [item for item in retained if item[1] == str(media)]
                if previous:
                    if len(previous) != 1 or previous[0][0] != "artifacts":
                        raise _Unsafe("recording_temporary_inventory_changed")
                    targets.append(previous[0])
                elif "original.mp4" in children:
                    with fs.file(directory, media) as descriptor:
                        size = _identity_of(os.fstat(descriptor))[2]
                        if size > 8 * 1024**3:
                            raise _Unsafe("recording_temporary_size_limit")
                        targets.append(
                            ["artifacts", str(media), _hash(descriptor, cancelled), size]
                        )
    if len(targets) > _MAX_TARGETS:
        raise _Unsafe("recording_cleanup_inventory_limit")
    return targets


def process_recording_discard_once(
    store,
    jobs_db,
    artifact_root,
    jobs_root,
    *,
    ring_root,
    session_ids,
    cancelled=lambda: False,
    now_utc=None,
    fault_hook=lambda point, obj: None,
    exclude_managed=False,
):
    """Erase at most sixteen pinned objects; durable progress survives every rename."""
    if len(session_ids) != 1:
        raise ValueError("one original recording context required")
    session = session_ids[0]
    now = now_utc or datetime.now(UTC)
    stamp = _time(now)
    token = uuid4().hex
    progress = None
    claimed = False
    try:
        root, jobs_root, jobs_db = map(_absolute, (artifact_root, jobs_root, jobs_db))
        ring_root = None if ring_root is None else _absolute(ring_root)
        if store.artifact_path_for_session(session) != root:
            raise _Unsafe("recording_cleanup_pin_root_mismatch")
        control = _absolute(store.path).parent / ".local-media-cleanup-control"
        control.mkdir(mode=0o700, exist_ok=True)
        control_id = "claims-v1:" + hashlib.sha256(str(store.path).encode()).hexdigest()
        with ExitStack() as stack:
            stack.enter_context(session_media_lock(control, control_id, exclusive=True))
            with store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM field_recording_cleanup WHERE session_id=? AND "
                    "((status='pending' AND next_attempt_at_utc<=?) OR "
                    "(status='leased' AND lease_until_utc<=?))",
                    (session, stamp, stamp),
                ).fetchone()
                if row is None:
                    return LocalDiscardAttempt("idle")
                if exclude_managed and (
                    session.startswith("managed-")
                    or db.execute(
                        "SELECT 1 FROM field_managed_open_requests WHERE session_id=?", (session,)
                    ).fetchone()
                ):
                    return LocalDiscardAttempt("idle")
                db.execute(
                    "UPDATE field_recording_cleanup SET status='leased',claim_token=?,"
                    "lease_until_utc=?,attempts=attempts+1 WHERE session_id=?",
                    (token, _time(now + timedelta(seconds=30)), session),
                )
            claimed = True
            stack.enter_context(session_media_lock(root, session, exclusive=True))
            recording = _authority(store, session, row["manifest_sha256"])
            fs = _port(os.name)
            inventory_raw = row["inventory_json"]
            if inventory_raw is None:
                targets = _targets(
                    fs,
                    store,
                    recording,
                    root,
                    jobs_root,
                    ring_root,
                    cancelled,
                    jobs_db=jobs_db,
                )
                inventory_raw = _json(targets, _MAX_INVENTORY)
                with store._connect() as db:
                    db.execute(
                        "UPDATE field_recording_cleanup SET inventory_json=?,inventory_sha256=? "
                        "WHERE session_id=? AND claim_token=? AND status='leased'",
                        (inventory_raw, _digest(inventory_raw), session, token),
                    )
            else:
                if (
                    len(inventory_raw.encode()) > _MAX_INVENTORY
                    or _digest(inventory_raw) != row["inventory_sha256"]
                ):
                    raise _Unsafe("recording_cleanup_inventory_changed")
                targets = json.loads(inventory_raw)
                current = _targets(
                    fs,
                    store,
                    recording,
                    root,
                    jobs_root,
                    ring_root,
                    cancelled,
                    jobs_db=jobs_db,
                    retained=targets,
                )
                if _json(current, _MAX_INVENTORY) != inventory_raw:
                    raise _Unsafe("recording_cleanup_inventory_authority_changed")
            if len(row["progress_json"].encode()) > 64 * 1024:
                raise _Unsafe("recording_cleanup_journal_limit")
            progress = json.loads(row["progress_json"])
            if not isinstance(progress, dict):
                raise _Unsafe("invalid_recording_cleanup_progress")
            if not progress:
                progress = dict(version=1, cursor=0, reclaimed_bytes=0, batch=None)
            if (
                set(progress) != {"version", "cursor", "reclaimed_bytes", "batch"}
                or type(progress["version"]) is not int
                or progress["version"] != 1
                or type(progress["cursor"]) is not int
                or not 0 <= progress["cursor"] <= len(targets)
                or type(progress["reclaimed_bytes"]) is not int
                or not 0 <= progress["reclaimed_bytes"] < 2**63
                or progress["batch"] is not None
                and not isinstance(progress["batch"], dict)
                or len(targets) > _MAX_TARGETS
            ):
                raise _Unsafe("invalid_recording_cleanup_progress")
            cursor = progress["cursor"]
            roots = {"artifacts": root, "jobs": jobs_root}
            if ring_root is not None:
                roots["raw"] = ring_root
            specs = []
            for kind, path, digest, size in targets[cursor : cursor + _BATCH]:
                path = _absolute(path)
                if kind not in roots or not path.is_relative_to(roots[kind]):
                    raise _Unsafe("recording_cleanup_target_escape")
                specs.append((kind, path, digest, size))
            claim = _batch_claim(session, row["manifest_sha256"], cursor)
            _require_completed_absent(
                fs,
                targets,
                cursor,
                roots,
                session,
                row["manifest_sha256"],
                cancelled,
            )

            def checkpoint(batch):
                progress["batch"] = batch
                with store._connect() as db:
                    changed = db.execute(
                        "UPDATE field_recording_cleanup SET progress_json=? WHERE session_id=? "
                        "AND claim_token=? AND status='leased'",
                        (_json(progress), session, token),
                    ).rowcount
                if changed != 1:
                    raise _Unsafe("recording_cleanup_claim_lost")

            batch = progress["batch"]
            if batch is None:
                batch = _inventory(fs, specs, roots, claim, cancelled, namespace=_NAMESPACE)
                checkpoint(batch)
            else:
                _validate_progress(batch, specs, roots, claim, namespace=_NAMESPACE)
            _dispose(fs, batch, roots, cancelled, fault_hook, checkpoint)
            _require_completed_absent(
                fs,
                targets,
                cursor + len(specs),
                roots,
                session,
                row["manifest_sha256"],
                cancelled,
            )
            progress["reclaimed_bytes"] += sum(obj["reclaimed_bytes"] for obj in batch["objects"])
            progress.update(cursor=cursor + len(specs), batch=None)
            status = "completed" if progress["cursor"] == len(targets) else "pending"
            with store._connect() as db:
                db.execute(
                    "UPDATE field_recording_cleanup SET status=?,claim_token=NULL,"
                    "lease_until_utc=NULL,"
                    "progress_json=?,next_attempt_at_utc=?,last_error_code=NULL "
                    "WHERE session_id=? AND claim_token=? AND status='leased'",
                    (status, _json(progress), stamp, session, token),
                )
            return LocalDiscardAttempt(
                status, reclaimed_bytes=progress["reclaimed_bytes"], session_id=session
            )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, RecordingPlaybackError) as exc:
        retry = _owner_retryable(exc)
        code = (
            str(exc)
            if isinstance(exc, _Unsafe)
            else exc.code
            if isinstance(exc, RecordingPlaybackError)
            else "recording_cleanup_unavailable"
        )
        if claimed:
            with store._connect() as db:
                db.execute(
                    "UPDATE field_recording_cleanup SET status=?,claim_token=NULL,"
                    "lease_until_utc=NULL,"
                    "next_attempt_at_utc=?,last_error_code=? WHERE session_id=? AND claim_token=?",
                    (
                        "pending" if retry else "blocked",
                        _time(now + timedelta(seconds=1)),
                        code,
                        session,
                        token,
                    ),
                )
        return LocalDiscardAttempt("retry" if retry else "blocked", code, session_id=session)
