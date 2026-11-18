"""Resolve original managed inputs without granting lifecycle or media authority."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from billiards_ai.edge.route_assistance.managed_sessions import ManagedSessionScope
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import FieldSessionStatus
from billiards_ai.edge.route_assistance.retention_inventory import _job, _sha
from billiards_ai.edge.route_assistance.runtime_resources import ManagedResourceSpecRecord
from billiards_ai.edge.route_assistance.shot_diagnostics import is_capture_interruption
from billiards_ai.lab.field_assistance.runtime_resources import (
    _native_path,
    verify_managed_runtime_resources,
)
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval
from billiards_ai.lab.mainline1_runtime import _json_object

Purpose = Literal["analysis", "media", "cleanup"]
_CODES = {
    "analysis": "POSTGAME_ORIGINAL_CONTEXT_UNAVAILABLE",
    "media": "POSTGAME_MEDIA_ORIGINAL_CONTEXT_UNAVAILABLE",
    "cleanup": "LOCAL_MEDIA_CLEANUP_ORIGINAL_CONTEXT_UNAVAILABLE",
}


class ManagedRuntimeResourceUnavailable(ValueError):
    """Only fixed private screen codes may leave resource resolution."""

    def __init__(self, code: str):
        if code not in _CODES.values():
            raise ValueError("invalid managed resource reason")
        self.reason_code = code
        super().__init__(code)


@dataclass(frozen=True)
class ResolvedManagedRuntimeResources:
    spec_record: ManagedResourceSpecRecord
    jobs_db: Path
    artifact_root: Path
    jobs_root: Path
    ring_root: Path


def _verify_interval_metadata(payload, ring_root, session_id):
    root = _native_path(Path(payload["media_root"]), directory=True)
    manifest = _json_object(
        _native_path(root / "interval.json", directory=False), label="original interval"
    )
    for name in (
        "schema_version",
        "source_epoch_id",
        "start_frame_id",
        "end_frame_id",
        "media_sha256",
    ):
        if manifest.get(name) != payload.get(name):
            raise ValueError("original interval lineage mismatch")
    segments = manifest.get("segment_sha256")
    if (
        not isinstance(segments, dict)
        or not segments
        or any(
            not isinstance(name, str)
            or name in {".", "..", ".project-archive"}
            or Path(name).name != name
            or "\\" in name
            or not _sha(digest)
            for name, digest in segments.items()
        )
    ):
        raise ValueError("original interval inventory mismatch")
    if any(
        type(manifest.get(name)) is not int or manifest[name] <= 0
        for name in ("width", "height", "frame_count")
    ):
        raise ValueError("original interval dimensions mismatch")
    if manifest["frame_count"] != manifest["end_frame_id"] - manifest["start_frame_id"] + 1:
        raise ValueError("original interval frame range mismatch")
    storage = manifest.get("storage_kind", "copied_segments")
    if storage == "session_segment_reference":
        source = _native_path(Path(manifest["segment_root"]), directory=True)
        source.relative_to(ring_root / "sessions" / session_id / payload["source_epoch_id"])
    elif storage != "copied_segments" or any(
        k in manifest for k in ("segment_root", "recording_root")
    ):
        raise ValueError("original interval storage mismatch")
    frozen = FrozenCaptureInterval(
        root=root,
        source_epoch_id=manifest["source_epoch_id"],
        width=manifest["width"],
        height=manifest["height"],
        start_frame_id=manifest["start_frame_id"],
        end_frame_id=manifest["end_frame_id"],
        frame_count=manifest["frame_count"],
        segment_sha256=segments,
    )
    if frozen.media_sha256 != payload["media_sha256"]:
        raise ValueError("original interval digest mismatch")


def _verify_recording_media_sources(store, session_id, jobs_root, ring_root, manifest, shots):
    """Bind a managed media launch to the frozen ordinary recording only."""

    if manifest.expected_jobs:
        raise ValueError("ordinary recording manifest contains analysis jobs")
    recording = store.get_recording_end_manifest(session_id)
    if recording is None:
        raise ValueError("frozen recording end is missing")
    if [item["shot_id"] for item in recording["shots"]] != [shot.shot_id for shot in shots]:
        raise ValueError("frozen recording shots changed")
    raw_manifest_path = _native_path(Path(recording["raw_manifest_path"]), directory=False)
    raw_root = raw_manifest_path.parent
    raw_root.relative_to(ring_root / "sessions" / session_id / recording["source_epoch_id"])
    deadline = time.monotonic() + 6.0
    raw_manifest = store._validate_recording_raw(recording, deadline=deadline)
    for frozen in recording["shots"]:
        evidence = store.get_recording_evidence(session_id, frozen["shot_id"])
        encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if sha256(encoded.encode()).hexdigest() != frozen["payload_sha256"]:
            raise ValueError("frozen recording evidence changed")
        media = evidence["media"]
        reference_path = media.get("reference_path")
        if reference_path is not None:
            reference = _native_path(Path(reference_path), directory=False)
            expected = jobs_root / "media" / session_id / media["evidence_ref_id"] / "interval.json"
            if reference != expected:
                raise ValueError("recording reference path changed")
        store._validate_recording_end_reference(
            evidence,
            raw_manifest_path,
            raw_manifest,
            deadline=deadline,
        )


def resolve_ordinary_recording_jobs(store, scope, session_id):
    """Recover only the queue proved by immutable ordinary recording evidence.

    Normal recording has no model specification. A stopped capture and its
    committed references suffice even before detection finishes. Missing proof
    never authorizes a current/default queue or a new database.
    """
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore

    record = store.get_managed_open_request_for_session(scope, session_id)
    session = store.validate_managed_local_session(scope, record.open_request_id)
    if session.status is not FieldSessionStatus.ENDED or session.session_id != session_id:
        raise ValueError("original recording is not ended")
    if store.get_session_policy(session_id) != "recording-v1":
        raise ValueError("original recording policy changed")
    _native_path(store.artifact_path_for_session(session_id), directory=True)
    authority = store.get_recording_capture_end(session_id) or store.get_recording_end_manifest(
        session_id
    )
    if authority is None:
        raise ValueError("original recording authority unavailable")
    epoch = authority["source_epoch_id"]
    raw = _native_path(Path(authority["raw_manifest_path"]), directory=False)
    if raw.parts[-5:] != (
        "sessions",
        session_id,
        epoch,
        "session-recording",
        "session-recording.json",
    ):
        raise ValueError("original recording root changed")
    manifest = _json_object(raw, label="original recording")
    if (manifest.get("session_id"), manifest.get("source_epoch_id")) != (session_id, epoch):
        raise ValueError("original recording lineage changed")
    roots = set()
    deadline = time.monotonic() + 6
    for shot in store.list_shots(session_id):
        evidence = store.get_recording_evidence(session_id, shot.shot_id)
        if evidence["media_status"] != "RECORDED":
            continue
        if evidence["start"]["source_epoch_id"] != epoch:
            raise ValueError("original recording epoch changed")
        media = evidence["media"]
        reference = _native_path(Path(media["reference_path"]), directory=False)
        if reference.parts[-4:] != ("media", session_id, media["evidence_ref_id"], "interval.json"):
            raise ValueError("original recording reference root changed")
        store._validate_recording_end_reference(evidence, raw, manifest, deadline=deadline)
        roots.add(reference.parents[3])
    if len(roots) != 1:
        raise ValueError("original recording queue cannot be proved")
    database = _native_path(roots.pop() / "analysis-jobs.sqlite3", directory=False)
    return SQLiteShotAnalysisJobStore.open_existing(database)


def resolve_managed_session_resources(
    store: SQLiteFieldRecordStore, scope: ManagedSessionScope, session_id: str, *, purpose: Purpose
) -> ResolvedManagedRuntimeResources:
    """Read targeted frozen jobs, strict retained identity and original resources.

    Discovery never constructs the jobs store and never follows active calibration.
    Cleanup reads metadata only: journaled media objects may already be absent.
    """
    if purpose not in _CODES:
        raise ValueError("invalid managed recovery purpose")
    try:
        record = store.get_managed_resource_spec_for_session(scope, session_id)
        session = store.validate_managed_local_session(scope, record.open_request_id)
        spec = record.spec
        if session.status is not FieldSessionStatus.ENDED or session.session_id != session_id:
            raise ValueError("session not ended")
        root = Path(spec.paths["artifact_root"])
        jobs_root, ring_root = Path(spec.paths["jobs_root"]), Path(spec.paths["ring_root"])
        if store.artifact_path_for_session(session_id) != root:
            raise ValueError("original artifact binding mismatch")
        # Roots and the DB are checked without creating/migrating a discovery DB.
        verify_managed_runtime_resources(spec, purpose="cleanup")
        db = jobs_root / "analysis-jobs.sqlite3"
        if _native_path(db, directory=False) != db:
            raise ValueError("original jobs DB changed")
        manifest = store.get_postgame_manifest(session_id)
        if manifest is None or manifest.ended_at_utc != datetime.fromisoformat(
            session.ended_at_utc.replace("Z", "+00:00")
        ):
            raise ValueError("frozen manifest missing")
        shots = store.list_shots(session_id)
        if purpose == "media" and store.get_session_policy(session_id) == "recording-v1":
            _verify_recording_media_sources(
                store, session_id, jobs_root, ring_root, manifest, shots
            )
        bound = [
            s.live_job_id
            for s in shots
            if s.live_job_id is not None and not is_capture_interruption(s)
        ]
        if len(bound) != len(set(bound)) or set(bound) != set(manifest.expected_jobs):
            raise ValueError("frozen jobs mismatch")
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            for shot in shots:
                if shot.live_job_id is None:
                    continue
                if is_capture_interruption(shot):
                    epoch = store.get_managed_capture_epoch(
                        scope, session_id, shot.media.source_epoch_id
                    )
                    if epoch.spec_sha256 != record.spec_sha256:
                        raise ValueError("original diagnostic epoch mismatch")
                    continue
                row, payload = _job(connection, shot.live_job_id, session_id, shot.shot_id)
                if row["payload_json"] != json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ) or shot.live_job_id in {".", ".."}:
                    raise ValueError("original job is not canonical")
                epoch = store.get_managed_capture_epoch(
                    scope, session_id, payload["source_epoch_id"]
                )
                media = store.get_live_shot_media(session_id, shot.shot_id)
                if (
                    epoch.spec_sha256 != record.spec_sha256
                    or shot.settled_ms is None
                    or shot.media.source_kind != "usb_live"
                    or shot.media.source_epoch_id != payload["source_epoch_id"]
                    or media is None
                    or (media.job_id, media.source_epoch_id, media.media_sha256, media.media_path)
                    != (
                        shot.live_job_id,
                        payload["source_epoch_id"],
                        payload["media_sha256"],
                        payload["media_root"],
                    )
                    or payload["media_root"]
                    != str(jobs_root / "media" / session_id / shot.live_job_id)
                ):
                    raise ValueError("original media lineage mismatch")
                _verify_interval_metadata(payload, ring_root, session_id)
                expected = dict(
                    capture_setup_id=spec.capture_setup_id,
                    calibration_artifact_id=spec.calibration_artifact_id,
                    config_sha256=spec.config_sha256,
                    model_package_id=spec.model_package_id,
                    identity_training_package_id=spec.identity_training_package_id,
                    identity_training_package_sha256=spec.identity_training_package_sha256,
                )
                if any(payload.get(k) != v for k, v in expected.items()):
                    raise ValueError("original job dependencies mismatch")
        if purpose == "analysis":
            verify_managed_runtime_resources(spec, purpose="analysis")
        return ResolvedManagedRuntimeResources(record, db, root, jobs_root, ring_root)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, sqlite3.Error):
        raise ManagedRuntimeResourceUnavailable(_CODES[purpose]) from None


class ManagedRuntimeResourceResolver:
    """Purpose-aware private notices projected onto the existing flat screen map."""

    def __init__(self, store, scope, diagnostics):
        self.store, self.scope, self.diagnostics = store, scope, diagnostics
        self._pending = {}
        self._previous = {}
        self._lock = threading.Lock()

    def _note(self, session_id, purpose, unavailable):
        # Never publish unknown or foreign session identities.
        try:
            self.store.get_managed_open_request_for_session(self.scope, session_id)
        except (OSError, ValueError):
            return
        with self._lock:
            current = self.diagnostics.get(session_id)
            if current is not None and current not in _CODES.values():
                self._previous[session_id] = current
            pending = self._pending.setdefault(session_id, set())
            # A reconstructed resolver inherits the visible pending purpose. It
            # must not clear an analysis failure when cleanup later succeeds.
            for existing_purpose, code in _CODES.items():
                if current == code:
                    pending.add(existing_purpose)
            if unavailable:
                pending.add(purpose)
            else:
                pending.discard(purpose)
            if pending:
                selected = (
                    "analysis"
                    if "analysis" in pending
                    else "media"
                    if "media" in pending
                    else "cleanup"
                )
                self.diagnostics[session_id] = _CODES[selected]
            else:
                self._pending.pop(session_id, None)
                if current in _CODES.values():
                    old = self._previous.pop(session_id, None)
                    if old is None:
                        self.diagnostics.pop(session_id, None)
                    else:
                        self.diagnostics[session_id] = old

    def resolve(self, session_id: str, *, purpose: Purpose) -> ResolvedManagedRuntimeResources:
        try:
            result = resolve_managed_session_resources(
                self.store, self.scope, session_id, purpose=purpose
            )
        except ManagedRuntimeResourceUnavailable:
            self._note(session_id, purpose, True)
            raise
        self._note(session_id, purpose, False)
        return result
