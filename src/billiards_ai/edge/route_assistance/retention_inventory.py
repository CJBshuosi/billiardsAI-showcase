"""Bounded ledger inventory. Metadata only; this module has no deletion authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from billiards_ai.edge.route_assistance.postgame import (
    ClipSaveReceipt,
    ClipSaveStatus,
    PostgameSessionManifest,
    session_scoped_clip_id,
)
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJob,
    ShotMediaEncodeJobV1,
    _media_job_from_payload,
)

_MAX_JSON = 1024 * 1024
_MAX_ROWS = 10000
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9_./:-]{1,1024}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _identity(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("invalid_identity")
    return value


def _capture_setup_metadata(value: object) -> str:
    """Setup identity is UTF-8 metadata, never a filesystem path component here."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid_capture_setup_id")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ValueError("invalid_capture_setup_id") from None
    return value


def _public_id(value: object) -> str:
    try:
        return _identity(value)
    except ValueError:
        return "invalid-" + hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _json(value: str, kind: type = dict):
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_JSON:
        raise ValueError("invalid_metadata")
    try:
        result = json.loads(value)
    except (ValueError, RecursionError):
        raise ValueError("invalid_metadata") from None
    if not isinstance(result, kind):
        raise ValueError("invalid_metadata")
    return result


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("invalid_timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone_required")
    return result.astimezone(UTC)


def _absolute(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts or ".project-archive" in path.parts:
        raise ValueError("unsafe_path")
    return path


@contextmanager
def _directory(path: Path):
    """Open each component without following symlinks, including root ancestors."""
    with ExitStack() as stack:
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, descriptor)
        for part in _absolute(path).parts[1:]:
            descriptor = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            stack.callback(os.close, descriptor)
        yield descriptor


def _relative(root: Path, path: Path) -> str:
    path = _absolute(path)
    relative = path.relative_to(root).as_posix()
    if relative == "." or _RELATIVE.fullmatch(relative) is None:
        raise ValueError("unsafe_path")
    return relative


def _metadata(root: Path, path: Path, *, read_json: bool = False):
    relative = _relative(root, path)
    with _directory(path.parent) as parent:
        info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("unsafe_media")
        if read_json:
            if info.st_size > _MAX_JSON or info.st_nlink != 1:
                raise ValueError("invalid_manifest")
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("changed_manifest")
                data = stream.read(_MAX_JSON + 1)
            if len(data) > _MAX_JSON:
                raise ValueError("invalid_manifest")
            return _json(data.decode("utf-8"))
    return relative, info


@contextmanager
def _database(path: Path):
    try:
        path = _absolute(path)
        with _directory(path.parent) as parent:
            if not stat.S_ISREG(os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_mode):
                raise ValueError("database_unavailable")
        # mode=ro includes committed WAL facts; immutable=1 would hide those facts.
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    except (OSError, ValueError, sqlite3.Error):
        raise ValueError("database_unavailable") from None
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def _rows(connection, query, parameters=(), *, limit=_MAX_ROWS):
    return connection.execute(query + " LIMIT ?", (*parameters, limit + 1)).fetchall()


class _Inventory:
    def __init__(self, artifacts: Path, jobs: Path):
        self.artifacts = artifacts
        self.jobs = jobs
        self.issues: set[str] = set()
        self.shots: list[dict] = []
        self.observed: dict[tuple[int, int], dict] = {}
        self.object_count = 0
        self.row_count = 0
        self.state_schema_version = 17

    def issue(self, shot: dict, reason: str):
        self.issues.add(reason)
        shot["protection_reasons"].add(reason)

    def object(self, shot: dict, root: Path, path: Path, kind: str, size=None):
        if self.object_count >= _MAX_ROWS:
            self.issue(shot, "object_limit_reached")
            return
        self.object_count += 1
        obj = {"kind": kind, "root": "artifacts" if root == self.artifacts else "jobs"}
        shot["objects"].append(obj)
        try:
            relative, info = _metadata(root, path)
            obj.update(relative_path=relative, observed_bytes=info.st_size, status="observed")
            if size is not None and info.st_size != size:
                self.issue(shot, "media_size_mismatch")
                obj["status"] = "size_mismatch"
            key = (info.st_dev, info.st_ino)
            duplicate = key in self.observed
            if duplicate:
                self.issue(shot, "aliased_media")
                self.observed[key]["protection_reasons"].add("aliased_media")
            else:
                self.observed[key] = shot
            if info.st_nlink != 1:
                self.issue(shot, "hardlinked_media")
                obj["status"] = "hardlinked"
            obj["unique_observed_bytes"] = 0 if duplicate else info.st_size
        except FileNotFoundError:
            obj["status"] = "missing"
            self.issue(shot, "missing_media")
        except (OSError, ValueError):
            obj["status"] = "unsafe"
            self.issue(shot, "unsafe_media")

    def source(self, shot: dict, payload: dict, media: dict):
        try:
            for key in ("session_id", "shot_id", "job_id", "source_epoch_id", "media_sha256"):
                if media.get(key) != payload.get(key):
                    raise ValueError("invalid_lineage")
            root = Path(payload["media_root"])
            if media.get("media_path") != str(root):
                raise ValueError("invalid_lineage")
            expected = self.jobs / "media" / payload["session_id"] / payload["job_id"]
            if root != expected:
                raise ValueError("unsupported_source_layout")
            manifest = _metadata(self.jobs, root / "interval.json", read_json=True)
            if any(key in manifest for key in ("storage_kind", "segment_root", "recording_root")):
                self.issue(shot, "shared_source_uncovered")
                return
            if manifest.get("schema_version") != "1.0":
                raise ValueError("invalid_manifest")
            for key in ("source_epoch_id", "start_frame_id", "end_frame_id", "media_sha256"):
                if manifest.get(key) != payload.get(key):
                    raise ValueError("invalid_manifest")
            segments = manifest.get("segment_sha256")
            if not isinstance(segments, dict) or not segments:
                raise ValueError("invalid_manifest")
            for name, digest in segments.items():
                if (
                    not isinstance(name, str)
                    or name in {".", ".."}
                    or Path(name).name != name
                    or "\\" in name
                    or not _sha(digest)
                ):
                    raise ValueError("invalid_manifest")
            identity = {
                key: manifest[key]
                for key in (
                    "schema_version",
                    "source_epoch_id",
                    "width",
                    "height",
                    "start_frame_id",
                    "end_frame_id",
                    "frame_count",
                )
            }
            if any(
                type(identity[key]) is not int or identity[key] <= 0
                for key in ("width", "height", "frame_count")
            ):
                raise ValueError("invalid_manifest")
            if identity["frame_count"] != identity["end_frame_id"] - identity["start_frame_id"] + 1:
                raise ValueError("invalid_manifest")
            identity["segments"] = [
                {"name": name, "sha256": digest} for name, digest in segments.items()
            ]
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(encoded).hexdigest() != payload["media_sha256"]:
                raise ValueError("invalid_manifest")
            for name in sorted(segments):
                if self.object_count >= _MAX_ROWS:
                    self.issue(shot, "object_limit_reached")
                    break
                self.object(shot, self.jobs, root / name, "copied_source")
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            self.issue(shot, "invalid_source_metadata")

    def playback(self, shot: dict, row, job: ShotMediaEncodeJobV1):
        """Account for media-task output without treating it as canonical availability."""

        if row["status"] != "available":
            if row["artifact_id"] is not None:
                self.issue(shot, "invalid_playback_task_metadata")
            return
        artifact_id = row["artifact_id"]
        if (
            not isinstance(artifact_id, str)
            or not artifact_id.startswith("recording-playback-")
            or not _sha(artifact_id.removeprefix("recording-playback-"))
        ):
            self.issue(shot, "invalid_playback_task_metadata")
            return
        output = self.artifacts / "postgame" / job.session_id / artifact_id
        try:
            manifest = _metadata(self.artifacts, output / "manifest.json", read_json=True)
            media = manifest["output"]
            if (
                manifest.get("schema_version") != "recording-playback-artifact/v1"
                or manifest.get("artifact_id") != artifact_id
                or manifest.get("job") != job.identity()
                or not isinstance(media, dict)
                or media.get("name") != "original.mp4"
                or media.get("content_type") != "video/mp4"
                or not _sha(media.get("content_sha256"))
                or type(media.get("byte_size")) is not int
                or media["byte_size"] <= 0
            ):
                raise ValueError("invalid_playback_manifest")
            self.object(
                shot,
                self.artifacts,
                output / "original.mp4",
                "recording_playback",
                media["byte_size"],
            )
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            self.issue(shot, "invalid_playback_artifact")


def _session_manifest(row, session) -> PostgameSessionManifest:
    if row is None or session["status"] != "ended":
        raise ValueError("session_unresolved")
    expected = _json(row["expected_jobs_json"], list)
    if len(expected) > _MAX_ROWS:
        raise ValueError("manifest_limit_reached")
    for job in expected:
        _identity(job)
    ended = _time(row["ended_at_utc"])
    deadline = _time(row["deadline_at_utc"])
    if ended != _time(session["ended_at_utc"]) or deadline < ended:
        raise ValueError("invalid_session_metadata")
    return PostgameSessionManifest(session["session_id"], ended, deadline, tuple(expected))


def _job(connection, job_id, session_id, shot_id):
    rows = _rows(
        connection,
        "SELECT job_id, substr(payload_json,1,1048577) AS payload_json, "
        "payload_sha256, status, artifact_id FROM shot_analysis_jobs WHERE job_id=?",
        (job_id,),
        limit=1,
    )
    if len(rows) != 1:
        raise ValueError("job_missing_or_ambiguous")
    row = rows[0]
    encoded = row["payload_json"]
    if (
        not isinstance(encoded, str)
        or hashlib.sha256(encoded.encode()).hexdigest() != row["payload_sha256"]
    ):
        raise ValueError("invalid_job_metadata")
    payload = _json(encoded)
    if (
        payload.get("schema_version"),
        payload.get("job_id"),
        payload.get("session_id"),
        payload.get("shot_id"),
    ) != ("1.0", job_id, session_id, shot_id):
        raise ValueError("invalid_job_lineage")
    if (
        not _sha(payload.get("media_sha256"))
        or not _sha(payload.get("config_sha256"))
        or type(payload.get("start_frame_id")) is not int
        or type(payload.get("end_frame_id")) is not int
        or payload["start_frame_id"] < 0
        or payload["end_frame_id"] < payload["start_frame_id"]
    ):
        raise ValueError("invalid_job_metadata")
    _capture_setup_metadata(payload.get("capture_setup_id"))
    for key in (
        "source_epoch_id",
        "calibration_artifact_id",
        "model_package_id",
    ):
        _identity(payload.get(key))
    if payload.get("identity_training_package_id") is not None:
        _identity(payload["identity_training_package_id"])
    canonical_fields = dict(payload)
    del canonical_fields["schema_version"]
    canonical_fields["media_root"] = Path(canonical_fields["media_root"])
    decoded = ShotAnalysisJob(**canonical_fields)
    if decoded.identity() != payload:
        raise ValueError("invalid_job_metadata")
    return row, payload


def _media_jobs(connection, session_id):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(shot_analysis_jobs)")}
    if "job_kind" not in columns:
        return ()
    result = []
    for row in _rows(
        connection,
        "SELECT job_id,substr(payload_json,1,1048577) AS payload_json,payload_sha256,"
        "status,artifact_id FROM shot_analysis_jobs WHERE job_kind='media'",
    ):
        encoded = row["payload_json"]
        if (
            not isinstance(encoded, str)
            or hashlib.sha256(encoded.encode()).hexdigest() != row["payload_sha256"]
        ):
            raise ValueError("invalid_playback_task_metadata")
        job = _media_job_from_payload(_json(encoded))
        if job.session_id == session_id:
            result.append((row, job))
    return tuple(result)


def _scan_session(inv, state, jobs, session):
    session_id = session["session_id"]
    session_reasons = set()
    manifest = None
    selection_complete = False
    try:
        _identity(session_id)
        manifests = _rows(
            state,
            "SELECT ended_at_utc, deadline_at_utc, "
            "substr(expected_jobs_json,1,1048577) AS expected_jobs_json "
            "FROM field_postgame_manifests WHERE session_id=?",
            (session_id,),
            limit=1,
        )
        if len(manifests) != 1:
            raise ValueError("session_unresolved")
        manifest = _session_manifest(manifests[0], session)
    except (ValueError, TypeError):
        session_reasons.add("session_unresolved")
        inv.issues.add("session_unresolved")

    reviews = _rows(
        state,
        "SELECT artifact_id, status FROM field_postgame_reviews WHERE session_id=?",
        (session_id,),
        limit=1,
    )
    if (
        len(reviews) != 1
        or reviews[0]["artifact_id"] != session["artifact_id"]
        or reviews[0]["status"] not in {"completed", "active", "pending"}
    ):
        session_reasons.add("review_missing_or_invalid")
        inv.issues.add("review_missing_or_invalid")
    elif reviews[0]["status"] == "completed" and manifest is not None:
        selection_complete = True
    else:
        session_reasons.add("selection_unfinished")
        inv.issues.add("selection_unfinished")

    tables = {
        "shots": "SELECT shot_id, substr(payload_json,1,1048577) AS payload_json "
        "FROM field_shots WHERE session_id=? ORDER BY sequence",
        "media": "SELECT shot_id, substr(payload_json,1,1048577) AS payload_json "
        "FROM field_live_shot_media WHERE session_id=?",
        "available": "SELECT shot_id, postgame_manifest_sha256, job_id, artifact_id, "
        "clip_id, local_path, content_sha256, byte_size, content_type "
        "FROM field_postgame_clip_availability WHERE session_id=?",
        "selections": "SELECT shot_id, request_id, postgame_manifest_sha256, "
        "substr(receipt_json,1,1048577) AS receipt_json "
        "FROM field_clip_save_requests WHERE session_id=?",
    }
    if inv.state_schema_version in {18, 19, 20, 21}:
        tables["discards"] = (
            "SELECT shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,local_path,"
            "content_sha256,byte_size,content_type,status FROM field_local_media_discards "
            "WHERE session_id=?"
        )
    facts = {}
    for name, query in tables.items():
        remaining = max(0, _MAX_ROWS - inv.row_count)
        rows = _rows(state, query, (session_id,), limit=remaining)
        if len(rows) > remaining:
            inv.issues.add("ledger_row_limit_reached")
            session_reasons.add("ledger_row_limit_reached")
        facts[name] = rows[:remaining]
        inv.row_count += len(facts[name])
    shot_ids = {row["shot_id"] for row in facts["shots"]}
    if any(
        row["shot_id"] not in shot_ids for name in tables if name != "shots" for row in facts[name]
    ):
        inv.issues.add("unbound_ledger_rows")
        session_reasons.add("unbound_ledger_rows")
    bound_jobs = set()
    try:
        media_tasks = _media_jobs(jobs, session_id)
    except (ValueError, TypeError, KeyError):
        media_tasks = ()
        inv.issues.add("invalid_playback_task_metadata")
        session_reasons.add("invalid_playback_task_metadata")
    bound_media_tasks = set()
    for fact in facts["shots"]:
        shot_id = fact["shot_id"]
        selections = [row for row in facts["selections"] if row["shot_id"] == shot_id]
        shot = {
            "session_id": _public_id(session_id),
            "shot_id": _public_id(shot_id),
            "job_id": None,
            "selection_count": len(selections),
            "selection_complete": selection_complete,
            "protection_reasons": set(session_reasons),
            "objects": [],
            "simulated_unselected_after_review": False,
            "local_discard_status": None,
        }
        inv.shots.append(shot)
        for media_row, media_job in media_tasks:
            if media_job.shot_id == shot_id:
                if media_job.job_id in bound_media_tasks:
                    inv.issue(shot, "duplicate_playback_task")
                bound_media_tasks.add(media_job.job_id)
                inv.playback(shot, media_row, media_job)
        discards = [row for row in facts.get("discards", []) if row["shot_id"] == shot_id]
        if discards:
            shot["protection_reasons"].add("local_discard_registered")
            if (
                len(discards) != 1
                or discards[0]["status"] not in {"pending", "leased", "completed", "blocked"}
                or selections
                or not selection_complete
            ):
                inv.issue(shot, "invalid_discard_metadata")
            else:
                shot["local_discard_status"] = discards[0]["status"]
        if selections:
            shot["protection_reasons"].add("save_intent")
            for selection in selections:
                try:
                    receipt_payload = _json(selection["receipt_json"])
                    receipt_payload["status"] = ClipSaveStatus(receipt_payload["status"])
                    receipt = ClipSaveReceipt(**receipt_payload)
                    if (
                        receipt.request_id,
                        receipt.session_id,
                        receipt.shot_id,
                        receipt.postgame_manifest_sha256,
                    ) != (
                        selection["request_id"],
                        session_id,
                        shot_id,
                        selection["postgame_manifest_sha256"],
                    ):
                        raise ValueError("invalid_selection")
                    if (
                        manifest is None
                        or selection["postgame_manifest_sha256"] != manifest.manifest_sha256
                    ):
                        raise ValueError("invalid_selection")
                    if receipt.status is ClipSaveStatus.SAVED:
                        available = [row for row in facts["available"] if row["shot_id"] == shot_id]
                        if len(available) != 1 or not _sha(available[0]["artifact_id"]):
                            raise ValueError("invalid_selection")
                        output = available[0]
                        job_id = _identity(output["job_id"])
                        expected_clip = session_scoped_clip_id(
                            session_id, shot_id, job_id, output["artifact_id"]
                        )
                        if receipt.clip_id != output["clip_id"] or receipt.clip_id != expected_clip:
                            raise ValueError("invalid_selection")
                except (ValueError, TypeError, KeyError):
                    inv.issue(shot, "invalid_selection_metadata")
        try:
            _identity(shot_id)
            payload = _json(fact["payload_json"])
            job_id = _identity(payload.get("live_job_id"))
            shot["job_id"] = job_id
            if (
                payload.get("session_id") != session_id
                or payload.get("shot_id") != shot_id
                or manifest is None
                or job_id not in manifest.expected_jobs
                or job_id in bound_jobs
                or payload.get("settled_ms") is None
                or not isinstance(payload.get("media"), dict)
                or payload["media"].get("source_kind") != "usb_live"
            ):
                raise ValueError("invalid_shot_lineage")
            bound_jobs.add(job_id)
            job, job_payload = _job(jobs, job_id, session_id, shot_id)
            if job["status"] != "available":
                inv.issue(shot, "job_unresolved")
            media = [row for row in facts["media"] if row["shot_id"] == shot_id]
            if len(media) != 1:
                inv.issue(shot, "media_binding_missing_or_ambiguous")
            else:
                inv.source(shot, job_payload, _json(media[0]["payload_json"]))
            available = [row for row in facts["available"] if row["shot_id"] == shot_id]
            if len(available) != 1:
                inv.issue(shot, "availability_missing_or_ambiguous")
                if job["status"] == "available" and _sha(job["artifact_id"]):
                    inv.object(
                        shot,
                        inv.artifacts,
                        inv.artifacts / job["artifact_id"] / "overlay.mp4",
                        "overlay",
                    )
                continue
            output = available[0]
            if discards and any(
                discard[key] != output[key] for discard in discards for key in output.keys()
            ):
                inv.issue(shot, "invalid_discard_metadata")
            artifact = output["artifact_id"]
            if (
                output["job_id"] != job_id
                or not _sha(artifact)
                or output["postgame_manifest_sha256"] != manifest.manifest_sha256
                or job["artifact_id"] != artifact
                or output["clip_id"]
                != session_scoped_clip_id(session_id, shot_id, job_id, artifact)
                or not _sha(output["content_sha256"])
                or type(output["byte_size"]) is not int
                or output["byte_size"] <= 0
                or output["content_type"] != "video/mp4"
            ):
                raise ValueError("invalid_availability_metadata")
            path = Path(output["local_path"])
            if path != inv.artifacts / artifact / "overlay.mp4":
                inv.issue(shot, "unsafe_media")
            else:
                inv.object(shot, inv.artifacts, path, "overlay", output["byte_size"])
            outbox = _rows(
                state,
                "SELECT save_request_id, status FROM field_saved_media_outbox WHERE clip_id=?",
                (output["clip_id"],),
                limit=1,
            )
            if outbox:
                shot["protection_reasons"].add("saved_outbox")
                if (
                    len(outbox) != 1
                    or outbox[0]["save_request_id"] not in {row["request_id"] for row in selections}
                    or outbox[0]["status"] not in {"pending", "leased", "acknowledged", "blocked"}
                ):
                    inv.issue(shot, "invalid_outbox_metadata")
        except (ValueError, TypeError, KeyError):
            inv.issue(shot, "invalid_or_missing_shot_metadata")
    if manifest and bound_jobs != set(manifest.expected_jobs):
        inv.issues.add("frozen_jobs_unbound")
    if bound_media_tasks != {job.job_id for _, job in media_tasks}:
        inv.issues.add("unbound_playback_tasks")


def build_retention_inventory(
    *,
    state_db: Path,
    jobs_db: Path,
    artifact_root: Path,
    jobs_root: Path,
    limit: int = 100,
) -> dict:
    """Inventory schema17/18/19/20/21 selections and discard state without video reads.

    The two database snapshots and file metadata are not a deletion transaction.
    Unknown objects, session recordings, and video content integrity are not audited.
    """
    if type(limit) is not int or not 1 <= limit <= _MAX_ROWS:
        raise ValueError("invalid_limit")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("unsupported_platform")
    try:
        artifacts, sources = _absolute(artifact_root), _absolute(jobs_root)
        with _directory(artifacts), _directory(sources):
            pass
    except (OSError, ValueError):
        raise ValueError("unsafe_roots") from None
    inv = _Inventory(artifacts, sources)
    scanned = 0
    try:
        with _database(state_db) as state, _database(jobs_db) as jobs:
            inv.state_schema_version = state.execute("PRAGMA user_version").fetchone()[0]
            if inv.state_schema_version not in {17, 18, 19, 20, 21}:
                raise ValueError("unsupported_state_schema")
            jobs.execute(
                "SELECT job_id, payload_json, payload_sha256, status, artifact_id "
                "FROM shot_analysis_jobs LIMIT 0"
            )
            # Find dangling ledger references without scanning media or exposing raw rows.
            for table in (
                "field_postgame_manifests",
                "field_shots",
                "field_live_shot_media",
                "field_postgame_clip_availability",
                "field_clip_save_requests",
                "field_postgame_reviews",
            ) + (
                ("field_local_media_discards",)
                if inv.state_schema_version in {18, 19, 20, 21}
                else ()
            ):
                orphan = state.execute(
                    # Table comes solely from the fixed schema-version tuple above.
                    f"SELECT 1 FROM {table} f LEFT JOIN field_sessions s "  # nosec B608
                    "ON s.session_id=f.session_id WHERE s.session_id IS NULL LIMIT 1"
                ).fetchone()
                if orphan:
                    inv.issues.add("unbound_ledger_rows")
            if state.execute(
                "SELECT 1 FROM field_saved_media_outbox o "
                "LEFT JOIN field_postgame_clip_availability a USING(clip_id) "
                "LEFT JOIN field_clip_save_requests r "
                "ON r.request_id=o.save_request_id "
                "WHERE a.clip_id IS NULL OR r.request_id IS NULL LIMIT 1"
            ).fetchone():
                inv.issues.add("unbound_ledger_rows")
            sessions = _rows(
                state,
                "SELECT session_id, artifact_id, status, ended_at_utc "
                "FROM field_sessions ORDER BY session_id",
                limit=limit,
            )
            if len(sessions) >= limit:
                inv.issues.add("session_limit_reached")
            for session in sessions[:limit]:
                _scan_session(inv, state, jobs, session)
                scanned += 1
    except sqlite3.Error:
        raise ValueError("database_query_failed") from None
    for shot in inv.shots:
        if inv.issues:
            shot["protection_reasons"].add("incomplete_coverage")
        shot["simulated_unselected_after_review"] = bool(
            shot["selection_complete"] and not shot["protection_reasons"]
        )
        shot["protection_reasons"] = sorted(shot["protection_reasons"])
    return {
        "schema_version": "1.0",
        "dry_run": True,
        "deletion_authorized": False,
        "policy_basis": "completed_review_selection_boundary",
        "state_schema_version": inv.state_schema_version,
        "session_limit": limit,
        "ledger_row_limit": _MAX_ROWS,
        "object_limit": _MAX_ROWS,
        "manifest_max_bytes": _MAX_JSON,
        "sessions_scanned": scanned,
        "shots": inv.shots,
        "coverage_complete": not inv.issues,
        "issues": sorted(inv.issues),
        "observed_unique_bytes": sum(
            obj.get("unique_observed_bytes", 0) for shot in inv.shots for obj in shot["objects"]
        ),
        "content_hashes_verified": False,
        "limitations": [
            "bounded_registered_metadata_only",
            "unknown_files_not_scanned",
            "shared_recordings_not_covered",
            "separate_database_snapshots",
            "file_metadata_may_change",
            "no_space_reclaimed",
            "no_deletion_authority",
        ],
    }
