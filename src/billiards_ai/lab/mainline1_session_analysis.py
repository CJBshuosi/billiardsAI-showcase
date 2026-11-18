"""Durable, development-scoped publication of Stage 3 session analysis state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_sync_envelope,
)
from billiards_ai.edge.route_assistance.postgame import (
    PostgameJobSnapshot,
    PostgameSessionManifest,
    SessionAnalysisStatus,
    project_session_analysis,
)
from billiards_ai.edge.route_assistance.records import FieldSessionRecord, FieldShotRecord
from billiards_ai.edge.route_assistance.shot_diagnostics import is_capture_interruption
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    SyncMediaItem,
    SyncOutboxJob,
)
from billiards_ai.lab.field_assistance.cloud_gateway import FieldCloudGateway
from billiards_ai.lab.mainline1_shot_jobs import (
    ShotAnalysisJobSnapshot,
    ShotAnalysisJobStatus,
)
from billiards_ai.shared.models import SessionAnalysisProjectionV1, canonical_sync_hash

_RETRY_DELAY = timedelta(seconds=30)
_PUBLICATION_STATUSES = ("pending", "published")


class SessionAnalysisRecords(Protocol):
    def get_postgame_manifest(self, session_id: str) -> PostgameSessionManifest | None: ...

    def is_postgame_finalized(self, session_id: str) -> bool: ...

    def load_session(self, session_id: str) -> FieldSessionRecord: ...

    def list_shots(self, session_id: str) -> tuple[FieldShotRecord, ...]: ...

    def enqueue_sync_job(
        self,
        *,
        session_id: str,
        manifest_hash: str,
        manifest_json: str,
        media: tuple[SyncMediaItem, ...] = (),
    ) -> SyncOutboxJob: ...

    def claim_due_sync_job_for_session(
        self, session_id: str, now: datetime
    ) -> SyncOutboxJob | None: ...

    def mark_sync_partial(
        self, sync_job_id: str, *, error_code: str, next_attempt_at: datetime
    ) -> SyncOutboxJob: ...

    def mark_sync_complete(self, sync_job_id: str) -> SyncOutboxJob: ...

    def mark_sync_blocked(self, sync_job_id: str, *, error_code: str) -> SyncOutboxJob: ...


class SessionAnalysisJobs(Protocol):
    def snapshot(self, job_id: str) -> ShotAnalysisJobSnapshot: ...


@dataclass(frozen=True, slots=True)
class SessionAnalysisOutboxEntry:
    session_id: str
    revision: int
    payload_sha256: str
    payload: dict[str, object]
    status: Literal["pending", "published"]
    attempt_count: int
    next_attempt_at_utc: datetime
    last_error: str | None


class SQLiteSessionAnalysisOutbox:
    """Append-only revisions whose exact request bytes survive retries and restarts."""

    def __init__(self, database: Path) -> None:
        self._database = database
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_analysis_publications (
                  session_id TEXT NOT NULL,
                  revision INTEGER NOT NULL CHECK(revision > 0),
                  semantic_sha256 TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','published')),
                  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
                  next_attempt_at_utc TEXT NOT NULL,
                  last_error TEXT,
                  PRIMARY KEY(session_id, revision)
                );
                CREATE INDEX IF NOT EXISTS ix_session_analysis_publications_due
                ON session_analysis_publications(session_id, revision, status, next_attempt_at_utc);
                """
            )

    def stage(
        self,
        *,
        session_id: str,
        semantic_payload: dict[str, object],
        now_utc: datetime,
    ) -> SessionAnalysisOutboxEntry:
        """Append only when user-visible semantics changed; polling is idempotent."""
        _require_aware(now_utc)
        semantic_json = _canonical_json(semantic_payload)
        semantic_sha256 = hashlib.sha256(semantic_json.encode()).hexdigest()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                "SELECT * FROM session_analysis_publications WHERE session_id=? "
                "ORDER BY revision DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if latest is not None and latest["semantic_sha256"] == semantic_sha256:
                connection.commit()
                return _entry(latest)
            revision = 1 if latest is None else int(latest["revision"]) + 1
            payload = {
                **semantic_payload,
                "revision": revision,
                "updated_at": _utc_text(now_utc),
            }
            # Validate public semantics before durable staging. Private lineage
            # fields are deliberately rejected by the public Pydantic contract.
            SessionAnalysisProjectionV1.model_validate(_public_payload(payload))
            payload_json = _canonical_json(payload)
            payload_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
            connection.execute(
                "INSERT INTO session_analysis_publications VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    revision,
                    semantic_sha256,
                    payload_sha256,
                    payload_json,
                    "pending",
                    0,
                    _utc_text(now_utc),
                    None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM session_analysis_publications WHERE session_id=? AND revision=?",
                (session_id, revision),
            ).fetchone()
            connection.commit()
        assert row is not None
        return _entry(row)

    def next_due(self, session_id: str, now_utc: datetime) -> SessionAnalysisOutboxEntry | None:
        """Return only the oldest unpublished revision, preserving monotonic delivery."""
        _require_aware(now_utc)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM session_analysis_publications WHERE session_id=? "
                "AND status='pending' ORDER BY revision LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None or _parse_utc(row["next_attempt_at_utc"]) > now_utc.astimezone(UTC):
            return None
        return _entry(row)

    def mark_published(self, entry: SessionAnalysisOutboxEntry) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE session_analysis_publications SET status='published',last_error=NULL "
                "WHERE session_id=? AND revision=? AND status='pending' AND payload_sha256=?",
                (entry.session_id, entry.revision, entry.payload_sha256),
            )
            connection.commit()
        return cursor.rowcount == 1

    def mark_retrying(
        self,
        entry: SessionAnalysisOutboxEntry,
        *,
        now_utc: datetime,
        error: str,
    ) -> bool:
        _require_aware(now_utc)
        message = error.strip() or "session analysis gateway failed"
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE session_analysis_publications SET attempt_count=attempt_count+1,"
                "next_attempt_at_utc=?,last_error=? WHERE session_id=? AND revision=? "
                "AND status='pending' AND payload_sha256=?",
                (
                    _utc_text(now_utc + _RETRY_DELAY),
                    message[:500],
                    entry.session_id,
                    entry.revision,
                    entry.payload_sha256,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def entries(self, session_id: str) -> tuple[SessionAnalysisOutboxEntry, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM session_analysis_publications WHERE session_id=? ORDER BY revision",
                (session_id,),
            ).fetchall()
        return tuple(_entry(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection


@dataclass(slots=True)
class DevelopmentSessionAnalysisPublisher:
    """Project locally and optionally drain through an explicitly injected dev gateway."""

    records: SessionAnalysisRecords
    jobs: SessionAnalysisJobs
    outbox: SQLiteSessionAnalysisOutbox
    gateway: FieldCloudGateway | None = None
    sync_client: FieldSyncClient | None = None
    artifact_root: Path | None = None

    def refresh_and_publish(
        self, session_id: str, now_utc: datetime
    ) -> SessionAnalysisProjectionV1:
        manifest = self.records.get_postgame_manifest(session_id)
        if manifest is None:
            raise ValueError("session analysis requires a frozen postgame manifest")
        snapshots, interrupted_shot_ids = self._snapshots(manifest)
        sync_job = self._sync_session_facts(manifest, now_utc)
        semantic = _semantic_projection(
            manifest,
            snapshots,
            now_utc,
            finalized=self.records.is_postgame_finalized(session_id),
            interrupted_shot_ids=interrupted_shot_ids,
            sync_status=None if sync_job is None else sync_job.status,
        )
        staged = self.outbox.stage(
            session_id=session_id,
            semantic_payload=semantic,
            now_utc=now_utc,
        )
        if self.gateway is not None:
            due = self.outbox.next_due(session_id, now_utc)
            if due is not None:
                try:
                    self.gateway.publish_session_analysis(due.payload)
                except OSError as exc:
                    self.outbox.mark_retrying(due, now_utc=now_utc, error=str(exc))
                else:
                    self.outbox.mark_published(due)
        return SessionAnalysisProjectionV1.model_validate(_public_payload(staged.payload))

    def _snapshots(
        self, manifest: PostgameSessionManifest
    ) -> tuple[tuple[ShotAnalysisJobSnapshot, ...], tuple[str, ...]]:
        shots = self.records.list_shots(manifest.session_id)
        shots_by_job = {shot.live_job_id: shot for shot in shots if shot.live_job_id is not None}
        snapshots: list[ShotAnalysisJobSnapshot] = []
        interrupted = tuple(shot.shot_id for shot in shots if is_capture_interruption(shot))
        for job_id in manifest.expected_jobs:
            shot = shots_by_job.get(job_id)
            if shot is not None and _is_capture_interruption(job_id, shot):
                continue
            snapshots.append(self.jobs.snapshot(job_id))
        return tuple(snapshots), interrupted

    def _sync_session_facts(
        self,
        manifest: PostgameSessionManifest,
        now_utc: datetime,
    ) -> SyncOutboxJob | None:
        # Historical projection-only callers omit both seams. Product worker
        # assembly supplies the artifact root even when it is temporarily
        # offline, so its immutable facts are still staged durably before any
        # optional postgame media exists.
        if self.sync_client is None and self.artifact_root is None:
            return None
        sync_job = self._enqueue_sync(manifest)
        if sync_job.status in {"complete", "blocked"} or self.sync_client is None:
            return sync_job
        claimed = self.records.claim_due_sync_job_for_session(manifest.session_id, now_utc)
        if claimed is None:
            return sync_job
        result = self.sync_client.run_once(claimed)
        if result.outcome == "complete":
            return self.records.mark_sync_complete(claimed.sync_job_id)
        error_code = result.error_code or "SYNC_FAILED"
        if result.outcome == "blocked":
            return self.records.mark_sync_blocked(claimed.sync_job_id, error_code=error_code)
        return self.records.mark_sync_partial(
            claimed.sync_job_id,
            error_code=error_code,
            next_attempt_at=now_utc + _RETRY_DELAY,
        )

    def _enqueue_sync(
        self,
        manifest: PostgameSessionManifest,
    ) -> SyncOutboxJob:
        session_id = manifest.session_id
        report = compose_field_fact_report(
            self.records.load_session(session_id), self.records.list_shots(session_id)
        )
        if any(shot.media.source_kind != "usb_live" for shot in report.shots):
            raise ValueError("development session analysis sync requires USB live evidence")
        envelope = field_sync_envelope(
            report,
            (),
            postgame_manifest_sha256=manifest.manifest_sha256,
            evidence_source_kind="usb_live",
        )
        manifest_json = _canonical_json(envelope.model_dump(mode="json", exclude_none=False))
        return self.records.enqueue_sync_job(
            session_id=session_id,
            manifest_hash=canonical_sync_hash(envelope),
            manifest_json=manifest_json,
            media=(),
        )


def _semantic_projection(
    manifest: PostgameSessionManifest,
    snapshots: tuple[ShotAnalysisJobSnapshot, ...],
    now_utc: datetime,
    *,
    finalized: bool,
    interrupted_shot_ids: tuple[str, ...] = (),
    sync_status: str | None = None,
) -> dict[str, object]:
    _require_aware(now_utc)
    shot_ids = [snapshot.job.shot_id for snapshot in snapshots] + list(interrupted_shot_ids)
    if len(shot_ids) != len(set(shot_ids)):
        raise ValueError("frozen postgame jobs must map one-to-one to shots")
    local_manifest = PostgameSessionManifest(
        session_id=manifest.session_id,
        ended_at_utc=manifest.ended_at_utc,
        deadline_at_utc=manifest.deadline_at_utc,
        expected_jobs=tuple(snapshot.job.job_id for snapshot in snapshots),
    )
    local = project_session_analysis(
        local_manifest,
        tuple(
            PostgameJobSnapshot(
                snapshot.job.job_id,
                snapshot.status.value,
                snapshot.artifact_id,
            )
            for snapshot in snapshots
        ),
        now_utc,
        finalized=finalized,
    )
    available = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.status is ShotAnalysisJobStatus.AVAILABLE and snapshot.artifact_id
    )
    retrying = sum(
        snapshot.status is ShotAnalysisJobStatus.FAILED_RETRYING for snapshot in snapshots
    ) + len(interrupted_shot_ids)
    unresolved = len(shot_ids) - len(available)
    if local.status is SessionAnalysisStatus.FAILED_RETRYING or (
        unresolved and now_utc >= manifest.deadline_at_utc
    ):
        # Once the deadline passes, every unresolved item is represented as
        # retrying even if its local lease was still pending/running.
        retrying = max(retrying, len(shot_ids) - len(available))
    facts_sync_failed = sync_status in {"partial", "blocked"}
    if retrying or facts_sync_failed:
        status = SessionAnalysisStatus.FAILED_RETRYING
    else:
        status = SessionAnalysisStatus.PROCESSING
    if facts_sync_failed:
        retrying = max(retrying, 1)
    # This legacy projection described cloud-published analysis clips. Session
    # facts now publish independently, so local artifact readiness must not be
    # presented as clip publication. The saved-media append path owns that
    # future transition.
    show_published = False
    frozen_digest = hashlib.sha256(_canonical_json(shot_ids).encode()).hexdigest()
    counts: dict[str, object]
    if status is SessionAnalysisStatus.PROCESSING:
        counts = {
            "total_shots": None,
            "available_shots": None,
            "retrying_shots": None,
        }
    else:
        counts = {
            "total_shots": len(shot_ids),
            "available_shots": 0,
            "retrying_shots": retrying,
        }
    return {
        "schema_version": "1.0",
        "session_id": manifest.session_id,
        "status": status.value,
        **counts,
        "report_available": show_published,
        "deadline_at": _utc_text(manifest.deadline_at_utc),
        "frozen_shots_digest": frozen_digest,
        "frozen_shot_ids": shot_ids,
        "published_shots": [],
        "published_clips": [],
    }


def _public_payload(payload: dict[str, object]) -> dict[str, object]:
    private = {
        "frozen_shots_digest",
        "frozen_shot_ids",
        "published_shots",
        "published_clips",
    }
    return {key: value for key, value in payload.items() if key not in private}


def _is_capture_interruption(job_id: str, shot: FieldShotRecord) -> bool:
    """Compatibility wrapper for callers carrying a separate frozen identity."""
    return job_id == shot.live_job_id and is_capture_interruption(shot)


def _entry(row: sqlite3.Row) -> SessionAnalysisOutboxEntry:
    payload_json = str(row["payload_json"])
    payload_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
    if payload_sha256 != row["payload_sha256"]:
        raise ValueError("session analysis outbox payload hash mismatch")
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError("session analysis outbox payload is invalid")
    return SessionAnalysisOutboxEntry(
        session_id=str(row["session_id"]),
        revision=int(row["revision"]),
        payload_sha256=payload_sha256,
        payload=payload,
        status=row["status"],
        attempt_count=int(row["attempt_count"]),
        next_attempt_at_utc=_parse_utc(row["next_attempt_at_utc"]),
        last_error=row["last_error"],
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("session analysis time must be timezone-aware")


def _utc_text(value: datetime) -> str:
    _require_aware(value)
    return value.astimezone(UTC).isoformat(timespec="milliseconds")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require_aware(parsed)
    return parsed.astimezone(UTC)
