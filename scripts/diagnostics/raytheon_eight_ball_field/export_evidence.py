#!/usr/bin/env python3
"""Export one ended field session without mutating either SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _readonly(database: Path) -> sqlite3.Connection:
    path = database.resolve()
    if not path.is_file():
        raise ValueError(f"evidence database is missing: {path}")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _json(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else {key: row[key] for key in row.keys()}


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("postgame deadline must include a timezone")
    return parsed.astimezone(UTC)


def export_evidence(
    state_db: Path,
    jobs_db: Path,
    session_id: str,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible, session-scoped evidence projection."""

    if not session_id.strip():
        raise ValueError("session ID must not be empty")
    now = datetime.now(UTC) if now_utc is None else now_utc.astimezone(UTC)
    with _readonly(state_db) as state:
        session = state.execute(
            "SELECT * FROM field_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if session is None:
            raise ValueError("field session does not exist")
        match_row = state.execute(
            "SELECT payload_json, revision FROM field_match_state WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if match_row is None:
            raise ValueError("persisted match state is missing")
        shots = state.execute(
            "SELECT shot_id, sequence, payload_json FROM field_shots "
            "WHERE session_id=? ORDER BY sequence",
            (session_id,),
        ).fetchall()
        manifest = state.execute(
            "SELECT ended_at_utc, deadline_at_utc, expected_jobs_json "
            "FROM field_postgame_manifests WHERE session_id=?",
            (session_id,),
        ).fetchone()
        finalization = state.execute(
            "SELECT manifest_sha256, manifest_json FROM field_postgame_finalizations "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()
        outbox = state.execute(
            "SELECT manifest_sha256, status FROM local_postgame_outbox WHERE session_id=?",
            (session_id,),
        ).fetchone()

    expected = () if manifest is None else tuple(_json(manifest["expected_jobs_json"]))
    if len(expected) != len(set(expected)) or any(
        not isinstance(item, str) or not item for item in expected
    ):
        raise ValueError("postgame expected job inventory is invalid")
    with _readonly(jobs_db) as jobs:
        snapshots = []
        for job_id in expected:
            item = jobs.execute(
                "SELECT job_id, status, artifact_id, failure_reason, attempt_count "
                "FROM shot_analysis_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            snapshots.append(
                {
                    "job_id": job_id,
                    "status": "missing" if item is None else item["status"],
                    "artifact_id": None if item is None else item["artifact_id"],
                    "failure_reason": None if item is None else item["failure_reason"],
                    "attempt_count": 0 if item is None else item["attempt_count"],
                }
            )

    available = sum(
        item["status"] == "available" and bool(item["artifact_id"]) for item in snapshots
    )
    finalized = finalization is not None
    if manifest is not None and available == len(expected) and finalized:
        status = "AVAILABLE"
    elif available:
        status = "PARTIAL"
    elif manifest is not None and (
        any(item["status"] == "failed_retrying" for item in snapshots)
        or now >= _utc(manifest["deadline_at_utc"])
    ):
        status = "FAILED_RETRYING"
    else:
        status = "PROCESSING"

    match = _json(match_row["payload_json"])
    return {
        "schema_version": "1.0",
        "exported_at_utc": now.isoformat().replace("+00:00", "Z"),
        "session": _row(session),
        "match": match,
        "match_revision": match_row["revision"],
        "shots": [
            {
                "shot_id": item["shot_id"],
                "sequence": item["sequence"],
                "payload": _json(item["payload_json"]),
            }
            for item in shots
        ],
        "postgame_manifest": None
        if manifest is None
        else {
            "ended_at_utc": manifest["ended_at_utc"],
            "deadline_at_utc": manifest["deadline_at_utc"],
            "expected_jobs": list(expected),
        },
        "jobs": snapshots,
        "analysis": {
            "status": status,
            "expected_count": len(expected),
            "available_count": available,
            "finalized": finalized,
        },
        "finalization": None
        if finalization is None
        else {
            "manifest_sha256": finalization["manifest_sha256"],
            "manifest": _json(finalization["manifest_json"]),
        },
        "local_postgame_outbox": _row(outbox),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--jobs-db", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = export_evidence(args.state_db, args.jobs_db, args.session_id)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (OSError, ValueError, sqlite3.DatabaseError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
