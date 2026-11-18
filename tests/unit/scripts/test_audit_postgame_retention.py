from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/algorithm/audit_postgame_retention.py"


def run(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_works_outside_checkout_and_has_no_deletion_mode(tmp_path):
    result = run(tmp_path, "--help")
    assert result.returncode == 0
    assert "--unsaved-retention-hours" not in result.stdout
    assert "--as-of" not in result.stdout
    assert "--delete" not in result.stdout
    assert "--apply" not in result.stdout


def test_failure_is_one_sanitized_json_report_and_does_not_create_databases(tmp_path):
    secret = tmp_path / "credential-private.sqlite"
    result = run(
        tmp_path,
        "--state-db",
        str(secret),
        "--jobs-db",
        str(secret),
        "--artifact-root",
        str(tmp_path),
        "--jobs-root",
        str(tmp_path),
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["deletion_authorized"] is False
    assert report["error"] == "database_unavailable"
    assert str(tmp_path) not in result.stdout + result.stderr
    assert "credential-private" not in result.stdout + result.stderr
    assert not secret.exists()


def test_unsupported_arguments_are_rejected_without_path_leaks(tmp_path):
    for args in [("--apply", str(tmp_path)), ("--as-of", "2026-10-29T12:00:00")]:
        result = run(tmp_path, *args)
        assert result.returncode == 2
        assert str(tmp_path) not in result.stdout + result.stderr


def test_empty_inventory_outside_checkout_rejects_ttl_asof_and_deletion_flags(tmp_path):
    state = tmp_path / "field.sqlite"
    jobs = tmp_path / "jobs.sqlite"
    with sqlite3.connect(state) as db:
        db.executescript("""
            PRAGMA user_version=17;
            CREATE TABLE field_sessions(session_id, artifact_id, status, ended_at_utc);
            CREATE TABLE field_postgame_manifests(session_id);
            CREATE TABLE field_shots(session_id);
            CREATE TABLE field_live_shot_media(session_id);
            CREATE TABLE field_postgame_clip_availability(session_id, clip_id);
            CREATE TABLE field_clip_save_requests(session_id, request_id);
            CREATE TABLE field_saved_media_outbox(clip_id, save_request_id);
            CREATE TABLE field_postgame_reviews(session_id, artifact_id, status);
        """)
    with sqlite3.connect(jobs) as db:
        db.execute(
            "CREATE TABLE shot_analysis_jobs("
            "job_id, payload_json, payload_sha256, status, artifact_id)"
        )
    arguments = (
        "--state-db",
        str(state),
        "--jobs-db",
        str(jobs),
        "--artifact-root",
        str(tmp_path),
        "--jobs-root",
        str(tmp_path),
    )
    result = run(tmp_path, *arguments)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["sessions_scanned"] == 0
    assert report["policy_basis"] == "completed_review_selection_boundary"
    assert report["coverage_complete"] is True
    for extra in [
        ("--apply",),
        ("--delete",),
        ("--as-of", "2026-10-29T00:00:00Z"),
        ("--unsaved-retention-hours", "48"),
    ]:
        rejected = run(tmp_path, *arguments, *extra)
        assert rejected.returncode == 2
