from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from billiards_ai.edge.route_assistance.postgame import session_scoped_clip_id
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    ShotMediaEncodeJobV1,
)

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)
ARTIFACT = "a" * 64
DIGEST = "b" * 64


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def interval_digest(interval):
    identity = {
        key: interval[key]
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
    identity["segments"] = [
        {"name": name, "sha256": digest} for name, digest in interval["segment_sha256"].items()
    ]
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


@pytest.fixture
def ledger(tmp_path):
    state = tmp_path / "field.sqlite"
    jobs = tmp_path / "jobs.sqlite"
    artifacts = tmp_path / "artifacts"
    sources = tmp_path / "jobs"
    artifacts.mkdir()
    sources.mkdir()
    with sqlite3.connect(state) as db:
        db.executescript("""
            PRAGMA user_version=17;
            CREATE TABLE field_sessions(session_id, artifact_id, status, ended_at_utc);
            CREATE TABLE field_postgame_manifests(
                session_id, ended_at_utc, deadline_at_utc, expected_jobs_json);
            CREATE TABLE field_shots(session_id, shot_id, sequence, payload_json);
            CREATE TABLE field_live_shot_media(session_id, shot_id, payload_json);
            CREATE TABLE field_postgame_clip_availability(
                session_id, shot_id, postgame_manifest_sha256, job_id, artifact_id,
                clip_id, local_path, content_sha256, byte_size, content_type);
            CREATE TABLE field_clip_save_requests(
                request_id, session_id, shot_id, postgame_manifest_sha256, receipt_json);
            CREATE TABLE field_saved_media_outbox(clip_id, save_request_id, status, attempts);
            CREATE TABLE field_postgame_reviews(session_id, artifact_id, status);
        """)
    with sqlite3.connect(jobs) as db:
        db.execute(
            "CREATE TABLE shot_analysis_jobs("
            "job_id, payload_json, payload_sha256, status, artifact_id)"
        )

    def add(
        session="session-a",
        *,
        selection=None,
        available=True,
        job_status="available",
        review="completed",
        artifact=ARTIFACT,
    ):
        job = f"job-{session}"
        shot = "shot-1"
        ended = (NOW - timedelta(hours=48)).isoformat()
        deadline = (NOW - timedelta(hours=48) + timedelta(seconds=120)).isoformat()
        manifest = {
            "session_id": session,
            "ended_at_utc": ended,
            "deadline_at_utc": deadline,
            "expected_jobs": [job],
        }
        digest = hashlib.sha256(canonical(manifest).encode()).hexdigest()
        media_root = sources / "media" / session / job
        media_root.mkdir(parents=True)
        (media_root / "segment-1.mjpeg").write_bytes(b"source-video")
        interval = {
            "schema_version": "1.0",
            "source_epoch_id": "epoch-1",
            "width": 640,
            "height": 480,
            "frame_count": 2,
            "start_frame_id": 1,
            "end_frame_id": 2,
            "media_sha256": DIGEST,
            "segment_sha256": {"segment-1.mjpeg": DIGEST},
        }
        interval["media_sha256"] = interval_digest(interval)
        (media_root / "interval.json").write_text(canonical(interval))
        payload = {
            "schema_version": "1.0",
            "job_id": job,
            "session_id": session,
            "shot_id": shot,
            "media_root": str(media_root),
            "source_epoch_id": "epoch-1",
            "media_sha256": interval["media_sha256"],
            "start_frame_id": 1,
            "end_frame_id": 2,
            "capture_setup_id": "setup",
            "calibration_artifact_id": "calibration",
            "config_sha256": DIGEST,
            "model_package_id": "model",
            "identity_training_package_id": None,
            "identity_training_package_sha256": None,
        }
        clip = session_scoped_clip_id(session, shot, job, artifact)
        output = artifacts / artifact / "overlay.mp4"
        output.parent.mkdir(exist_ok=True)
        if available:
            output.write_bytes(b"overlay")
        with sqlite3.connect(state) as db:
            db.execute(
                "INSERT INTO field_sessions VALUES(?,?,?,?)",
                (session, "field-artifact", "ended", ended),
            )
            if review is not None:
                db.execute(
                    "INSERT INTO field_postgame_reviews VALUES(?,?,?)",
                    (session, "field-artifact", review),
                )
            db.execute(
                "INSERT INTO field_postgame_manifests VALUES(?,?,?,?)",
                (session, ended, deadline, canonical([job])),
            )
            db.execute(
                "INSERT INTO field_shots VALUES(?,?,?,?)",
                (
                    session,
                    shot,
                    1,
                    canonical(
                        {
                            "shot_id": shot,
                            "session_id": session,
                            "live_job_id": job,
                            "settled_ms": 100,
                            "media": {"source_kind": "usb_live"},
                        }
                    ),
                ),
            )
            db.execute(
                "INSERT INTO field_live_shot_media VALUES(?,?,?)",
                (
                    session,
                    shot,
                    canonical(
                        {
                            "session_id": session,
                            "shot_id": shot,
                            "job_id": job,
                            "source_epoch_id": "epoch-1",
                            "media_sha256": interval["media_sha256"],
                            "media_path": str(media_root),
                            "match_revision": 1,
                        }
                    ),
                ),
            )
            if available:
                db.execute(
                    "INSERT INTO field_postgame_clip_availability VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        session,
                        shot,
                        digest,
                        job,
                        artifact,
                        clip,
                        str(output),
                        DIGEST,
                        7,
                        "video/mp4",
                    ),
                )
            if selection:
                db.execute(
                    "INSERT INTO field_clip_save_requests VALUES(?,?,?,?,?)",
                    (
                        "request-1",
                        session,
                        shot,
                        digest,
                        canonical(
                            {
                                "request_id": "request-1",
                                "session_id": session,
                                "shot_id": shot,
                                "postgame_manifest_sha256": digest,
                                "clip_id": clip if available else None,
                                "status": "SAVE_PENDING" if not available else "SAVED",
                            }
                        ),
                    ),
                )
                if available:
                    db.execute(
                        "INSERT INTO field_saved_media_outbox VALUES(?,?,?,?)",
                        (clip, "request-1", selection, 2),
                    )
        encoded = canonical(payload)
        with sqlite3.connect(jobs) as db:
            db.execute(
                "INSERT INTO shot_analysis_jobs VALUES(?,?,?,?,?)",
                (
                    job,
                    encoded,
                    hashlib.sha256(encoded.encode()).hexdigest(),
                    job_status,
                    artifact if job_status == "available" else None,
                ),
            )
        return media_root, output

    return {
        "state_db": state,
        "jobs_db": jobs,
        "artifact_root": artifacts,
        "jobs_root": sources,
        "add": add,
    }


def inventory(ledger, **overrides):
    module = importlib.util.find_spec("billiards_ai.edge.route_assistance.retention_inventory")
    assert module is not None, "read-only retention inventory is not implemented"
    build = importlib.import_module(module.name).build_retention_inventory
    args = {key: value for key, value in ledger.items() if key != "add"}
    return build(**(args | overrides))


def test_inventory_accounts_for_typed_playback_without_calling_it_canonical(ledger):
    media_root, _ = ledger["add"]()
    job = ShotMediaEncodeJobV1.create(
        session_id="session-a",
        shot_id="shot-1",
        source_epoch_id="epoch-1",
        start_frame_id=1,
        end_frame_id=2,
        media_root=media_root,
        media_sha256=json.loads((media_root / "interval.json").read_text())["media_sha256"],
        source=LegacyFrozenJobMediaSourceV1(
            kind="legacy_frozen_job",
            postgame_manifest_sha256="c" * 64,
            source_job_id="job-session-a",
            source_job_payload_sha256="d" * 64,
        ),
    )
    artifact_id = "recording-playback-" + "e" * 64
    output = ledger["artifact_root"] / "postgame" / job.session_id / artifact_id
    output.mkdir(parents=True)
    media = output / "original.mp4"
    media.write_bytes(b"typed-playback")
    manifest = {
        "schema_version": "recording-playback-artifact/v1",
        "artifact_id": artifact_id,
        "job": job.identity(),
        "source_interval": {},
        "output": {
            "name": "original.mp4",
            "content_type": "video/mp4",
            "content_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "byte_size": media.stat().st_size,
        },
    }
    (output / "manifest.json").write_text(canonical(manifest))
    encoded = canonical(job.identity())
    with sqlite3.connect(ledger["jobs_db"]) as db:
        db.execute("ALTER TABLE shot_analysis_jobs ADD COLUMN job_kind")
        db.execute(
            "INSERT INTO shot_analysis_jobs("
            "job_id,payload_json,payload_sha256,status,artifact_id,job_kind) "
            "VALUES(?,?,?,?,?,?)",
            (
                job.job_id,
                encoded,
                hashlib.sha256(encoded.encode()).hexdigest(),
                "available",
                artifact_id,
                "media",
            ),
        )

    report = inventory(ledger)

    objects = report["shots"][0]["objects"]
    playback = [obj for obj in objects if obj["kind"] == "recording_playback"]
    assert len(playback) == 1
    assert playback[0]["status"] == "observed"
    assert not any(obj["kind"] == "overlay" and obj == playback[0] for obj in objects)


@pytest.mark.parametrize("status", ["pending", "leased", "completed", "blocked"])
@pytest.mark.parametrize("version", [18, 19, 20, 21])
def test_schema18_through21_report_discard_state_without_deletion_authority(
    ledger, status, version
):
    from billiards_ai.edge.route_assistance.persistence import _LOCAL_DISCARD_SCHEMA

    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.executescript(_LOCAL_DISCARD_SCHEMA + f"PRAGMA user_version={version};")
        if version in {19, 20, 21}:
            from billiards_ai.edge.route_assistance.persistence import (
                _MANAGED_OPEN_ACTIVE_INDEXES,
                _MANAGED_OPEN_SCHEMA,
                _MANAGED_OPEN_SCHEMA20,
                _MANAGED_RESOURCE_SCHEMA21,
            )

            db.executescript(
                _MANAGED_OPEN_SCHEMA
                if version == 19
                else (_MANAGED_OPEN_SCHEMA20 + _MANAGED_OPEN_ACTIVE_INDEXES)
            )
            db.execute(
                "INSERT INTO field_managed_open_requests "
                "(environment_id,device_id,table_id,artifact_id,rule_profile_id,open_request_id,"
                "phase,session_id,birth_id,birth_contract,remote_status,created_at_utc,updated_at_utc"
                + (",reservation_order" if version >= 20 else "")
                + ") VALUES "
                "('dev','device-1','table-1','field-artifact',NULL,"
                "'11111111-1111-4111-8111-111111111111','bound',"
                "'managed-22222222-2222-4222-8222-222222222222',"
                "'22222222-2222-4222-8222-222222222222','managed_session_birth_v1','ended',"
                "'2026-10-30T00:00:00Z','2026-10-30T00:00:00Z'" + (",1)" if version >= 20 else ")")
            )
        if version == 21:
            db.executescript(_MANAGED_RESOURCE_SCHEMA21)
        db.execute(
            "INSERT INTO field_local_media_discards "
            "(session_id,shot_id,postgame_manifest_sha256,job_id,artifact_id,clip_id,"
            "local_path,content_sha256,byte_size,content_type,status,attempts,next_attempt_at_utc) "
            "SELECT a.*,?,0,? FROM field_postgame_clip_availability a",
            (status, NOW.isoformat()),
        )
    result = inventory(ledger)
    assert result["state_schema_version"] == version
    assert result["shots"][0]["local_discard_status"] == status
    assert result["deletion_authorized"] is False and result["dry_run"] is True


@pytest.mark.parametrize(
    ("review", "complete"),
    [("completed", True), ("active", False), ("pending", False), (None, False), ("corrupt", False)],
)
def test_only_completed_review_finishes_selection_and_both_video_copies(ledger, review, complete):
    ledger["add"](review=review)
    report = inventory(ledger)
    shot = report["shots"][0]
    assert shot["selection_complete"] is complete
    assert shot["simulated_unselected_after_review"] is complete
    assert report["observed_unique_bytes"] == 19
    assert {obj["kind"] for obj in shot["objects"]} == {"overlay", "copied_source"}
    assert report["dry_run"] is True
    assert report["deletion_authorized"] is False
    assert str(ledger["state_db"].parent) not in json.dumps(report)


@pytest.mark.parametrize("selection", ["pending", "leased", "acknowledged", "blocked"])
def test_every_selection_is_protected_even_retrying_or_blocked(ledger, selection):
    ledger["add"](selection=selection)
    shot = inventory(ledger)["shots"][0]
    assert "save_intent" in shot["protection_reasons"]
    assert shot["simulated_unselected_after_review"] is False


def test_selection_before_availability_and_without_outbox_is_protected(ledger):
    ledger["add"](selection="pending", available=False)
    report = inventory(ledger)
    assert "save_intent" in report["shots"][0]["protection_reasons"]
    assert report["observed_unique_bytes"] == 12
    assert report["coverage_complete"] is False


@pytest.mark.parametrize("status", ["pending", "running", "failed_retrying"])
def test_unresolved_jobs_are_protected(ledger, status):
    ledger["add"](available=False, job_status=status)
    shot = inventory(ledger)["shots"][0]
    assert "job_unresolved" in shot["protection_reasons"]
    assert shot["simulated_unselected_after_review"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_json",
        "bad_digest",
        "wrong_shot",
        "missing_job",
        "wrong_manifest",
        "unbound_selection",
        "open_session",
    ],
)
def test_corrupt_or_unbound_ledger_fails_closed(ledger, mutation):
    ledger["add"]()
    if mutation in {"bad_json", "bad_digest", "wrong_shot", "missing_job"}:
        with sqlite3.connect(ledger["jobs_db"]) as db:
            if mutation == "bad_json":
                db.execute("UPDATE shot_analysis_jobs SET payload_json='{private-token'")
            elif mutation == "bad_digest":
                db.execute("UPDATE shot_analysis_jobs SET payload_sha256=?", ("0" * 64,))
            elif mutation == "missing_job":
                db.execute("DELETE FROM shot_analysis_jobs")
            else:
                payload = json.loads(
                    db.execute("SELECT payload_json FROM shot_analysis_jobs").fetchone()[0]
                )
                payload["shot_id"] = "another-shot"
                encoded = canonical(payload)
                db.execute(
                    "UPDATE shot_analysis_jobs SET payload_json=?, payload_sha256=?",
                    (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
                )
    else:
        with sqlite3.connect(ledger["state_db"]) as db:
            if mutation == "wrong_manifest":
                db.execute(
                    "UPDATE field_postgame_clip_availability SET postgame_manifest_sha256='wrong'"
                )
            elif mutation == "unbound_selection":
                db.execute(
                    "INSERT INTO field_clip_save_requests VALUES('r','session-a',"
                    "'unknown','bad','{}')"
                )
            else:
                db.execute("UPDATE field_sessions SET status='open', ended_at_utc=NULL")
    report = inventory(ledger)
    assert report["coverage_complete"] is False
    assert all(not shot["simulated_unselected_after_review"] for shot in report["shots"])
    assert "private-token" not in json.dumps(report)


@pytest.mark.parametrize(
    "unsafe",
    [
        "traversal",
        "symlink",
        "directory",
        "hardlink",
        "shared",
        "oversize_manifest",
        "bad_interval",
    ],
)
def test_unsafe_or_unsupported_media_never_becomes_candidate(ledger, unsafe, tmp_path):
    media, output = ledger["add"]()
    if unsafe == "traversal":
        with sqlite3.connect(ledger["state_db"]) as db:
            db.execute(
                "UPDATE field_postgame_clip_availability SET local_path=?",
                (str(output.parent / ".." / ARTIFACT / "overlay.mp4"),),
            )
    elif unsafe == "symlink":
        output.unlink()
        output.symlink_to(media / "segment-1.mjpeg")
    elif unsafe == "directory":
        output.unlink()
        output.mkdir()
    elif unsafe == "hardlink":
        (tmp_path / "alias").hardlink_to(output)
    elif unsafe == "shared":
        interval = json.loads((media / "interval.json").read_text())
        interval["recording_root"] = str(tmp_path / "private-recording")
        (media / "interval.json").write_text(canonical(interval))
    elif unsafe == "oversize_manifest":
        (media / "interval.json").write_bytes(b" " * (1024 * 1024 + 1))
    else:
        (media / "interval.json").write_text('{"secret":"private-token"}')
    report = inventory(ledger)
    assert report["coverage_complete"] is False
    assert report["shots"][0]["simulated_unselected_after_review"] is False
    assert "private-recording" not in json.dumps(report)


def test_duplicate_objects_are_counted_once_and_query_limit_is_incomplete(ledger):
    ledger["add"]()
    ledger["add"]("session-b")
    report = inventory(ledger)
    assert report["observed_unique_bytes"] == 31
    assert report["coverage_complete"] is False
    limited = inventory(ledger, limit=1)
    assert limited["sessions_scanned"] == 1
    assert "session_limit_reached" in limited["issues"]
    assert limited["coverage_complete"] is False


def test_missing_overlay_keeps_source_visible_and_blocks_candidate(ledger):
    _, output = ledger["add"]()
    output.unlink()
    report = inventory(ledger)
    assert report["observed_unique_bytes"] == 12
    assert report["coverage_complete"] is False


def test_read_only_inventory_sees_wal_and_preserves_inputs(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO field_clip_save_requests VALUES(?,?,?,?,?)",
            ("wal-save", "session-a", "shot-1", "unknown", "{}"),
        )
        writer.commit()
        before = {
            path: path.read_bytes()
            for path in ledger["state_db"].parent.rglob("*")
            if path.is_file() and not path.name.endswith("-shm")
        }
        report = inventory(ledger)
        after = {path: path.read_bytes() for path in before}
        assert before == after
        assert "save_intent" in report["shots"][0]["protection_reasons"]


def test_missing_database_is_not_created_and_unknown_schema_rejected(ledger):
    missing = ledger["state_db"].parent / "missing.sqlite"
    with pytest.raises(ValueError, match="database_unavailable"):
        inventory(ledger, state_db=missing)
    assert not missing.exists()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("PRAGMA user_version=16")
    with pytest.raises(ValueError, match="unsupported_state_schema"):
        inventory(ledger)


def test_multiple_references_to_same_inode_are_counted_once(ledger):
    media, _ = ledger["add"]()
    (media / "segment-2.mjpeg").hardlink_to(media / "segment-1.mjpeg")
    interval = json.loads((media / "interval.json").read_text())
    interval["segment_sha256"]["segment-2.mjpeg"] = DIGEST
    interval["media_sha256"] = interval_digest(interval)
    (media / "interval.json").write_text(canonical(interval))
    with sqlite3.connect(ledger["jobs_db"]) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM shot_analysis_jobs").fetchone()[0]
        )
        payload["media_sha256"] = interval["media_sha256"]
        encoded = canonical(payload)
        db.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?, payload_sha256=?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    with sqlite3.connect(ledger["state_db"]) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM field_live_shot_media").fetchone()[0]
        )
        payload["media_sha256"] = interval["media_sha256"]
        db.execute("UPDATE field_live_shot_media SET payload_json=?", (canonical(payload),))
    report = inventory(ledger)
    assert report["observed_unique_bytes"] == 19
    assert report["coverage_complete"] is False


def test_orphan_save_for_unknown_session_makes_coverage_incomplete(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("INSERT INTO field_clip_save_requests VALUES('r','missing','shot','bad','{}')")
    report = inventory(ledger)
    assert "unbound_ledger_rows" in report["issues"]
    assert report["coverage_complete"] is False
    assert report["shots"][0]["simulated_unselected_after_review"] is False


def test_malformed_save_receipt_still_protects_and_marks_incomplete(ledger):
    ledger["add"](selection="acknowledged")
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("UPDATE field_clip_save_requests SET receipt_json='{}'")
    report = inventory(ledger)
    assert "save_intent" in report["shots"][0]["protection_reasons"]
    assert report["coverage_complete"] is False


def test_registered_job_overlay_is_visible_before_availability_commit(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("DELETE FROM field_postgame_clip_availability")
    report = inventory(ledger)
    assert report["observed_unique_bytes"] == 19
    assert report["coverage_complete"] is False


def test_numeric_persisted_timestamp_fails_closed(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("UPDATE field_postgame_manifests SET ended_at_utc=123")
    assert inventory(ledger)["coverage_complete"] is False


def test_video_files_are_never_opened(ledger, monkeypatch):
    import os

    ledger["add"]()
    original = os.open

    def metadata_only(path, flags, *args, **kwargs):
        assert not str(path).endswith((".mp4", ".mjpeg")), "video bytes must never be opened"
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", metadata_only)
    assert inventory(ledger)["observed_unique_bytes"] == 19


def test_incomplete_sql_schema_is_sanitized(ledger):
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("DROP TABLE field_sessions")
    with pytest.raises(ValueError, match="database_query_failed"):
        inventory(ledger)


def test_shot_payload_identity_must_match_ledger_row(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        row = json.loads(db.execute("SELECT payload_json FROM field_shots").fetchone()[0])
        row["shot_id"] = "other-shot"
        db.execute("UPDATE field_shots SET payload_json=?", (canonical(row),))
    assert inventory(ledger)["coverage_complete"] is False


def test_ledger_row_budget_limits_output_and_marks_incomplete(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.executemany(
            "INSERT INTO field_clip_save_requests VALUES(?,?,?,?,?)",
            ((f"r-{index}", "session-a", "shot-1", "bad", "{}") for index in range(10001)),
        )
    report = inventory(ledger)
    assert "ledger_row_limit_reached" in report["issues"]
    assert report["shots"][0]["selection_count"] < 10001
    assert report["shots"][0]["simulated_unselected_after_review"] is False


def test_archive_and_symlink_roots_are_rejected_without_scanning(ledger, tmp_path):
    archive = tmp_path / ".project-archive"
    # It need not exist: the lexical guard must reject it before opening directories.
    with pytest.raises(ValueError, match="unsafe_roots"):
        inventory(ledger, artifact_root=archive)
    link = tmp_path / "link"
    link.symlink_to(ledger["artifact_root"], target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe_roots"):
        inventory(ledger, artifact_root=link)


def test_changed_segment_digest_cannot_reuse_frozen_media_identity(ledger):
    media, _ = ledger["add"]()
    manifest = json.loads((media / "interval.json").read_text())
    manifest["segment_sha256"]["segment-1.mjpeg"] = "0" * 64
    (media / "interval.json").write_text(canonical(manifest))
    report = inventory(ledger)
    assert report["coverage_complete"] is False
    assert report["shots"][0]["simulated_unselected_after_review"] is False


def test_completed_review_must_match_session_artifact(ledger):
    ledger["add"]()
    with sqlite3.connect(ledger["state_db"]) as db:
        db.execute("UPDATE field_postgame_reviews SET artifact_id='other-artifact'")
    report = inventory(ledger)
    assert report["shots"][0]["selection_complete"] is False
    assert report["coverage_complete"] is False


def test_unsupported_directory_safety_primitives_fail_without_traceback(ledger, monkeypatch):
    import os

    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="unsupported_platform"):
        inventory(ledger)


@pytest.mark.parametrize("overrides", [{"limit": 0}, {"limit": 10001}])
def test_inventory_limits_are_validated(ledger, overrides):
    with pytest.raises(ValueError):
        inventory(ledger, **overrides)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_package_id", None),
        ("model_package_id", ""),
        ("model_package_id", True),
        ("capture_setup_id", ""),
        ("calibration_artifact_id", None),
        ("identity_training_package_id", "training-without-digest"),
        ("identity_training_package_sha256", "0" * 64),
        ("unexpected_field", "unknown"),
    ],
)
def test_incomplete_canonical_job_identity_is_not_a_cleanup_candidate(ledger, field, value):
    ledger["add"]()
    with sqlite3.connect(ledger["jobs_db"]) as db:
        payload = json.loads(
            db.execute("SELECT payload_json FROM shot_analysis_jobs").fetchone()[0]
        )
        if value is None:
            payload.pop(field)
        else:
            payload[field] = value
        encoded = canonical(payload)
        db.execute(
            "UPDATE shot_analysis_jobs SET payload_json=?, payload_sha256=?",
            (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    report = inventory(ledger)
    assert report["coverage_complete"] is False
    assert report["shots"][0]["simulated_unselected_after_review"] is False


@pytest.mark.parametrize(
    ("status", "clip_id"),
    [
        ("SAVED", "clip-from-another-session"),
        ("SAVED", None),
        ("SAVE_PENDING", "clip-should-be-null"),
    ],
)
def test_misbound_receipt_protects_selection_and_blocks_other_candidates(ledger, status, clip_id):
    ledger["add"](selection="acknowledged", artifact="c" * 64)
    ledger["add"]("session-b")
    with sqlite3.connect(ledger["state_db"]) as db:
        receipt = json.loads(
            db.execute("SELECT receipt_json FROM field_clip_save_requests").fetchone()[0]
        )
        receipt.update(status=status, clip_id=clip_id)
        db.execute("UPDATE field_clip_save_requests SET receipt_json=?", (canonical(receipt),))
    report = inventory(ledger)
    selected, unselected = report["shots"]
    assert "save_intent" in selected["protection_reasons"]
    assert "invalid_selection_metadata" in report["issues"]
    assert report["coverage_complete"] is False
    assert unselected["selection_count"] == 0
    assert unselected["simulated_unselected_after_review"] is False


@pytest.mark.parametrize("status", ["active", "pending"])
def test_unfinished_review_makes_all_coverage_incomplete(ledger, status):
    ledger["add"](review=status, artifact="c" * 64)
    ledger["add"]("session-b")
    report = inventory(ledger)
    assert "selection_unfinished" in report["issues"]
    assert report["coverage_complete"] is False
    assert "selection_unfinished" in report["shots"][0]["protection_reasons"]
    assert all(not shot["simulated_unselected_after_review"] for shot in report["shots"])
