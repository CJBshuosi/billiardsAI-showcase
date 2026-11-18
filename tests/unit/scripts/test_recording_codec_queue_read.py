import hashlib
import importlib.util
import pathlib
import sqlite3

import pytest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs/operations/diagnostics/recording-codec-queue-read-2026-11-10.py"
)
S = importlib.util.spec_from_file_location("queue_probe", SOURCE)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)
N = {"__name__": "test_worker"}
exec(M.WORKER, N)


def database(tmp_path):
    p = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(p) as c:
        c.execute(
            "CREATE TABLE shot_analysis_jobs "
            "(status,error_code,attempt_count,terminal_failure,"
            "lease_token,job_kind,payload_json)"
        )
        rows = [
            ("pending", None, 0, 0, None, "media", '{"session_id":"' + N["SESSION"] + '"}'),
            (
                "failed_retrying",
                "MEDIA_ENCODING_FAILED",
                2,
                0,
                "PRIVATE_TOKEN",
                "media",
                '{"session_id":"' + N["SESSION"] + '"}',
            ),
            ("pending", "PRIVATE_ERROR", 1, 0, None, "media", '{"session_id":"FOREIGN"}'),
        ]
        c.executemany("INSERT INTO shot_analysis_jobs VALUES (?,?,?,?,?,?,?)", rows)
    return p


def test_real_read_is_scoped_sanitized_and_does_not_write(tmp_path):
    p = database(tmp_path)
    before = p.read_bytes()
    files = {x.name for x in tmp_path.iterdir()}
    value = N["inspect_jobs"](p)
    assert value["job_count"] == 2 and value["status_counts"]["failed_retrying"] == 1
    assert value["maximum_attempt"] == 2 and value["claim_count"] == 1
    assert value["error_code_counts"] == {"NONE": 1, "MEDIA_ENCODING_FAILED": 1}
    assert value["database_sha256"] == hashlib.sha256(before).hexdigest()
    assert p.read_bytes() == before and {x.name for x in tmp_path.iterdir()} == files
    assert "PRIVATE" not in str(value) and "FOREIGN" not in str(value)


def test_nonempty_wal_is_rejected_without_querying_stale_main(tmp_path):
    p = database(tmp_path)
    p.with_name(p.name + "-wal").write_bytes(b"PRIVATE WAL")
    with pytest.raises(ValueError, match="WAL_PRESENT"):
        N["inspect_jobs"](p)


def test_unknown_error_is_not_returned(tmp_path):
    p = database(tmp_path)
    with sqlite3.connect(p) as c:
        c.execute(
            "UPDATE shot_analysis_jobs SET error_code='PRIVATE SENTINEL' WHERE error_code IS NULL"
        )
    r = N["inspect_jobs"](p)
    assert r["error_code_counts"]["UNCLASSIFIED"] == 1 and "PRIVATE" not in str(r)


def test_inventory_overrun_is_rejected(tmp_path):
    p = database(tmp_path)
    with sqlite3.connect(p) as c:
        c.executemany(
            "INSERT INTO shot_analysis_jobs VALUES (?,?,?,?,?,?,?)",
            [("pending", None, 0, 0, None, "media", '{"session_id":"' + N["SESSION"] + '"}')] * 28,
        )
    with pytest.raises(ValueError, match="QUEUE_INVENTORY"):
        N["inspect_jobs"](p)
