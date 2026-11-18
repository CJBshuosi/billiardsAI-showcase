"""Producer-derived metadata controls; no runtime initialization or media decoding."""

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from billiards_ai.edge.route_assistance.retention_inventory import _identity, _job
from billiards_ai.lab.mainline1_product_realtime_runtime import ProductRealtimeRuntime
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval
from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore


def producer_queue(tmp_path, capture_setup_id):
    runtime = object.__new__(ProductRealtimeRuntime)
    runtime._session_id = "session-1"
    runtime._capture_setup_id = capture_setup_id
    runtime._calibration_artifact_id = "calibration-1"
    runtime._config_sha256 = "a" * 64
    runtime._job_media_root = tmp_path / "media" / "session-1"
    runtime._detector = NS(package_identity=NS(package_id="model-1"))
    runtime._identity_training_package_id = "package-1"
    runtime._identity_training_package_sha256 = "b" * 64
    frozen = FrozenCaptureInterval(
        tmp_path / "ring",
        "epoch-1",
        1920,
        1080,
        0,
        2,
        3,
        {"00000000000000000000-00000000000000000002.mjpgseg": "c" * 64},
    )
    job = runtime._analysis_job(NS(shot_id="shot-1"), frozen)
    database = tmp_path / "jobs.sqlite3"
    SQLiteShotAnalysisJobStore(database).enqueue(job)
    return database, job


def read_job(database, job):
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return _job(db, job.job_id, job.session_id, job.shot_id)


def rewrite(database, mutation, *, correct_hash=True):
    with sqlite3.connect(database) as db:
        value = json.loads(db.execute("select payload_json from shot_analysis_jobs").fetchone()[0])
        mutation(value)
        # Escape malformed surrogates here so the reader receives their decoded value.
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest() if correct_hash else "0" * 64
        db.execute("update shot_analysis_jobs set payload_json=?,payload_sha256=?", (raw, digest))


@pytest.mark.parametrize(
    "setup_id",
    ["台球桌", "台球桌" * 100, "台球桌 / metadata"],
    ids=["unicode", "long_utf8", "metadata_only_slash"],
)
def test_actual_producer_unicode_setup_metadata_is_readonly_and_not_a_path(
    tmp_path, monkeypatch, setup_id
):
    monkeypatch.setattr(
        ProductRealtimeRuntime,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("no runtime initialization"),
    )
    database, job = producer_queue(tmp_path, setup_id)
    before = database.read_bytes()
    row, payload = read_job(database, job)
    assert payload == job.identity()
    assert payload["capture_setup_id"] == setup_id
    assert Path(payload["media_root"]) == job.media_root
    assert setup_id not in job.media_root.parts
    assert hashlib.sha256(row["payload_json"].encode()).hexdigest() == row["payload_sha256"]
    assert database.read_bytes() == before


@pytest.mark.parametrize("bad", [None, "", True, 7, [], {}, "setup\x00id", "\ud800", "\udfff"])
def test_malformed_capture_setup_metadata_stays_rejected(tmp_path, bad):
    database, job = producer_queue(tmp_path, "valid-setup")
    rewrite(database, lambda value: value.update(capture_setup_id=bad))
    with pytest.raises((ValueError, TypeError)):
        read_job(database, job)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_epoch_id", "epoch-台球"),
        ("calibration_artifact_id", "cal-台球"),
        ("model_package_id", "model-台球"),
        ("identity_training_package_id", "package-台球"),
        ("session_id", "session-other"),
        ("shot_id", "shot-other"),
        ("schema_version", "2.0"),
        ("start_frame_id", True),
        ("start_frame_id", -1),
        ("end_frame_id", -1),
        ("media_sha256", "invalid"),
        ("unexpected_field", "invalid"),
    ],
)
def test_other_identity_hash_schema_and_frame_guards_are_not_widened(tmp_path, field, value):
    database, job = producer_queue(tmp_path, "valid-setup")
    rewrite(database, lambda data: data.update({field: value}))
    with pytest.raises((ValueError, TypeError)):
        read_job(database, job)


def test_payload_hash_guard_and_global_path_identity_remain_strict(tmp_path):
    database, job = producer_queue(tmp_path, "valid-setup")
    rewrite(database, lambda value: value.update(capture_setup_id="台球桌"), correct_hash=False)
    with pytest.raises(ValueError, match="invalid_job_metadata"):
        read_job(database, job)
    for value in ("台球桌", "../escape", "part/name", "x" * 257):
        with pytest.raises(ValueError, match="invalid_identity"):
            _identity(value)


def test_utf8_capture_metadata_still_obeys_existing_json_byte_limit(tmp_path):
    database, job = producer_queue(tmp_path, "桌" * 350_000)
    with pytest.raises(ValueError, match="invalid_metadata"):
        read_job(database, job)
