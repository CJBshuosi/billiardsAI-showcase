"""Durable replay origin and segment processing fences in the boundary SQLite DB.

One ordinary detector owns this journal through the existing runtime lifecycle.
Input admission precedes any events; a checkpoint follows synchronous evidence
persistence. A restart replays the original prefix, rebuilding motion/shot state,
then idempotently processes events beyond the last successful checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

import cv2


def recording_processing_config_sha256(
    *,
    capture_config,
    calibration,
    window,
    motion,
    runtime_config,
    config_sha256,
):
    effective = dict(
        implementation="recording-replay/v1",
        opencv=cv2.__version__,
        capture=asdict(capture_config),
        window=asdict(window),
        motion=asdict(motion),
        calibration=calibration.model_dump(mode="json", exclude={"calibrated_at"}),
        pre_roll_frames=runtime_config.pre_roll_frames,
        post_roll_frames=runtime_config.post_roll_frames,
        maximum_shot_frames=runtime_config.maximum_shot_frames,
        config_sha256=config_sha256,
    )
    return hashlib.sha256(
        json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class RecordingProcessingJournal:
    def __init__(
        self,
        *,
        path: Path,
        session_id: str,
        source_epoch_id: str,
        config_sha256: str,
        initial_shot_number: int,
        require_existing: bool = False,
    ) -> None:
        if not session_id.strip() or not source_epoch_id.strip():
            raise ValueError("recording processing identity is invalid")
        _require_sha(config_sha256)
        if type(initial_shot_number) is not int or initial_shot_number < 0:
            raise ValueError("recording processing initial shot number is invalid")
        self.path = path
        self._key = (session_id, source_epoch_id)
        if require_existing and not path.is_file():
            raise ValueError("recording replay origin is missing")
        with closing(self._connect()) as db, db:
            if (
                require_existing
                and db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recording_processing'"
                ).fetchone()
                is None
            ):
                raise ValueError("recording replay origin is missing")
            db.execute("""
                CREATE TABLE IF NOT EXISTS recording_processing (
                    session_id TEXT NOT NULL,
                    source_epoch_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, source_epoch_id)
                )
            """)
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT 1 FROM recording_processing WHERE session_id=? AND source_epoch_id=?",
                self._key,
            ).fetchone()
            if existing is None and require_existing:
                raise ValueError("recording replay origin is missing")
            if (
                existing is None
                and db.execute(
                    "SELECT 1 FROM shot_boundary_ledger "
                    "WHERE session_id=? AND source_epoch_id=? LIMIT 1",
                    self._key,
                ).fetchone()
                is not None
            ):
                raise ValueError("recording replay origin is missing for existing boundaries")
            initial = dict(
                contract="recording-processing/v1",
                recovery="replay-from-start/v1",
                config_sha256=config_sha256,
                initial_shot_number=initial_shot_number,
                input_prefix=None,
                checkpoint=None,
            )
            db.execute(
                "INSERT OR IGNORE INTO recording_processing VALUES(?,?,?)",
                (*self._key, _canonical(initial)),
            )
            stored = self._read(db)
            if stored["config_sha256"] != config_sha256:
                raise ValueError("recording processing config conflicts with replay origin")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def _read(self, db):
        return self._read_in(db, self._key)

    @classmethod
    def read_state(cls, path, session_id, source_epoch_id):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            return cls._read_in(db, (session_id, source_epoch_id))

    @staticmethod
    def _read_in(db, key):
        row = db.execute(
            "SELECT payload_json FROM recording_processing "
            "WHERE session_id=? AND source_epoch_id=?",
            key,
        ).fetchone()
        if row is None:
            raise ValueError("recording processing origin is missing")
        try:
            value = json.loads(row[0])
            if (
                not isinstance(value, dict)
                or set(value)
                != {
                    "contract",
                    "recovery",
                    "config_sha256",
                    "initial_shot_number",
                    "input_prefix",
                    "checkpoint",
                }
                or value["contract"] != "recording-processing/v1"
                or value["recovery"] != "replay-from-start/v1"
                or type(value["initial_shot_number"]) is not int
                or value["initial_shot_number"] < 0
            ):
                raise ValueError("shape")
            _require_sha(value["config_sha256"])
            for key in ("input_prefix", "checkpoint"):
                if value[key] is not None:
                    _require_segment(value[key])
            admitted, processed = value["input_prefix"], value["checkpoint"]
            if processed is not None and (
                admitted is None
                or processed["segment_index"] > admitted["segment_index"]
                or (
                    processed["segment_index"] == admitted["segment_index"]
                    and processed != admitted
                )
            ):
                raise ValueError("checkpoint exceeds admission")
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("recording processing journal is invalid") from exc
        return value

    @property
    def state(self):
        with closing(self._connect()) as db:
            return self._read(db)

    @contextmanager
    def _change(self):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            value = self._read(db)
            yield value
            db.execute(
                "UPDATE recording_processing SET payload_json=? "
                "WHERE session_id=? AND source_epoch_id=?",
                (_canonical(value), *self._key),
            )

    def admit_segment(self, segment):
        _require_segment(segment)
        with self._change() as state:
            prior = state["input_prefix"]
            if prior == segment:
                return
            if prior is not None and segment["segment_index"] <= prior["segment_index"]:
                raise ValueError("recording processing input identity conflicts")
            if prior != state["checkpoint"]:
                raise ValueError("previous admitted segment has not been processed")
            expected = 0 if prior is None else prior["segment_index"] + 1
            if segment["segment_index"] != expected or (
                prior is not None and segment["last_frame_id"] <= prior["last_frame_id"]
            ):
                raise ValueError("recording processing segment sequence is invalid")
            state["input_prefix"] = dict(segment)

    def checkpoint(self, segment):
        _require_segment(segment)
        with self._change() as state:
            previous = state["checkpoint"]
            if previous == segment:
                return
            if previous is not None and segment["segment_index"] < previous["segment_index"]:
                raise ValueError("recording processing checkpoint cannot move backward")
            if segment != state["input_prefix"]:
                raise ValueError("recording processing checkpoint is not the admitted input")
            state["checkpoint"] = dict(segment)


def _require_sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("recording processing SHA256 is invalid")


def _require_segment(value):
    if not isinstance(value, dict) or set(value) != {
        "segment_index",
        "last_frame_id",
        "last_captured_at_ns",
        "prefix_sha256",
    }:
        raise ValueError("recording processing segment descriptor is invalid")
    for key in ("segment_index", "last_frame_id", "last_captured_at_ns"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("recording processing segment position is invalid")
    _require_sha(value["prefix_sha256"])


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
