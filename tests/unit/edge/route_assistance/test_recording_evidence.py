import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from billiards_ai.edge.route_assistance.control import MatchControlSnapshot, WristbandRole
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    RecordedMediaV1,
    RecordingBoundaryV1,
    validate_shot_ownership,
)
from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_session_recording import (
    ContinuousSessionRecorder,
    SessionRecordingConfig,
    SessionRecordingEvidenceStore,
)
from billiards_ai.lab.mainline1_shot_jobs import materialize_recorded_interval_reference


def _store(tmp_path):
    store = SQLiteFieldRecordStore(tmp_path / "state.sqlite3")
    session = store.open_or_resume_session(artifact_id="recording", session_policy="recording-v1")
    store.save_match(
        session.session_id,
        MatchControlSnapshot(
            red_user_id="alice",
            blue_user_id="bob",
            breaker=WristbandRole.RED,
            active_role=WristbandRole.RED,
            revision=1,
        ),
        expected_revision=0,
    )
    return store, session.session_id


def test_policy_is_frozen_and_schema26_does_not_backfill_old_sessions(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = SQLiteFieldRecordStore(path)
    session = store.open_or_resume_session(artifact_id="old")
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE field_recording_ends")
        db.execute("DROP TABLE field_recording_evidence")
        db.execute("DROP TABLE field_session_recording_policy")
        db.execute("PRAGMA user_version=26")
    store = SQLiteFieldRecordStore(path)
    assert store.get_session_policy(session.session_id) == "legacy"
    with pytest.raises(ValueError, match="policy changed"):
        store.open_or_resume_session(artifact_id="old", session_policy="recording-v1")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM field_session_recording_policy").fetchone()[0] == 0
    recording = store.open_or_resume_session(artifact_id="new", session_policy="recording-v1")
    with pytest.raises(ValueError, match="policy changed"):
        store.open_or_resume_session(artifact_id="new")
    assert store.get_session_policy(recording.session_id) == "recording-v1"


def test_late_boundary_does_not_require_latest_or_change_unknown_owner(tmp_path):
    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "epoch", 1, 100)
    original = store.get_shot_ownership(session, "shot-1")
    store.record_recording_start(session, "shot-1", "epoch", 1, 100)
    for epoch, frame in [("epoch", 2), ("foreign", 1)]:
        with pytest.raises(ValueError, match="conflicting recording start"):
            store.record_recording_start(session, "shot-1", epoch, frame, 100)
    store.record_recording_start(session, "shot-2", "epoch", 4, 140)
    boundary = RecordingBoundaryV1("shot-1", "epoch", 1, 2, 100, 120, True)
    store.record_recording_boundary(session, boundary)
    store.record_recording_boundary(session, boundary)
    with pytest.raises(ValueError, match="lineage"):
        store.record_recording_boundary(session, replace(boundary, source_epoch_id="foreign"))
    assert store.list_shots(session)[1].settled_ms is None
    assert store.get_shot_ownership(session, "shot-1") == original
    assert original.evidence_status == "unknown"
    assert original.actor_role is original.actor_user_id is None
    with pytest.raises(ValueError, match="ownership is unavailable"):
        validate_shot_ownership(original, store.list_shots(session)[0])
    assert store.get_rule_decision(session, "shot-1") is None


def test_actual_reference_hash_and_late_media_are_transactional_and_idempotent(tmp_path):
    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "epoch", 1, 100)
    boundary = RecordingBoundaryV1("shot-1", "epoch", 1, 2, 100, 120, True)
    store.record_recording_boundary(session, boundary)
    store.record_recording_start(session, "shot-2", "epoch", 4, 140)
    recorder = ContinuousSessionRecorder(
        root=tmp_path / "raw",
        session_id=session,
        expected_fps=60,
        config=SessionRecordingConfig(reserve_bytes=1, minimum_write_mbps=0),
    )
    try:
        for frame in range(6):
            recorder.append(CaptureFrame("epoch", frame, frame * 16_666_667, 16, 12, b"jpeg"))
        evidence = SessionRecordingEvidenceStore(
            recorder=recorder,
            session_id=session,
            pre_roll_frames=1,
            post_roll_frames=3,
            maximum_shot_frames=60,
        )
        frozen = evidence.freeze_window(
            shot_id="shot-1",
            source_epoch_id="epoch",
            start_frame_id=1,
            end_frame_id=5,
        )
        materialize_recorded_interval_reference(
            frozen, tmp_path / "reference", recording_identity=True
        )
        path = tmp_path / "reference/interval.json"
        raw = path.read_bytes()
        media = RecordedMediaV1(
            boundary,
            "RECORDED",
            "recording-unit",
            str(path),
            hashlib.sha256(raw).hexdigest(),
            frozen.media_sha256,
            frozen.start_frame_id,
            frozen.end_frame_id,
        )
        path.write_bytes(raw + b" ")
        with pytest.raises(ValueError, match="hash"):
            store.record_recording_media(session, media)
        assert store.get_recording_evidence(session, "shot-1")["media_status"] == "PENDING"
        path.write_bytes(raw)
        original_reference = json.loads(raw)
        for key, value in (("session_id", "foreign-session"), ("shot_id", "shot-other")):
            altered = json.dumps(original_reference | {key: value}).encode()
            path.write_bytes(altered)
            with pytest.raises(ValueError, match="lineage"):
                store.record_recording_media(
                    session,
                    replace(
                        media,
                        reference_sha256=hashlib.sha256(altered).hexdigest(),
                    ),
                )
        path.write_bytes(raw)
        manifest_path = frozen.root / "session-recording.json"
        manifest_raw = manifest_path.read_bytes()
        altered_manifest = json.loads(manifest_raw) | {"session_id": "other-session"}
        manifest_path.write_text(json.dumps(altered_manifest))
        with pytest.raises(ValueError, match="manifest lineage"):
            store.record_recording_media(session, media)
        manifest_path.write_bytes(manifest_raw)
        segment = frozen.root / next(iter(frozen.segment_sha256))
        segment_raw = segment.read_bytes()
        segment.write_bytes(segment_raw + b"corrupt")
        with pytest.raises(ValueError, match="verification"):
            store.record_recording_media(session, media)
        assert store.get_recording_evidence(session, "shot-1")["media_status"] == "PENDING"
        segment.write_bytes(segment_raw)
        store.record_recording_media(session, media)
        store.record_recording_media(session, media)
        assert store.get_recording_evidence(session, "shot-1")["media_status"] == "RECORDED"
        assert store.get_recording_evidence(session, "shot-1")["encoding_status"] == "NOT_SCHEDULED"
        assert store.list_shots(session)[1].settled_ms is None
        assert store.get_live_shot_media(session, "shot-1") is None  # No invented job attachment.
    finally:
        recorder.close()


def test_failed_and_single_frame_interrupted_boundaries_never_claim_recorded(tmp_path):
    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "epoch", 1, 100)
    boundary = RecordingBoundaryV1("shot-1", "epoch", 1, 1, 100, 100, False, "interrupted")
    store.record_recording_boundary(session, boundary)
    store.record_recording_media(
        session,
        RecordedMediaV1(
            boundary,
            "FAILED",
            failure_code="RECORDING_OSError",
        ),
    )
    data = store.get_recording_evidence(session, "shot-1")
    assert data["media_status"] == "FAILED" and data["media"]["reference_path"] is None
    assert store.list_shots(session)[0].settled_ms is None  # No invented +1ms settlement.


@pytest.mark.parametrize("with_boundary", [False, True])
def test_restart_refuses_pending_original_epoch_instead_of_losing_it(tmp_path, with_boundary):
    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "old-epoch", 1, 100)
    if with_boundary:
        store.record_recording_boundary(
            session,
            RecordingBoundaryV1(
                "shot-1",
                "old-epoch",
                1,
                2,
                100,
                120,
                True,
            ),
        )
    before = store.get_recording_evidence(session, "shot-1")
    restarted = SQLiteFieldRecordStore(store.path)
    for fresh in (False, True):
        with pytest.raises(ValueError, match="explicit recovery"):
            restarted.open_or_resume_session(
                artifact_id="recording",
                session_policy="recording-v1",
                fresh_session_on_start=fresh,
            )
    assert restarted.get_recording_evidence(session, "shot-1") == before
    assert restarted.load_session(session).status.value == "open"


@pytest.mark.parametrize(
    "mutate", ["extra", "frame_bool", "boundary_epoch", "status", "coherent_wrong_time"]
)
def test_rehashed_corrupt_recording_metadata_is_not_accepted_on_restart(tmp_path, mutate):
    store, session = _store(tmp_path)
    store.record_recording_start(session, "shot-1", "epoch", 1, 100)
    store.record_recording_boundary(
        session,
        RecordingBoundaryV1(
            "shot-1",
            "epoch",
            1,
            2,
            100,
            120,
            True,
        ),
    )
    data = store.get_recording_evidence(session, "shot-1")
    if mutate == "extra":
        data["unexpected"] = True
    elif mutate == "frame_bool":
        data["start"]["frame_id"] = True
    elif mutate == "boundary_epoch":
        data["boundary"]["source_epoch_id"] = "foreign"
    elif mutate == "status":
        data["media_status"] = "RECORDED"
    else:
        data["start"]["started_ms"] += 1
        data["boundary"]["started_ms"] += 1
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE field_recording_evidence SET payload_json=?,payload_sha256=?",
            (raw, hashlib.sha256(raw.encode()).hexdigest()),
        )
    with pytest.raises(ValueError):
        SQLiteFieldRecordStore(store.path).get_recording_evidence(session, "shot-1")


@pytest.mark.parametrize("fault", [None, "store", "raw_close", "boundary"])
def test_failed_ui_channel_still_persists_interrupted_media_and_closes_raw(
    tmp_path, monkeypatch, fault
):
    import threading
    from collections import deque
    from types import SimpleNamespace

    from billiards_ai.lab.field_assistance.live_source import USBLiveProjectionSource
    from billiards_ai.lab.mainline1_live_window import LiveShotBoundaryKind
    from billiards_ai.lab.mainline1_product_realtime_runtime import (
        ProductRealtimeEvent,
        ProductRealtimeEventChannel,
        ProductRealtimeRuntime,
    )

    store, session = _store(tmp_path)
    source = USBLiveProjectionSource.__new__(USBLiveProjectionSource)
    source.store, source.session_id, source.source_epoch_id = store, session, "epoch"
    channel = ProductRealtimeEventChannel(
        maximum_pending_events=1, recording_evidence_sink=source._persist_recording_event
    )
    if fault != "boundary":
        channel.publish(ProductRealtimeEvent("CAPTURE_DEGRADED", reason="fill UI queue"))
    window = SimpleNamespace(
        shot_id="shot-1", start_frame_id=1, end_frame_id=1, start_ms=16, end_ms=16, complete=False
    )
    started = SimpleNamespace(kind=LiveShotBoundaryKind.SHOT_STARTED, window=window)
    interrupted = SimpleNamespace(
        kind=LiveShotBoundaryKind.SHOT_INTERRUPTED, window=window, reason="UI_CHANNEL_FAILED"
    )
    settled = SimpleNamespace(
        kind=LiveShotBoundaryKind.SHOT_SETTLED,
        window=SimpleNamespace(
            shot_id="shot-1",
            start_frame_id=1,
            end_frame_id=2,
            start_ms=16,
            end_ms=33,
            complete=True,
        ),
    )
    closed = []
    if fault == "store":

        def broken_store(*args, **kwargs):
            raise ValueError("controlled SQLite failure")

        monkeypatch.setattr(store, "record_recording_start", broken_store)

    def close_raw():
        closed.append("raw")
        if fault == "raw_close":
            raise OSError("controlled close failure")

    def unavailable(**kwargs):
        raise OSError("no complete post-roll")

    runtime = ProductRealtimeRuntime.__new__(ProductRealtimeRuntime)
    runtime._session_policy = "recording-v1"
    runtime._recording_processing = None
    runtime._recording_replay_through = None
    runtime._recording_replaying = False
    runtime._recording_user_end = False
    runtime._session_id = session
    runtime._event_channel = channel
    runtime._event_channel_failed = False
    runtime._recording_active = None
    runtime._recording_windows = deque()
    runtime._latest_capture_frame = None
    runtime._latest_capture_lock = threading.Lock()
    frames = [CaptureFrame("epoch", 1, 16_000_000, 16, 12, b"jpeg")]
    if fault == "boundary":
        frames.append(CaptureFrame("epoch", 2, 33_000_000, 16, 12, b"jpeg"))
    runtime._source = SimpleNamespace(frames=lambda stop: iter(frames))
    runtime._config = SimpleNamespace(
        maximum_shot_frames=1800, post_roll_frames=120, background_shutdown_timeout_seconds=0.01
    )
    runtime._capture_monitor = SimpleNamespace(
        observe=lambda frame: None, close=lambda: closed.append("monitor")
    )
    runtime._ring = SimpleNamespace(
        append=lambda frame: None, freeze_window=unavailable, close=close_raw
    )
    runtime._validate_frame_mode = lambda frame: None
    runtime._cancel_source = lambda: closed.append("source")
    runtime._visual_boundary_detector = SimpleNamespace(interrupt=lambda reason: interrupted)
    runtime._handle_visual_boundary = lambda frame, events: runtime._apply_recording_boundary(
        started if frame.capture_frame_id == 1 else settled,
        events,
    )
    events = runtime.run()
    assert [event.kind for event in events][-1] == "CAPTURE_DEGRADED"
    assert closed == ["source", "raw", "monitor"]
    if fault == "store":
        assert store.list_shots(session) == ()
        assert any("SQLite failure" in (event.reason or "") for event in events)
        return
    data = store.get_recording_evidence(session, "shot-1")
    assert data["boundary"]["complete"] is (fault == "boundary")
    assert data["media_status"] == "FAILED" and data["media"]["reference_path"] is None
    assert store.list_shots(session)[0].settled_ms == (33 if fault == "boundary" else None)
