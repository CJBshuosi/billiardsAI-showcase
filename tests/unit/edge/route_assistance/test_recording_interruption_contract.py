"""The interruption version cannot masquerade as a normal stop or deletion grant."""

import hashlib
from dataclasses import asdict

import pytest

from billiards_ai.edge.route_assistance.recording_capture import (
    public_recording_capture,
    recording_capture_from_payload,
)
from billiards_ai.edge.route_assistance.records import (
    RecordingEndV1,
    RecordingInterruptedEndV2,
    recording_end_from_payload,
)
from tests.unit.edge.route_assistance.test_recording_capture_end import _stop
from tests.unit.edge.route_assistance.test_recording_evidence import _store


def test_interruption_capture_is_strict_version_and_publicly_tail_unknown(tmp_path):
    store, sid = _store(tmp_path)
    normal = _stop(store, sid, tmp_path / "raw")
    core = {
        k: v
        for k, v in normal.items()
        if k not in {"tail_status", "processing_status", "failure_code", "capture_end_sha256"}
    }
    assert recording_capture_from_payload(core) == core
    interrupted = core | dict(
        contract="recording-capture-end/v2",
        end_reason="PROCESS_INTERRUPTED",
        tail_completeness="UNKNOWN",
        raw_manifest_sha256="b" * 64,
    )
    assert recording_capture_from_payload(interrupted) == interrupted
    public = public_recording_capture(
        interrupted | dict(tail_status="SAVED", processing_status="PROCESSING", failure_code=None)
    )
    assert (
        public["capture_status"] == "INTERRUPTED" and public["boundary_kind"] == "COMMITTED_PREFIX"
    )
    assert public["tail_completeness"] == "UNKNOWN" and "raw_manifest_sha256" not in public
    for value in (
        interrupted | dict(contract="recording-capture-end/v1"),
        interrupted | dict(end_reason="USER_ENDED_RECORDING"),
        interrupted | dict(tail_completeness="COMPLETE"),
        core | dict(end_reason="PROCESS_INTERRUPTED"),
    ):
        with pytest.raises(ValueError):
            recording_capture_from_payload(value)


def test_interruption_final_version_does_not_expand_v1():
    core = dict(
        contract="recording-end/v1",
        session_id="session",
        source_epoch_id="epoch",
        policy="recording-v1",
        end_reason="USER_ENDED_RECORDING",
        cutoff=None,
        raw_manifest_path="/raw/session-recording.json",
        raw_manifest_sha256="a" * 64,
        raw_segments=[],
        first_frame_id=None,
        last_frame_id=None,
        width=0,
        height=0,
        shots=[],
        media_status="EMPTY",
        analysis_status="NOT_REQUESTED",
        encoding_status="NOT_SCHEDULED",
        ended_at_utc=None,
    )
    assert recording_end_from_payload(core) == core
    interrupted = core | dict(
        contract="recording-end/v2", end_reason="PROCESS_INTERRUPTED", tail_completeness="UNKNOWN"
    )
    assert recording_end_from_payload(interrupted) == interrupted
    with pytest.raises(ValueError):
        RecordingEndV1(
            **(core | dict(contract="recording-end/v2", end_reason="PROCESS_INTERRUPTED"))
        )
    with pytest.raises(ValueError):
        recording_end_from_payload(interrupted | dict(tail_completeness="COMPLETE"))
    assert asdict(RecordingInterruptedEndV2(**interrupted)) == interrupted


def test_interruption_registration_and_session_end_are_atomic_and_retry_immutable(tmp_path):
    store, sid = _store(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = raw / "session-recording.json"
    manifest.write_bytes(b"original snapshot")
    values = dict(
        first_frame_id=0,
        width=16,
        height=12,
        last_frame_sha256="a" * 64,
        interruption_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    result = store.record_recording_capture_end(
        sid, "epoch", {"frame_id": 9, "captured_at_ns": 150000003}, manifest, **values
    )
    assert store.load_session(sid).status.value == "ended"
    assert store.get_postgame_manifest(sid) is not None
    assert store.get_recording_end_manifest(sid) is None
    assert store.record_recording_capture_end(
        sid, "epoch", result["cutoff"], manifest, **values
    ) == store.get_recording_capture_end(sid)
    with pytest.raises(ValueError, match="conflicts"):
        store.record_recording_capture_end(
            sid, "epoch", {"frame_id": 10, "captured_at_ns": 166666670}, manifest, **values
        )
