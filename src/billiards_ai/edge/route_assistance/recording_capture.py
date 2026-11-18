"""Immutable capture-stop receipt, distinct from a final detected-shot manifest."""

from datetime import datetime
from pathlib import Path

RECORDING_CAPTURE_SCHEMA30 = """
CREATE TABLE IF NOT EXISTS field_recording_captures (
    session_id TEXT PRIMARY KEY REFERENCES field_sessions(session_id),
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    tail_status TEXT NOT NULL CHECK(tail_status IN ('PENDING','SAVED','FAILED')),
    processing_status TEXT NOT NULL CHECK(processing_status IN ('PROCESSING','COMPLETE','FAILED')),
    failure_code TEXT
);
"""


def recording_capture_from_payload(value):
    keys = {
        "contract",
        "session_id",
        "source_epoch_id",
        "end_reason",
        "cutoff",
        "raw_manifest_path",
        "first_frame_id",
        "width",
        "height",
        "last_frame_sha256",
        "accepted_at_utc",
    }
    interrupted = isinstance(value, dict) and value.get("contract") == "recording-capture-end/v2"
    if interrupted:
        keys |= {"tail_completeness", "raw_manifest_sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("recording capture receipt shape is invalid")
    if interrupted:
        digest = value["raw_manifest_sha256"]
        if (
            value["end_reason"] != "PROCESS_INTERRUPTED"
            or value["tail_completeness"] != "UNKNOWN"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("recording interruption contract is invalid")
    elif (
        value["contract"] != "recording-capture-end/v1"
        or value["end_reason"] != "USER_ENDED_RECORDING"
    ):
        raise ValueError("recording capture contract is invalid")
    for key in ("session_id", "source_epoch_id", "raw_manifest_path", "accepted_at_utc"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("recording capture identity is invalid")
    path = Path(value["raw_manifest_path"])
    if not path.is_absolute() or ".." in path.parts or path.name != "session-recording.json":
        raise ValueError("recording capture raw path is invalid")
    try:
        timestamp = datetime.fromisoformat(value["accepted_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError("timezone")
    except ValueError as exc:
        raise ValueError("recording capture timestamp is invalid") from exc
    cutoff = value["cutoff"]
    if cutoff is None:
        if (
            value["first_frame_id"],
            value["width"],
            value["height"],
            value["last_frame_sha256"],
        ) != (
            None,
            0,
            0,
            None,
        ):
            raise ValueError("empty recording capture geometry is invalid")
    else:
        if not isinstance(cutoff, dict) or set(cutoff) != {"frame_id", "captured_at_ns"}:
            raise ValueError("recording capture cutoff is invalid")
        if any(type(v) is not int or v < 0 for v in cutoff.values()):
            raise ValueError("recording capture cutoff is invalid")
        if (
            type(value["first_frame_id"]) is not int
            or not 0 <= value["first_frame_id"] <= cutoff["frame_id"]
            or type(value["width"]) is not int
            or value["width"] <= 0
            or type(value["height"]) is not int
            or value["height"] <= 0
        ):
            raise ValueError("recording capture geometry is invalid")
        digest = value["last_frame_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("recording capture final-frame hash is invalid")
    return dict(value)


def public_recording_capture(value):
    """The field screen needs progress and identity, never private filesystem roots."""
    return {
        key: value[key]
        for key in (
            "contract",
            "session_id",
            "source_epoch_id",
            "end_reason",
            "cutoff",
            "accepted_at_utc",
            "tail_status",
            "processing_status",
            "failure_code",
        )
    } | (
        {
            "capture_status": "INTERRUPTED",
            "tail_completeness": "UNKNOWN",
            "boundary_kind": "COMMITTED_PREFIX",
        }
        if value["contract"] == "recording-capture-end/v2"
        else {"capture_status": "STOPPED"}
    )


def public_recording_end(value):
    """Project final status without raw paths or private segment identity receipts."""
    result = {
        key: value[key]
        for key in (
            "contract",
            "session_id",
            "source_epoch_id",
            "policy",
            "end_reason",
            "cutoff",
            "first_frame_id",
            "last_frame_id",
            "width",
            "height",
            "shots",
            "media_status",
            "analysis_status",
            "encoding_status",
            "ended_at_utc",
        )
    }

    if value["contract"] == "recording-end/v2":
        result["tail_completeness"] = "UNKNOWN"
        result["boundary_kind"] = "COMMITTED_PREFIX"
    return result
