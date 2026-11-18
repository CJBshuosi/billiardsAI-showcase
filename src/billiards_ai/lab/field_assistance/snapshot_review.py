"""Attach optional guidance observations to existing authoritative shot intervals."""

import json
from pathlib import Path


def snapshot_shot_review(capture, evidence):
    if capture is None or evidence is None or evidence.get("boundary") is None:
        return None
    path = Path(capture["raw_manifest_path"]).parent.parent / "snapshot-guidance.json"
    try:
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if (
            data.get("schema_version") != "snapshot-guidance/1"
            or data.get("session_id") != capture["session_id"]
            or data.get("source_epoch_id") != capture["source_epoch_id"]
        ):
            return None
        boundary = evidence["boundary"]
        rows = [
            row
            for row in data["reviews"]
            if boundary["start_frame_id"] <= row["motion_frame_id"] <= boundary["end_frame_id"]
            and row["recommendation"]["source_frame_id"] <= row["motion_frame_id"]
        ]
        # Ambiguous associations never become a per-shot verdict.
        return rows[0] if len(rows) == 1 else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
