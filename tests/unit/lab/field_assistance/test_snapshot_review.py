import json

from billiards_ai.lab.field_assistance.snapshot_review import snapshot_shot_review


def test_review_requires_original_epoch_and_unique_motion_interval(tmp_path):
    capture = {
        "session_id": "s",
        "source_epoch_id": "e",
        "raw_manifest_path": str(tmp_path / "raw" / "manifest.json"),
    }
    evidence = {"boundary": {"start_frame_id": 100, "end_frame_id": 150}}
    row = {"motion_frame_id": 105, "recommendation": {"source_frame_id": 80}}
    payload = {
        "schema_version": "snapshot-guidance/1",
        "session_id": "s",
        "source_epoch_id": "e",
        "reviews": [row],
    }
    path = tmp_path / "snapshot-guidance.json"
    path.write_text(json.dumps(payload))
    assert snapshot_shot_review(capture, evidence) == row
    payload["reviews"] = [row, row]
    path.write_text(json.dumps(payload))
    assert snapshot_shot_review(capture, evidence) is None
    payload["reviews"] = [row]
    payload["source_epoch_id"] = "wrong"
    path.write_text(json.dumps(payload))
    assert snapshot_shot_review(capture, evidence) is None


def test_corrupt_optional_review_cannot_break_postgame(tmp_path):
    capture = {"raw_manifest_path": str(tmp_path / "raw" / "manifest.json")}
    (tmp_path / "snapshot-guidance.json").write_text("[]")
    assert snapshot_shot_review(capture, {"boundary": {}}) is None
