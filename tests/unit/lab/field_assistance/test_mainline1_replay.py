from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.lab.field_assistance.mainline1_replay import (
    load_mainline1_replay_inputs,
)
from billiards_ai.shared.calibration.homography import calibrate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "input.avi"
    video.write_bytes(b"verified-mainline1-video")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payloads = {
        "detections.json": {"schema_version": "1.0", "frames": []},
        "shot-windows.json": {"schema_version": "1.0", "shots": []},
        "world-events.json": {"schema_version": "1.0", "events": []},
        "event-timeline.json": {"schema_version": "1.0", "shots": []},
        "table-state-after.json": {"schema_version": "1.0", "shots": []},
    }
    for name, payload in payloads.items():
        (artifact / name).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    inventory = {
        name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name in payloads
        for path in (artifact / name,)
    }
    calibration = calibrate(
        "synthetic-camera",
        ((0, 180), (320, 180), (320, 0), (0, 0)),
    )
    manifest = {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "pipeline_version": "shot-local-unified-product-v1",
        "artifact_id": "a" * 64,
        "video_sha256": _sha256(video),
        "calibration": calibration.model_dump(mode="json"),
        "inventory": inventory,
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return video, artifact


def test_verified_artifact_resolves_existing_replay_files(tmp_path: Path) -> None:
    video, artifact = _write_artifact(tmp_path)

    inputs = load_mainline1_replay_inputs(
        artifact_path=artifact,
        video_path=video,
    )

    assert inputs.artifact_id == "a" * 64
    assert inputs.video_path == video.resolve()
    assert inputs.observations_path == artifact / "detections.json"
    assert inputs.shot_windows_path == artifact / "shot-windows.json"
    assert inputs.events_path == artifact / "world-events.json"
    assert inputs.timeline_path == artifact / "event-timeline.json"
    assert inputs.table_state_after_path == artifact / "table-state-after.json"
    assert inputs.calibration.device_id == "synthetic-camera"


def test_replay_inputs_reject_mutated_inventory(tmp_path: Path) -> None:
    video, artifact = _write_artifact(tmp_path)
    (artifact / "world-events.json").write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="inventory mismatch"):
        load_mainline1_replay_inputs(artifact_path=artifact, video_path=video)


def test_replay_inputs_reject_foreign_video(tmp_path: Path) -> None:
    _, artifact = _write_artifact(tmp_path)
    foreign = tmp_path / "foreign.avi"
    foreign.write_bytes(b"different-video")

    with pytest.raises(ValueError, match="does not belong"):
        load_mainline1_replay_inputs(artifact_path=artifact, video_path=foreign)


def test_replay_inputs_reject_manifest_that_omits_required_product_file(
    tmp_path: Path,
) -> None:
    video, artifact = _write_artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["inventory"]["world-events.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="required product files"):
        load_mainline1_replay_inputs(artifact_path=artifact, video_path=video)
