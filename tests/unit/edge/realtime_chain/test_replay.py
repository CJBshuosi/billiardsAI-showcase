from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.edge.realtime_chain.replay import run_replay, verify_replay_artifact


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _source_artifact(root: Path) -> Path:
    root.mkdir()
    payloads = {
        "detections.json": {"schema_version": "1.0", "frames": []},
        "event-timeline.json": {
            "schema_version": "1.0",
            "shots": [{"shot_id": "shot-0001", "events": []}],
        },
        "shot-windows.json": {
            "schema_version": "1.0",
            "shots": [
                {
                    "shot_id": "shot-0001",
                    "start_frame_id": 11,
                    "end_frame_id": 20,
                    "before_frame_id": 10,
                    "after_frame_id": 20,
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "complete": True,
                }
            ],
        },
        "anchors.json": {
            "schema_version": "1.0",
            "shots": [
                {
                    "shot_id": "shot-0001",
                    "anchors": [
                        {
                            "shot_id": "shot-0001",
                            "anchor_id": "shot-0001-anchor-01",
                            "source_detection_id": "cue-before",
                            "group": "cue",
                            "x_mm": 100,
                            "y_mm": 200,
                            "confidence": 0.9,
                        },
                        {
                            "shot_id": "shot-0001",
                            "anchor_id": "shot-0001-anchor-02",
                            "source_detection_id": "target-before",
                            "group": "solid",
                            "x_mm": 400,
                            "y_mm": 200,
                            "confidence": 0.9,
                        },
                    ],
                }
            ],
        },
        "cue-track.json": {
            "schema_version": "1.0",
            "shots": [
                {
                    "shot_id": "shot-0001",
                    "samples": [
                        {
                            "shot_id": "shot-0001",
                            "local_id": "shot-0001-cue",
                            "frame_id": 11,
                            "timestamp_ms": 1000,
                            "state": "confirmed",
                            "x_mm": 100,
                            "y_mm": 200,
                            "confidence": 0.9,
                            "source_detection_id": "cue-11",
                        },
                        {
                            "shot_id": "shot-0001",
                            "local_id": "shot-0001-cue",
                            "frame_id": 12,
                            "timestamp_ms": 1100,
                            "state": "confirmed",
                            "x_mm": 140,
                            "y_mm": 200,
                            "confidence": 0.88,
                            "source_detection_id": "cue-12",
                        },
                    ],
                }
            ],
        },
        "events.json": {
            "schema_version": "1.0",
            "shots": [
                {
                    "shot_id": "shot-0001",
                    "events": [
                        {
                            "shot_id": "shot-0001",
                            "event_id": "shot-0001-event-first-contact",
                            "event_type": "first_contact",
                            "status": "observed",
                            "timestamp_ms": 1100,
                            "actor_id": "shot-0001-cue",
                            "target_id": "shot-0001-anchor-02",
                            "evidence_frame_ids": [11, 12],
                            "confidence": 0.84,
                            "reason": None,
                        }
                    ],
                }
            ],
        },
        "table-state-after.json": {
            "schema_version": "1.0",
            "shots": [
                {
                    "shot_id": "shot-0001",
                    "frame_id": 20,
                    "available": True,
                    "balls": [
                        {
                            "detection_id": "cue-after",
                            "center_mm": [140, 200],
                            "detection_confidence": 0.9,
                            "group": "cue",
                            "group_confidence": 0.9,
                            "bbox_xyxy_px": [1, 1, 2, 2],
                        }
                    ],
                }
            ],
        },
        "world-events.json": {"schema_version": "1.0", "events": []},
    }
    inventory = {}
    for name, payload in payloads.items():
        data = _canonical(payload)
        (root / name).write_bytes(data)
        inventory[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "pipeline_version": "shot-local-calibration-overlay-v4",
        "artifact_id": "a" * 64,
        "video_sha256": "a" * 64,
        "inventory": inventory,
    }
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return root


def test_replay_publishes_traceable_profile_and_measured_async_latency(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path / "source")

    artifact = run_replay(
        source_artifact=source,
        output_root=tmp_path / "output",
        user_id="user-1",
        session_id="session-43",
    )

    assert {item.name for item in artifact.path.iterdir()} == {
        "realtime-chain.json",
        "profile-increment.json",
        "profile-aggregate.json",
        "performance.json",
        "manifest.json",
    }
    chain = json.loads((artifact.path / "realtime-chain.json").read_text())
    assert chain["mode"] == "artifact_replay_async"
    assert chain["claims"]["global_fixed_ball_id"] is False
    assert chain["claims"]["sam_realtime_main_chain"] is False
    assert chain["shots"][0]["status"] == "admitted"
    performance = json.loads((artifact.path / "performance.json").read_text())
    assert performance["detector_runtime_included"] is False
    assert performance["shot_count"] == 1
    assert len(performance["per_shot_projection_ms"]) == 1
    assert performance["projection_p95_ms"] >= 0
    aggregate = json.loads((artifact.path / "profile-aggregate.json").read_text())
    assert aggregate["admitted_shot_count"] == 1
    assert verify_replay_artifact(artifact.path)["source_artifact_id"] == "a" * 64


def test_replay_refuses_tampered_source_before_writing_outputs(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path / "source")
    (source / "events.json").write_text("tampered")

    with pytest.raises(ValueError, match="inventory mismatch"):
        run_replay(
            source_artifact=source,
            output_root=tmp_path / "output",
            user_id="user-1",
            session_id="session-43",
        )
    assert not (tmp_path / "output").exists()


def test_replay_refuses_source_directory_symlink_before_resolving_it(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path / "source")
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="missing or unsafe"):
        run_replay(
            source_artifact=alias,
            output_root=tmp_path / "output",
            user_id="user-1",
            session_id="session-43",
        )
    assert not (tmp_path / "output").exists()


def test_replay_is_content_addressed_for_same_source_and_policy(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path / "source")
    first = run_replay(
        source_artifact=source,
        output_root=tmp_path / "output",
        user_id="user-1",
        session_id="session-43",
    )
    repeated = run_replay(
        source_artifact=source,
        output_root=tmp_path / "output",
        user_id="user-1",
        session_id="session-43",
    )

    assert repeated.artifact_id == first.artifact_id
    assert repeated.path == first.path
