from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.edge.route_assistance.commands import (
    FieldCommandEnvelope,
    reduce_field_command,
)
from billiards_ai.edge.route_assistance.fact_report import (
    compose_field_fact_report,
    field_fact_report_payload,
)
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.records import (
    FieldShotRecord,
    SessionLimitCode,
    ShotMediaRange,
)
from billiards_ai.edge.route_assistance.stability import StabilityConfig
from billiards_ai.lab.field_assistance.mainline1_replay import (
    load_mainline1_replay_inputs,
)
from billiards_ai.lab.field_assistance.replay import (
    FieldReplayConfig,
    run_field_route_replay,
    verify_field_route_replay_artifact,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.calibration.homography import calibrate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    assert writer.isOpened()
    cue_positions = (50, 50, 90, 130, 170, 170, 170, 170)
    for cue_x in cue_positions:
        image = np.full((180, 320, 3), (20, 90, 20), dtype=np.uint8)
        cv2.circle(image, (cue_x, 90), 7, (245, 245, 245), -1)
        cv2.circle(image, (220, 90), 7, (40, 40, 200), -1)
        writer.write(image)
    writer.release()
    return path


def _observation(
    frame_id: int,
    cue_x: int,
    target_x: int,
    *,
    cue_group: str = "cue",
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp_ms": frame_id * 100,
        "observations": [
            {
                "detection_id": f"cue-{frame_id}",
                "center_mm": [cue_x, 500],
                "detection_confidence": 0.95,
                "group": cue_group,
                "group_confidence": 0.95,
                "bbox_xyxy_px": [40, 80, 55, 95],
            },
            {
                "detection_id": f"target-{frame_id}",
                "center_mm": [target_x, 650],
                "detection_confidence": 0.92,
                "group": "solid",
                "group_confidence": 0.91,
                "bbox_xyxy_px": [210, 80, 225, 95],
            },
        ],
    }


def _write_mainline1_artifact(
    tmp_path: Path,
    *,
    cue_group: str = "cue",
) -> tuple[Path, Path]:
    video = _write_video(tmp_path / "input.avi")
    calibration = calibrate(
        "synthetic-camera",
        ((0, 180), (320, 180), (320, 0), (0, 0)),
    )
    artifact = tmp_path / "mainline1-artifact"
    artifact.mkdir()
    positions = (
        (500, 1500),
        (500, 1500),
        (520, 1500),
        (540, 1510),
        (560, 1520),
        (560, 1520),
        (560, 1520),
        (560, 1520),
    )
    payloads = {
        "detections.json": {
            "schema_version": "1.0",
            "frames": [
                _observation(frame_id, cue_x, target_x, cue_group=cue_group)
                for frame_id, (cue_x, target_x) in enumerate(positions)
            ],
        },
        "shot-windows.json": {
            "schema_version": "1.0",
            "shots": [{"shot_id": "shot-0001", "start_frame_id": 2, "end_frame_id": 5}],
        },
        "world-events.json": {
            "schema_version": "1.0",
            "events": [
                {"shot_id": "shot-0001", "event_type": "shot_start"},
                {"shot_id": "shot-0001", "event_type": "shot_end"},
            ],
        },
        "event-timeline.json": {
            "schema_version": "1.0",
            "shots": [{"shot_id": "shot-0001", "events": []}],
        },
        "table-state-after.json": {
            "schema_version": "1.0",
            "shots": [{"shot_id": "shot-0001", "frame_id": 5, "available": True}],
        },
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
    manifest = {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "pipeline_version": "shot-local-unified-product-v1",
        "artifact_id": "a" * 64,
        "video_sha256": _sha256(video),
        "calibration": calibration.model_dump(mode="json"),
        "inventory": inventory,
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return video, artifact


def _command(
    command_id: str,
    *,
    sequence: int,
    revision: int,
    actor: str,
    command_type: str,
    payload: dict[str, str],
) -> FieldCommandEnvelope:
    return FieldCommandEnvelope(
        command_id=command_id,
        session_id="session-vertical-slice",
        actor_user_id=actor,
        command_type=command_type,
        payload=payload,
        expected_revision=revision,
        server_sequence=sequence,
    )


def test_verified_mainline1_artifact_drives_route_users_and_fact_report(tmp_path: Path) -> None:
    video, mainline1_path = _write_mainline1_artifact(tmp_path)
    inputs = load_mainline1_replay_inputs(
        artifact_path=mainline1_path,
        video_path=video,
    )
    replay = run_field_route_replay(
        video=inputs.video_path,
        observations_path=inputs.observations_path,
        shot_windows_path=inputs.shot_windows_path,
        calibration=inputs.calibration,
        context=EightBallContext(group="solid"),
        config=FieldReplayConfig(
            stability=StabilityConfig(
                motion_threshold_mm=8,
                motion_confirm_frames=2,
                stable_duration_ms=100,
            )
        ),
        output_root=tmp_path / "field-runs",
        source_mainline1_artifact_id=inputs.artifact_id,
    )
    replay_manifest = verify_field_route_replay_artifact(replay.path)
    summary = json.loads((replay.path / "summary.json").read_text(encoding="utf-8"))
    timeline = json.loads((replay.path / "timeline.json").read_text(encoding="utf-8"))
    recommendations = json.loads((replay.path / "recommendations.json").read_text(encoding="utf-8"))

    assert replay_manifest["source_mainline1_artifact_id"] == inputs.artifact_id
    assert replay_manifest["pipeline_version"] == "field-route-file-replay/2"
    assert summary["shot_count"] == 1
    assert summary["guidance_generation_count"] >= 1
    assert any(row["guidance_visible"] for row in timeline["frames"][:2])
    assert not any(row["guidance_visible"] for row in timeline["frames"][2:5])
    assert any(row["guidance_visible"] for row in timeline["frames"][5:])

    display = next(
        generation["display"]
        for generation in recommendations["generations"]
        if generation["display"] is not None
    )
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: datetime(2026, 8, 29, 10, 0, tzinfo=UTC),
        session_id_factory=lambda: "session-vertical-slice",
    )
    store.open_or_resume_session(artifact_id=replay.artifact_id)
    commands = (
        _command(
            "cmd-red",
            sequence=1,
            revision=0,
            actor="user-red",
            command_type="claim_role",
            payload={"role": "red"},
        ),
        _command(
            "cmd-blue",
            sequence=2,
            revision=1,
            actor="user-blue",
            command_type="claim_role",
            payload={"role": "blue"},
        ),
        _command(
            "cmd-breaker",
            sequence=3,
            revision=2,
            actor="user-red",
            command_type="set_breaker",
            payload={"role": "red"},
        ),
        _command(
            "cmd-groups",
            sequence=4,
            revision=3,
            actor="user-red",
            command_type="assign_groups",
            payload={"red_group": "solid"},
        ),
    )
    for command in commands:
        assert store.apply_field_command(command, reduce_field_command).outcome == "applied"

    session = store.load_session("session-vertical-slice")
    shot = FieldShotRecord(
        session_id=session.session_id,
        shot_id="shot-0001",
        sequence=1,
        actor_role=session.match.active_role,
        started_ms=200,
        settled_ms=500,
        recommendation_id=display["recommendation_id"],
        power_band=display["power"]["band"],
        power_percent=display["power"]["approximate_percent"],
        media=ShotMediaRange(
            artifact_id=replay.artifact_id,
            source_locator="presentation.mp4",
            source_kind="development_replay",
            start_ms=0,
            end_ms=700,
        ),
        limitations=(SessionLimitCode.DEVELOPMENT_REPLAY_MEDIA,),
    )
    store.record_shot_start(replace(shot, settled_ms=None))
    store.record_shot_settle(shot)
    assert (
        store.apply_field_command(
            _command(
                "cmd-end",
                sequence=5,
                revision=4,
                actor="user-red",
                command_type="end_session",
                payload={},
            ),
            reduce_field_command,
        ).outcome
        == "applied"
    )
    ended = store.load_session(session.session_id)
    report = field_fact_report_payload(
        compose_field_fact_report(ended, store.list_shots(session.session_id))
    )

    assert report["match"]["red_participant_ref"] == "user-red"
    assert report["match"]["blue_participant_ref"] == "user-blue"
    assert report["shots"][0]["actor_role"] == "red"
    assert report["shots"][0]["recommendation_id"] == display["recommendation_id"]
    assert not ({"score", "weaknesses", "training_plan", "highlight"} & report.keys())


def test_unplannable_mainline1_state_abstains_without_stale_guidance(tmp_path: Path) -> None:
    video, mainline1_path = _write_mainline1_artifact(tmp_path, cue_group="unknown")
    inputs = load_mainline1_replay_inputs(
        artifact_path=mainline1_path,
        video_path=video,
    )

    replay = run_field_route_replay(
        video=inputs.video_path,
        observations_path=inputs.observations_path,
        shot_windows_path=inputs.shot_windows_path,
        calibration=inputs.calibration,
        context=EightBallContext(group="solid"),
        config=FieldReplayConfig(
            stability=StabilityConfig(
                motion_threshold_mm=8,
                motion_confirm_frames=2,
                stable_duration_ms=100,
            )
        ),
        output_root=tmp_path / "field-runs",
        source_mainline1_artifact_id=inputs.artifact_id,
    )
    timeline = json.loads((replay.path / "timeline.json").read_text(encoding="utf-8"))
    recommendations = json.loads((replay.path / "recommendations.json").read_text(encoding="utf-8"))

    assert not any(row["guidance_visible"] for row in timeline["frames"])
    assert recommendations["generations"]
    assert all(row["status"] == "abstained" for row in recommendations["generations"])
