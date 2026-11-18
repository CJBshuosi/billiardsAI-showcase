from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.edge.route_assistance.stability import StabilityConfig
from billiards_ai.lab.field_assistance.replay import (
    FieldReplayConfig,
    run_field_route_replay,
    verify_field_route_replay_artifact,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.calibration.homography import calibrate


def _write_video(path: Path, *, frame_count: int = 8) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (320, 180),
    )
    assert writer.isOpened()
    cue_x_positions = (50, 50, 90, 130, 170, 170, 170, 170)
    for frame_id in range(frame_count):
        image = np.full((180, 320, 3), (20, 90, 20), dtype=np.uint8)
        cv2.circle(image, (cue_x_positions[frame_id], 90), 7, (245, 245, 245), -1)
        writer.write(image)
    writer.release()
    return path


def _observation(
    frame_id: int,
    cue_x: int,
    target_x: int,
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp_ms": frame_id * 100,
        "observations": [
            {
                "detection_id": f"cue-{frame_id}",
                "center_mm": [cue_x, 500],
                "detection_confidence": 0.95,
                "group": "cue",
                "group_confidence": 0.95,
                "bbox_xyxy_px": [40, 80, 55, 95],
            },
            {
                "detection_id": f"target-{frame_id}",
                "center_mm": [target_x, 650],
                "detection_confidence": 0.9,
                "group": "solid",
                "group_confidence": 0.9,
                "bbox_xyxy_px": [160, 70, 175, 85],
            },
        ],
    }


def _write_observations(path: Path, *, frame_count: int = 8) -> Path:
    positions = (
        (500, 1500),
        (500, 1500),
        (520, 1500),
        (540, 1510),
        (560, 1520),
        (560, 1520),
        (560, 1520),
        (560, 1520),
    )[:frame_count]
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": [
                    _observation(frame_id, cue_x, target_x)
                    for frame_id, (cue_x, target_x) in enumerate(positions)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _calibration():
    return calibrate(
        "synthetic-camera",
        ((0, 180), (320, 180), (320, 0), (0, 0)),
    )


def _config() -> FieldReplayConfig:
    return FieldReplayConfig(
        stability=StabilityConfig(
            motion_threshold_mm=8,
            motion_confirm_frames=2,
            stable_duration_ms=100,
        )
    )


def _run(tmp_path: Path):
    return run_field_route_replay(
        video=_write_video(tmp_path / "input.avi"),
        observations_path=_write_observations(tmp_path / "detections.json"),
        calibration=_calibration(),
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )


def _write_windows(path: Path, *, end_frame_id: int = 5) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "shots": [
                    {
                        "shot_id": "shot-0001",
                        "start_frame_id": 2,
                        "end_frame_id": end_frame_id,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_boundary_events(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "events": [
                    {
                        "shot_id": "shot-0001",
                        "event_type": "shot_start",
                        "status": "observed",
                        "confidence": 1.0,
                        "contact_frame_id": 4,
                        "decision_frame_id": 5,
                    },
                    {
                        "shot_id": "shot-0001",
                        "event_type": "shot_end",
                        "status": "abstained",
                        "contact_frame_id": 7,
                        "decision_frame_id": 7,
                        "reason": "shot_end_not_observed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_observed_boundary_events(
    path: Path,
    *,
    start_confidence: float = 1.0,
    end_confidence: float = 1.0,
    end_frame_id: int = 6,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "events": [
                    {
                        "shot_id": "shot-0001",
                        "event_type": "shot_start",
                        "status": "observed",
                        "confidence": start_confidence,
                        "contact_frame_id": 2,
                        "decision_frame_id": 3,
                    },
                    {
                        "shot_id": "shot-0001",
                        "event_type": "shot_end",
                        "status": "observed",
                        "confidence": end_confidence,
                        "contact_frame_id": end_frame_id,
                        "decision_frame_id": end_frame_id,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_replay_shows_hides_and_recomputes_guidance(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    timeline = json.loads((artifact.path / "timeline.json").read_text(encoding="utf-8"))

    assert timeline["frames"][0]["guidance_visible"] is False
    assert timeline["frames"][1]["guidance_visible"] is True
    assert timeline["frames"][2]["guidance_visible"] is True
    assert not any(row["guidance_visible"] for row in timeline["frames"][3:5])
    assert any(
        row["guidance_visible"] and row["scene_generation"] == 1 for row in timeline["frames"][5:]
    )
    assert (artifact.path / "presentation.mp4").is_file()
    manifest = verify_field_route_replay_artifact(artifact.path)
    assert manifest["artifact_id"] == artifact.artifact_id


def test_replay_consumes_existing_authoritative_shot_window(tmp_path: Path) -> None:
    artifact = run_field_route_replay(
        video=_write_video(tmp_path / "input.avi"),
        observations_path=_write_observations(tmp_path / "detections.json"),
        shot_windows_path=_write_windows(tmp_path / "shot-windows.json"),
        calibration=_calibration(),
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )

    assert artifact.shot_count == 1
    assert artifact.guidance_generation_count == 2


def test_replay_uses_confirmed_cue_start_and_authoritative_window_end_when_end_abstains(
    tmp_path: Path,
) -> None:
    artifact = run_field_route_replay(
        video=_write_video(tmp_path / "input.avi"),
        observations_path=_write_observations(tmp_path / "detections.json"),
        shot_windows_path=_write_windows(tmp_path / "shot-windows.json", end_frame_id=7),
        events_path=_write_boundary_events(tmp_path / "world-events.json"),
        calibration=_calibration(),
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )

    timeline = json.loads((artifact.path / "timeline.json").read_text(encoding="utf-8"))
    assert all(row["guidance_visible"] for row in timeline["frames"][:5])
    assert not any(row["guidance_visible"] for row in timeline["frames"][5:7])
    assert timeline["frames"][7]["guidance_visible"] is True
    assert timeline["frames"][5]["effects"][-1] == "SHOT_STARTED"
    assert timeline["frames"][7]["effects"][0] == "SHOT_SETTLED"
    assert artifact.guidance_generation_count == 2
    assert artifact.shot_count == 1
    manifest = verify_field_route_replay_artifact(artifact.path)
    assert manifest["pipeline_version"] == "field-route-file-replay/2"
    assert manifest["events_sha256"] is not None


def test_replay_recomputes_only_after_confirmed_shot_end(tmp_path: Path) -> None:
    artifact = run_field_route_replay(
        video=_write_video(tmp_path / "input.avi"),
        observations_path=_write_observations(tmp_path / "detections.json"),
        shot_windows_path=_write_windows(tmp_path / "shot-windows.json", end_frame_id=7),
        events_path=_write_observed_boundary_events(tmp_path / "world-events.json"),
        calibration=_calibration(),
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )

    timeline = json.loads((artifact.path / "timeline.json").read_text(encoding="utf-8"))
    assert all(row["guidance_visible"] for row in timeline["frames"][:3])
    assert not any(row["guidance_visible"] for row in timeline["frames"][3:6])
    assert all(row["guidance_visible"] for row in timeline["frames"][6:])
    assert artifact.guidance_generation_count == 2
    assert artifact.shot_count == 1


@pytest.mark.parametrize("confidence", [0.69, float("nan"), float("inf")])
def test_replay_rejects_untrusted_observed_boundary_confidence(
    tmp_path: Path, confidence: float
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi"),
            observations_path=_write_observations(tmp_path / "detections.json"),
            shot_windows_path=_write_windows(tmp_path / "shot-windows.json", end_frame_id=7),
            events_path=_write_observed_boundary_events(
                tmp_path / "world-events.json", start_confidence=confidence
            ),
            calibration=_calibration(),
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
        )


def test_replay_rejects_non_object_boundary_event(tmp_path: Path) -> None:
    events = tmp_path / "world-events.json"
    events.write_text(json.dumps({"schema_version": "1.0", "events": [None]}))

    with pytest.raises(ValueError, match="world events are invalid"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi"),
            observations_path=_write_observations(tmp_path / "detections.json"),
            shot_windows_path=_write_windows(tmp_path / "shot-windows.json", end_frame_id=7),
            events_path=events,
            calibration=_calibration(),
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
        )


def test_replay_rejects_confirmed_boundary_outside_authoritative_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside authoritative shot window"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi"),
            observations_path=_write_observations(tmp_path / "detections.json"),
            shot_windows_path=_write_windows(tmp_path / "shot-windows.json", end_frame_id=5),
            events_path=_write_observed_boundary_events(
                tmp_path / "world-events.json", end_frame_id=6
            ),
            calibration=_calibration(),
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
        )


def test_replay_rejects_video_and_observation_frame_count_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame count"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi", frame_count=7),
            observations_path=_write_observations(tmp_path / "detections.json"),
            calibration=_calibration(),
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
        )


def test_replay_identity_is_reused_for_identical_inputs(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "input.avi")
    observations = _write_observations(tmp_path / "detections.json")
    calibration = _calibration()
    first = run_field_route_replay(
        video=video,
        observations_path=observations,
        calibration=calibration,
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )
    second = run_field_route_replay(
        video=video,
        observations_path=observations,
        calibration=calibration,
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
    )

    assert second == first


def test_replay_manifest_records_verified_mainline1_lineage(tmp_path: Path) -> None:
    artifact = run_field_route_replay(
        video=_write_video(tmp_path / "input.avi"),
        observations_path=_write_observations(tmp_path / "detections.json"),
        shot_windows_path=_write_windows(tmp_path / "shot-windows.json"),
        calibration=_calibration(),
        context=EightBallContext(group="solid"),
        config=_config(),
        output_root=tmp_path / "runs",
        source_mainline1_artifact_id="a" * 64,
    )

    manifest = verify_field_route_replay_artifact(artifact.path)
    assert manifest["source_mainline1_artifact_id"] == "a" * 64


def test_replay_lineage_participates_in_derived_artifact_identity(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "input.avi")
    observations = _write_observations(tmp_path / "detections.json")
    windows = _write_windows(tmp_path / "shot-windows.json")
    common = {
        "video": video,
        "observations_path": observations,
        "shot_windows_path": windows,
        "calibration": _calibration(),
        "context": EightBallContext(group="solid"),
        "config": _config(),
        "output_root": tmp_path / "runs",
    }

    first = run_field_route_replay(**common, source_mainline1_artifact_id="a" * 64)
    same = run_field_route_replay(**common, source_mainline1_artifact_id="a" * 64)
    different = run_field_route_replay(**common, source_mainline1_artifact_id="b" * 64)

    assert same == first
    assert different.artifact_id != first.artifact_id


def test_replay_rejects_invalid_mainline1_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi"),
            observations_path=_write_observations(tmp_path / "detections.json"),
            calibration=_calibration(),
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
            source_mainline1_artifact_id="NOT-A-HASH",
        )


def test_artifact_verifier_rejects_mutated_timeline(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    timeline = artifact.path / "timeline.json"
    timeline.write_bytes(timeline.read_bytes() + b" ")

    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_field_route_replay_artifact(artifact.path)


def test_replay_rejects_calibration_outside_video_viewport(tmp_path: Path) -> None:
    bad = calibrate(
        "bad-camera",
        ((0, 180), (640, 180), (640, 0), (0, 0)),
    )

    with pytest.raises(ValueError, match="calibration.*viewport"):
        run_field_route_replay(
            video=_write_video(tmp_path / "input.avi"),
            observations_path=_write_observations(tmp_path / "detections.json"),
            calibration=bad,
            context=EightBallContext(group="solid"),
            config=_config(),
            output_root=tmp_path / "runs",
        )
