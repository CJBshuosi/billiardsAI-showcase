from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.field_assistance.observations import (
    build_snapshot_table_state,
    load_cached_observations,
    scene_observation,
)
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.shared.models.table_state import BallKind, MotionStatus


def _ball(
    detection_id: str,
    x_mm: int,
    y_mm: int,
    group: BallGroup,
) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=detection_id,
        x_mm=x_mm,
        y_mm=y_mm,
        confidence=0.9,
        group=group,
        group_confidence=0.8,
        bbox_xyxy_px=(10, 20, 30, 40),
    )


def _frame(*balls: CalibratedBallObservation) -> ObservationFrame:
    return ObservationFrame(frame_id=8, timestamp_ms=240, observations=tuple(balls))


def _payload(*, frame_id: int = 8, observations: list[dict] | None = None) -> dict:
    return {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": frame_id,
                "timestamp_ms": 240,
                "observations": observations
                if observations is not None
                else [
                    {
                        "detection_id": "cue-8",
                        "center_mm": [100, 200],
                        "detection_confidence": 0.9,
                        "group": "cue",
                        "group_confidence": 0.8,
                        "bbox_xyxy_px": [10, 20, 30, 40],
                    }
                ],
            }
        ],
    }


def test_snapshot_ids_are_local_to_one_scene_generation() -> None:
    frame = _frame(
        _ball("stripe", 900, 500, BallGroup.STRIPE),
        _ball("cue", 100, 200, BallGroup.CUE),
        _ball("solid", 600, 300, BallGroup.SOLID),
    )

    state = build_snapshot_table_state(
        frame,
        calibration_id="cal-1",
        source_ref="artifact:detections.json#frame=8",
        scene_generation=3,
    )

    assert [ball.track_id for ball in state.balls] == [
        "scene-0003-ball-01",
        "scene-0003-ball-02",
        "scene-0003-ball-03",
    ]
    assert [ball.kind for ball in state.balls] == [
        BallKind.CUE,
        BallKind.SOLID,
        BallKind.STRIPE,
    ]
    assert state.motion_status is MotionStatus.STATIONARY
    state.require_plannable()


def test_scene_observation_exposes_centers_without_detection_ids() -> None:
    frame = _frame(
        _ball("second", 400, 300, BallGroup.SOLID),
        _ball("first", 100, 200, BallGroup.CUE),
    )

    scene = scene_observation(frame)

    assert scene.centers_mm == ((100, 200), (400, 300))
    assert not hasattr(scene, "detection_ids")


def test_cached_observation_loader_reconstructs_validated_frames(tmp_path: Path) -> None:
    source = tmp_path / "detections.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    frames = load_cached_observations(source)

    assert frames == (_frame(_ball("cue-8", 100, 200, BallGroup.CUE)),)


@pytest.mark.parametrize("schema_version", (None, "0.9", "2.0"))
def test_cached_observation_loader_rejects_unknown_schema(
    tmp_path: Path, schema_version: str | None
) -> None:
    source = tmp_path / "detections.json"
    payload = _payload()
    payload["schema_version"] = schema_version
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_cached_observations(source)


def test_cached_observation_loader_rejects_duplicate_frames(tmp_path: Path) -> None:
    source = tmp_path / "detections.json"
    payload = _payload()
    payload["frames"].append(dict(payload["frames"][0]))
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique ordered"):
        load_cached_observations(source)


def test_cached_observation_loader_rejects_missing_group_confidence(tmp_path: Path) -> None:
    source = tmp_path / "detections.json"
    payload = _payload()
    del payload["frames"][0]["observations"][0]["group_confidence"]
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_cached_observations(source)


@pytest.mark.parametrize(
    "balls",
    (
        (_ball("solid", 600, 300, BallGroup.SOLID),),
        (
            _ball("cue-a", 100, 200, BallGroup.CUE),
            _ball("cue-b", 200, 200, BallGroup.CUE),
        ),
    ),
)
def test_snapshot_builder_rejects_missing_or_duplicate_cue(
    balls: tuple[CalibratedBallObservation, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one cue"):
        build_snapshot_table_state(
            _frame(*balls),
            calibration_id="cal-1",
            source_ref="artifact:detections.json#frame=8",
            scene_generation=0,
        )
