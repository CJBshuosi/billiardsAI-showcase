from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.guidance_render import (
    build_guidance_scene,
    mobile_scene_payload,
    render_scene_frame,
    render_scene_video,
)
from billiards_ai.shared.calibration.homography import calibrate
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PlannedShot,
    PocketId,
    Recommendation,
)
from billiards_ai.shared.models.table_state import BallKind


def _shot() -> PlannedShot:
    return PlannedShot(
        candidate_id="candidate-1",
        target_track_id="target",
        target_identity="1",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=1900, y_mm=500),
        cue_path=(Point(x_mm=500, y_mm=500), Point(x_mm=1900, y_mm=500)),
        object_path=(Point(x_mm=1957, y_mm=500), Point(x_mm=2540, y_mm=1270)),
        cut_angle_deg=10,
        cue_travel_mm=1400,
        object_travel_mm=966,
        evidence_confidence=0.9,
        score_components={"pot_tolerance": 0.8},
        score=0.8,
        evidence_refs=("frame:1",),
    )


def _recommendation() -> Recommendation:
    return Recommendation(
        status=GuidanceStatus.GEOMETRY_ONLY,
        candidates=(_shot(),),
        calibration_id="cal-1",
        evidence_refs=("frame:1",),
        planner_config_hash="a" * 64,
    )


def _calibration():
    return calibrate("cam", [(0, 479), (853, 479), (853, 0), (0, 0)])


def test_scene_is_device_independent_and_mobile_serializable() -> None:
    scene = build_guidance_scene(_recommendation())
    payload = mobile_scene_payload(scene)
    assert payload["coordinate_space"] == "table_mm"
    assert payload["routes"][0]["ghost_ball_center_mm"] == [1900, 500]
    assert "pixel" not in str(payload)


def test_frame_renderer_draws_route_at_inverse_homography_pixels() -> None:
    original = np.zeros((480, 854, 3), dtype=np.uint8)
    rendered = render_scene_frame(original, build_guidance_scene(_recommendation()), _calibration())
    assert rendered.shape == original.shape
    assert np.count_nonzero(rendered != original) > 100
    assert np.count_nonzero(original) == 0


def _write_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (854, 480))
    assert writer.isOpened()
    for value in range(frame_count):
        writer.write(np.full((480, 854, 3), value * 10, dtype=np.uint8))
    writer.release()


def _count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def test_video_renderer_preserves_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    output = tmp_path / "overlay.avi"
    _write_video(source)
    result = render_scene_video(
        source, output, build_guidance_scene(_recommendation()), _calibration()
    )
    assert result == output
    assert _count_frames(output) == 3


def test_video_renderer_rejects_undecodable_input(tmp_path: Path) -> None:
    source = tmp_path / "bad.avi"
    source.write_bytes(b"not-video")
    try:
        render_scene_video(
            source,
            tmp_path / "output.avi",
            build_guidance_scene(_recommendation()),
            _calibration(),
        )
    except ValueError as exc:
        assert "decode" in str(exc)
    else:
        raise AssertionError("invalid video was accepted")
