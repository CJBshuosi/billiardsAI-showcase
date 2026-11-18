"""Qualify raw physics input; these tests never claim model/display success."""

import importlib
from functools import lru_cache
from math import hypot
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path("scripts/development/s1_physics_input.py")


def tool():
    assert SCRIPT.is_file(), "physical S1 input generator is missing"
    return importlib.import_module("scripts.development.s1_physics_input")


@lru_cache
def inside_case():
    return tool().simulate_case("inside")


def test_physics_has_cue_first_then_contact_then_object_motion():
    inside = inside_case()
    shot = inside["shots"][0]
    assert inside["engine"]["version"] == "0.6.0"
    contact = next(e for e in shot["events"] if e["kind"] == "ball_ball")
    assert set(contact["ids"]) == {"0", "1"}
    assert contact["seconds"] > 0
    before = [p for p in shot["samples"] if p["seconds"] < contact["seconds"]]
    assert len(before) >= 2
    assert all(p["balls"]["1"] == [1270, 800] for p in before)
    assert any(p["balls"]["0"] != [1270, 300] for p in before)
    assert any(p["balls"].get("1") not in (None, [1270, 800]) for p in shot["samples"])
    assert any(e["kind"] == "ball_pocket" and e["ids"] == ["1", "rc"] for e in shot["events"])
    assert not any(e["kind"] == "ball_pocket" and "0" in e["ids"] for e in shot["events"])


def test_next_shot_starts_from_the_previous_physical_terminal_pose():
    inside = inside_case()
    assert len(inside["shots"]) == 2
    first, second = inside["shots"]
    assert "1" not in first["final"]
    assert second["initial"] == first["final"]
    assert "engine_final" in first, "raw terminal state must be preserved, not reconstructed"
    assert second["engine_initial"] == first["engine_final"]
    assert all(b["balls"]["8"] == [700, 400] for s in inside["shots"] for b in s["samples"])
    assert any(e["kind"] == "ball_pocket" and e["ids"] == ["5", "rt"] for e in second["events"])
    assert not any(e["kind"] == "ball_pocket" and "0" in e["ids"] for e in second["events"])
    x, y = first["final"]["0"]
    assert hypot(x - 1270, y - 743) <= 80


def test_outside_case_really_misses_the_original_stop_region_without_a_scratch():
    case = tool().simulate_case("outside")
    assert len(case["shots"]) == 1
    shot = case["shots"][0]
    assert "1" not in shot["final"] and "0" in shot["final"]
    x, y = shot["final"]["0"]
    assert hypot(x - 1270, y - 743) >= 120
    assert not any(e["kind"] == "ball_pocket" and "0" in e["ids"] for e in shot["events"])


def test_rendered_pixels_follow_physical_poses_and_have_a_real_calibration():
    inside = inside_case()
    module = tool()
    background = module.table_background()
    first = module.render_pose(background, inside["shots"][0]["initial"])
    assert first.shape == (1080, 1920, 3) and first.dtype == np.uint8
    assert np.std(background[180:900, 220:1700]) > 1
    from billiards_ai.shared.calibration.homography import pixel_to_table

    calibration = module.input_calibration()
    white = pixel_to_table(calibration, 960, 756)
    assert abs(white.x_mm - 1270) <= 2 and abs(white.y_mm - 300) <= 2
    # White/yellow/orange/black centres are actual ball pixels, not drawn advice.
    assert min(first[756, 960]) > 170
    assert first[449, 972, 2] > first[449, 972, 0]
    assert np.array_equal(background, module.table_background())


def test_generator_refuses_to_overwrite_an_existing_input(tmp_path):
    module = tool()
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        module.build_inputs(tmp_path)
    assert marker.read_text() == "original"


def test_explicit_cold_start_hold_preserves_the_same_physical_shots():
    module = tool()
    original = module.frame_segments(inside_case())
    extended = module.frame_segments(inside_case(), initial_hold_seconds=120)
    assert original[0]["frame_count"] == 75 * 60
    assert extended[0]["frame_count"] == 120 * 60
    assert extended[0]["balls"] == original[0]["balls"]
    assert extended[1:] == original[1:]
    assert sum(item["frame_count"] for item in extended) < 12000


@pytest.mark.parametrize("hold", [0, 74, 121, True, 120.5])
def test_cold_start_input_extension_is_bounded_before_creating_files(tmp_path, hold):
    destination = tmp_path / "input"
    with pytest.raises(ValueError, match="initial hold"):
        tool().build_inputs(destination, initial_hold_seconds=hold)
    assert not destination.exists()


def test_source_pixels_keep_the_object_still_until_the_real_contact():
    module = tool()
    shot = inside_case()["shots"][0]
    background = module.table_background()
    before = module.render_pose(background, shot["initial"])
    early = next(p for p in shot["samples"] if p["seconds"] >= 0.1)
    first_contact = next(e["seconds"] for e in shot["events"] if e["kind"] == "ball_ball")
    assert early["seconds"] < first_contact
    after = module.render_pose(background, early["balls"])
    assert not np.array_equal(before[736:776, 940:980], after[736:776, 940:980])
    assert np.array_equal(before[435:463, 946:974], after[435:463, 946:974])


def test_fixture_geometry_has_real_s1_value_but_is_not_a_model_observation():
    from billiards_ai.lab.planning.rules import EightBallContext
    from billiards_ai.lab.planning.s1 import recommend_s1
    from billiards_ai.shared.models.geometry import Point
    from billiards_ai.shared.models.table_state import (
        BallKind,
        CanonicalBall,
        CanonicalTableState,
        MotionStatus,
    )

    module = tool()
    # This is input qualification against test truth, never passed to a field runtime.
    state = CanonicalTableState(
        frame_id=0,
        timestamp_ms=0,
        calibration_id="fixture-geometry-only",
        motion_status=MotionStatus.STATIONARY,
        source_refs=("test-oracle-not-product-input",),
        balls=tuple(
            CanonicalBall(
                track_id=key,
                identity="cue" if key == "0" else key,
                kind=BallKind.CUE
                if key == "0"
                else BallKind.EIGHT
                if key == "8"
                else BallKind.SOLID,
                center=Point(x_mm=xy[0], y_mm=xy[1]),
                radius_mm=28.575,
                identity_confidence=1,
                detection_confidence=1,
            )
            for key, xy in module.INITIAL.items()
        ),
    )
    result = recommend_s1(state, EightBallContext(group="open"))
    assert result.route.target_track_id == "1"
    assert result.region is not None and result.region.next_target_track_id == "5"
    assert result.region.physical_witnesses >= 2


def test_absent_fixed_engine_never_falls_back_to_handwritten_motion(monkeypatch):
    from billiards_ai.cloud.engine import pooltool_adapter
    from billiards_ai.cloud.engine.ideal_path import EngineUnavailableError

    module = tool()
    monkeypatch.setattr(pooltool_adapter, "pt", None)
    with pytest.raises(EngineUnavailableError, match="unavailable"):
        module.simulate_case("inside")


def test_actual_input_schedule_allows_cold_preparation_then_preserves_every_motion_sample():
    module = tool()
    case = inside_case()
    segments = module.frame_segments(case)
    # Preparation cap 60s + request cap 8s + at least 4s visible before striking.
    assert segments[0]["kind"] == "hold"
    assert segments[0]["frame_count"] / 60 >= 72
    assert segments[0]["balls"] == case["shots"][0]["initial"]
    motion = [part for part in segments if part["kind"] == "motion"]
    assert len(motion) == 2
    for part, shot in zip(motion, case["shots"], strict=True):
        assert part["samples"] is shot["samples"]
        assert part["frame_count"] == len(shot["samples"])
    assert segments[2]["kind"] == "hold"
    assert segments[2]["frame_count"] / 60 >= 12
    assert segments[2]["balls"] == case["shots"][0]["final"]
    assert segments[-1]["balls"] == case["shots"][-1]["final"]
    assert segments[-1]["frame_count"] / 60 >= 12
    assert sum(part["frame_count"] for part in segments) / 60 <= 130


def test_calibration_reference_comes_from_the_written_video_not_the_renderer(tmp_path, monkeypatch):
    import hashlib
    import json

    import cv2

    module = tool()
    # Isolate codec/calibration wiring with one real source frame, not a full scenario claim.
    monkeypatch.setattr(
        module,
        "frame_segments",
        lambda case, **kw: [
            {"kind": "hold", "frame_count": 1, "balls": case["shots"][0]["initial"]}
        ],
    )
    root = module.build_inputs(tmp_path / "short-codec-fixture")
    for name in ("inside", "outside"):
        directory = root / name
        capture = cv2.VideoCapture(str(directory / "input.avi"))
        try:
            ok, decoded = capture.read()
            assert ok
        finally:
            capture.release()
        ok, encoded = cv2.imencode(".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 94])
        assert ok
        manifest = json.loads((directory / "manifest.json").read_bytes())
        calibration = directory / "calibration-registry" / manifest["calibration_id"]
        reference = (calibration / "reference-frame.jpg").read_bytes()
        assert reference == encoded.tobytes()
        identity = json.loads((calibration / "manifest.json").read_bytes())
        assert identity["reference_frame_sha256"] == hashlib.sha256(reference).hexdigest()
        assert identity["source_video_sha256"] == module.pin(directory / "input.avi")["sha256"]


def test_selected_case_does_not_regenerate_the_already_completed_other_case(tmp_path, monkeypatch):
    import json

    module = tool()
    original = module.simulate_case
    calls = []

    def simulate(name):
        calls.append(name)
        return original(name)

    monkeypatch.setattr(module, "simulate_case", simulate)
    monkeypatch.setattr(
        module,
        "frame_segments",
        lambda case, **kw: [
            {"kind": "hold", "frame_count": 1, "balls": case["shots"][0]["initial"]}
        ],
    )
    root = module.build_inputs(tmp_path / "only-outside", case_name="outside")
    assert calls == ["outside"]
    assert not (root / "inside").exists()
    manifest = json.loads((root / "outside/manifest.json").read_bytes())
    assert manifest["case"] == "outside" and manifest["decoded_frames"] == 1


def test_invalid_case_is_rejected_before_output_or_physics(tmp_path, monkeypatch):
    module = tool()
    monkeypatch.setattr(
        module, "simulate_case", lambda name: pytest.fail("invalid case ran physics")
    )
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="case"):
        module.build_inputs(root, case_name="other")
    assert not root.exists()


@pytest.mark.parametrize("hold", [75, 120])
def test_cli_passes_selected_case_to_the_existing_builder(tmp_path, monkeypatch, hold):
    import sys

    module = tool()
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["s1_physics_input", "--output", str(tmp_path), "--case", "outside"]
        + ([] if hold == 75 else ["--initial-hold-seconds", str(hold)]),
    )
    monkeypatch.setattr(module, "build_inputs", lambda path, **kw: calls.append((path, kw)) or path)
    module.main()
    assert calls == [(tmp_path, {"case_name": "outside", "initial_hold_seconds": hold})]
