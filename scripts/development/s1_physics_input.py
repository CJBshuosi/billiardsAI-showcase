"""Build raw, physically sourced S1 input, never a rendered recommendation demo.

Only video pixels and retained calibration are product inputs. Physics samples,
events and scenario expectations stay in a separate test-only oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.development.blackbox_report import atomic_json  # noqa: E402
from scripts.development.blackbox_source import SOURCE_ASSETS, pin  # noqa: E402

FPS = 60
INITIAL_HOLD_SECONDS = 75
HOLD_SECONDS = 12
CORNERS = [(180, 940), (1740, 940), (1740, 160), (180, 160)]
FELT = Path(__file__).with_name("assets") / "s1-felt.png"
FELT_SHA256 = SOURCE_ASSETS["scripts/development/assets/s1-felt.png"]["sha256"]
INITIAL = {"0": [1270, 300], "1": [1270, 800], "5": [1850, 950], "8": [700, 400]}


def _simulate(initial, target, pocket, *, speed, offset, previous=None):
    from billiards_ai.cloud.engine.pooltool_adapter import PooltoolIdealPathAdapter, pt
    from billiards_ai.shared.models.ideal_path import IdealPathPocket, IdealPathRequest

    adapter = PooltoolIdealPathAdapter()
    adapter._require_engine()
    request = IdealPathRequest(
        cue_ball={"ball_id": 0, "position": dict(zip(("x_mm", "y_mm"), initial["0"], strict=True))},
        object_balls=tuple(
            {"ball_id": int(key), "position": dict(zip(("x_mm", "y_mm"), value, strict=True))}
            for key, value in initial.items()
            if key != "0"
        ),
        target_ball_id=int(target),
        target_pocket=IdealPathPocket(pocket),
        intent="DIRECT_POT",
        target_ball_reviewed=True,
        target_pocket_reviewed=True,
        intent_reviewed=True,
        calibration_status="ACTIVE",
        table=dict(
            configuration_version="chinese-eight-ball-9ft-v1",
            length_mm=2540,
            width_mm=1270,
            ball_diameter_mm=57,
            corner_pocket_mouth_mm=86,
            side_pocket_mouth_mm=92,
            cushion_nose_width_mm=40,
        ),
    )
    if previous is None:
        system = adapter._build_system(request)
    else:
        system = previous.copy()
        system.reset_history()
        system.balls = {key: ball for key, ball in system.balls.items() if key in initial}
    engine_initial = {key: ball.state.rvw.tolist() for key, ball in system.balls.items()}
    pocket_id = {"TOP_CENTER": "rc", "TOP_RIGHT": "rt"}[pocket]
    angle = pt.pot.PottingConfig.default().calculate_angle(
        system.balls["0"], system.balls[target], system.table, system.table.pockets[pocket_id]
    )
    system.strike(V0=speed, phi=angle, theta=0, a=0, b=offset)
    simulated = pt.simulate(
        system, engine=adapter._physics_engine(), continuous=True, dt=1 / FPS, max_events=100
    )
    mapping = adapter._coordinate_mapping(simulated.table, request)
    events = [
        {"kind": str(e.event_type), "ids": list(e.ids), "seconds": float(e.time)}
        for e in simulated.events
    ]
    if not any(e["kind"] == "ball_pocket" and e["ids"] == [target, pocket_id] for e in events):
        raise ValueError("fixed physical input did not pot its designated object")
    if any(e["kind"] == "ball_pocket" and "0" in e["ids"] for e in events):
        raise ValueError("fixed physical input scratched the cue ball")
    histories = {key: tuple(ball.history_cts) for key, ball in simulated.balls.items()}
    if any(not history for history in histories.values()):
        raise ValueError("physical input lacks continuous histories")
    times = {key: [float(state.t) for state in history] for key, history in histories.items()}
    duration = max(values[-1] for values in times.values())
    if not math.isfinite(duration) or not 0 < duration <= 15:
        raise ValueError("physical input motion duration exceeds fixed budget")
    samples = []
    for frame in range(math.ceil(duration * FPS) + 1):
        seconds, balls = frame / FPS, {}
        for key, history in histories.items():
            index = max(0, bisect_right(times[key], seconds + 1e-9) - 1)
            state = history[index]
            if int(state.s) == pt.constants.pocketed:
                continue
            point = mapping.to_business_point(state.rvw[0])
            balls[key] = [point.x_mm, point.y_mm]
        samples.append({"seconds": seconds, "balls": balls})
    final = samples[-1]["balls"]
    if any(int(ball.state.s) != 0 for key, ball in simulated.balls.items() if key in final):
        raise ValueError("physical terminal pose still has moving/spinning balls")
    return (
        {
            "initial": initial,
            "final": final,
            "samples": samples,
            "events": events,
            "engine_initial": engine_initial,
            "engine_final": {
                key: ball.state.rvw.tolist()
                for key, ball in simulated.balls.items()
                if key in final
            },
            "parameters": {"speed_m_s": speed, "vertical_offset": offset},
            "target_ball": target,
            "pocket": pocket_id,
        },
        simulated,
    )


def simulate_case(name):
    if name not in {"inside", "outside"}:
        raise ValueError("unknown fixed S1 physical scenario")
    first, terminal = _simulate(
        INITIAL, "1", "TOP_CENTER", speed=1.5, offset=-0.4 if name == "inside" else -0.08
    )
    cue = first["final"]["0"]
    distance = math.hypot(cue[0] - 1270, cue[1] - 743)
    if (name == "inside" and distance > 80) or (name == "outside" and distance < 120):
        raise ValueError("physical endpoint does not qualify the fixed test case")
    shots = [first]
    if name == "inside":
        second, _ = _simulate(
            first["final"], "5", "TOP_RIGHT", speed=2.0, offset=0, previous=terminal
        )
        shots.append(second)
    return {
        "contract": "s1-physics-case/v1",
        "name": name,
        "fps": FPS,
        "engine": {"name": "pooltool-billiards", "version": "0.6.0"},
        "shots": shots,
    }


def input_calibration(device_id="s1-physical-input"):
    from billiards_ai.shared.calibration.homography import calibrate

    return calibrate(device_id, CORNERS)


def table_background():
    if pin(FELT)["sha256"] != FELT_SHA256:
        raise ValueError("approved felt asset changed")
    felt = cv2.imread(str(FELT))
    if felt is None:
        raise ValueError("felt asset cannot be decoded")
    frame = np.full((1080, 1920, 3), (19, 26, 25), dtype=np.uint8)
    cv2.rectangle(frame, (118, 98), (1802, 1002), (30, 42, 46), -1)
    cv2.rectangle(frame, (128, 108), (1792, 992), (42, 57, 65), 3)
    cv2.rectangle(frame, (158, 138), (1762, 962), (28, 59, 43), -1)
    # Keep the approved texture while bounding MJPEG size over the cold-start hold.
    texture = cv2.resize(felt, (1560, 780), interpolation=cv2.INTER_AREA)
    frame[160:940, 180:1740] = cv2.GaussianBlur(texture, (3, 3), 0)
    for x, y in [*CORNERS, (960, 160), (960, 940)]:
        cv2.circle(frame, (x, y), 30, (12, 18, 17), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 30, (59, 74, 65), 2, cv2.LINE_AA)
    for x in (375, 570, 765, 1155, 1350, 1545):
        for y in (126, 974):
            cv2.fillConvexPoly(
                frame,
                np.array([(x, y - 3), (x + 3, y), (x, y + 3), (x - 3, y)]),
                (143, 163, 149),
                cv2.LINE_AA,
            )
    return frame


def render_pose(background, balls):
    output = background.copy()
    colors = {"0": (238, 244, 243), "1": (25, 205, 245), "5": (23, 116, 236), "8": (28, 33, 35)}
    radius = round(28.5 * 1560 / 2540)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1] / radius
    inside = xx * xx + yy * yy <= 1
    z = np.sqrt(np.maximum(0, 1 - xx * xx - yy * yy))
    light = 0.50 + 0.50 * np.maximum(0, -0.28 * xx - 0.36 * yy + 0.88 * z)
    shine = np.exp(-((xx + 0.32) ** 2 + (yy + 0.42) ** 2) / 0.025) * 0.45
    for key, (x_mm, y_mm) in balls.items():
        x, y = round(180 + x_mm * 1560 / 2540), round(940 - y_mm * 780 / 1270)
        if not (radius <= x < 1920 - radius and radius <= y < 1080 - radius):
            raise ValueError("physical ball cannot be drawn within the source image")
        cv2.ellipse(
            output, (x + 3, y + 5), (radius + 2, radius), 0, 0, 360, (20, 45, 31), -1, cv2.LINE_AA
        )
        patch = output[y - radius : y + radius + 1, x - radius : x + radius + 1]
        base = np.array(colors[key], dtype=float)
        shaded = np.clip(
            base[None, None, :] * light[:, :, None] + 255 * shine[:, :, None], 0, 255
        ).astype(np.uint8)
        patch[inside] = shaded[inside]
        if key != "0":
            cv2.circle(output, (x, y), round(radius * 0.43), (238, 239, 233), -1, cv2.LINE_AA)
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(
                output,
                key,
                (x - text_size[0] // 2, y + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (28, 32, 31),
                1,
                cv2.LINE_AA,
            )
    return output


def _validate_initial_hold(seconds):
    if type(seconds) is not int or not INITIAL_HOLD_SECONDS <= seconds <= 120:
        raise ValueError("initial hold must be an integer from 75 to 120 seconds")


def frame_segments(case, *, initial_hold_seconds=INITIAL_HOLD_SECONDS):
    _validate_initial_hold(initial_hold_seconds)
    segments = []
    for index, shot in enumerate(case["shots"]):
        segments.append(
            {
                "kind": "hold",
                "frame_count": (initial_hold_seconds if index == 0 else HOLD_SECONDS) * FPS,
                "balls": shot["initial"],
            }
        )
        segments.append(
            {"kind": "motion", "frame_count": len(shot["samples"]), "samples": shot["samples"]}
        )
    segments.append(
        {
            "kind": "hold",
            "frame_count": HOLD_SECONDS * FPS,
            "balls": case["shots"][-1]["final"],
        }
    )
    return segments


def build_inputs(output, *, case_name=None, initial_hold_seconds=INITIAL_HOLD_SECONDS):
    if case_name not in (None, "inside", "outside"):
        raise ValueError("unknown physical input case")
    _validate_initial_hold(initial_hold_seconds)
    from billiards_ai.lab.mainline1_calibration import create_calibration_artifact
    from billiards_ai.lab.mainline1_calibration_registry import register_calibration
    from billiards_ai.lab.mainline1_capture import CaptureSetup

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    background = table_background()
    for name in ("inside", "outside") if case_name is None else (case_name,):
        root = output / name
        root.mkdir()
        case = simulate_case(name)
        video = root / "input.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (1920, 1080))
        if not writer.isOpened():
            raise ValueError("physical input MJPEG writer unavailable")
        frames, intervals = 0, []
        try:
            for segment in frame_segments(case, initial_hold_seconds=initial_hold_seconds):
                start = frames
                if segment["kind"] == "hold":
                    still = render_pose(background, segment["balls"])
                    images = (still for _ in range(segment["frame_count"]))
                else:
                    images = (render_pose(background, p["balls"]) for p in segment["samples"])
                for image in images:
                    writer.write(image)
                    frames += 1
                    if frames % FPS == 0 and video.stat().st_size > 1024**3:
                        raise ValueError("physical input exceeds video budget")
                if segment["kind"] == "motion":
                    intervals.append(
                        {"motion_start_frame": start, "final_static_frame": frames - 1}
                    )
        finally:
            writer.release()
        if video.stat().st_size > 1024**3:
            raise ValueError("physical input exceeds video budget")
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        reference_frame = None
        try:
            if capture.get(cv2.CAP_PROP_FPS) != FPS:
                raise ValueError("physical input violates source FPS")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape != (1080, 1920, 3):
                    raise ValueError("physical input violates source geometry")
                if decoded == 0:
                    reference_frame = frame.copy()
                decoded += 1
        finally:
            capture.release()
        if decoded != frames or reference_frame is None:
            raise ValueError("physical input frame decode mismatch")
        setup = CaptureSetup.model_validate(
            dict(
                capture_setup_id=f"s1-physics-{name}-v1",
                table_id="s1-synthetic-table",
                camera_device_id=f"s1-physics-{name}",
                lens_id="orthographic",
                mount_revision="1",
                distortion_revision="1",
                source_kind="file",
                requested_fps=FPS,
                geometry=dict(width=1920, height=1080, crop=[0, 0, 1920, 1080]),
            )
        )
        (root / "capture-setup.json").write_text(setup.model_dump_json(indent=2))
        calibration_root = root / "calibration-registry"
        artifact = create_calibration_artifact(
            video=video,
            frame_index=0,
            frame=reference_frame,
            corners_px=CORNERS,
            device_id=setup.camera_device_id,
            output_root=calibration_root,
            capture_setup=setup,
        )
        register_calibration(calibration_root, artifact, setup)
        atomic_json(root / "oracle.not-product-input.json", case)
        atomic_json(
            root / "manifest.json",
            {
                "contract": "s1-physical-input/v1",
                "synthetic_physics_only": True,
                "model_acceptance": "NOT_RUN",
                "display_acceptance": "NOT_RUN",
                "field_accuracy": "OPEN",
                "case": name,
                "fps": FPS,
                "frame_count": frames,
                "decoded_frames": decoded,
                "initial_hold_seconds": initial_hold_seconds,
                "hold_seconds": HOLD_SECONDS,
                "intervals": intervals,
                "video": pin(video),
                "generator": pin(Path(__file__)),
                "felt": pin(FELT),
                "calibration_id": artifact.artifact_id,
                "product_inputs": ["input.avi", "capture-setup.json", "calibration-registry"],
                "oracle_is_product_input": False,
            },
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("inside", "outside"))
    parser.add_argument("--initial-hold-seconds", type=int, default=INITIAL_HOLD_SECONDS)
    args = parser.parse_args()
    destination = build_inputs(
        args.output, case_name=args.case, initial_hold_seconds=args.initial_hold_seconds
    )
    print(json.dumps({"physical_input": str(destination), "model_acceptance": "NOT_RUN"}))


if __name__ == "__main__":
    main()
