from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.shared.calibration.homography import calibrate

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (320, 180),
    )
    assert writer.isOpened()
    cue_x_positions = (50, 50, 90, 130, 130, 130)
    for frame_id in range(6):
        image = np.full((180, 320, 3), (20, 90, 20), dtype=np.uint8)
        cv2.circle(image, (cue_x_positions[frame_id], 90), 7, (245, 245, 245), -1)
        writer.write(image)
    writer.release()
    return path


def _write_observations(path: Path) -> Path:
    cue_positions = (500, 500, 520, 540, 540, 540)
    frames = []
    for frame_id, cue_x in enumerate(cue_positions):
        frames.append(
            {
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
                        "center_mm": [1500, 650],
                        "detection_confidence": 0.9,
                        "group": "solid",
                        "group_confidence": 0.9,
                        "bbox_xyxy_px": [160, 70, 175, 85],
                    },
                ],
            }
        )
    path.write_text(
        json.dumps({"schema_version": "1.0", "frames": frames}),
        encoding="utf-8",
    )
    return path


def _write_calibration(path: Path) -> Path:
    calibration = calibrate(
        "synthetic-camera",
        ((0, 180), (320, 180), (320, 0), (0, 0)),
    )
    path.write_text(calibration.model_dump_json(), encoding="utf-8")
    return path


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    return environment


def test_cli_publishes_machine_readable_result(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/run_field_route_replay.py",
            "--video",
            str(_write_video(tmp_path / "input.avi")),
            "--observations",
            str(_write_observations(tmp_path / "detections.json")),
            "--calibration",
            str(_write_calibration(tmp_path / "calibration.json")),
            "--context-group",
            "solid",
            "--output-root",
            str(tmp_path / "runs"),
            "--stable-duration-ms",
            "100",
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert Path(result["presentation"]).name == "presentation.mp4"
    assert result["frame_count"] == 6
    assert result["guidance_generation_count"] == 2


def test_cli_returns_usage_exit_for_invalid_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/algorithm/run_field_route_replay.py"],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 64
    assert "arguments invalid" in completed.stderr


def test_cli_returns_evidence_exit_for_invalid_calibration(tmp_path: Path) -> None:
    bad_calibration = tmp_path / "calibration.json"
    bad_calibration.write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm/run_field_route_replay.py",
            "--video",
            str(_write_video(tmp_path / "input.avi")),
            "--observations",
            str(_write_observations(tmp_path / "detections.json")),
            "--calibration",
            str(bad_calibration),
            "--context-group",
            "solid",
            "--output-root",
            str(tmp_path / "runs"),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "field route replay failed" in completed.stderr


def test_windows_wrapper_does_not_change_execution_policy() -> None:
    wrapper = (REPOSITORY_ROOT / "scripts/algorithm/windows/run-field-route-replay.ps1").read_text(
        encoding="utf-8"
    )

    assert "ExecutionPolicy" not in wrapper
    assert "Resolve-Path -LiteralPath" in wrapper
    assert "$LASTEXITCODE" in wrapper


def test_windows_wrapper_accepts_verified_mainline1_artifact() -> None:
    wrapper = (REPOSITORY_ROOT / "scripts/algorithm/windows/run-field-route-replay.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$Mainline1Artifact" in wrapper
    assert '"--mainline1-artifact", $ResolvedMainline1Artifact' in wrapper
    assert "Specify exactly one of -Mainline1Artifact or -Observations." in wrapper
    assert "-Mainline1Artifact cannot be combined with -Calibration or -ShotWindows." in wrapper


def test_windows_wrapper_preserves_legacy_observation_mode() -> None:
    wrapper = (REPOSITORY_ROOT / "scripts/algorithm/windows/run-field-route-replay.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$Observations" in wrapper
    assert "Legacy -Observations mode requires -Calibration." in wrapper
    assert '"--observations", $ResolvedObservations' in wrapper
    assert '"--calibration", $ResolvedCalibration' in wrapper
