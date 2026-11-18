import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import types
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
FOLDER = ROOT / "scripts/diagnostics/raytheon_table"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(FOLDER))
    path = FOLDER / "board_check.py"
    assert path.is_file(), "Board diagnostic is not implemented"
    spec = importlib.util.spec_from_file_location("board_check", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def geometry():
    lens = {
        "width": 1920,
        "height": 1080,
        "camera_matrix": [[800, 0, 960], [0, 800, 540], [0, 0, 1]],
        "coefficients": [0.1, -0.02, 0.001, -0.002, 0.005],
        "status": "DIAGNOSTIC_CANDIDATE_NOT_REGISTERED",
        "production_ready": False,
    }

    def project(points):
        xyz = np.array([[x - 1270, 635 - y, 1900] for x, y in points], np.float64)
        return cv2.projectPoints(
            xyz,
            np.zeros(3),
            np.zeros(3),
            np.array(lens["camera_matrix"], np.float64),
            np.array(lens["coefficients"]),
        )[0].reshape(-1, 2)

    rail_mm = [
        (300, 0),
        (2240, 0),
        (2540, 300),
        (2540, 970),
        (2240, 1270),
        (300, 1270),
        (0, 970),
        (0, 300),
    ]
    return lens, project, {"rails_px": project(rail_mm).tolist(), "confirmed": True}


def test_known_board_lengths_are_evaluated_without_using_board_to_fit(module):
    lens, project, payload = geometry()
    rails = module.evaluate_rails(payload, lens)
    h = np.array(rails["homography_undistorted_px_to_mm"])
    points = project([(1145 + x * 25, 547.5 + y * 25) for y in range(8) for x in range(11)])
    result = module.measure_board(points, lens, h, "center")
    assert result["placement_matches"] is True
    assert result["max_pair_distance_error_mm"] < 0.02
    assert result["mean_cell_mm"] == pytest.approx(25, abs=0.002)
    assert result["long_span_mean_mm"] == pytest.approx(250, abs=0.02)
    wrong = h.copy()
    wrong[0, :] *= 1.1
    bad = module.measure_board(points, lens, wrong, "center")
    assert bad["max_pair_distance_error_mm"] > 24
    np.testing.assert_array_equal(h, np.array(rails["homography_undistorted_px_to_mm"]))
    assert result["absolute_position_validated"] is False


def test_repeated_center_board_does_not_count_as_corner_coverage(module):
    lens, project, payload = geometry()
    h = module.evaluate_rails(payload, lens)["homography_undistorted_px_to_mm"]
    points = project([(1145 + x * 25, 547.5 + y * 25) for y in range(8) for x in range(11)])
    assert module.measure_board(points, lens, h, "lower_left")["placement_matches"] is False


def test_auto_roi_finds_small_board_in_original_frame_without_manual_crosses(module):
    frame = np.full((1080, 1920, 3), (30, 90, 30), np.uint8)
    frame[380:570, 690:934] = 220
    for row in range(9):
        for col in range(12):
            if (row + col) % 2 == 0:
                frame[
                    394 + row * 18 : 394 + (row + 1) * 18, 704 + col * 18 : 704 + (col + 1) * 18
                ] = 30
    detected = module.detect_board(frame)
    assert np.array(detected["image_points"]).shape == (88, 2)
    points = np.array(detected["image_points"])
    assert 700 < points[:, 0].min() < 730
    assert 880 < points[:, 0].max() < 930
    assert 390 < points[:, 1].min() < 430
    assert 515 < points[:, 1].max() < 555


def test_missing_board_is_not_accepted(module):
    with pytest.raises(ValueError, match="棋盘"):
        module.detect_board(np.full((1080, 1920, 3), (30, 90, 30), np.uint8))


def test_unconfirmed_flatness_cannot_finish_rail_stage(module):
    lens, _, payload = geometry()
    payload["confirmed"] = False
    with pytest.raises(ValueError):
        module.evaluate_rails(payload, lens)


def test_five_positions_report_local_lengths_but_never_absolute_acceptance(module):
    lens, project, payload = geometry()
    centers = [(1270, 635), (400, 300), (2140, 300), (400, 970), (2140, 970)]
    boards = []
    for name, (cx, cy) in zip(module.POSITIONS, centers, strict=True):
        points = project(
            [(cx - 125 + x * 25, cy - 87.5 + y * 25) for y in range(8) for x in range(11)]
        )
        boards.append({"position": name, "image_points": points.tolist()})
    result = module.evaluate_all(payload, lens, boards)
    assert result["coverage"] == "FIVE_REGIONS_COLLECTED"
    assert len(result["boards"]) == 5
    assert result["absolute_position_validated"] is False
    assert result["production_ready"] is False
    assert result["reference_check"] == "NOT_EVALUATED"
    boards[1]["image_points"] = boards[0]["image_points"]
    assert module.evaluate_all(payload, lens, boards)["coverage"] == "INCOMPLETE_REVIEW_REQUIRED"
    with pytest.raises(ValueError):
        module.evaluate_all(payload, lens, boards[:4])


def test_board_packet_clean_import_cli_detection_and_hashes(tmp_path, diagnostic_packet_assets):
    target = tmp_path / "board.zip"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_table.py"),
            "--output",
            str(target),
            "--lens-candidate",
            str(diagnostic_packet_assets.lens),
            "--board-check",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["packet_version"] == "rer-table-board-20260905-v2"
        archive.extractall(tmp_path / "packet")
    root = tmp_path / "packet"
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    cmd = [sys.executable, str(root / "scripts/diagnostics/raytheon_table/board_check.py")]
    help_result = subprocess.run(
        cmd + ["--help"], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    assert help_result.returncode == 0, help_result.stderr
    frame = np.full((1080, 1920, 3), (30, 90, 30), np.uint8)
    frame[380:570, 690:934] = 220
    for row in range(9):
        for col in range(12):
            if (row + col) % 2 == 0:
                frame[
                    394 + row * 18 : 394 + (row + 1) * 18, 704 + col * 18 : 704 + (col + 1) * 18
                ] = 30
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), frame)
    dest = tmp_path / "detected.json"
    result = subprocess.run(
        cmd + ["--detect-frame", str(source), "--detection-output", str(dest)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(dest.read_text())["image_points"]) == 88
    again = subprocess.run(
        cmd + ["--detect-frame", str(source), "--detection-output", str(dest)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert again.returncode != 0


def packet(tmp_path, diagnostic_packet_assets):
    target = tmp_path / "packet.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_table.py"),
            "--output",
            str(target),
            "--lens-candidate",
            str(diagnostic_packet_assets.lens),
            "--board-check",
        ],
        check=True,
        capture_output=True,
    )
    return target


def test_board_one_liner_selects_board_entry_and_restores_environment(
    tmp_path, diagnostic_packet_assets
):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    packet(tmp_path, diagnostic_packet_assets=diagnostic_packet_assets).rename(
        downloads / "billiardsAI-rer-table-board-20260905-v2.zip"
    )
    fake = tmp_path / "python with spaces"
    fake.write_text("#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    command = (
        (ROOT / "docs/operations/rer-table-board-20260905.md")
        .read_text()
        .split("```powershell\n")[1]
        .split("\n```")[0]
    )
    assert "\n" not in command
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe", str(fake)
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; $env:PYTHONPATH='original'; "
        f"function powershell.exe {{ & '{pwsh}' @args; "
        "$global:LASTEXITCODE=$LASTEXITCODE }; "
    )
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            prefix
            + command
            + "; if($env:PYTHONPATH -ne 'original'){throw 'Environment not restored'}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
    assert args[0].endswith("board_check.py")
    assert args[1] == "--output"


@pytest.mark.parametrize("outcome", ["save", "cancel", "capture_timeout", "detect_failure"])
def test_board_session_always_returns_evidence_without_activation(
    module, tmp_path, monkeypatch, outcome, diagnostic_packet_assets
):
    root = tmp_path / "packet"
    with zipfile.ZipFile(
        packet(tmp_path, diagnostic_packet_assets=diagnostic_packet_assets)
    ) as archive:
        archive.extractall(root)
    lens, project, payload = geometry()
    lens["device_fingerprint"] = "test"
    (root / "lens-candidate.json").write_text(json.dumps(lens))
    manifest = json.loads((root / "packet-manifest.json").read_text())
    manifest["files"]["lens-candidate.json"]["sha256"] = hashlib.sha256(
        (root / "lens-candidate.json").read_bytes()
    ).hexdigest()
    (root / "packet-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        module, "__file__", str(root / "scripts/diagnostics/raytheon_table/board_check.py")
    )
    monkeypatch.setattr("builtins.input", lambda *_: "")
    monkeypatch.setattr(
        module,
        "enumerate_usb_devices",
        lambda **_: [types.SimpleNamespace(durable_fingerprint="test", model_dump=lambda **_: {})],
    )
    camera_spec = importlib.util.spec_from_file_location(
        "camera_check", ROOT / "scripts/diagnostics/raytheon_camera/camera_check.py"
    )
    camera = importlib.util.module_from_spec(camera_spec)
    camera_spec.loader.exec_module(camera)
    centers = dict(
        zip(
            module.POSITIONS,
            [(1270, 635), (400, 300), (2140, 300), (400, 970), (2140, 970)],
            strict=True,
        )
    )
    captures = []

    def bounded(command, *_, **kwargs):
        if "--worker" in command:
            if outcome == "capture_timeout":
                return {"status": "TIMEOUT"}
            assert kwargs["timeout"] == 45
            assert "--project-root" not in command
            folder = Path(command[command.index("--output") + 1])
            folder.mkdir()
            captures.append(folder.name)
            report = {
                "status": "READBACK_COLLECTED_REVIEW_REQUIRED",
                "mode_matches_request": True,
                "decoded_geometry": [1920, 1080],
                "sample_completed": True,
                "cadence_without_recording": {"within_5_percent": True},
                "controls_after_mode": module.table.EXPECTED_CONTROLS,
                "controls_after_capture": module.table.EXPECTED_CONTROLS,
            }
            (folder / "report.json").write_text(json.dumps(report))
            cv2.imwrite(str(folder / "baseline.png"), np.zeros((1080, 1920, 3), np.uint8))
        else:
            assert kwargs["timeout"] == 30
            if outcome == "detect_failure":
                return {"status": "ERROR"}
            dest = Path(command[command.index("--detection-output") + 1])
            cx, cy = centers[dest.parent.name]
            points = project(
                [(cx - 125 + x * 25, cy - 87.5 + y * 25) for y in range(8) for x in range(11)]
            )
            dest.write_text(json.dumps({"image_points": points.tolist()}))
        return {"status": "COMPLETED"}

    monkeypatch.setattr(camera, "run_bounded", bounded)
    monkeypatch.setitem(sys.modules, "camera_check", camera)
    threads = []
    errors = []

    def browser(url):
        def post():
            try:
                req = urllib.request.Request(
                    url + outcome,
                    data=json.dumps(payload if outcome == "save" else {}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    assert response.status == 200
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=post, daemon=True)
        t.start()
        threads.append(t)
        return True

    monkeypatch.setattr(module.webbrowser, "open", browser)
    output = tmp_path / "result"
    code = module.main(["--output", str(output)])
    for t in threads:
        t.join(timeout=12)
    assert not errors
    assert code == (0 if outcome == "save" else 2)
    if outcome in ("save", "cancel"):
        assert captures == list(module.POSITIONS)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["production_ready"] is False
    assert summary["absolute_position_validated"] is False
    assert summary["active_calibration_written"] is False
    if outcome == "save":
        assert summary["coverage"] == "FIVE_REGIONS_COLLECTED"
    with zipfile.ZipFile(tmp_path / "SEND_BACK-result.zip") as archive:
        assert archive.testzip() is None
        for name, info in summary["inventory"].items():
            assert hashlib.sha256(archive.read("result/" + name)).hexdigest() == info["sha256"]
