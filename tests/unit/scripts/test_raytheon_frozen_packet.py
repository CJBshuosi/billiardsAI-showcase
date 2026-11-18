import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
FOLDER = ROOT / "scripts/diagnostics/raytheon_table"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(FOLDER))
    path = FOLDER / "frozen_check.py"
    assert path.exists(), "Frozen evaluator not implemented"
    spec = importlib.util.spec_from_file_location("frozen_check", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_geometry():
    lens = dict(
        width=1920,
        height=1080,
        camera_matrix=[[800, 0, 960], [0, 800, 540], [0, 0, 1]],
        coefficients=[0] * 5,
        status="DIAGNOSTIC_CANDIDATE_NOT_REGISTERED",
        production_ready=False,
    )
    candidate = dict(
        schema_version="1.0",
        status="FROZEN_LOCAL_METRIC_CANDIDATE_NOT_ACTIVE",
        production_ready=False,
        active_calibration_written=False,
        absolute_position_validated=False,
        normal_camera=[0, 0, 1],
        camera_distance_mm=1600,
        checkerboard=dict(columns=11, rows=8, square_size_mm=25),
        homography_for_placement_only=[[2, 0, -650], [0, -2, 1715], [0, 0, 1]],
    )

    def points(cx=960, cy=540):
        return np.array(
            [[cx + (x - 5) * 12.5, cy + (y - 3.5) * 12.5] for y in range(8) for x in range(11)]
        )

    return lens, candidate, points


def test_frozen_metric_does_not_fit_away_new_frame_error(module):
    lens, candidate, points = fixture_geometry()
    before = copy.deepcopy(candidate)
    good = module.measure_frozen(points(), lens, candidate, "center")
    assert good["max_pair_distance_error_mm"] < 1e-8
    assert good["mean_cell_mm"] == pytest.approx(25)
    bad_points = points()
    bad_points[:, 0] = 960 + (bad_points[:, 0] - 960) * 1.1
    bad = module.measure_frozen(bad_points, lens, candidate, "center")
    assert bad["max_pair_distance_error_mm"] >= 24.99
    assert bad["refit_performed"] is False
    assert candidate == before
    assert good["absolute_position_validated"] is False
    assert good["acceptance"] == "NOT_EVALUATED"


def test_tilted_metric_plane_uses_3d_distance_and_undistorts_lens(module):
    import cv2

    lens, candidate, _ = fixture_geometry()
    # Analytic orthonormal basis: n=(0,.6,.8), u=(1,0,0), v=(0,.8,-.6).
    # An incorrect raw image distance or camera-Z depth gives wrong lengths here.
    lens["coefficients"] = [0.08, -0.03, 0.001, -0.002, 0.005]
    candidate["normal_camera"] = [0, 0.6, 0.8]
    xyz = np.array(
        [
            [25 * (x - 5), 0.8 * 25 * (y - 3.5), 2000 - 0.6 * 25 * (y - 3.5)]
            for y in range(8)
            for x in range(11)
        ],
        dtype=float,
    )
    points = cv2.projectPoints(
        xyz,
        np.zeros(3),
        np.zeros(3),
        np.array(lens["camera_matrix"], float),
        np.array(lens["coefficients"]),
    )[0].reshape(-1, 2)
    result = module.measure_frozen(points, lens, candidate, "center")
    assert result["mean_cell_mm"] == pytest.approx(25, abs=1e-5)
    assert result["long_span_mean_mm"] == pytest.approx(250, abs=1e-5)
    assert result["short_span_mean_mm"] == pytest.approx(175, abs=1e-5)
    assert result["max_pair_distance_error_mm"] < 1e-5


@pytest.mark.parametrize("center", [(560, 340), (1360, 340), (560, 740), (1360, 740)])
def test_any_corner_is_accepted_without_direction_entry(module, center):
    lens, candidate, points = fixture_geometry()
    assert module.measure_frozen(points(*center), lens, candidate, "corner")["placement_matches"]
    assert not module.measure_frozen(points(), lens, candidate, "corner")["placement_matches"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_distance_mm", 0),
        ("camera_distance_mm", float("nan")),
        ("normal_camera", [0, 0, -1]),
        ("normal_camera", [0, 0, 2]),
        ("production_ready", True),
        ("absolute_position_validated", True),
    ],
)
def test_invalid_or_active_candidate_fails_closed(module, field, value):
    lens, candidate, points = fixture_geometry()
    candidate[field] = value
    with pytest.raises(ValueError):
        module.measure_frozen(points(), lens, candidate, "center")


def build_packet(target, diagnostic_packet_assets):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_table.py"),
            "--output",
            str(target),
            "--lens-candidate",
            str(diagnostic_packet_assets.lens),
            "--frozen-candidate",
            str(diagnostic_packet_assets.frozen),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "outcome",
    [
        "save",
        "repeat_center",
        "cancel",
        "second_timeout",
        "detect_fail",
        "wrong_lens",
        "wrong_controls",
    ],
)
def test_two_capture_session_returns_evidence_and_never_refits(
    module, tmp_path, monkeypatch, outcome, diagnostic_packet_assets
):
    target = tmp_path / "packet.zip"
    build_packet(target, diagnostic_packet_assets=diagnostic_packet_assets)
    root = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root)
    lens, candidate, points = fixture_geometry()
    lens["device_fingerprint"] = "test"
    (root / "lens-candidate.json").write_text(json.dumps(lens))
    candidate["lens_file_sha256"] = hashlib.sha256(
        (root / "lens-candidate.json").read_bytes()
    ).hexdigest()
    if outcome == "wrong_lens":
        candidate["lens_file_sha256"] = "0" * 64
    (root / "frozen-plane-candidate.json").write_text(json.dumps(candidate))
    manifest = json.loads((root / "packet-manifest.json").read_text())
    for name in ("lens-candidate.json", "frozen-plane-candidate.json"):
        data = (root / name).read_bytes()
        manifest["files"][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (root / "packet-manifest.json").write_text(json.dumps(manifest))
    frozen_before = (root / "frozen-plane-candidate.json").read_bytes()
    monkeypatch.setattr(
        module, "__file__", str(root / "scripts/diagnostics/raytheon_table/frozen_check.py")
    )

    def prompt(*_):
        if outcome == "cancel":
            raise KeyboardInterrupt
        return ""

    monkeypatch.setattr("builtins.input", prompt)
    assert hasattr(module, "main"), "Two-capture session not implemented"
    monkeypatch.setattr(
        module,
        "enumerate_usb_devices",
        lambda **_: [types.SimpleNamespace(durable_fingerprint="test", model_dump=lambda **_: {})],
    )
    spec = importlib.util.spec_from_file_location(
        "camera_check", ROOT / "scripts/diagnostics/raytheon_camera/camera_check.py"
    )
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    import cv2

    def bounded(command, *_, **kwargs):
        if "--worker" in command:
            assert kwargs["timeout"] == 45
            folder = Path(command[command.index("--output") + 1])
            if outcome == "second_timeout" and folder.name == "corner":
                return dict(status="TIMEOUT")
            folder.mkdir()
            controls = dict(module.table.EXPECTED_CONTROLS)
            if outcome == "wrong_controls":
                controls["exposure"] = -7
            report = dict(
                status="READBACK_COLLECTED_REVIEW_REQUIRED",
                mode_matches_request=True,
                decoded_geometry=[1920, 1080],
                sample_completed=True,
                cadence_without_recording=dict(within_5_percent=True),
                controls_after_mode=controls,
                controls_after_capture=controls,
            )
            (folder / "report.json").write_text(json.dumps(report))
            cv2.imwrite(str(folder / "baseline.png"), np.zeros((1080, 1920, 3), np.uint8))
        else:
            assert kwargs["timeout"] == 30
            if outcome == "detect_fail":
                return dict(status="ERROR")
            dest = Path(command[command.index("--detection-output") + 1])
            p = (
                points()
                if dest.parent.name == "center" or outcome == "repeat_center"
                else points(1360, 740)
            )
            dest.write_text(
                json.dumps(
                    dict(
                        image_points=p.tolist(),
                        source_sha256=hashlib.sha256(
                            (dest.parent / "baseline.png").read_bytes()
                        ).hexdigest(),
                    )
                )
            )
        return dict(status="COMPLETED")

    monkeypatch.setattr(camera, "run_bounded", bounded)
    monkeypatch.setitem(sys.modules, "camera_check", camera)
    output = tmp_path / "result"
    code = module.main(["--output", str(output)])
    assert code == (0 if outcome in ("save", "repeat_center") else 2)
    assert (root / "frozen-plane-candidate.json").read_bytes() == frozen_before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["refit_performed"] is False
    assert summary["active_calibration_written"] is False
    assert summary["absolute_position_validated"] is False
    assert summary["acceptance"] == "NOT_EVALUATED"
    if outcome == "save":
        review = json.loads((output / "frozen-review.json").read_text())
        assert len(review["boards"]) == 2
        assert max(b["max_pair_distance_error_mm"] for b in review["boards"]) < 1e-7
        assert summary["coverage"] == "CENTER_AND_CORNER_COLLECTED"
        assert (output / "frozen-plane-candidate.json").read_bytes() == frozen_before
    if outcome == "repeat_center":
        assert summary["coverage"] == "INCOMPLETE_REVIEW_REQUIRED"
    if outcome == "second_timeout":
        assert (output / "center/frozen-measurement.json").exists()
    with zipfile.ZipFile(tmp_path / "SEND_BACK-result.zip") as archive:
        assert archive.testzip() is None
        for name, info in summary["inventory"].items():
            assert hashlib.sha256(archive.read("result/" + name)).hexdigest() == info["sha256"]


def test_frozen_one_line_command_uses_frozen_entry(tmp_path, diagnostic_packet_assets):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    build_packet(
        downloads / "billiardsAI-rer-plane-recheck-20260905-v1.zip",
        diagnostic_packet_assets=diagnostic_packet_assets,
    )
    doc = ROOT / "docs/operations/rer-plane-recheck-20260905.md"
    assert doc.exists(), "One-line recheck instructions missing"
    command = doc.read_text().split("```powershell\n")[1].split("\n```")[0]
    assert "\n" not in command
    fake = tmp_path / "python with spaces"
    fake.write_text("#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe", str(fake)
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; function powershell.exe {{ & '{pwsh}' @args; "
        "$global:LASTEXITCODE=$LASTEXITCODE }; "
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", prefix + command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("[")))
    assert args[0].endswith("frozen_check.py")
    assert args[1] == "--output"


def test_clean_packet_import_and_candidate_binding(tmp_path, diagnostic_packet_assets):
    target = tmp_path / "packet.zip"
    build_packet(target, diagnostic_packet_assets=diagnostic_packet_assets)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("packet-manifest.json"))
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        c = json.loads(archive.read("frozen-plane-candidate.json"))
        assert (
            c["lens_file_sha256"] == hashlib.sha256(archive.read("lens-candidate.json")).hexdigest()
        )
        # Package the explicit synthetic flat plane, not an unavailable field result.
        assert c["test_fixture_only"] is True
        assert c["normal_camera"] == [0, 0, 1]
        assert c["method"] == "synthetic_flat_plane_for_ci_only"
        assert c["production_ready"] is False
        archive.extractall(tmp_path / "packet")
    root = tmp_path / "packet"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/diagnostics/raytheon_table/frozen_check.py"),
            "--help",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
