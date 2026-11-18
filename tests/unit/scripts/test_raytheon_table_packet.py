import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/diagnostics/raytheon_table/table_check.py"


@pytest.fixture
def module():
    assert SCRIPT.is_file(), "Table diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("table_check", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scene():
    k = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], np.float64)
    d = np.array([0.12, -0.02, 0.001, -0.002, 0.005])

    def project(points):
        xyz = np.array([[x - 1270, 635 - y, 1900] for x, y in points], np.float64)
        return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2).tolist()

    rails = [
        (300, 0),
        (2240, 0),
        (2540, 300),
        (2540, 970),
        (2240, 1270),
        (300, 1270),
        (0, 970),
        (0, 300),
    ]
    refs = [[300, 300], [2240, 300], [1270, 635], [300, 970], [2240, 970]]
    lens = {
        "width": 1920,
        "height": 1080,
        "camera_matrix": k.tolist(),
        "coefficients": d.tolist(),
        "status": "DIAGNOSTIC_CANDIDATE_NOT_REGISTERED",
        "production_ready": False,
    }
    payload = {
        "rails_px": project(rails),
        "references_px": project(refs),
        "references_mm": refs,
        "confirmed": True,
    }
    return lens, payload


def test_distorted_rails_and_independent_points_return_mm_without_activation(module):
    lens, payload = scene()
    r = module.evaluate(payload, lens)
    assert r["reference_check"] == "WITHIN_20MM_REVIEW_REQUIRED"
    assert r["max_reference_error_mm"] < 0.01
    np.testing.assert_allclose(r["predicted_reference_mm"], payload["references_mm"], atol=0.01)
    assert r["production_ready"] is False
    assert r["active_calibration_written"] is False
    assert r["coordinate_origin"] == "virtual_lower_left_cushion_nose_intersection"


def test_bad_independent_reference_is_reported_not_fitted_away(module):
    lens, p = scene()
    original = module.evaluate(p, lens)
    p["references_mm"][2][0] += 40
    r = module.evaluate(p, lens)
    assert r["reference_check"] == "FAILED"
    assert r["max_reference_error_mm"] == pytest.approx(40, abs=0.01)
    assert r["homography_undistorted_px_to_mm"] == original["homography_undistorted_px_to_mm"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unconfirmed",
        "nan",
        "duplicate",
        "swapped",
        "tiny",
        "outside",
        "ref_duplicate",
        "ref_cluster",
    ],
)
def test_bad_geometry_or_unmeasured_input_is_rejected(module, change):
    lens, p = scene()
    if change == "missing":
        p["rails_px"].pop()
    if change == "unconfirmed":
        p["confirmed"] = False
    if change == "nan":
        p["references_mm"][0][0] = float("nan")
    if change == "duplicate":
        p["rails_px"][1] = p["rails_px"][0]
    if change == "swapped":
        p["rails_px"][2:4], p["rails_px"][4:6] = p["rails_px"][4:6], p["rails_px"][2:4]
    if change == "tiny":
        p["rails_px"][1] = [p["rails_px"][0][0] + 1, p["rails_px"][0][1]]
    if change == "outside":
        p["references_px"][0] = [-1, 20]
    if change == "ref_duplicate":
        p["references_mm"][1] = p["references_mm"][0]
    if change == "ref_cluster":
        p["references_mm"] = [[500 + i * 20, 500 + i * 5] for i in range(5)]
    with pytest.raises(ValueError):
        module.evaluate(p, lens)


def test_capture_report_does_not_accept_fallback_or_controls_drift(module):
    report = {
        "status": "READBACK_COLLECTED_REVIEW_REQUIRED",
        "mode_matches_request": True,
        "decoded_geometry": [1920, 1080],
        "sample_completed": True,
        "cadence_without_recording": {"within_5_percent": True},
        "controls_after_mode": module.EXPECTED_CONTROLS.copy(),
        "controls_after_capture": module.EXPECTED_CONTROLS.copy(),
    }
    module.validate_capture(report)
    report["controls_after_capture"]["exposure"] = -7
    with pytest.raises(ValueError):
        module.validate_capture(report)
    report["controls_after_capture"]["exposure"] = -8
    report["cadence_without_recording"]["within_5_percent"] = False
    with pytest.raises(ValueError):
        module.validate_capture(report)


def test_http_save_is_token_scoped_and_keeps_failure_evidence(module, tmp_path):
    lens, payload = scene()
    state = {}
    server = module.make_server(
        tmp_path, np.zeros((1080, 1920, 3), np.uint8), lens, "test-secret", state, port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, p):
        return urllib.request.urlopen(
            urllib.request.Request(
                base + path,
                data=json.dumps(p).encode(),
                headers={"Content-Type": "application/json"},
            )
        )

    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            post("/bad/save", payload)
        assert error.value.code == 404
        assert not (tmp_path / "table-review.json").exists()
        payload["references_mm"][2][0] += 40
        with post("/test-secret/save", payload) as response:
            result = json.load(response)
        assert result["reference_check"] == "FAILED"
        assert (
            json.loads((tmp_path / "table-review.json").read_text())["max_reference_error_mm"] > 39
        )
        assert state["done"]
        assert (
            hashlib.sha256((tmp_path / "reference-frame.png").read_bytes()).hexdigest()
            == result["source_frame_sha256"]
        )
        assert not (tmp_path / "registry.json").exists()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def build_packet(tmp_path, diagnostic_packet_assets):
    target = tmp_path / "table.zip"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/diagnostics/package_raytheon_table.py"),
            "--output",
            str(target),
            "--lens-candidate",
            str(diagnostic_packet_assets.lens),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return target


def test_packet_hashes_clean_import_and_no_overwrite(tmp_path, diagnostic_packet_assets):
    target = build_packet(tmp_path, diagnostic_packet_assets=diagnostic_packet_assets)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert manifest["baseline_commit"].startswith("9a14495")
        assert manifest["production_ready"] is False
        for name, info in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == info["sha256"]
        assert not any(
            name.endswith((".db", ".sqlite", ".avi", ".mp4")) for name in archive.namelist()
        )
        archive.extractall(tmp_path / "unpacked")
    root = tmp_path / "unpacked"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/diagnostics/raytheon_table/table_check.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    before = target.read_bytes()
    with pytest.raises(AssertionError):
        build_packet(tmp_path, diagnostic_packet_assets=diagnostic_packet_assets)
    assert target.read_bytes() == before


def test_delivered_one_liner_spaces_and_environment_restored(tmp_path, diagnostic_packet_assets):
    pwsh = os.environ.get("RAYTHEON_TEST_PWSH")
    if not pwsh:
        pytest.skip("PowerShell not provided")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    build_packet(downloads, diagnostic_packet_assets=diagnostic_packet_assets).rename(
        downloads / "billiardsAI-rer-table-check-20260905-v1.zip"
    )
    fake = tmp_path / "python with spaces"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,sys,os\n"
        'print(json.dumps({"args":sys.argv[1:],"path":os.environ["PYTHONPATH"]}))\n'
    )
    fake.chmod(0o755)
    doc = (ROOT / "docs/operations/rer-table-check-20260905.md").read_text()
    command = doc.split("```powershell\n")[1].split("\n```")[0]
    assert "\n" not in command
    command = command.replace(
        r"C:\Users\developer\Documents\billiardsAI-raytheon3\.venv-rfdetr\Scripts\python.exe", str(fake)
    )
    prefix = (
        f"$env:USERPROFILE='{tmp_path}'; $env:PYTHONPATH='original value'; "
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
            + "; if($env:PYTHONPATH -ne 'original value'){throw 'Environment not restored'}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{")))
    assert data["args"][0].endswith("table_check.py")
    assert data["args"][1] == "--output"
    assert "rer-table-result-" in data["args"][2]
    assert data["path"].endswith("/src") or data["path"].endswith("\\src")


@pytest.mark.parametrize("outcome", ["save", "cancel", "wrong_identity", "tamper"])
def test_session_packages_evidence_and_never_activates(
    module, tmp_path, monkeypatch, outcome, diagnostic_packet_assets
):
    import types

    from billiards_ai.lab import mainline1_usb_capture

    target = build_packet(tmp_path, diagnostic_packet_assets=diagnostic_packet_assets)
    root = tmp_path / "packet"
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root)
    lens, payload = scene()
    lens["device_fingerprint"] = "test-fingerprint"
    (root / "lens-candidate.json").write_text(json.dumps(lens))
    manifest = json.loads((root / "packet-manifest.json").read_text())
    manifest["files"]["lens-candidate.json"]["sha256"] = hashlib.sha256(
        (root / "lens-candidate.json").read_bytes()
    ).hexdigest()
    (root / "packet-manifest.json").write_text(json.dumps(manifest))
    if outcome == "tamper":
        (root / "scripts/diagnostics/raytheon_table/page.html").write_text("tampered")
    monkeypatch.setattr(
        module, "__file__", str(root / "scripts/diagnostics/raytheon_table/table_check.py")
    )
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["table_check.py", "--output", str(output)])
    monkeypatch.setattr("builtins.input", lambda *_: "")
    device = types.SimpleNamespace(
        durable_fingerprint="wrong" if outcome == "wrong_identity" else "test-fingerprint",
        model_dump=lambda **_: {"device_locator": "test"},
    )
    monkeypatch.setattr(mainline1_usb_capture, "enumerate_usb_devices", lambda **_: [device])
    spec = importlib.util.spec_from_file_location(
        "camera_check", ROOT / "scripts/diagnostics/raytheon_camera/camera_check.py"
    )
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    worker_calls = []

    def worker(command, *_, **kwargs):
        worker_calls.append(command)
        assert kwargs["timeout"] == 45
        assert "--worker" in command and "--require-mode" in command
        assert "--project-root" not in command
        assert (
            json.loads(command[command.index("--expected-controls") + 1])
            == module.EXPECTED_CONTROLS
        )
        dest = Path(command[command.index("--output") + 1])
        dest.mkdir()
        report = {
            "status": "READBACK_COLLECTED_REVIEW_REQUIRED",
            "mode_matches_request": True,
            "decoded_geometry": [1920, 1080],
            "sample_completed": True,
            "cadence_without_recording": {"within_5_percent": True},
            "controls_after_mode": module.EXPECTED_CONTROLS,
            "controls_after_capture": module.EXPECTED_CONTROLS,
        }
        (dest / "report.json").write_text(json.dumps(report))
        cv2.imwrite(str(dest / "baseline.png"), np.zeros((1080, 1920, 3), np.uint8))
        return {"status": "COMPLETED"}

    monkeypatch.setattr(camera, "run_bounded", worker)
    monkeypatch.setitem(sys.modules, "camera_check", camera)
    threads = []
    errors = []

    def browser(url):
        def submit():
            try:
                request = urllib.request.Request(
                    url + outcome,
                    data=json.dumps(payload if outcome == "save" else {}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    assert response.status == 200
            except Exception as error:
                errors.append(error)

        t = threading.Thread(target=submit, daemon=True)
        t.start()
        threads.append(t)
        return True

    monkeypatch.setattr(module.webbrowser, "open", browser)
    code = module.main()
    for t in threads:
        t.join(timeout=12)
    assert not errors
    assert code == (0 if outcome == "save" else 2)
    assert len(worker_calls) == (0 if outcome in ("wrong_identity", "tamper") else 1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["production_ready"] is False
    assert summary["active_calibration_written"] is False
    assert summary["acceptance"] == "NOT_EVALUATED"
    assert not list(tmp_path.rglob("registry.json"))
    with zipfile.ZipFile(tmp_path / "SEND_BACK-result.zip") as archive:
        assert archive.testzip() is None
        for name, info in summary["inventory"].items():
            assert hashlib.sha256(archive.read("result/" + name)).hexdigest() == info["sha256"]
