from __future__ import annotations

import json
from pathlib import Path

import pytest

from billiards_ai.lab.shot_local.regression_suite import (
    RegressionCalibrationBlocked,
    RegressionProductRun,
    run_regression_suite,
)


def _product_artifact(root: Path, *, wall_seconds: float = 1.25) -> Path:
    root.mkdir(parents=True)
    (root / "world-events.json").write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot-0001",
                        "events": [
                            {
                                "event_type": "cushion",
                                "status": "observed",
                                "contact_frame_id": 70,
                                "actor_id": "shot-0001-anchor-10",
                                "rail": "left",
                            }
                        ],
                    }
                ]
            }
        )
    )
    (root / "active-tracks.json").write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot-0001",
                        "tracks": [
                            {
                                "track_id": "shot-0001-anchor-10",
                                "samples": [{"frame_id": 70, "x_mm": 20.0, "y_mm": 40.0}],
                            }
                        ],
                    }
                ]
            }
        )
    )
    (root / "summary.json").write_text(json.dumps({"shot_count": 1, "wall_seconds": wall_seconds}))
    (root / "performance.json").write_text(
        json.dumps({"schema_version": "1.0", "total_artifact_ms": wall_seconds * 1000})
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": root.name,
                "kind": "mainline1_shot_local_video",
                "pipeline_version": "shot-local-unified-product-v1",
            }
        )
    )
    return root


def _truth(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_id": "shot-0001",
                        "events": [
                            {
                                "event_type": "cushion",
                                "contact_frame_id": 70,
                                "actor_id": "shot-0001-anchor-10",
                                "rail": "left",
                            }
                        ],
                        "identity_checkpoints": [
                            {
                                "track_id": "shot-0001-anchor-10",
                                "frame_id": 70,
                                "x_mm": 20.0,
                                "y_mm": 40.0,
                                "tolerance_mm": 2.0,
                            }
                        ],
                    }
                ]
            }
        )
    )


def _capture_setup(path: Path, *, setup_id: str = "table-a-camera-a") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "capture_setup_id": setup_id,
                "table_id": "table-a",
                "camera_device_id": "camera-a",
                "lens_id": "lens-a",
                "mount_revision": "mount-v1",
                "distortion_revision": "distortion-v1",
                "source_kind": "file",
                "geometry": {
                    "width": 1920,
                    "height": 1080,
                    "crop": [0, 0, 1920, 1080],
                    "rotation_degrees": 0,
                },
            }
        )
    )


def _manifest(path: Path, cases: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "suite_id": "venue-regression-v1",
                "frame_tolerance": 2,
                "gates": {
                    "minimum_overall_precision": 0.9,
                    "minimum_overall_recall": 0.9,
                    "minimum_identity_checkpoint_pass_rate": 1.0,
                    "maximum_case_wall_seconds": 5.0,
                    "minimum_recall_by_event_type": {"cushion": 0.9},
                },
                "cases": cases,
            }
        )
    )


def test_suite_resolves_paths_runs_product_and_publishes_pass(tmp_path: Path) -> None:
    video = tmp_path / "video-a.mp4"
    source = tmp_path / "source-a"
    config = tmp_path / "shot-local.yaml"
    truth = tmp_path / "truth-a.json"
    setup = tmp_path / "setup-a.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "video-a",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )
    calls = []

    def product_runner(**kwargs):
        calls.append(kwargs)
        return RegressionProductRun(
            path=_product_artifact(tmp_path / "product-a"),
            calibration_gate={
                "status": "READY",
                "capture_setup_id": "table-a-camera-a",
                "artifact_id": "calibration-a",
                "revision": 1,
                "validation_mode": "corner_drift",
            },
        )

    calibration_root = tmp_path / "calibrations"
    calibration_root.mkdir()
    artifact = run_regression_suite(
        manifest, tmp_path / "outputs", calibration_root, product_runner
    )
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "PASS"
    assert report["status"] == "PASS"
    assert report["aggregate"]["overall"]["recall"] == 1.0
    assert report["cases"][0]["product_artifact_id"] == "product-a"
    assert report["cases"][0]["calibration_gate"]["status"] == "READY"
    assert calls == [
        {
            "video": video.resolve(),
            "source_artifact": source.resolve(),
            "capture_setup": setup.resolve(),
            "calibration_root": calibration_root.resolve(),
            "config_path": config.resolve(),
            "output_root": (tmp_path / "outputs" / "product-artifacts").resolve(),
        }
    ]


def test_suite_rejects_duplicate_case_ids_before_running(tmp_path: Path) -> None:
    manifest = tmp_path / "suite.json"
    case = {
        "case_id": "duplicate",
        "video": "video.mp4",
        "source_artifact": "source",
        "shot_local_config": "config.yaml",
        "truth": "truth.json",
        "capture_setup": "setup.json",
    }
    _manifest(manifest, [case, case])

    with pytest.raises(ValueError, match="case IDs must be unique"):
        run_regression_suite(manifest, tmp_path / "outputs", tmp_path, lambda **_: None)


def test_suite_records_one_execution_error_and_continues_remaining_cases(tmp_path: Path) -> None:
    cases = []
    for name in ("broken", "good"):
        (tmp_path / f"{name}.mp4").write_bytes(b"video")
        (tmp_path / f"source-{name}").mkdir()
        (tmp_path / f"{name}.yaml").write_text("config")
        _truth(tmp_path / f"truth-{name}.json")
        _capture_setup(tmp_path / f"setup-{name}.json", setup_id=f"setup-{name}")
        cases.append(
            {
                "case_id": name,
                "video": f"{name}.mp4",
                "source_artifact": f"source-{name}",
                "shot_local_config": f"{name}.yaml",
                "truth": f"truth-{name}.json",
                "capture_setup": f"setup-{name}.json",
            }
        )
    manifest = tmp_path / "suite.json"
    _manifest(manifest, cases)
    call_count = 0

    def product_runner(**_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("detector unavailable")
        return RegressionProductRun(
            path=_product_artifact(tmp_path / "product-good"),
            calibration_gate={"status": "READY", "capture_setup_id": "setup-good"},
        )

    artifact = run_regression_suite(manifest, tmp_path / "outputs", tmp_path, product_runner)
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "ERROR"
    assert [row["status"] for row in report["cases"]] == ["ERROR", "PASS"]
    assert report["cases"][0]["error"] == "detector unavailable"


def test_suite_refuses_symlinked_case_input(tmp_path: Path) -> None:
    target = tmp_path / "real.mp4"
    target.write_bytes(b"video")
    link = tmp_path / "video.mp4"
    link.symlink_to(target)
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("config")
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "unsafe",
                "video": link.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )

    with pytest.raises(ValueError, match="unsafe"):
        run_regression_suite(manifest, tmp_path / "outputs", tmp_path, lambda **_: None)


def test_suite_rejects_legacy_tracker_artifact(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "legacy",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )

    def legacy_runner(**_):
        artifact = _product_artifact(tmp_path / "legacy-artifact")
        (artifact / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": artifact.name,
                    "kind": "shot_local_event_tracker",
                    "pipeline_version": "shot-local-event-tracker-v1",
                }
            )
        )
        return RegressionProductRun(
            path=artifact,
            calibration_gate={"status": "READY", "capture_setup_id": "table-a-camera-a"},
        )

    artifact = run_regression_suite(manifest, tmp_path / "outputs", tmp_path, legacy_runner)
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "ERROR"
    assert report["cases"][0]["status"] == "ERROR"
    assert "canonical Mainline 1 product artifact" in report["cases"][0]["error"]


def test_calibration_blocked_case_is_not_scored_as_algorithm_failure(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "blocked",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )

    def blocked_runner(**_):
        raise RegressionCalibrationBlocked(
            status="CALIBRATION_REQUIRED",
            calibration_gate={
                "status": "CALIBRATION_REQUIRED",
                "capture_setup_id": "table-a-camera-a",
                "reasons": ["calibration_not_registered"],
            },
        )

    artifact = run_regression_suite(manifest, tmp_path / "outputs", tmp_path, blocked_runner)
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "CALIBRATION_REQUIRED"
    assert report["cases"][0]["status"] == "CALIBRATION_REQUIRED"
    assert report["cases"][0].get("evaluation") is None
    assert report["aggregate"]["overall"]["tp"] == 0
    assert report["aggregate"]["overall"]["fp"] == 0
    assert report["aggregate"]["overall"]["fn"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_suite_rejects_non_finite_numeric_gate(tmp_path: Path, value: float) -> None:
    video = tmp_path / "video.mp4"
    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "finite-gate",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )
    payload = json.loads(manifest.read_text())
    payload["gates"]["maximum_case_wall_seconds"] = value
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="maximum case wall seconds"):
        run_regression_suite(manifest, tmp_path / "outputs", tmp_path, lambda **_: None)


@pytest.mark.parametrize("wall_seconds", [None, float("nan"), float("inf")])
def test_suite_rejects_missing_or_non_finite_case_wall_time(
    tmp_path: Path, wall_seconds: float | None
) -> None:
    video = tmp_path / "video.mp4"
    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "wall-time",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )

    def product_runner(**_):
        artifact = _product_artifact(tmp_path / "product")
        performance = json.loads((artifact / "performance.json").read_text())
        if wall_seconds is None:
            performance.pop("total_artifact_ms")
        else:
            performance["total_artifact_ms"] = wall_seconds * 1000
        (artifact / "performance.json").write_text(json.dumps(performance))
        return RegressionProductRun(
            path=artifact,
            calibration_gate={"status": "READY", "capture_setup_id": "table-a-camera-a"},
        )

    artifact = run_regression_suite(manifest, tmp_path / "outputs", tmp_path, product_runner)
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "ERROR"
    assert "wall seconds" in report["cases"][0]["error"]


def test_suite_applies_wall_gate_to_product_performance(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    source = tmp_path / "source"
    config = tmp_path / "config.yaml"
    truth = tmp_path / "truth.json"
    setup = tmp_path / "setup.json"
    video.write_bytes(b"video")
    source.mkdir()
    config.write_text("config")
    _truth(truth)
    _capture_setup(setup)
    manifest = tmp_path / "suite.json"
    _manifest(
        manifest,
        [
            {
                "case_id": "slow",
                "video": video.name,
                "source_artifact": source.name,
                "shot_local_config": config.name,
                "truth": truth.name,
                "capture_setup": setup.name,
            }
        ],
    )

    def product_runner(**_):
        product = _product_artifact(tmp_path / "slow-product", wall_seconds=6.0)
        (product / "summary.json").write_text(json.dumps({"shot_count": 1, "wall_seconds": 1.0}))
        return RegressionProductRun(
            path=product,
            calibration_gate={"status": "READY", "capture_setup_id": "table-a-camera-a"},
        )

    artifact = run_regression_suite(manifest, tmp_path / "outputs", tmp_path, product_runner)
    report = json.loads((artifact.path / "report.json").read_text())

    assert artifact.status == "FAIL"
    assert report["cases"][0]["wall_seconds"] == 6.0
    assert report["gate_failures"] == ["slow wall seconds above 5.0"]
