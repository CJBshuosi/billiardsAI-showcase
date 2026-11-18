from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.diagnostics.package_raytheon_product_realtime import build_packet


def test_product_realtime_packet_is_scoped_and_contains_native_bridge(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "calibrations").mkdir(parents=True)
    (candidate / "capture-setup.json").write_text("{}\n", encoding="utf-8")
    (candidate / "calibrations" / "registry.json").write_text("{}\n", encoding="utf-8")
    archive = build_packet(Path.cwd(), candidate, tmp_path / "out")

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest_name = next(name for name in names if name.endswith("packet-manifest.json"))
        root = manifest_name.removesuffix("packet-manifest.json")
        manifest = json.loads(bundle.read(manifest_name))

        assert manifest["kind"] == "mainline1_product_realtime_single_shot_diagnostic"
        assert manifest["production_ready"] is False
        assert manifest["writes_installed_service_or_state_db"] is False
        assert manifest["run"]["stop_after_terminal_shots"] == 1
        assert manifest["run"]["diagnostic_duration_seconds"] == 60
        assert manifest["run"]["replay_grayscale_cache"] == (
            "ordered_parallel_ingress_once_960x540"
        )
        assert manifest["run"]["working_preprocessor_workers"] == 3
        assert manifest["run"]["maximum_preprocessing_lag_frames"] == 2
        assert manifest["run"]["anchor_pending_policy"] == ("replace_with_latest_scheduled_frame")
        assert manifest["run"]["identity_refresh_interval_ms"] == 1000
        assert manifest["run"]["unclassified_anchor_identity_policy"] == (
            "inherit_only_from_spatially_matched_trusted_track_else_unknown"
        )
        assert root + "run.ps1" in names
        assert root + "scripts/algorithm/run_mainline1_product_realtime.py" in names
        assert root + "native/windows/mainline1_uvc_capture_csharp/main.cs" in names
        assert root + "src/billiards_ai/lab/mainline1_native_capture.py" in names
        assert root + "candidate/capture-setup.json" in names
        native = bundle.read(root + "native/windows/mainline1_uvc_capture_csharp/main.cs").decode(
            "utf-8"
        )
        assert "frames.Count >= 32" in native
        launcher = bundle.read(root + "run.ps1").decode("utf-8-sig")
        assert '$env:PYTHONWARNINGS = "ignore::FutureWarning"' in launcher
        assert "ForEach-Object { $_.ToString() }" in launcher
        assert "realtime_diagnostics = $(if ($ProductSummary.Count -eq 1)" in launcher
        readme = bundle.read(root + "README.txt").decode("utf-8")
        assert "60 秒" in readme
        assert "Ctrl+C" not in readme
        assert archive.name == "billiardsAI-raytheon-product-realtime-20260906-v13.zip"
