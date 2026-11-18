from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_supported_shot_local_entrypoints_exist() -> None:
    required = (
        "scripts/algorithm/run_mainline1_shot_local_video.py",
        "scripts/algorithm/windows/run-mainline1-shot-local-video.ps1",
        "scripts/algorithm/run_shot_local_regression_suite.py",
        "scripts/algorithm/windows/run-shot-local-regression-suite.ps1",
        "configs/algorithm/shot-local-regression-suite.template.json",
        "configs/algorithm/shot-local-regression-truth.template.json",
        "src/billiards_ai/lab/shot_local/active_tracks.py",
        "src/billiards_ai/lab/shot_local/product.py",
        "src/billiards_ai/lab/shot_local/product_overlay.py",
        "src/billiards_ai/lab/shot_local/regression.py",
        "src/billiards_ai/lab/shot_local/regression_suite.py",
        "src/billiards_ai/edge/realtime_chain/profile.py",
    )
    assert all((ROOT / path).is_file() for path in required)


def test_experimental_spike_entrypoint_is_not_product_surface() -> None:
    forbidden = (
        "scripts/algorithm/run_active_track_collision_spike.py",
        "scripts/algorithm/windows/run-active-track-collision-spike.ps1",
    )
    assert all(not (ROOT / path).exists() for path in forbidden)


def test_archived_tracking_research_is_not_in_product_tree() -> None:
    forbidden = (
        "src/billiards_ai/lab/sam2_audit",
        "src/billiards_ai/lab/sam2_handoff.py",
        "src/billiards_ai/lab/propagation/sam2_backend.py",
        "src/billiards_ai/lab/mainline1_tracking.py",
        "src/billiards_ai/lab/mainline1_tracking_video.py",
        "src/billiards_ai/lab/adapters/tracker.py",
        "scripts/algorithm/sam2_real_video_pilot.py",
        "scripts/algorithm/sam2_wsl_worker.py",
        "scripts/algorithm/run_mainline1_tracking_video.py",
        "scripts/algorithm/windows/run-mainline1-tracking-video.ps1",
        "scripts/algorithm/wsl",
    )
    assert all(not (ROOT / path).exists() for path in forbidden)


def test_four_mainline_product_roots_remain() -> None:
    required = (
        "src/billiards_ai/lab/mainline1_shot_local_video.py",
        "src/billiards_ai/lab/planning",
        "src/billiards_ai/lab/field_assistance",
        "src/billiards_ai/deployment/field_bundle.py",
        "apps/miniprogram/miniprogram",
        "apps/miniprogram/cloudfunctions",
    )
    assert all((ROOT / path).exists() for path in required)


def test_frozen_shot_local_product_milestone_contract() -> None:
    manifest_path = ROOT / "configs/milestones/shot-local-product-v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0"
    assert manifest["milestone_id"] == "shot-local-product-v1"
    assert manifest["status"] == "frozen_for_regression"
    assert manifest["runtime"]["requires_sam21"] is False
    assert manifest["runtime"]["requires_global_fixed_id"] is False
    assert manifest["validated"]["shot_local_id_stability"] is True
    assert manifest["validated"]["ball_contact_events"] is True
    assert manifest["validated"]["cushion_events"] is True
    assert manifest["pending"]["multi_video_regression"] is True
    assert manifest["pending"]["live_camera_validation"] is True


def test_mainline1_owns_the_unified_product_pipeline() -> None:
    entry_source = (ROOT / "src/billiards_ai/lab/mainline1_shot_local_video.py").read_text(
        encoding="utf-8"
    )
    product_source = (ROOT / "src/billiards_ai/lab/mainline1_shot_product.py").read_text(
        encoding="utf-8"
    )
    source = entry_source + product_source

    assert '"shot-local-unified-product-v1"' in entry_source
    assert "build_mainline1_shot_product" in entry_source
    assert "active-tracks.json" in product_source
    assert "world-events.json" in product_source
    assert "observe_ball_ball_collisions" not in source
    assert "observe_shot_cushions" not in source
    assert "def _render_overlay(" not in source
    assert "def _event_timeline_payload(" not in source
    assert "ACTIVATE " not in source


def test_raytheon_runbook_uses_one_mainline_product_stage() -> None:
    text = (ROOT / "docs/operations/lightweight-realtime-chain-raytheon.md").read_text(
        encoding="utf-8"
    )

    assert "run-mainline1-shot-local-video.ps1" in text
    assert "run-shot-local-event-tracker.ps1" not in text
    assert "mainline1-shot-local-video-runs" in text
    assert "shot-local-event-tracker" not in text
