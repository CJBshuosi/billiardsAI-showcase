from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from billiards_ai.lab.field_assistance.replay import consume_live_mainline1_artifact
from billiards_ai.lab.mainline1_live_runtime import LiveRuntimeEventKind
from tests.unit.lab.test_mainline1_live_runtime import _runtime


def _published_live_artifact(tmp_path: Path) -> Path:
    events = _runtime(tmp_path, 5).run()
    published = next(event for event in events if event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED)
    assert published.artifact_path is not None
    return published.artifact_path


def test_complete_activated_live_shot_opens_only_safe_downstream_surfaces(
    tmp_path: Path,
) -> None:
    artifact = _published_live_artifact(tmp_path)

    projection = consume_live_mainline1_artifact(artifact)

    assert projection.artifact_id == artifact.name
    assert projection.source_kind == "usb_live"
    assert projection.shot_id == "shot-0001"
    assert projection.route is not None
    assert projection.route["available"] is True
    assert [fact["event_type"] for fact in projection.facts] == ["shot_start", "shot_end"]
    assert projection.clip_path == artifact / "overlay.mp4"
    assert projection.turn_effects == ()
    assert projection.growth_writes == ()
    assert projection.display.phase == "READY_FOR_ROUTE"
    assert projection.display.guidance_visible is False
    assert projection.display.reason is None


def test_interrupted_live_shot_changes_no_rule_memory_or_route_authority(
    tmp_path: Path,
) -> None:
    events = _runtime(tmp_path, 4).run()
    assert any(event.kind is LiveRuntimeEventKind.SHOT_ABSTAINED for event in events)
    assert not any(event.kind is LiveRuntimeEventKind.SHOT_PUBLISHED for event in events)

    projection = consume_live_mainline1_artifact(None, health_reason="source_stopped")

    assert projection.route is None
    assert projection.facts == ()
    assert projection.clip_path is None
    assert projection.turn_effects == ()
    assert projection.growth_writes == ()
    assert projection.display.phase == "DEGRADED"
    assert projection.display.guidance_visible is False
    assert projection.display.reason == "source_stopped"


def test_tampered_live_product_fails_artifact_verification(tmp_path: Path) -> None:
    artifact = _published_live_artifact(tmp_path / "source")
    tampered = tmp_path / "tampered"
    shutil.copytree(artifact, tampered)
    windows_path = tampered / "shot-windows.json"
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    windows["shots"][0]["complete"] = False
    windows_path.write_text(json.dumps(windows), encoding="utf-8")
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inventory"]["shot-windows.json"] = {
        "bytes": windows_path.stat().st_size,
        "sha256": hashlib.sha256(windows_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    projection = consume_live_mainline1_artifact(tampered)

    assert projection.route is None
    assert projection.facts == ()
    assert projection.clip_path is None
    assert projection.turn_effects == ()
    assert projection.growth_writes == ()
    assert projection.display.phase == "DEGRADED"
    assert projection.display.guidance_visible is False
    assert projection.display.reason == "LIVE_ARTIFACT_INVALID"


def test_capture_degradation_hides_route_but_preserves_activated_historical_evidence(
    tmp_path: Path,
) -> None:
    artifact = _published_live_artifact(tmp_path)

    projection = consume_live_mainline1_artifact(
        artifact,
        health_reason="CAMERA_DISCONNECTED",
    )

    assert projection.route is None
    assert projection.facts
    assert projection.clip_path == artifact / "overlay.mp4"
    assert projection.turn_effects == ()
    assert projection.growth_writes == ()
    assert projection.display.phase == "DEGRADED"
    assert projection.display.guidance_visible is False
    assert projection.display.reason == "CAMERA_DISCONNECTED"


def test_manifest_lineage_tampering_fails_closed(tmp_path: Path) -> None:
    artifact = _published_live_artifact(tmp_path / "source")
    tampered = tmp_path / "tampered-lineage"
    shutil.copytree(artifact, tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture_setup_id"] = "forged-setup"
    manifest["artifact_id"] = "forged-artifact"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    projection = consume_live_mainline1_artifact(tampered)

    assert projection.route is None
    assert projection.facts == ()
    assert projection.display.phase == "DEGRADED"
    assert projection.display.reason == "LIVE_ARTIFACT_INVALID"
