from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from billiards_ai.lab import mainline1_shot_local_video as shot_local_video_module
from billiards_ai.lab.mainline1_live_publisher import publish_live_shot
from billiards_ai.lab.mainline1_shot_local_video import (
    Mainline1ShotLocalConfig,
    build_shot_evidence,
    run_cached_mainline1_shot_local_video,
    run_mainline1_shot_local_video,
    verify_mainline1_shot_local_artifact,
)
from billiards_ai.lab.shot_local.cue import CueTrackingConfig
from billiards_ai.lab.shot_local.events import ShotLocalEventConfig
from billiards_ai.lab.shot_local.fragments import FragmentTrackingConfig
from billiards_ai.lab.shot_local.models import (
    BallGroup,
    CalibratedBallObservation,
    ObservationFrame,
)
from billiards_ai.lab.shot_local.window import ShotWindowConfig
from billiards_ai.lab.vision.contracts import (
    BallInstanceObservation,
    PixelBox,
    PixelPoint,
    VisionFrameObservation,
    VisionPackageIdentity,
)
from billiards_ai.shared.calibration.homography import calibrate


def _synthetic_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (160, 120),
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    return path


def _detection(detection_id: str, x: int, y: int = 60) -> BallInstanceObservation:
    return BallInstanceObservation(
        detection_id=detection_id,
        bbox=PixelBox(left=x - 4, top=y - 4, right=x + 4, bottom=y + 4),
        center=PixelPoint(x=x, y=y),
        confidence=0.9,
        class_label="ball",
    )


class RecordingBackend:
    package_identity = VisionPackageIdentity(
        package_id="a" * 64,
        manifest_sha256="b" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer(self, frame) -> VisionFrameObservation:
        self.calls.append(frame.frame_id)
        cue_x = (50, 50, 60, 70, 70)[frame.frame_id]
        target_x = (100, 100, 100, 110, 110)[frame.frame_id]
        return VisionFrameObservation(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            width_px=frame.width_px,
            height_px=frame.height_px,
            pixel_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            package=self.package_identity,
            threshold=0.25,
            detections=(
                _detection(f"cue-{frame.frame_id}", cue_x),
                _detection(f"target-{frame.frame_id}", target_x),
            ),
            runtime_ref="synthetic/1",
        )


class RecordingClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def classify(
        self,
        *,
        image: np.ndarray,
        detections: tuple[BallInstanceObservation, ...],
    ) -> Mapping[str, Mapping[str, Any]]:
        self.calls += 1
        return {
            item.detection_id: {
                "group": "cue" if item.detection_id.startswith("cue-") else "solid",
                "confidence": 0.9,
            }
            for item in detections
        }


def _config() -> Mainline1ShotLocalConfig:
    return Mainline1ShotLocalConfig(
        window=ShotWindowConfig(
            motion_threshold_mm=8,
            motion_confirm_frames=2,
            still_duration_ms=100,
        ),
        cue=CueTrackingConfig(
            ambiguity_margin=0.08,
            reacquire_confidence_min=0.55,
        ),
        fragment=FragmentTrackingConfig(
            motion_threshold_mm=5,
            continuation_distance_mm=40,
            ambiguity_mm=3,
        ),
        event=ShotLocalEventConfig(
            first_contact_distance_mm=35,
            first_contact_ambiguity_mm=3,
            cue_motion_min_mm=5,
            contact_time_tolerance_ms=120,
            cushion_distance_mm=10,
            pocket_distance_mm=15,
            pot_missing_frames=3,
            after_match_distance_mm=20,
            table_length_mm=160,
            table_width_mm=120,
        ),
    )


def _calibration():
    return calibrate(
        "synthetic-camera",
        ((0, 120), (160, 120), (160, 0), (0, 0)),
        table_length_mm=160,
        table_width_mm=120,
    )


def _evidence_observation(
    frame_id: int,
    detection_id: str,
    x_mm: int,
    group: BallGroup,
) -> CalibratedBallObservation:
    return CalibratedBallObservation(
        detection_id=f"{detection_id}-{frame_id}",
        x_mm=x_mm,
        y_mm=60,
        confidence=0.95,
        group=group,
        group_confidence=0.95,
        bbox_xyxy_px=(x_mm - 4, 56, x_mm + 4, 64),
    )


def _evidence_frames() -> tuple[ObservationFrame, ...]:
    cue_positions = (50, 50, 60, 70, 70)
    timestamps = (0, 40, 80, 120, 240)
    return tuple(
        ObservationFrame(
            frame_id=frame_id,
            timestamp_ms=timestamps[frame_id],
            observations=(
                _evidence_observation(frame_id, "cue", cue_x, BallGroup.CUE),
                _evidence_observation(frame_id, "target", 100, BallGroup.SOLID),
            ),
        )
        for frame_id, cue_x in enumerate(cue_positions)
    )


def test_shot_evidence_excludes_legacy_collision_and_cushion_events() -> None:
    results = build_shot_evidence(observations=_evidence_frames(), config=_config())

    event_types = {event.event_type for event in results[0].events}
    assert {"shot_start", "first_contact", "shot_end"} <= event_types
    assert "ball_ball_collision" not in event_types
    assert "cushion" not in event_types


def test_table_dimensions_must_match_calibration_before_inference(tmp_path: Path) -> None:
    backend = RecordingBackend()
    mismatched = replace(
        _config(),
        event=replace(_config().event, table_length_mm=161),
    )

    with pytest.raises(ValueError, match="table dimensions do not match calibration"):
        run_mainline1_shot_local_video(
            video=_synthetic_video(tmp_path / "shot.avi"),
            calibration=_calibration(),
            vision_backend=backend,
            identity_classifier=RecordingClassifier(),
            config=mismatched,
            output_root=tmp_path / "artifacts",
        )

    assert backend.calls == []


def test_shot_local_video_publishes_one_consistent_artifact(tmp_path: Path) -> None:
    backend = RecordingBackend()
    classifier = RecordingClassifier()
    artifact = run_mainline1_shot_local_video(
        video=_synthetic_video(tmp_path / "shot.avi"),
        calibration=_calibration(),
        vision_backend=backend,
        identity_classifier=classifier,
        config=_config(),
        output_root=tmp_path / "artifacts",
    )
    expected = {
        "detections.json",
        "shot-windows.json",
        "anchors.json",
        "cue-track.json",
        "motion-fragments.json",
        "active-tracks.json",
        "events.json",
        "event-timeline.json",
        "world-events.json",
        "table-state-before.json",
        "table-state-after.json",
        "summary.json",
        "performance.json",
        "overlay.mp4",
        "manifest.json",
    }

    assert {path.name for path in artifact.path.iterdir()} == expected
    assert json.loads((artifact.path / "summary.json").read_text())["shot_count"] == 1
    assert artifact.frame_count == 5
    assert backend.calls == [0, 1, 2, 3, 4]
    assert classifier.calls == 5
    manifest = verify_mainline1_shot_local_artifact(artifact.path)
    assert manifest["pipeline_version"] == "shot-local-unified-product-v1"
    timeline = json.loads((artifact.path / "event-timeline.json").read_text())
    assert {event["event_type"] for event in timeline["shots"][0]["events"]} >= {
        "shot_start",
        "first_contact",
        "shot_end",
    }
    world = json.loads((artifact.path / "world-events.json").read_text())
    assert world == json.loads((artifact.path / "events.json").read_text())
    assert all(event["event_type"] != "ball_ball_collision" for event in world["events"])
    assert all("fragment" not in str(event.get("actor_id")) for event in world["events"])
    summary = json.loads((artifact.path / "summary.json").read_text())
    assert summary["track_count"] == 2
    assert summary["incomplete_shot_count"] == 0
    performance = json.loads((artifact.path / "performance.json").read_text())
    assert performance["mode"] == "full_file_replay"
    assert performance["detection_and_cache_ms"] >= 0
    assert performance["shot_evidence_ms"] >= 0
    assert performance["stable_track_ms"] >= 0
    assert performance["event_observation_ms"] >= 0


def test_file_replay_publishes_the_shared_product_payloads_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    original = shot_local_video_module._publish_shot_local_artifact

    def recording_publisher(**kwargs):
        captured["product"] = kwargs["product"]
        return original(**kwargs)

    monkeypatch.setattr(
        shot_local_video_module,
        "_publish_shot_local_artifact",
        recording_publisher,
    )
    artifact = run_mainline1_shot_local_video(
        video=_synthetic_video(tmp_path / "shot.avi"),
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "artifacts",
    )

    product = captured["product"]
    for name, payload in product.payloads.items():
        assert (artifact.path / name).read_bytes() == shot_local_video_module._canonical_bytes(
            payload
        )


def test_incomplete_live_product_cannot_activate_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    original = shot_local_video_module._publish_shot_local_artifact

    def recording_publisher(**kwargs):
        captured["product"] = kwargs["product"]
        return original(**kwargs)

    monkeypatch.setattr(
        shot_local_video_module,
        "_publish_shot_local_artifact",
        recording_publisher,
    )
    run_mainline1_shot_local_video(
        video=_synthetic_video(tmp_path / "shot.avi"),
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "file-artifacts",
    )
    product = captured["product"]
    incomplete_window = replace(
        product.results[0].window,
        after_frame_id=None,
        complete=False,
    )
    incomplete_result = replace(product.results[0], window=incomplete_window)
    incomplete_product = replace(product, results=(incomplete_result,))
    live_root = tmp_path / "live-artifacts"

    with pytest.raises(ValueError, match="incomplete live shot"):
        publish_live_shot(
            product=incomplete_product,
            identity={"source_kind": "usb_live"},
            output_root=live_root,
        )

    assert not live_root.exists()


@pytest.mark.parametrize(
    "tampered_name",
    ["events.json", "world-events.json", "active-tracks.json", "overlay.mp4"],
)
def test_manifest_inventory_detects_tampering(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    artifact = run_mainline1_shot_local_video(
        video=_synthetic_video(tmp_path / "shot.avi"),
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "artifacts",
    )
    (artifact.path / tampered_name).write_text("tampered")

    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_mainline1_shot_local_artifact(artifact.path)


def test_verifier_rejects_content_addressed_manifest_with_empty_inventory(
    tmp_path: Path,
) -> None:
    identity = {
        "artifact_id_algorithm": "identity-semantic-inventory-v2",
        "inventory": {},
        "kind": "mainline1_shot_local_video",
    }
    encoded = (
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    artifact_id = hashlib.sha256(encoded).hexdigest()
    artifact = tmp_path / artifact_id
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(identity | {"artifact_id": artifact_id}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="required product files"):
        verify_mainline1_shot_local_artifact(artifact)


def test_same_input_and_build_reuse_same_artifact_despite_runtime_timing(
    tmp_path: Path,
) -> None:
    video = _synthetic_video(tmp_path / "shot.avi")
    calibration = _calibration()
    first = run_mainline1_shot_local_video(
        video=video,
        calibration=calibration,
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "artifacts",
    )
    second = run_mainline1_shot_local_video(
        video=video,
        calibration=calibration,
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "artifacts",
    )

    assert second.artifact_id == first.artifact_id
    assert second.path == first.path


def test_verifier_accepts_legacy_v1_identity_without_inventory_hash(tmp_path: Path) -> None:
    artifact = run_mainline1_shot_local_video(
        video=_synthetic_video(tmp_path / "shot.avi"),
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "artifacts",
    )
    manifest = json.loads((artifact.path / "manifest.json").read_text())
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_id", "artifact_id_algorithm", "inventory"}
    }
    legacy_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    ).hexdigest()
    manifest.pop("artifact_id_algorithm", None)
    manifest["artifact_id"] = legacy_id
    (artifact.path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    legacy_path = artifact.path.with_name(legacy_id)
    artifact.path.rename(legacy_path)

    verified = verify_mainline1_shot_local_artifact(legacy_path)

    assert verified["artifact_id"] == legacy_id


def test_cached_detection_reobserver_reuses_verified_observations(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "shot.avi")
    source = run_mainline1_shot_local_video(
        video=video,
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "source",
    )

    observed = run_cached_mainline1_shot_local_video(
        video=video,
        source_artifact=source.path,
        config=_config(),
        output_root=tmp_path / "observed",
    )

    manifest = verify_mainline1_shot_local_artifact(observed.path)
    performance = json.loads((observed.path / "performance.json").read_text())
    assert manifest["observation_mode"] == "verified_cached_detections"
    assert manifest["source_artifact_id"] == source.artifact_id
    assert performance["mode"] == "cached_detection_file_replay"
    assert performance["detection_and_cache_ms"] == 0
    assert performance["cached_observation_load_ms"] >= 0


def test_cached_detection_artifact_identity_includes_build_code(tmp_path: Path) -> None:
    video = _synthetic_video(tmp_path / "shot.avi")
    source = run_mainline1_shot_local_video(
        video=video,
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "source",
    )

    first = run_cached_mainline1_shot_local_video(
        video=video,
        source_artifact=source.path,
        config=_config(),
        output_root=tmp_path / "observed",
        build_code_sha256="a" * 64,
    )
    second = run_cached_mainline1_shot_local_video(
        video=video,
        source_artifact=source.path,
        config=_config(),
        output_root=tmp_path / "observed",
        build_code_sha256="b" * 64,
    )

    assert first.artifact_id != second.artifact_id
    assert verify_mainline1_shot_local_artifact(first.path)["build_code_sha256"] == "a" * 64
    assert verify_mainline1_shot_local_artifact(second.path)["build_code_sha256"] == "b" * 64


def test_cached_detection_reobserver_rejects_symlinked_source_before_resolve(
    tmp_path: Path,
) -> None:
    video = _synthetic_video(tmp_path / "shot.avi")
    source = run_mainline1_shot_local_video(
        video=video,
        calibration=_calibration(),
        vision_backend=RecordingBackend(),
        identity_classifier=RecordingClassifier(),
        config=_config(),
        output_root=tmp_path / "source",
    )
    link = tmp_path / "source-link"
    link.symlink_to(source.path, target_is_directory=True)

    with pytest.raises(ValueError, match="missing or unsafe"):
        run_cached_mainline1_shot_local_video(
            video=video,
            source_artifact=link,
            config=_config(),
            output_root=tmp_path / "observed",
        )
