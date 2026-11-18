"""Publish one-pass, shot-local tracking evidence and a review overlay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from billiards_ai.lab.field_assistance.observations import load_cached_observations
from billiards_ai.lab.mainline1_frame_source import FileFrameSource, decode_frame_source
from billiards_ai.lab.mainline1_shot_product import (
    Mainline1ShotProductPayloads,
    build_mainline1_shot_product,
)
from billiards_ai.lab.shot_local.active_tracks import ActiveTrackConfig
from billiards_ai.lab.shot_local.cue import CueTrackingConfig, track_cue
from billiards_ai.lab.shot_local.events import (
    ShotLocalEventConfig,
    observe_first_anchor_contact,
    observe_fragment_pots,
    observe_shot_boundaries,
)
from billiards_ai.lab.shot_local.fragments import (
    FragmentTrackingConfig,
    track_motion_fragments,
)
from billiards_ai.lab.shot_local.models import (
    ObservationFrame,
    ShotLocalResult,
)
from billiards_ai.lab.shot_local.observations import (
    DecodedTrackingFrame,
    GroupIdentityClassifier,
    collect_observation_frames,
)
from billiards_ai.lab.shot_local.product import (
    CanonicalShotAnalysis,
    build_canonical_shot_analyses,
    canonical_world_event_counts,
    canonical_world_event_rows,
    shot_local_product_active_config,
)
from billiards_ai.lab.shot_local.product_overlay import render_canonical_overlay
from billiards_ai.lab.shot_local.window import (
    ShotWindowConfig,
    build_shot_anchors,
    segment_shot_windows,
)
from billiards_ai.lab.vision.backend import VisionBackend
from billiards_ai.shared.calibration.homography import TableCalibration

_SHOT_LOCAL_PIPELINE_VERSION = "shot-local-unified-product-v1"


@dataclass(frozen=True, slots=True)
class Mainline1ShotLocalConfig:
    window: ShotWindowConfig
    cue: CueTrackingConfig
    fragment: FragmentTrackingConfig
    event: ShotLocalEventConfig


@dataclass(frozen=True, slots=True)
class Mainline1ShotLocalArtifact:
    path: Path
    artifact_id: str
    frame_count: int
    shot_count: int


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_REQUIRED_ARTIFACT_INVENTORY = frozenset(
    {
        "active-tracks.json",
        "anchors.json",
        "cue-track.json",
        "detections.json",
        "event-timeline.json",
        "events.json",
        "motion-fragments.json",
        "overlay.mp4",
        "performance.json",
        "shot-windows.json",
        "summary.json",
        "table-state-after.json",
        "table-state-before.json",
        "world-events.json",
    }
)
_LEGACY_REQUIRED_ARTIFACT_INVENTORY = frozenset(
    {
        "detections.json",
        "event-timeline.json",
        "shot-windows.json",
        "table-state-after.json",
        "world-events.json",
    }
)


def _artifact_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in manifest.items() if key != "artifact_id"}
    algorithm = payload.get("artifact_id_algorithm")
    if algorithm is None:
        payload.pop("inventory", None)
        return payload
    if algorithm != "identity-semantic-inventory-v2":
        raise ValueError("shot-local artifact identity algorithm is unsupported")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("shot-local artifact inventory is invalid")
    payload["inventory"] = {
        name: evidence for name, evidence in inventory.items() if name != "performance.json"
    }
    return payload


def _decode_video(path: Path) -> tuple[tuple[DecodedTrackingFrame, ...], float]:
    frames, metadata = decode_frame_source(FileFrameSource(path))
    return frames, metadata.fps


def _observation_row(item) -> dict[str, object]:
    return {
        "detection_id": item.detection_id,
        "center_mm": [item.x_mm, item.y_mm],
        "detection_confidence": item.confidence,
        "group": item.group.value,
        "group_confidence": item.group_confidence,
        "bbox_xyxy_px": list(item.bbox_xyxy_px) if item.bbox_xyxy_px else None,
    }


def _detections_payload(frames: tuple[ObservationFrame, ...]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "observations": [_observation_row(item) for item in frame.observations],
            }
            for frame in frames
        ],
    }


def _window_payload(results: tuple[ShotLocalResult, ...]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "shots": [asdict(result.window) for result in results],
    }


def _anchors_payload(results: tuple[ShotLocalResult, ...]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "shots": [
            {
                "shot_id": result.shot_id,
                "anchors": [asdict(anchor) for anchor in result.anchors],
            }
            for result in results
        ],
    }


def _cue_payload(results: tuple[ShotLocalResult, ...]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "shots": [
            {
                "shot_id": result.shot_id,
                "samples": [asdict(sample) for sample in result.cue_samples],
            }
            for result in results
        ],
    }


def _fragment_payload(results: tuple[ShotLocalResult, ...]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "shots": [
            {
                "shot_id": result.shot_id,
                "fragments": [asdict(fragment) for fragment in result.fragments],
            }
            for result in results
        ],
    }


def _active_tracks_payload(
    analyses: tuple[CanonicalShotAnalysis, ...],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "shots": [
            {
                "shot_id": analysis.evidence.shot_id,
                "tracks": [asdict(track) for track in analysis.active_tracks],
            }
            for analysis in analyses
        ],
    }


def _world_events_payload(
    analyses: tuple[CanonicalShotAnalysis, ...],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "events": [row for analysis in analyses for row in canonical_world_event_rows(analysis)],
    }


def _canonical_event_timeline_payload(
    analyses: tuple[CanonicalShotAnalysis, ...],
    frames_by_id: dict[int, ObservationFrame],
) -> dict[str, object]:
    shots = []
    for analysis in analyses:
        evidence = analysis.evidence
        after = (
            None
            if evidence.window.after_frame_id is None
            else frames_by_id.get(evidence.window.after_frame_id)
        )
        shots.append(
            {
                "shot_id": evidence.shot_id,
                "start_frame_id": evidence.window.start_frame_id,
                "end_frame_id": evidence.window.end_frame_id,
                "after_frame_id": evidence.window.after_frame_id,
                "complete": evidence.window.complete,
                "table_difference": {
                    "before_frame_id": evidence.window.before_frame_id,
                    "after_frame_id": evidence.window.after_frame_id,
                    "before_ball_count": len(evidence.anchors),
                    "after_ball_count": None if after is None else len(after.observations),
                    "net_ball_count": (
                        None if after is None else len(after.observations) - len(evidence.anchors)
                    ),
                },
                "activated_tracks": [
                    {
                        "track_id": track.track_id,
                        "frame_id": track.samples[0].frame_id if track.samples else None,
                        "state": track.samples[0].state if track.samples else None,
                    }
                    for track in analysis.active_tracks
                ],
                "events": canonical_world_event_rows(analysis),
            }
        )
    return {"schema_version": "1.0", "shots": shots}


def _table_state_payload(
    results: tuple[ShotLocalResult, ...],
    frames_by_id: dict[int, ObservationFrame],
    *,
    after: bool,
) -> dict[str, object]:
    rows = []
    for result in results:
        frame_id = result.window.after_frame_id if after else result.window.before_frame_id
        frame = None if frame_id is None else frames_by_id.get(frame_id)
        rows.append(
            {
                "shot_id": result.shot_id,
                "frame_id": frame_id,
                "available": frame is not None,
                "balls": []
                if frame is None
                else [_observation_row(item) for item in frame.observations],
            }
        )
    return {"schema_version": "1.0", "shots": rows}


def _summary_payload(
    *,
    frame_count: int,
    analyses: tuple[CanonicalShotAnalysis, ...],
) -> dict[str, object]:
    results = tuple(analysis.evidence for analysis in analyses)
    cue_states: dict[str, int] = {}
    for result in results:
        for sample in result.cue_samples:
            cue_states[sample.state.value] = cue_states.get(sample.state.value, 0) + 1
    world_events = [row for analysis in analyses for row in canonical_world_event_rows(analysis)]
    return {
        "schema_version": "1.0",
        "status": "DEVELOPMENT_EVIDENCE_ONLY",
        "frame_count": frame_count,
        "shot_count": len(results),
        "complete_shot_count": sum(result.window.complete for result in results),
        "incomplete_shot_count": sum(not result.window.complete for result in results),
        "track_count": sum(len(analysis.active_tracks) for analysis in analyses),
        "fragment_count": sum(len(result.fragments) for result in results),
        "event_count": len(world_events),
        **canonical_world_event_counts(world_events),
        "cue_state_counts": cue_states,
    }


def _validate_calibration_dimensions(
    calibration: TableCalibration,
    config: Mainline1ShotLocalConfig,
) -> None:
    if (
        config.event.table_length_mm != calibration.table_length_mm
        or config.event.table_width_mm != calibration.table_width_mm
    ):
        raise ValueError(
            "shot-local table dimensions do not match calibration: "
            f"config={config.event.table_length_mm}x{config.event.table_width_mm} mm, "
            f"calibration={calibration.table_length_mm}x{calibration.table_width_mm} mm"
        )


def _decoded_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def read_shot_local_artifact_inventory(path: Path) -> dict[str, Any]:
    """Validate retained artifact identity without reopening erased media."""

    if path.is_symlink() or not path.is_dir():
        raise ValueError("shot-local artifact is missing or unsafe")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("shot-local manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "mainline1_shot_local_video":
        raise ValueError("shot-local manifest kind is invalid")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("shot-local artifact inventory is invalid")
    identity_algorithm = manifest.get("artifact_id_algorithm")
    required_inventory = (
        _LEGACY_REQUIRED_ARTIFACT_INVENTORY
        if identity_algorithm is None
        else _REQUIRED_ARTIFACT_INVENTORY
    )
    if (
        not required_inventory.issubset(inventory)
        or identity_algorithm is not None
        and set(inventory) != _REQUIRED_ARTIFACT_INVENTORY
    ):
        raise ValueError("shot-local artifact required product files are incomplete")
    artifact_id = manifest.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or any(character not in "0123456789abcdef" for character in artifact_id)
        or identity_algorithm is not None
        and path.name != artifact_id
    ):
        raise ValueError("shot-local artifact identity is invalid")
    if identity_algorithm is not None:
        identity_payload = _artifact_identity_payload(manifest)
        if hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest() != artifact_id:
            raise ValueError("shot-local artifact identity mismatch")
    return manifest


def verify_mainline1_shot_local_artifact(path: Path) -> dict[str, Any]:
    """Verify the immutable inventory before reusing a shot-local artifact."""
    manifest = read_shot_local_artifact_inventory(path)
    inventory = manifest["inventory"]
    for name, expected in inventory.items():
        candidate = path / name
        if (
            not isinstance(expected, dict)
            or not candidate.is_file()
            or candidate.is_symlink()
            or expected.get("bytes") != candidate.stat().st_size
            or expected.get("sha256") != _sha256(candidate)
        ):
            raise ValueError(f"shot-local artifact inventory mismatch: {name}")
    return manifest


def build_shot_evidence(
    *,
    observations: tuple[ObservationFrame, ...],
    config: Mainline1ShotLocalConfig,
) -> tuple[ShotLocalResult, ...]:
    frames_by_id = {frame.frame_id: frame for frame in observations}
    results = []
    for window in segment_shot_windows(observations, config.window):
        anchors = build_shot_anchors(window, observations)
        cue_samples = track_cue(window, observations, config.cue)
        fragments = track_motion_fragments(
            window,
            observations,
            anchors,
            cue_samples,
            config.fragment,
        )
        first_contact = observe_first_anchor_contact(
            cue_samples,
            anchors,
            fragments,
            config.event,
        )
        start, end = observe_shot_boundaries(window, cue_samples, config.event)
        events = [
            start,
            first_contact,
        ]
        if window.after_frame_id is not None:
            events.extend(
                observe_fragment_pots(
                    fragments,
                    frames_by_id[window.after_frame_id],
                    config.event,
                )
            )
        events.append(end)
        results.append(
            ShotLocalResult(
                shot_id=window.shot_id,
                window=window,
                anchors=anchors,
                cue_samples=cue_samples,
                fragments=fragments,
                events=tuple(events),
            )
        )
    return tuple(results)


def run_mainline1_shot_local_video(
    *,
    video: Path,
    calibration: TableCalibration,
    vision_backend: VisionBackend,
    identity_classifier: GroupIdentityClassifier,
    config: Mainline1ShotLocalConfig,
    output_root: Path,
    active_track_config: ActiveTrackConfig | None = None,
) -> Mainline1ShotLocalArtifact:
    """Run the one-pass shot-local pipeline and publish one immutable artifact."""

    source = video.resolve()
    if video.is_symlink() or not source.is_file():
        raise ValueError("shot-local video is missing or unsafe")
    _validate_calibration_dimensions(calibration, config)
    total_started = time.perf_counter()
    decode_started = total_started
    decoded_frames, fps = _decode_video(source)
    decode_seconds = time.perf_counter() - decode_started
    detection_started = time.perf_counter()
    observations = collect_observation_frames(
        frames=decoded_frames,
        vision_backend=vision_backend,
        identity_classifier=identity_classifier,
        calibration=calibration,
    )
    detection_seconds = time.perf_counter() - detection_started
    event_started = time.perf_counter()
    results = build_shot_evidence(observations=observations, config=config)
    event_seconds = time.perf_counter() - event_started
    effective_active_config = active_track_config or shot_local_product_active_config()
    canonical_timing: dict[str, float] = {}
    analyses = build_canonical_shot_analyses(
        evidence_results=results,
        observations=observations,
        images_by_frame={frame.frame_id: frame.image for frame in decoded_frames},
        event_config=config.event,
        active_config=effective_active_config,
        timing_ms=canonical_timing,
    )
    identity = {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "pipeline_version": _SHOT_LOCAL_PIPELINE_VERSION,
        "video_sha256": _sha256(source),
        "vision_package_id": vision_backend.package_identity.package_id,
        "observation_mode": "full_detection",
        "calibration": calibration.model_dump(mode="json"),
        "config": asdict(config),
        "active_track_config": asdict(effective_active_config),
        "event_policy": "stable_anchor_tracks_three_layer_contacts_v1",
        "frame_count": len(decoded_frames),
    }
    product = build_mainline1_shot_product(
        decoded_frames=decoded_frames,
        observations=observations,
        results=results,
        analyses=analyses,
        calibration=calibration,
        config=config,
        fps=fps,
    )
    return _publish_shot_local_artifact(
        product=product,
        identity=identity,
        output_root=output_root,
        total_started=total_started,
        decode_seconds=decode_seconds,
        detection_seconds=detection_seconds,
        event_seconds=event_seconds,
        canonical_timing=canonical_timing,
        performance_mode="full_file_replay",
    )


def run_cached_mainline1_shot_local_video(
    *,
    video: Path,
    source_artifact: Path,
    config: Mainline1ShotLocalConfig,
    output_root: Path,
    active_track_config: ActiveTrackConfig | None = None,
    build_code_sha256: str | None = None,
) -> Mainline1ShotLocalArtifact:
    """Re-observe verified cached detections through the same shot-local pipeline."""

    total_started = time.perf_counter()
    source = video.resolve()
    if video.is_symlink() or not source.is_file():
        raise ValueError("shot-local video is missing or unsafe")
    if source_artifact.is_symlink():
        raise ValueError("shot-local artifact is missing or unsafe")
    source_root = source_artifact.resolve()
    source_manifest = verify_mainline1_shot_local_artifact(source_root)
    if _sha256(source) != source_manifest.get("video_sha256"):
        raise ValueError("cached observations do not belong to the supplied video")
    calibration = TableCalibration.model_validate(source_manifest["calibration"])
    _validate_calibration_dimensions(calibration, config)
    decode_started = time.perf_counter()
    decoded_frames, fps = _decode_video(source)
    decode_seconds = time.perf_counter() - decode_started
    cache_started = time.perf_counter()
    observations = load_cached_observations(source_root / "detections.json")
    cache_seconds = time.perf_counter() - cache_started
    if len(observations) != len(decoded_frames) or any(
        observation.frame_id != decoded.frame_id or observation.timestamp_ms != decoded.timestamp_ms
        for observation, decoded in zip(observations, decoded_frames, strict=True)
    ):
        raise ValueError("cached observations do not match decoded video frames")
    event_started = time.perf_counter()
    results = build_shot_evidence(observations=observations, config=config)
    event_seconds = time.perf_counter() - event_started
    effective_active_config = active_track_config or shot_local_product_active_config()
    if build_code_sha256 is None:
        code_digest = hashlib.sha256()
        code_paths = [
            Path(__file__),
            Path(__file__).with_name("mainline1_shot_product.py"),
            *sorted(Path(__file__).with_name("shot_local").glob("*.py")),
        ]
        for code_path in code_paths:
            code_digest.update(code_path.name.encode())
            code_digest.update(code_path.read_bytes())
        build_code_sha256 = code_digest.hexdigest()
    if len(build_code_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in build_code_sha256
    ):
        raise ValueError("build code SHA-256 must be lowercase hexadecimal")
    canonical_timing: dict[str, float] = {}
    analyses = build_canonical_shot_analyses(
        evidence_results=results,
        observations=observations,
        images_by_frame={frame.frame_id: frame.image for frame in decoded_frames},
        event_config=config.event,
        active_config=effective_active_config,
        timing_ms=canonical_timing,
    )
    identity = {
        "schema_version": "1.0",
        "kind": "mainline1_shot_local_video",
        "pipeline_version": _SHOT_LOCAL_PIPELINE_VERSION,
        "video_sha256": _sha256(source),
        "vision_package_id": source_manifest["vision_package_id"],
        "observation_mode": "verified_cached_detections",
        "source_artifact_id": source_manifest["artifact_id"],
        "source_detections_sha256": _sha256(source_root / "detections.json"),
        "build_code_sha256": build_code_sha256,
        "calibration": calibration.model_dump(mode="json"),
        "config": asdict(config),
        "active_track_config": asdict(effective_active_config),
        "event_policy": "stable_anchor_tracks_three_layer_contacts_v1",
        "frame_count": len(decoded_frames),
    }
    product = build_mainline1_shot_product(
        decoded_frames=decoded_frames,
        observations=observations,
        results=results,
        analyses=analyses,
        calibration=calibration,
        config=config,
        fps=fps,
    )
    return _publish_shot_local_artifact(
        product=product,
        identity=identity,
        output_root=output_root,
        total_started=total_started,
        decode_seconds=decode_seconds,
        detection_seconds=0.0,
        cache_load_seconds=cache_seconds,
        event_seconds=event_seconds,
        canonical_timing=canonical_timing,
        performance_mode="cached_detection_file_replay",
    )


def _publish_shot_local_artifact(
    *,
    product: Mainline1ShotProductPayloads,
    identity: dict[str, object],
    output_root: Path,
    total_started: float,
    decode_seconds: float,
    detection_seconds: float,
    event_seconds: float,
    canonical_timing: dict[str, float],
    performance_mode: str,
    cache_load_seconds: float = 0.0,
) -> Mainline1ShotLocalArtifact:
    """Publish either fresh or cached observations with one artifact contract."""

    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = ".pending-"
    if identity.get("source_kind") == "usb_live":
        prefix = ".pending-shot-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest() + "-"
    temporary = root / f"{prefix}{uuid4().hex}"
    try:
        temporary.mkdir()
        for name, payload in product.payloads.items():
            (temporary / name).write_bytes(_canonical_bytes(payload))
        overlay_started = time.perf_counter()
        render_canonical_overlay(
            decoded_frames=product.decoded_frames,
            analyses=product.analyses,
            calibration=product.calibration,
            cushion_distance_mm=product.cushion_distance_mm,
            fps=product.fps,
            output=temporary / "overlay.mp4",
        )
        if _decoded_count(temporary / "overlay.mp4") != len(product.decoded_frames):
            raise ValueError("shot-local overlay frame count mismatch")
        overlay_seconds = time.perf_counter() - overlay_started
        total_seconds = time.perf_counter() - total_started
        performance = {
            "schema_version": "1.0",
            "mode": performance_mode,
            "frame_count": len(product.decoded_frames),
            "video_duration_ms": product.decoded_frames[-1].timestamp_ms,
            "decode_ms": round(decode_seconds * 1000, 3),
            "detection_and_cache_ms": round(detection_seconds * 1000, 3),
            "detection_and_cache_ms_per_frame": round(
                detection_seconds * 1000 / len(product.decoded_frames), 3
            ),
            "cached_observation_load_ms": round(cache_load_seconds * 1000, 3),
            "shot_evidence_ms": round(event_seconds * 1000, 3),
            "stable_track_ms": canonical_timing["stable_track_ms"],
            "event_observation_ms": canonical_timing["event_observation_ms"],
            "overlay_ms": round(overlay_seconds * 1000, 3),
            "total_artifact_ms": round(total_seconds * 1000, 3),
            "shots": [
                {
                    "shot_id": result.shot_id,
                    "source_end_ms": result.window.end_ms,
                    "artifact_ready_after_shot_end_ms": round(
                        max(0.0, total_seconds * 1000 - result.window.end_ms), 3
                    ),
                    "complete": result.window.complete,
                }
                for result in product.results
            ],
            "latency_note": (
                "Batch file replay availability; not a streaming latency claim. "
                "Detector/cache and event-chain timings are reported separately."
            ),
        }
        (temporary / "performance.json").write_bytes(_canonical_bytes(performance))
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
        }
        manifest_identity = identity | {
            "artifact_id_algorithm": "identity-semantic-inventory-v2",
            "inventory": inventory,
        }
        artifact_id = hashlib.sha256(
            _canonical_bytes(_artifact_identity_payload(manifest_identity))
        ).hexdigest()
        destination = root / artifact_id
        manifest = manifest_identity | {"artifact_id": artifact_id}
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
        if destination.exists():
            verify_mainline1_shot_local_artifact(destination)
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Mainline1ShotLocalArtifact(
        path=destination,
        artifact_id=artifact_id,
        frame_count=len(product.decoded_frames),
        shot_count=len(product.results),
    )
