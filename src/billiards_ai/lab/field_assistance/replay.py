"""Deterministic file-video replay for the field route-assistance loop."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from billiards_ai.edge.route_assistance.capabilities import capability_registry
from billiards_ai.edge.route_assistance.models import (
    AssistanceEffectKind,
    AssistanceSnapshot,
    MotionState,
)
from billiards_ai.edge.route_assistance.session import (
    AcceptGuidance,
    AcceptMotion,
    transition,
)
from billiards_ai.edge.route_assistance.stability import (
    StabilityConfig,
    StabilitySnapshot,
    observe_scene,
)
from billiards_ai.lab.field_assistance.guidance import build_field_guidance
from billiards_ai.lab.field_assistance.observations import (
    build_snapshot_table_state,
    load_cached_observations,
    scene_observation,
)
from billiards_ai.lab.field_assistance.visual_motion import (
    VisualMotionConfig,
    table_visual_motion_ratio,
)
from billiards_ai.lab.guidance_render import render_scene_frame
from billiards_ai.lab.mainline1_shot_local_video import (
    verify_mainline1_shot_local_artifact,
)
from billiards_ai.lab.planning.rules import EightBallContext
from billiards_ai.shared.calibration.homography import TableCalibration

_PIPELINE_VERSION = "field-route-file-replay/2"


@dataclass(frozen=True, slots=True)
class FieldReplayConfig:
    stability: StabilityConfig
    visual_motion: VisualMotionConfig = field(default_factory=VisualMotionConfig)


@dataclass(frozen=True, slots=True)
class _AuthoritativeMotionWindow:
    shot_id: str
    start_frame_id: int
    end_frame_id: int


@dataclass(frozen=True, slots=True)
class FieldReplayArtifact:
    path: Path
    artifact_id: str
    frame_count: int
    shot_count: int
    guidance_generation_count: int


@dataclass(frozen=True, slots=True)
class LiveDisplayProjection:
    """Mainline 4-safe status derived without inventing video facts."""

    phase: str
    guidance_visible: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class LiveMainline1Consumption:
    """One verified live shot projected onto downstream authority boundaries."""

    artifact_id: str | None
    source_kind: str | None
    shot_id: str | None
    route: dict[str, Any] | None
    facts: tuple[dict[str, Any], ...]
    clip_path: Path | None
    turn_effects: tuple[object, ...]
    growth_writes: tuple[object, ...]
    display: LiveDisplayProjection


class _LiveProductRejected(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _degraded_live_consumption(reason: str) -> LiveMainline1Consumption:
    return LiveMainline1Consumption(
        artifact_id=None,
        source_kind=None,
        shot_id=None,
        route=None,
        facts=(),
        clip_path=None,
        turn_effects=(),
        growth_writes=(),
        display=LiveDisplayProjection("DEGRADED", False, reason),
    )


def _read_product_payload(root: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _LiveProductRejected("LIVE_ARTIFACT_INVALID") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
    return payload


def consume_live_mainline1_artifact(
    artifact_path: Path | None,
    *,
    health_reason: str | None = None,
) -> LiveMainline1Consumption:
    """Fail closed while projecting one activated USB shot to Mainlines 2/3/4.

    The canonical Mainline 1 verifier remains the integrity authority. This
    adapter only assigns downstream permissions: stable after-state may feed
    route planning, observed facts and the verified overlay may feed the facts
    report/ordinary clip surface, and no rule or personal-growth write is
    inferred here.
    """

    if artifact_path is None:
        return _degraded_live_consumption(health_reason or "LIVE_ARTIFACT_MISSING")
    try:
        root = artifact_path.resolve()
        manifest = verify_mainline1_shot_local_artifact(root)
        if manifest.get("source_kind") != "usb_live":
            raise _LiveProductRejected("LIVE_SOURCE_KIND_UNSUPPORTED")
        inventory = manifest.get("inventory")
        required = {
            "shot-windows.json",
            "world-events.json",
            "table-state-after.json",
            "overlay.mp4",
        }
        if not isinstance(inventory, dict) or not required <= set(inventory):
            raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
        windows = _read_product_payload(root, "shot-windows.json").get("shots")
        after_rows = _read_product_payload(root, "table-state-after.json").get("shots")
        events = _read_product_payload(root, "world-events.json").get("events")
        if (
            not isinstance(windows, list)
            or len(windows) != 1
            or not isinstance(windows[0], dict)
            or not isinstance(after_rows, list)
            or len(after_rows) != 1
            or not isinstance(after_rows[0], dict)
            or not isinstance(events, list)
            or any(not isinstance(row, dict) for row in events)
        ):
            raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
        window = windows[0]
        after = after_rows[0]
        shot_id = window.get("shot_id")
        if not isinstance(shot_id, str) or not shot_id.startswith("shot-"):
            raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
        if not bool(window.get("complete")) or window.get("after_frame_id") is None:
            raise _LiveProductRejected("LIVE_PRODUCT_INCOMPLETE")
        if after.get("shot_id") != shot_id:
            raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
        facts = tuple(
            row
            for row in events
            if row.get("shot_id") == shot_id
            and row.get("status") == "observed"
            and isinstance(row.get("confidence"), int | float)
            and float(row["confidence"]) >= 0.7
        )
        has_observed_end = any(row.get("event_type") == "shot_end" for row in facts)
        route = after if bool(after.get("available")) and has_observed_end else None
        effective_reason = health_reason
        if effective_reason is None and route is None:
            effective_reason = "STABLE_AFTER_STATE_UNAVAILABLE"
        if effective_reason is not None:
            route = None
        phase = "READY_FOR_ROUTE" if route is not None else "DEGRADED"
        artifact_id = manifest.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise _LiveProductRejected("LIVE_ARTIFACT_INVALID")
        return LiveMainline1Consumption(
            artifact_id=artifact_id,
            source_kind="usb_live",
            shot_id=shot_id,
            route=route,
            facts=facts,
            clip_path=root / "overlay.mp4",
            turn_effects=(),
            growth_writes=(),
            display=LiveDisplayProjection(phase, False, effective_reason),
        )
    except _LiveProductRejected as exc:
        return _degraded_live_consumption(exc.reason)
    except (OSError, ValueError, TypeError, KeyError):
        return _degraded_live_consumption("LIVE_ARTIFACT_INVALID")


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


def _decode_video(path: Path) -> tuple[tuple[np.ndarray, ...], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("field replay video cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    if not frames:
        raise ValueError("field replay video contains no frames")
    return tuple(frames), fps


def _validate_calibration_viewport(
    calibration: TableCalibration,
    *,
    width: int,
    height: int,
) -> None:
    if any(x < 0 or x > width or y < 0 or y > height for x, y in calibration.corners_px):
        raise ValueError("calibration corners are outside the video viewport")


def _calibration_payload(calibration: TableCalibration) -> dict[str, object]:
    return calibration.model_dump(mode="json")


def _calibration_id(calibration: TableCalibration) -> str:
    return hashlib.sha256(_canonical_bytes(_calibration_payload(calibration))).hexdigest()


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


def _load_authoritative_shot_windows(path: Path) -> tuple[_AuthoritativeMotionWindow, ...]:
    source = path.resolve()
    if path.is_symlink() or not source.is_file():
        raise ValueError("authoritative shot windows are missing or unsafe")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload["shots"]
        windows = tuple(
            _AuthoritativeMotionWindow(
                shot_id=str(row["shot_id"]),
                start_frame_id=int(row["start_frame_id"]),
                end_frame_id=int(row["end_frame_id"]),
            )
            for row in rows
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("authoritative shot windows are invalid") from exc
    if payload.get("schema_version") != "1.0" or not isinstance(rows, list):
        raise ValueError("authoritative shot windows are invalid")
    if any(
        not window.shot_id
        or window.start_frame_id < 0
        or window.end_frame_id <= window.start_frame_id
        for window in windows
    ) or any(
        current.start_frame_id <= previous.end_frame_id
        for previous, current in zip(windows, windows[1:], strict=False)
    ):
        raise ValueError("authoritative shot windows overlap or are unordered")
    return windows


def _load_confirmed_motion_windows(
    windows: tuple[_AuthoritativeMotionWindow, ...],
    events_path: Path,
) -> tuple[_AuthoritativeMotionWindow, ...]:
    source = events_path.resolve()
    if events_path.is_symlink() or not source.is_file():
        raise ValueError("authoritative world events are missing or unsafe")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload["events"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("authoritative world events are invalid") from exc
    if (
        payload.get("schema_version") != "1.0"
        or not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("authoritative world events are invalid")

    refined = []
    for window in windows:
        shot_events = [row for row in rows if row.get("shot_id") == window.shot_id]
        for row in shot_events:
            if row.get("status") != "observed" or row.get("event_type") not in {
                "shot_start",
                "shot_end",
            }:
                continue
            confidence = row.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(float(confidence))
                or float(confidence) < 0.7
            ):
                raise ValueError("authoritative shot boundary confidence is invalid")
        observed_starts = [
            row
            for row in shot_events
            if row.get("event_type") == "shot_start" and row.get("status") == "observed"
        ]
        observed_ends = [
            row
            for row in shot_events
            if row.get("event_type") == "shot_end" and row.get("status") == "observed"
        ]
        if len(observed_starts) > 1 or len(observed_ends) > 1:
            raise ValueError("authoritative shot boundary events are ambiguous")
        try:
            start = (
                window.start_frame_id
                if not observed_starts
                else int(observed_starts[0]["decision_frame_id"])
            )
            end = (
                window.end_frame_id
                if not observed_ends
                else int(observed_ends[0]["decision_frame_id"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("authoritative shot boundary events are invalid") from exc
        if not window.start_frame_id <= start < end <= window.end_frame_id:
            raise ValueError("authoritative shot boundary is outside authoritative shot window")
        refined.append(
            _AuthoritativeMotionWindow(
                shot_id=window.shot_id,
                start_frame_id=start,
                end_frame_id=end,
            )
        )
    return tuple(refined)


def _render_status(
    image: np.ndarray,
    snapshot: AssistanceSnapshot,
) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 54), (12, 18, 24), -1)
    cv2.putText(
        image,
        (
            f"{snapshot.phase.value}  shot={snapshot.shot_sequence:04d} "
            f"scene={snapshot.scene_generation:04d}"
        ),
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if snapshot.guidance is not None:
        power = snapshot.guidance.power
        detail = f"TOP-1  POWER={power.band} ~{power.approximate_percent}% GENERIC"
        color = (90, 240, 180)
    elif snapshot.degraded_reason:
        detail = f"NO ROUTE: {snapshot.degraded_reason}"
        color = (80, 150, 255)
    else:
        detail = "ROUTE HIDDEN"
        color = (180, 180, 180)
    cv2.putText(
        image,
        detail,
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )


def _artifact_from_existing(path: Path, artifact_id: str) -> FieldReplayArtifact:
    verify_field_route_replay_artifact(path)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return FieldReplayArtifact(
        path=path,
        artifact_id=artifact_id,
        frame_count=int(summary["frame_count"]),
        shot_count=int(summary["shot_count"]),
        guidance_generation_count=int(summary["guidance_generation_count"]),
    )


def run_field_route_replay(
    *,
    video: Path,
    observations_path: Path,
    shot_windows_path: Path | None = None,
    events_path: Path | None = None,
    calibration: TableCalibration,
    context: EightBallContext,
    config: FieldReplayConfig,
    output_root: Path,
    source_mainline1_artifact_id: str | None = None,
) -> FieldReplayArtifact:
    """Replay cached observations through the online product state boundary."""

    if source_mainline1_artifact_id is not None and (
        len(source_mainline1_artifact_id) != 64
        or any(character not in "0123456789abcdef" for character in source_mainline1_artifact_id)
    ):
        raise ValueError("source Mainline 1 artifact ID must be lowercase SHA-256")
    source = video.resolve()
    evidence = observations_path.resolve()
    if video.is_symlink() or not source.is_file():
        raise ValueError("field replay video is missing or unsafe")
    if observations_path.is_symlink() or not evidence.is_file():
        raise ValueError("field replay observations are missing or unsafe")
    decoded_frames, fps = _decode_video(source)
    observations = load_cached_observations(evidence)
    if events_path is not None and shot_windows_path is None:
        raise ValueError("authoritative world events require shot windows")
    authoritative_windows = (
        () if shot_windows_path is None else _load_authoritative_shot_windows(shot_windows_path)
    )
    if events_path is not None:
        authoritative_windows = _load_confirmed_motion_windows(
            authoritative_windows,
            events_path,
        )
    if len(decoded_frames) != len(observations):
        raise ValueError("video and observation frame count do not match")
    if [item.frame_id for item in observations] != list(range(len(decoded_frames))):
        raise ValueError("observation frame IDs do not match decoded video")
    height, width = decoded_frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in decoded_frames):
        raise ValueError("decoded video frame dimensions changed")
    _validate_calibration_viewport(calibration, width=width, height=height)

    identity = {
        "schema_version": "1.0",
        "kind": "field_route_file_replay",
        "pipeline_version": _PIPELINE_VERSION,
        "video_sha256": _sha256(source),
        "observations_sha256": _sha256(evidence),
        "shot_windows_sha256": (
            None if shot_windows_path is None else _sha256(shot_windows_path.resolve())
        ),
        "events_sha256": None if events_path is None else _sha256(events_path.resolve()),
        "source_mainline1_artifact_id": source_mainline1_artifact_id,
        "calibration": _calibration_payload(calibration),
        "context_group": context.group,
        "config": asdict(config),
        "frame_count": len(decoded_frames),
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / artifact_id
    if destination.exists():
        return _artifact_from_existing(destination, artifact_id)

    staging = Path(tempfile.mkdtemp(prefix=".field-route-", dir=root))
    try:
        presentation = staging / "presentation.mp4"
        writer = cv2.VideoWriter(
            str(presentation),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("field replay presentation cannot be opened")

        gate = StabilitySnapshot.initial()
        session = AssistanceSnapshot()
        scenes: dict[str, Any] = {}
        timeline_rows = []
        recommendation_rows = []
        try:
            previous_image = None
            for image, observation in zip(decoded_frames, observations, strict=True):
                visual_motion = None
                visual_motion_ratio = None
                if previous_image is not None:
                    visual_motion_ratio = table_visual_motion_ratio(
                        previous_image,
                        image,
                        calibration,
                        config.visual_motion,
                    )
                    visual_motion = visual_motion_ratio >= config.visual_motion.changed_pixel_ratio
                if authoritative_windows:
                    completed = sum(
                        window.end_frame_id <= observation.frame_id
                        for window in authoritative_windows
                    )
                    inside = any(
                        window.start_frame_id <= observation.frame_id < window.end_frame_id
                        for window in authoritative_windows
                    )
                    gate = StabilitySnapshot(
                        motion=MotionState.MOVING if inside else MotionState.STABLE,
                        previous_observation=scene_observation(
                            observation, visual_motion=visual_motion
                        ),
                        scene_generation=completed,
                    )
                else:
                    gate = observe_scene(
                        gate,
                        scene_observation(observation, visual_motion=visual_motion),
                        config.stability,
                    )
                previous_image = image
                effect_kinds = []
                if (
                    gate.motion is not session.motion
                    or gate.scene_generation != session.scene_generation
                ):
                    motion_result = transition(
                        session,
                        AcceptMotion(gate.motion, gate.scene_generation),
                    )
                    session = motion_result.snapshot
                    effect_kinds.extend(effect.kind.value for effect in motion_result.effects)
                    if any(
                        effect.kind is AssistanceEffectKind.REQUEST_ANALYSIS
                        for effect in motion_result.effects
                    ):
                        source_ref = (
                            f"sha256:{identity['observations_sha256']}#frame={observation.frame_id}"
                        )
                        try:
                            state = build_snapshot_table_state(
                                observation,
                                calibration_id=_calibration_id(calibration),
                                source_ref=source_ref,
                                scene_generation=gate.scene_generation,
                            )
                            guidance = build_field_guidance(
                                state=state,
                                context=context,
                                scene_generation=gate.scene_generation,
                            )
                        except ValueError as exc:
                            guidance_result = transition(
                                session,
                                AcceptGuidance(
                                    None,
                                    abstained_reason=f"snapshot_unplannable:{exc}",
                                ),
                            )
                            recommendation_rows.append(
                                {
                                    "scene_generation": gate.scene_generation,
                                    "frame_id": observation.frame_id,
                                    "status": "abstained",
                                    "reason": str(exc),
                                }
                            )
                        else:
                            recommendation_rows.append(
                                {
                                    "scene_generation": gate.scene_generation,
                                    "frame_id": observation.frame_id,
                                    "status": guidance.recommendation.status.value,
                                    "reason": guidance.abstained_reason,
                                    "recommendation": guidance.recommendation.model_dump(
                                        mode="json"
                                    ),
                                    "scene": (
                                        None
                                        if guidance.scene is None
                                        else guidance.scene.model_dump(mode="json")
                                    ),
                                    "display": (
                                        None
                                        if guidance.display is None
                                        else asdict(guidance.display)
                                    ),
                                }
                            )
                            if guidance.display is None:
                                guidance_result = transition(
                                    session,
                                    AcceptGuidance(
                                        None,
                                        abstained_reason=(
                                            guidance.abstained_reason or "no_supported_candidates"
                                        ),
                                    ),
                                )
                            else:
                                assert guidance.scene is not None
                                scenes[guidance.display.recommendation_id] = guidance.scene
                                guidance_result = transition(
                                    session,
                                    AcceptGuidance(guidance.display),
                                )
                        session = guidance_result.snapshot
                        effect_kinds.extend(effect.kind.value for effect in guidance_result.effects)

                rendered = image.copy()
                if session.guidance is not None:
                    scene = scenes.get(session.guidance.recommendation_id)
                    if scene is None:
                        raise ValueError("visible guidance scene is missing")
                    rendered = render_scene_frame(rendered, scene, calibration)
                _render_status(rendered, session)
                writer.write(rendered)
                timeline_rows.append(
                    {
                        "frame_id": observation.frame_id,
                        "timestamp_ms": observation.timestamp_ms,
                        "phase": session.phase.value,
                        "motion": session.motion.value,
                        "scene_generation": session.scene_generation,
                        "shot_sequence": session.shot_sequence,
                        "guidance_visible": session.guidance is not None,
                        "recommendation_id": (
                            None if session.guidance is None else session.guidance.recommendation_id
                        ),
                        "effects": effect_kinds,
                        "reason": session.degraded_reason,
                        "visual_motion": visual_motion,
                        "visual_motion_ratio": visual_motion_ratio,
                    }
                )
        finally:
            writer.release()
        if _decoded_count(presentation) != len(decoded_frames):
            raise ValueError("field replay presentation frame count mismatch")

        summary = {
            "schema_version": "1.0",
            "status": "PASS",
            "frame_count": len(decoded_frames),
            "shot_count": session.shot_sequence,
            "guidance_generation_count": sum(
                row.get("display") is not None for row in recommendation_rows
            ),
        }
        payloads = {
            "timeline.json": {"schema_version": "1.0", "frames": timeline_rows},
            "recommendations.json": {
                "schema_version": "1.0",
                "generations": recommendation_rows,
            },
            "summary.json": summary,
            "capabilities.json": {
                "schema_version": "1.0",
                "capabilities": [asdict(item) for item in capability_registry()],
            },
        }
        for name, payload in payloads.items():
            (staging / name).write_bytes(_canonical_bytes(payload))

        inventory = {}
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if path.name == "manifest.json":
                continue
            inventory[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        manifest = {**identity, "artifact_id": artifact_id, "inventory": inventory}
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
        os.replace(staging, destination)
        return FieldReplayArtifact(
            path=destination,
            artifact_id=artifact_id,
            frame_count=len(decoded_frames),
            shot_count=session.shot_sequence,
            guidance_generation_count=int(summary["guidance_generation_count"]),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_field_route_replay_artifact(path: Path) -> dict[str, Any]:
    """Verify all immutable output hashes before reuse or review."""

    if path.is_symlink() or not path.is_dir():
        raise ValueError("field replay artifact is missing or unsafe")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("field replay manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "field_route_file_replay":
        raise ValueError("field replay manifest kind is invalid")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("field replay inventory is invalid")
    for name, expected in inventory.items():
        candidate = path / name
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, dict)
            or candidate.is_symlink()
            or not candidate.is_file()
            or expected.get("bytes") != candidate.stat().st_size
            or expected.get("sha256") != _sha256(candidate)
        ):
            raise ValueError(f"field replay inventory mismatch: {name}")
    return manifest
