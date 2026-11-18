"""Verified shot-local artifact replay with auditable performance evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from billiards_ai.edge.realtime_chain.models import (
    AnchorEvidence,
    CueEvidence,
    EventEvidence,
    EvidenceSource,
    ShotReplayEvidence,
    TableBallEvidence,
)
from billiards_ai.edge.realtime_chain.profile import aggregate_profile
from billiards_ai.edge.realtime_chain.projection import ProjectionConfig, project_shot_evidence
from billiards_ai.lab.mainline1_shot_local_video import verify_mainline1_shot_local_artifact

_REPLAY_POLICY_VERSION = "lightweight-realtime-chain-v1"


@dataclass(frozen=True, slots=True)
class ReplayArtifact:
    path: Path
    artifact_id: str
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


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"shot-local replay input is invalid: {path.name}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError(f"shot-local replay input is invalid: {path.name}")
    return value


def _shot_index(path: Path, name: str) -> dict[str, dict[str, Any]]:
    rows = _json(path / name).get("shots")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"shot-local replay shot rows are invalid: {name}")
    index = {str(row.get("shot_id", "")): row for row in rows}
    if "" in index or len(index) != len(rows):
        raise ValueError(f"shot-local replay shot IDs are invalid: {name}")
    return index


def _require_matching_shots(indexes: tuple[dict[str, dict[str, Any]], ...]) -> tuple[str, ...]:
    if not indexes:
        return ()
    expected = set(indexes[0])
    if any(set(index) != expected for index in indexes[1:]):
        raise ValueError("shot-local replay files contain mismatched shot IDs")
    return tuple(sorted(expected))


def _center(row: dict[str, Any]) -> tuple[int, int]:
    center = row.get("center_mm")
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("shot-local table ball center is invalid")
    return int(center[0]), int(center[1])


def _load_shots(path: Path, session_id: str) -> tuple[ShotReplayEvidence, ...]:
    windows = _shot_index(path, "shot-windows.json")
    anchors = _shot_index(path, "anchors.json")
    cue = _shot_index(path, "cue-track.json")
    events = _shot_index(path, "events.json")
    after = _shot_index(path, "table-state-after.json")
    shot_ids = _require_matching_shots((windows, anchors, cue, events, after))
    loaded = []
    try:
        for shot_id in shot_ids:
            window = windows[shot_id]
            after_row = after[shot_id]
            loaded.append(
                ShotReplayEvidence(
                    session_id=session_id,
                    shot_id=shot_id,
                    complete=bool(window["complete"]),
                    before_frame_id=int(window["before_frame_id"]),
                    after_frame_id=(
                        None
                        if window.get("after_frame_id") is None
                        else int(window["after_frame_id"])
                    ),
                    anchors=tuple(
                        AnchorEvidence(
                            anchor_id=str(row["anchor_id"]),
                            group=str(row["group"]),
                            x_mm=int(row["x_mm"]),
                            y_mm=int(row["y_mm"]),
                            confidence=float(row["confidence"]),
                        )
                        for row in anchors[shot_id]["anchors"]
                    ),
                    cue_samples=tuple(
                        CueEvidence(
                            frame_id=int(row["frame_id"]),
                            timestamp_ms=int(row["timestamp_ms"]),
                            state=str(row["state"]),
                            x_mm=None if row.get("x_mm") is None else int(row["x_mm"]),
                            y_mm=None if row.get("y_mm") is None else int(row["y_mm"]),
                            confidence=float(row["confidence"]),
                        )
                        for row in cue[shot_id]["samples"]
                    ),
                    events=tuple(
                        EventEvidence(
                            event_id=str(row["event_id"]),
                            event_type=str(row["event_type"]),
                            status=str(row["status"]),
                            timestamp_ms=(
                                None
                                if row.get("timestamp_ms") is None
                                else int(row["timestamp_ms"])
                            ),
                            actor_id=None if row.get("actor_id") is None else str(row["actor_id"]),
                            target_id=(
                                None if row.get("target_id") is None else str(row["target_id"])
                            ),
                            evidence_frame_ids=tuple(
                                int(item) for item in row["evidence_frame_ids"]
                            ),
                            confidence=float(row["confidence"]),
                            reason=None if row.get("reason") is None else str(row["reason"]),
                        )
                        for row in events[shot_id]["events"]
                    ),
                    after_balls=tuple(
                        TableBallEvidence(
                            group=str(row["group"]),
                            x_mm=_center(row)[0],
                            y_mm=_center(row)[1],
                            confidence=min(
                                float(row["detection_confidence"]),
                                float(row["group_confidence"]),
                            ),
                        )
                        for row in after_row["balls"]
                    )
                    if bool(after_row["available"])
                    else (),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("shot-local replay evidence contract is invalid") from exc
    return tuple(loaded)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def verify_replay_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("replay artifact is missing or unsafe")
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("replay manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != "lightweight_realtime_chain":
        raise ValueError("replay manifest kind is invalid")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("replay inventory is invalid")
    for name, expected in inventory.items():
        candidate = path / name
        if (
            not isinstance(expected, dict)
            or not candidate.is_file()
            or candidate.is_symlink()
            or expected.get("bytes") != candidate.stat().st_size
            or expected.get("sha256") != _sha256(candidate)
        ):
            raise ValueError(f"replay artifact inventory mismatch: {name}")
    return manifest


def run_replay(
    *,
    source_artifact: Path,
    output_root: Path,
    user_id: str,
    session_id: str,
) -> ReplayArtifact:
    """Replay an immutable shot-local artifact without rerunning detector inference."""

    started_ns = time.perf_counter_ns()
    if source_artifact.is_symlink():
        raise ValueError("shot-local artifact is missing or unsafe")
    source_path = source_artifact.resolve()
    manifest = verify_mainline1_shot_local_artifact(source_path)
    source = EvidenceSource(
        artifact_id=str(manifest["artifact_id"]),
        source_video_sha256=str(manifest["video_sha256"]),
        manifest_sha256=_sha256(source_path / "manifest.json"),
    )
    load_started_ns = time.perf_counter_ns()
    shots = _load_shots(source_path, session_id)
    loaded_ns = time.perf_counter_ns()
    config = ProjectionConfig()
    projections = []
    per_shot_ms = []
    for shot in shots:
        shot_started_ns = time.perf_counter_ns()
        projections.append(
            project_shot_evidence(user_id=user_id, source=source, shot=shot, config=config)
        )
        per_shot_ms.append((time.perf_counter_ns() - shot_started_ns) / 1_000_000)
    projected_ns = time.perf_counter_ns()
    aggregate = aggregate_profile(
        user_id,
        tuple(projection.profile_increment for projection in projections),
    )
    aggregated_ns = time.perf_counter_ns()

    identity = {
        "schema_version": "1.0",
        "kind": "lightweight_realtime_chain",
        "policy_version": _REPLAY_POLICY_VERSION,
        "source_artifact_id": source.artifact_id,
        "source_manifest_sha256": source.manifest_sha256,
        "user_id": user_id,
        "session_id": session_id,
        "config": asdict(config),
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    root = output_root.resolve()
    destination = root / artifact_id
    if destination.exists():
        existing = verify_replay_artifact(destination)
        if existing.get("artifact_id") != artifact_id:
            raise ValueError("replay artifact identity mismatch")
        return ReplayArtifact(destination, artifact_id, len(shots))

    payloads = {
        "realtime-chain.json": {
            "schema_version": "1.0",
            "mode": "artifact_replay_async",
            "source": asdict(source),
            "claims": {
                "global_fixed_ball_id": False,
                "sam_realtime_main_chain": False,
            },
            "shots": [asdict(item) for item in projections],
        },
        "profile-increment.json": {
            "schema_version": "1.0",
            "increments": [asdict(item.profile_increment) for item in projections],
        },
        "profile-aggregate.json": asdict(aggregate) | {"schema_version": "1.0"},
        "performance.json": {
            "schema_version": "1.0",
            "mode": "artifact_replay_async",
            "detector_runtime_included": False,
            "shot_count": len(shots),
            "load_ms": (loaded_ns - load_started_ns) / 1_000_000,
            "projection_total_ms": (projected_ns - loaded_ns) / 1_000_000,
            "aggregate_ms": (aggregated_ns - projected_ns) / 1_000_000,
            "total_ms_before_publish": (aggregated_ns - started_ns) / 1_000_000,
            "per_shot_projection_ms": per_shot_ms,
            "projection_p50_ms": _percentile(tuple(per_shot_ms), 0.5),
            "projection_p95_ms": _percentile(tuple(per_shot_ms), 0.95),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".pending-{uuid4().hex}"
    try:
        temporary.mkdir()
        for name, payload in payloads.items():
            (temporary / name).write_bytes(_canonical_bytes(payload))
        inventory = {
            item.name: {"bytes": item.stat().st_size, "sha256": _sha256(item)}
            for item in sorted(temporary.iterdir(), key=lambda candidate: candidate.name)
        }
        replay_manifest = identity | {"artifact_id": artifact_id, "inventory": inventory}
        (temporary / "manifest.json").write_bytes(_canonical_bytes(replay_manifest))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ReplayArtifact(destination, artifact_id, len(shots))
