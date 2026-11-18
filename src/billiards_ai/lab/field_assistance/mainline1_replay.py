"""Resolve one verified Mainline 1 artifact into the existing field replay inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from billiards_ai.lab.mainline1_shot_local_video import (
    verify_mainline1_shot_local_artifact,
)
from billiards_ai.shared.calibration.homography import TableCalibration

_REQUIRED_PRODUCT_FILES = frozenset(
    {
        "detections.json",
        "shot-windows.json",
        "world-events.json",
        "event-timeline.json",
        "table-state-after.json",
    }
)


@dataclass(frozen=True, slots=True)
class Mainline1ReplayInputs:
    artifact_path: Path
    artifact_id: str
    video_path: Path
    observations_path: Path
    shot_windows_path: Path
    events_path: Path
    timeline_path: Path
    table_state_after_path: Path
    calibration: TableCalibration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_mainline1_replay_inputs(
    *,
    artifact_path: Path,
    video_path: Path,
) -> Mainline1ReplayInputs:
    """Verify artifact integrity and bind it to the supplied source video."""

    manifest = verify_mainline1_shot_local_artifact(artifact_path)
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict) or not _REQUIRED_PRODUCT_FILES <= inventory.keys():
        raise ValueError("Mainline 1 artifact is missing required product files")
    source = video_path.resolve()
    if video_path.is_symlink() or not source.is_file():
        raise ValueError("Mainline 1 replay video is missing or unsafe")
    if _sha256(source) != manifest.get("video_sha256"):
        raise ValueError("Mainline 1 artifact does not belong to the supplied video")
    root = artifact_path.resolve()
    return Mainline1ReplayInputs(
        artifact_path=root,
        artifact_id=str(manifest["artifact_id"]),
        video_path=source,
        observations_path=root / "detections.json",
        shot_windows_path=root / "shot-windows.json",
        events_path=root / "world-events.json",
        timeline_path=root / "event-timeline.json",
        table_state_after_path=root / "table-state-after.json",
        calibration=TableCalibration.model_validate(manifest["calibration"]),
    )
