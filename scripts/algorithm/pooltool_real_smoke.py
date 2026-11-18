"""Run one local synthetic shot through the real optional pooltool backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

from billiards_ai.lab.adapters.pooltool_physics import PooltoolPhysicsBackend
from billiards_ai.shared.models.geometry import Point
from billiards_ai.shared.models.guidance import (
    GuidanceStatus,
    PhysicsStatus,
    PlannedShot,
    PocketId,
    PowerBand,
)
from billiards_ai.shared.models.table_state import (
    BallKind,
    CanonicalBall,
    CanonicalTableState,
    MotionStatus,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _fixture() -> tuple[CanonicalTableState, PlannedShot]:
    balls = (
        CanonicalBall(
            track_id="cue-track",
            identity="cue",
            kind=BallKind.CUE,
            identity_confidence=1,
            center=Point(x_mm=500, y_mm=600),
            radius_mm=28.5,
            detection_confidence=1,
        ),
        CanonicalBall(
            track_id="target",
            identity="1",
            kind=BallKind.SOLID,
            identity_confidence=1,
            center=Point(x_mm=1500, y_mm=600),
            radius_mm=28.5,
            detection_confidence=1,
        ),
    )
    state = CanonicalTableState(
        frame_id=0,
        timestamp_ms=0,
        calibration_id="synthetic-pooltool-smoke",
        motion_status=MotionStatus.STATIONARY,
        balls=balls,
        source_refs=("synthetic:pooltool-smoke",),
    )
    shot = PlannedShot(
        candidate_id="synthetic-direct-shot",
        target_track_id="target",
        target_identity="1",
        target_kind=BallKind.SOLID,
        pocket=PocketId.TOP_RIGHT,
        status=GuidanceStatus.PLANNED,
        ghost_ball_center=Point(x_mm=1443, y_mm=600),
        cue_path=(Point(x_mm=500, y_mm=600), Point(x_mm=1443, y_mm=600)),
        object_path=(Point(x_mm=1500, y_mm=600), Point(x_mm=2540, y_mm=1270)),
        cut_angle_deg=0,
        cue_travel_mm=943,
        object_travel_mm=1200,
        evidence_confidence=1,
        evidence_refs=("synthetic:pooltool-smoke",),
    )
    return state, shot


def run_smoke(output_root: Path) -> Path:
    state, shot = _fixture()
    result = PooltoolPhysicsBackend().simulate(state, shot, PowerBand.MEDIUM)
    if result.status is not PhysicsStatus.SIMULATED:
        raise RuntimeError(f"pooltool real smoke failed: {result.reason}")
    document = {
        "schema_version": "1.0",
        "status": "PASS",
        "backend_version": result.backend_version,
        "candidate_id": result.candidate_id,
        "cue_path_points": len(result.cue_path),
        "object_path_ids": sorted(result.object_paths),
        "events": list(result.events),
        "synthetic_only": True,
        "real_video_accuracy": False,
    }
    payload = _canonical(document)
    artifact_id = hashlib.sha256(payload).hexdigest()
    destination = output_root.resolve() / artifact_id
    if destination.exists():
        if (destination / "result.json").read_bytes() == payload:
            return destination
        raise RuntimeError("pooltool smoke artifact destination conflicts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{artifact_id}.tmp-{uuid.uuid4()}"
    temporary.mkdir()
    try:
        (temporary / "result.json").write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.rmdir()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    destination = run_smoke(arguments.output_root)
    print(json.dumps({"status": "PASS", "output": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
