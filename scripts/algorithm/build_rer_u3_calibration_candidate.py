#!/usr/bin/env python3
"""Build an immutable RER U3 calibration candidate from frozen field evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from billiards_ai.lab.mainline1_calibration import create_calibration_artifact
from billiards_ai.lab.mainline1_calibration_registry import register_calibration
from billiards_ai.lab.mainline1_capture import (
    CaptureSetup,
    CaptureSourceKind,
    FrameGeometry,
    capture_setup_fingerprint,
)
from billiards_ai.lab.mainline1_table_validation import (
    detect_middle_pocket_references,
    detect_table_validation_corners,
)
from billiards_ai.shared.models.calibration import (
    CalibrationReference,
    DistortionModel,
    DistortionParameters,
)
from billiards_ai.shared.models.geometry import Point


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one {suffix}")
    return matches[0]


def _json_member(archive: zipfile.ZipFile, suffix: str) -> dict | list:
    return json.loads(archive.read(_member(archive, suffix)))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def build_candidate(
    *,
    motion_archive: Path,
    board_validation_archive: Path,
    output_root: Path,
) -> Path:
    destination = output_root.resolve()
    if destination.exists():
        raise ValueError("output_root already exists; calibration candidates are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.parent / f".{destination.name}-pending-{uuid4().hex}"
    pending.mkdir()
    try:
        with (
            zipfile.ZipFile(motion_archive) as motion,
            zipfile.ZipFile(board_validation_archive) as board,
        ):
            devices = _json_member(motion, "/devices.json")
            lens = _json_member(motion, "/lens-candidate.json")
            slow_report = _json_member(motion, "/slow/report.json")
            frozen_review = _json_member(board, "/frozen-review.json")
            baseline_raw = motion.read(_member(motion, "/slow/baseline.png"))
            source_video = pending / "source-camera.avi"
            source_video.write_bytes(motion.read(_member(motion, "/slow/camera.avi")))

        if not isinstance(devices, list) or len(devices) != 1:
            raise ValueError("field evidence must identify exactly one camera")
        if not isinstance(lens, dict) or int(lens.get("sample_count", 0)) < 24:
            raise ValueError("lens evidence has insufficient accepted samples")
        if float(lens.get("rms_reprojection_error", float("inf"))) > 0.5:
            raise ValueError("lens reprojection error exceeds the candidate threshold")
        boards = frozen_review.get("boards") if isinstance(frozen_review, dict) else None
        if not isinstance(boards, list) or len(boards) < 2:
            raise ValueError("independent checkerboard scale validation is incomplete")
        if any(
            item.get("refit_performed")
            or not item.get("placement_matches")
            or float(item.get("max_pair_distance_error_mm", float("inf"))) > 20.0
            for item in boards
        ):
            raise ValueError("checkerboard validation was refitted, misplaced, or inaccurate")

        frame = cv2.imdecode(np.frombuffer(baseline_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("baseline frame could not be decoded")
        corners = detect_table_validation_corners(frame)
        middle_pockets = detect_middle_pocket_references(frame, corners)
        if corners is None or middle_pockets is None:
            raise ValueError("table rails or independent middle pockets could not be detected")

        assert isinstance(slow_report, dict)
        expected = slow_report.get("expected_controls_readback") or {}
        mapping = (
            ("exposure", "exposure"),
            ("gain", "gain"),
            ("auto_white_balance", "auto_wb"),
            ("white_balance_temperature", "wb_temperature"),
            ("focus", "focus"),
            ("auto_focus", "autofocus"),
            ("zoom", "zoom"),
        )
        image_controls = tuple(
            (canonical, float(expected[source])) for canonical, source in mapping
        )
        if dict(image_controls)["auto_focus"] != 2.0:
            raise ValueError("RER manual-focus flag was not frozen")
        if dict(image_controls)["auto_white_balance"] != 0.0:
            raise ValueError("automatic white balance was not disabled")

        device = devices[0]
        motion_sha = _sha256_file(motion_archive)
        lens_sha = hashlib.sha256(_canonical_bytes(lens)).hexdigest()
        height, width = frame.shape[:2]
        setup = CaptureSetup(
            capture_setup_id="rer-u3-qiaoshi-overhead-20260905-v1",
            table_id="qiaoshi-silver-leg-standard-2540x1270",
            camera_device_id="rer-u3-32e4-6678",
            lens_id="rer-u3-integrated-lens",
            mount_revision=f"raytheon-overhead-{motion_sha[:12]}",
            distortion_revision=f"opencv-{lens_sha[:12]}",
            source_kind=CaptureSourceKind.USB,
            geometry=FrameGeometry(width=width, height=height, crop=(0, 0, width, height)),
            requested_fps=60.0,
            observed_fps=float(slow_report["observed"]["fps"]),
            device_locator_fingerprint=str(device["fingerprint"]),
            validation_corners_px=corners,
            image_controls=image_controls,
        )
        distortion = DistortionParameters(
            model=DistortionModel.OPENCV,
            verified=True,
            camera_matrix=tuple(
                tuple(float(value) for value in row) for row in lens["camera_matrix"]
            ),
            coefficients=tuple(float(value) for value in lens["coefficients"]),
        )
        references = (
            CalibrationReference(pixel=middle_pockets[0], table=Point(x_mm=1270, y_mm=0)),
            CalibrationReference(pixel=middle_pockets[1], table=Point(x_mm=1270, y_mm=1270)),
        )
        calibration_root = pending / "calibrations"
        artifact = create_calibration_artifact(
            video=source_video,
            frame_index=0,
            frame=frame,
            corners_px=corners,
            device_id=setup.camera_device_id,
            output_root=calibration_root,
            capture_setup=setup,
            distortion=distortion,
            references=references,
        )
        record = register_calibration(calibration_root, artifact, setup)
        source_video.unlink()
        (pending / "capture-setup.json").write_bytes(
            _canonical_bytes(setup.model_dump(mode="json"))
        )
        manifest = json.loads((artifact.path / "manifest.json").read_bytes())
        (pending / "candidate-receipt.json").write_bytes(
            _canonical_bytes(
                {
                    "schema_version": "1.0",
                    "status": "FORMAL_CANDIDATE_NOT_DEPLOYED",
                    "production_ready": False,
                    "capture_setup_fingerprint": capture_setup_fingerprint(setup),
                    "calibration_artifact_id": artifact.artifact_id,
                    "calibration_revision": record.revision,
                    "device_fingerprint": device["fingerprint"],
                    "lens_rms_reprojection_error": lens["rms_reprojection_error"],
                    "maximum_reference_error_mm": manifest["maximum_reference_error_mm"],
                    "independent_references": "detected_lower_and_upper_middle_pockets",
                    "source_motion_archive_sha256": motion_sha,
                    "source_board_archive_sha256": _sha256_file(board_validation_archive),
                }
            )
        )
        os.replace(pending, destination)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-archive", type=Path, required=True)
    parser.add_argument("--board-validation-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        build_candidate(
            motion_archive=arguments.motion_archive,
            board_validation_archive=arguments.board_validation_archive,
            output_root=arguments.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
