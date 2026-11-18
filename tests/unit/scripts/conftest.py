"""Synthetic diagnostic packet inputs; never imply real camera/calibration acceptance."""

import hashlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def diagnostic_packet_assets(tmp_path):
    lens = {
        "test_fixture_only": True,
        "width": 1920,
        "height": 1080,
        "camera_matrix": [[800, 0, 960], [0, 800, 540], [0, 0, 1]],
        "coefficients": [0, 0, 0, 0, 0],
        "sample_count": 36,
        "status": "DIAGNOSTIC_CANDIDATE_NOT_REGISTERED",
        "production_ready": False,
        "device_fingerprint": "865a64805fad161c4f09aeb35214fa23b061be0d8a08c9e6fc99f39c86481a93",
    }
    lens_path = tmp_path / "ci-synthetic-lens.json"
    lens_path.write_text(json.dumps(lens))
    plane = {
        "test_fixture_only": True,
        "schema_version": "1.0",
        "status": "FROZEN_LOCAL_METRIC_CANDIDATE_NOT_ACTIVE",
        "method": "synthetic_flat_plane_for_ci_only",
        "production_ready": False,
        "active_calibration_written": False,
        "absolute_position_validated": False,
        "normal_camera": [0, 0, 1],
        "camera_distance_mm": 1600,
        "checkerboard": {"columns": 11, "rows": 8, "square_size_mm": 25},
        "homography_for_placement_only": [[2, 0, -650], [0, -2, 1715], [0, 0, 1]],
        "lens_file_sha256": hashlib.sha256(lens_path.read_bytes()).hexdigest(),
    }
    plane_path = tmp_path / "ci-synthetic-frozen-plane.json"
    plane_path.write_text(json.dumps(plane))
    return SimpleNamespace(lens=lens_path, frozen=plane_path)
