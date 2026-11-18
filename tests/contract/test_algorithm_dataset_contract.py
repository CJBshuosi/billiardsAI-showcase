from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from billiards_ai.lab.contracts import (
    BallFrameAnnotation,
    CalibrationReference,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    ShotAnnotation,
    VideoAsset,
    Visibility,
)
from billiards_ai.shared.models import Point

BENCHMARK_CONFIG = Path("configs/algorithm/benchmark-v1.yaml")


def _asset(
    *,
    asset_id: str,
    digest: str,
    session_id: str,
    person_id: str = "person-1",
) -> VideoAsset:
    return VideoAsset(
        asset_id=asset_id,
        relative_path=f"assets/{asset_id}.bin",
        sha256=digest,
        is_real_person=True,
        person_group_id=person_id,
        session_group_id=session_id,
        table_group_id="table-1",
        camera_group_id="camera-1",
        calibration=CalibrationReference(
            revision="calibration-r1",
            sha256="c" * 64,
        ),
        gate_revision_id=uuid4(),
        approved_uses=(DatasetPurpose.BENCHMARK,),
    )


def test_manifest_rejects_duplicate_asset_hashes() -> None:
    first = _asset(asset_id="asset-a", digest="a" * 64, session_id="session-a")
    duplicate = _asset(asset_id="asset-b", digest="a" * 64, session_id="session-b")

    with pytest.raises(ValidationError, match="asset sha256 values must be unique"):
        DatasetManifest(
            revision="benchmark-v1-r1",
            assets=(first, duplicate),
            splits={
                DatasetSplit.DEV: (first.asset_id,),
                DatasetSplit.LOCKED_TEST: (duplicate.asset_id,),
            },
        )


def test_manifest_rejects_a_person_group_crossing_splits() -> None:
    dev = _asset(asset_id="asset-dev", digest="a" * 64, session_id="session-dev")
    locked = _asset(asset_id="asset-locked", digest="b" * 64, session_id="session-locked")

    with pytest.raises(ValidationError, match="group isolation violation"):
        DatasetManifest(
            revision="benchmark-v1-r1",
            assets=(dev, locked),
            splits={
                DatasetSplit.DEV: (dev.asset_id,),
                DatasetSplit.LOCKED_TEST: (locked.asset_id,),
            },
        )


def test_real_asset_requires_a_calibration_revision() -> None:
    with pytest.raises(ValidationError):
        VideoAsset(
            asset_id="asset-a",
            relative_path="assets/a.bin",
            sha256="a" * 64,
            is_real_person=True,
            person_group_id="person-a",
            session_group_id="session-a",
            table_group_id="table-a",
            camera_group_id="camera-a",
            gate_revision_id=uuid4(),
            approved_uses=(DatasetPurpose.BENCHMARK,),
        )


def test_ball_annotations_reject_pixel_only_positions() -> None:
    with pytest.raises(ValidationError):
        BallFrameAnnotation(
            frame_index=3,
            timestamp_ms=100,
            track_id="track-1",
            identity="cue",
            visibility=Visibility.VISIBLE,
            center_px=(320, 240),
        )


def test_shot_annotations_reject_overlapping_intervals() -> None:
    asset = _asset(
        asset_id="asset-a",
        digest="a" * 64,
        session_id="session-a",
        person_id="person-a",
    )
    frame = BallFrameAnnotation(
        frame_index=3,
        timestamp_ms=100,
        track_id="track-1",
        identity="cue",
        visibility=Visibility.VISIBLE,
        center_mm=Point(x_mm=100, y_mm=200),
    )
    first = ShotAnnotation(
        shot_id="shot-1",
        asset_id=asset.asset_id,
        start_ms=0,
        end_ms=1000,
        ball_frames=(frame,),
    )
    overlapping = ShotAnnotation(
        shot_id="shot-2",
        asset_id=asset.asset_id,
        start_ms=900,
        end_ms=1500,
        ball_frames=(),
    )

    with pytest.raises(ValidationError, match="shot intervals overlap"):
        DatasetManifest(
            revision="benchmark-v1-r1",
            assets=(asset,),
            splits={DatasetSplit.DEV: (asset.asset_id,)},
            shots=(first, overlapping),
        )


def test_benchmark_policy_freezes_minimum_data_and_hard_gates() -> None:
    config = yaml.safe_load(BENCHMARK_CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == "1.0"
    assert config["revision"] == "benchmark-v1-policy-r1"
    assert config["data"]["minimum_sessions"] == 12
    assert config["data"]["minimum_shots"] == 120
    assert config["data"]["dev_shots"] == 60
    assert config["data"]["locked_test_shots"] == 60
    assert config["hard_gates"]["position_mm"] == {"median_max": 10, "p95_max": 20}
    assert config["hard_gates"]["first_hit"] == {
        "top1_accuracy_min": 0.70,
        "coverage_min": 0.80,
    }
    assert config["hard_gates"]["unsupported_field_false_claims_max"] == 0
