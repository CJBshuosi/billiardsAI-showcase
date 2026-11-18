import hashlib
import json
from pathlib import Path

import pytest

from billiards_ai.lab.contracts import (
    CalibrationReference,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    VideoAsset,
)
from billiards_ai.lab.training.detector_dataset import (
    DetectorAnnotation,
    DetectorFrame,
    DetectorTrainingDataError,
    export_detector_dataset,
)


def _asset(asset_id: str, *, session: str, digest: str) -> VideoAsset:
    return VideoAsset(
        asset_id=asset_id,
        relative_path=f"assets/{asset_id}.bin",
        sha256=digest,
        is_real_person=False,
        person_group_id=f"person-{asset_id}",
        session_group_id=session,
        table_group_id=f"table-{asset_id}",
        camera_group_id=f"camera-{asset_id}",
        calibration=CalibrationReference(revision="cal-r1", sha256="c" * 64),
        approved_uses=(DatasetPurpose.LOCAL_TRAINING,),
    )


def _frame(source_root: Path, *, asset_id: str = "train-a") -> DetectorFrame:
    image = source_root / "frames" / "frame-0001.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"synthetic table crop without audio")
    return DetectorFrame(
        frame_id="frame-0001",
        source_asset_id=asset_id,
        relative_path="frames/frame-0001.png",
        sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        width=640,
        height=360,
        table_cropped=True,
        audio_free=True,
        annotations=(
            DetectorAnnotation(
                polygon_px=((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
                bbox_xywh=(10.0, 20.0, 20.0, 20.0),
                visibility="occluded",
                difficult=True,
            ),
        ),
    )


def test_export_preserves_polygons_hashes_and_difficult_examples(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    frame = _frame(source_root)
    asset = _asset("train-a", session="session-train", digest="a" * 64)
    manifest = DatasetManifest(
        revision="dataset-r1",
        locked=True,
        assets=(asset,),
        splits={DatasetSplit.TRAIN: (asset.asset_id,)},
    )

    result = export_detector_dataset(
        manifest=manifest,
        frames=(frame,),
        source_root=source_root,
        output=tmp_path / "export",
    )
    coco = json.loads((result.output / "annotations" / "instances_train.json").read_text())
    derived = json.loads((result.output / "derived-manifest.json").read_text())

    assert coco["images"][0]["sha256"] == frame.sha256
    assert coco["annotations"][0]["segmentation"] == [
        [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0]
    ]
    assert coco["annotations"][0]["difficult"] is True
    assert coco["annotations"][0]["visibility"] == "occluded"
    assert derived["source_assets"][0]["sha256"] == asset.sha256
    assert derived["source_assets"][0]["gate_revision_id"] is None
    assert len(result.data_hash) == 64


def test_export_rejects_session_present_in_non_training_split(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    frame = _frame(source_root)
    train = _asset("train-a", session="leaked-session", digest="a" * 64)
    dev = _asset("dev-a", session="leaked-session", digest="b" * 64)
    manifest = DatasetManifest.model_construct(
        revision="dataset-leaked",
        locked=True,
        assets=(train, dev),
        splits={DatasetSplit.TRAIN: (train.asset_id,), DatasetSplit.DEV: (dev.asset_id,)},
        shots=(),
        schema_version="1.0",
        coordinate_unit="mm",
    )

    with pytest.raises(DetectorTrainingDataError, match="session leakage"):
        export_detector_dataset(
            manifest=manifest,
            frames=(frame,),
            source_root=source_root,
            output=tmp_path / "export",
        )


def test_export_rejects_non_crop_or_changed_frame_before_copy(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    frame = _frame(source_root).model_copy(update={"table_cropped": False})
    asset = _asset("train-a", session="session-train", digest="a" * 64)
    manifest = DatasetManifest(
        revision="dataset-r1",
        locked=True,
        assets=(asset,),
        splits={DatasetSplit.TRAIN: (asset.asset_id,)},
    )

    with pytest.raises(DetectorTrainingDataError, match="table-cropped and audio-free"):
        export_detector_dataset(
            manifest=manifest,
            frames=(frame,),
            source_root=source_root,
            output=tmp_path / "export",
        )
    assert not (tmp_path / "export").exists()
