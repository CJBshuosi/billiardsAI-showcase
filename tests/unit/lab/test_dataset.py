import hashlib
from pathlib import Path
from uuid import uuid4

import pytest

from billiards_ai.lab.cli import main
from billiards_ai.lab.contracts import (
    CalibrationReference,
    DatasetManifest,
    DatasetPurpose,
    DatasetSplit,
    VideoAsset,
)
from billiards_ai.lab.dataset import DatasetIntegrityError, freeze_dataset, load_frozen_dataset


class _PurposeAuthorizer:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed

    def allows(self, asset: VideoAsset, purpose: DatasetPurpose) -> bool:
        return self.allowed and purpose in asset.approved_uses


def _write_draft(tmp_path: Path) -> tuple[Path, Path]:
    asset_path = tmp_path / "assets" / "video.bin"
    asset_path.parent.mkdir()
    asset_path.write_bytes(b"approved synthetic stand-in")
    digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    asset = VideoAsset(
        asset_id="asset-a",
        relative_path="assets/video.bin",
        sha256=digest,
        is_real_person=True,
        person_group_id="person-a",
        session_group_id="session-a",
        table_group_id="table-a",
        camera_group_id="camera-a",
        calibration=CalibrationReference(revision="cal-r1", sha256="c" * 64),
        gate_revision_id=uuid4(),
        approved_uses=(DatasetPurpose.BENCHMARK,),
    )
    manifest = DatasetManifest(
        revision="benchmark-v1-r1",
        assets=(asset,),
        splits={DatasetSplit.DEV: (asset.asset_id,)},
    )
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return draft_path, asset_path


def test_freeze_and_load_verify_manifest_and_asset_hashes(tmp_path: Path) -> None:
    draft_path, _ = _write_draft(tmp_path)
    frozen_path = tmp_path / "manifest.json"

    frozen = freeze_dataset(draft_path, frozen_path)
    loaded = load_frozen_dataset(
        frozen_path,
        required_use=DatasetPurpose.BENCHMARK,
        authorizer=_PurposeAuthorizer(allowed=True),
    )

    assert frozen.locked is True
    assert loaded == frozen
    assert frozen_path.with_suffix(".json.sha256").is_file()


def test_load_fails_before_decoding_when_gate_use_is_not_authorized(tmp_path: Path) -> None:
    draft_path, _ = _write_draft(tmp_path)
    frozen_path = tmp_path / "manifest.json"
    freeze_dataset(draft_path, frozen_path)

    with pytest.raises(DatasetIntegrityError, match="purpose is not authorized"):
        load_frozen_dataset(
            frozen_path,
            required_use=DatasetPurpose.BENCHMARK,
            authorizer=_PurposeAuthorizer(allowed=False),
        )


def test_load_rejects_media_changed_after_freeze(tmp_path: Path) -> None:
    draft_path, asset_path = _write_draft(tmp_path)
    frozen_path = tmp_path / "manifest.json"
    freeze_dataset(draft_path, frozen_path)
    asset_path.write_bytes(b"tampered")

    with pytest.raises(DatasetIntegrityError, match="asset hash mismatch"):
        load_frozen_dataset(
            frozen_path,
            required_use=DatasetPurpose.BENCHMARK,
            authorizer=_PurposeAuthorizer(allowed=True),
        )


def test_dataset_freeze_cli_writes_a_locked_manifest(tmp_path: Path) -> None:
    draft_path, _ = _write_draft(tmp_path)
    frozen_path = tmp_path / "manifest.json"

    exit_code = main(
        [
            "dataset",
            "freeze",
            "--source",
            str(draft_path),
            "--output",
            str(frozen_path),
        ]
    )

    assert exit_code == 0
    assert DatasetManifest.model_validate_json(frozen_path.read_text()).locked is True
