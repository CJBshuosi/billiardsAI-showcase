from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from billiards_ai.lab.training.identity_crop_export import export_identity_crop_candidates
from billiards_ai.lab.training.identity_representative_selection import (
    IdentityRepresentativeSelectionError,
    _select_representatives,
    build_identity_representative_queue,
)
from tests.unit.lab.training.test_identity_crop_export import _bootstrap_dataset


class _FakeEmbedder:
    version = "fake-dinov2/1"

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray:
        assert len(crops) == 2
        return np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)


def test_farthest_point_selection_is_deterministic_and_keeps_mandatory_coverage() -> None:
    embeddings = np.asarray(
        ((1.0, 0.0), (0.95, 0.05), (0.0, 1.0), (-1.0, 0.0)),
        dtype=np.float32,
    )

    selected = _select_representatives(
        embeddings=embeddings,
        crop_ids=("a", "b", "c", "d"),
        mandatory_indices=(1,),
        target_count=3,
    )

    assert selected == (1, 3, 2)


def test_queue_publishes_only_verified_representative_crops(tmp_path: Path) -> None:
    source = export_identity_crop_candidates(
        bootstrap_dataset=_bootstrap_dataset(tmp_path),
        output_root=tmp_path / "all-candidates",
        crop_size_px=64,
    )

    result = build_identity_representative_queue(
        candidate_artifact=source.path,
        output_root=tmp_path / "representatives",
        embedder=_FakeEmbedder(),
        target_count=2,
    )

    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert result.crop_count == 2
    assert manifest["source_candidate_artifact_id"] == source.artifact_id
    assert manifest["feature_backend_version"] == "fake-dinov2/1"
    assert {item["source_asset_id"] for item in manifest["crops"]} == {
        "video-train",
        "video-valid",
    }


def test_queue_refuses_to_drop_per_video_coverage(tmp_path: Path) -> None:
    source = export_identity_crop_candidates(
        bootstrap_dataset=_bootstrap_dataset(tmp_path),
        output_root=tmp_path / "all-candidates",
        crop_size_px=64,
    )

    with pytest.raises(IdentityRepresentativeSelectionError, match="source assets"):
        build_identity_representative_queue(
            candidate_artifact=source.path,
            output_root=tmp_path / "representatives",
            embedder=_FakeEmbedder(),
            target_count=1,
        )
