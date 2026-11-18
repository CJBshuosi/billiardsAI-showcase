"""DINO feature diversity sampling for a smaller exact-identity review queue."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from uuid import uuid4

import numpy as np

from billiards_ai.lab.training.identity_crop_export import (
    _canonical_bytes,
    _make_immutable,
    _verify_published,
)


class IdentityRepresentativeSelectionError(RuntimeError):
    """Candidate features or requested representative coverage are invalid."""


class IdentityFeatureBackend(Protocol):
    version: str

    def embed(self, crops: tuple[Path, ...]) -> np.ndarray: ...


@dataclass(frozen=True)
class IdentityRepresentativeQueue:
    path: Path
    artifact_id: str
    crop_count: int
    source_crop_count: int


def _select_representatives(
    *,
    embeddings: np.ndarray,
    crop_ids: Sequence[str],
    mandatory_indices: Sequence[int],
    target_count: int,
) -> tuple[int, ...]:
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(crop_ids)
        or not len(crop_ids)
        or not np.isfinite(embeddings).all()
        or target_count < 1
        or target_count > len(crop_ids)
    ):
        raise IdentityRepresentativeSelectionError("identity feature matrix is invalid")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0):
        raise IdentityRepresentativeSelectionError("identity feature vector is empty")
    normalised = embeddings / norms[:, None]
    mandatory = tuple(dict.fromkeys(int(index) for index in mandatory_indices))
    if any(index < 0 or index >= len(crop_ids) for index in mandatory):
        raise IdentityRepresentativeSelectionError("mandatory representative is invalid")
    if len(mandatory) > target_count:
        raise IdentityRepresentativeSelectionError("target count cannot cover all source assets")
    selected = list(mandatory or (min(range(len(crop_ids)), key=lambda index: crop_ids[index]),))
    remaining = set(range(len(crop_ids))) - set(selected)
    while len(selected) < target_count:
        similarities = normalised[list(remaining)] @ normalised[selected].T
        minimum_distances = np.min(1.0 - similarities, axis=1)
        ordered_remaining = list(remaining)
        winner = min(
            range(len(ordered_remaining)),
            key=lambda offset: (
                -float(minimum_distances[offset]),
                crop_ids[ordered_remaining[offset]],
            ),
        )
        selected_index = ordered_remaining[winner]
        selected.append(selected_index)
        remaining.remove(selected_index)
    return tuple(selected)


def build_identity_representative_queue(
    *,
    candidate_artifact: Path,
    output_root: Path,
    embedder: IdentityFeatureBackend,
    target_count: int,
) -> IdentityRepresentativeQueue:
    """Select a video-covered, feature-diverse subset without assigning ball labels."""

    try:
        source = _verify_published(candidate_artifact.resolve())
        manifest = json.loads((source.path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        raise IdentityRepresentativeSelectionError("identity candidates are invalid") from exc
    records = manifest.get("crops")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise IdentityRepresentativeSelectionError("identity candidates are invalid")
    ordered = sorted(records, key=lambda record: str(record.get("crop_id")))
    crop_ids = tuple(str(record["crop_id"]) for record in ordered)
    crop_paths = tuple(
        source.path.joinpath(*PurePosixPath(str(record["relative_path"])).parts)
        for record in ordered
    )
    source_assets: dict[str, list[int]] = {}
    for index, record in enumerate(ordered):
        asset_id = record.get("source_asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise IdentityRepresentativeSelectionError("identity source asset is invalid")
        source_assets.setdefault(asset_id, []).append(index)
    if target_count < len(source_assets):
        raise IdentityRepresentativeSelectionError("target count cannot cover all source assets")
    try:
        embeddings = np.asarray(embedder.embed(crop_paths), dtype=np.float32)
    except Exception as exc:
        raise IdentityRepresentativeSelectionError("identity feature extraction failed") from exc
    mandatory = tuple(
        min(indices, key=lambda index: crop_ids[index])
        for _, indices in sorted(source_assets.items())
    )
    selected_indices = _select_representatives(
        embeddings=embeddings,
        crop_ids=crop_ids,
        mandatory_indices=mandatory,
        target_count=target_count,
    )
    selected = [ordered[index] for index in selected_indices]
    selected.sort(key=lambda record: str(record["crop_id"]))
    identity = {
        "schema_version": "1.0",
        "kind": "ball_identity_representative_candidates",
        "source_candidate_artifact_id": source.artifact_id,
        "source_dataset_id": manifest.get("source_dataset_id"),
        "source_crop_count": source.crop_count,
        "feature_backend_version": embedder.version,
        "selection_method": "video_coverage_then_cosine_farthest_point_v1",
        "target_count": target_count,
        "split_assets": manifest.get("split_assets"),
        "crop_count": len(selected),
        "crops": selected,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / artifact_id
    if target.exists():
        verified = _verify_published(target)
        return IdentityRepresentativeQueue(
            path=verified.path,
            artifact_id=verified.artifact_id,
            crop_count=verified.crop_count,
            source_crop_count=source.crop_count,
        )
    temporary = output_root / f".pending-{uuid4().hex}"
    try:
        (temporary / "crops").mkdir(parents=True)
        for record in selected:
            relative = PurePosixPath(str(record["relative_path"]))
            shutil.copyfile(
                source.path.joinpath(*relative.parts),
                temporary.joinpath(*relative.parts),
            )
        (temporary / "manifest.json").write_bytes(
            _canonical_bytes({"artifact_id": artifact_id, **identity})
        )
        _make_immutable(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verified = _verify_published(target)
    return IdentityRepresentativeQueue(
        path=verified.path,
        artifact_id=verified.artifact_id,
        crop_count=verified.crop_count,
        source_crop_count=source.crop_count,
    )
