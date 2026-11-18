"""Freeze a deterministic video-level split for an approved external dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256

SplitName = Literal["train", "dev", "locked-test"]


class ExternalVideoSplitError(RuntimeError):
    """The external dataset cannot be split without identity or holdout leakage."""


class ExternalVideoAsset(FrozenContract):
    asset_id: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: Sha256
    split: SplitName
    previously_used: bool


class ExternalVideoSplitManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["external_video_training_split"] = "external_video_training_split"
    artifact_id: Sha256
    source_dataset_id: NonEmptyStr
    source_repository: Literal["ATA-space/Det-SAM2"]
    source_commit: NonEmptyStr
    source_descriptor_sha256: Sha256
    seed: int
    split_policy: Literal["previously_used_forced_train_then_sha256_ranked_video_holdout"] = (
        "previously_used_forced_train_then_sha256_ranked_video_holdout"
    )
    locked_test_opened: Literal[False] = False
    assets: tuple[ExternalVideoAsset, ...] = Field(min_length=3)
    splits: dict[SplitName, tuple[str, ...]]
    previously_used_asset_ids: tuple[str, ...]
    scan_asset_ids: tuple[str, ...]

    @model_validator(mode="after")
    def identity_and_splits_are_consistent(self) -> ExternalVideoSplitManifest:
        if set(self.splits) != {"train", "dev", "locked-test"}:
            raise ValueError("external split must contain train, dev and locked-test")
        asset_ids = {asset.asset_id for asset in self.assets}
        assigned = [asset_id for values in self.splits.values() for asset_id in values]
        if len(asset_ids) != len(self.assets) or len(assigned) != len(set(assigned)):
            raise ValueError("external split assets must be unique")
        if set(assigned) != asset_ids:
            raise ValueError("every external asset must be assigned exactly once")
        used = set(self.previously_used_asset_ids)
        if not used <= set(self.splits["train"]):
            raise ValueError("previously used assets must remain in train")
        expected_scan = (set(self.splits["train"]) | set(self.splits["dev"])) - used
        if set(self.scan_asset_ids) != expected_scan:
            raise ValueError("scan assets must exclude prior-use and locked-test videos")
        split_by_asset = {asset.asset_id: asset.split for asset in self.assets}
        if any(
            split_by_asset[asset_id] != split
            for split, ids in self.splits.items()
            for asset_id in ids
        ):
            raise ValueError("asset split metadata changed")
        return self


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _video_records(descriptor: dict[str, object]) -> list[dict[str, str]]:
    files = descriptor.get("files")
    if not isinstance(files, list):
        raise ExternalVideoSplitError("external descriptor files are invalid")
    records: list[dict[str, str]] = []
    for item in files:
        if not isinstance(item, dict) or item.get("media_kind") != "video":
            continue
        path_value = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(sha256, str):
            raise ExternalVideoSplitError("external video descriptor is invalid")
        path = PurePosixPath(path_value)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
            raise ExternalVideoSplitError("external video path is unsafe")
        records.append({"asset_id": path.stem, "relative_path": path.as_posix(), "sha256": sha256})
    asset_ids = [record["asset_id"] for record in records]
    if not records or len(asset_ids) != len(set(asset_ids)):
        raise ExternalVideoSplitError("external video identities are invalid")
    return sorted(records, key=lambda item: item["asset_id"])


def build_external_video_split(
    descriptor: dict[str, object],
    *,
    descriptor_sha256: str,
    previously_used_asset_ids: tuple[str, ...],
    seed: int,
    train_count: int,
    dev_count: int,
    locked_test_count: int,
) -> ExternalVideoSplitManifest:
    """Build one deterministic split while forcing prior-use videos into train."""

    records = _video_records(descriptor)
    total = len(records)
    if min(train_count, dev_count, locked_test_count) <= 0 or (
        train_count + dev_count + locked_test_count != total
    ):
        raise ExternalVideoSplitError("external split counts do not match the video set")
    used = tuple(sorted(set(previously_used_asset_ids)))
    asset_ids = {record["asset_id"] for record in records}
    if len(used) != len(previously_used_asset_ids) or not set(used) <= asset_ids:
        raise ExternalVideoSplitError("previously used assets are unknown or duplicated")
    if len(used) > train_count:
        raise ExternalVideoSplitError("train split cannot contain all previously used assets")

    remaining = sorted(
        asset_ids - set(used),
        key=lambda asset_id: (
            hashlib.sha256(f"{seed}:{asset_id}".encode()).hexdigest(),
            asset_id,
        ),
    )
    locked = tuple(sorted(remaining[:locked_test_count]))
    dev = tuple(sorted(remaining[locked_test_count : locked_test_count + dev_count]))
    train = tuple(sorted((*used, *remaining[locked_test_count + dev_count :])))
    if len(train) != train_count:
        raise ExternalVideoSplitError("deterministic external split produced invalid counts")
    splits: dict[SplitName, tuple[str, ...]] = {
        "train": train,
        "dev": dev,
        "locked-test": locked,
    }
    split_by_asset = {asset_id: split for split, values in splits.items() for asset_id in values}
    assets = tuple(
        ExternalVideoAsset(
            **record,
            split=split_by_asset[record["asset_id"]],
            previously_used=record["asset_id"] in used,
        )
        for record in records
    )
    scan_asset_ids = tuple(sorted((set(train) | set(dev)) - set(used)))
    identity = {
        "schema_version": "1.0",
        "kind": "external_video_training_split",
        "source_dataset_id": descriptor.get("dataset_id"),
        "source_repository": descriptor.get("source_repository"),
        "source_commit": descriptor.get("source_commit"),
        "source_descriptor_sha256": descriptor_sha256,
        "seed": seed,
        "split_policy": "previously_used_forced_train_then_sha256_ranked_video_holdout",
        "locked_test_opened": False,
        "assets": [asset.model_dump(mode="json") for asset in assets],
        "splits": splits,
        "previously_used_asset_ids": used,
        "scan_asset_ids": scan_asset_ids,
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    try:
        return ExternalVideoSplitManifest.model_validate({**identity, "artifact_id": artifact_id})
    except ValueError as exc:
        raise ExternalVideoSplitError("external split manifest is invalid") from exc


def load_external_video_split(path: Path) -> ExternalVideoSplitManifest:
    """Load only a schema-valid split whose content identity still matches."""

    try:
        manifest = ExternalVideoSplitManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExternalVideoSplitError("external split manifest identity is invalid") from exc
    identity = manifest.model_dump(mode="json", exclude={"artifact_id"})
    if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != manifest.artifact_id:
        raise ExternalVideoSplitError("external split manifest identity changed")
    return manifest


def freeze_external_video_split(
    *,
    descriptor_path: Path,
    descriptor_sha256: str,
    output_path: Path,
    previously_used_asset_ids: tuple[str, ...],
    seed: int,
    train_count: int,
    dev_count: int,
    locked_test_count: int,
) -> ExternalVideoSplitManifest:
    """Load a pinned descriptor and create a no-overwrite canonical split artifact."""

    if output_path.exists():
        raise ExternalVideoSplitError("external split output already exists")
    try:
        descriptor_bytes = descriptor_path.read_bytes()
        descriptor = yaml.safe_load(descriptor_bytes.decode("utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ExternalVideoSplitError("external descriptor cannot be loaded") from exc
    if hashlib.sha256(descriptor_bytes).hexdigest() != descriptor_sha256:
        raise ExternalVideoSplitError("external descriptor hash mismatch")
    if not isinstance(descriptor, dict):
        raise ExternalVideoSplitError("external descriptor is invalid")
    manifest = build_external_video_split(
        descriptor,
        descriptor_sha256=descriptor_sha256,
        previously_used_asset_ids=previously_used_asset_ids,
        seed=seed,
        train_count=train_count,
        dev_count=dev_count,
        locked_test_count=locked_test_count,
    )
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ExternalVideoSplitError("external split output already exists") from exc
    return manifest
