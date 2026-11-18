from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from billiards_ai.lab.training.external_video_split import (
    ExternalVideoSplitError,
    build_external_video_split,
    freeze_external_video_split,
    load_external_video_split,
)


def _descriptor(count: int) -> dict[str, object]:
    files: list[dict[str, object]] = [
        {"path": "only_test.jsonl", "sha256": "a" * 64, "media_kind": "annotation"}
    ]
    for index in range(count):
        files.append(
            {
                "path": f"videos/video{index}.mp4",
                "sha256": hashlib.sha256(f"video-{index}".encode()).hexdigest(),
                "media_kind": "video",
            }
        )
    return {
        "schema_version": "1.0",
        "dataset_id": "fixture",
        "source_repository": "ATA-space/Det-SAM2",
        "source_commit": "7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
        "files": files,
    }


def test_split_is_deterministic_and_keeps_used_assets_out_of_locked_test() -> None:
    descriptor = _descriptor(8)

    first = build_external_video_split(
        descriptor,
        descriptor_sha256="b" * 64,
        previously_used_asset_ids=("video0", "video1"),
        seed=20260821,
        train_count=4,
        dev_count=2,
        locked_test_count=2,
    )
    second = build_external_video_split(
        descriptor,
        descriptor_sha256="b" * 64,
        previously_used_asset_ids=("video1", "video0"),
        seed=20260821,
        train_count=4,
        dev_count=2,
        locked_test_count=2,
    )

    assert first == second
    assert len(first.splits["train"]) == 4
    assert len(first.splits["dev"]) == 2
    assert len(first.splits["locked-test"]) == 2
    assert {"video0", "video1"} <= set(first.splits["train"])
    assert not ({"video0", "video1"} & set(first.splits["locked-test"]))
    assert set(first.scan_asset_ids) == (set(first.splits["train"]) | set(first.splits["dev"])) - {
        "video0",
        "video1",
    }
    assert first.locked_test_opened is False
    assert len(first.artifact_id) == 64


def test_split_rejects_unknown_previously_used_asset() -> None:
    with pytest.raises(ExternalVideoSplitError, match="previously used"):
        build_external_video_split(
            _descriptor(4),
            descriptor_sha256="b" * 64,
            previously_used_asset_ids=("missing",),
            seed=1,
            train_count=2,
            dev_count=1,
            locked_test_count=1,
        )


def test_freeze_refuses_overwrite_and_round_trips(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.yaml"
    descriptor_path.write_text(yaml.safe_dump(_descriptor(6)), encoding="utf-8")
    descriptor_sha256 = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    output = tmp_path / "split.json"

    manifest = freeze_external_video_split(
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        output_path=output,
        previously_used_asset_ids=("video0",),
        seed=9,
        train_count=4,
        dev_count=1,
        locked_test_count=1,
    )

    assert output.is_file()
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert manifest.artifact_id in output.read_text(encoding="utf-8")
    assert load_external_video_split(output) == manifest
    with pytest.raises(ExternalVideoSplitError, match="already exists"):
        freeze_external_video_split(
            descriptor_path=descriptor_path,
            descriptor_sha256=descriptor_sha256,
            output_path=output,
            previously_used_asset_ids=("video0",),
            seed=9,
            train_count=4,
            dev_count=1,
            locked_test_count=1,
        )


def test_load_rejects_manifest_identity_tampering(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.yaml"
    descriptor_path.write_text(yaml.safe_dump(_descriptor(6)), encoding="utf-8")
    output = tmp_path / "split.json"
    freeze_external_video_split(
        descriptor_path=descriptor_path,
        descriptor_sha256=hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
        output_path=output,
        previously_used_asset_ids=("video0",),
        seed=9,
        train_count=4,
        dev_count=1,
        locked_test_count=1,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["seed"] = 10
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExternalVideoSplitError, match="identity"):
        load_external_video_split(output)


def test_freeze_rejects_descriptor_hash_mismatch(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.yaml"
    descriptor_path.write_text(yaml.safe_dump(_descriptor(4)), encoding="utf-8")

    with pytest.raises(ExternalVideoSplitError, match="hash mismatch"):
        freeze_external_video_split(
            descriptor_path=descriptor_path,
            descriptor_sha256="b" * 64,
            output_path=tmp_path / "split.json",
            previously_used_asset_ids=("video0",),
            seed=9,
            train_count=2,
            dev_count=1,
            locked_test_count=1,
        )
