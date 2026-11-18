"""Local-only input-bundle integrity tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.propagation.pilot_bundle import PilotBundleError, build_pilot_input_bundle
from billiards_ai.lab.propagation.pilot_data import FROZEN_VIDEO_IDS


def _write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def _write_video(path: Path) -> None:
    """A real, decodable source video so build_pilot_manifests can probe geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(2):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    export, review, dataset, output = (
        tmp_path / "export",
        tmp_path / "review",
        tmp_path / "dataset",
        tmp_path / "out",
    )
    images, annotations = [], []
    for image_id, (video, role, frame) in enumerate(
        (("video1", "start", 10), ("video1", "middle", 15), ("video1", "end", 19)), start=1
    ):
        name = f"shot-000-{role}-frame-{frame:06d}.jpg"
        raw = f"{video}:{frame}".encode()
        _write(review / "videos" / video / "review-frames" / name, raw)
        _write(export / "images" / f"image-{image_id}.jpg", raw)
        images.append(
            {
                "id": image_id,
                "file_name": f"image-{image_id}.jpg",
                "width": 480,
                "height": 270,
                "source_video": video,
                "source_frame_index": frame,
            }
        )
        annotations.append(
            {"id": image_id, "image_id": image_id, "bbox": [0, 0, 20, 20], "area": math.pi * 100}
        )
    _write(
        review / "videos" / "video1" / "shot-review.jsonl",
        (
            json.dumps(
                {
                    "shot_index": 0,
                    "candidate": {"start_frame": 10, "end_frame": 20},
                    "evidence_frames": {
                        "start": "shot-000-start-frame-000010.jpg",
                        "middle": "shot-000-middle-frame-000015.jpg",
                        "end": "shot-000-end-frame-000019.jpg",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    _write(
        export / "annotations.json",
        json.dumps({"images": images, "annotations": annotations}).encode(),
    )
    _write(export / "summary.json", b'{"annotation_count":3}')
    _write_video(dataset / "videos" / "video1.mp4")
    return export, review, dataset, output


def _production_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    export, review, dataset, output = (
        tmp_path / "export",
        tmp_path / "review",
        tmp_path / "dataset",
        tmp_path / "out",
    )
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    image_id = annotation_id = 1
    shot_counts = {video_id: 2 for video_id in FROZEN_VIDEO_IDS}
    for video_id in FROZEN_VIDEO_IDS[:5]:
        shot_counts[video_id] += 1
    for video_id, shot_count in shot_counts.items():
        rows: list[dict[str, object]] = []
        for shot_index in range(shot_count):
            start = 100 * (shot_index + 1)
            evidence: dict[str, str] = {}
            for role, frame in (("start", start), ("middle", start + 10), ("end", start + 19)):
                filename = f"shot-{shot_index:03d}-{role}-frame-{frame:06d}.jpg"
                raw = f"{video_id}:{frame}".encode()
                _write(review / "videos" / video_id / "review-frames" / filename, raw)
                _write(export / "images" / f"image-{image_id}.jpg", raw)
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"image-{image_id}.jpg",
                        "width": 480,
                        "height": 270,
                        "source_video": video_id,
                        "source_frame_index": frame,
                    }
                )
                circle_count = 11 if image_id <= 18 else 10
                for circle_index in range(circle_count):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "bbox": [circle_index * 20, 0, 20, 20],
                            "area": math.pi * 100,
                        }
                    )
                    annotation_id += 1
                image_id += 1
                evidence[role] = filename
            rows.append(
                {
                    "shot_index": shot_index,
                    "candidate": {"start_frame": start, "end_frame": start + 20},
                    "evidence_frames": evidence,
                }
            )
        _write(
            review / "videos" / video_id / "shot-review.jsonl",
            ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode(),
        )
        _write_video(dataset / "videos" / f"{video_id}.mp4")
    _write(
        export / "annotations.json",
        json.dumps({"images": images, "annotations": annotations}).encode(),
    )
    _write(export / "summary.json", json.dumps({"annotation_count": len(annotations)}).encode())
    assert len(images) == 75
    assert len(annotations) == 768
    return export, review, dataset, output


def test_input_bundle_writes_content_addressed_confined_inventory(tmp_path: Path) -> None:
    export, review, dataset, output = _production_fixture(tmp_path)

    bundle = build_pilot_input_bundle(
        annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
    )

    inventory = json.loads((bundle.path / "inventory.json").read_text(encoding="utf-8"))
    paths = [item["path"] for item in inventory["files"]]
    assert paths == sorted(paths)
    assert len(paths) == 97
    assert (
        sum(path.startswith("review-pack/") and "/review-frames/" in path for path in paths) == 75
    )
    assert not any(path.startswith("annotation-export/images/") for path in paths)
    assert (
        inventory["files"][0]["sha256"]
        == hashlib.sha256((export / "annotations.json").read_bytes()).hexdigest()
    )
    assert bundle.path.name == bundle.bundle_id


@pytest.mark.parametrize("failure", ("extra", "missing", "drift"))
def test_input_bundle_rejects_extra_missing_and_changed_payloads(
    tmp_path: Path, failure: str
) -> None:
    export, review, dataset, output = _production_fixture(tmp_path)
    if failure == "extra":
        _write(dataset / "videos" / "other.mp4", b"other")
    elif failure == "missing":
        (export / "summary.json").unlink()
    else:
        _write(export / "images" / "image-1.jpg", b"changed")

    with pytest.raises(PilotBundleError):
        build_pilot_input_bundle(
            annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
        )


def test_input_bundle_rejects_tampered_existing_content_address(tmp_path: Path) -> None:
    export, review, dataset, output = _production_fixture(tmp_path)
    bundle = build_pilot_input_bundle(
        annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
    )
    (bundle.path / "videos" / "video1.mp4").write_bytes(b"tampered")

    with pytest.raises(PilotBundleError, match="existing"):
        build_pilot_input_bundle(
            annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
        )


@pytest.mark.parametrize(
    "relative",
    (
        "videos/video1/review-frames/rogue.png",
        "videos/video1/review-frames/rogue.webp",
    ),
)
def test_input_bundle_rejects_every_extra_regular_review_payload(
    tmp_path: Path, relative: str
) -> None:
    export, review, dataset, output = _production_fixture(tmp_path)
    _write(review / relative, b"rogue")

    with pytest.raises(PilotBundleError, match="payload mismatch"):
        build_pilot_input_bundle(
            annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
        )


def test_input_bundle_requires_production_preflight(tmp_path: Path) -> None:
    export, review, dataset, output = _fixture(tmp_path)

    with pytest.raises(PilotBundleError, match="exactly 10"):
        build_pilot_input_bundle(
            annotation_export=export, review_pack=review, dataset_root=dataset, output_root=output
        )
