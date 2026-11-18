"""Contracts for the prompt/evaluation data boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.propagation.pilot_data import (
    FROZEN_VIDEO_IDS,
    EvaluationCircle,
    EvaluationManifest,
    PilotDataError,
    PilotManifestBundle,
    PromptCircle,
    PromptManifest,
    ShotDefinition,
    build_pilot_manifests,
    preflight_frozen_pilot_counts,
)

# Review frames are annotated at 480x270; the source video is exactly 2x, so the
# review->source mapping applied to evaluation circles is a clean 2x scale.
_REVIEW_SIZE = (480, 270)
_SOURCE_SIZE = (960, 540)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path: Path, *, size: tuple[int, int], frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    try:
        for _ in range(frames):
            writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    finally:
        writer.release()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    annotation_export = tmp_path / "export"
    review_pack = tmp_path / "review-pack"
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    for video_id, shot_index, frames in (("video1", 0, (10, 15, 19)), ("video2", 0, (20, 25, 29))):
        evidence: dict[str, str] = {}
        for role, frame_index in zip(("start", "middle", "end"), frames, strict=True):
            filename = f"shot-{shot_index:03d}-{role}-frame-{frame_index:06d}.jpg"
            raw = f"{video_id}:{frame_index}".encode()
            source = review_pack / "videos" / video_id / "review-frames" / filename
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(raw)
            exported_name = f"{video_id}-{frame_index}.jpg"
            target = annotation_export / "images" / exported_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            images.append(
                {
                    "id": image_id,
                    "file_name": exported_name,
                    "width": 480,
                    "height": 270,
                    "source_video": video_id,
                    "source_frame_index": frame_index,
                }
            )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "annotation_id": f"{video_id}-{frame_index}-ball",
                    "bbox": [100.0, 50.0, 20.0, 20.0],
                    "area": math.pi * 100.0,
                }
            )
            annotation_id += 1
            image_id += 1
            evidence[role] = filename
        _write_jsonl(
            review_pack / "videos" / video_id / "shot-review.jsonl",
            [
                {
                    "shot_index": shot_index,
                    "candidate": {"start_frame": frames[0], "end_frame": 30},
                    "evidence_frames": evidence,
                }
            ],
        )
        video = dataset_root / "videos" / f"{video_id}.mp4"
        _write_video(video, size=_SOURCE_SIZE)
    _write_json(
        annotation_export / "annotations.json",
        {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "ball"}]},
    )
    _write_json(annotation_export / "summary.json", {"annotation_count": len(annotations)})
    return annotation_export, review_pack, dataset_root, output_root


def test_build_splits_start_prompts_from_held_out_evaluation_and_derives_circles(
    tmp_path: Path,
) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)

    bundle = build_pilot_manifests(
        annotation_export=annotation_export,
        review_pack=review_pack,
        dataset_root=dataset_root,
        output_root=output_root,
    )

    assert [
        (item.video_id, item.source_frame_index) for item in bundle.prompt_manifest.circles
    ] == [
        ("video1", 10),
        ("video2", 20),
    ]
    assert [
        (item.video_id, item.source_frame_index) for item in bundle.evaluation_manifest.circles
    ] == [
        ("video1", 15),
        ("video1", 19),
        ("video2", 25),
        ("video2", 29),
    ]
    prompt = bundle.prompt_manifest.circles[0]
    # Prompt circles stay in review pixels (mapped to source at execution time).
    assert (prompt.center_x_px, prompt.center_y_px, prompt.radius_px) == (110.0, 60.0, 10.0)
    # Evaluation circles are mapped review->source (a clean 2x here) so they share
    # the SAM2 prediction pixel space the blind evaluator scores against.
    evaluation = bundle.evaluation_manifest.circles[0]
    assert (evaluation.center_x_px, evaluation.center_y_px, evaluation.radius_px) == (
        220.0,
        120.0,
        20.0,
    )
    prompt_payload = bundle.prompt_manifest.model_dump(mode="json")
    assert "evaluation" not in json.dumps(prompt_payload)
    assert bundle.prompt_manifest_path.name == "prompt-manifest.json"
    assert bundle.evaluation_manifest_path.name == "evaluation-manifest.json"
    assert bundle.prompt_manifest_path.parent != bundle.evaluation_manifest_path.parent
    descriptors = [
        {
            "video_id": video_id,
            "path": f"videos/{video_id}.mp4",
            "bytes": (dataset_root / "videos" / f"{video_id}.mp4").stat().st_size,
            "sha256": _sha256(dataset_root / "videos" / f"{video_id}.mp4"),
        }
        for video_id in ("video1", "video2")
    ]
    expected_dataset_id = hashlib.sha256(
        json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert bundle.prompt_manifest.schema_version == "2.0"
    assert bundle.prompt_manifest.dataset_id == expected_dataset_id
    assert bundle.evaluation_manifest.dataset_id == expected_dataset_id
    assert bundle.evaluation_manifest.prompt_manifest_id == bundle.prompt_manifest.manifest_id


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda export, pack: _duplicate_role(pack), "duplicate video/shot/role"),
        (lambda export, pack: _remove_role(pack), "missing"),
        (lambda export, pack: _change_exported_frame(export), "hash"),
        (lambda export, pack: _add_unexpected_video(pack), "missing|unexpected"),
        (lambda export, pack: _make_non_circular(export), "circular"),
    ],
)
def test_rejects_untrusted_review_join(tmp_path: Path, mutate: object, message: str) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)
    mutate(annotation_export, review_pack)  # type: ignore[operator]

    with pytest.raises(PilotDataError, match=message):
        build_pilot_manifests(
            annotation_export=annotation_export,
            review_pack=review_pack,
            dataset_root=dataset_root,
            output_root=output_root,
        )


def test_production_preflight_is_explicit_and_enforces_frozen_counts(tmp_path: Path) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)
    bundle = build_pilot_manifests(
        annotation_export=annotation_export,
        review_pack=review_pack,
        dataset_root=dataset_root,
        output_root=output_root,
    )

    with pytest.raises(PilotDataError, match="exactly 10"):
        preflight_frozen_pilot_counts(bundle)


def test_production_preflight_requires_the_approved_video_set(tmp_path: Path) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)
    bundle = build_pilot_manifests(
        annotation_export=annotation_export,
        review_pack=review_pack,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    wrong_shots = tuple(
        ShotDefinition(video_id=f"other-{index}", shot_index=0, start_frame=0, end_frame=1)
        for index in range(10)
    )
    changed = bundle.model_copy(
        update={"prompt_manifest": bundle.prompt_manifest.model_copy(update={"shots": wrong_shots})}
    )

    with pytest.raises(PilotDataError, match="approved video set"):
        preflight_frozen_pilot_counts(changed)


def test_rejects_annotation_summary_count_mismatch(tmp_path: Path) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)
    _write_json(annotation_export / "summary.json", {"annotation_count": 999})

    with pytest.raises(PilotDataError, match="count mismatch"):
        build_pilot_manifests(
            annotation_export=annotation_export,
            review_pack=review_pack,
            dataset_root=dataset_root,
            output_root=output_root,
        )


def test_production_preflight_rejects_count_preserving_tampered_evaluation_shots() -> None:
    bundle = _production_count_bundle()
    changed_shots = list(bundle.evaluation_manifest.shots)
    changed_shots[0] = changed_shots[0].model_copy(update={"shot_index": 99})
    changed = bundle.model_copy(
        update={
            "evaluation_manifest": bundle.evaluation_manifest.model_copy(
                update={"shots": tuple(changed_shots)}
            )
        }
    )

    with pytest.raises(PilotDataError, match="evaluation shots"):
        preflight_frozen_pilot_counts(changed)


def test_production_preflight_rejects_evaluation_circle_bound_to_wrong_shot() -> None:
    bundle = _production_count_bundle()
    circles = list(bundle.evaluation_manifest.circles)
    circles[0] = circles[0].model_copy(update={"shot_index": 10})
    changed = bundle.model_copy(
        update={
            "evaluation_manifest": bundle.evaluation_manifest.model_copy(
                update={"circles": tuple(circles)}
            )
        }
    )

    with pytest.raises(PilotDataError, match="source frame"):
        preflight_frozen_pilot_counts(changed)


def test_rejects_review_roles_outside_strict_shot_order(tmp_path: Path) -> None:
    annotation_export, review_pack, dataset_root, output_root = _fixture(tmp_path)
    path = review_pack / "videos" / "video1" / "shot-review.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["evidence_frames"]["middle"] = "shot-000-middle-frame-000035.jpg"
    (path.parent / "review-frames" / "shot-000-middle-frame-000035.jpg").write_bytes(b"late")
    _write_jsonl(path, [row])

    with pytest.raises(PilotDataError, match="role frames"):
        build_pilot_manifests(
            annotation_export=annotation_export,
            review_pack=review_pack,
            dataset_root=dataset_root,
            output_root=output_root,
        )


def _production_count_bundle() -> PilotManifestBundle:
    shots = tuple(
        ShotDefinition(
            video_id=video_id,
            shot_index=shot_index,
            start_frame=shot_index * 100,
            end_frame=(shot_index + 1) * 100,
        )
        for shot_index, video_id in enumerate(
            (*FROZEN_VIDEO_IDS, *FROZEN_VIDEO_IDS, *FROZEN_VIDEO_IDS[:5])
        )
    )
    prompts = tuple(
        PromptCircle(
            annotation_id=f"prompt-{index}",
            video_id=shot.video_id,
            shot_index=shot.shot_index,
            source_frame_index=shot.start_frame,
            source_frame_sha256="a" * 64,
            center_x_px=1,
            center_y_px=1,
            radius_px=1,
        )
        for index, shot in enumerate(shots)
    )
    evaluations: list[EvaluationCircle] = []
    frame_count = 0
    for shot in shots:
        for role, frame_index in (
            ("middle", shot.start_frame + 10),
            ("end", shot.start_frame + 20),
        ):
            circle_count = 15 if frame_count < 43 else 14
            evaluations.extend(
                EvaluationCircle(
                    annotation_id=f"evaluation-{frame_count}-{circle_index}",
                    video_id=shot.video_id,
                    shot_index=shot.shot_index,
                    source_frame_index=frame_index,
                    source_frame_sha256="a" * 64,
                    center_x_px=1,
                    center_y_px=1,
                    radius_px=1,
                    role=role,
                )
                for circle_index in range(circle_count)
            )
            frame_count += 1
    return PilotManifestBundle(
        prompt_manifest=PromptManifest(
            manifest_id="b" * 64, dataset_id="d" * 64, shots=shots, circles=prompts
        ),
        evaluation_manifest=EvaluationManifest(
            manifest_id="c" * 64,
            dataset_id="d" * 64,
            prompt_manifest_id="b" * 64,
            shots=shots,
            circles=tuple(evaluations),
        ),
        prompt_manifest_path=Path("prompt.json"),
        evaluation_manifest_path=Path("evaluation.json"),
    )


def _duplicate_role(review_pack: Path) -> None:
    path = review_pack / "videos" / "video1" / "shot-review.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")


def _remove_role(review_pack: Path) -> None:
    path = review_pack / "videos" / "video1" / "shot-review.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    del row["evidence_frames"]["end"]
    _write_jsonl(path, [row])


def _change_exported_frame(annotation_export: Path) -> None:
    (annotation_export / "images" / "video1-10.jpg").write_bytes(b"changed")


def _add_unexpected_video(review_pack: Path) -> None:
    _write_jsonl(
        review_pack / "videos" / "video999" / "shot-review.jsonl",
        [{"shot_index": 0, "candidate": {"start_frame": 1, "end_frame": 2}, "evidence_frames": {}}],
    )


def _make_non_circular(annotation_export: Path) -> None:
    path = annotation_export / "annotations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["annotations"][0]["bbox"] = [100.0, 50.0, 20.0, 30.0]
    _write_json(path, payload)
