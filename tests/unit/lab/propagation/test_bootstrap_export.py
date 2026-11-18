from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from billiards_ai.lab.propagation.bootstrap_export import (
    BootstrapFrame,
    BootstrapSelection,
    BootstrapShot,
    assign_video_splits,
    publish_bootstrap_dataset,
    select_bootstrap_frames,
    verify_bootstrap_gate,
)
from billiards_ai.lab.propagation.contracts import InstanceAbsence, MaskObservation


def _observation(
    *,
    asset_id: str,
    frame_index: int,
    instance_id: str,
    qa_flags: tuple[str, ...] = (),
    box: tuple[float, float, float, float] = (10.0, 10.0, 20.0, 20.0),
    area: int = 80,
) -> MaskObservation:
    return MaskObservation(
        asset_id=asset_id,
        frame_index=frame_index,
        instance_id=instance_id,
        class_name="ball",
        prompt_frame_index=0,
        center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        box_xyxy_px=box,
        area_px=area,
        mask_ref=f"masks/{instance_id}/{frame_index:08d}.png",
        qa_flags=qa_flags,
    )


def _absence(*, asset_id: str, frame_index: int, instance_id: str) -> InstanceAbsence:
    return InstanceAbsence(
        asset_id=asset_id,
        frame_index=frame_index,
        instance_id=instance_id,
        class_name="ball",
        prompt_frame_index=0,
        reason="empty_mask",
    )


def _shot(*, asset_id: str = "video1") -> BootstrapShot:
    return BootstrapShot(
        asset_id=asset_id,
        shot_index=0,
        start_frame=0,
        end_frame=46,
        prompt_boxes={
            "ball-a": (10.0, 10.0, 20.0, 20.0),
            "ball-b": (30.0, 30.0, 40.0, 40.0),
        },
    )


def _complete_frame(asset_id: str, frame_index: int) -> tuple[MaskObservation, ...]:
    return (
        _observation(asset_id=asset_id, frame_index=frame_index, instance_id="ball-a"),
        _observation(
            asset_id=asset_id,
            frame_index=frame_index,
            instance_id="ball-b",
            box=(30.0, 30.0, 40.0, 40.0),
        ),
    )


def test_selection_excludes_every_frozen_evaluation_frame() -> None:
    observations = tuple(
        item for frame_index in (0, 15, 30, 45) for item in _complete_frame("video1", frame_index)
    )

    selection = select_bootstrap_frames(
        shots=(_shot(),),
        observations=observations,
        absences=(),
        frozen_evaluation_frames=frozenset({("video1", 15), ("video1", 30)}),
        frame_stride=15,
    )

    assert [(item.asset_id, item.frame_index) for item in selection.frames] == [
        ("video1", 0),
        ("video1", 45),
    ]
    assert selection.rejection_counts == {"frozen_evaluation_frame": 2}


def test_selection_rejects_the_whole_frame_for_absence_qa_or_incomplete_inventory() -> None:
    observations = (
        *_complete_frame("video1", 0),
        _observation(
            asset_id="video1",
            frame_index=15,
            instance_id="ball-a",
            qa_flags=("area_jump",),
        ),
        _observation(
            asset_id="video1",
            frame_index=15,
            instance_id="ball-b",
            box=(30.0, 30.0, 40.0, 40.0),
        ),
        _observation(asset_id="video1", frame_index=30, instance_id="ball-a"),
        _observation(asset_id="video1", frame_index=45, instance_id="ball-a"),
    )
    absences = (_absence(asset_id="video1", frame_index=45, instance_id="ball-b"),)

    selection = select_bootstrap_frames(
        shots=(_shot(),),
        observations=observations,
        absences=absences,
        frozen_evaluation_frames=frozenset(),
        frame_stride=15,
    )

    assert [(item.asset_id, item.frame_index) for item in selection.frames] == [("video1", 0)]
    assert selection.rejection_counts == {
        "explicit_absence": 1,
        "incomplete_instance_inventory": 1,
        "qa_flagged": 1,
    }


def test_selection_rejects_stable_but_geometrically_leaking_masks() -> None:
    observations = (
        _observation(
            asset_id="video1",
            frame_index=0,
            instance_id="ball-a",
            box=(0.0, 10.0, 40.0, 20.0),
            area=80,
        ),
        _observation(
            asset_id="video1",
            frame_index=0,
            instance_id="ball-b",
            box=(30.0, 30.0, 40.0, 40.0),
        ),
    )

    selection = select_bootstrap_frames(
        shots=(
            BootstrapShot(
                asset_id="video1",
                shot_index=0,
                start_frame=0,
                end_frame=1,
                prompt_boxes=_shot().prompt_boxes,
            ),
        ),
        observations=observations,
        absences=(),
        frozen_evaluation_frames=frozenset(),
        frame_stride=15,
    )

    assert selection.frames == ()
    assert selection.rejection_counts == {"geometry_outlier": 1}


def test_video_split_is_deterministic_and_group_isolated() -> None:
    asset_ids = ("video1", "video2", "video3", "video4", "video5")

    first = assign_video_splits(asset_ids, seed=20260810, valid_fraction=0.2)
    second = assign_video_splits(tuple(reversed(asset_ids)), seed=20260810, valid_fraction=0.2)

    assert first == second
    assert set(first) == set(asset_ids)
    assert list(first.values()).count("valid") == 1
    assert list(first.values()).count("train") == 4


def _write_image(path: Path) -> None:
    image = np.full((64, 64, 3), (20, 100, 50), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_mask(path: Path, *, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (15 + offset, 15), 5, 255, -1)
    assert cv2.imwrite(str(path), mask)


def test_publication_writes_video_disjoint_rfdetr_coco_and_frozen_provenance(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "raw"
    images_root = tmp_path / "frames"
    payload_root.mkdir()
    images_root.mkdir()
    frames: list[BootstrapFrame] = []
    frame_paths: dict[tuple[str, int], Path] = {}
    for offset, asset_id in enumerate(("video1", "video2")):
        instance_id = f"ball-{asset_id}"
        frame_index = offset * 15
        image_path = images_root / f"{asset_id}.jpg"
        _write_image(image_path)
        frame_paths[(asset_id, frame_index)] = image_path
        mask_ref = f"masks/{instance_id}/{frame_index:08d}.png"
        _write_mask(payload_root / mask_ref, offset=offset * 20)
        frames.append(
            BootstrapFrame(
                asset_id=asset_id,
                shot_index=0,
                frame_index=frame_index,
                observations=(
                    _observation(
                        asset_id=asset_id,
                        frame_index=frame_index,
                        instance_id=instance_id,
                        box=(10.0 + offset * 20, 10.0, 21.0 + offset * 20, 21.0),
                        area=81,
                    ).model_copy(update={"mask_ref": mask_ref}),
                ),
            )
        )

    published = publish_bootstrap_dataset(
        selection=BootstrapSelection(frames=tuple(frames), rejection_counts={"qa_flagged": 3}),
        splits={"video1": "train", "video2": "valid"},
        frame_paths=frame_paths,
        payload_root=payload_root,
        source_provenance={
            "raw_run_hash": "a" * 64,
            "input_bundle_id": "b" * 64,
            "evaluation_manifest_hash": "c" * 64,
            "verdict_artifact_hash": "d" * 64,
        },
        output_root=tmp_path / "outputs",
    )

    train = json.loads((published.path / "train" / "_annotations.coco.json").read_text())
    valid = json.loads((published.path / "valid" / "_annotations.coco.json").read_text())
    manifest = json.loads((published.path / "manifest.json").read_text())
    assert [item["source_asset_id"] for item in train["images"]] == ["video1"]
    assert [item["source_asset_id"] for item in valid["images"]] == ["video2"]
    assert len(train["annotations"][0]["segmentation"][0]) >= 6
    assert manifest["artifact_id"] == published.artifact_id == published.path.name
    assert manifest["policy"] == {
        "benchmark_frames_excluded": True,
        "frame_complete_labels_required": True,
        "qa_flags_allowed": False,
        "scientific_use": "development_bootstrap_not_independent_accuracy",
    }
    assert manifest["rejection_counts"] == {"qa_flagged": 3}
    assert published.image_count == 2
    assert published.annotation_count == 2


def test_bootstrap_gate_requires_the_exact_passed_human_verdict_binding() -> None:
    binding = {
        "raw_run_hash": "a" * 64,
        "input_bundle_id": "b" * 64,
        "dataset_id": "c" * 64,
        "prompt_manifest_id": "d" * 64,
        "evaluation_manifest_hash": "e" * 64,
        "evaluation_artifact_hash": "f" * 64,
        "review_sample_hash": "1" * 64,
    }
    verdict = {
        **binding,
        "pilot_status": "passed",
        "summary": {"status": "human_gates_passed"},
    }

    assert verify_bootstrap_gate(expected=binding, verdict_payload=verdict) == binding

    verdict["pilot_status"] = "failed"
    try:
        verify_bootstrap_gate(expected=binding, verdict_payload=verdict)
    except ValueError as exc:
        assert "human-approved" in str(exc)
    else:  # pragma: no cover - the failed gate must never become training data.
        raise AssertionError("failed human gate was accepted")
