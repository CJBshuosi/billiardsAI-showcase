from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.propagation import (
    InstanceAbsence,
    InstancePrompt,
    MaskObservation,
    PropagationResult,
    PropagationWindow,
)
from billiards_ai.lab.propagation.frame_mapping import (
    extract_shot_frames,
    materialize_window_frames,
)
from billiards_ai.lab.propagation.pilot_data import ShotDefinition
from billiards_ai.lab.propagation.raw_overlays import render_raw_overlays


def _prepared_window(tmp_path: Path):
    source = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 12))
    assert writer.isOpened()
    for value in range(6):
        writer.write(np.full((12, 16, 3), value * 20, dtype=np.uint8))
    writer.release()
    shot = extract_shot_frames(
        video_path=source,
        shot=ShotDefinition(video_id="synthetic-video", shot_index=2, start_frame=4, end_frame=6),
        output_dir=tmp_path / "prepared",
    )
    return materialize_window_frames(
        shot=shot, start_frame=4, end_frame=6, output_dir=tmp_path / "prepared"
    )


def _result() -> PropagationResult:
    window = PropagationWindow(asset_id="synthetic-video", ordinal=2, start_frame=4, end_frame=6)
    prompt = InstancePrompt(
        asset_id="synthetic-video",
        frame_index=4,
        instance_id="ball-a",
        class_name="ball",
        source_kind="gold",
        source_id="seed-a",
        box_xyxy_px=(1.0, 1.0, 5.0, 5.0),
    )
    observation = MaskObservation(
        asset_id="synthetic-video",
        frame_index=4,
        instance_id="ball-a",
        class_name="ball",
        prompt_frame_index=4,
        center_px=(3.0, 3.0),
        box_xyxy_px=(1.0, 1.0, 5.0, 5.0),
        area_px=16,
        mask_ref="masks/ball-a/00000004.png",
        qa_flags=("center_jump",),
    )
    absence = InstanceAbsence(
        asset_id="synthetic-video",
        frame_index=5,
        instance_id="ball-a",
        class_name="ball",
        prompt_frame_index=4,
        reason="empty_mask",
        qa_flags=("empty_mask",),
    )
    return PropagationResult(
        backend_id="synthetic",
        backend_version="1.0",
        window=window,
        prompts=(prompt,),
        observations=(observation,),
        absences=(absence,),
    )


def test_raw_overlays_render_only_inference_evidence_for_synthetic_prepared_frames(
    tmp_path: Path,
) -> None:
    masks = tmp_path / "staging" / "masks" / "ball-a"
    masks.mkdir(parents=True)
    prepared = _prepared_window(tmp_path)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    assert cv2.imwrite(str(masks / "00000004.png"), mask)

    overlays = render_raw_overlays(
        prepared_windows=(prepared,),
        results=(_result(),),
        payload_root=tmp_path / "staging",
        output_root=tmp_path / "staging" / "overlays",
    )

    assert [item.path for item in overlays] == [
        "overlays/synthetic-video/00000004.png",
        "overlays/synthetic-video/00000005.png",
    ]
    assert all(item.bytes > 0 and len(item.sha256) == 64 for item in overlays)
    assert (tmp_path / "staging" / overlays[0].path).read_bytes() != (
        prepared.frame_directory / "00000.jpg"
    ).read_bytes()
    evidence = [item.evidence.model_dump(mode="json") for item in overlays]
    assert evidence == [
        {
            "asset_id": "synthetic-video",
            "frame_index": 4,
            "instance_ids": ["ball-a"],
            "qa_flags": ["center_jump"],
            "trajectory": [[3.0, 3.0]],
            "window": {"end_frame": 6, "ordinal": 2, "start_frame": 4},
        },
        {
            "asset_id": "synthetic-video",
            "frame_index": 5,
            "instance_ids": ["ball-a"],
            "qa_flags": ["empty_mask"],
            "trajectory": [[3.0, 3.0]],
            "window": {"end_frame": 6, "ordinal": 2, "start_frame": 4},
        },
    ]
    assert not {
        "evaluation_manifest",
        "held_out",
        "labels",
        "metrics",
    } & set().union(*(set(item) for item in evidence))


def test_raw_overlays_reject_unverified_or_non_contract_payloads(tmp_path: Path) -> None:
    (tmp_path / "staging").mkdir()
    with pytest.raises(ValueError, match="prepared window"):
        render_raw_overlays(
            prepared_windows=(),
            results=(_result(),),
            payload_root=tmp_path / "staging",
            output_root=tmp_path / "staging" / "overlays",
        )


def test_raw_overlays_never_render_promoted_observations(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    mask = staging / "masks" / "ball-a" / "00000004.png"
    mask.parent.mkdir(parents=True)
    assert cv2.imwrite(str(mask), np.full((12, 16), 255, dtype=np.uint8))
    promoted = _result().model_copy(
        update={
            "observations": (
                _result().observations[0].model_copy(update={"review_status": "approved"}),
            )
        }
    )

    with pytest.raises(ValueError, match="pseudo/unreviewed"):
        render_raw_overlays(
            prepared_windows=(_prepared_window(tmp_path),),
            results=(promoted,),
            payload_root=staging,
            output_root=staging / "overlays",
        )


@pytest.mark.parametrize("asset_id", ["../../outside", "linked", "C:outside"])
def test_raw_overlays_reject_unsafe_asset_destinations_without_touching_sentinel(
    tmp_path: Path, asset_id: str
) -> None:
    staging = tmp_path / "staging"
    mask = staging / "masks" / "ball-a" / "00000004.png"
    mask.parent.mkdir(parents=True)
    assert cv2.imwrite(str(mask), np.full((12, 16), 255, dtype=np.uint8))
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "00000004.png"
    sentinel.write_bytes(b"do-not-overwrite")
    if asset_id == "linked":
        (staging / "overlays").mkdir()
        (staging / "overlays" / "linked").symlink_to(outside, target_is_directory=True)
    prepared = _prepared_window(tmp_path)
    unsafe_prepared = prepared.model_copy(
        update={"shot": prepared.shot.model_copy(update={"video_id": asset_id})}
    )
    unsafe_result = _result().model_copy(
        update={"window": _result().window.model_copy(update={"asset_id": asset_id})}
    )

    with pytest.raises(ValueError, match="asset|symlink"):
        render_raw_overlays(
            prepared_windows=(unsafe_prepared,),
            results=(unsafe_result,),
            payload_root=staging,
            output_root=staging / "overlays",
        )
    assert sentinel.read_bytes() == b"do-not-overwrite"
