from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from billiards_ai.lab.propagation import (
    PilotQaConfig,
    PilotQaIdentity,
    Sam2WindowInput,
    sam2_worker_input,
)
from billiards_ai.lab.propagation.frame_mapping import (
    FrameExtractionError,
    extract_shot_frames,
    materialize_window_frames,
)
from billiards_ai.lab.propagation.pilot_data import ShotDefinition


def _numbered_video(path: Path, *, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), (index * 20, 30, 240 - index * 20), dtype=np.uint8)
        cv2.putText(
            frame,
            str(index),
            (4, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def test_extracts_exact_half_open_source_frames_and_materializes_window_only(
    tmp_path: Path,
) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    shot = ShotDefinition(video_id="synthetic", shot_index=3, start_frame=2, end_frame=6)

    extracted = extract_shot_frames(
        video_path=video,
        shot=shot,
        output_dir=tmp_path / "artifacts",
    )
    window = materialize_window_frames(
        shot=extracted,
        start_frame=3,
        end_frame=5,
        output_dir=tmp_path / "artifacts",
    )

    assert [(frame.source_frame_index, frame.shot_local_index) for frame in extracted.frames] == [
        (2, 0),
        (3, 1),
        (4, 2),
        (5, 3),
    ]
    assert [
        (frame.source_frame_index, frame.shot_local_index, frame.sam2_local_index)
        for frame in window.frames
    ] == [
        (3, 1, 0),
        (4, 2, 1),
    ]
    assert {path.name for path in extracted.frame_directory.iterdir()} == {
        "00000002.jpg",
        "00000003.jpg",
        "00000004.jpg",
        "00000005.jpg",
    }
    assert {path.name for path in window.frame_directory.iterdir()} == {"00000.jpg", "00001.jpg"}
    assert window.frame_directory != extracted.frame_directory
    assert not (window.frame_directory / "00000002.jpg").exists()
    assert (
        extracted.source_metadata.display_width,
        extracted.source_metadata.display_height,
    ) == (64, 48)
    for frame in window.frames:
        payload = window.frame_directory / f"{frame.sam2_local_index:05d}.jpg"
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == frame.jpeg_sha256
    worker_input = sam2_worker_input(window)
    assert worker_input.frame_directory == window.frame_directory
    assert worker_input.qa_identity.artifact_kind == "worker_job"
    assert worker_input.qa_identity.pilot_qa_config_hash == PilotQaConfig().canonical_hash
    with pytest.raises(ValidationError, match="worker-job QA identity"):
        Sam2WindowInput(window=window, qa_identity=PilotQaIdentity(artifact_kind="raw_run"))
    with pytest.raises(TypeError, match="ExtractedWindow"):
        sam2_worker_input(extracted)


def test_refuses_to_materialize_a_window_when_the_verified_shot_cache_changes(
    tmp_path: Path,
) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    extracted = extract_shot_frames(
        video_path=video,
        shot=ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4),
        output_dir=tmp_path / "artifacts",
    )
    payload = extracted.frame_directory / "00000002.jpg"
    payload.chmod(payload.stat().st_mode | stat.S_IWUSR)
    payload.write_bytes(b"changed")

    with pytest.raises(FrameExtractionError, match="changed"):
        materialize_window_frames(
            shot=extracted,
            start_frame=1,
            end_frame=3,
            output_dir=tmp_path / "artifacts",
        )


def test_refuses_to_materialize_a_window_when_the_source_video_changes(tmp_path: Path) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    extracted = extract_shot_frames(
        video_path=video,
        shot=ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4),
        output_dir=tmp_path / "artifacts",
    )
    video.write_bytes(b"not the frozen source video")

    with pytest.raises(FrameExtractionError, match="source video changed"):
        materialize_window_frames(
            shot=extracted,
            start_frame=1,
            end_frame=3,
            output_dir=tmp_path / "artifacts",
        )


def test_reuses_a_content_addressed_window_directory_and_rejects_hash_drift(
    tmp_path: Path,
) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    extracted = extract_shot_frames(
        video_path=video,
        shot=ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4),
        output_dir=tmp_path / "artifacts",
    )
    first = materialize_window_frames(
        shot=extracted,
        start_frame=1,
        end_frame=3,
        output_dir=tmp_path / "artifacts",
    )
    second = materialize_window_frames(
        shot=extracted,
        start_frame=1,
        end_frame=3,
        output_dir=tmp_path / "artifacts",
    )

    assert first.frame_directory == second.frame_directory
    assert first.frame_directory.parent.name != "artifacts"
    assert first.frame_directory.parent.parent.parent == tmp_path / "artifacts"
    payload = first.frame_directory / "00000.jpg"
    payload.chmod(payload.stat().st_mode | stat.S_IWUSR)
    payload.write_bytes(b"changed")

    with pytest.raises(FrameExtractionError, match="changed"):
        materialize_window_frames(
            shot=extracted,
            start_frame=1,
            end_frame=3,
            output_dir=tmp_path / "artifacts",
        )


def test_rejects_an_output_directory_nested_under_a_symbolic_link(tmp_path: Path) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    target = tmp_path / "target"
    nested = target / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises(FrameExtractionError, match="unsafe output directory"):
        extract_shot_frames(
            video_path=video,
            shot=ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4),
            output_dir=linked / "nested" / "shot-cache",
        )


def test_reuses_a_content_addressed_shot_directory_under_the_output_root(tmp_path: Path) -> None:
    video = tmp_path / "numbered.mp4"
    _numbered_video(video)
    root = tmp_path / "artifacts"
    shot = ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4)

    first = extract_shot_frames(video_path=video, shot=shot, output_dir=root)
    second = extract_shot_frames(video_path=video, shot=shot, output_dir=root)

    assert first.frame_directory == second.frame_directory
    assert first.frame_directory.parent.name != root.name
    assert first.frame_directory.parent.parent.parent == root
    assert not (root / "00000001.jpg").exists()
    assert not first.frame_directory.stat().st_mode & stat.S_IWUSR


def test_rejects_a_nested_symbolic_link_in_the_source_video_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    video = nested / "numbered.mp4"
    _numbered_video(video)
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)

    with pytest.raises(FrameExtractionError, match="unsafe source video"):
        extract_shot_frames(
            video_path=linked / "nested" / "numbered.mp4",
            shot=ShotDefinition(video_id="synthetic", shot_index=0, start_frame=1, end_frame=4),
            output_dir=tmp_path / "artifacts",
        )
