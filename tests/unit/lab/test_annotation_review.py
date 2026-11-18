from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.annotation_review import (
    AnnotationCircle,
    FrameDraft,
    ReviewConflictError,
    ReviewStore,
    build_review_session,
    export_reviewed_dataset,
    migrate_reviewed_drafts,
)
from billiards_ai.lab.annotation_review_page import render_annotation_page


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    review_root = tmp_path / "review"
    sparse_run = tmp_path / "sparse"
    sparse_run.mkdir()
    rows = []
    for index in range(3):
        frame = (
            review_root
            / "videos"
            / f"video{index + 1}"
            / "review-frames"
            / f"shot-000-middle-frame-{index * 10:06d}.jpg"
        )
        frame.parent.mkdir(parents=True)
        assert cv2.imwrite(str(frame), np.zeros((64, 96, 3), dtype=np.uint8))
        if index < 2:
            rows.append(
                {
                    "detection_id": f"detection-{index}",
                    "source_frame": {
                        "path": frame.relative_to(review_root).as_posix(),
                        "bytes": frame.stat().st_size,
                        "sha256": _sha256(frame),
                    },
                    "confidence": 0.8 if index == 0 else 0.2,
                    "box_xyxy_px": [40.0, 25.0, 50.0, 35.0],
                    "teacher_id": "rfdetr-seg-small-coco",
                    "teacher_weight_sha256": "a" * 64,
                    "provenance": "pseudo",
                    "review_status": "unreviewed",
                }
            )
    (sparse_run / "summary.json").write_text(
        json.dumps({"run_id": "b" * 64, "accuracy_claim": "none"}), encoding="utf-8"
    )
    (sparse_run / "detections.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return review_root, sparse_run


def test_session_selects_high_confidence_and_zero_candidate_frames(tmp_path: Path) -> None:
    review_root, sparse_run = _fixture(tmp_path)

    first = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=1,
    )
    second = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=1,
    )

    assert first.path == second.path
    assert first.frame_count == 2
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert [frame["queue_reason"] for frame in manifest["frames"]] == [
        "high_confidence",
        "no_candidate",
    ]
    assert len(manifest["frames"][0]["initial_annotations"]) == 1


def test_session_all_mode_selects_every_review_frame(tmp_path: Path) -> None:
    review_root, sparse_run = _fixture(tmp_path)

    session = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=0,
        selection_mode="all",
    )

    manifest = json.loads((session.path / "manifest.json").read_text())
    assert [frame["queue_reason"] for frame in manifest["frames"]] == [
        "high_confidence",
        "low_confidence",
        "no_candidate",
    ]


def test_draft_save_is_versioned_and_stale_tab_cannot_overwrite(tmp_path: Path) -> None:
    review_root, sparse_run = _fixture(tmp_path)
    session = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=0,
    )
    store = ReviewStore(session.path)
    frame_id = store.manifest.frames[0].frame_id
    draft = store.load_draft(frame_id)
    changed = FrameDraft(
        frame_id=frame_id,
        revision=draft.revision,
        annotations=(
            AnnotationCircle(
                annotation_id="human-ball-1",
                center_x_px=30.0,
                center_y_px=20.0,
                radius_px=5.0,
                origin="human_added",
            ),
        ),
        whole_frame_complete=True,
        difficult=False,
    )

    saved = store.save_draft(changed, expected_revision=0, actor="Boyu")

    assert saved.revision == 1
    assert store.load_draft(frame_id).annotations[0].center_x_px == 30.0
    with pytest.raises(ReviewConflictError):
        store.save_draft(changed, expected_revision=0, actor="Boyu")


def test_reviewed_drafts_migrate_to_expanded_session(tmp_path: Path) -> None:
    review_root, sparse_run = _fixture(tmp_path)
    priority = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "priority",
        high_confidence=0.3,
        zero_candidate_limit=0,
    )
    source = ReviewStore(priority.path)
    source_frame = source.manifest.frames[0]
    source.save_draft(
        source.load_draft(source_frame.frame_id).model_copy(update={"whole_frame_complete": True}),
        expected_revision=0,
        actor="Boyu",
    )
    expanded = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "expanded",
        high_confidence=0.3,
        zero_candidate_limit=0,
        selection_mode="all",
    )
    target = ReviewStore(expanded.path)

    migrated = migrate_reviewed_drafts(source=source, target=target, actor="Boyu")

    assert migrated == 1
    carried = target.load_draft(source_frame.frame_id)
    assert carried.whole_frame_complete is True
    assert carried.annotations == source.load_draft(source_frame.frame_id).annotations


def test_export_includes_only_complete_non_difficult_frames(tmp_path: Path) -> None:
    review_root, sparse_run = _fixture(tmp_path)
    session = build_review_session(
        review_root=review_root,
        sparse_run=sparse_run,
        output_root=tmp_path / "sessions",
        high_confidence=0.3,
        zero_candidate_limit=1,
    )
    store = ReviewStore(session.path)
    first, second = store.manifest.frames
    first_draft = store.load_draft(first.frame_id)
    store.save_draft(
        first_draft.model_copy(update={"whole_frame_complete": True}),
        expected_revision=0,
        actor="Boyu",
    )
    second_draft = store.load_draft(second.frame_id)
    store.save_draft(
        second_draft.model_copy(update={"difficult": True}),
        expected_revision=0,
        actor="Boyu",
    )

    exported = export_reviewed_dataset(
        store=store,
        review_root=review_root,
        output_root=tmp_path / "exports",
    )

    assert exported.included_frame_count == 1
    assert exported.excluded_frame_count == 1
    coco = json.loads((exported.path / "annotations.json").read_text())
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["segmentation"]


def test_review_page_exposes_required_annotation_controls() -> None:
    page = render_annotation_page()

    assert 'id="review-canvas"' in page
    assert 'id="complete-frame"' in page
    assert 'id="save-next"' in page
    assert 'id="export-labels"' in page
    assert "新增漏检球" in page
    assert "低置信候选" in page
    assert "/api/session" in page
