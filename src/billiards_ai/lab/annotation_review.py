"""Minimal local circle-annotation sessions and training export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

import cv2
from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256


class ReviewError(RuntimeError):
    """A review input or operation is invalid."""


class ReviewConflictError(ReviewError):
    """A stale browser revision attempted to overwrite a newer draft."""


class AnnotationCircle(FrozenContract):
    annotation_id: NonEmptyStr
    center_x_px: float = Field(ge=0)
    center_y_px: float = Field(ge=0)
    radius_px: float = Field(gt=0)
    origin: Literal["teacher", "teacher_modified", "human_added"]
    source_detection_id: str | None = None

    @model_validator(mode="after")
    def values_are_finite(self) -> AnnotationCircle:
        if not all(
            math.isfinite(value) for value in (self.center_x_px, self.center_y_px, self.radius_px)
        ):
            raise ValueError("circle geometry must be finite")
        return self


class ReviewFrame(FrozenContract):
    frame_id: Sha256
    source_path: NonEmptyStr
    source_sha256: Sha256
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    video_id: NonEmptyStr
    frame_index: int = Field(ge=0)
    queue_reason: Literal["high_confidence", "low_confidence", "no_candidate"]
    initial_annotations: tuple[AnnotationCircle, ...] = ()


class ReviewManifest(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    session_id: Sha256
    sparse_run_id: Sha256
    high_confidence: float = Field(gt=0, le=1)
    zero_candidate_limit: int = Field(ge=0)
    selection_mode: Literal["priority", "all"] = "priority"
    frames: tuple[ReviewFrame, ...] = Field(min_length=1)


class FrameDraft(FrozenContract):
    frame_id: Sha256
    revision: int = Field(ge=0)
    annotations: tuple[AnnotationCircle, ...] = ()
    whole_frame_complete: bool = False
    difficult: bool = False

    @model_validator(mode="after")
    def status_is_unambiguous(self) -> FrameDraft:
        if self.whole_frame_complete and self.difficult:
            raise ValueError("a difficult frame cannot be complete")
        ids = [item.annotation_id for item in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("annotation IDs must be unique within a frame")
        return self


class ReviewSession(FrozenContract):
    session_id: Sha256
    path: Path
    frame_count: int = Field(gt=0)
    reused: bool = False


class ReviewExport(FrozenContract):
    export_id: Sha256
    path: Path
    included_frame_count: int = Field(ge=0)
    excluded_frame_count: int = Field(ge=0)
    annotation_count: int = Field(ge=0)
    reused: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ReviewError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _safe_source(review_root: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ReviewError("review source path must be confined")
    candidate = (review_root / path).resolve()
    if not candidate.is_relative_to(review_root.resolve()) or not candidate.is_file():
        raise ReviewError(f"review source is missing or unsafe: {relative_path}")
    return candidate


def _frame_index(path: str) -> int:
    stem = PurePosixPath(path).stem
    try:
        return int(stem.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ReviewError(f"review frame name lacks a numeric frame index: {path}") from exc


def _initial_circle(row: dict[str, object]) -> AnnotationCircle:
    box_value = row.get("box_xyxy_px")
    if not isinstance(box_value, list) or len(box_value) != 4:
        raise ReviewError("teacher detection requires box_xyxy_px")
    left, top, right, bottom = (float(value) for value in box_value)
    detection_id = str(row.get("detection_id") or "")
    return AnnotationCircle(
        annotation_id=detection_id,
        center_x_px=(left + right) / 2,
        center_y_px=(top + bottom) / 2,
        radius_px=max(right - left, bottom - top) / 2,
        origin="teacher",
        source_detection_id=detection_id,
    )


def build_review_session(
    *,
    review_root: Path,
    sparse_run: Path,
    output_root: Path,
    high_confidence: float = 0.3,
    zero_candidate_limit: int = 10,
    selection_mode: Literal["priority", "all"] = "priority",
) -> ReviewSession:
    """Freeze a small queue from high-confidence and no-candidate review frames."""

    if not 0 < high_confidence <= 1 or zero_candidate_limit < 0:
        raise ValueError("invalid queue thresholds")
    summary = json.loads((sparse_run / "summary.json").read_text(encoding="utf-8"))
    sparse_run_id = str(summary.get("run_id") or "")
    if len(sparse_run_id) != 64:
        raise ReviewError("sparse summary lacks a valid run ID")
    rows = _load_jsonl(sparse_run / "detections.jsonl")
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        source = row.get("source_frame")
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ReviewError("detection lacks source-frame evidence")
        by_path[source["path"]].append(row)

    all_images = tuple(sorted(review_root.rglob("*.jpg")))
    relative_images = tuple(path.relative_to(review_root).as_posix() for path in all_images)
    high_paths = sorted(
        path
        for path, detections in by_path.items()
        if any(float(row.get("confidence", 0.0)) >= high_confidence for row in detections)
    )
    zero_paths = [path for path in relative_images if path not in by_path][:zero_candidate_limit]
    if selection_mode == "all":
        high_path_set = set(high_paths)
        selected = [
            (
                path,
                "high_confidence"
                if path in high_path_set
                else "low_confidence"
                if path in by_path
                else "no_candidate",
            )
            for path in relative_images
        ]
    else:
        selected = [(path, "high_confidence") for path in high_paths]
        selected.extend((path, "no_candidate") for path in zero_paths)
    if not selected:
        raise ReviewError("review queue is empty")

    frames = []
    for relative_path, reason in selected:
        source_path = _safe_source(review_root, relative_path)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ReviewError(f"review source cannot be decoded: {relative_path}")
        source_sha256 = _sha256(source_path)
        for row in by_path.get(relative_path, []):
            source = row["source_frame"]
            if isinstance(source, dict) and source.get("sha256") != source_sha256:
                raise ReviewError(f"source hash mismatch: {relative_path}")
        parts = PurePosixPath(relative_path).parts
        video_id = parts[1] if len(parts) > 1 and parts[0] == "videos" else parts[0]
        frames.append(
            ReviewFrame(
                frame_id=_digest({"path": relative_path, "sha256": source_sha256}),
                source_path=relative_path,
                source_sha256=source_sha256,
                width_px=int(image.shape[1]),
                height_px=int(image.shape[0]),
                video_id=video_id,
                frame_index=_frame_index(relative_path),
                queue_reason=reason,  # type: ignore[arg-type]
                initial_annotations=tuple(
                    _initial_circle(row) for row in by_path.get(relative_path, [])
                ),
            )
        )
    manifest_payload = {
        "schema_version": "1.0",
        "sparse_run_id": sparse_run_id,
        "high_confidence": high_confidence,
        "zero_candidate_limit": zero_candidate_limit,
        "selection_mode": selection_mode,
        "frames": [frame.model_dump(mode="json") for frame in frames],
    }
    session_id = _digest(manifest_payload)
    manifest = ReviewManifest(session_id=session_id, **manifest_payload)
    final_path = output_root / f"session-{session_id}"
    if (final_path / "manifest.json").is_file():
        return ReviewSession(
            session_id=session_id,
            path=final_path,
            frame_count=len(frames),
            reused=True,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f"session-{session_id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    (temporary / "drafts").mkdir()
    (temporary / "events.jsonl").touch()
    _write_json_atomic(temporary / "manifest.json", manifest.model_dump(mode="json"))
    os.replace(temporary, final_path)
    return ReviewSession(session_id=session_id, path=final_path, frame_count=len(frames))


class ReviewStore:
    def __init__(self, session_path: Path) -> None:
        self.session_path = session_path
        self.manifest = ReviewManifest.model_validate_json(
            (session_path / "manifest.json").read_text(encoding="utf-8")
        )
        self._frames = {frame.frame_id: frame for frame in self.manifest.frames}

    def _draft_path(self, frame_id: str) -> Path:
        if frame_id not in self._frames:
            raise ReviewError("unknown review frame")
        return self.session_path / "drafts" / f"{frame_id}.json"

    def load_draft(self, frame_id: str) -> FrameDraft:
        path = self._draft_path(frame_id)
        if path.is_file():
            return FrameDraft.model_validate_json(path.read_text(encoding="utf-8"))
        frame = self._frames[frame_id]
        return FrameDraft(
            frame_id=frame_id,
            revision=0,
            annotations=frame.initial_annotations,
        )

    def save_draft(self, draft: FrameDraft, *, expected_revision: int, actor: str) -> FrameDraft:
        if not actor.strip():
            raise ValueError("review actor must be non-empty")
        current = self.load_draft(draft.frame_id)
        if current.revision != expected_revision or draft.revision != expected_revision:
            raise ReviewConflictError("stale frame revision")
        frame = self._frames[draft.frame_id]
        for circle in draft.annotations:
            if (
                circle.center_x_px - circle.radius_px < 0
                or circle.center_y_px - circle.radius_px < 0
                or circle.center_x_px + circle.radius_px > frame.width_px
                or circle.center_y_px + circle.radius_px > frame.height_px
            ):
                raise ReviewError("circle must remain inside the source image")
        saved = draft.model_copy(update={"revision": expected_revision + 1})
        event = {
            "session_id": self.manifest.session_id,
            "frame_id": draft.frame_id,
            "previous_revision": expected_revision,
            "revision": saved.revision,
            "actor": actor.strip(),
            "draft_sha256": _digest(saved.model_dump(mode="json")),
        }
        with (self.session_path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _write_json_atomic(self._draft_path(draft.frame_id), saved.model_dump(mode="json"))
        return saved


def migrate_reviewed_drafts(*, source: ReviewStore, target: ReviewStore, actor: str) -> int:
    """Carry persisted human drafts into a larger content-addressed review session."""

    target_frame_ids = {frame.frame_id for frame in target.manifest.frames}
    migrated = 0
    for frame in source.manifest.frames:
        source_path = source.session_path / "drafts" / f"{frame.frame_id}.json"
        if not source_path.is_file() or frame.frame_id not in target_frame_ids:
            continue
        target_path = target.session_path / "drafts" / f"{frame.frame_id}.json"
        if target_path.exists():
            raise ReviewConflictError("expanded session already contains a migrated draft")
        carried = source.load_draft(frame.frame_id).model_copy(update={"revision": 0})
        target.save_draft(
            carried,
            expected_revision=0,
            actor=f"{actor.strip()} migrated from {source.manifest.session_id}",
        )
        migrated += 1
    return migrated


def _circle_polygon(circle: AnnotationCircle, points: int = 24) -> list[float]:
    polygon = []
    for index in range(points):
        angle = 2 * math.pi * index / points
        polygon.extend(
            [
                circle.center_x_px + circle.radius_px * math.cos(angle),
                circle.center_y_px + circle.radius_px * math.sin(angle),
            ]
        )
    return polygon


def export_reviewed_dataset(
    *, store: ReviewStore, review_root: Path, output_root: Path
) -> ReviewExport:
    """Export only complete, non-difficult drafts as a small COCO segmentation set."""

    included = []
    excluded = []
    for frame in store.manifest.frames:
        draft = store.load_draft(frame.frame_id)
        if draft.whole_frame_complete and not draft.difficult:
            included.append((frame, draft))
        else:
            excluded.append(frame.frame_id)
    if not included:
        raise ReviewError("no complete reviewed frames are eligible for export")
    export_payload = {
        "session_id": store.manifest.session_id,
        "included": [
            {
                "frame_id": frame.frame_id,
                "revision": draft.revision,
                "draft": draft.model_dump(mode="json"),
            }
            for frame, draft in included
        ],
    }
    export_id = _digest(export_payload)
    final_path = output_root / f"export-{export_id}"
    summary_path = final_path / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return ReviewExport(path=final_path, reused=True, **payload)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f"export-{export_id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    images_directory = temporary / "images"
    images_directory.mkdir()
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_id, (frame, draft) in enumerate(included, start=1):
        source = _safe_source(review_root, frame.source_path)
        if _sha256(source) != frame.source_sha256:
            raise ReviewError(f"source hash changed before export: {frame.source_path}")
        filename = f"{frame.frame_id}.jpg"
        shutil.copy2(source, images_directory / filename)
        coco_images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": frame.width_px,
                "height": frame.height_px,
                "source_video": frame.video_id,
                "source_frame_index": frame.frame_index,
            }
        )
        for circle in draft.annotations:
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        circle.center_x_px - circle.radius_px,
                        circle.center_y_px - circle.radius_px,
                        circle.radius_px * 2,
                        circle.radius_px * 2,
                    ],
                    "area": math.pi * circle.radius_px**2,
                    "segmentation": [_circle_polygon(circle)],
                    "iscrowd": 0,
                    "annotation_id": circle.annotation_id,
                    "origin": circle.origin,
                }
            )
            annotation_id += 1
    _write_json_atomic(
        temporary / "annotations.json",
        {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "ball"}],
        },
    )
    summary = ReviewExport(
        export_id=export_id,
        path=final_path,
        included_frame_count=len(included),
        excluded_frame_count=len(excluded),
        annotation_count=len(coco_annotations),
    )
    _write_json_atomic(
        temporary / "summary.json",
        summary.model_dump(mode="json", exclude={"path", "reused"}),
    )
    os.replace(temporary, final_path)
    return summary
