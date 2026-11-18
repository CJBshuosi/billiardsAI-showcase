"""Content-addressed, fail-isolated bootstrap processing for approved videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np
from pydantic import Field, model_validator

from billiards_ai.lab.contracts import FrozenContract, Sha256
from billiards_ai.lab.external_dataset import (
    canonical_external_bundle_sha256,
    load_external_descriptor,
    verify_external_dataset,
)
from billiards_ai.lab.motion import (
    MotionConfig,
    MotionInputFrame,
    motion_series,
    segment_shots,
)
from billiards_ai.lab.pseudo_labels import (
    FrameGeometry,
    InstanceCandidate,
    MotionContext,
    ReviewStrata,
    fuse_candidates,
    select_review_items,
)
from billiards_ai.lab.video_probe import FfprobeRunner, VideoProbe, probe_video


class BootstrapError(RuntimeError):
    """The batch contract or its atomic output is invalid."""


class CandidateDetector(Protocol):
    def detect(self, frames: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]: ...


@dataclass(frozen=True)
class TeacherBinding:
    teacher_id: str
    teacher_version: str
    teacher_sha256: str
    detector: CandidateDetector = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.teacher_id.strip() or not self.teacher_version.strip():
            raise ValueError("teacher identity and version must be non-empty")
        if len(self.teacher_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.teacher_sha256
        ):
            raise ValueError("teacher sha256 must be lowercase hexadecimal")


class BootstrapConfig(FrozenContract):
    analysis_width_px: int = Field(default=480, ge=64, le=3840)
    frame_stride: int = Field(default=2, ge=1)
    detector_stride: int = Field(default=15, ge=1)
    table_bounds_normalized: tuple[float, float, float, float] = (0.08, 0.08, 0.96, 0.92)
    motion: MotionConfig = MotionConfig(
        max_component_area=200,
        start_threshold=0.12,
        stop_threshold=0.025,
    )
    review: ReviewStrata = ReviewStrata()
    review_limit: int = Field(default=500, ge=0)
    export_review_frames: bool = False

    @model_validator(mode="after")
    def table_bounds_are_normalized(self) -> BootstrapConfig:
        left, top, right, bottom = self.table_bounds_normalized
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("normalized table bounds must be ordered inside [0, 1]")
        return self


@dataclass(frozen=True)
class BootstrapInputs:
    descriptor_path: Path
    dataset_root: Path
    output_root: Path
    code_revision: str
    selected_video_paths: tuple[str, ...]
    config: BootstrapConfig = field(default_factory=BootstrapConfig)
    teachers: tuple[TeacherBinding, ...] = ()
    ffprobe: FfprobeRunner | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.code_revision.strip():
            raise ValueError("code revision must be non-empty")
        if not self.selected_video_paths or len(set(self.selected_video_paths)) != len(
            self.selected_video_paths
        ):
            raise ValueError("selected video paths must be non-empty and unique")
        for value in self.selected_video_paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("selected video paths must be confined relative paths")
        if self.output_root.resolve().is_relative_to(self.dataset_root.resolve()):
            raise ValueError("bootstrap output root must be outside the dataset root")


class BootstrapRun(FrozenContract):
    run_id: Sha256
    path: Path
    expected_video_count: int = Field(ge=1)
    succeeded_video_count: int = Field(ge=0)
    failed_video_count: int = Field(ge=0)
    review_item_count: int = Field(ge=0)
    accuracy_claim: Literal["none"] = "none"
    reused: bool = False


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[union-attr]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(
                json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _probe_payload(probe: VideoProbe) -> dict[str, object]:
    return probe.model_dump(mode="json", exclude={"path"})


def _decode_analysis_frames(
    path: Path, *, width_px: int, stride: int, rotation_degrees: int
) -> tuple[MotionInputFrame, ...]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BootstrapError(f"video cannot be opened: {path}")
    frames: list[MotionInputFrame] = []
    source_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_index % stride:
                source_index += 1
                continue
            if rotation_degrees == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_degrees == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation_degrees == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            height_px = max(1, round(image.shape[0] * width_px / image.shape[1]))
            resized = cv2.resize(image, (width_px, height_px), interpolation=cv2.INTER_AREA)
            frames.append(
                MotionInputFrame(
                    frame_index=source_index,
                    timestamp_ms=float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                    image=resized,
                )
            )
            source_index += 1
    finally:
        capture.release()
    if not frames:
        raise BootstrapError(f"video decoded zero frames: {path}")
    return tuple(frames)


def _table_geometry(
    shape: tuple[int, int], bounds: tuple[float, float, float, float]
) -> tuple[np.ndarray, FrameGeometry]:
    height, width = shape
    left, top, right, bottom = bounds
    box = (
        float(round(left * width)),
        float(round(top * height)),
        float(round(right * width)),
        float(round(bottom * height)),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = 255
    return mask, FrameGeometry(
        width_px=width,
        height_px=height,
        table_bounds_px=box,
        minimum_ball_diameter_px=max(2.0, width * 0.006),
        maximum_ball_diameter_px=max(12.0, width * 0.05),
        fusion_center_distance_px=max(4.0, width * 0.015),
    )


def _teacher_candidates(
    inputs: BootstrapInputs,
    relative_path: str,
    frames: tuple[MotionInputFrame, ...],
    geometry: FrameGeometry,
) -> tuple[InstanceCandidate, ...]:
    frame_by_index = {frame.frame_index: frame for frame in frames}
    selected_frames = tuple(
        frame for index, frame in enumerate(frames) if index % inputs.config.detector_stride == 0
    )
    observations = tuple(
        {
            "frame_id": frame.frame_index,
            "timestamp_ms": round(frame.timestamp_ms),
            "image": frame.image,
        }
        for frame in selected_frames
    )
    candidates: list[InstanceCandidate] = []
    for teacher in inputs.teachers:
        for detection in teacher.detector.detect(observations):
            if not detection.get("supported", True):
                continue
            frame_index = int(detection["frame_id"])
            frame = frame_by_index.get(frame_index)
            center_value = detection.get("center_px")
            radius_value = detection.get("radius_px")
            if frame is None or not isinstance(center_value, (list, tuple)) or radius_value is None:
                continue
            center = (float(center_value[0]), float(center_value[1]))
            radius = float(radius_value)
            mask = detection.get("mask")
            polygon = None
            if isinstance(mask, Mapping) and mask.get("format") == "polygon_px":
                points = mask.get("points")
                if isinstance(points, (list, tuple)):
                    polygon = tuple((float(point[0]), float(point[1])) for point in points)
            candidates.append(
                InstanceCandidate(
                    asset_id=relative_path,
                    frame_index=frame_index,
                    timestamp_ms=round(frame.timestamp_ms),
                    teacher_id=teacher.teacher_id,
                    teacher_version=teacher.teacher_version,
                    teacher_sha256=teacher.teacher_sha256,
                    class_name="ball",
                    confidence=float(detection.get("confidence", 0.0)),
                    center_px=center,
                    box_xyxy_px=(
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    mask_polygon_px=polygon,
                    identity_candidates=tuple(detection.get("identity_candidates", ())),
                    occluded=detection.get("abstention_reason") == "occluded",
                )
            )
    return tuple(candidates)


def _process_video(
    *,
    inputs: BootstrapInputs,
    relative_path: str,
    probe: VideoProbe,
    video_directory: Path,
) -> tuple[int, list[object]]:
    path = inputs.dataset_root / relative_path
    frames = _decode_analysis_frames(
        path,
        width_px=inputs.config.analysis_width_px,
        stride=inputs.config.frame_stride,
        rotation_degrees=probe.rotation_degrees,
    )
    mask, geometry = _table_geometry(
        frames[0].image.shape[:2], inputs.config.table_bounds_normalized
    )
    motion = motion_series(frames, mask, inputs.config.motion)
    shots = segment_shots(motion, inputs.config.motion)
    candidates = _teacher_candidates(inputs, relative_path, frames, geometry)
    motion_by_frame = {item.frame_index: item for item in motion}
    grouped: dict[int, list[InstanceCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.frame_index].append(candidate)
    labels = []
    for frame_index, frame_candidates in sorted(grouped.items()):
        motion_frame = motion_by_frame[frame_index]
        labels.extend(
            fuse_candidates(
                frame_candidates,
                geometry,
                MotionContext(
                    active_ratio=motion_frame.active_pixel_ratio,
                    speed_px_per_s=0.0,
                ),
            )
        )

    video_directory.mkdir(parents=True)
    _write_json(
        video_directory / "qa.json",
        {
            "path": relative_path,
            "probe": _probe_payload(probe),
            "analysis_frame_count": len(frames),
            "audio_policy": "strip-derived-assets" if probe.audio_stream_count else "no-audio",
            "accuracy_claim": "none",
        },
    )
    _write_jsonl(video_directory / "motion.jsonl", motion)
    _write_jsonl(video_directory / "shots.jsonl", shots)
    _write_jsonl(video_directory / "pseudo-labels.jsonl", labels)
    if inputs.config.export_review_frames:
        review_directory = video_directory / "review-frames"
        review_directory.mkdir()
        frame_by_index = {frame.frame_index: frame for frame in frames}
        available_indices = tuple(frame_by_index)
        instructions = []
        for shot_index, shot in enumerate(shots):
            targets = {
                "start": shot.start_frame,
                "middle": (shot.start_frame + shot.end_frame - 1) // 2,
                "end": shot.end_frame - 1,
            }
            evidence = {}
            for role, target in targets.items():
                nearest = min(available_indices, key=lambda index: (abs(index - target), index))
                filename = f"shot-{shot_index:03d}-{role}-frame-{nearest:06d}.jpg"
                if not cv2.imwrite(str(review_directory / filename), frame_by_index[nearest].image):
                    raise BootstrapError(f"failed to write review frame: {filename}")
                evidence[role] = filename
            instructions.append(
                {
                    "shot_index": shot_index,
                    "candidate": shot.model_dump(mode="json"),
                    "evidence_frames": evidence,
                    "review_status": "unreviewed",
                    "accuracy_claim": "none",
                }
            )
        _write_jsonl(video_directory / "shot-review.jsonl", instructions)
    return len(shots), labels


def _existing_run(path: Path) -> BootstrapRun | None:
    aggregate_path = path / "aggregate.json"
    if not aggregate_path.is_file():
        return None
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return BootstrapRun.model_validate(
        {
            "run_id": payload["run_id"],
            "path": path,
            "expected_video_count": payload["expected_video_count"],
            "succeeded_video_count": payload["succeeded_video_count"],
            "failed_video_count": payload["failed_video_count"],
            "review_item_count": payload["review_item_count"],
            "accuracy_claim": payload["accuracy_claim"],
            "reused": True,
        }
    )


def run_bootstrap_batch(inputs: BootstrapInputs) -> BootstrapRun:
    """Verify, probe and process each selected video into one atomic run directory."""

    descriptor = load_external_descriptor(inputs.descriptor_path)
    verified = verify_external_dataset(descriptor, inputs.dataset_root)
    known_videos = {item.path for item in descriptor.files if item.media_kind == "video"}
    unknown = sorted(set(inputs.selected_video_paths) - known_videos)
    if unknown:
        raise BootstrapError(f"selected video is not in verified descriptor: {unknown[0]}")

    probes: dict[str, VideoProbe] = {}
    probe_failures: dict[str, str] = {}
    for relative_path in inputs.selected_video_paths:
        try:
            probes[relative_path] = probe_video(inputs.dataset_root / relative_path, inputs.ffprobe)
        except Exception as exc:  # noqa: BLE001 - per-video typed isolation boundary
            probe_failures[relative_path] = f"{type(exc).__name__}: {exc}"
    run_payload = {
        "code_revision": inputs.code_revision,
        "descriptor_sha256": canonical_external_bundle_sha256(descriptor),
        "verified_file_count": verified.file_count,
        "config": inputs.config.model_dump(mode="json"),
        "teachers": [
            {
                "id": teacher.teacher_id,
                "version": teacher.teacher_version,
                "sha256": teacher.teacher_sha256,
            }
            for teacher in inputs.teachers
        ],
        "probes": {path: _probe_payload(probe) for path, probe in sorted(probes.items())},
        "probe_failures": probe_failures,
        "selected_video_paths": inputs.selected_video_paths,
    }
    run_id = _canonical_sha256(run_payload)
    final_path = inputs.output_root / f"run-{run_id}"
    existing = _existing_run(final_path)
    if existing is not None:
        return existing

    inputs.output_root.mkdir(parents=True, exist_ok=True)
    temporary_path = inputs.output_root / f"run-{run_id}.tmp-{uuid4().hex}"
    temporary_path.mkdir()
    failures = dict(probe_failures)
    all_labels: list[Any] = []
    total_shots = 0
    try:
        for relative_path in inputs.selected_video_paths:
            video_id = PurePosixPath(relative_path).stem
            video_directory = temporary_path / "videos" / video_id
            if relative_path in failures:
                video_directory.mkdir(parents=True)
                _write_json(
                    video_directory / "failure.json",
                    {"path": relative_path, "failure": failures[relative_path]},
                )
                continue
            try:
                shot_count, labels = _process_video(
                    inputs=inputs,
                    relative_path=relative_path,
                    probe=probes[relative_path],
                    video_directory=video_directory,
                )
                total_shots += shot_count
                all_labels.extend(labels)
            except Exception as exc:  # noqa: BLE001 - per-video typed isolation boundary
                failures[relative_path] = f"{type(exc).__name__}: {exc}"
                video_directory.mkdir(parents=True, exist_ok=True)
                _write_json(
                    video_directory / "failure.json",
                    {"path": relative_path, "failure": failures[relative_path]},
                )

        review_items = select_review_items(
            all_labels, limit=inputs.config.review_limit, strata=inputs.config.review
        )
        review_directory = temporary_path / "review"
        review_directory.mkdir()
        _write_jsonl(review_directory / "review-items.jsonl", review_items)
        succeeded = len(inputs.selected_video_paths) - len(failures)
        aggregate = BootstrapRun(
            run_id=run_id,
            path=final_path,
            expected_video_count=len(inputs.selected_video_paths),
            succeeded_video_count=succeeded,
            failed_video_count=len(failures),
            review_item_count=len(review_items),
        )
        aggregate_payload = aggregate.model_dump(mode="json", exclude={"path", "reused"})
        aggregate_payload.update(
            {
                "descriptor_sha256": verified.descriptor_sha256,
                "shot_candidate_count": total_shots,
                "pseudo_label_count": len(all_labels),
                "failures": failures,
            }
        )
        _write_json(temporary_path / "run-inputs.json", run_payload)
        _write_json(temporary_path / "aggregate.json", aggregate_payload)
        os.replace(temporary_path, final_path)
        return aggregate
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify the complete frozen descriptor")
    verify.add_argument("--descriptor", type=Path, required=True)
    verify.add_argument("--dataset-root", type=Path, required=True)
    for command in ("probe", "motion", "review-pack", "aggregate"):
        stage = subparsers.add_parser(command, help=f"materialize and report the {command} stage")
        stage.add_argument("--descriptor", type=Path, required=True)
        stage.add_argument("--dataset-root", type=Path, required=True)
        stage.add_argument("--output-root", type=Path, required=True)
        stage.add_argument("--code-revision", required=True)
        stage.add_argument("--video", action="append", default=[])
        stage.add_argument("--classic-cv", action="store_true")
        stage.add_argument("--analysis-width", type=int, default=480)
        stage.add_argument("--frame-stride", type=int, default=2)
    return parser


def _classic_teacher() -> TeacherBinding:
    from billiards_ai.lab.adapters.classic_cv import ClassicCvAdapter, ClassicDetectorBackend

    source = Path(__file__).parent / "adapters" / "classic_cv.py"
    return TeacherBinding(
        teacher_id="classic-cv",
        teacher_version=ClassicDetectorBackend.version,
        teacher_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        detector=ClassicCvAdapter(
            ClassicDetectorBackend(
                minimum_radius_px=2.0,
                maximum_radius_px=12.0,
                foreground_distance=20.0,
                minimum_circularity=0.65,
            )
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local bootstrap CLI without cloud or training side effects."""

    arguments = _cli_parser().parse_args(argv)
    descriptor = load_external_descriptor(arguments.descriptor)
    if arguments.command == "verify":
        verified = verify_external_dataset(descriptor, arguments.dataset_root)
        print(json.dumps(verified.model_dump(mode="json"), sort_keys=True))
        return 0
    selected = tuple(arguments.video) or tuple(
        item.path for item in descriptor.files if item.media_kind == "video"
    )
    teachers = (_classic_teacher(),) if arguments.classic_cv else ()
    run = run_bootstrap_batch(
        BootstrapInputs(
            descriptor_path=arguments.descriptor,
            dataset_root=arguments.dataset_root,
            output_root=arguments.output_root,
            code_revision=arguments.code_revision,
            selected_video_paths=selected,
            config=BootstrapConfig(
                analysis_width_px=arguments.analysis_width,
                frame_stride=arguments.frame_stride,
                export_review_frames=arguments.command == "review-pack",
            ),
            teachers=teachers,
        )
    )
    print(json.dumps(run.model_dump(mode="json"), sort_keys=True))
    return 0
