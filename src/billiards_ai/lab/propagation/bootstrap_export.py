"""Safe selection and local RF-DETR bootstrap export from verified SAM2 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

import cv2
import numpy as np

from billiards_ai.lab.propagation.contracts import InstanceAbsence, MaskObservation
from billiards_ai.lab.propagation.evaluation import (
    EvaluationReport,
    _load_evaluation_manifest,
    _load_raw_run,
)
from billiards_ai.lab.propagation.frame_mapping import extract_shot_frames
from billiards_ai.lab.propagation.overlays import _load_sample, _validate_json_artifact
from billiards_ai.lab.propagation.pilot_artifacts import PilotRunManifest
from billiards_ai.lab.propagation.pilot_bundle import (
    PilotInputBundle,
    PilotInventoryItem,
    verify_pilot_input_bundle,
)

BootstrapSplit = Literal["train", "valid"]
FrameKey = tuple[str, int]

_MAX_GEOMETRY_RATIO = 2.0


@dataclass(frozen=True)
class BootstrapShot:
    """One propagated shot and the complete gold-seeded instance inventory."""

    asset_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    prompt_boxes: Mapping[str, tuple[float, float, float, float]]

    def __post_init__(self) -> None:
        if not self.asset_id or self.shot_index < 0 or self.start_frame < 0:
            raise ValueError("bootstrap shot identity is invalid")
        if self.end_frame <= self.start_frame or not self.prompt_boxes:
            raise ValueError("bootstrap shot must be non-empty and prompted")
        for instance_id, (left, top, right, bottom) in self.prompt_boxes.items():
            if not instance_id or right <= left or bottom <= top:
                raise ValueError("bootstrap prompt boxes must be non-empty")


@dataclass(frozen=True)
class BootstrapFrame:
    asset_id: str
    shot_index: int
    frame_index: int
    observations: tuple[MaskObservation, ...]


@dataclass(frozen=True)
class BootstrapSelection:
    frames: tuple[BootstrapFrame, ...]
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class BootstrapDatasetExport:
    path: Path
    artifact_id: str
    image_count: int
    annotation_count: int


def _ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return math.inf
    return max(first, second) / min(first, second)


def _geometry_is_usable(
    observation: MaskObservation,
    prompt_box: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = observation.box_xyxy_px
    prompt_left, prompt_top, prompt_right, prompt_bottom = prompt_box
    width = right - left
    height = bottom - top
    prompt_width = prompt_right - prompt_left
    prompt_height = prompt_bottom - prompt_top
    box_area = width * height
    return (
        _ratio(width, prompt_width) <= _MAX_GEOMETRY_RATIO
        and _ratio(height, prompt_height) <= _MAX_GEOMETRY_RATIO
        and _ratio(width, height) <= _MAX_GEOMETRY_RATIO
        and observation.area_px <= box_area
        and box_area / observation.area_px <= _MAX_GEOMETRY_RATIO
    )


def select_bootstrap_frames(
    *,
    shots: Sequence[BootstrapShot],
    observations: Sequence[MaskObservation],
    absences: Sequence[InstanceAbsence],
    frozen_evaluation_frames: frozenset[FrameKey],
    frame_stride: int = 15,
) -> BootstrapSelection:
    """Select complete, QA-clean pseudo-label frames outside the frozen benchmark."""

    if frame_stride <= 0:
        raise ValueError("bootstrap frame stride must be positive")
    if len({(shot.asset_id, shot.shot_index) for shot in shots}) != len(shots):
        raise ValueError("bootstrap shots must have unique identities")

    observations_by_frame: dict[FrameKey, list[MaskObservation]] = defaultdict(list)
    for observation in observations:
        observations_by_frame[(observation.asset_id, observation.frame_index)].append(observation)
    absences_by_frame: dict[FrameKey, list[InstanceAbsence]] = defaultdict(list)
    for absence in absences:
        absences_by_frame[(absence.asset_id, absence.frame_index)].append(absence)

    selected: list[BootstrapFrame] = []
    rejected: Counter[str] = Counter()
    for shot in sorted(shots, key=lambda item: (item.asset_id, item.shot_index)):
        expected_ids = set(shot.prompt_boxes)
        for frame_index in range(shot.start_frame, shot.end_frame, frame_stride):
            key = (shot.asset_id, frame_index)
            if key in frozen_evaluation_frames:
                rejected["frozen_evaluation_frame"] += 1
                continue
            if absences_by_frame.get(key):
                rejected["explicit_absence"] += 1
                continue
            frame_observations = observations_by_frame.get(key, [])
            observed_ids = [item.instance_id for item in frame_observations]
            if len(observed_ids) != len(expected_ids) or set(observed_ids) != expected_ids:
                rejected["incomplete_instance_inventory"] += 1
                continue
            if any(item.qa_flags for item in frame_observations):
                rejected["qa_flagged"] += 1
                continue
            if any(
                not _geometry_is_usable(item, shot.prompt_boxes[item.instance_id])
                for item in frame_observations
            ):
                rejected["geometry_outlier"] += 1
                continue
            selected.append(
                BootstrapFrame(
                    asset_id=shot.asset_id,
                    shot_index=shot.shot_index,
                    frame_index=frame_index,
                    observations=tuple(
                        sorted(frame_observations, key=lambda item: item.instance_id)
                    ),
                )
            )
    return BootstrapSelection(
        frames=tuple(selected),
        rejection_counts=dict(sorted(rejected.items())),
    )


def assign_video_splits(
    asset_ids: Sequence[str],
    *,
    seed: int,
    valid_fraction: float,
) -> dict[str, BootstrapSplit]:
    """Assign complete videos to deterministic train/valid groups."""

    unique = sorted(set(asset_ids))
    if len(unique) != len(asset_ids) or len(unique) < 2:
        raise ValueError("video-disjoint splitting requires at least two unique assets")
    if not 0 < valid_fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    ranked = sorted(
        unique,
        key=lambda asset_id: (
            hashlib.sha256(f"{seed}:{asset_id}".encode()).hexdigest(),
            asset_id,
        ),
    )
    valid_count = min(len(ranked) - 1, max(1, math.ceil(len(ranked) * valid_fraction)))
    valid = set(ranked[:valid_count])
    return {asset_id: "valid" if asset_id in valid else "train" for asset_id in unique}


def verify_bootstrap_gate(
    *, expected: Mapping[str, str], verdict_payload: Mapping[str, object]
) -> dict[str, str]:
    """Require an exact immutable source binding and a passed human gate."""

    if (
        verdict_payload.get("pilot_status") != "passed"
        or not isinstance(verdict_payload.get("summary"), Mapping)
        or verdict_payload["summary"].get("status") != "human_gates_passed"
    ):  # type: ignore[union-attr]
        raise ValueError("bootstrap export requires a human-approved SAM2 pilot")
    if any(verdict_payload.get(key) != value for key, value in expected.items()):
        raise ValueError("bootstrap verdict source binding changed")
    return dict(expected)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_payload(root: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or reference in {"", "."}:
        raise ValueError("bootstrap mask reference is unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("bootstrap mask payload is missing or unsafe")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("bootstrap mask payload escapes its root") from exc
    return candidate


def _mask_annotation(
    *, mask_path: Path, observation: MaskObservation, image_shape: tuple[int, int]
) -> dict[str, object]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image_shape or set(np.unique(mask)) - {0, 255}:
        raise ValueError("bootstrap mask must be a full-size binary PNG")
    ys, xs = np.nonzero(mask)
    if len(xs) != observation.area_px or not len(xs):
        raise ValueError("bootstrap mask area changed")
    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    if box != observation.box_xyxy_px:
        raise ValueError("bootstrap mask bounding box changed")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [
        contour.reshape(-1, 2).astype(float).reshape(-1).tolist()
        for contour in contours
        if len(contour) >= 3
    ]
    if not polygons:
        raise ValueError("bootstrap mask has no trainable polygon")
    left, top, right, bottom = box
    return {
        "category_id": 1,
        "segmentation": polygons,
        "bbox": [left, top, right - left, bottom - top],
        "area": observation.area_px,
        "iscrowd": 0,
        "instance_id": observation.instance_id,
    }


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _verify_publication(path: Path, manifest: dict[str, object]) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bootstrap publication is missing or unsafe")
    stored = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if stored != manifest or stored.get("artifact_id") != path.name:
        raise ValueError("bootstrap publication manifest changed")
    expected = {"manifest.json"}
    for item in manifest["inventory"]:  # type: ignore[index]
        relative = str(item["path"])
        expected.add(relative)
        candidate = path.joinpath(*PurePosixPath(relative).parts)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != item["bytes"]
            or _sha256_file(candidate) != item["sha256"]
        ):
            raise ValueError("bootstrap publication payload changed")
    actual = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
    if actual != expected:
        raise ValueError("bootstrap publication inventory changed")
    if os.name != "nt" and any(
        stat.S_IMODE(item.stat().st_mode) & 0o222 for item in (path, *path.rglob("*"))
    ):
        raise ValueError("bootstrap publication is mutable")


def publish_bootstrap_dataset(
    *,
    selection: BootstrapSelection,
    splits: Mapping[str, BootstrapSplit],
    frame_paths: Mapping[FrameKey, Path],
    payload_root: Path,
    source_provenance: Mapping[str, str],
    output_root: Path,
    frame_stride: int = 15,
) -> BootstrapDatasetExport:
    """Publish an immutable, content-addressed RF-DETR COCO development dataset."""

    if not selection.frames:
        raise ValueError("bootstrap selection is empty")
    assets = {frame.asset_id for frame in selection.frames}
    if set(splits) != assets or set(splits.values()) != {"train", "valid"}:
        raise ValueError("bootstrap splits must cover all selected videos and both partitions")
    required_provenance = {
        "raw_run_hash",
        "input_bundle_id",
        "evaluation_manifest_hash",
        "verdict_artifact_hash",
    }
    if set(source_provenance) != required_provenance or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in source_provenance.values()
    ):
        raise ValueError("bootstrap source provenance is incomplete")

    output_root = output_root.absolute()
    publication_root = output_root / "bootstrap-datasets"
    publication_root.mkdir(parents=True, exist_ok=True)
    temporary = publication_root / f".pending-{uuid4().hex}"
    temporary.mkdir()
    image_count = 0
    annotation_count = 0
    try:
        for split in ("train", "valid"):
            (temporary / split).mkdir()
            images: list[dict[str, object]] = []
            annotations: list[dict[str, object]] = []
            split_frames = [frame for frame in selection.frames if splits[frame.asset_id] == split]
            for image_id, frame in enumerate(
                sorted(
                    split_frames,
                    key=lambda item: (item.asset_id, item.shot_index, item.frame_index),
                ),
                start=1,
            ):
                source = frame_paths.get((frame.asset_id, frame.frame_index))
                if source is None or source.is_symlink() or not source.is_file():
                    raise ValueError("bootstrap source frame is missing or unsafe")
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("bootstrap source frame cannot be decoded")
                height, width = image.shape[:2]
                filename = (
                    f"{frame.asset_id}-shot-{frame.shot_index:03d}-"
                    f"frame-{frame.frame_index:08d}.jpg"
                )
                shutil.copyfile(source, temporary / split / filename)
                images.append(
                    {
                        "id": image_id,
                        "file_name": filename,
                        "width": width,
                        "height": height,
                        "source_asset_id": frame.asset_id,
                        "source_frame_index": frame.frame_index,
                    }
                )
                for observation in frame.observations:
                    annotation = _mask_annotation(
                        mask_path=_safe_payload(payload_root, observation.mask_ref),
                        observation=observation,
                        image_shape=(height, width),
                    )
                    annotation_count += 1
                    annotations.append({"id": annotation_count, "image_id": image_id, **annotation})
                image_count += 1
            coco = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "ball", "supercategory": "billiards"}],
            }
            (temporary / split / "_annotations.coco.json").write_bytes(_canonical_bytes(coco))

        inventory = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        identity = {
            "schema_version": "1.0",
            "kind": "sam2_rfdetr_bootstrap",
            "source_provenance": dict(sorted(source_provenance.items())),
            "policy": {
                "benchmark_frames_excluded": True,
                "frame_complete_labels_required": True,
                "qa_flags_allowed": False,
                "scientific_use": "development_bootstrap_not_independent_accuracy",
            },
            "frame_stride": frame_stride,
            "split_assets": {
                split: sorted(asset for asset, assigned in splits.items() if assigned == split)
                for split in ("train", "valid")
            },
            "rejection_counts": selection.rejection_counts,
            "image_count": image_count,
            "annotation_count": annotation_count,
            "inventory": inventory,
        }
        artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        manifest = {"artifact_id": artifact_id, **identity}
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
        target = publication_root / artifact_id
        if target.exists():
            _verify_publication(target, manifest)
            shutil.rmtree(temporary)
        else:
            _make_immutable(temporary)
            os.replace(temporary, target)
            _verify_publication(target, manifest)
        return BootstrapDatasetExport(
            path=target,
            artifact_id=artifact_id,
            image_count=image_count,
            annotation_count=annotation_count,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_input_bundle(path: Path, *, expected_id: str) -> PilotInputBundle:
    try:
        raw = json.loads((path / "inventory.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "files"}:
            raise ValueError("input bundle inventory schema is invalid")
        inventory = tuple(PilotInventoryItem.model_validate(item) for item in raw["files"])
        bundle = PilotInputBundle(bundle_id=expected_id, path=path, inventory=inventory)
        verify_pilot_input_bundle(bundle)
        return bundle
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("bootstrap input bundle is invalid") from exc


def _bootstrap_shots(raw_manifest: PilotRunManifest) -> tuple[BootstrapShot, ...]:
    plans = {(item.asset_id, item.start_frame, item.end_frame): item for item in raw_manifest.plans}
    grouped: dict[tuple[str, int, int, int], dict[str, tuple[float, float, float, float]]] = (
        defaultdict(dict)
    )
    for executed in raw_manifest.executed_gold_prompts:
        prompt = executed.prompt
        if prompt.box_xyxy_px is None:
            raise ValueError("bootstrap requires box-seeded gold prompts")
        key = (executed.asset_id, executed.shot_index, executed.shot_start, executed.shot_end)
        if prompt.instance_id in grouped[key]:
            raise ValueError("bootstrap prompt instance identity is duplicated")
        grouped[key][prompt.instance_id] = prompt.box_xyxy_px
    shots = tuple(
        BootstrapShot(
            asset_id=asset_id,
            shot_index=shot_index,
            start_frame=start_frame,
            end_frame=end_frame,
            prompt_boxes=prompt_boxes,
        )
        for (asset_id, shot_index, start_frame, end_frame), prompt_boxes in sorted(grouped.items())
    )
    if {(item.asset_id, item.start_frame, item.end_frame) for item in shots} != set(plans):
        raise ValueError("bootstrap prompts do not cover every successful shot plan")
    return shots


def export_sam2_bootstrap(
    *,
    input_bundle: Path,
    raw_run: Path,
    evaluation_manifest: Path,
    evaluation_artifact: Path,
    review_sample: Path,
    verdict_artifact: Path,
    output_root: Path,
    frame_stride: int = 15,
    split_seed: int = 20260810,
) -> BootstrapDatasetExport:
    """Verify the passed pilot and export only non-benchmark, complete pseudo-label frames."""

    loaded = _load_raw_run(raw_run)
    manifest = loaded.manifest
    bundle = _load_input_bundle(input_bundle, expected_id=manifest.input_bundle_id)
    evaluation = _load_evaluation_manifest(evaluation_manifest)
    report_payload = _validate_json_artifact(
        evaluation_artifact, filename="report.json", label="evaluation artifact"
    )
    sample_payload = _validate_json_artifact(
        review_sample, filename="review-sample.json", label="review sample"
    )
    verdict_payload = _validate_json_artifact(
        verdict_artifact, filename="verdict.json", label="verdict artifact"
    )
    report = EvaluationReport.model_validate(report_payload)
    sample = _load_sample(sample_payload)
    if (
        report.raw_run_hash != raw_run.name
        or report.input_bundle_id != bundle.bundle_id
        or report.dataset_id != manifest.dataset_id
        or report.prompt_manifest_id != manifest.prompt_manifest_id
        or report.evaluation_manifest_hash != evaluation.manifest_id
        or sample.population_counts != report.review_population_counts
    ):
        raise ValueError("bootstrap evaluation provenance changed")
    shared = {
        "raw_run_hash": raw_run.name,
        "input_bundle_id": bundle.bundle_id,
        "dataset_id": manifest.dataset_id,
        "prompt_manifest_id": manifest.prompt_manifest_id,
        "evaluation_manifest_hash": evaluation.manifest_id,
    }
    if any(report_payload.get(key) != value for key, value in shared.items()) or any(
        sample_payload.get(key) != value for key, value in shared.items()
    ):
        raise ValueError("bootstrap report/sample binding changed")
    gate_binding = {
        **shared,
        "evaluation_artifact_hash": evaluation_artifact.name,
        "review_sample_hash": review_sample.name,
    }
    verify_bootstrap_gate(expected=gate_binding, verdict_payload=verdict_payload)

    shots = _bootstrap_shots(manifest)
    frozen_frames = frozenset(
        (circle.video_id, circle.source_frame_index) for circle in evaluation.circles
    )
    selection = select_bootstrap_frames(
        shots=shots,
        observations=loaded.observations,
        absences=loaded.absences,
        frozen_evaluation_frames=frozen_frames,
        frame_stride=frame_stride,
    )
    selected_assets = tuple(sorted({frame.asset_id for frame in selection.frames}))
    splits = assign_video_splits(selected_assets, seed=split_seed, valid_fraction=0.2)

    frame_paths: dict[FrameKey, Path] = {}
    selected_keys = {(frame.asset_id, frame.frame_index) for frame in selection.frames}
    if len(selected_keys) != len(selection.frames):
        raise ValueError("bootstrap selected duplicate source frames")
    cache_root = output_root / "bootstrap-frame-cache"
    for shot in shots:
        wanted = {
            frame
            for asset, frame in selected_keys
            if asset == shot.asset_id and shot.start_frame <= frame < shot.end_frame
        }
        if not wanted:
            continue
        source_shot = next(
            (
                item
                for item in manifest.prompt_manifest.shots
                if (item.video_id, item.shot_index) == (shot.asset_id, shot.shot_index)
            ),
            None,
        )
        if source_shot is None:
            raise ValueError("bootstrap shot identity is absent from the prompt manifest")
        extracted = extract_shot_frames(
            video_path=bundle.path / "videos" / f"{shot.asset_id}.mp4",
            shot=source_shot,
            output_dir=cache_root,
        )
        by_index = {
            frame.source_frame_index: extracted.frame_directory / frame.jpeg_filename
            for frame in extracted.frames
        }
        for frame_index in wanted:
            try:
                frame_paths[(shot.asset_id, frame_index)] = by_index[frame_index]
            except KeyError as exc:
                raise ValueError("bootstrap selected frame was not decoded") from exc

    return publish_bootstrap_dataset(
        selection=selection,
        splits=splits,
        frame_paths=frame_paths,
        payload_root=raw_run,
        source_provenance={
            "raw_run_hash": raw_run.name,
            "input_bundle_id": bundle.bundle_id,
            "evaluation_manifest_hash": evaluation.manifest_id,
            "verdict_artifact_hash": verdict_artifact.name,
        },
        output_root=output_root,
        frame_stride=frame_stride,
    )
