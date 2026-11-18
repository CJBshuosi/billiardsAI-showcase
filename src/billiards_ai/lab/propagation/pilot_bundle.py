"""Content-addressed local input bundles for the frozen SAM2 pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import Field, field_validator

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr, Sha256
from billiards_ai.lab.propagation.pilot_data import (
    PilotDataError,
    _read_jsonl,
    build_pilot_manifests,
    preflight_frozen_pilot_counts,
)


class PilotBundleError(RuntimeError):
    """A local pilot input payload is missing, unsafe, extra, or changed."""


class PilotInventoryItem(FrozenContract):
    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("inventory path must be relative and confined")
        return value


class PilotInputBundle(FrozenContract):
    bundle_id: Sha256
    path: Path
    inventory: tuple[PilotInventoryItem, ...]


def dataset_id_from_inventory(inventory: tuple[PilotInventoryItem, ...]) -> str:
    """Derive the dataset identity only from verified video inventory descriptors."""

    descriptors: list[dict[str, object]] = []
    for item in inventory:
        path = PurePosixPath(item.path)
        if len(path.parts) == 2 and path.parts[0] == "videos" and path.suffix == ".mp4":
            descriptors.append(
                {
                    "video_id": path.stem,
                    "path": item.path,
                    "bytes": item.bytes,
                    "sha256": item.sha256,
                }
            )
    if not descriptors:
        raise PilotBundleError("input bundle has no video payload descriptors")
    descriptors.sort(key=lambda item: (str(item["video_id"]), str(item["path"])))
    return hashlib.sha256(_canonical_bytes(descriptors)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        result = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PilotBundleError(f"unsafe payload path: {path}") from exc
    if path.is_symlink() or not path.is_file() or result in {"", "."}:
        raise PilotBundleError(f"unsafe payload path: {path}")
    return result


def _review_payloads(review_pack: Path) -> set[str]:
    expected: set[str] = set()
    for review_file in sorted((review_pack / "videos").glob("*/shot-review.jsonl")):
        video_id = review_file.parent.name
        expected.add(_relative(review_pack, review_file))
        for row in _read_jsonl(review_file):
            frames = row.get("evidence_frames")
            if not isinstance(frames, dict):
                raise PilotBundleError("review row lacks evidence frames")
            for filename in frames.values():
                if not isinstance(filename, str):
                    raise PilotBundleError("review evidence filename is not a string")
                path = review_pack / "videos" / video_id / "review-frames" / filename
                expected.add(_relative(review_pack, path))
    _require_exact_regular_files(review_pack / "videos", expected, prefix="videos")
    return expected


def _annotation_payloads(annotation_export: Path) -> set[str]:
    required = {"annotations.json", "summary.json"}
    payload = json.loads((annotation_export / "annotations.json").read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise PilotBundleError("annotation export lacks images")
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
            raise PilotBundleError("annotation image lacks file name")
        relative = PurePosixPath(image["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PilotBundleError("unsafe annotation image name")
        required.add((PurePosixPath("images") / relative).as_posix())
    for relative in required:
        _relative(annotation_export, annotation_export / relative)
    _require_exact_regular_files(annotation_export, required)
    return {"annotations.json", "summary.json"}


def _video_payloads(dataset_root: Path, video_ids: set[str]) -> set[str]:
    expected = {f"videos/{video_id}.mp4" for video_id in video_ids}
    for relative in expected:
        _relative(dataset_root, dataset_root / relative)
    _require_exact_regular_files(dataset_root / "videos", expected, prefix="videos")
    return expected


def _require_exact_regular_files(root: Path, expected: set[str], *, prefix: str = "") -> None:
    if not root.is_dir() or root.is_symlink():
        raise PilotBundleError(f"payload scope is missing or unsafe: {root}")
    discovered: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise PilotBundleError(f"payload scope contains a symlink: {candidate}")
        if candidate.is_file():
            relative = _relative(root, candidate)
            discovered.add(f"{prefix}/{relative}" if prefix else relative)
    if discovered != expected:
        unexpected = sorted(discovered - expected)
        missing = sorted(expected - discovered)
        raise PilotBundleError(f"payload mismatch: {unexpected or missing}")


def _inventory_item(
    root: Path, relative: str, *, stored_path: str
) -> tuple[Path, PilotInventoryItem]:
    source = root / relative
    _relative(root, source)
    return source, PilotInventoryItem(
        path=stored_path, bytes=source.stat().st_size, sha256=_sha256(source)
    )


def _verify_existing_bundle(
    path: Path, *, identity: dict[str, object], inventory: tuple[PilotInventoryItem, ...]
) -> None:
    expected_inventory = _canonical_bytes(identity) + b"\n"
    inventory_path = path / "inventory.json"
    if not inventory_path.is_file() or inventory_path.read_bytes() != expected_inventory:
        raise PilotBundleError("existing input bundle inventory is invalid")
    expected_paths = {"inventory.json", *(item.path for item in inventory)}
    found_paths = {
        _relative(path, candidate) for candidate in path.rglob("*") if candidate.is_file()
    }
    if found_paths != expected_paths:
        raise PilotBundleError("existing input bundle payload set is invalid")
    for item in inventory:
        candidate = path / item.path
        if not candidate.is_file() or candidate.stat().st_size != item.bytes:
            raise PilotBundleError(f"existing input bundle payload changed: {item.path}")
        if _sha256(candidate) != item.sha256:
            raise PilotBundleError(f"existing input bundle payload changed: {item.path}")


def build_pilot_input_bundle(
    *, annotation_export: Path, review_pack: Path, dataset_root: Path, output_root: Path
) -> PilotInputBundle:
    """Freeze exactly the local videos, reviewed frames, and annotation export payloads."""

    try:
        manifests = build_pilot_manifests(
            annotation_export=annotation_export,
            review_pack=review_pack,
            dataset_root=dataset_root,
            output_root=output_root,
        )
        preflight_frozen_pilot_counts(manifests)
    except PilotDataError as exc:
        raise PilotBundleError(str(exc)) from exc
    annotation_export = annotation_export.resolve()
    review_pack = review_pack.resolve()
    dataset_root = dataset_root.resolve()
    annotation_payloads = _annotation_payloads(annotation_export)
    review_payloads = _review_payloads(review_pack)
    video_payloads = _video_payloads(
        dataset_root, {shot.video_id for shot in manifests.prompt_manifest.shots}
    )
    entries: list[tuple[Path, PilotInventoryItem]] = []
    entries.extend(
        _inventory_item(annotation_export, relative, stored_path=f"annotation-export/{relative}")
        for relative in annotation_payloads
    )
    entries.extend(
        _inventory_item(review_pack, relative, stored_path=f"review-pack/{relative}")
        for relative in review_payloads
    )
    entries.extend(
        _inventory_item(dataset_root, relative, stored_path=relative) for relative in video_payloads
    )
    entries.sort(key=lambda item: item[1].path)
    inventory = tuple(item for _, item in entries)
    identity = {
        "schema_version": "1.0",
        "files": [item.model_dump(mode="json") for item in inventory],
    }
    bundle_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    final_path = output_root / "input-bundles" / bundle_id
    if final_path.is_dir():
        _verify_existing_bundle(final_path, identity=identity, inventory=inventory)
        return PilotInputBundle(bundle_id=bundle_id, path=final_path, inventory=inventory)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.parent / f".{bundle_id}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for source, item in entries:
            destination = temporary / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if destination.stat().st_size != item.bytes or _sha256(destination) != item.sha256:
                raise PilotBundleError(f"payload changed while freezing: {item.path}")
        (temporary / "inventory.json").write_bytes(_canonical_bytes(identity) + b"\n")
        os.replace(temporary, final_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return PilotInputBundle(bundle_id=bundle_id, path=final_path, inventory=inventory)


def verify_pilot_input_bundle(bundle: PilotInputBundle) -> None:
    """Revalidate a caller-supplied input bundle contract against its frozen tree."""

    if not isinstance(bundle, PilotInputBundle):
        raise TypeError("pilot input provenance requires a PilotInputBundle")
    identity = {
        "schema_version": "1.0",
        "files": [item.model_dump(mode="json") for item in bundle.inventory],
    }
    expected_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    if bundle.bundle_id != expected_id or bundle.path.name != expected_id:
        raise PilotBundleError("input bundle identity does not match its verified inventory")
    _verify_existing_bundle(bundle.path, identity=identity, inventory=bundle.inventory)
