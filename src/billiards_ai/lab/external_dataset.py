"""Fail-closed descriptors for approved external algorithm datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import (
    DatasetPurpose,
    FrozenContract,
    NonEmptyStr,
    Sha256,
)


class ExternalDatasetError(RuntimeError):
    """An external dataset is missing, changed, unsafe, or unauthorized."""


class ExternalFile(FrozenContract):
    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256
    media_kind: Literal["video", "annotation"]

    @field_validator("path")
    @classmethod
    def path_is_relative_and_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("external file path must be relative and confined")
        return value


class ExternalDatasetDescriptor(FrozenContract):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmptyStr
    source_repository: NonEmptyStr
    source_commit: NonEmptyStr
    license: NonEmptyStr
    commercial_status: Literal["legal_review_required"]
    approved_purposes: tuple[DatasetPurpose, ...]
    production_use: Literal[False]
    expected_file_count: int = Field(gt=0)
    expected_total_bytes: int = Field(gt=0)
    files: tuple[ExternalFile, ...]

    @field_validator("approved_purposes")
    @classmethod
    def purposes_are_local_and_unique(
        cls, value: tuple[DatasetPurpose, ...]
    ) -> tuple[DatasetPurpose, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("approved purposes must be non-empty and unique")
        if DatasetPurpose.CLOUD_TRAINING in value:
            raise ValueError("external dataset descriptor cannot authorize cloud training")
        return tuple(sorted(value, key=lambda item: item.value))

    @model_validator(mode="after")
    def totals_and_files_are_consistent(self) -> ExternalDatasetDescriptor:
        if not self.files:
            raise ValueError("external dataset requires at least one file")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("external dataset file paths must be unique")
        hashes = [item.sha256 for item in self.files]
        if len(hashes) != len(set(hashes)):
            raise ValueError("external dataset file sha256 values must be unique")
        if self.expected_file_count != len(self.files):
            raise ValueError("external dataset file count is inconsistent")
        if self.expected_total_bytes != sum(item.bytes for item in self.files):
            raise ValueError("external dataset byte total is inconsistent")
        return self


class VerifiedExternalFile(FrozenContract):
    path: NonEmptyStr
    bytes: int = Field(gt=0)
    sha256: Sha256


class VerifiedExternalDataset(FrozenContract):
    dataset_id: NonEmptyStr
    root: NonEmptyStr
    file_count: int = Field(gt=0)
    total_bytes: int = Field(gt=0)
    descriptor_sha256: Sha256
    files: tuple[VerifiedExternalFile, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_external_bundle_sha256(descriptor: ExternalDatasetDescriptor) -> str:
    """Return the stable digest of the complete approved descriptor."""

    payload = json.dumps(
        descriptor.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_external_descriptor(path: Path) -> ExternalDatasetDescriptor:
    """Load and validate an external dataset descriptor from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExternalDatasetDescriptor.model_validate(payload)


def _contains_symlink(root: Path, relative_path: str) -> bool:
    candidate = root
    for part in PurePosixPath(relative_path).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def verify_external_dataset(
    descriptor: ExternalDatasetDescriptor,
    root: Path,
) -> VerifiedExternalDataset:
    """Verify every approved file and return immutable verification evidence."""

    root = root.resolve()
    verified_files: list[VerifiedExternalFile] = []
    for item in descriptor.files:
        if _contains_symlink(root, item.path):
            raise ExternalDatasetError(f"symlink is not allowed: {item.path}")
        candidate = (root / item.path).resolve()
        if not candidate.is_relative_to(root):
            raise ExternalDatasetError(f"unsafe file path: {item.path}")
        if not candidate.is_file():
            raise ExternalDatasetError(f"missing file: {item.path}")
        actual_bytes = candidate.stat().st_size
        if actual_bytes != item.bytes:
            raise ExternalDatasetError(
                f"size mismatch for {item.path}: expected {item.bytes}, got {actual_bytes}"
            )
        actual_sha256 = _sha256(candidate)
        if actual_sha256 != item.sha256:
            raise ExternalDatasetError(
                f"sha256 mismatch for {item.path}: expected {item.sha256}, got {actual_sha256}"
            )
        verified_files.append(
            VerifiedExternalFile(
                path=item.path,
                bytes=actual_bytes,
                sha256=actual_sha256,
            )
        )

    expected_paths = {item.path for item in descriptor.files}
    discovered_media = {
        path.relative_to(root).as_posix()
        for pattern in ("*.mp4", "*.jsonl")
        for path in root.rglob(pattern)
    }
    unexpected = sorted(discovered_media - expected_paths)
    if unexpected:
        raise ExternalDatasetError(f"unexpected file: {unexpected[0]}")

    return VerifiedExternalDataset(
        dataset_id=descriptor.dataset_id,
        root=str(root),
        file_count=len(verified_files),
        total_bytes=sum(item.bytes for item in verified_files),
        descriptor_sha256=canonical_external_bundle_sha256(descriptor),
        files=tuple(verified_files),
    )
