from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_verifier_requires_every_expected_file(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    descriptor = external_dataset.ExternalDatasetDescriptor(
        dataset_id="fixture-external-dataset",
        source_repository="owner/repository",
        source_commit="7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
        license="Apache-2.0-repository-declaration-video-provenance-unresolved",
        commercial_status="legal_review_required",
        approved_purposes=("local_training",),
        production_use=False,
        expected_file_count=1,
        expected_total_bytes=3,
        files=(
            external_dataset.ExternalFile(
                path="videos/video1.mp4",
                bytes=3,
                sha256="a" * 64,
                media_kind="video",
            ),
        ),
    )

    with pytest.raises(external_dataset.ExternalDatasetError, match="missing file"):
        external_dataset.verify_external_dataset(descriptor, tmp_path)


def _descriptor(external_dataset: object, *, sha256: str, bytes_: int = 3) -> object:
    return external_dataset.ExternalDatasetDescriptor(
        dataset_id="fixture-external-dataset",
        source_repository="owner/repository",
        source_commit="7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
        license="Apache-2.0-repository-declaration-video-provenance-unresolved",
        commercial_status="legal_review_required",
        approved_purposes=("local_training", "local_view"),
        production_use=False,
        expected_file_count=1,
        expected_total_bytes=bytes_,
        files=(
            external_dataset.ExternalFile(
                path="videos/video1.mp4",
                bytes=bytes_,
                sha256=sha256,
                media_kind="video",
            ),
        ),
    )


def test_verifier_rejects_size_mismatch(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    video = tmp_path / "videos/video1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"wrong")
    descriptor = _descriptor(external_dataset, sha256=hashlib.sha256(b"abc").hexdigest())

    with pytest.raises(external_dataset.ExternalDatasetError, match="size mismatch"):
        external_dataset.verify_external_dataset(descriptor, tmp_path)


def test_verifier_rejects_hash_mismatch(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    video = tmp_path / "videos/video1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"abc")
    descriptor = _descriptor(external_dataset, sha256="a" * 64)

    with pytest.raises(external_dataset.ExternalDatasetError, match="sha256 mismatch"):
        external_dataset.verify_external_dataset(descriptor, tmp_path)


def test_descriptor_rejects_duplicate_paths_and_inconsistent_totals() -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    file_row = external_dataset.ExternalFile(
        path="videos/video1.mp4",
        bytes=3,
        sha256="a" * 64,
        media_kind="video",
    )

    with pytest.raises(ValidationError):
        external_dataset.ExternalDatasetDescriptor(
            dataset_id="fixture-external-dataset",
            source_repository="owner/repository",
            source_commit="7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
            license="unresolved",
            commercial_status="legal_review_required",
            approved_purposes=("local_training",),
            production_use=False,
            expected_file_count=1,
            expected_total_bytes=6,
            files=(file_row, file_row),
        )


def test_external_file_rejects_parent_path() -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")

    with pytest.raises(ValidationError):
        external_dataset.ExternalFile(
            path="../video1.mp4",
            bytes=3,
            sha256="a" * 64,
            media_kind="video",
        )


def test_successful_verification_returns_canonical_evidence(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    video = tmp_path / "videos/video1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    descriptor = _descriptor(external_dataset, sha256=digest)

    verified = external_dataset.verify_external_dataset(descriptor, tmp_path)

    assert verified.file_count == 1
    assert verified.total_bytes == 3
    assert verified.files[0].sha256 == digest
    assert len(verified.descriptor_sha256) == 64


def test_descriptor_rejects_duplicate_content_hashes() -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")

    with pytest.raises(ValidationError, match="sha256"):
        external_dataset.ExternalDatasetDescriptor(
            dataset_id="fixture-external-dataset",
            source_repository="owner/repository",
            source_commit="7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
            license="unresolved",
            commercial_status="legal_review_required",
            approved_purposes=("local_training",),
            production_use=False,
            expected_file_count=2,
            expected_total_bytes=6,
            files=(
                external_dataset.ExternalFile(
                    path="videos/video1.mp4",
                    bytes=3,
                    sha256="a" * 64,
                    media_kind="video",
                ),
                external_dataset.ExternalFile(
                    path="videos/video2.mp4",
                    bytes=3,
                    sha256="a" * 64,
                    media_kind="video",
                ),
            ),
        )


def test_verifier_rejects_symlinked_expected_file(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside.write_bytes(b"abc")
    video = tmp_path / "videos/video1.mp4"
    video.parent.mkdir()
    video.symlink_to(outside)
    descriptor = _descriptor(external_dataset, sha256=hashlib.sha256(b"abc").hexdigest())

    with pytest.raises(external_dataset.ExternalDatasetError, match="symlink"):
        external_dataset.verify_external_dataset(descriptor, tmp_path)


def test_verifier_rejects_unapproved_extra_media(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    video = tmp_path / "videos/video1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"abc")
    (tmp_path / "unexpected.jsonl").write_text("{}\n", encoding="utf-8")
    descriptor = _descriptor(external_dataset, sha256=hashlib.sha256(b"abc").hexdigest())

    with pytest.raises(external_dataset.ExternalDatasetError, match="unexpected file"):
        external_dataset.verify_external_dataset(descriptor, tmp_path)


def test_load_external_descriptor_from_yaml(tmp_path: Path) -> None:
    external_dataset = importlib.import_module("billiards_ai.lab.external_dataset")
    digest = hashlib.sha256(b"abc").hexdigest()
    descriptor_path = tmp_path / "descriptor.yaml"
    descriptor_path.write_text(
        "\n".join(
            (
                "schema_version: '1.0'",
                "dataset_id: fixture-external-dataset",
                "source_repository: owner/repository",
                "source_commit: 7d3fb98497a2103251f7f3d6bb9a0f32e9b2aa41",
                "license: unresolved",
                "commercial_status: legal_review_required",
                "approved_purposes: [local_training]",
                "production_use: false",
                "expected_file_count: 1",
                "expected_total_bytes: 3",
                "files:",
                "  - path: videos/video1.mp4",
                "    bytes: 3",
                f"    sha256: {digest}",
                "    media_kind: video",
            )
        ),
        encoding="utf-8",
    )

    descriptor = external_dataset.load_external_descriptor(descriptor_path)

    assert descriptor.files[0].sha256 == digest
    assert external_dataset.canonical_external_bundle_sha256(descriptor) == (
        external_dataset.canonical_external_bundle_sha256(descriptor)
    )
