"""Offline-only preflight for a proposed cloud training job."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Sha256 = str


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _confined_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or ".." in path.parts:
        raise ValueError("path must be relative and confined")
    return value


class Approval(_FrozenModel):
    approval_status: str


class Provider(Approval):
    name: str = Field(min_length=1)
    terms_url: str = Field(pattern=r"^https://")
    dpa_url: str = Field(pattern=r"^https://")


class Region(Approval):
    id: str = Field(min_length=1)
    jurisdiction: str


class Instance(Approval):
    gpu: str = Field(min_length=1)
    secure_cloud: bool


class ContainerImage(Approval):
    reference: str = Field(min_length=1)


class Budget(_FrozenModel):
    maximum_usd: float = Field(gt=0)
    estimated_hours: float = Field(gt=0)
    hourly_usd: float = Field(gt=0)


class ArtifactReference(_FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _path_is_confined = field_validator("path")(_confined_path)


class DatasetReference(_FrozenModel):
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upload_bytes: int = Field(ge=0)

    _path_is_confined = field_validator("manifest_path")(_confined_path)


class AssetManifestReference(ArtifactReference):
    required_asset_ids: tuple[str, ...] = Field(min_length=1)


class Shutdown(_FrozenModel):
    enabled: bool
    idle_minutes: int = Field(gt=0)
    hard_deadline_minutes: int = Field(gt=0)


class Deletion(_FrozenModel):
    approval_status: str
    delete_instance: bool
    delete_attached_volume: bool
    verification_required: bool


class Collection(_FrozenModel):
    local_output_dir: str
    expected_output_bytes: int = Field(gt=0)
    free_space_multiplier: float = Field(ge=1)

    _path_is_confined = field_validator("local_output_dir")(_confined_path)


class CloudJobSpec(_FrozenModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    provider: Provider
    region: Region
    instance: Instance
    image: ContainerImage
    budget: Budget
    policy: ArtifactReference
    dataset: DatasetReference
    asset_manifest: AssetManifestReference
    training_configs: tuple[ArtifactReference, ...] = Field(min_length=1)
    shutdown: Shutdown
    deletion: Deletion
    collection: Collection


@dataclass(frozen=True)
class CloudPreflightResult:
    passed: bool
    blockers: tuple[str, ...]
    upload_bytes: int
    projected_cost_usd: float


def _evidence_is_valid(evidence_dir: Path, job_hash: str) -> bool:
    try:
        record = json.loads((evidence_dir / "preflight.json").read_text(encoding="utf-8"))
        hashes = record["input_sha256"]
        if record.get("job_hash") != job_hash or record.get("passed") is not True:
            return False
        if not isinstance(hashes, dict) or not hashes:
            return False
        declared = set(hashes) | {"preflight.json"}
        actual = {
            path.relative_to(evidence_dir).as_posix()
            for path in evidence_dir.rglob("*")
            if path.is_file()
        }
        if actual != declared or evidence_dir.stat().st_mode & stat.S_IWUSR:
            return False
        for relative_path, expected_sha256 in hashes.items():
            confined = _confined_path(relative_path)
            path = evidence_dir.joinpath(*PurePosixPath(confined).parts)
            if path.is_symlink() or not path.is_file() or path.stat().st_mode & stat.S_IWUSR:
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
                return False
    except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    return True


def create_job_evidence(spec_path: Path, evidence_root: Path) -> Path:
    result = run_preflight(spec_path)
    if not result.passed:
        raise ValueError(f"preflight failed: {', '.join(result.blockers)}")
    spec_bytes = spec_path.read_bytes()
    job_hash = hashlib.sha256(spec_bytes).hexdigest()
    evidence_dir = evidence_root / job_hash
    if evidence_dir.exists():
        if not _evidence_is_valid(evidence_dir, job_hash):
            raise ValueError(f"existing job evidence is corrupt: {evidence_dir}")
        return evidence_dir

    spec = CloudJobSpec.model_validate(yaml.safe_load(spec_bytes))
    root = spec_path.parent.resolve(strict=True)
    sources = [
        ("cloud-job.yaml", spec_path),
        ("inputs/experiment-policy.yaml", root / spec.policy.path),
        ("inputs/derived-dataset.yaml", root / spec.dataset.manifest_path),
        ("inputs/asset-manifest.yaml", root / spec.asset_manifest.path),
        *(
            (f"inputs/training-config-{index:03d}.yaml", root / reference.path)
            for index, reference in enumerate(spec.training_configs)
        ),
    ]
    temporary = evidence_root.parent / f".tmp-{evidence_root.name}-{uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        for relative_path, source in sources:
            destination = temporary.joinpath(*PurePosixPath(relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        input_sha256 = {
            relative_path: hashlib.sha256(
                temporary.joinpath(*PurePosixPath(relative_path).parts).read_bytes()
            ).hexdigest()
            for relative_path, _ in sources
        }
        record = {
            "schema_version": "1.0",
            "job_hash": job_hash,
            "passed": result.passed,
            "blockers": list(result.blockers),
            "upload_bytes": result.upload_bytes,
            "projected_cost_usd": result.projected_cost_usd,
            "network_calls": 0,
            "cloud_resources_created": 0,
            "input_sha256": input_sha256,
        }
        (temporary / "preflight.json").write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        for path in sorted(temporary.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        evidence_root.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, evidence_dir)
        evidence_dir.chmod(0o555)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return evidence_dir


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_and_verify(root: Path, relative_path: str, expected_sha256: str) -> Path | None:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink() or not candidate.is_file():
        return None
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        return None
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256:
        return None
    return resolved


def _valid_digest_reference(reference: str) -> bool:
    marker = "@sha256:"
    if reference.count(marker) != 1:
        return False
    digest = reference.rsplit(marker, 1)[1]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _validate_dataset(payload: Any, expected_upload_bytes: int) -> list[str]:
    blockers = []
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        return ["derived_dataset_manifest_invalid"]
    assets = payload["assets"]
    total_bytes = 0
    for asset in assets:
        if not isinstance(asset, dict):
            blockers.append("derived_dataset_manifest_invalid")
            continue
        size = asset.get("bytes")
        if not isinstance(size, int) or size < 0:
            blockers.append("derived_dataset_manifest_invalid")
        else:
            total_bytes += size
        if asset.get("is_real_person_source") and "cloud_training" not in asset.get(
            "approved_uses", []
        ):
            blockers.append("real_asset_missing_cloud_training_approval")
        if asset.get("table_crop_only") is not True:
            blockers.append("dataset_not_table_cropped")
        if asset.get("audio_removed") is not True:
            blockers.append("dataset_audio_not_removed")
        if asset.get("contains_user_identity") is not False:
            blockers.append("dataset_contains_user_identity")
    if total_bytes != expected_upload_bytes:
        blockers.append("dataset_upload_bytes_mismatch")
    return blockers


def _validate_assets(payload: Any, required_ids: tuple[str, ...]) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        return ["asset_manifest_invalid"]
    by_id = {
        asset.get("id"): asset
        for asset in payload["assets"]
        if isinstance(asset, dict) and isinstance(asset.get("id"), str)
    }
    blockers = []
    for asset_id in required_ids:
        asset = by_id.get(asset_id)
        digest = asset.get("sha256") if asset else None
        if (
            asset is None
            or asset.get("approval_status") != "approved"
            or asset.get("commercial_status") != "allowed"
            or not isinstance(asset.get("version"), str)
            or not asset["version"].strip()
            or not isinstance(asset.get("license"), str)
            or not asset["license"].strip()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            blockers.append(f"asset_not_approved:{asset_id}")
    return blockers


def run_preflight(spec_path: Path) -> CloudPreflightResult:
    try:
        spec = CloudJobSpec.model_validate(_load_yaml(spec_path))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError):
        return CloudPreflightResult(False, ("job_spec_invalid",), 0, 0)

    root = spec_path.parent.resolve(strict=True)
    blockers = []
    for label, approval in (
        ("provider_not_approved", spec.provider),
        ("region_not_approved", spec.region),
        ("instance_not_approved", spec.instance),
        ("container_image_not_approved", spec.image),
    ):
        if approval.approval_status != "approved":
            blockers.append(label)
    if spec.region.jurisdiction != "EU":
        blockers.append("region_not_eu")
    if not spec.instance.secure_cloud:
        blockers.append("instance_not_secure_cloud")
    if not _valid_digest_reference(spec.image.reference):
        blockers.append("container_image_not_digest_pinned")

    projected_cost = spec.budget.estimated_hours * spec.budget.hourly_usd
    policy_path = _resolve_and_verify(root, spec.policy.path, spec.policy.sha256)
    if policy_path is None:
        blockers.append("experiment_policy_hash_mismatch")
        policy = {}
    else:
        policy = _load_yaml(policy_path)
    if not isinstance(policy, dict) or policy.get("cloud_budget_status") != "approved":
        blockers.append("budget_not_approved")
    cap = policy.get("cloud_budget_usd") if isinstance(policy, dict) else None
    if (
        not isinstance(cap, (int, float))
        or spec.budget.maximum_usd > cap
        or projected_cost > spec.budget.maximum_usd
    ):
        blockers.append("budget_cap_exceeded")

    dataset_path = _resolve_and_verify(
        root, spec.dataset.manifest_path, spec.dataset.manifest_sha256
    )
    if dataset_path is None:
        blockers.append("dataset_manifest_hash_mismatch")
    else:
        blockers.extend(_validate_dataset(_load_yaml(dataset_path), spec.dataset.upload_bytes))

    asset_path = _resolve_and_verify(root, spec.asset_manifest.path, spec.asset_manifest.sha256)
    if asset_path is None:
        blockers.append("asset_manifest_hash_mismatch")
    else:
        blockers.extend(
            _validate_assets(_load_yaml(asset_path), spec.asset_manifest.required_asset_ids)
        )
    if any(
        _resolve_and_verify(root, reference.path, reference.sha256) is None
        for reference in spec.training_configs
    ):
        blockers.append("training_config_hash_mismatch")

    allowed_idle = policy.get("automatic_shutdown_minutes") if isinstance(policy, dict) else None
    if (
        not spec.shutdown.enabled
        or not isinstance(allowed_idle, int)
        or spec.shutdown.idle_minutes > allowed_idle
    ):
        blockers.append("automatic_shutdown_not_enforced")
    deletion = spec.deletion
    if not (
        deletion.approval_status == "approved"
        and deletion.delete_instance
        and deletion.delete_attached_volume
        and deletion.verification_required
    ):
        blockers.append("deletion_policy_not_approved")

    free_bytes = shutil.disk_usage(root).free
    required_free = spec.collection.expected_output_bytes * spec.collection.free_space_multiplier
    if required_free > free_bytes:
        blockers.append("insufficient_local_collection_space")
    unique_blockers = tuple(dict.fromkeys(blockers))
    return CloudPreflightResult(
        passed=not unique_blockers,
        blockers=unique_blockers,
        upload_bytes=spec.dataset.upload_bytes,
        projected_cost_usd=projected_cost,
    )
