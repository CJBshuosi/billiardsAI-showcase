"""Fail-closed release packaging for a verified algorithm-brain finalist."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]

_REQUIRED_ARTIFACT_ROLES = {
    "detector_checkpoint",
    "detector_onnx",
    "identity_checkpoint",
    "identity_onnx",
    "finalist_config",
    "tracker_config",
    "event_config",
    "model_card",
    "benchmark_report",
    "locked_test_claim",
    "locked_test_metrics",
}


class ReleasePackagingError(ValueError):
    """Raised when release evidence is incomplete or inconsistent."""


def claim_locked_test_opening(
    *,
    ledger_root: Path,
    generation_id: str,
    dataset_revision: str,
    authorization_id: str,
) -> Path:
    try:
        claim = LockedTestClaim(
            schema_version="1.0",
            generation_id=generation_id,
            dataset_revision=dataset_revision,
            authorization_id=authorization_id,
        )
    except ValidationError as exc:
        raise ReleasePackagingError("locked-test claim is incomplete or invalid") from exc
    ledger_root.mkdir(parents=True, exist_ok=True)
    claim_path = ledger_root / f"{claim.generation_id}.json"
    payload = _canonical_bytes(claim.model_dump(mode="json")) + b"\n"
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ReleasePackagingError(
            f"locked test has already been opened for generation {claim.generation_id}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as claim_file:
            claim_file.write(payload)
            claim_file.flush()
            os.fsync(claim_file.fileno())
    except Exception:
        claim_path.unlink(missing_ok=True)
        raise
    return claim_path


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LockedTestClaim(_FrozenModel):
    schema_version: Literal["1.0"]
    generation_id: Identifier
    dataset_revision: Identifier
    authorization_id: Identifier


def _confined_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or ".." in path.parts:
        raise ValueError("path must be relative and confined")
    return value


class PackageArtifact(_FrozenModel):
    role: Identifier
    source_path: str
    package_path: str
    sha256: Sha256

    _source_is_confined = field_validator("source_path")(_confined_path)
    _package_is_confined = field_validator("package_path")(_confined_path)


class LicenseNotice(_FrozenModel):
    source_path: str
    package_path: str
    sha256: Sha256

    _source_is_confined = field_validator("source_path")(_confined_path)
    _package_is_confined = field_validator("package_path")(_confined_path)

    @field_validator("package_path")
    @classmethod
    def notice_stays_in_license_directory(cls, value: str) -> str:
        if PurePosixPath(value).parts[0] != "LICENSES":
            raise ValueError("license notice must be packaged below LICENSES")
        return value


class LockedTestEvidence(_FrozenModel):
    generation_id: Identifier
    dataset_revision: Identifier
    open_count: int = Field(ge=0)
    all_hard_gates_passed: bool
    metrics_sha256: Sha256
    thresholds_sha256: Sha256
    authorization_id: Identifier
    claim_sha256: Sha256


class LockedTestMetrics(_FrozenModel):
    schema_version: Literal["1.0"]
    generation_id: Identifier
    dataset_revision: Identifier
    result: Literal["PASS"]
    all_hard_gates_passed: bool
    thresholds_sha256: Sha256


class ReleaseSpec(_FrozenModel):
    schema_version: Literal["1.0"]
    release_id: Identifier
    generation_id: Identifier
    generation_status: Literal["frozen_before_locked_test"]
    result: Literal["PASS"]
    dataset_revision: Identifier
    dataset_manifest_sha256: Sha256
    training_seeds: tuple[int, ...] = Field(min_length=1)
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    dependency_lock_sha256: Sha256
    class_map: dict[str, int] = Field(min_length=1)
    calibration_assumptions: tuple[str, ...] = Field(min_length=1)
    supported_events: tuple[Identifier, ...] = Field(min_length=1)
    unsupported_events: tuple[Identifier, ...] = Field(min_length=1)
    runtime_requirements: tuple[str, ...] = Field(min_length=1)
    asset_ids: tuple[Identifier, ...] = Field(min_length=1)
    artifacts: tuple[PackageArtifact, ...] = Field(min_length=1)
    license_notices: tuple[LicenseNotice, ...] = Field(min_length=1)
    locked_test: LockedTestEvidence

    @model_validator(mode="after")
    def release_lists_are_consistent(self) -> ReleaseSpec:
        unique_fields = {
            "training seeds": self.training_seeds,
            "supported events": self.supported_events,
            "unsupported events": self.unsupported_events,
            "asset IDs": self.asset_ids,
        }
        for label, values in unique_fields.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if set(self.supported_events) & set(self.unsupported_events):
            raise ValueError("supported and unsupported events must be disjoint")
        if any(not value.strip() for value in self.calibration_assumptions):
            raise ValueError("calibration assumptions cannot be blank")
        if any(not value.strip() for value in self.runtime_requirements):
            raise ValueError("runtime requirements cannot be blank")
        if len(set(self.class_map.values())) != len(self.class_map) or any(
            value < 0 for value in self.class_map.values()
        ):
            raise ValueError("class map indices must be unique and non-negative")
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)) or not _REQUIRED_ARTIFACT_ROLES <= set(roles):
            raise ValueError("release artifacts are missing required unique roles")
        package_paths = [
            *(artifact.package_path for artifact in self.artifacts),
            *(notice.package_path for notice in self.license_notices),
        ]
        if len(package_paths) != len(set(package_paths)):
            raise ValueError("package paths must be unique")
        return self


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ReleasePackagingError(f"cannot read release evidence: {path}") from exc


def _load_release(path: Path) -> tuple[ReleaseSpec, str]:
    try:
        content = path.read_bytes()
        payload = yaml.safe_load(content)
        return ReleaseSpec.model_validate(payload), hashlib.sha256(content).hexdigest()
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ReleasePackagingError("release configuration is incomplete or invalid") from exc


def _validate_locked_test(spec: ReleaseSpec) -> None:
    evidence = spec.locked_test
    if evidence.open_count != 1:
        raise ReleasePackagingError("locked test must be opened exactly once for a generation")
    if not evidence.all_hard_gates_passed:
        raise ReleasePackagingError("locked-test hard gates did not all pass")
    if evidence.generation_id != spec.generation_id:
        raise ReleasePackagingError("locked-test generation does not match finalist generation")
    if evidence.dataset_revision != spec.dataset_revision:
        raise ReleasePackagingError("locked-test dataset revision does not match release")


def _validate_claim(spec: ReleaseSpec, claim_path: Path, artifact_sha256: str) -> None:
    if artifact_sha256 != spec.locked_test.claim_sha256:
        raise ReleasePackagingError("locked-test claim hash does not match release evidence")
    try:
        claim = LockedTestClaim.model_validate_json(claim_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ReleasePackagingError("locked-test claim is incomplete or invalid") from exc
    if claim.generation_id != spec.generation_id:
        raise ReleasePackagingError("locked-test claim generation does not match finalist")
    if claim.dataset_revision != spec.dataset_revision:
        raise ReleasePackagingError("locked-test claim dataset does not match finalist")
    if claim.authorization_id != spec.locked_test.authorization_id:
        raise ReleasePackagingError("locked-test claim authorization does not match finalist")


def _validate_locked_metrics(spec: ReleaseSpec, metrics_path: Path, artifact_sha256: str) -> None:
    if artifact_sha256 != spec.locked_test.metrics_sha256:
        raise ReleasePackagingError("locked-test metrics hash does not match release evidence")
    try:
        metrics = LockedTestMetrics.model_validate_json(metrics_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ReleasePackagingError("locked-test metrics are incomplete or invalid") from exc
    if metrics.generation_id != spec.generation_id:
        raise ReleasePackagingError("locked-test metrics generation does not match finalist")
    if metrics.dataset_revision != spec.dataset_revision:
        raise ReleasePackagingError("locked-test metrics dataset does not match finalist")
    if not metrics.all_hard_gates_passed:
        raise ReleasePackagingError("locked-test metrics contain a failed hard gate")
    if metrics.thresholds_sha256 != spec.locked_test.thresholds_sha256:
        raise ReleasePackagingError("locked-test metrics thresholds do not match finalist")


def _approved_assets(path: Path, required_ids: tuple[str, ...]) -> dict[str, dict]:
    payload = _load_yaml(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("assets"), list):
        raise ReleasePackagingError("asset manifest is incomplete or invalid")
    by_id = {
        asset.get("id"): asset
        for asset in payload["assets"]
        if isinstance(asset, Mapping) and isinstance(asset.get("id"), str)
    }
    approved = {}
    for asset_id in required_ids:
        asset = by_id.get(asset_id)
        digest = asset.get("sha256") if asset is not None else None
        if (
            asset is None
            or asset.get("approval_status") != "approved"
            or asset.get("commercial_status") != "allowed"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(asset.get("version"), str)
            or not asset["version"].strip()
            or not isinstance(asset.get("license"), str)
            or not asset["license"].strip()
        ):
            raise ReleasePackagingError(
                f"referenced asset {asset_id!r} is not approved for commercial packaging"
            )
        approved[asset_id] = dict(asset)
    return approved


def _verified_source(source_root: Path, relative_path: str, expected_sha256: str) -> Path:
    root = source_root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ReleasePackagingError(f"release artifact is missing: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ReleasePackagingError(f"release artifact escapes source root: {relative_path}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ReleasePackagingError(f"release artifact hash mismatch: {relative_path}")
    return resolved


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _make_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def package_release(
    *,
    source_root: Path,
    output_root: Path,
    release_config: Path,
    asset_manifest: Path,
) -> Path:
    spec, release_config_sha256 = _load_release(release_config)
    _validate_locked_test(spec)
    assets = _approved_assets(asset_manifest, spec.asset_ids)

    verified_files = [
        (
            artifact.package_path,
            _verified_source(source_root, artifact.source_path, artifact.sha256),
        )
        for artifact in spec.artifacts
    ]
    verified_files.extend(
        (
            notice.package_path,
            _verified_source(source_root, notice.source_path, notice.sha256),
        )
        for notice in spec.license_notices
    )
    claim_artifact = next(
        artifact for artifact in spec.artifacts if artifact.role == "locked_test_claim"
    )
    claim_path = next(
        source
        for package_path, source in verified_files
        if package_path == claim_artifact.package_path
    )
    _validate_claim(spec, claim_path, claim_artifact.sha256)
    metrics_artifact = next(
        artifact for artifact in spec.artifacts if artifact.role == "locked_test_metrics"
    )
    metrics_path = next(
        source
        for package_path, source in verified_files
        if package_path == metrics_artifact.package_path
    )
    _validate_locked_metrics(spec, metrics_path, metrics_artifact.sha256)

    release_dir = output_root / spec.release_id
    if release_dir.exists():
        raise ReleasePackagingError(f"release package already exists: {release_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-{spec.release_id}-{uuid4().hex}"
    temporary.mkdir()
    try:
        for package_path, source in verified_files:
            destination = temporary.joinpath(*PurePosixPath(package_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        manifest = spec.model_dump(mode="json")
        manifest["referenced_assets"] = {
            asset_id: {
                "version": assets[asset_id].get("version"),
                "sha256": assets[asset_id]["sha256"],
                "license": assets[asset_id].get("license"),
            }
            for asset_id in sorted(assets)
        }
        manifest["release_config_sha256"] = release_config_sha256
        (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        _make_immutable(temporary)
        os.replace(temporary, release_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return release_dir
