"""Strict contracts for the isolated pooltool dependency metadata audit."""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from billiards_ai.lab.dependency_audit.models import (
    CandidateDiagnostic,
    ResolvedPackage,
    TargetPlatform,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PooltoolAuditRequest(FrozenModel):
    schema_version: Literal["1.0"]
    request_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    purpose: Literal["synthetic_smoke"]
    target: TargetPlatform
    root: Literal["pooltool-billiards==0.6.0"]
    pypi_index: Literal["https://pypi.org/simple/"]
    panda3d_index: Literal["https://archive.panda3d.org/simple/"]
    single_response_bytes_max: int = Field(gt=0, le=12 * 1024 * 1024)
    total_metadata_bytes_max: int = Field(gt=0, le=128 * 1024 * 1024)

    @model_validator(mode="after")
    def target_is_the_approved_raytheon_runtime(self) -> "PooltoolAuditRequest":
        expected = {
            "os": "windows",
            "os_version": "11.0",
            "architecture": "x86_64",
            "python_implementation": "cp",
            "python_version": "3.12.10",
        }
        actual = self.target.model_dump()
        if actual != expected:
            raise ValueError("target must be the approved Raytheon Windows Python 3.12 runtime")
        return self


class PandaArchiveArtifact(FrozenModel):
    name: Literal["panda3d"] = "panda3d"
    version: Literal["1.11.0.dev3702"] = "1.11.0.dev3702"
    filename: str = Field(pattern=r"^panda3d-1\.11\.0\.dev3702-.*\.whl$")
    url: str
    tags: tuple[str, ...] = Field(min_length=1)
    bytes: int = Field(gt=0)
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    last_modified: str = Field(min_length=1)
    etag: str = Field(min_length=1)
    approval_status: Literal["sha256_required", "hash_complete"]

    @model_validator(mode="after")
    def hash_status_matches_sha256(self) -> "PandaArchiveArtifact":
        expected = "hash_complete" if self.sha256 is not None else "sha256_required"
        if self.approval_status != expected:
            raise ValueError("approval status does not match SHA-256 availability")
        return self


class PooltoolPyPIClosure(FrozenModel):
    root: Literal["pooltool-billiards==0.6.0"]
    panda3d_requirement: Literal['panda3d==1.11.0.dev3702; sys_platform != "darwin"']
    root_requires_dist: tuple[str, ...] = Field(min_length=1)
    external_panda_constraints: tuple[str, ...] = Field(min_length=1)
    packages: tuple[ResolvedPackage, ...] = Field(min_length=1)
    diagnostics: tuple[CandidateDiagnostic, ...]

    @model_validator(mode="after")
    def closure_excludes_only_the_external_panda_payload(self) -> "PooltoolPyPIClosure":
        names = {package.name for package in self.packages}
        if "pooltool-billiards" not in names or "panda3d" in names:
            raise ValueError(
                "PyPI closure must retain pooltool and isolate Panda3D; "
                f"resolved names: {', '.join(sorted(names))}"
            )
        if self.panda3d_requirement not in self.root_requires_dist:
            raise ValueError("root metadata must retain the isolated Panda3D requirement")
        if not any(
            item.startswith("pooltool-billiards==0.6.0: ")
            for item in self.external_panda_constraints
        ):
            raise ValueError("external Panda3D constraints must retain the root pin")
        return self


class ArchiveMetadataAccess(FrozenModel):
    url: Literal["https://archive.panda3d.org/simple/panda3d/"]
    content_type: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PooltoolMetadataDraft(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    request: PooltoolAuditRequest
    closure: PooltoolPyPIClosure
    panda3d: PandaArchiveArtifact
    pypi_metadata_bytes: int = Field(ge=0)
    panda_index_access: ArchiveMetadataAccess
    payload_count: int = Field(gt=0)
    payload_bytes: int = Field(gt=0)
    approval_status: Literal["panda3d_sha256_required"]
    side_effects: dict[str, Literal[0]]

    @model_validator(mode="after")
    def derived_counts_are_exact(self) -> "PooltoolMetadataDraft":
        expected_count = len(self.closure.packages) + 1
        expected_bytes = sum(package.wheel.bytes for package in self.closure.packages)
        expected_bytes += self.panda3d.bytes
        if self.payload_count != expected_count or self.payload_bytes != expected_bytes:
            raise ValueError("payload totals do not match the resolved draft")
        if self.panda3d.sha256 is not None:
            raise ValueError("metadata draft must not claim an unpublished Panda3D SHA-256")
        expected_side_effects = {
            "archive_downloads": 0,
            "installs": 0,
            "package_imports": 0,
            "weights_loaded": 0,
        }
        if self.side_effects != expected_side_effects:
            raise ValueError("metadata draft side effects must all be zero")
        return self


class PandaHashEvidence(FrozenModel):
    draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str
    filename: str = Field(pattern=r"^panda3d-1\.11\.0\.dev3702-.*\.whl$")
    bytes: int = Field(gt=0)
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_path: Path


class PooltoolPacketSideEffects(FrozenModel):
    archive_downloads: Literal[1]
    installs: Literal[0]
    package_imports: Literal[0]
    weights_loaded: Literal[0]


class PooltoolApprovalPacket(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: PooltoolAuditRequest
    closure: PooltoolPyPIClosure
    panda3d: PandaArchiveArtifact
    pypi_metadata_bytes: int = Field(ge=0)
    panda_index_access: ArchiveMetadataAccess
    payload_count: int = Field(gt=0)
    payload_bytes: int = Field(gt=0)
    approval_status: Literal["hash_complete"]
    side_effects: PooltoolPacketSideEffects

    @model_validator(mode="after")
    def packet_is_fully_hashed(self) -> "PooltoolApprovalPacket":
        expected_count = len(self.closure.packages) + 1
        expected_bytes = sum(package.wheel.bytes for package in self.closure.packages)
        expected_bytes += self.panda3d.bytes
        if self.payload_count != expected_count or self.payload_bytes != expected_bytes:
            raise ValueError("packet payload totals do not match the resolved artifacts")
        if self.panda3d.sha256 is None or self.panda3d.approval_status != "hash_complete":
            raise ValueError("packet Panda3D artifact must have a verified SHA-256")
        return self


def archive_access(payload: bytes, content_type: str) -> ArchiveMetadataAccess:
    return ArchiveMetadataAccess(
        url="https://archive.panda3d.org/simple/panda3d/",
        content_type=content_type,
        bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
    )
