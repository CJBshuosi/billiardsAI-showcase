"""Typed, non-approvable evidence for the PyTorch CUDA HTML indexes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.dependency_audit.models import FrozenModel, MetadataAccess, SideEffects

_INDEX_URLS = {
    "https://download.pytorch.org/whl/cu126/torch/",
    "https://download.pytorch.org/whl/cu126/torchvision/",
}


def _sorted_unique_by[CollectionItem](
    values: tuple[CollectionItem, ...],
    *,
    key: str,
    collection: str,
) -> tuple[CollectionItem, ...]:
    identities = tuple(getattr(value, key) for value in values)
    if len(identities) != len(set(identities)):
        raise ValueError(f"{collection} contains a duplicate {key}")
    return tuple(sorted(values, key=lambda value: getattr(value, key)))


class IndexAccess(FrozenModel):
    url: str
    status_code: int = 200
    content_type: str = "text/html"
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("url")
    @classmethod
    def is_allowlisted_index(cls, value: str) -> str:
        if value not in _INDEX_URLS:
            raise ValueError("PyTorch index URL is not allowlisted")
        return value


class WheelObservation(FrozenModel):
    package: Literal["torch", "torchvision"]
    filename: str
    archive_url: str
    sha256_fragment: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def has_exact_allowlisted_path(self) -> WheelObservation:
        expected = {
            "torch": "torch-2.13.0+cu126-cp312-cp312-win_amd64.whl",
            "torchvision": "torchvision-0.28.0+cu126-cp312-cp312-win_amd64.whl",
        }
        parsed = urlsplit(self.archive_url)
        if expected.get(self.package) != self.filename or parsed.scheme != "https":
            raise ValueError("wheel observation is not an expected CUDA target")
        if parsed.hostname != "download-r2.pytorch.org" or parsed.query or parsed.fragment:
            raise ValueError("wheel archive URL is not allowlisted")
        if not parsed.path.endswith(self.filename.replace("+", "%2B")):
            raise ValueError("wheel archive URL does not bind filename")
        return self


class LicenseCommercialState(FrozenModel):
    package: Literal["torch", "torchvision"]
    license_conclusion: Literal["unresolved"]
    commercial_conclusion: Literal["unresolved"]
    acquisition_status: Literal["forbidden"]


class AuditFailureSummary(FrozenModel):
    approvable: Literal[False]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    error: str = Field(min_length=1)
    error_category: Literal["dependency_resolution"]
    failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_accesses: tuple[MetadataAccess, ...]

    @field_validator("metadata_accesses")
    @classmethod
    def canonicalize_metadata_accesses(
        cls, values: tuple[MetadataAccess, ...]
    ) -> tuple[MetadataAccess, ...]:
        return _sorted_unique_by(
            values,
            key="url",
            collection="audit failure metadata_accesses",
        )


class KnownPartialBytes(FrozenModel):
    complete_closure_disk_bytes: None = None
    complete_closure_installed_bytes: None = None
    complete_closure_network_bytes: None = None
    pypi_metadata_bytes: int = Field(ge=0)
    pytorch_index_bytes: int = Field(ge=0)


class WindowsCudaEvidence(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    closure_status: Literal["unresolved"]
    acquisition_status: Literal["forbidden"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_failure_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_failure: AuditFailureSummary
    index_accesses: tuple[IndexAccess, ...]
    wheel_observations: tuple[WheelObservation, ...]
    license_commercial_states: tuple[LicenseCommercialState, ...]
    known_partial_bytes: KnownPartialBytes
    side_effects: SideEffects

    @field_validator("index_accesses")
    @classmethod
    def canonicalize_index_accesses(
        cls, values: tuple[IndexAccess, ...]
    ) -> tuple[IndexAccess, ...]:
        return _sorted_unique_by(values, key="url", collection="index_accesses")

    @field_validator("wheel_observations")
    @classmethod
    def canonicalize_wheel_observations(
        cls, values: tuple[WheelObservation, ...]
    ) -> tuple[WheelObservation, ...]:
        return _sorted_unique_by(values, key="package", collection="wheel_observations")

    @field_validator("license_commercial_states")
    @classmethod
    def canonicalize_license_commercial_states(
        cls, values: tuple[LicenseCommercialState, ...]
    ) -> tuple[LicenseCommercialState, ...]:
        return _sorted_unique_by(
            values,
            key="package",
            collection="license_commercial_states",
        )

    @model_validator(mode="after")
    def stays_non_approvable(self) -> WindowsCudaEvidence:
        if self.audit_failure.failure_sha256 != self.audit_failure_sha256:
            raise ValueError("audit failure hash does not bind failure evidence")
        if self.audit_failure.config_sha256 != self.request_sha256:
            raise ValueError("request hash does not bind audit failure configuration")
        if self.side_effects != SideEffects():
            raise ValueError("evidence must record zero execution side effects")
        return self


def canonical_evidence_json(value: object) -> bytes:
    """Validate then canonically serialize non-approvable index evidence."""

    evidence = WindowsCudaEvidence.model_validate(value)
    return (
        json.dumps(evidence.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def parse_target_anchor(body: bytes, access: IndexAccess, package: str) -> WheelObservation:
    """Re-parse one retained official index body; no network or payload access."""

    actual_sha256 = hashlib.sha256(body).hexdigest()
    if len(body) != access.bytes or actual_sha256 != access.sha256:
        raise ValueError("retained index body does not match its recorded access")
    version = "2.13.0" if package == "torch" else "0.28.0"
    filename = f"{package}-{version}%2Bcu126-cp312-cp312-win_amd64.whl"
    pattern = (
        rf'href="(?P<url>https://[^"#]+{re.escape(filename)})'
        r'(?:#sha256=(?P<sha>[0-9a-f]{64}))?"'
    )
    match = re.search(pattern, body.decode("utf-8"))
    if match is None:
        raise ValueError("target wheel anchor is absent from retained index body")
    return WheelObservation(
        package=package,
        filename=filename.replace("%2B", "+"),
        archive_url=match.group("url"),
        sha256_fragment=match.group("sha"),
    )
