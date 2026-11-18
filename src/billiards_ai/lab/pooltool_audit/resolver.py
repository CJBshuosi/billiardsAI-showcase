"""Resolve the PyPI portion of pooltool while isolating its Panda3D archive pin."""

from __future__ import annotations

import json
from typing import Protocol, cast

from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from billiards_ai.lab.dependency_audit.index import resolve_target
from billiards_ai.lab.dependency_audit.models import (
    AuditRequest,
    RootRequirement,
    TargetSnapshot,
    requirement_marker_applies,
)
from billiards_ai.lab.dependency_audit.resolver import resolve_dependency_audit
from billiards_ai.lab.pooltool_audit.models import (
    PooltoolAuditRequest,
    PooltoolPyPIClosure,
)

PANDA_REQUIREMENT = 'panda3d==1.11.0.dev3702; sys_platform != "darwin"'
ALLOWED_RESEARCH_LICENSES = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MIT",
    "PSF-2.0",
)


class MetadataClient(Protocol):
    def simple(self, name: str) -> bytes: ...

    def version(self, name: str, version: str) -> bytes: ...


class PooltoolMetadataError(ValueError):
    """Raised when pooltool metadata does not contain the frozen Panda3D pin."""


def _is_frozen_panda_requirement(value: str) -> bool:
    try:
        actual = Requirement(value)
        expected = Requirement(PANDA_REQUIREMENT)
    except InvalidRequirement:
        return False
    return (
        canonicalize_name(actual.name) == canonicalize_name(expected.name)
        and not actual.extras
        and actual.url is None
        and str(actual.specifier) == str(expected.specifier)
        and str(actual.marker) == str(expected.marker)
    )


class _PandaFilteringClient:
    def __init__(self, delegate: MetadataClient, target: TargetSnapshot) -> None:
        self._delegate = delegate
        self._target = target
        self.root_requires_dist: tuple[str, ...] | None = None
        self.external_constraints: set[str] = set()

    def simple(self, name: str) -> bytes:
        return self._delegate.simple(name)

    def version(self, name: str, version: str) -> bytes:
        payload = self._delegate.version(name, version)
        try:
            document = json.loads(payload)
            info = document["info"]
            requirements = info["requires_dist"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise PooltoolMetadataError("pooltool version metadata is malformed") from error
        if requirements is None:
            requirements = []
        if not isinstance(requirements, list) or not all(
            isinstance(item, str) for item in requirements
        ):
            raise PooltoolMetadataError(f"{name} requires_dist is malformed")
        if name == "pooltool-billiards" and version == "0.6.0":
            panda_pins = [item for item in requirements if _is_frozen_panda_requirement(item)]
            if len(panda_pins) != 1:
                raise PooltoolMetadataError(
                    "pooltool Panda3D pin does not match the frozen request"
                )
            self.root_requires_dist = tuple(requirements)
        retained: list[str] = []
        for item in requirements:
            try:
                parsed = Requirement(item)
            except InvalidRequirement as error:
                raise PooltoolMetadataError(f"{name} requires_dist is malformed") from error
            if canonicalize_name(parsed.name) != "panda3d":
                retained.append(item)
                continue
            try:
                active = requirement_marker_applies(parsed, (), self._target)
            except UndefinedEnvironmentName as error:
                raise PooltoolMetadataError(f"{name} Panda3D marker is invalid") from error
            if not active:
                retained.append(item)
                continue
            if not parsed.specifier.contains(Version("1.11.0.dev3702"), prereleases=True):
                raise PooltoolMetadataError(
                    f"frozen Panda3D 1.11.0.dev3702 does not satisfy {name}: {item}"
                )
            self.external_constraints.add(f"{name}=={version}: {item}")
        info["requires_dist"] = retained
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def resolve_pooltool_pypi_closure(
    request: PooltoolAuditRequest,
    client: MetadataClient,
) -> PooltoolPyPIClosure:
    """Resolve every PyPI wheel and retain the one external Panda3D requirement."""

    compatibility_request = AuditRequest.model_construct(
        schema_version="1.0",
        request_id=request.request_id,
        purpose=request.purpose,
        target=request.target,
        roots=(RootRequirement.parse(request.root),),
        allowed_licenses=ALLOWED_RESEARCH_LICENSES,
        single_response_bytes_max=request.single_response_bytes_max,
        total_metadata_bytes_max=request.total_metadata_bytes_max,
        existing_payload_bytes=0,
        installed_size_multipliers={"pure_python": 4.0, "native": 2.5},
        free_space_safety_ratio=0.2,
    )
    target = resolve_target(request.target)
    filtering_client = _PandaFilteringClient(client, target)
    result = resolve_dependency_audit(
        compatibility_request,
        cast("object", filtering_client),
        target,
    )
    if filtering_client.root_requires_dist is None:
        raise PooltoolMetadataError("pooltool root metadata was not resolved")
    return PooltoolPyPIClosure(
        root=request.root,
        panda3d_requirement=PANDA_REQUIREMENT,
        root_requires_dist=filtering_client.root_requires_dist,
        external_panda_constraints=tuple(sorted(filtering_client.external_constraints)),
        packages=result.packages,
        diagnostics=result.diagnostics,
    )
