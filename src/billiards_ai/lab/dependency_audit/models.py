from collections import Counter
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from packaging._parser import Variable
from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT_PARENT_SENTINEL = "<root>"
LICENSE_CLASSIFIER_EXPRESSIONS = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}


class UnavailableTargetMarkerError(ValueError):
    """Raised when a marker needs target kernel/build data that is unavailable."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetPlatform(FrozenModel):
    os: Literal["macos", "windows", "linux"]
    os_version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)*$")
    architecture: Literal["arm64", "x86_64"]
    python_implementation: Literal["cp"]
    python_version: str = Field(pattern=r"^3\.[0-9]+\.[0-9]+$")


class MarkerEnvironment(FrozenModel):
    implementation_name: str
    implementation_version: str
    os_name: str
    platform_machine: str
    platform_python_implementation: str
    platform_release: str | None
    platform_system: str
    platform_version: str | None
    python_full_version: str
    python_version: str
    sys_platform: str


class TargetSnapshot(FrozenModel):
    os: Literal["macos", "windows", "linux"]
    os_version: str
    architecture: Literal["arm64", "x86_64"]
    python_implementation: Literal["cp"]
    python_version: str
    wheel_tags: tuple[str, ...] = Field(min_length=1)
    marker_environment: MarkerEnvironment


class RootRequirement(FrozenModel):
    raw: str
    name: str
    extras: tuple[str, ...]
    specifier: str

    @classmethod
    def parse(cls, value: str) -> "RootRequirement":
        requirement = Requirement(value)
        return cls(
            raw=value,
            name=requirement.name.lower().replace("_", "-"),
            extras=tuple(sorted(requirement.extras)),
            specifier=str(requirement.specifier),
        )

    @model_validator(mode="after")
    def parsed_fields_match_raw_requirement(self) -> "RootRequirement":
        try:
            requirement = Requirement(self.raw)
        except InvalidRequirement as error:
            raise ValueError("root requirement raw value is invalid") from error
        expected = (
            canonicalize_name(requirement.name),
            tuple(sorted(requirement.extras)),
            str(requirement.specifier),
        )
        if (canonicalize_name(self.name), self.extras, self.specifier) != expected:
            raise ValueError("root requirement fields do not match raw requirement")
        return self


class AuditRequest(FrozenModel):
    schema_version: Literal["1.0"]
    request_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    purpose: Literal["synthetic_smoke", "local_training", "cloud_training"]
    target: TargetPlatform
    roots: tuple[RootRequirement, ...] = Field(min_length=1)
    allowed_licenses: tuple[str, ...] = Field(min_length=1)
    single_response_bytes_max: int = Field(gt=0, le=12 * 1024 * 1024)
    total_metadata_bytes_max: int = Field(gt=0, le=128 * 1024 * 1024)
    existing_payload_bytes: int = Field(ge=0)
    installed_size_multipliers: dict[Literal["pure_python", "native"], float]
    free_space_safety_ratio: float = Field(ge=0, le=1)

    @field_validator("roots", mode="before")
    @classmethod
    def parse_roots(
        cls,
        value: list[str | dict[str, object] | RootRequirement],
    ) -> tuple[RootRequirement, ...]:
        return tuple(
            item
            if isinstance(item, RootRequirement)
            else (
                RootRequirement.parse(item)
                if isinstance(item, str)
                else RootRequirement.model_validate(item)
            )
            for item in value
        )

    @field_validator("installed_size_multipliers")
    @classmethod
    def installed_size_multipliers_are_positive(
        cls, value: dict[Literal["pure_python", "native"], float]
    ) -> dict[Literal["pure_python", "native"], float]:
        required = {"pure_python", "native"}
        if set(value) != required or any(multiplier <= 0 for multiplier in value.values()):
            raise ValueError("installed size multipliers must be positive")
        return value

    @model_validator(mode="after")
    def primary_packages_are_pinned(self) -> "AuditRequest":
        primary = tuple(item for item in self.roots if item.name in {"rfdetr", "trackers"})

        def is_concrete_exact_pin(specifier: str) -> bool:
            pins = tuple(SpecifierSet(specifier))
            return len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version

        if not primary or any(not is_concrete_exact_pin(item.specifier) for item in primary):
            raise ValueError("rfdetr and trackers roots must be exactly pinned")
        return self


class WheelArtifact(FrozenModel):
    name: str
    version: str
    filename: str
    url: str
    tags: tuple[str, ...]
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requires_python: str | None
    upload_time: str
    is_pure_python: bool


class DependencyEdge(FrozenModel):
    parent: str
    child: str
    marker: str | None


class LicenseDecision(FrozenModel):
    expression: str | None
    evidence: tuple[str, ...]
    status: Literal["allowed", "legal_review_required", "unknown"]
    reason: (
        Literal[
            "allowlisted",
            "not_allowlisted",
            "conflicting_evidence",
            "missing_or_unrecognized",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def decision_relations_are_consistent(self) -> "LicenseDecision":
        expression = self.expression
        if expression is not None and (not expression.strip() or expression != expression.strip()):
            raise ValueError("license decision expression must be nonempty and normalized")

        expression_evidence = tuple(
            item.removeprefix("license_expression: ")
            for item in self.evidence
            if item.startswith("license_expression: ")
        )
        classifier_evidence = tuple(
            item.removeprefix("classifier: ")
            for item in self.evidence
            if item.startswith("classifier: ")
        )
        matching_expression_evidence = expression is not None and expression in expression_evidence
        matching_classifier_evidence = expression is not None and any(
            LICENSE_CLASSIFIER_EXPRESSIONS.get(classifier) == expression
            for classifier in classifier_evidence
        )
        conflicting_expression_evidence = expression is not None and any(
            item != expression for item in expression_evidence
        )
        conflicting_classifier_evidence = expression is not None and any(
            LICENSE_CLASSIFIER_EXPRESSIONS.get(classifier) != expression
            for classifier in classifier_evidence
        )
        decision_is_supported = (
            matching_expression_evidence or matching_classifier_evidence
        ) and not (conflicting_expression_evidence or conflicting_classifier_evidence)

        if self.status == "allowed":
            if expression is None or self.reason != "allowlisted" or not decision_is_supported:
                raise ValueError("license decision allowed relation is inconsistent")
            return self
        if self.status == "legal_review_required":
            if expression is None or self.reason != "not_allowlisted" or not decision_is_supported:
                raise ValueError("license decision legal review relation is inconsistent")
            return self

        if self.reason not in {"conflicting_evidence", "missing_or_unrecognized"}:
            raise ValueError("license decision unknown relation is inconsistent")
        if self.reason == "missing_or_unrecognized" and expression is not None:
            raise ValueError("license decision missing evidence cannot retain an expression")
        if (
            self.reason == "missing_or_unrecognized"
            and len(classifier_evidence) == 1
            and classifier_evidence[0] in LICENSE_CLASSIFIER_EXPRESSIONS
        ):
            raise ValueError("license decision known classifier cannot be missing evidence")
        if self.reason == "conflicting_evidence":
            if expression is not None:
                if not matching_expression_evidence or not (
                    conflicting_expression_evidence or conflicting_classifier_evidence
                ):
                    raise ValueError("license decision conflict evidence is inconsistent")
            elif not self.evidence:
                raise ValueError("license decision conflict evidence is required")
        return self


class ProjectMetadata(FrozenModel):
    name: str
    version: str
    summary: str | None
    requires_python: str | None
    requires_dist: tuple[str, ...]


class ResolvedPackage(FrozenModel):
    name: str
    version: str
    extras: tuple[str, ...]
    parents: tuple[str, ...] = ()
    wheel: WheelArtifact
    dependencies: tuple[DependencyEdge, ...]
    project_metadata: ProjectMetadata
    license: LicenseDecision


class CandidateDiagnostic(FrozenModel):
    package: str
    filename: str
    category: Literal["excluded", "blocked"]
    reason: str
    detail: str | None = None


class WheelCandidateIndex(FrozenModel):
    candidates: tuple[WheelArtifact, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]


class DependencyAuditResult(FrozenModel):
    target: TargetSnapshot
    packages: tuple[ResolvedPackage, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]


class MetadataAccess(FrozenModel):
    url: str
    content_type: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SizeEstimate(FrozenModel):
    archive_bytes: int = Field(ge=0)
    metadata_bytes: int = Field(ge=0)
    existing_payload_bytes: int = Field(ge=0)
    estimated_installed_bytes: int = Field(ge=0)
    estimated_temporary_cache_peak_bytes: int = Field(ge=0)
    required_free_bytes: int = Field(ge=0)


class SideEffects(FrozenModel):
    archive_downloads: Literal[0] = 0
    installs: Literal[0] = 0
    package_imports: Literal[0] = 0
    weights_loaded: Literal[0] = 0


class AuditBlocker(FrozenModel):
    category: Literal["license", "candidate", "resolution", "provenance"]
    package: str | None
    filename: str | None
    reason: str
    detail: str | None = None


class SourceHash(FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CodeProvenance(FrozenModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    sources: tuple[SourceHash, ...] = Field(min_length=1)


class ToolRuntime(FrozenModel):
    python: str
    packaging: str
    pydantic: str
    pyyaml: str
    pip: str
    pip_vendored_resolvelib: str


class DependencyApprovalPacket(FrozenModel):
    schema_version: Literal["2.0"]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: AuditRequest
    target: TargetSnapshot
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_sources: tuple[SourceHash, ...] = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_runtime: ToolRuntime
    packages: tuple[ResolvedPackage, ...] = Field(min_length=1)
    metadata_accesses: tuple[MetadataAccess, ...] = Field(min_length=1)
    access_hosts: tuple[str, ...]
    size: SizeEstimate
    exclusions: tuple[CandidateDiagnostic, ...]
    blockers: tuple[AuditBlocker, ...]
    license_preflight: Literal["PASS", "BLOCKED"]
    side_effects: SideEffects

    @model_validator(mode="after")
    def approval_relations_are_complete(self) -> "DependencyApprovalPacket":
        target_fields = (
            "os",
            "os_version",
            "architecture",
            "python_implementation",
            "python_version",
        )
        if any(
            getattr(self.request.target, field) != getattr(self.target, field)
            for field in target_fields
        ):
            raise ValueError("target does not match request target")

        canonical_packages = tuple(canonicalize_name(package.name) for package in self.packages)
        if len(canonical_packages) != len(set(canonical_packages)):
            raise ValueError("canonical package names must be unique")
        package_names = set(canonical_packages)
        packages_by_name = dict(zip(canonical_packages, self.packages, strict=True))
        package_identities = {
            name: f"{name}=={package.version}" for name, package in packages_by_name.items()
        }
        active_roots: dict[str, list[Requirement]] = {}
        for root in self.request.roots:
            requirement = Requirement(root.raw)
            if _root_requirement_is_active(requirement, self.target):
                active_roots.setdefault(canonicalize_name(requirement.name), []).append(requirement)
        if not active_roots:
            raise ValueError("approval packet must contain at least one active root")
        missing_roots = sorted(name for name in active_roots if name not in package_names)
        if missing_roots:
            raise ValueError(f"root packages are missing from closure: {', '.join(missing_roots)}")

        for name, requirements in active_roots.items():
            package = packages_by_name[name]
            if ROOT_PARENT_SENTINEL not in package.parents:
                raise ValueError(f"active root package is missing root parent sentinel: {name}")
            try:
                selected_version = Version(package.version)
            except InvalidVersion as error:
                raise ValueError(f"root package version is invalid: {name}") from error
            for requirement in requirements:
                if selected_version not in requirement.specifier:
                    raise ValueError(
                        f"root requirement version is not satisfied by selected package: {name}"
                    )
                if not set(package.extras).issuperset(requirement.extras):
                    raise ValueError(
                        f"root requirement extras are not satisfied by selected package: {name}"
                    )

        graph_edges: set[tuple[str, str]] = set()
        adjacency: dict[str, set[str]] = {name: set() for name in package_names}
        for package, canonical_name in zip(self.packages, canonical_packages, strict=True):
            project = package.project_metadata
            if (
                canonicalize_name(project.name) != canonical_name
                or project.version != package.version
            ):
                raise ValueError(f"project_metadata identity does not match package {package.name}")
            if (
                canonicalize_name(package.wheel.name) != canonical_name
                or package.wheel.version != package.version
            ):
                raise ValueError(f"wheel identity does not match package {package.name}")
            expected_dependencies: Counter[tuple[str, tuple[str, ...], str, str | None]] = Counter()
            for raw_requirement in project.requires_dist:
                try:
                    requirement = Requirement(raw_requirement)
                except InvalidRequirement as error:
                    raise ValueError(
                        f"project requires_dist is invalid for package {package.name}"
                    ) from error
                if requirement.url is not None:
                    raise ValueError(
                        f"project requires_dist direct URL is invalid for package {package.name}"
                    )
                try:
                    active = requirement_marker_applies(
                        requirement,
                        package.extras,
                        self.target,
                    )
                except UndefinedEnvironmentName as error:
                    raise ValueError(
                        f"project requires_dist marker is invalid for package {package.name}"
                    ) from error
                if active:
                    original_marker = raw_requirement.partition(";")[2].strip() or None
                    expected_dependencies[_semantic_requirement(requirement, original_marker)] += 1

            actual_dependencies: Counter[tuple[str, tuple[str, ...], str, str | None]] = Counter()
            for edge in package.dependencies:
                try:
                    edge_requirement = Requirement(edge.child)
                except InvalidRequirement as error:
                    raise ValueError(
                        f"dependency child is outside package closure: {edge.child}"
                    ) from error
                if edge_requirement.url is not None or edge_requirement.marker is not None:
                    raise ValueError(
                        f"dependency edge is not a normalized requires_dist entry: {edge.child}"
                    )
                actual_dependencies[_semantic_requirement(edge_requirement, edge.marker)] += 1
            if expected_dependencies != actual_dependencies:
                raise ValueError(
                    f"project requires_dist does not match dependency edges for {package.name}"
                )

            expected_parent = package_identities[canonical_name]
            for edge in package.dependencies:
                if edge.parent != expected_parent:
                    raise ValueError(
                        f"dependency edge parent does not match containing package: {edge.parent}"
                    )
                try:
                    child_requirement = Requirement(edge.child)
                except InvalidRequirement as error:
                    raise ValueError(
                        f"dependency child is outside package closure: {edge.child}"
                    ) from error
                child_name = canonicalize_name(child_requirement.name)
                child_package = packages_by_name.get(child_name)
                if child_package is None:
                    raise ValueError(f"dependency child is outside package closure: {edge.child}")
                try:
                    child_version = Version(child_package.version)
                except InvalidVersion as error:
                    raise ValueError(f"selected child version is invalid: {child_name}") from error
                if child_version not in child_requirement.specifier:
                    raise ValueError(
                        f"selected child version does not satisfy dependency edge: {edge.child}"
                    )
                if not set(child_package.extras).issuperset(child_requirement.extras):
                    raise ValueError(
                        f"selected child extras do not satisfy dependency edge: {edge.child}"
                    )
                if edge.parent not in child_package.parents:
                    raise ValueError(f"dependency edge is missing from child parents: {edge.child}")
                graph_edges.add((edge.parent, child_name))
                adjacency[canonical_name].add(child_name)

        known_parent_identities = set(package_identities.values())
        for package, canonical_name in zip(self.packages, canonical_packages, strict=True):
            for parent in package.parents:
                if parent == ROOT_PARENT_SENTINEL:
                    if canonical_name not in active_roots:
                        raise ValueError(
                            "root parent sentinel is only allowed for active roots: "
                            f"{canonical_name}"
                        )
                    continue
                if parent not in known_parent_identities:
                    raise ValueError(f"parent reference is outside package closure: {parent}")
                if (parent, canonical_name) not in graph_edges:
                    raise ValueError(
                        f"parent entry has no reciprocal edge: {parent} -> {canonical_name}"
                    )

        reachable = set(active_roots)
        pending = list(active_roots)
        while pending:
            parent = pending.pop()
            for child in adjacency[parent]:
                if child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        unreachable = sorted(package_names - reachable)
        if unreachable:
            raise ValueError(
                f"all packages must be reachable from active roots: {', '.join(unreachable)}"
            )

        derived_hosts = _metadata_access_hosts(self.metadata_accesses)
        if self.access_hosts != derived_hosts:
            raise ValueError("access_hosts do not match metadata_accesses")
        if self.license_preflight == "PASS" and self.access_hosts != ("pypi.org",):
            raise ValueError("access_hosts must be exactly pypi.org for an approvable packet")

        blocked_diagnostics = {
            _diagnostic_relation(item) for item in self.exclusions if item.category == "blocked"
        }
        candidate_blockers = {
            _blocker_relation(item) for item in self.blockers if item.category == "candidate"
        }
        if blocked_diagnostics != candidate_blockers:
            raise ValueError("every blocked diagnostic must have exactly one candidate blocker")

        expected_license_blockers = {
            (
                f"{canonicalize_name(package.name)}=={package.version}",
                (
                    "legal_review_required"
                    if package.license.status == "legal_review_required"
                    else "unknown_license"
                ),
                package.license.reason,
            )
            for package in self.packages
            if package.license.status != "allowed"
        }
        actual_license_blockers = {
            (item.package, item.reason, item.detail)
            for item in self.blockers
            if item.category == "license"
        }
        if expected_license_blockers != actual_license_blockers:
            raise ValueError("license blockers do not match package license decisions")

        expected_preflight = (
            "PASS"
            if not self.blockers
            and all(package.license.status == "allowed" for package in self.packages)
            else "BLOCKED"
        )
        if self.license_preflight != expected_preflight:
            raise ValueError("license_preflight does not match blockers and licenses")
        return self


class DependencyAuditFailure(FrozenModel):
    schema_version: Literal["1.0"]
    approvable: Literal[False]
    failure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: AuditRequest
    target: TargetSnapshot
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_sources: tuple[SourceHash, ...] = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_runtime: ToolRuntime
    metadata_accesses: tuple[MetadataAccess, ...]
    access_hosts: tuple[str, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]
    error_category: Literal["dependency_resolution"]
    error: str = Field(min_length=1)
    side_effects: SideEffects

    @model_validator(mode="after")
    def failure_evidence_is_consistent(self) -> "DependencyAuditFailure":
        target_fields = (
            "os",
            "os_version",
            "architecture",
            "python_implementation",
            "python_version",
        )
        if any(
            getattr(self.request.target, field) != getattr(self.target, field)
            for field in target_fields
        ):
            raise ValueError("target does not match request target")
        if self.access_hosts != _metadata_access_hosts(self.metadata_accesses):
            raise ValueError("access_hosts do not match metadata_accesses")
        return self


def _root_requirement_is_active(
    requirement: Requirement,
    target: TargetSnapshot,
) -> bool:
    try:
        return requirement_marker_applies(
            requirement,
            tuple(sorted(requirement.extras)),
            target,
        )
    except UndefinedEnvironmentName as error:
        raise ValueError("root requirement marker is invalid") from error


def requirement_marker_applies(
    requirement: Requirement,
    extras: tuple[str, ...],
    target: TargetSnapshot,
) -> bool:
    if requirement.marker is None:
        return True
    variables = _marker_variable_names(requirement.marker._markers)
    for name in ("platform_release", "platform_version"):
        if name in variables and getattr(target.marker_environment, name) is None:
            raise UnavailableTargetMarkerError(f"unavailable target marker: {name}")
    selected_extras = extras or ("",)
    environment = target.marker_environment.model_dump(mode="json")
    return any(
        requirement.marker.evaluate({**environment, "extra": extra}) for extra in selected_extras
    )


def _marker_variable_names(value: object) -> set[str]:
    if isinstance(value, Variable):
        return {value.value}
    if isinstance(value, (tuple, list)):
        return set().union(*(_marker_variable_names(item) for item in value))
    return set()


def _semantic_requirement(
    requirement: Requirement,
    original_marker: str | None,
) -> tuple[str, tuple[str, ...], str, str | None]:
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        original_marker,
    )


def _metadata_access_hosts(accesses: tuple[MetadataAccess, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                hostname
                for access in accesses
                if (hostname := urlsplit(access.url).hostname) is not None
            }
        )
    )


def _diagnostic_relation(item: CandidateDiagnostic) -> tuple[str, str, str, str | None]:
    return (canonicalize_name(item.package), item.filename, item.reason, item.detail)


def _blocker_relation(item: AuditBlocker) -> tuple[str, str, str, str | None]:
    return (
        canonicalize_name(item.package or ""),
        item.filename or "",
        item.reason,
        item.detail,
    )


def load_audit_request(path: Path) -> AuditRequest:
    return AuditRequest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
