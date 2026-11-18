"""Offline dependency resolution and conservative licence classification."""

import json
import re
from dataclasses import dataclass
from importlib import metadata, resources
from typing import Any

import pip
from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pip._vendor import resolvelib

from billiards_ai.lab.dependency_audit.index import (
    WheelIndexError,
    parse_simple_index,
    resolve_target,
)
from billiards_ai.lab.dependency_audit.models import (
    LICENSE_CLASSIFIER_EXPRESSIONS,
    ROOT_PARENT_SENTINEL,
    AuditRequest,
    CandidateDiagnostic,
    DependencyAuditResult,
    DependencyEdge,
    LicenseDecision,
    ProjectMetadata,
    ResolvedPackage,
    TargetSnapshot,
    UnavailableTargetMarkerError,
    WheelArtifact,
    WheelCandidateIndex,
    requirement_marker_applies,
)
from billiards_ai.lab.dependency_audit.transport import (
    MetadataSecurityError,
    MetadataTransportError,
    PyPIMetadataClient,
)


class DependencyResolutionError(RuntimeError):
    """Raised when metadata cannot produce a complete, safe dependency graph."""

    def __init__(
        self,
        message: str,
        diagnostics: tuple[CandidateDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class _MetadataError(ValueError):
    pass


def _diagnostic_key(item: CandidateDiagnostic) -> tuple[str, str, str, str, str]:
    return (
        canonicalize_name(item.package),
        item.filename,
        item.category,
        item.reason,
        item.detail or "",
    )


def _merge_diagnostics(
    *groups: tuple[CandidateDiagnostic, ...],
) -> tuple[CandidateDiagnostic, ...]:
    by_key = {_diagnostic_key(item): item for group in groups for item in group}
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True, slots=True)
class _ResolverRequirement:
    parsed: Requirement
    display: str

    @property
    def name(self) -> str:
        return canonicalize_name(self.parsed.name)


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    version: str
    extras: tuple[str, ...]
    wheel: WheelArtifact


def _vendored_resolvelib_version() -> str:
    direct = getattr(resolvelib, "__version__", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    try:
        vendor_manifest = resources.files("pip._vendor").joinpath("vendor.txt").read_text("utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        vendor_manifest = ""
    match = re.search(r"(?m)^resolvelib==([^\s]+)$", vendor_manifest)
    if match is None:
        raise DependencyResolutionError(
            "installed pip does not expose its vendored resolvelib version"
        )
    return match.group(1)


def resolver_runtime_versions() -> dict[str, str]:
    """Return exact resolver-owned runtime versions for approval-packet evidence.

    Pip's bundled ``vendor.txt`` manifest is the deterministic fallback when the
    vendored resolvelib module does not expose ``__version__`` directly.
    """

    pip_version = getattr(pip, "__version__", None)
    if not isinstance(pip_version, str) or not pip_version.strip():
        try:
            pip_version = metadata.version("pip")
        except metadata.PackageNotFoundError as error:
            raise DependencyResolutionError("installed pip version is unavailable") from error
    return {
        "pip": pip_version.strip(),
        "pip_vendored_resolvelib": _vendored_resolvelib_version(),
    }


def _license_classifiers(metadata_record: dict[str, object]) -> tuple[str, ...]:
    raw_classifiers = metadata_record.get("classifiers", [])
    if not isinstance(raw_classifiers, list) or not all(
        isinstance(classifier, str) for classifier in raw_classifiers
    ):
        raise _MetadataError("classifiers must be a list of strings")
    return tuple(
        classifier for classifier in raw_classifiers if classifier.startswith("License :: ")
    )


def classify_license(
    metadata_record: dict[str, object], allowed_licenses: tuple[str, ...]
) -> LicenseDecision:
    """Classify standard licence evidence without interpreting SPDX expressions."""

    expression = metadata_record.get("license_expression")
    if expression is not None and not isinstance(expression, str):
        raise _MetadataError("license_expression must be a string or null")
    classifiers = _license_classifiers(metadata_record)
    free_text = metadata_record.get("license")
    if free_text is not None and not isinstance(free_text, str):
        raise _MetadataError("license must be a string or null")
    evidence: list[str] = []

    if isinstance(expression, str) and expression.strip():
        expression = expression.strip()
        evidence.append(f"license_expression: {expression}")
        evidence.extend(f"classifier: {classifier}" for classifier in classifiers)
        if isinstance(free_text, str) and free_text.strip():
            evidence.append(f"license: {free_text.strip()}")
        if any(
            LICENSE_CLASSIFIER_EXPRESSIONS.get(classifier) != expression
            for classifier in classifiers
        ):
            return LicenseDecision(
                expression=expression,
                evidence=tuple(evidence),
                status="unknown",
                reason="conflicting_evidence",
            )
        status = "allowed" if expression in allowed_licenses else "legal_review_required"
        return LicenseDecision(
            expression=expression,
            evidence=tuple(evidence),
            status=status,
            reason="allowlisted" if status == "allowed" else "not_allowlisted",
        )

    evidence.extend(f"classifier: {classifier}" for classifier in classifiers)
    if len(classifiers) == 1:
        classifier_expression = LICENSE_CLASSIFIER_EXPRESSIONS.get(classifiers[0])
    else:
        classifier_expression = None
    if classifier_expression is not None:
        status = "allowed" if classifier_expression in allowed_licenses else "legal_review_required"
        return LicenseDecision(
            expression=classifier_expression,
            evidence=tuple(evidence),
            status=status,
            reason="allowlisted" if status == "allowed" else "not_allowlisted",
        )

    if isinstance(free_text, str) and free_text.strip():
        evidence.append(f"license: {free_text.strip()}")
    return LicenseDecision(
        expression=None,
        evidence=tuple(evidence),
        status="unknown",
        reason="missing_or_unrecognized",
    )


def _base_requirement(parsed: Requirement) -> str:
    name = canonicalize_name(parsed.name)
    extras = f"[{','.join(sorted(parsed.extras))}]" if parsed.extras else ""
    return f"{name}{extras}{parsed.specifier}"


def _marker_applies(
    parsed: Requirement,
    extras: tuple[str, ...],
    target: TargetSnapshot,
) -> bool:
    try:
        return requirement_marker_applies(parsed, extras, target)
    except UnavailableTargetMarkerError as error:
        raise DependencyResolutionError(str(error)) from error


class _ResolverProvider(resolvelib.AbstractProvider):
    def __init__(
        self,
        request: AuditRequest,
        client: PyPIMetadataClient,
        target: TargetSnapshot,
    ) -> None:
        self.request = request
        self.client = client
        self.target = target
        self.context = "root requirements"
        self._indexes: dict[str, WheelCandidateIndex] = {}
        self._metadata: dict[tuple[str, str], dict[str, object]] = {}
        self._dependencies: dict[_Candidate, tuple[_ResolverRequirement, ...]] = {}
        self._edges: dict[_Candidate, tuple[DependencyEdge, ...]] = {}

    def identify(self, requirement_or_candidate: _ResolverRequirement | _Candidate) -> str:
        return canonicalize_name(requirement_or_candidate.name)

    def get_preference(
        self,
        identifier: str,
        resolutions: Any,
        candidates: Any,
        information: Any,
        backtrack_causes: Any,
    ) -> str:
        return identifier

    def find_matches(
        self,
        identifier: str,
        requirements: Any,
        incompatibilities: Any,
    ) -> tuple[_Candidate, ...]:
        active_requirements = tuple(requirements[identifier])
        constraints = tuple(sorted(item.display for item in active_requirements))
        self.context = f"{identifier} constraints [{', '.join(constraints)}]"
        extras = tuple(
            sorted({extra for item in active_requirements for extra in item.parsed.extras})
        )
        if identifier not in self._indexes:
            self._indexes[identifier] = parse_simple_index(
                self.client.simple(identifier),
                self.target,
                expected_package=identifier,
            )
            if any(wheel.name != identifier for wheel in self._indexes[identifier].candidates):
                raise _MetadataError(
                    f"{identifier}: simple metadata package name does not match request"
                )
        rejected = set(incompatibilities[identifier])
        matches = []
        for wheel in self._indexes[identifier].candidates:
            candidate = _Candidate(identifier, wheel.version, extras, wheel)
            if candidate in rejected:
                continue
            if all(
                self.is_satisfied_by(requirement, candidate) for requirement in active_requirements
            ):
                matches.append(candidate)
        return tuple(matches)

    def is_satisfied_by(self, requirement: _ResolverRequirement, candidate: _Candidate) -> bool:
        return (
            requirement.name == candidate.name
            and Version(candidate.version) in requirement.parsed.specifier
            and set(candidate.extras).issuperset(requirement.parsed.extras)
        )

    def get_dependencies(self, candidate: _Candidate) -> tuple[_ResolverRequirement, ...]:
        if candidate in self._dependencies:
            return self._dependencies[candidate]
        info = self.metadata_for(candidate)
        raw_dependencies = info.get("requires_dist")
        if raw_dependencies is None:
            raw_dependencies = []
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_dependencies
        ):
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: requires_dist must be a list of strings"
            )

        dependencies: list[_ResolverRequirement] = []
        edges: list[DependencyEdge] = []
        for raw_requirement in raw_dependencies:
            try:
                parsed = Requirement(raw_requirement)
            except InvalidRequirement as error:
                raise _MetadataError(
                    f"{candidate.name}=={candidate.version}: malformed requirement "
                    f"{raw_requirement!r}"
                ) from error
            if parsed.url is not None:
                raise _MetadataError(
                    f"{candidate.name}=={candidate.version}: direct URL requirement is not allowed"
                )
            try:
                active = _marker_applies(parsed, candidate.extras, self.target)
            except UndefinedEnvironmentName as error:
                raise _MetadataError(
                    f"{candidate.name}=={candidate.version}: requirement marker is invalid"
                ) from error
            if not active:
                continue
            display = _base_requirement(parsed)
            original_marker = raw_requirement.partition(";")[2].strip() or None
            dependencies.append(_ResolverRequirement(Requirement(display), display))
            edges.append(
                DependencyEdge(
                    parent=f"{candidate.name}=={candidate.version}",
                    child=display,
                    marker=original_marker,
                )
            )
        self._dependencies[candidate] = tuple(dependencies)
        self._edges[candidate] = tuple(
            sorted(edges, key=lambda edge: (edge.parent, edge.child, edge.marker or ""))
        )
        return self._dependencies[candidate]

    def metadata_for(self, candidate: _Candidate) -> dict[str, object]:
        key = (candidate.name, candidate.version)
        self.context = f"{candidate.name}=={candidate.version} version metadata"
        if key in self._metadata:
            return self._metadata[key]
        try:
            document = json.loads(self.client.version(candidate.name, candidate.version))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: invalid version metadata JSON"
            ) from error
        if not isinstance(document, dict) or not isinstance(document.get("info"), dict):
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: version metadata info is required"
            )
        info = document["info"]
        metadata_name = info.get("name")
        metadata_version = info.get("version")
        if not isinstance(metadata_name, str) or canonicalize_name(metadata_name) != candidate.name:
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: metadata package name does not match"
            )
        try:
            normalized_version = str(Version(metadata_version))
        except (InvalidVersion, TypeError) as error:
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: metadata version is invalid"
            ) from error
        if normalized_version != candidate.version:
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: metadata version does not match"
            )
        _license_classifiers(info)
        self._metadata[key] = info
        return info

    def edges_for(self, candidate: _Candidate) -> tuple[DependencyEdge, ...]:
        self.get_dependencies(candidate)
        return self._edges[candidate]

    def project_metadata_for(self, candidate: _Candidate) -> ProjectMetadata:
        info = self.metadata_for(candidate)
        summary = info.get("summary")
        requires_python = info.get("requires_python")
        raw_dependencies = info.get("requires_dist")
        if raw_dependencies is None:
            raw_dependencies = []
        if summary is not None and not isinstance(summary, str):
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: summary must be a string or null"
            )
        if requires_python is not None and not isinstance(requires_python, str):
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: requires_python must be a string or null"
            )
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_dependencies
        ):
            raise _MetadataError(
                f"{candidate.name}=={candidate.version}: requires_dist must be a list of strings"
            )
        return ProjectMetadata(
            name=candidate.name,
            version=candidate.version,
            summary=summary,
            requires_python=requires_python,
            requires_dist=tuple(raw_dependencies),
        )

    def diagnostics_for(
        self,
        mapping: dict[str, _Candidate],
    ) -> tuple[CandidateDiagnostic, ...]:
        diagnostics = [
            diagnostic
            for identifier in sorted(self._indexes)
            for diagnostic in self._indexes[identifier].diagnostics
        ]
        for identifier, index in sorted(self._indexes.items()):
            selected = mapping.get(identifier)
            if selected is None:
                continue
            for candidate in index.candidates:
                if candidate.filename == selected.wheel.filename:
                    continue
                diagnostics.append(
                    CandidateDiagnostic(
                        package=identifier,
                        filename=candidate.filename,
                        category="excluded",
                        reason="not_selected",
                        detail=f"selected {selected.wheel.filename}",
                    )
                )
        return tuple(sorted(diagnostics, key=_diagnostic_key))

    @property
    def accumulated_diagnostics(self) -> tuple[CandidateDiagnostic, ...]:
        return tuple(
            diagnostic
            for identifier in sorted(self._indexes)
            for diagnostic in self._indexes[identifier].diagnostics
        )


def _root_requirements(
    request: AuditRequest,
    target: TargetSnapshot,
) -> tuple[_ResolverRequirement, ...]:
    requirements: list[_ResolverRequirement] = []
    for root in request.roots:
        try:
            parsed = Requirement(root.raw)
        except InvalidRequirement as error:
            raise DependencyResolutionError(f"malformed root requirement {root.raw!r}") from error
        if parsed.url is not None:
            raise DependencyResolutionError(
                f"root requirement {root.raw!r} uses a disallowed direct URL"
            )
        try:
            if not _marker_applies(parsed, tuple(sorted(parsed.extras)), target):
                continue
        except UndefinedEnvironmentName as error:
            raise DependencyResolutionError(
                f"root requirement {root.raw!r} has an invalid marker"
            ) from error
        display = _base_requirement(parsed)
        requirements.append(_ResolverRequirement(Requirement(display), display))
    if not requirements:
        raise DependencyResolutionError("no root requirements are active for the frozen target")
    return tuple(requirements)


def _impossible_message(error: resolvelib.ResolutionImpossible) -> str:
    constraints = sorted(
        {
            cause.requirement.display
            for cause in error.causes
            if isinstance(cause.requirement, _ResolverRequirement)
        }
    )
    packages = sorted(
        {
            cause.requirement.name
            for cause in error.causes
            if isinstance(cause.requirement, _ResolverRequirement)
        }
    )
    package_context = ", ".join(packages) or "unknown package"
    return (
        f"dependency resolution impossible for {package_context}; constraints "
        f"[{', '.join(constraints)}]"
    )


def _target_matches_request(request: AuditRequest, target: TargetSnapshot) -> bool:
    return all(
        getattr(request.target, field) == getattr(target, field)
        for field in (
            "os",
            "os_version",
            "architecture",
            "python_implementation",
            "python_version",
        )
    )


def resolve_dependency_audit(
    request: AuditRequest,
    client: PyPIMetadataClient,
    target: TargetSnapshot | None = None,
) -> DependencyAuditResult:
    """Resolve a complete graph and retain its exact target and candidate evidence."""

    frozen_target = target or resolve_target(request.target)
    if not _target_matches_request(request, frozen_target):
        raise DependencyResolutionError("resolved target does not match the audit request")
    provider = _ResolverProvider(request, client, frozen_target)
    resolver = resolvelib.Resolver(provider, resolvelib.BaseReporter())
    try:
        root_requirements = _root_requirements(request, frozen_target)
        result = resolver.resolve(root_requirements, max_rounds=100)
        mapping = dict(result.mapping)
        parents: dict[str, set[str]] = {identifier: set() for identifier in mapping}
        for root in root_requirements:
            if root.name in parents:
                parents[root.name].add(ROOT_PARENT_SENTINEL)
        for candidate in mapping.values():
            for edge in provider.edges_for(candidate):
                child = canonicalize_name(Requirement(edge.child).name)
                if child in parents:
                    parents[child].add(edge.parent)
        packages: list[ResolvedPackage] = []
        for candidate in sorted(mapping.values(), key=lambda item: item.name):
            info = provider.metadata_for(candidate)
            packages.append(
                ResolvedPackage(
                    name=candidate.name,
                    version=candidate.version,
                    extras=candidate.extras,
                    parents=tuple(sorted(parents[candidate.name])),
                    wheel=candidate.wheel,
                    dependencies=provider.edges_for(candidate),
                    project_metadata=provider.project_metadata_for(candidate),
                    license=classify_license(info, request.allowed_licenses),
                )
            )
        return DependencyAuditResult(
            target=frozen_target,
            packages=tuple(packages),
            diagnostics=provider.diagnostics_for(mapping),
        )
    except resolvelib.ResolutionImpossible as error:
        raise DependencyResolutionError(
            _impossible_message(error),
            provider.accumulated_diagnostics,
        ) from error
    except resolvelib.ResolutionTooDeep as error:
        roots = ", ".join(sorted(root.raw for root in request.roots))
        raise DependencyResolutionError(
            f"dependency resolution too deep after {error.round_count} rounds; roots [{roots}]",
            provider.accumulated_diagnostics,
        ) from error
    except DependencyResolutionError as error:
        raise DependencyResolutionError(
            str(error),
            _merge_diagnostics(provider.accumulated_diagnostics, error.diagnostics),
        ) from error
    except (
        InvalidRequirement,
        InvalidVersion,
        MetadataSecurityError,
        MetadataTransportError,
        WheelIndexError,
        _MetadataError,
    ) as error:
        diagnostics = _merge_diagnostics(
            provider.accumulated_diagnostics,
            error.diagnostics if isinstance(error, WheelIndexError) else (),
        )
        raise DependencyResolutionError(
            f"dependency resolution failed for {provider.context}: {error}",
            diagnostics,
        ) from error


def resolve_dependencies(
    request: AuditRequest, client: PyPIMetadataClient
) -> tuple[ResolvedPackage, ...]:
    """Compatibility wrapper returning only the resolved package tuple."""

    return resolve_dependency_audit(request, client).packages
