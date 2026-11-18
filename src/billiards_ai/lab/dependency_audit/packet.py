"""Deterministic construction and immutable publication of approval packets."""

import ctypes
import errno
import hashlib
import hmac
import html
import json
import math
import os
import stat
import sys
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name
from pydantic import BaseModel

from billiards_ai.lab.dependency_audit.models import (
    AuditBlocker,
    AuditRequest,
    CandidateDiagnostic,
    CodeProvenance,
    DependencyApprovalPacket,
    DependencyAuditFailure,
    DependencyAuditResult,
    DependencyEdge,
    MetadataAccess,
    ResolvedPackage,
    SideEffects,
    SizeEstimate,
    SourceHash,
    TargetSnapshot,
    ToolRuntime,
)
from billiards_ai.lab.dependency_audit.resolver import (
    DependencyResolutionError,
    resolver_runtime_versions,
)


class PacketIntegrityError(RuntimeError):
    """Raised when a content-addressed packet does not match its expected bytes."""


def _revalidate_model[ModelT: BaseModel](
    model_type: type[ModelT],
    value: object,
    field_name: str,
) -> ModelT:
    if not isinstance(value, model_type):
        raise TypeError(f"{field_name} must be {model_type.__name__}")
    return model_type.model_validate(value.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class _OwnedFile:
    descriptor: int | None
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _CleanupCapabilities:
    descriptor_relative: bool
    directory_open_flags: int | None
    file_nofollow_flag: int | None


@dataclass(slots=True)
class _OwnedTemporary:
    parent: Path
    name: str
    parent_device: int
    parent_inode: int
    directory_device: int
    directory_inode: int
    capabilities: _CleanupCapabilities
    parent_descriptor: int | None
    directory_descriptor: int | None
    files: dict[str, _OwnedFile] = field(default_factory=dict)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _edge_key(edge: DependencyEdge) -> tuple[str, str, bool, str]:
    return (edge.parent, edge.child, edge.marker is not None, edge.marker or "")


def _package_key(package: ResolvedPackage) -> tuple[str, bytes]:
    return (
        canonicalize_name(package.name),
        _canonical_json(package.model_dump(mode="json")),
    )


def _access_key(access: MetadataAccess) -> tuple[str, bytes]:
    return (access.url, _canonical_json(access.model_dump(mode="json")))


def _sorted_packages(packages: Iterable[ResolvedPackage]) -> tuple[ResolvedPackage, ...]:
    normalized = (
        package.model_copy(
            update={
                "dependencies": tuple(sorted(package.dependencies, key=_edge_key)),
                "extras": tuple(sorted(package.extras)),
                "parents": tuple(sorted(package.parents)),
            }
        )
        for package in packages
    )
    return tuple(sorted(normalized, key=_package_key))


def _diagnostic_key(item: CandidateDiagnostic) -> tuple[str, str, str, str, str]:
    return (
        canonicalize_name(item.package),
        item.filename,
        item.category,
        item.reason,
        item.detail or "",
    )


def _blocker_key(item: AuditBlocker) -> tuple[str, str, str, str, str]:
    return (
        item.category,
        item.package or "",
        item.filename or "",
        item.reason,
        item.detail or "",
    )


def _source_key(item: SourceHash) -> tuple[str, str]:
    return (item.path, item.sha256)


def tool_runtime_versions() -> ToolRuntime:
    """Return exact typed versions for every package used by the auditor boundary."""

    resolver_versions = resolver_runtime_versions()
    return ToolRuntime(
        python=".".join(str(item) for item in sys.version_info[:3]),
        packaging=metadata.version("packaging"),
        pydantic=metadata.version("pydantic"),
        pyyaml=metadata.version("PyYAML"),
        pip=resolver_versions["pip"],
        pip_vendored_resolvelib=resolver_versions["pip_vendored_resolvelib"],
    )


def _packet_payload_json(packet: DependencyApprovalPacket) -> bytes:
    return _canonical_json(packet.model_dump(mode="json", exclude={"packet_sha256"}))


def canonical_packet_json(packet: DependencyApprovalPacket) -> bytes:
    """Return compact canonical UTF-8 JSON, including one trailing newline."""

    return _canonical_json(packet.model_dump(mode="json"))


def _failure_payload_json(failure: DependencyAuditFailure) -> bytes:
    return _canonical_json(failure.model_dump(mode="json", exclude={"failure_sha256"}))


def canonical_failure_json(failure: DependencyAuditFailure) -> bytes:
    """Return compact canonical failure JSON, including one trailing newline."""

    return _canonical_json(failure.model_dump(mode="json"))


def build_dependency_audit_failure(
    request: AuditRequest,
    target: TargetSnapshot,
    metadata_accesses: Iterable[MetadataAccess],
    error: DependencyResolutionError,
    code_commit: str,
    config_sha256: str,
    *,
    code_sources: Iterable[SourceHash],
    tool_runtime: ToolRuntime | None = None,
) -> DependencyAuditFailure:
    """Build immutable, explicitly non-approvable resolution-failure evidence."""

    if not isinstance(error, DependencyResolutionError):
        raise TypeError("error must be DependencyResolutionError")
    request = _revalidate_model(AuditRequest, request, "request")
    target = _revalidate_model(TargetSnapshot, target, "target")
    sorted_accesses = tuple(
        sorted(
            (
                _revalidate_model(MetadataAccess, access, "metadata access")
                for access in metadata_accesses
            ),
            key=_access_key,
        )
    )
    sorted_sources = tuple(
        sorted(
            (_revalidate_model(SourceHash, source, "code source") for source in code_sources),
            key=_source_key,
        )
    )
    provenance = CodeProvenance.model_validate(
        {
            "commit": code_commit,
            "sources": [source.model_dump(mode="json") for source in sorted_sources],
        }
    )
    diagnostics = tuple(
        sorted(
            (
                _revalidate_model(CandidateDiagnostic, item, "diagnostic")
                for item in error.diagnostics
            ),
            key=_diagnostic_key,
        )
    )
    validated_runtime = _revalidate_model(
        ToolRuntime,
        tool_runtime or tool_runtime_versions(),
        "tool runtime",
    )
    access_hosts = tuple(
        sorted(
            {
                hostname
                for access in sorted_accesses
                if (hostname := urlsplit(access.url).hostname) is not None
            }
        )
    )
    failure = DependencyAuditFailure(
        schema_version="1.0",
        approvable=False,
        failure_sha256="0" * 64,
        request=request,
        target=target,
        code_commit=provenance.commit,
        code_sources=provenance.sources,
        config_sha256=config_sha256,
        tool_runtime=validated_runtime,
        metadata_accesses=sorted_accesses,
        access_hosts=access_hosts,
        diagnostics=diagnostics,
        error_category="dependency_resolution",
        error=str(error),
        side_effects=SideEffects(),
    )
    failure_sha256 = hashlib.sha256(_failure_payload_json(failure)).hexdigest()
    document = failure.model_dump(mode="json")
    document["failure_sha256"] = failure_sha256
    return DependencyAuditFailure.model_validate(document)


def build_approval_packet(
    request: AuditRequest,
    audit: DependencyAuditResult,
    metadata_accesses: Iterable[MetadataAccess],
    code_commit: str,
    config_sha256: str,
    *,
    code_sources: Iterable[SourceHash],
    tool_runtime: ToolRuntime | None = None,
) -> DependencyApprovalPacket:
    """Build a canonical packet from exact source provenance and metadata evidence."""

    if not isinstance(audit, DependencyAuditResult):
        raise TypeError("audit must be a complete DependencyAuditResult")
    request = _revalidate_model(AuditRequest, request, "request")
    audit = _revalidate_model(DependencyAuditResult, audit, "audit")
    validated_accesses = tuple(
        _revalidate_model(MetadataAccess, access, "metadata access") for access in metadata_accesses
    )
    validated_sources = tuple(
        _revalidate_model(SourceHash, source, "code source") for source in code_sources
    )
    provenance = CodeProvenance.model_validate(
        {
            "commit": code_commit,
            "sources": [source.model_dump(mode="json") for source in validated_sources],
        }
    )
    validated_runtime = _revalidate_model(
        ToolRuntime,
        tool_runtime or tool_runtime_versions(),
        "tool runtime",
    )
    target = audit.target
    packages = audit.packages
    diagnostics = audit.diagnostics
    sorted_packages = _sorted_packages(packages)
    sorted_accesses = tuple(sorted(validated_accesses, key=_access_key))
    sorted_sources = tuple(sorted(provenance.sources, key=_source_key))
    sorted_diagnostics = tuple(sorted(diagnostics, key=_diagnostic_key))
    archive_bytes = sum(package.wheel.bytes for package in sorted_packages)
    metadata_bytes = sum(access.bytes for access in sorted_accesses)
    estimated_installed = sum(
        Decimal(package.wheel.bytes)
        * Decimal(
            str(
                request.installed_size_multipliers[
                    "pure_python" if package.wheel.is_pure_python else "native"
                ]
            )
        )
        for package in sorted_packages
    )
    estimated_installed_bytes = math.ceil(estimated_installed)
    estimated_temporary_cache_peak_bytes = archive_bytes + metadata_bytes
    required_subtotal = (
        request.existing_payload_bytes
        + estimated_installed_bytes
        + estimated_temporary_cache_peak_bytes
    )
    required_free_bytes = math.ceil(
        Decimal(required_subtotal) * (Decimal(1) + Decimal(str(request.free_space_safety_ratio)))
    )
    blockers = [
        AuditBlocker(
            category="license",
            package=f"{canonicalize_name(package.name)}=={package.version}",
            filename=None,
            reason=(
                "legal_review_required"
                if package.license.status == "legal_review_required"
                else "unknown_license"
            ),
            detail=package.license.reason,
        )
        for package in sorted_packages
        if package.license.status != "allowed"
    ]
    blockers.extend(
        AuditBlocker(
            category="candidate",
            package=diagnostic.package,
            filename=diagnostic.filename,
            reason=diagnostic.reason,
            detail=diagnostic.detail,
        )
        for diagnostic in sorted_diagnostics
        if diagnostic.category == "blocked"
    )
    sorted_blockers = tuple(sorted(blockers, key=_blocker_key))
    access_hosts = tuple(
        sorted(
            {
                hostname
                for access in sorted_accesses
                if (hostname := urlsplit(access.url).hostname) is not None
            }
        )
    )
    packet = DependencyApprovalPacket(
        schema_version="2.0",
        packet_sha256="0" * 64,
        request=request,
        target=target,
        code_commit=provenance.commit,
        code_sources=sorted_sources,
        config_sha256=config_sha256,
        tool_runtime=validated_runtime,
        packages=sorted_packages,
        metadata_accesses=sorted_accesses,
        access_hosts=access_hosts,
        size=SizeEstimate(
            archive_bytes=archive_bytes,
            metadata_bytes=metadata_bytes,
            existing_payload_bytes=request.existing_payload_bytes,
            estimated_installed_bytes=estimated_installed_bytes,
            estimated_temporary_cache_peak_bytes=estimated_temporary_cache_peak_bytes,
            required_free_bytes=required_free_bytes,
        ),
        exclusions=sorted_diagnostics,
        blockers=sorted_blockers,
        license_preflight="PASS" if not sorted_blockers else "BLOCKED",
        side_effects=SideEffects(),
    )
    packet_sha256 = hashlib.sha256(_packet_payload_json(packet)).hexdigest()
    document = packet.model_dump(mode="json")
    document["packet_sha256"] = packet_sha256
    return DependencyApprovalPacket.model_validate(document)


def _markdown_scalar(value: str | None) -> str:
    encoded = json.dumps(value if value is not None else "", ensure_ascii=False)
    return html.escape(encoded, quote=True)


def render_packet_markdown(packet: DependencyApprovalPacket) -> bytes:
    """Render packet evidence without interpreting untrusted text as Markdown."""

    target_lines = [
        f"{position}. {_markdown_scalar(tag)}"
        for position, tag in enumerate(packet.target.wheel_tags, start=1)
    ]
    root_lines = [f"- {_markdown_scalar(root.raw)}" for root in packet.request.roots]
    source_lines = [
        f"- {_markdown_scalar(source.path)} — SHA-256 `{source.sha256}`"
        for source in packet.code_sources
    ] or ["- None recorded"]
    access_host_lines = [f"- {_markdown_scalar(host)}" for host in packet.access_hosts] or [
        "- None"
    ]
    blocker_lines = [
        f"- {_markdown_scalar(blocker.category)} / {_markdown_scalar(blocker.reason)}; "
        f"package {_markdown_scalar(blocker.package)}; file "
        f"{_markdown_scalar(blocker.filename)}; detail: {_markdown_scalar(blocker.detail)}"
        for blocker in packet.blockers
    ] or ["- None"]
    exclusion_lines = [
        f"- {_markdown_scalar(item.package)} / {_markdown_scalar(item.filename)} — "
        f"{_markdown_scalar(item.reason)}; {_markdown_scalar(item.detail)}"
        for item in packet.exclusions
    ] or ["- None"]
    package_lines: list[str] = []
    for package in packet.packages:
        metadata_record = package.project_metadata
        extras = ", ".join(_markdown_scalar(extra) for extra in package.extras) or "None"
        parents = ", ".join(_markdown_scalar(parent) for parent in package.parents) or "None"
        dependency_lines: list[str] = []
        for edge in package.dependencies:
            dependency_lines.append(
                f"  - {_markdown_scalar(edge.child)} from {_markdown_scalar(edge.parent)}; "
                f"marker: {_markdown_scalar(edge.marker)}"
            )
        if not dependency_lines:
            dependency_lines.append("  - None")
        evidence_lines = [f"  - {_markdown_scalar(item)}" for item in package.license.evidence] or [
            "  - None"
        ]
        raw_requires = (
            ", ".join(_markdown_scalar(item) for item in metadata_record.requires_dist)
            if metadata_record.requires_dist
            else "None"
        )
        package_lines.extend(
            [
                f"### Package {_markdown_scalar(f'{package.name}=={package.version}')}",
                "",
                f"- Extras: {extras}",
                f"- Parents: {parents}",
                f"- Wheel filename: {_markdown_scalar(package.wheel.filename)}",
                f"- Official wheel URL (data only): {_markdown_scalar(package.wheel.url)}",
                f"- Wheel SHA-256: `{package.wheel.sha256}`",
                f"- Exact wheel bytes: {package.wheel.bytes}",
                f"- Upload time: {_markdown_scalar(package.wheel.upload_time)}",
                f"- Wheel requires Python: {_markdown_scalar(package.wheel.requires_python)}",
                "- Matching target tags: "
                + ", ".join(_markdown_scalar(tag) for tag in package.wheel.tags),
                f"- Project name: {_markdown_scalar(metadata_record.name)}",
                f"- Project version: {_markdown_scalar(metadata_record.version)}",
                f"- Project summary: {_markdown_scalar(metadata_record.summary)}",
                f"- Project requires Python: {_markdown_scalar(metadata_record.requires_python)}",
                f"- Raw requires-dist: {raw_requires}",
                f"- Licence expression: {_markdown_scalar(package.license.expression)}",
                f"- Licence status: {_markdown_scalar(package.license.status)}",
                f"- Licence decision reason: {_markdown_scalar(package.license.reason)}",
                "- Licence evidence:",
                *evidence_lines,
                "- Final dependency edges:",
                *dependency_lines,
                "",
            ]
        )
    metadata_lines = [
        f"- {_markdown_scalar(access.url)} — {access.bytes} bytes; SHA-256 `{access.sha256}`"
        for access in packet.metadata_accesses
    ]
    runtime = packet.tool_runtime
    lines = [
        "# Dependency approval packet",
        "",
        f"- Packet SHA-256 / JSON integrity anchor: `{packet.packet_sha256}`",
        f"- Request: {_markdown_scalar(packet.request.request_id)}",
        "",
        "## Target and ordered wheel tags",
        "",
        f"- OS: {_markdown_scalar(packet.target.os)}",
        f"- OS version: {_markdown_scalar(packet.target.os_version)}",
        f"- Architecture: {_markdown_scalar(packet.target.architecture)}",
        f"- Python implementation: {_markdown_scalar(packet.target.python_implementation)}",
        f"- Python version: {_markdown_scalar(packet.target.python_version)}",
        "- Complete ordered wheel-tag tuple:",
        *target_lines,
        "",
        "## Root requirements",
        "",
        *root_lines,
        "",
        "## Code, config, and runtime provenance",
        "",
        f"- Code commit: `{packet.code_commit}`",
        f"- Config SHA-256: `{packet.config_sha256}`",
        f"- Python runtime: {_markdown_scalar(runtime.python)}",
        f"- packaging: {_markdown_scalar(runtime.packaging)}",
        f"- pydantic: {_markdown_scalar(runtime.pydantic)}",
        f"- PyYAML: {_markdown_scalar(runtime.pyyaml)}",
        f"- pip: {_markdown_scalar(runtime.pip)}",
        f"- pip-vendored resolvelib: {_markdown_scalar(runtime.pip_vendored_resolvelib)}",
        "- Auditor source hashes:",
        *source_lines,
        "",
        "## Access hosts",
        "",
        *access_host_lines,
        "",
        "## Exact and estimated size accounting",
        "",
        f"- Exact archive bytes: {packet.size.archive_bytes}",
        f"- Exact metadata bytes: {packet.size.metadata_bytes}",
        f"- Exact existing payload bytes: {packet.size.existing_payload_bytes}",
        f"- Estimated installed bytes: {packet.size.estimated_installed_bytes}",
        "- Estimated temporary/cache peak bytes: "
        f"{packet.size.estimated_temporary_cache_peak_bytes}",
        f"- Required free bytes: {packet.size.required_free_bytes}",
        "",
        "## Packages",
        "",
        *package_lines,
        "## Metadata evidence",
        "",
        *metadata_lines,
        "",
        "## Candidate exclusions",
        "",
        *exclusion_lines,
        "",
        "## Blockers",
        "",
        *blocker_lines,
        "",
        "## Licence preflight",
        "",
        f"- `{packet.license_preflight}`",
        "",
        "## Zero side effects",
        "",
        "- Archive downloads: 0",
        "- Installs: 0",
        "- Package imports: 0",
        "- Weights loaded: 0",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _packet_markdown(packet: DependencyApprovalPacket) -> bytes:
    return render_packet_markdown(packet)


def render_failure_markdown(failure: DependencyAuditFailure) -> bytes:
    """Render non-approvable failure evidence with inert untrusted scalars."""

    diagnostics = [
        f"- {_markdown_scalar(item.package)} / {_markdown_scalar(item.filename)} — "
        f"{_markdown_scalar(item.category)} / {_markdown_scalar(item.reason)}; "
        f"detail: {_markdown_scalar(item.detail)}"
        for item in failure.diagnostics
    ] or ["- None recorded"]
    metadata_accesses = [
        f"- {_markdown_scalar(access.url)} — {access.bytes} bytes; SHA-256 `{access.sha256}`"
        for access in failure.metadata_accesses
    ] or ["- None recorded"]
    source_lines = [
        f"- {_markdown_scalar(source.path)} — SHA-256 `{source.sha256}`"
        for source in failure.code_sources
    ]
    runtime = failure.tool_runtime
    lines = [
        "# Dependency audit failure",
        "",
        "- Approvable: `false`",
        f"- Failure SHA-256 / JSON integrity anchor: `{failure.failure_sha256}`",
        f"- Request: {_markdown_scalar(failure.request.request_id)}",
        "",
        "## Failure",
        "",
        f"- Category: {_markdown_scalar(failure.error_category)}",
        f"- Error: {_markdown_scalar(failure.error)}",
        "",
        "## Target",
        "",
        f"- OS: {_markdown_scalar(failure.target.os)}",
        f"- OS version: {_markdown_scalar(failure.target.os_version)}",
        f"- Architecture: {_markdown_scalar(failure.target.architecture)}",
        f"- Python implementation: {_markdown_scalar(failure.target.python_implementation)}",
        f"- Python version: {_markdown_scalar(failure.target.python_version)}",
        "",
        "## Code, config, and runtime provenance",
        "",
        f"- Code commit: `{failure.code_commit}`",
        f"- Config SHA-256: `{failure.config_sha256}`",
        f"- Python runtime: {_markdown_scalar(runtime.python)}",
        f"- packaging: {_markdown_scalar(runtime.packaging)}",
        f"- pydantic: {_markdown_scalar(runtime.pydantic)}",
        f"- PyYAML: {_markdown_scalar(runtime.pyyaml)}",
        f"- pip: {_markdown_scalar(runtime.pip)}",
        f"- pip-vendored resolvelib: {_markdown_scalar(runtime.pip_vendored_resolvelib)}",
        "- Auditor source hashes:",
        *source_lines,
        "",
        "## Metadata evidence",
        "",
        *metadata_accesses,
        "",
        "## Diagnostics",
        "",
        *diagnostics,
        "",
        "## Zero side effects",
        "",
        "- Archive downloads: 0",
        "- Installs: 0",
        "- Package imports: 0",
        "- Weights loaded: 0",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _validate_packet_hash(packet: DependencyApprovalPacket) -> None:
    actual = hashlib.sha256(_packet_payload_json(packet)).hexdigest()
    if not hmac.compare_digest(actual, packet.packet_sha256):
        raise PacketIntegrityError("packet SHA-256 is invalid")


def _validate_failure_hash(failure: DependencyAuditFailure) -> None:
    actual = hashlib.sha256(_failure_payload_json(failure)).hexdigest()
    if not hmac.compare_digest(actual, failure.failure_sha256):
        raise PacketIntegrityError("failure SHA-256 is invalid")


def _validate_existing(target: Path, expected_json: bytes, expected_markdown: bytes) -> None:
    try:
        target_mode = target.lstat().st_mode
        if (
            stat.S_ISLNK(target_mode)
            or not stat.S_ISDIR(target_mode)
            or stat.S_IMODE(target_mode) != 0o555
        ):
            raise PacketIntegrityError("existing packet is corrupt")
        entries = {entry.name: entry for entry in target.iterdir()}
        if set(entries) != {"packet.json", "packet.md"}:
            raise PacketIntegrityError("existing packet is corrupt")
        for name, expected in (
            ("packet.json", expected_json),
            ("packet.md", expected_markdown),
        ):
            path = entries[name]
            mode = path.lstat().st_mode
            if (
                not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
                or stat.S_IMODE(mode) != 0o444
                or path.read_bytes() != expected
            ):
                raise PacketIntegrityError("existing packet is corrupt")
    except PacketIntegrityError:
        raise
    except OSError as error:
        raise PacketIntegrityError("existing packet is corrupt") from error


def _validate_existing_failure(
    target: Path,
    expected_json: bytes,
    expected_markdown: bytes,
) -> None:
    try:
        target_mode = target.lstat().st_mode
        if (
            stat.S_ISLNK(target_mode)
            or not stat.S_ISDIR(target_mode)
            or stat.S_IMODE(target_mode) != 0o555
        ):
            raise PacketIntegrityError("existing failure is corrupt")
        entries = {entry.name: entry for entry in target.iterdir()}
        if set(entries) != {"failure.json", "failure.md"}:
            raise PacketIntegrityError("existing failure is corrupt")
        for name, expected in (
            ("failure.json", expected_json),
            ("failure.md", expected_markdown),
        ):
            path = entries[name]
            mode = path.lstat().st_mode
            if (
                not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
                or stat.S_IMODE(mode) != 0o444
                or path.read_bytes() != expected
            ):
                raise PacketIntegrityError("existing failure is corrupt")
    except PacketIntegrityError:
        raise
    except OSError as error:
        raise PacketIntegrityError("existing failure is corrupt") from error


def _cleanup_capabilities() -> _CleanupCapabilities:
    if sys.platform == "win32":
        return _CleanupCapabilities(
            descriptor_relative=False,
            directory_open_flags=None,
            file_nofollow_flag=None,
        )
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    descriptor_apis = (os.open, os.stat, os.unlink, os.rmdir)
    if (
        directory_flag is None
        or nofollow_flag is None
        or not all(api in os.supports_dir_fd for api in descriptor_apis)
    ):
        return _CleanupCapabilities(
            descriptor_relative=False,
            directory_open_flags=None,
            file_nofollow_flag=None,
        )
    return _CleanupCapabilities(
        descriptor_relative=True,
        directory_open_flags=os.O_RDONLY | directory_flag | nofollow_flag,
        file_nofollow_flag=nofollow_flag,
    )


def _owned_file(descriptor: int) -> _OwnedFile:
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        raise PacketIntegrityError("temporary packet file is unsafe")
    return _OwnedFile(
        descriptor=descriptor,
        device=status.st_dev,
        inode=status.st_ino,
    )


def _write_and_sync(owned: _OwnedTemporary, name: str, content: bytes) -> None:
    if owned.directory_descriptor is None:
        path = owned.parent / owned.name / name
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise PacketIntegrityError("temporary packet file is unsafe")
        owned.files[name] = _OwnedFile(
            descriptor=None,
            device=status.st_dev,
            inode=status.st_ino,
        )
        return
    nofollow_flag = owned.capabilities.file_nofollow_flag
    if nofollow_flag is None:
        raise PacketIntegrityError("descriptor cleanup capability is incomplete")
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow_flag,
        0o666,
        dir_fd=owned.directory_descriptor,
    )
    try:
        owned.files[name] = _owned_file(descriptor)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        if owned.files.pop(name, None) is not None:
            os.close(descriptor)
        raise


def _read_owned_file(owned: _OwnedTemporary, name: str) -> bytes:
    descriptor = owned.files[name].descriptor
    if descriptor is None:
        return (owned.parent / owned.name / name).read_bytes()
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _rename_no_replace(source: Path, target: Path) -> None:
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = library.renamex_np(source_bytes, target_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        result = library.renameat2(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "win32":
        os.rename(source, target)
        return
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def _owned_temporary(temporary: Path) -> _OwnedTemporary:
    capabilities = _cleanup_capabilities()
    if not capabilities.descriptor_relative:
        parent_status = temporary.parent.lstat()
        directory_status = temporary.lstat()
        if (
            not stat.S_ISDIR(parent_status.st_mode)
            or stat.S_ISLNK(parent_status.st_mode)
            or not stat.S_ISDIR(directory_status.st_mode)
            or stat.S_ISLNK(directory_status.st_mode)
        ):
            raise PacketIntegrityError("temporary packet directory is unsafe")
        owned = _OwnedTemporary(
            parent=temporary.parent,
            name=temporary.name,
            parent_device=parent_status.st_dev,
            parent_inode=parent_status.st_ino,
            directory_device=directory_status.st_dev,
            directory_inode=directory_status.st_ino,
            capabilities=capabilities,
            parent_descriptor=None,
            directory_descriptor=None,
        )
        for entry in sorted(temporary.iterdir()):
            if entry.name not in {"packet.json", "packet.md"}:
                continue
            status = entry.lstat()
            if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
                raise PacketIntegrityError("temporary packet file is unsafe")
            owned.files[entry.name] = _OwnedFile(
                descriptor=None,
                device=status.st_dev,
                inode=status.st_ino,
            )
        return owned

    directory_flags = capabilities.directory_open_flags
    nofollow_flag = capabilities.file_nofollow_flag
    if directory_flags is None or nofollow_flag is None:
        raise PacketIntegrityError("descriptor cleanup capability is incomplete")
    parent_descriptor = os.open(temporary.parent, directory_flags)
    try:
        directory_descriptor = os.open(
            temporary.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    owned: _OwnedTemporary | None = None
    try:
        parent_status = os.fstat(parent_descriptor)
        directory_status = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(parent_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
            raise PacketIntegrityError("temporary packet directory is unsafe")
        owned = _OwnedTemporary(
            parent=temporary.parent,
            name=temporary.name,
            parent_device=parent_status.st_dev,
            parent_inode=parent_status.st_ino,
            directory_device=directory_status.st_dev,
            directory_inode=directory_status.st_ino,
            capabilities=capabilities,
            parent_descriptor=parent_descriptor,
            directory_descriptor=directory_descriptor,
        )
        for entry in sorted(os.listdir(directory_descriptor)):
            if entry not in {"packet.json", "packet.md"}:
                continue
            descriptor = os.open(
                entry,
                os.O_RDWR | nofollow_flag,
                dir_fd=directory_descriptor,
            )
            try:
                owned.files[entry] = _owned_file(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
        return owned
    except BaseException:
        if owned is None:
            os.close(directory_descriptor)
            os.close(parent_descriptor)
        else:
            _close_owned_temporary(owned)
        raise


def _same_identity(status: os.stat_result, *, device: int, inode: int) -> bool:
    return status.st_dev == device and status.st_ino == inode


def _name_has_identity(
    name: str,
    *,
    directory_descriptor: int,
    device: int,
    inode: int,
) -> bool:
    try:
        status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return _same_identity(status, device=device, inode=inode)


def _close_owned_temporary(owned: _OwnedTemporary) -> None:
    for owned_file in owned.files.values():
        if owned_file.descriptor is None:
            continue
        try:
            os.close(owned_file.descriptor)
        except OSError:
            pass
    for descriptor in (owned.directory_descriptor, owned.parent_descriptor):
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError:
            pass


_CleanupHook = Callable[[str, str | None], None]


def _remove_owned_temporary_path(owned: _OwnedTemporary) -> None:
    """Fail closed where Python cannot bind destructive operations to held handles."""

    temporary = owned.parent / owned.name
    try:
        parent_status = owned.parent.lstat()
        directory_status = temporary.lstat()
        if (
            not _same_identity(
                parent_status,
                device=owned.parent_device,
                inode=owned.parent_inode,
            )
            or not stat.S_ISDIR(directory_status.st_mode)
            or stat.S_ISLNK(directory_status.st_mode)
            or not _same_identity(
                directory_status,
                device=owned.directory_device,
                inode=owned.directory_inode,
            )
        ):
            return
        entries = {entry.name: entry for entry in temporary.iterdir()}
        if set(entries) != set(owned.files):
            return
        for name, owned_file in owned.files.items():
            status = entries[name].lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or stat.S_ISLNK(status.st_mode)
                or not _same_identity(status, device=owned_file.device, inode=owned_file.inode)
            ):
                return
        # Python's Windows path APIs do not expose conditional delete-by-file-identity.
        # Preserve the verified directory rather than risk deleting replacement state.
    except OSError:
        return


def _remove_owned_temporary(
    owned: _OwnedTemporary,
    *,
    hook: _CleanupHook | None = None,
) -> None:
    run_hook = hook or (lambda _event, _entry: None)
    try:
        if not owned.capabilities.descriptor_relative:
            _remove_owned_temporary_path(owned)
            return
        parent_descriptor = owned.parent_descriptor
        directory_descriptor = owned.directory_descriptor
        if parent_descriptor is None or directory_descriptor is None:
            return
        if not _same_identity(
            os.fstat(parent_descriptor),
            device=owned.parent_device,
            inode=owned.parent_inode,
        ) or not _same_identity(
            os.fstat(directory_descriptor),
            device=owned.directory_device,
            inode=owned.directory_inode,
        ):
            return
        if not _name_has_identity(
            owned.name,
            directory_descriptor=parent_descriptor,
            device=owned.directory_device,
            inode=owned.directory_inode,
        ):
            return
        entries = tuple(sorted(os.listdir(directory_descriptor)))
        if set(entries) != set(owned.files):
            return
        for entry in entries:
            owned_file = owned.files[entry]
            if owned_file.descriptor is None:
                return
            if not _same_identity(
                os.fstat(owned_file.descriptor),
                device=owned_file.device,
                inode=owned_file.inode,
            ) or not _name_has_identity(
                entry,
                directory_descriptor=directory_descriptor,
                device=owned_file.device,
                inode=owned_file.inode,
            ):
                return
        os.fchmod(directory_descriptor, 0o700)
        for entry in entries:
            run_hook("before_entry_unlink", entry)
            owned_file = owned.files[entry]
            if not _name_has_identity(
                entry,
                directory_descriptor=directory_descriptor,
                device=owned_file.device,
                inode=owned_file.inode,
            ):
                return
            os.unlink(entry, dir_fd=directory_descriptor)
        run_hook("before_directory_rmdir", None)
        if not _name_has_identity(
            owned.name,
            directory_descriptor=parent_descriptor,
            device=owned.directory_device,
            inode=owned.directory_inode,
        ):
            return
        os.rmdir(owned.name, dir_fd=parent_descriptor)
    except OSError:
        return
    finally:
        _close_owned_temporary(owned)


def write_approval_packet(packet: DependencyApprovalPacket, root: Path) -> Path:
    """Atomically publish ``packet`` or reuse an identical immutable target."""

    packet = _revalidate_model(DependencyApprovalPacket, packet, "packet")
    _validate_packet_hash(packet)
    expected_json = canonical_packet_json(packet)
    expected_markdown = _packet_markdown(packet)
    target_parent = root / packet.request.request_id
    target = target_parent / packet.packet_sha256
    target_parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_existing(target, expected_json, expected_markdown)
        return target

    temporary = target_parent / f".{packet.packet_sha256}.tmp-{uuid.uuid4()}"
    temporary.mkdir(mode=0o700)
    owned_temporary = _owned_temporary(temporary)
    try:
        _write_and_sync(owned_temporary, "packet.json", expected_json)
        _write_and_sync(owned_temporary, "packet.md", expected_markdown)
        if (
            _read_owned_file(owned_temporary, "packet.json") != expected_json
            or _read_owned_file(owned_temporary, "packet.md") != expected_markdown
        ):
            raise PacketIntegrityError("temporary packet validation failed")
        directory_descriptor = owned_temporary.directory_descriptor
        if directory_descriptor is None:
            (temporary / "packet.json").chmod(0o444)
            (temporary / "packet.md").chmod(0o444)
            temporary.chmod(0o555)
        else:
            for name in ("packet.json", "packet.md"):
                file_descriptor = owned_temporary.files[name].descriptor
                if file_descriptor is None:
                    raise PacketIntegrityError("descriptor cleanup capability is incomplete")
                os.fchmod(file_descriptor, 0o444)
            os.fsync(directory_descriptor)
            # The evidence directory is intentionally immutable and world-readable.
            os.fchmod(directory_descriptor, 0o555)  # nosec B103
        try:
            _rename_no_replace(temporary, target)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _validate_existing(target, expected_json, expected_markdown)
        else:
            if owned_temporary.parent_descriptor is not None:
                os.fsync(owned_temporary.parent_descriptor)
        return target
    finally:
        _remove_owned_temporary(owned_temporary)


def write_dependency_audit_failure(failure: DependencyAuditFailure, root: Path) -> Path:
    """Publish immutable failure evidence below a non-installable namespace."""

    failure = _revalidate_model(DependencyAuditFailure, failure, "failure")
    _validate_failure_hash(failure)
    expected_json = canonical_failure_json(failure)
    expected_markdown = render_failure_markdown(failure)
    target_parent = root / failure.request.request_id / "failures"
    target = target_parent / failure.failure_sha256
    target_parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_existing_failure(target, expected_json, expected_markdown)
        return target

    temporary = target_parent / f".{failure.failure_sha256}.tmp-{uuid.uuid4()}"
    temporary.mkdir(mode=0o700)
    owned_temporary = _owned_temporary(temporary)
    try:
        _write_and_sync(owned_temporary, "failure.json", expected_json)
        _write_and_sync(owned_temporary, "failure.md", expected_markdown)
        if (
            _read_owned_file(owned_temporary, "failure.json") != expected_json
            or _read_owned_file(owned_temporary, "failure.md") != expected_markdown
        ):
            raise PacketIntegrityError("temporary failure validation failed")
        directory_descriptor = owned_temporary.directory_descriptor
        if directory_descriptor is None:
            (temporary / "failure.json").chmod(0o444)
            (temporary / "failure.md").chmod(0o444)
            temporary.chmod(0o555)
        else:
            for name in ("failure.json", "failure.md"):
                file_descriptor = owned_temporary.files[name].descriptor
                if file_descriptor is None:
                    raise PacketIntegrityError("descriptor cleanup capability is incomplete")
                os.fchmod(file_descriptor, 0o444)
            os.fsync(directory_descriptor)
            # The evidence directory is intentionally immutable and world-readable.
            os.fchmod(directory_descriptor, 0o555)  # nosec B103
        try:
            _rename_no_replace(temporary, target)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _validate_existing_failure(target, expected_json, expected_markdown)
        else:
            if owned_temporary.parent_descriptor is not None:
                os.fsync(owned_temporary.parent_descriptor)
        return target
    finally:
        _remove_owned_temporary(owned_temporary)
