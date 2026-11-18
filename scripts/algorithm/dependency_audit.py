"""Create a metadata-only, content-addressed dependency approval packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml
from packaging.requirements import InvalidRequirement
from pydantic import ValidationError

from billiards_ai.lab.dependency_audit.index import WheelIndexError, resolve_target
from billiards_ai.lab.dependency_audit.models import (
    AuditRequest,
    CodeProvenance,
    SourceHash,
)
from billiards_ai.lab.dependency_audit.packet import (
    PacketIntegrityError,
    build_approval_packet,
    build_dependency_audit_failure,
    tool_runtime_versions,
    write_approval_packet,
    write_dependency_audit_failure,
)
from billiards_ai.lab.dependency_audit.resolver import (
    DependencyResolutionError,
    resolve_dependency_audit,
)
from billiards_ai.lab.dependency_audit.transport import (
    MetadataSecurityError,
    MetadataTransportError,
    PyPIMetadataClient,
    UrllibMetadataTransport,
)

MetadataClientFactory = Callable[[AuditRequest], PyPIMetadataClient]
ProvenanceFactory = Callable[[], CodeProvenance]

_AUDITOR_SOURCE_ROOTS = (
    "scripts/algorithm/dependency_audit.py",
    "src/billiards_ai/lab/dependency_audit",
)


class AuditConfigurationError(RuntimeError):
    """Raised when the request file cannot be safely read or validated."""


class AuditOutputError(RuntimeError):
    """Raised when an approval packet cannot be published."""


class AuditProvenanceError(RuntimeError):
    """Raised when the executed auditor sources cannot be tied to clean HEAD."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise SystemExit(64)


def live_client_factory(request: AuditRequest) -> PyPIMetadataClient:
    """Construct the production metadata client without making a request yet."""

    return PyPIMetadataClient(
        UrllibMetadataTransport(),
        single_response_bytes_max=request.single_response_bytes_max,
        total_metadata_bytes_max=request.total_metadata_bytes_max,
    )


def _git_output(repository_root: Path, arguments: list[str]) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository_root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditProvenanceError("auditor source provenance check failed") from error


def _discover_source_files(
    repository_root: Path,
    source_roots: tuple[str, ...],
) -> tuple[str, ...]:
    discovered: set[str] = set()
    for relative_root in source_roots:
        root = Path(relative_root)
        if root.is_absolute() or ".." in root.parts:
            raise AuditProvenanceError("auditor source root is unsafe")
        path = repository_root / root
        try:
            status = path.lstat()
        except OSError as error:
            raise AuditProvenanceError("auditor source root is missing") from error
        if stat.S_ISLNK(status.st_mode):
            raise AuditProvenanceError("auditor source root is a symlink")
        if stat.S_ISREG(status.st_mode):
            if path.suffix == ".py":
                discovered.add(path.relative_to(repository_root).as_posix())
            continue
        if not stat.S_ISDIR(status.st_mode):
            raise AuditProvenanceError("auditor source root is not a file or directory")
        for source in path.rglob("*.py"):
            source_status = source.lstat()
            if not stat.S_ISREG(source_status.st_mode) or stat.S_ISLNK(source_status.st_mode):
                raise AuditProvenanceError("auditor source file is unsafe")
            discovered.add(source.relative_to(repository_root).as_posix())
    return tuple(sorted(discovered))


def collect_code_provenance(
    repository_root: Path,
    source_roots: tuple[str, ...] = _AUDITOR_SOURCE_ROOTS,
) -> CodeProvenance:
    """Reject dirty auditor paths and hash the exact clean source set."""

    repository_root = repository_root.resolve()
    status_arguments = [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *source_roots,
    ]
    commit_before = _git_output(repository_root, ["rev-parse", "HEAD"])
    if _git_output(repository_root, status_arguments):
        raise AuditProvenanceError("auditor source tree is dirty")
    tracked = tuple(
        sorted(
            item.decode("utf-8")
            for item in _git_output(
                repository_root,
                ["ls-files", "-z", "--", *source_roots],
            ).split(b"\0")
            if item and item.endswith(b".py")
        )
    )
    discovered = _discover_source_files(repository_root, source_roots)
    if not tracked or discovered != tracked:
        raise AuditProvenanceError("auditor source set does not match tracked HEAD")
    try:
        sources = tuple(
            SourceHash(
                path=relative_path,
                sha256=hashlib.sha256((repository_root / relative_path).read_bytes()).hexdigest(),
            )
            for relative_path in tracked
        )
    except OSError as error:
        raise AuditProvenanceError("auditor source hashing failed") from error
    if _git_output(repository_root, status_arguments):
        raise AuditProvenanceError("auditor source tree changed during hashing")
    commit_after = _git_output(repository_root, ["rev-parse", "HEAD"])
    if commit_after != commit_before:
        raise AuditProvenanceError("repository HEAD changed during auditor source hashing")
    commit = commit_after.decode("ascii").strip()
    return CodeProvenance(commit=commit, sources=sources)


def live_code_provenance() -> CodeProvenance:
    return collect_code_provenance(Path(__file__).resolve().parents[2])


def _read_request_bytes(request_path: Path) -> bytes:
    try:
        return request_path.read_bytes()
    except OSError as error:
        raise AuditConfigurationError(f"request read failed: {request_path}") from error


def _parse_request(raw_request: bytes, request_path: Path) -> object:
    try:
        document = raw_request.decode("utf-8")
        return yaml.safe_load(document)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise AuditConfigurationError(f"request parse failed: {request_path}") from error


def _validate_request(document: object, request_path: Path) -> AuditRequest:
    try:
        return AuditRequest.model_validate(document)
    except (InvalidRequirement, ValidationError) as error:
        raise AuditConfigurationError(f"request validation failed: {request_path}") from error


def _load_request(request_path: Path) -> tuple[bytes, AuditRequest]:
    raw_request = _read_request_bytes(request_path)
    document = _parse_request(raw_request, request_path)
    return raw_request, _validate_request(document, request_path)


def audit_dependencies(
    request_path: Path,
    output_root: Path,
    *,
    client_factory: MetadataClientFactory = live_client_factory,
    provenance_factory: ProvenanceFactory | None = None,
) -> Path:
    """Resolve a request using metadata only and write its immutable packet."""

    raw_request, request = _load_request(request_path)
    provenance = (provenance_factory or live_code_provenance)()
    client = client_factory(request)
    target = resolve_target(request.target)
    runtime = tool_runtime_versions()
    config_sha256 = hashlib.sha256(raw_request).hexdigest()
    try:
        audit = resolve_dependency_audit(request, client, target)
    except DependencyResolutionError as error:
        failure = build_dependency_audit_failure(
            request,
            target,
            client.metadata_accesses,
            error,
            code_commit=provenance.commit,
            code_sources=provenance.sources,
            config_sha256=config_sha256,
            tool_runtime=runtime,
        )
        try:
            return write_dependency_audit_failure(failure, output_root)
        except OSError as output_error:
            raise AuditOutputError(f"failure output failed: {output_root}") from output_error
    packet = build_approval_packet(
        request,
        audit,
        client.metadata_accesses,
        code_commit=provenance.commit,
        code_sources=provenance.sources,
        config_sha256=config_sha256,
        tool_runtime=runtime,
    )
    try:
        return write_approval_packet(packet, output_root)
    except OSError as error:
        raise AuditOutputError(f"packet output failed: {output_root}") from error


_DECLARED_AUDIT_ERRORS = (
    AuditConfigurationError,
    AuditOutputError,
    AuditProvenanceError,
    DependencyResolutionError,
    MetadataSecurityError,
    MetadataTransportError,
    PacketIntegrityError,
    WheelIndexError,
)


def _summary(packet_dir: Path) -> dict[str, object]:
    failure_path = packet_dir / "failure.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        blockers = [
            f"candidate:{item['package']}:{item['reason']}"
            for item in failure["diagnostics"]
            if item["category"] == "blocked"
        ]
        if not blockers:
            blockers = [f"resolution:{failure['error_category']}"]
        return {
            "blockers": blockers,
            "failure_dir": str(packet_dir),
            "failure_sha256": failure["failure_sha256"],
            "passed": False,
        }
    packet = json.loads((packet_dir / "packet.json").read_text(encoding="utf-8"))
    blockers = [f"{item['reason']}:{item['package']}" for item in packet["blockers"]]
    return {
        "archive_bytes": packet["size"]["archive_bytes"],
        "blockers": blockers,
        "package_count": len(packet["packages"]),
        "packet_dir": str(packet_dir),
        "packet_sha256": packet["packet_sha256"],
        "passed": packet["license_preflight"] == "PASS" and not blockers,
    }


def main(
    argv: list[str] | None = None,
    *,
    client_factory: MetadataClientFactory = live_client_factory,
    provenance_factory: ProvenanceFactory | None = None,
) -> int:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        summary = _summary(
            audit_dependencies(
                arguments.request,
                arguments.output,
                client_factory=client_factory,
                provenance_factory=provenance_factory,
            )
        )
    except _DECLARED_AUDIT_ERRORS as error:
        print(f"dependency audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
