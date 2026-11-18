"""Public contracts for metadata-only dependency audits."""

from billiards_ai.lab.dependency_audit.models import (
    AuditRequest,
    CandidateDiagnostic,
    DependencyApprovalPacket,
    DependencyAuditFailure,
    DependencyAuditResult,
    TargetSnapshot,
    load_audit_request,
)
from billiards_ai.lab.dependency_audit.packet import (
    PacketIntegrityError,
    build_approval_packet,
    build_dependency_audit_failure,
    write_approval_packet,
    write_dependency_audit_failure,
)
from billiards_ai.lab.dependency_audit.resolver import (
    DependencyResolutionError,
    resolve_dependencies,
    resolve_dependency_audit,
)
from billiards_ai.lab.dependency_audit.transport import (
    MetadataTransportError,
    PyPIMetadataClient,
    UrllibMetadataTransport,
)

__all__ = [
    "AuditRequest",
    "CandidateDiagnostic",
    "DependencyAuditResult",
    "DependencyAuditFailure",
    "DependencyApprovalPacket",
    "DependencyResolutionError",
    "MetadataTransportError",
    "PacketIntegrityError",
    "PyPIMetadataClient",
    "UrllibMetadataTransport",
    "build_approval_packet",
    "build_dependency_audit_failure",
    "load_audit_request",
    "resolve_dependency_audit",
    "resolve_dependencies",
    "TargetSnapshot",
    "write_approval_packet",
    "write_dependency_audit_failure",
]
