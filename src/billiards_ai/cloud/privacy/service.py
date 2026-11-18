"""State transitions for REAL_VIDEO_GATE revisions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from billiards_ai.shared.models import (
    EvidenceReference,
    GateApproval,
    GateRevision,
    GateScope,
    RealVideoGateStatus,
)

from .errors import GateApprovalError, GateConflictError
from .identity import APPROVER_CREDENTIAL_ID, FileApproverIdentityProvider
from .repository import (
    _REPOSITORY_WRITER_TOKEN,
    JsonGateRepository,
    _approval_security_payload,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RealVideoGateService:
    REQUIRED_APPROVER_ID = "Boyu"

    def __init__(
        self,
        repository: JsonGateRepository,
    ):
        self.repository = repository

    def create_gate(
        self,
        scope: GateScope,
        inspector_id: str,
        inspected_at: datetime,
        evidence: tuple[EvidenceReference, ...],
    ) -> GateRevision:
        created_at = _utc_now()
        revision = GateRevision(
            gate_id=uuid4(),
            revision_id=uuid4(),
            revision_number=1,
            scope=scope,
            inspector_id=inspector_id,
            inspected_at=inspected_at,
            evidence=evidence,
            created_at=created_at,
        )
        return self.repository._append_revision(
            revision,
            expected_current_revision_id=None,
            _writer_token=_REPOSITORY_WRITER_TOKEN,
        )

    def approve(
        self,
        gate_id: UUID,
        approval_evidence_sha256: str,
    ) -> GateRevision:
        identity = FileApproverIdentityProvider().current_identity()
        if identity.approver_id != self.REQUIRED_APPROVER_ID:
            raise GateApprovalError("current identity is not authorized to approve this gate")
        if identity.credential_id != APPROVER_CREDENTIAL_ID:
            raise GateApprovalError("approval credential is not on the trusted allowlist")
        history, previous_chain_digest = self.repository._get_history_with_chain_digest(gate_id)
        if not history:
            raise GateApprovalError("gate was not found")
        current = history[-1]
        if current.status is RealVideoGateStatus.APPROVED:
            if (
                current.approval is not None
                and current.approval.approver_id == identity.approver_id
                and current.approval.approval_evidence_sha256 == approval_evidence_sha256
            ):
                return current
            raise GateApprovalError("approved gate cannot be overwritten")
        if not current.scope.checklist.is_complete:
            missing = ", ".join(current.scope.checklist.missing_items)
            raise GateApprovalError(f"gate checklist is incomplete: {missing}")
        if not current.evidence:
            raise GateApprovalError("gate requires on-site evidence")
        approved_at = _utc_now()
        revision_id = uuid4()
        approval = GateApproval(
            approver_id=identity.approver_id,
            credential_id=identity.credential_id,
            key_id=identity.active_key_id,
            mac_version="3.0",
            approved_at=approved_at,
            approval_evidence_sha256=approval_evidence_sha256,
            gate_fingerprint=current.scope.fingerprint,
            approval_mac="0" * 64,
        )
        revision = GateRevision(
            gate_id=current.gate_id,
            revision_id=revision_id,
            revision_number=current.revision_number + 1,
            supersedes_revision_id=current.revision_id,
            status=RealVideoGateStatus.APPROVED,
            scope=current.scope,
            inspector_id=current.inspector_id,
            inspected_at=current.inspected_at,
            evidence=current.evidence,
            approval=approval,
            created_at=approved_at,
        )
        approval_mac = identity.approval_mac(
            key_id=identity.active_key_id,
            payload=_approval_security_payload(
                revision,
                previous_chain_digest=previous_chain_digest,
                mac_version="3.0",
            ),
        )
        revision = revision.model_copy(
            update={"approval": approval.model_copy(update={"approval_mac": approval_mac})}
        )
        try:
            return self.repository._append_revision(
                revision,
                expected_current_revision_id=current.revision_id,
                _writer_token=_REPOSITORY_WRITER_TOKEN,
            )
        except GateConflictError:
            latest = self.repository.get_current(gate_id)
            if (
                latest.status is RealVideoGateStatus.APPROVED
                and latest.approval is not None
                and latest.approval.approver_id == identity.approver_id
                and latest.approval.approval_evidence_sha256 == approval_evidence_sha256
            ):
                return latest
            raise

    def revise(
        self,
        gate_id: UUID,
        scope: GateScope,
        inspector_id: str,
        inspected_at: datetime,
        evidence: tuple[EvidenceReference, ...],
        reason: str,
    ) -> GateRevision:
        current = self.repository.get_current(gate_id)
        created_at = _utc_now()
        revision = GateRevision(
            gate_id=current.gate_id,
            revision_id=uuid4(),
            revision_number=current.revision_number + 1,
            supersedes_revision_id=current.revision_id,
            scope=scope,
            inspector_id=inspector_id,
            inspected_at=inspected_at,
            evidence=evidence,
            invalidation_reason=reason,
            created_at=created_at,
        )
        return self.repository._append_revision(
            revision,
            expected_current_revision_id=current.revision_id,
            _writer_token=_REPOSITORY_WRITER_TOKEN,
        )

    def revoke(self, gate_id: UUID, *, reason: str) -> GateRevision:
        current = self.repository.get_current(gate_id)
        created_at = _utc_now()
        revision = GateRevision(
            gate_id=current.gate_id,
            revision_id=uuid4(),
            revision_number=current.revision_number + 1,
            supersedes_revision_id=current.revision_id,
            scope=current.scope,
            inspector_id=current.inspector_id,
            inspected_at=current.inspected_at,
            evidence=(),
            invalidation_reason=reason,
            created_at=created_at,
        )
        return self.repository._append_revision(
            revision,
            expected_current_revision_id=current.revision_id,
            _writer_token=_REPOSITORY_WRITER_TOKEN,
        )
