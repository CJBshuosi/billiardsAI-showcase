"""Trusted current-gate and immutable-notice authority for session consent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from billiards_ai.cloud.table_binding.repository import BindingSnapshot
from billiards_ai.shared.models import (
    ConsentDecision,
    ConsentNotice,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    GateRevision,
    MaterialUsePurpose,
    RealVideoGateStatus,
    SessionAdmissionErrorCode,
)

from .errors import SessionAdmissionDomainError


class CurrentGateRepository(Protocol):
    def get_current(self, gate_id: UUID) -> GateRevision: ...


@dataclass(frozen=True, slots=True)
class NoticeRegistryEntry:
    purpose: MaterialUsePurpose
    version: str
    canonical_bytes: bytes
    collection_scope: str
    use_description: str
    retention_description: str
    access_description: str
    deletion_description: str
    operator_name: str
    operator_contact: str
    canonicalization_schema_version: str = "raw-utf8-v1"


@dataclass(frozen=True, slots=True)
class ResolvedConsentPolicy:
    view: ConsentPolicyView
    approved_purposes: frozenset[MaterialUsePurpose]

    def verify_submission(self, request: ConsentSubmissionRequest) -> None:
        view = self.view
        if (
            request.policy_id != view.policy_id
            or request.gate_revision_id != view.gate_revision_id
            or request.service_notice_version != view.service_notice.version
            or request.training_notice_version != view.training_notice.version
        ):
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        required: set[MaterialUsePurpose] = set()
        if request.service_decision is ConsentDecision.GRANTED:
            required.update(
                {MaterialUsePurpose.SERVICE_ANALYSIS, MaterialUsePurpose.NECESSARY_REVIEW}
            )
        if request.training_decision is ConsentDecision.GRANTED:
            required.update({MaterialUsePurpose.ANNOTATION, MaterialUsePurpose.TRAINING})
        if not required.issubset(self.approved_purposes):
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            )


class ConsentPolicyProvider:
    """Resolve policy solely from server-owned table mapping and registry."""

    def __init__(
        self,
        *,
        gate_repository: CurrentGateRepository,
        table_gate_mapping: Mapping[tuple[str, str, str], UUID],
        notice_registry: Mapping[tuple[UUID, MaterialUsePurpose], NoticeRegistryEntry],
    ):
        self._gate_repository = gate_repository
        self._table_gate_mapping = dict(table_gate_mapping)
        self._notice_registry = dict(notice_registry)

    @staticmethod
    def _notice(entry: NoticeRegistryEntry, revision: GateRevision) -> ConsentNotice:
        if entry.canonicalization_schema_version != "raw-utf8-v1":
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        try:
            content = entry.canonical_bytes.decode("utf-8", errors="strict")
        except UnicodeError:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED
            ) from None
        digest = hashlib.sha256(entry.canonical_bytes).hexdigest()
        scope = revision.scope
        expected_version = (
            scope.service_consent_version
            if entry.purpose is MaterialUsePurpose.SERVICE_ANALYSIS
            else scope.training_consent_version
        )
        expected_digest = (
            scope.service_consent_sha256
            if entry.purpose is MaterialUsePurpose.SERVICE_ANALYSIS
            else scope.training_consent_sha256
        )
        if (
            entry.version != expected_version
            or digest != expected_digest
            or entry.operator_name != scope.operator_name
            or entry.operator_contact != scope.operator_contact
        ):
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        try:
            return ConsentNotice(
                purpose=entry.purpose,
                version=entry.version,
                content=content,
                content_sha256=digest,
                scope_fingerprint=scope.fingerprint,
                gate_revision_id=revision.revision_id,
                canonicalization_schema_version=entry.canonicalization_schema_version,
                collection_scope=entry.collection_scope,
                use_description=entry.use_description,
                retention_description=entry.retention_description,
                access_description=entry.access_description,
                deletion_description=entry.deletion_description,
                operator_name=entry.operator_name,
                operator_contact=entry.operator_contact,
            )
        except Exception:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED
            ) from None

    def resolve(self, snapshot: BindingSnapshot) -> ResolvedConsentPolicy:
        table = snapshot.table
        mapping_key = (table.venue_id, table.table_id, table.device_id)
        gate_id = self._table_gate_mapping.get(mapping_key)
        if gate_id is None:
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        try:
            candidate = self._gate_repository.get_current(gate_id)
            revision = GateRevision.model_validate(candidate.model_dump(mode="python"))
        except SessionAdmissionDomainError:
            raise
        except Exception:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            ) from None
        if revision.gate_id != gate_id:
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        if revision.status is not RealVideoGateStatus.APPROVED:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            )
        scope = revision.scope
        if (scope.venue_id, scope.table_id, scope.device_id) != mapping_key:
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_SCOPE_MISMATCH)
        try:
            service_entry = self._notice_registry[(gate_id, MaterialUsePurpose.SERVICE_ANALYSIS)]
            training_entry = self._notice_registry[(gate_id, MaterialUsePurpose.TRAINING)]
        except KeyError:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED
            ) from None
        service_notice = self._notice(service_entry, revision)
        training_notice = self._notice(training_entry, revision)
        identity = {
            "canonicalization_schema_version": "raw-utf8-v1",
            "gate_revision_id": str(revision.revision_id),
            "scope_fingerprint": scope.fingerprint,
            "service_version": service_notice.version,
            "service_sha256": service_notice.content_sha256,
            "training_version": training_notice.version,
            "training_sha256": training_notice.content_sha256,
        }
        policy_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        try:
            view = ConsentPolicyView(
                binding_id=snapshot.binding.binding_id,
                policy_id=policy_id,
                gate_revision_id=revision.revision_id,
                scope_fingerprint=scope.fingerprint,
                service_notice=service_notice,
                training_notice=training_notice,
            )
        except Exception:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED
            ) from None
        return ResolvedConsentPolicy(view, frozenset(scope.approved_purposes))

    def verify_capture(self, table) -> None:
        """Re-read the trusted current Gate for capture start/finalize."""

        key = (table.venue_id, table.table_id, table.device_id)
        gate_id = self._table_gate_mapping.get(key)
        if gate_id is None:
            raise SessionAdmissionDomainError(SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED)
        try:
            revision = self._gate_repository.get_current(gate_id)
        except Exception:
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            ) from None
        required = {
            MaterialUsePurpose.SERVICE_ANALYSIS,
            MaterialUsePurpose.NECESSARY_REVIEW,
        }
        if (
            revision.gate_id != gate_id
            or revision.status is not RealVideoGateStatus.APPROVED
            or (revision.scope.venue_id, revision.scope.table_id, revision.scope.device_id) != key
            or not required.issubset(revision.scope.approved_purposes)
        ):
            raise SessionAdmissionDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            )
