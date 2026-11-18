"""Owner-only orchestration tests for Story 6.2."""

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.session_admission import (
    SessionAdmissionService,
    SessionAdmissionServiceError,
)
from billiards_ai.cloud.session_admission.errors import SessionAdmissionDomainError
from billiards_ai.cloud.session_admission.policy import (
    ConsentPolicyProvider,
    NoticeRegistryEntry,
    ResolvedConsentPolicy,
)
from billiards_ai.cloud.table_binding.repository import AdmissionSnapshot, BindingSnapshot
from billiards_ai.shared.models import (
    CloudSession,
    ConsentDecision,
    ConsentDecisionRecord,
    ConsentNotice,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    ConsentWithdrawalRequest,
    EvidenceReference,
    GateApproval,
    GateChecklist,
    GateRevision,
    GateScope,
    MaterialUsePurpose,
    RealVideoGateStatus,
    SessionAdmissionErrorCode,
    SessionAdmissionNextAction,
    SessionParticipant,
    TableBinding,
    TableBindingStatus,
    TableCapacityStatus,
    TableRegistration,
    TableRegistrationStatus,
)

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)
KEY = "B" * 43
DIGEST = "b" * 64


class Identity:
    def __init__(self, user_id: UUID):
        self.user_id = user_id

    def current_user_id(self) -> UUID:
        return self.user_id


class Policy:
    def __init__(self, resolved: ResolvedConsentPolicy):
        self.resolved = resolved

    def resolve(self, snapshot: BindingSnapshot) -> ResolvedConsentPolicy:
        assert snapshot.binding.binding_id == self.resolved.view.binding_id
        return self.resolved


class Repository:
    def __init__(self, snapshot: BindingSnapshot | None):
        self.snapshot = snapshot
        self.admit_calls = 0

    def find_owned(self, *, binding_id: UUID, user_id: UUID) -> BindingSnapshot | None:
        if self.snapshot is None or self.snapshot.binding.user_id != user_id:
            return None
        return self.snapshot

    def admit(self, **kwargs):
        self.admit_calls += 1
        return kwargs["candidate_factory"]()

    def withdraw(self, **kwargs):
        return kwargs["candidate_factory"]()


def test_foreign_binding_is_resource_not_found_without_candidate_generation() -> None:
    user_id, foreign_user = uuid4(), uuid4()
    binding_id, table_registration_id, qr_id = uuid4(), uuid4(), uuid4()
    snapshot = BindingSnapshot(
        TableBinding(
            binding_id=binding_id,
            user_id=foreign_user,
            table_registration_id=table_registration_id,
            qr_registration_id=qr_id,
            qr_version=1,
            status=TableBindingStatus.PENDING_CONSENT,
            created_at=NOW,
        ),
        TableRegistration(
            table_registration_id=table_registration_id,
            venue_id="synthetic-venue",
            table_id="synthetic-table",
            device_id="synthetic-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        ),
        TableCapacityStatus.AVAILABLE,
    )
    repository = Repository(snapshot)
    service = SessionAdmissionService(
        repository=repository,
        identity_provider=Identity(user_id),
        policy_provider=object(),
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock called")),
        uuid_factory=lambda: (_ for _ in ()).throw(AssertionError("uuid called")),
    )
    with pytest.raises(SessionAdmissionServiceError) as raised:
        service.read_policy(binding_id)
    assert raised.value.code is SessionAdmissionErrorCode.RESOURCE_NOT_FOUND
    assert repository.admit_calls == 0


def test_model_construct_request_is_revalidated_before_authority_calls() -> None:
    service = SessionAdmissionService(
        repository=object(),
        identity_provider=Identity(uuid4()),
        policy_provider=object(),
    )
    bypass = ConsentSubmissionRequest.model_construct(
        schema_version="1.0",
        binding_id=uuid4(),
        service_decision=True,
        training_decision=ConsentDecision.DECLINED,
        policy_id=DIGEST,
        gate_revision_id=uuid4(),
        service_notice_version="v1",
        training_notice_version="v1",
        idempotency_key=KEY,
    )
    with pytest.raises(SessionAdmissionServiceError) as raised:
        service.submit(bypass)
    assert raised.value.code is SessionAdmissionErrorCode.CONSENT_REQUEST_INVALID


def _approved_gate(snapshot: BindingSnapshot, notice_bytes: bytes) -> GateRevision:
    scope = GateScope(
        venue_id=snapshot.table.venue_id,
        table_id=snapshot.table.table_id,
        device_id=snapshot.table.device_id,
        camera_config_id="synthetic-camera",
        checklist=GateChecklist(**{name: True for name in GateChecklist.model_fields}),
        operator_name="Synthetic operator",
        operator_contact="privacy@example.invalid",
        service_consent_version="synthetic-v1",
        service_consent_sha256=hashlib.sha256(notice_bytes).hexdigest(),
        training_consent_version="synthetic-v1",
        training_consent_sha256=hashlib.sha256(notice_bytes).hexdigest(),
        access_policy_ref="synthetic-access",
        retention_policy_ref="synthetic-retention",
        deletion_policy_ref="synthetic-deletion",
        approved_purposes=tuple(MaterialUsePurpose),
    )
    return GateRevision(
        gate_id=uuid4(),
        revision_id=uuid4(),
        revision_number=2,
        supersedes_revision_id=uuid4(),
        status=RealVideoGateStatus.APPROVED,
        scope=scope,
        inspector_id="synthetic-inspector",
        inspected_at=NOW,
        evidence=(
            EvidenceReference(evidence_ref="synthetic-evidence", sha256=DIGEST, captured_at=NOW),
        ),
        approval=GateApproval(
            approver_id="synthetic-approver",
            credential_id="synthetic-credential",
            approved_at=NOW,
            approval_evidence_sha256=DIGEST,
            gate_fingerprint=scope.fingerprint,
            approval_mac=DIGEST,
        ),
        created_at=NOW,
    )


def test_policy_provider_uses_exact_server_mapping_raw_bytes_and_gate_scope() -> None:
    user_id, binding_id, table_id, qr_id = uuid4(), uuid4(), uuid4(), uuid4()
    snapshot = BindingSnapshot(
        TableBinding(
            binding_id=binding_id,
            user_id=user_id,
            table_registration_id=table_id,
            qr_registration_id=qr_id,
            qr_version=1,
            status=TableBindingStatus.PENDING_CONSENT,
            created_at=NOW,
        ),
        TableRegistration(
            table_registration_id=table_id,
            venue_id="synthetic-venue",
            table_id="synthetic-table",
            device_id="synthetic-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        ),
        TableCapacityStatus.AVAILABLE,
    )
    notice_bytes = b"Synthetic notice\n"
    gate = _approved_gate(snapshot, notice_bytes)

    class Gates:
        def get_current(self, gate_id):
            assert gate_id == gate.gate_id
            return gate

    def entry(purpose):
        return NoticeRegistryEntry(
            purpose=purpose,
            version="synthetic-v1",
            canonical_bytes=notice_bytes,
            collection_scope="Synthetic marked clips",
            use_description="Synthetic use",
            retention_description="Synthetic retention",
            access_description="Synthetic access",
            deletion_description="Synthetic deletion",
            operator_name="Synthetic operator",
            operator_contact="privacy@example.invalid",
        )

    provider = ConsentPolicyProvider(
        gate_repository=Gates(),
        table_gate_mapping={
            (
                snapshot.table.venue_id,
                snapshot.table.table_id,
                snapshot.table.device_id,
            ): gate.gate_id
        },
        notice_registry={
            (gate.gate_id, MaterialUsePurpose.SERVICE_ANALYSIS): entry(
                MaterialUsePurpose.SERVICE_ANALYSIS
            ),
            (gate.gate_id, MaterialUsePurpose.TRAINING): entry(MaterialUsePurpose.TRAINING),
        },
    )
    resolved = provider.resolve(snapshot)
    assert resolved.view.service_notice.content == "Synthetic notice\n"
    assert resolved.view.service_notice.content_sha256 == hashlib.sha256(notice_bytes).hexdigest()

    stale = ConsentSubmissionRequest(
        schema_version="1.0",
        binding_id=binding_id,
        service_decision=ConsentDecision.GRANTED,
        training_decision=ConsentDecision.DECLINED,
        policy_id="0" * 64,
        gate_revision_id=gate.revision_id,
        service_notice_version="synthetic-v1",
        training_notice_version="synthetic-v1",
        idempotency_key="A" * 43,
    )
    with pytest.raises(Exception) as raised:
        resolved.verify_submission(stale)
    assert raised.value.code is SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED

    wrong_gate = gate.model_copy(update={"gate_id": uuid4()})

    class WrongGateRepository:
        def get_current(self, gate_id):
            assert gate_id == gate.gate_id
            return wrong_gate

    wrong_provider = ConsentPolicyProvider(
        gate_repository=WrongGateRepository(),
        table_gate_mapping={
            (
                snapshot.table.venue_id,
                snapshot.table.table_id,
                snapshot.table.device_id,
            ): gate.gate_id
        },
        notice_registry={
            (gate.gate_id, MaterialUsePurpose.SERVICE_ANALYSIS): entry(
                MaterialUsePurpose.SERVICE_ANALYSIS
            ),
            (gate.gate_id, MaterialUsePurpose.TRAINING): entry(MaterialUsePurpose.TRAINING),
        },
    )
    with pytest.raises(SessionAdmissionDomainError) as mismatch:
        wrong_provider.resolve(snapshot)
    assert mismatch.value.code is SessionAdmissionErrorCode.CONSENT_POLICY_CHANGED


def _admission_snapshot(*, service_decision: ConsentDecision) -> AdmissionSnapshot:
    user_id, binding_id, table_id, qr_id = uuid4(), uuid4(), uuid4(), uuid4()
    session_id, participant_id = uuid4(), uuid4()
    service_id, training_id = uuid4(), uuid4()
    binding = TableBinding(
        binding_id=binding_id,
        user_id=user_id,
        table_registration_id=table_id,
        qr_registration_id=qr_id,
        qr_version=1,
        status=TableBindingStatus.ACTIVE,
        created_at=NOW,
        activated_at=NOW,
    )
    table = TableRegistration(
        table_registration_id=table_id,
        venue_id="synthetic-venue",
        table_id="synthetic-table",
        device_id="synthetic-device",
        venue_display_label="Synthetic venue",
        table_display_label="Table 1",
        current_qr_version=1,
        status=TableRegistrationStatus.ACTIVE,
    )

    def consent(consent_id, purpose, decision, *, supersedes=None):
        return ConsentDecisionRecord(
            consent_id=consent_id,
            user_id=user_id,
            binding_id=binding_id,
            session_id=session_id,
            participant_id=participant_id,
            purpose=purpose,
            decision=decision,
            notice_version="synthetic-v1",
            notice_sha256="a" * 64,
            gate_revision_id=uuid4(),
            gate_scope_fingerprint="b" * 64,
            decided_at=NOW,
            withdrawn_at=NOW if decision is ConsentDecision.WITHDRAWN else None,
            supersedes=supersedes,
        )

    service = consent(
        service_id,
        MaterialUsePurpose.SERVICE_ANALYSIS,
        service_decision,
        supersedes=uuid4() if service_decision is ConsentDecision.WITHDRAWN else None,
    )
    training = consent(training_id, MaterialUsePurpose.TRAINING, ConsentDecision.DECLINED)
    return AdmissionSnapshot(
        binding=binding,
        table=table,
        service_consent=service,
        training_consent=training,
        session=CloudSession(
            session_id=session_id,
            table_registration_id=table_id,
            created_at=NOW,
        ),
        participant=SessionParticipant(
            participant_id=participant_id,
            session_id=session_id,
            user_id=user_id,
            binding_id=binding_id,
            service_consent_id=service_id,
            training_consent_id=training_id,
            joined_at=NOW,
        ),
    )


def test_review_service_withdrawal_view_retains_audit_links_but_blocks_upload() -> None:
    snapshot = _admission_snapshot(service_decision=ConsentDecision.WITHDRAWN)
    view = SessionAdmissionService._view(snapshot)
    assert view.session_id == snapshot.session.session_id
    assert view.participant_id == snapshot.participant.participant_id
    assert view.service_decision is ConsentDecision.WITHDRAWN
    assert view.next_action is SessionAdmissionNextAction.REVIEW_CONSENT


def _resolved_policy(binding_id) -> ResolvedConsentPolicy:
    revision_id = uuid4()
    digest = hashlib.sha256(b"Synthetic controlled notice").hexdigest()

    def notice(purpose):
        return ConsentNotice(
            purpose=purpose,
            version="synthetic-v1",
            content="Synthetic controlled notice",
            content_sha256=digest,
            scope_fingerprint="c" * 64,
            gate_revision_id=revision_id,
            collection_scope="Synthetic marked clips",
            use_description="Synthetic use",
            retention_description="Synthetic retention",
            access_description="Synthetic access",
            deletion_description="Synthetic deletion",
            operator_name="Synthetic operator",
            operator_contact="privacy@example.invalid",
        )

    return ResolvedConsentPolicy(
        ConsentPolicyView(
            binding_id=binding_id,
            policy_id="d" * 64,
            gate_revision_id=revision_id,
            scope_fingerprint="c" * 64,
            service_notice=notice(MaterialUsePurpose.SERVICE_ANALYSIS),
            training_notice=notice(MaterialUsePurpose.TRAINING),
        ),
        frozenset(MaterialUsePurpose),
    )


def test_review_service_submit_read_and_withdraw_orchestration() -> None:
    admitted = _admission_snapshot(service_decision=ConsentDecision.GRANTED)
    withdrawn = _admission_snapshot(service_decision=ConsentDecision.WITHDRAWN)
    binding_snapshot = BindingSnapshot(
        admitted.binding,
        admitted.table,
        TableCapacityStatus.ONE_ACTIVE,
    )
    resolved = _resolved_policy(admitted.binding.binding_id)

    class OrchestrationRepository:
        def __init__(self):
            self.calls = []

        def find_owned(self, **kwargs):
            self.calls.append(("find_owned", kwargs))
            return binding_snapshot

        def recover_admission(self, **kwargs):
            self.calls.append(("recover_admission", kwargs))
            return None

        def admit(self, **kwargs):
            self.calls.append(("admit", kwargs))
            return admitted

        def read_admission(self, **kwargs):
            self.calls.append(("read_admission", kwargs))
            return admitted

        def withdraw(self, **kwargs):
            self.calls.append(("withdraw", kwargs))
            return withdrawn

    repository = OrchestrationRepository()
    service = SessionAdmissionService(
        repository=repository,
        identity_provider=Identity(admitted.binding.user_id),
        policy_provider=Policy(resolved),
    )
    submission = ConsentSubmissionRequest(
        schema_version="1.0",
        binding_id=admitted.binding.binding_id,
        service_decision=ConsentDecision.GRANTED,
        training_decision=ConsentDecision.DECLINED,
        policy_id=resolved.view.policy_id,
        gate_revision_id=resolved.view.gate_revision_id,
        service_notice_version=resolved.view.service_notice.version,
        training_notice_version=resolved.view.training_notice.version,
        idempotency_key="A" * 43,
    )
    assert service.submit(submission).next_action is SessionAdmissionNextAction.WAITING_FOR_UPLOAD
    assert (
        service.read(admitted.binding.binding_id).next_action
        is SessionAdmissionNextAction.WAITING_FOR_UPLOAD
    )
    withdrawal = ConsentWithdrawalRequest(
        participant_id=admitted.participant.participant_id,
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        idempotency_key="A" * 43,
    )
    assert service.withdraw(withdrawal).next_action is SessionAdmissionNextAction.REVIEW_CONSENT
    assert [name for name, _ in repository.calls] == [
        "find_owned",
        "recover_admission",
        "admit",
        "read_admission",
        "withdraw",
    ]
