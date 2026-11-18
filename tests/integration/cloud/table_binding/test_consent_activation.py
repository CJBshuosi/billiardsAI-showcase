"""Atomic store smoke tests for Story 6.2.

The exhaustive repository failure/race coverage remains colocated with the
existing Story 6.1 repository integration suite; these tests pin the v2 shape.
"""

import base64
import hashlib
import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.session_admission import SessionAdmissionService
from billiards_ai.cloud.session_admission.policy import ResolvedConsentPolicy
from billiards_ai.cloud.session_capture.identity import StaffPrincipal
from billiards_ai.cloud.table_binding import JsonTableBindingRepository
from billiards_ai.cloud.table_binding.errors import (
    SessionAdmissionRepositoryDomainError,
    TableBindingRepositoryError,
)
from billiards_ai.cloud.table_binding.repository import (
    AdmissionCandidate,
    PendingBindingCandidate,
    WithdrawalCandidate,
)
from billiards_ai.shared.models import (
    CaptureOperationRequest,
    CloudSessionStatus,
    ConsentDecision,
    ConsentDecisionRecord,
    ConsentNotice,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    ConsentWithdrawalRequest,
    MaterialUsePurpose,
    SessionAdmissionErrorCode,
    SessionAdmissionNextAction,
    TableBindingStatus,
    TableRegistration,
    TableRegistrationStatus,
    canonical_payload_digest,
    idempotency_key_digest,
)

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)
LATER = datetime(2026, 7, 29, 12, 1, tzinfo=UTC)
DIGEST = hashlib.sha256(b"Synthetic controlled notice").hexdigest()


def _policy(binding_id):
    gate_revision_id = uuid4()
    scope = "c" * 64

    def notice(purpose):
        return ConsentNotice(
            purpose=purpose,
            version="synthetic-v1",
            content="Synthetic controlled notice",
            content_sha256=DIGEST,
            scope_fingerprint=scope,
            gate_revision_id=gate_revision_id,
            collection_scope="Synthetic marked clips",
            use_description="Synthetic purpose",
            retention_description="Synthetic retention",
            access_description="Synthetic access",
            deletion_description="Synthetic deletion",
            operator_name="Synthetic operator",
            operator_contact="privacy@example.invalid",
        )

    view = ConsentPolicyView(
        binding_id=binding_id,
        policy_id="d" * 64,
        gate_revision_id=gate_revision_id,
        scope_fingerprint=scope,
        service_notice=notice(MaterialUsePurpose.SERVICE_ANALYSIS),
        training_notice=notice(MaterialUsePurpose.TRAINING),
    )
    return ResolvedConsentPolicy(view, frozenset(MaterialUsePurpose))


def _request(
    binding_id,
    key,
    *,
    training=ConsentDecision.DECLINED,
    service=ConsentDecision.GRANTED,
):
    policy = _policy(binding_id)
    request = ConsentSubmissionRequest(
        schema_version="1.0",
        binding_id=binding_id,
        service_decision=service,
        training_decision=training,
        policy_id=policy.view.policy_id,
        gate_revision_id=policy.view.gate_revision_id,
        service_notice_version=policy.view.service_notice.version,
        training_notice_version=policy.view.training_notice.version,
        idempotency_key=key,
    )
    return policy, request


def _admit(repository, user_id, request, policy, candidate):
    return repository.admit(
        request=request,
        user_id=user_id,
        policy=policy,
        key_digest=idempotency_key_digest(request.idempotency_key),
        payload_digest=canonical_payload_digest(request),
        candidate_factory=lambda service_granted, need_session: candidate,
    )


def _private_write(store: Path, payload: dict) -> None:
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.chmod(0o600)


def _provision_with_pending(repository, *, suffix: str, count: int):
    provisioned = repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id=f"synthetic-venue-{suffix}",
            table_id=f"synthetic-table-{suffix}",
            device_id=f"synthetic-device-{suffix}",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    users = [uuid4() for _ in range(count)]
    bindings = [
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=user,
            candidate_factory=lambda: PendingBindingCandidate(uuid4(), NOW),
        ).binding
        for user in users
    ]
    return users, bindings


class _CapturePolicy:
    def verify_capture(self, table):
        assert table.status is TableRegistrationStatus.ACTIVE


def _admit_worker(
    store_text,
    user_text,
    request_payload,
    policy_payload,
    candidate_payload,
    start,
    queue,
):
    try:
        request = ConsentSubmissionRequest.model_validate(request_payload)
        policy = ResolvedConsentPolicy(
            ConsentPolicyView.model_validate(policy_payload), frozenset(MaterialUsePurpose)
        )
        candidate = AdmissionCandidate(
            UUID(candidate_payload[0]),
            UUID(candidate_payload[1]),
            None,
            UUID(candidate_payload[2]),
            datetime.fromisoformat(candidate_payload[3]),
        )
        start.wait(timeout=10)
        _admit(JsonTableBindingRepository(store_text), UUID(user_text), request, policy, candidate)
        queue.put("ok")
    except SessionAdmissionRepositoryDomainError as exc:
        queue.put(exc.code.value)
    except Exception as exc:  # pragma: no cover - surfaced by the parent assertion
        queue.put(type(exc).__name__)


def test_new_store_uses_explicit_v3_root_without_parallel_session_file(tmp_path: Path) -> None:
    repository = JsonTableBindingRepository(tmp_path / "authority.json")
    repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id="synthetic-venue",
            table_id="synthetic-table",
            device_id="synthetic-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    payload = json.loads((tmp_path / "authority.json").read_text())
    assert payload["schema_version"] == "3.0"
    assert set(payload) == {
        "schema_version",
        "revision",
        "table_registrations",
        "qr_registrations",
        "bindings",
        "consent_decisions",
        "sessions",
        "session_participants",
        "idempotency_records",
        "capture_idempotency_records",
        "capture_audit_events",
        "upload_attempts",
        "media_bindings",
    }
    assert list(tmp_path.glob("*session*")) == []


def test_capture_start_freezes_roster_and_end_releases_capacity(tmp_path: Path) -> None:
    repository = JsonTableBindingRepository(tmp_path / "capture" / "authority.json")
    users, bindings = _provision_with_pending(repository, suffix="capture", count=1)
    policy, request = _request(bindings[0].binding_id, "A" * 43)
    admitted = _admit(
        repository,
        users[0],
        request,
        policy,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), NOW),
    )
    principal = StaffPrincipal(
        staff_id=uuid4(), role="STAFF", venue_ids=("synthetic-venue-capture",)
    )
    capture = CaptureOperationRequest(
        schema_version="1.0", session_id=admitted.session.session_id, idempotency_key="A" * 43
    )
    started = repository.start_capture(
        request=capture, principal=principal, now=LATER, policy_provider=_CapturePolicy()
    )
    assert started.roster_frozen_at == LATER
    ended = repository.end_capture(
        request=CaptureOperationRequest(
            schema_version="1.0",
            session_id=admitted.session.session_id,
            idempotency_key=base64.urlsafe_b64encode(bytes([1]) * 32).decode().rstrip("="),
        ),
        principal=principal,
        now=LATER.replace(minute=2),
        policy_provider=_CapturePolicy(),
    )
    assert ended.status is CloudSessionStatus.ENDED_AWAITING_UPLOAD
    snapshot = repository.find_owned(binding_id=bindings[0].binding_id, user_id=users[0])
    assert snapshot.binding.status is TableBindingStatus.COMPLETED
    assert snapshot.capacity_status.value == "AVAILABLE"


def test_atomic_admission_shared_session_capacity_idempotency_and_withdrawal(
    tmp_path: Path,
) -> None:
    store = tmp_path / "authority.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id="synthetic-venue",
            table_id="synthetic-table",
            device_id="synthetic-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    users = [uuid4() for _ in range(3)]
    bindings = [
        repository.resolve_or_create_pending(
            qr_token=provisioned.qr_token,
            user_id=user,
            candidate_factory=lambda: PendingBindingCandidate(uuid4(), NOW),
        ).binding
        for user in users
    ]
    before = json.loads(store.read_text())["revision"]

    policy1, request1 = _request(bindings[0].binding_id, "A" * 43, training=ConsentDecision.GRANTED)
    first = _admit(
        repository,
        users[0],
        request1,
        policy1,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
    )
    assert first.session is not None and first.participant is not None
    after_first = json.loads(store.read_text())["revision"]
    assert after_first == before + 1
    replay = _admit(
        repository,
        users[0],
        request1,
        policy1,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
    )
    assert replay == first
    assert json.loads(store.read_text())["revision"] == after_first

    changed = request1.model_copy(update={"training_decision": ConsentDecision.DECLINED})
    with pytest.raises(SessionAdmissionRepositoryDomainError) as conflict:
        _admit(
            repository,
            users[0],
            changed,
            policy1,
            AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
        )
    assert conflict.value.code is SessionAdmissionErrorCode.IDEMPOTENCY_CONFLICT

    policy2, request2 = _request(bindings[1].binding_id, "A" * 43)
    second = _admit(
        repository,
        users[1],
        request2,
        policy2,
        AdmissionCandidate(uuid4(), uuid4(), None, uuid4(), LATER),
    )
    assert second.session == first.session
    assert second.participant.user_id != first.participant.user_id

    policy3, request3 = _request(bindings[2].binding_id, "A" * 43)
    raw_before_capacity = store.read_bytes()
    with pytest.raises(SessionAdmissionRepositoryDomainError) as capacity:
        _admit(
            repository,
            users[2],
            request3,
            policy3,
            AdmissionCandidate(uuid4(), uuid4(), None, uuid4(), LATER),
        )
    assert capacity.value.code is SessionAdmissionErrorCode.TABLE_CAPACITY_REACHED
    assert store.read_bytes() == raw_before_capacity

    withdrawal = ConsentWithdrawalRequest(
        participant_id=first.participant.participant_id,
        purpose=MaterialUsePurpose.TRAINING,
        idempotency_key="A" * 43,
    )
    withdrawal_result = repository.withdraw(
        request=withdrawal,
        user_id=users[0],
        key_digest=idempotency_key_digest(withdrawal.idempotency_key),
        payload_digest=canonical_payload_digest(withdrawal),
        candidate_factory=lambda: WithdrawalCandidate(uuid4(), LATER),
    )
    assert withdrawal_result.training_consent.decision is ConsentDecision.WITHDRAWN
    withdrawal_revision = json.loads(store.read_text())["revision"]
    replay_withdrawal = repository.withdraw(
        request=withdrawal,
        user_id=users[0],
        key_digest=idempotency_key_digest(withdrawal.idempotency_key),
        payload_digest=canonical_payload_digest(withdrawal),
        candidate_factory=lambda: (_ for _ in ()).throw(AssertionError("candidate called")),
    )
    assert replay_withdrawal == withdrawal_result
    assert json.loads(store.read_text())["revision"] == withdrawal_revision
    current = repository.read_admission(binding_id=first.binding.binding_id, user_id=users[0])
    assert current.training_consent.decision is ConsentDecision.WITHDRAWN
    assert repository.read_admission(binding_id=first.binding.binding_id, user_id=users[1]) is None


def test_service_decline_is_a_single_audit_mutation_without_capacity_or_session(
    tmp_path: Path,
) -> None:
    store = tmp_path / "authority.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id="synthetic-venue",
            table_id="synthetic-table",
            device_id="synthetic-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    user_id = uuid4()
    binding = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=user_id,
        candidate_factory=lambda: PendingBindingCandidate(uuid4(), NOW),
    ).binding
    policy, request = _request(
        binding.binding_id,
        "A" * 43,
        service=ConsentDecision.DECLINED,
    )
    result = _admit(
        repository,
        user_id,
        request,
        policy,
        AdmissionCandidate(uuid4(), uuid4(), None, None, LATER),
    )
    assert result.session is None and result.participant is None
    assert result.binding.status.value == "PENDING_CONSENT"
    payload = json.loads(store.read_text())
    assert payload["sessions"] == {}
    assert payload["session_participants"] == {}
    assert repository.read_admission(binding_id=binding.binding_id, user_id=user_id) == result


def test_last_seat_thread_and_process_races_have_one_winner(tmp_path: Path) -> None:
    for mode in ("thread", "process"):
        store = tmp_path / mode / "authority.json"
        repository = JsonTableBindingRepository(store)
        provisioned = repository.provision_table(
            TableRegistration(
                table_registration_id=uuid4(),
                venue_id=f"synthetic-venue-{mode}",
                table_id="synthetic-table",
                device_id=f"synthetic-device-{mode}",
                venue_display_label="Synthetic venue",
                table_display_label="Table 1",
                current_qr_version=1,
                status=TableRegistrationStatus.ACTIVE,
            )
        )
        users = [uuid4() for _ in range(3)]
        bindings = [
            repository.resolve_or_create_pending(
                qr_token=provisioned.qr_token,
                user_id=user,
                candidate_factory=lambda: PendingBindingCandidate(uuid4(), NOW),
            ).binding
            for user in users
        ]
        first_policy, first_request = _request(bindings[0].binding_id, "A" * 43)
        _admit(
            repository,
            users[0],
            first_request,
            first_policy,
            AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
        )
        attempts = [_request(bindings[index].binding_id, "A" * 43) for index in (1, 2)]
        candidates = [AdmissionCandidate(uuid4(), uuid4(), None, uuid4(), LATER) for _ in range(2)]
        if mode == "thread":
            barrier = threading.Barrier(2)

            def attempt(
                index,
                *,
                current_barrier=barrier,
                current_store=store,
                current_users=users,
                current_attempts=attempts,
                current_candidates=candidates,
            ):
                current_barrier.wait(timeout=10)
                try:
                    _admit(
                        JsonTableBindingRepository(current_store),
                        current_users[index + 1],
                        current_attempts[index][1],
                        current_attempts[index][0],
                        current_candidates[index],
                    )
                    return "ok"
                except SessionAdmissionRepositoryDomainError as exc:
                    return exc.code.value

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(attempt, range(2)))
        else:
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            queue = context.Queue()
            processes = []
            for index in range(2):
                candidate = candidates[index]
                process = context.Process(
                    target=_admit_worker,
                    args=(
                        str(store),
                        str(users[index + 1]),
                        attempts[index][1].model_dump(mode="json"),
                        attempts[index][0].view.model_dump(mode="json"),
                        (
                            str(candidate.service_consent_id),
                            str(candidate.training_consent_id),
                            str(candidate.participant_id),
                            candidate.created_at.isoformat(),
                        ),
                        start,
                        queue,
                    ),
                )
                processes.append(process)
                process.start()
            start.set()
            results = [queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                assert process.exitcode == 0
        assert sorted(results) == ["TABLE_CAPACITY_REACHED", "ok"]


def test_v1_reads_are_byte_stable_and_first_mutation_migrates_once(tmp_path: Path) -> None:
    store = tmp_path / "legacy" / "authority.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id="synthetic-legacy-venue",
            table_id="synthetic-table",
            device_id="synthetic-legacy-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    payload = json.loads(store.read_text())
    payload["schema_version"] = "1.0"
    for key in (
        "consent_decisions",
        "sessions",
        "session_participants",
        "idempotency_records",
        "capture_idempotency_records",
        "capture_audit_events",
        "upload_attempts",
        "media_bindings",
    ):
        payload.pop(key)
    store.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    store.chmod(0o600)
    before = store.read_bytes()
    assert repository.find_owned(binding_id=uuid4(), user_id=uuid4()) is None
    assert store.read_bytes() == before
    revision = payload["revision"]
    repository.rotate_qr(provisioned.table_registration.table_registration_id)
    migrated = json.loads(store.read_text())
    assert migrated["schema_version"] == "3.0"
    assert migrated["revision"] == revision + 1


def test_admission_write_failure_keeps_all_authority_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import billiards_ai.cloud.table_binding.repository as module

    store = tmp_path / "failure" / "authority.json"
    repository = JsonTableBindingRepository(store)
    provisioned = repository.provision_table(
        TableRegistration(
            table_registration_id=uuid4(),
            venue_id="synthetic-failure-venue",
            table_id="synthetic-table",
            device_id="synthetic-failure-device",
            venue_display_label="Synthetic venue",
            table_display_label="Table 1",
            current_qr_version=1,
            status=TableRegistrationStatus.ACTIVE,
        )
    )
    user_id = uuid4()
    binding = repository.resolve_or_create_pending(
        qr_token=provisioned.qr_token,
        user_id=user_id,
        candidate_factory=lambda: PendingBindingCandidate(uuid4(), NOW),
    ).binding
    policy, request = _request(binding.binding_id, "A" * 43)
    before = store.read_bytes()
    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(TableBindingRepositoryError):
        _admit(
            repository,
            user_id,
            request,
            policy,
            AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
        )
    assert store.read_bytes() == before


def test_review_decline_while_table_session_exists_keeps_both_consent_links_null(
    tmp_path: Path,
) -> None:
    store = tmp_path / "decline-with-open" / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix="decline", count=2)
    first_policy, first_request = _request(bindings[0].binding_id, "A" * 43)
    _admit(
        repository,
        users[0],
        first_request,
        first_policy,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
    )
    decline_policy, decline_request = _request(
        bindings[1].binding_id,
        "A" * 43,
        service=ConsentDecision.DECLINED,
    )
    declined = _admit(
        repository,
        users[1],
        decline_request,
        decline_policy,
        AdmissionCandidate(uuid4(), uuid4(), None, None, LATER),
    )
    assert declined.session is None and declined.participant is None
    assert declined.service_consent.session_id is None
    assert declined.service_consent.participant_id is None
    assert declined.training_consent.session_id is None
    assert declined.training_consent.participant_id is None


def test_review_old_decline_replay_after_grant_preserves_original_pending_outcome(
    tmp_path: Path,
) -> None:
    store = tmp_path / "decline-replay" / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix="replay", count=1)
    decline_policy, decline_request = _request(
        bindings[0].binding_id,
        "A" * 43,
        service=ConsentDecision.DECLINED,
    )
    declined = _admit(
        repository,
        users[0],
        decline_request,
        decline_policy,
        AdmissionCandidate(uuid4(), uuid4(), None, None, LATER),
    )
    grant_policy, grant_request = _request(bindings[0].binding_id, "B" * 42 + "A")
    granted = _admit(
        repository,
        users[0],
        grant_request,
        grant_policy,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
    )
    assert granted.binding.status is TableBindingStatus.ACTIVE
    replay = repository.recover_admission(
        request=decline_request,
        user_id=users[0],
        key_digest=idempotency_key_digest(decline_request.idempotency_key),
        payload_digest=canonical_payload_digest(decline_request),
    )
    assert replay == declined
    assert replay.binding.status is TableBindingStatus.PENDING_CONSENT
    assert replay.binding.activated_at is None
    assert replay.session is None and replay.participant is None

    class ReplayIdentity:
        def current_user_id(self):
            return users[0]

    class PolicyMustNotRun:
        def resolve(self, snapshot):
            raise AssertionError("idempotent replay must not resolve current policy")

    service = SessionAdmissionService(
        repository=repository,
        identity_provider=ReplayIdentity(),
        policy_provider=PolicyMustNotRun(),
    )
    replay_view = service.submit(decline_request)
    assert replay_view.next_action is SessionAdmissionNextAction.REVIEW_CONSENT
    assert replay_view.session_id is None and replay_view.activated_at is None


def test_review_repository_reverifies_policy_inside_atomic_admit(tmp_path: Path) -> None:
    store = tmp_path / "policy-reverify" / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix="policy", count=1)
    resolved, request = _request(bindings[0].binding_id, "A" * 43)
    calls = 0

    class RejectingEvidence:
        view = resolved.view

        def verify_submission(self, submitted):
            nonlocal calls
            calls += 1
            assert submitted == request
            raise SessionAdmissionRepositoryDomainError(
                SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
            )

    before = store.read_bytes()
    with pytest.raises(SessionAdmissionRepositoryDomainError) as raised:
        _admit(
            repository,
            users[0],
            request,
            RejectingEvidence(),
            AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
        )
    assert raised.value.code is SessionAdmissionErrorCode.REAL_VIDEO_GATE_NOT_APPROVED
    assert calls == 1
    assert store.read_bytes() == before


def test_review_candidate_factory_internal_typeerror_is_closed_and_called_once(
    tmp_path: Path,
) -> None:
    store = tmp_path / "factory-typeerror" / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix="factory", count=1)
    policy, request = _request(bindings[0].binding_id, "A" * 43)
    calls = 0

    def broken_factory(service_granted: bool, need_session: bool):
        nonlocal calls
        calls += 1
        assert service_granted and need_session
        raise TypeError("synthetic internal factory failure")

    before = store.read_bytes()
    with pytest.raises(TableBindingRepositoryError):
        repository.admit(
            request=request,
            user_id=users[0],
            policy=policy,
            key_digest=idempotency_key_digest(request.idempotency_key),
            payload_digest=canonical_payload_digest(request),
            candidate_factory=broken_factory,
        )
    assert calls == 1
    assert store.read_bytes() == before


@pytest.mark.parametrize(
    "corruption",
    [
        "admit_result_participant_cross_owner",
        "withdraw_result_participant_cross_owner",
        "withdraw_result_consent_wrong_purpose",
    ],
)
def test_review_idempotency_results_cannot_cross_owner_operation_or_purpose(
    tmp_path: Path, corruption: str
) -> None:
    store = tmp_path / corruption / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix=corruption, count=2)
    outcomes = []
    for index in range(2):
        policy, request = _request(
            bindings[index].binding_id,
            "A" * 43,
            training=ConsentDecision.GRANTED,
        )
        outcomes.append(
            _admit(
                repository,
                users[index],
                request,
                policy,
                AdmissionCandidate(
                    uuid4(),
                    uuid4(),
                    uuid4() if index == 0 else None,
                    uuid4(),
                    LATER,
                ),
            )
        )
    withdrawal = ConsentWithdrawalRequest(
        participant_id=outcomes[0].participant.participant_id,
        purpose=MaterialUsePurpose.TRAINING,
        idempotency_key="A" * 43,
    )
    repository.withdraw(
        request=withdrawal,
        user_id=users[0],
        key_digest=idempotency_key_digest(withdrawal.idempotency_key),
        payload_digest=canonical_payload_digest(withdrawal),
        candidate_factory=lambda: WithdrawalCandidate(uuid4(), LATER),
    )
    payload = json.loads(store.read_text())
    admission_record = next(
        value
        for value in payload["idempotency_records"].values()
        if value["operation"] == "ADMIT" and value["user_id"] == str(users[0])
    )
    withdrawal_record = next(
        value
        for value in payload["idempotency_records"].values()
        if value["operation"] == "WITHDRAW" and value["user_id"] == str(users[0])
    )
    if corruption == "admit_result_participant_cross_owner":
        admission_record["result_participant_id"] = str(outcomes[1].participant.participant_id)
    elif corruption == "withdraw_result_participant_cross_owner":
        withdrawal_record["result_participant_id"] = str(outcomes[1].participant.participant_id)
    else:
        withdrawal_record["result_consent_ids"] = [str(outcomes[0].service_consent.consent_id)]
    _private_write(store, payload)
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=bindings[0].binding_id, user_id=users[0])


@pytest.mark.parametrize("corruption", ["disconnected_grant", "withdrawal_of_withdrawal"])
def test_review_each_linked_purpose_has_one_connected_grant_to_head_chain(
    tmp_path: Path, corruption: str
) -> None:
    store = tmp_path / corruption / "authority.json"
    repository = JsonTableBindingRepository(store)
    users, bindings = _provision_with_pending(repository, suffix=corruption, count=1)
    policy, request = _request(
        bindings[0].binding_id,
        "A" * 43,
        training=ConsentDecision.GRANTED,
    )
    admitted = _admit(
        repository,
        users[0],
        request,
        policy,
        AdmissionCandidate(uuid4(), uuid4(), uuid4(), uuid4(), LATER),
    )
    payload = json.loads(store.read_text())
    training = payload["consent_decisions"][str(admitted.training_consent.consent_id)]
    if corruption == "disconnected_grant":
        disconnected = ConsentDecisionRecord.model_validate(
            {**training, "consent_id": str(uuid4())}
        )
        payload["consent_decisions"][str(disconnected.consent_id)] = disconnected.model_dump(
            mode="json"
        )
    else:
        first_id = uuid4()
        first = ConsentDecisionRecord.model_validate(
            {
                **training,
                "consent_id": str(first_id),
                "decision": "WITHDRAWN",
                "decided_at": "2026-07-29T12:02:00Z",
                "withdrawn_at": "2026-07-29T12:02:00Z",
                "supersedes": str(admitted.training_consent.consent_id),
            }
        )
        second_id = uuid4()
        second = ConsentDecisionRecord.model_validate(
            {
                **first.model_dump(mode="json"),
                "consent_id": str(second_id),
                "decided_at": "2026-07-29T12:03:00Z",
                "withdrawn_at": "2026-07-29T12:03:00Z",
                "supersedes": str(first_id),
            }
        )
        payload["consent_decisions"][str(first_id)] = first.model_dump(mode="json")
        payload["consent_decisions"][str(second_id)] = second.model_dump(mode="json")
        payload["session_participants"][str(admitted.participant.participant_id)][
            "training_consent_id"
        ] = str(second_id)
    _private_write(store, payload)
    with pytest.raises(TableBindingRepositoryError):
        repository.find_owned(binding_id=bindings[0].binding_id, user_id=users[0])
