"""Trusted-identity orchestration for consent-backed session admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import UUID4, TypeAdapter

from billiards_ai.cloud.table_binding.errors import SessionAdmissionRepositoryDomainError
from billiards_ai.cloud.table_binding.identity import AuthenticatedUserProvider
from billiards_ai.cloud.table_binding.repository import (
    AdmissionCandidate,
    AdmissionSnapshot,
    BindingSnapshot,
    TableBindingRepository,
    WithdrawalCandidate,
)
from billiards_ai.shared.models import (
    ConsentDecision,
    ConsentPolicyView,
    ConsentSubmissionRequest,
    ConsentWithdrawalRequest,
    SessionAdmissionErrorCode,
    SessionAdmissionNextAction,
    SessionAdmissionView,
    canonical_payload_digest,
    idempotency_key_digest,
)

from .errors import (
    SessionAdmissionDomainError,
    SessionAdmissionServiceError,
)
from .policy import ConsentPolicyProvider, ResolvedConsentPolicy

_UUID4 = TypeAdapter(UUID4)


class SessionAdmissionService:
    def __init__(
        self,
        *,
        repository: TableBindingRepository,
        identity_provider: AuthenticatedUserProvider,
        policy_provider: ConsentPolicyProvider,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ):
        self._repository = repository
        self._identity_provider = identity_provider
        self._policy_provider = policy_provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    def _identity(self) -> UUID:
        try:
            return _UUID4.validate_python(self._identity_provider.current_user_id())
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.IDENTITY_REQUIRED
            ) from None

    @staticmethod
    def _submission(
        value: ConsentSubmissionRequest | Mapping[str, object],
    ) -> ConsentSubmissionRequest:
        try:
            payload = dict(value.__dict__) if isinstance(value, ConsentSubmissionRequest) else value
            return ConsentSubmissionRequest.model_validate(payload)
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.CONSENT_REQUEST_INVALID
            ) from None

    @staticmethod
    def _withdrawal(
        value: ConsentWithdrawalRequest | Mapping[str, object],
    ) -> ConsentWithdrawalRequest:
        try:
            payload = dict(value.__dict__) if isinstance(value, ConsentWithdrawalRequest) else value
            return ConsentWithdrawalRequest.model_validate(payload)
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.CONSENT_REQUEST_INVALID
            ) from None

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock is invalid")
        return value.astimezone(UTC)

    def _uuid(self) -> UUID:
        return _UUID4.validate_python(self._uuid_factory())

    def _candidate(
        self, service_granted: bool = True, need_session: bool = True
    ) -> AdmissionCandidate:
        try:
            service_id = self._uuid()
            training_id = self._uuid()
            now = self._now()
            if not service_granted:
                return AdmissionCandidate(service_id, training_id, None, None, now)
            return AdmissionCandidate(
                service_id,
                training_id,
                self._uuid() if need_session else None,
                self._uuid(),
                now,
            )
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None

    def _withdrawal_candidate(self) -> WithdrawalCandidate:
        try:
            return WithdrawalCandidate(self._uuid(), self._now())
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None

    def _owned(self, *, binding_id: UUID, user_id: UUID) -> BindingSnapshot:
        try:
            snapshot = self._repository.find_owned(binding_id=binding_id, user_id=user_id)
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None
        if snapshot is None:
            raise SessionAdmissionServiceError(SessionAdmissionErrorCode.RESOURCE_NOT_FOUND)
        return snapshot

    def read_policy(self, binding_id: UUID) -> ConsentPolicyView:
        user_id = self._identity()
        snapshot = self._owned(binding_id=binding_id, user_id=user_id)
        try:
            return self._policy_provider.resolve(snapshot).view
        except SessionAdmissionDomainError as error:
            raise SessionAdmissionServiceError(error.code) from None
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None

    def read(self, binding_id: UUID) -> SessionAdmissionView:
        user_id = self._identity()
        try:
            snapshot = self._repository.read_admission(
                binding_id=_UUID4.validate_python(binding_id), user_id=user_id
            )
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None
        if snapshot is None:
            raise SessionAdmissionServiceError(SessionAdmissionErrorCode.RESOURCE_NOT_FOUND)
        return self._view(snapshot)

    @staticmethod
    def _view(snapshot: AdmissionSnapshot) -> SessionAdmissionView:
        try:
            return SessionAdmissionView(
                binding_id=snapshot.binding.binding_id,
                session_id=None if snapshot.session is None else snapshot.session.session_id,
                participant_id=(
                    None if snapshot.participant is None else snapshot.participant.participant_id
                ),
                venue_display_label=snapshot.table.venue_display_label,
                table_display_label=snapshot.table.table_display_label,
                service_decision=snapshot.service_consent.decision,
                training_decision=snapshot.training_consent.decision,
                service_notice_version=snapshot.service_consent.notice_version,
                training_notice_version=snapshot.training_consent.notice_version,
                session_status=None if snapshot.session is None else snapshot.session.status,
                created_at=None if snapshot.session is None else snapshot.session.created_at,
                activated_at=snapshot.binding.activated_at,
                next_action=(
                    SessionAdmissionNextAction.REVIEW_CONSENT
                    if snapshot.service_consent.decision is not ConsentDecision.GRANTED
                    else SessionAdmissionNextAction.WAITING_FOR_UPLOAD
                ),
            )
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None

    def submit(
        self, value: ConsentSubmissionRequest | Mapping[str, object]
    ) -> SessionAdmissionView:
        request = self._submission(value)
        user_id = self._identity()
        snapshot = self._owned(binding_id=request.binding_id, user_id=user_id)
        key_digest = idempotency_key_digest(request.idempotency_key)
        payload_digest = canonical_payload_digest(request)
        try:
            recovered = self._repository.recover_admission(
                request=request,
                user_id=user_id,
                key_digest=key_digest,
                payload_digest=payload_digest,
            )
            if recovered is not None:
                return self._view(recovered)
            policy: ResolvedConsentPolicy = self._policy_provider.resolve(snapshot)
            policy.verify_submission(request)
            result = self._repository.admit(
                request=request,
                user_id=user_id,
                policy=policy,
                key_digest=key_digest,
                payload_digest=payload_digest,
                candidate_factory=self._candidate,
            )
        except SessionAdmissionDomainError as error:
            raise SessionAdmissionServiceError(error.code) from None
        except SessionAdmissionRepositoryDomainError as error:
            raise SessionAdmissionServiceError(error.code) from None
        except SessionAdmissionServiceError:
            raise
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None
        return self._view(result)

    def withdraw(
        self, value: ConsentWithdrawalRequest | Mapping[str, object]
    ) -> SessionAdmissionView:
        request = self._withdrawal(value)
        user_id = self._identity()
        try:
            result = self._repository.withdraw(
                request=request,
                user_id=user_id,
                key_digest=idempotency_key_digest(request.idempotency_key),
                payload_digest=canonical_payload_digest(request),
                candidate_factory=self._withdrawal_candidate,
            )
        except SessionAdmissionDomainError as error:
            raise SessionAdmissionServiceError(error.code) from None
        except SessionAdmissionRepositoryDomainError as error:
            raise SessionAdmissionServiceError(error.code) from None
        except SessionAdmissionServiceError:
            raise
        except Exception:
            raise SessionAdmissionServiceError(
                SessionAdmissionErrorCode.SESSION_ADMISSION_REPOSITORY_UNAVAILABLE
            ) from None
        return self._view(result)
