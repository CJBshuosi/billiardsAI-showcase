"""Trusted-identity orchestration for pre-consent table admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import UUID4, TypeAdapter

from billiards_ai.shared.models import (
    JoinTableRequest,
    TableBinding,
    TableBindingErrorCode,
    TableBindingNextAction,
    TableBindingView,
    TableCapacityStatus,
    TableRegistration,
)

from .errors import (
    TableBindingDomainError,
    TableBindingServiceError,
)
from .identity import AuthenticatedUserProvider
from .repository import BindingSnapshot, PendingBindingCandidate, TableBindingRepository

_UUID4_ADAPTER = TypeAdapter(UUID4)


class TableBindingService:
    def __init__(
        self,
        *,
        repository: TableBindingRepository,
        identity_provider: AuthenticatedUserProvider,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] | None = None,
    ):
        self._repository = repository
        self._identity_provider = identity_provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._uuid_factory = uuid_factory or uuid4

    def _identity(self) -> UUID:
        try:
            return _UUID4_ADAPTER.validate_python(self._identity_provider.current_user_id())
        except Exception:
            raise TableBindingServiceError(TableBindingErrorCode.IDENTITY_REQUIRED) from None

    @staticmethod
    def _request(value: JoinTableRequest | Mapping[str, object]) -> JoinTableRequest:
        schema_version: object = None
        try:
            payload = (
                value.model_dump(mode="python") if isinstance(value, JoinTableRequest) else value
            )
            if isinstance(payload, Mapping):
                schema_version = payload.get("schema_version")
            return JoinTableRequest.model_validate(payload)
        except Exception:
            code = (
                TableBindingErrorCode.QR_VERSION_UNSUPPORTED
                if schema_version is not None and schema_version != "1.0"
                else TableBindingErrorCode.QR_INVALID
            )
            raise TableBindingServiceError(code) from None

    def _candidate(self) -> PendingBindingCandidate:
        try:
            binding_id = _UUID4_ADAPTER.validate_python(self._uuid_factory())
            created_at = self._clock()
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise ValueError("clock must return an aware datetime")
            return PendingBindingCandidate(
                binding_id=binding_id,
                created_at=created_at.astimezone(UTC),
            )
        except Exception:
            raise TableBindingServiceError(
                TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
            ) from None

    @staticmethod
    def _view(snapshot: BindingSnapshot) -> TableBindingView:
        try:
            binding_value = snapshot.binding
            table_value = snapshot.table
            binding = TableBinding.model_validate(binding_value.model_dump(mode="python"))
            table = TableRegistration.model_validate(table_value.model_dump(mode="python"))
            capacity = TableCapacityStatus(snapshot.capacity_status)
            return TableBindingView(
                binding_id=binding.binding_id,
                venue_display_label=table.venue_display_label,
                table_display_label=table.table_display_label,
                qr_version=binding.qr_version,
                binding_status=binding.status,
                capacity_status=capacity,
                created_at=binding.created_at,
                next_action=TableBindingNextAction.REVIEW_CONSENT,
            )
        except Exception:
            raise TableBindingServiceError(
                TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
            ) from None

    def join(self, request: JoinTableRequest | Mapping[str, object]) -> TableBindingView:
        parsed = self._request(request)
        user_id = self._identity()
        try:
            snapshot = self._repository.resolve_or_create_pending(
                qr_token=parsed.qr_token,
                user_id=user_id,
                candidate_factory=self._candidate,
            )
        except TableBindingDomainError as error:
            raise TableBindingServiceError(error.code) from None
        except TableBindingServiceError:
            raise
        except Exception:
            raise TableBindingServiceError(
                TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
            ) from None
        return self._view(snapshot)

    def read(self, binding_id: UUID) -> TableBindingView:
        user_id = self._identity()
        try:
            snapshot = self._repository.find_owned(binding_id=binding_id, user_id=user_id)
        except Exception:
            raise TableBindingServiceError(
                TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
            ) from None
        if snapshot is None:
            raise TableBindingServiceError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
        return self._view(snapshot)
