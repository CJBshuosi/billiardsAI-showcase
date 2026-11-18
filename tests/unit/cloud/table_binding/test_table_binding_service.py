"""Story 6.1 unit coverage for the trusted identity and owner-only service."""

from __future__ import annotations

import inspect
import traceback
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from billiards_ai.cloud.table_binding import (
    TableBindingDomainError,
    TableBindingRepositoryError,
    TableBindingService,
    TableBindingServiceError,
)
from billiards_ai.cloud.table_binding.repository import BindingSnapshot
from billiards_ai.shared.models import (
    JoinTableRequest,
    TableBinding,
    TableBindingErrorCode,
    TableBindingStatus,
    TableCapacityStatus,
    TableRegistration,
    TableRegistrationStatus,
)

TOKEN = "A" * 43
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class IdentityProvider:
    def __init__(self, user_id: UUID | None):
        self.user_id = user_id

    def current_user_id(self) -> UUID:
        if self.user_id is None:
            raise RuntimeError("unverified synthetic identity")
        return self.user_id


class UnexpectedIdentityFailure:
    def current_user_id(self) -> UUID:
        raise KeyError("synthetic private identity detail")


class FakeRepository:
    def __init__(self, snapshot: BindingSnapshot | None = None):
        self.snapshot = snapshot
        self.join_calls = 0
        self.read_calls = 0
        self.failure: Exception | None = None

    def resolve_or_create_pending(
        self,
        *,
        qr_token: str,
        user_id: UUID,
        candidate_factory: object,
    ) -> BindingSnapshot:
        self.join_calls += 1
        if self.failure is not None:
            raise self.failure
        assert qr_token == TOKEN
        assert callable(candidate_factory)
        candidate_binding = candidate_factory()
        assert candidate_binding.binding_id is not None
        assert candidate_binding.created_at == NOW
        if self.snapshot is None:
            raise AssertionError("test must configure a repository snapshot")
        return self.snapshot

    def find_owned(self, *, binding_id: UUID, user_id: UUID) -> BindingSnapshot | None:
        self.read_calls += 1
        if self.failure is not None:
            raise self.failure
        if self.snapshot is None:
            return None
        binding = self.snapshot.binding
        if binding.binding_id != binding_id or binding.user_id != user_id:
            return None
        return self.snapshot


def _snapshot(*, user_id: UUID | None = None) -> BindingSnapshot:
    owner_id = user_id or uuid4()
    table_id = uuid4()
    qr_id = uuid4()
    table = TableRegistration(
        table_registration_id=table_id,
        venue_id="venue-synthetic",
        table_id="table-03",
        device_id="device-synthetic-03",
        venue_display_label="Dilan 球厅",
        table_display_label="3 号台",
        current_qr_version=1,
        status=TableRegistrationStatus.ACTIVE,
    )
    binding = TableBinding(
        binding_id=uuid4(),
        user_id=owner_id,
        table_registration_id=table_id,
        qr_registration_id=qr_id,
        qr_version=1,
        status=TableBindingStatus.PENDING_CONSENT,
        created_at=NOW,
        activated_at=None,
    )
    return BindingSnapshot(
        binding=binding,
        table=table,
        capacity_status=TableCapacityStatus.AVAILABLE,
    )


def _service(*, owner_id: UUID | None, repository: FakeRepository) -> TableBindingService:
    return TableBindingService(
        repository=repository,
        identity_provider=IdentityProvider(owner_id),
        clock=lambda: NOW,
    )


def test_public_methods_do_not_accept_caller_identity_or_credentials() -> None:
    """Adding caller-controlled identity parameters would reopen the authority boundary."""
    join_parameters = set(inspect.signature(TableBindingService.join).parameters)
    read_parameters = set(inspect.signature(TableBindingService.read).parameters)
    for forbidden in {
        "user_id",
        "openid",
        "jwt",
        "session_key",
        "code",
        "device_id",
    }:
        assert forbidden not in join_parameters
        assert forbidden not in read_parameters


def test_identity_failure_is_closed_before_repository_access() -> None:
    repository = FakeRepository()
    service = _service(owner_id=None, repository=repository)
    with pytest.raises(TableBindingServiceError) as raised:
        service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
    assert raised.value.code is TableBindingErrorCode.IDENTITY_REQUIRED
    assert repository.join_calls == 0


def test_unexpected_identity_failure_is_also_closed_before_repository_access() -> None:
    repository = FakeRepository()
    service = TableBindingService(
        repository=repository,
        identity_provider=UnexpectedIdentityFailure(),
        clock=lambda: NOW,
    )
    with pytest.raises(TableBindingServiceError) as raised:
        service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
    assert raised.value.code is TableBindingErrorCode.IDENTITY_REQUIRED
    assert "synthetic private identity detail" not in str(raised.value)
    assert repository.join_calls == 0


def test_join_builds_only_pending_and_projects_safe_registration_labels() -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    candidate_id = snapshot.binding.binding_id
    repository = FakeRepository(snapshot)
    service = _service(owner_id=owner_id, repository=repository)

    view = service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))

    assert view.binding_id == candidate_id
    assert view.binding_status is TableBindingStatus.PENDING_CONSENT
    assert view.venue_display_label == "Dilan 球厅"
    assert view.table_display_label == "3 号台"
    assert view.next_action.value == "REVIEW_CONSENT"
    assert repository.join_calls == 1


def test_join_uses_injected_uuid_factory_only_when_repository_creates() -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    candidate_id = uuid4()
    calls = 0

    def uuid_factory() -> UUID:
        nonlocal calls
        calls += 1
        return candidate_id

    service = TableBindingService(
        repository=FakeRepository(snapshot),
        identity_provider=IdentityProvider(owner_id),
        clock=lambda: NOW,
        uuid_factory=uuid_factory,
    )

    service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))

    assert calls == 1


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"schema_version": "2.0", "qr_token": TOKEN}, "QR_VERSION_UNSUPPORTED"),
        ({"schema_version": "1.0", "qr_token": "bad"}, "QR_INVALID"),
        ({"schema_version": "1.0", "qr_token": TOKEN, "user_id": str(uuid4())}, "QR_INVALID"),
    ],
)
def test_join_maps_invalid_untrusted_payloads_without_repository_access(
    payload: dict[str, object], expected: str
) -> None:
    owner_id = uuid4()
    repository = FakeRepository(_snapshot(user_id=owner_id))
    service = _service(owner_id=owner_id, repository=repository)
    with pytest.raises(TableBindingServiceError) as raised:
        service.join(payload)
    assert raised.value.code.value == expected
    assert repository.join_calls == 0


def test_repository_failure_maps_to_one_closed_non_sensitive_code() -> None:
    owner_id = uuid4()
    repository = FakeRepository(_snapshot(user_id=owner_id))
    repository.failure = TableBindingRepositoryError("synthetic raw store detail")
    service = _service(owner_id=owner_id, repository=repository)
    with pytest.raises(TableBindingServiceError) as raised:
        service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
    assert raised.value.code is TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
    assert "synthetic raw store detail" not in str(raised.value)
    assert TOKEN not in str(raised.value)


def test_sensitive_locator_is_not_logged_on_failure(caplog: pytest.LogCaptureFixture) -> None:
    owner_id = uuid4()
    repository = FakeRepository(_snapshot(user_id=owner_id))
    repository.failure = TableBindingRepositoryError("synthetic repository failure")
    service = _service(owner_id=owner_id, repository=repository)
    with pytest.raises(TableBindingServiceError):
        service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
    assert TOKEN not in caplog.text


def test_unexpected_repository_exception_still_fails_closed() -> None:
    owner_id = uuid4()
    repository = FakeRepository(_snapshot(user_id=owner_id))
    repository.failure = KeyError("synthetic private detail")
    service = _service(owner_id=owner_id, repository=repository)
    with pytest.raises(TableBindingServiceError) as raised:
        service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
    assert raised.value.code is TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
    assert "synthetic private detail" not in str(raised.value)


def test_owner_can_read_but_foreign_and_absent_are_indistinguishable() -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    owner_service = _service(owner_id=owner_id, repository=FakeRepository(snapshot))
    assert owner_service.read(snapshot.binding.binding_id).binding_id == snapshot.binding.binding_id

    foreign_service = _service(owner_id=uuid4(), repository=FakeRepository(snapshot))
    absent_service = _service(owner_id=owner_id, repository=FakeRepository())
    codes: list[TableBindingErrorCode] = []
    for service in (foreign_service, absent_service):
        with pytest.raises(TableBindingServiceError) as raised:
            service.read(snapshot.binding.binding_id)
        codes.append(raised.value.code)
    assert codes == [
        TableBindingErrorCode.RESOURCE_NOT_FOUND,
        TableBindingErrorCode.RESOURCE_NOT_FOUND,
    ]


def test_read_reauthenticates_on_every_call() -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    identity = IdentityProvider(owner_id)
    service = TableBindingService(
        repository=FakeRepository(snapshot),
        identity_provider=identity,
        clock=lambda: NOW,
    )
    service.read(snapshot.binding.binding_id)
    identity.user_id = None
    with pytest.raises(TableBindingServiceError) as raised:
        service.read(snapshot.binding.binding_id)
    assert raised.value.code is TableBindingErrorCode.IDENTITY_REQUIRED


def test_review_service_revalidates_constructed_request_and_hides_traceback_details() -> None:
    raw_locator = "raw-private-qr-that-must-not-enter-a-public-traceback"
    owner_id = uuid4()
    repository = FakeRepository(_snapshot(user_id=owner_id))
    request = JoinTableRequest.model_construct(schema_version="1.0", qr_token=raw_locator)
    service = _service(owner_id=owner_id, repository=repository)

    with pytest.raises(TableBindingServiceError) as raised:
        service.join(request)

    rendered = "".join(
        traceback.format_exception(type(raised.value), raised.value, raised.value.__traceback__)
    )
    assert raised.value.code is TableBindingErrorCode.QR_INVALID
    assert raised.value.__cause__ is None
    assert raw_locator not in rendered
    assert "ValidationError" not in rendered
    assert repository.join_calls == 0


def test_review_service_revalidates_snapshot_and_closes_view_failures() -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    poisoned = BindingSnapshot(
        binding=TableBinding.model_construct(
            **{**snapshot.binding.model_dump(), "schema_version": "private-invalid-schema"}
        ),
        table=snapshot.table,
        capacity_status=snapshot.capacity_status,
    )
    service = _service(owner_id=owner_id, repository=FakeRepository(poisoned))

    with pytest.raises(TableBindingServiceError) as raised:
        service.read(snapshot.binding.binding_id)

    rendered = "".join(
        traceback.format_exception(type(raised.value), raised.value, raised.value.__traceback__)
    )
    assert raised.value.code is TableBindingErrorCode.TABLE_BINDING_REPOSITORY_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert "private-invalid-schema" not in rendered
    assert "ValidationError" not in rendered


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("idempotent", None),
        ("unknown_qr", TableBindingErrorCode.QR_INVALID),
        ("full", TableBindingErrorCode.TABLE_CAPACITY_REACHED),
        ("cross_table", TableBindingErrorCode.USER_ALREADY_BOUND),
    ],
)
def test_review_candidate_generation_is_lazy_for_noncreating_decisions(
    decision: str,
    expected: TableBindingErrorCode | None,
) -> None:
    owner_id = uuid4()
    snapshot = _snapshot(user_id=owner_id)
    calls = 0

    class DecisionRepository:
        def resolve_or_create_pending(self, **kwargs: object) -> BindingSnapshot:
            assert callable(kwargs["candidate_factory"])
            if decision == "idempotent":
                return snapshot
            raise TableBindingDomainError(expected)

        def find_owned(self, **kwargs: object) -> None:
            return None

    def broken_clock() -> datetime:
        nonlocal calls
        calls += 1
        raise RuntimeError("clock must remain unused")

    def broken_uuid() -> UUID:
        raise RuntimeError("UUID must remain unused")

    service = TableBindingService(
        repository=DecisionRepository(),
        identity_provider=IdentityProvider(owner_id),
        clock=broken_clock,
        uuid_factory=broken_uuid,
    )
    if expected is None:
        assert service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN)).binding_id
    else:
        with pytest.raises(TableBindingServiceError) as raised:
            service.join(JoinTableRequest(schema_version="1.0", qr_token=TOKEN))
        assert raised.value.code is expected
    assert calls == 0
