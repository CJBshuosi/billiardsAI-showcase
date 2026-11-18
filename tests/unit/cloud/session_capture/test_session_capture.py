from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from billiards_ai.cloud.session_capture.errors import SessionCaptureServiceError
from billiards_ai.cloud.session_capture.identity import StaffPrincipal
from billiards_ai.cloud.session_capture.service import SessionCaptureService
from billiards_ai.shared.models.session_capture import SessionCaptureErrorCode

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)
KEY = "A" * 43


class Identity:
    def __init__(self, principal=None):
        self.principal = principal

    def current_staff(self):
        return self.principal


class Repository:
    def __init__(self):
        self.calls = []

    def start_capture(self, **kwargs):
        self.calls.append(("start", kwargs))
        return kwargs

    def correct_interval(self, **kwargs):
        self.calls.append(("correct", kwargs))
        return kwargs


def service(repository, identity, now=NOW):
    return SessionCaptureService(
        repository=repository,
        identity_provider=identity,
        policy_provider=object(),
        clock=lambda: now,
    )


def test_missing_staff_identity_fails_before_repository_or_candidate_generation() -> None:
    repository = Repository()
    with pytest.raises(SessionCaptureServiceError) as raised:
        service(repository, Identity()).start(
            {"schema_version": "1.0", "session_id": str(uuid4()), "idempotency_key": KEY}
        )
    assert raised.value.code is SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED
    assert repository.calls == []


def test_start_uses_server_time_and_trusted_staff_scope() -> None:
    repository = Repository()
    principal = StaffPrincipal(staff_id=uuid4(), role="STAFF", venue_ids=("venue-a",))
    service(repository, Identity(principal)).start(
        {"schema_version": "1.0", "session_id": str(uuid4()), "idempotency_key": KEY}
    )
    _, call = repository.calls[0]
    assert call["now"] == NOW
    assert call["principal"] == principal


@pytest.mark.parametrize(
    "start,end,code",
    [
        (NOW - timedelta(hours=48, microseconds=1), NOW, "BACKFILL_WINDOW_EXPIRED"),
        (NOW, NOW, "INVALID_TIME_RANGE"),
        (NOW, NOW + timedelta(microseconds=1), "INVALID_TIME_RANGE"),
    ],
)
def test_correction_enforces_server_48h_window(start, end, code) -> None:
    repository = Repository()
    principal = StaffPrincipal(staff_id=uuid4(), role="STAFF", venue_ids=("venue-a",))
    with pytest.raises(SessionCaptureServiceError) as raised:
        service(repository, Identity(principal)).correct_interval(
            {
                "schema_version": "1.0",
                "session_id": str(uuid4()),
                "expected_interval_revision": 1,
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "reason": "staff attestation",
                "idempotency_key": KEY,
            }
        )
    assert raised.value.code.value == code
    assert repository.calls == []
