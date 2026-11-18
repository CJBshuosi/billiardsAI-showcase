from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.session_capture import (
    CaptureIntervalProvenance,
    CaptureIntervalRevision,
    CaptureOperationRequest,
    CloudSessionStatus,
    CorrectIntervalRequest,
    SessionCaptureErrorCode,
    SessionCaptureLimits,
    SessionCaptureStatus,
    UploadAttemptStatus,
    UploadPrepareRequest,
)

KEY = "A" * 43
NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)


def test_public_requests_are_strict_and_exclude_authoritative_fields() -> None:
    session_id = uuid4()
    request = CaptureOperationRequest(
        schema_version="1.0", session_id=session_id, idempotency_key=KEY
    )
    assert request.session_id == session_id
    for field in (
        "staff_id",
        "role",
        "venue_id",
        "table_id",
        "device_id",
        "participant_id",
        "gate_id",
        "storage_locator",
        "content_sha256",
        "size_bytes",
    ):
        with pytest.raises(ValidationError):
            CaptureOperationRequest.model_validate(
                {**request.model_dump(mode="json"), field: "attacker-controlled"}
            )
    with pytest.raises(ValidationError):
        CaptureOperationRequest.model_validate(
            {"schema_version": "9.9", "session_id": session_id, "idempotency_key": KEY}
        )


def test_interval_and_limits_reject_coercion_and_untrusted_time() -> None:
    with pytest.raises(ValidationError):
        CorrectIntervalRequest(
            schema_version="1.0",
            session_id=uuid4(),
            expected_interval_revision=True,
            start_at=NOW,
            end_at=NOW,
            reason="repair",
            idempotency_key=KEY,
        )
    with pytest.raises(ValidationError):
        CorrectIntervalRequest(
            schema_version="1.0",
            session_id=uuid4(),
            expected_interval_revision=1,
            start_at=1_722_256_400,
            end_at=1_722_256_401,
            reason="repair",
            idempotency_key=KEY,
        )
    with pytest.raises(ValidationError):
        SessionCaptureLimits(
            allowed_containers=(),
            allowed_video_codecs=("h264",),
            max_upload_bytes=1,
            max_duration_seconds=1,
            probe_timeout_seconds=1,
            interval_tolerance_seconds=1,
        )


def test_frozen_versioned_state_contract_covers_the_only_lifecycle() -> None:
    assert set(CloudSessionStatus) == {
        CloudSessionStatus.OPEN,
        CloudSessionStatus.ENDED_AWAITING_UPLOAD,
        CloudSessionStatus.MEDIA_BOUND,
        CloudSessionStatus.CANCELLED,
    }
    assert set(SessionCaptureStatus) == {SessionCaptureStatus.RECORDING}
    assert set(UploadAttemptStatus) == {
        UploadAttemptStatus.STAGED,
        UploadAttemptStatus.FAILED,
        UploadAttemptStatus.COMPLETED,
    }
    interval = CaptureIntervalRevision(
        interval_id=uuid4(),
        revision=1,
        start_at=NOW,
        end_at=NOW.replace(minute=1),
        provenance=CaptureIntervalProvenance.SERVER_MARKED,
        recorded_by=uuid4(),
        recorded_at=NOW,
    )
    with pytest.raises(ValidationError):
        interval.revision = 2


def test_upload_prepare_accepts_no_path_or_media_facts() -> None:
    request = UploadPrepareRequest(schema_version="1.0", session_id=uuid4(), idempotency_key=KEY)
    for field in ("path", "filename", "locator", "mime", "hash", "size", "duration"):
        with pytest.raises(ValidationError):
            UploadPrepareRequest.model_validate({**request.model_dump(), field: "x"})


def test_error_contract_contains_all_frozen_codes() -> None:
    required = {
        "STAFF_IDENTITY_REQUIRED",
        "RESOURCE_NOT_FOUND",
        "SESSION_NOT_ACTIVE",
        "SESSION_ROSTER_FROZEN",
        "SESSION_ALREADY_STARTED",
        "SESSION_NOT_STARTED",
        "SESSION_ALREADY_ENDED",
        "INVALID_TIME_RANGE",
        "INTERVAL_OVERLAP",
        "BACKFILL_WINDOW_EXPIRED",
        "REVISION_CONFLICT",
        "SERVICE_CONSENT_NOT_CURRENT",
        "REAL_VIDEO_GATE_NOT_APPROVED",
        "ANONYMIZATION_OR_CONSENT_REQUIRED",
        "UPLOAD_NOT_READY",
        "MEDIA_FORMAT_UNSUPPORTED",
        "MEDIA_TOO_LARGE",
        "MEDIA_INSPECTION_FAILED",
        "MEDIA_AUDIO_PRESENT",
        "CLIP_INTERVAL_MISMATCH",
        "DUPLICATE_UPLOAD",
        "SESSION_MEDIA_ALREADY_BOUND",
        "IDEMPOTENCY_CONFLICT",
        "REPOSITORY_UNAVAILABLE",
        "UPLOAD_COMMIT_STATE_UNKNOWN",
    }
    assert required <= {item.value for item in SessionCaptureErrorCode}
