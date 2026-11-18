from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models import Recording
from billiards_ai.shared.models.retention import (
    DeletionAttempt,
    DeletionRequest,
    DeletionRequestStatus,
    DeletionScope,
    MediaAssetKind,
    MediaDeletionStatus,
    PublicDeletionStatus,
    RetainedMediaAsset,
)

NOW = datetime(2026, 7, 27, 18, tzinfo=UTC)
SHA = "a" * 64


def test_protocol_enums_are_exact() -> None:
    assert {item.value for item in DeletionRequestStatus} == {
        "RECEIVED",
        "IN_PROGRESS",
        "COMPLETED",
        "FAILED",
    }
    assert {item.value for item in MediaAssetKind} == {
        "VIDEO",
        "FRAME",
        "KEYFRAME",
        "SCREENSHOT",
        "TRACEABLE_DERIVATIVE",
    }


def test_scope_is_canonical_and_all_media_is_exclusive() -> None:
    first, second = uuid4(), uuid4()
    scope = DeletionScope(session_ids=(second, first, second))
    assert scope.session_ids == tuple(sorted({first, second}, key=str))
    with pytest.raises(ValidationError):
        DeletionScope(all_user_media=True, session_ids=(first,))


def test_asset_rejects_naive_time_and_unknown_fields() -> None:
    values = dict(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="session/video.mp4",
        content_sha256=SHA,
        captured_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    asset = RetainedMediaAsset(**values)
    assert asset.deletion_status is MediaDeletionStatus.ACTIVE
    with pytest.raises(ValidationError):
        RetainedMediaAsset(**{**values, "captured_at": NOW.replace(tzinfo=None)})
    with pytest.raises(ValidationError):
        RetainedMediaAsset(**values, hidden_path="nope")
    with pytest.raises(ValidationError):
        RetainedMediaAsset(**{**values, "storage_locator": "../escape.mp4"})
    with pytest.raises(ValidationError, match="future"):
        RetainedMediaAsset(
            **{
                **values,
                "reviewed_at": NOW + timedelta(minutes=6),
                "updated_at": NOW + timedelta(minutes=6),
            }
        )


def test_public_status_cannot_contain_internal_storage_locator() -> None:
    request = DeletionRequest(
        request_id=uuid4(),
        user_id="user-1",
        scope=DeletionScope(all_user_media=True),
        idempotency_key="delete-all-once",
        status=DeletionRequestStatus.RECEIVED,
        created_at=NOW,
        updated_at=NOW,
    )
    public = PublicDeletionStatus.from_request(request)
    assert set(public.model_dump()) == {
        "request_id",
        "status",
        "updated_at",
        "scope_summary",
        "next_action",
    }
    assert "locator" not in public.model_dump_json()


def test_uuid4_and_control_character_contracts_fail_closed() -> None:
    with pytest.raises(ValidationError):
        DeletionRequest(
            request_id=UUID(int=0),
            user_id="user-1",
            scope=DeletionScope(all_user_media=True),
            idempotency_key="key",
            status=DeletionRequestStatus.RECEIVED,
            created_at=NOW,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="control"):
        DeletionRequest(
            request_id=uuid4(),
            user_id="user\x00split",
            scope=DeletionScope(all_user_media=True),
            idempotency_key="key",
            status=DeletionRequestStatus.RECEIVED,
            created_at=NOW,
            updated_at=NOW,
        )


def test_expiry_is_exact_and_attempt_counts_are_real_constraints() -> None:
    values = dict(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.VIDEO,
        storage_locator="session/video.mp4",
        content_sha256=SHA,
        captured_at=NOW,
        reviewed_at=NOW,
        delivered_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    with pytest.raises(ValidationError, match="exactly"):
        RetainedMediaAsset(**values, retention_expires_at=NOW + timedelta(days=29))
    with pytest.raises(ValidationError, match="verify every"):
        DeletionAttempt(
            attempt_number=1,
            executor_id="Boyu",
            started_at=NOW,
            finished_at=NOW,
            planned_asset_count=2,
            verified_deleted_count=1,
            result="COMPLETED",
            evidence_summary="only one verified",
        )
    with pytest.raises(ValidationError):
        DeletionRequest(
            request_id=uuid4(),
            user_id="user-1",
            scope=DeletionScope(all_user_media=True),
            idempotency_key="failed-without-code",
            status=DeletionRequestStatus.FAILED,
            created_at=NOW,
            updated_at=NOW,
        )


def test_recording_retention_fields_are_non_authoritative_compatibility_projection() -> None:
    recording = Recording(
        id=uuid4(),
        device_id="camera-1",
        start_ts=NOW,
        end_ts=NOW,
        oss_key="legacy/object.mp4",
        duration_ms=0,
        size_bytes=0,
    )
    assert recording.model_dump()["delete_after"] is None
    assert recording.retention_asset_id is None
