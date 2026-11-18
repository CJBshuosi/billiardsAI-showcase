import hashlib
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.privacy.retention_repository import RetentionRepositoryError
from billiards_ai.cloud.session_capture.identity import StaffPrincipal
from billiards_ai.cloud.session_capture.storage import (
    BinarySource,
    ControlledStagingObjectStore,
    SessionCaptureLimits,
)
from billiards_ai.cloud.session_capture.upload import SessionUploadCoordinator
from billiards_ai.shared.models import MediaDeletionStatus
from billiards_ai.shared.models.session_capture import UploadAttemptStatus, UploadPrepareRequest


class OneShotSource(BinarySource):
    def __init__(self, payload: bytes):
        self.payload = payload
        self.opened = 0

    def open(self):
        self.opened += 1
        if self.opened != 1:
            raise RuntimeError("single use")
        return BytesIO(self.payload)


def test_binary_source_is_deployment_owned_and_single_read() -> None:
    source = OneShotSource(b"synthetic-media")
    assert source.open().read() == b"synthetic-media"
    with pytest.raises(RuntimeError):
        source.open()


def test_capture_limits_are_required_not_silent_defaults() -> None:
    with pytest.raises(ValidationError):
        SessionCaptureLimits()


def test_controlled_staging_streams_hashes_and_deletes(
    tmp_path, controlled_media_store_factory
) -> None:
    media_store = controlled_media_store_factory(tmp_path)
    store = ControlledStagingObjectStore(media_store)
    payload = b"synthetic-media-without-real-people"
    locator, digest, size = store.stage(
        attempt_id=uuid4(), source=OneShotSource(payload), max_bytes=len(payload)
    )
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
    assert media_store.exists(locator)
    store.delete(locator)
    assert not media_store.exists(locator)


def test_controlled_staging_cleans_partial_object_on_limit(
    tmp_path, controlled_media_store_factory
) -> None:
    media_store = controlled_media_store_factory(tmp_path)
    store = ControlledStagingObjectStore(media_store)
    with pytest.raises(RetentionRepositoryError):
        store.stage(attempt_id=uuid4(), source=OneShotSource(b"too-large"), max_bytes=1)
    assert list((media_store._resolved_root / "staged").iterdir()) == []


class SagaObjects:
    def __init__(self, calls):
        self.calls = calls

    def stage(self, **kwargs):
        self.calls.append("OBJECT_STAGED")
        return "staged/object.bin", "c" * 64, 10

    def delete(self, locator):
        self.calls.append("OBJECT_DELETED")


class SagaSessions:
    def __init__(self, calls, fail_bind=False):
        self.calls, self.fail_bind, self.binding = calls, fail_bind, None
        now = datetime(2026, 7, 29, tzinfo=UTC)
        self.session = SimpleNamespace(
            revision=2,
            interval_history=(SimpleNamespace(start_at=now, end_at=now + timedelta(seconds=10)),),
        )
        self.subjects = (uuid4(), uuid4())
        self.attempt = None

    def prepare_upload_attempt(self, *, request, principal, **kwargs):
        self.calls.append("ATTEMPT_PERSISTED")
        self.attempt = SimpleNamespace(
            attempt_id=uuid4(),
            session_id=request.session_id,
            status=UploadAttemptStatus.STAGED,
            object_staged=False,
            expected_session_revision=2,
            interval_revision=1,
            content_sha256=None,
            asset_id=None,
            size_bytes=None,
            storage_locator=None,
            gate_revision_id=None,
            subject_user_ids=(),
        )
        return self.attempt, self.session, object(), self.subjects

    def complete_upload_attempt(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.attempt, key, value)
        self.attempt.object_staged = True
        self.attempt.subject_user_ids = tuple(sorted(kwargs["subject_user_ids"]))
        return self.attempt

    def get_upload_attempt(self, **kwargs):
        return self.attempt

    def prepare_upload_authority(self, **kwargs):
        return self.session, object(), self.subjects

    def bind_staged_upload(self, **kwargs):
        if self.fail_bind:
            raise RuntimeError("bind failed")
        self.calls.append("SESSION_BOUND")
        self.binding = {"asset_id": str(kwargs["asset_id"])}

    def get_media_binding(self, **kwargs):
        return self.binding

    def mark_media_bound(self, **kwargs):
        self.calls.append("SESSION_MEDIA_BOUND")
        return "MEDIA_BOUND"

    def pending_media_bindings(self, **kwargs):
        return ()

    def pending_upload_attempts(self, **kwargs):
        return ()

    def pending_upload_cleanups(self, **kwargs):
        return ()

    def record_upload_failed(self, **kwargs):
        return None


class SagaRetention:
    def __init__(self, calls):
        self.calls = calls
        self.repository = SimpleNamespace(
            get_asset=lambda asset_id: SimpleNamespace(deletion_status=MediaDeletionStatus.ACTIVE)
        )

    def register_pending_asset(self, *args, **kwargs):
        self.calls.append("RETENTION_PENDING")
        return "PENDING_BINDING"

    def activate_pending_asset(self, **kwargs):
        self.calls.append("RETENTION_ACTIVE")

    def abandon_pending_asset(self, **kwargs):
        self.calls.append("RETENTION_ABANDONED")

    def quarantine_unbound_asset(self, **kwargs):
        self.calls.append("RETENTION_QUARANTINED")


class SagaInspector:
    def inspect(self, *args, **kwargs):
        return SimpleNamespace(
            container="mp4", video_codec="h264", duration_seconds=10.0, audio_present=False
        )


def _coordinator(calls, fail_bind=False):
    gate_revision_id = uuid4()
    return SessionUploadCoordinator(
        session_authority=SagaSessions(calls, fail_bind),
        retention_service=SagaRetention(calls),
        object_store=SagaObjects(calls),
        asset_context_provider=SimpleNamespace(
            resolve=lambda **kwargs: SimpleNamespace(captured_under_revision_id=gate_revision_id)
        ),
        policy_provider=object(),
        media_inspector=SagaInspector(),
        limits=SessionCaptureLimits(
            allowed_containers=("mp4",),
            allowed_video_codecs=("h264",),
            max_upload_bytes=100,
            max_duration_seconds=20.0,
            probe_timeout_seconds=1.0,
            interval_tolerance_seconds=0.1,
        ),
        identity_provider=SimpleNamespace(
            current_staff=lambda: StaffPrincipal(uuid4(), "STAFF", ("venue",))
        ),
        clock=lambda: datetime(2026, 7, 29, tzinfo=UTC),
    )


def test_upload_saga_uses_fixed_inaccessible_then_bound_order() -> None:
    calls = []
    _coordinator(calls).finalize(
        request=UploadPrepareRequest(
            schema_version="1.0", session_id=uuid4(), idempotency_key="A" * 43
        ),
        principal=StaffPrincipal(uuid4(), "STAFF", ("venue",)),
        source=OneShotSource(b"ignored"),
    )
    assert calls == [
        "ATTEMPT_PERSISTED",
        "OBJECT_STAGED",
        "RETENTION_PENDING",
        "SESSION_BOUND",
        "RETENTION_ACTIVE",
        "SESSION_MEDIA_BOUND",
    ]


def test_failure_before_session_binding_compensates_pending_and_object() -> None:
    calls = []
    with pytest.raises(RuntimeError, match="REPOSITORY_UNAVAILABLE"):
        _coordinator(calls, True).finalize(
            request=UploadPrepareRequest(
                schema_version="1.0", session_id=uuid4(), idempotency_key="A" * 43
            ),
            principal=StaffPrincipal(uuid4(), "STAFF", ("venue",)),
            source=OneShotSource(b"ignored"),
        )
    assert calls[-2:] == ["RETENTION_ABANDONED", "OBJECT_DELETED"]


def test_inspection_failure_cleans_staged_object() -> None:
    calls = []
    coordinator = _coordinator(calls)
    coordinator.inspector = SimpleNamespace(
        inspect=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe failed"))
    )
    with pytest.raises(RuntimeError, match="MEDIA_INSPECTION_FAILED"):
        coordinator.finalize(
            request=UploadPrepareRequest(
                schema_version="1.0", session_id=uuid4(), idempotency_key="A" * 43
            ),
            principal=StaffPrincipal(uuid4(), "STAFF", ("venue",)),
            source=OneShotSource(b"ignored"),
        )
    assert calls == ["ATTEMPT_PERSISTED", "OBJECT_STAGED", "OBJECT_DELETED"]
