import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.privacy.retention_repository import MediaUnavailableError
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.cloud.session_capture.errors import SessionCaptureServiceError
from billiards_ai.cloud.session_capture.identity import StaffPrincipal
from billiards_ai.cloud.session_capture.policy import (
    JsonAssetContextEvidenceRepository,
    TrustedAssetContextEvidenceService,
)
from billiards_ai.cloud.session_capture.service import SessionCaptureService
from billiards_ai.cloud.session_capture.storage import (
    ControlledStagingObjectStore,
    FFmpegMediaInspector,
)
from billiards_ai.cloud.session_capture.upload import SessionUploadCoordinator
from billiards_ai.cloud.table_binding.errors import (
    TableBindingDomainError,
    TableBindingRepositoryError,
)
from billiards_ai.cloud.table_binding.repository import JsonTableBindingRepository
from billiards_ai.shared.models import (
    AssetContext,
    DeletionScope,
    MediaDeletionStatus,
    TableBindingErrorCode,
)
from billiards_ai.shared.models.session_capture import (
    CaptureOperationRequest,
    SessionCaptureErrorCode,
    SessionCaptureLimits,
    UploadAttemptStatus,
    UploadFinalizeRequest,
    UploadPrepareRequest,
)

NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)
KEY = "A" * 43


def _attempt(session_id=None, *, object_staged=False):
    return SimpleNamespace(
        attempt_id=uuid4(),
        session_id=session_id or uuid4(),
        status=UploadAttemptStatus.STAGED,
        object_staged=object_staged,
        expected_session_revision=2,
        interval_revision=1,
        content_sha256="c" * 64 if object_staged else None,
        asset_id=uuid4() if object_staged else None,
        size_bytes=1 if object_staged else None,
        storage_locator="staged/object.bin" if object_staged else None,
        gate_revision_id=uuid4() if object_staged else None,
        subject_user_ids=("user",),
    )


def _coordinator(authority, objects, *, identity=True):
    return SessionUploadCoordinator(
        session_authority=authority,
        retention_service=SimpleNamespace(),
        object_store=objects,
        asset_context_provider=SimpleNamespace(),
        policy_provider=SimpleNamespace(),
        media_inspector=SimpleNamespace(
            inspect=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after stage"))
        ),
        limits=SessionCaptureLimits(
            allowed_containers=("mp4",),
            allowed_video_codecs=("h264",),
            max_upload_bytes=8,
            max_duration_seconds=10.0,
            probe_timeout_seconds=1.0,
            interval_tolerance_seconds=0.1,
        ),
        identity_provider=(
            SimpleNamespace(
                current_staff=lambda: StaffPrincipal(uuid4(), "CAPTURE_OPERATOR", ("v",))
            )
            if identity
            else None
        ),
        clock=lambda: NOW,
    )


def test_access_requires_active_retention_and_media_bound_session() -> None:
    authority = SimpleNamespace(is_media_bound=lambda **kwargs: False)
    service = MediaRetentionService(SimpleNamespace(), session_media_authority=authority)
    with pytest.raises(MediaUnavailableError, match="not bound"):
        service._require_session_media_bound(SimpleNamespace(session_id=uuid4(), asset_id=uuid4()))


def test_prepare_persists_attempt_before_object_stage() -> None:
    calls = []
    session_id = uuid4()
    attempt = _attempt(session_id)
    session = SimpleNamespace(
        interval_history=(SimpleNamespace(start_at=NOW, end_at=NOW),), revision=2
    )
    authority = SimpleNamespace(
        prepare_upload_attempt=lambda **kwargs: (
            calls.append("ATTEMPT") or (attempt, session, object(), ("user",))
        ),
        record_upload_failed=lambda **kwargs: None,
    )
    objects = SimpleNamespace(
        stage=lambda **kwargs: calls.append("OBJECT") or ("staged/object.bin", "c" * 64, 1),
        delete=lambda locator: None,
    )
    with pytest.raises(SessionCaptureServiceError):
        _coordinator(authority, objects).prepare(
            request=UploadPrepareRequest(
                schema_version="1.0", session_id=session_id, idempotency_key=KEY
            ),
            source=SimpleNamespace(open=lambda: BytesIO(b"synthetic")),
        )
    assert calls[:2] == ["ATTEMPT", "OBJECT"]


def test_prepare_and_finalize_are_separate_revisioned_operations() -> None:
    session_id = uuid4()
    attempt = _attempt(session_id, object_staged=True)
    authority = SimpleNamespace(get_upload_attempt=lambda **kwargs: attempt)
    coordinator = _coordinator(authority, SimpleNamespace())
    with pytest.raises(SessionCaptureServiceError) as raised:
        coordinator.finalize(
            request=UploadFinalizeRequest(
                schema_version="1.0",
                session_id=session_id,
                idempotency_key=KEY,
                attempt_id=attempt.attempt_id,
                expected_session_revision=3,
            )
        )
    assert raised.value.code is SessionCaptureErrorCode.REVISION_CONFLICT


def test_correction_invalidates_staged_attempt_before_new_interval() -> None:
    session_id = uuid4()
    payload = {
        "media_bindings": {},
        "upload_attempts": {str(uuid4()): {"session_id": str(session_id), "status": "STAGED"}},
    }
    assert JsonTableBindingRepository._capture_has_upload_authority(payload, session_id)


def test_reconciler_revalidates_gate_consent_interval_evidence_and_retention() -> None:
    calls = []
    authority = SimpleNamespace(
        pending_upload_attempts=lambda **kwargs: (_attempt(object_staged=True),),
        pending_upload_cleanups=lambda **kwargs: (),
    )
    reconciler = _coordinator(authority, SimpleNamespace())
    reconciler.finalize = lambda *, request: calls.append(request)
    reconciler.reconcile_pending_uploads()
    assert calls and calls[0].expected_session_revision == 2


def test_pending_retention_is_excluded_from_user_deletion_scope() -> None:
    pending = SimpleNamespace(
        asset_id=uuid4(),
        session_id=uuid4(),
        owner_scope=SimpleNamespace(subject_user_ids=("user",)),
        source_asset_ids=(),
        deletion_status=MediaDeletionStatus.PENDING_BINDING,
    )
    repository = SimpleNamespace(list_assets=lambda: (pending,))
    from billiards_ai.cloud.privacy.retention_service import MediaRetentionService

    assert (
        MediaRetentionService(repository)._resolve_scope("user", DeletionScope(all_user_media=True))
        == ()
    )


def test_service_maps_repository_not_found_to_frozen_capture_error() -> None:
    repository = SimpleNamespace(
        start_capture=lambda **kwargs: (_ for _ in ()).throw(
            TableBindingDomainError(TableBindingErrorCode.RESOURCE_NOT_FOUND)
        )
    )
    identity = SimpleNamespace(
        current_staff=lambda: StaffPrincipal(uuid4(), "CAPTURE_OPERATOR", ("venue",))
    )
    service = SessionCaptureService(
        repository=repository,
        identity_provider=identity,
        policy_provider=object(),
        clock=lambda: NOW,
    )
    with pytest.raises(SessionCaptureServiceError) as raised:
        service.start(
            CaptureOperationRequest(schema_version="1.0", session_id=uuid4(), idempotency_key=KEY)
        )
    assert raised.value.code is SessionCaptureErrorCode.RESOURCE_NOT_FOUND


def test_v3_rejects_malformed_capture_authority_record() -> None:
    payload = JsonTableBindingRepository._empty()
    payload["capture_audit_events"]["1"] = {"sequence": True}
    with pytest.raises(TableBindingRepositoryError):
        JsonTableBindingRepository._validated_store(payload)


def test_upload_and_evidence_use_trusted_identity_not_caller_principal(tmp_path) -> None:
    coordinator = _coordinator(SimpleNamespace(), SimpleNamespace(), identity=False)
    with pytest.raises(SessionCaptureServiceError) as raised:
        coordinator.prepare(
            request=UploadPrepareRequest(
                schema_version="1.0", session_id=uuid4(), idempotency_key=KEY
            ),
            source=SimpleNamespace(open=lambda: BytesIO(b"x")),
        )
    assert raised.value.code is SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED
    evidence = TrustedAssetContextEvidenceService(
        JsonAssetContextEvidenceRepository(tmp_path / "evidence" / "store.json"),
        SimpleNamespace(current_staff=lambda: None),
    )
    with pytest.raises(SessionCaptureServiceError) as evidence_error:
        evidence.record(attempt_id=uuid4(), context=object())
    assert evidence_error.value.code is SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED


def test_staging_retries_short_os_write_until_all_bytes_persisted(
    tmp_path, controlled_media_store_factory, monkeypatch
) -> None:
    store = ControlledStagingObjectStore(controlled_media_store_factory(tmp_path))
    real_write = os.write

    def short_write(fd, value):
        return real_write(fd, value[: max(1, len(value) // 2)])

    monkeypatch.setattr(os, "write", short_write)
    locator, _, size = store.stage(
        attempt_id=uuid4(), source=SimpleNamespace(open=lambda: BytesIO(b"abcdefgh")), max_bytes=8
    )
    assert store.internal_path(locator).read_bytes() == b"abcdefgh"
    assert size == 8


def test_evidence_store_rejects_unsafe_parent_before_writing(tmp_path: Path) -> None:
    parent = tmp_path / "unsafe"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    repository = JsonAssetContextEvidenceRepository(parent / "evidence.json")
    context = AssetContext(
        asset_id=str(uuid4()),
        asset_sha256="a" * 64,
        gate_scope_fingerprint="b" * 64,
        captured_at=NOW,
        captured_under_revision_id=uuid4(),
        bystanders_present=False,
    )
    with pytest.raises(RuntimeError, match="unsafe"):
        repository.save(attempt_id=uuid4(), context=context, operator_id=uuid4())


def test_capture_idempotency_result_is_strict_and_immutable() -> None:
    import billiards_ai.shared.models.session_capture as contracts

    record = contracts.CaptureIdempotencyRecord(
        record_id="a" * 64,
        operation="START",
        staff_id=uuid4(),
        session_id=uuid4(),
        key_digest="b" * 64,
        payload_digest="c" * 64,
        result_session_revision=2,
        result_interval_revision=1,
        result_status="OPEN",
        created_at=NOW,
    )
    with pytest.raises(ValidationError):
        record.result_session_revision = 3


def test_capture_audit_is_strict_versioned_and_complete() -> None:
    import billiards_ai.shared.models.session_capture as contracts

    valid = contracts.CaptureAuditEvent(
        schema_version="1.0",
        event_id=uuid4(),
        sequence=1,
        event_type="START_MARKED",
        operator_id=uuid4(),
        session_id=uuid4(),
        table_registration_id=uuid4(),
        occurred_at=NOW,
        result="SUCCESS",
        idempotency_digest="a" * 64,
    )
    assert valid.sequence == 1
    with pytest.raises(ValidationError):
        contracts.CaptureAuditEvent(
            schema_version="1.0",
            event_id=uuid4(),
            sequence=True,
            event_type="START_MARKED",
            operator_id=uuid4(),
            session_id=uuid4(),
            table_registration_id=uuid4(),
            occurred_at=NOW,
            result="SUCCESS",
            idempotency_digest="a" * 64,
        )


def test_media_inspector_normalizes_mp4_alias_and_includes_last_frame_duration(monkeypatch) -> None:
    probe = SimpleNamespace(
        container="mov,mp4,m4a,3gp,3g2,mj2",
        video_codec="h264",
        frame_pts=(0, 1, 2),
        frame_durations=(1, 1, 1),
        time_base=SimpleNamespace(numerator=1, denominator=10),
        audio_present=False,
    )
    monkeypatch.setattr(
        "billiards_ai.cloud.session_capture.storage.SubprocessFFmpegRunner.probe",
        lambda self, path: probe,
    )
    inspector = FFmpegMediaInspector(SimpleNamespace(internal_path=lambda locator: Path("x")))
    facts = inspector.inspect("safe", timeout_seconds=1)
    assert facts.container == "mp4"
    assert facts.duration_seconds == pytest.approx(0.3)


def test_direct_staff_read_does_not_route_through_paginated_list(monkeypatch) -> None:
    repository = JsonTableBindingRepository("unused.json")
    monkeypatch.setattr(
        repository,
        "list_staff_sessions",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pagination used")),
    )
    monkeypatch.setattr(
        repository, "_read_staff_capture_direct", lambda **kwargs: "direct", raising=False
    )
    assert (
        repository.read_staff_session(
            session_id=uuid4(), principal=StaffPrincipal(uuid4(), "CAPTURE_OPERATOR", ("venue",))
        )
        == "direct"
    )
