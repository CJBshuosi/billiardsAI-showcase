"""Crash-recoverable, authority-first session upload saga."""

from __future__ import annotations

import hashlib
from contextlib import closing
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    MaterialUsePurpose,
    MediaAssetKind,
    MediaDeletionStatus,
    RetainedMediaAssetCreate,
)
from billiards_ai.shared.models.retention import MediaOwnerScope
from billiards_ai.shared.models.session_capture import (
    SessionCaptureErrorCode,
    SessionCaptureLimits,
    UploadAttemptStatus,
    UploadFinalizeRequest,
    UploadPrepareRequest,
)

from .errors import SessionCaptureServiceError
from .identity import AuthenticatedStaffProvider, StaffPrincipal
from .policy import AssetContextProvider
from .storage import BinarySource, ObjectStore


class MediaFacts(Protocol):
    container: str
    video_codec: str
    duration_seconds: float
    audio_present: bool


class MediaInspector(Protocol):
    def inspect(self, locator: str, *, timeout_seconds: float) -> MediaFacts: ...


class SessionUploadCoordinator:
    def __init__(
        self,
        *,
        session_authority,
        retention_service: MediaRetentionService,
        object_store: ObjectStore,
        asset_context_provider: AssetContextProvider,
        policy_provider,
        media_inspector: MediaInspector,
        limits: SessionCaptureLimits,
        identity_provider: AuthenticatedStaffProvider | None = None,
        clock=lambda: datetime.now(UTC),
    ):
        self.sessions = session_authority
        self.retention = retention_service
        self.objects = object_store
        self.contexts = asset_context_provider
        self.policy = policy_provider
        self.inspector = media_inspector
        self.limits = SessionCaptureLimits.model_validate(limits.model_dump())
        self.identity_provider = identity_provider
        self.clock = clock

    def _principal(self) -> StaffPrincipal:
        try:
            principal = self.identity_provider.current_staff() if self.identity_provider else None
            if principal is None:
                raise ValueError
            return StaffPrincipal(principal.staff_id, principal.role, tuple(principal.venue_ids))
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED
            ) from None

    def _session_call(self, operation: str, **kwargs):
        try:
            return getattr(self.sessions, operation)(**kwargs)
        except SessionCaptureServiceError:
            raise
        except Exception as error:
            raw = getattr(getattr(error, "code", None), "value", None)
            try:
                code = SessionCaptureErrorCode(raw)
            except (TypeError, ValueError):
                code = SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            raise SessionCaptureServiceError(code) from None

    @staticmethod
    def _asset_id(session_id: UUID, content_sha256: str) -> UUID:
        raw = bytearray(
            hashlib.sha256(f"billiardsAI:source:{session_id}:{content_sha256}".encode()).digest()[
                :16
            ]
        )
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        return UUID(bytes=bytes(raw))

    def _source_digest(self, source: BinarySource) -> str:
        digest = hashlib.sha256()
        size = 0
        with closing(source.open()) as stream:
            while chunk := stream.read(1024 * 1024):
                if not isinstance(chunk, bytes):
                    raise SessionCaptureServiceError(
                        SessionCaptureErrorCode.MEDIA_INSPECTION_FAILED
                    )
                size += len(chunk)
                if size > self.limits.max_upload_bytes:
                    raise SessionCaptureServiceError(SessionCaptureErrorCode.MEDIA_TOO_LARGE)
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_facts(self, facts: MediaFacts, expected_duration: float) -> None:
        if facts.container not in self.limits.allowed_containers:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.MEDIA_FORMAT_UNSUPPORTED)
        if facts.video_codec not in self.limits.allowed_video_codecs:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.MEDIA_FORMAT_UNSUPPORTED)
        if facts.audio_present:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.MEDIA_AUDIO_PRESENT)
        if facts.duration_seconds > self.limits.max_duration_seconds:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.MEDIA_TOO_LARGE)
        if abs(facts.duration_seconds - expected_duration) > self.limits.interval_tolerance_seconds:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.CLIP_INTERVAL_MISMATCH)

    def _revalidate_attempt(self, attempt, principal: StaffPrincipal):
        session, _, subjects = self._session_call(
            "prepare_upload_authority",
            session_id=attempt.session_id,
            principal=principal,
            policy_provider=self.policy,
        )
        if len(session.interval_history) != attempt.interval_revision:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.REVISION_CONFLICT)
        if tuple(sorted(str(item) for item in subjects)) != attempt.subject_user_ids:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.SERVICE_CONSENT_NOT_CURRENT)
        try:
            context = self.contexts.resolve(
                attempt_id=attempt.attempt_id,
                asset_id=attempt.asset_id,
                content_sha256=attempt.content_sha256,
            )
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.ANONYMIZATION_OR_CONSENT_REQUIRED
            ) from None
        if context.captured_under_revision_id != attempt.gate_revision_id:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.ANONYMIZATION_OR_CONSENT_REQUIRED
            )
        return session, context

    def prepare(self, *, request: UploadPrepareRequest, source: BinarySource):
        principal = self._principal()
        now = self.clock().astimezone(UTC)
        attempt, session, _, subjects = self._session_call(
            "prepare_upload_attempt",
            request=request,
            principal=principal,
            policy_provider=self.policy,
            now=now,
        )
        if attempt is None:
            binding = self._session_call("get_media_binding", session_id=request.session_id)
            if binding is None:
                raise SessionCaptureServiceError(
                    SessionCaptureErrorCode.UPLOAD_COMMIT_STATE_UNKNOWN
                )
            if self._source_digest(source) == binding["content_sha256"]:
                return binding
            raise SessionCaptureServiceError(SessionCaptureErrorCode.SESSION_MEDIA_ALREADY_BOUND)
        if attempt.object_staged:
            return attempt
        locator = f"staged/{attempt.attempt_id}.bin"
        try:
            locator, digest, size = self.objects.stage(
                attempt_id=attempt.attempt_id,
                source=source,
                max_bytes=self.limits.max_upload_bytes,
            )
        except Exception as error:
            self._fail_attempt(attempt, principal, locator)
            code = (
                SessionCaptureErrorCode.MEDIA_TOO_LARGE
                if "upload limit" in str(error)
                else SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            )
            raise SessionCaptureServiceError(code) from None
        try:
            facts = self.inspector.inspect(
                locator, timeout_seconds=self.limits.probe_timeout_seconds
            )
        except Exception:
            self._fail_attempt(attempt, principal, locator)
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.MEDIA_INSPECTION_FAILED
            ) from None
        interval = session.interval_history[-1]
        try:
            self._validate_facts(facts, (interval.end_at - interval.start_at).total_seconds())
        except SessionCaptureServiceError:
            self._fail_attempt(attempt, principal, locator)
            raise
        asset_id = self._asset_id(request.session_id, digest)
        try:
            context = self.contexts.resolve(
                attempt_id=attempt.attempt_id, asset_id=asset_id, content_sha256=digest
            )
        except Exception:
            self._fail_attempt(attempt, principal, locator)
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.ANONYMIZATION_OR_CONSENT_REQUIRED
            ) from None
        try:
            return self._session_call(
                "complete_upload_attempt",
                attempt_id=attempt.attempt_id,
                content_sha256=digest,
                asset_id=asset_id,
                size_bytes=size,
                storage_locator=locator,
                gate_revision_id=context.captured_under_revision_id,
                subject_user_ids=tuple(str(item) for item in subjects),
            )
        except Exception:
            self._fail_attempt(attempt, principal, locator)
            raise

    def _fail_attempt(self, attempt, principal: StaffPrincipal, locator: str) -> None:
        if attempt.asset_id is not None:
            try:
                self.retention.quarantine_unbound_asset(
                    asset_id=attempt.asset_id, session_id=attempt.session_id
                )
            except Exception:
                pass
        try:
            self.objects.delete(locator)
        except Exception:
            pass
        try:
            self.sessions.record_upload_failed(
                session_id=attempt.session_id,
                attempt_id=attempt.attempt_id,
                content_sha256=attempt.content_sha256 or "0" * 64,
                principal=principal,
                now=self.clock().astimezone(UTC),
            )
        except Exception:
            pass

    def finalize(
        self,
        *,
        request: UploadFinalizeRequest | UploadPrepareRequest,
        source: BinarySource | None = None,
        principal: StaffPrincipal | None = None,
    ):
        # Compatibility for the original one-call deployment adapter. The supplied
        # principal is deliberately ignored; authorization always comes from the provider.
        if isinstance(request, UploadPrepareRequest):
            if source is None:
                raise SessionCaptureServiceError(SessionCaptureErrorCode.UPLOAD_NOT_READY)
            prepared = self.prepare(request=request, source=source)
            if isinstance(prepared, dict):
                return prepared
            request = UploadFinalizeRequest(
                schema_version="1.0",
                session_id=request.session_id,
                idempotency_key=request.idempotency_key,
                attempt_id=prepared.attempt_id,
                expected_session_revision=prepared.expected_session_revision,
            )
        trusted = self._principal()
        attempt = self._session_call("get_upload_attempt", attempt_id=request.attempt_id)
        if attempt is None or attempt.session_id != request.session_id:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.RESOURCE_NOT_FOUND)
        if attempt.status is UploadAttemptStatus.COMPLETED:
            return self._session_call("get_media_binding", session_id=request.session_id)
        if attempt.status is not UploadAttemptStatus.STAGED:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.UPLOAD_NOT_READY)
        if request.expected_session_revision != attempt.expected_session_revision:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.REVISION_CONFLICT)
        session, context = self._revalidate_attempt(attempt, trusted)
        if session.revision != request.expected_session_revision:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.REVISION_CONFLICT)
        interval = session.interval_history[-1]
        create = RetainedMediaAssetCreate(
            asset_id=attempt.asset_id,
            session_id=request.session_id,
            owner_scope=MediaOwnerScope(subject_user_ids=attempt.subject_user_ids),
            kind=MediaAssetKind.VIDEO,
            storage_locator=attempt.storage_locator,
            content_sha256=attempt.content_sha256,
            captured_at=interval.start_at,
        )
        pending = self.retention.register_pending_asset(
            create,
            gate_revision_id=context.captured_under_revision_id,
            asset_context=context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )
        now = self.clock().astimezone(UTC)
        try:
            self._session_call(
                "bind_staged_upload",
                session_id=request.session_id,
                attempt_id=attempt.attempt_id,
                asset_id=attempt.asset_id,
                content_sha256=attempt.content_sha256,
                size_bytes=attempt.size_bytes,
                storage_locator=attempt.storage_locator,
                principal=trusted,
                policy_provider=self.policy,
                now=now,
            )
            self._revalidate_attempt(attempt, trusted)
            # Access remains closed until both authorities reach their final states.
            self.retention.activate_pending_asset(
                asset_id=attempt.asset_id, session_id=request.session_id
            )
            self._revalidate_attempt(attempt, trusted)
            active = self.retention.repository.get_asset(attempt.asset_id)
            if active.deletion_status is not MediaDeletionStatus.ACTIVE:
                raise SessionCaptureServiceError(
                    SessionCaptureErrorCode.UPLOAD_COMMIT_STATE_UNKNOWN
                )
            result = self._session_call(
                "mark_media_bound",
                session_id=request.session_id,
                asset_id=attempt.asset_id,
                now=now,
            )
        except Exception as error:
            try:
                accepted = self.sessions.get_media_binding(session_id=request.session_id)
            except Exception:
                raise SessionCaptureServiceError(
                    SessionCaptureErrorCode.UPLOAD_COMMIT_STATE_UNKNOWN
                ) from None
            if accepted is not None:
                raise SessionCaptureServiceError(
                    SessionCaptureErrorCode.UPLOAD_COMMIT_STATE_UNKNOWN
                ) from None
            self.retention.abandon_pending_asset(
                asset_id=attempt.asset_id, session_id=request.session_id
            )
            self.objects.delete(attempt.storage_locator)
            self.sessions.record_upload_failed(
                session_id=request.session_id,
                attempt_id=attempt.attempt_id,
                content_sha256=attempt.content_sha256,
                principal=trusted,
                now=now,
            )
            if isinstance(error, SessionCaptureServiceError):
                raise error
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            ) from None
        return result, pending

    def reconcile_pending_uploads(self, *, limit: int = 100) -> int:
        reconciled = 0
        for attempt in self.sessions.pending_upload_attempts(limit=limit):
            if not attempt.object_staged:
                self.objects.delete(f"staged/{attempt.attempt_id}.bin")
                self.sessions.record_upload_failed(
                    session_id=attempt.session_id,
                    attempt_id=attempt.attempt_id,
                    content_sha256="0" * 64,
                    principal=self._principal(),
                    now=self.clock().astimezone(UTC),
                )
                reconciled += 1
                continue
            try:
                self.finalize(
                    request=UploadFinalizeRequest(
                        schema_version="1.0",
                        session_id=attempt.session_id,
                        idempotency_key="A" * 43,
                        attempt_id=attempt.attempt_id,
                        expected_session_revision=attempt.expected_session_revision,
                    )
                )
            except Exception:
                self._fail_attempt(attempt, self._principal(), attempt.storage_locator)
            reconciled += 1
        for attempt in self.sessions.pending_upload_cleanups(limit=limit):
            if attempt.get("asset_id"):
                self.retention.quarantine_unbound_asset(
                    asset_id=UUID(attempt["asset_id"]), session_id=UUID(attempt["session_id"])
                )
            if attempt.get("storage_locator"):
                self.objects.delete(attempt["storage_locator"])
            self.sessions.mark_upload_cleanup_complete(attempt_id=UUID(attempt["attempt_id"]))
            reconciled += 1
        return reconciled
