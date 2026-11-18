"""Trusted orchestration for server-timed staff capture operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol

from billiards_ai.shared.models.session_capture import (
    CaptureOperationRequest,
    CorrectIntervalRequest,
    SessionCaptureErrorCode,
    SessionCaptureNextAction,
    SessionCaptureStatus,
    StaffSessionView,
)

from .errors import SessionCaptureDomainError, SessionCaptureServiceError
from .identity import AuthenticatedStaffProvider, StaffPrincipal


def _capture_code(error: Exception) -> SessionCaptureErrorCode | None:
    raw = getattr(getattr(error, "code", None), "value", None)
    try:
        return SessionCaptureErrorCode(raw)
    except (TypeError, ValueError):
        aliases = {
            "CONSENT_ALREADY_WITHDRAWN": SessionCaptureErrorCode.SERVICE_CONSENT_NOT_CURRENT,
        }
        return aliases.get(raw)


class SessionCaptureRepository(Protocol):
    def start_capture(self, **kwargs): ...
    def correct_interval(self, **kwargs): ...


class SessionCaptureService:
    def __init__(
        self,
        *,
        repository: SessionCaptureRepository,
        identity_provider: AuthenticatedStaffProvider,
        policy_provider: object,
        clock: Callable[[], datetime] | None = None,
    ):
        self._repository = repository
        self._identity_provider = identity_provider
        self._policy_provider = policy_provider
        self._clock = clock or (lambda: datetime.now(UTC))

    def _principal(self) -> StaffPrincipal:
        try:
            principal = self._identity_provider.current_staff()
            if principal is None:
                raise ValueError
            return StaffPrincipal(
                staff_id=principal.staff_id,
                role=principal.role,
                venue_ids=tuple(principal.venue_ids),
            )
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.STAFF_IDENTITY_REQUIRED
            ) from None

    def _now(self) -> datetime:
        try:
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError
            return now.astimezone(UTC)
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            ) from None

    @staticmethod
    def _parse(model, value):
        try:
            payload = value.model_dump(mode="python") if hasattr(value, "model_dump") else value
            return model.model_validate(payload)
        except Exception:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.INVALID_TIME_RANGE) from None

    def _call(self, operation: str, **kwargs):
        try:
            return getattr(self._repository, operation)(**kwargs)
        except SessionCaptureDomainError as error:
            raise SessionCaptureServiceError(error.code) from None
        except SessionCaptureServiceError:
            raise
        except Exception as error:
            code = _capture_code(error)
            if code is not None:
                raise SessionCaptureServiceError(code) from None
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            ) from None

    def start(self, value: CaptureOperationRequest | Mapping[str, object]):
        request = self._parse(CaptureOperationRequest, value)
        principal = self._principal()
        return self._call(
            "start_capture",
            request=request,
            principal=principal,
            now=self._now(),
            policy_provider=self._policy_provider,
        )

    def end(self, value: CaptureOperationRequest | Mapping[str, object]):
        request = self._parse(CaptureOperationRequest, value)
        principal = self._principal()
        return self._call(
            "end_capture",
            request=request,
            principal=principal,
            now=self._now(),
            policy_provider=self._policy_provider,
        )

    def cancel(self, value: CaptureOperationRequest | Mapping[str, object]):
        request = self._parse(CaptureOperationRequest, value)
        principal = self._principal()
        return self._call(
            "cancel_capture",
            request=request,
            principal=principal,
            now=self._now(),
            policy_provider=self._policy_provider,
        )

    @staticmethod
    def _view(snapshot) -> StaffSessionView:
        session, table, count = snapshot
        interval = None
        if session.interval_history and session.interval_history[-1].end_at is not None:
            from billiards_ai.shared.models.session_capture import CaptureIntervalRevision

            source = session.interval_history[-1]
            interval = CaptureIntervalRevision(
                interval_id=session.session_id,
                revision=source.revision,
                start_at=source.start_at,
                end_at=source.end_at,
                provenance=source.provenance,
                recorded_by=source.operator_id,
                recorded_at=source.recorded_at,
                reason=source.reason,
                supersedes_revision=None if source.revision == 1 else source.revision - 1,
            )
        return StaffSessionView(
            session_id=session.session_id,
            venue_display_label=table.venue_display_label,
            table_display_label=table.table_display_label,
            participant_count=count,
            session_status=session.status,
            capture_status=(
                SessionCaptureStatus.RECORDING
                if session.interval_history and session.interval_history[-1].end_at is None
                else None
            ),
            interval=interval,
            upload_status=None,
            next_action=(
                SessionCaptureNextAction.MARK_END
                if session.interval_history and session.interval_history[-1].end_at is None
                else SessionCaptureNextAction.PREPARE_UPLOAD
                if session.status.value == "ENDED_AWAITING_UPLOAD"
                else SessionCaptureNextAction.MARK_START
            ),
        )

    def list_authorized_active_sessions(self, *, limit: int = 100) -> tuple[StaffSessionView, ...]:
        principal = self._principal()
        try:
            return tuple(
                self._view(item)
                for item in self._repository.list_staff_sessions(principal=principal, limit=limit)
            )
        except Exception:
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            ) from None

    def read_staff_session(self, session_id) -> StaffSessionView:
        principal = self._principal()
        try:
            snapshot = self._repository.read_staff_session(
                session_id=session_id, principal=principal
            )
        except Exception as error:
            code = _capture_code(error)
            if code is not None:
                raise SessionCaptureServiceError(code) from None
            raise SessionCaptureServiceError(
                SessionCaptureErrorCode.REPOSITORY_UNAVAILABLE
            ) from None
        if snapshot is None:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.RESOURCE_NOT_FOUND)
        return self._view(snapshot)

    def correct_interval(self, value: CorrectIntervalRequest | Mapping[str, object]):
        request = self._parse(CorrectIntervalRequest, value)
        principal = self._principal()
        now = self._now()
        if request.start_at < now - timedelta(hours=48):
            raise SessionCaptureServiceError(SessionCaptureErrorCode.BACKFILL_WINDOW_EXPIRED)
        if request.end_at > now or request.end_at <= request.start_at:
            raise SessionCaptureServiceError(SessionCaptureErrorCode.INVALID_TIME_RANGE)
        return self._call(
            "correct_interval",
            request=request,
            principal=principal,
            now=now,
            policy_provider=self._policy_provider,
        )
