"""Trusted reviewer and session-subject authority boundaries."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import UUID4, Field, StrictInt, TypeAdapter, ValidationError, field_validator

from billiards_ai.shared.models.access import ReviewerIdentity
from billiards_ai.shared.models.events import CoreContractModel, require_utc
from billiards_ai.shared.models.session_capture import SessionMediaBindingRecord


class ReviewerIdentityProvider(Protocol):
    def current_identity(self) -> ReviewerIdentity: ...


class TableBindingMediaRepository(Protocol):
    def get_media_binding(self, *, session_id: UUID) -> object: ...


class SessionSubjectAuthorityProvider(Protocol):
    def resolve_subject(self, *, session_id: UUID) -> ResolvedSessionSubject: ...


class SessionSubjectAuthorityError(RuntimeError):
    """The accepted media binding cannot identify exactly one UUID4 subject."""


class ResolvedSessionSubject(CoreContractModel):
    session_id: UUID4
    user_id: UUID4
    attempt_id: UUID4
    interval_revision: StrictInt = Field(ge=1)
    accepted_at: datetime

    _accepted_at_utc = field_validator("accepted_at")(require_utc)


_UUID4 = TypeAdapter(UUID4)


class TableBindingSessionSubjectAuthority:
    """Resolve one training subject from the accepted table-binding media fact."""

    def __init__(self, repository: TableBindingMediaRepository) -> None:
        self._repository = repository

    def resolve_subject(self, *, session_id: UUID) -> ResolvedSessionSubject:
        try:
            raw = self._repository.get_media_binding(session_id=session_id)
            if isinstance(raw, SessionMediaBindingRecord):
                binding = SessionMediaBindingRecord.model_validate(
                    raw.model_dump(mode="python"), strict=True
                )
            else:
                binding = SessionMediaBindingRecord.model_validate_json(
                    json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    strict=True,
                )
            if binding.session_id != session_id or len(binding.subject_user_ids) != 1:
                raise ValueError("media binding does not identify one requested-session subject")
            user_id = _UUID4.validate_python(UUID(binding.subject_user_ids[0]), strict=True)
            return ResolvedSessionSubject(
                session_id=binding.session_id,
                user_id=user_id,
                attempt_id=binding.attempt_id,
                interval_revision=binding.interval_revision,
                accepted_at=binding.accepted_at,
            )
        except (
            AttributeError,
            json.JSONDecodeError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise SessionSubjectAuthorityError("session subject authority is unavailable") from exc
