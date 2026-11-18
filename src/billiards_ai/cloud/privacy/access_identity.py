"""Trusted local reviewer identity for the Phase 0 controlled access entrypoint."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from billiards_ai.shared.models import ReviewerIdentity

from .access_repository import AccessRepositoryError
from .errors import GateRepositoryError
from .identity import (
    _account_home,
    _is_inside_git_worktree,
    _read_private_json,
    _require_private_directory,
)

_REVIEWER_CREDENTIAL_RELATIVE_PATH = Path(".config/billiards-ai/reviewer.json")


class FileReviewerIdentityProvider:
    """Resolve reviewer/admin roles from one fixed private account credential."""

    @property
    def path(self) -> Path:
        return _account_home() / _REVIEWER_CREDENTIAL_RELATIVE_PATH

    def current_identity(self) -> ReviewerIdentity:
        try:
            if _is_inside_git_worktree(self.path):
                raise AccessRepositoryError("reviewer credential must stay outside Git")
            _require_private_directory(self.path.parent, label="reviewer credential directory")
            payload = _read_private_json(self.path, label="reviewer credential")
            return ReviewerIdentity.model_validate(payload)
        except (
            AccessRepositoryError,
            GateRepositoryError,
            ValidationError,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise AccessRepositoryError("reviewer identity could not be verified") from exc
