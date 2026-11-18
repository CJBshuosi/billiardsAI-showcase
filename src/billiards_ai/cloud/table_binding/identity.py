"""Trusted upstream identity boundary consumed by Story 6.1."""

from typing import Protocol
from uuid import UUID


class AuthenticatedUserProvider(Protocol):
    def current_user_id(self) -> UUID:
        """Return the already-verified internal user UUID."""
