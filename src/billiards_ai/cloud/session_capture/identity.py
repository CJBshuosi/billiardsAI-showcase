"""Trusted staff identity boundary; no request field may substitute for it."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

CAPTURE_ROLES = frozenset({"CAPTURE_OPERATOR", "VENUE_MANAGER", "STAFF"})


@dataclass(frozen=True, slots=True)
class StaffPrincipal:
    staff_id: UUID
    role: str
    venue_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.staff_id.version != 4
            or self.role not in CAPTURE_ROLES
            or not self.venue_ids
            or any(not venue.strip() for venue in self.venue_ids)
        ):
            raise ValueError("trusted staff capture permission is invalid")


class AuthenticatedStaffProvider(Protocol):
    def current_staff(self) -> StaffPrincipal | None: ...
