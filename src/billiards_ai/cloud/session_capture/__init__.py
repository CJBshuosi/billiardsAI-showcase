"""Story 6.3 staff capture boundary."""

from .errors import SessionCaptureServiceError
from .identity import AuthenticatedStaffProvider, StaffPrincipal
from .service import SessionCaptureService
from .storage import BinarySource, ControlledStagingObjectStore, ObjectStore

__all__ = [
    "AuthenticatedStaffProvider",
    "BinarySource",
    "ControlledStagingObjectStore",
    "ObjectStore",
    "SessionCaptureService",
    "SessionCaptureServiceError",
    "StaffPrincipal",
]
