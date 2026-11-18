"""Stable public API for privacy gates and evidence services."""

from .access_identity import FileReviewerIdentityProvider
from .access_repository import (
    AccessConflictError,
    AccessError,
    AccessNotFoundError,
    AccessRepositoryError,
    JsonAccessRepository,
)
from .access_service import AccessDeniedError, MediaAccessService
from .authorization import authorize_real_video_use
from .errors import GateApprovalError, GateConflictError, GateError, GateRepositoryError
from .identity import FileApproverIdentityProvider
from .media_store import ControlledFilesystemMediaStore
from .repository import JsonGateRepository
from .retention_repository import (
    JsonRetentionRepository,
    MediaUnavailableError,
    RetentionConflictError,
    RetentionError,
    RetentionNotFoundError,
    RetentionOwnershipError,
    RetentionRepositoryError,
)
from .retention_service import MediaRetentionService
from .service import RealVideoGateService

__all__ = [
    "AccessConflictError",
    "AccessDeniedError",
    "AccessError",
    "AccessNotFoundError",
    "AccessRepositoryError",
    "FileApproverIdentityProvider",
    "FileReviewerIdentityProvider",
    "GateApprovalError",
    "GateConflictError",
    "GateError",
    "GateRepositoryError",
    "JsonGateRepository",
    "JsonAccessRepository",
    "JsonRetentionRepository",
    "ControlledFilesystemMediaStore",
    "MediaRetentionService",
    "MediaAccessService",
    "MediaUnavailableError",
    "RealVideoGateService",
    "RetentionConflictError",
    "RetentionError",
    "RetentionNotFoundError",
    "RetentionOwnershipError",
    "RetentionRepositoryError",
    "authorize_real_video_use",
]
