"""PostgreSQL persistence boundary for quality-gate authority."""

from .approved_data import PostgresApprovedDataRepository
from .approved_publication import PostgresApprovedEvidencePublisher
from .errors import (
    IdempotencyConflictError,
    NoProjectableApprovedEvidenceError,
    PersistenceIntegrityError,
    PublicationAuthorityError,
    QualityGatePersistenceError,
    ResourceNotFoundError,
    RevisionConflictError,
)
from .outbox import OutboxReservation, PostgresOutboxRepository
from .publication import (
    ReviewPublicationAuthority,
    ReviewPublicationRecord,
    ReviewSessionOwnership,
    StoredApprovedEvidencePublication,
)
from .repository import (
    IdempotencyIdentity,
    IdempotencyReceipt,
    PostgresQualityGateRepository,
    StoredRejectedReviewResult,
    StoredReviewTask,
    StoredReviewTaskPage,
    StoredSourceShot,
)
from .unit_of_work import QualityGateUnitOfWork

__all__ = [
    "IdempotencyConflictError",
    "IdempotencyIdentity",
    "IdempotencyReceipt",
    "NoProjectableApprovedEvidenceError",
    "OutboxReservation",
    "PersistenceIntegrityError",
    "PostgresOutboxRepository",
    "PostgresApprovedEvidencePublisher",
    "PostgresApprovedDataRepository",
    "PostgresQualityGateRepository",
    "QualityGatePersistenceError",
    "QualityGateUnitOfWork",
    "PublicationAuthorityError",
    "ResourceNotFoundError",
    "RevisionConflictError",
    "StoredReviewTask",
    "StoredReviewTaskPage",
    "StoredRejectedReviewResult",
    "StoredSourceShot",
    "ReviewPublicationAuthority",
    "ReviewPublicationRecord",
    "ReviewSessionOwnership",
    "StoredApprovedEvidencePublication",
]
