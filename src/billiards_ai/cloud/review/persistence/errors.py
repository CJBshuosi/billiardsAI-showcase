"""Stable quality-gate persistence failures."""


class QualityGatePersistenceError(RuntimeError):
    """Base class for persistence failures that may cross the application boundary."""


class PersistenceIntegrityError(QualityGatePersistenceError):
    """Stored authority failed digest, column, or strict-contract validation."""


class RevisionConflictError(QualityGatePersistenceError):
    """A conditional mutation did not match the current task revision."""


class ResourceNotFoundError(QualityGatePersistenceError):
    """The requested authority does not exist."""


class IdempotencyConflictError(QualityGatePersistenceError):
    """An idempotency identity was reused with a different request digest."""


class NoProjectableApprovedEvidenceError(QualityGatePersistenceError):
    """A reviewed session has no safe approved evidence that can be published."""


class PublicationAuthorityError(QualityGatePersistenceError):
    """Trusted ownership, metric, or publication lineage is invalid or incomplete."""
