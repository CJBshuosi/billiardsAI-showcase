"""Domain errors for REAL_VIDEO_GATE operations."""


class GateError(RuntimeError):
    """Base error for the real-video gate."""


class GateRepositoryError(GateError):
    """The gate store cannot be trusted or persisted safely."""


class GateConflictError(GateRepositoryError):
    """A writer attempted to append after a stale revision."""


class GateApprovalError(GateError):
    """Approval identity or evidence does not satisfy the gate."""
