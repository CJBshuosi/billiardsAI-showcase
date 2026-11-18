"""Closed Story 6.1 table-binding errors."""

from billiards_ai.shared.models import (
    SessionAdmissionErrorCode,
    SessionCaptureErrorCode,
    TableBindingErrorCode,
)


class TableBindingError(RuntimeError):
    """Base table-binding error carrying only a stable public code."""

    def __init__(self, code: TableBindingErrorCode):
        super().__init__(code.value)
        self.code = code


class TableBindingDomainError(TableBindingError):
    """Authoritative state denied a table-binding operation."""


class TableBindingServiceError(TableBindingError):
    """Safe public service failure."""


class TableBindingRepositoryError(RuntimeError):
    """The repository cannot prove or persist authoritative state."""


class TableBindingCommitStateUnknownError(TableBindingRepositoryError):
    """A failed durable commit could not prove either the old or new snapshot."""


class SessionAdmissionRepositoryDomainError(RuntimeError):
    """Stable denial raised inside the shared table-binding transaction."""

    def __init__(self, code: SessionAdmissionErrorCode):
        super().__init__(code.value)
        self.code = code


class SessionCaptureRepositoryDomainError(RuntimeError):
    def __init__(self, code: SessionCaptureErrorCode):
        super().__init__(code.value)
        self.code = code
