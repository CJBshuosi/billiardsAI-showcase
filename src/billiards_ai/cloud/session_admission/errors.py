"""Closed public and internal errors for Story 6.2."""

from billiards_ai.shared.models import SessionAdmissionErrorCode


class SessionAdmissionError(RuntimeError):
    def __init__(self, code: SessionAdmissionErrorCode):
        super().__init__(code.value)
        self.code = code


class SessionAdmissionDomainError(SessionAdmissionError):
    """An authoritative state safely denied the requested operation."""


class SessionAdmissionServiceError(SessionAdmissionError):
    """Stable public error with no untrusted exception text."""
