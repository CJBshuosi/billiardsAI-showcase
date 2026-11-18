"""Closed public failures for staff capture operations."""

from billiards_ai.shared.models.session_capture import SessionCaptureErrorCode


class SessionCaptureServiceError(RuntimeError):
    def __init__(self, code: SessionCaptureErrorCode):
        self.code = code
        super().__init__(code.value)


class SessionCaptureDomainError(RuntimeError):
    def __init__(self, code: SessionCaptureErrorCode):
        self.code = code
        super().__init__(code.value)
