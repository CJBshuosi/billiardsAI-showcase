"""Typed failures for fail-closed media ingestion."""


class MediaProcessingError(RuntimeError):
    """Base class for media failures safe for callers to classify."""


class FFmpegUnavailableError(MediaProcessingError):
    pass


class FFmpegVersionError(MediaProcessingError):
    pass


class ToolExecutionError(MediaProcessingError):
    pass


class ResourceLimitError(MediaProcessingError):
    """The caller-provided media resource policy rejected the input."""


class CorruptMediaError(MediaProcessingError):
    pass


class UnsupportedMediaError(MediaProcessingError):
    pass


class MediaIntegrityError(MediaProcessingError):
    pass


class MediaPublicationError(MediaProcessingError):
    pass
