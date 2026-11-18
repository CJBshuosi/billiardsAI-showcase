"""Typed failures for normalized-frame ball localization."""


class DetectionError(RuntimeError):
    """Base class for Story 2.5 localization failures."""


class DetectionInputError(DetectionError):
    """The normalized media or frame image violates its declared contract."""
