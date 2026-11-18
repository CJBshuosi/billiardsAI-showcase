"""Typed fail-closed calibration lifecycle and persistence errors."""


class CalibrationError(RuntimeError):
    """Base error for calibration lifecycle operations."""


class CalibrationRepositoryError(CalibrationError):
    """The calibration store cannot be trusted or persisted."""


class CalibrationConflictError(CalibrationRepositoryError):
    """The current calibration revision changed before append."""


class CalibrationStaleError(CalibrationError):
    """The calibration no longer matches a usable current configuration."""


class CalibrationInvalidError(CalibrationError):
    """The calibration was explicitly invalidated."""


class CalibrationNotYetValidError(CalibrationError):
    """The frame predates the calibration validation timestamp."""


class CalibrationExpiredError(CalibrationStaleError):
    """The calibration validity window has expired."""


class CalibrationConfigurationError(CalibrationStaleError):
    """The requested camera or table configuration differs from calibration."""
