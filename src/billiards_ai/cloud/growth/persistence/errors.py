class GrowthPersistenceIntegrityError(RuntimeError):
    """A durable authority graph failed strict reconstruction."""


class GrowthIdentityError(RuntimeError):
    """Trusted user identity is unavailable or invalid."""
