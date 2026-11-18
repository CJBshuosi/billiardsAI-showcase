"""Concrete Coach model provider adapters."""

from billiards_ai.cloud.coach_agent.providers.deepseek import (
    AsyncProviderJsonTransport,
    DeepSeekCoachModelProvider,
    DeepSeekProviderConfig,
    ProviderHttpRequest,
    ProviderHttpResponse,
    StdlibHttpsJsonTransport,
)

__all__ = [
    "AsyncProviderJsonTransport",
    "DeepSeekCoachModelProvider",
    "DeepSeekProviderConfig",
    "ProviderHttpRequest",
    "ProviderHttpResponse",
    "StdlibHttpsJsonTransport",
]
