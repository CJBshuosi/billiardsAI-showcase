import re
from typing import Protocol

from .contracts import ModelTurn, ValidationDiagnostic


class ModelError(Exception):
    """Public-safe provider failure category, never raw response content."""

    def __init__(self, code="MODEL_UNAVAILABLE", *, diagnostics=()):
        known = {
            "LIVE_MODEL_NOT_CONFIGURED",
            "PROVIDER_RESPONSE_TOO_LARGE",
            "PROVIDER_JSON_INVALID",
            "PROVIDER_NETWORK_ERROR",
            "PROVIDER_INVALID_OR_UNAVAILABLE",
            "MODEL_ENVELOPE_INVALID",
            "MODEL_OUTPUT_TRUNCATED",
            "MODEL_USAGE_INVALID",
            "MODEL_TOOL_CALLS_INVALID",
            "MODEL_TOOL_ARGUMENTS_INVALID",
            "MODEL_EMPTY_CONTENT",
            "MODEL_FINAL_JSON_INVALID",
            "MODEL_FINAL_CONTRACT_INVALID",
            "MODEL_UNAVAILABLE",
        }
        self.code = (
            code
            if code in known or re.fullmatch(r"PROVIDER_HTTP_[1-5][0-9]{2}", str(code))
            else "MODEL_UNAVAILABLE"
        )
        self.diagnostics = tuple(
            ValidationDiagnostic.model_validate(d) for d in diagnostics
        )[:8]
        super().__init__(self.code)


class ReplayModelPort(Protocol):
    async def next_turn(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        *,
        timeout_s: float,
        required_tool: str | None = None,
    ) -> ModelTurn: ...
