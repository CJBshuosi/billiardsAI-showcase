"""Provider-neutral structured generation contracts for Coach Agent workflows."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import (
    UUID4,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from billiards_ai.cloud.coach_agent.context import CoachTaskType
from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr


def _validate_finite_json(value: JsonValue, active: set[int] | None = None) -> JsonValue:
    """Reject non-finite numbers and recursive containers at the provider boundary."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if not isinstance(value, (dict, list)):
        return value

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("JSON containers must not be recursive")
    active.add(identity)
    try:
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            _validate_finite_json(item, active)
    finally:
        active.remove(identity)
    return value


def _immutable_json_error(*_: object, **__: object) -> None:
    raise TypeError("JSON value is immutable")


class _FrozenJsonDict(dict[str, JsonValue]):
    __setitem__ = _immutable_json_error
    __delitem__ = _immutable_json_error
    clear = _immutable_json_error
    pop = _immutable_json_error
    popitem = _immutable_json_error
    setdefault = _immutable_json_error
    update = _immutable_json_error
    __ior__ = _immutable_json_error


class _FrozenJsonList(list[JsonValue]):
    __setitem__ = _immutable_json_error
    __delitem__ = _immutable_json_error
    append = _immutable_json_error
    clear = _immutable_json_error
    extend = _immutable_json_error
    insert = _immutable_json_error
    pop = _immutable_json_error
    remove = _immutable_json_error
    reverse = _immutable_json_error
    sort = _immutable_json_error
    __iadd__ = _immutable_json_error
    __imul__ = _immutable_json_error


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _FrozenJsonDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenJsonList(_freeze_json(item) for item in value)
    return value


def _validate_and_freeze_json(value: JsonValue) -> JsonValue:
    return _freeze_json(_validate_finite_json(value))


class CoachModelResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class CoachModelFailureCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    TIMEOUT = "TIMEOUT"
    INVALID_JSON = "INVALID_JSON"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    REQUEST_REJECTED = "REQUEST_REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


_RETRYABLE_FAILURES = frozenset(
    {
        CoachModelFailureCode.TIMEOUT,
        CoachModelFailureCode.RATE_LIMITED,
        CoachModelFailureCode.PROVIDER_UNAVAILABLE,
    }
)


class CoachModelRequest(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    task_type: CoachTaskType
    prompt_version: NonEmptyStr
    system_prompt: NonEmptyStr = Field(repr=False)
    input_payload: dict[str, JsonValue] = Field(repr=False)
    output_schema: dict[str, JsonValue] = Field(repr=False, min_length=1)
    request_id: UUID4
    timeout_ms: StrictInt = Field(ge=1, le=600_000)

    _input_is_finite = field_validator("input_payload")(_validate_and_freeze_json)
    _schema_is_finite = field_validator("output_schema")(_validate_and_freeze_json)


class CoachModelResponse(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal[CoachModelResultStatus.SUCCEEDED] = CoachModelResultStatus.SUCCEEDED
    provider: NonEmptyStr
    model: NonEmptyStr
    provider_request_id: NonEmptyStr | None
    structured_output: dict[str, JsonValue]
    finish_reason: NonEmptyStr
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    latency_ms: StrictInt = Field(ge=0)

    _output_is_finite = field_validator("structured_output")(_validate_and_freeze_json)


class CoachModelFailure(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal[CoachModelResultStatus.FAILED] = CoachModelResultStatus.FAILED
    provider: NonEmptyStr
    model: NonEmptyStr
    code: CoachModelFailureCode
    retryable: StrictBool
    latency_ms: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def retryability_matches_failure_code(self) -> CoachModelFailure:
        if self.retryable is not (self.code in _RETRYABLE_FAILURES):
            raise ValueError("retryability must match the stable failure policy")
        return self


type CoachModelResult = CoachModelResponse | CoachModelFailure


@runtime_checkable
class CoachModelProvider(Protocol):
    async def generate_structured(self, request: CoachModelRequest) -> CoachModelResult: ...


@runtime_checkable
class CoachModelProviderAuditPort(Protocol):
    """Read-only provider identity captured before an invocation can fail."""

    @property
    def audit_provider_name(self) -> str: ...

    @property
    def audit_model_name(self) -> str: ...


def revalidate_coach_model_request(request: object) -> CoachModelRequest:
    """Deeply revalidate values, including objects produced by unsafe copy/construct APIs."""

    if not isinstance(request, CoachModelRequest):
        raise ValueError("invalid coach model request")
    try:
        stored = dict(request.__dict__)
        return CoachModelRequest.model_validate(stored, strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise ValueError("invalid coach model request") from None


def revalidate_coach_model_result(result: object) -> CoachModelResult:
    """Revalidate provider implementations before business workflows consume their result."""

    if isinstance(result, CoachModelResponse):
        model = CoachModelResponse
    elif isinstance(result, CoachModelFailure):
        model = CoachModelFailure
    else:
        raise ValueError("invalid coach model result")
    try:
        return model.model_validate(dict(result.__dict__), strict=True)
    except (AttributeError, RecursionError, TypeError, ValidationError, ValueError):
        raise ValueError("invalid coach model result") from None


__all__ = [
    "CoachModelFailure",
    "CoachModelFailureCode",
    "CoachModelProvider",
    "CoachModelProviderAuditPort",
    "CoachModelRequest",
    "CoachModelResponse",
    "CoachModelResult",
    "CoachModelResultStatus",
    "revalidate_coach_model_request",
    "revalidate_coach_model_result",
]
