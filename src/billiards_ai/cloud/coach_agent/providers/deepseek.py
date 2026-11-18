"""DeepSeek JSON Output adapter behind the provider-neutral Coach Agent protocol."""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import Field, SecretStr, StrictInt, field_validator

from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
    CoachModelRequest,
    CoachModelResponse,
    CoachModelResult,
    revalidate_coach_model_request,
)
from billiards_ai.shared.models.events import CoreContractModel, NonEmptyStr

_PROVIDER = "deepseek"
_OFFICIAL_HOST = "api.deepseek.com"
_CHAT_COMPLETIONS_PATH = "/chat/completions"
_CONTENT_TYPE = "application/json"


class DeepSeekProviderConfig(CoreContractModel):
    schema_version: Literal["1.0"] = "1.0"
    model: NonEmptyStr
    api_key: SecretStr = Field(repr=False)
    max_tokens: StrictInt = Field(default=4096, ge=1, le=384_000)
    max_response_bytes: StrictInt = Field(default=2_000_000, ge=1, le=10_000_000)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if (
            not secret
            or secret != secret.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in secret)
        ):
            raise ValueError("api key contains invalid characters")
        return value


@dataclass(frozen=True)
class ProviderHttpRequest:
    host: str
    path: str
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    body: bytes = field(repr=False)
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True)
class ProviderHttpResponse:
    status_code: int
    body: bytes = field(repr=False)


class AsyncProviderJsonTransport(Protocol):
    async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse: ...


class StdlibHttpsJsonTransport:
    """Small production transport restricted to DeepSeek's official HTTPS endpoint."""

    async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse:
        if request.host != _OFFICIAL_HOST or request.path != _CHAT_COMPLETIONS_PATH:
            raise OSError("provider endpoint is not allowed")
        connection = http.client.HTTPSConnection(request.host, timeout=request.timeout_seconds)
        try:
            return await asyncio.to_thread(self._post_sync, connection, request)
        except asyncio.CancelledError:
            connection.close()
            raise
        except http.client.HTTPException:
            raise OSError("provider transport failed") from None
        except OSError:
            raise

    @staticmethod
    def _post_sync(
        connection: http.client.HTTPSConnection,
        request: ProviderHttpRequest,
    ) -> ProviderHttpResponse:
        try:
            connection.request(
                "POST",
                request.path,
                body=request.body,
                headers=dict(request.headers),
            )
            response = connection.getresponse()
            body = response.read(request.max_response_bytes + 1)
            return ProviderHttpResponse(status_code=response.status, body=body)
        finally:
            connection.close()


class _DuplicateJsonKey(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _require_bounded_json_depth(value: object, *, max_depth: int = 128) -> None:
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > max_depth:
            raise ValueError("JSON nesting is too deep")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _load_json_object(raw: bytes | str) -> dict[str, object]:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="strict")
    elif isinstance(raw, str):
        text = raw
    else:
        raise ValueError("JSON input must be text")
    value = json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_non_finite_constant,
        parse_float=_parse_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    _require_bounded_json_depth(value)
    return value


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token count must be a non-negative integer")
    return value


class DeepSeekCoachModelProvider:
    """Map the official DeepSeek JSON Output API to neutral Coach model results."""

    def __init__(
        self,
        *,
        config: DeepSeekProviderConfig,
        transport: AsyncProviderJsonTransport | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._config = DeepSeekProviderConfig.model_validate(config.__dict__, strict=True)
        self._transport = transport if transport is not None else StdlibHttpsJsonTransport()
        self._monotonic_ns = monotonic_ns

    @property
    def audit_provider_name(self) -> str:
        return _PROVIDER

    @property
    def audit_model_name(self) -> str:
        return self._config.model

    async def generate_structured(self, request: CoachModelRequest) -> CoachModelResult:
        started_ns = self._read_clock()
        try:
            validated_request = revalidate_coach_model_request(request)
            wire_request = self._build_request(validated_request)
        except (TypeError, ValueError):
            return self._failure(CoachModelFailureCode.INVALID_REQUEST, started_ns)

        try:
            async with asyncio.timeout(validated_request.timeout_ms / 1000):
                response = await self._transport.post(wire_request)
        except TimeoutError:
            return self._failure(CoachModelFailureCode.TIMEOUT, started_ns)
        except OSError:
            return self._failure(CoachModelFailureCode.PROVIDER_UNAVAILABLE, started_ns)
        except Exception:
            return self._failure(CoachModelFailureCode.INTERNAL_FAILURE, started_ns)

        if not isinstance(response, ProviderHttpResponse):
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)
        try:
            result = self._parse_response(response, started_ns)
        except (AttributeError, RecursionError, TypeError, ValueError):
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)
        except Exception:
            return self._failure(CoachModelFailureCode.INTERNAL_FAILURE, started_ns)
        if result.latency_ms > validated_request.timeout_ms:
            return self._failure_with_latency(
                CoachModelFailureCode.TIMEOUT,
                result.latency_ms,
            )
        return result

    def _build_request(self, request: CoachModelRequest) -> ProviderHttpRequest:
        instruction = {
            "task_type": request.task_type.value,
            "prompt_version": request.prompt_version,
            "input_payload": request.input_payload,
            "output_schema": request.output_schema,
        }
        user_content = json.dumps(
            instruction,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        body = json.dumps(
            {
                "model": self._config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": request.system_prompt
                        + "\n\nReturn valid json only and match the supplied output_schema.",
                    },
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": self._config.max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return ProviderHttpRequest(
            host=_OFFICIAL_HOST,
            path=_CHAT_COMPLETIONS_PATH,
            headers=(
                ("Authorization", f"Bearer {self._config.api_key.get_secret_value()}"),
                ("Content-Type", _CONTENT_TYPE),
                ("Accept", _CONTENT_TYPE),
            ),
            body=body,
            timeout_seconds=request.timeout_ms / 1000,
            max_response_bytes=self._config.max_response_bytes,
        )

    def _parse_response(
        self,
        response: ProviderHttpResponse,
        started_ns: int | None,
    ) -> CoachModelResult:
        if (
            isinstance(response.status_code, bool)
            or not isinstance(response.status_code, int)
            or not isinstance(response.body, bytes)
        ):
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)

        status_failure = self._status_failure(response.status_code)
        if status_failure is not None:
            return self._failure(status_failure, started_ns)
        if len(response.body) > self._config.max_response_bytes:
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)

        try:
            envelope = _load_json_object(response.body)
            provider_request_id = envelope.get("id")
            provider_model = envelope.get("model")
            if not isinstance(provider_request_id, str) or not provider_request_id.strip():
                raise ValueError("invalid provider request id")
            if not isinstance(provider_model, str) or not provider_model.strip():
                raise ValueError("invalid provider model")
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("invalid choices")
            choice = choices[0]
            message = choice.get("message")
            finish_reason = choice.get("finish_reason")
            choice_index = choice.get("index")
            if (
                isinstance(choice_index, bool)
                or choice_index != 0
                or not isinstance(message, dict)
                or message.get("role") != "assistant"
                or not isinstance(finish_reason, str)
            ):
                raise ValueError("invalid choice")
            content = message.get("content")
            if not isinstance(content, str) or not finish_reason.strip():
                raise ValueError("invalid content")

            usage = envelope.get("usage")
            if usage is None:
                input_tokens = output_tokens = None
            elif isinstance(usage, dict):
                required_usage = {"prompt_tokens", "completion_tokens", "total_tokens"}
                if not required_usage <= usage.keys():
                    raise ValueError("incomplete usage")
                input_tokens = _non_negative_int(usage["prompt_tokens"])
                output_tokens = _non_negative_int(usage["completion_tokens"])
                total_tokens = _non_negative_int(usage["total_tokens"])
                if total_tokens != input_tokens + output_tokens:
                    raise ValueError("inconsistent usage")
            else:
                raise ValueError("invalid usage")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKey,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)

        try:
            structured_output = _load_json_object(content)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKey,
            RecursionError,
            TypeError,
            ValueError,
        ):
            return self._failure(CoachModelFailureCode.INVALID_JSON, started_ns)

        try:
            return CoachModelResponse(
                provider=_PROVIDER,
                model=provider_model,
                provider_request_id=provider_request_id,
                structured_output=structured_output,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=self._latency_ms(started_ns),
            )
        except (TypeError, ValueError):
            return self._failure(CoachModelFailureCode.INVALID_RESPONSE, started_ns)

    @staticmethod
    def _status_failure(status_code: int) -> CoachModelFailureCode | None:
        if not 100 <= status_code <= 599 or status_code != 200 and 200 <= status_code < 300:
            return CoachModelFailureCode.INVALID_RESPONSE
        if status_code == 200:
            return None
        if status_code == 401:
            return CoachModelFailureCode.AUTHENTICATION_FAILED
        if status_code == 408:
            return CoachModelFailureCode.TIMEOUT
        if status_code == 429:
            return CoachModelFailureCode.RATE_LIMITED
        if status_code >= 500:
            return CoachModelFailureCode.PROVIDER_UNAVAILABLE
        return CoachModelFailureCode.REQUEST_REJECTED

    def _failure(
        self,
        code: CoachModelFailureCode,
        started_ns: int | None,
    ) -> CoachModelFailure:
        return self._failure_with_latency(code, self._latency_ms(started_ns))

    def _failure_with_latency(
        self,
        code: CoachModelFailureCode,
        latency_ms: int,
    ) -> CoachModelFailure:
        return CoachModelFailure(
            provider=_PROVIDER,
            model=self._config.model,
            code=code,
            retryable=code
            in {
                CoachModelFailureCode.TIMEOUT,
                CoachModelFailureCode.RATE_LIMITED,
                CoachModelFailureCode.PROVIDER_UNAVAILABLE,
            },
            latency_ms=latency_ms,
        )

    def _read_clock(self) -> int | None:
        try:
            value = self._monotonic_ns()
        except Exception:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _latency_ms(self, started_ns: int | None) -> int:
        completed_ns = self._read_clock()
        if started_ns is None or completed_ns is None or completed_ns < started_ns:
            return 0
        return (completed_ns - started_ns) // 1_000_000


__all__ = [
    "AsyncProviderJsonTransport",
    "DeepSeekCoachModelProvider",
    "DeepSeekProviderConfig",
    "ProviderHttpRequest",
    "ProviderHttpResponse",
    "StdlibHttpsJsonTransport",
]
