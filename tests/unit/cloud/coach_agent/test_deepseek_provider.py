from __future__ import annotations

import asyncio
import http.client
import json
import threading
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

import pytest
from pydantic import SecretStr

from billiards_ai.cloud.coach_agent import CoachTaskType
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
    CoachModelProvider,
    CoachModelRequest,
    CoachModelResponse,
    revalidate_coach_model_result,
)
from billiards_ai.cloud.coach_agent.providers.deepseek import (
    AsyncProviderJsonTransport,
    DeepSeekCoachModelProvider,
    DeepSeekProviderConfig,
    ProviderHttpRequest,
    ProviderHttpResponse,
    StdlibHttpsJsonTransport,
)


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def model_request(*, timeout_ms: int = 30_000) -> CoachModelRequest:
    return CoachModelRequest(
        task_type=CoachTaskType.GROWTH_ASSESSMENT,
        prompt_version="growth-assessment.v1",
        system_prompt="You are a bounded coach. Return json only.",
        input_payload={
            "context_id": "00000000-0000-4000-8000-000000009301",
            "private_marker": "user-marker-must-not-leak",
            "metrics": [{"code": "POT_SUCCESS_RATE", "value": 0.31}],
        },
        output_schema={
            "type": "object",
            "required": ["primary_finding"],
            "properties": {"primary_finding": {"type": "string"}},
        },
        request_id=uid(9302),
        timeout_ms=timeout_ms,
    )


def response_body(
    *,
    content: str = '{"primary_finding":"LARGE_ANGLE_POTTING"}',
    finish_reason: object = "stop",
    usage: object = None,
) -> bytes:
    payload = {
        "id": "deepseek-request-1",
        "model": "provider-side-model-label",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


@dataclass
class ScriptedTransport:
    response: ProviderHttpResponse | None = None
    error: BaseException | None = None
    requests: list[ProviderHttpRequest] = field(default_factory=list)

    async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeProvider:
    def __init__(self, result: CoachModelResponse):
        self.result = result
        self.requests: list[CoachModelRequest] = []

    async def generate_structured(self, request: CoachModelRequest) -> CoachModelResponse:
        self.requests.append(request)
        return self.result


async def consume_common_result(
    provider: CoachModelProvider, request: CoachModelRequest
) -> tuple[str, str, dict[str, object]]:
    result = revalidate_coach_model_result(await provider.generate_structured(request))
    assert isinstance(result, CoachModelResponse)
    return result.provider, result.model, cast(dict[str, object], result.structured_output)


def provider(
    transport: AsyncProviderJsonTransport,
    *,
    clock_values: tuple[int, int] = (1_000_000_000, 1_042_000_000),
    max_response_bytes: int = 1_000_000,
) -> DeepSeekCoachModelProvider:
    values = iter(clock_values)
    return DeepSeekCoachModelProvider(
        config=DeepSeekProviderConfig(
            model="deepseek-v4-flash",
            api_key=SecretStr("sk-story-9-3-secret"),
            max_tokens=2048,
            max_response_bytes=max_response_bytes,
        ),
        transport=transport,
        monotonic_ns=lambda: next(values),
    )


@pytest.mark.asyncio
async def test_fake_and_deepseek_adapter_share_protocol_and_common_result() -> None:
    request = model_request()
    expected = CoachModelResponse(
        provider="fake",
        model="fake-model",
        provider_request_id="fake-1",
        structured_output={"primary_finding": "LARGE_ANGLE_POTTING"},
        finish_reason="stop",
        input_tokens=120,
        output_tokens=30,
        latency_ms=42,
    )
    fake: CoachModelProvider = FakeProvider(expected)
    transport = ScriptedTransport(ProviderHttpResponse(status_code=200, body=response_body()))
    real: CoachModelProvider = provider(transport)

    assert isinstance(real, CoachModelProvider)
    fake_result = await consume_common_result(fake, request)
    real_result = await consume_common_result(real, request)

    assert fake_result[2] == real_result[2]
    assert real_result[:2] == ("deepseek", "provider-side-model-label")


@pytest.mark.asyncio
async def test_adapter_builds_official_json_output_wire_request_and_redacts_secret() -> None:
    transport = ScriptedTransport(ProviderHttpResponse(status_code=200, body=response_body()))
    result = await provider(transport).generate_structured(model_request())

    assert isinstance(result, CoachModelResponse)
    assert result.provider_request_id == "deepseek-request-1"
    assert result.model == "provider-side-model-label"
    assert result.latency_ms == 42
    wire = transport.requests[0]
    assert (wire.host, wire.path) == ("api.deepseek.com", "/chat/completions")
    assert dict(wire.headers)["Authorization"] == "Bearer sk-story-9-3-secret"
    assert "sk-story-9-3-secret" not in repr(wire)
    assert "user-marker-must-not-leak" not in repr(wire)
    body = json.loads(wire.body)
    assert body["model"] == "deepseek-v4-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["max_tokens"] == 2048
    assert body["messages"][1]["content"].find("growth-assessment.v1") >= 0
    assert "output_schema" in body["messages"][1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_code", "retryable"),
    (
        (400, CoachModelFailureCode.REQUEST_REJECTED, False),
        (401, CoachModelFailureCode.AUTHENTICATION_FAILED, False),
        (402, CoachModelFailureCode.REQUEST_REJECTED, False),
        (422, CoachModelFailureCode.REQUEST_REJECTED, False),
        (429, CoachModelFailureCode.RATE_LIMITED, True),
        (500, CoachModelFailureCode.PROVIDER_UNAVAILABLE, True),
        (503, CoachModelFailureCode.PROVIDER_UNAVAILABLE, True),
    ),
)
async def test_http_failures_map_to_stable_payload_free_results(
    status_code: int,
    expected_code: CoachModelFailureCode,
    retryable: bool,
) -> None:
    transport = ScriptedTransport(
        ProviderHttpResponse(
            status_code=status_code,
            body=b'{"error":{"message":"sk-secret prompt user-marker-must-not-leak"}}',
        )
    )
    result = await provider(transport).generate_structured(model_request())

    assert isinstance(result, CoachModelFailure)
    assert (result.code, result.retryable) == (expected_code, retryable)
    rendered = repr(result) + result.model_dump_json()
    assert "sk-secret" not in rendered
    assert "prompt" not in rendered.lower()
    assert "user-marker-must-not-leak" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected_code"),
    (
        (b"not-json", CoachModelFailureCode.INVALID_RESPONSE),
        (b'{"id":"one","id":"two","choices":[]}', CoachModelFailureCode.INVALID_RESPONSE),
        (b'{"id":"one","choices":[]}', CoachModelFailureCode.INVALID_RESPONSE),
        (response_body(content=""), CoachModelFailureCode.INVALID_JSON),
        (response_body(content="[]"), CoachModelFailureCode.INVALID_JSON),
        (response_body(content='{"x":1,"x":2}'), CoachModelFailureCode.INVALID_JSON),
        (response_body(content='{"x":NaN}'), CoachModelFailureCode.INVALID_JSON),
        (response_body(content='{"x":1} trailing'), CoachModelFailureCode.INVALID_JSON),
        (
            response_body(usage={"prompt_tokens": True, "completion_tokens": 3}),
            CoachModelFailureCode.INVALID_RESPONSE,
        ),
    ),
)
async def test_invalid_envelope_or_content_json_fails_closed(
    body: bytes, expected_code: CoachModelFailureCode
) -> None:
    result = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=body))
    ).generate_structured(model_request())

    assert isinstance(result, CoachModelFailure)
    assert result.code is expected_code
    assert result.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        b"",
        b"[]",
        b"\xff",
        response_body(content="1"),
        response_body(content='{"x":1e400}'),
        response_body(usage={}),
        response_body(usage={"prompt_tokens": 1, "completion_tokens": 2}),
        response_body(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 99}),
    ),
)
async def test_empty_utf8_root_overflow_and_incomplete_usage_fail_closed(body: bytes) -> None:
    result = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=body))
    ).generate_structured(model_request())

    assert isinstance(result, CoachModelFailure)
    assert result.code in {
        CoachModelFailureCode.INVALID_JSON,
        CoachModelFailureCode.INVALID_RESPONSE,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.pop("model"),
        lambda payload: payload["choices"][0].update(index=True),
        lambda payload: payload["choices"][0].update(index=1),
        lambda payload: payload["choices"][0]["message"].update(role="user"),
        lambda payload: payload["choices"][0].update(finish_reason=""),
    ),
)
async def test_invalid_official_envelope_shape_is_rejected(mutation) -> None:
    payload = json.loads(response_body())
    mutation(payload)
    result = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=json.dumps(payload).encode()))
    ).generate_structured(model_request())

    assert isinstance(result, CoachModelFailure)
    assert result.code is CoachModelFailureCode.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_deep_json_and_invalid_transport_object_never_escape() -> None:
    deep_content = '{"x":' * 2_000 + "0" + "}" * 2_000
    deep_result = await provider(
        ScriptedTransport(
            ProviderHttpResponse(status_code=200, body=response_body(content=deep_content))
        )
    ).generate_structured(model_request())

    class InvalidTransport:
        async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse:
            return cast(ProviderHttpResponse, None)

    invalid_result = await provider(InvalidTransport()).generate_structured(model_request())

    assert isinstance(deep_result, CoachModelFailure)
    assert deep_result.code is CoachModelFailureCode.INVALID_JSON
    assert isinstance(invalid_result, CoachModelFailure)
    assert invalid_result.code is CoachModelFailureCode.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_timeout_transport_and_unexpected_failures_are_stable() -> None:
    class NeverReturns:
        async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    timeout = await provider(
        NeverReturns(), clock_values=(1_000_000_000, 1_002_000_000)
    ).generate_structured(model_request(timeout_ms=1))
    unavailable = await provider(
        ScriptedTransport(error=OSError("provider body and secret"))
    ).generate_structured(model_request())
    internal = await provider(
        ScriptedTransport(error=RuntimeError("prompt user-marker-must-not-leak"))
    ).generate_structured(model_request())

    assert isinstance(timeout, CoachModelFailure)
    assert (timeout.code, timeout.retryable) == (CoachModelFailureCode.TIMEOUT, True)
    assert isinstance(unavailable, CoachModelFailure)
    assert (unavailable.code, unavailable.retryable) == (
        CoachModelFailureCode.PROVIDER_UNAVAILABLE,
        True,
    )
    assert isinstance(internal, CoachModelFailure)
    assert (internal.code, internal.retryable) == (
        CoachModelFailureCode.INTERNAL_FAILURE,
        False,
    )
    assert "prompt" not in repr(internal).lower()


@pytest.mark.asyncio
async def test_timeout_covers_complete_adapter_call() -> None:
    result = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=response_body())),
        clock_values=(1_000_000_000, 1_002_000_000),
    ).generate_structured(model_request(timeout_ms=1))

    assert isinstance(result, CoachModelFailure)
    assert result.code is CoachModelFailureCode.TIMEOUT


@pytest.mark.asyncio
async def test_response_body_limit_and_construct_bypass_fail_closed() -> None:
    oversized = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=b"x" * 101)),
        max_response_bytes=100,
    ).generate_structured(model_request())
    bypassed = model_request().model_copy(update={"timeout_ms": True})
    invalid_request = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=200, body=response_body()))
    ).generate_structured(bypassed)

    assert isinstance(oversized, CoachModelFailure)
    assert oversized.code is CoachModelFailureCode.INVALID_RESPONSE
    assert isinstance(invalid_request, CoachModelFailure)
    assert invalid_request.code is CoachModelFailureCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_falsey_transport_is_not_replaced_by_live_transport() -> None:
    class FalseyTransport(ScriptedTransport):
        def __bool__(self) -> bool:
            return False

    transport = FalseyTransport(ProviderHttpResponse(status_code=200, body=response_body()))
    adapter = provider(transport)

    assert adapter._transport is transport
    assert isinstance(await adapter.generate_structured(model_request()), CoachModelResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (99, 206, 600, 700))
async def test_invalid_or_partial_http_status_is_invalid_response(status_code: int) -> None:
    result = await provider(
        ScriptedTransport(ProviderHttpResponse(status_code=status_code, body=response_body()))
    ).generate_structured(model_request())

    assert isinstance(result, CoachModelFailure)
    assert result.code is CoachModelFailureCode.INVALID_RESPONSE


@pytest.mark.asyncio
async def test_invalid_first_clock_read_never_creates_huge_latency() -> None:
    calls = 0

    def clock() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("clock unavailable")
        return 10_000_000_000

    adapter = DeepSeekCoachModelProvider(
        config=DeepSeekProviderConfig(
            model="deepseek-v4-flash",
            api_key=SecretStr("sk-story-9-3-secret"),
        ),
        transport=ScriptedTransport(ProviderHttpResponse(status_code=200, body=response_body())),
        monotonic_ns=clock,
    )

    result = await adapter.generate_structured(model_request())
    assert isinstance(result, CoachModelResponse)
    assert result.latency_ms == 0


@pytest.mark.asyncio
async def test_stdlib_transport_rejects_wrong_endpoint_and_preserves_internal_bug(
    monkeypatch,
) -> None:
    transport = StdlibHttpsJsonTransport()
    wrong_endpoint = ProviderHttpRequest(
        host="example.com",
        path="/chat/completions",
        headers=(),
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=100,
    )
    with pytest.raises(OSError, match="not allowed"):
        await transport.post(wrong_endpoint)

    valid_request = ProviderHttpRequest(
        host="api.deepseek.com",
        path="/chat/completions",
        headers=(),
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=100,
    )
    monkeypatch.setattr(
        StdlibHttpsJsonTransport,
        "_post_sync",
        lambda *args: (_ for _ in ()).throw(RuntimeError("implementation bug")),
    )
    with pytest.raises(RuntimeError, match="implementation bug"):
        await transport.post(valid_request)


@pytest.mark.asyncio
async def test_stdlib_transport_maps_http_failure_without_leaking_detail(monkeypatch) -> None:
    transport = StdlibHttpsJsonTransport()
    request = ProviderHttpRequest(
        host="api.deepseek.com",
        path="/chat/completions",
        headers=(),
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=100,
    )
    monkeypatch.setattr(
        StdlibHttpsJsonTransport,
        "_post_sync",
        lambda *args: (_ for _ in ()).throw(http.client.RemoteDisconnected("secret body")),
    )
    with pytest.raises(OSError, match="provider transport failed") as raised:
        await transport.post(request)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_stdlib_transport_limits_read_closes_connection_and_cancels_active_io(
    monkeypatch,
) -> None:
    connections = []

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.read_limit = None

        def read(self, limit: int) -> bytes:
            self.read_limit = limit
            return b"{}"

    class FakeConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.response = FakeResponse()
            self.closed = threading.Event()
            connections.append(self)

        def request(self, *args, **kwargs) -> None:
            return None

        def getresponse(self) -> FakeResponse:
            return self.response

        def close(self) -> None:
            self.closed.set()

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    transport = StdlibHttpsJsonTransport()
    request = ProviderHttpRequest(
        host="api.deepseek.com",
        path="/chat/completions",
        headers=(),
        body=b"{}",
        timeout_seconds=1,
        max_response_bytes=100,
    )

    response = await transport.post(request)
    assert response.body == b"{}"
    assert connections[0].response.read_limit == 101
    assert connections[0].closed.is_set()

    started = threading.Event()
    released = threading.Event()

    def blocking_post(connection, provider_request) -> ProviderHttpResponse:
        started.set()
        released.wait(timeout=1)
        return ProviderHttpResponse(status_code=200, body=b"{}")

    monkeypatch.setattr(
        StdlibHttpsJsonTransport,
        "_post_sync",
        staticmethod(blocking_post),
    )
    task = asyncio.create_task(transport.post(request))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connections[1].closed.is_set()
    released.set()


@pytest.mark.asyncio
async def test_cancellation_propagates_without_becoming_provider_failure() -> None:
    started = asyncio.Event()

    class CancellableTransport:
        async def post(self, request: ProviderHttpRequest) -> ProviderHttpResponse:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    task = asyncio.create_task(
        provider(CancellableTransport()).generate_structured(model_request())
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
