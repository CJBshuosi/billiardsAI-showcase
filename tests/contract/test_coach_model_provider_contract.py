from __future__ import annotations

from typing import get_args
from uuid import UUID

import pytest
from pydantic import SecretStr, ValidationError

from billiards_ai.cloud.coach_agent import CoachTaskType
from billiards_ai.cloud.coach_agent.provider import (
    CoachModelFailure,
    CoachModelFailureCode,
    CoachModelRequest,
    CoachModelResponse,
    CoachModelResult,
    CoachModelResultStatus,
    revalidate_coach_model_result,
)


def uid(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012d}")


def request_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "task_type": CoachTaskType.GROWTH_ASSESSMENT,
        "prompt_version": "growth-assessment.v1",
        "system_prompt": "Return json only.",
        "input_payload": {"metric": {"value": 0.31, "sample_count": 40}},
        "output_schema": {"type": "object", "required": ["primary_finding"]},
        "request_id": uid(9301),
        "timeout_ms": 30_000,
    }


def test_request_is_strict_frozen_versioned_and_json_bounded() -> None:
    request = CoachModelRequest.model_validate(request_payload(), strict=True)

    assert request.request_id.version == 4
    assert "system_prompt=" not in repr(request)
    assert "input_payload=" not in repr(request)
    with pytest.raises(ValidationError):
        request.timeout_ms = 1
    with pytest.raises(TypeError):
        request.input_payload["mutated"] = True
    with pytest.raises(TypeError):
        request.output_schema["required"].append("mutated")

    invalid_payloads = (
        request_payload() | {"schema_version": "2.0"},
        request_payload() | {"timeout_ms": True},
        request_payload() | {"timeout_ms": 0},
        request_payload() | {"request_id": UUID(int=0)},
        request_payload() | {"input_payload": {"value": float("nan")}},
        request_payload() | {"output_schema": {"maximum": float("inf")}},
        request_payload() | {"unknown": "forbidden"},
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            CoachModelRequest.model_validate(payload, strict=True)


def test_success_and_failure_are_disjoint_payload_free_results() -> None:
    success = CoachModelResponse(
        provider="deepseek",
        model="deepseek-v4-flash",
        provider_request_id="provider-request-1",
        structured_output={"primary_finding": "LARGE_ANGLE_POTTING"},
        finish_reason="stop",
        input_tokens=120,
        output_tokens=30,
        latency_ms=42,
    )
    failure = CoachModelFailure(
        provider="deepseek",
        model="deepseek-v4-flash",
        code=CoachModelFailureCode.TIMEOUT,
        retryable=True,
        latency_ms=30_000,
    )

    assert success.status is CoachModelResultStatus.SUCCEEDED
    assert failure.status is CoachModelResultStatus.FAILED
    assert set(get_args(CoachModelResult.__value__)) == {CoachModelResponse, CoachModelFailure}
    assert set(failure.model_dump(mode="json")) == {
        "schema_version",
        "status",
        "provider",
        "model",
        "code",
        "retryable",
        "latency_ms",
    }
    with pytest.raises(TypeError):
        success.structured_output["mutated"] = float("nan")

    with pytest.raises(ValidationError):
        CoachModelResponse(
            provider="deepseek",
            model="deepseek-v4-flash",
            provider_request_id=None,
            structured_output=[],
            finish_reason="stop",
            input_tokens=None,
            output_tokens=None,
            latency_ms=1,
        )
    with pytest.raises(ValidationError):
        CoachModelFailure(
            provider="deepseek",
            model="deepseek-v4-flash",
            code=CoachModelFailureCode.TIMEOUT,
            retryable=False,
            latency_ms=1,
        )


def test_result_revalidation_rejects_unsafe_construct_and_copy() -> None:
    unsafe_success = CoachModelResponse.model_construct(
        provider="deepseek",
        model="deepseek-v4-flash",
        provider_request_id=None,
        structured_output=None,
        finish_reason="stop",
        input_tokens=None,
        output_tokens=None,
        latency_ms=1,
    )
    unsafe_failure = CoachModelFailure.model_construct(
        provider="deepseek",
        model="deepseek-v4-flash",
        code=CoachModelFailureCode.TIMEOUT,
        retryable=False,
        latency_ms=1,
    )

    with pytest.raises(ValueError, match="invalid coach model result"):
        revalidate_coach_model_result(unsafe_success)
    with pytest.raises(ValueError, match="invalid coach model result"):
        revalidate_coach_model_result(unsafe_failure)


def test_provider_configuration_masks_api_key() -> None:
    from billiards_ai.cloud.coach_agent.providers.deepseek import DeepSeekProviderConfig

    config = DeepSeekProviderConfig(
        model="deepseek-v4-flash",
        api_key=SecretStr("sk-story-9-3-secret"),
    )

    assert "sk-story-9-3-secret" not in repr(config)
    assert "sk-story-9-3-secret" not in config.model_dump_json()
    for invalid_secret in ("abc\ndef", "abc\rdef", "abc\x00def", " abc", "abc "):
        with pytest.raises(ValidationError):
            DeepSeekProviderConfig(
                model="deepseek-v4-flash",
                api_key=SecretStr(invalid_secret),
            )


def test_provider_contract_modules_keep_vendor_and_mainline_boundaries() -> None:
    neutral_source = (
        __import__("pathlib")
        .Path("src/billiards_ai/cloud/coach_agent/provider.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    adapter_source = (
        __import__("pathlib")
        .Path("src/billiards_ai/cloud/coach_agent/providers/deepseek.py")
        .read_text(encoding="utf-8")
        .lower()
    )

    assert all(name not in neutral_source for name in ("openai", "anthropic", "deepseek", "kimi"))
    assert all(
        name not in adapter_source
        for name in (
            "billiards_ai.cloud.growth",
            "billiards_ai.cloud.perception",
            "billiards_ai.cloud.plan_publication",
            "billiards_ai.shared.models.growth",
        )
    )
    assert "import openai" not in adapter_source
