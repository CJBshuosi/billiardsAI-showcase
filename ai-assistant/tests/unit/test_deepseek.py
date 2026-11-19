import json

import httpx
import pytest

from cueorbit_replay.model import ModelError
from cueorbit_replay.providers.deepseek import DeepSeekModel
from cueorbit_replay.tools import tool_schemas


async def test_native_http_contract_and_hidden_thinking():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "configured-model"
        assert body["thinking"] == {"type": "disabled"}
        assert len(body["tools"]) == 3
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "hidden-secret",
                            "tool_calls": [
                                {
                                    "id": "native-id",
                                    "type": "function",
                                    "function": {
                                        "name": "get_shot",
                                        "arguments": '{"sequence":7}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        turn = await DeepSeekModel(
            "test-key", "configured-model", client=client
        ).next_turn(
            [{"role": "user", "content": "shot 7"}], tool_schemas(), timeout_s=15
        )
    assert turn.tool_calls[0].call_id == "native-id"
    assert turn.usage.input_tokens is None
    assert "hidden-secret" not in turn.model_dump_json()


async def test_selected_clip_uses_named_native_tool_choice():
    def handler(request):
        body = json.loads(request.content)
        assert body["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_shot"},
        }
        assert body["thinking"] == {"type": "disabled"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "selected",
                                    "function": {
                                        "name": "get_shot",
                                        "arguments": '{"shot_id":"complete-7"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        turn = await DeepSeekModel("key", "model", client=client).next_turn(
            [], tool_schemas(), timeout_s=15, required_tool="get_shot"
        )
    assert turn.tool_calls[0].arguments == {"shot_id": "complete-7"}


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "not json"}}]},
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "a",
                                "function": {"name": "get_shot", "arguments": "{"},
                            }
                        ]
                    }
                }
            ]
        },
    ],
)
async def test_invalid_provider_response(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ModelError):
            await DeepSeekModel("key", "model", client=client).next_turn(
                [], tool_schemas(), timeout_s=15
            )


async def test_provider_rejection_never_exposes_raw_payload():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="key-secret"))
    ) as client:
        with pytest.raises(ModelError) as error:
            await DeepSeekModel("key", "model", client=client).next_turn(
                [], tool_schemas(), timeout_s=15
            )
    assert "key-secret" not in str(error.value)
