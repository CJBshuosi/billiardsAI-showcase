import json

import httpx
import pytest

from cueorbit_replay.model import ModelError
from cueorbit_replay.providers.deepseek import DeepSeekModel
from cueorbit_replay.rules.model import RulesDeepSeekModel


@pytest.mark.parametrize("adapter", [RulesDeepSeekModel, DeepSeekModel])
async def test_native_null_tool_calls_is_a_valid_final_answer(adapter):
    draft = {"status": "unsupported", "text": "Not supported."}
    if adapter is RulesDeepSeekModel:
        draft.update(rule_citations=[], shot_evidence_refs=[])
    else:
        draft["evidence_refs"] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"tool_calls": None, "content": json.dumps(draft)}}
                    ]
                },
            )
        )
    ) as client:
        turn = await adapter("fixture-key", "fixture-model", client=client).next_turn(
            [], [], timeout_s=1
        )
    assert turn.answer.status == "unsupported"


@pytest.mark.parametrize(
    "message,finish,code",
    [
        ({"content": ""}, "stop", "MODEL_EMPTY_CONTENT"),
        ({"content": '{"status":'}, "length", "MODEL_OUTPUT_TRUNCATED"),
        ({"content": "not-json"}, "stop", "MODEL_FINAL_JSON_INVALID"),
        (
            {"content": '{"status":"answered","text":"x","unknown":"SECRET"}'},
            "stop",
            "MODEL_FINAL_CONTRACT_INVALID",
        ),
        (
            {
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "get_rule", "arguments": "{SECRET"},
                    }
                ]
            },
            "tool_calls",
            "MODEL_TOOL_ARGUMENTS_INVALID",
        ),
        ({"tool_calls": {"SECRET": "bad"}}, "tool_calls", "MODEL_TOOL_CALLS_INVALID"),
    ],
)
async def test_failed_stage_has_safe_distinct_error_code(message, finish, code):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"choices": [{"finish_reason": finish, "message": message}]}
            )
        )
    ) as client:
        with pytest.raises(ModelError) as error:
            await RulesDeepSeekModel(
                "KEY_SECRET", "fixture-model", client=client
            ).next_turn([], [], timeout_s=1)
    assert error.value.code == code and "SECRET" not in str(error.value)


async def test_provider_bad_json_transport_is_not_mislabeled_as_final_schema():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="SECRET{"))
    ) as client:
        with pytest.raises(ModelError) as error:
            await RulesDeepSeekModel(
                "KEY_SECRET", "fixture-model", client=client
            ).next_turn([], [], timeout_s=1)
    assert error.value.code == "PROVIDER_JSON_INVALID"


async def test_failed_native_turn_stays_error_with_sources_preserved_not_fallback_success():
    from tests.rules.unit.test_assistant import Runtime, call, invoke

    r = await invoke(
        "Explain II.11",
        [
            call("get_rule", {"rule_number": "II.11"}),
            ModelError("MODEL_OUTPUT_TRUNCATED"),
        ],
        runtime=Runtime(),
    )
    assert r.status == "error" and r.model_rounds == 2 and len(r.tool_trace) == 1
    assert r.execution_events[-1].error_code == "MODEL_OUTPUT_TRUNCATED"
    assert r.usage.input_tokens is None and r.rule_citations == ()


@pytest.mark.parametrize(
    "bad,expected_path,expected_type",
    [
        (
            {
                "status": "answered",
                "text": "ok",
                "rule_citations": [
                    {"chunk_id": "known", "quote": "SENSITIVE_QUOTE_" * 200}
                ],
            },
            "rule_citations.0.quote",
            "string_too_long",
        ),
        (
            {
                "status": "answered",
                "text": "ok",
                "SENSITIVE_FIELD_NAME": "SENSITIVE_VALUE",
            },
            "extra",
            "extra_forbidden",
        ),
        ({"status": "SENSITIVE_STATUS", "text": "ok"}, "status", "literal_error"),
    ],
)
async def test_final_contract_diagnostics_include_only_allowlisted_path_and_type(
    bad, expected_path, expected_type
):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(bad)}}]}
            )
        )
    ) as client:
        with pytest.raises(ModelError) as caught:
            await RulesDeepSeekModel(
                "KEY_SECRET", "fixture-model", client=client
            ).next_turn([], [], timeout_s=1)
    safe = [d.model_dump() for d in caught.value.diagnostics]
    assert {"path": expected_path, "type": expected_type} in safe
    assert "SENSITIVE" not in json.dumps(safe) and "SECRET" not in str(caught.value)
    assert all(set(d) == {"path", "type"} for d in safe)
