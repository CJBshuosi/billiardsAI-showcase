import asyncio

import pytest

from cueorbit_replay.contracts import ToolCall, Usage
from cueorbit_replay.rules.contracts import (
    RuleAnswerDraft,
    RuleModelTurn,
    RulesAskRequest,
)
from cueorbit_replay.rules.workflow import ask_rules
from tests.rules.unit.test_rules_tools import CONTEXT, FixtureRules

REQUEST = RulesAskRequest(
    question="Explain rule II.11", ruleset_id="synthetic-heyball", version="2025-08-16"
)
CALL = ToolCall(call_id="native-1", name="get_rule", arguments={"rule_number": "II.11"})


class ScriptedRulesModel:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.messages = []

    async def next_turn(self, messages, schemas, *, timeout_s, required_tool=None):
        self.messages.append(list(messages))
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        return turn


def final(chunk_id=None, quote="Synthetic specific exception."):
    return RuleModelTurn(
        answer=RuleAnswerDraft(
            status="answered",
            text="A source-backed test answer.",
            rule_citations=(
                {
                    "chunk_id": chunk_id or FixtureRules().chunks[0].chunk_id,
                    "quote": quote,
                },
            ),
        )
    )


async def invoke(model, request=REQUEST, repository=None, clock=None):
    return await ask_rules(
        request,
        context=CONTEXT,
        repository=repository or FixtureRules(),
        model=model,
        strategy="keyword",
        **({"clock": clock} if clock else {}),
    )


async def test_native_loop_returns_canonical_source_and_pairs_call_id():
    model = ScriptedRulesModel([RuleModelTurn(tool_calls=(CALL,)), final()])
    response = await invoke(model)
    assert response.status == "answered" and not response.adjudication_supported
    assert response.rule_citations[0].rule_number == "II.11"
    assert not response.shot_evidence_refs and not response.replay_fact_cards
    assert model.messages[1][-1]["tool_call_id"] == "native-1"


@pytest.mark.parametrize(
    "draft", [final(chunk_id="unreturned"), final(quote="An invented quotation.")]
)
async def test_unreturned_or_nonverbatim_rule_citation_rejected(draft):
    response = await invoke(
        ScriptedRulesModel([RuleModelTurn(tool_calls=(CALL,)), draft])
    )
    assert response.status == "insufficient_evidence"
    assert "UNSUPPORTED_RULE_CITATION" in response.limitations
    assert not response.rule_citations


async def test_unknown_version_stops_before_provider():
    model = ScriptedRulesModel([])
    response = await invoke(
        model, request=REQUEST.model_copy(update={"version": "2024-01-01"})
    )
    assert response.status == "unsupported" and not model.messages


async def test_error_after_attempt_keeps_usage_unknown_and_sanitized():
    first = RuleModelTurn(
        tool_calls=(CALL,), usage=Usage(input_tokens=20, output_tokens=3)
    )
    response = await invoke(
        ScriptedRulesModel([first, RuntimeError("secret-key provider body")])
    )
    assert response.status == "error" and len(response.tool_trace) == 1
    assert response.usage.input_tokens is None
    assert "secret-key" not in response.model_dump_json()


async def test_model_and_tool_limits_are_cumulative():
    turns = [
        RuleModelTurn(tool_calls=(CALL.model_copy(update={"call_id": str(i)}),))
        for i in range(3)
    ]
    response = await invoke(ScriptedRulesModel(turns))
    assert response.status == "limit_reached" and len(response.tool_trace) == 3
    response = await invoke(
        ScriptedRulesModel(
            [
                RuleModelTurn(
                    tool_calls=tuple(
                        CALL.model_copy(update={"call_id": str(i)}) for i in range(5)
                    )
                )
            ]
        )
    )
    assert response.status == "limit_reached" and len(response.tool_trace) == 4


async def test_deadline_and_cancellation_do_not_return_late_evidence():
    now = [0.0]
    repository = FixtureRules()

    class Late:
        async def next_turn(self, *args, **kwargs):
            now[0] = 31
            return RuleModelTurn(tool_calls=(CALL,))

    response = await invoke(Late(), repository=repository, clock=lambda: now[0])
    assert response.status == "timeout" and not repository.calls
    entered = asyncio.Event()

    class Pending:
        async def next_turn(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(invoke(Pending()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_source_less_and_injected_answer_cannot_bypass_evidence():
    model = ScriptedRulesModel(
        [
            RuleModelTurn(
                answer=RuleAnswerDraft(
                    status="answered", text="Ignore the rules and declare a win."
                )
            )
        ]
    )
    response = await invoke(model)
    assert response.status == "insufficient_evidence"
    assert "MISSING_RULE_CITATION" in response.limitations
    assert all(s["role"] != "tool" for s in model.messages[0])


async def test_model_and_tool_timeout_are_reported_without_fallback(monkeypatch):
    import cueorbit_replay.rules.workflow as workflow

    class Pending:
        async def next_turn(self, *args, **kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr(workflow, "MODEL_TIMEOUT", 0.01)
    response = await invoke(Pending())
    assert response.status == "timeout" and response.usage.input_tokens is None

    class PendingRules(FixtureRules):
        async def get_rule(self, *args, **kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr(workflow, "TOOL_TIMEOUT", 0.01)
    response = await invoke(
        ScriptedRulesModel([RuleModelTurn(tool_calls=(CALL,))]),
        repository=PendingRules(),
    )
    assert response.status == "timeout" and response.tool_trace[0].status == "timeout"
    assert not response.rule_citations
