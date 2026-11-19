import asyncio
import json

import pytest

from cueorbit_replay.contracts import AnswerDraft, AskRequest, ModelTurn, ToolCall
from cueorbit_replay.providers.offline import OfflineModel
from cueorbit_replay.workflow import ask
from tests.helpers import FixtureRepository, ScriptedModel

REQ = AskRequest(session_id="demo-complete", question="What happened?")
CALL = ToolCall(call_id="a", name="get_session_summary", arguments={})


def final(refs=(), status="answered"):
    return ModelTurn(
        answer=AnswerDraft(status=status, text="Recorded facts.", evidence_refs=refs)
    )


async def test_native_result_pairing_and_fact_cards():
    model = ScriptedModel([ModelTurn(tool_calls=(CALL,)), final(("ev-complete-1",))])
    response = await ask(REQ, repository=FixtureRepository(), model=model)
    assert response.status == "answered"
    assert response.fact_cards[0].data["recorded_shot_count"] == 8
    assert response.usage.input_tokens is None
    history = model.messages[1]
    assert history[-1]["tool_call_id"] == "a"
    assert history[-2]["tool_calls"][0]["id"] == "a"
    assert json.loads(history[-1]["content"])["data"]["pocket_observed_count"] == 3


async def test_forged_and_unqueried_answers_fail_closed():
    for turns in [
        [final(("fake",))],
        [final()],
        [ModelTurn(tool_calls=(CALL,)), final(("ev-partial-1",))],
    ]:
        r = await ask(REQ, repository=FixtureRepository(), model=ScriptedModel(turns))
        assert r.status == "insufficient_evidence"


async def test_round_and_parallel_call_limits():
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [
                ModelTurn(tool_calls=(CALL.model_copy(update={"call_id": str(i)}),))
                for i in range(3)
            ]
        ),
    )
    assert r.status == "limit_reached"
    assert len(r.tool_trace) == 3
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [
                ModelTurn(
                    tool_calls=tuple(
                        CALL.model_copy(update={"call_id": str(i)}) for i in range(5)
                    )
                )
            ]
        ),
    )
    assert r.status == "limit_reached"
    assert len(r.tool_trace) == 4


async def test_mixed_success_failure_visible():
    missing = ToolCall(call_id="b", name="get_shot", arguments={"sequence": 99})
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [ModelTurn(tool_calls=(CALL, missing)), final(("ev-complete-1",))]
        ),
    )
    assert r.status == "partial"
    assert [t.status for t in r.tool_trace] == ["ok", "not_found"]
    assert "RECORD_NOT_FOUND" in r.limitations


async def test_provider_error_is_sanitized_and_keeps_cards():
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [ModelTurn(tool_calls=(CALL,)), RuntimeError("sk-secret reasoning_content")]
        ),
    )
    assert r.status == "error" and len(r.fact_cards) == 1
    assert "sk-secret" not in r.model_dump_json()


async def test_clock_deadline_no_execution_after_budget():
    now = [0.0]

    class SlowModel:
        async def next_turn(self, messages, tool_schemas, *, timeout_s):
            assert timeout_s <= 15
            now[0] = 31
            return ModelTurn(tool_calls=(CALL,))

    r = await ask(
        REQ, repository=FixtureRepository(), model=SlowModel(), clock=lambda: now[0]
    )
    assert r.status == "timeout" and not r.tool_trace


async def test_provider_timeout():
    r = await ask(
        REQ, repository=FixtureRepository(), model=ScriptedModel([TimeoutError()])
    )
    assert r.status == "timeout"


async def test_cancellation_propagates():
    entered = asyncio.Event()

    class Waiting:
        async def next_turn(self, *args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        ask(REQ, repository=FixtureRepository(), model=Waiting())
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_tool_timeout_and_failure(monkeypatch):
    import cueorbit_replay.workflow as workflow

    async def timeout(*args):
        raise TimeoutError()

    monkeypatch.setattr(workflow, "execute_tool", timeout)
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [ModelTurn(tool_calls=(CALL,)), final(status="insufficient_evidence")]
        ),
    )
    assert r.tool_trace[0].status == "timeout"
    assert "TOOL_TIMEOUT" in r.limitations


@pytest.mark.parametrize(
    "question,status,session",
    [
        ("Why did I miss?", "insufficient_evidence", "demo-complete"),
        ("为什么没进？", "insufficient_evidence", "demo-partial"),
        ("Explain the rules", "unsupported", "demo-complete"),
        ("Tell me about that one", "clarification", "demo-complete"),
        ("What happened on shot 99?", "insufficient_evidence", "demo-complete"),
        ("What happened in this session?", "partial", "demo-partial"),
        ("What happened in this session?", "answered", "demo-empty"),
        ("Show the shots with observed pocket events.", "answered", "demo-complete"),
    ],
)
async def test_offline_scripted_classes(question, status, session):
    r = await ask(
        AskRequest(session_id=session, question=question),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert r.status == status
    assert r.mode == "offline"


async def test_unknown_session():
    r = await ask(
        REQ.model_copy(update={"session_id": "other"}),
        repository=FixtureRepository(),
        model=ScriptedModel([]),
    )
    assert r.status == "insufficient_evidence"


async def test_real_provider_timeout_cancels_pending_call(monkeypatch):
    import cueorbit_replay.workflow as workflow

    monkeypatch.setattr(workflow, "MODEL_TIMEOUT", 0.01)
    cancelled = asyncio.Event()

    class Slow:
        async def next_turn(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    response = await ask(REQ, repository=FixtureRepository(), model=Slow())
    assert response.status == "timeout"
    assert cancelled.is_set()


async def test_cumulative_deadline_shrinks_model_timeout():
    now = [0.0]

    class Consumes:
        def __init__(self):
            self.timeouts = []

        async def next_turn(self, messages, schemas, *, timeout_s):
            self.timeouts.append(timeout_s)
            now[0] += 14
            return ModelTurn(
                tool_calls=(
                    CALL.model_copy(update={"call_id": str(len(self.timeouts))}),
                )
            )

    model = Consumes()
    r = await ask(
        REQ, repository=FixtureRepository(), model=model, clock=lambda: now[0]
    )
    assert model.timeouts == [15, 15, 2]
    assert r.status == "timeout"
    assert len(r.tool_trace) == 2


async def test_duplicate_call_ids_stop_without_reexecution():
    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel([ModelTurn(tool_calls=(CALL, CALL))]),
    )
    assert r.status == "error"
    assert len(r.tool_trace) == 1


async def test_offline_list_all_recorded_shots():
    r = await ask(
        REQ.model_copy(update={"question": "Show recorded shots."}),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert r.status == "answered"
    assert len(r.fact_cards[0].data["shots"]) == 8


@pytest.mark.parametrize(
    "question,session,sequences,status",
    [
        (
            "Show shots with no pocket observed.",
            "demo-complete",
            [1, 3, 5, 6, 8],
            "answered",
        ),
        ("显示没有进球观测的单杆", "demo-partial", [3], "partial"),
        ("Show shots with unknown outcomes.", "demo-partial", [2], "partial"),
    ],
)
async def test_offline_negative_and_unknown_filters(
    question, session, sequences, status
):
    r = await ask(
        AskRequest(session_id=session, question=question),
        repository=FixtureRepository(),
        model=OfflineModel(),
    )
    assert r.status == status
    assert [s["sequence"] for s in r.fact_cards[0].data["shots"]] == sequences


@pytest.mark.parametrize(
    "failure", [TimeoutError(), RuntimeError("malformed response")]
)
async def test_failed_model_attempt_makes_aggregate_usage_unknown(failure):
    from cueorbit_replay.contracts import Usage

    r = await ask(
        REQ,
        repository=FixtureRepository(),
        model=ScriptedModel(
            [
                ModelTurn(
                    tool_calls=(CALL,), usage=Usage(input_tokens=11, output_tokens=7)
                ),
                failure,
            ]
        ),
    )
    assert r.status in ("timeout", "error")
    assert r.usage.input_tokens is None
    assert r.usage.output_tokens is None
