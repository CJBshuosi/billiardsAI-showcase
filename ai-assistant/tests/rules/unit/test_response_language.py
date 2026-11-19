import pytest

from cueorbit_replay.contracts import AskRequest
from cueorbit_replay.model import ModelError
from cueorbit_replay.rules.contracts import RuleAnswerDraft, RuleModelTurn
from cueorbit_replay.workflow import ask
from tests.helpers import FixtureRepository, ScriptedModel
from tests.rules.unit.test_assistant import Runtime, call, invoke
from tests.rules.unit.test_rules_workflow import final


@pytest.mark.parametrize(
    "status",
    [
        "unsupported",
        "clarification",
        "error",
        "timeout",
        "insufficient_evidence",
        "limit_reached",
    ],
)
async def test_chinese_nonanswer_branches_remain_chinese(status):
    if status == "error":
        turns = [ModelError("MODEL_FINAL_JSON_INVALID")]
    elif status == "timeout":
        turns = [TimeoutError()]
    elif status == "limit_reached":
        turns = [
            call("get_rule", {"rule_number": "II.11"}).model_copy(
                update={
                    "tool_calls": tuple(
                        call("get_rule", {"rule_number": "II.11"})
                        .tool_calls[0]
                        .model_copy(update={"call_id": str(i)})
                        for i in range(5)
                    )
                }
            )
        ]
    else:
        turns = [
            RuleModelTurn(
                answer=RuleAnswerDraft(
                    status=status, text="English fallback should not leak."
                )
            )
        ]
    r = await invoke("请回答这个问题", turns, runtime=Runtime())
    assert any("\u4e00" <= c <= "\u9fff" for c in r.answer)
    assert "English fallback" not in r.answer
    if status == "error":
        assert (
            "MODEL_FINAL_JSON_INVALID" in r.limitations
            and r.execution_events[-1].error_code == "MODEL_FINAL_JSON_INVALID"
        )


async def test_wrong_language_substantive_answer_is_not_accepted_as_translated_success():
    r = await invoke(
        "需要指袋吗？",
        [call("get_rule", {"rule_number": "II.11"}), final()],
        runtime=Runtime(),
        mode="live",
    )
    assert (
        r.status == "insufficient_evidence"
        and "RESPONSE_LANGUAGE_MISMATCH" in r.limitations
    )
    assert "中文" in r.answer and r.rule_citations


async def test_legacy_replay_error_uses_question_language_and_safe_category():
    r = await ask(
        AskRequest(question="请概括这场记录", session_id="demo-complete", mode="live"),
        repository=FixtureRepository(),
        model=ScriptedModel([ModelError("MODEL_EMPTY_CONTENT")]),
    )
    assert r.status == "error" and "MODEL_EMPTY_CONTENT" in r.limitations
    assert "模型" in r.answer


@pytest.mark.parametrize(
    "question,scope",
    [
        (
            "这一杆发生了什么？",
            {"session_id": "demo-complete", "selected_shot_id": "complete-7"},
        ),
        ("请解释 II.11", {}),
    ],
)
async def test_default_scripted_success_narrative_is_chinese(question, scope):
    from cueorbit_replay.assistant import ask_assistant
    from cueorbit_replay.assistant_model import OfflineAssistantModel
    from cueorbit_replay.contracts import UnifiedAskRequest

    r = await ask_assistant(
        UnifiedAskRequest(question=question, **scope),
        repository=FixtureRepository(),
        model=OfflineAssistantModel(),
        rules_runtime=Runtime(),
    )
    assert r.status == "answered" and any("\u4e00" <= c <= "\u9fff" for c in r.answer)
