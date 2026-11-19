"""Unified orchestration: native choices with real scoped tool dispatch, no paid calls."""

import pytest
from pydantic import ValidationError

from cueorbit_replay.assistant import ask_assistant
from cueorbit_replay.contracts import ToolCall, UnifiedAskRequest
from cueorbit_replay.rules.contracts import RuleAnswerDraft, RuleModelTurn
from tests.helpers import FixtureRepository
from tests.rules.unit.test_rules_tools import CONTEXT, FixtureRules
from tests.rules.unit.test_rules_workflow import ScriptedRulesModel, final


class Runtime:
    ready = True
    context = CONTEXT
    encoder = None
    strategy = "keyword"

    def __init__(self):
        self.repository = FixtureRules()


def call(name, args):
    return RuleModelTurn(
        tool_calls=(ToolCall(call_id="call-" + name, name=name, arguments=args),)
    )


def replay_final(ref="ev-complete-7", status="answered"):
    return RuleModelTurn(
        answer=RuleAnswerDraft(
            status=status,
            text="Synthetic observation only.",
            shot_evidence_refs=(ref,) if ref else (),
        )
    )


async def invoke(question, turns, *, runtime=None, **scope):
    return await ask_assistant(
        UnifiedAskRequest(question=question, **scope),
        repository=FixtureRepository(),
        model=ScriptedRulesModel(turns),
        rules_runtime=runtime,
    )


@pytest.mark.parametrize(
    "selection", [{}, {"session_id": "demo-complete", "selected_shot_id": "complete-7"}]
)
async def test_rule_question_uses_only_rules_even_with_selection(selection):
    response = await invoke(
        "Explain rule II.11",
        [call("get_rule", {"rule_number": "II.11"}), final()],
        runtime=Runtime(),
        **selection,
    )
    assert (
        response.status == "answered"
        and response.rule_citations[0].rule_number == "II.11"
    )
    assert not response.replay_fact_cards and not response.shot_evidence_refs
    assert response.model_rounds == 2 and len(response.tool_trace) == 1
    assert [e.phase for e in response.execution_events] == ["model", "model"]


async def test_replay_only_works_without_rag_or_irrelevant_rule_citation():
    response = await invoke(
        "What happened in this shot?",
        [call("get_shot", {"shot_id": "complete-7"}), replay_final()],
        session_id="demo-complete",
        selected_shot_id="complete-7",
    )
    assert response.status == "answered" and response.shot_evidence_refs == (
        "ev-complete-7",
    )
    assert not response.rule_citations
    assert response.replay_fact_cards[0].data["sequence"] == 7


@pytest.mark.parametrize(
    "question,args,ref",
    [
        ("What happened on shot 2?", {"sequence": 2}, "ev-complete-2"),
        ("Summarize the session", {}, "ev-complete-7"),
        ("List recorded shots", {}, "ev-complete-7"),
    ],
)
async def test_explicit_and_session_wide_questions_override_selection(
    question, args, ref
):
    name = (
        "get_shot"
        if args
        else "list_shots"
        if "List" in question
        else "get_session_summary"
    )
    response = await invoke(
        question,
        [call(name, args), replay_final(ref)],
        session_id="demo-complete",
        selected_shot_id="complete-7",
    )
    assert response.status == "answered" and response.tool_trace[0].name == name
    assert response.tool_trace[0].arguments == args


async def test_missing_referent_and_foreign_locator_never_leak_other_session():
    response = await invoke("这一杆发生了什么？", [], session_id="demo-complete")
    assert response.status == "clarification" and response.model_rounds == 0
    response = await invoke(
        "this shot",
        [call("get_shot", {"shot_id": "partial-2"})],
        session_id="demo-complete",
        selected_shot_id="partial-2",
    )
    assert response.status == "insufficient_evidence" and not response.replay_fact_cards


async def test_mixed_evidence_and_unknown_conditions_remain_separate():
    chunk = FixtureRules().chunks[0]
    answer = RuleModelTurn(
        answer=RuleAnswerDraft(
            status="insufficient_evidence",
            text="First contact and cue-ball status are missing.",
            shot_evidence_refs=("ev-partial-2",),
            rule_citations=({"chunk_id": chunk.chunk_id, "quote": chunk.text},),
        )
    )
    response = await invoke(
        "Can this shot prove a foul?",
        [
            RuleModelTurn(
                tool_calls=(
                    ToolCall(
                        call_id="s", name="get_shot", arguments={"shot_id": "partial-2"}
                    ),
                    ToolCall(
                        call_id="r", name="get_rule", arguments={"rule_number": "II.11"}
                    ),
                )
            ),
            answer,
        ],
        session_id="demo-partial",
        selected_shot_id="partial-2",
        runtime=Runtime(),
    )
    assert (
        response.status == "insufficient_evidence"
        and response.rule_citations
        and response.shot_evidence_refs
    )
    assert (
        response.replay_fact_cards[0].data["outcome"] == "unknown"
        and not response.adjudication_supported
    )


async def test_unavailable_rules_do_not_fabricate_rule_answer_or_block_replay():
    response = await invoke(
        "Explain II.11", [call("get_rule", {"rule_number": "II.11"})]
    )
    assert (
        response.status in ("unsupported", "insufficient_evidence")
        and not response.rule_citations
    )
    assert response.tool_trace[0].status == "denied"


def test_optional_scope_is_strict_and_server_pins_rule_edition():
    for extra in (
        {"version": "2024"},
        {"ruleset_id": "other"},
        {"selected_shot_id": "complete-7"},
    ):
        with pytest.raises(ValidationError):
            UnifiedAskRequest(question="q", **extra)


async def test_empty_session_can_report_zero_without_fictitious_reference():
    response = await invoke(
        "Session summary",
        [call("get_session_summary", {}), replay_final(None)],
        session_id="demo-empty",
    )
    assert (
        response.status == "answered"
        and response.replay_fact_cards[0].data["recorded_shot_count"] == 0
    )


@pytest.mark.parametrize(
    "question,name,args,ref",
    [
        ("What happened on shot 2?", "get_shot", {"sequence": 2}, "ev-complete-2"),
        ("Summarize this session", "get_session_summary", {}, "ev-complete-7"),
    ],
)
async def test_stale_irrelevant_selection_cannot_block_explicit_or_session_scope(
    question, name, args, ref
):
    response = await invoke(
        question,
        [call(name, args), replay_final(ref)],
        session_id="demo-complete",
        selected_shot_id="obsolete-clip",
    )
    assert response.status == "answered" and response.shot_evidence_refs == (ref,)


async def test_explicit_replay_question_cannot_answer_from_rules_alone():
    r = await invoke(
        "What happened on shot 2?",
        [call("get_rule", {"rule_number": "II.11"}), final()],
        session_id="demo-complete",
        runtime=Runtime(),
    )
    assert (
        r.status == "insufficient_evidence"
        and "MISSING_REPLAY_EVIDENCE" in r.limitations
    )


async def test_session_question_cannot_be_narrowed_by_model_to_selected_shot():
    r = await invoke(
        "Summarize this session",
        [call("get_shot", {"shot_id": "complete-7"}), replay_final()],
        session_id="demo-complete",
        selected_shot_id="complete-7",
    )
    assert r.status == "insufficient_evidence" and not r.replay_fact_cards
    assert r.tool_trace[0].status == "denied"


@pytest.mark.parametrize("omit", ["rules", "replay"])
async def test_both_successful_evidence_families_require_their_own_citations(omit):
    chunk = FixtureRules().chunks[0]
    draft = RuleModelTurn(
        answer=RuleAnswerDraft(
            status="answered",
            text="A combined statement.",
            rule_citations=({"chunk_id": chunk.chunk_id, "quote": chunk.text},)
            if omit != "rules"
            else (),
            shot_evidence_refs=("ev-complete-7",) if omit != "replay" else (),
        )
    )
    first = RuleModelTurn(
        tool_calls=(
            ToolCall(call_id="s", name="get_shot", arguments={"shot_id": "complete-7"}),
            ToolCall(call_id="r", name="get_rule", arguments={"rule_number": "II.11"}),
        )
    )
    r = await invoke(
        "Can this shot be explained by the rules?",
        [first, draft],
        session_id="demo-complete",
        selected_shot_id="complete-7",
        runtime=Runtime(),
    )
    assert r.status == "insufficient_evidence"


@pytest.mark.parametrize(
    "scope", [{}, {"session_id": "demo-complete", "selected_shot_id": "complete-7"}]
)
async def test_offline_jump_shots_rule_question_does_not_require_replay(scope):
    from cueorbit_replay.assistant_model import OfflineAssistantModel

    r = await ask_assistant(
        UnifiedAskRequest(question="Are jump shots legal under the rules?", **scope),
        repository=FixtureRepository(),
        model=OfflineAssistantModel(),
        rules_runtime=Runtime(),
    )
    assert r.status == "answered" and r.rule_citations and not r.replay_fact_cards
